/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ErrorCheckMacros.h"
#include "ref/bundle_adjustment_ref.h"

#include <cupva_host_common.hpp>

#include <OpBundleAdjustment.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

/* Must match the operator's device-side capacities (bundle_adjustment_param.h). */
constexpr int32_t kMaxCameraCount = 21;
constexpr int32_t kMaxTrackCount  = 20480;
constexpr int32_t kTrackCountTile = 128;

/* kTrackCount must be a multiple of kTrackCountTile. */
constexpr int32_t kCameraCount       = 4;
constexpr int32_t kTrackCount        = kTrackCountTile;
constexpr int32_t kActiveCameraCount = kCameraCount;
constexpr int32_t kActiveCameraBegin = 0;

/** @brief Host-side view of a strided NVCV tensor buffer. */
struct TensorView
{
    uint8_t *host                         = nullptr;
    int64_t strides[NVCV_TENSOR_MAX_RANK] = {0};
    int32_t rank                          = 0;

    template<typename T>
    T &At(int64_t i0) const
    {
        return *reinterpret_cast<T *>(host + i0 * strides[0]);
    }
    template<typename T>
    T &At(int64_t i0, int64_t i1) const
    {
        return *reinterpret_cast<T *>(host + i0 * strides[0] + i1 * strides[1]);
    }
    template<typename T>
    T &At(int64_t i0, int64_t i1, int64_t i2) const
    {
        return *reinterpret_cast<T *>(host + i0 * strides[0] + i1 * strides[1] + i2 * strides[2]);
    }
};

static int GetTensorView(NVCVTensorHandle tensor, TensorView *view)
{
    NVCVTensorData data;
    int32_t err   = 0;
    void *hostPtr = nullptr;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensor, &data), err, Error);

    CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer(&hostPtr, (void *)data.buffer.strided.basePtr), err, Error);

    view->host = reinterpret_cast<uint8_t *>(hostPtr);
    view->rank = data.rank;
    for (int32_t i = 0; i < data.rank; i++)
    {
        view->strides[i] = data.buffer.strided.strides[i];
    }
    return 0;

Error:
    return err;
}

/** @brief Synthetic scene: noiseless measurements from ground truth; solvers start from perturbed poses/points. */
struct Scene
{
    std::vector<float> gtPoses;  // kCameraCount * 16, row-major 4x4 world-to-rig
    std::vector<float> gtPoints; // kTrackCount * 4, homogeneous
    std::vector<float> initPoses;
    std::vector<float> initPoints;
    std::vector<float> measurements; // kTrackCount * kCameraCount * 2
    std::vector<int32_t> bitmask;    // bit i set => camera i observes the track
    std::vector<float> rig2cam;      // identity
};

/* Identity rotations, so world_point_in_camera = world_point - t.
 * nearPlanePoint puts track 0 inside the operator's minDistance guard; perturb args of 0 start at the optimum. */
static Scene BuildScene(bool nearPlanePoint, float poseTranslationPerturb, float pointPerturb)
{
    Scene scene;
    scene.gtPoses.assign(kCameraCount * 16, 0.0f);
    scene.gtPoints.assign(kTrackCount * 4, 0.0f);
    scene.initPoses.assign(kCameraCount * 16, 0.0f);
    scene.initPoints.assign(kTrackCount * 4, 0.0f);
    scene.measurements.assign(static_cast<size_t>(kTrackCount) * kCameraCount * 2, 0.0f);
    scene.bitmask.assign(kTrackCount, 0);
    scene.rig2cam.assign(16, 0.0f);
    for (int32_t i = 0; i < 4; i++)
    {
        scene.rig2cam[i * 4 + i] = 1.0f;
    }

    for (int32_t c = 0; c < kCameraCount; c++)
    {
        float tx = (c - 0.5f * (kCameraCount - 1)) * 1.5f;
        float *p = &scene.gtPoses[c * 16];
        for (int32_t i = 0; i < 4; i++)
        {
            p[i * 4 + i] = 1.0f;
        }
        p[0 * 4 + 3] = tx;
        p[1 * 4 + 3] = 0.0f;
        p[2 * 4 + 3] = 0.0f;
    }

    for (int32_t t = 0; t < kTrackCount; t++)
    {
        float x = ((t % 8) - 3.5f) * 0.5f;
        float y = (((t / 8) % 8) - 3.5f) * 0.5f;
        float z = 5.0f;
        if (nearPlanePoint && t == 0)
        {
            z = 0.3f; /* below minDistance (1.0) */
        }
        scene.gtPoints[t * 4 + 0] = x;
        scene.gtPoints[t * 4 + 1] = y;
        scene.gtPoints[t * 4 + 2] = z;
        scene.gtPoints[t * 4 + 3] = 1.0f;
        scene.bitmask[t] = static_cast<int32_t>(((1u << kCameraCount) - 1u) | (1u << 31)); /* bit 31 = track active */
    }

    for (int32_t t = 0; t < kTrackCount; t++)
    {
        for (int32_t c = 0; c < kCameraCount; c++)
        {
            const float *pose                                                       = &scene.gtPoses[c * 16];
            float tx                                                                = pose[0 * 4 + 3];
            float ty                                                                = pose[1 * 4 + 3];
            float tz                                                                = pose[2 * 4 + 3];
            float xc                                                                = scene.gtPoints[t * 4 + 0] - tx;
            float yc                                                                = scene.gtPoints[t * 4 + 1] - ty;
            float zc                                                                = scene.gtPoints[t * 4 + 2] - tz;
            float zSafe                                                             = (zc < 1e-3f) ? 1e-3f : zc;
            scene.measurements[(static_cast<size_t>(t) * kCameraCount + c) * 2 + 0] = xc / zSafe;
            scene.measurements[(static_cast<size_t>(t) * kCameraCount + c) * 2 + 1] = yc / zSafe;
        }
    }

    /* Deterministic (RNG-free) perturbation so runs are reproducible. */
    scene.initPoses = scene.gtPoses;
    for (int32_t c = 0; c < kCameraCount; c++)
    {
        float *p = &scene.initPoses[c * 16];
        p[0 * 4 + 3] += poseTranslationPerturb * std::sin(0.7f * c + 0.3f);
        p[1 * 4 + 3] += poseTranslationPerturb * std::sin(1.3f * c + 1.1f);
        p[2 * 4 + 3] += poseTranslationPerturb * std::sin(1.9f * c + 2.1f);
    }
    scene.initPoints = scene.gtPoints;
    for (int32_t t = 0; t < kTrackCount; t++)
    {
        scene.initPoints[t * 4 + 0] += pointPerturb * std::sin(0.11f * t + 0.2f);
        scene.initPoints[t * 4 + 1] += pointPerturb * std::sin(0.17f * t + 0.5f);
        scene.initPoints[t * 4 + 2] += pointPerturb * std::sin(0.23f * t + 0.9f);
    }

    return scene;
}

static BundleAdjustmentSetting MakeRefSetting(const PvaBundleAdjustmentParams &params)
{
    BundleAdjustmentSetting setting;
    setting.maxIterations        = params.maxIterations;
    setting.minDistance          = params.minDistance;
    setting.robustifierScaleMeas = params.robustifierScaleMeas;
    setting.rinvMeas             = params.rinvMeas;
    setting.lambdaInit           = params.lambdaInit;
    return setting;
}

static PvaBundleAdjustmentParams MakeParams()
{
    PvaBundleAdjustmentParams params;
    params.cameraCount          = kCameraCount;
    params.trackCount           = kTrackCount;
    params.maxIterations        = 10;
    params.minDistance          = 1.0f;
    params.robustifierScaleMeas = 4.f / 1900.f;
    params.rinvMeas             = 1.f / (0.01f * 0.01f);
    params.lambdaInit           = 1e-4f;
    return params;
}

/** @brief Tensors for one Create()/Submit() run, with inputs filled from a Scene. */
struct Tensors
{
    NVCVAllocatorHandle allocatorHandle = NULL;
    NVCVTensorHandle rigPoses           = NULL;
    NVCVTensorHandle points3d           = NULL;
    NVCVTensorHandle measurements       = NULL;
    NVCVTensorHandle cameraBitmask      = NULL;
    NVCVTensorHandle rig2cam            = NULL;
    NVCVTensorHandle updatedRigPoses    = NULL;
    NVCVTensorHandle updatedPoints3d    = NULL;
    NVCVTensorHandle finalCost          = NULL;
    NVCVTensorHandle iterationsUsed     = NULL;

    NVCVTensorRequirements rigPosesReq;
    NVCVTensorRequirements points3dReq;
    NVCVTensorRequirements measurementsReq;
    NVCVTensorRequirements cameraBitmaskReq;
    NVCVTensorRequirements rig2camReq;
    NVCVTensorRequirements finalCostReq;
    NVCVTensorRequirements iterationsUsedReq;
};

static void FreeTensors(Tensors *t)
{
    nvcvTensorDecRef(t->rigPoses, NULL);
    nvcvTensorDecRef(t->points3d, NULL);
    nvcvTensorDecRef(t->measurements, NULL);
    nvcvTensorDecRef(t->cameraBitmask, NULL);
    nvcvTensorDecRef(t->rig2cam, NULL);
    nvcvTensorDecRef(t->updatedRigPoses, NULL);
    nvcvTensorDecRef(t->updatedPoints3d, NULL);
    nvcvTensorDecRef(t->finalCost, NULL);
    nvcvTensorDecRef(t->iterationsUsed, NULL);
    nvcvAllocatorDecRef(t->allocatorHandle, NULL);
    *t = Tensors();
}

static int AllocateAndFillTensors(const Scene &scene, Tensors *t)
{
    int32_t err = 0;
    NVCVTensorLayout layoutHW, layoutHWC, layoutW;
    int64_t rigPosesShape[]      = {kMaxCameraCount, 16};
    int64_t points3dShape[]      = {kMaxTrackCount, 4};
    int64_t measurementsShape[]  = {kMaxTrackCount, kMaxCameraCount, 2};
    int64_t cameraBitmaskShape[] = {kMaxTrackCount};
    int64_t rig2camShape[]       = {16};
    int64_t scalarShape[]        = {1};

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &layoutHW), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &layoutHWC), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layoutW), err, Error);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(2, rigPosesShape, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, &t->rigPosesReq), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(2, points3dShape, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, &t->points3dReq), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(3, measurementsShape, NVCV_DATA_TYPE_F32, layoutHWC, 0, 0, &t->measurementsReq),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, cameraBitmaskShape, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &t->cameraBitmaskReq),
        err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, rig2camShape, NVCV_DATA_TYPE_F32, layoutW, 0, 0, &t->rig2camReq), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, scalarShape, NVCV_DATA_TYPE_F32, layoutW, 0, 0, &t->finalCostReq), err, Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, scalarShape, NVCV_DATA_TYPE_S32, layoutW, 0, 0, &t->iterationsUsedReq), err,
        Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&t->allocatorHandle), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->rigPosesReq, t->allocatorHandle, &t->rigPoses), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->points3dReq, t->allocatorHandle, &t->points3d), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->measurementsReq, t->allocatorHandle, &t->measurements), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->cameraBitmaskReq, t->allocatorHandle, &t->cameraBitmask), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->rig2camReq, t->allocatorHandle, &t->rig2cam), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->rigPosesReq, t->allocatorHandle, &t->updatedRigPoses), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->points3dReq, t->allocatorHandle, &t->updatedPoints3d), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->finalCostReq, t->allocatorHandle, &t->finalCost), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&t->iterationsUsedReq, t->allocatorHandle, &t->iterationsUsed), err,
                          Error);

    {
        TensorView v;
        NVCV_CHECK_ERROR_GOTO(GetTensorView(t->rigPoses, &v), err, Error);
        std::memset(v.host, 0, static_cast<size_t>(kMaxCameraCount) * v.strides[0]);
        for (int32_t c = 0; c < kCameraCount; c++)
        {
            for (int32_t k = 0; k < 16; k++)
            {
                v.At<float>(c, k) = scene.initPoses[c * 16 + k];
            }
        }
    }
    {
        TensorView v;
        NVCV_CHECK_ERROR_GOTO(GetTensorView(t->points3d, &v), err, Error);
        std::memset(v.host, 0, static_cast<size_t>(kMaxTrackCount) * v.strides[0]);
        for (int32_t tr = 0; tr < kTrackCount; tr++)
        {
            for (int32_t k = 0; k < 4; k++)
            {
                v.At<float>(tr, k) = scene.initPoints[tr * 4 + k];
            }
        }
    }
    {
        TensorView v;
        NVCV_CHECK_ERROR_GOTO(GetTensorView(t->measurements, &v), err, Error);
        std::memset(v.host, 0, static_cast<size_t>(kMaxTrackCount) * v.strides[0]);
        for (int32_t tr = 0; tr < kTrackCount; tr++)
        {
            for (int32_t c = 0; c < kCameraCount; c++)
            {
                v.At<float>(tr, c, 0) = scene.measurements[(static_cast<size_t>(tr) * kCameraCount + c) * 2 + 0];
                v.At<float>(tr, c, 1) = scene.measurements[(static_cast<size_t>(tr) * kCameraCount + c) * 2 + 1];
            }
        }
    }
    {
        TensorView v;
        NVCV_CHECK_ERROR_GOTO(GetTensorView(t->cameraBitmask, &v), err, Error);
        std::memset(v.host, 0, static_cast<size_t>(kMaxTrackCount) * v.strides[0]);
        for (int32_t tr = 0; tr < kTrackCount; tr++)
        {
            v.At<int32_t>(tr) = scene.bitmask[tr];
        }
    }
    {
        TensorView v;
        NVCV_CHECK_ERROR_GOTO(GetTensorView(t->rig2cam, &v), err, Error);
        for (int32_t k = 0; k < 16; k++)
        {
            v.At<float>(k) = scene.rig2cam[k];
        }
    }

    return 0;

Error:
    FreeTensors(t);
    return err;
}

/**
 * @brief Compare operator output against the CPU reference using loose tolerances (not bit-exact).
 *
 * The two LM solvers differ in arithmetic ordering, so float error accumulates across iterations.
 */
static int CompareOutputs(const Tensors &t, const Scene &scene, const PvaBundleAdjustmentParams &params,
                          const char *testName)
{
    std::vector<float> refPoses(kCameraCount * 16, 0.0f);
    std::vector<float> refPoints(kTrackCount * 4, 0.0f);
    float refCost             = 0.0f;
    int32_t refIterationsUsed = 0;

    BundleAdjustmentSetting setting = MakeRefSetting(params);
    BundleAdjustmentRef(scene.initPoses.data(), scene.initPoints.data(), scene.measurements.data(),
                        scene.bitmask.data(), scene.rig2cam.data(), kCameraCount, kTrackCount, kActiveCameraCount,
                        kActiveCameraBegin, setting, refPoses.data(), refPoints.data(), &refCost, &refIterationsUsed);

    TensorView poseView, pointView, costView, itersView;
    int32_t err = 0;
    NVCV_CHECK_ERROR_GOTO(GetTensorView(t.updatedRigPoses, &poseView), err, Error);
    NVCV_CHECK_ERROR_GOTO(GetTensorView(t.updatedPoints3d, &pointView), err, Error);
    NVCV_CHECK_ERROR_GOTO(GetTensorView(t.finalCost, &costView), err, Error);
    NVCV_CHECK_ERROR_GOTO(GetTensorView(t.iterationsUsed, &itersView), err, Error);

    /* Scene scale is O(1-5): loose enough for two LM solvers, tight enough to catch divergence. */
    {
        const float absEps = 5e-2f;
        const float relEps = 0.05f;
        bool ok            = true;
        for (int32_t c = 0; c < kCameraCount && ok; c++)
        {
            for (int32_t k = 0; k < 16; k++)
            {
                float a = poseView.At<float>(c, k);
                float b = refPoses[c * 16 + k];
                if (std::isnan(a) || std::isinf(a))
                {
                    printf("%s: pose[%d][%d] is NaN/Inf (pva=%e)\n", testName, c, k, a);
                    ok = false;
                    break;
                }
                float tol = absEps + relEps * std::fabs(b);
                if (std::fabs(a - b) > tol)
                {
                    printf("%s: pose mismatch [cam %d, elem %d]: pva=%e ref=%e (tol=%e)\n", testName, c, k, a, b, tol);
                    ok = false;
                    break;
                }
            }
        }
        for (int32_t tr = 0; tr < kTrackCount && ok; tr++)
        {
            for (int32_t k = 0; k < 3; k++) /* w is always 1 */
            {
                float a = pointView.At<float>(tr, k);
                float b = refPoints[tr * 4 + k];
                if (std::isnan(a) || std::isinf(a))
                {
                    printf("%s: point[%d][%d] is NaN/Inf (pva=%e)\n", testName, tr, k, a);
                    ok = false;
                    break;
                }
                float tol = absEps + relEps * std::fabs(b);
                if (std::fabs(a - b) > tol)
                {
                    printf("%s: point mismatch [track %d, elem %d]: pva=%e ref=%e (tol=%e)\n", testName, tr, k, a, b,
                           tol);
                    ok = false;
                    break;
                }
            }
        }

        float pvaCost = costView.At<float>(0);
        if (std::isnan(pvaCost) || std::isinf(pvaCost))
        {
            printf("%s: finalCost is NaN/Inf (pva=%e)\n", testName, pvaCost);
            ok = false;
        }
        else
        {
            /* Cost is shallow near convergence, so small parameter diffs give larger relative cost diffs. */
            float costTol = 1e-3f + 0.20f * std::fabs(refCost);
            if (std::fabs(pvaCost - refCost) > costTol)
            {
                printf("%s: cost mismatch: pva=%e ref=%e (tol=%e)\n", testName, pvaCost, refCost, costTol);
                ok = false;
            }
        }

        /* Only range-checked: the solvers may differ by one iteration at the lambda-adapt threshold. */
        int32_t pvaIters = itersView.At<int32_t>(0);
        if (pvaIters < 1 || pvaIters > params.maxIterations)
        {
            printf("%s: iterationsUsed out of range: %d (maxIterations=%d)\n", testName, pvaIters,
                   params.maxIterations);
            ok = false;
        }

        if (!ok)
        {
            printf("%s test failed\n", testName);
            return 1;
        }
    }
    return 0;

Error:
    printf("%s test failed (tensor export error %d)\n", testName, err);
    return err;
}

int RunOneCase(const Scene &scene, const char *caseName, bool useCudaStream)
{
    Tensors t;
    NVCVOperatorHandle operatorHandle = NULL;
    cudaStream_t cudaStream           = NULL;
    cupvaSyncObj_t sync               = NULL;
    cupvaStream_t cupvaStream         = NULL;
    int32_t err                       = 0;
    cudaError_t cudaErr               = cudaSuccess;
    char testName[128];
    PvaBundleAdjustmentConfig config;
    PvaBundleAdjustmentParams params = MakeParams();
    std::snprintf(testName, sizeof(testName), "%s (%s stream)", caseName, useCudaStream ? "CUDA" : "cuPVA");

    err = AllocateAndFillTensors(scene, &t);
    if (err)
        return err;

    config.activeCameraCount = kActiveCameraCount;
    config.activeCameraBegin = kActiveCameraBegin;

    NVCV_CHECK_ERROR_GOTO(pvaBundleAdjustmentCreate(&operatorHandle, &t.rigPosesReq, &t.points3dReq, &t.measurementsReq,
                                                    &t.cameraBitmaskReq, &t.rig2camReq, &t.rigPosesReq, &t.points3dReq,
                                                    &t.finalCostReq, &t.iterationsUsedReq, &config),
                          err, FreeResources);

    if (useCudaStream)
    {
        CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking), cudaErr, FreeResources);
        NVCV_CHECK_ERROR_GOTO(pvaBundleAdjustmentSubmit(operatorHandle, cudaStream, t.rigPoses, t.points3d,
                                                        t.measurements, t.cameraBitmask, t.rig2cam, t.updatedRigPoses,
                                                        t.updatedPoints3d, t.finalCost, t.iterationsUsed, &params),
                              err, FreeResources);
        CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(cudaStream), cudaErr, FreeResources);
    }
    else
    {
        CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                               FreeResources);

        cupvaFence_t fence;
        CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

        cupvaCmd_t rf;
        CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

        CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&cupvaStream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

        cupvaCmd_t const *cmds[1]  = {&rf};
        cupvaCmdStatus_t status[1] = {NULL};

        NVCV_CHECK_ERROR_GOTO(pvaBundleAdjustmentSubmit(operatorHandle, cupvaStream, t.rigPoses, t.points3d,
                                                        t.measurements, t.cameraBitmask, t.rig2cam, t.updatedRigPoses,
                                                        t.updatedPoints3d, t.finalCost, t.iterationsUsed, &params),
                              err, FreeResources);

        CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(cupvaStream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err,
                               FreeResources);

        bool waitSuccess;
        CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);
    }

    err = CompareOutputs(t, scene, params, testName);

FreeResources:
    if (cudaStream)
        cudaStreamDestroy(cudaStream);
    if (cupvaStream)
        CupvaStreamDestroy(cupvaStream);
    if (sync)
        CupvaSyncObjDestroy(sync);
    nvcvOperatorDestroy(operatorHandle);
    FreeTensors(&t);

    return err;
}

} // namespace

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    /* Needs a 4th VMEM bank, which PVA GEN2 (Orin) lacks. */
    if (cupva::GetHardwareInfo().pvaGen != cupva::PVA_GEN3)
    {
        printf("SKIPPED: bundle_adjustment sample requires PVA GEN3\n");
        return 0;
    }

    struct Case
    {
        const char *name;
        bool nearPlanePoint;
        float poseTranslationPerturb;
        float pointPerturb;
    };

    /* zero-motion: start at the optimum, must not diverge.
     * small-perturbation: convergence from a nearby start.
     * near-plane-point: track inside minDistance, must not produce NaN. */
    Case cases[] = {
        {"zero-motion", false, 0.0f, 0.0f},
        {"small-perturbation", false, 0.15f, 0.1f},
        {"near-plane-point", true, 0.15f, 0.1f},
    };

    int32_t overallResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    bool cudaAvailable = false;
    {
        int32_t err            = 0;
        cudaError_t cudaResult = cudaSuccess;
        uint32_t driverVersion = 0;
        CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaResult, SkipCuda);

        CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), err, SkipCuda);
        if (driverVersion >= 2007)
        {
            cudaAvailable = true;
        }
    SkipCuda:
        (void)err;
        (void)cudaResult;
    }
#endif

    for (const Case &c : cases)
    {
        Scene scene = BuildScene(c.nearPlanePoint, c.poseTranslationPerturb, c.pointPerturb);

        int32_t cupvaResult = RunOneCase(scene, c.name, /*useCudaStream=*/false);
        if (cupvaResult == 0)
        {
            printf("%s (cuPVA stream): PASSED\n", c.name);
        }
        else
        {
            printf("%s (cuPVA stream): FAILED\n", c.name);
            overallResult = 1;
        }

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
        if (cudaAvailable)
        {
            int32_t cudaResult = RunOneCase(scene, c.name, /*useCudaStream=*/true);
            if (cudaResult == 0)
            {
                printf("%s (CUDA stream): PASSED\n", c.name);
            }
            else
            {
                printf("%s (CUDA stream): FAILED\n", c.name);
                overallResult = 1;
            }
        }
#endif
    }

    if (overallResult)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

    return overallResult;
}
