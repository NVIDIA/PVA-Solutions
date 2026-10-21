/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file capon_roi_preprocess.c
 * @brief VPU-vs-reference verify for the Capon ROI preprocessing (projection-core) operator.
 *
 * Stages the operator inputs (twiddle tables, per-detection detected Doppler, element snapshot,
 * range-Doppler plane, and steering LUT), submits the VPU operator, and compares the device
 * snapshotsROI / steeringVectorsROI against the double-precision C reference in
 * ref/capon_roi_preprocess_ref.{h,cpp}:
 *   - the device steering LUT is built with caponROIPreprocessSteeringLut() (no duplicated steering
 *     math here), and
 *   - the golden snapshots/steering come from caponROIPreprocess(), fed raw-array inputs
 *     reconstructed from the same host buffers.
 *
 * A non-degenerate 4(az) x 2(el) receive layout (numElVirt = 2) keeps the coarse-DoA elevation ROI
 * unambiguous so the device and the reference select the same ROI (coarse DoA and ROI selection,
 * stages 5-6). Snapshots (stages 7-9) match to a few LSB (fixed-point device projection vs double
 * reference) and steering (stage 10) is a verbatim LUT gather that must match exactly.
 *
 * The operator is submitted once to a CUDA stream (when the platform and driver support it) and once to
 * a cuPVA stream; each run's outputs are checked against the same reference.
 */

#include "capon_roi_preprocess_ref.h"

#include <ErrorCheckMacros.h>
#include <OpCaponROIPreprocess.h>
#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <cupva_host_nonsafety.h>
#include <math.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double kPi = 3.14159265358979323846;

/* Host pointer of a PVA tensor, or NULL (with a message) if the tensor cannot be exported or mapped. */
static void *hostPtr(NVCVTensorHandle t)
{
    NVCVTensorData d;
    memset(&d, 0, sizeof(d));
    const NVCVStatus st = nvcvTensorExportData(t, &d);
    if (st != NVCV_SUCCESS)
    {
        printf("Capon ROI preprocess: tensor export failed (%d)\n", (int)st);
        return NULL;
    }
    void *p                 = NULL;
    const cupvaError_t cerr = CupvaMemGetHostPointer(&p, (void *)d.buffer.strided.basePtr);
    if (cerr != CUPVA_ERROR_NONE || p == NULL)
    {
        printf("Capon ROI preprocess: host mapping failed (%d)\n", (int)cerr);
        return NULL;
    }
    return p;
}

/* Construct one PVA tensor and its requirements; returns 0 on success. */
static int makeTensor(NVCVAllocatorHandle alloc, int32_t rank, int64_t *shp, NVCVDataType dt, NVCVTensorLayout layout,
                      NVCVTensorRequirements *r, NVCVTensorHandle *t)
{
    NVCVStatus st = nvcvTensorCalcRequirementsPva(rank, shp, dt, layout, 0, 0, r);
    if (st == NVCV_SUCCESS)
        st = nvcvTensorConstruct(r, alloc, t);
    if (st != NVCV_SUCCESS)
    {
        printf("Capon ROI preprocess: tensor construction failed (%d)\n", (int)st);
        return 1;
    }
    return 0;
}

/* The operator's tensors, as passed to every Submit, and the host views of its outputs. */
typedef struct CaponTensors
{
    NVCVTensorHandle count, detDop, elem, plane, lut, snap, sv;
    int32_t *snapHost, *svHost;
} CaponTensors;

/* Fill the outputs with a sentinel so a run that does not write them cannot pass on stale data. */
static void ClearOutputs(const CaponTensors *t, size_t snapInts, size_t svInts)
{
    for (size_t i = 0; i < snapInts; ++i) t->snapHost[i] = 0x5A5A5A5A;
    for (size_t i = 0; i < svInts; ++i) t->svHost[i] = 0x5A5A5A5A;
}

/* Compare the device outputs with the reference; returns 0 on a match. */
static int CompareWithReference(const char *path, const CaponTensors *t, const int32_t *refSnap, const int32_t *refSv,
                                size_t snapInts, size_t svInts, int maxDet)
{
    const int32_t *devOut = t->snapHost;
    const int32_t *devSv  = t->svHost;
    const long tol        = 4; /* a few LSB: fixed-point (device) vs double (ref) projection slack */
    long maxDiff = 0, nMismatch = 0;
    long svMaxDiff = 0, svMismatch = 0;
    for (size_t i = 0; i < snapInts; ++i)
    {
        long df = labs((long)devOut[i] - (long)refSnap[i]);
        if (df > maxDiff)
            maxDiff = df;
        if (df > tol)
            ++nMismatch;
    }
    for (size_t i = 0; i < svInts; ++i)
    {
        long df = labs((long)devSv[i] - (long)refSv[i]);
        if (df > svMaxDiff)
            svMaxDiff = df;
        if (df > 0)
            ++svMismatch;
    }
    printf(
        "Capon ROI preprocess VPU-vs-ref (%s): snapshots maxDiff=%ld mismatches(>%ld)=%ld; steering maxDiff=%ld "
        "mismatches=%ld; over %d dets\n",
        path, maxDiff, tol, nMismatch, svMaxDiff, svMismatch, maxDet);
    /* Steering is a verbatim gather of the precomputed LUT block, so it must match exactly. */
    return (nMismatch == 0 && svMismatch == 0) ? 0 : 1;
}

/* Submit to a cuPVA stream and wait for it; the stream and sync object are released on every path.
 * Returns 0 on success. *inFlight is set when submitted work could not be waited for, so the caller
 * must not release the tensors or the operator it uses. */
static int SubmitToCupvaStream(NVCVOperatorHandle op, const CaponTensors *t, bool *inFlight)
{
    int err              = 0;
    cupvaStream_t stream = NULL;
    cupvaSyncObj_t sync  = NULL;
    cupvaFence_t fence;
    cupvaCmd_t rf;
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};
    bool ok                    = false;

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err, CLEANUP);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, CLEANUP);
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, CLEANUP);
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, CLEANUP);

    NVCV_CHECK_ERROR_GOTO(pvaCaponROIPreprocessSubmit(op, stream, NULL, NULL, t->count, t->detDop, t->elem, t->plane,
                                                      t->lut, t->snap, t->sv),
                          err, CLEANUP);
    /* The operator's work is queued from here until the fence wait succeeds. */
    *inFlight = true;
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, CLEANUP);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &ok), err, CLEANUP);
    *inFlight = false;
    /* Without this the caller would go on to compare untouched output buffers, reporting a data
     * mismatch against every element instead of the execution failure that actually happened. */
    if (!ok)
    {
        printf("Capon ROI preprocess: PVA execution did not complete successfully\n");
        err = 1;
    }

CLEANUP:
    if (*inFlight)
    {
        /* The fence was not submitted or not waited for: the work may still be running, so keep the
         * stream as well as the resources it uses. */
        printf("Capon ROI preprocess: submitted work could not be waited for\n");
        return 1;
    }
    if (stream && CupvaStreamDestroy(stream) != CUPVA_ERROR_NONE)
        err = err ? err : 1;
    if (sync && CupvaSyncObjDestroy(sync) != CUPVA_ERROR_NONE)
        err = err ? err : 1;
    return err != 0 ? 1 : 0;
}

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
/* Submit to a CUDA stream and synchronize it. Returns 0 on success; *inFlight is set, and the stream
 * kept, when submitted work could not be synchronized (see SubmitToCupvaStream). */
static int SubmitToCudaStream(NVCVOperatorHandle op, const CaponTensors *t, bool *inFlight)
{
    int err             = 0;
    cudaError_t cudaErr = cudaSuccess;
    cudaStream_t stream = NULL;

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaErr, CLEANUP);
    NVCV_CHECK_ERROR_GOTO(pvaCaponROIPreprocessSubmit(op, stream, NULL, NULL, t->count, t->detDop, t->elem, t->plane,
                                                      t->lut, t->snap, t->sv),
                          err, CLEANUP);
    *inFlight = true;
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), cudaErr, CLEANUP);
    *inFlight = false;

CLEANUP:
    if (*inFlight)
    {
        printf("Capon ROI preprocess: submitted work could not be synchronized\n");
        return 1;
    }
    if (stream && cudaStreamDestroy(stream) != cudaSuccess)
        err = err ? err : 1;
    return (err != 0 || cudaErr != cudaSuccess) ? 1 : 0;
}
#endif

int main(void)
{
    int err = 0;

    /* CUDA stream submission needs a CUDA device and cuPVA driver 2007 or newer. Select the device before
     * any PVA allocation or operator creation. */
#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    bool useCudaStream = false;
    {
        uint32_t driverVersion = 0;
        if (cudaSetDevice(0) != cudaSuccess || CupvaGetDriverVersion(&driverVersion) != CUPVA_ERROR_NONE)
        {
            printf("Capon ROI preprocess: CUDA device or cuPVA driver query failed\n");
            return 1;
        }
        useCudaStream = (driverVersion >= 2007);
    }
#endif

    /* Config: 8x8 with a 4(az) x 2(el) receive layout, i.e. a non-degenerate virtual array with real
     * elevation resolution (numElVirt = 2). This keeps the coarse-DoA elevation ROI unambiguous so the
     * device and the double-precision reference select the same ROI, letting both snapshots AND steering
     * be compared against the shared caponROIPreprocess reference. */
    const int numAzElements = 4, numElElements = 2; /* receive layout: numRx = 4 x 2 */
    PVACaponROIPreprocessParams gp;
    memset(&gp, 0, sizeof(gp));
    gp.numTx                = 8;
    gp.numRx                = 8;
    gp.numChirps            = 64;
    gp.numAzVirt            = numAzElements * gp.numTx;                                    /* 32 */
    gp.numElVirt            = numElElements;                                               /* 2 */
    gp.beamAz               = gp.numAzVirt;                                                /* 32 */
    gp.beamEl               = (gp.numElVirt > gp.numAzVirt) ? gp.numElVirt : gp.numAzVirt; /* 32 */
    gp.roiAz                = 4;
    gp.roiEl                = 4;
    gp.svDim                = gp.roiAz * gp.roiEl; /* 16 */
    gp.numAzBins            = 16;
    gp.numElBins            = 8;
    gp.numDopplerBinsCovMat = 64; /* H */
    gp.dopplerNeighborhood  = 33;
    gp.enableFBA            = true;
    gp.inputQBits           = 12;
    gp.steeringQBits        = 12;
    gp.maxDet               = 4;

    const int numAzVirt = gp.numAzVirt, numElVirt = gp.numElVirt, beamAz = gp.beamAz, beamEl = gp.beamEl;
    const int roiAz = gp.roiAz, roiEl = gp.roiEl, svDim = gp.svDim, H = gp.numDopplerBinsCovMat, maxDet = gp.maxDet;
    const int numTx = gp.numTx, numRx = gp.numRx, numChirps = gp.numChirps;
    const int nvc = numAzVirt * numElVirt;

    /* DDM decode terms (ddm[tx] = tx/numTx): relDopplerOffset + ddmaWeight = exp(+j2pi ddm[tx]). */
    for (int tx = 0; tx < numTx; ++tx)
    {
        double ddm              = (double)tx / numTx;
        gp.relDopplerOffset[tx] = (int)lround((0.0 - ddm) * numChirps);
        gp.ddmaWeightRe[tx]     = (float)cos(2.0 * kPi * ddm);
        gp.ddmaWeightIm[tx]     = (float)sin(2.0 * kPi * ddm);
    }

    NVCVAllocatorHandle alloc = NULL;
    if (nvcvAllocatorConstructPva(&alloc) != NVCV_SUCCESS)
    {
        printf("Capon ROI preprocess: PVA allocator construction failed\n");
        return 1;
    }

    /* Tensor shapes (HW / NHW / NCHW), complex packed as interleaved 2-channel types. */
    NVCVTensorLayout W, HW, NHW, NCHW;
    nvcvTensorLayoutMake("W", &W);
    nvcvTensorLayoutMake("HW", &HW);
    nvcvTensorLayoutMake("NHW", &NHW);
    nvcvTensorLayoutMake("NCHW", &NCHW);

    const int numAzBins = gp.numAzBins, numElBins = gp.numElBins;
    const int numRoiAzPos = beamAz - roiAz + 1, numRoiElPos = beamEl - roiEl + 1;
    const int numRoiPos   = numRoiAzPos * numRoiElPos;
    const int svBlockCplx = svDim * numElBins * numAzBins; /* complex entries per steering block */

    NVCVTensorHandle countT = NULL, detDopT = NULL, elemT = NULL, planeT = NULL, snapT = NULL, lutT = NULL, svT = NULL;
    NVCVTensorRequirements countR, detDopR, elemR, planeR, snapR, lutR, svR;
    /* Declared before any `goto CLEANUP` so cleanup never reads them uninitialized. */
    NVCVOperatorHandle op = NULL;
    float *snapRaw        = NULL;
    int32_t *detList = NULL, *refSnap = NULL, *refSv = NULL;
    bool inFlight = false; /* submitted work that could not be waited for: keep its resources */
    int64_t shp1[1], shp2[2], shp3[3], shp4[4];

    /* The operator generates the DFT twiddle tables itself (gp.useCustomTwiddles is false), so waz/wel are
     * NULL here and at Submit. */
    shp1[0] = 1;
    err |= makeTensor(alloc, 1, shp1, NVCV_DATA_TYPE_S32, W, &countR, &countT);
    shp2[0] = 1;
    shp2[1] = maxDet;
    err |= makeTensor(alloc, 2, shp2, NVCV_DATA_TYPE_S32, HW, &detDopR, &detDopT);
    shp2[0] = maxDet;
    shp2[1] = nvc;
    err |= makeTensor(alloc, 2, shp2, NVCV_DATA_TYPE_2F32, HW, &elemR, &elemT);
    shp2[1] = (int64_t)numChirps * numRx;
    err |= makeTensor(alloc, 2, shp2, NVCV_DATA_TYPE_2F32, HW, &planeR, &planeT);

    shp3[0] = maxDet;
    shp3[1] = H;
    shp3[2] = svDim;
    err |= makeTensor(alloc, 3, shp3, NVCV_DATA_TYPE_2S32, NHW, &snapR, &snapT);

    /* Steering LUT (input) holds one quantized block per ROI start position; the device gathers the
     * block for each detection's computed ROI. steeringVectorsROI (output) receives the gather. */
    shp2[0] = numRoiPos;
    shp2[1] = svBlockCplx;
    err |= makeTensor(alloc, 2, shp2, NVCV_DATA_TYPE_2S32, HW, &lutR, &lutT);
    shp4[0] = maxDet;
    shp4[1] = svDim;
    shp4[2] = numElBins;
    shp4[3] = numAzBins;
    err |= makeTensor(alloc, 4, shp4, NVCV_DATA_TYPE_2S32, NCHW, &svR, &svT);
    if (err)
        goto CLEANUP;

    float *elem      = (float *)hostPtr(elemT);
    int32_t *count   = (int32_t *)hostPtr(countT);
    int32_t *detDop  = (int32_t *)hostPtr(detDopT);
    float *plane     = (float *)hostPtr(planeT);
    int32_t *lut     = (int32_t *)hostPtr(lutT);
    int32_t *snapOut = (int32_t *)hostPtr(snapT);
    int32_t *svOut   = (int32_t *)hostPtr(svT);
    if (!elem || !count || !detDop || !plane || !lut || !snapOut || !svOut)
    {
        err = 1;
        goto CLEANUP;
    }

    /* Per-detection element snapshot: a distinct 2D beamspace tone per detection (sharp, unique coarse
     * peak in both azimuth and elevation). Element index i maps to virtual (a, e): i = e*numAzVirt + a. */
    for (int d = 0; d < maxDet; ++d)
    {
        int peakAzBin = 6 + d * 5; /* distinct az/el beamspace peak bins per detection, within [0, beam) */
        int peakElBin = 6 + d * 4;
        for (int i = 0; i < nvc; ++i)
        {
            int a                       = i % numAzVirt;
            int e                       = i / numAzVirt;
            double ph                   = (double)peakAzBin * a / beamAz + (double)peakElBin * e / beamEl;
            elem[(d * nvc + i) * 2 + 0] = (float)cos(-2.0 * kPi * ph);
            elem[(d * nvc + i) * 2 + 1] = (float)sin(-2.0 * kPi * ph);
        }
    }

    /* Number of detections to process: every row here. A pipeline passes its CFAR detection count. */
    *count = maxDet;

    /* Per-detection detected Doppler bin. */
    for (int d = 0; d < maxDet; ++d) detDop[d] = (20 + d * 3) % numChirps;

    /* Per-detection range-Doppler plane [numChirps, numRx] (deterministic pseudo-signal). */
    for (int d = 0; d < maxDet; ++d)
        for (int ch = 0; ch < numChirps; ++ch)
            for (int rx = 0; rx < numRx; ++rx)
            {
                plane[((d * numChirps + ch) * numRx + rx) * 2 + 0] = cosf(0.09f * (d + 1) * ch + 0.05f * rx);
                plane[((d * numChirps + ch) * numRx + rx) * 2 + 1] = sinf(0.06f * (d + 1) * ch - 0.04f * rx);
            }

    /* Shared reference config, mirroring the operator params gp. Used both to build the device's
     * steering LUT and to compute the golden snapshots/steering. */
    CaponROIPreprocConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.numTx                = numTx;
    cfg.numRx                = numRx;
    cfg.numAzimuthElements   = numAzElements;
    cfg.numElevationElements = numElElements;
    cfg.numChirps            = numChirps;
    cfg.roiAzSize            = roiAz;
    cfg.roiElSize            = roiEl;
    cfg.numAzBins            = numAzBins;
    cfg.numElBins            = numElBins;
    cfg.enableFBA            = gp.enableFBA ? 1 : 0;
    cfg.dopplerNeighborhood  = gp.dopplerNeighborhood;
    cfg.numDopplerBinsCovMat = H;
    cfg.txSpacing            = 0.5;
    cfg.rxAzSpacing          = 4.0;
    cfg.rxElSpacing          = 0.5;
    cfg.inputQBits           = gp.inputQBits;
    cfg.steeringQBits        = gp.steeringQBits;
    for (int tx = 0; tx < numTx; ++tx)
    {
        cfg.relDopplerOffset[tx] = gp.relDopplerOffset[tx];
        cfg.ddmaWeightRe[tx]     = gp.ddmaWeightRe[tx];
        cfg.ddmaWeightIm[tx]     = gp.ddmaWeightIm[tx];
    }

    /* Build the device's steering LUT via the shared reference (no duplicated steering math here). */
    {
        int rc = caponROIPreprocessSteeringLut(&cfg, lut);
        if (rc != 0)
        {
            printf("Capon ROI preprocess steering LUT build failed (%d)\n", rc);
            err = 1;
            goto CLEANUP;
        }
    }

    NVCV_CHECK_ERROR_GOTO(
        pvaCaponROIPreprocessCreate(&op, NULL, NULL, &countR, &detDopR, &elemR, &planeR, &lutR, &snapR, &svR, &gp), err,
        CLEANUP);

    /* Host reference (shared caponROIPreprocess), computed once and compared with every submission.
     * Its inputs are reconstructed from the same host buffers fed to the VPU:
     *  - snapshots CHW [maxDet, numTx, numRx]: permute the element snapshot so vc0 = rx*numTx+tx
     *    (the VPU element index i maps to tx = i % numTx, rx = i / numTx).
     *  - rangeDopplerCube [maxDet, numChirps, numRx]: identical interleaved layout to the plane
     *    tensor, one range plane per detection.
     *  - detectionList [maxDet, 2] = (rangeIdx = det, dopplerIdx). */
    const size_t snapInts = (size_t)maxDet * H * svDim * 2;
    const size_t svInts   = (size_t)maxDet * svBlockCplx * 2;
    snapRaw               = (float *)malloc((size_t)maxDet * numTx * numRx * 2 * sizeof(float));
    detList               = (int32_t *)malloc((size_t)maxDet * 2 * sizeof(int32_t));
    refSnap               = (int32_t *)calloc(snapInts, sizeof(int32_t));
    refSv                 = (int32_t *)calloc(svInts, sizeof(int32_t));
    if (!snapRaw || !detList || !refSnap || !refSv)
    {
        printf("Capon ROI preprocess: host allocation failed\n");
        err = 1;
        goto CLEANUP;
    }
    for (int d = 0; d < maxDet; ++d)
        for (int i = 0; i < nvc; ++i)
        {
            int tx           = i % numTx;
            int rx           = i / numTx;
            size_t dst       = ((size_t)(d * numTx + tx) * numRx + rx) * 2;
            size_t src       = ((size_t)d * nvc + i) * 2;
            snapRaw[dst]     = elem[src];
            snapRaw[dst + 1] = elem[src + 1];
        }
    for (int d = 0; d < maxDet; ++d)
    {
        detList[d * 2 + 0] = d;
        detList[d * 2 + 1] = detDop[d];
    }
    {
        CaponROIPreprocInput in;
        memset(&in, 0, sizeof(in));
        in.numDetections    = maxDet;
        in.snapshots        = snapRaw;
        in.rangeDopplerCube = plane; /* [maxDet, numChirps, numRx], same interleaved layout */
        in.numRangeBins     = maxDet;
        in.detectionList    = detList;

        CaponROIPreprocOutput refOut;
        memset(&refOut, 0, sizeof(refOut));
        refOut.snapshotsROI       = refSnap;
        refOut.steeringVectorsROI = refSv;

        int prc = caponROIPreprocess(&cfg, &in, &refOut);
        if (prc != 0)
        {
            printf("Capon ROI preprocess reference failed (%d)\n", prc);
            err = 1;
            goto CLEANUP;
        }
    }

    {
        const CaponTensors tensors = {countT, detDopT, elemT, planeT, lutT, snapT, svT, snapOut, svOut};
        int cudaResult             = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
        if (useCudaStream)
        {
            ClearOutputs(&tensors, snapInts, svInts);
            cudaResult = SubmitToCudaStream(op, &tensors, &inFlight);
            if (cudaResult == 0)
                cudaResult = CompareWithReference("CUDA stream", &tensors, refSnap, refSv, snapInts, svInts, maxDet);
        }
#endif

        int cupvaResult = 1;
        if (!inFlight)
        {
            ClearOutputs(&tensors, snapInts, svInts);
            cupvaResult = SubmitToCupvaStream(op, &tensors, &inFlight);
        }
        if (cupvaResult == 0)
            cupvaResult = CompareWithReference("cuPVA stream", &tensors, refSnap, refSv, snapInts, svInts, maxDet);

        err = (cudaResult || cupvaResult) ? 1 : 0;
    }

CLEANUP:
    free(snapRaw);
    free(detList);
    free(refSnap);
    free(refSv);
    if (inFlight)
    {
        /* Work that may still be running uses the operator and the tensors: leave them to process exit. */
        printf("CaponROIPreprocessTest: FAILED\n");
        return 1;
    }
    if (op)
        nvcvOperatorDestroy(op);
    {
        NVCVTensorHandle all[] = {countT, detDopT, elemT, planeT, snapT, lutT, svT};
        for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i)
            if (all[i] && nvcvTensorDecRef(all[i], NULL) != NVCV_SUCCESS)
                err = 1;
    }
    if (alloc && nvcvAllocatorDecRef(alloc, NULL) != NVCV_SUCCESS)
        err = 1;
    printf(err ? "CaponROIPreprocessTest: FAILED\n" : "CaponROIPreprocessTest: PASSED\n");
    return err;
}
