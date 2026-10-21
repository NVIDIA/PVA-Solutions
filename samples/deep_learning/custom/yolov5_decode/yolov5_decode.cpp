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
 * @file yolov5_decode.cpp
 * @brief Sample demonstrating the YOLOv5 decode operator on the three scales of a YOLOv5 model.
 *
 * One operator is created per scale (stride 8, 16 and 32). All three are submitted to the same boxes, scores,
 * classes and count tensors: the count tensor is an append cursor, so the detections of all scales are collected
 * in one list. A few detections are placed in otherwise empty predictions and the result is compared with a CPU
 * reference, for CHW and HWC predictions and for cuPVA and CUDA stream submission.
 */

#include "ErrorCheckMacros.h"
#include "ref/yolov5_decode_ref.h"

#include <OpYoloV5Decode.h>
#include <PvaAllocator.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define NUM_SCALES 3

static const int32_t kImageSize      = 672; /* Square network input, in pixels. */
static const int32_t kNumClasses     = 80;
static const int32_t kAnchorsPerCell = 3;
static const int32_t kItemSize       = kNumClasses + 5; /* tx, ty, tw, th, obj, one logit per class */
static const int32_t kNumChannels    = kAnchorsPerCell * kItemSize;
static const int32_t kMaxOutputBoxes = 5120;
static const float kConfThreshold    = 0.25f;

static const int32_t kStrides[NUM_SCALES] = {8, 16, 32};

/* YOLOv5 default anchor sizes in pixels, [scale][anchor][width, height]. */
static const float kAnchorSizes[NUM_SCALES][3][2] = {{{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}},
                                                     {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}},
                                                     {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}};

/* Detections placed in the predictions: scale, grid cell (x, y), anchor and class. */
struct SampleDetection
{
    int32_t scale, x, y, anchor, classId;
};

static const SampleDetection kDetections[] = {
    {0, 10, 20, 0, 0},
    {0, 40, 50, 2, 2},
    {1, 20, 10, 1, 15},
    {2, 5, 5, 0, 79},
};

/* Tensors shared by the operators of all scales. */
struct SampleTensors
{
    NVCVAllocatorHandle allocator;
    NVCVTensorRequirements predReqs[NUM_SCALES];
    NVCVTensorRequirements anchorReqs[NUM_SCALES];
    NVCVTensorRequirements boxesReqs, scoresReqs, classesReqs, countReqs;
    NVCVTensorHandle predictions[NUM_SCALES];
    NVCVTensorHandle anchors[NUM_SCALES];
    NVCVTensorHandle boxes, scores, classes, count;
};

static const char *LayoutName(PVAYoloV5DecodeInputLayout layout)
{
    return layout == PVA_YOLOV5_DECODE_INPUT_HWC ? "HWC" : "CHW";
}

static int32_t GridSize(int32_t scale)
{
    return kImageSize / kStrides[scale];
}

/**
 * @brief Index of value k of an anchor of a grid cell in the prediction tensor.
 */
static int64_t PredictionIndex(PVAYoloV5DecodeInputLayout layout, int32_t numCells, int32_t cell, int32_t anchor,
                               int32_t k)
{
    const int64_t channel = (int64_t)anchor * kItemSize + k;
    return layout == PVA_YOLOV5_DECODE_INPUT_HWC ? (int64_t)cell * kNumChannels + channel : channel * numCells + cell;
}

/**
 * @brief Map a tensor into host memory.
 */
static int32_t GetHostPointer(NVCVTensorHandle tensor, void **hostPtr)
{
    int32_t err = 0;
    NVCVTensorData tensorData;

    NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensor, &tensorData), err, Error);
    CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer(hostPtr, (void *)tensorData.buffer.strided.basePtr), err, Error);

Error:
    return err;
}

/**
 * @brief Allocate the input and output tensors of all scales.
 */
static int32_t AllocateTensors(PVAYoloV5DecodeInputLayout layout, SampleTensors *tensors)
{
    int32_t err = 0;
    NVCVTensorLayout predLayout;
    NVCVTensorLayout layout2d;
    NVCVTensorLayout layout1d;
    int64_t boxesShape[2]  = {4, kMaxOutputBoxes};
    int64_t outputShape[1] = {kMaxOutputBoxes};
    int64_t countShape[1]  = {1};

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake(LayoutName(layout), &predLayout), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HW", &layout2d), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &layout1d), err, Error);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&tensors->allocator), err, Error);

    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        const int64_t grid     = GridSize(sc);
        int64_t chwShape[3]    = {kNumChannels, grid, grid};
        int64_t hwcShape[3]    = {grid, grid, kNumChannels};
        int64_t anchorShape[2] = {grid * grid * kAnchorsPerCell, 5};

        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(3, layout == PVA_YOLOV5_DECODE_INPUT_HWC ? hwcShape : chwShape,
                                          NVCV_DATA_TYPE_F32, predLayout, 0, 0, &tensors->predReqs[sc]),
            err, Error);
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorCalcRequirementsPva(2, anchorShape, NVCV_DATA_TYPE_F32, layout2d, 0, 0, &tensors->anchorReqs[sc]),
            err, Error);
        NVCV_CHECK_ERROR_GOTO(
            nvcvTensorConstruct(&tensors->predReqs[sc], tensors->allocator, &tensors->predictions[sc]), err, Error);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->anchorReqs[sc], tensors->allocator, &tensors->anchors[sc]),
                              err, Error);
    }

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(2, boxesShape, NVCV_DATA_TYPE_F32, layout2d, 0, 0, &tensors->boxesReqs), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, outputShape, NVCV_DATA_TYPE_F16, layout1d, 0, 0, &tensors->scoresReqs), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, outputShape, NVCV_DATA_TYPE_S16, layout1d, 0, 0, &tensors->classesReqs), err,
        Error);
    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(1, countShape, NVCV_DATA_TYPE_S32, layout1d, 0, 0, &tensors->countReqs), err,
        Error);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->boxesReqs, tensors->allocator, &tensors->boxes), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->scoresReqs, tensors->allocator, &tensors->scores), err, Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->classesReqs, tensors->allocator, &tensors->classes), err,
                          Error);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&tensors->countReqs, tensors->allocator, &tensors->count), err, Error);

Error:
    return err;
}

/**
 * @brief Fill the predictions and anchors of all scales and reset the count tensor.
 *
 * All predictions are set to a large negative logit, so that nothing is detected, except for the values of the
 * detections in kDetections. The anchors tensor holds [grid_x, grid_y, anchor_w, anchor_h, stride] for every anchor
 * row (y * W + x) * 3 + a.
 */
static int32_t InitializeInputs(PVAYoloV5DecodeInputLayout layout, SampleTensors *tensors)
{
    int32_t err      = 0;
    void *hostPtr    = NULL;
    float *pred      = NULL;
    float *anchors   = NULL;
    int32_t numCells = 0;

    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        const int32_t grid = GridSize(sc);
        numCells           = grid * grid;

        err = GetHostPointer(tensors->predictions[sc], &hostPtr);
        if (err)
            goto Error;
        pred = (float *)hostPtr;
        for (int64_t i = 0; i < (int64_t)kNumChannels * numCells; i++)
        {
            pred[i] = -10.0f;
        }

        err = GetHostPointer(tensors->anchors[sc], &hostPtr);
        if (err)
            goto Error;
        anchors = (float *)hostPtr;
        for (int32_t y = 0; y < grid; y++)
        {
            for (int32_t x = 0; x < grid; x++)
            {
                for (int32_t a = 0; a < kAnchorsPerCell; a++)
                {
                    float *row = anchors + ((int64_t)(y * grid + x) * kAnchorsPerCell + a) * 5;
                    row[0]     = (float)x;
                    row[1]     = (float)y;
                    row[2]     = kAnchorSizes[sc][a][0];
                    row[3]     = kAnchorSizes[sc][a][1];
                    row[4]     = (float)kStrides[sc];
                }
            }
        }
    }

    /* Box logits of 0 center the box in its grid cell with the anchor size; objectness 0.73, class score 0.88. */
    for (size_t d = 0; d < sizeof(kDetections) / sizeof(kDetections[0]); d++)
    {
        const SampleDetection *det = &kDetections[d];
        const int32_t grid         = GridSize(det->scale);
        const int32_t cell         = det->y * grid + det->x;
        numCells                   = grid * grid;

        err = GetHostPointer(tensors->predictions[det->scale], &hostPtr);
        if (err)
            goto Error;
        pred = (float *)hostPtr;
        for (int32_t k = 0; k < 4; k++)
        {
            pred[PredictionIndex(layout, numCells, cell, det->anchor, k)] = 0.0f;
        }
        pred[PredictionIndex(layout, numCells, cell, det->anchor, 4)]                = 1.0f;
        pred[PredictionIndex(layout, numCells, cell, det->anchor, 5 + det->classId)] = 2.0f;
    }

    /* The count tensor is the append cursor: it must be zero before the first scale is submitted. */
    err = GetHostPointer(tensors->count, &hostPtr);
    if (err)
        goto Error;
    *(int32_t *)hostPtr = 0;

Error:
    return err;
}

/**
 * @brief Create one operator per scale. All operators share the output tensors.
 */
static int32_t CreateOperators(PVAYoloV5DecodeInputLayout layout, SampleTensors *tensors,
                               NVCVOperatorHandle operatorHandles[NUM_SCALES])
{
    int32_t err = 0;

    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        NVCV_CHECK_ERROR_GOTO(
            pvaYoloV5DecodeCreate(&operatorHandles[sc], &tensors->predReqs[sc], &tensors->anchorReqs[sc],
                                  &tensors->boxesReqs, &tensors->scoresReqs, &tensors->classesReqs, &tensors->countReqs,
                                  kNumClasses, kConfThreshold, kImageSize, kImageSize, layout),
            err, Error);
    }

Error:
    return err;
}

/**
 * @brief Allocate and initialize the tensors and create the operators.
 */
static int32_t SetUp(PVAYoloV5DecodeInputLayout layout, SampleTensors *tensors,
                     NVCVOperatorHandle operatorHandles[NUM_SCALES])
{
    int32_t err = AllocateTensors(layout, tensors);
    if (err == 0)
    {
        err = InitializeInputs(layout, tensors);
    }
    if (err == 0)
    {
        err = CreateOperators(layout, tensors, operatorHandles);
    }
    return err;
}

/**
 * @brief Compare the decoded detections of all scales with the CPU reference.
 */
static int32_t ValidateOutput(PVAYoloV5DecodeInputLayout layout, SampleTensors *tensors)
{
    const float boxTolerance   = 1.0f;
    const float scoreTolerance = 1e-3f;

    int32_t err      = 0;
    int32_t numDets  = 0;
    int64_t boxPitch = 0;
    void *hostPtr    = NULL;
    const float *boxes;
    const __half *scores;
    const int16_t *classes;
    NVCVTensorData boxesData;
    std::vector<YoloV5RefDetection> refDets;
    std::vector<float> anchorMajor;

    /* CPU reference. It reads anchor-major predictions, so CHW predictions are transposed first. */
    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        const int32_t numCells   = GridSize(sc) * GridSize(sc);
        const int32_t numAnchors = numCells * kAnchorsPerCell;
        const float *pred;
        const float *anchors;

        err = GetHostPointer(tensors->predictions[sc], &hostPtr);
        if (err)
            goto Error;
        pred = (const float *)hostPtr;
        err  = GetHostPointer(tensors->anchors[sc], &hostPtr);
        if (err)
            goto Error;
        anchors = (const float *)hostPtr;

        if (layout == PVA_YOLOV5_DECODE_INPUT_CHW)
        {
            anchorMajor.resize((size_t)numAnchors * kItemSize);
            yolov5ChwToFlatRef(pred, anchorMajor.data(), numCells, kNumClasses);
            pred = anchorMajor.data();
        }

        std::vector<YoloV5RefDetection> scaleDets =
            yolov5DecodeRef(pred, anchors, numAnchors, kNumClasses, kConfThreshold, (float)kImageSize,
                            (float)kImageSize, kMaxOutputBoxes - (int)refDets.size());
        refDets.insert(refDets.end(), scaleDets.begin(), scaleDets.end());
    }

    NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors->boxes, &boxesData), err, Error);
    boxPitch = boxesData.buffer.strided.strides[0] / (int64_t)sizeof(float);

    err = GetHostPointer(tensors->boxes, &hostPtr);
    if (err)
        goto Error;
    boxes = (const float *)hostPtr;
    err   = GetHostPointer(tensors->scores, &hostPtr);
    if (err)
        goto Error;
    scores = (const __half *)hostPtr;
    err    = GetHostPointer(tensors->classes, &hostPtr);
    if (err)
        goto Error;
    classes = (const int16_t *)hostPtr;
    err     = GetHostPointer(tensors->count, &hostPtr);
    if (err)
        goto Error;
    numDets = *(const int32_t *)hostPtr;

    printf("  Detections: PVA %d, reference %d\n", numDets, (int)refDets.size());
    if (numDets != (int32_t)refDets.size())
    {
        err = 1;
    }

    for (int32_t i = 0; i < numDets && i < (int32_t)refDets.size(); i++)
    {
        const YoloV5RefDetection &ref = refDets[i];
        const float left              = boxes[i];
        const float top               = boxes[boxPitch + i];
        const float right             = boxes[2 * boxPitch + i];
        const float bottom            = boxes[3 * boxPitch + i];
        const float score             = __half2float(scores[i]);
        const bool match = std::fabs(left - ref.left) <= boxTolerance && std::fabs(top - ref.top) <= boxTolerance &&
                           std::fabs(right - ref.right) <= boxTolerance &&
                           std::fabs(bottom - ref.bottom) <= boxTolerance &&
                           std::fabs(score - ref.confidence) <= scoreTolerance && classes[i] == ref.classId;

        printf("  [%d] box=(%.1f, %.1f, %.1f, %.1f) score=%.3f class=%d%s\n", i, left, top, right, bottom, score,
               (int)classes[i], match ? "" : "  <- differs from the reference");
        if (!match)
        {
            printf("       reference box=(%.1f, %.1f, %.1f, %.1f) score=%.3f class=%d\n", ref.left, ref.top, ref.right,
                   ref.bottom, ref.confidence, ref.classId);
            err = 1;
        }
    }

Error:
    return err;
}

/**
 * @brief Release the operators and tensors.
 */
static void FreeTensorsAndOperators(SampleTensors *tensors, NVCVOperatorHandle operatorHandles[NUM_SCALES])
{
    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        nvcvOperatorDestroy(operatorHandles[sc]);
        nvcvTensorDecRef(tensors->predictions[sc], NULL);
        nvcvTensorDecRef(tensors->anchors[sc], NULL);
    }
    nvcvTensorDecRef(tensors->boxes, NULL);
    nvcvTensorDecRef(tensors->scores, NULL);
    nvcvTensorDecRef(tensors->classes, NULL);
    nvcvTensorDecRef(tensors->count, NULL);
    nvcvAllocatorDecRef(tensors->allocator, NULL);
}

static void PrintConfiguration(PVAYoloV5DecodeInputLayout layout, const char *streamName)
{
    printf("YOLOv5 decode, %s input, %s stream\n", LayoutName(layout), streamName);
    printf("  Image %dx%d, %d classes, threshold %.2f, grids", kImageSize, kImageSize, kNumClasses, kConfThreshold);
    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        printf(" %dx%d (stride %d)", GridSize(sc), GridSize(sc), kStrides[sc]);
    }
    printf("\n");
}

/**
 * @brief Decode the three scales using a CUDA stream.
 */
int32_t SubmitYoloV5DecodeToCudaStream(PVAYoloV5DecodeInputLayout layout)
{
    /* Set the handles of resources that should be freed in case of an error to NULL. */
    SampleTensors tensors                          = {};
    NVCVOperatorHandle operatorHandles[NUM_SCALES] = {};
    cudaStream_t stream                            = NULL;
    cudaError_t cudaErr                            = cudaSuccess;
    int32_t err                                    = 0;
    NVCVStatus submitStatus                        = NVCV_SUCCESS;
    bool skipped                                   = false;

    PrintConfiguration(layout, "CUDA");

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaErr, FreeResources);

    err = SetUp(layout, &tensors, operatorHandles);
    if (err)
        goto FreeResources;

    /* The scales are submitted in order; each one appends its detections after those of the previous one. */
    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        submitStatus = pvaYoloV5DecodeSubmit(operatorHandles[sc], stream, tensors.predictions[sc], tensors.anchors[sc],
                                             tensors.boxes, tensors.scores, tensors.classes, tensors.count);
        if (submitStatus == NVCV_ERROR_NOT_IMPLEMENTED)
        {
            printf("  Submit to CUDA stream: not supported by the cuPVA runtime, skipped\n");
            skipped = true;
            goto FreeResources;
        }
        NVCV_CHECK_ERROR_GOTO(submitStatus, err, FreeResources);
    }

    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), cudaErr, FreeResources);

    err = ValidateOutput(layout, &tensors);

FreeResources: /* clean up all allocated resources */
    if (cudaErr != cudaSuccess)
    {
        err = 1;
    }
    if (stream != NULL)
    {
        cudaStreamDestroy(stream);
    }
    FreeTensorsAndOperators(&tensors, operatorHandles);

    printf("  %s input, CUDA stream: %s\n\n", LayoutName(layout), err ? "FAILED" : (skipped ? "SKIPPED" : "PASSED"));

    return err;
}

/**
 * @brief Decode the three scales using a cuPVA stream.
 */
int32_t SubmitYoloV5DecodeToCupvaStream(PVAYoloV5DecodeInputLayout layout)
{
    /* Set the handles of resources that should be freed in case of an error to NULL. */
    SampleTensors tensors                          = {};
    NVCVOperatorHandle operatorHandles[NUM_SCALES] = {};
    cupvaSyncObj_t sync                            = NULL;
    cupvaStream_t stream                           = NULL;
    cupvaFence_t fence;
    cupvaCmd_t rf;
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};
    bool waitSuccess           = false;
    int32_t err                = 0;

    PrintConfiguration(layout, "cuPVA");

    err = SetUp(layout, &tensors, operatorHandles);
    if (err)
        goto FreeResources;

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* The scales are submitted in order; each one appends its detections after those of the previous one. */
    for (int32_t sc = 0; sc < NUM_SCALES; sc++)
    {
        NVCV_CHECK_ERROR_GOTO(
            pvaYoloV5DecodeSubmit(operatorHandles[sc], stream, tensors.predictions[sc], tensors.anchors[sc],
                                  tensors.boxes, tensors.scores, tensors.classes, tensors.count),
            err, FreeResources);
    }

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);
    if (!waitSuccess)
    {
        printf("Fence wait did not succeed\n");
        err = 1;
        goto FreeResources;
    }

    err = ValidateOutput(layout, &tensors);

FreeResources: /* clean up all allocated resources */
    if (stream != NULL)
    {
        CupvaStreamDestroy(stream);
    }
    if (sync != NULL)
    {
        CupvaSyncObjDestroy(sync);
    }
    FreeTensorsAndOperators(&tensors, operatorHandles);

    printf("  %s input, cuPVA stream: %s\n\n", LayoutName(layout), err ? "FAILED" : "PASSED");

    return err;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    const PVAYoloV5DecodeInputLayout layouts[] = {PVA_YOLOV5_DECODE_INPUT_CHW, PVA_YOLOV5_DECODE_INPUT_HWC};

    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    {
        uint32_t driverVersion = 0;
        cudaError_t cudaErr    = cudaSuccess;

        /* Initialize the CUDA context before the first cuPVA call. */
        CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaErr, CudaSetupFailed);

        /* Submit to CUDA stream if driver version is greater than or equal to 2007 */
        CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CudaSetupFailed);
        if (driverVersion >= 2007)
        {
            for (size_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); i++)
            {
                cudaStreamSubmitTestResult |= SubmitYoloV5DecodeToCudaStream(layouts[i]);
            }
        }

    CudaSetupFailed:
        if (cudaErr != cudaSuccess)
        {
            cudaStreamSubmitTestResult = 1;
        }
    }
#endif

    /* Submit to cuPVA stream */
    for (size_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); i++)
    {
        cupvaStreamSubmitTestResult |= SubmitYoloV5DecodeToCupvaStream(layouts[i]);
    }

    /* Overall result */
    if (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

    return (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult);
}
