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
#include "coco_evaluation.h"
#include "yolo_postprocess.h"

#include <ImageIO.h>
#include <OpDLInference.h>
#include <PvaAllocator.h>
#include <nvcv/Fwd.h>
#include <nvcv/Size.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Function to get model name from command line arguments
inline int GetModelName(int argc, char **argv, char *modelName, int maxNameLength)
{
    int err = -1;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model"))
        {
            if (i + 1 < argc)
            {
                strncpy(modelName, argv[++i], maxNameLength);
                modelName[maxNameLength - 1] = '\0';
                printf("Using model: %s\n", modelName);
                err = 0;
            }
            else
            {
                printf("--model option requires one argument.\n");
            }
            break;
        }
    }

    if (err != 0)
    {
        printf("Please specify model name using -m or --model option\n");
    }

    return err;
}

int main(int argc, char **argv)
{
    NVCVAllocatorHandle allocatorHandle      = NULL;
    NVCVTensorHandle inTensorHandle          = NULL;
    NVCVTensorHandle outTensorHandle0        = NULL;
    NVCVTensorHandle outTensorHandle1        = NULL;
    NVCVTensorHandle outTensorHandle2        = NULL;
    NVCVTensorHandle postprocOutTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle        = NULL;
    cupvaSyncObj_t sync                      = NULL;
    cupvaStream_t stream                     = NULL;
    int32_t err                              = 0;
    int32_t imgHeight                        = 640;
    int32_t imgWidth                         = 640;
    int32_t imgChannels                      = 3;
    int32_t numClasses                       = 80;

    const int32_t maxPathLength = 200;
    char assetsDirectory[maxPathLength];
    if (GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength) != 0)
    {
        throw std::runtime_error("Failed to get assets directory");
    }

    const int32_t maxNameLength = 100;
    char modelName[maxNameLength];
    if (GetModelName(argc, argv, modelName, maxNameLength) != 0)
    {
        throw std::runtime_error("Failed to get model name");
    }

    if (strcmp(modelName, "yolov5s") != 0 && strcmp(modelName, "yolov26s") != 0)
    {
        throw std::runtime_error(std::string("Unsupported detection model: ") + modelName);
    }

    const bool isYolov5  = strstr(modelName, "yolov5") != nullptr;
    const bool isYolov26 = strstr(modelName, "yolov26") != nullptr;

    NVCV_CHECK_ERROR(nvcvAllocatorConstructPva(&allocatorHandle));

    // Input tensor
    NVCVTensorLayout inputLayout;
    NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HWC", &inputLayout));

    NVCVTensorRequirements inputRequirements;
    int64_t imageTensorShape[] = {imgHeight, imgWidth, imgChannels};
    NVCV_CHECK_ERROR(
        nvcvTensorCalcRequirementsPva(3, imageTensorShape, NVCV_DATA_TYPE_S8, inputLayout, 0, 0, &inputRequirements));

    NVCV_CHECK_ERROR(nvcvTensorConstruct(&inputRequirements, allocatorHandle, &inTensorHandle));

    NVCVTensorData inputTensorData;
    NVCV_CHECK_ERROR(nvcvTensorExportData(inTensorHandle, &inputTensorData));

    uint8_t *image_h;
    CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void **)&image_h, (void *)inputTensorData.buffer.strided.basePtr));

    if (ReadImageBuffer("/onnx/traffic.bin", assetsDirectory, image_h,
                        sizeof(uint8_t) * imgHeight * imgWidth * imgChannels) != 0)
    {
        throw std::runtime_error("Failed to read input image: assets/onnx/traffic.bin");
    }

    // Output tensors — shapes and types differ per model
    NVCVTensorRequirements outputRequirements0, outputRequirements1, outputRequirements2;
    int32_t numOutputs = 0;

    if (isYolov5)
    {
        // YOLOv5: three feature levels, each [H, W, (num_classes+5)*3]
        NVCVTensorLayout layout3d;
        NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HWC", &layout3d));

        int64_t shape0[] = {80, 80, 255};
        int64_t shape1[] = {40, 40, 255};
        int64_t shape2[] = {20, 20, 255};
        NVCV_CHECK_ERROR(
            nvcvTensorCalcRequirementsPva(3, shape0, NVCV_DATA_TYPE_F32, layout3d, 0, 0, &outputRequirements0));
        NVCV_CHECK_ERROR(
            nvcvTensorCalcRequirementsPva(3, shape1, NVCV_DATA_TYPE_F32, layout3d, 0, 0, &outputRequirements1));
        NVCV_CHECK_ERROR(
            nvcvTensorCalcRequirementsPva(3, shape2, NVCV_DATA_TYPE_F32, layout3d, 0, 0, &outputRequirements2));
        NVCV_CHECK_ERROR(nvcvTensorConstruct(&outputRequirements0, allocatorHandle, &outTensorHandle0));
        NVCV_CHECK_ERROR(nvcvTensorConstruct(&outputRequirements1, allocatorHandle, &outTensorHandle1));
        NVCV_CHECK_ERROR(nvcvTensorConstruct(&outputRequirements2, allocatorHandle, &outTensorHandle2));
        numOutputs = 3;
    }
    else if (isYolov26)
    {
        // YOLO26: three post-TopK outputs (NMS-free, 300 candidates)
        // Output 0: scores   [1, 300]    confidence scores  (TopK_1_output_0)
        // Output 1: cls_ids  [1, 300]    class indices, i32 (Mod_output_0)
        // Output 2: boxes    [1, 300, 4] x1,y1,x2,y2       (GatherElements_1_output_0)
        NVCVTensorLayout layout2d, layout3d;
        NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HW", &layout2d));
        NVCV_CHECK_ERROR(nvcvTensorLayoutMake("HWC", &layout3d));

        int64_t shapeScores[] = {1, 300};
        int64_t shapeClsIds[] = {1, 300};
        int64_t shapeBoxes[]  = {1, 300, 4};
        NVCV_CHECK_ERROR(
            nvcvTensorCalcRequirementsPva(2, shapeScores, NVCV_DATA_TYPE_F32, layout2d, 0, 0, &outputRequirements0));
        NVCV_CHECK_ERROR(
            nvcvTensorCalcRequirementsPva(2, shapeClsIds, NVCV_DATA_TYPE_S32, layout2d, 0, 0, &outputRequirements1));
        NVCV_CHECK_ERROR(
            nvcvTensorCalcRequirementsPva(3, shapeBoxes, NVCV_DATA_TYPE_F32, layout3d, 0, 0, &outputRequirements2));
        NVCV_CHECK_ERROR(nvcvTensorConstruct(&outputRequirements0, allocatorHandle, &outTensorHandle0));
        NVCV_CHECK_ERROR(nvcvTensorConstruct(&outputRequirements1, allocatorHandle, &outTensorHandle1));
        NVCV_CHECK_ERROR(nvcvTensorConstruct(&outputRequirements2, allocatorHandle, &outTensorHandle2));
        numOutputs = 3;
    }

    PvaDLInferenceCreateParams params;
    params.inReqs[0]   = &inputRequirements;
    params.outReqs[0]  = &outputRequirements0;
    params.outReqs[1]  = &outputRequirements1;
    params.outReqs[2]  = &outputRequirements2;
    params.numInReqs   = 1;
    params.numOutReqs  = numOutputs;
    params.networkName = modelName;
    NVCV_CHECK_ERROR(pvaDLInferenceCreate(&operatorHandle, &params));

    CUPVA_CHECK_ERROR(CupvaSyncObjCreate(&sync, true, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD));

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR(CupvaFenceInit(&fence, sync));

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR(CupvaCmdRequestFencesInit(&rf, &fence, 1));

    CUPVA_CHECK_ERROR(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY));

    PvaDLInferenceSubmitParams submitParams;
    submitParams.inTensors[0]  = inTensorHandle;
    submitParams.outTensors[0] = outTensorHandle0;
    submitParams.outTensors[1] = outTensorHandle1;
    submitParams.outTensors[2] = outTensorHandle2; // NULL for 2-output models
    submitParams.numInTensors  = 1;
    submitParams.numOutTensors = numOutputs;
    NVCV_CHECK_ERROR(pvaDLInferenceSubmit(operatorHandle, stream, &submitParams));

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};
    CUPVA_CHECK_ERROR(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1));

    bool waitSuccess;
    CUPVA_CHECK_ERROR(CupvaFenceWait(&fence, -1, &waitSuccess));

    // Post processing — extract host pointers and decode detections
    NVCVTensorData outTensorData0, outTensorData1, outTensorData2;
    NVCV_CHECK_ERROR(nvcvTensorExportData(outTensorHandle0, &outTensorData0));
    NVCV_CHECK_ERROR(nvcvTensorExportData(outTensorHandle1, &outTensorData1));
    NVCV_CHECK_ERROR(nvcvTensorExportData(outTensorHandle2, &outTensorData2));

    std::vector<Box> boxes;
    if (isYolov5)
    {
        float *outPtr0, *outPtr1, *outPtr2;
        CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void **)&outPtr0, (void *)outTensorData0.buffer.strided.basePtr));
        CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void **)&outPtr1, (void *)outTensorData1.buffer.strided.basePtr));
        CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void **)&outPtr2, (void *)outTensorData2.buffer.strided.basePtr));

        constexpr int outputSize0     = 80 * 80 * 255;
        constexpr int outputSize1     = 40 * 40 * 255;
        constexpr int outputSize2     = 20 * 20 * 255;
        constexpr int totalOutputSize = outputSize0 + outputSize1 + outputSize2;

        std::unique_ptr<float[]> postprocInput = std::make_unique<float[]>(totalOutputSize);
        memcpy(postprocInput.get(), outPtr0, outputSize0 * sizeof(float));
        memcpy(postprocInput.get() + outputSize0, outPtr1, outputSize1 * sizeof(float));
        memcpy(postprocInput.get() + outputSize0 + outputSize1, outPtr2, outputSize2 * sizeof(float));

        int totalNumBoxes = totalOutputSize / (numClasses + 5);
        boxes             = yolov5PostProcess(postprocInput.get(), totalNumBoxes, numClasses, imgHeight, imgWidth);
    }
    else if (isYolov26)
    {
        // Output 0: scores  [1, 300]    — confidence values
        // Output 1: cls_ids [1, 300]    — class indices (int32, cast from int64 by pva compiler)
        // Output 2: boxes   [1, 300, 4] — x1,y1,x2,y2 in letterboxed pixel coords
        float *scores, *bboxes;
        int32_t *clsIds;
        CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void **)&scores, (void *)outTensorData0.buffer.strided.basePtr));
        CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void **)&clsIds, (void *)outTensorData1.buffer.strided.basePtr));
        CUPVA_CHECK_ERROR(CupvaMemGetHostPointer((void **)&bboxes, (void *)outTensorData2.buffer.strided.basePtr));

        // TopK sorts the candidates by score; keep those above the YOLOv5 confidence threshold.
        constexpr int32_t kNumCandidates = 300;
        constexpr float kMinConfidence   = 0.25f;
        for (int i = 0; i < kNumCandidates && scores[i] >= kMinConfidence; i++)
        {
            const float *b = bboxes + i * 4;
            boxes.emplace_back(b[0], b[1], b[2], b[3], scores[i], clsIds[i]);
        }
    }
    printf("Number of detections=%d\n", (int)boxes.size());
    printf("left, top, right, bottom, confidence, class:\n");

    // Project bbox from model space (640x640) to original image (1280x720)
    // Top-left letterbox: scale = min(640/1280, 640/720) = 0.5, inv_scale = 2
    constexpr float kInvScale = 2.0f;

    // These outputs are generated from YOLOv5s ONNX model
    static const Box groundTruthYolov5[] = {
        {474.5f, 10.1f, 1280.0f, 706.1f, 5}, // bus
        {0.1f, 266.4f, 137.4f, 681.0f, 0},   // person
        {196.3f, 296.7f, 314.4f, 570.6f, 0}, // person
        {404.3f, 325.7f, 475.5f, 380.9f, 2}, // car
        {361.5f, 312.5f, 399.6f, 348.6f, 2}, // car
        {70.9f, 300.6f, 138.7f, 513.5f, 0},  // person
        {0.0f, 327.7f, 55.6f, 455.0f, 24},   // backpack
        {392.0f, 315.7f, 435.4f, 348.3f, 2}, // car
    };

    // These outputs are generated from the float YOLO26s ONNX model (confidence >= 0.25)
    static const Box groundTruthYolov26[] = {
        {471.8f, 1.7f, 1276.8f, 708.5f, 5},  // bus
        {193.2f, 291.9f, 316.9f, 576.3f, 0}, // person
        {0.0f, 258.9f, 131.4f, 684.1f, 0},   // person
        {407.7f, 328.8f, 476.9f, 381.4f, 2}, // car
        {362.0f, 314.8f, 399.6f, 347.2f, 2}, // car
        {75.4f, 301.8f, 137.7f, 513.1f, 0},  // person
        {396.6f, 315.6f, 436.9f, 353.0f, 2}, // car
        {306.5f, 321.9f, 326.7f, 371.0f, 0}, // person
    };

    const Box *groundTruth   = isYolov26 ? groundTruthYolov26 : groundTruthYolov5;
    const int numGroundTruth = isYolov26 ? sizeof(groundTruthYolov26) / sizeof(groundTruthYolov26[0])
                                         : sizeof(groundTruthYolov5) / sizeof(groundTruthYolov5[0]);

    // Note: These thresholds are application-specific and may need adjustment
    constexpr float kIoUThreshold = 0.5f; // 0.5 for AP@0.5 metric
    constexpr float kMinPrecision = 0.7f;
    constexpr float kMinRecall    = 0.7f;

    bool validationPassed =
        evaluateCocoStyle(boxes, groundTruth, numGroundTruth, kInvScale, kIoUThreshold, kMinPrecision, kMinRecall);

    if (!validationPassed)
    {
        err++;
    }

    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle0, NULL);
    nvcvTensorDecRef(outTensorHandle1, NULL);
    if (outTensorHandle2 != NULL)
        nvcvTensorDecRef(outTensorHandle2, NULL);
    nvcvTensorDecRef(postprocOutTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}
