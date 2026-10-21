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

#include "radar_cfar_test_ref.h"

#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <unistd.h>

#include <cstring>
#include <random>
/* Include the CFAR library headers */
#include "radar_cfar_top.hpp"

#include "radar_cfar_param.h"

int32_t radarCFARRefSubmit(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, PvaRadarCFARParams cfg,
                           NVCVTensorHandle detectionCountTensorHandle, PVARadarCFARType algorithm)
{
    int result = 0;

    /* Validate input tensor handles */
    if (inTensorHandle == nullptr || outTensorHandle == nullptr || detectionCountTensorHandle == nullptr)
    {
        printf("Error: Invalid tensor handles (null pointers)\n");
        return -1;
    }

    /* Validate CFAR configuration parameters */
    /* Check training cells range */
    if (cfg.numHorTrain <= 0 || cfg.numHorTrain > CFAR_NUM_TRAIN_MAX || cfg.numVerTrain <= 0 ||
        cfg.numVerTrain > CFAR_NUM_TRAIN_MAX)
    {
        printf("Error: Invalid training cells (numHorTrain=%d, numVerTrain=%d), must be 1-%d\n", cfg.numHorTrain,
               cfg.numVerTrain, CFAR_NUM_TRAIN_MAX);
        return -1;
    }

    /* Check guard cells range */
    if (cfg.numHorGuard >= cfg.numHorTrain)
    {
        printf("Error: Invalid guard cells (numHorGuard=%d,), must be 0-%d\n", cfg.numHorGuard, cfg.numHorTrain);
        return -1;
    }

    /* Check guard cells are less than training cells */
    if (cfg.numVerGuard >= cfg.numVerTrain)
    {
        printf("Error: Invalid guard cells (numVerGuard=%d), must be 0-%d\n", cfg.numVerGuard, cfg.numVerTrain);
        return -1;
    }

    /* Check algorithm type */
    if (algorithm != PVA_CFAR_CA && algorithm != PVA_CFAR_CA_GO && algorithm != PVA_CFAR_CA_SO)
    {
        printf("Error: Invalid algorithm type (%d), must be %d, %d, or %d\n", algorithm, PVA_CFAR_CA, PVA_CFAR_CA_GO,
               PVA_CFAR_CA_SO);
        return -1;
    }

    /* Export tensor data for input tensor */
    NVCVTensorData inTensorData;
    NVCVStatus status = nvcvTensorExportData(inTensorHandle, &inTensorData);
    if (status != NVCV_SUCCESS)
    {
        printf("Failed to export input tensor data: %d\n", status);
        return -1;
    }

    /* Export tensor data for output tensor */
    NVCVTensorData outTensorData;
    status = nvcvTensorExportData(outTensorHandle, &outTensorData);
    if (status != NVCV_SUCCESS)
    {
        printf("Failed to export output tensor data: %d\n", status);
        return -1;
    }

    /* Export tensor data for detection count tensor */
    NVCVTensorData detectionCountTensorData;
    status = nvcvTensorExportData(detectionCountTensorHandle, &detectionCountTensorData);
    if (status != NVCV_SUCCESS)
    {
        printf("Failed to export detection count tensor data: %d\n", status);
        return -1;
    }

    /* Validate tensor dimensions */
    if (inTensorData.shape[0] <= 0 || inTensorData.shape[1] <= 0)
    {
        printf("Error: Invalid tensor dimensions (height=%ld, width=%ld)\n", inTensorData.shape[0],
               inTensorData.shape[1]);
        return -1;
    }

    /* Check tensor data type */
    if ((inTensorData.dtype != NVCV_DATA_TYPE_S32 && inTensorData.dtype != NVCV_DATA_TYPE_U32) ||
        outTensorData.dtype != NVCV_DATA_TYPE_S32)
    {
        printf("Error: Unsupported tensor data type, expected S32\n");
        return -1;
    }

    /* Get host pointers to the tensor data */
    int32_t *inData = nullptr;
    CupvaMemGetHostPointer((void **)&inData, (void *)inTensorData.buffer.strided.basePtr);

    int32_t *outData = nullptr;
    CupvaMemGetHostPointer((void **)&outData, (void *)outTensorData.buffer.strided.basePtr);

    int32_t *detectionCountData = nullptr;
    CupvaMemGetHostPointer((void **)&detectionCountData, (void *)detectionCountTensorData.buffer.strided.basePtr);
    /* Validate host pointers */
    if (inData == nullptr || outData == nullptr || detectionCountData == nullptr)
    {
        printf("Error: Failed to get host pointers to tensor data\n");
        return -1;
    }

    int32_t height = inTensorData.shape[0];
    int32_t width  = inTensorData.shape[1];

    if (width > 1024 || ((width % 8) != 0) || width < 1)
    {
        printf("Error: Unsupported Width:%d. Valid range (1-1024) and must be divisible by 8\n", width);
        return -1;
    }
    if (height > 1024 || ((height % 8) != 0) || height < 1)
    {
        printf("Error: Unsupported Height:%d. Valid range (1-1024) and must be divisible by 8\n", height);
        return -1;
    }
    if (height * width > (1024 * 512))
    {
        printf("Error: Unsupported Height:%d and width:%d.  Valid range for width * height is (1 - 512*1024).\n",
               height, width);
        return -1;
    }

    if (cfg.numHorTrain > (width - 1) / 2)
    {
        printf("Error: Invalid number of training cells %d valid range 1 - %d\n", cfg.numHorTrain, (width - 1) / 2);
        return -1;
    }
    if (cfg.numVerTrain > (height - 1) / 2)
    {
        printf("Error: Invalid number of training cells %d valid range 1 - %d\n", cfg.numVerTrain, (height - 1) / 2);
        return -1;
    }
    if (cfg.numHorTrain + cfg.numHorGuard > 255)
    {
        printf("Error: Invalid number of training cells + guard cells %d valid range 1 - %d\n",
               cfg.numHorTrain + cfg.numHorGuard, 255);
        return -1;
    }
    if (cfg.numVerTrain + cfg.numVerGuard > 255)
    {
        printf("Error: Invalid number of training cells + guard cells %d valid range 1 - %d\n",
               cfg.numVerTrain + cfg.numVerGuard, 255);
        return -1;
    }
    /* Call the CFAR execution function directly */
    result = radar_cfar_func(inData, outData, cfg, width, height, algorithm);
    if (result < 0)
    {
        printf("CFAR processing failed with error: %d\n", result);
    }
    else
    {
        *detectionCountData = result;
        result              = 0;
    }
    return result;
}