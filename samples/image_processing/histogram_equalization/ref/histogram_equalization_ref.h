/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * @file HISTOGRAM_EQUALIZATION_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef HISTOGRAM_EQUALIZATION_REF_H
#define HISTOGRAM_EQUALIZATION_REF_H

#include <ImageIO.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t loadHistogramEqualizationData(NVCVTensorHandle tensorHandle, const char *assetDir, const char *filename,
                                      const int32_t tensorSizeInBytes);

void generateHistogramEqualizationFixedImgInput(NVCVTensorRequirements *tensorRequirements,
                                                NVCVTensorHandle inTensorHandle, const char *assetDir);

void generateRandomInput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle);

int32_t checkHistogramEqualizationOutput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                         NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle);

int32_t getHistogramEqualizationRefData(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                        NVCVTensorHandle ref_outTensorHandle);

int32_t saveHistogramEqualizationOutput(NVCVTensorHandle tensorHandle, const char *filePath, int image_width,
                                        int image_height);

#ifdef __cplusplus
}
#endif

#endif /* IMAGE_HISTOGRAM_REF_H */
