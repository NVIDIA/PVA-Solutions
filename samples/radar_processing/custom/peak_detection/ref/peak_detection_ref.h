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
 * @file peak_detection_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef PEAK_DETECTION_REF_H
#define PEAK_DETECTION_REF_H

#include "peak_detection_param.h"

#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomPeakDetectionInput(NVCVTensorHandle tensorHandle);

int32_t loadTensorInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename);

void peakDetectionRef(NVCVTensorHandle *inHandles, int32_t NofTx, int32_t repeatFold, NVCVTensorHandle *outHandles);

int32_t checkPeakDetectionOutput(int32_t NofTx, int32_t NofRx, NVCVTensorHandle refPeakCountTensorHandle,
                                 NVCVTensorHandle refPeakIndexTensorHandle, NVCVTensorHandle refSnapTensorHandle,
                                 NVCVTensorHandle pvaPeakCountTensorHandle, NVCVTensorHandle pvaPeakIndexTensorHandle,
                                 NVCVTensorHandle pvaSnapTensorHandle, float maxErrRatio);

#ifdef __cplusplus
}
#endif

#endif /* PEAK_DETECTION_REF_H */
