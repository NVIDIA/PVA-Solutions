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
 * @file fast_corner_detector_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef FAST_CORNER_DETECTOR_REF_H
#define FAST_CORNER_DETECTOR_REF_H

#include <OpFastCorner.h>
#include <nvcv/BorderType.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif
int32_t generateFastCornerDetectorInput(NVCVTensorHandle tensorHandle, char const *fileAbsPathName);

void fastCornerDetectorRef(NVCVTensorHandle inTensorHandle, PvaFastCornerDetectorParams const *params,
                           int32_t intensityThreshold, const NVCVBorderType borderMode, int32_t borderValue,
                           NVCVTensorHandle locHandle, NVCVTensorHandle numLocHandle);

int32_t checkFastCornerDetectorOutput(NVCVTensorHandle refHandle, NVCVTensorHandle refNumLocHandle,
                                      NVCVTensorHandle pvaLocHandle, NVCVTensorHandle pvaNumLocHandle);

#ifdef __cplusplus
}
#endif

#endif /* FAST_CORNER_DETECTOR_REF_H */
