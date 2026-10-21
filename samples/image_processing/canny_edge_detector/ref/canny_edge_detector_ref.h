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
 * @file canny_edge_detector_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef CANNY_EDGE_DETECTOR_REF_H
#define CANNY_EDGE_DETECTOR_REF_H

#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t loadCannyImageInput(NVCVTensorHandle imageTensorHandle, const char *directory, const char *filename,
                            const int64_t size);

int32_t checkCannyEdgeDetectorOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);

void CannyEdgeDetectorFixedPointRef(NVCVTensorHandle imageTensorHandle, NVCVTensorHandle edgeMapTensorHandle,
                                    const int32_t gradientSize, const int32_t thresholdStrong,
                                    const int32_t thresholdWeak);

#ifdef __cplusplus
}
#endif

#endif /* CANNY_EDGE_DETECTOR_REF_H */