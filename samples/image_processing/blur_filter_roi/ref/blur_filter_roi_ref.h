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
 * @file blur_filter_roi_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef BLUR_FILTER_ROI_REF_H
#define BLUR_FILTER_ROI_REF_H

#include <OpBlurFilterROI.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
#    include <vector>
extern "C" {
#endif

void generateBlurFilterROIInput(NVCVImageHandle imageHandle);

int32_t checkBlurFilterROIOutput(NVCVImageHandle refImageHandle, NVCVImageHandle pvaImageHandle);

void blurFilterROIRefC(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle, const NVCVRectI *rectArray,
                       int32_t numRects, const NVCVSize2D blockSize);

void memcpyToTensor(NVCVTensorHandle tensorHandle, const void *src, size_t count);

#ifdef __cplusplus
}

void blurFilterROIRef(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle,
                      const std::vector<NVCVRectI> &rectVec, const NVCVSize2D blockSize);
#endif

#endif /* BLUR_FILTER_ROI_REF_H */
