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

#ifndef IMAGE_FLIP_REF_H
#define IMAGE_FLIP_REF_H

#include <PvaOperatorTypes.h>
#include <nvcv/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate random input data for image flip testing
 * @param imageRequirements Image requirements specifying shape and format
 * @param inImageHandle Handle to input image to fill with random data
 */
void generateImageFlipRandomInput(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle);

/**
 * @brief Generate reference output data for image flip
 * @param imageRequirements Image requirements specifying shape and format
 * @param inImageHandle Handle to input image
 * @param refImageHandle Handle to reference image to fill with expected output
 * @param flipDirection Direction of the flip operation (PVA_FLIP_HORIZONTAL, PVA_FLIP_VERTICAL, PVA_FLIP_BOTH)
 */
void getImageFlipRefData(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                         NVCVImageHandle refImageHandle, PVAFlipDirection flipDirection);

/**
 * @brief Check if image flip output matches reference
 * @param imageRequirements Image requirements specifying shape and format
 * @param inImageHandle Handle to input image
 * @param refImageHandle Handle to reference image
 * @param outImageHandle Handle to output image to check
 * @return 0 if match, non-zero if different
 */
int32_t checkImageFlipOutput(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                             NVCVImageHandle refImageHandle, NVCVImageHandle outImageHandle);

/**
 * @brief Generate fixed input from image file for visual testing
 * @param imageRequirements Image requirements specifying shape and format
 * @param inImageHandle Handle to input image to fill
 * @param assetDir Directory containing input images
 */
void generateImageFlipFixedImgInput(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                                    const char *assetDir);

/**
 * @brief Save image flip output to file for visual inspection
 * @param outImageHandle Handle to output image to save
 * @param filename Output filename
 * @param width Image width
 * @param height Image height
 */
void saveImageFlipOutput(NVCVImageHandle outImageHandle, const char *filename, int32_t width, int32_t height);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_FLIP_REF_H
