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

/*
 * @file hog_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef HOG_REF_H
#define HOG_REF_H

#include <OpHOG.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int32_t w;
    int32_t h;
} Size2D;

typedef struct
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} RGB;

void generateRandomHOGInput(NVCVImageHandle imageHandle);
int32_t checkHOGOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);
void hogRef(NVCVImageHandle inImageHandle, NVCVTensorHandle refTensorHandle);
void loadHOGDataFromFile(NVCVImageHandle imageHandle, const char *directory, const char *filename);

#ifdef __cplusplus
}
#endif

#endif /* HOG_REF_H */
