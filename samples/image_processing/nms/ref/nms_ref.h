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
 * @file nms_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef NMS_REF_H
#define NMS_REF_H

#include <OpNms.h>
#include <nvcv/BorderType.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fill the input tensor with random values.
 *
 * A narrow value range makes equal and nearly equal neighbors common, exercising the tie-break and
 * accuracy rules documented in OpNms.h.
 */
void generateRandomNmsInput(NVCVTensorHandle tensorHandle);

/**
 * @brief CPU model of the Non-Maximum Suppression (NMS) operator.
 *
 * Implements the tie-break and accuracy behavior documented in OpNms.h, so the comparison against
 * the PVA output is bit-exact.
 */
void nmsRef(NVCVTensorHandle in, NVCVTensorHandle out, int32_t nmsSize, NVCVBorderType borderMode, int32_t borderValue);

/**
 * @brief Compare the reference and PVA outputs element by element.
 *
 * @return 0 when every element matches, non-zero otherwise.
 */
int32_t checkNmsOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);

#ifdef __cplusplus
}
#endif

#endif /* NMS_REF_H */
