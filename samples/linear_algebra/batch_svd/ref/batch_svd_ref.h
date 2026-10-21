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

/*
 * @file batch_svd_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef BATCH_SVD_REF_H
#define BATCH_SVD_REF_H

#include <OpBatchSVD.h>
#include <nvcv/Tensor.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateBatchSVDRandomInput(NVCVTensorHandle tensorHandle, const float rangeMin, const float rangeMax,
                                 bool rankDeficient);

void batchSVDRef(NVCVTensorHandle srcTensorHandle, NVCVTensorHandle uTensorHandle, NVCVTensorHandle sTensorHandle,
                 NVCVTensorHandle vTensorHandle, const int maxIters, const float tol);

int32_t checkBatchSVDOutput(NVCVTensorHandle uPvaTensorHandle, NVCVTensorHandle sPvaTensorHandle,
                            NVCVTensorHandle vPvaTensorHandle, NVCVTensorHandle uRefTensorHandle,
                            NVCVTensorHandle sRefTensorHandle, NVCVTensorHandle vRefTensorHandle, const float tol);

#ifdef __cplusplus
}
#endif

#endif
