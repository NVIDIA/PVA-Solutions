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
 * @file floyd_warshall_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef FLOYD_WARSHALL_REF_H
#define FLOYD_WARSHALL_REF_H

#include <nvcv/Tensor.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomFloydWarshallInput(NVCVTensorHandle lenTensor, NVCVTensorHandle nextTensor, bool directed,
                                      bool connected);

void copyInputTensor(NVCVTensorHandle srcTensor, NVCVTensorHandle dstTensor);

int32_t checkFloydWarshallOutput(NVCVTensorHandle refLenTensor, NVCVTensorHandle refNextTensor,
                                 NVCVTensorHandle pvaLenTensor, NVCVTensorHandle pvaNextTensor);

void floydWarshallRef(NVCVTensorHandle refLenTensor, NVCVTensorHandle refNextTensor);

#ifdef __cplusplus
}
#endif

#endif /* FLOYD_WARSHALL_REF_H */