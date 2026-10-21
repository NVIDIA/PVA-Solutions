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
 * @file ccl_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef CCL_REF_H
#define CCL_REF_H

#include <PvaOperatorTypes.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void CCLRef(NVCVTensorHandle imageHdl, NVCVTensorHandle labelHdl, NVCVTensorHandle numLabelsHdl,
            PVAConnectivityType connectivity, const uint8_t threshold);

int32_t checkCCLOutput(NVCVTensorHandle refLabelHdl, NVCVTensorHandle pvaLabelHdl, NVCVTensorHandle refNumLabelsHdl,
                       NVCVTensorHandle pvaNumLabelsHdl);

void generateRandomCCLInput(NVCVTensorHandle imageHdl, const int32_t width, const int32_t height, const int32_t lp);

void generateRandomCCLBinaryInput(NVCVTensorHandle imageHdl, const int32_t width, const int32_t height,
                                  const int32_t lp);

#ifdef __cplusplus
}
#endif

#endif /* CCL_REF_H */