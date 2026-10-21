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
 * @file orb_desc_extractor_ref.h
 *
 * @brief Defines functions for ORB descriptor extraction reference model verification
 */

#ifndef ORB_DESC_EXTRACTOR_REF_H
#define ORB_DESC_EXTRACTOR_REF_H

#include <PvaOperatorTypes.h>
#include <nvcv/BorderType.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateORBDescExtractorRandomInput(NVCVImageHandle imageHandle, NVCVTensorHandle inCornersTensorHandle);

void ORBDescExtractorRef(NVCVImageHandle imageHandle, NVCVTensorHandle inCornersTensorHandle,
                         NVCVTensorHandle refDescriptorsTensorHandle, uint32_t disableRBRIEFFlag,
                         NVCVBorderType borderType, int32_t borderValue);

int32_t checkORBDescExtractorOutput(NVCVTensorHandle refDescriptorsTensorHandle,
                                    NVCVTensorHandle pvaDescriptorsTensorHandle);

#ifdef __cplusplus
}
#endif

#endif /* ORB_DESC_EXTRACTOR_REF_H */
