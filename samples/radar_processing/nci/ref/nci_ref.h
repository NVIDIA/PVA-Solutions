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
 * @file nci_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef NCI_REF_H
#define NCI_REF_H

#include <PvaOperatorTypes.h>
#include <nvcv/Tensor.h>
#ifdef __cplusplus
extern "C" {
#endif

int32_t loadNCIRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename);

void generateRandomNciInput(NVCVTensorHandle tensorHandle, int32_t qBits);

int32_t nciRef(NVCVTensorHandle const *inTensorHandles, NVCVTensorHandle const *outTensorHandles, int inTensorCount,
               int outTensorCount, uint32_t numDopplerFolds, PVAFFTLayout inputLayout);

int32_t checkNCIOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle, uint32_t tolerance);

#ifdef __cplusplus
}
#endif

#endif /* NCI_REF_H */