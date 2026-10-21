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
 * @file range_fft_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef RANGE_FFT_REF_H
#define RANGE_FFT_REF_H

#include <OpRangeFFT.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void loadRangeFFTInput(NVCVTensorHandle tensorHandle, int32_t *inBuf, int32_t inHeight, int32_t inChannels,
                       int32_t inWidth);

void generateRangeFFTInput(NVCVTensorHandle tensorHandle);

void generateRangeFFTWindow(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType);

int32_t checkRangeFFTOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                            const float maxDiffFloat);

void rangeFFTRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle, NVCVTensorHandle outTensorHandle,
                 PVARangeFFTParams *params);

void loadRangeFFTRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename, bool convertToFixed);

void storeRangeFFTRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename);

void dumpRangeFFTSNR(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);

#ifdef __cplusplus
}

// C++ only functions (cannot be used from C due to std::vector)
#    include <vector>
void load_range_fft_input(NVCVTensorHandle tensorHandle, std::vector<std::vector<std::vector<int32_t>>> &inBuf);

#endif

#endif /* RANGE_FFT_REF_H */
