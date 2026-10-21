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
 * @file doppler_fft_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef DOPPLER_FFT_REF_H
#define DOPPLER_FFT_REF_H

#include <OpDopplerFFT.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateDopplerFFTInput(NVCVTensorHandle tensorHandle);

void generateDopplerFFTWindow(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType);

int32_t checkDopplerFFTOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                              const float maxDiffFloat);

void dopplerFFTRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle, NVCVTensorHandle outTensorHandle,
                   PVADopplerFFTParams *params);

int32_t loadDopplerFFTRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                          bool convertToFixed);

void dumpDopplerFFTSNR(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle);

#ifdef __cplusplus
}
#endif

#endif /* DOPPLER_FFT_REF_H */
