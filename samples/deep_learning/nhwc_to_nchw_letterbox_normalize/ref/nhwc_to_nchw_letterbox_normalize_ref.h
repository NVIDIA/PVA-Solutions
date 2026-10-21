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

#ifndef NHWC_TO_NCHW_LETTERBOX_NORMALIZE_REF_H
#define NHWC_TO_NCHW_LETTERBOX_NORMALIZE_REF_H

#include <nvcv/Tensor.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fill an HWC U8 input tensor with a deterministic gradient pattern.
 */
void generateNhwcToNchwLetterboxNormalizeInput(NVCVTensorHandle imgIn);

/**
 * CPU reference: fused NHWC->NCHW + letterbox + normalize.
 *
 * Input:  HWC U8  (imgH x imgW x 3)
 * Output: CHW F32 (3 x outH x outW)
 *
 * Letterbox centers the image in the output with padding.
 * Image pixels: (float)pixel * scale
 * Pad pixels:   padValue
 */
void nhwcToNchwLbNormRef(NVCVTensorHandle imgIn, NVCVTensorHandle floatOut, float scale, float padValue);

/**
 * Compare two CHW F32 output tensors element-wise.
 *
 * @return 0 if all elements match within tolerance, else the number of mismatches found,
 *         capped at 20 (the comparison stops at the 20th mismatch).
 */
int32_t checkNhwcToNchwLetterboxNormalizeOutput(NVCVTensorHandle refOut, NVCVTensorHandle testOut);

#ifdef __cplusplus
}
#endif

#endif /* NHWC_TO_NCHW_LETTERBOX_NORMALIZE_REF_H */
