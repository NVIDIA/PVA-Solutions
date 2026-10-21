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

#ifndef RADAR_PIPELINE_HOST_UTILS_HPP
#define RADAR_PIPELINE_HOST_UTILS_HPP

#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

/** Host pointer to a PVA tensor's storage, and optionally its export data. Fails unless the tensor is densely
 *  packed (no row padding), so callers can index it as a plain row-major array. */
inline bool radarTensorHostPtr(NVCVTensorHandle handle, void **out, NVCVTensorData *dataOut = nullptr)
{
    if (!handle || !out)
        return false;
    NVCVTensorData data;
    if (nvcvTensorExportData(handle, &data) != NVCV_SUCCESS || data.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA)
        return false;
    const NVCVTensorBufferStrided &b = data.buffer.strided;
    for (int32_t d = data.rank - 2; d >= 0; --d)
    {
        if (b.strides[d] != b.strides[d + 1] * data.shape[d + 1])
            return false;
    }
    if (CupvaMemGetHostPointer(out, (void *)b.basePtr) != CUPVA_ERROR_NONE)
        return false;
    if (dataOut)
        *dataOut = data;
    return true;
}

/** 1 / max |src[i]| (1 for all-zero input): the scale radarDequantNormalized applies. */
inline float radarDequantScale(const int32_t *src, size_t numInts)
{
    int64_t maxAbs = 1;
    for (size_t i = 0; i < numInts; ++i)
    {
        const int64_t a = std::llabs(static_cast<int64_t>(src[i]));
        if (a > maxAbs)
            maxAbs = a;
    }
    return 1.0f / static_cast<float>(maxAbs);
}

/** Interleaved 2S32 -> float, normalized by the max magnitude (all-zero input stays zero). The PVA path and
 *  the CPU reference both use it, so the Capon preprocessing sees identical inputs. */
inline bool radarDequantNormalized(const int32_t *src, size_t numInts, std::vector<float> &dst)
{
    if (!src)
        return false;
    dst.resize(numInts);
    const float inv = radarDequantScale(src, numInts);
    for (size_t i = 0; i < numInts; ++i) dst[i] = static_cast<float>(src[i]) * inv;
    return true;
}

#endif // RADAR_PIPELINE_HOST_UTILS_HPP
