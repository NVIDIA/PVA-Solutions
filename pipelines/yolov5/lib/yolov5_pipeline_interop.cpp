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

#include "yolov5_pipeline_interop.hpp"

// CudaInteropUtils.h declares C functions but carries no extern "C" guards of
// its own, so every C++ consumer has to add them. Its CUDA includes must be
// pulled in first: they contain templates, which cannot have C linkage.
#include <cuda_runtime.h>

extern "C" {
#include "CudaInteropUtils.h" // nvcvTensorWrapCudaBuf
}

#include <cstdio>

namespace yolov5_pipeline {

int PvaTrtInterop::wrapLetterboxOutput(const NVCVTensorRequirements &reqs, int modelHeight, int modelWidth)
{
    if (!inputPinned)
        return -1;

    // PVA must lay the plane out exactly as TensorRT expects to read it.
    const int64_t naturalPlaneStride = static_cast<int64_t>(modelHeight) * modelWidth * sizeof(float);
    if (reqs.strides[0] != naturalPlaneStride)
    {
        fprintf(stderr, "  PVA letterbox output stride (%ld) != natural plane stride (%ld)\n",
                static_cast<long>(reqs.strides[0]), static_cast<long>(naturalPlaneStride));
        return -1;
    }

    for (int s = 0; s < NUM_PIPE; s++)
    {
        if (nvcvTensorWrapCudaBuf(&reqs, trtInput[s], &lbOut[s]) != NVCV_SUCCESS)
        {
            fprintf(stderr, "  nvcvTensorWrapCudaBuf failed for letterbox output slot %d\n", s);
            for (int r = 0; r < NUM_PIPE; r++)
            {
                if (lbOut[r] != nullptr)
                {
                    nvcvTensorDecRef(lbOut[r], nullptr);
                    lbOut[r] = nullptr;
                }
            }
            return -1;
        }
    }

    return 0;
}

int PvaTrtInterop::wrapPrediction(int slot, int scale, const NVCVTensorRequirements &reqs, int gridH, int gridW)
{
    if (trtOutput[slot][scale] == nullptr)
        return -1;

    const int64_t naturalStride0 = static_cast<int64_t>(gridH) * gridW * sizeof(float);
    const int64_t naturalStride1 = static_cast<int64_t>(gridW) * sizeof(float);
    const int64_t naturalStride2 = static_cast<int64_t>(sizeof(float));
    if (reqs.strides[0] != naturalStride0 || reqs.strides[1] != naturalStride1 || reqs.strides[2] != naturalStride2)
        return -1;

    if (nvcvTensorWrapCudaBuf(&reqs, trtOutput[slot][scale], &predictions[slot][scale]) != NVCV_SUCCESS)
        return -1;

    return 0;
}

void PvaTrtInterop::release()
{
    // The views alias the buffers, so they go first.
    for (int s = 0; s < NUM_PIPE; s++)
    {
        for (int sc = 0; sc < NUM_YOLO_SCALES; sc++)
        {
            if (predictions[s][sc] != nullptr)
            {
                nvcvTensorDecRef(predictions[s][sc], nullptr);
                predictions[s][sc] = nullptr;
            }
        }
        if (lbOut[s] != nullptr)
        {
            nvcvTensorDecRef(lbOut[s], nullptr);
            lbOut[s] = nullptr;
        }
    }

    for (int s = 0; s < NUM_PIPE; s++)
    {
        if (trtInput[s] != nullptr)
        {
            if (inputPinned)
                cudaFreeHost(trtInput[s]);
            else
                cudaFree(trtInput[s]);
            trtInput[s] = nullptr;
        }
        for (int sc = 0; sc < NUM_YOLO_SCALES; sc++)
        {
            if (trtOutput[s][sc] != nullptr)
            {
                cudaFree(trtOutput[s][sc]);
                trtOutput[s][sc] = nullptr;
            }
        }
    }
}

} // namespace yolov5_pipeline
