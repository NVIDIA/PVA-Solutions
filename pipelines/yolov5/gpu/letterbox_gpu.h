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

/*
 * GPU letterbox (resize + pad) using the sample's CUDA kernel.
 */

#ifndef YOLOV5_COMMON_LETTERBOX_GPU_H
#define YOLOV5_COMMON_LETTERBOX_GPU_H

#include "yolov5_common.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

namespace yolov5_common {

/** Optional timings for loadImageLetterboxGPU (all in ms). */
struct LetterboxGPUTimings
{
    float decodeMs         = 0.f;   /**< PNG decode or .bin file-read time. */
    float h2dCopyMs        = 0.f;   /**< Host-to-device copy of the decoded image. */
    float resizePadMs      = 0.f;   /**< Letterbox kernel time, including stream synchronization. */
    bool usedPredecodedBin = false; /**< True for .bin input (decodeMs is file read; h2dCopyMs is zero). */
};

/**
 * Letterbox from an input GPU tensor to an output GPU tensor.
 * Input and output are NHWC uint8, [1, H, W, 3]. The work is enqueued on
 * `stream` and is not synchronized.
 *
 * @param inputTensor   Input tensor [1, srcH, srcW, 3] uint8 (already on GPU).
 * @param srcW          Input image width .
 * @param srcH          Input image height.
 * @param outputTensor  Output tensor [1, modelH, modelW, 3] uint8 (caller-allocated).
 * @param modelW        Output width (e.g. 672).
 * @param modelH        Output height (e.g. 672).
 * @param lbInfo        Output letterbox params for coordinate remapping (scale, pad, orig size).
 * @param stream        CUDA stream.
 * @return 0 on success, -1 on error.
 */
int letterboxGPU(NVCVTensorHandle inputTensor, int srcW, int srcH, NVCVTensorHandle outputTensor, int modelW,
                 int modelH, LetterboxInfo *lbInfo, cudaStream_t stream);

/**
 * Load image from path, upload to GPU, and letterbox into output tensor.
 * - .bin (pre-decoded): 8-byte header (int32 width, int32 height) + raw RGB,
 *   read into managed memory, so there is no explicit host-to-device copy.
 * - PNG: LodePNG decode + host-to-device copy + GPU resize+pad.
 * The stream is synchronized before returning.
 *
 * @param imagePath     Path to image file.
 * @param outputTensor  Output tensor [1, modelH, modelW, 3] uint8 (caller-allocated).
 * @param modelW        Output width.
 * @param modelH        Output height.
 * @param lbInfo        Output letterbox params.
 * @param stream        CUDA stream.
 * @param timings       Optional; if non-NULL, filled with timing breakdown.
 * @return 0 on success, -1 on error.
 */
int loadImageLetterboxGPU(const char *imagePath, NVCVTensorHandle outputTensor, int modelW, int modelH,
                          LetterboxInfo *lbInfo, cudaStream_t stream, LetterboxGPUTimings *timings = nullptr);

/**
 * Letterbox from a preloaded RGB buffer (e.g. from preloadBinImages) into the output tensor.
 * The buffer must be device-accessible (e.g. pinned or managed memory). Caller keeps ownership.
 * The work is enqueued on `stream` and is not synchronized.
 *
 * @param buffer       Device-accessible RGB buffer (row-major, w*h*3 bytes).
 * @param w            Width.
 * @param h            Height.
 * @param outputTensor Output tensor [1, modelH, modelW, 3] uint8 (caller-allocated).
 * @param modelW       Output width.
 * @param modelH       Output height.
 * @param lbInfo       Output letterbox params.
 * @param stream       CUDA stream.
 * @return 0 on success, -1 on error.
 */
int letterboxFromBufferGPU(const void *buffer, int w, int h, NVCVTensorHandle outputTensor, int modelW, int modelH,
                           LetterboxInfo *lbInfo, cudaStream_t stream);

} // namespace yolov5_common

#endif // YOLOV5_COMMON_LETTERBOX_GPU_H
