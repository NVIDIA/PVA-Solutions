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
 * GPU letterbox implemented as a small CUDA kernel. Uses 114 for padding.
 */

#include "letterbox_gpu.h"

#include "image_io.h"
#include "pipeline_error.h"

#include "letterbox_cuda.cuh"

#include <nvcv/Tensor.h>
#include <nvcv/TensorData.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <type_traits>

namespace yolov5_common {

namespace {

constexpr uint8_t PAD_VALUE      = 114;
constexpr int64_t MAX_BIN_PIXELS = 4 * 4096 * 4096; // ~67M pixels max

static bool pathEndsWithBin(const char *path)
{
    if (!path)
        return false;
    size_t len = std::strlen(path);
    if (len < 4u)
        return false;
    const char *suffix = path + len - 4;
    return (suffix[0] == '.' && (suffix[1] == 'b' || suffix[1] == 'B') && (suffix[2] == 'i' || suffix[2] == 'I') &&
            (suffix[3] == 'n' || suffix[3] == 'N'));
}

/** Cleanup for nvcvTensorWrapDataConstruct: frees managed buffer. */
static void cleanupManagedBuffer(void *ctx, const NVCVTensorData *)
{
    if (ctx)
        cudaFree(ctx);
}

/** No-op cleanup when wrapping a buffer we do not own (e.g. preloaded). */
static void cleanupNoOp(void *, const NVCVTensorData *) {}

struct ImageDeleter
{
    void operator()(uint8_t *pixels) const
    {
        freeImage(pixels);
    }
};

struct TensorDeleter
{
    void operator()(NVCVTensorHandle tensor) const
    {
        if (tensor)
            nvcvTensorDecRef(tensor, nullptr);
    }
};

using ImagePtr  = std::unique_ptr<uint8_t, ImageDeleter>;
using TensorPtr = std::unique_ptr<std::remove_pointer<NVCVTensorHandle>::type, TensorDeleter>;

} // namespace

int letterboxGPU(NVCVTensorHandle inputTensor, int srcW, int srcH, NVCVTensorHandle outputTensor, int modelW,
                 int modelH, LetterboxInfo *lbInfo, cudaStream_t stream)
{
    if (srcW <= 0 || srcH <= 0)
        return -1;

    float scale = std::min(static_cast<float>(modelW) / srcW, static_cast<float>(modelH) / srcH);
    int scaledW = static_cast<int>(std::round(srcW * scale));
    int scaledH = static_cast<int>(std::round(srcH * scale));
    scaledW     = std::max(1, std::min(scaledW, modelW));
    scaledH     = std::max(1, std::min(scaledH, modelH));
    int padX    = (modelW - scaledW) / 2;
    int padY    = (modelH - scaledH) / 2;

    if (lbInfo)
    {
        lbInfo->scaleX = scale;
        lbInfo->scaleY = scale;
        lbInfo->padX   = static_cast<float>(padX);
        lbInfo->padY   = static_cast<float>(padY);
        lbInfo->origW  = static_cast<int>(srcW);
        lbInfo->origH  = static_cast<int>(srcH);
    }

    NVCVTensorData inputData{}, outputData{};
    YOLOV5_NVCV_CHECK(nvcvTensorExportData(inputTensor, &inputData));
    YOLOV5_NVCV_CHECK(nvcvTensorExportData(outputTensor, &outputData));
    YOLOV5_CUDA_CHECK(launchLetterboxRgbU8(
        reinterpret_cast<const uint8_t *>(inputData.buffer.strided.basePtr), inputData.buffer.strided.strides[1], srcW,
        srcH, reinterpret_cast<uint8_t *>(outputData.buffer.strided.basePtr), outputData.buffer.strided.strides[1],
        modelW, modelH, scaledW, scaledH, padX, padY, PAD_VALUE, stream));
    return 0;
}

int loadImageLetterboxGPU(const char *imagePath, NVCVTensorHandle outputTensor, int modelW, int modelH,
                          LetterboxInfo *lbInfo, cudaStream_t stream, LetterboxGPUTimings *timings)
{
    using Clock = std::chrono::high_resolution_clock;

    // -------------------------------------------------------------------------
    // Pre-decoded .bin path: the payload is read into managed memory and
    // wrapped, so there is no explicit host-to-device copy.
    // -------------------------------------------------------------------------
    if (pathEndsWithBin(imagePath))
    {
        auto t0 = Clock::now();
        FILE *f = fopen(imagePath, "rb");
        if (!f)
        {
            fprintf(stderr, "Failed to open .bin: %s\n", imagePath);
            return -1;
        }
        int32_t w = 0, h = 0;
        if (fread(&w, sizeof(int32_t), 1, f) != 1 || fread(&h, sizeof(int32_t), 1, f) != 1)
        {
            fclose(f);
            fprintf(stderr, "Failed to read .bin header: %s\n", imagePath);
            return -1;
        }
        if (w <= 0 || h <= 0 || (int64_t)w * h > MAX_BIN_PIXELS)
        {
            fclose(f);
            fprintf(stderr, "Invalid .bin dimensions %dx%d: %s\n", w, h, imagePath);
            return -1;
        }
        size_t size          = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
        void *buf            = nullptr;
        cudaError_t allocErr = cudaMallocManaged(&buf, size);
        if (allocErr != cudaSuccess)
        {
            fclose(f);
            fprintf(stderr, "Failed to allocate managed memory for .bin (%s): %s\n", imagePath,
                    cudaGetErrorString(allocErr));
            return -1;
        }
        if (fread(buf, 1, size, f) != size)
        {
            cudaFree(buf);
            fclose(f);
            fprintf(stderr, "Failed to read .bin payload: %s\n", imagePath);
            return -1;
        }
        fclose(f);
        float readMs = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();

        NVCVTensorData wrapData            = {};
        wrapData.dtype                     = NVCV_DATA_TYPE_U8;
        wrapData.layout                    = NVCV_TENSOR_NHWC;
        wrapData.rank                      = 4;
        wrapData.shape[0]                  = 1;
        wrapData.shape[1]                  = h;
        wrapData.shape[2]                  = w;
        wrapData.shape[3]                  = 3;
        wrapData.bufferType                = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
        wrapData.buffer.strided.strides[3] = 1;
        wrapData.buffer.strided.strides[2] = 3;
        wrapData.buffer.strided.strides[1] = w * 3;
        wrapData.buffer.strided.strides[0] = static_cast<int64_t>(h) * w * 3;
        wrapData.buffer.strided.basePtr    = static_cast<NVCVByte *>(buf);

        NVCVTensorHandle inputHandle = nullptr;
        NVCVStatus wrapStatus        = nvcvTensorWrapDataConstruct(&wrapData, cleanupManagedBuffer, buf, &inputHandle);
        if (wrapStatus != NVCV_SUCCESS)
        {
            cudaFree(buf);
            fprintf(stderr, "NVCV error at %s:%d: %d\n", __FILE__, __LINE__, wrapStatus);
            return -1;
        }
        // The tensor owns the managed buffer from here on.
        TensorPtr inputTensor(inputHandle);

        // Synchronize before the input tensor, and with it the managed buffer,
        // is released.
        t0      = Clock::now();
        int ret = letterboxGPU(inputTensor.get(), w, h, outputTensor, modelW, modelH, lbInfo, stream);
        YOLOV5_CUDA_CHECK(cudaStreamSynchronize(stream));
        float resizePadMs = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();

        if (timings)
        {
            timings->decodeMs          = readMs;
            timings->h2dCopyMs         = 0.f; // Managed memory: no explicit copy
            timings->resizePadMs       = resizePadMs;
            timings->usedPredecodedBin = true;
        }
        return ret;
    }

    // -------------------------------------------------------------------------
    // PNG path: LodePNG decode + H->D + GPU letterbox
    // -------------------------------------------------------------------------
    int imgW = 0, imgH = 0;
    auto t0 = Clock::now();
    ImagePtr imgData(loadPngRgb(imagePath, &imgW, &imgH));
    if (!imgData)
    {
        return -1;
    }
    float decodeMs = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();

    NVCVTensorHandle inputHandle = nullptr;
    NVCVTensorRequirements reqs;
    int64_t shape[] = {1, imgH, imgW, 3};
    YOLOV5_NVCV_CHECK(nvcvTensorCalcRequirements(4, shape, NVCV_DATA_TYPE_U8, NVCV_TENSOR_NHWC, 0, 0, &reqs));
    YOLOV5_NVCV_CHECK(nvcvTensorConstruct(&reqs, nullptr, &inputHandle));
    TensorPtr inputTensor(inputHandle);

    NVCVTensorData tensorData;
    YOLOV5_NVCV_CHECK(nvcvTensorExportData(inputTensor.get(), &tensorData));
    void *devPtr      = tensorData.buffer.strided.basePtr;
    int64_t rowStride = tensorData.buffer.strided.strides[1];

    t0 = Clock::now();
    for (int y = 0; y < imgH; y++)
    {
        YOLOV5_CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t *>(devPtr) + y * rowStride, imgData.get() + y * imgW * 3,
                                          imgW * 3, cudaMemcpyHostToDevice, stream));
    }
    YOLOV5_CUDA_CHECK(cudaStreamSynchronize(stream));
    float h2dMs = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    imgData.reset();

    // Synchronize before the input tensor is released.
    t0      = Clock::now();
    int ret = letterboxGPU(inputTensor.get(), imgW, imgH, outputTensor, modelW, modelH, lbInfo, stream);
    YOLOV5_CUDA_CHECK(cudaStreamSynchronize(stream));
    float resizePadMs = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();

    if (timings)
    {
        timings->decodeMs    = decodeMs;
        timings->h2dCopyMs   = h2dMs;
        timings->resizePadMs = resizePadMs;
    }
    return ret;
}

int letterboxFromBufferGPU(const void *buffer, int w, int h, NVCVTensorHandle outputTensor, int modelW, int modelH,
                           LetterboxInfo *lbInfo, cudaStream_t stream)
{
    if (!buffer || w <= 0 || h <= 0 || (int64_t)w * h > MAX_BIN_PIXELS)
        return -1;

    NVCVTensorData wrapData            = {};
    wrapData.dtype                     = NVCV_DATA_TYPE_U8;
    wrapData.layout                    = NVCV_TENSOR_NHWC;
    wrapData.rank                      = 4;
    wrapData.shape[0]                  = 1;
    wrapData.shape[1]                  = h;
    wrapData.shape[2]                  = w;
    wrapData.shape[3]                  = 3;
    wrapData.bufferType                = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    wrapData.buffer.strided.strides[3] = 1;
    wrapData.buffer.strided.strides[2] = 3;
    wrapData.buffer.strided.strides[1] = w * 3;
    wrapData.buffer.strided.strides[0] = static_cast<int64_t>(h) * w * 3;
    wrapData.buffer.strided.basePtr    = static_cast<NVCVByte *>(const_cast<void *>(buffer));

    NVCVTensorHandle inputTensor = nullptr;
    YOLOV5_NVCV_CHECK(nvcvTensorWrapDataConstruct(&wrapData, cleanupNoOp, nullptr, &inputTensor));

    int ret = letterboxGPU(inputTensor, w, h, outputTensor, modelW, modelH, lbInfo, stream);
    nvcvTensorDecRef(inputTensor, nullptr);
    return ret;
}

} // namespace yolov5_common
