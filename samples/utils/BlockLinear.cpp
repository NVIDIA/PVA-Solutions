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

#include "BlockLinear.h"

#include <cupva_host_utils.hpp>

#include <vector>

namespace {
void plToBlImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    uint8_t *basePtr_h{nullptr};
    CupvaMemGetHostPointer((void **)&basePtr_h, (void *)inImageData.buffer.strided.planes[0].basePtr);
    std::array<uint8_t *, 2> input = {basePtr_h, basePtr_h + inImageData.buffer.strided.planes[0].height *
                                                                 inImageData.buffer.strided.planes[0].rowStride};

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(inImageData.format, &numPlanes);
    for (int32_t i = 0; i < numPlanes; i++)
    {
        int32_t height    = inImageData.buffer.strided.planes[i].height;
        int32_t width     = inImageData.buffer.strided.planes[i].width;
        int32_t linePitch = inImageData.buffer.strided.planes[i].rowStride;

        int32_t bpp;
        nvcvImageFormatGetPlaneBitsPerPixel(inImageData.format, i, &bpp);
        int32_t bytesPerPixel = bpp / 8;

        uint8_t *_input;
        std::vector<uint8_t> inputPacked;
        if (linePitch == width * bytesPerPixel)
        {
            _input = input[i];
        }
        else
        {
            // cupva_utils::CopyToPlane does not support line pitch to be larger than the image, so it needs an intermediate buffer
            inputPacked.resize(height * width * bytesPerPixel);
            for (int j = 0; j < height; j++)
            {
                std::memcpy(inputPacked.data() + j * width, input[i] + j * linePitch, width * bytesPerPixel);
            }
            _input = inputPacked.data();
        }
        // Copy from the raw buffer which is in pitch linear layout to the surface with block linear layout. The conversion occurs
        // during the copy. This API is per plane operation.
        cupva_utils::CopyToPlane(outImageData.buffer.strided.planes[i].basePtr, _input, width * bytesPerPixel, height);
    }
}

void blToPlImpl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
{
    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);

    NVCVImageData outImageData;
    nvcvImageExportData(outImageHandle, &outImageData);

    uint8_t *basePtr_h{nullptr};
    CupvaMemGetHostPointer((void **)&basePtr_h, (void *)outImageData.buffer.strided.planes[0].basePtr);
    std::array<uint8_t *, 2> output = {basePtr_h, basePtr_h + outImageData.buffer.strided.planes[0].height *
                                                                  outImageData.buffer.strided.planes[0].rowStride};

    int32_t numPlanes;
    nvcvImageFormatGetNumPlanes(outImageData.format, &numPlanes);
    for (int32_t i = 0; i < numPlanes; i++)
    {
        int32_t height    = outImageData.buffer.strided.planes[i].height;
        int32_t width     = outImageData.buffer.strided.planes[i].width;
        int32_t linePitch = outImageData.buffer.strided.planes[i].rowStride;

        int32_t bpp;
        nvcvImageFormatGetPlaneBitsPerPixel(inImageData.format, i, &bpp);
        int32_t bytesPerPixel = bpp / 8;

        uint8_t *_output;
        std::vector<uint8_t> outputPacked;
        if (linePitch == width * bytesPerPixel)
        {
            _output = output[i];
        }
        else
        {
            // cupva_utils::CopyFromPlane does not support line pitch to be larger than the image, so it needs an intermediate buffer
            outputPacked.resize(height * width * bytesPerPixel);
            _output = outputPacked.data();
        }
        // Copy from the surface with block linear layout to the raw buffer which is in pitch linear layout. The conversion occurs
        // during the copy. This API is per plane operation.
        cupva_utils::CopyFromPlane(_output, inImageData.buffer.strided.planes[i].basePtr, width * bytesPerPixel,
                                   height);

        if (linePitch != width * bytesPerPixel)
        {
            for (int j = 0; j < height; j++)
            {
                std::memcpy(output[i] + j * linePitch, _output + j * width * bytesPerPixel, width * bytesPerPixel);
            }
        }
    }
}

template<int32_t NumPlanes>
NVCVStatus allocBlSurfaceImpl(void **in, const int32_t imageWidth, const int32_t imageHeight)
{
    static_assert(NumPlanes == 1 || NumPlanes == 2, "NumPlanes must be either 1 or 2");

    try
    {
        cupva_utils::PlaneSize pc[NumPlanes];
        pc[0].height = imageHeight;
        pc[0].width  = imageWidth;

        if constexpr (NumPlanes == 2)
        {
            pc[1].height = imageHeight / 2;
            pc[1].width  = imageWidth;
        }

        *in = cupva_utils::AllocSurface(pc, NumPlanes, cupva::SurfaceFormatType::BLOCK_LINEAR);
        return NVCV_SUCCESS;
    }
    catch (const std::runtime_error &e)
    {
        return NVCV_ERROR_INTERNAL;
    }
}

NVCVImageData nvcvImageDataConstructBLImpl(void *devPtr, int32_t imageWidth, int32_t imageHeight,
                                           NVCVImageFormat format)
{
    NVCVImageData imageData;
    NVCVImageFormat _format;
    // need to convert to their PL layouts since the potential called nvcvImageWrapDataConstruct does not support BL formats
    if (format == NVCV_IMAGE_FORMAT_NV12_BL)
    {
        _format = NVCV_IMAGE_FORMAT_NV12;
    }
    else if (format == NVCV_IMAGE_FORMAT_NV12_ER_BL)
    {
        _format = NVCV_IMAGE_FORMAT_NV12_ER;
    }
    else if (format == NVCV_IMAGE_FORMAT_U8_BL)
    {
        _format = NVCV_IMAGE_FORMAT_U8;
    }
    else if (format == NVCV_IMAGE_FORMAT_Y8_BL)
    {
        _format = NVCV_IMAGE_FORMAT_Y8;
    }
    else if (format == NVCV_IMAGE_FORMAT_Y8_ER_BL)
    {
        _format = NVCV_IMAGE_FORMAT_Y8_ER;
    }

    imageData.format     = _format;
    imageData.bufferType = NVCV_IMAGE_BUFFER_STRIDED_CUDA;

    cupva::mem::SurfaceAttributes attr;
    cupva::mem::GetSurfaceAttributes(devPtr, attr);

    imageData.buffer.strided.numPlanes = attr.numPlanes;

    // For single or Y plane, it has the image's original width and height and having implied dtype of NVCV_DATA_TYPE_U8.
    imageData.buffer.strided.planes[0].width     = imageWidth;
    imageData.buffer.strided.planes[0].height    = imageHeight;
    imageData.buffer.strided.planes[0].rowStride = attr.planes[0].linePitch;
    imageData.buffer.strided.planes[0].basePtr   = (NVCVByte *)devPtr;

    if (format == NVCV_IMAGE_FORMAT_NV12_BL || format == NVCV_IMAGE_FORMAT_NV12_ER_BL)
    {
        // For UV plane, it has the image's half of width and height and having implied dtype of NVCV_DATA_TYPE_2U8.
        imageData.buffer.strided.planes[1].width     = imageWidth / 2;
        imageData.buffer.strided.planes[1].height    = imageHeight / 2;
        imageData.buffer.strided.planes[1].rowStride = attr.planes[1].linePitch;
        imageData.buffer.strided.planes[1].basePtr   = (NVCVByte *)devPtr + attr.planes[1].offset;
    }

    return imageData;
}
} // namespace

void plToBl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
{
    plToBlImpl(inImageHandle, outImageHandle);
}

void blToPl(NVCVImageHandle inImageHandle, NVCVImageHandle outImageHandle)
{
    blToPlImpl(inImageHandle, outImageHandle);
}

NVCVStatus allocBlSurfaceNv12(void **in, const int32_t imageWidth, const int32_t imageHeight)
{
    return allocBlSurfaceImpl<2>(in, imageWidth, imageHeight);
}

NVCVImageData nvcvImageDataConstructBL(void *devPtr, int32_t imageWidth, int32_t imageHeight, NVCVImageFormat format)
{
    return nvcvImageDataConstructBLImpl(devPtr, imageWidth, imageHeight, format);
}

NVCVStatus nvcvImageConstructBL(const NVCVImageRequirements *reqs, NVCVImageHandle *handle, void **devPtr)
{
    const int32_t width{reqs->width};
    const int32_t height{reqs->height};
    void *blSurfaceDevPtr{nullptr};

    if (reqs->format == NVCV_IMAGE_FORMAT_NV12_BL || reqs->format == NVCV_IMAGE_FORMAT_NV12_ER_BL)
    {
        if (reqs->width % 2 || reqs->height % 2)
        {
            printf("%s only accepts even number of width or height.\n", __FUNCTION__);
            return NVCV_ERROR_INVALID_ARGUMENT;
        }
        NVCVStatus ret = allocBlSurfaceImpl<2>(&blSurfaceDevPtr, width, height);
        if (ret != NVCV_SUCCESS)
        {
            return ret;
        }
    }
    else if (reqs->format == NVCV_IMAGE_FORMAT_U8_BL || reqs->format == NVCV_IMAGE_FORMAT_Y8_BL ||
             reqs->format == NVCV_IMAGE_FORMAT_Y8_ER_BL)
    {
        NVCVStatus ret = allocBlSurfaceImpl<1>(&blSurfaceDevPtr, width, height);
        if (ret != NVCV_SUCCESS)
        {
            return ret;
        }
    }
    else
    {
        printf(
            "%s only accepts format to be NVCV_IMAGE_FORMAT_NV12_BL or NVCV_IMAGE_FORMAT_NV12_ER_BL or NVCV_IMAGE_FORMAT_U8_BL \
                or NVCV_IMAGE_FORMAT_Y8_BL or NVCV_IMAGE_FORMAT_Y8_ER_BL.\n",
            __FUNCTION__);
        return NVCV_ERROR_INVALID_IMAGE_FORMAT;
    }

    if (devPtr != nullptr)
    {
        *devPtr = blSurfaceDevPtr;
    }

    NVCVImageData imageData = nvcvImageDataConstructBL(blSurfaceDevPtr, width, height, reqs->format);

    nvcvImageWrapDataConstruct(&imageData, nvcvImageConstructBLCleanup, NULL, handle);

    return NVCV_SUCCESS;
}

void nvcvImageConstructBLCleanup(void *ctx, const NVCVImageData *data)
{
    void *blSurfaceDevPtr = (void *)data->buffer.strided.planes[0].basePtr;
    cupvaError_t ret      = CupvaMemFree(blSurfaceDevPtr);
    if (ret != CUPVA_ERROR_NONE)
    {
        printf("Failed to free the block linear surface.\n");
    }
}