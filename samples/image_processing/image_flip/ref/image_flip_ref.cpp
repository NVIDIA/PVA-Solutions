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

#include "image_flip_ref.h"

#include <nvcv/Image.hpp>
#include <pngUtils.hpp>

#include <ImageIO.h>
#include <cupva_host.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace std;

// get image width, height and line pitch from NVCVImageRequirements
std::tuple<int, int, int> getImageShapeFromImage(NVCVImageRequirements *imageRequirements)
{
    int32_t image_width  = static_cast<int32_t>(imageRequirements->width);
    int32_t image_height = static_cast<int32_t>(imageRequirements->height);

    // For simple grayscale images, line pitch equals width
    // For more complex formats, this would need to be calculated from the format
    int32_t line_pitch = image_width;

    return std::make_tuple(image_width, image_height, line_pitch);
}

// Reference implementation for horizontal flip
void flip_horizontal_ref(uint8_t *src, uint8_t *dst, int width, int height, int pitch)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            dst[i * pitch + j] = src[i * pitch + (width - 1 - j)];
        }
    }
}

// Reference implementation for vertical flip
void flip_vertical_ref(uint8_t *src, uint8_t *dst, int width, int height, int pitch)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            dst[i * pitch + j] = src[(height - 1 - i) * pitch + j];
        }
    }
}

// Reference implementation for both horizontal and vertical flip
void flip_both_ref(uint8_t *src, uint8_t *dst, int width, int height, int pitch)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            dst[i * pitch + j] = src[(height - 1 - i) * pitch + (width - 1 - j)];
        }
    }
}

void getImageFlipRefData(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                         NVCVImageHandle refImageHandle, PVAFlipDirection flipDirection)
{
    NVCVImageData inputImageData, refImageData;
    nvcvImageExportData(inImageHandle, &inputImageData);
    nvcvImageExportData(refImageHandle, &refImageData);

    int width, height, pitch;
    auto shape_tuple = getImageShapeFromImage(imageRequirements);
    width            = std::get<0>(shape_tuple);
    height           = std::get<1>(shape_tuple);
    pitch            = std::get<2>(shape_tuple);

    uint8_t *input_h{nullptr}, *ref_h{nullptr};
    CupvaMemGetHostPointer((void **)&input_h, (void *)inputImageData.buffer.strided.planes[0].basePtr);
    CupvaMemGetHostPointer((void **)&ref_h, (void *)refImageData.buffer.strided.planes[0].basePtr);

    // Choose flip mode based on runtime parameter
    switch (flipDirection)
    {
    case PVA_FLIP_HORIZONTAL:
        flip_horizontal_ref(&input_h[0], &ref_h[0], width, height, pitch);
        break;
    case PVA_FLIP_VERTICAL:
        flip_vertical_ref(&input_h[0], &ref_h[0], width, height, pitch);
        break;
    case PVA_FLIP_BOTH:
        flip_both_ref(&input_h[0], &ref_h[0], width, height, pitch);
        break;
    default:
        printf("Error: Invalid flip direction %d\n", flipDirection);
        break;
    }
}

void generateImageFlipRandomInput(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle)
{
    NVCVImageData inputImageData;
    nvcvImageExportData(inImageHandle, &inputImageData);

    int width, height, pitch;
    auto shape_tuple = getImageShapeFromImage(imageRequirements);
    width            = std::get<0>(shape_tuple);
    height           = std::get<1>(shape_tuple);
    pitch            = std::get<2>(shape_tuple);

    uint8_t *input_h{nullptr};
    CupvaMemGetHostPointer((void **)&input_h, (void *)inputImageData.buffer.strided.planes[0].basePtr);

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);

    // Fill image with random values
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            input_h[i * pitch + j] = dis(gen);
        }
    }
}

template<typename T>
int32_t checkImageFlipOutputImpl(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                                 NVCVImageHandle refImageHandle, NVCVImageHandle outImageHandle)
{
    int image_width, image_height, line_pitch;
    auto shape_tuple = getImageShapeFromImage(imageRequirements);
    image_width      = std::get<0>(shape_tuple);
    image_height     = std::get<1>(shape_tuple);
    line_pitch       = std::get<2>(shape_tuple);

    NVCVImageData refImageData;
    nvcvImageExportData(refImageHandle, &refImageData);

    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refImageData.buffer.strided.planes[0].basePtr);

    NVCVImageData pvaImageData;
    nvcvImageExportData(outImageHandle, &pvaImageData);

    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaImageData.buffer.strided.planes[0].basePtr);

    int fail = 0;
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            T ref_val = ref[(i * line_pitch) + j];
            T pva_val = pva[(i * line_pitch) + j];
            if (abs(static_cast<int>(ref_val) - static_cast<int>(pva_val)) > 0)
            {
                fail++;
                if (fail < 10) // Only print first 10 errors to avoid spam
                {
                    printf("Error at: [%d][%d], pva = %d, ref = %d\n", i, j, pva_val, ref_val);
                }
            }
        }
    }
    return fail;
}

int32_t checkImageFlipOutput(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                             NVCVImageHandle refImageHandle, NVCVImageHandle outImageHandle)
{
    // For image requirements, we need to extract data type from format
    NVCVImageFormat format = imageRequirements->format;
    if (format == NVCV_IMAGE_FORMAT_U8 || format == NVCV_IMAGE_FORMAT_Y8 || format == NVCV_IMAGE_FORMAT_Y8_ER)
    {
        return checkImageFlipOutputImpl<uint8_t>(imageRequirements, inImageHandle, refImageHandle, outImageHandle);
    }
    else
    {
        printf("Unsupported image format for image flip output check\n");
        return -1;
    }
}

void generateImageFlipFixedImgInput(NVCVImageRequirements *imageRequirements, NVCVImageHandle inImageHandle,
                                    const char *assetDir)
{
    NVCVImageData imageData;
    nvcvImageExportData(inImageHandle, &imageData);

    uint8_t *image_h = nullptr;
    CupvaMemGetHostPointer((void **)&image_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    int32_t height = static_cast<int32_t>(imageRequirements->height);
    int32_t width  = static_cast<int32_t>(imageRequirements->width);

    // Try to load a default test image from assets
    std::string assetPath = std::string(assetDir) + "/image_flip/kodim08_1920x1280_GS.png";

    int32_t err = loadGrayscalePNG<uint8_t>(assetPath, image_h, height, width);
    if (err != 0)
    {
        printf("Note: Failed to load PNG from %s, using generated test pattern\n", assetPath.c_str());
        // Fallback to checkboard pattern if PNG loading fails
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                // Create a simple checkboard pattern for visual verification
                image_h[i * width + j] = ((i / 8 + j / 8) % 2) * 255;
            }
        }
    }
    else
    {
        printf("Successfully loaded PNG image from %s\n", assetPath.c_str());
    }
}

void saveImageFlipOutput(NVCVImageHandle imageHandle, const char *filename, int32_t width, int32_t height)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    uint8_t *image_h = nullptr;
    CupvaMemGetHostPointer((void **)&image_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    // Save as PNG using pngUtils
    int32_t err = writeGrayscalePNG<uint8_t>(filename, image_h, height, width);
    if (err != 0)
    {
        printf("Error: Failed to save PNG to %s (error code: %d)\n", filename, err);
    }
    else
    {
        printf("Successfully saved PNG image to %s\n", filename);
    }
}
