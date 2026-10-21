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

#include "histogram_equalization_ref.h"

#include <nvcv/Tensor.hpp>
#include <pngUtils.hpp>

#include <ImageIO.h>

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

#define NBINS 256
using namespace std;

// get image width, height and line pitch from NVCVTensorRequirements
std::tuple<int, int, int> getImageShapeFromTensor(NVCVTensorRequirements *tensorRequirements)
{
    int32_t image_width, image_height, line_pitch;
    // Permute shape and strides to NCHW to robustly get H, W, and line pitch
    int64_t imageShapeNCHW[4];
    int64_t imageStridesNCHW[4];
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->shape, NVCV_TENSOR_NCHW, imageShapeNCHW);
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->strides, NVCV_TENSOR_NCHW, imageStridesNCHW);

    image_height = static_cast<int32_t>(imageShapeNCHW[2]); // H from NCHW
    image_width  = static_cast<int32_t>(imageShapeNCHW[3]); // W from NCHW
    line_pitch   = imageStridesNCHW[2] / (nvcv::DataType(tensorRequirements->dtype).bitsPerPixel() / 8);

    return std::make_tuple(image_width, image_height, line_pitch);
}

void histogram_ref(uint8_t *src, int *hist, int width, int height, int pitch)
{
    int i, j;

    uint8_t pix_s;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            pix_s = src[i * pitch + j];
            hist[pix_s] += 1;
        }
    }
}

void equalize_hist_ref(uint8_t *src, unsigned short *hist, uint8_t *dst, int32_t width, int32_t height, int pitch)
{
    int i, j;

    uint8_t pix_s, pix_h, pix_d;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {

            pix_s              = src[i * pitch + j];
            pix_h              = hist[pix_s];
            dst[i * pitch + j] = pix_h;
        }
    }
}

void normalize_hist_ref(int32_t *hist, uint16_t *lut, int32_t nbins, int32_t width, int32_t height)
{
    int i, j, i0 = -1;
    int32_t x, y, srcstride, dststride, delta = 0, scale = 1; //0,1

    for (i = 0; i < nbins; i++)
    {
        if (i0 < 0 && hist[i] > 0)
        {
            i0    = i;
            delta = hist[i0];
        }

        hist[i] += i > 0 ? hist[i - 1] : 0;

        /* A constant image makes delta == width * height; clamp the divisor to avoid /0. */
        const int32_t denom = width * height - delta;
        scale               = ((255 * (1 << 15)) / (denom > 0 ? denom : 1));

        int val0 = (hist[i] - delta) * scale;
        lut[i]   = (uint16_t)(val0 >> 15);
    }
}

int32_t getHistogramEqualizationRefData(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                        NVCVTensorHandle ref_outTensorHandle)
{
    NVCVTensorData inputTensorData, refTensorData;
    nvcvTensorExportData(inTensorHandle, &inputTensorData);
    nvcvTensorExportData(ref_outTensorHandle, &refTensorData);

    int width, height, pitch;
    auto shape_tuple = getImageShapeFromTensor(tensorRequirements);
    width            = std::get<0>(shape_tuple);
    height           = std::get<1>(shape_tuple);
    pitch            = std::get<2>(shape_tuple);

    uint8_t *input_h{nullptr}, *ref_h{nullptr};
    CupvaMemGetHostPointer((void **)&input_h, (void *)inputTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&ref_h, (void *)refTensorData.buffer.strided.basePtr);

    std::vector<int32_t> hist_ref_vec(NBINS, 0);
    std::vector<uint16_t> hist_ref_norm_vec(NBINS, 0);

    histogram_ref(&input_h[0], hist_ref_vec.data(), width, height, pitch);
    normalize_hist_ref(hist_ref_vec.data(), hist_ref_norm_vec.data(), NBINS, width, height);
    equalize_hist_ref(&input_h[0], hist_ref_norm_vec.data(), &ref_h[0], width, height, pitch);

    return 0;
}

int32_t loadHistogramEqualizationData(NVCVTensorHandle tensorHandle, const char *assetDir, const char *filename,
                                      const int32_t tensorSizeInBytes)
{
    int32_t err = 0;
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *tensor_h = nullptr;
    CupvaMemGetHostPointer((void **)&tensor_h, (void *)tensorData.buffer.strided.basePtr);

    int32_t height = tensorData.shape[0];
    int32_t width  = tensorData.shape[1];

    std::string assetPath = std::string(assetDir) + "/" + std::string(filename);

    err = loadGrayscalePNG<uint8_t>(assetPath, tensor_h, height, width);

    return err;
}

template<typename T>
int32_t checkHistogramEqualizationOutputImpl(NVCVTensorRequirements *tensorRequirements,
                                             NVCVTensorHandle inTensorHandle, NVCVTensorHandle refTensorHandle,
                                             NVCVTensorHandle outTensorHandle)
{
    int image_width, image_height, line_pitch;
    auto shape_tuple = getImageShapeFromTensor(tensorRequirements);
    image_width      = std::get<0>(shape_tuple);
    image_height     = std::get<1>(shape_tuple);
    line_pitch       = std::get<2>(shape_tuple);

    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(outTensorHandle, &pvaTensorData);

    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr);

    int i, j;
    int fail = 0;
    for (i = 0; i < image_height; i++)
    {
        for (j = 0; j < image_width; j++)
        {
            T ref_val = ref[(i * line_pitch) + j];
            T pva_val = pva[(i * line_pitch) + j];
            if (abs(static_cast<int>(ref_val) - static_cast<int>(pva_val)) > 0) // Cast to int for abs if T is unsigned
            {
                fail++;
                printf("Error at: [%d][%d], pva = %d, ref = %d\n", i, j, pva_val, ref_val);
            }
        }
    }
    return fail;
}

// Function to save histogram equalization output data to a file
int32_t saveHistogramEqualizationOutput(NVCVTensorHandle tensorHandle, const char *filePath, int image_width,
                                        int image_height)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *image_h = NULL;
    CupvaMemGetHostPointer((void **)&image_h, (void *)tensorData.buffer.strided.basePtr);

    //Write outputfiles for the visual check:
    std::string filePathStr = std::string(filePath);
    writeGrayscalePNG<uint8_t>(filePathStr, &image_h[0], image_height, image_width);

    return 0;
}

void generateHistogramEqualizationFixedImgInput(NVCVTensorRequirements *tensorRequirements,
                                                NVCVTensorHandle inTensorHandle, const char *assetDir)
{
    NVCVDataType inDataType;
    nvcvTensorGetDataType(inTensorHandle, &inDataType);
    auto shape_tuple     = getImageShapeFromTensor(tensorRequirements);
    int32_t image_width  = std::get<0>(shape_tuple);
    int32_t image_height = std::get<1>(shape_tuple);

    loadHistogramEqualizationData(inTensorHandle, assetDir, "/histogram_equalization/kodim08_1920x1280_GS.png",
                                  image_width * image_height * sizeof(uint8_t));
}

void generateRandomInput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle)
{
    auto shape_tuple     = getImageShapeFromTensor(tensorRequirements);
    int32_t image_width  = std::get<0>(shape_tuple);
    int32_t image_height = std::get<1>(shape_tuple);
    int32_t line_pitch   = std::get<2>(shape_tuple); // Pitch in elements for uint8_t

    const int32_t total_pixels = image_width * image_height;

    // 1. Generate a random histogram
    std::vector<int32_t> random_hist(NBINS, 0);
    std::random_device rd_hist;
    std::mt19937 gen_hist(rd_hist());
    // Distribution to pick a bin index (0 to NBINS-1)
    // std::uniform_int_distribution<int32_t> bin_dist(0, NBINS - 1); // Old uniform distribution

    // Use a normal (Gaussian) distribution
    double mean   = static_cast<double>(NBINS - 1) / 2.0;
    double stddev = static_cast<double>(NBINS - 1) / 6.0; // So that ~99.7% values are within [0, NBINS-1]
    std::normal_distribution<double> gaussian_dist(mean, stddev);

    for (int32_t i = 0; i < total_pixels; ++i)
    {
        // random_hist[bin_dist(gen_hist)]++; // Old uniform update
        double generated_value = gaussian_dist(gen_hist);
        int32_t bin_index      = static_cast<int32_t>(std::round(generated_value));

        // Clamp the bin_index to be within [0, NBINS - 1]
        bin_index = std::max(0, std::min(bin_index, NBINS - 1));
        random_hist[bin_index]++;
    }

    // 2. Create a list of pixel values to assign
    std::vector<uint8_t> pixel_values_to_assign;
    pixel_values_to_assign.reserve(total_pixels);

    for (int32_t bin_val = 0; bin_val < NBINS; ++bin_val)
    {
        for (int32_t count = 0; count < random_hist[bin_val]; ++count)
        {
            pixel_values_to_assign.push_back(static_cast<uint8_t>(bin_val));
        }
    }

    // 3. Shuffle the list
    std::random_device rd_shuffle;
    std::mt19937 gen_shuffle(rd_shuffle());
    std::shuffle(pixel_values_to_assign.begin(), pixel_values_to_assign.end(), gen_shuffle);

    // 4. Populate the tensor
    NVCVTensorData tensorData;
    nvcvTensorExportData(inTensorHandle, &tensorData);

    uint8_t *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    int k = 0;
    for (int32_t r = 0; r < image_height; ++r)
    {
        for (int32_t c = 0; c < image_width; ++c)
        {
            // Ensure we do not write past the allocated pixels or read past shuffled values
            if (k < total_pixels)
            {
                tensorPtr_h[r * line_pitch + c] = pixel_values_to_assign[k++];
            }
        }
    }
}

int32_t checkHistogramEqualizationOutput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                         NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle)
{
    nvcv::TensorShape shape(tensorRequirements->shape, tensorRequirements->rank, tensorRequirements->layout);

    NVCVDataType inDataType;
    nvcvTensorGetDataType(inTensorHandle, &inDataType);

    int32_t err = checkHistogramEqualizationOutputImpl<uint8_t>(tensorRequirements, inTensorHandle, refTensorHandle,
                                                                outTensorHandle);

    return err;
}