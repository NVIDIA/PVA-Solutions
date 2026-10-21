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

#include "orb_desc_extractor_ref.h"

#include "../../../../src/operator/image_processing/orb_desc_extractor/device/orb_desc_extr_patterns.h"

#include <nvcv/Tensor.hpp>

#include <OpORBDescExtractor.h>

#include <bitset>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

namespace {

const int32_t DESCRIPTOR_ARRAY_NUM_BITS = PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH * 8;
const int32_t PATCH_CENTROID_SIZE       = 15;
const int32_t PATCH_CENTER_X_OFFSET     = 19;
const int32_t PATCH_CENTER_Y_OFFSET     = 19;
const int32_t ANGLE_QBITS               = 6;
const int32_t ERROR_THRESHOLD_PERCENT   = 5;

template<typename T>
void generateRandomImageImpl(NVCVImageHandle imageHandle)
{
    uint32_t seed = 365;
    srand(seed);

    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);

    T *imagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&imagePtr_h, (void *)imageData.buffer.strided.planes[0].basePtr);

    NVCVImageFormat format = imageData.format;
    int32_t numPlanes      = imageData.buffer.strided.numPlanes;

    int64_t minVal = std::numeric_limits<T>::min();
    int64_t maxVal = std::numeric_limits<T>::max();
    // fix limits for image formats with special range
    if (format == NVCV_IMAGE_FORMAT_Y8)
    {
        minVal = 16;
        maxVal = 235;
    }
    else if (format == NVCV_IMAGE_FORMAT_Y16)
    {
        minVal = 4096;
        maxVal = 60160;
    }

    for (int32_t p = 0; p < numPlanes; p++)
    {
        int32_t height    = imageData.buffer.strided.planes[p].height;
        int32_t width     = imageData.buffer.strided.planes[p].width;
        int32_t linePitch = imageData.buffer.strided.planes[p].rowStride / sizeof(T);

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(format, p, &numChannels);

        for (int32_t i = 0; i < height; i++)
        {
            for (int32_t j = 0; j < width; j++)
            {
                for (int32_t c = 0; c < numChannels; c++)
                {
                    int64_t val                                     = (int64_t)rand() % (maxVal - minVal + 1) + minVal;
                    imagePtr_h[i * linePitch + j * numChannels + c] = static_cast<T>(val);
                }
            }
        }
        imagePtr_h += height * linePitch;
    }
}

void generateORBDescExtractorRandomFeaturesImpl(NVCVImageHandle imageHandle, NVCVTensorHandle inCornersTensorHandle)
{
    NVCVImageData imageData;
    nvcvImageExportData(imageHandle, &imageData);
    int32_t height = imageData.buffer.strided.planes[0].height;
    int32_t width  = imageData.buffer.strided.planes[0].width;

    NVCVTensorData inCornersTensorData;
    nvcvTensorExportData(inCornersTensorHandle, &inCornersTensorData);
    float *featuresPtr_h = NULL;

    int32_t numFeatures  = inCornersTensorData.shape[0];
    int32_t keypointSize = inCornersTensorData.shape[1];
    int32_t linePitch    = inCornersTensorData.buffer.strided.strides[0] / sizeof(float);

    CupvaMemGetHostPointer((void **)&featuresPtr_h, (void *)inCornersTensorData.buffer.strided.basePtr);
    memset(featuresPtr_h, 0, linePitch * keypointSize * sizeof(float));

    uint32_t seed = 250;
    srand(seed);

    for (int32_t j = 0; j < numFeatures; j++)
    {
        int32_t idx                           = j;
        int32_t x                             = rand() % width;
        int32_t y                             = rand() % height;
        featuresPtr_h[idx * keypointSize + 0] = x;
        featuresPtr_h[idx * keypointSize + 1] = y;
    }
}

// CUDA does round-to-nearest-even, C/C++ functions that do the same are roundeven*, however they are available
// only in recent C23 which may not be readily available, therefore we need our own implementation of roundeven
template<typename T>
inline T roundEven(T val)
{
    T rounded = std::round(val);
    if (std::abs(rounded - val) == T(0.5))
    {
        if (static_cast<int64_t>(rounded) & 1)
        {
            rounded -= std::copysign(T(1.0), val);
        }
    }
    return rounded;
}

std::vector<int32_t> getCircleLimits()
{
    std::vector<int32_t> circleLimits(PATCH_CENTROID_SIZE + 2);

    for (int32_t v = 0; v <= (int32_t)std::floor(PATCH_CENTROID_SIZE * std::sqrt(2.f) / 2 + 1); ++v)
    {
        circleLimits[v] = roundEven(std::sqrt(PATCH_CENTROID_SIZE * PATCH_CENTROID_SIZE - v * v));
    }

    for (int32_t v = PATCH_CENTROID_SIZE, v0 = 0; v >= (int32_t)std::ceil(PATCH_CENTROID_SIZE * std::sqrt(2.f) / 2);
         --v)
    {
        while (circleLimits[v0] == circleLimits[v0 + 1])
        {
            ++v0;
        }

        circleLimits[v] = v0;
        ++v0;
    }

    return circleLimits;
}

template<typename T>
T extendBorder(T *in, int32_t width, int32_t height, int32_t linePitch, int32_t x, int32_t y,
               const NVCVBorderType borderType, int32_t borderValue)
{
    switch (borderType)
    {
    case NVCV_BORDER_REPLICATE:
    {
        int32_t idx_x = std::clamp(x, 0, width - 1);
        int32_t idx_y = std::clamp(y, 0, height - 1);
        return in[idx_y * linePitch + idx_x];
    }
    case NVCV_BORDER_CONSTANT:
    {
        T val = (x < 0 || x >= width || y < 0 || y >= height) ? static_cast<T>(borderValue) : in[y * linePitch + x];
        return val;
    }
    default:
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid border type!");
    }
    }
}

template<typename T>
std::pair<float, float> getOrientationAngle(T *in, int32_t x, int32_t y, int32_t width, int32_t height,
                                            int32_t linePitch, NVCVBorderType borderType, int32_t borderValue,
                                            const std::vector<int32_t> &circleLimits)
{

    int32_t moment01 = 0, moment10 = 0;
    // calculate the moments
    for (int32_t v = -PATCH_CENTROID_SIZE; v <= PATCH_CENTROID_SIZE; v++)
    {
        int32_t limit = circleLimits[std::abs(v)];
        for (int32_t u = -limit; u <= limit; u++)
        {
            T val = extendBorder(in, width, height, linePitch, x + u, y + v, borderType, borderValue);
            moment10 += u * val;
            moment01 += v * val;
        }
    }

    float m01  = moment01;
    float m10  = moment10;
    float norm = std::sqrt(m01 * m01 + m10 * m10) + 1e-8f;
    float sina = m01 / norm;
    float cosa = m10 / norm;
    return std::make_pair(sina, cosa);
}

template<typename T>
PVABriefDescriptor computeDescriptor(T *in, int32_t width, int32_t height, int32_t linePitch, int32_t x, int32_t y,
                                     const std::vector<int32_t> &circleLimits, bool enableRBRIEF,
                                     NVCVBorderType borderType, int32_t borderValue)
{
    PVABriefDescriptor descriptor;
    memset(descriptor.data, 0, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH * sizeof(uint8_t));

    // Initialize with default rotation angle 0
    float sina = 0.0f;
    float cosa = 1.0f;

    if (enableRBRIEF)
    {
        std::tie(sina, cosa) =
            getOrientationAngle<T>(in, x, y, width, height, linePitch, borderType, borderValue, circleLimits);
    }
    int sina_qbit = roundEven(sina * (1 << ANGLE_QBITS));
    int cosa_qbit = roundEven(cosa * (1 << ANGLE_QBITS));

    for (int32_t i = 0, stringIndex = 0, bitIndex = 0; i < DESCRIPTOR_ARRAY_NUM_BITS; i++)
    {
        int32_t u1 = orbBitPatterns[i * 4 + 0];
        int32_t v1 = orbBitPatterns[i * 4 + 1];
        int32_t u2 = orbBitPatterns[i * 4 + 2];
        int32_t v2 = orbBitPatterns[i * 4 + 3];

        int32_t x1, y1, x2, y2;
        if (enableRBRIEF)
        {
            // use fixed-point representation to apply rotation
            int32_t x1_qbit = cosa_qbit * u1 - sina_qbit * v1;
            int32_t y1_qbit = sina_qbit * u1 + cosa_qbit * v1;
            int32_t x2_qbit = cosa_qbit * u2 - sina_qbit * v2;
            int32_t y2_qbit = sina_qbit * u2 + cosa_qbit * v2;
            // round to get int indices
            x1 = ((x1_qbit + (1 << ANGLE_QBITS) / 2) >> ANGLE_QBITS) + x;
            y1 = ((y1_qbit + (1 << ANGLE_QBITS) / 2) >> ANGLE_QBITS) + y;
            x2 = ((x2_qbit + (1 << ANGLE_QBITS) / 2) >> ANGLE_QBITS) + x;
            y2 = ((y2_qbit + (1 << ANGLE_QBITS) / 2) >> ANGLE_QBITS) + y;
        }
        else
        {
            x1 = u1 + x;
            y1 = v1 + y;
            x2 = u2 + x;
            y2 = v2 + y;
        }

        T pixel1    = extendBorder(in, width, height, linePitch, x1, y1, borderType, borderValue);
        T pixel2    = extendBorder(in, width, height, linePitch, x2, y2, borderType, borderValue);
        int32_t val = (pixel1 < pixel2) ? 1 : 0;

        descriptor.data[stringIndex] |= (val << bitIndex);

        bitIndex++;
        if (bitIndex == 8)
        {
            stringIndex++;
            bitIndex = 0;
        }
    }

    return descriptor;
}

template<typename T>
void computeDescriptors(NVCVImageHandle inImageHandle, NVCVTensorHandle inCornersTensorHandle,
                        NVCVTensorHandle refDescriptorsTensorHandle, bool enableRBRIEF, NVCVBorderType borderType,
                        int32_t borderValue)
{

    NVCVImageData inImageData;
    nvcvImageExportData(inImageHandle, &inImageData);
    T *inImagePtr_h = NULL;
    CupvaMemGetHostPointer((void **)&inImagePtr_h, (void *)inImageData.buffer.strided.planes[0].basePtr);

    int32_t height    = inImageData.buffer.strided.planes[0].height;
    int32_t width     = inImageData.buffer.strided.planes[0].width;
    int32_t linePitch = inImageData.buffer.strided.planes[0].rowStride / sizeof(T);

    // preprocess the input corners tensor
    NVCVTensorData inCornersTensorData;
    nvcvTensorExportData(inCornersTensorHandle, &inCornersTensorData);
    float *featuresPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&featuresPtr_h, (void *)inCornersTensorData.buffer.strided.basePtr);

    int32_t numFeatures  = inCornersTensorData.shape[0];
    int32_t keypointSize = inCornersTensorData.shape[1];

    NVCVTensorData refDescriptorsTensorData;
    nvcvTensorExportData(refDescriptorsTensorHandle, &refDescriptorsTensorData);
    uint8_t *refDescriptorsPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&refDescriptorsPtr_h, (void *)refDescriptorsTensorData.buffer.strided.basePtr);

    // get circle limits for centroid moment calculation
    auto circleLimits = getCircleLimits();

    for (int32_t i = 0; i < numFeatures; i++)
    {
        int32_t x = static_cast<int32_t>(featuresPtr_h[i * keypointSize + 0]);
        int32_t y = static_cast<int32_t>(featuresPtr_h[i * keypointSize + 1]);

        PVABriefDescriptor descriptor = computeDescriptor(inImagePtr_h, width, height, linePitch, x, y, circleLimits,
                                                          enableRBRIEF, borderType, borderValue);
        // copy descriptor to ref descriptors tensor
        for (int32_t j = 0; j < PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH; j++)
        {
            refDescriptorsPtr_h[i * PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH + j] = descriptor.data[j];
        }
    }
}

int32_t computeHammingDistance(uint8_t *a, uint8_t *b, int32_t length)
{
    int32_t distance = 0;
    for (int32_t i = 0; i < length; i++)
    {
        distance += std::bitset<8>(a[i] ^ b[i]).count();
    }
    return distance;
}

} // namespace

void generateORBDescExtractorRandomInput(NVCVImageHandle imageHandle, NVCVTensorHandle inCornersTensorHandle)
{
    // generate input image pyramid
    if (imageHandle == NULL)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input image handle pointer is NULL!");
    }

    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);

    switch (format)
    {
    case NVCV_IMAGE_FORMAT_U8:
    case NVCV_IMAGE_FORMAT_Y8:
    case NVCV_IMAGE_FORMAT_Y8_ER:
        generateRandomImageImpl<uint8_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_S8:
        generateRandomImageImpl<int8_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_U16:
    case NVCV_IMAGE_FORMAT_Y16:
    case NVCV_IMAGE_FORMAT_Y16_ER:
        generateRandomImageImpl<uint16_t>(imageHandle);
        break;
    case NVCV_IMAGE_FORMAT_S16:
        generateRandomImageImpl<int16_t>(imageHandle);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input image format!");
    }

    // generate input features
    generateORBDescExtractorRandomFeaturesImpl(imageHandle, inCornersTensorHandle);
}

void ORBDescExtractorRef(NVCVImageHandle imageHandle, NVCVTensorHandle inCornersTensorHandle,
                         NVCVTensorHandle refDescriptorsTensorHandle, uint32_t disableRBRIEFFlag,
                         NVCVBorderType borderType, int32_t borderValue)
{
    if (imageHandle == NULL)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input image handle pointer is NULL!");
    }

    NVCVImageFormat format;
    nvcvImageGetFormat(imageHandle, &format);

    bool enableRBRIEF = !disableRBRIEFFlag;

    switch (format)
    {
    case NVCV_IMAGE_FORMAT_U8:
    case NVCV_IMAGE_FORMAT_Y8:
    case NVCV_IMAGE_FORMAT_Y8_ER:
        computeDescriptors<uint8_t>(imageHandle, inCornersTensorHandle, refDescriptorsTensorHandle, enableRBRIEF,
                                    borderType, borderValue);
        break;
    case NVCV_IMAGE_FORMAT_S8:
        computeDescriptors<int8_t>(imageHandle, inCornersTensorHandle, refDescriptorsTensorHandle, enableRBRIEF,
                                   borderType, borderValue);
        break;
    case NVCV_IMAGE_FORMAT_U16:
    case NVCV_IMAGE_FORMAT_Y16:
    case NVCV_IMAGE_FORMAT_Y16_ER:
        computeDescriptors<uint16_t>(imageHandle, inCornersTensorHandle, refDescriptorsTensorHandle, enableRBRIEF,
                                     borderType, borderValue);
        break;
    case NVCV_IMAGE_FORMAT_S16:
        computeDescriptors<int16_t>(imageHandle, inCornersTensorHandle, refDescriptorsTensorHandle, enableRBRIEF,
                                    borderType, borderValue);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input image format!");
    }
}

int32_t checkORBDescExtractorOutput(NVCVTensorHandle refDescriptorsTensorHandle,
                                    NVCVTensorHandle pvaDescriptorsTensorHandle)
{
    // Maximum expected number of bits different between descriptors
    constexpr int32_t maximumDifference = DESCRIPTOR_ARRAY_NUM_BITS * ERROR_THRESHOLD_PERCENT / 100;

    NVCVTensorData refDescriptorsTensorData;
    nvcvTensorExportData(refDescriptorsTensorHandle, &refDescriptorsTensorData);
    int32_t numFeatures          = refDescriptorsTensorData.shape[0];
    uint8_t *refDescriptorsPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&refDescriptorsPtr_h, (void *)refDescriptorsTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaDescriptorsTensorData;
    nvcvTensorExportData(pvaDescriptorsTensorHandle, &pvaDescriptorsTensorData);
    uint8_t *pvaDescriptorsPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&pvaDescriptorsPtr_h, (void *)pvaDescriptorsTensorData.buffer.strided.basePtr);

    for (int32_t i = 0; i < numFeatures; i++)
    {
        int32_t hammingDistance = computeHammingDistance(&refDescriptorsPtr_h[i * PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH],
                                                         &pvaDescriptorsPtr_h[i * PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH],
                                                         PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH);

        if (hammingDistance > maximumDifference)
        {
            printf("Mismatch at feature %d, hamming distance = %d\n", i, hammingDistance);
            return 1;
        }
    }
    return 0;
}