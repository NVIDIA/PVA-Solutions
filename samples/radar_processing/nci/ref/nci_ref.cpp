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

#include "nci_ref.h"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <ImageIO.h>
#include <cupva_host_scheduling.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <type_traits>
namespace {

// Generic fixed-point multiplication functions
template<typename T>
static inline T fixed_point_mult(T a, T b, uint32_t qBits)
{
    if constexpr (std::is_signed_v<T>)
    {
        // multiplication result is limited to 48-bit signed
        int64_t result = (int64_t)a * b;

        // Truncate to 48-bit signed
        const int64_t mask_48bit = (1LL << 48) - 1;
        result                   = result & mask_48bit;
        // Sign extend from bit 47
        if (result & (1LL << 47))
        {
            result |= ~mask_48bit;
        }

        return static_cast<T>(result >> qBits);
    }
    else
    {
        return static_cast<T>(((uint64_t)a * b) >> qBits);
    }
}

static inline uint64_t fixed_point_magnitude(int64_t a, uint32_t qBits)
{
    uint64_t a_uint64 = (uint64_t)std::abs(a);
    uint64_t result   = a_uint64 * a_uint64;
    return result >> qBits;
}

// Generic fixed-point conversion functions
template<typename T>
static inline float fixed_to_float(T val, uint32_t qBits)
{
    return (float)val / (1 << qBits);
}

template<typename T>
static inline T float_to_fixed(float val, uint32_t qBits)
{
    return static_cast<T>(val * (1 << qBits));
}

/**
 * @brief Truncates a 64-bit signed integer to 48 bits by clearing the upper 16 bits
 * @param value The 64-bit value to truncate
 * @return The truncated 48-bit value as int64_t
 */
static inline int64_t truncateTo48Bits(int64_t value)
{
    return (int64_t)((value << 16) >> 16);
}

int32_t nciRefImpl(const int32_t *inFt2d, uint32_t *outNciRx, uint32_t *outNciFinal, uint32_t *outNoiseEstimate,
                   uint32_t Nr, uint32_t NofRx, uint32_t Ns, uint32_t numDopplerFolds, PVAFFTLayout inputLayout)
{
    int32_t status = -1;

    // Map function parameters to variable names used in the code
    const int32_t *ft2d     = inFt2d;
    uint32_t *nciRx         = outNciRx;
    uint32_t *nciFinal      = outNciFinal;
    uint32_t *noiseEstimate = outNoiseEstimate;

    const uint32_t roundNumDopplerFolds = 0u;  // no rounding
    const uint32_t roundNofRx           = 0u;  // no rounding
    const uint32_t threshold            = 10u; // Threshold for noise estimation (UQ16 format)
    const uint32_t NsPerFold            = Ns / numDopplerFolds;

    if (Ns % numDopplerFolds != 0)
    {
        goto FreeResources;
    }

    // Step 1: Compute absolute values of complex numbers and average along RX dimension
    // FT2D dimensions:
    // - [Nr][NofRx][Ns] for HCW layout (Range, RX, Doppler)
    // - [Nr][Ns][NofRx] for HWC layout (Range, Doppler, RX)
    // Output NCI_RX dimensions: [Nr][Ns]
    for (uint32_t iRange = 0; iRange < Nr; iRange++)
    {
        for (uint32_t iDoppler = 0; iDoppler < Ns; iDoppler++)
        {
            uint64_t sum = 0;
            // Average absolute values across all RX channels
            for (uint32_t iRx = 0; iRx < NofRx; iRx++)
            {
                // Layout: interleaved real/imag data
                uint32_t baseIdx = 0;
                if (inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
                {
                    // [Nr][Ns][NofRx]
                    baseIdx = ((iRange * Ns * NofRx) + (iDoppler * NofRx) + iRx) * 2;
                }
                else
                {
                    // [Nr][NofRx][Ns]
                    baseIdx = ((iRange * NofRx * Ns) + (iRx * Ns) + iDoppler) * 2;
                }

                float real = (float)ft2d[baseIdx];     // Fixed-point format with configurable Q bits
                float imag = (float)ft2d[baseIdx + 1]; // Fixed-point format with configurable Q bits

                float magnitude_squared = real * real + imag * imag;

                float magnitude_float = sqrtf(magnitude_squared);

                // Convert to output UQ16 fixed-point format
                uint64_t magnitude = static_cast<uint64_t>(magnitude_float);

                sum += magnitude;

                // Truncate sum to a 48-bit signed value
                sum = truncateTo48Bits(sum);
            }
            // Store average magnitude in NCI_RX (with rounding)
            // Perform averaging in 64-bit to avoid overflow, then cast result to uint32_t
            nciRx[iRange * Ns + iDoppler] = (uint32_t)((sum + roundNofRx) / NofRx);
        }
    }

    // Step 2: Compute NCI_FINAL by reshaping and averaging over Doppler folds
    // Reshape NCI_RX from [Nr][Ns] to [Nr][Ns/numDopplerFolds][numDopplerFolds] and average over last dimension
    // Output NCI_FINAL dimensions: [Nr][Ns/numDopplerFolds]

    for (uint32_t iRange = 0; iRange < Nr; iRange++)
    {
        for (uint32_t iFold = 0; iFold < NsPerFold; iFold++)
        {
            int64_t sum = 0;

            // Average across Doppler folds (Fortran-style column-major reshaping)
            for (uint32_t iDopplerFold = 0; iDopplerFold < numDopplerFolds; iDopplerFold++)
            {
                uint32_t dopplerIdx = iFold + iDopplerFold * NsPerFold;
                uint32_t val        = nciRx[((iRange * Ns) + dopplerIdx)];

                sum += val;

                // Truncate sum to a 48-bit signed value
                sum = truncateTo48Bits(sum);
            }

            // Store average in NCI_FINAL (with rounding)
            // Perform averaging in 64-bit to avoid overflow, then cast result to uint32_t
            nciFinal[iRange * NsPerFold + iFold] = (uint32_t)((sum + roundNumDopplerFolds) / numDopplerFolds);
        }
    }

    // Step 3: Compute noise estimation if noise estimation is enabled
    if (outNoiseEstimate != NULL)
    {
        // Find minimum value across Doppler folds for each range bin and multiply by threshold
        for (uint32_t iRange = 0; iRange < Nr; iRange++)
        {
            uint32_t minVal = nciFinal[iRange * NsPerFold]; // Initialize with first value
            // Find minimum across all Doppler folds for this range bin
            for (uint32_t iFold = 1; iFold < NsPerFold; iFold++)
            {
                uint32_t val = nciFinal[((iRange * NsPerFold) + iFold)];
                if (val < minVal)
                {
                    minVal = val;
                }
            }

            // Apply threshold multiplier using UQ16 fixed-point multiplication
            noiseEstimate[iRange] = minVal * threshold;
        }
    }
    status = 0;
FreeResources:
    return status;
}

template<typename T>
void generateRandomNciInputImpl(NVCVTensorHandle tensorHandle, int32_t qBits)
{
    static std::mt19937 gen = []() {
        std::mt19937 g;
        g.seed(2);
        return g;
    }();
    using Dist = std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
                                    std::uniform_int_distribution<T>>;

    // For fixed-point format (int32_t), limit range to prevent overflow when squaring
    T minVal, maxVal;
    if constexpr (std::is_same_v<T, int32_t>)
    {
        minVal = static_cast<T>(-100 * (1 << qBits));
        maxVal = static_cast<T>(100 * (1 << qBits));
    }
    else
    {
        minVal = std::numeric_limits<T>::lowest();
        maxVal = std::numeric_limits<T>::max();
    }

    Dist distrib(minVal, maxVal);

    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    NVCVDataType dataType;
    nvcvTensorGetDataType(tensorHandle, &dataType);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(dataType, &bitsPerPixel);

    // For NVCV_DATA_TYPE_2S32, treated as 2x of the original number of elements, which is derived from shape.
    int32_t ratio = bitsPerPixel / 8 / sizeof(T);

    int64_t numel = std::accumulate(std::begin(tensorData.shape), std::begin(tensorData.shape) + tensorData.rank, 1,
                                    std::multiplies<int64_t>()) *
                    ratio;

    for (size_t i = 0; i < numel; i++)
    {
        tensorPtr_h[i] = static_cast<T>(distrib(gen));
    }
}

int32_t loadNCIRawImpl(NVCVTensorHandle tensorHandle, const char *directory, const char *filename)
{
    int32_t err   = -1;
    uint8_t *data = nullptr;
    int64_t numel;
    int32_t bitsPerPixel;
    NVCVDataType dataType;
    NVCVTensorData tensorData;
    if (0 != nvcvTensorExportData(tensorHandle, &tensorData))
    {
        goto FreeResources;
    }

    if (CupvaMemGetHostPointer((void **)&data, (void *)tensorData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    numel = std::accumulate(std::begin(tensorData.shape), std::begin(tensorData.shape) + tensorData.rank, 1,
                            std::multiplies<int64_t>());

    nvcvTensorGetDataType(tensorHandle, &dataType);

    nvcvDataTypeGetBitsPerPixel(dataType, &bitsPerPixel);

    err = ReadImageBuffer(filename, directory, data, numel * (bitsPerPixel / 8));

FreeResources:
    return err;
}
} // namespace

int32_t checkNCIOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle, uint32_t tolerance)
{
    int32_t err            = 1;
    uint32_t mismatchCount = 0;
    int32_t height;
    int32_t width;
    int32_t dataSize;
    int32_t rank;

    NVCVTensorData refTensorData;
    NVCVTensorData pvaTensorData;

    uint32_t *ref = NULL;
    uint32_t *pva = NULL;

    if (0 != nvcvTensorExportData(refTensorHandle, &refTensorData))
    {
        goto FreeResources;
    }

    if (0 != CupvaMemGetHostPointer((void **)&ref, (void *)refTensorData.buffer.strided.basePtr))
    {
        goto FreeResources;
    }

    if (0 != nvcvTensorExportData(pvaTensorHandle, &pvaTensorData))
    {
        goto FreeResources;
    }

    if (0 != CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr))
    {
        goto FreeResources;
    }

    height = refTensorData.shape[0];
    width  = refTensorData.shape[1];
    rank   = refTensorData.layout.rank;

    if (2 == rank)
    {
        dataSize = height * width;
    }
    else if (1 == rank)
    {
        dataSize = height;
    }
    else
    {
        printf("Unsupported data layout for comparison\n");
        goto FreeResources;
    }

    for (int32_t x = 0; x < dataSize; x++)
    {
        uint32_t refData = ref[x];
        uint32_t cmpData = pva[x];
        uint32_t diff    = (refData > cmpData) ? (refData - cmpData) : (cmpData - refData);
        if (diff > tolerance)
        {
            mismatchCount++;

            if (mismatchCount < 10)
            {
                printf("Mismatch[%d]: ref = %u, pva = %u\n", x, refData, cmpData);
            }
        }
    }

    if (mismatchCount == 0)
    {
        err = 0;
    }
    else if (mismatchCount >= 10)
    {
        printf("Mismatch count: %u\n", mismatchCount);
    }

FreeResources:
    return err;
}

int32_t nciRef(NVCVTensorHandle const *inTensorHandles, NVCVTensorHandle const *outTensorHandles, int inTensorCount,
               int outTensorCount, uint32_t numDopplerFolds, PVAFFTLayout inputLayout)
{
    int32_t err = 1;

    NVCVTensorData inTensorData;
    NVCVTensorData outNciRxTensorData;
    NVCVTensorData outNciFinalTensorData;
    NVCVTensorData outNoiseEstimateTensorData;

    int32_t *inTensorPtr                = nullptr;
    uint32_t *outNciRxTensorPtr         = nullptr;
    uint32_t *outNciFinalTensorPtr      = nullptr;
    uint32_t *outNoiseEstimateTensorPtr = nullptr;

    if ((1 != inTensorCount) || (3 != outTensorCount))
    {
        goto FreeResources;
    }

    if (0 != nvcvTensorExportData(inTensorHandles[0], &inTensorData))
    {
        goto FreeResources;
    }

    if (0 != CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr))
    {
        goto FreeResources;
    }

    if (0 != nvcvTensorExportData(outTensorHandles[0], &outNciRxTensorData))
    {
        goto FreeResources;
    }

    if (0 != CupvaMemGetHostPointer((void **)&outNciRxTensorPtr, (void *)outNciRxTensorData.buffer.strided.basePtr))
    {
        goto FreeResources;
    }

    if (0 != nvcvTensorExportData(outTensorHandles[1], &outNciFinalTensorData))
    {
        goto FreeResources;
    }

    if (0 !=
        CupvaMemGetHostPointer((void **)&outNciFinalTensorPtr, (void *)outNciFinalTensorData.buffer.strided.basePtr))
    {
        goto FreeResources;
    }

    if (outTensorHandles[2] != NULL)
    {
        if (0 != nvcvTensorExportData(outTensorHandles[2], &outNoiseEstimateTensorData))
        {
            goto FreeResources;
        }

        if (0 != CupvaMemGetHostPointer((void **)&outNoiseEstimateTensorPtr,
                                        (void *)outNoiseEstimateTensorData.buffer.strided.basePtr))
        {
            goto FreeResources;
        }
    }
    {
        int64_t inTensorShapeNCHW[4];
        nvcvTensorShapePermute(inTensorData.layout, inTensorData.shape, NVCV_TENSOR_NCHW, inTensorShapeNCHW);
        uint32_t Nr    = inTensorShapeNCHW[2];
        uint32_t NofRx = 0;
        uint32_t Ns    = 0;

        if (inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
        {
            Ns    = inTensorShapeNCHW[1];
            NofRx = inTensorShapeNCHW[3];
        }
        else
        {
            NofRx = inTensorShapeNCHW[1];
            Ns    = inTensorShapeNCHW[3];
        }
        err = nciRefImpl(inTensorPtr, outNciRxTensorPtr, outNciFinalTensorPtr, outNoiseEstimateTensorPtr, Nr, NofRx, Ns,
                         numDopplerFolds, inputLayout);
    }

FreeResources:
    return err;
}

int32_t loadNCIRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename)
{
    return loadNCIRawImpl(tensorHandle, directory, filename);
}

void generateRandomNciInput(NVCVTensorHandle tensorHandle, int32_t qBits)
{
    NVCVDataType dataType;
    nvcvTensorGetDataType(tensorHandle, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_2S32:
        generateRandomNciInputImpl<int32_t>(tensorHandle, qBits);
        break;
    default:
        throw std::runtime_error("Unsupported data type for random NCI input generation");
    }
}