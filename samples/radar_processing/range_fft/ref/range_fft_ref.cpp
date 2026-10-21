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

#include "range_fft_ref.h"

#include "RadarRefUtils.hpp"
#include "RadarUtils.hpp"

#include "../../../../src/operator/radar_processing/range_fft/device/range_fft_param.h"

#include <TensorUtils.hpp>

#include <ImageIO.h>

#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {

// Mixed-radix DIF FFT implementation (supports combinations of Radix-5 and Radix-2)
void fft_fixedp_mixed_radix_dif(int32_t *x_interleaved, int32_t *y_final_interleaved, FFTFactorization &factorization)
{
    int32_t n = factorization.n;
    // Allocate buffers
    int32_t *stage_data          = (int32_t *)malloc(2 * n * sizeof(int32_t));
    int32_t *twiddle_interleaved = (int32_t *)malloc(2 * n * sizeof(int32_t));
    int32_t *temp_buffer         = (int32_t *)malloc(2 * n * sizeof(int32_t));

    // Copy input data
    for (int32_t i = 0; i < 2 * n; i++)
    {
        stage_data[i] = x_interleaved[i];
    }

    // Pre-compute twiddle factors
    for (int32_t i = 0; i < n; i++)
    {
        double angle                   = 2 * M_PI * i / n;
        twiddle_interleaved[2 * i]     = (int32_t)round(cos(angle) * TWIDDLE_SCALE);
        twiddle_interleaved[2 * i + 1] = (int32_t)round(-sin(angle) * TWIDDLE_SCALE);
    }

    int32_t *stage_in  = stage_data;
    int32_t *stage_out = temp_buffer;

    int32_t groupsz_radix5 = factorization.groups[RADIX5_GROUP_IDX].groupSize;
    // Process Radix-5 stages first (outermost in DIF)
    for (int32_t k = 0; k < factorization.groups[RADIX5_GROUP_IDX].numStages; k++)
    {
        RadarRefUtils::fft_fixedp_r5_dif_stage(n, k, groupsz_radix5, stage_in, twiddle_interleaved, stage_out);
        // Swap buffers
        int32_t *temp = stage_in;
        stage_in      = stage_out;
        stage_out     = temp;
    }

    int32_t groupsz_radix3 = factorization.groups[RADIX3_GROUP_IDX].groupSize;

    for (int32_t k = 0; k < factorization.groups[RADIX3_GROUP_IDX].numStages; k++)
    {
        RadarRefUtils::fft_fixedp_r3_dif_stage(n, k, groupsz_radix3, stage_in, twiddle_interleaved, stage_out);
        int32_t *temp = stage_in;
        stage_in      = stage_out;
        stage_out     = temp;
    }

    int32_t groupsz_radix4 = factorization.groups[RADIX4_GROUP_IDX].groupSize;

    for (int32_t k = 0; k < factorization.groups[RADIX4_GROUP_IDX].numStages; k++)
    {
        RadarRefUtils::fft_fixedp_r4_dif_stage(n, k, groupsz_radix4, stage_in, twiddle_interleaved, stage_out);
        // Swap buffers
        int32_t *temp = stage_in;
        stage_in      = stage_out;
        stage_out     = temp;
    }

    int32_t groupsz_radix2 = groupsz_radix4;
    // Process radix-2 stage if odd number of radix-2 bits
    if (factorization.groups[RADIX4_GROUP_IDX].isOdd)
    {
        int32_t log2n = factorization.groups[RADIX4_GROUP_IDX].numStages * 2;
        RadarRefUtils::fft_fixedp_r2_dif_stage(n, log2n, groupsz_radix2, stage_in, twiddle_interleaved, stage_out);

        int32_t *temp = stage_in;
        stage_in      = stage_out;
        stage_out     = temp;
    }

    // Final permutation (mixed-radix digit reversal)
    for (int32_t i = 0; i < n; i++)
    {
        int32_t j                      = RadarRefUtils::mixed_radix_reverse(i, n, factorization);
        y_final_interleaved[2 * i]     = stage_in[2 * j];
        y_final_interleaved[2 * i + 1] = stage_in[2 * j + 1];
    }

    free(twiddle_interleaved);
    free(stage_data);
    free(temp_buffer);
}

// Real FFT post-processing function
void fft_real_post_process(int32_t n, int32_t *x, int32_t *y)
{
    int32_t qbits = 1 + (TWIDDLE_QBITS - TRUNCATE_QBITS);

    // Prepare A and B arrays
    int32_t *A = (int32_t *)malloc(2 * n * sizeof(int32_t));
    int32_t *B = (int32_t *)malloc(2 * n * sizeof(int32_t));

    for (int32_t i = 0; i < n; i++)
    {
        A[2 * i]     = (int32_t)round(0.5 * (1.0 - sin(2 * M_PI / (2 * n) * i)) * TWIDDLE_SCALE);
        A[2 * i + 1] = (int32_t)round(0.5 * (-1.0 * cos(2 * M_PI / (2 * n) * i)) * TWIDDLE_SCALE);
        B[2 * i]     = (int32_t)round(0.5 * (1.0 + sin(2 * M_PI / (2 * n) * i)) *
                                      std::min(1UL << (TWIDDLE_QBITS - TRUNCATE_QBITS), TWIDDLE_SCALE));
        B[2 * i + 1] = (int32_t)round(0.5 * (1.0 * cos(2 * M_PI / (2 * n) * i)) *
                                      std::min(1UL << (TWIDDLE_QBITS - TRUNCATE_QBITS), TWIDDLE_SCALE));
    }

    // Extend input for symmetry: x[2*n] = x[0], x[2*n+1] = x[1]
    int32_t *x_ext = (int32_t *)malloc((2 * n + 2) * sizeof(int32_t));
    for (int32_t i = 0; i < 2 * n; i++)
    {
        x_ext[i] = x[i];
    }
    x_ext[2 * n]     = x[0];
    x_ext[2 * n + 1] = x[1];

    // Initialize output array
    for (int32_t i = 0; i < 4 * n + 2; i++)
    {
        y[i] = 0;
    }

    for (int32_t i = 0; i < n; i++)
    {
        int32_t j = n - i;

        int64_t x0r = x_ext[2 * i];
        int64_t x0i = x_ext[2 * i + 1];
        int64_t x1r = x_ext[2 * j];
        int64_t x1i = x_ext[2 * j + 1];
        int64_t ar  = A[2 * i];
        int64_t ai  = A[2 * i + 1];
        int64_t br  = B[2 * i];
        int64_t bi  = B[2 * i + 1];

        int64_t tr = ((x0r * ar) >> TRUNCATE_QBITS) - ((x0i * ai) >> TRUNCATE_QBITS) + ((x1i * bi)) + ((x1r * br));
        int64_t ti = ((x0i * ar) >> TRUNCATE_QBITS) + ((x0r * ai) >> TRUNCATE_QBITS) + ((x1r * bi)) - ((x1i * br));

        y[2 * i]     = my_round(tr, qbits);
        y[2 * i + 1] = my_round(ti, qbits);
    }

    y[2 * n]     = (x_ext[0] - x_ext[1] + 1) >> 1;
    y[2 * n + 1] = 0;

    free(A);
    free(B);
    free(x_ext);
}

void rangeFFTFixedPoint(const int32_t *input, const int32_t *window, int32_t *output, int32_t n)
{
    // Prepare FFT input with windowing
    std::vector<int32_t> fft_input(n);
    for (int32_t i = 0; i < n; i++)
    {
        fft_input[i] = my_round(((int64_t)input[i] * window[i]) >> TRUNCATE_QBITS, WINDOW_QBITS - TRUNCATE_QBITS);
    }

    FFTFactorization factorization = {n, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, {0}, {0}, 0, 0, {0}, 0};
    if ((n % 2) == 0)
    {
        factorization = pva_solutions::priv::FactorizeFFTSize(n / 2, false);
        if (factorization.isValid == 1)
        {
            fft_fixedp_mixed_radix_dif(fft_input.data(), fft_input.data(), factorization);
        }
    }

    if (factorization.isValid == 0)
    {
        printf("Error: FFT size '%d' is not supported.\n", n);
        return;
    }

    // Apply real FFT post-processing
    std::vector<int32_t> post_processed(4 * n + 2);
    fft_real_post_process(n / 2, fft_input.data(), post_processed.data());

    // Copy results to output
    for (int32_t i = 0; i < n; i++)
    {
        output[2 * i]     = (int32_t)post_processed[2 * i];
        output[2 * i + 1] = (int32_t)post_processed[2 * i + 1];
    }
}

// Complex-input reference: N complex samples in, N complex bins out.  No real-FFT post-process — factorize N
// directly and run the mixed-radix DIF.  The window is real and broadcast over (re, im).
void rangeFFTComplexFixedPoint(const int32_t *input, const int32_t *window, int32_t *output, int32_t n)
{
    std::vector<int32_t> fft_input(2 * n);
    for (int32_t i = 0; i < n; i++)
    {
        const int64_t w  = window[i];
        fft_input[2 * i] = my_round(((int64_t)input[2 * i] * w) >> TRUNCATE_QBITS, WINDOW_QBITS - TRUNCATE_QBITS);
        fft_input[2 * i + 1] =
            my_round(((int64_t)input[2 * i + 1] * w) >> TRUNCATE_QBITS, WINDOW_QBITS - TRUNCATE_QBITS);
    }

    FFTFactorization factorization = pva_solutions::priv::FactorizeFFTSize(n, false);
    if (factorization.isValid == 0)
    {
        printf("Error: complex FFT size '%d' is not supported.\n", n);
        return;
    }

    std::vector<int32_t> fft_output(2 * n);
    fft_fixedp_mixed_radix_dif(fft_input.data(), fft_output.data(), factorization);

    for (int32_t i = 0; i < 2 * n; i++)
    {
        output[i] = fft_output[i];
    }
}

template<typename T>
void generateInputImpl(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(tensorData.dtype, &bitsPerPixel);
    int32_t elements = bitsPerPixel / CHAR_BIT / sizeof(T);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t channels = tensorShapeNCHW[1];
    const int32_t height   = tensorShapeNCHW[2];
    const int32_t width    = tensorShapeNCHW[3];

    // Use random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    // Generate random value for input data
    for (int32_t h = 0; h < height; h++)
    {
        for (int32_t c = 0; c < channels; c++)
        {
            for (int32_t w = 0; w < width; w++)
            {
                for (int32_t e = 0; e < elements; e++)
                {
                    int32_t i = ((c * height + h) * width + w) * elements + e;
                    // Input can't be greater than 31-bit
                    tensorPtr[i] = dist(gen) / 2;
                }
            }
        }
    }
}

template<typename T>
void loadRawImpl(NVCVTensorHandle tensorHandle, const char *directory, const char *filename, bool convertToFixed)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(tensorData.dtype, &bitsPerPixel);
    int32_t elements = bitsPerPixel / CHAR_BIT / sizeof(T);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t channels = tensorShapeNCHW[1];
    const int32_t height   = tensorShapeNCHW[2];
    const int32_t width    = tensorShapeNCHW[3];

    const int32_t count = height * channels * width * elements;

    ReadImageBuffer(filename, directory, tensorPtr, count * sizeof(T));

    if (convertToFixed)
    {
        float *floatPtr = (float *)tensorPtr;
        // Inplace conversion
        for (int32_t i = 0; i < count; i++)
        {
            tensorPtr[i] = static_cast<T>(DATA_SCALE * floatPtr[i]);
        }
    }
    else
    {
        for (int32_t i = 0; i < count; i++)
        {
            tensorPtr[i] = static_cast<T>(DATA_SCALE * tensorPtr[i]);
        }
    }
}

template<typename T>
void storeRawImpl(NVCVTensorHandle tensorHandle, const char *directory, const char *filename)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(tensorData.dtype, &bitsPerPixel);
    int32_t elements = bitsPerPixel / CHAR_BIT / sizeof(T);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t channels = tensorShapeNCHW[1];
    const int32_t height   = tensorShapeNCHW[2];
    const int32_t width    = tensorShapeNCHW[3];

    const int32_t count = height * channels * width * elements;

    WriteImageBuffer(filename, directory, tensorPtr, count * sizeof(T));
}

template<typename T>
void generateRangeFFTWindowImpl(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);
    T *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    int64_t tensorShapeNCHW[4];
    nvcvTensorShapePermute(tensorData.layout, tensorData.shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);

    const int32_t width = tensorShapeNCHW[3];

    for (int32_t w = 0; w < width; w++)
    {
        switch (windowType)
        {
        case PVA_BATCH_FFT_WINDOW_HANNING:
        {
            tensorPtr[w] = static_cast<T>(WINDOW_SCALE * 0.5f * (1.0f - cosf(2.0f * M_PI * w / (width - 1))));
        }
        break;
        case PVA_BATCH_FFT_WINDOW_USER_DEFINED:
        {
            tensorPtr[w] = static_cast<T>(WINDOW_SCALE * 1.0f);
        }
        break;
        default:
            break;
        }
    }
}

template<typename T>
int32_t checkOutputImpl(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle, const float maxDiffFloat)
{
    const T maxDiffFixed = static_cast<T>(ceil(DATA_SCALE * maxDiffFloat));

    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);
    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);
    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(refTensorData.dtype, &bitsPerPixel);
    int32_t elements = bitsPerPixel / CHAR_BIT / sizeof(T);

    int64_t refTensorShapeNCHW[4];
    nvcvTensorShapePermute(refTensorData.layout, refTensorData.shape, NVCV_TENSOR_NCHW, refTensorShapeNCHW);

    const int32_t height   = refTensorShapeNCHW[2];
    const int32_t channels = refTensorShapeNCHW[1];
    const int32_t width    = refTensorShapeNCHW[3];

    for (int32_t h = 0; h < height; h++)
    {
        for (int32_t c = 0; c < channels; c++)
        {
            for (int32_t w = 0; w < width; w++)
            {
                for (int32_t e = 0; e < elements; e++)
                {
                    int32_t i = ((h * channels + c) * width + w) * elements + e;
                    T refData = ref[i];
                    T cmpData = pva[i];
                    // Use absolute difference instead of relative difference ratio
                    // to avoid division by very small data near to zero
                    bool isError = (abs(refData - cmpData) > maxDiffFixed);
                    if (isError)
                    {
                        printf("Mismatch[%d, %d, %d, %d]: ref = %u, pva = %u\n", h, c, w, e, refData, cmpData);
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

template<typename TI, typename TO>
void rangeFFTRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle,
                     NVCVTensorHandle outTensorHandle, PVARangeFFTParams *params)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    TI *inTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr);

    NVCVTensorData winTensorData;
    nvcvTensorExportData(winTensorHandle, &winTensorData);
    TI *winTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&winTensorPtr, (void *)winTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    TO *outTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(outTensorData.dtype, &bitsPerPixel);
    int32_t elements = bitsPerPixel / CHAR_BIT / sizeof(TO);

    int64_t inTensorShapeNCHW[4], outTensorShapeNCHW[4];
    nvcvTensorShapePermute(inTensorData.layout, inTensorData.shape, NVCV_TENSOR_NCHW, inTensorShapeNCHW);
    nvcvTensorShapePermute(outTensorData.layout, outTensorData.shape, NVCV_TENSOR_NCHW, outTensorShapeNCHW);

    // RX_DOPPLER_RANGE input is HCW[NofRx, chirps, samples], so the Rx count and the chirp count swap
    // places in the NCHW permutation relative to the default HCW[chirps, NofRx, samples].
    const bool isRxDopplerRangeInputLayout = (params && params->inputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);

    const int32_t channels = isRxDopplerRangeInputLayout ? inTensorShapeNCHW[2] : inTensorShapeNCHW[1];
    const int32_t inHeight = isRxDopplerRangeInputLayout ? inTensorShapeNCHW[1] : inTensorShapeNCHW[2];
    const int32_t inWidth  = inTensorShapeNCHW[3];

    const int32_t outHeight = outTensorShapeNCHW[2];
    const int32_t outWidth  = outTensorShapeNCHW[3];

    const int32_t samples = inWidth;
    const int32_t chirps  = inHeight;

    // RANGE_RX_DOPPLER output is HCW[rangeBins, NofRx, chirps]: the bin count comes from H (NCHW[2]).
    const bool isRangeRxDopplerOutputLayout = (params && params->outputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER);
    const int32_t outRangeBins              = isRangeRxDopplerOutputLayout ? outHeight : outWidth;

    std::vector<TI> inBuf;
    inBuf.resize(samples);

    std::vector<TO> outBuf;
    outBuf.resize(samples * elements);

    const bool isRxDopplerRangeOutputLayout = (params && params->outputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);

    for (int32_t c = 0; c < channels; c++)
    {
        for (int32_t b = 0; b < chirps; b++)
        {
            // RX_DOPPLER_RANGE input rows are ordered Rx-major: offset = (c * chirps + b) * samples.
            // Default input rows are chirp-major:           offset = (b * channels + c) * samples.
            const int32_t inOffset =
                isRxDopplerRangeInputLayout ? (c * chirps + b) * samples : (b * channels + c) * samples;
            for (int32_t s = 0; s < samples; s++)
            {
                inBuf[s] = inTensorPtr[inOffset + s];
            }

            rangeFFTFixedPoint(inBuf.data(), winTensorPtr, outBuf.data(), samples);

            if (isRangeRxDopplerOutputLayout)
            {
                /* HCW[rangeBins, NofRx, chirps]: the bins of one chirp are strided by the channel
                 * and chirp extents rather than contiguous, so this is a scatter, not a memcpy. */
                for (int32_t r = 0; r < outRangeBins; r++)
                {
                    const int32_t dstIdx = ((r * channels + c) * chirps + b) * elements;
                    for (int32_t e = 0; e < elements; e++)
                    {
                        outTensorPtr[dstIdx + e] = outBuf[r * elements + e];
                    }
                }
                continue;
            }

            // RX_DOPPLER_RANGE: output shape is HCW[NofRx, chirps, rangeBins]
            // → offset = (c * chirps + b) * outRangeBins
            // Default: output shape is HCW[chirps, NofRx, rangeBins]
            // → offset = (b * channels + c) * outRangeBins
            const int32_t outOffset =
                isRxDopplerRangeOutputLayout ? (c * chirps + b) * outRangeBins : (b * channels + c) * outRangeBins;
            memcpy(outTensorPtr + outOffset * elements, outBuf.data(), outRangeBins * elements * sizeof(TO));
        }
    }
}

// Complex-input ref dispatch: input is 2S32 (interleaved re,im pairs), window is S32 (real), output is 2S32.
void rangeFFTRefComplexImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle,
                            NVCVTensorHandle outTensorHandle, PVARangeFFTParams *params)
{
    NVCVTensorData inTensorData, winTensorData, outTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    nvcvTensorExportData(winTensorHandle, &winTensorData);
    nvcvTensorExportData(outTensorHandle, &outTensorData);

    int32_t *inTensorPtr  = nullptr;
    int32_t *winTensorPtr = nullptr;
    int32_t *outTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&inTensorPtr, (void *)inTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&winTensorPtr, (void *)winTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    int64_t inTensorShapeNCHW[4], outTensorShapeNCHW[4];
    nvcvTensorShapePermute(inTensorData.layout, inTensorData.shape, NVCV_TENSOR_NCHW, inTensorShapeNCHW);
    nvcvTensorShapePermute(outTensorData.layout, outTensorData.shape, NVCV_TENSOR_NCHW, outTensorShapeNCHW);

    const bool isRxDopplerRangeInputLayout  = (params && params->inputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);
    const bool isRxDopplerRangeOutputLayout = (params && params->outputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);

    const int32_t channels = isRxDopplerRangeInputLayout ? inTensorShapeNCHW[2] : inTensorShapeNCHW[1];
    const int32_t chirps   = isRxDopplerRangeInputLayout ? inTensorShapeNCHW[1] : inTensorShapeNCHW[2];
    const int32_t samples  = inTensorShapeNCHW[3];

    const int32_t outRangeBins = outTensorShapeNCHW[3];

    std::vector<int32_t> inBuf(2 * samples);
    std::vector<int32_t> outBuf(2 * samples);

    for (int32_t c = 0; c < channels; c++)
    {
        for (int32_t b = 0; b < chirps; b++)
        {
            const int32_t inRowIdx =
                isRxDopplerRangeInputLayout ? (c * chirps + b) * samples : (b * channels + c) * samples;
            const int32_t *inRow = inTensorPtr + inRowIdx * 2;
            for (int32_t s = 0; s < 2 * samples; s++)
            {
                inBuf[s] = inRow[s];
            }

            rangeFFTComplexFixedPoint(inBuf.data(), winTensorPtr, outBuf.data(), samples);

            const int32_t outOffset =
                isRxDopplerRangeOutputLayout ? (c * chirps + b) * outRangeBins : (b * channels + c) * outRangeBins;
            memcpy(outTensorPtr + outOffset * 2, outBuf.data(), outRangeBins * 2 * sizeof(int32_t));
        }
    }
}

inline double calc_snr(int n, double *exp, double *out)
{
    double signal_power, noise_power;

    signal_power = noise_power = 0;

    for (int32_t i = 0; i < n; i++)
    {
        double ref_r = exp[i * 2];
        double ref_i = exp[i * 2 + 1];
        double cmp_r = out[i * 2];
        double cmp_i = out[i * 2 + 1];

        signal_power += ref_r * ref_r + ref_i * ref_i;
        noise_power += (ref_r - cmp_r) * (ref_r - cmp_r) + (ref_i - cmp_i) * (ref_i - cmp_i);
    }

    double snr = 10 * log10(signal_power / noise_power);

    return snr;
}

template<typename T>
void dumpSNRImpl(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);
    T *ref = NULL;
    CupvaMemGetHostPointer((void **)&ref, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);
    T *pva = NULL;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(refTensorData.dtype, &bitsPerPixel);
    const int32_t elements = bitsPerPixel / CHAR_BIT / sizeof(T);

    int64_t refTensorShapeNCHW[4];
    nvcvTensorShapePermute(refTensorData.layout, refTensorData.shape, NVCV_TENSOR_NCHW, refTensorShapeNCHW);

    const int32_t height   = refTensorShapeNCHW[2];
    const int32_t channels = refTensorShapeNCHW[1];
    const int32_t width    = refTensorShapeNCHW[3];

    const int32_t n = height * channels * width;

    std::vector<double> refVec(n * elements);
    std::vector<double> pvaVec(n * elements);

    for (int32_t i = 0; i < n * elements; i++)
    {
        refVec[i] = ref[i] / (double)DATA_SCALE;
        pvaVec[i] = pva[i] / (double)DATA_SCALE;
    }

    double snr = calc_snr(n, refVec.data(), pvaVec.data());
    printf("SNR: %f\n", snr);
}

} // namespace

void loadRangeFFTInput(NVCVTensorHandle tensorHandle, int32_t *inBuf, int32_t inHeight, int32_t inChannels,
                       int32_t inWidth)
{
    if (!inBuf)
    {
        throw std::invalid_argument("Input buffer is null");
    }

    NVCVTensorData tensorData;
    int32_t status = nvcvTensorExportData(tensorHandle, &tensorData);
    if (status != NVCV_SUCCESS)
    {
        throw std::runtime_error("Failed to export tensor data");
    }

    int32_t *tensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);
    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(tensorData.dtype, &bitsPerPixel);

    int32_t height   = tensorData.shape[0];
    int32_t channels = tensorData.shape[1];
    int32_t width    = tensorData.shape[2];

    if (height != inHeight || channels != inChannels || width != inWidth)
    {
        throw std::invalid_argument("Tensor dimensions don't match input dimensions");
    }
    if ((bitsPerPixel / CHAR_BIT) != sizeof(int32_t))
    {
        throw std::invalid_argument("Tensor data type doesn't match int32_t");
    }

    std::copy(inBuf, inBuf + (height * channels * width), tensorPtr);
}

void load_range_fft_input(NVCVTensorHandle tensorHandle, std::vector<std::vector<std::vector<int32_t>>> &inBuf)
{
    std::unique_ptr<int32_t[]> inBufPtr(new int32_t[inBuf.size() * inBuf[0].size() * inBuf[0][0].size()]);

    for (int32_t r = 0; r < inBuf.size(); r++)
    {
        for (int32_t c = 0; c < inBuf[0].size(); c++)
        {
            for (int32_t p = 0; p < inBuf[0][0].size(); p++)
            {
                inBufPtr[r * inBuf[0].size() * inBuf[0][0].size() + c * inBuf[0][0].size() + p] = inBuf[r][c][p];
            }
        }
    }

    loadRangeFFTInput(tensorHandle, inBufPtr.get(), inBuf.size(), inBuf[0].size(), inBuf[0][0].size());
}

void generateRangeFFTInput(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_S32:
    case NVCV_DATA_TYPE_2S32:
        // Both S32 (real) and 2S32 (complex, interleaved re/im pairs) are backed by int32_t storage; the loop walks
        // bitsPerPixel/sizeof(T) elements per pixel so 2S32 generates twice the values per pixel automatically.
        generateInputImpl<int32_t>(tensorHandle);
        break;
    default:
        break;
    }
}

void loadRangeFFTRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename, bool convertToFixed)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_S32:
    case NVCV_DATA_TYPE_2S32:
        loadRawImpl<int32_t>(tensorHandle, directory, filename, convertToFixed);
        break;
    default:
        break;
    }
}

void storeRangeFFTRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_2S32:
        storeRawImpl<int32_t>(tensorHandle, directory, filename);
        break;
    default:
        break;
    }
}

void generateRangeFFTWindow(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_S32:
        generateRangeFFTWindowImpl<int32_t>(tensorHandle, windowType);
        break;
    default:
        break;
    }
}

int32_t checkRangeFFTOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                            const float maxDiffFloat)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    switch (refTensorData.dtype)
    {
    case NVCV_DATA_TYPE_2S32:
        return checkOutputImpl<int32_t>(refTensorHandle, pvaTensorHandle, maxDiffFloat);
    default:
        return -1;
    }
}

void rangeFFTRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle, NVCVTensorHandle outTensorHandle,
                 PVARangeFFTParams *params)
{
    NVCVTensorData inTensorData, winTensorData, outTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    nvcvTensorExportData(winTensorHandle, &winTensorData);
    nvcvTensorExportData(outTensorHandle, &outTensorData);

    if ((inTensorData.dtype == NVCV_DATA_TYPE_S32) && (outTensorData.dtype == NVCV_DATA_TYPE_2S32))
    {
        rangeFFTRefImpl<int32_t, int32_t>(inTensorHandle, winTensorHandle, outTensorHandle, params);
    }
    else if ((inTensorData.dtype == NVCV_DATA_TYPE_2S32) && (outTensorData.dtype == NVCV_DATA_TYPE_2S32))
    {
        rangeFFTRefComplexImpl(inTensorHandle, winTensorHandle, outTensorHandle, params);
    }
}

void dumpRangeFFTSNR(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    switch (refTensorData.dtype)
    {
    case NVCV_DATA_TYPE_2S32:
        dumpSNRImpl<int32_t>(refTensorHandle, pvaTensorHandle);
        break;
    default:
        break;
    }
}