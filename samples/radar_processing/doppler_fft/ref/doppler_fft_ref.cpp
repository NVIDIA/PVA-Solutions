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

#include "doppler_fft_ref.h"

#include "RadarRefUtils.hpp"
#include "RadarUtils.hpp"

#include "doppler_fft_param.h"

#include <TensorUtils.hpp>

#include <ImageIO.h>

#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

void dopplerFFTFixedPoint(const int32_t *input, const int32_t *window, int32_t *output, int32_t n)
{
    // Prepare FFT input with windowing
    std::vector<int32_t> fft_interleaved(2 * n);
    for (int32_t i = 0; i < n; i++)
    {
        fft_interleaved[2 * i] =
            my_round(((int64_t)input[2 * i] * window[i]) >> TRUNCATE_QBITS, WINDOW_QBITS - TRUNCATE_QBITS);
        fft_interleaved[2 * i + 1] =
            my_round(((int64_t)input[2 * i + 1] * window[i]) >> TRUNCATE_QBITS, WINDOW_QBITS - TRUNCATE_QBITS);
    }

    FFTFactorization factorization = pva_solutions::priv::FactorizeFFTSize(n, false);

    if (factorization.isValid == 1u)
    {
        fft_fixedp_mixed_radix_dif(fft_interleaved.data(), output, factorization);
    }
    else
    {

        printf("Error: FFT size '%d' is not supported.\n", n);
        return;
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
int32_t loadRawImpl(NVCVTensorHandle tensorHandle, const char *directory, const char *filename, bool convertToFixed)
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

    int32_t err = ReadImageBuffer(filename, directory, tensorPtr, count * sizeof(T));
    if (err)
    {
        return err;
    }

    if (convertToFixed)
    {
        float *floatPtr = (float *)tensorPtr;
        // Inplace conversion
        for (int32_t i = 0; i < count; i++)
        {
            tensorPtr[i] = static_cast<T>(DATA_SCALE * floatPtr[i]);
        }
    }

    return 0;
}

template<typename T>
void generateDopplerFFTWindowImpl(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType)
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
                        printf("Mismatch[%d, %d, %d, %d]: ref = %d, pva = %d\n", h, c, w, e, refData, cmpData);
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

template<typename TI, typename TO>
void dopplerFFTRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle,
                       NVCVTensorHandle outTensorHandle, PVADopplerFFTParams *params)
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

    /* The tensors are HCW, so the NCHW permutation puts the *middle* layout axis at index 1 (C), the
     * first at index 2 (H) and the last at index 3 (W). Spelled out per layout (Nr = chirps / Doppler
     * bins, Nb = range bins):
     *
     *   layout             HCW shape            NCHW[1]   NCHW[2]   NCHW[3]
     *   DOPPLER_RX_RANGE   [Nr, NofRx, Nb]      NofRx     Nr        Nb
     *   RANGE_RX_DOPPLER   [Nb, NofRx, Nr]      NofRx     Nb        Nr
     *   RX_DOPPLER_RANGE   [NofRx, Nr, Nb]      Nr        NofRx     Nb      (input only)
     *   RANGE_DOPPLER_RX   [Nb, Nr, NofRx]      Nr        Nb        NofRx   (output only)
     *   RX_RANGE_DOPPLER   [NofRx, Nb, Nr]      Nb        NofRx     Nr      (output only)
     */
    const bool isRxDopplerRangeInputLayout = (params->inputLayout == PVA_FFT_LAYOUT_RX_DOPPLER_RANGE);
    // RANGE_RX_DOPPLER input keeps the chirps in W and the range bins in H.
    const bool isRangeRxDopplerInputLayout = (params->inputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER);
    const int32_t channels                 = isRxDopplerRangeInputLayout ? inTensorShapeNCHW[2] : inTensorShapeNCHW[1];
    const int32_t chirps                   = isRangeRxDopplerInputLayout   ? inTensorShapeNCHW[3]
                                             : isRxDopplerRangeInputLayout ? inTensorShapeNCHW[1]
                                                                           : inTensorShapeNCHW[2];
    const int32_t rangeBins                = isRangeRxDopplerInputLayout ? inTensorShapeNCHW[2] : inTensorShapeNCHW[3];

    int32_t outDopplerBins;
    int32_t outRangeBins;
    switch (params->outputLayout)
    {
    case PVA_FFT_LAYOUT_DOPPLER_RX_RANGE:
        outDopplerBins = outTensorShapeNCHW[2];
        outRangeBins   = outTensorShapeNCHW[3];
        break;
    case PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
        outDopplerBins = outTensorShapeNCHW[3];
        outRangeBins   = outTensorShapeNCHW[2];
        break;
    case PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
        outDopplerBins = outTensorShapeNCHW[1];
        outRangeBins   = outTensorShapeNCHW[2];
        break;
    case PVA_FFT_LAYOUT_RX_RANGE_DOPPLER:
        outDopplerBins = outTensorShapeNCHW[3];
        outRangeBins   = outTensorShapeNCHW[1];
        break;
    default:
        printf("Error: Doppler FFT output layout '%d' is not supported.\n", static_cast<int32_t>(params->outputLayout));
        return;
    }

    std::vector<TI> inBuf;
    inBuf.resize(chirps * elements);

    std::vector<TO> outBuf;
    outBuf.resize(chirps * elements);

    for (int32_t c = 0; c < channels; c++)
    {
        for (int32_t b = 0; b < rangeBins; b++)
        {
            for (int32_t s = 0; s < chirps; s++)
            {
                for (int32_t e = 0; e < elements; e++)
                {
                    // Chirp-innermost input reads along the row; the others stride by the bin count.
                    const int32_t srcIdx =
                        isRangeRxDopplerInputLayout
                            ? ((b * channels + c) * chirps + s)
                            : ((isRxDopplerRangeInputLayout ? (c * chirps + s) : (s * channels + c)) * rangeBins + b);
                    inBuf[s * elements + e] = inTensorPtr[srcIdx * elements + e];
                }
            }

            dopplerFFTFixedPoint(inBuf.data(), winTensorPtr, outBuf.data(), chirps);

            if (params->outputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
            {
                // Output layout: [rangeBins][channels][dopplerBins]
                memcpy(outTensorPtr + ((b * channels + c) * outDopplerBins) * elements, outBuf.data(),
                       outDopplerBins * elements * sizeof(TO));
            }
            else if (params->outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
            {
                // Output layout: [rangeBins][dopplerBins][channels]
                for (int32_t d = 0; d < outDopplerBins; d++)
                {
                    for (int32_t e = 0; e < elements; e++)
                    {
                        outTensorPtr[((b * outDopplerBins + d) * channels + c) * elements + e] =
                            outBuf[d * elements + e];
                    }
                }
            }
            else if (params->outputLayout == PVA_FFT_LAYOUT_RX_RANGE_DOPPLER)
            {
                // Output layout: [channels][rangeBins][dopplerBins]
                for (int32_t d = 0; d < outDopplerBins; d++)
                {
                    for (int32_t e = 0; e < elements; e++)
                    {
                        outTensorPtr[((c * outRangeBins + b) * outDopplerBins + d) * elements + e] =
                            outBuf[d * elements + e];
                    }
                }
            }
            else
            {
                // Output layout: [dopplerBins][channels][rangeBins]
                for (int32_t d = 0; d < outDopplerBins; d++)
                {
                    for (int32_t e = 0; e < elements; e++)
                    {
                        outTensorPtr[((d * channels + c) * rangeBins + b) * elements + e] = outBuf[d * elements + e];
                    }
                }
            }
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

void generateDopplerFFTInput(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_2S32:
        generateInputImpl<int32_t>(tensorHandle);
        break;
    default:
        break;
    }
}

int32_t loadDopplerFFTRaw(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                          bool convertToFixed)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_2S32:
        return loadRawImpl<int32_t>(tensorHandle, directory, filename, convertToFixed);
    default:
        return -1; // Unsupported data type
    }
}

void generateDopplerFFTWindow(NVCVTensorHandle tensorHandle, PVABatchFFTWindowType windowType)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    switch (tensorData.dtype)
    {
    case NVCV_DATA_TYPE_S32:
        generateDopplerFFTWindowImpl<int32_t>(tensorHandle, windowType);
        break;
    default:
        break;
    }
}

int32_t checkDopplerFFTOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
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

void dopplerFFTRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle winTensorHandle, NVCVTensorHandle outTensorHandle,
                   PVADopplerFFTParams *params)
{
    NVCVTensorData inTensorData, winTensorData, outTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);
    nvcvTensorExportData(winTensorHandle, &winTensorData);
    nvcvTensorExportData(outTensorHandle, &outTensorData);

    if ((inTensorData.dtype == NVCV_DATA_TYPE_2S32) && (outTensorData.dtype == NVCV_DATA_TYPE_2S32))
    {
        dopplerFFTRefImpl<int32_t, int32_t>(inTensorHandle, winTensorHandle, outTensorHandle, params);
    }
}

void dumpDopplerFFTSNR(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
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
