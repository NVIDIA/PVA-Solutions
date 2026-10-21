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

#include "peak_detection_ref.h"

#include "peak_detection_param.h"

#include <cupva_host_utils.hpp>

#include <ImageIO.h>
#include <cupva_host.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace {

template<typename T>
void generateRandomPeakDetectionInputImpl(NVCVTensorHandle tensorHandle)
{
    static std::mt19937 gen = []() {
        std::mt19937 g;
        g.seed(2);
        return g;
    }();
    using Dist = std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
                                    std::uniform_int_distribution<T>>;
    Dist distrib(std::numeric_limits<T>::lowest(), // lowest() for floats, 0 for unsigned
                 std::numeric_limits<T>::max());

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

    // For NVCV_DATA_TYPE_2S32 or NVCV_DATA_TYPE_2F32, treated as 2x of the original number of elements, which is derived from shape.
    int32_t ratio = bitsPerPixel / 8 / sizeof(T);

    int64_t numel = std::accumulate(std::begin(tensorData.shape), std::begin(tensorData.shape) + tensorData.rank, 1,
                                    std::multiplies<int64_t>()) *
                    ratio;

    for (size_t i = 0; i < numel; i++)
    {
        tensorPtr_h[i] = static_cast<T>(distrib(gen));
    }
}

int32_t loadTensorInputImpl(NVCVTensorHandle tensorHandle, const char *directory, const char *filename)
{
    int32_t err = 0;
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint8_t *data = nullptr;
    if (CupvaMemGetHostPointer((void **)&data, (void *)tensorData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    int64_t numel = std::accumulate(std::begin(tensorData.shape), std::begin(tensorData.shape) + tensorData.rank, 1,
                                    std::multiplies<int64_t>());

    NVCVDataType dataType;
    nvcvTensorGetDataType(tensorHandle, &dataType);

    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(dataType, &bitsPerPixel);

    err = ReadImageBuffer(filename, directory, data, numel * bitsPerPixel / 8);
    return err;
}

template<typename T>
void peakDetectionRefImpl(NVCVTensorHandle FT2DTensorHandle, NVCVTensorHandle nciRxTensorHandle,
                          NVCVTensorHandle nciFinalTensorHandle, NVCVTensorHandle noiseEstTensorHandle, int32_t NofTx,
                          int32_t repeatFold, NVCVTensorHandle peakCountTensorHandle,
                          NVCVTensorHandle peakIndexTensorHandle, NVCVTensorHandle snapTensorHandle)
{
    NVCVTensorData nciRxData;
    nvcvTensorExportData(nciRxTensorHandle, &nciRxData);
    T *nciRx_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&nciRx_h, (void *)nciRxData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    const int32_t nciRxLp = nciRxData.buffer.strided.strides[0] / sizeof(T);
    const int32_t Nb      = nciRxData.shape[0];
    const int32_t Nr      = nciRxData.shape[1];

    NVCVTensorData nciFinalData;
    nvcvTensorExportData(nciFinalTensorHandle, &nciFinalData);
    T *nciFinal_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&nciFinal_h, (void *)nciFinalData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    const int32_t nciFinalLp = nciFinalData.buffer.strided.strides[0] / sizeof(T);

    NVCVTensorData FT2DData;
    nvcvTensorExportData(FT2DTensorHandle, &FT2DData);
    // For the actual FT2D_h type when T is integral type, it should be its signed type version, but since only assignment
    // operation is used, we just ignore this.
    std::complex<T> *FT2D_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&FT2D_h, (void *)FT2DData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    const int32_t NofRx = FT2DData.shape[1];

    NVCVTensorData noiseEstData;
    nvcvTensorExportData(noiseEstTensorHandle, &noiseEstData);
    T *noiseEst_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&noiseEst_h, (void *)noiseEstData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    NVCVTensorData peakCountData;
    nvcvTensorExportData(peakCountTensorHandle, &peakCountData);
    int32_t *peakCount_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&peakCount_h, (void *)peakCountData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    NVCVTensorData peakIndexData;
    nvcvTensorExportData(peakIndexTensorHandle, &peakIndexData);
    uint32_t *peakIndex_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&peakIndex_h, (void *)peakIndexData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    uint32_t *rangeIdxPtr       = peakIndex_h;
    uint32_t *dopplerFoldIdxPtr = peakIndex_h + PEAKDET_MAX_PEAK_COUNT;
    uint32_t *dopplerIdxPtr     = peakIndex_h + PEAKDET_MAX_PEAK_COUNT * 2;

    NVCVTensorData snapData;
    nvcvTensorExportData(snapTensorHandle, &snapData);
    T *snap_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&snap_h, (void *)snapData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    const std::vector<int32_t> DDMspace{0 * Nr / repeatFold, 7 * Nr / repeatFold, 6 * Nr / repeatFold,
                                        4 * Nr / repeatFold};
    int32_t peakCount = 0;
    // Loop over range bins and skips the first and last range bin.
    for (int32_t ibin = 1; ibin < Nb - 1; ibin++)
    {
        // Loop over doppler fold bins.
        for (int32_t ifold = 0; ifold < Nr / repeatFold; ifold++)
        {
            // Compare actual value with estimated noise level.
            if (nciFinal_h[ibin * nciFinalLp + ifold] > noiseEst_h[ibin])
            {
                // Compare with lower neighbor bin.
                if (nciFinal_h[ibin * nciFinalLp + ifold] > nciFinal_h[(ibin - 1) * nciFinalLp + ifold])
                {
                    // Compare with higher neighbor bin.
                    if (nciFinal_h[ibin * nciFinalLp + ifold] > nciFinal_h[(ibin + 1) * nciFinalLp + ifold])
                    {
                        // Calculate doppler bin neighbors (with modulo).
                        int32_t preiFold  = ((ifold - 1) + Nr / repeatFold) % (Nr / repeatFold);
                        int32_t postiFold = (ifold + 1) % (Nr / repeatFold);

                        // Compare with left doppler bin.
                        if (nciFinal_h[ibin * nciFinalLp + ifold] > nciFinal_h[ibin * nciFinalLp + preiFold])
                        {
                            // Compare with right doppler bin.
                            if (nciFinal_h[ibin * nciFinalLp + ifold] > nciFinal_h[ibin * nciFinalLp + postiFold])
                            {
                                if (peakCount >= PEAKDET_MAX_PEAK_COUNT)
                                {
                                    std::cerr << "peakDetectionRefImpl: peak buffer overflow, truncating results\n";
                                    break;
                                }

                                // Peak found.
                                rangeIdxPtr[peakCount]       = ibin;
                                dopplerFoldIdxPtr[peakCount] = ifold;

                                // Search for TX peak position in doppler domain.
                                std::vector<T> candi(repeatFold);
                                for (int32_t iCand = 0; iCand < repeatFold; iCand++)
                                {
                                    // Sum values at the 4 supposed locations in doppler domain.
                                    for (int32_t tx = 0; tx < NofTx; tx++)
                                    {
                                        int32_t dopplerIdx = (DDMspace[tx] + ifold + (Nr / repeatFold * iCand)) % Nr;
                                        candi[iCand] += nciRx_h[ibin * nciRxLp + dopplerIdx];
                                    }
                                }

                                // Get TX doppler-fold position by searching max value.
                                int32_t mi = std::distance(candi.begin(), std::max_element(candi.begin(), candi.end()));

                                // Calculate and store doppler index of TX.
                                dopplerIdxPtr[peakCount] = (ifold + (Nr / repeatFold * mi)) % Nr;

                                // Get RD map value at peak location for all RX channels.
                                for (int32_t tx = 0; tx < NofTx; tx++)
                                {
                                    for (int32_t rx = 0; rx < NofRx; rx++)
                                    {
                                        int32_t dopplerIdx = (DDMspace[tx] + ifold + (Nr / repeatFold * mi)) % Nr;
                                        int32_t snapIdx    = rx + tx * NofRx;

                                        // Store complex values as real/imag pairs.
                                        auto val0 = FT2D_h[ibin * NofRx * Nr + rx * Nr + dopplerIdx].real();
                                        auto val1 = FT2D_h[ibin * NofRx * Nr + rx * Nr + dopplerIdx].imag();

                                        // The template parameter T is either float or int32_t. When T is int32_t, it is just
                                        // an assignment. When T is float, it is a bit-wise copy. So use memcpy to handle both cases.
                                        memcpy(&snap_h[peakCount * NofTx * NofRx * 2 + snapIdx * 2], &val0,
                                               sizeof(val0));
                                        memcpy(&snap_h[peakCount * NofTx * NofRx * 2 + snapIdx * 2 + 1], &val1,
                                               sizeof(val1));
                                    }
                                }

                                peakCount++;
                            }
                        }
                    }
                }
            }
        }
    }

    *peakCount_h = peakCount;
}

} // namespace

int32_t checkPeakDetectionOutput(int32_t NofTx, int32_t NofRx, NVCVTensorHandle refPeakCountTensorHandle,
                                 NVCVTensorHandle refPeakIndexTensorHandle, NVCVTensorHandle refSnapTensorHandle,
                                 NVCVTensorHandle pvaPeakCountTensorHandle, NVCVTensorHandle pvaPeakIndexTensorHandle,
                                 NVCVTensorHandle pvaSnapTensorHandle, float maxErrRatio)
{
    NVCVTensorData refPeakCountData;
    nvcvTensorExportData(refPeakCountTensorHandle, &refPeakCountData);
    uint32_t *refPeakCount_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&refPeakCount_h, (void *)refPeakCountData.buffer.strided.basePtr) !=
        CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    const uint32_t refPeakCount = *refPeakCount_h;

    NVCVTensorData pvaPeakCountData;
    nvcvTensorExportData(pvaPeakCountTensorHandle, &pvaPeakCountData);
    uint32_t *pvaPeakCount_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&pvaPeakCount_h, (void *)pvaPeakCountData.buffer.strided.basePtr) !=
        CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    const uint32_t pvaPeakCount = *pvaPeakCount_h;

    if (refPeakCount != pvaPeakCount)
    {
        std::cout << __func__ << ": refPeakCount != pvaPeakCount. refPeakCount = " << refPeakCount
                  << ", pvaPeakCount = " << pvaPeakCount << std::endl;
        return 1;
    }

    NVCVTensorData refPeakIndexData;
    nvcvTensorExportData(refPeakIndexTensorHandle, &refPeakIndexData);
    uint32_t *refPeakIndex_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&refPeakIndex_h, (void *)refPeakIndexData.buffer.strided.basePtr) !=
        CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    uint32_t *refRangeIdxPtr       = refPeakIndex_h;
    uint32_t *refDopplerFoldIdxPtr = refPeakIndex_h + PEAKDET_MAX_PEAK_COUNT;
    uint32_t *refDopplerIdxPtr     = refPeakIndex_h + PEAKDET_MAX_PEAK_COUNT * 2;

    NVCVTensorData pvaPeakIndexData;
    nvcvTensorExportData(pvaPeakIndexTensorHandle, &pvaPeakIndexData);
    uint32_t *pvaPeakIndex_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&pvaPeakIndex_h, (void *)pvaPeakIndexData.buffer.strided.basePtr) !=
        CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }
    uint32_t *pvaRangeIdxPtr       = pvaPeakIndex_h;
    uint32_t *pvaDopplerFoldIdxPtr = pvaPeakIndex_h + PEAKDET_MAX_PEAK_COUNT;
    uint32_t *pvaDopplerIdxPtr     = pvaPeakIndex_h + PEAKDET_MAX_PEAK_COUNT * 2;

    NVCVTensorData refSnapData;
    nvcvTensorExportData(refSnapTensorHandle, &refSnapData);
    std::complex<int32_t> *refSnap_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&refSnap_h, (void *)refSnapData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    NVCVTensorData pvaSnapData;
    nvcvTensorExportData(pvaSnapTensorHandle, &pvaSnapData);
    std::complex<int32_t> *pvaSnap_h = nullptr;
    if (CupvaMemGetHostPointer((void **)&pvaSnap_h, (void *)pvaSnapData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw std::runtime_error("Host pointer mapping failed");
    }

    int32_t err = 0;
    for (int32_t i = 0; i < refPeakCount; i++)
    {
        bool indexMismatch =
            (refRangeIdxPtr[i] != pvaRangeIdxPtr[i] || refDopplerFoldIdxPtr[i] != pvaDopplerFoldIdxPtr[i] ||
             refDopplerIdxPtr[i] != pvaDopplerIdxPtr[i]);
        const int32_t snapLen = NofTx * NofRx;
        bool snapMismatch =
            memcmp(&refSnap_h[i * snapLen], &pvaSnap_h[i * snapLen], snapLen * sizeof(std::complex<int32_t>));

        if (indexMismatch || snapMismatch)
        {
            if (!err)
            {
                std::cout << "Mismatch format[rangeIdx, dopplerFoldIdx, dopplerIdx]: ref = (rangeIdx, dopplerFoldIdx, "
                          << "dopplerIdx), pva = (rangeIdx, dopplerFoldIdx, dopplerIdx)" << std::endl;
            }

            std::cout << "Mismatch[" << i << "]: ref = (" << refRangeIdxPtr[i] << ", " << refDopplerFoldIdxPtr[i]
                      << ", " << refDopplerIdxPtr[i] << "), pva = (" << pvaRangeIdxPtr[i] << ", "
                      << pvaDopplerFoldIdxPtr[i] << ", " << pvaDopplerIdxPtr[i] << ")" << std::endl;

            std::cout << "ref snap: ";
            for (int32_t j = 0; j < snapLen; j++)
            {
                std::cout << refSnap_h[i * snapLen + j].real() << " " << refSnap_h[i * snapLen + j].imag();
            }
            std::cout << std::endl;

            std::cout << "pva snap: ";
            for (int32_t j = 0; j < snapLen; j++)
            {
                std::cout << pvaSnap_h[i * snapLen + j].real() << " " << pvaSnap_h[i * snapLen + j].imag();
            }
            std::cout << std::endl;
            err++;
        }

        const int32_t maxErr = std::max(5, static_cast<int32_t>(refPeakCount * maxErrRatio));
        if (err > maxErr)
        {
            break;
        }
    }

    return err;
}

void generateRandomPeakDetectionInput(NVCVTensorHandle tensorHandle)
{
    NVCVDataType dataType;
    nvcvTensorGetDataType(tensorHandle, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_F32:
    case NVCV_DATA_TYPE_2F32:
        generateRandomPeakDetectionInputImpl<float>(tensorHandle);
        break;
    case NVCV_DATA_TYPE_U32:
        generateRandomPeakDetectionInputImpl<uint32_t>(tensorHandle);
        break;
    case NVCV_DATA_TYPE_S32:
    case NVCV_DATA_TYPE_2S32:
        generateRandomPeakDetectionInputImpl<int32_t>(tensorHandle);
        break;
    }
}

int32_t loadTensorInput(NVCVTensorHandle tensorHandle, const char *directory, const char *filename)
{
    return loadTensorInputImpl(tensorHandle, directory, filename);
}

void peakDetectionRef(NVCVTensorHandle *inHandles, int32_t NofTx, int32_t repeatFold, NVCVTensorHandle *outHandles)
{
    NVCVTensorHandle FT2DTensorHandle      = inHandles[0];
    NVCVTensorHandle nciRxTensorHandle     = inHandles[1];
    NVCVTensorHandle nciFinalTensorHandle  = inHandles[2];
    NVCVTensorHandle noiseEstTensorHandle  = inHandles[3];
    NVCVTensorHandle peakCountTensorHandle = outHandles[0];
    NVCVTensorHandle peakIndexTensorHandle = outHandles[1];
    NVCVTensorHandle snapTensorHandle      = outHandles[2];

    NVCVDataType dataType;
    nvcvTensorGetDataType(nciFinalTensorHandle, &dataType);

    switch (dataType)
    {
    case NVCV_DATA_TYPE_F32:
        return peakDetectionRefImpl<float>(FT2DTensorHandle, nciRxTensorHandle, nciFinalTensorHandle,
                                           noiseEstTensorHandle, NofTx, repeatFold, peakCountTensorHandle,
                                           peakIndexTensorHandle, snapTensorHandle);
    case NVCV_DATA_TYPE_U32:
        return peakDetectionRefImpl<uint32_t>(FT2DTensorHandle, nciRxTensorHandle, nciFinalTensorHandle,
                                              noiseEstTensorHandle, NofTx, repeatFold, peakCountTensorHandle,
                                              peakIndexTensorHandle, snapTensorHandle);
    default:
        std::cerr << "Unsupported data type in " << __func__ << "\n";
        break;
    }
}
