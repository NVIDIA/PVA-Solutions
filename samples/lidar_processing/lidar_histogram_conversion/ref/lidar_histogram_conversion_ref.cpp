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

#include "lidar_histogram_conversion_ref.h"

#include "debug_options.h"

#include <OpLidarHistogramConversion.h>
#include <cupva_host.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>

// Speed of light in meters per nanosecond
constexpr float SPEED_OF_LIGHT_M_PER_NS = 0.299792458f;

namespace {

/**
 * @brief Helper to get host pointer from tensor data
 */
template<typename T>
T *getTensorHostPointer(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);
    return tensorPtr_h;
}

/**
 * @brief Unpack RAW12 data from 3 bytes to 2 uint16_t values
 *
 * RAW12 format packs two 12-bit values into 3 bytes:
 *   byte[0] = A[11:4]
 *   byte[1] = B[11:4]
 *   byte[2] = A[3:0] | B[3:0]  (upper nibble A, lower nibble B)
 *
 * @param packed Pointer to 3-byte RAW12 data
 * @param val0 Output first 12-bit value
 * @param val1 Output second 12-bit value
 */
inline void UnpackRAW12(std::byte const *packed, uint16_t &val0, uint16_t &val1)
{
    val0 = (static_cast<uint16_t>(packed[0]) << 4) | ((static_cast<uint16_t>(packed[2]) >> 4) & 0x0F);
    val1 = (static_cast<uint16_t>(packed[1]) << 4) | (static_cast<uint16_t>(packed[2]) & 0x0F);
}

/**
  * @brief Pack 2 uint16_t values to RAW12 format (3 bytes)
  *
  * RAW12 format packs two 12-bit values into 3 bytes:
  *   byte[0] = A[11:4]
  *   byte[1] = B[11:4]
  *   byte[2] = A[3:0] | B[3:0]  (upper nibble A, lower nibble B)
  *
  * @param val0 First 12-bit value (upper bits will be masked)
  * @param val1 Second 12-bit value (upper bits will be masked)
  * @param packed Output pointer to 3-byte RAW12 data
  */
inline void PackRAW12(uint16_t val0, uint16_t val1, std::byte *packed)
{
    val0 &= 0x0FFF; // Mask to 12 bits
    val1 &= 0x0FFF; // Mask to 12 bits

    packed[0] = static_cast<std::byte>(static_cast<uint8_t>(val0 >> 4));                            // A[11:4]
    packed[1] = static_cast<std::byte>(static_cast<uint8_t>(val1 >> 4));                            // B[11:4]
    packed[2] = static_cast<std::byte>(static_cast<uint8_t>(((val0 & 0x0F) << 4) | (val1 & 0x0F))); // A[3:0]|B[3:0]
}

/**
  * @brief Read and unpack pixel data from tensor to local buffer
  *
  * Reads data for a single pixel from the tensor and unpacks it if necessary.
  * For RAW12 packed data, unpacks 24b pairs of 12b values into pairs of uint16_t.
  * For unpacked data, directly copies uint16_t values.
  *
  * @param tensorPtr Pointer to tensor data (uint16_t* or uint8_t* depending on packing type)
  * @param elementsPerPixel Number of elements (histogram values) per pixel
  * @param packingType Packing type (PVA_LIDAR_DATA_PACKING_TYPE_NONE or PVA_LIDAR_DATA_PACKING_TYPE_RAW12)
  * @param outputBuffer Output buffer for unpacked uint16_t values (must be pre-allocated)
  */
void UnpackPixelData(std::byte const *tensorPtr, int32_t elementsPerPixel, int32_t packingType, uint16_t *outputBuffer)
{
    if (packingType == PVA_LIDAR_DATA_PACKING_TYPE_NONE)
    {
        // Direct copy for unpacked data
        memcpy(outputBuffer, tensorPtr, elementsPerPixel * sizeof(uint16_t));
    }
    else if (packingType == PVA_LIDAR_DATA_PACKING_TYPE_RAW12)
    {
        // Unpack RAW12 data
        // For RAW12: each pair of 12-bit values uses 3 bytes
        std::byte const *src = tensorPtr;

        for (int32_t i = 0; i < elementsPerPixel; i += 2)
        {
            uint16_t val0, val1;
            UnpackRAW12(src, val0, val1);

            outputBuffer[i] = val0;
            if (i + 1 < elementsPerPixel)
            {
                outputBuffer[i + 1] = val1;
            }

            src += 3; // Move to next 3-byte group
        }
    }
}

/**
  * @brief Pack and write pixel data from local buffer to tensor
  *
  * Writes histogram data for a single pixel from local buffer to tensor, packing if necessary.
  * For RAW12 packed data, packs pairs of uint16_t into to 24-bit chunks.
  * For unpacked data, directly copies uint16_t values.
  *
  * @param inputBuffer Input buffer with uint16_t values to write
  * @param tensorPtr Pointer to tensor data (uint16_t* or uint8_t* depending on packing type)
  * @param elementsPerPixel Number of elements (histogram values) per pixel
  * @param packingType Packing type (PVA_LIDAR_DATA_PACKING_TYPE_NONE or PVA_LIDAR_DATA_PACKING_TYPE_RAW12)
  */
void PackPixelData(uint16_t const *inputBuffer, std::byte *tensorPtr, int32_t elementsPerPixel, int32_t packingType)
{
    if (packingType == PVA_LIDAR_DATA_PACKING_TYPE_NONE)
    {
        // Direct copy for unpacked data
        memcpy(tensorPtr, inputBuffer, elementsPerPixel * sizeof(uint16_t));
    }
    else if (packingType == PVA_LIDAR_DATA_PACKING_TYPE_RAW12)
    {
        // Pack to RAW12 data
        // For RAW12: each pair of 12-bit values uses 3 bytes
        std::byte *dst = tensorPtr;

        for (int32_t i = 0; i < elementsPerPixel; i += 2)
        {
            uint16_t val0 = inputBuffer[i];
            uint16_t val1 = (i + 1 < elementsPerPixel) ? inputBuffer[i + 1] : 0;

            PackRAW12(val0, val1, dst);
            dst += 3; // Move to next 3-byte group
        }
    }
}

/**
 * @brief Add a Gaussian peak to the histogram
 *
 * Whatever energy from a gaussian at location `center` overlaps the histogram is added to the histogram. ie, center
 * does not necessarily have to be within the bounds of the histogram.
 *
 * @param bins Output histogram bins
 * @param numBins Total number of bins
 * @param center Center bin of the peak
 * @param amplitude Peak amplitude
 * @param sigma Standard deviation (controls peak width)
 */
void addGaussianPeak(std::vector<float> &bins, int32_t numBins, float center, float amplitude, float sigma)
{
    for (int32_t i = 0; i < numBins; i++)
    {
        float x        = static_cast<float>(i) - center;
        float gaussian = amplitude * std::exp(-(x * x) / (2.0f * sigma * sigma));
        bins[i] += gaussian;
    }
}

} // namespace

void generateRandomLidarHistogramInput(NVCVTensorHandle histogramHandle,
                                       PvaLidarHistogramSpec const *const histogramSpec,
                                       PvaLidarHistogramParams const *const params)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(histogramHandle, &tensorData);

    // base address of the histogram tensor
    std::byte *histogramPtr = getTensorHostPointer<std::byte>(histogramHandle);

    int32_t height   = tensorData.shape[1];
    int32_t width    = tensorData.shape[2];
    int32_t channels = tensorData.shape[3];

    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the maximum value based on maxIntensity divided by 3 (sum of 3 bins at peak)
    const float maxValue = params->maxIntensity / 3.0f;

    // Calculate elements per pixel (unpacked representation)
    int32_t const elementsPerPixel =
        histogramSpec->pixelHeaderSize +
        histogramSpec->numHistPerPixel * (histogramSpec->histHeaderSize + histogramSpec->numBinsPerHist);

    // Allocate local buffer for one pixel of unpacked data
    std::vector<uint16_t> pixelBuffer(elementsPerPixel);

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            int32_t idx = 0;

            // Generate pixel header with random values
            for (int32_t h = 0; h < histogramSpec->pixelHeaderSize; h++)
            {
                std::uniform_int_distribution<uint16_t> headerDist(0, 1000);
                uint16_t val       = headerDist(gen);
                pixelBuffer[idx++] = val;
            }

            // Generate each histogram for this pixel
            for (int32_t histIdx = 0; histIdx < histogramSpec->numHistPerPixel; histIdx++)
            {
                // Generate histogram header
                for (int32_t h = 0; h < histogramSpec->histHeaderSize; h++)
                {
                    std::uniform_int_distribution<uint16_t> headerDist(0, 100);
                    uint16_t val       = headerDist(gen);
                    pixelBuffer[idx++] = val;
                }

                // Generate realistic histogram bins
                int32_t numBins = histogramSpec->numBinsPerHist;
                std::vector<float> bins(numBins, 0.0f);

                // Determine number of peaks
                std::uniform_int_distribution<int32_t> numPeaksDist(0, 4);
                int32_t numStrongPeaks = numPeaksDist(gen);
                int32_t numWeakPeaks   = numPeaksDist(gen);

                // Generate strong peaks (50%-80% of max)
                std::uniform_real_distribution<float> strongAmpDist(0.5f, 0.8f);
                std::uniform_real_distribution<float> centerDist(5.0f, static_cast<float>(numBins - 6));
                std::uniform_real_distribution<float> sigmaDist(1.0f, 2.0f); // 3-6 bin spread (2.5*sigma ≈ spread)

                for (int32_t p = 0; p < numStrongPeaks; p++)
                {
                    float center    = centerDist(gen);
                    float amplitude = strongAmpDist(gen) * maxValue;
                    float sigma     = sigmaDist(gen);
                    addGaussianPeak(bins, numBins, center, amplitude, sigma);
                }

                // Generate weak peaks (<50% of max)
                std::uniform_real_distribution<float> weakAmpDist(0.1f, 0.49f);

                for (int32_t p = 0; p < numWeakPeaks; p++)
                {
                    float center    = centerDist(gen);
                    float amplitude = weakAmpDist(gen) * maxValue;
                    float sigma     = sigmaDist(gen);
                    addGaussianPeak(bins, numBins, center, amplitude, sigma);
                }

                // Add small noise (<5% of max)
                std::uniform_real_distribution<float> noiseDist(0.0f, 0.05f * maxValue);
                for (int32_t b = 0; b < numBins; b++)
                {
                    bins[b] += noiseDist(gen);
                }

                // Clamp to 12-bit dynamic range and convert to uint16
                for (int32_t b = 0; b < numBins; b++)
                {
                    float value        = std::clamp(bins[b], 0.0f, 4095.0f);
                    pixelBuffer[idx++] = static_cast<uint16_t>(value);
                }
            }

            // byte address for the start of the pixel in the tensor
            std::byte *histogramPixelPtr =
                histogramPtr + y * tensorData.buffer.strided.strides[1] + x * tensorData.buffer.strided.strides[2];
            PackPixelData(pixelBuffer.data(), histogramPixelPtr, elementsPerPixel, histogramSpec->packingType);
        }
    }
}

void generateRangeBiasTensor(NVCVTensorHandle rangeBiasHandle, float biasValue)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(rangeBiasHandle, &tensorData);

    int32_t height   = tensorData.shape[1];
    int32_t width    = tensorData.shape[2];
    int32_t channels = tensorData.shape[3];

    // Verify data type is F32
    if (tensorData.dtype != NVCV_DATA_TYPE_F32)
    {
        printf("ERROR: Expected F32 data type for range bias tensor\n");
        return;
    }

    float *tensorPtr_h = getTensorHostPointer<float>(rangeBiasHandle);
    if (!tensorPtr_h)
    {
        printf("ERROR: Failed to get host pointer for range bias tensor\n");
        return;
    }

    int32_t linePitch = tensorData.buffer.strided.strides[1] / sizeof(float);

    // Fill the tensor with the constant bias value
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            for (int32_t c = 0; c < channels; c++)
            {
                tensorPtr_h[y * linePitch + x * channels + c] = biasValue;
            }
        }
    }
}

void generateXyzCalibrationTensor(NVCVTensorHandle xyzCalibrationHandle, float calibrationValue)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(xyzCalibrationHandle, &tensorData);

    int32_t height   = tensorData.shape[1];
    int32_t width    = tensorData.shape[2];
    int32_t channels = tensorData.shape[3];

    // Verify data type is F32
    if (tensorData.dtype != NVCV_DATA_TYPE_F32)
    {
        printf("ERROR: Expected F32 data type for XYZ calibration tensor\n");
        return;
    }

    float *tensorPtr_h = getTensorHostPointer<float>(xyzCalibrationHandle);
    if (!tensorPtr_h)
    {
        printf("ERROR: Failed to get host pointer for XYZ calibration tensor\n");
        return;
    }

    int32_t linePitch = tensorData.buffer.strided.strides[1] / sizeof(float);

    // Fill the tensor with the constant calibration value
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            for (int32_t c = 0; c < channels; c++)
            {
                tensorPtr_h[y * linePitch + x * channels + c] = calibrationValue;
            }
        }
    }
}

int32_t checkLidarHistogramConversionOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                                            float tolerance)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);

    int32_t height   = refTensorData.shape[1];
    int32_t width    = refTensorData.shape[2];
    int32_t channels = refTensorData.shape[3];

    // Check if data types match
    if (refTensorData.dtype != pvaTensorData.dtype)
    {
        printf("Data type mismatch: ref = %ld, pva = %ld\n", refTensorData.dtype, pvaTensorData.dtype);
        return 1;
    }

    // Handle float comparison
    if (refTensorData.dtype == NVCV_DATA_TYPE_F32)
    {
        float *ref = getTensorHostPointer<float>(refTensorHandle);
        float *pva = getTensorHostPointer<float>(pvaTensorHandle);

        int32_t linePitch  = refTensorData.buffer.strided.strides[1] / sizeof(float);
        int32_t pixelPitch = refTensorData.buffer.strided.strides[2] / sizeof(float);

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                for (int32_t c = 0; c < channels; c++)
                {
                    float refData = ref[y * linePitch + x * pixelPitch + c];
                    float cmpData = pva[y * linePitch + x * pixelPitch + c];
                    float diff    = std::abs(refData - cmpData);

                    if (diff > tolerance)
                    {
                        printf("Mismatch[%d, %d, %d]: ref = %f, pva = %f, diff = %f\n", y, x, c, refData, cmpData,
                               diff);
                        return 1;
                    }
                }
            }
        }
    }
    else if (refTensorData.dtype == NVCV_DATA_TYPE_U8)
    {
        uint8_t *ref = getTensorHostPointer<uint8_t>(refTensorHandle);
        uint8_t *pva = getTensorHostPointer<uint8_t>(pvaTensorHandle);

        int32_t linePitch  = refTensorData.buffer.strided.strides[1] / sizeof(uint8_t);
        int32_t pixelPitch = refTensorData.buffer.strided.strides[2] / sizeof(uint8_t);

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                for (int32_t c = 0; c < channels; c++)
                {
                    uint8_t refData = ref[y * linePitch + x * pixelPitch + c];
                    uint8_t cmpData = pva[y * linePitch + x * pixelPitch + c];

                    if (refData != cmpData)
                    {
                        printf("Mismatch[%d, %d, %d]: ref = %u, pva = %u\n", y, x, c, refData, cmpData);
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

namespace {

/**
 * @brief Apply 7-tap Gaussian filter to smooth histogram
 *
 * Uses a 7-tap Gaussian kernel with sigma=1.0:
 * [0.0044, 0.054, 0.242, 0.399, 0.242, 0.054, 0.0044]
 *
 * Edge handling: Uses symmetric boundary conditions (reflects values at edges)
 *
 * @param histogram Input histogram bins
 * @param smoothedHistogram Output smoothed histogram bins
 */
void SmoothHistogram(std::vector<uint16_t> const &histogram, std::vector<uint16_t> &smoothedHistogram)
{
    int32_t numBins = static_cast<int32_t>(histogram.size());

    // 7-tap Gaussian kernel scaled by 1024 (matches VPU SmoothHistogram_vi16).
    // sum = 5+55+248+408+248+55+5 = 1024; output right-shifted by 10 (floor, not round-to-nearest).
    const int32_t kernel_int[7] = {5, 55, 248, 408, 248, 55, 5};
    const int32_t kernelRadius  = 3; // (7-1)/2

    smoothedHistogram.resize(numBins);

    for (int32_t i = 0; i < numBins; i++)
    {
        int32_t sum = 0;

        // Apply convolution with the integer kernel
        for (int32_t k = -kernelRadius; k <= kernelRadius; k++)
        {
            int32_t idx = i + k;

            // Handle boundary conditions with symmetric reflection
            if (idx < 0)
            {
                idx = -idx; // Reflect at left edge
            }
            else if (idx >= numBins)
            {
                idx = 2 * numBins - idx - 2; // Reflect at right edge
            }

            // Clamp to valid range (safety check)
            idx = std::clamp(idx, 0, numBins - 1);

            sum += static_cast<int32_t>(histogram[idx]) * kernel_int[k + kernelRadius];
        }

        // Divide by 1024 via floor right-shift (matches VPU >> 10 behavior)
        smoothedHistogram[i] = static_cast<uint16_t>(sum >> 10);
    }
}

/**
 * @brief Detect peaks in a histogram
 *
 * Uses a simple peak detection algorithm that identifies all local maxima and sorts them using lowest index to break ties.
 *
 * @param histogram Input histogram bins
 * @param peaks Output peak indices
 * @param noiseThreshold Minimum bin value a peak must exceed to be a candidate
 */
void DetectPeaks(std::vector<uint16_t> const &histogram, std::vector<int32_t> &peaks, uint16_t noiseThreshold)
{
    int32_t numBins          = static_cast<int32_t>(histogram.size());
    int32_t numPeaksToDetect = static_cast<int32_t>(peaks.size());

    // Structure to hold candidate peaks with their values
    struct PeakCandidate
    {
        int32_t index;
        uint16_t value;
    };
    std::vector<PeakCandidate> candidates;

    // Find all local maxima (peaks)
    for (int32_t i = 0; i < numBins; i++)
    {
        uint16_t currentValue = histogram[i];

        // Find the start of the plateau (contiguous bins with equal values)
        int32_t plateauStart = i;
        while (plateauStart > 0 && histogram[plateauStart - 1] == currentValue)
        {
            plateauStart--;
        }

        // Find the end of the plateau
        int32_t plateauEnd = i;
        while (plateauEnd < numBins - 1 && histogram[plateauEnd + 1] == currentValue)
        {
            plateauEnd++;
        }

        // Check if this is a peak (plateau value > neighbors on both sides)
        uint16_t prevValue = (plateauStart > 0) ? histogram[plateauStart - 1] : 0;
        uint16_t nextValue = (plateauEnd < numBins - 1) ? histogram[plateauEnd + 1] : 0;

        if (currentValue > prevValue && currentValue > nextValue)
        {
            // Skip candidates that do not exceed the noise threshold.
            if (currentValue > noiseThreshold)
            {
                // Calculate the middle bin of the plateau
                int32_t middleBin = (plateauStart + plateauEnd) / 2;
                candidates.push_back({middleBin, currentValue});
            }

            // Skip to the end of the plateau to avoid detecting the same peak multiple times
            i = plateauEnd;
        }
    }

    // Sort candidates by peak value (descending), then by index (ascending) for stability
    std::sort(candidates.begin(), candidates.end(), [](PeakCandidate const &a, PeakCandidate const &b) {
        if (a.value != b.value)
            return a.value > b.value; // Higher value first
        return a.index < b.index;     // Earlier index first for equal values
    });

    // Fill the peaks vector with the top N peak indices
    // If fewer peaks were found than requested, fill remaining with -1
    int32_t numPeaksFound = std::min(numPeaksToDetect, static_cast<int32_t>(candidates.size()));

    for (int32_t i = 0; i < numPeaksFound; i++)
    {
        peaks[i] = candidates[i].index;
    }

    // Fill remaining slots with -1 to indicate no peak found
    for (int32_t i = numPeaksFound; i < numPeaksToDetect; i++)
    {
        peaks[i] = -1;
    }
}

/**
 * @brief Compute range values for detected peaks
 *
 * Range is computed from time-of-flight (TOF) using the formula:
 *   range = rangeBias + rangeScale * (tof * SPEED_OF_LIGHT_M_PER_NS)
 *
 * TOF is computed by:
 *   1. Finding sub-bin peak position via parabolic interpolation of 3 bins centered at peak
 *   2. Converting to time: tof = offsetNs + (sub_bin_index * binSizeNs)
 *
 * @param histogram Input histogram bins
 * @param peaks Input peak indices
 * @param offsetNs Time offset in nanoseconds
 * @param binSizeNs Bin size in nanoseconds
 * @param rangeScale Range scaling factor (e.g., 100.0 for m to cm conversion)
 * @param rangeBias Range bias value for this pixel
 * @param rangePtr Output range values (one per peak)
 */
void ComputeRange(std::vector<uint16_t> const &histogram, std::vector<int32_t> const &peaks, float offsetNs,
                  float binSizeNs, float rangeScale, float rangeBias, float *rangePtr)
{
    int32_t numBins  = static_cast<int32_t>(histogram.size());
    int32_t numPeaks = static_cast<int32_t>(peaks.size());

    // Pre-collapse constants to match the VPU vectorized formula:
    //   range = subBinIndex * rangeMul + rangeConst
    // where rangeMul = rangeScale * binSizeNs * c, rangeConst = rangeBias + rangeScale * offsetNs * c
    // This is algebraically equivalent to the expanded tof form and matches the VPU operation order.
    float rangeMul   = rangeScale * binSizeNs * SPEED_OF_LIGHT_M_PER_NS;
    float rangeConst = rangeBias + rangeScale * offsetNs * SPEED_OF_LIGHT_M_PER_NS;

    for (int32_t p = 0; p < numPeaks; p++)
    {
        int32_t peakIndex = peaks[p];

        // If no peak was detected (index -1), set range to 0
        if (peakIndex < 0)
        {
            rangePtr[p] = 0.0f;
            continue;
        }

        // Get the 3 bins centered at the peak for sub-bin interpolation
        float y_center = static_cast<float>(histogram[peakIndex]);
        float y_left   = (peakIndex > 0) ? static_cast<float>(histogram[peakIndex - 1]) : y_center;
        float y_right  = (peakIndex < numBins - 1) ? static_cast<float>(histogram[peakIndex + 1]) : y_center;

        // Parabolic interpolation to find sub-bin peak offset
        // Formula: offset = 0.5 * (y_left - y_right) / (y_left - 2*y_center + y_right)
        float denominator  = y_left - 2.0f * y_center + y_right;
        float subBinOffset = 0.0f;

        if (std::abs(denominator) > 1e-6f) // Avoid division by zero
        {
            subBinOffset = 0.5f * (y_left - y_right) / denominator;
            // Clamp offset to reasonable range [-0.5, 0.5]
            subBinOffset = std::clamp(subBinOffset, -0.5f, 0.5f);
        }

        // Compute sub-bin index, clamped to >= 0
        float subBinIndex = static_cast<float>(peakIndex) + subBinOffset;
        if (subBinIndex < 0.0f)
        {
            subBinIndex = 0.0f;
        }

        // Compute range: range = subBinIndex * rangeMul + rangeConst
        rangePtr[p] = subBinIndex * rangeMul + rangeConst;
    }
}

/**
 * @brief Compute XYZ coordinates for detected peaks
 *
 * XYZ coordinates are computed by multiplying the range values by the XYZ calibration
 * coefficients for this pixel:
 *   xyz[p + 0]        = xyzCalibration[0] * range[p]  (X coordinate)
 *   xyz[p + stride]   = xyzCalibration[1] * range[p]  (Y coordinate)
 *   xyz[p + 2*stride] = xyzCalibration[2] * range[p]  (Z coordinate)
 *
 * @param rangePtr Input range values (one per peak)
 * @param numPeaksPerHist Number of peaks per histogram
 * @param numHistPerPixel Number of histograms per pixel
 * @param xyzCalPtr XYZ calibration coefficients for this pixel (3 values: X, Y, Z)
 * @param xyzPtr Output XYZ coordinates (3 values per peak)
 */
void ComputeXYZ(float const *const rangePtr, int32_t numPeaksPerHist, int32_t numHistPerPixel,
                float const *const xyzCalPtr, float *xyzPtr)
{
    // stride from x to corresponding y for a given peak
    int32_t xyzStride = numPeaksPerHist * numHistPerPixel;

    for (int32_t p = 0; p < numPeaksPerHist; p++)
    {
        float range = rangePtr[p];

        // Compute XYZ coordinates by multiplying range with calibration coefficients
        xyzPtr[p]                 = xyzCalPtr[0] * range; // X coordinate
        xyzPtr[p + xyzStride]     = xyzCalPtr[1] * range; // Y coordinate
        xyzPtr[p + 2 * xyzStride] = xyzCalPtr[2] * range; // Z coordinate
    }
}

/**
 * @brief Compute reflectance of peaks in the histogram
 *
 * Reflectance is computed as the ratio of the intensity of the peak to the maximum intensity.
 * Intensity of a peak is the sum of the 3 bins centered at the peak.
 *
 * @param histogram Input histogram bins
 * @param peaks Input peak indices
 * @param maxIntensity Maximum intensity value (used for normalization)
 * @param reflectanceValues Output reflectance values
 */
void ComputeReflectance(std::vector<uint16_t> const &histogram, std::vector<int32_t> const &peaks, float maxIntensity,
                        std::vector<float> &reflectanceValues)
{
    int32_t numBins  = static_cast<int32_t>(histogram.size());
    int32_t numPeaks = static_cast<int32_t>(peaks.size());

    reflectanceValues.resize(numPeaks);

    for (int32_t p = 0; p < numPeaks; p++)
    {
        int32_t peakIndex = peaks[p];

        // If no peak was detected (index -1), set reflectance to 0
        if (peakIndex < 0)
        {
            reflectanceValues[p] = 0.0f;
            continue;
        }

        // Sum the 3 bins centered at the peak: [peakIndex-1, peakIndex, peakIndex+1]
        float intensity = 0.0f;

        for (int32_t offset = -1; offset <= 1; offset++)
        {
            int32_t binIndex = peakIndex + offset;

            // Clamp to valid histogram range
            if (binIndex >= 0 && binIndex < numBins)
            {
                intensity += static_cast<float>(histogram[binIndex]);
            }
        }

        // Compute reflectance as the ratio of intensity to maxIntensity
        reflectanceValues[p] = (maxIntensity > 0.0f) ? (intensity / maxIntensity) : 0.0f;
    }
}
} // namespace

void lidarHistogramConversionRef(NVCVTensorHandle histogram, NVCVTensorHandle rangeBias,
                                 NVCVTensorHandle xyzCalibration, NVCVTensorHandle range, NVCVTensorHandle xyz,
                                 NVCVTensorHandle reflectance, PvaLidarHistogramParams const *const params,
                                 PvaLidarHistogramSpec const *const histogramSpec)
{
    //
    // High-level steps:
    // 1. Parse histogram data according to histogramSpec (handle packing, headers, etc.)
    // 2. For each pixel (i,j) and histogram (n):
    //    a. Detect peaks in the histogram (up to numPeaksPerHist)
    //    b. For each peak (p):
    //       - Estimate time-of-flight using weighted average around peak
    //       - If range output requested:
    //         range[i][j][n*numPeaksPerHist+p] = rangeBias[i][j] + rangeScale * (tof * SPEED_OF_LIGHT_M_PER_NS)
    //       - If xyz output requested (xyz C dim is 'planar' layout x..,y...,z...):
    //         xyz[i][j][n*numPeaksPerHist+p] = xyzCalibration[i][j][0] * range
    //         xyz[i][j][n*numPeaksPerHist+p + (numPeaksPerHist*numHistPerPixel)] = xyzCalibration[i][j][1] * range
    //         xyz[i][j][n*numPeaksPerHist+p + 2*(numPeaksPerHist*numHistPerPixel)] = xyzCalibration[i][j][2] * range
    //         xyz[i][j][p + 2*(numPeaksPerHist*numHistPerPixel)] = xyzCalibration[i][j][2] * range
    //       - If reflectance output requested:
    //         reflectance[i][j][n*numPeaksPerHist+p] = intensity / maxIntensity
    //         (intensity is sum of 3 bins centered at peak)

    // Get histogram tensor info
    NVCVTensorData histogramData;
    assert(histogram);
    nvcvTensorExportData(histogram, &histogramData);

    int32_t height   = histogramData.shape[1];
    int32_t width    = histogramData.shape[2];
    int32_t channels = histogramData.shape[3];

    // base address of the histogram tensor
    std::byte *histogramPtr = getTensorHostPointer<std::byte>(histogram);

    NVCVTensorData rangeData, rangeBiasData;
    if (range)
    {
        nvcvTensorExportData(range, &rangeData);
        assert(rangeBias);
        nvcvTensorExportData(rangeBias, &rangeBiasData);
    }
    NVCVTensorData xyzData, xyzCalData;
    if (xyz)
    {
        nvcvTensorExportData(xyz, &xyzData);
        assert(xyzCalibration);
        nvcvTensorExportData(xyzCalibration, &xyzCalData);
    }
    NVCVTensorData reflectanceData;
    if (reflectance)
    {
        nvcvTensorExportData(reflectance, &reflectanceData);
    }

    // total size of a pixel in elements (headers + histograms)
    int32_t const elementsPerPixel =
        histogramSpec->pixelHeaderSize +
        histogramSpec->numHistPerPixel * (histogramSpec->histHeaderSize + histogramSpec->numBinsPerHist);

    // Allocate local buffer for one pixel of unpacked data
    std::vector<uint16_t> pixelBuffer(elementsPerPixel);

    // Pre-calculate line pitches for output tensors (converting byte strides to element counts)
    int32_t rangeLinePitch     = range ? rangeData.buffer.strided.strides[1] / sizeof(float) : 0;
    int32_t rangeBiasLinePitch = rangeBias ? rangeBiasData.buffer.strided.strides[1] / sizeof(float) : 0;
    int32_t xyzLinePitch       = xyz ? xyzData.buffer.strided.strides[1] / sizeof(float) : 0;
    int32_t xyzCalLinePitch    = xyzCalibration ? xyzCalData.buffer.strided.strides[1] / sizeof(float) : 0;
    int32_t reflectanceLinePitch =
        reflectance ? reflectanceData.buffer.strided.strides[1] /
                          (reflectanceData.dtype == NVCV_DATA_TYPE_U8 ? sizeof(uint8_t) : sizeof(float))
                    : 0;

    // Loop over pixels, extract histograms, detect peaks, compute outputs
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            // byte address for the start of the pixel in the tensor
            std::byte *histogramPixelPtr = histogramPtr + y * histogramData.buffer.strided.strides[1] +
                                           x * histogramData.buffer.strided.strides[2];

            // Unpack pixel data from tensor to local buffer
            UnpackPixelData(histogramPixelPtr, elementsPerPixel, histogramSpec->packingType, pixelBuffer.data());

            // skip pixel header
            /// NOTE: pixel header is for vendor-specific info. For example purposes, we assume it has no impact on the algorithm.
            uint16_t *histogramStartPtr = pixelBuffer.data() + histogramSpec->pixelHeaderSize;

            // For each histogram in this pixel, detect the peaks and compute the outputs
            std::vector<uint16_t> smoothedHistogramBins(histogramSpec->numBinsPerHist);
            std::vector<int32_t> peaks(histogramSpec->numPeaksPerHist);
            for (int32_t n = 0; n < histogramSpec->numHistPerPixel; n++)
            {
                // skip histogram header
                /// NOTE: additional vendor-specific info may be located in the histogram header. In this example we ignore it.
                uint16_t *histogramBinsPtr = histogramStartPtr +
                                             (histogramSpec->histHeaderSize + histogramSpec->numBinsPerHist) * n +
                                             histogramSpec->histHeaderSize;

                // smooth the histogram bins
                std::vector<uint16_t> histogramBins(histogramBinsPtr, histogramBinsPtr + histogramSpec->numBinsPerHist);
                SmoothHistogram(histogramBins, smoothedHistogramBins);

#if DEBUG_OVERWRITE_INPUT_WITH_SMOOTHED_HISTOGRAM
                // Copy smoothed data back to pixel buffer
                std::copy(smoothedHistogramBins.begin(), smoothedHistogramBins.end(), histogramBinsPtr);
#endif

                // Compute noise floor threshold as 1/8 of dynamic range above min
                uint16_t histMin        = *std::min_element(smoothedHistogramBins.begin(), smoothedHistogramBins.end());
                uint16_t histMax        = *std::max_element(smoothedHistogramBins.begin(), smoothedHistogramBins.end());
                uint16_t noiseThreshold = histMin + ((histMax - histMin) >> 3);

                // identify the indices of the peaks in the smoothed histogram
                DetectPeaks(smoothedHistogramBins, peaks, noiseThreshold);

                // Compute outputs
                if (range)
                {
                    float *rangePtr = getTensorHostPointer<float>(range);
                    NVCVTensorData rangeData_local;
                    nvcvTensorExportData(range, &rangeData_local);
                    int32_t rangeChannels = rangeData_local.shape[3];
                    int32_t rangeIndex    = y * rangeLinePitch + x * rangeChannels + n * histogramSpec->numPeaksPerHist;

#if DEBUG_WRITE_PEAK_INDICES_TO_RANGE
                    // Debug mode: Write peak indices directly to range tensor
                    for (size_t p = 0; p < peaks.size(); p++)
                    {
                        rangePtr[rangeIndex + p] = static_cast<float>(peaks[p]);
                    }
#else
                    float *rangeBiasPtr = getTensorHostPointer<float>(rangeBias);
                    NVCVTensorData rangeBiasData_local;
                    nvcvTensorExportData(rangeBias, &rangeBiasData_local);
                    int32_t rangeBiasChannels = rangeBiasData_local.shape[3];
                    int32_t rangeBiasIndex    = y * rangeBiasLinePitch + x * rangeBiasChannels;

                    ComputeRange(smoothedHistogramBins, peaks, params->offsetNs, params->binSizeNs, params->rangeScale,
                                 rangeBiasPtr[rangeBiasIndex], rangePtr + rangeIndex);

                    if (xyz)
                    {
                        float *xyzPtr = getTensorHostPointer<float>(xyz);
                        NVCVTensorData xyzData_local;
                        nvcvTensorExportData(xyz, &xyzData_local);
                        int32_t xyzChannels = xyzData_local.shape[3];
                        int32_t xyzIndex    = y * xyzLinePitch + x * xyzChannels + n * histogramSpec->numPeaksPerHist;

                        float *xyzCalPtr = getTensorHostPointer<float>(xyzCalibration);
                        NVCVTensorData xyzCalData_local;
                        nvcvTensorExportData(xyzCalibration, &xyzCalData_local);
                        int32_t xyzCalChannels = xyzCalData_local.shape[3];
                        int32_t xyzCalIndex    = y * xyzCalLinePitch + x * xyzCalChannels;

                        ComputeXYZ(rangePtr + rangeIndex, peaks.size(), histogramSpec->numHistPerPixel,
                                   xyzCalPtr + xyzCalIndex, xyzPtr + xyzIndex);
                    }
#endif
                }
                if (reflectance)
                {
                    std::vector<float> reflectanceValues(peaks.size());
                    ComputeReflectance(smoothedHistogramBins, peaks, params->maxIntensity, reflectanceValues);

                    NVCVTensorData reflectanceData_local;
                    nvcvTensorExportData(reflectance, &reflectanceData_local);
                    int32_t reflectanceChannels = reflectanceData_local.shape[3];
                    int32_t reflectanceIndex =
                        y * reflectanceLinePitch + x * reflectanceChannels + n * histogramSpec->numPeaksPerHist;

                    if (reflectanceData.dtype == NVCV_DATA_TYPE_U8)
                    {
                        uint8_t *reflectancePtr_u8 = getTensorHostPointer<uint8_t>(reflectance);
                        for (float v : reflectanceValues)
                        {
                            reflectancePtr_u8[reflectanceIndex++] =
                                static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f));
                        }
                    }
                    else
                    {
                        float *reflectancePtr_f32 = getTensorHostPointer<float>(reflectance);
                        for (float v : reflectanceValues)
                        {
                            reflectancePtr_f32[reflectanceIndex++] = v;
                        }
                    }
                }
            } // end of for (int32_t n = 0; n < histogramSpec->numHistPerPixel; n++)

#if DEBUG_OVERWRITE_INPUT_WITH_SMOOTHED_HISTOGRAM
            // Pack modified pixel data back to tensor (with smoothed histogram values and unmodified header data)
            PackPixelData(pixelBuffer.data(), histogramPixelPtr, elementsPerPixel, histogramSpec->packingType);
#endif
        }
    }
}
