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

/**
 * @file OpLidarHistogramConversion.h
 *
 * @brief Defines types and functions for converting histogram signals from LiDAR systems to XYZ, range, and reflectance.
 * @defgroup PVA_OPERATOR_ALGORITHM_LIDAR_HISTOGRAM_CONVERSION LidarHistogramConversion
 * @{
 */

#ifndef PVA_SOLUTIONS_OPLIDARHISTOGRAMCONVERSION_H
#define PVA_SOLUTIONS_OPLIDARHISTOGRAMCONVERSION_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief LiDAR data packing type.
 *
 * LiDAR histogram data may be packed in one of the following ways:
 * - NONE: The data is not packed. Each element in the C dimension has the data type of the input tensor.
 * - RAW12: Each element is 12 bits, with every pair of elements (A,B) packed together in 24 bits (3 bytes):
 *          {[A4:11], [B4:11], [A0:3, B0:3]}
 *   (The input tensor data type is ignored other than to understand the storage size)
 */
typedef enum
{
    /// LiDAR histogram data is not packed.
    PVA_LIDAR_DATA_PACKING_TYPE_NONE = 0,
    /// LiDAR histogram data is packed as 12 bits per element, 2 elements per 3 bytes.
    PVA_LIDAR_DATA_PACKING_TYPE_RAW12 = 1
} PvaLidarDataPackingType;

/**
 * @brief LiDAR histogram specification.
 *
 * Metadata that defines how to interpret the layout in the C dimension of the input and output tensors. A 'pixel' in this
 * context is defined as the collection of all elements in the C dimension for a given 2D index i,j where i is in [0, H)
 * and j is in [0, W).
 *
 * - packingType: The PvaLidarDataPackingType of the input tensor data. See PvaLidarDataPackingType for more details.
 * - numHistPerPixel: Number of histograms within one pixel (1 to 8).
 * - numBinsPerHist: Number of bins per histogram (3 to 2048).
 * - numPeaksPerHist: Number of peaks to detect and output per histogram (1 to 8).
 * - pixelHeaderSize: Size in elements of the header preceding the pixel data (0 to 64).
 * - histHeaderSize: Size in elements of the header preceding each histogram within a pixel (0 to 16).
 *
 * The packingType determines the size of one pixel element, either 16 bits (NONE) or 12 bits (RAW12).
 *
 * Multiple histograms may be present at each pixel.
 * Layout of one pixel: {[pixelHeader,] histogram0, histogram1, ..., histogramN-1} where the pixelHeader size in bits is
 * pixelHeaderSize*pixelElementBits and N = numHistPerPixel.
 *
 * Layout of one histogram: {[histHeader,] bin0, bin1, bin2, ..., binK-1} where histHeader size in bits is
 * histHeaderSize*pixelElementBits, each bin is one pixel element, and K = numBinsPerHist.
 *
 * Total size in bits of one input pixel is therefore:
 * pixelBits = (pixelHeaderSize + numHistPerPixel*(histHeaderSize + numBinsPerHist))*pixelElementBits
 * The tensor C dimension must be at least large enough to hold all pixel elements, ie C*sizeof(T) >= pixelBits/8.
 *
 * The size of an output pixel depends on the output tensor and numPeaksPerHist. See pvaLidarHistogramConversionSubmit.
 *
 * Note: The pixel and histogram headers (if present) contain sensor-specific information and therefore are treated
 * as opaque data at the host API level. The implementation of this operator ignores the header data, but unpacks it
 * into separate buffers in VMEM prior to processing the histogram data.
 *
 * Note: When using RAW12 packing type, the pixelHeaderSize, histHeaderSize, and numBinsPerHist must be even numbers.
 */
typedef struct PvaLidarHistogramSpecRec
{
    /// The packing type of the input tensor data.
    PvaLidarDataPackingType packingType;
    /// The number of histograms per pixel.
    int32_t numHistPerPixel;
    /// The number of bins per histogram of the input.
    int32_t numBinsPerHist;
    /// The number of peaks per histogram to detect and output.
    int32_t numPeaksPerHist;
    /// The size of the pixel header in elements.
    int32_t pixelHeaderSize;
    /// The size of the histogram header in elements.
    int32_t histHeaderSize;
} PvaLidarHistogramSpec;

/** Constructs an instance of the LidarHistogramConversion operator.
 *
 * This operator converts LiDAR histograms to XYZ, range and/or reflectance maps.
 *
 * **Input Limitations:**
 *
 *  Histogram tensor requirements:
 *  - Layout: NHWC
 *  - Data type (T) if packingType is PVA_LIDAR_DATA_PACKING_TYPE_NONE: U16
 *  - Histogram bin values if packingType is PVA_LIDAR_DATA_PACKING_TYPE_NONE: at most 8191. The smoothing filter
 *    sums up to 1024 times a bin value in a signed 24-bit accumulator, so larger values give wrong results. Header
 *    elements are not limited.
 *  - Data type (T) if packingType is PVA_LIDAR_DATA_PACKING_TYPE_RAW12: U8, U16, or U32
 *  - N == 1
 *  - H >= 1
 *  - W >= 1
 *  - C must be chosen such that there is enough space for the histogram data. See PvaLidarHistogramSpec.
 *
 * **Parameters:**
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] histogramRequirements Pointer to the NVCVTensorRequirements structure that defines the histogram tensor shape.
 *
 * @param [in] histogramSpec Pointer to the PvaLidarHistogramSpec defining the histogram metadata.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT A pointer is null or the tensor requirements and/or specification are incompatible.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaLidarHistogramConversionCreate(NVCVOperatorHandle *handle,
                                             NVCVTensorRequirements const *const histogramRequirements,
                                             PvaLidarHistogramSpec const *const histogramSpec);

/**
 * @brief LiDAR histogram parameters.
 *
 * Parameters that define the time and intensity scales of the histogram data.
 *
 * - offsetNs: The global offset in nanoseconds that defines the start of bin index 0.
 * - binSizeNs: The time interval in nanoseconds spanned by one bin index.
 * - rangeScale: Range output is in meters when scale is 1.0. Set to 100.0 to output in centimeters, etc.
 * - maxIntensity: The maximum intensity value for computing reflectance.
 *
 * A histogram bin value at an index k represents the number of photons captured over a time interval of binSizeNs
 * beginning at offsetNs + k * binSizeNs. The sum of 3 bins centered at a peak is the measured intensity of the peak.
 * The reflectance output is computed as the ratio of the measured intensity to the maxIntensity.
 */
typedef struct PvaLidarHistogramParamsRec
{
    /// The global offset in nanoseconds that defines the start of bin index 0.
    float offsetNs;
    /// The time interval in nanoseconds spanned by one bin index.
    float binSizeNs;
    /// The scale factor to convert range output where 1.0 is meters.
    float rangeScale;
    /// The maximum intensity value for computing reflectance.
    float maxIntensity;
} PvaLidarHistogramParams;

#ifdef __cplusplus
}

/**
 * Executes the conversion of LiDAR histograms to XYZ, range and/or reflectance.
 *
 * First, the locations of peaks are estimated from the histogram data. The time-of-flight in nanoseconds of a given
 * histogram bin index k can be expressed in simplified form as:
 *
 *   tof = offsetNs + binSizeNs * (k)
 *
 * The actual estimated time-of-flight for a given peak refines k by parabolic interpolation through the smoothed
 * histogram bins at k-1, k and k+1.
 *
 * Calibration tensors are required to convert to range and/or xyz outputs from the time-of-flight. For a given pixel
 * location (i,j), histogram (n), and peak (p), the range is computed as:
 *
 *   range[i][j][n*numPeaksPerHist+p] = rangeBias[i][j] + rangeScale * (tof * C)
 *
 * Where C is the speed of light through air in meters per nanosecond.
 *
 * The XYZ calibration tensor is used to convert from polar coordinates (range and angle, where angle is inherent in the
 * pixel location) to cartesian coordinates. XYZ is computed from the range as:
 *
 *  xyz[i][j][n*numPeaksPerHist+p]       = xyzCalibration[i][j][0] * range[n*numPeaksPerHist+p]
 *  xyz[i][j][T + n*numPeaksPerHist+p]   = xyzCalibration[i][j][1] * range[n*numPeaksPerHist+p]
 *  xyz[i][j][2*T + n*numPeaksPerHist+p] = xyzCalibration[i][j][2] * range[n*numPeaksPerHist+p]
 *  where T = numHistPerPixel*numPeaksPerHist.
 *
 * The reflectance is computed as the ratio of the measured intensity to the maxIntensity. Intensity of a peak is
 * estimated as the sum of 3 histogram bins centered at the peak bin index. The reflectance value is not clipped. ie,
 * if measured intensity is > maxIntensity the reflectance output is > 1.0.
 *
 * **Input Limitations:**
 *
 * Histogram tensor:
 * - Must exactly match the histogram tensor requirements specified in pvaLidarHistogramConversionCreate.
 *
 * Range bias tensor (required if range and/or xyz outputs are not NULL):
 * - Layout: NHWC
 * - Data type: [F32]
 * - N == 1
 * - H,W == histogram H,W
 * - C == 1
 *
 * XYZ calibration tensor (required if xyz output is not NULL):
 * - Layout: NHWC
 * - Data type: [F32]
 * - N == 1
 * - H,W == histogram H,W
 * - C == 3
 *
 * **Output Limitations:**
 *
 * Outputs are optional, but at least one must be non-NULL.
 *
 * Range tensor (optional):
 * - Layout: NHWC
 * - Data type: [F32]
 * - N == 1
 * - H,W == histogram H,W
 * - C == numHistPerPixel*numPeaksPerHist
 *
 * XYZ tensor (optional):
 * - Layout: NHWC
 * - Data type: [F32]
 * - N == 1
 * - H,W == histogram H,W
 * - C == 3*numHistPerPixel*numPeaksPerHist
 *
 * Reflectance tensor (optional):
 * - Layout: NHWC
 * - Data type: [F32]
 * - N == 1
 * - H,W == histogram H,W
 * - C == numHistPerPixel*numPeaksPerHist
 *
 * **Parameters:**
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid CUPVA stream.
 *
 * @param [in] params Pointer to the PvaLidarHistogramParams with constants that apply to this submission.
 *
 * @param [in] histogram Input histogram tensor handle.
 *
 * @param [in] rangeBias Input range bias tensor handle (optional).
 *
 * @param [in] xyzCalibration Input XYZ calibration tensor handle (optional).
 *
 * @param [out] range Output range tensor handle (optional).
 *
 * @param [out] xyz Output XYZ tensor handle (optional).
 *
 * @param [out] reflectance Output reflectance tensor handle (optional).
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or one or more input and/or output tensors are invalid
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaLidarHistogramConversionSubmit(NVCVOperatorHandle const handle, cupvaStream_t const stream,
                                             PvaLidarHistogramParams const *const params,
                                             NVCVTensorHandle const histogram, NVCVTensorHandle const rangeBias,
                                             NVCVTensorHandle const xyzCalibration, NVCVTensorHandle range,
                                             NVCVTensorHandle xyz, NVCVTensorHandle reflectance);

/**
 * Submits the LidarHistogramConversion operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] params Pointer to the PvaLidarHistogramParams with constants that apply to this submission.
 * @param [in] histogram Input histogram tensor handle.
 * @param [in] rangeBias Input range bias tensor handle (optional).
 * @param [in] xyzCalibration Input XYZ calibration tensor handle (optional).
 * @param [out] range Output range tensor handle (optional).
 * @param [out] xyz Output XYZ tensor handle (optional).
 * @param [out] reflectance Output reflectance tensor handle (optional).
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or one or more input and/or output tensors are invalid
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval NVCV_ERROR_NOT_IMPLEMENTED  CUDA stream support is not available.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaLidarHistogramConversionSubmit(NVCVOperatorHandle const handle, cudaStream_t const stream,
                                             PvaLidarHistogramParams const *const params,
                                             NVCVTensorHandle const histogram, NVCVTensorHandle const rangeBias,
                                             NVCVTensorHandle const xyzCalibration, NVCVTensorHandle range,
                                             NVCVTensorHandle xyz, NVCVTensorHandle reflectance);
#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPLIDARHISTOGRAMCONVERSION_H */
