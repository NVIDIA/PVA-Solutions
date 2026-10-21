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

/*
 * @file lidar_histogram_conversion_ref.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef LIDAR_HISTOGRAM_CONVERSION_REF_H
#define LIDAR_HISTOGRAM_CONVERSION_REF_H

#include <OpLidarHistogramConversion.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate random histogram input data
 *
 * @param histogramHandle Handle to the histogram tensor to fill with random data
 * @param histogramSpec Pointer to the histogram specification
 * @param params Pointer to the histogram parameters (used for maxIntensity scaling)
 */
void generateRandomLidarHistogramInput(NVCVTensorHandle histogramHandle,
                                       PvaLidarHistogramSpec const *const histogramSpec,
                                       PvaLidarHistogramParams const *const params);

/**
 * @brief Generate range bias tensor data
 *
 * Fills the range bias tensor with a constant value.
 *
 * @param rangeBiasHandle Handle to the range bias tensor to fill
 * @param biasValue Constant bias value to fill the tensor with
 */
void generateRangeBiasTensor(NVCVTensorHandle rangeBiasHandle, float biasValue);

/**
 * @brief Generate XYZ calibration tensor data
 *
 * Fills the XYZ calibration tensor with a constant value for all channels (X, Y, Z).
 *
 * @param xyzCalibrationHandle Handle to the XYZ calibration tensor to fill
 * @param calibrationValue Constant calibration value to fill the tensor with
 */
void generateXyzCalibrationTensor(NVCVTensorHandle xyzCalibrationHandle, float calibrationValue);

/**
 * @brief Compare reference and PVA output tensors
 *
 * @param refTensorHandle Handle to the reference tensor
 * @param pvaTensorHandle Handle to the PVA output tensor
 * @param tolerance Tolerance for floating point comparison
 * @return 0 if outputs match, non-zero otherwise
 */
int32_t checkLidarHistogramConversionOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                                            float tolerance);

/**
 * @brief Reference implementation of LiDAR histogram conversion
 *
 * @param histogram Input histogram tensor handle
 * @param rangeBias Input range bias tensor handle (optional)
 * @param xyzCalibration Input XYZ calibration tensor handle (optional)
 * @param range Output range tensor handle (optional)
 * @param xyz Output XYZ tensor handle (optional)
 * @param reflectance Output reflectance tensor handle (optional)
 * @param params Pointer to the histogram parameters
 * @param histogramSpec Pointer to the histogram specification
 */
void lidarHistogramConversionRef(NVCVTensorHandle histogram, NVCVTensorHandle rangeBias,
                                 NVCVTensorHandle xyzCalibration, NVCVTensorHandle range, NVCVTensorHandle xyz,
                                 NVCVTensorHandle reflectance, PvaLidarHistogramParams const *const params,
                                 PvaLidarHistogramSpec const *const histogramSpec);

#ifdef __cplusplus
}
#endif

#endif /* LIDAR_HISTOGRAM_CONVERSION_REF_H */
