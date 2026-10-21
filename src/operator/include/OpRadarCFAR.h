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
 * @file OpRadarCFAR.h
 *
 * @brief Defines types and functions to handle the Radar CFAR operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_CFAR RadarCFAR
 * @{
 */

#ifndef PVA_SOLUTIONS_OPRADARCFAR_H
#define PVA_SOLUTIONS_OPRADARCFAR_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates and initializes a Radar CFAR operator instance for target detection.
 *
 * @param [out] handle Pointer to store the created operator handle.
 *                     Must not be NULL. The handle is used for subsequent operator operations.
 *
 * @param [in] inputTensorRequirements Pointer to NVCVTensorRequirements structure defining input tensor specifications.
 *
 *  \b Input \b Tensor \b Requirements:
 *      - Data Layout:    [HW, HWC] (2D or 3D tensor)
 *      - Channels:       [1] (single channel - magnitude data)
 *      - Rank:           2 for [HW] 3 for [HWC] layout
 *      - Width:          Valid range (1-1024) and must be divisible by 8
 *      - Height:         Valid range (1-1024) and must be divisible by 8
 *      - Height * Width  Valid range (1-1024*512)
 *
 *  \b Supported \b Data \b Types:
 *      Data Type      | Support
 *      -------------- | -------------
 *      8bit  Unsigned | ❌ Not supported
 *      8bit  Signed   | ❌ Not supported
 *      16bit Unsigned | ❌ Not supported
 *      16bit Signed   | ❌ Not supported
 *      32bit Unsigned | ✅ Supported
 *      32bit Signed   | ✅ Supported
 *      32bit Float    | ❌ Not supported
 *      64bit Float    | ❌ Not supported
 *
 * @param [in] outDetectionListTensorRequirements Pointer to NVCVTensorRequirements structure defining output tensor specifications.
 *
 *  \b Detection \b List \b Format:
 *      The output tensor contains a detection list where each element contains the following fields:
 *       | Data Fields |               |
 *       |-------------|---------------|
 *       | int32_t     | verticalIdx   |
 *       |             |               |
 *       | int32_t     | horizontalIdx |
 *
 *  \b Output \b Tensor \b Characteristics:
 *      - Data Layout:    [HW, HWC] (2D or 3D)
 *      - Height:         Maximum Detection count. Should be between 1 and 8192
 *      - Width:          2 (number of fields in a detection list entry)
 *      - Channels:       [1]
 *      - Rank:           2 for [HW], 3 for [HWC]
 *
 *  \b Output \b Data \b Types:
 *      Data Type      | Support
 *      -------------- | -------------
 *      8bit  Unsigned | ❌ Not supported
 *      8bit  Signed   | ❌ Not supported
 *      16bit Unsigned | ❌ Not supported
 *      16bit Signed   | ❌ Not supported
 *      32bit Unsigned | ❌ Not supported
 *      32bit Signed   | ✅ Supported
 *      32bit Float    | ❌ Not supported
 *      64bit Float    | ❌ Not supported
 *
 *  \b Input/Output \b Tensor \b Dependencies:
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | ✅ Must match
 *      Channels      | ✅ Must match (1 channel)
 *
 * @param [in] algorithm PVARadarCFARType containing CFAR algorithm type.
 *                       Supported algorithms:
 *                       - PVA_CFAR_CA: Cell averaging with leading and trailing cells.
 *                       - PVA_CFAR_CA_GO: Greatest of leading and trailing cell averages.
 *                       - PVA_CFAR_CA_SO: Smallest of leading and trailing cell averages.
 *
 * @param [in] radarCFARParams Pointer to PvaRadarCFARParams structure containing CFAR algorithm configuration.
 *                  Must not be NULL.
 *                  See PvaRadarCFARParams structure documentation for detailed parameter constraints and behavior.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Invalid handle pointer or parameter values outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Insufficient memory to allocate operator resources.
 * @retval NVCV_SUCCESS                    Operator created successfully and ready for use.
 *
 * @note The operator handle must be destroyed using the appropriate destroy function
 *       when no longer needed to prevent memory leaks.
 */
NVCVStatus pvaRadarCFARCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inputTensorRequirements,
                              NVCVTensorRequirements const *const outDetectionListTensorRequirements,
                              PVARadarCFARType const algorithm, PvaRadarCFARParams const *const radarCFARParams);

/**
 * @brief Executes the Radar CFAR target detection operation on input radar data.
 *
 * This function performs Constant False Alarm Rate (CFAR) target detection on the input
 * radar range-Doppler map data and produces detection list.
 * The operation uses the CFAR algorithm and parameters configured during operator creation
 * to maintain a constant false alarm rate while maximizing target detection probability.
 *
 * @param [in] handle Valid operator handle created by pvaRadarCFARCreate().
 *                    Must not be NULL and must reference a properly initialized operator.
 *
 * @param [in] stream Valid cuPVA stream handle for asynchronous execution.
 *
 * @param [in] inputTensor Input tensor handle containing input data.
 *
 * @param [out] outDetectionListTensor Output tensor handle for storing detection results.
 *
 * @param [out] outDetectionCountTensor to store the number of detections found.
 *                  - Data type: NVCV_DATA_TYPE_S32
 *                  - Data layout: [W]
 *                  - Length: W = 1;
 *                  - Must not be NULL
 *                  - Contains the count of valid detections in the detection list
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Invalid handle pointer or mismatched tensor dimensions.
 * @retval NVCV_ERROR_INTERNAL             Internal processing error, invalid data types, or execution failure.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Insufficient memory for processing.
 * @retval NVCV_SUCCESS                    CFAR detection completed successfully.
 *
 */

#ifdef __cplusplus
}

NVCVStatus pvaRadarCFARSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const inputTensor,
                              NVCVTensorHandle outDetectionListTensor, NVCVTensorHandle outDetectionCountTensor);

/**
 * Submits the RadarCFAR operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Valid operator handle created by pvaRadarCFARCreate().
 * @param [in] stream Valid CUDA stream handle for asynchronous execution.
 * @param [in] inputTensor Input tensor handle containing input data.
 * @param [out] outDetectionListTensor Output tensor handle for storing detection results.
 * @param [out] outDetectionCountTensor Output tensor to store the number of detections found.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Invalid handle pointer or mismatched tensor dimensions.
 * @retval NVCV_ERROR_INTERNAL             Internal processing error, invalid data types, or execution failure.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Insufficient memory for processing.
 * @retval NVCV_SUCCESS                    CFAR detection completed successfully.
 */
NVCVStatus pvaRadarCFARSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const inputTensor,
                              NVCVTensorHandle outDetectionListTensor, NVCVTensorHandle outDetectionCountTensor);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPRADARCFAR_H */
