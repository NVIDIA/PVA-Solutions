/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file radar_cfar_test_ref.hpp
 *
 * @brief Reference implementation and verification functions for Radar CFAR operations.
 *
 * This header file defines functions for reference model verification, test data generation,
 * and validation of the Radar CFAR (Constant False Alarm Rate) target detection algorithm.
 * It provides both CPU-based reference implementations and utility functions for testing
 * and validation purposes.
 *
 * The reference functions are used to:
 * - Generate test input data for CFAR algorithms
 * - Execute reference CFAR implementations for comparison
 * - Validate PVA operator results against golden reference data
 * - Calculate CFAR threshold parameters (Alpha constant)
 *
 * @defgroup PVA_SAMPLES_RADAR_CFAR_REF Radar CFAR Reference Functions
 * @{
 */

#ifndef RADAR_CFAR_TEST_REF_H
#define RADAR_CFAR_TEST_REF_H

#include "radar_cfar_param.h"

#include <OpRadarCFAR.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute reference CFAR algorithm implementation for verification.
 *
 * This function runs a CPU-based reference implementation of the CFAR algorithm
 * using the same parameters as the PVA operator. It serves as a golden reference
 * for validating the PVA operator's correctness.
 *
 * @param [in] inTensorHandle Input tensor handle containing radar magnitude data.
 *                            Expected format: [Height, Width, Channels] with single channel.
 *                            Data type: 32-bit signed integer.
 * @param [out] outTensorHandle Output tensor handle for storing CFAR detection results.
 *                              Same dimensions as input tensor.
 *                              Data type: 32-bit signed integer.
 * @param [in] cfg CFAR configuration parameters including:
 *                 - numHorTrain/numVerTrain: Number of training cells
 *                 - numHorGuard/numVerGuard: Number of guard cells
 *                 - alpha: CFAR threshold multiplier
 *                 - algorithm: CFAR algorithm type (PVA_CFAR_CA, PVA_CFAR_CA_OS, PVA_CFAR_CA_GO, PVA_CFAR_CA_SO)
 * @param [in] detectionCountTensorHandle Detection count tensor handle.
 *                              Data type: 32-bit signed integer.
 * @param [in] algorithm CFAR algorithm type (PVA_CFAR_CA, PVA_CFAR_CA_OS, PVA_CFAR_CA_GO, PVA_CFAR_CA_SO).
 *
 * @retval 0 Success - Reference CFAR execution completed successfully
 * @retval -1 Error - Invalid parameters or execution failure
 *
 * @note This function implements the same CFAR logic as the PVA operator but runs
 *       on CPU for verification purposes. Performance is not optimized.
 */
int32_t radarCFARRefSubmit(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, PvaRadarCFARParams cfg,
                           NVCVTensorHandle detectionCountTensorHandle, PVARadarCFARType algorithm);

/**
 * @brief Validate PVA operator output against golden reference data.
 *
 * This function compares the output from the PVA CFAR operator against pre-computed
 * golden reference data stored in a binary file. It performs element-wise comparison
 * and reports any differences found.
 *
 * @param [in] pvaTensorHandle Tensor handle containing PVA operator output to validate.
 *                             Expected format: [Height, Width, Channels] with single channel.
 *                             Data type: 32-bit signed integer.
 * @param [in] input_file Path to binary file containing golden reference data.
 *                        File format: Raw binary data with same dimensions as input tensor.
 *
 * @retval 0 Success - Output matches golden reference data
 * @retval -1 Error - Mismatch found or file access error
 *
 * @note The golden reference file should contain the expected CFAR output for the
 *       corresponding input data and parameters used in the test.
 */
int32_t checkagainstRadarCFARGolden(NVCVTensorHandle pvaTensorHandle, const char *input_file);

/**
 * @brief Generate test input data for CFAR algorithm testing.
 *
 * This function generates synthetic radar magnitude data for testing CFAR algorithms.
 * It can either load data from a file or generate random test patterns based on
 * the tensor dimensions.
 *
 * @param [out] tensorHandle Tensor handle to populate with input data.
 *                           Expected format: [Height, Width, Channels] with single channel.
 *                           Data type: 32-bit signed integer.
 * @param [in] useLargeValues True to generate large valued input for stress testing,
 *                            false to generate normal test data with smaller values.
 *
 * @retval 0 Success - Input data generated/loaded successfully
 * @retval -1 Error - File access error or invalid tensor handle
 *
 * @note The function generates random magnitude data
 *       suitable for CFAR testing with configurable noise and target characteristics.
 */
int32_t generateRadarCFARInput(NVCVTensorHandle tensorHandle, bool useLargeValues);

/**
 * @brief Compare reference implementation output with PVA operator output.
 *
 * This function performs a direct comparison between the reference CFAR implementation
 * output and the PVA operator output to validate correctness. It provides detailed
 * difference reporting for debugging purposes.
 *
 * @param [in] refTensorHandle Tensor handle containing reference implementation output.
 *                             Expected format: [Height, Width, Channels] with single channel.
 *                             Data type: 32-bit signed integer.
 * @param [in] outTensorHandle Tensor handle containing PVA operator output to validate.
 *                             Expected format: [Height, Width, Channels] with single channel.
 *                             Data type: 32-bit signed integer.
 * @param [in] detectionCountRefTensorHandle Detection count ref tensor handle.
 *                              Data type: 32-bit signed integer.
 * @param [in] detectionCountOutTensorHandle Detection count out tensor handle.
 *                              Data type: 32-bit signed integer.
 *
 * @retval 0 Success - Outputs match within acceptable tolerance
 * @retval -1 Error - Significant differences found between outputs
 *
 * @note This function performs element-wise comparison and may allow for small
 *       numerical differences due to implementation variations while flagging
 *       significant discrepancies.
 */
int32_t checkagainstRadarCFARRef(NVCVTensorHandle refTensorHandle, NVCVTensorHandle outTensorHandle,
                                 NVCVTensorHandle detectionCountRefTensorHandle,
                                 NVCVTensorHandle detectionCountOutTensorHandle);

#ifdef __cplusplus
}
#endif

/** @} */ // end of PVA_SAMPLES_RADAR_CFAR_REF group

#endif /* RADAR_CFAR_TEST_REF_H */