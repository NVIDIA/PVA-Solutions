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
 * @file radar_cfar_top.hpp
 *
 * @brief Top-level interface for Radar CFAR (Constant False Alarm Rate) processing.
 *
 * This header file defines the main entry point function for executing CFAR target
 * detection algorithms on radar magnitude data. It provides a simplified interface
 * that handles memory management and orchestrates the CFAR processing pipeline.
 *
 * The CFAR algorithm is used in radar signal processing to maintain a constant
 * false alarm rate while maximizing target detection probability in varying noise
 * conditions. This implementation supports multiple CFAR variants including Cell
 * Averaging (CA), Order Statistic (OS), Greatest Of (GO), and Smallest Of (SO).
 *
 * @defgroup PVA_SAMPLES_RADAR_CFAR_TOP Radar CFAR Top-Level Interface
 * @{
 */

#ifndef RADAR_CFAR_TOP_H
#    define RADAR_CFAR_TOP_H

#    include "radar_cfar_param.h"

#    include <stdint.h>

/**
 * @brief Execute CFAR target detection algorithm on radar magnitude data.
 *
 * This function serves as the main entry point for CFAR processing, providing a
 * simplified interface that handles internal memory management and orchestrates
 * the complete CFAR detection pipeline. It allocates necessary intermediate buffers,
 * executes the appropriate CFAR algorithm based on configuration, and returns
 * detection results.
 *
 * The function supports multiple CFAR algorithms:
 * - **CA (Cell Averaging)**: Uses arithmetic mean of training cells
 * - **CA_OS (Order Statistic)**: Uses median of training cells for robust estimation
 * - **CA_GO (Greatest Of)**: Uses maximum of leading/trailing training cells
 * - **CA_SO (Smallest Of)**: Uses minimum of leading/trailing training cells
 *
 * @param [in] mag Pointer to input radar magnitude data array.
 *                 Expected format: 2D array with dimensions [height][width].
 *                 Data type: 32-bit signed integer representing magnitude values.
 *                 Memory layout: Row-major order (y * width + x).
 * @param [out] peaks Pointer to output detection results array.
 *                    Same dimensions as input: [height][width].
 *                    Data type: 32-bit signed integer.
 *                    Output values: 1 for detected targets, 0 for no detection.
 * @param [in] cfg CFAR configuration parameters structure containing:
 *                 - numHorTrain/numVerTrain: Number of training cells (1-24)
 *                 - numHorGuard/numVerGuard: Number of guard cells (0-12)
 *                 - alpha: CFAR threshold multiplier (typically 1.0-100.0)
 *                 - algorithm: CFAR algorithm type (CA, CA_OS, CA_GO, CA_SO)
 * @param [in] width Width of the input/output arrays in pixels.
 *                   Must be positive integer.
 * @param [in] height Height of the input/output arrays in pixels.
 *                    Must be positive integer.
 * @param [in] algorithm CFAR algorithm type.
 *
 * @retval 0 Success - CFAR processing completed successfully
 * @retval -1 Error - Invalid parameters, memory allocation failure, or processing error
 *
 * @see PvaRadarCFARParams for detailed parameter descriptions and constraints.
 */
int32_t radar_cfar_func(int32_t *mag, int32_t *peaks, PvaRadarCFARParams cfg, int32_t width, int32_t height,
                        PVARadarCFARType algorithm);

#endif // RADAR_CFAR_TOP_H

/** @} */ // end of PVA_SAMPLES_RADAR_CFAR_TOP group