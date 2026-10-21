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

#ifndef RADAR_PIPELINE_REFERENCE_HPP
#define RADAR_PIPELINE_REFERENCE_HPP

#include "radar_pipeline_pva.hpp"
#include "radar_pipeline_tensors.hpp"

#include <cstdint>

// ============================================================================
// Reference Implementation Functions
// ============================================================================

// Individual processing stage functions
int range_fft_processing(RadarPipelineTensors &tensors);
int doppler_fft_processing(RadarPipelineTensors &tensors);
int nci_processing(RadarPipelineTensors &tensors);
int peak_detection(RadarPipelineTensors &tensors, int32_t NofTx, int32_t NofRx, int32_t repeatFold, int32_t &peakCount);
int doa_processing(RadarPipelineTensors &tensors, int32_t peakCount);

// Function to execute all reference workloads
int radar_execute_ref_workloads(RadarPipelineTensors &tensors, int32_t NofTx, int32_t NofRx, int32_t repeatFold);

#endif // RADAR_PIPELINE_REFERENCE_HPP
