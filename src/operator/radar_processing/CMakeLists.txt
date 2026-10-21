#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(CMAKE_COMMON_FFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common/fft)
set(CMAKE_COMMON_RADAR_DEFINITIONS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../priv)

pva_op_add_subdirectory(radar_processing common/steering_vector)
pva_op_add_subdirectory(radar_processing common/capon_roi_beamspace)

pva_op_add_subdirectory(radar_processing range_fft)
pva_op_add_subdirectory(radar_processing doppler_fft)
pva_op_add_subdirectory(radar_processing coherent_integration)
pva_op_add_subdirectory(radar_processing Nci)
pva_op_add_subdirectory(radar_processing cfar)
pva_op_add_subdirectory(radar_processing snapshot_extraction)
pva_op_add_subdirectory(radar_processing doa_bartlett_beamforming)
pva_op_add_subdirectory(radar_processing doa_capon_roi)
pva_op_add_subdirectory(radar_processing capon_roi_preprocess)
pva_op_add_subdirectory(radar_processing custom)
pva_op_add_subdirectory(radar_processing radar_transpose)
