#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_subdirectory(nci)
add_subdirectory(coherent_integration)
add_subdirectory(range_fft)
add_subdirectory(doppler_fft)
add_subdirectory(radar_cfar)
add_subdirectory(snapshot_extraction)
add_subdirectory(doa_bartlett_beamforming)
add_subdirectory(capon_roi_preprocess)
add_subdirectory(custom)
add_subdirectory(radar_transpose)

# doa_capon_roi device kernels are C++ and require the LLVM front end
if(CUPVA_LLVM_SUPPORTED)
    add_subdirectory(doa_capon_roi)
endif()
