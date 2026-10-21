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
# Host-side Capon ROI beamspace geometry and steering-LUT builder: used by applications that drive the
# Capon ROI preprocessing operator (e.g. the 8T8R radar pipeline) and by the operator's CPU reference.
add_library(pva_capon_roi_beamspace STATIC
    CaponROIBeamspace.cpp
)

target_include_directories(pva_capon_roi_beamspace
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(pva_capon_roi_beamspace PUBLIC cxx_std_17)

set_target_properties(pva_capon_roi_beamspace PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
