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
add_library(capon_roi_preprocess_ref STATIC
    capon_roi_preprocess_ref.cpp
)

target_include_directories(capon_roi_preprocess_ref
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# The beamspace geometry and steering-LUT builder the reference shares with applications.
target_link_libraries(capon_roi_preprocess_ref PUBLIC pva_capon_roi_beamspace)

set_target_properties(capon_roi_preprocess_ref PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
