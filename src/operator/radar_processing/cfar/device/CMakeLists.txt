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

find_package(cupva REQUIRED)

# One executable per CFAR algorithm. A runtime flag would put a branch in the marching
# loop, making every variant pay the scheduler slots and register pressure of the others.
set(CFAR_ALGOS CA GO SO)

foreach(algo ${CFAR_ALGOS})
    string(TOLOWER ${algo} algo_lower)
    pva_solutions_device(pva_radar_cfar_${algo_lower}_dev
        radar_cfar_knl.c
        radar_cfar_top.c
        CFLAGS
            -DCFAR_ALGO_${algo}
            "-I$<TARGET_PROPERTY:pva_operator_includes,INTERFACE_INCLUDE_DIRECTORIES>"
    )
endforeach()
