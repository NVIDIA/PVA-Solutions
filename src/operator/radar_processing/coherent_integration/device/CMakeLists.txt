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

find_package(cupva REQUIRED)

set(COHERENT_INTEGRATION_DEVICE_SOURCES
    coherent_integration_knl.c
    coherent_integration_full_grid.c
    coherent_integration_top.c)

set(COHERENT_INTEGRATION_DEVICE_CFLAGS
    "-I$<TARGET_PROPERTY:pva_operator_includes,INTERFACE_INCLUDE_DIRECTORIES>")

# Keep Rx compile-time constant in the preprocessing and exhaustive-search hot paths.
foreach(COHERENT_INTEGRATION_RX IN ITEMS 4 8 12 16)
    set(COHERENT_INTEGRATION_RX_DEVICE_CFLAGS
        ${COHERENT_INTEGRATION_DEVICE_CFLAGS}
        -DPVASOL_COHERENT_INTEGRATION_RX=${COHERENT_INTEGRATION_RX})

    pva_solutions_device(pva_coherent_integration_rx${COHERENT_INTEGRATION_RX}_dev
        ${COHERENT_INTEGRATION_DEVICE_SOURCES}
        CFLAGS
            ${COHERENT_INTEGRATION_RX_DEVICE_CFLAGS})
endforeach()
