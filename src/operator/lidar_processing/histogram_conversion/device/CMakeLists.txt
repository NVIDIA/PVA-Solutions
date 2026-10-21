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

if(PVASOL_LIDAR_PROFILING)
    # enable profiling hooks, as well as extra chess outputs useful for analysis
    pva_solutions_device(pva_lidar_histogram_conversion_dev
        lidar_histogram_conversion_top.c
        CFLAGS_DEVICE
            ${PVA_VPU_NO_PIPELINING_WARN_FLAG}
            +Wmist2,-gall
            +k
            -d
            -DPVASOL_LIDAR_PROFILING
            -DCUPVA_PROFILING_FLAG=1)
else()
    pva_solutions_device(pva_lidar_histogram_conversion_dev
        lidar_histogram_conversion_top.c
        CFLAGS_DEVICE
            ${PVA_VPU_NO_PIPELINING_WARN_FLAG})
endif()

