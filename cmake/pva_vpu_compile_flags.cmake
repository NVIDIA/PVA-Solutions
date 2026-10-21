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

# Common VPU device compile flags.

# Device / VPU C++17 flag.
if(PVA_USE_NVVPU OR CUPVA_BUILD_MODE STREQUAL "NATIVE")
    set(PVA_VPU_CXX17_FLAG "-std=c++17")
else()
    set(PVA_VPU_CXX17_FLAG "+Wllvm,-std=c++17")
endif()

# Suppress Chess mist2 software-pipelining warnings. Empty on NVVPU/NATIVE
# (no equivalent flag, and the host compiler would reject '+Wmist2,...').
# Use without quotes so an empty value expands to no argument:
#     CFLAGS_DEVICE ${PVA_VPU_NO_PIPELINING_WARN_FLAG}
if(PVA_USE_NVVPU OR CUPVA_BUILD_MODE STREQUAL "NATIVE")
    set(PVA_VPU_NO_PIPELINING_WARN_FLAG)
else()
    set(PVA_VPU_NO_PIPELINING_WARN_FLAG "+Wmist2,-Wno_warn_pipelining")
endif()
