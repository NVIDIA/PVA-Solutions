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

# Set GSDF flag if CUPVA version is less than 2.7.0
if(${CUPVA_VERSION} VERSION_LESS "2.7.0")
    set(GSDF_FLAG -DUSE_GSDF)
else()
    set(GSDF_FLAG)
endif()

set(INPUT_FORMATS U8 NV12)

pva_solutions_device(remap_u8_dev
            remap_top_u8.c
            remap_knl.c
            CFLAGS
                -DINPUT_FORMAT_U8
                ${GSDF_FLAG}
            CFLAGS_DEVICE
                -DINPUT_FORMAT_U8
                ${GSDF_FLAG})

pva_solutions_device(remap_nv12_dev
            remap_top_nv12.c
            remap_knl.c
            CFLAGS
                -DINPUT_FORMAT_NV12
                ${GSDF_FLAG}
            CFLAGS_DEVICE
                -DINPUT_FORMAT_NV12
                ${GSDF_FLAG})