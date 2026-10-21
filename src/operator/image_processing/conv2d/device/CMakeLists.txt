#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(KERNEL_SIZES 3 5 7)
set(IO_TYPES U8 S8 U16 S16)

set(VPUBIN_NAMES)

# Get the include directories from the pvaAplConv2dVpu target
get_target_property(PVA_APL_INCLUDE_DIR pvaAplVpu INTERFACE_INCLUDE_DIRECTORIES)

## Loop over all supported configs
foreach(io_type ${IO_TYPES})
    foreach(k_size ${KERNEL_SIZES})
        pva_solutions_device(pva_conv_K${k_size}_${io_type}_dev
                conv2d_top.cpp
                CFLAGS
                    -D${io_type}_IO
                    -DKSIZE_${k_size}X${k_size}
                    "-I${PVA_APL_INCLUDE_DIR}"
                CFLAGS_DEVICE
                    -D${io_type}_IO
                    -DKSIZE_${k_size}X${k_size}
                    "-I${PVA_APL_INCLUDE_DIR}"
                    "${PVA_VPU_CXX17_FLAG}")
        list(APPEND VPUBIN_NAMES pva_conv_K${k_size}_${io_type}_dev)
    endforeach()
endforeach()

set(VPUBIN_NAMES ${VPUBIN_NAMES} PARENT_SCOPE)