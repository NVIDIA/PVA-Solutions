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

set(IO_TYPES U8 S8 U16 S16)
set(OP_TYPES ERODE DILATE)
set(KERNEL_SIZES 3 5)
set(KERNEL_SHAPES RECT CROSS)

set(VPUBIN_NAMES)

## Loop over all supported configs
foreach(op_type ${OP_TYPES})
    foreach(io_type ${IO_TYPES})
        foreach(k_size ${KERNEL_SIZES})
            foreach(k_shape ${KERNEL_SHAPES})
                ## Skip 5x5_CROSS config
                if(NOT (${k_size} STREQUAL "5" AND ${k_shape} STREQUAL "CROSS"))
                    pva_solutions_device(filter_${op_type}_${io_type}_K${k_size}_${k_shape}_dev
                            morphological_top.c
                            morphological_knl.c
                            CFLAGS
                                -D${op_type}_OP
                                -D${io_type}_IO
                                -DKSIZE_${k_size}X${k_size}
                                -DKSHAPE_${k_shape}
                            CFLAGS_DEVICE
                                -D${op_type}_OP
                                -D${io_type}_IO
                                -DKSIZE_${k_size}X${k_size}
                                -DKSHAPE_${k_shape})
                    ## create list of VPU binaries
                    list(APPEND VPUBIN_NAMES filter_${op_type}_${io_type}_K${k_size}_${k_shape}_dev)
                endif()
            endforeach()
        endforeach()
    endforeach()
endforeach()

set(VPUBIN_NAMES ${VPUBIN_NAMES} PARENT_SCOPE)