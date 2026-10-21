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

set(IO_TYPES U16 U8 S8 S16)
set(NMS_OPTIONS ON OFF)
set(TOPK_OPTIONS ON OFF)

set(VPUBIN_NAMES)

foreach(io_type ${IO_TYPES})
    foreach(nms_option ${NMS_OPTIONS})
        foreach(topk_option ${TOPK_OPTIONS})
            set(fastcorner_vpu_bin pva_fastcorner_detector_${io_type}_NMS_${nms_option}_TOPK_${topk_option}_dev)
            pva_solutions_device(${fastcorner_vpu_bin}
                fast_corner_detector_top.cpp
                fast_corner_detector_knls.cpp
                CFLAGS
                    -D${io_type}_IO
                    -DNMS_${nms_option}
                    -DTOPK_${topk_option}
                CFLAGS_VPU
                    -D${io_type}_IO
                    -DNMS_${nms_option}
                    -DTOPK_${topk_option}
                    "${PVA_VPU_CXX17_FLAG}")
            list(APPEND VPUBIN_NAMES ${fastcorner_vpu_bin})
            unset(fastcorner_vpu_bin)
        endforeach()
    endforeach()
endforeach()

set(VPUBIN_NAMES ${VPUBIN_NAMES} PARENT_SCOPE)