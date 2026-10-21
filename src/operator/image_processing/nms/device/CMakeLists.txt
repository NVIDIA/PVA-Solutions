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

set(NMS_SIZES 3 5)
set(VPUBIN_NAMES)

# The primitive is a header-only device template; pull its include directory off the INTERFACE target.
get_target_property(PVA_APL_INCLUDE_DIR pvaAplVpu INTERFACE_INCLUDE_DIRECTORIES)

foreach(nms_size ${NMS_SIZES})
    pva_solutions_device(pva_nms_K${nms_size}_S32_dev
            nms_top.cpp
            CFLAGS
                -DNMS_SIZE_${nms_size}
                "-I${PVA_APL_INCLUDE_DIR}"
            CFLAGS_DEVICE
                -DNMS_SIZE_${nms_size}
                "-I${PVA_APL_INCLUDE_DIR}"
                "${PVA_VPU_CXX17_FLAG}")
    list(APPEND VPUBIN_NAMES pva_nms_K${nms_size}_S32_dev)
endforeach()

set(VPUBIN_NAMES ${VPUBIN_NAMES} PARENT_SCOPE)
