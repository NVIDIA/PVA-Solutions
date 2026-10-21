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

set(NCIW_WIDTHS 512 1024)

foreach(width ${NCIW_WIDTHS})
    pva_solutions_device(pva_peak_detection_nciw${width}_dev
        peak_detection_knls.c
        peak_detection_top.c
        CFLAGS
            -DNCIRX_WIDTH_${width}
            "-I$<TARGET_PROPERTY:pva_operator_includes,INTERFACE_INCLUDE_DIRECTORIES>"
    )
endforeach()
