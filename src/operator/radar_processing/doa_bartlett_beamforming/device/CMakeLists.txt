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

# cuPVA < 2.7.0: 8-arg cupvaSQDFUpdateAddr overload not available on VPU compiler
if(${CUPVA_VERSION} VERSION_LESS "2.7.0")
    set(SQDF_COMPAT_FLAG -DSQDF_NO_PITCH_OVERLOAD)
else()
    set(SQDF_COMPAT_FLAG)
endif()

# Build separable mode ELF
pva_solutions_device(pva_bartlett_separable_dev
    bartlett_top_separable.c
    bartlett_common.c
    bartlett_separable_knl.c
    CFLAGS
        "-I$<TARGET_PROPERTY:pva_operator_includes,INTERFACE_INCLUDE_DIRECTORIES>"
        ${SQDF_COMPAT_FLAG})

# Build full 2D mode ELF
pva_solutions_device(pva_bartlett_full2d_dev
    bartlett_top_full2d.c
    bartlett_common.c
    bartlett_full2d_knl.c
    CFLAGS
        "-I$<TARGET_PROPERTY:pva_operator_includes,INTERFACE_INCLUDE_DIRECTORIES>"
        ${SQDF_COMPAT_FLAG})
