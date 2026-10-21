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

# u8
pva_solutions_device(pva_minmaxloc_u8_dev
    minmaxloc_top.c
    minmaxloc_opt_interface.c
    minmaxloc_opt_u8.c
    minmaxloc_opt_s8.c
    minmaxloc_common.c
    CFLAGS
        -DDATA_TYPE_U8)

# s8
pva_solutions_device(pva_minmaxloc_s8_dev
    minmaxloc_top.c
    minmaxloc_opt_interface.c
    minmaxloc_opt_s8.c
    minmaxloc_common.c
    CFLAGS
        -DDATA_TYPE_S8)


# u16
pva_solutions_device(pva_minmaxloc_u16_dev
    minmaxloc_top.c
    minmaxloc_opt_interface.c
    minmaxloc_opt_u16.c
    minmaxloc_opt_s16.c
    minmaxloc_common.c
    CFLAGS
        -DDATA_TYPE_U16)

# s16
pva_solutions_device(pva_minmaxloc_s16_dev
    minmaxloc_top.c
    minmaxloc_opt_interface.c
    minmaxloc_opt_s16.c
    minmaxloc_common.c
    CFLAGS
        -DDATA_TYPE_S16)

# u32
pva_solutions_device(pva_minmaxloc_u32_dev
    minmaxloc_top.c
    minmaxloc_opt_interface.c
    minmaxloc_opt_u32.c
    minmaxloc_opt_s32.c
    minmaxloc_common.c
    CFLAGS
        -DDATA_TYPE_U32)

# s32
pva_solutions_device(pva_minmaxloc_s32_dev
    minmaxloc_top.c
    minmaxloc_opt_interface.c
    minmaxloc_opt_s32.c
    minmaxloc_common.c
    CFLAGS
        -DDATA_TYPE_S32)

# f32
pva_solutions_device(pva_minmaxloc_f32_dev
    minmaxloc_top.c
    minmaxloc_opt_interface.c
    minmaxloc_opt_f32.c
    minmaxloc_opt_s32.c
    minmaxloc_common.c
    CFLAGS
        -DDATA_TYPE_F32)
