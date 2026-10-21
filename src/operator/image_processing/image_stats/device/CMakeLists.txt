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

# set(IO_TYPES U8 RGB RGBA NV12)
set(IO_TYPES U8 RGB NV12)
set(MULTI_CHANNEL_IO_TYPES RGB NV12)


# device executable for all formats
foreach(io_type ${IO_TYPES})
    string(TOLOWER ${io_type} io_type_lower)
    # without mask
    pva_solutions_device(pva_image_stats_${io_type_lower}_cnt_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                     -D${io_type}_IO
                     -DCNT_IO)

    pva_solutions_device(pva_image_stats_${io_type_lower}_sum_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                     -D${io_type}_IO
                     -DSUM_IO)

    pva_solutions_device(pva_image_stats_${io_type_lower}_var_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                         -D${io_type}_IO
                         -DSUM_IO
                         -DVAR_IO)

    # with mask
    pva_solutions_device(pva_image_stats_${io_type_lower}_mask_cnt_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                     -D${io_type}_IO
                     -DCNT_IO
                     -DMASK_IO)

    pva_solutions_device(pva_image_stats_${io_type_lower}_mask_sum_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                     -D${io_type}_IO
                     -DSUM_IO
                     -DMASK_IO)

    pva_solutions_device(pva_image_stats_${io_type_lower}_mask_cnt_sum_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                     -D${io_type}_IO
                     -DCNT_IO
                     -DSUM_IO
                     -DMASK_IO)

    pva_solutions_device(pva_image_stats_${io_type_lower}_mask_var_dev
                     image_stats_top.c
                     image_stats_opt_${io_type_lower}.c
                     longlong_div.c
                     CFLAGS
                         -D${io_type}_IO
                         -DCNT_IO
                         -DSUM_IO
                         -DVAR_IO
                         -DMASK_IO)
endforeach()

# for multi-channel formats
foreach(io_type ${MULTI_CHANNEL_IO_TYPES})
    string(TOLOWER ${io_type} io_type_lower)
    # without mask
    pva_solutions_device(pva_image_stats_${io_type_lower}_cov_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                     -D${io_type}_IO
                     -DSUM_IO
                     -DVAR_IO
                     -DCOV_IO)

    # with mask
    pva_solutions_device(pva_image_stats_${io_type_lower}_mask_cov_dev
                 image_stats_top.c
                 image_stats_opt_${io_type_lower}.c
                 longlong_div.c
                 CFLAGS
                     -D${io_type}_IO
                     -DCNT_IO
                     -DSUM_IO
                     -DVAR_IO
                     -DCOV_IO
                     -DMASK_IO)
endforeach()
