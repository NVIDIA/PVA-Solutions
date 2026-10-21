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

set(IO_TYPES U8 S8 U16 S16 U32 S32)
set(SEPA_2X2_KNL_TYPES 5x5 5x7 7x7)

# 3x3: use the classic no-reuse implementation
# 3x3, single channel
foreach(io_type ${IO_TYPES})
    string(TOLOWER ${io_type} io_type_lower)
    pva_solutions_device(pva_median_filter_${io_type_lower}_3x3_dev
                         median_filter_top.c
                         median_filter_knl_3x3.c
                         CFLAGS
                             -D${io_type}_IO
                             -DKNL_3x3)
endforeach()

# 3x3, 2S16
pva_solutions_device(pva_median_filter_i2s16_3x3_dev
                     median_filter_top.c
                     median_filter_knl_3x3.c
                     CFLAGS
                         -DI2S16_IO
                         -DKNL_3x3
                         -DINTERLEAVED2C)

# kernel sizes > 3x3: use the separable 2x2 pipeline
# kernel sizes > 3x3, single channel
foreach(io_type ${IO_TYPES})
    string(TOLOWER ${io_type} io_type_lower)
    foreach(knl_type ${SEPA_2X2_KNL_TYPES})
        set(sepa_2x2_sources
            median_filter_sepa_2x2_top.c
            median_filter_common_sepa_2x2.c
            median_filter_knl_sepa_2x2_${knl_type}.c)

        pva_solutions_device(pva_median_filter_sepa_2x2_${io_type_lower}_${knl_type}_dev
                             ${sepa_2x2_sources}
                             CFLAGS
                                 -DSEPA_2x2
                                 -D${io_type}_IO
                                 -DKNL_${knl_type})
    endforeach()
endforeach()

# kernel sizes > 3x3, 2S16
foreach(knl_type ${SEPA_2X2_KNL_TYPES})
    set(sepa_2x2_i2s16_sources
        median_filter_sepa_2x2_i2s16_top.c
        median_filter_sepa_2x2_i2s16.c
        median_filter_common_sepa_2x2.c
        median_filter_knl_sepa_2x2_${knl_type}.c)

    pva_solutions_device(pva_median_filter_sepa_2x2_i2s16_${knl_type}_dev
                         ${sepa_2x2_i2s16_sources}
                         CFLAGS
                             -DSEPA_2x2
                             -DI2S16_IO
                             -DKNL_${knl_type}
                             -DINTERLEAVED2C)
endforeach()