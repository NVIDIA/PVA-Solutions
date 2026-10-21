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

set (TYPE_STR RBRIEF BRIEF)
set (X8_SET U8 S8)
set (X16_SET U16 S16)

foreach(type ${TYPE_STR})
    string(TOLOWER ${type} type_lower)
    # X8 data types
    foreach(io_type ${X8_SET})
        string(TOLOWER ${io_type} io_type_lower)
        pva_solutions_device(pva_orb_desc_extractor_${io_type_lower}_${type_lower}_dev
                             orb_desc_extr_top.c
                             orb_desc_extr_knl_x8.c
                             orb_desc_extr_common.c
                             CFLAGS
                                 -D${io_type}_IO
                                 -DKNL_${type})
    endforeach()

    # X16 data types
    foreach(io_type ${X16_SET})
        string(TOLOWER ${io_type} io_type_lower)
        pva_solutions_device(pva_orb_desc_extractor_${io_type_lower}_${type_lower}_dev
                             orb_desc_extr_top.c
                             orb_desc_extr_knl_x16.c
                             orb_desc_extr_common.c
                             CFLAGS
                                 -D${io_type}_IO
                                 -DKNL_${type})
    endforeach()
endforeach()
