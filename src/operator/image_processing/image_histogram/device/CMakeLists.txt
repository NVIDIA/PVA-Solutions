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

pva_solutions_device(pva_image_histogram_u8_dev
    image_histogram_top.c
    image_histogram_knl_u8.c
    CFLAGS -DDATA_TYPE_U8=1)

# The host picks which of these to load from numBins at create.
foreach(WAYS 16 8 4 2)
    pva_solutions_device(pva_image_histogram_u16_${WAYS}w_dev
        image_histogram_top.c
        image_histogram_knl_u16.c
        CFLAGS -DDATA_TYPE_U16=1 -DHIST_WAYS=${WAYS})
endforeach()
