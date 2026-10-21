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

set(INPUT_FORMATS U8 NV12)

pva_solutions_device(warp_perspective_u8_dev
                            warpPerspective_top_u8.c
                            warpPerspective_knl.c
                            CFLAGS
                                -DINPUT_FORMAT_U8
                            CFLAGS_DEVICE
                                -DINPUT_FORMAT_U8)

pva_solutions_device(warp_perspective_nv12_dev
                            warpPerspective_top_nv12.c
                            warpPerspective_knl.c
                            CFLAGS
                                -DINPUT_FORMAT_NV12
                            CFLAGS_DEVICE
                                -DINPUT_FORMAT_NV12)
