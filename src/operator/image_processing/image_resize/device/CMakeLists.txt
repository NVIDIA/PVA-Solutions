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

set(IO_TYPES U8 RGB8 RGBA8 NV12 U16)

## Loop over all supported configs
# Nearest neighbor
foreach(io_type ${IO_TYPES})
    pva_solutions_device(pva_image_resize_${io_type}_nn_dev
                 image_resize_top.c
                 image_resize_opt_nn.c
                 image_resize_common.c
                 CFLAGS
                     -DNN_OP
                     -D${io_type}_IO)
endforeach()

# Bilinear interpolation
foreach(io_type ${IO_TYPES})
    pva_solutions_device(pva_image_resize_${io_type}_bl_dev
                     image_resize_top.c
                     image_resize_opt_bl.c
                     image_resize_common.c
                     CFLAGS
                         -DBL_OP
                         -D${io_type}_IO)
endforeach()

