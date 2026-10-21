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

# Full kernel: RANGE_DOPPLER_RX layout (L2-buffered RxPacked path)
pva_solutions_device(pva_doppler_fft_dev
    doppler_fft_knl.c
    doppler_fft_top.c
    ${CMAKE_COMMON_FFT_DIR}/fft_knl.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix2.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix3.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix4.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix5.c
    CFLAGS_DEVICE
        -I${CMAKE_COMMON_RADAR_DEFINITIONS_DIR}
        -I${CMAKE_COMMON_FFT_DIR}
    CFLAGS_NATIVE
        -I${CMAKE_COMMON_RADAR_DEFINITIONS_DIR}
        -I${CMAKE_COMMON_FFT_DIR})

# Direct kernel: DOPPLER_RX_RANGE, RANGE_RX_DOPPLER and RX_RANGE_DOPPLER outputs (no L2 buffering)
pva_solutions_device(pva_doppler_fft_dev_direct
    doppler_fft_knl.c
    doppler_fft_top_direct.c
    ${CMAKE_COMMON_FFT_DIR}/fft_knl.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix2.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix3.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix4.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix5.c
    CFLAGS_DEVICE
        -I${CMAKE_COMMON_RADAR_DEFINITIONS_DIR}
        -I${CMAKE_COMMON_FFT_DIR}
    CFLAGS_NATIVE
        -I${CMAKE_COMMON_RADAR_DEFINITIONS_DIR}
        -I${CMAKE_COMMON_FFT_DIR})