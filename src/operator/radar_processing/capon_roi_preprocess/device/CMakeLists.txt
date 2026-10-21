#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Three VPU programs from the same sources; the host picks one per operator (OpCaponROIPreprocess.cpp):
# dft and dftgen run the direct-DFT coarse DoA, fft the batched FFT-64 one (CRP_COARSE_DOA_FFT selects
# the coarse-DoA path compiled into top.c). CRP_PROJ_GENERAL = 0 builds dft, which only runs the
# dot-product projection on azimuth-only arrays; CRP_PROJ_GENERAL = 1 builds dftgen, which serves every
# other direct-DFT geometry: it picks the dot-product projection or the per-tap MAC paths at run time
# and also links the elevation projection (elproj.c) and the per-tap paths (pertap.c).
set(CRP_DEVICE_INCLUDES
    "-I$<TARGET_PROPERTY:pva_operator_includes,INTERFACE_INCLUDE_DIRECTORIES>"
    -I${CMAKE_CURRENT_SOURCE_DIR}
    -I${CMAKE_COMMON_RADAR_DEFINITIONS_DIR}
    -I${CMAKE_COMMON_FFT_DIR})

# Direct-DFT coarse DoA.
pva_solutions_device(pva_capon_roi_preprocess_dft_dev
    capon_roi_preprocess_init.c
    capon_roi_preprocess_top.c
    capon_roi_preprocess_dft.c
    capon_roi_preprocess_knl.c
    capon_roi_preprocess_rank.c
    capon_roi_preprocess_dft2d.c
    CFLAGS_DEVICE ${CRP_DEVICE_INCLUDES} -DCRP_COARSE_DOA_FFT=0 -DCRP_PROJ_GENERAL=0
    CFLAGS_NATIVE ${CRP_DEVICE_INCLUDES} -DCRP_COARSE_DOA_FFT=0 -DCRP_PROJ_GENERAL=0)

pva_solutions_device(pva_capon_roi_preprocess_dftgen_dev
    capon_roi_preprocess_init.c
    capon_roi_preprocess_top.c
    capon_roi_preprocess_dft.c
    capon_roi_preprocess_knl.c
    capon_roi_preprocess_elproj.c
    capon_roi_preprocess_dft2d.c
    capon_roi_preprocess_rank.c
    capon_roi_preprocess_pertap.c
    CFLAGS_DEVICE ${CRP_DEVICE_INCLUDES} -DCRP_COARSE_DOA_FFT=0 -DCRP_PROJ_GENERAL=1
    CFLAGS_NATIVE ${CRP_DEVICE_INCLUDES} -DCRP_COARSE_DOA_FFT=0 -DCRP_PROJ_GENERAL=1)

# Batched FFT-64 coarse DoA.
pva_solutions_device(pva_capon_roi_preprocess_fft_dev
    capon_roi_preprocess_init.c
    ${CMAKE_COMMON_FFT_DIR}/fft_knl.c
    ${CMAKE_COMMON_FFT_DIR}/fft_radix4.c
    capon_roi_preprocess_top.c
    capon_roi_preprocess_knl.c
    capon_roi_preprocess_fft.c
    capon_roi_preprocess_rank.c
    capon_roi_preprocess_dft2d.c
    CFLAGS_DEVICE ${CRP_DEVICE_INCLUDES} -DCRP_COARSE_DOA_FFT=1
    CFLAGS_NATIVE ${CRP_DEVICE_INCLUDES} -DCRP_COARSE_DOA_FFT=1)
