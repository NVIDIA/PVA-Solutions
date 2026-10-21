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

set(_capon_roi_common_cflags
    "-I$<TARGET_PROPERTY:pva_operator_includes,INTERFACE_INCLUDE_DIRECTORIES>"
    "-I${CMAKE_CURRENT_SOURCE_DIR}")

set(_capon_roi_part1_cflags
    ${_capon_roi_common_cflags}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/part1"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/part1/cov_mat"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/part1/block_cholesky")

set(_capon_roi_combined_cflags
    ${_capon_roi_part1_cflags}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/part2")

set(_capon_roi_fp_bcf "${CMAKE_CURRENT_BINARY_DIR}/capon_roi_fp_bs4.bcf")
set(_capon_roi_s32_bcf "${CMAKE_CURRENT_BINARY_DIR}/capon_roi_s32_bs4.bcf")

# The block-Cholesky kernels take VMEM-bank-qualified pointers, and the bank
# qualifier is part of their mangled names. The Chess frontend shipped with
# SDK 2.10 (ASIP Programmer 2025.12) mangles chess_storage(RAM_Xb) as the vendor
# qualifier "chess_storage_RAM_Xb"; earlier SDKs mangle it as an address-space
# number. The layout scripts are templates that take the right spelling here.
if(CUPVA_VERSION VERSION_GREATER_EQUAL "2.10.0")
    set(CAPON_ROI_MANGLED_RAM_Ab "U20chess_storage_RAM_Ab")
    set(CAPON_ROI_MANGLED_RAM_Bb "U20chess_storage_RAM_Bb")
    set(CAPON_ROI_MANGLED_RAM_Cb "U20chess_storage_RAM_Cb")
else()
    set(CAPON_ROI_MANGLED_RAM_Ab "U3AS4")
    set(CAPON_ROI_MANGLED_RAM_Bb "U3AS5")
    set(CAPON_ROI_MANGLED_RAM_Cb "U3AS6")
endif()

if(NOT CUPVA_BUILD_MODE STREQUAL "NATIVE")
    configure_file(capon_roi_fp_bs4.bcf.in "${_capon_roi_fp_bcf}" @ONLY)
    configure_file(capon_roi_s32_bs4.bcf.in "${_capon_roi_s32_bcf}" @ONLY)
endif()

# Part 2 kernel sources are shared by every combined variant.
set(_capon_roi_part2_kernel_srcs
    part2/capon_roi_part2_config.cpp
    part2/compute_L_inv_times_steering_matrix.cpp
    part2/compute_fp_rcp_spectrum.cpp
    part2/estimate_noise_floor.cpp
    part2/extract_3x3filtered_spectrum_in16az8el_out14az6el.cpp
    part2/get_top3_peaks.cpp
    part2/mark_peak_candidates.cpp
    part2/obtain_fixed_pt_L_inv.cpp
    part2/obtain_fixed_pt_steering_matrix.cpp
    part2/obtain_spectrum.cpp)

# Combined command program executes both memory passes in sequence.
foreach(covmat_type s32 fp)
    if(covmat_type STREQUAL "fp")
        set(_cov_knl_src part1/cov_mat/capon_utils_knl.c)
        set(_use_fp_covmat 1)
        set(_capon_roi_bcf "${_capon_roi_fp_bcf}")
    else()
        set(_cov_knl_src part1/cov_mat/capon_utils_knl_s32_nhw.c)
        set(_use_fp_covmat 0)
        set(_capon_roi_bcf "${_capon_roi_s32_bcf}")
    endif()

    foreach(interp_flag interp no_interp)
        if(interp_flag STREQUAL "interp")
            set(_enable_subbin_interp 1)
        else()
            set(_enable_subbin_interp 0)
        endif()

        pva_solutions_device(pva_capon_roi_combined_${covmat_type}_${interp_flag}_dev
            capon_roi_top.c
            part1/capon_roi_part1_config.c
            part1/cov_mat/cov_mat_common.c
            ${_cov_knl_src}
            part1/block_cholesky/block_cholesky_opt_bs4_knl.c
            part1/block_cholesky/block_linv_opt_knl_bs4.c
            ${_capon_roi_part2_kernel_srcs}
            CFLAGS ${_capon_roi_combined_cflags}
                "-DCAPON_ROI_BUILD_PART1=1"
                "-DCAPON_ROI_BUILD_PART2=1"
                "-DCAPON_ROI_USE_FP_COVMAT=${_use_fp_covmat}"
                "-DCAPON_ROI_ENABLE_SUBBIN_INTERP=${_enable_subbin_interp}"
            BCF
                ${_capon_roi_bcf})
    endforeach()
endforeach()
