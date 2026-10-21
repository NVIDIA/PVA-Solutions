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

# Include guard: this file may be included from both the repo-root CMakeLists.txt
# and public/CMakeLists.txt depending on the build entry point.
if(_PVA_OP_FILTER_INCLUDED)
    return()
endif()
set(_PVA_OP_FILTER_INCLUDED TRUE)

# Multi-op build filter.
#
# When PVA_BUILD_OP_FILTER is set (e.g. -DPVA_BUILD_OP_FILTER='range_fft;doppler_fft;nci'),
# only those operators' sources are compiled. A single op
# (e.g. -DPVA_BUILD_OP_FILTER=morphology) is just a 1-element list. Empty value
# builds everything and is fully backward-compatible.
#
# Adding a new operator: write one pva_declare_op(...) block below.
#
# Discovery: run `cmake -DPVA_LIST_OPS=ON ..` from any build dir to print every
# declared op (with its flags) and exit before configuring further.
#
# After this file is included the following cache variables are set:
#   PVA_OPS_AVAILABLE             List of ops enabled in this configuration
#                                 (public + LLVM/CUPVA conditions met).
# If a filter is active, additionally:
#   PVA_FILTER_NAMES              List of canonical CamelCase op names
#   PVA_FILTER_EXTRA_OP_TARGETS   Extra cmake targets to add to OPERATORS_LIST
#   PVA_FILTER_REF_LIBS           Union of ref-lib targets (with imageIO_utils)
#   PVA_FILTER_OP_PATHS           Union of operator subdir paths (category/op)
#   PVA_FILTER_IS_RADAR_OP        ON if any selected op is a radar op
# Per-op metadata is exposed as _PVA_OP_<FIELD>_<op>; consumers (including the
# test-side cmake files) read those directly when they need additional info.

set(PVA_BUILD_OP_FILTER "" CACHE STRING
    "Semicolon-separated list of operator short names to build (e.g. 'morphology' or 'range_fft;doppler_fft;nci'). Empty = build all.")

option(PVA_LIST_OPS "Print all declared operators and exit before configuring further." OFF)

# ── pva_declare_op — single source of truth per operator ────────────────────
#
# Usage:
#   pva_declare_op(<op>
#       CATEGORY        <image_processing|radar_processing|deep_learning|linear_algebra|optimization|lidar_processing>
#       NAME            <CamelCase>            # canonical CamelCase op name
#       [REF            <lib> [<lib>...]]      # reference libraries
#       [EXTRA          <target> [<target>...]]# extra cmake targets to link into pva_operator
#       [DIRS           <dir> [<dir>...]]      # source subdirs; defaults to <CATEGORY>/<op>
#       [RADAR]                                # op is in the radar family (links radar_pipeline_utils)
#       [LLVM_ONLY]                            # requires CUPVA_LLVM_SUPPORTED (CUPVA >= 2.6.0)
#       [PRIVATE_ONLY]                         # lives in private/ tree; filter unsupported
#       [MIN_CUPVA      <version>])            # gate on CUPVA_VERSION >= <version>

set(_PVA_DECLARED_OPS "" CACHE INTERNAL "Ops registered by pva_declare_op")

macro(pva_declare_op _op)
    set(_opts  RADAR LLVM_ONLY PRIVATE_ONLY)
    set(_one   CATEGORY NAME MIN_CUPVA)
    set(_multi REF EXTRA DIRS)
    cmake_parse_arguments(_PDO "${_opts}" "${_one}" "${_multi}" ${ARGN})

    if(NOT _PDO_CATEGORY)
        message(FATAL_ERROR "pva_declare_op(${_op}): CATEGORY is required")
    endif()
    if(NOT _PDO_NAME)
        message(FATAL_ERROR "pva_declare_op(${_op}): NAME is required")
    endif()
    if(_PDO_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "pva_declare_op(${_op}): unknown argument(s): ${_PDO_UNPARSED_ARGUMENTS}")
    endif()

    if(NOT _PDO_DIRS)
        set(_PDO_DIRS "${_PDO_CATEGORY}/${_op}")
    endif()

    list(APPEND _PVA_DECLARED_OPS "${_op}")
    set(_PVA_OP_CATEGORY_${_op}     "${_PDO_CATEGORY}")
    set(_PVA_OP_NAME_${_op}         "${_PDO_NAME}")
    set(_PVA_OP_REF_${_op}          "${_PDO_REF}")
    set(_PVA_OP_EXTRA_${_op}        "${_PDO_EXTRA}")
    set(_PVA_OP_DIRS_${_op}         "${_PDO_DIRS}")
    set(_PVA_OP_RADAR_${_op}        "${_PDO_RADAR}")
    set(_PVA_OP_LLVM_ONLY_${_op}    "${_PDO_LLVM_ONLY}")
    set(_PVA_OP_PRIVATE_ONLY_${_op} "${_PDO_PRIVATE_ONLY}")
    set(_PVA_OP_MIN_CUPVA_${_op}    "${_PDO_MIN_CUPVA}")
endmacro()

# ── Operator declarations ───────────────────────────────────────────────────
# image_processing ----------------------------------------------------------
pva_declare_op(morphology             CATEGORY image_processing  NAME Morphology               REF morphology_ref)
pva_declare_op(image_histogram        CATEGORY image_processing  NAME ImageHistogram           REF image_histogram_ref)
pva_declare_op(corner_subpix          CATEGORY image_processing  NAME CornerSubPix             REF cornersubpix_ref
    DIRS image_processing/cornersubpix)
pva_declare_op(minmaxloc              CATEGORY image_processing  NAME MinMaxLoc                REF minmaxloc_ref)
pva_declare_op(template_matching      CATEGORY image_processing  NAME TemplateMatching         REF template_matching_ref)
pva_declare_op(mix_channels           CATEGORY image_processing  NAME MixChannels              REF mix_channels_ref)
pva_declare_op(bltopl                 CATEGORY image_processing  NAME BlToPl                   REF bltopl_ref)
pva_declare_op(convert_image_format   CATEGORY image_processing  NAME ConvertImageFormat       REF convert_image_format_ref)
pva_declare_op(image_blend            CATEGORY image_processing  NAME ImageBlend               REF image_blend_ref)
pva_declare_op(image_resize           CATEGORY image_processing  NAME ImageResize              REF image_resize_ref)
pva_declare_op(blur_filter_roi        CATEGORY image_processing  NAME BlurFilterROI            REF blur_filter_roi_ref)
pva_declare_op(distance_transform     CATEGORY image_processing  NAME DistanceTransform        REF distance_transform_ref)
pva_declare_op(bilateral_filter       CATEGORY image_processing  NAME BilateralFilter          REF bilateral_filter_ref)
pva_declare_op(brute_force_matcher    CATEGORY image_processing  NAME BruteForceMatcher        REF brute_force_matcher_ref)
pva_declare_op(warp_perspective       CATEGORY image_processing  NAME WarpPerspective          REF warp_perspective_ref)
pva_declare_op(image_stats            CATEGORY image_processing  NAME ImageStats               REF image_stats_ref)
pva_declare_op(canny_edge_detector    CATEGORY image_processing  NAME CannyEdgeDetector        REF canny_edge_detector_ref)
pva_declare_op(median_filter          CATEGORY image_processing  NAME MedianFilter             REF median_filter_ref)
pva_declare_op(histogram_equalization CATEGORY image_processing  NAME HistogramEqualization    REF histogram_equalization_ref)
pva_declare_op(background_subtractor  CATEGORY image_processing  NAME BackgroundSubtractor     REF background_subtractor_ref)
pva_declare_op(gaussian_pyramid       CATEGORY image_processing  NAME GaussianPyramid          REF gaussian_pyramid_ref)
pva_declare_op(remap                  CATEGORY image_processing  NAME Remap                    REF remap_ref)
pva_declare_op(orb_desc_extractor     CATEGORY image_processing  NAME ORBDescExtractor         REF orb_desc_extractor_ref)
pva_declare_op(image_flip             CATEGORY image_processing  NAME ImageFlip                REF image_flip_ref)
pva_declare_op(ccl                    CATEGORY image_processing  NAME CCL                      REF ccl_ref)
pva_declare_op(hog                    CATEGORY image_processing  NAME HOG                      REF hog_ref)
pva_declare_op(fastcorner             CATEGORY image_processing  NAME FastCornerDetector       REF fastcorner_ref
    LLVM_ONLY)
pva_declare_op(contour_tracing        CATEGORY image_processing  NAME ContourTracing           REF contour_tracing_ref
    LLVM_ONLY
    MIN_CUPVA 2.7.0)
pva_declare_op(nms                    CATEGORY image_processing  NAME Nms                      REF nms_ref
    LLVM_ONLY)
# conv2d / gaussian_filter / box_filter form an indivisible cluster:
#   - OpConv2d.cpp directly instantiates GaussianFilter<> / BoxFilter<>
#   - gaussian_filter and box_filter both inherit BaseConv2d (which lives in conv2d/)
#   - all three share the pva_conv_K{3,5,7}_* device kernels
# Filtering on any one of them therefore needs the other two's sources + targets.
pva_declare_op(conv2d                 CATEGORY image_processing  NAME Conv2d                   REF conv2d_ref
    EXTRA pva_gaussian_filter_op pva_box_filter_op
    DIRS image_processing/conv2d image_processing/gaussian_filter image_processing/box_filter
    LLVM_ONLY)
pva_declare_op(gaussian_filter        CATEGORY image_processing  NAME GaussianFilter           REF gaussian_filter_ref
    EXTRA pva_conv2d_op pva_box_filter_op
    DIRS image_processing/gaussian_filter image_processing/conv2d image_processing/box_filter
    LLVM_ONLY)
pva_declare_op(box_filter             CATEGORY image_processing  NAME BoxFilter                REF box_filter_ref
    EXTRA pva_conv2d_op pva_gaussian_filter_op
    DIRS image_processing/box_filter image_processing/conv2d image_processing/gaussian_filter
    LLVM_ONLY)

# deep_learning -------------------------------------------------------------
pva_declare_op(depthtospace           CATEGORY deep_learning     NAME DepthToSpace             REF depthtospace_ref)
pva_declare_op(nhwc_to_nchw_letterbox_normalize CATEGORY deep_learning NAME NhwcToNchwLetterboxNormalize
    REF nhwc_to_nchw_letterbox_normalize_ref)
# dl_inference also links the pva_dl_lib helper target and has no *_ref library
# (its sample assets — swin_t, resnet50, etc. — are handled by the test-side
# cmake, not by a reference library).
pva_declare_op(dl_inference           CATEGORY deep_learning     NAME DLInference
    EXTRA pva_dl_lib
    DIRS deep_learning/dl_inference deep_learning/pva-dl-lib)

pva_declare_op(batched_nms            CATEGORY deep_learning     NAME BatchedNMS               REF batched_nms_ref
    LLVM_ONLY)
pva_declare_op(yolov5_decode          CATEGORY deep_learning     NAME YoloV5Decode
    REF yolov5_decode_ref
    DIRS deep_learning/custom)

# linear_algebra ------------------------------------------------------------
pva_declare_op(batch_svd              CATEGORY linear_algebra    NAME BatchSVD                 REF batch_svd_ref)
pva_declare_op(bundle_adjustment      CATEGORY linear_algebra    NAME BundleAdjustment          REF bundle_adjustment_ref)

# optimization --------------------------------------------------------------
pva_declare_op(floyd_warshall         CATEGORY optimization      NAME FloydWarshall            REF floyd_warshall_ref)

# lidar_processing ----------------------------------------------------------
pva_declare_op(lidar_histogram_conversion CATEGORY lidar_processing NAME LidarHistogramConversion REF lidar_histogram_conversion_ref
    DIRS lidar_processing/histogram_conversion
    LLVM_ONLY
    MIN_CUPVA 2.6.0)

# radar_processing ----------------------------------------------------------
# range_fft_ref and doppler_fft_ref are OBJECT libs that PRIVATE-link
# radar_ref_utils; PRIVATE deps of OBJECT libs do not propagate at link
# time, so consumers of reference_c_models need both libraries explicitly.
pva_declare_op(range_fft              CATEGORY radar_processing  NAME RangeFFT                 REF range_fft_ref radar_ref_utils
    RADAR)
pva_declare_op(doppler_fft            CATEGORY radar_processing  NAME DopplerFFT               REF doppler_fft_ref radar_ref_utils
    RADAR)
pva_declare_op(coherent_integration   CATEGORY radar_processing  NAME CoherentIntegration      REF coherent_integration_ref
    RADAR)
pva_declare_op(nci                    CATEGORY radar_processing  NAME Nci                      REF nci_ref
    DIRS radar_processing/Nci
    RADAR)
pva_declare_op(radar_cfar             CATEGORY radar_processing  NAME RadarCFAR                REF radar_cfar_ref
    DIRS radar_processing/cfar
    RADAR)
pva_declare_op(snapshot_extraction    CATEGORY radar_processing  NAME SnapshotExtraction       REF snapshot_extraction_ref
    RADAR)
pva_declare_op(radar_transpose        CATEGORY radar_processing  NAME RadarTranspose           REF radar_transpose_ref
    RADAR)
# bartlett's ref-lib chain pulls doa_angle_fft_ref + snapshot_extraction_ref
# headers, and the .cpp references checkTargetProcessingOutput from
# target_processing_ref. Same shape for target_processing.
pva_declare_op(bartlett               CATEGORY radar_processing  NAME Bartlett                 REF doa_bartlett_beamforming_ref doa_angle_fft_ref snapshot_extraction_ref target_processing_ref
    DIRS radar_processing/doa_bartlett_beamforming
    RADAR)
pva_declare_op(capon_roi              CATEGORY radar_processing  NAME DoaCaponROI              REF doa_capon_roi_ref
    DIRS radar_processing/doa_capon_roi
    RADAR
    LLVM_ONLY)
# capon_roi_preprocess produces the snapshotsROI + steeringVectorsROI consumed by a Capon
# DoA-on-ROI stage. Its system tests check the reference rejects what the operator rejects.
pva_declare_op(capon_roi_preprocess   CATEGORY radar_processing  NAME CaponROIPreprocess REF capon_roi_preprocess_ref
    DIRS radar_processing/capon_roi_preprocess radar_processing/common/capon_roi_beamspace
    RADAR)
# doa_da_fft / peak_detection / doa_angle_fft / target_processing all live in
# the shared radar_processing/custom/ subdir. The inner CMakeLists.txt
# EXCLUDE_FROM_ALLs the non-selected three when filtering on any one of them,
# so the final binary is still clean.
pva_declare_op(doa_da_fft             CATEGORY radar_processing  NAME DoaDAFFT                 REF doa_target_processing_ref
    DIRS radar_processing/custom
    RADAR)
pva_declare_op(peak_detection         CATEGORY radar_processing  NAME PeakDetection            REF peak_detection_ref
    DIRS radar_processing/custom
    RADAR)
pva_declare_op(doa_angle_fft          CATEGORY radar_processing  NAME DoaAngleFFT              REF doa_angle_fft_ref snapshot_extraction_ref
    DIRS radar_processing/custom
    RADAR)
pva_declare_op(target_processing      CATEGORY radar_processing  NAME TargetProcessing         REF target_processing_ref doa_angle_fft_ref snapshot_extraction_ref
    DIRS radar_processing/custom
    RADAR)
# Private-only operators: declared so the filter recognizes the name and can
# emit a useful FATAL_ERROR; cannot actually be filter-built from public tree.
pva_declare_op(radar_fixed_point          CATEGORY radar_processing NAME RadarFixedPoint         REF radar_ref_utils RADAR PRIVATE_ONLY)
pva_declare_op(radar_advanced_fixed_point CATEGORY radar_processing NAME RadarAdvancedFixedPoint REF radar_ref_utils RADAR PRIVATE_ONLY)

# ── Discovery: print declared ops and exit ──────────────────────────────────
if(PVA_LIST_OPS)
    message(STATUS "")
    message(STATUS "Declared operators (use any of these as PVA_BUILD_OP_FILTER values):")
    foreach(_op IN LISTS _PVA_DECLARED_OPS)
        set(_flags "")
        if(_PVA_OP_RADAR_${_op})
            list(APPEND _flags RADAR)
        endif()
        if(_PVA_OP_LLVM_ONLY_${_op})
            list(APPEND _flags LLVM_ONLY)
        endif()
        if(_PVA_OP_PRIVATE_ONLY_${_op})
            list(APPEND _flags PRIVATE_ONLY)
        endif()
        if(_PVA_OP_MIN_CUPVA_${_op})
            list(APPEND _flags "CUPVA>=${_PVA_OP_MIN_CUPVA_${_op}}")
        endif()
        string(REPLACE ";" "," _flags_pretty "${_flags}")
        if(_flags_pretty)
            message(STATUS "  ${_op}  [${_flags_pretty}]")
        else()
            message(STATUS "  ${_op}")
        endif()
    endforeach()
    message(STATUS "")
    list(LENGTH _PVA_DECLARED_OPS _n_ops)
    message(FATAL_ERROR "PVA_LIST_OPS=ON: ${_n_ops} operators listed; stopping before further configuration.")
endif()

# ── Derived op sets ─────────────────────────────────────────────────────────
# PVA_OPS_AVAILABLE  : ops built into libpva_operator.so unfiltered. Excludes
#                     PRIVATE_ONLY (their source lives in private/).
# PVA_OPS_WITH_TESTS : ops with test/workload/kpi source in the public tests/
#                     tree. Includes PRIVATE_ONLY (they have public test files
#                     even though the operator implementation is private).
# Both apply the LLVM_ONLY and MIN_CUPVA gates.
set(PVA_OPS_AVAILABLE "")
set(PVA_OPS_WITH_TESTS "")
foreach(_op IN LISTS _PVA_DECLARED_OPS)
    if(_PVA_OP_LLVM_ONLY_${_op} AND NOT CUPVA_LLVM_SUPPORTED)
        continue()
    endif()
    if(_PVA_OP_MIN_CUPVA_${_op} AND CUPVA_VERSION VERSION_LESS "${_PVA_OP_MIN_CUPVA_${_op}}")
        continue()
    endif()
    list(APPEND PVA_OPS_WITH_TESTS "${_op}")
    if(NOT _PVA_OP_PRIVATE_ONLY_${_op})
        list(APPEND PVA_OPS_AVAILABLE "${_op}")
    endif()
endforeach()
set(PVA_OPS_AVAILABLE  "${PVA_OPS_AVAILABLE}"  CACHE INTERNAL "")
set(PVA_OPS_WITH_TESTS "${PVA_OPS_WITH_TESTS}" CACHE INTERNAL "")

# ── Helper used by category CMakeLists.txt ──────────────────────────────────
# Adds <category>/<dir> with EXCLUDE_FROM_ALL when a filter is active and the
# path is not in the union of selected ops' DIRS. The full-build path is
# unchanged.
macro(pva_op_add_subdirectory category dir)
    if(PVA_BUILD_OP_FILTER AND NOT "${category}/${dir}" IN_LIST PVA_FILTER_OP_PATHS)
        add_subdirectory(${dir} EXCLUDE_FROM_ALL)
    else()
        add_subdirectory(${dir})
    endif()
endmacro()

# ── Process PVA_BUILD_OP_FILTER ─────────────────────────────────────────────
if(NOT PVA_BUILD_OP_FILTER)
    set(PVA_FILTER_NAMES            "" CACHE INTERNAL "")
    set(PVA_FILTER_EXTRA_OP_TARGETS "" CACHE INTERNAL "")
    set(PVA_FILTER_REF_LIBS         "" CACHE INTERNAL "")
    set(PVA_FILTER_IS_RADAR_OP      "" CACHE INTERNAL "")
    set(PVA_FILTER_OP_PATHS         "" CACHE INTERNAL "")
    return()
endif()

set(_filter_names "")
set(_filter_extras "")
set(_filter_refs "imageIO_utils")
set(_filter_paths "")
set(_filter_is_radar OFF)

foreach(_op IN LISTS PVA_BUILD_OP_FILTER)
    if(NOT "${_op}" IN_LIST _PVA_DECLARED_OPS)
        message(FATAL_ERROR
            "PVA_BUILD_OP_FILTER contains '${_op}' which is not declared.\n"
            "Add a pva_declare_op(${_op} ...) block in public/cmake/pva_op_filter.cmake,\n"
            "or run 'cmake -DPVA_LIST_OPS=ON ..' to see all known ops.\n"
            "Known ops: ${_PVA_DECLARED_OPS}")
    endif()
    if(_PVA_OP_PRIVATE_ONLY_${_op})
        message(FATAL_ERROR
            "PVA_BUILD_OP_FILTER contains '${_op}' which is a private-only operator.\n"
            "Single-op filter builds are not supported for this operator from the public tree.")
    endif()
    if(_PVA_OP_LLVM_ONLY_${_op} AND NOT CUPVA_LLVM_SUPPORTED)
        message(FATAL_ERROR
            "PVA_BUILD_OP_FILTER contains '${_op}' which requires LLVM support (CUPVA >= 2.6.0).\n"
            "Current CUPVA version is ${CUPVA_VERSION}.")
    endif()
    if(_PVA_OP_MIN_CUPVA_${_op} AND CUPVA_VERSION VERSION_LESS "${_PVA_OP_MIN_CUPVA_${_op}}")
        message(FATAL_ERROR
            "PVA_BUILD_OP_FILTER contains '${_op}' which requires CUPVA >= ${_PVA_OP_MIN_CUPVA_${_op}}.\n"
            "Current CUPVA version is ${CUPVA_VERSION}.")
    endif()

    list(APPEND _filter_names "${_PVA_OP_NAME_${_op}}")
    list(APPEND _filter_extras ${_PVA_OP_EXTRA_${_op}})
    list(APPEND _filter_refs   ${_PVA_OP_REF_${_op}})
    list(APPEND _filter_paths  ${_PVA_OP_DIRS_${_op}})
    if(_PVA_OP_RADAR_${_op})
        set(_filter_is_radar ON)
    endif()
endforeach()

list(REMOVE_DUPLICATES _filter_extras)
list(REMOVE_DUPLICATES _filter_refs)
list(REMOVE_DUPLICATES _filter_paths)

set(PVA_FILTER_NAMES            "${_filter_names}"      CACHE INTERNAL "")
set(PVA_FILTER_EXTRA_OP_TARGETS "${_filter_extras}"     CACHE INTERNAL "")
set(PVA_FILTER_REF_LIBS         "${_filter_refs}"       CACHE INTERNAL "")
set(PVA_FILTER_IS_RADAR_OP      "${_filter_is_radar}"   CACHE INTERNAL "")
set(PVA_FILTER_OP_PATHS         "${_filter_paths}"      CACHE INTERNAL "")

message(STATUS "PVA op filter: ${PVA_BUILD_OP_FILTER}  (radar=${_filter_is_radar})")
