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

# Check PVA version to conditionally include BCF file
if(${CUPVA_VERSION} VERSION_GREATER_EQUAL "2.6.0")
    # The mangled name of cupvaRasterDataFlowAdvProgImpl differs by SDK version.
    if(${CUPVA_VERSION} VERSION_GREATER_EQUAL "2.10.0")
        set(PVA_RDF_ADVPROG_SYMBOL "_Z30cupvaRasterDataFlowAdvProgImplR21RasterDataFlowHandler")
    else()
        set(PVA_RDF_ADVPROG_SYMBOL "_Z31_cupvaRasterDataFlowAdvProgImplR21RasterDataFlowHandler")
    endif()
    configure_file(canny_edge_detector.bcf.in
                   ${CMAKE_CURRENT_BINARY_DIR}/canny_edge_detector.bcf @ONLY)
    set(CANNY_BCF_FILE ${CMAKE_CURRENT_BINARY_DIR}/canny_edge_detector.bcf)
else()
    set(CANNY_BCF_FILE canny_edge_detector_noodle.bcf)
endif()

pva_solutions_device(pva_canny_edge_detector_dev
    canny_edge_detector_top.c
    canny_edge_detector_funcs.c
    canny_edge_detector_knls.c
    canny_edge_detector_ccl_funcs.c
    canny_edge_detector_ccl_knls.c
    BCF
        ${CANNY_BCF_FILE})
