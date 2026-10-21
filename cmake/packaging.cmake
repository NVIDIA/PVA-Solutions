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

# Generate required packaging rules

# ------------------------------------------------------------------------------
# Top level CPack options
# ------------------------------------------------------------------------------
set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_CONTACT "CUPVA Support <cupva-support@nvidia.com>")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${pva-solutions_SOURCE_DIR}/DESCRIPTION")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PVA Solutions operators and primitives")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://www.nvidia.com")

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

# We append CUPVA_VERSION_SUFFIX to debian package versions
set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${CPACK_PACKAGE_VERSION}")

set(CPACK_GENERATOR DEB)
set(CPACK_STRIP_FILES TRUE)

# Create a separate deb for each component
set(CPACK_MONOLITHIC_INSTALL OFF)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_GROUPING IGNORE)

# Install under /opt/nvidia/pva-solutions-<maj>.<min>
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
set(CMAKE_INSTALL_PREFIX "/opt/nvidia/${CPACK_PACKAGE_NAME}")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/nvidia/${CPACK_PACKAGE_NAME}")

# ------------------------------------------------------------------------------
# For L4T, expect to install the packages on arm64
# ------------------------------------------------------------------------------
set(CPACK_DEBIAN_L4T_PACKAGE_ARCHITECTURE arm64)
set(CPACK_DEBIAN_L4T-SAMPLES_PACKAGE_ARCHITECTURE arm64)

# ------------------------------------------------------------------------------
# Install sample assets for all architectures
# ------------------------------------------------------------------------------
set(CPACK_DEBIAN_SAMPLE-ASSETS_PACKAGE_DEPENDS "")
set(CPACK_DEBIAN_SAMPLE-ASSETS_PACKAGE_ARCHITECTURE all)

# ------------------------------------------------------------------------------
# Include ldconf step in runtime packages, except for QNX
# ------------------------------------------------------------------------------
if(NOT CUPVA_BUILD_MODE STREQUAL QNX)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/ld.so.conf.in" ld.so.conf @ONLY)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/pvasol_postinst.in" postinst @ONLY)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/pvasol_postrm.in" postrm @ONLY)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ld.so.conf
        DESTINATION "/etc/ld.so.conf.d"
        RENAME ${CPACK_PACKAGE_NAME}-${PVASOL_RUNTIME_PACKAGE}.conf
        COMPONENT ${PVASOL_RUNTIME_PACKAGE})

    string(TOUPPER ${PVASOL_RUNTIME_PACKAGE} _PVASOL_RUNTIME_PACKAGE_UPPER)
    set(CPACK_DEBIAN_${_PVASOL_RUNTIME_PACKAGE_UPPER}_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/postrm")
endif()

# ------------------------------------------------------------------------------
# Finalize CPack rules
# ------------------------------------------------------------------------------
include(CPack)
