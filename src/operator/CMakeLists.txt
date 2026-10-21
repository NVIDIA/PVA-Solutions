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

add_library(pva_operator_includes INTERFACE)
target_include_directories(pva_operator_includes
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(pva_allocator OBJECT
    PvaAllocator.cpp
)

target_link_libraries(pva_allocator
    PRIVATE pva_operator_priv
    PRIVATE pva_operator_includes
    PRIVATE cupva_host
    PRIVATE nvcv_types
)

add_library(pva_operator SHARED
    PvaOperator.cpp
)

target_link_libraries(pva_operator
    PRIVATE pva_operator_priv
    PRIVATE cupva_host
    PRIVATE nvcv_types
    PUBLIC pva_allocator
    PUBLIC pva_operator_includes
)

# Link submit macros with whole-archive to ensure all symbols are exported
target_link_libraries(pva_operator PRIVATE -Wl,--whole-archive pva_operator_priv_submit_definitions -Wl,--no-whole-archive)

# Check if IMPORT_DEVICE_TARGETS is set for NATIVE build
if(IMPORT_DEVICE_TARGETS AND CUPVA_BUILD_MODE STREQUAL "NATIVE")
    message(WARNING "IMPORT_DEVICE_TARGETS cannot be used with NATIVE build mode. Forcing IMPORT_DEVICE_TARGETS=OFF")
    set(IMPORT_DEVICE_TARGETS OFF)
endif()

add_subdirectory(priv)
add_subdirectory(image_processing)
add_subdirectory(deep_learning)
add_subdirectory(linear_algebra)
add_subdirectory(radar_processing)
add_subdirectory(lidar_processing)
add_subdirectory(utils)
add_subdirectory(optimization)

# OPERATORS_LIST is derived from the declarations in pva_op_filter.cmake.
# Unfiltered: every public op enabled in this configuration + their EXTRA targets.
# Filtered:   the selected ops + their EXTRA targets (e.g. pva_dl_lib for dl_inference,
#             the conv-cluster siblings for conv2d/gaussian_filter/box_filter).
if(PVA_BUILD_OP_FILTER)
    set(OPERATORS_LIST "${PVA_FILTER_EXTRA_OP_TARGETS}")
    foreach(_op IN LISTS PVA_BUILD_OP_FILTER)
        list(APPEND OPERATORS_LIST "pva_${_op}_op")
    endforeach()
else()
    set(OPERATORS_LIST "")
    foreach(_op IN LISTS PVA_OPS_AVAILABLE)
        list(APPEND OPERATORS_LIST "pva_${_op}_op")
        list(APPEND OPERATORS_LIST ${_PVA_OP_EXTRA_${_op}})
    endforeach()
endif()
list(REMOVE_DUPLICATES OPERATORS_LIST)

add_operators(pva_operator ${OPERATORS_LIST})

# For now, we don't add 'exports' - this would be needed along with headers in a dev package
# if we ever want to ship binaries
install(TARGETS pva_operator
        EXPORT pvasol_runtime_libs
        LIBRARY DESTINATION ${PVASOL_LIB_INSTALL_PATH}
        COMPONENT ${PVASOL_RUNTIME_PACKAGE})

# Set the allowlist file path
if(IMPORT_DEVICE_TARGETS)
    set(source_allowlist_file "${DEVICE_BINARIES_ROOT}/${CUPVA_RUNTIME_VERSION}/pva_allowlist_default_all")
else()
    set(source_allowlist_file "${CMAKE_BINARY_DIR}/pva_allowlist_default_all")
endif()

# Set the destination file name
set(destination_allowlist_file "${CMAKE_BINARY_DIR}/cupva_allowlist_pva_operators_all")

# Determine the correct dependency for the allowlist
if(IMPORT_DEVICE_TARGETS)
    set(allowlist_dependency ${source_allowlist_file})
else()
    # When generating, depend on the target that creates pva_allowlist_default_all
    set(allowlist_dependency pva_allowlist_default_all_target)
endif()

if(PVA_BUILD_OP_FILTER)
    # In filtered single-op builds skip allowlist generation — it requires all
    # operators' device kernels and would defeat the purpose of the filter.
    # The stub target keeps downstream DEPENDS references valid.
    add_custom_target(copy_pva_allowlist
        COMMAND ${CMAKE_COMMAND} -E true
        COMMENT "Allowlist generation skipped (single-op filter build)"
    )
else()
    #Create a target for copying the allowlist file
    # This ensures the library is fully built before attempting to copy the file
    add_custom_target(copy_pva_allowlist ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${source_allowlist_file} ${destination_allowlist_file}
        DEPENDS pva_operator ${allowlist_dependency}
        COMMENT "Copying and renaming allowlist file"
    )

    get_target_property(dl_lib_allowlist_file pva_dl_lib ALLOWLIST_FILE)

    if(dl_lib_allowlist_file)
        # Merge the DL allowlist into the main allowlist
        add_custom_target(merge_pva_dl_lib_allowlist ALL
            COMMAND python3 "${CUPVA_ALLOWLIST_SCRIPT}"
                    -i "${destination_allowlist_file}" "${dl_lib_allowlist_file}"
                    -o "${destination_allowlist_file}"
            DEPENDS copy_pva_allowlist # Ensures destination_allowlist_file exists from the copy step
            COMMENT "Merging pva-dl-lib allowlist into main allowlist"
            VERBATIM
        )
    endif()
endif()

# Install the renamed allowlist file (skipped in filtered builds — the file is not generated)
if(NOT PVA_BUILD_OP_FILTER)
    install(FILES "${destination_allowlist_file}"
            DESTINATION ${PVASOL_LIB_INSTALL_PATH}
            COMPONENT ${PVASOL_RUNTIME_PACKAGE})
endif()

# Install nvcv_types library
install(TARGETS nvcv_types
        EXPORT pvasol_runtime_libs
        LIBRARY DESTINATION ${PVASOL_LIB_INSTALL_PATH}
        COMPONENT ${PVASOL_RUNTIME_PACKAGE})
