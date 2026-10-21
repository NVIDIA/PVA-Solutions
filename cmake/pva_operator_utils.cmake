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

# Function to handle device targets with either compilation or import
# Note: DEVICE_BINARIES_ROOT must be set before calling this function when IMPORT_DEVICE_TARGETS is enabled
function(pva_solutions_device TARGET)
    if(IMPORT_DEVICE_TARGETS)
        set(device_binary_path "${DEVICE_BINARIES_ROOT}/${CUPVA_RUNTIME_VERSION}/${target_name}")
        set(OUTPUT_FILE "${device_binary_path}/${TARGET}__merged.bin")

        add_custom_target(${TARGET} ALL DEPENDS ${OUTPUT_FILE})
        set_property(TARGET ${TARGET} APPEND PROPERTY MERGED_BIN_OUTPUT ${OUTPUT_FILE})
    else()
        # When not importing, pass through to pva_device
        pva_device(${TARGET} ${ARGN})
    endif()
endfunction()

# Function to add operators with whole-archive linking
function(add_operators target)
    # Construct the whole-archive flags
    # This forces the linker to include all object files from the static libraries,
    # even if they appear unused.
    set(whole_archive_flag "-Wl,--whole-archive")
    set(no_whole_archive_flag "-Wl,--no-whole-archive")

    # Initialize an empty list to hold the libraries with flags
    set(operators "")

    # Iterate over the libraries passed to the function
    foreach(library IN LISTS ARGN)
        # Check if the library exists as a target
        if(TARGET ${library})
            # Append the whole-archive flags around each library
            list(APPEND operators ${whole_archive_flag} ${library} ${no_whole_archive_flag})
        endif()
    endforeach()

    # Apply the flags by modifying the target's link options
    target_link_libraries(${target} PRIVATE ${operators})
endfunction()

