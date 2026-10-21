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

set(PVA_ONNX_COMPILER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tools/pva-onnx-compiler)
set(GENERATED_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/operator/deep_learning/inference/)

if (CUPVA_BUILD_MODE STREQUAL "QNX" OR CUPVA_BUILD_MODE STREQUAL "L4T")
    set(LLC_ADDITIONAL_FLAGS "-mtriple=aarch64-linux-gnu")
else()
    set(LLC_ADDITIONAL_FLAGS "")
endif()

function(pva_compile_networks NETWORK_TARGET)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs NETWORKS)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create an object library to collect all .s files and compile definitions
    add_library(${NETWORK_TARGET} OBJECT)
    set_source_files_properties(${NETWORK_TARGET} PROPERTIES GENERATED TRUE)

    foreach(network ${ARGS_NETWORKS})
        string(REPLACE "|" ";" args_networks "${network}")

        list(LENGTH args_networks list_length)
        if(list_length LESS 4)
            message(FATAL_ERROR "Network must have 4 components: input_model|network_name|onnx_format|operator_format")
        endif()

        list(GET args_networks 0 input_model)
        list(GET args_networks 1 network_name)
        list(GET args_networks 2 onnx_format)
        list(GET args_networks 3 operator_format)

        # Create a custom command to generate the .s file
        add_custom_command(
            OUTPUT ${network_name}.s
            COMMAND ${PVA_ONNX_COMPILER_PATH} -onnx-format="${onnx_format}" -engine-format="${operator_format}" -name=${network_name} ${input_model} -o ${network_name}.ll
            COMMAND llc -relocation-model=pic ${LLC_ADDITIONAL_FLAGS} ${network_name}.ll -o ${network_name}.s
            COMMAND mv ${input_model}.tmp.mlir ${network_name}.onnx.mlir
            COMMENT "Compiling ONNX model ${input_model} to ${network_name}.s"
        )

        # Add the .s file to the target's sources
        target_sources(${NETWORK_TARGET} PRIVATE ${network_name}.s)
    endforeach()
endfunction()

function(prepare_onnx_model TARGET_NAME ASSETS_DIR INPUT_MODEL_NAME APP_TYPE OUTPUT_NODES)
    set(OUTPUT_MODEL_NAME "${INPUT_MODEL_NAME}_processed.onnx")
    set(OUTPUT_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_MODEL_NAME}")

    set(INPUT_MODEL_PATH "${ASSETS_DIR}/${INPUT_MODEL_NAME}")
    set(PROCESSING_SCRIPT_PATH "${ASSETS_DIR}/quantize_model.py")

    if(NOT APP_TYPE STREQUAL "classification" AND NOT APP_TYPE STREQUAL "detection")
        message(FATAL_ERROR "Invalid APP_TYPE: '${APP_TYPE}'. Must be 'classification' or 'detection'")
    endif()
    # TODO make this more flexible
    if(APP_TYPE STREQUAL "classification")
        set(INPUT_IMAGE_PATH "${ASSETS_DIR}/daisy.jpg")
    else()
        set(INPUT_IMAGE_PATH "${ASSETS_DIR}/traffic.jpg")
    endif()

    # If the input model is not found in the assets directory, check if it exists in the default ONNX model directory
    if(NOT EXISTS "${INPUT_MODEL_PATH}")
        if(EXISTS "${DEFAULT_ONNX_MODEL_DIR}/${INPUT_MODEL_NAME}")
            set(INPUT_MODEL_PATH "${DEFAULT_ONNX_MODEL_DIR}/${INPUT_MODEL_NAME}")
            message(STATUS "Input model ${INPUT_MODEL_PATH} not found, using ${DEFAULT_ONNX_MODEL_DIR}/${INPUT_MODEL_NAME}")
        endif()
    endif()

    if(EXISTS "${INPUT_MODEL_PATH}")
        # Essential check for the main processing script itself
        if(NOT EXISTS "${PROCESSING_SCRIPT_PATH}")
            message(FATAL_ERROR "Main processing script not found: ${PROCESSING_SCRIPT_PATH}.")
        endif()

        # Prepare a list of dependencies for the custom command.
        # Start with the input model and the main processing script.
        set(CMD_DEPENDENCIES "${INPUT_MODEL_PATH}" "${PROCESSING_SCRIPT_PATH}")

        set(_output_nodes_arg "")
        if(OUTPUT_NODES)
            set(_output_nodes_arg "--output-nodes" "${OUTPUT_NODES}")
        endif()

        add_custom_command(
            OUTPUT ${OUTPUT_MODEL_PATH}
            COMMAND python3 ${PROCESSING_SCRIPT_PATH}
                    --image-path "${INPUT_IMAGE_PATH}"
                    --original-model-path "${INPUT_MODEL_PATH}"
                    --quantized-model-path "${OUTPUT_MODEL_PATH}"
                    --app "${APP_TYPE}"
                    --remove-first-quantize-node
                    ${_output_nodes_arg}
                    2> /dev/null
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${CMD_DEPENDENCIES}
            COMMENT "Processing ONNX model '${INPUT_MODEL_NAME}' using script '${PROCESSING_SCRIPT_PATH}' -> '${OUTPUT_MODEL_NAME}' (in ${CMAKE_CURRENT_BINARY_DIR})"
            VERBATIM
        )

        add_custom_target(${TARGET_NAME}
            DEPENDS ${OUTPUT_MODEL_PATH}
        )

        message(STATUS "Added custom target '${TARGET_NAME}' to generate ${OUTPUT_MODEL_NAME} in ${CMAKE_CURRENT_BINARY_DIR} using ${PROCESSING_SCRIPT_PATH}")
        message(STATUS "The prepared model will be at: ${OUTPUT_MODEL_PATH}")

        set_target_properties(${TARGET_NAME} PROPERTIES GENERATED_MODEL_PATH ${OUTPUT_MODEL_PATH})

    else()
        get_filename_component(INPUT_MODEL_DIR "${INPUT_MODEL_PATH}" DIRECTORY)
        message(WARNING "Original model (${INPUT_MODEL_NAME}) not found in ${INPUT_MODEL_DIR}. '${TARGET_NAME}' target will not be created.")
    endif()

endfunction()

function(pva_compile_multiple_networks COMPILED_MODELS_OUT MODEL_CONFIGS ASSETS_DIR APP_TYPE)
    set(_compiled_models "")

    foreach(MODEL_CONFIG ${MODEL_CONFIGS})
        string(REPLACE "|" ";" CONFIG_LIST "${MODEL_CONFIG}")
        list(GET CONFIG_LIST 0 MODEL_NAME)
        list(GET CONFIG_LIST 1 ALIAS)
        list(GET CONFIG_LIST 2 ONNX_FORMAT)
        list(GET CONFIG_LIST 3 OPERATOR_FORMAT)

        # Optional 5th field: comma-separated output node names
        list(LENGTH CONFIG_LIST _config_len)
        set(_output_nodes "")
        if(_config_len GREATER 4)
            list(GET CONFIG_LIST 4 _output_nodes)
        endif()

        set(PROCESSED_TARGET "processed_${ALIAS}_model")
        set(COMPILE_TARGET "${ALIAS}")

        prepare_onnx_model(${PROCESSED_TARGET}
            "${ASSETS_DIR}"
            "${MODEL_NAME}"
            "${APP_TYPE}"
            "${_output_nodes}"
        )

        if(NOT TARGET ${PROCESSED_TARGET})
            message(WARNING "Input model ${MODEL_NAME} not found or processing failed")
            continue()
        endif()

        get_target_property(PROCESSED_MODEL_PATH ${PROCESSED_TARGET} GENERATED_MODEL_PATH)

        # Compile the model
        pva_compile_networks(${COMPILE_TARGET}
            NETWORKS "${PROCESSED_MODEL_PATH}|${ALIAS}|${ONNX_FORMAT}|${OPERATOR_FORMAT}")

        add_dependencies(${COMPILE_TARGET} ${PROCESSED_TARGET})

        # Add compiled model to the list
        list(APPEND _compiled_models $<TARGET_OBJECTS:${COMPILE_TARGET}>)
    endforeach()

    set(${COMPILED_MODELS_OUT} "${_compiled_models}" PARENT_SCOPE)
endfunction()
