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

add_library(convert_image_format_ref OBJECT
    convert_image_format_ref.cpp
)

target_link_libraries(convert_image_format_ref
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE cupva_host
    PUBLIC host_pointer_utils
    PRIVATE CUDA::cudart_static
)

target_include_directories(convert_image_format_ref
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
