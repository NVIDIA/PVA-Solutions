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

add_library(imageIO_utils OBJECT
    ImageIO.c
    BlockLinear.cpp
)


target_link_libraries(imageIO_utils
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE cupva_host
    PRIVATE cupva_host_utils
    PUBLIC lodepng
)

target_include_directories(imageIO_utils
    PUBLIC .
)

add_library(host_pointer_utils STATIC
    HostPointer.c
)

target_link_libraries(host_pointer_utils
    PRIVATE nvcv_types
    PRIVATE cupva_host
)

target_include_directories(host_pointer_utils
    PUBLIC .
)

add_library(math_utils OBJECT
    MathUtils.c
)

target_link_libraries(math_utils
    PRIVATE nvcv_types
    PRIVATE pva_operator
    PRIVATE cupva_host
    PRIVATE cupva_host_utils
)

target_include_directories(math_utils
    PUBLIC .
)

add_library(radar_ref_utils OBJECT
    RadarRefUtils.cpp
)

target_link_libraries(radar_ref_utils
    PRIVATE nvcv_types
    PRIVATE pva_operator_priv
    PUBLIC pva_operator_priv_includes
)

target_include_directories(radar_ref_utils
    PUBLIC .
)

if(CUPVA_BUILD_MODE STREQUAL "L4T" OR CUPVA_BUILD_MODE STREQUAL "QNX")
    add_library(interop_utils OBJECT
        priv/InteropUtilsCommon.c
        $<$<BOOL:${NVSCI_SUPPORTED}>:NvSciInteropUtils.c>
        CudaInteropUtils.c
    )

    target_link_libraries(interop_utils
        PRIVATE nvcv_types
        PRIVATE cupva_host
        PUBLIC host_pointer_utils
        PRIVATE pva_operator
        PRIVATE $<$<BOOL:${NVSCI_SUPPORTED}>:nvscibuf>
        PRIVATE $<$<BOOL:${NVSCI_SUPPORTED}>:nvscisync>
        PRIVATE CUDA::cudart_static
    )

    target_include_directories(interop_utils
        PUBLIC .
        PRIVATE priv
    )
endif()


