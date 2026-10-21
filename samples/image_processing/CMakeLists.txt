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

add_subdirectory(morphology)
add_subdirectory(image_histogram)
add_subdirectory(cornersubpix)
add_subdirectory(minmaxloc)
add_subdirectory(template_matching)
add_subdirectory(mix_channels)
add_subdirectory(bltopl)
add_subdirectory(convert_image_format)
add_subdirectory(image_blend)
add_subdirectory(image_resize)
add_subdirectory(blur_filter_roi)
add_subdirectory(distance_transform)
add_subdirectory(bilateral_filter)
add_subdirectory(brute_force_matcher)
add_subdirectory(warp_perspective)
add_subdirectory(image_stats)
add_subdirectory(canny_edge_detector)
add_subdirectory(median_filter)
add_subdirectory(histogram_equalization)
add_subdirectory(background_subtractor)
add_subdirectory(gaussian_pyramid)
add_subdirectory(remap)
add_subdirectory(orb_desc_extractor)
add_subdirectory(image_flip)
add_subdirectory(ccl)
add_subdirectory(hog)

# gaussian_filter, fastcorner, conv2d, box_filter, contour_tracing, nms are only supported on LLVM
if(CUPVA_LLVM_SUPPORTED)
    add_subdirectory(gaussian_filter)
    add_subdirectory(fastcorner)
    add_subdirectory(conv2d)
    add_subdirectory(box_filter)
    if(${CUPVA_VERSION} VERSION_GREATER_EQUAL "2.7.0")
        add_subdirectory(contour_tracing)
    endif()
    add_subdirectory(nms)
endif()
