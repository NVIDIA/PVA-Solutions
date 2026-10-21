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

pva_op_add_subdirectory(deep_learning pva-dl-lib)
pva_op_add_subdirectory(deep_learning depthtospace)
pva_op_add_subdirectory(deep_learning dl_inference)
pva_op_add_subdirectory(deep_learning nhwc_to_nchw_letterbox_normalize)
if(CUPVA_VERSION VERSION_GREATER_EQUAL "2.6.0")
    # Batched NMS uses the C++ SortPayload primitive. Device C++ source support
    # was added in PVA SDK 2.6.0.
    pva_op_add_subdirectory(deep_learning batched_nms)
endif()
pva_op_add_subdirectory(deep_learning custom)
