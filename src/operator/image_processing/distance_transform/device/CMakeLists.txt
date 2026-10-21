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

find_package(cupva REQUIRED)

set(VPUBIN_NAMES)

pva_solutions_device(pva_distance_transform_U16_dev
        distance_transform_vpu.c
        distance_transform_knl.c)

list(APPEND VPUBIN_NAMES pva_distance_transform_U16_dev)

set(VPUBIN_NAMES ${VPUBIN_NAMES} PARENT_SCOPE)