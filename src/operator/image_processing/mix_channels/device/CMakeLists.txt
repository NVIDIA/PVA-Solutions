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

pva_solutions_device(pva_split_rgba8_to_u8_dev
    split_rgba8_to_u8/split_rgba8_to_u8_top.c
    split_rgba8_to_u8/split_rgba8_to_u8_knl.c)

pva_solutions_device(pva_merge_u8_to_rgba8_dev
    merge_u8_to_rgba8/merge_u8_to_rgba8_top.c
    merge_u8_to_rgba8/merge_u8_to_rgba8_knl.c)