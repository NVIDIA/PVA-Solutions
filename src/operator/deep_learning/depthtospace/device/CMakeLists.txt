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

# The operator ships two data-flow paths and the host picks between them at Create
# time, so both device sides must always be present.

# General fallback: one GatherScatterDataFlow + DLUT executable that serves every
# supported shape. blk_sz is a runtime field of depth_to_space_host2dev_param_t here,
# so a single executable covers all three block sizes.
pva_solutions_device(pva_depthtospace_gsdf_dev
        depthtospace_gsdf_top.c)

# Fast path: one specialized RasterDataFlow executable per block size. BLK_SZ is a
# build-time constant so each variant gets its own interleave instruction sequence with
# no runtime branch in the inner loop; the host selects the matching executable at
# Create time.
set(BLOCK_SIZES 2 3 4)

foreach(blk_sz ${BLOCK_SIZES})
    pva_solutions_device(pva_depthtospace_rdf_b${blk_sz}_dev
            depthtospace_rdf_top.c
            depthtospace_rdf_knl.c
            CFLAGS
                -DBLK_SZ=${blk_sz}
            CFLAGS_DEVICE
                -DBLK_SZ=${blk_sz})
endforeach()
