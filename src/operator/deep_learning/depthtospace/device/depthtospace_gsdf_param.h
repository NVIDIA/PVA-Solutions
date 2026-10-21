/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DEPTHTOSPACE_GSDF_PARAM_H
#define DEPTHTOSPACE_GSDF_PARAM_H

#ifdef __cplusplus
extern "C" {
#endif

#define SCRATCH_SIZE (120 * 1024)

typedef struct
{
    /// Which VMEM the mapping table belongs to. Value 0,1 and 2 represents VMEM A,B and C respectively.
    int tbl_id;
    /// The offset of VMEM for the mapping table.
    int tbl_ofst;
    /// The mapping table's size in bytes unit.
    int tbl_sz;
    /// Which VMEM the DLUT input belongs to. Value 0,1 and 2 represents VMEM A,B and C respectively.
    int dut_in_id;
    /// The offset of VMEM for the DLUT input.
    int dut_in_ofst;
    /// The DLUT input's size in bytes unit.
    int dut_in_sz;
    /// Which VMEM the DLUT output belongs to. Value 0,1 and 2 represents VMEM A,B and C respectively.
    int dut_out_id;
    /// The offset of VMEM for the DLUT output.
    int dut_out_ofst;
    /// The DLUT output's size in bytes unit.
    int dut_out_sz;
} depth_to_space_buffer_info_t;

typedef struct
{
    /// Tile width.
    int tile_x;
    /// Tile height.
    int tile_y;
    /// Block size.
    int blk_sz;
    /// Number of tiles.
    int num_tiles;
    /// Input width.
    int iw;
    /// Input height.
    int ih;
    /// Input channels.
    int ic;
    /// Output width.
    int ow;
    /// Output height.
    int oh;
    /// Output channels.
    int oc;
    /// Number of GSDF handlers per input (default 1).
    int numGsdfHdl;
    /// Distance between two consecutive input rows, in elements. Equals iw for a tightly packed row.
    int in_row_stride;
    /// Distance between two consecutive input channels, in elements. Equals ih * in_row_stride when
    /// the channels are contiguous.
    int in_chan_stride;
} depth_to_space_host2dev_param_t;

#ifdef __cplusplus
}
#endif

#endif