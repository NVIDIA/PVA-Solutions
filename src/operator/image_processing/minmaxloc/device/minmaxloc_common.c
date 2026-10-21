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

#include "minmaxloc_common.h"

// SQDF write from VMEM to DRAM
void write_minmax_buf(int &write_iter, SequenceDataFlowHandler &sqdf_hdl, ExtMemPointer &extLocPtr, int &pingPong,
                      unsigned short *LocBuf, int &buf_start, int &buf_end, int &glb_write_cnt, unsigned short *p_loc_x,
                      unsigned short *p_loc_y, int pos_tile_size, int loc_capacity)
{
    if (write_iter > 0)
    {
        cupvaSQDFSync(sqdf_hdl);
    }

    int buf_cnt   = buf_end - buf_start;
    int write_cnt = (buf_cnt > loc_capacity - glb_write_cnt) ? loc_capacity - glb_write_cnt : buf_cnt;

    unsigned short *vmem_addr = pingPong ? (LocBuf + pos_tile_size) : LocBuf;
    pingPong ^= 1;

    interleave_loc_xy_vshort(&p_loc_x[0], &p_loc_y[0], write_cnt, vmem_addr);

    // update vmem and dram address
    uint64_t dram_addr = extLocPtr.base + extLocPtr.offset + 2 * glb_write_cnt * sizeof(uint16_t);
    cupvaSQDFUpdateAddr(sqdf_hdl, 0, cupvaGetVmemAddress((void *)vmem_addr), MEMTYPE_VMEM, dram_addr, MEMTYPE_DRAM);

    // update current tile size
    uint16_t lx = 2 * write_cnt * sizeof(uint16_t), ly = 1;
    cupvaSQDFUpdateTileSize(sqdf_hdl, 0, lx, ly);

    cupvaSQDFFlushAndTrig(sqdf_hdl);

    write_iter++;
    glb_write_cnt += write_cnt;
    // clear vmem write buffer
    buf_start = 0;
    buf_end   = 0;
}
