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

#include "minmaxloc_opt_interface.h"
#include "minmaxloc_param.h"

// src RDF handler and buffer
VMEM_RDF_UNIFIED(A, srcRDFHdl);
VMEM(A, STYPE, srcBuf, RDF_DOUBLE(STYPE, MM_TILE_WIDTH, MM_TILE_HEIGHT));

// SQDF: dst values and counts
VMEM_SEQDF_HANDLER(A, dstValCntSQDFHdl);
VMEM(A, STYPE, dstMinValBuf, 1);
VMEM(A, STYPE, dstMaxValBuf, 1);
VMEM(A, int, dstMinCntBuf, 1);
VMEM(A, int, dstMaxCntBuf, 1);

// SQDF: dst locations
VMEM_SEQDF_HANDLER(A, dstMinLocSQDFHdl);
VMEM_SEQDF_HANDLER(A, dstMaxLocSQDFHdl);
// external memory pointer
VMEM_POINTER(A, dstMinLocExtPtr);
VMEM_POINTER(A, dstMaxLocExtPtr);

// ping-pong write buffer for XY-interleaved coordinates, thus of size 4 times of the number of locations
VMEM(B, unsigned short, dstMinLocBuf, 4 * USHORT_LOC_BUF_SIZE);
VMEM(C, unsigned short, dstMaxLocBuf, 4 * USHORT_LOC_BUF_SIZE);

VMEM(A, int, loc_capacity);
VMEM(A, int, ntile_w);
VMEM(A, int, ntile_h);
VMEM(A, int, has_last_column);
VMEM(A, int, has_last_row);
VMEM(A, int, last_row_loc_no_unroll);
VMEM(A, unsigned short, last_row_tile_height);

VMEM(A, int, pos_tile_size);
VMEM(A, unsigned short, tile_width);
VMEM(A, unsigned short, tile_height);
VMEM(A, unsigned short, image_width);
VMEM(A, MINMAX_VAL_CFG, cfg_val);
VMEM(A, MINMAX_LOC_CFG, cfg_loc);
VMEM(A, MINMAX_VAL_CFG, cfg_val_last_row);
VMEM(A, MINMAX_LOC_CFG, cfg_loc_last_row);

VMEM(A, unsigned short, p_lane_id, 32);
VMEM(A, unsigned short, p_min_loc_x, USHORT_LOC_BUF_SIZE);
VMEM(B, unsigned short, p_min_loc_y, USHORT_LOC_BUF_SIZE);
VMEM(A, unsigned short, p_max_loc_x, USHORT_LOC_BUF_SIZE);
VMEM(C, unsigned short, p_max_loc_y, USHORT_LOC_BUF_SIZE);
// calculate min/max values for F32 requires auxiliary buffers
#if defined(DATA_TYPE_F32) && (DATA_TYPE_F32 == 1)
VMEM(A, float, p_float_dvmin, pva_elementsof(dvfloatx));
VMEM(A, float, p_float_dvmax, pva_elementsof(dvfloatx));
#endif

CUPVA_VPU_MAIN()
{
    // open SQDF write data flows
    cupvaSQDFOpen(dstMinLocSQDFHdl);
    cupvaSQDFOpen(dstMaxLocSQDFHdl);

    cupvaSQDFOpen(dstValCntSQDFHdl);

    // init minmax value and location configs
    minmax_val_cfg_init(srcBuf, tile_width, tile_height, tile_width, cfg_val);
    for (int i = 0; i < 32; i++)
    {
        p_lane_id[i] = i;
    }
    // [Note] unsigned data types share location functions with signed data types, since they are of the same data size
    minmax_loc_cfg_init((STYPE_LOC *)srcBuf, tile_width, tile_height, tile_width, p_lane_id, cfg_loc);
    // init configs with different tile height for the last row
    if (has_last_row)
    {
        minmax_val_cfg_init(srcBuf, tile_width, last_row_tile_height, tile_width, cfg_val_last_row);
        minmax_loc_cfg_init((STYPE_LOC *)srcBuf, tile_width, last_row_tile_height, tile_width, p_lane_id,
                            cfg_loc_last_row);
    }

    STYPE glb_min_val, glb_max_val;
    STYPE tile_min_val, tile_max_val;

    int glb_min_cnt       = 0;
    int glb_max_cnt       = 0;
    int glb_write_min_cnt = 0;
    int glb_write_max_cnt = 0;
    int buf_min_start = 0, buf_min_end = 0;
    int buf_max_start = 0, buf_max_end = 0;

    int min_write_iter = 0, max_write_iter = 0;

    // open RDF read data flow
    cupvaRasterDataFlowOpen(srcRDFHdl, &srcBuf[0]);

    int min_pingPong = 0, max_pingPong = 0;
    unsigned short *vmem_min_addr, *vmem_max_addr;

    unsigned short tile_offset_y = 0;
    for (int i = 0; i < ntile_h; i++)
    {
        unsigned short tile_offset_x = 0;

        for (int j = 0; j < ntile_w; j++)
        {
            STYPE *src = (STYPE *)cupvaRasterDataFlowAcquire(srcRDFHdl);

            bool in_last_row = has_last_row && i == ntile_h - 1;
            // determine the config for current tile
            MINMAX_VAL_CFG &cfg_val_ij = in_last_row ? cfg_val_last_row : cfg_val;
            MINMAX_LOC_CFG &cfg_loc_ij = in_last_row ? cfg_loc_last_row : cfg_loc;
            cupvaModifyAgenCfgBase(&cfg_val_ij._src_agen_cfg, src);
            cupvaModifyAgenCfgBase(&cfg_loc_ij._src_agen_cfg, src);

            // init global min/max values by the first element of src
            if (i == 0 && j == 0)
            {
                glb_min_val = src[0];
                glb_max_val = src[0];
            }
            // reset tmp min/max values to src tile start
            tile_min_val = src[0];
            tile_max_val = src[0];

            // find the local min/max values in the src tile
#if defined(DATA_TYPE_F32) && (DATA_TYPE_F32 == 1)
            find_minmax_values_float(cfg_val_ij, &tile_min_val, &tile_max_val, p_float_dvmin, p_float_dvmax);
#else
            FIND_MINMAX_VALUES(cfg_val_ij, &tile_min_val, &tile_max_val);
#endif
            // If the local min/max value is worse than the temporary global min/max value, there is no need to calculate its locations.
            bool update_min = tile_min_val <= glb_min_val;
            bool update_max = tile_max_val >= glb_max_val;

            int tile_min_cnt = 0, tile_max_cnt = 0;
            // Find min/max locations: check if the tile is in the last column, or the last row and requires no unroll, and will use special kernels for these corner cases.
            if (update_min && update_max)
            {
                // If both local min and max values are no worse than temporary global values, calculate locations for both min and max.
                find_minmax_locations_check_last(
                    cfg_loc_ij, *(STYPE_LOC *)&tile_min_val, *(STYPE_LOC *)&tile_max_val, tile_min_cnt, tile_max_cnt,
                    &p_min_loc_x[buf_min_end], &p_min_loc_y[buf_min_end], &p_max_loc_x[buf_max_end],
                    &p_max_loc_y[buf_max_end], tile_offset_x, tile_offset_y, image_width, ntile_w, j, has_last_column);
            }
            else if (update_min)
            {
                // if only the min locations requires update
                tile_min_cnt = find_value_locations_check_last(cfg_loc_ij, *(STYPE_LOC *)&tile_min_val,
                                                               &p_min_loc_x[buf_min_end], &p_min_loc_y[buf_min_end],
                                                               tile_offset_x, tile_offset_y, image_width, ntile_w,
                                                               ntile_h, i, j, has_last_column, last_row_loc_no_unroll);
            }
            else if (update_max)
            {
                // if only the max locations requires update
                tile_max_cnt = find_value_locations_check_last(cfg_loc_ij, *(STYPE_LOC *)&tile_max_val,
                                                               &p_max_loc_x[buf_max_end], &p_max_loc_y[buf_max_end],
                                                               tile_offset_x, tile_offset_y, image_width, ntile_w,
                                                               ntile_h, i, j, has_last_column, last_row_loc_no_unroll);
            }

            cupvaRasterDataFlowRelease(srcRDFHdl);

            // update min locations
            if (update_min)
            {
                if (tile_min_val < glb_min_val)
                {
                    // if new min value occurs, clear previous temporary locations
                    glb_min_val       = tile_min_val;
                    glb_min_cnt       = tile_min_cnt;
                    glb_write_min_cnt = 0;
                    buf_min_start     = buf_min_end;
                    buf_min_end += tile_min_cnt;
                }
                else
                {
                    // else if local min value equals the temporary global min, append locations to the end (when the number of locations already written out does not reach the capacity limit)
                    glb_min_cnt += tile_min_cnt;
                    if (glb_write_min_cnt < loc_capacity)
                    {
                        buf_min_end += tile_min_cnt;
                    }
                }

                // trigger write if the location buffer does not have enough space for the next tile
                if (USHORT_LOC_BUF_SIZE - buf_min_end < MM_TILE_SIZE)
                {

                    write_minmax_buf(min_write_iter, dstMinLocSQDFHdl, dstMinLocExtPtr, min_pingPong, dstMinLocBuf,
                                     buf_min_start, buf_min_end, glb_write_min_cnt, &p_min_loc_x[buf_min_start],
                                     &p_min_loc_y[buf_min_start], pos_tile_size, loc_capacity);
                }
            }
            // update max locations
            if (update_max)
            {
                if (tile_max_val > glb_max_val)
                {
                    // if new max value occurs, clear previous temporary locations
                    glb_max_val       = tile_max_val;
                    glb_max_cnt       = tile_max_cnt;
                    glb_write_max_cnt = 0;
                    buf_max_start     = buf_max_end;
                    buf_max_end += tile_max_cnt;
                }
                else
                {
                    // else if local max value equals the temporary global max, append locations to the end (when the number of locations already written out does not reach the capacity limit)
                    glb_max_cnt += tile_max_cnt;
                    if (glb_write_max_cnt < loc_capacity)
                    {
                        buf_max_end += tile_max_cnt;
                    }
                }
                // trigger write if the location buffer does not have enough space for the next tile
                if (USHORT_LOC_BUF_SIZE - buf_max_end < MM_TILE_SIZE)
                {

                    write_minmax_buf(max_write_iter, dstMaxLocSQDFHdl, dstMaxLocExtPtr, max_pingPong, dstMaxLocBuf,
                                     buf_max_start, buf_max_end, glb_write_max_cnt, &p_max_loc_x[buf_max_start],
                                     &p_max_loc_y[buf_max_start], pos_tile_size, loc_capacity);
                }
            }

            tile_offset_x += tile_width;
        }
        tile_offset_y += tile_height;
    }

    // write rest data in location buffers (when the number of locations already written out does not reach the capacity limit)
    if ((glb_write_min_cnt < loc_capacity) && (buf_min_end > buf_min_start))
    {

        write_minmax_buf(min_write_iter, dstMinLocSQDFHdl, dstMinLocExtPtr, min_pingPong, dstMinLocBuf, buf_min_start,
                         buf_min_end, glb_write_min_cnt, &p_min_loc_x[buf_min_start], &p_min_loc_y[buf_min_start],
                         pos_tile_size, loc_capacity);
    }

    if ((glb_write_max_cnt < loc_capacity) && (buf_max_end > buf_max_start))
    {

        write_minmax_buf(max_write_iter, dstMaxLocSQDFHdl, dstMaxLocExtPtr, max_pingPong, dstMaxLocBuf, buf_max_start,
                         buf_max_end, glb_write_max_cnt, &p_max_loc_x[buf_max_start], &p_max_loc_y[buf_max_start],
                         pos_tile_size, loc_capacity);
    }

    cupvaRasterDataFlowClose(srcRDFHdl);

    cupvaSQDFSync(dstMinLocSQDFHdl);
    cupvaSQDFClose(dstMinLocSQDFHdl);

    cupvaSQDFSync(dstMaxLocSQDFHdl);
    cupvaSQDFClose(dstMaxLocSQDFHdl);

    // SQDF to write min/max values and counts
    dstMinValBuf[0] = glb_min_val;
    dstMaxValBuf[0] = glb_max_val;
    dstMinCntBuf[0] = glb_min_cnt;
    dstMaxCntBuf[0] = glb_max_cnt;

    cupvaSQDFTrig(dstValCntSQDFHdl);
    cupvaSQDFSync(dstValCntSQDFHdl);
    cupvaSQDFClose(dstValCntSQDFHdl);

    return 0;
}
