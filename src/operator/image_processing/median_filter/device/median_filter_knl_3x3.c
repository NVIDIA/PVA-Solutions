/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "median_filter_knl_3x3.h"

/*
* VPU kernel to compute 3x3 Median Filter -
*
* To perform 3x3 Median filter we divide the kernel into two parts:
*
* In 1st Pass, we compute the min, mid and max of every 3 consecutive
* elements in horizontal direction and store them in 3 intermediate
* buffers (tile_min, tile_mid, tild_max).
*
* In 2nd Pass, we load 3 consecutive elements in vertical direction
* from each of the intermediate tiles and compute
* min of the elements from tile_max,
* max of the elements from tile_min, and
* mid of the elements from tile_mid.
*
* Next we take the median of these elements to find the
* overall median of the 3x3 kernel window.
*/

void sort_horz3_exec(MedianFilterConfig *cfg)
{
    agen_B a_in = init_agen_B_from_cfg(cfg->_in_agen_cfg);

    agen_A a_store_tmin = init_agen_A_from_cfg(cfg->_tmp_st_agen_cfg[0]);
    agen_B a_store_tmid = init_agen_B_from_cfg(cfg->_tmp_st_agen_cfg[1]);
    agen_C a_store_tmax = init_agen_C_from_cfg(cfg->_tmp_st_agen_cfg[2]);

    int niter = max(cfg->_sort_horz_niter, 12);
    niter     = (niter + 3) / 4 * 4;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        // find min, max and mid in horizontal direction
        DVECX_TYPE dv_min, dv_mid, dv_max;

#ifdef INTERLEAVED2C
        // using double vector loads we load 3 interleaved input pixels horizontally
        // to compute min, mid, max for 1 interleaved output pixel per loop.
        //   pixel index:     0   |   1   |   2
        // First channel:   v0.lo | v1.lo | v2.lo
        // Second channel:  v0.hi | v1.hi | v2.hi
        DVECX_TYPE dv0 = DVLOAD_DI(a_in);
        DVECX_TYPE dv1 = DVLOAD_DI(a_in);
        DVECX_TYPE dv2 = DVLOAD_DI(a_in);

        dv_min.lo = vmin3(dv0.lo, dv1.lo, dv2.lo);
        dv_mid.lo = vmid3(dv0.lo, dv1.lo, dv2.lo);
        dv_max.lo = vmax3(dv0.lo, dv1.lo, dv2.lo);

        dv_min.hi = vmin3(dv0.hi, dv1.hi, dv2.hi);
        dv_mid.hi = vmid3(dv0.hi, dv1.hi, dv2.hi);
        dv_max.hi = vmax3(dv0.hi, dv1.hi, dv2.hi);
#else
        // using double vector loads we load 4 input pixels horizontally
        // to compute 2 output pixels per loop.
        // The odd pixels are present in v0.hi and v1.hi
        //           index:         0   |   1   |   2   |   3
        // First set of pixels:   v0.lo | v0.hi | v1.lo |   x
        // Second set of pixels :   x   | v0.hi | v1.lo | v1.hi
        DVECX_TYPE dv0 = DVLOAD_DI(a_in);
        DVECX_TYPE dv1 = DVLOAD_DI(a_in);

        dv_min.lo = vmin3(dv0.lo, dv0.hi, dv1.lo);
        dv_mid.lo = vmid3(dv0.lo, dv0.hi, dv1.lo);
        dv_max.lo = vmax3(dv0.lo, dv0.hi, dv1.lo);

        dv_min.hi = vmin3(dv0.hi, dv1.lo, dv1.hi);
        dv_mid.hi = vmid3(dv0.hi, dv1.lo, dv1.hi);
        dv_max.hi = vmax3(dv0.hi, dv1.lo, dv1.hi);
#endif

        vstore_i(dv_min, a_store_tmin);
        vstore_i(dv_mid, a_store_tmid);
        vstore_i(dv_max, a_store_tmax);
    }
}

void find_median_3x3_exec(MedianFilterConfig *cfg)
{

    // Perform 2nd Pass to process Intermediate Min, Mid and Max Tiles and compute the Median.
    agen_A a_load_tmin = init_agen_A_from_cfg(cfg->_tmp_ld_agen_cfg[0]);
    agen_B a_load_tmid = init_agen_B_from_cfg(cfg->_tmp_ld_agen_cfg[1]);
    agen_C a_load_tmax = init_agen_C_from_cfg(cfg->_tmp_ld_agen_cfg[2]);

    agen_C a_out = init_agen_C_from_cfg(cfg->_out_agen_cfg);

    int niter = max(cfg->_merge_sorted_rows_niter, 6);
    niter     = (niter + 1) / 2 * 2;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        // load 3 elements vertically from intermediate min tile
        DVECX_TYPE dv_min0 = DVLOAD(a_load_tmin);
        DVECX_TYPE dv_min1 = DVLOAD(a_load_tmin);
        DVECX_TYPE dv_min2 = DVLOAD(a_load_tmin);

        // load 3 elements vertically from intermediate mid tile
        DVECX_TYPE dv_mid0 = DVLOAD(a_load_tmid);
        DVECX_TYPE dv_mid1 = DVLOAD(a_load_tmid);
        DVECX_TYPE dv_mid2 = DVLOAD(a_load_tmid);

        // load 3 elements vertically from intermediate max tile
        DVECX_TYPE dv_max0 = DVLOAD(a_load_tmax);
        DVECX_TYPE dv_max1 = DVLOAD(a_load_tmax);
        DVECX_TYPE dv_max2 = DVLOAD(a_load_tmax);

        // compute min of max
        DVECX_TYPE dv0 = dvmin3(dv_max0, dv_max1, dv_max2);
        // compute mid of mid
        DVECX_TYPE dv1 = dvmid3(dv_mid0, dv_mid1, dv_mid2);
        // compute max of min
        DVECX_TYPE dv2 = dvmax3(dv_min0, dv_min1, dv_min2);

        // find mid of all 3 elements
        DVECX_TYPE dv_median = dvmid3(dv0, dv1, dv2);

        vstore(dv_median, a_out);
    }
}

void median_filter_3x3_exec(MedianFilterConfig *cfg)
{
    // Perform 1st Pass to populate Intermediate Min, Mid and Max Tiles
    sort_horz3_exec(cfg);

    // Perform 2nd Pass to process Intermediate Min, Mid and Max Tiles and compute the Median.
    find_median_3x3_exec(cfg);
}
