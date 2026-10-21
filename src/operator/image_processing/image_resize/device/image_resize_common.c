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

#include "image_resize_common.h"

void fixMultiChannelPadding(uint8_t *src, int32_t fixColPixel, int32_t refColPixel, uint8_t channelNum,
                            uint16_t srcTileHeight, uint16_t srcTileLp, int32_t leftFlag)
{
    int vecw    = pva_elementsof(vcharx);
    int niter_h = (srcTileHeight + vecw - 1) / vecw;

    AgenWrapper w0;
    w0.size = sizeof(uint8_t);
    w0.n1   = niter_h;
    w0.s1   = srcTileLp * vecw;
    w0.n2   = channelNum - 1;
    w0.s2   = 1;

    int niter = w0.n1 * w0.n2;

    // the left border: fix from the second channel to the last channel
    // the right border: fix from the first channel to the second last channel
    // init agen for the reference pixel
    agen a_ref = init((vuchar *)(src + refColPixel * channelNum + leftFlag));
    INIT_AGEN2(a_ref, w0);
    int srcLaneOfst = (srcTileLp - 2) / 64;
    a_ref.lane_ofst = srcLaneOfst;

    // init agen for the pixel to be fixed
    agen a_fix = init((vuchar *)(src + fixColPixel * channelNum + leftFlag));
    INIT_AGEN2(a_fix, w0);
    a_fix.lane_ofst = srcLaneOfst;

    // compute predicate for the last row block
    int r         = srcTileHeight - srcTileHeight / vecw * vecw;
    int last_pred = (r == 0) ? -1 : (1 << r) - 1;

    int row_cnt      = 0;
    int last_row_cnt = niter_h - 1;

    niter = max(niter, 24);
    niter = (niter + 7) / 8 * 8;

    chess_separator_scheduler();

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        vcharx vdata = vuchar_load_transp(a_ref);

        // row_cnt == 0 means the last row block
        row_cnt  = mod_inc(row_cnt, last_row_cnt);
        int pred = mux(row_cnt, -1, last_pred);

        vstore_transp(vdata, a_fix, pred);
    }
}
