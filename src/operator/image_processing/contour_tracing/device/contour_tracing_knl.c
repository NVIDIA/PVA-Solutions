/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "contour_tracing_knl.h"

#include <cupva_device.h>

void fill_zero_init(AgenCFG *cfgs, uint8_t *evenBuf, uint8_t *oddBuf)
{
    AgenWrapper w0;
    w0.size = sizeof(uint8_t);
    w0.n1   = (CONV_BUF_SIZE + 63) / 64;
    w0.s1   = 64;

    agen_B a_even = init_B((void *)evenBuf);
    agen_A a_odd  = init_A((void *)oddBuf);

    INIT_AGEN1(a_even, w0);
    INIT_AGEN1(a_odd, w0);

    cfgs[FILL_ZERO_EVEN] = extract_agen_B_cfg(a_even);
    cfgs[FILL_ZERO_ODD]  = extract_agen_A_cfg(a_odd);
}

void fill_zero_exec(AgenCFG *cfgs, BoundingRect *roi)
{
    agen_B a_even = init_agen_B_from_cfg(cfgs[FILL_ZERO_EVEN]);
    agen_A a_odd  = init_agen_A_from_cfg(cfgs[FILL_ZERO_ODD]);

    int32_t width  = roi->width + 2 * CONV_KWR;
    int32_t height = roi->height / 2 + CONV_KWR;
    int32_t niter  = (width * height + 63) / 64;

    dvcharx vzero = chess_dont_care(dvcharx) & 0;

    for (int32_t i = 0; i < niter; i++)
    {
        vstore(vzero, a_even);
        vstore(vzero, a_odd);
    }
}

void split_even_odd_exec(uint8_t *input, uint8_t *evenBuf, uint8_t *oddBuf, BoundingRect *roi)
{
    int32_t roiWidth  = roi->width;
    int32_t roiHeight = roi->height;
    int32_t linePitch = roiWidth + 2 * CONV_KWR;

    int32_t niter1    = roiWidth / 64;
    int32_t niter2    = roiHeight / 2;
    int32_t copyNiter = niter1 * niter2;
    int32_t copyTail  = roiWidth - niter1 * 64;

    if (copyNiter > 0)
    {
        AgenWrapper w_in;
        w_in.size = sizeof(uint8_t);
        w_in.n1   = 2;
        w_in.s1   = roiWidth;
        w_in.n2   = niter1;
        w_in.s2   = 64;
        w_in.n3   = niter2;
        w_in.s3   = 2 * roiWidth;

        AgenWrapper w_out;
        w_out.size = sizeof(uint8_t);
        w_out.n1   = niter1 * 2;
        w_out.s1   = 32;
        w_out.n2   = niter2;
        w_out.s2   = linePitch;

        agen_B a_in   = init_B((void *)input);
        agen_B a_even = init_B((void *)(evenBuf + CONV_KWR));
        agen_A a_odd  = init_A((void *)(oddBuf + linePitch + CONV_KWR));

        INIT_AGEN3(a_in, w_in);
        INIT_AGEN2(a_even, w_out);
        INIT_AGEN2(a_odd, w_out);

        copyNiter = (copyNiter + 3) & ~3;
        copyNiter = (copyNiter >= 8) ? copyNiter : 8;

        // 2 loads (bank conflict), 4 stores -> 2 VLIW
        for (int32_t i = 0; i < copyNiter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            dvcharx vEven = dvuchar_load(a_in);
            dvcharx vOdd  = dvuchar_load(a_in);

            // Use single-vector stores because the destination address
            // (evenBuf + CONV_KWR) can be odd. A double vector store
            // requires the destination address to be halfword-aligned.
            vstore(vEven.lo, a_even);
            vstore(vOdd.lo, a_odd);
            vstore(vEven.hi, a_even);
            vstore(vOdd.hi, a_odd);
        }
    }

    if (copyTail > 0)
    {
        int32_t predLo = (copyTail >= 32) ? -1 : ((1U << copyTail) - 1);
        int32_t predHi = (copyTail <= 32) ? 0 : ((1U << (copyTail - 32)) - 1);

        AgenWrapper w_inT;
        w_inT.size = sizeof(uint8_t);
        w_inT.n1   = roiHeight;
        w_inT.s1   = roiWidth;

        AgenWrapper w_outT;
        w_outT.size = sizeof(uint8_t);
        w_outT.n1   = 2;
        w_outT.s1   = 32;
        w_outT.n2   = niter2;
        w_outT.s2   = linePitch;

        agen_B a_inT   = init_B((void *)(input + niter1 * 64));
        agen_B a_evenT = init_B((void *)(evenBuf + CONV_KWR + niter1 * 64));
        agen_A a_oddT  = init_A((void *)(oddBuf + linePitch + CONV_KWR + niter1 * 64));

        INIT_AGEN1(a_inT, w_inT);
        INIT_AGEN2(a_evenT, w_outT);
        INIT_AGEN2(a_oddT, w_outT);

        niter2 = (niter2 + 3) & ~3;
        niter2 = (niter2 >= 8) ? niter2 : 8;

        // 2 loads (bank conflict), 4 stores -> 2 VLIW
        for (int32_t i = 0; i < niter2; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            dvcharx vEven = dvuchar_load(a_inT);
            dvcharx vOdd  = dvuchar_load(a_inT);

            vstore(vEven.lo, a_evenT, predLo);
            vstore(vOdd.lo, a_oddT, predLo);
            vstore(vEven.hi, a_evenT, predHi);
            vstore(vOdd.hi, a_oddT, predHi);
        }
    }
}

void repack_bitmap_exec(uint8_t *buf, BoundingRect *roi)
{
    int32_t roiWidth  = roi->width;
    int32_t roiHeight = roi->height;
    int32_t linePitch = (roiWidth + 63) / 64 * 64;
    int32_t niter     = linePitch * roiHeight / 64;

    AgenWrapper w_in;
    w_in.size = sizeof(uint8_t);
    w_in.n1   = niter;
    w_in.s1   = 64;

    AgenWrapper w_out;
    w_out.size = sizeof(uint8_t);
    w_out.n1   = linePitch / 64;
    w_out.s1   = 64;
    w_out.n2   = roiHeight;
    w_out.s2   = roiWidth;

    agen_A a_src = init_A((void *)buf);
    agen_A a_dst = init_A((void *)buf);
    INIT_AGEN1(a_src, w_in);
    INIT_AGEN2(a_dst, w_out);

    niter = (niter + 3) & ~3;
    niter = (niter >= 8) ? niter : 8;

    // 1 load, 1 store -> 1 VLIW
    for (int32_t i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvcharx data = dvuchar_load(a_src);
        vstore(data, a_dst);
    }
}

void find_start_pixel_init(AgenCFG *cfgs)
{
    AgenWrapper w0;
    w0.size = sizeof(uint8_t);
    w0.n1   = MAX_RECT_WIDTH / 64;
    w0.s1   = 64;

    agen_B a_in = init_B(NULL);
    agen_A a_bm = init_A(NULL);

    INIT_AGEN1(a_in, w0);
    INIT_AGEN1(a_bm, w0);

    cfgs[FIND_START_IN] = extract_agen_B_cfg(a_in);
    cfgs[FIND_START_BM] = extract_agen_A_cfg(a_bm);
}

void find_start_pixel_exec(AgenCFG *cfgs, uint8_t *input, uint8_t *bitmap, BoundingRect *roi, int16_t *startIndices)
{
    agen_B a_in = init_agen_B_from_cfg(cfgs[FIND_START_IN]);
    agen_A a_bm = init_agen_A_from_cfg(cfgs[FIND_START_BM]);

    int32_t width  = roi->width;
    int32_t height = roi->height;

    int32_t maxValLo = 0, maxLaneLo = 0;
    int32_t maxValHi = 0, maxLaneHi = 0;
    int32_t leftIndex = -1, rightIndex = -1;
    int32_t candIndex = -1;
    dvcharx vInput, vBitmap, vCond;

    dvcharx v0x55;
    v0x55.lo = replicateb(0x55);
    v0x55.hi = v0x55.lo;

    dvshortx vSeq32;
    vSeq32.lo = vcollate_idx(replicateh(0)) + 1;
    vSeq32.hi = vSeq32.lo + 16;

    dvcharx vSeq64;
    vSeq64.lo = vdemote(vSeq32);
    vSeq64.hi = vSeq64.lo + 32;

    dvcharx vLaneId  = vSeq64 - 1;
    int32_t halfSize = width * height / 2;

    // Scan horizontal midline
    // 2 loads, 8 vector computes, 14 scalar computes
    for (int32_t iter = 0; iter < MAX_RECT_WIDTH / 64; iter++) chess_unroll_loop(*)
    {
        vInput  = dvuchar_load(a_in) & (vLaneId < width);
        vBitmap = dvuchar_load(a_bm);
        vCond   = dvCmpNE_andL((vBitmap & 0x55), v0x55, vInput); // pseudo intrinsics

        vmaxrid_s(vCond.lo, maxValLo, maxLaneLo);
        vmaxrid_s(vCond.hi, maxValHi, maxLaneHi);

        int32_t base = halfSize + iter * 64;
        candIndex    = maxValLo ? (base + maxLaneLo) : (maxValHi ? (base + 32 + maxLaneHi) : -1);
        leftIndex    = (leftIndex < 0) ? candIndex : leftIndex;

        vCond = dvmux(vCond, vSeq64, vCond);
        vmaxrid_s(vCond.lo, maxValLo, maxLaneLo);
        vmaxrid_s(vCond.hi, maxValHi, maxLaneHi);

        // Always overwrite with the latest hit so we end up with the last hit across iterations
        rightIndex = maxValHi ? (base + 32 + maxLaneHi) : (maxValLo ? (base + maxLaneLo) : rightIndex);

        vLaneId += 64;
    }

    startIndices[0] = leftIndex;
    startIndices[2] = rightIndex;

    // Scan vertical midline
    int32_t topOfst  = width / 2;
    int32_t botOfst  = (height - 1) * width + width / 2;
    int32_t topIndex = -1, botIndex = -1;

    for (int32_t y = 0; y < height && topIndex < 0; y++)
    {
        topIndex = (input[topOfst] && ((bitmap[topOfst] & 0x55) != 0x55)) ? topOfst : -1;
        topOfst += width;
    }

    for (int32_t y = 0; y < height && botIndex < 0; y++)
    {
        botIndex = (input[botOfst] && ((bitmap[botOfst] & 0x55) != 0x55)) ? botOfst : -1;
        botOfst -= width;
    }

    startIndices[1] = topIndex;
    startIndices[3] = botIndex;
}

void reformat_table_init(AgenCFG *cfgs)
{
    int32_t niter = MAX_RECT_AREA / 64;

    AgenWrapper w_in;
    w_in.size = sizeof(uint8_t);
    w_in.n1   = niter;
    w_in.s1   = 64;

    AgenWrapper w_out;
    w_out.size = sizeof(uint8_t);
    w_out.n1   = niter * 4;
    w_out.s1   = 64;

    agen a_src = init(NULL);
    agen a_dst = init(NULL);
    INIT_AGEN1(a_src, w_in);
    INIT_AGEN1(a_dst, w_out);

    cfgs[REFORMAT_SRC] = extract_agen_cfg(a_src);
    cfgs[REFORMAT_DST] = extract_agen_cfg(a_dst);
}

void reformat_table_exec(AgenCFG *cfgs, int32_t size)
{
    agen a_src = init_agen_from_cfg(cfgs[REFORMAT_SRC]);
    agen a_dst = init_agen_from_cfg(cfgs[REFORMAT_DST]);

    int32_t niter = (size + 63) / 64;

    dvshortx vSeq32;
    vSeq32.lo = vcollate_idx(replicateh(0));
    vSeq32.hi = vSeq32.lo + 16;

    vcharx vPerm0 = vdemote(vSeq32) & 0xF; // [0..15, 0..15]
    vcharx vPerm1 = vPerm0 | 0x10;         // [16..31, 16..31]

    niter = (niter + 1) & ~1;
    niter = (niter >= 6) ? niter : 6;

    // 1 load, 8 computes, 4 stores (bank conflicts) -> 4 VLIW
    for (int32_t i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvcharx vData = dvuchar_load(a_src);

        vcharx v0, v1, v2, v3;
        v0 = vpermute(vData.lo, vPerm0); // [0..15]
        v1 = vpermute(vData.lo, vPerm1); // [16..31]
        v2 = vpermute(vData.hi, vPerm0); // [32..47]
        v3 = vpermute(vData.hi, vPerm1); // [48..63]

        dvcharx vOut0, vOut1, vOut2, vOut3;
        dvmov(v0, v0, vOut0.lo, vOut0.hi);
        dvmov(v1, v1, vOut1.lo, vOut1.hi);
        dvmov(v2, v2, vOut2.lo, vOut2.hi);
        dvmov(v3, v3, vOut3.lo, vOut3.hi);

        vstore(vOut0, a_dst);
        vstore(vOut1, a_dst);
        vstore(vOut2, a_dst);
        vstore(vOut3, a_dst);
    }
}

void trace_contour_init(AgenCFG *cfgs)
{
    AgenWrapper w_out;
    w_out.size = sizeof(uint16_t);
    w_out.n1   = MAX_RECT_AREA / 4;
    w_out.s1   = 2;

    agen_C a_seg0 = init_C(NULL);
    agen_C a_seg1 = init_C(NULL);
    agen_B a_seg2 = init_B(NULL);
    agen_B a_seg3 = init_B(NULL);

    INIT_AGEN1(a_seg0, w_out);
    INIT_AGEN1(a_seg1, w_out);
    INIT_AGEN1(a_seg2, w_out);
    INIT_AGEN1(a_seg3, w_out);

    cfgs[TRACE_SEG0] = extract_agen_C_cfg(a_seg0);
    cfgs[TRACE_SEG1] = extract_agen_C_cfg(a_seg1);
    cfgs[TRACE_SEG2] = extract_agen_B_cfg(a_seg2);
    cfgs[TRACE_SEG3] = extract_agen_B_cfg(a_seg3);
}

int32_t trace_contour_exec(AgenCFG *cfgs, uint8_t *bitmap4Way, uint16_t *dirTable4Way, BoundingRect *roi,
                           int16_t *startIndices, uint16_t *segLens)
{
    agen_C a_seg0 = init_agen_C_from_cfg(cfgs[TRACE_SEG0]);
    agen_C a_seg1 = init_agen_C_from_cfg(cfgs[TRACE_SEG1]);
    agen_B a_seg2 = init_agen_B_from_cfg(cfgs[TRACE_SEG2]);
    agen_B a_seg3 = init_agen_B_from_cfg(cfgs[TRACE_SEG3]);
    int32_t width = roi->width;

    vshortx vOffsetX = sign_extend(*(vshort *)X_OFFSETS);
    vshortx vOffsetY = sign_extend(*(vshort *)Y_OFFSETS);
    vshortx vOffsetP = vmaddh(vOffsetY, width, vOffsetX, VPU_ROUND_0);

    int16_t coords[16];
    *((vshort *)coords) = extract(replicateh(0));

    dvshortx vCoords; // .lo = x, .hi = y
    for (int32_t i = 0; i < 4; i++) coords[i] = startIndices[i] % width;
    vCoords.lo = sign_extend(*(vshort *)coords);
    for (int32_t i = 0; i < 4; i++) coords[i] = startIndices[i] / width;
    vCoords.hi = sign_extend(*(vshort *)coords);

    // Build current position vP and target vT
    // seg0 (left -> top), seg1 (top -> right), seg2 (right -> bot), seg3 (bot -> left)
    vshortx vP, vT, vPerm;
    vP    = vmaddh(vCoords.hi, width, vCoords.lo, VPU_ROUND_0);
    vPerm = (vcollate_idx(replicateh(0)) + 1) & 0x3; // [1, 2, 3, 0]
    vT    = vpermute(vP, vPerm);

    vshortx vCond = vCmpNE_andL(vP, vT, vbitunpackh(0xF)); // vCond[0..3] = (vP != vT)
    vshortx vLen  = replicateh(0);

    // Shift LUT: next direction -> shift amount for extracting the next direction
    vshortx vShiftLut = sign_extend(*(vshort *)SHIFT_LUT);
    vshortx vShift    = replicateh(0);

    vCoords.lo += roi->x;
    vCoords.hi += roi->y;

    // 2 lookups (M0 only), 13 computes, 4 stores
    while (vorr_s(vCond)) chess_prepare_for_pipelining
    {
        vstore_i(vCoords, a_seg0, 0x03);
        vstore_i(vCoords, a_seg1, 0x0C);
        vstore_i(vCoords, a_seg2, 0x30);
        vstore_i(vCoords, a_seg3, 0xC0);
        vLen += vCond;

        // 4-way bitmap lookup: byte table, halfword index
        vcharx vBitmap = vlookup_4hbu(bitmap4Way, vP);

        // 4-way dirTable lookup: halfword table, halfword index
        vshortx vBm16  = vpromote(vBitmap).lo;
        vshortx vEntry = vlookup_4hu(dirTable4Way, vBm16);

        // Extract next direction
        vshortx vNextDir = vextract_bits(vEntry, vShift, 0x0300);

        // Update state
        vP += vpermute(vOffsetP, vNextDir);
        vCoords.lo += vpermute(vOffsetX, vNextDir);
        vCoords.hi += vpermute(vOffsetY, vNextDir);
        vShift = vpermute(vShiftLut, vNextDir);

        // Once a lane meets its target it stays off
        vCond = vCmpNE_andL(vP, vT, vCond);
    }

    *((vshort *)segLens) = extract(vLen);
    return vsumr_s(vLen);
}

void memcpy_u16_init(AgenCFG *cfgs)
{
    AgenWrapper w;
    w.size = sizeof(uint16_t);
    w.n1   = MAX_RECT_AREA / 32;
    w.s1   = 32;

    agen a_src = init(NULL);
    agen a_dst = init(NULL);
    INIT_AGEN1(a_src, w);
    INIT_AGEN1(a_dst, w);

    cfgs[MEMCPY_SRC] = extract_agen_cfg(a_src);
    cfgs[MEMCPY_DST] = extract_agen_cfg(a_dst);
}

void memcpy_u16_exec(AgenCFG *cfgs, uint16_t length)
{
    if (length == 0)
        return;

    agen a_src = init_agen_from_cfg(cfgs[MEMCPY_SRC]);
    agen a_dst = init_agen_from_cfg(cfgs[MEMCPY_DST]);

    int32_t niter = (length + 31) / 32;

    niter = (niter + 3) & ~3;
    niter = (niter >= 8) ? niter : 8;

    // 1 load, 1 store -> 1 VLIW
    for (int32_t i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvshortx data = dvushort_load(a_src);
        vstore(data, a_dst);
    }
}
