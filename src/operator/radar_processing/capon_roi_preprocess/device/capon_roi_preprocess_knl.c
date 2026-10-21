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

/**
 * @file capon_roi_preprocess_knl.c
 * @brief Azimuth projection (dot-product path), quantize and steering gather, plus
 *        the VD build. The coarse-DoA ranking both methods share is in _rank.c, the methods in _dft.c /
 *        _fft.c, the dispatch in _top.c.
 */

#include "capon_roi_preprocess_knl.h"

#include "capon_roi_preprocess_agens.h"
#include "capon_roi_preprocess_params.h"

#define CRP_PROJ_BEAMS 4
#define CRP_PROJ_OUT(J, TR, TI)                                                               \
    do                                                                                        \
    {                                                                                         \
        const vintx t_r_  = (TR);                                                             \
        const vintx t_i_  = (TI);                                                             \
        const int32_t on_ = ((J) < nValid) ? -1 : 0;                                          \
        const int32_t el_ = numElVirt > 1; /* elevation arrays: unnarrowed sums to tmpRoi */  \
        vintx o_r_        = t_r_;                                                             \
        vintx o_i_        = t_i_;                                                             \
        if (!el_)                                                                             \
        {                                                                                     \
            o_r_ = ((t_r_ + rnd) >> rshift) << lshift;                                        \
            o_i_ = ((t_i_ + rnd) >> rshift) << lshift;                                        \
        }                                                                                     \
        vstore(o_r_, sr, on_);                                                                \
        vstore(o_i_, si, on_);                                                                \
        if (fbaSym)                                                                           \
        {                                                                                     \
            dvintx t_;                                                                        \
            t_.lo             = t_r_;                                                         \
            t_.hi             = t_i_;                                                         \
            const dvshortx h_ = dvshort_load(rtA); /* (hR | hI), built once per beam group */ \
            const dvshortx l_ = dvshort_load(rtA); /* (lR | lI) */                            \
            const vshortx hR_ = h_.lo, hI_ = h_.hi, lR_ = l_.lo, lI_ = l_.hi;                 \
            const vintx aR_  = vdotp4_whw(t_, t_, hR_, t_r_, 0);                              \
            const vintx aI_  = vdotp4_whw(t_, t_, hI_, t_i_, 0);                              \
            const vintx lRv_ = (vdotp4_whw(t_, t_, lR_, t_r_, 0) + (1 << 14)) >> 15;          \
            const vintx lIv_ = (vdotp4_whw(t_, t_, lI_, t_i_, 0) + (1 << 14)) >> 15;          \
            const vintx u_r_ = (aR_ + lRv_ + (1 << 13)) >> 14;                                \
            const vintx u_i_ = (aI_ + lIv_ + (1 << 13)) >> 14;                                \
            vintx b_r_       = u_r_;                                                          \
            vintx b_i_       = u_i_;                                                          \
            if (!el_)                                                                         \
            {                                                                                 \
                b_r_ = ((u_r_ + rnd) >> rshift) << lshift;                                    \
                b_i_ = ((u_i_ + rnd) >> rshift) << lshift;                                    \
            }                                                                                 \
            const int32_t bp_ = el_ ? on_ : (on_ & bLn); /* tmpRoi: whole blocks */           \
            vstore(b_r_, br, bp_);                                                            \
            vstore(b_i_, bi, bp_);                                                            \
        }                                                                                     \
    } while (0)
static void crpProjectAzDotP4(CaponKnl *k, int32_t roiAzStart)
{
    /* Elevation rows only in the general kernel; elsewhere numElVirt == 1 is a compile-time constant. */
    const int32_t roiAz = k->roiAz, numAzVirt = k->numAzVirt, numElVirt = CRP_PROJ_GENERAL ? k->numElVirt : 1;
    const int32_t beamAz = k->beamAz, shAz = k->shAz, H = k->H;
    const int32_t np = numAzVirt >> 1, nC = H / CRP_VECW;
    const int32_t fbaSym = k->fbaSym, numSel = k->numSel;
    const int32_t nCf = fbaSym ? (numSel + CRP_VECW - 1) / CRP_VECW : nC; /* chunks projected */
    int16_t *coef     = (int16_t *)k->coarseTmpReBuf;
    /* Per-chunk staging of the 8 sums: tmpRoi, unless elevation arrays use it for the azimuth sums. */
    int32_t *stg = (numElVirt > 1) ? (int32_t *)k->coarseTmpImBuf : (int32_t *)k->tmpRoiReBuf;

    /* Permute indices (halfwords from the pair's first coefficient): lanes 0-15 (0,1,2,3) x 4, lanes 16-31
     * (1,0,3,2) x 4. vdemote_i interleaves its two sources, so a gives the even lanes and b the odd ones. */
    const vshortx lane = vcollate_idx(replicateh(0));
    const vshortx e    = (lane & 1) * 2;
    const vshortx up   = vmux(lane >= 8, replicateh(1), replicateh(0));
    vcharx idx         = chess_dont_care(vcharx);
    vdemote_i(e + up, e + 1 - up, idx);
    const vshortx odd  = lane & 1;
    const vshortx role = lane & 3;
    const vshortx zero = replicateh(0), maxQ = replicateh(32767);
    const int32_t shift  = (k->vdQBits - 1) - k->inputQBits;
    const int32_t rshift = (shift > 0) ? shift : 0;
    const int32_t lshift = (shift < 0) ? -shift : 0;
    const int32_t rnd    = (shift > 0) ? (1 << (shift - 1)) : 0;

    if (nCf < nC && numElVirt == 1)
    {
        /* Rows past the forward and backward blocks are zero; the backward stores below overwrite the rest. */
        const int32_t nz = nC - nCf;
        agen_A zr        = initAgenA2_i((int32_t *)k->patchReBuf + nCf * CRP_VECW, nz, CRP_VECW, roiAz, H);
        agen_B zi        = initAgenB2_i((int32_t *)k->patchImBuf + nCf * CRP_VECW, nz, CRP_VECW, roiAz, H);
        const vintx z    = replicatew(0);
        for (int32_t i = 0; i < nz * roiAz; i++) chess_prepare_for_pipelining
        chess_loop_range(1, )
        {
            vstore(z, zr);
            vstore(z, zi);
        }
    }

    for (int32_t g = 0; g < roiAz; g += CRP_PROJ_BEAMS)
    {
        /* Beams g..g+3; past roiAz the last beam is repeated and its stores are disabled. */
        int32_t bj[CRP_PROJ_BEAMS];
        for (int32_t j = 0; j < CRP_PROJ_BEAMS; j++) bj[j] = (g + j < roiAz) ? (g + j) : (roiAz - 1);
        const int32_t nValid = (roiAz - g < CRP_PROJ_BEAMS) ? (roiAz - g) : CRP_PROJ_BEAMS;

        /* Rotation coefficients per beam, (hR | hI) and (lR | lI) for the CRP_PROJ_OUT backward rows. */
        dvshort *rotTab = (dvshort *)((int32_t *)k->coarseTmpImBuf + 512);
        for (int32_t j = 0; j < CRP_PROJ_BEAMS; j++)
        {
            int32_t ka = (roiAzStart + bj[j] - shAz) % beamAz;
            if (ka < 0)
                ka += beamAz;
            AgenWrapper ws = {.size = sizeof(int16_t), .n1 = (uint16_t)np, .n2 = 1, .s1 = 4, .s2 = 0};
            agen_B src     = init_B((void *)(k->waz16Buf + ka * numAzVirt * 2));
            INIT_AGEN2(src, ws);
            dvshort *dst = (dvshort *)((j < 2) ? (coef + j * np * 32) : (k->coefB + (j - 2) * np * 32));
            /* Two pairs per iteration: the load -> negate -> store chains of consecutive pairs interleave. */
            int32_t p = 0;
            for (; p + 1 < np; p += 2) chess_prepare_for_pipelining
            {
                dvshortx c0 = dvshort_load_perm(src, idx);
                dvshortx c1 = dvshort_load_perm(src, idx);
                /* -wi in the odd lanes; -(-32768) is clamped to 32767 (one Q15 LSB). */
                c0.lo      = vmux(odd != 0, vmin(zero - c0.lo, maxQ), c0.lo);
                c1.lo      = vmux(odd != 0, vmin(zero - c1.lo, maxQ), c1.lo);
                dst[p]     = extract(c0);
                dst[p + 1] = extract(c1);
            }
            if (p < np)
            {
                dvshortx c = dvshort_load_perm(src, idx);
                c.lo       = vmux(odd != 0, vmin(zero - c.lo, maxQ), c.lo);
                dst[p]     = extract(c);
            }
            const int16_t *c_ = k->fbaRot + 4 * ka;
            const int32_t hr = c_[0], hi = c_[1], lr = c_[2], li = c_[3];
            dvshortx h, l;
            h.lo              = vmux(role == 0, replicateh(hr), vmux(role == 1, replicateh(hi), zero));
            h.hi              = vmux(role == 0, replicateh(hi), vmux(role == 1, replicateh(-hr), zero));
            l.lo              = vmux(role == 0, replicateh(lr), vmux(role == 1, replicateh(li), zero));
            l.hi              = vmux(role == 0, replicateh(li), vmux(role == 1, replicateh(-lr), zero));
            rotTab[2 * j]     = extract(h);
            rotTab[2 * j + 1] = extract(l);
        }
        CRP_SEPARATOR();

        /* Elevation arrays: one pass per element row e (VD rows e*numAzVirt + a), the sums to tmpRoi [e][beam][H]. */
        for (int32_t e = 0; e < numElVirt; e++)
        {
            /* data: tap a (n1, rows 2H apart), then chunk (n2) */
            agen_C xd =
                initAgenC4_i(k->vdFxReBuf + e * numAzVirt * 2 * H, numAzVirt, 2 * H, nCf, 2 * CRP_VECW, 1, 0, 1, 0);
            /* Coefficients: one agen per bank alternating its two beams (n1), then the pair (n2), rewinding per chunk
             * (n3), so the inner loop holds three agens. */
            agen_A cA = initAgenA3_i((int32_t *)coef, 2, np * 16, np, 16, nCf, 0);
            agen_B cB = initAgenB3_i((int32_t *)k->coefB, 2, np * 16, np, 16, nCf, 0);
            /* Elevation arrays: tmpRoi rows of length Hb, forward blocks then (fbaSym) the c_ka * conj backward ones. */
            const int32_t Hb  = (numElVirt > 1) ? (fbaSym ? 2 : 1) * nCf * CRP_VECW : H; /* row pitch */
            const int32_t bOf = (numElVirt > 1) ? nCf * CRP_VECW : numSel;               /* backward block offset */
            int32_t *fr =
                (numElVirt > 1) ? (int32_t *)k->tmpRoiReBuf + (e * roiAz + g) * Hb : (int32_t *)k->patchReBuf + g * H;
            int32_t *fi =
                (numElVirt > 1) ? (int32_t *)k->tmpRoiImBuf + (e * roiAz + g) * Hb : (int32_t *)k->patchImBuf + g * H;
            for (int32_t c = 0; c < nCf; c++)
            {
                vintx r0 = replicatew(0), i0 = replicatew(0), r1 = replicatew(0), i1 = replicatew(0);
                vintx r2 = replicatew(0), i2 = replicatew(0), r3 = replicatew(0), i3 = replicatew(0);
                for (int32_t p = 0; p < np; p++) chess_prepare_for_pipelining
                {
                    const dvintx da   = dvint_load(xd);
                    const dvintx db   = dvint_load(xd);
                    const dvshortx w0 = dvshort_load(cA);
                    const dvshortx w1 = dvshort_load(cA);
                    const dvshortx w2 = dvshort_load(cB);
                    const dvshortx w3 = dvshort_load(cB);
                    r0                = vdotp4_whw(da, db, w0.lo, r0, -1);
                    i0                = vdotp4_whw(da, db, w0.hi, i0, -1);
                    r1                = vdotp4_whw(da, db, w1.lo, r1, -1);
                    i1                = vdotp4_whw(da, db, w1.hi, i1, -1);
                    r2                = vdotp4_whw(da, db, w2.lo, r2, -1);
                    i2                = vdotp4_whw(da, db, w2.hi, i2, -1);
                    r3                = vdotp4_whw(da, db, w3.lo, r3, -1);
                    i3                = vdotp4_whw(da, db, w3.hi, i3, -1);
                }
                /* Backward rows numSel + 8c .. +7, clipped at H. */
                const int32_t left = H - numSel - c * CRP_VECW;
                const int32_t bLn  = (left >= CRP_VECW) ? -1 : CRP_LOW_LANES(left);
                /* Stage the sums, rounded to 32 bits, so that the per-beam output below is one loop, which keeps the
                 * per-detection code small. */
                agen_A ts = initAgenA2_i(stg, 2 * CRP_PROJ_BEAMS, CRP_VECW, 1, 0);
                vstore((r0 + (1 << 15)) >> 16, ts);
                vstore((i0 + (1 << 15)) >> 16, ts);
                vstore((r1 + (1 << 15)) >> 16, ts);
                vstore((i1 + (1 << 15)) >> 16, ts);
                vstore((r2 + (1 << 15)) >> 16, ts);
                vstore((i2 + (1 << 15)) >> 16, ts);
                vstore((r3 + (1 << 15)) >> 16, ts);
                vstore((i3 + (1 << 15)) >> 16, ts);
                CRP_SEPARATOR();
                agen_A tl = initAgenA2_i(stg, 2 * CRP_PROJ_BEAMS, CRP_VECW, 1, 0);
                /* rotation table, beam j: (hR | hI), (lR | lI) */
                agen_A rtA = initAgenA2_i((int32_t *)rotTab, 2 * CRP_PROJ_BEAMS, 16, 1, 0);
                /* Stores of this chunk: beam j (consecutive patch rows), one row per beam; the quantize reads it for every
             * elevation slot (crpBuild_patchRdRe). */
                const int32_t co = c * CRP_VECW;
                agen_A sr        = initAgenA2_i(fr + co, CRP_PROJ_BEAMS, Hb, 1, 0);
                agen_B si        = initAgenB2_i(fi + co, CRP_PROJ_BEAMS, Hb, 1, 0);
                agen_A br        = initAgenA2_i(fr + bOf + co, CRP_PROJ_BEAMS, Hb, 1, 0);
                agen_B bi        = initAgenB2_i(fi + bOf + co, CRP_PROJ_BEAMS, Hb, 1, 0);
                for (int32_t j = 0; j < CRP_PROJ_BEAMS; j++)
                {
                    const vintx tr = vint_load(tl);
                    const vintx ti = vint_load(tl);
                    CRP_PROJ_OUT(j, tr, ti);
                }
            }
        }
    }
}

void caponProjectAzFx(CaponKnl *k, int32_t roiAzStart)
{
#if CRP_PROJ_GENERAL
    if (k->useDotProj)
        crpProjectAzDotP4(k, roiAzStart); /* elevation arrays (numElVirt == 2) */
    else
        caponProjectAzPerTapFx(k, roiAzStart);
#else
    crpProjectAzDotP4(k, roiAzStart);
#endif
}

/* Quantize: plain-load patch h-contiguously and transpose-STORE into the TRANS_MODE_2 outBuf, so the
 * DMA yields [H][svDim] without a scalar corner-turn. */
void caponQuantizeSnapshotsFxT(CaponKnl *k, int32_t *out, int32_t laneOfst, int32_t linePitchDst)
{
    const int32_t svDim = k->svDim, H = k->H;
    const int32_t nHb = H / CRP_VECW; /* h-blocks of CRP_VECW(=8) rows */

    /* Plain loads over patch[svDim][H]: hb inner (s1=CRP_VECW), c outer (s2=H). */
    agen_A rA = CRP_AG_RELOAD(k, patchRdRe);
    agen_B iA = CRP_AG_RELOAD(k, patchRdIm);

    /* Transpose store over outBuf (dst [H][svDim] complex, TRANS_MODE_2):
     * s1 = CRP_VECW*linePitchDst, s2 = 1, plus the lane offset. */
    AgenWrapper wo = {.size = sizeof(int32_t),
                      .n1   = (uint16_t)nHb,
                      .n2   = (uint16_t)svDim,
                      .s1   = CRP_VECW * linePitchDst,
                      .s2   = 2};
    agen outg      = init((void *)out);
    outg.lane_ofst = laneOfst;
    INIT_AGEN2(outg, wo);
    agen_C oA = init_agen_C_from_cfg(extract_agen_cfg(outg));

    const int32_t niter =
        (nHb * svDim + CRP_QUANT_TRANSP_UNROLL - 1) / CRP_QUANT_TRANSP_UNROLL * CRP_QUANT_TRANSP_UNROLL;
    for (int32_t i = 0; i < niter; i++) chess_loop_range(CRP_QUANT_TRANSP_UNROLL, )
    chess_unroll_loop(CRP_QUANT_TRANSP_UNROLL)
    chess_prepare_for_pipelining
    {
        vintx re = vint_load(rA);
        vintx im = vint_load(iA);
        dvintx v;
        v.lo = re;
        v.hi = im;
        vstore_transp2_i(v, oA);
    }
}

/* The VD staging block (one TX window of up to CRP_VD_STAGE_ROWS rows at CRP_VD_STAGE_PITCH) reuses coarseTmpRe/Im. */
typedef char crpVdStageFits[(BUF_SZ_COARSE_TMP >= (CRP_VD_STAGE_ROWS + 1) * CRP_VD_STAGE_PITCH) ? 1 : -1];

void caponBuildVdFx(CaponKnl *k, float *plane, int32_t detDoppler)
{
    const int32_t numTx = k->numTx, numRx = k->numRx, N = k->numChirps, numSel = k->numSel, H = k->H;
    const int32_t nvc    = k->nvcProj;
    const int32_t rowLen = 2 * numRx; /* floats per chirp: numRx complex */
    const int32_t fold   = k->vdFold;

    /* A TX's window is numSel rows starting anywhere in [0, N) and wrapping at N, so it is read in segments of
     * consecutive rows (one, unless it wraps). The plane tile may be double-buffered: never write past it. */
    const int32_t d0 = (detDoppler % N + N) % N;
    int32_t start[CRP_MAX_TX];
    for (int32_t tx = 0; tx < numTx; tx++)
    {
        int32_t s = d0 + k->vdTxBase[tx]; /* both in [0, N) */
        start[tx] = (s >= N) ? s - N : s;
    }

    float *stRe = k->coarseTmpReBuf;
    float *stIm = k->coarseTmpImBuf;
    const vintx keep =
        vmux(vcollate_idx(replicatew(0)) < replicatew(numRx), replicatew(-1), replicatew(0)); /* RX lanes */
    const vintx absMask  = replicatew(0x7fffffff);
    const vintx zeroI    = replicatew(0);
    const int32_t predRx = (numRx == CRP_VECW) ? -1 : CRP_LOW_LANES(numRx);
    vintx maxBitsV       = zeroI;
    for (int32_t tx = 0; tx < numTx; tx++)
    {
        const float wr = k->ddmaRe[tx], wi = k->ddmaIm[tx]; /* device decode (no fold) */
        /* re = pr*wr - pi*wi; im = -(pr*wi + pi*wr) = pr*(-wi) + pi*(-wr) */

        AgenWrapper ws = {.size = sizeof(float), .n1 = (uint16_t)numSel, .n2 = 1, .s1 = CRP_VD_STAGE_PITCH, .s2 = 0};
        agen_A sr      = init_A((void *)(stRe + tx * numRx));
        INIT_AGEN2(sr, ws);
        agen_A si = init_A((void *)(stIm + tx * numRx));
        INIT_AGEN2(si, ws);
        int32_t pos = start[tx];
        for (int32_t left = numSel; left > 0;)
        {
            const int32_t seg = (left < N - pos) ? left : (N - pos);
            AgenWrapper wl    = {.size = sizeof(float), .n1 = (uint16_t)seg, .n2 = 1, .s1 = rowLen, .s2 = 0};
            agen_B ld         = init_B((void *)(plane + pos * rowLen));
            INIT_AGEN2(ld, wl);
            if (fold)
            {
                /* Stage the plane as is; pass 2 applies the conjugate through the sign of its im scale. The RX lane
                 * mask is applied once after the loop (the values are non-negative). */
                const int32_t row0 = numSel - left;
                const int32_t segE = (seg + 1) & ~1;
                AgenWrapper wsE    = {.size = sizeof(float),
                                      .n1   = (uint16_t)segE,
                                      .n2   = 1,
                                      .s1   = CRP_VD_STAGE_PITCH,
                                      .s2   = 0};
                agen_A srE         = init_A((void *)(stRe + tx * numRx + row0 * CRP_VD_STAGE_PITCH));
                INIT_AGEN2(srE, wsE);
                agen_A siE = init_A((void *)(stIm + tx * numRx + row0 * CRP_VD_STAGE_PITCH));
                INIT_AGEN2(siE, wsE);
                for (int32_t r = 0; r < segE; r++) chess_prepare_for_pipelining
                chess_loop_range(2, )
                chess_unroll_loop(2)
                {
                    const dvfloatx p0 = dvfloat_load_di(ld); /* .lo = re, .hi = im of RX 0..7 */
                    vstore(p0.lo, srE, predRx);
                    vstore(p0.hi, siE, predRx);
                    maxBitsV = vmax(maxBitsV, vmax(as_vintx(p0.lo) & absMask, as_vintx(p0.hi) & absMask));
                }
                maxBitsV = maxBitsV & keep;
            }
            else
            {
                /* DDM decode on the device (no fold). Keep this inline, so that the function stays a leaf function. */
                const vfloatx vwr = replicatef(wr), vwi = replicatef(wi), vnwr = replicatef(-wr),
                              vnwi = replicatef(-wi);
                for (int32_t r = 0; r < seg; r++) chess_prepare_for_pipelining
                chess_loop_range(1, )
                {
                    const dvfloatx p = dvfloat_load_di(ld); /* .lo = re, .hi = im of RX 0..7 */
                    const vfloatx re = vsubf(vmulf(p.lo, vwr), vmulf(p.hi, vwi));
                    const vfloatx im = vaddf(vmulf(p.lo, vnwi), vmulf(p.hi, vnwr));
                    vstore(re, sr, predRx);
                    vstore(im, si, predRx);
                    const vintx m = vmax(as_vintx(re) & absMask, as_vintx(im) & absMask);
                    maxBitsV      = vmax(maxBitsV, m & keep);
                }
            }
            left -= seg;
            pos = 0;
        }
    }

    const int32_t padRows = ((numSel + CRP_VECW - 1) / CRP_VECW) * CRP_VECW - numSel;
    if (padRows > 0)
    {
        const int32_t nCol = (numTx * numRx + CRP_VECW - 1) / CRP_VECW;
        agen_A zr =
            initAgenA2_i((int32_t *)stRe + numSel * CRP_VD_STAGE_PITCH, nCol, CRP_VECW, padRows, CRP_VD_STAGE_PITCH);
        agen_A zi =
            initAgenA2_i((int32_t *)stIm + numSel * CRP_VD_STAGE_PITCH, nCol, CRP_VECW, padRows, CRP_VD_STAGE_PITCH);
        for (int32_t i = 0; i < nCol * padRows; i++) chess_prepare_for_pipelining
        {
            vstore(zeroI, zr);
            vstore(zeroI, zi);
        }
    }
    int maxBits = 0, ignored = 0;
    vmaxrid_s(maxBitsV, maxBits, ignored);

    int32_t q = 24;
    if (maxBits != 0)
    {
        q = 150 - ((maxBits >> 23) & 0xff);
        q = (q > 30) ? 30 : ((q < 0) ? 0 : q);
    }
    k->vdQBits = q;
    CRP_SEPARATOR();

    union
    {
        int32_t i;
        float f;
    } sc;
    sc.i                  = (q + 127) << 23; /* 2^q, q in [0, 30] */
    const vfloatx scale   = replicatef(sc.f);
    const vfloatx scaleIm = replicatef(fold ? -sc.f : sc.f); /* vdFold: the staged im is the plane's; conj here */
    const int32_t nPhys   = numTx * numRx;
    const int32_t nBlk    = (numSel + CRP_VECW - 1) / CRP_VECW;
    /* The dot-product projection reads the VD interleaved: [re 8 | im 8] per Doppler block; the per-tap paths read
     * a re plane and an im plane. */
    const int32_t interleaved = k->useDotProj;
    const int32_t rowPitch    = interleaved ? 2 * H : H;
    const int32_t colMul      = interleaved ? 2 : 1;
    int32_t *vdRe             = k->vdBufInt;
    int32_t *vdIm             = interleaved ? (k->vdBufInt + CRP_VECW) : (k->vdBufInt + nvc * H);
    const int32_t fba         = k->enableFBA && !k->fbaSym; /* fbaSym: the projection derives the backward rows */

    /* Stage column c = tx*numRx + rx -> vdBuf row rx*numTx + tx: rx steps numTx rows, tx one row. The mirror
     * row is nvc-1 minus that, so its agen walks the rows backwards. Outer dimension: the Doppler block. */
    AgenWrapper wt = {.size = sizeof(float),
                      .n1   = (uint16_t)nPhys,
                      .n2   = (uint16_t)nBlk,
                      .s1   = 1,
                      .s2   = CRP_VECW * CRP_VD_STAGE_PITCH};
    agen_A tr      = init_A((void *)stRe);
    tr.lane_ofst   = CRP_VD_STAGE_LANE_OFST;
    INIT_AGEN2(tr, wt);
    agen_A ti    = init_A((void *)stIm);
    ti.lane_ofst = CRP_VD_STAGE_LANE_OFST;
    INIT_AGEN2(ti, wt);
    agen_C vr = initAgenC4_i(vdRe, numRx, numTx * rowPitch, numTx, rowPitch, nBlk, colMul * CRP_VECW, 1, 0);
    agen_C vi = initAgenC4_i(vdIm, numRx, numTx * rowPitch, numTx, rowPitch, nBlk, colMul * CRP_VECW, 1, 0);
    /* Without mirror rows the mirror stores write no lane, and their agens alias the forward rows. */
    agen_C mr = fba ? initAgenC4_i(vdRe + (nvc - 1) * rowPitch + colMul * numSel, numRx, -numTx * rowPitch, numTx,
                                   -rowPitch, nBlk, colMul * CRP_VECW, 1, 0)
                    : initAgenC4_i(vdRe, numRx, numTx * rowPitch, numTx, rowPitch, nBlk, colMul * CRP_VECW, 1, 0);
    agen_C mi = fba ? initAgenC4_i(vdIm + (nvc - 1) * rowPitch + colMul * numSel, numRx, -numTx * rowPitch, numTx,
                                   -rowPitch, nBlk, colMul * CRP_VECW, 1, 0)
                    : initAgenC4_i(vdIm, numRx, numTx * rowPitch, numTx, rowPitch, nBlk, colMul * CRP_VECW, 1, 0);
    if (interleaved && !fba)
    {
        /* Interleaved layout without mirror rows: one 64 B (re 8 | im 8) store per channel and block, so the
         * loop is not bound by the bank-C store port. */
        /* Unrolled by two so the load -> scale -> round -> store chains interleave (both compilers honor the hint). */
        const int32_t nPhysE = (nPhys + 1) & ~1;
        AgenWrapper wtB      = {.size = sizeof(float), .n1 = (uint16_t)nPhys, .n2 = 1, .s1 = 1, .s2 = 0};
        for (int32_t b = 0; b < nBlk; b++)
        {
            const int32_t nb   = numSel - b * CRP_VECW; /* valid rows in this block (>= CRP_VECW but the last) */
            const int32_t low  = (nb >= CRP_VECW) ? 0xff : CRP_LOW_LANES(nb);
            const int32_t pred = low | (low << CRP_VECW); /* the same rows in both halves */
            agen_A trB         = init_A((void *)(stRe + b * CRP_VECW * CRP_VD_STAGE_PITCH));
            trB.lane_ofst      = CRP_VD_STAGE_LANE_OFST;
            INIT_AGEN2(trB, wtB);
            agen_A tiB    = init_A((void *)(stIm + b * CRP_VECW * CRP_VD_STAGE_PITCH));
            tiB.lane_ofst = CRP_VD_STAGE_LANE_OFST;
            INIT_AGEN2(tiB, wtB);
            agen_C vd = initAgenC4_i(vdRe + b * 2 * CRP_VECW, numRx, numTx * rowPitch, numTx, rowPitch, 1, 0, 1, 0);
            for (int32_t c = 0; c < nPhysE; c++) chess_prepare_for_pipelining
            chess_loop_range(2, )
            chess_unroll_loop(2)
            {
                dvintx x;
                x.lo = vfp_vintx_round(vmulf(vfloat_load_transp(trB), scale));
                x.hi = vfp_vintx_round(vmulf(vfloat_load_transp(tiB), scaleIm));
                vstore(x, vd, pred);
            }
        }
    }
    else
    {
        for (int32_t b = 0; b < nBlk; b++)
        {
            const int32_t nb      = numSel - b * CRP_VECW; /* valid rows in this block (>= CRP_VECW but the last) */
            const int32_t predNb  = (nb >= CRP_VECW) ? -1 : CRP_LOW_LANES(nb);
            const int32_t predFba = fba ? predNb : 0;
            for (int32_t c = 0; c < nPhys; c++) chess_prepare_for_pipelining
            chess_loop_range(1, )
            {
                const vintx xr = vfp_vintx_round(vmulf(vfloat_load_transp(tr), scale));
                const vintx xi = vfp_vintx_round(vmulf(vfloat_load_transp(ti), scaleIm));
                vstore(xr, vr, predNb);
                vstore(xi, vi, predNb);
                vstore(xr, mr, predFba);
                vstore(zeroI - xi, mi, predFba);
            }
        }
    }
    CRP_SEPARATOR();

    k->vdFxReBuf = vdRe;
    k->vdFxImBuf = vdIm;
}

/* Steering block: DMA this ROI position's LUT block straight to steeringVectorsROI row det; it overlaps the rest of
 * this detection and the next one's start. */
void caponIssueSteering(CaponKnl *k, int32_t roiAzStart, int32_t roiElStart, int32_t det)
{
    const int64_t blockBytes = (int64_t)k->svBlockInts * (int64_t)sizeof(int32_t);
    const int32_t roiPos     = roiElStart * k->numRoiAzPos + roiAzStart;
    const int64_t srcAddr    = k->lutBase + (int64_t)roiPos * blockBytes;
    const int64_t dstAddr    = k->svBase + (int64_t)det * blockBytes;
    if (k->svPending)
        cupvaSQDFSync(*k->lutGatherHdl); /* the previous detection's copy */
    cupvaSQDFUpdateAddr(*k->lutGatherHdl, 0, (uint64_t)srcAddr, MEMTYPE_DRAM, (uint64_t)dstAddr, MEMTYPE_DRAM);
    cupvaSQDFFlushAndTrig(*k->lutGatherHdl);
    k->svPending = 1;
}

void caponCompleteSteering(CaponKnl *k)
{
    if (!k->svPending)
        return;
    k->svPending = 0;
    cupvaSQDFSync(*k->lutGatherHdl);
}
