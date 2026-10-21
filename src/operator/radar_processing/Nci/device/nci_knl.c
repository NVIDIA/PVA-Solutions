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

#include "nci_knl.h"

#include "nci_param.h"

#include <stdint.h>

void nciRx_params_init(param_nci_t *params, uint32_t NrPerTile, uint32_t Ns, uint32_t NsPerTile, uint32_t NofRx)
{
    params->nciRxWrappers[0].size = sizeof(Q16_15);
    params->nciRxWrappers[0].n1   = (NsPerTile * 2) / chess_elementsof(dvintx);
    params->nciRxWrappers[0].s1   = chess_elementsof(dvintx);
    params->nciRxWrappers[0].n2   = NrPerTile;
    params->nciRxWrappers[0].s2   = (NsPerTile * 2) * NofRx;

    params->nciRxWrappers[1].size = sizeof(Q16_15);
    params->nciRxWrappers[1].n1   = (NsPerTile * 2) / chess_elementsof(dvintx);
    params->nciRxWrappers[1].s1   = chess_elementsof(dvintx);
    params->nciRxWrappers[1].n2   = NrPerTile;
    params->nciRxWrappers[1].s2   = (NsPerTile * 2) * NofRx;

    params->nciRxWrappers[2].size = sizeof(UQ16_16);
    params->nciRxWrappers[2].n1   = NsPerTile / chess_elementsof(vintx);
    params->nciRxWrappers[2].s1   = chess_elementsof(vintx);
    params->nciRxWrappers[2].n2   = NrPerTile;
    params->nciRxWrappers[2].s2   = Ns;

    params->nciRxWrappers[3].size = sizeof(UQ16_16);
    params->nciRxWrappers[3].n1   = NsPerTile / chess_elementsof(vintx);
    params->nciRxWrappers[3].s1   = chess_elementsof(vintx);
    params->nciRxWrappers[3].n2   = NrPerTile;
    params->nciRxWrappers[3].s2   = Ns;

    /* Below Settings are applicable for Range Doppler Rx Layout*/
    /* Source: Channel 1 double vector transpose load.*/
    params->nciRxWrappers[4].size = sizeof(Q16_15);
    params->nciRxWrappers[4].n1   = (NsPerTile);
    params->nciRxWrappers[4].s1   = 2; // real and imaginary
    params->nciRxWrappers[4].n2   = (NrPerTile * NofRx) / chess_elementsof(vintx);
    params->nciRxWrappers[4].s2 =
        (NsPerTile * 2 + 2) * chess_elementsof(vintx); // additional 2 added for transpose interleaved buffer

    /* Nci Temp Buffer transpose store */
    params->nciRxWrappers[5].size = sizeof(Q16_15);
    params->nciRxWrappers[5].n1   = (NofRx / 2);
    params->nciRxWrappers[5].s1   = 0;
    params->nciRxWrappers[5].n2   = NsPerTile / NofRx;
    params->nciRxWrappers[5].s2   = 1;
    params->nciRxWrappers[5].n3   = (NrPerTile * NofRx) / chess_elementsof(vintx);
    params->nciRxWrappers[5].s3   = ((Ns / NofRx) + 1) * chess_elementsof(vintx);

    //Copy Agens from NciRx to NciTemp for sequential data transfer;
    //TBD: use double vector loads to improve performance
    params->nciRxWrappers[6].size = sizeof(Q16_15);
    params->nciRxWrappers[6].n1   = NrPerTile * Ns / (2 * chess_elementsof(vintx));
    params->nciRxWrappers[6].s1   = 2 * chess_elementsof(vintx);

    params->nciRxWrappers[7].size = sizeof(Q16_15);
    params->nciRxWrappers[7].n1   = Ns / (2 * chess_elementsof(vintx) * NofRx);
    params->nciRxWrappers[7].s1   = 2 * chess_elementsof(vintx);
    params->nciRxWrappers[7].n2   = (NrPerTile * NofRx);
    params->nciRxWrappers[7].s2   = (Ns / NofRx) + 1; // 1 is added as padding for transposed copy

    return;
}

int32_t copy_nciRxTemp_to_nciRx(param_nci_t *params, uint32_t Ns, uint32_t NrPerTile, UQ16_16 *nciRx,
                                uint32_t *nciRxTemp, uint32_t nciRxShiftValue)
{
    int niter             = (NrPerTile * Ns) / (2 * chess_elementsof(vintx));
    agen_A nciRxTemp_agen = init_A((vuint *)(nciRxTemp));
    INIT_AGEN2(nciRxTemp_agen, params->nciRxWrappers[7]);

    agen_C nciRx_agen = init_C((vuint *)nciRx);
    INIT_AGEN1(nciRx_agen, params->nciRxWrappers[6]);
    nciRx_agen.round = (agenTruncateMask | nciRxShiftValue);

    for (int i = 0; i < niter; i++) chess_unroll_loop(2)
    chess_prepare_for_pipelining
    chess_loop_range(2, )
    {
        dvintx vData = dvuint_load(nciRxTemp_agen);
        vstore(vData, nciRx_agen);
    }
    return 0;
}

int32_t compute_nciRx_RangeDopplerRx(param_nci_t *params, Q16_15 *ft2d, uint32_t NrPerTile, uint32_t NsPerTile,
                                     uint32_t NofRx, uint32_t tileIndex, uint32_t *nciRxTemp, uint32_t Ns)
{
    uint32_t tilesperRow = Ns / NsPerTile;
    uint32_t baseIndex   = (tileIndex % tilesperRow) * (NsPerTile / NofRx);

    // Input Read
    agen_B ft2d_RxCh1_agen    = init_B((vint *)ft2d);
    ft2d_RxCh1_agen.lane_ofst = (NsPerTile * 2) / 16;
    INIT_AGEN2(ft2d_RxCh1_agen, params->nciRxWrappers[4]);

    //temp output for store
    agen_A nciRxTemp_agen    = init_A((vuint *)(nciRxTemp + baseIndex));
    nciRxTemp_agen.lane_ofst = (Ns / NofRx) / 16;
    INIT_AGEN3(nciRxTemp_agen, params->nciRxWrappers[5]);

    vintx vNciRx;
    int32_t iter        = ((NrPerTile * NofRx) + chess_elementsof(vintx) - 1) / chess_elementsof(vintx);
    iter                = iter * ((NsPerTile * 2) / (NofRx * 2)) * (NofRx / 2);
    vNciRx              = replicatew(0);
    int numrxIterMinux1 = (NofRx >> 1) - 1;
    int modOp           = numrxIterMinux1;

    for (int i = 0; i < iter; i++) chess_unroll_loop(8)
    chess_prepare_for_pipelining
    chess_loop_range(8, )
    {
        modOp          = mod_inc(modOp, numrxIterMinux1);
        int pred_store = mux(modOp == numrxIterMinux1, 0x0FF, 0);

        dvintx vRxCh1 = dvint_load_transp2_di(ft2d_RxCh1_agen);
        dvintx vRxCh2 = dvint_load_transp2_di(ft2d_RxCh1_agen);

        dvfloatx vRxCh1_float, vRxCh2_float;
        vRxCh1_float.lo = vintx_vfp(vRxCh1.lo);
        vRxCh1_float.hi = vintx_vfp(vRxCh1.hi);
        vRxCh2_float.lo = vintx_vfp(vRxCh2.lo);
        vRxCh2_float.hi = vintx_vfp(vRxCh2.hi);

        vfloatx vRxCh1_sumsq = vmaddf(vRxCh1_float.hi, vRxCh1_float.hi, vmulf(vRxCh1_float.lo, vRxCh1_float.lo));
        vfloatx vRxCh2_sumsq = vmaddf(vRxCh2_float.hi, vRxCh2_float.hi, vmulf(vRxCh2_float.lo, vRxCh2_float.lo));

        vfloatx vRxCh1_magnitudef = vfsqrt(vRxCh1_sumsq);
        vfloatx vRxCh2_magnitudef = vfsqrt(vRxCh2_sumsq);

        vintx vRxCh1_magnitude = vfp_vintx_trunc(vRxCh1_magnitudef);
        vintx vRxCh2_magnitude = vfp_vintx_trunc(vRxCh2_magnitudef);

        vNciRx = vNciRx + vRxCh1_magnitude + vRxCh2_magnitude;
        vstore_transp(vNciRx, nciRxTemp_agen, pred_store);
        vNciRx = vmux(replicatew(pred_store == 0x0FF), replicatew(0), vNciRx);
    }

    return 0;
}

int32_t compute_nciRx_first(param_nci_t *params, Q16_15 *ft2d, UQ16_16 *nciRx, uint32_t NrPerTile, uint32_t NsPerTile,
                            uint32_t nciRxShiftValue)
{
    int vecw = chess_elementsof(dvintx);
    int iter = NrPerTile * (NsPerTile * 2) / vecw;

    agen_B ft2d_RxCh1_agen = init_B((vint *)ft2d);
    agen_B ft2d_RxCh2_agen = init_B((vint *)(ft2d + (NsPerTile * 2)));
    agen_C nciRx_agen      = init_C((vuint *)nciRx);
    INIT_AGEN2(ft2d_RxCh1_agen, params->nciRxWrappers[0]);
    INIT_AGEN2(ft2d_RxCh2_agen, params->nciRxWrappers[1]);
    INIT_AGEN2(nciRx_agen, params->nciRxWrappers[2]);

    nciRx_agen.round = ((1u == nciRxShiftValue) ? (agenTruncateMask | nciRxShiftValue) : 0);

    for (int i = 0; i < iter; i++) chess_unroll_loop(2)
    chess_prepare_for_pipelining
    chess_loop_range(4, )
    {
        dvintx vRxCh1 = dvint_load_di(ft2d_RxCh1_agen);
        dvintx vRxCh2 = dvint_load_di(ft2d_RxCh2_agen);

        dvfloatx vRxCh1_float, vRxCh2_float;
        vRxCh1_float.lo = vintx_vfp(vRxCh1.lo);
        vRxCh1_float.hi = vintx_vfp(vRxCh1.hi);
        vRxCh2_float.lo = vintx_vfp(vRxCh2.lo);
        vRxCh2_float.hi = vintx_vfp(vRxCh2.hi);

        vfloatx vRxCh1_sumsq = vmaddf(vRxCh1_float.hi, vRxCh1_float.hi, vmulf(vRxCh1_float.lo, vRxCh1_float.lo));
        vfloatx vRxCh2_sumsq = vmaddf(vRxCh2_float.hi, vRxCh2_float.hi, vmulf(vRxCh2_float.lo, vRxCh2_float.lo));

        vfloatx vRxCh1_magnitudef = vfsqrt(vRxCh1_sumsq);
        vfloatx vRxCh2_magnitudef = vfsqrt(vRxCh2_sumsq);

        vintx vRxCh1_magnitude = vfp_vint_trunc(vRxCh1_magnitudef);
        vintx vRxCh2_magnitude = vfp_vint_trunc(vRxCh2_magnitudef);

        vintx vNciRx = vRxCh1_magnitude + vRxCh2_magnitude;
        vstore(vNciRx, nciRx_agen);
    }

    return 0;
}

int32_t compute_nciRx(param_nci_t *params, Q16_15 *ft2d, UQ16_16 *nciRx, uint32_t isLast, uint32_t NrPerTile,
                      uint32_t NsPerTile, uint32_t nciRxShiftValue)
{
    int vecw = chess_elementsof(dvintx);
    int iter = NrPerTile * (NsPerTile * 2) / vecw;

    agen_B ft2d_RxCh1_agen        = init_B((vint *)ft2d);
    agen_B ft2d_RxCh2_agen        = init_B((vint *)(ft2d + (NsPerTile * 2)));
    agen_C nciRx_agen             = init_C((vuint *)nciRx);
    agen_C nciRxIntermediate_agen = init_C((vuint *)nciRx);
    INIT_AGEN2(ft2d_RxCh1_agen, params->nciRxWrappers[0]);
    INIT_AGEN2(ft2d_RxCh2_agen, params->nciRxWrappers[1]);
    INIT_AGEN2(nciRx_agen, params->nciRxWrappers[2]);
    INIT_AGEN2(nciRxIntermediate_agen, params->nciRxWrappers[3]);

    nciRx_agen.round = ((1u == isLast) ? (agenTruncateMask | nciRxShiftValue) : 0);

    for (int i = 0; i < iter; i++) chess_unroll_loop(2)
    chess_prepare_for_pipelining
    chess_loop_range(4, )
    {
        dvintx vRxCh1 = dvint_load_di(ft2d_RxCh1_agen);
        dvintx vRxCh2 = dvint_load_di(ft2d_RxCh2_agen);

        dvfloatx vRxCh1_float, vRxCh2_float;
        vRxCh1_float.lo = vintx_vfp(vRxCh1.lo);
        vRxCh1_float.hi = vintx_vfp(vRxCh1.hi);
        vRxCh2_float.lo = vintx_vfp(vRxCh2.lo);
        vRxCh2_float.hi = vintx_vfp(vRxCh2.hi);

        vfloatx vRxCh1_sumsq = vmaddf(vRxCh1_float.hi, vRxCh1_float.hi, vmulf(vRxCh1_float.lo, vRxCh1_float.lo));
        vfloatx vRxCh2_sumsq = vmaddf(vRxCh2_float.hi, vRxCh2_float.hi, vmulf(vRxCh2_float.lo, vRxCh2_float.lo));

        vfloatx vRxCh1_magnitudef = vfsqrt(vRxCh1_sumsq);
        vfloatx vRxCh2_magnitudef = vfsqrt(vRxCh2_sumsq);

        vintx vRxCh1_magnitude = vfp_vintx_trunc(vRxCh1_magnitudef);
        vintx vRxCh2_magnitude = vfp_vintx_trunc(vRxCh2_magnitudef);

        vintx vNciRx = vuint_load(nciRxIntermediate_agen);
        vNciRx       = vNciRx + vRxCh1_magnitude + vRxCh2_magnitude;

        vstore(vNciRx, nciRx_agen);
    }

    return 0;
}

void nciFinal_params_init(param_nci_t *params, uint32_t NrPerTile, uint32_t NsPerFold, uint32_t Ns)
{
    params->nciFinalWrappers[0].size = sizeof(UQ16_16);
    params->nciFinalWrappers[0].n1   = NsPerFold / chess_elementsof(dvintx);
    params->nciFinalWrappers[0].s1   = chess_elementsof(dvintx);
    params->nciFinalWrappers[0].n2   = NrPerTile;
    params->nciFinalWrappers[0].s2   = Ns;

    params->nciFinalWrappers[1].size = sizeof(UQ16_16);
    params->nciFinalWrappers[1].n1   = NsPerFold / chess_elementsof(dvintx);
    params->nciFinalWrappers[1].s1   = chess_elementsof(dvintx);
    params->nciFinalWrappers[1].n2   = NrPerTile;
    params->nciFinalWrappers[1].s2   = Ns;

    params->nciFinalWrappers[2].size = sizeof(UQ16_16);
    params->nciFinalWrappers[2].n1   = NsPerFold / chess_elementsof(dvintx);
    params->nciFinalWrappers[2].s1   = chess_elementsof(dvintx);
    params->nciFinalWrappers[2].n2   = NrPerTile;
    params->nciFinalWrappers[2].s2   = Ns;

    params->nciFinalWrappers[3].size = sizeof(UQ16_16);
    params->nciFinalWrappers[3].n1   = NsPerFold / chess_elementsof(dvintx);
    params->nciFinalWrappers[3].s1   = chess_elementsof(dvintx);
    params->nciFinalWrappers[3].n2   = NrPerTile;
    params->nciFinalWrappers[3].s2   = Ns;

    params->nciFinalWrappers[4].size = sizeof(UQ16_16);
    params->nciFinalWrappers[4].n1   = NsPerFold / chess_elementsof(dvintx);
    params->nciFinalWrappers[4].s1   = chess_elementsof(dvintx);
    params->nciFinalWrappers[4].n2   = NrPerTile;
    params->nciFinalWrappers[4].s2   = NsPerFold;

    params->nciFinalWrappers[5].size = sizeof(UQ16_16);
    params->nciFinalWrappers[5].n1   = NsPerFold / chess_elementsof(dvintx);
    params->nciFinalWrappers[5].s1   = chess_elementsof(dvintx);
    params->nciFinalWrappers[5].n2   = NrPerTile;
    params->nciFinalWrappers[5].s2   = NsPerFold;

    return;
}

int32_t compute_nciFinal(param_nci_t *params, UQ16_16 *nciRxFold0, UQ16_16 *nciRxFold1, UQ16_16 *nciRxFold2,
                         UQ16_16 *nciRxFold3, UQ16_16 *nciFinal, UQ16_16 *noiseEstimate, uint32_t isFirstIter,
                         uint32_t isLastDopplerFold, uint32_t NrPerTile, uint32_t NsPerFold,
                         uint32_t nciFinalShiftValue)
{
    int vecw    = chess_elementsof(dvintx);
    int rowIter = (NsPerFold / vecw);
    int iter    = NrPerTile * rowIter;

    agen_C nciRxFold0_agen   = init_C((vuint *)nciRxFold0);
    agen_C nciRxFold1_agen   = init_C((vuint *)nciRxFold1);
    agen_C nciRxFold2_agen   = init_C((vuint *)nciRxFold2);
    agen_C nciRxFold3_agen   = init_C((vuint *)nciRxFold3);
    agen_C nciFinal_agen     = init_C((vuint *)nciFinal);
    agen_C nciFinalPrev_agen = init_C((vuint *)nciFinal);
    INIT_AGEN2(nciRxFold0_agen, params->nciFinalWrappers[0]);
    INIT_AGEN2(nciRxFold1_agen, params->nciFinalWrappers[1]);
    INIT_AGEN2(nciRxFold2_agen, params->nciFinalWrappers[2]);
    INIT_AGEN2(nciRxFold3_agen, params->nciFinalWrappers[3]);
    INIT_AGEN2(nciFinal_agen, params->nciFinalWrappers[4]);
    INIT_AGEN2(nciFinalPrev_agen, params->nciFinalWrappers[5]);

    dvintx vNciRx;

    nciFinal_agen.round = ((lastDopplerFold == isLastDopplerFold) ? (agenTruncateMask | nciFinalShiftValue) : 0);

    for (int rowIdx = 0; rowIdx < (int)NrPerTile; rowIdx++) chess_unroll_loop(1)
    chess_prepare_for_pipelining
    chess_loop_range(1, )
    {
        nciFinal_agen.minmax_opt = 3;

        for (int i = 0; i < rowIter; i++) chess_unroll_loop(2)
        chess_prepare_for_pipelining
        chess_loop_range(1, )
        {
            vNciRx.lo          = replicatew(0);
            vNciRx.hi          = replicatew(0);
            dvintx vNciRxFold0 = dvuint_load(nciRxFold0_agen);
            dvintx vNciRxFold1 = dvuint_load(nciRxFold1_agen);
            dvintx vNciRxFold2 = dvuint_load(nciRxFold2_agen);
            dvintx vNciRxFold3 = dvuint_load(nciRxFold3_agen);

            dvuint_load(vNciRx, nciFinalPrev_agen, (int)(!isFirstIter));

            vNciRx = vNciRx + vNciRxFold0 + vNciRxFold1 + vNciRxFold2 + vNciRxFold3;

            vstore(vNciRx, nciFinal_agen);
        }

        noiseEstimate[rowIdx] = nciFinal_agen.min_val * threshold;
    }
    return 0;
}

void noiseEstimate_params_init(param_nci_t *params, uint32_t NrPerTile, uint32_t NsPerFold)
{
    params->nciNoiseEstimateWrappers[0].size = sizeof(UQ16_16);
    params->nciNoiseEstimateWrappers[0].n1   = NsPerFold / chess_elementsof(dvintx);
    params->nciNoiseEstimateWrappers[0].s1   = chess_elementsof(dvintx);
    params->nciNoiseEstimateWrappers[0].n2   = NrPerTile;
    params->nciNoiseEstimateWrappers[0].s2   = NsPerFold;

    return;
}

int32_t compute_noiseEstimate(param_nci_t *params, UQ16_16 *nciFinal, UQ16_16 *noiseEstimate, uint32_t NrPerTile,
                              uint32_t NsPerFold)
{
    int iter = (NsPerFold / chess_elementsof(dvintx));

    agen_C nciFinal_agen = init_C((vuint *)nciFinal);
    INIT_AGEN2(nciFinal_agen, params->nciNoiseEstimateWrappers[0]);
    vintx vNciFinal;

    for (int rowIdx = 0; rowIdx < (int)NrPerTile; rowIdx++)
    {
        uint32_t minVal = MAX_UINT32;
        vNciFinal       = replicatew(MAX_UINT32);
        for (int i = 0; i < iter; i++) chess_unroll_loop(4)
        chess_prepare_for_pipelining
        chess_loop_range(4, )
        {
            dvintx vNciFinalCurr = dvuint_load(nciFinal_agen);

            vNciFinal = vmin3(vNciFinalCurr.lo, vNciFinalCurr.hi, vNciFinal);
        }

        minVal = min(minVal, (uint32_t)vminr_s(vNciFinal));

        noiseEstimate[rowIdx] = minVal * threshold;
    }

    return 0;
}
