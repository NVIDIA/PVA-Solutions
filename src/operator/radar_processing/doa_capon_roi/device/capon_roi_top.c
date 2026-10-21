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
 * @file capon_roi_top.c
 * @brief Combined Capon ROI entry point executing Part 1 and Part 2 in one command program.
 */

#include "capon_roi_vmem_layout.h"
#include "part1/block_cholesky/block_cholesky_linv_bs4_knl.h"
#include "part1/capon_roi_part1_config.h"
#include "part1/cov_mat/cov_mat_common.h"
#include "part2/capon_roi_part2_config.h"

#include <PvaOperatorTypes.h>
#include <cupva_device.h>
#include <cupva_device/device_core.h>

// define Icache prefetch bytes
#if CUPVA_PVA_GEN_NUMBER >= 3
// gen3, part2
#    if CAPON_ROI_ENABLE_SUBBIN_INTERP
#        define CAPON_PART2_INIT_PREFETCH_BYTES 16000u
#    else
#        define CAPON_PART2_INIT_PREFETCH_BYTES 15360u
#    endif
#else
// gen2, part1
#    if CAPON_ROI_USE_FP_COVMAT
#        define CAPON_PART1_HOT_PREFETCH_BYTES 11776u
#    else
#        define CAPON_PART1_HOT_PREFETCH_BYTES 11904u
#    endif
#    define CAPON_PART2_INIT_PREFETCH_BYTES 6656u
// gen2, part2
#    if CAPON_ROI_USE_FP_COVMAT
#        if CAPON_ROI_ENABLE_SUBBIN_INTERP
#            define CAPON_PART2_HOT_PREFETCH_BYTES 9088u
#        else
#            define CAPON_PART2_HOT_PREFETCH_BYTES 8192u
#        endif
#    else
#        if CAPON_ROI_ENABLE_SUBBIN_INTERP
#            define CAPON_PART2_HOT_PREFETCH_BYTES 9216u
#        else
#            define CAPON_PART2_HOT_PREFETCH_BYTES 8320u
#        endif
#    endif

void obtain_fixed_pt_L_inv_stage0(const float *const restrict L_inv_real, const float *const restrict L_inv_imag,
                                  const Para_capon_roi_part2_config &para);

#endif

VMEM_SEQDF_HANDLER(A, snapshotSQDFHdl);
VMEM_SEQDF_HANDLER(A, interOutSQDFHdl);
VMEM_SEQDF_HANDLER(A, interInSQDFHdl);
VMEM_SEQDF_HANDLER(A, svCh0SQDFHdl);
VMEM_SEQDF_HANDLER(A, svCh1SQDFHdl);
VMEM_SEQDF_HANDLER(A, outSQDFHdl);

// Part 1 reuses the leading regions of the steering allocations before Part 2 starts.
VMEM(B, int32_t, arenaBPtr, BUF_SV_SINGLE_TILE_PIX * 2);
VMEM(C, int32_t, arenaCPtr, BUF_SV_SINGLE_TILE_PIX * 2);

VMEM(A, float, LinvRealBuffer, BUF_SZ_LINV * 2);
#if CUPVA_PVA_GEN_NUMBER >= 3
VMEM(D, float, LinvImagBuffer, BUF_SZ_LINV * 2);
VMEM(D, float, scaleBuffer, BUF_SZ_SCALE);
VMEM(D, Para_capon_roi_part1_config, paraPart1Storage);
#else
VMEM(A, float, LinvImagBuffer, BUF_SZ_LINV * 2);
VMEM(A, float, scaleBuffer, BUF_SZ_SCALE);
#endif

VMEM(A, float, covRealBuffer, BUF_SZ_COV_REAL);
VMEM(A, float, rcpSpectrumBuffer, BUF_SZ_RCP_SPECTRUM);
VMEM(A, int32_t, LinvFixedPtBuffer, BUF_SZ_LINV_FIXED);
VMEM(A, float, spectrumMinBuffer, BUF_SZ_SPEC_MIN);
VMEM(A, uint8_t, noiseHistBuffer, BUF_SZ_NOISE_HIST);
VMEM(A, uint16_t, localMaxBuffer, BUF_SZ_LOCAL_MAX);
VMEM(A, uint16_t, candidateBuffer, BUF_SZ_CANDIDATE);
VMEM(A, float, peakPowerBuffer, BUF_SZ_OUTPUT * 2);
VMEM(A, float, azBinBuffer, BUF_SZ_OUTPUT * 2);
VMEM(A, float, elBinBuffer, BUF_SZ_OUTPUT * 2);

VMEM(A, int32_t, numTilesDoppler);
VMEM(A, int32_t, numTilesROIdv);
VMEM(A, int32_t, numTilesROIsv);
VMEM(A, int32_t, numROIs);
#if !CAPON_ROI_USE_FP_COVMAT
VMEM(A, int32_t, numDopplerBinsCovMatBits);
#endif
VMEM(A, int32_t, dopplerBatchSize);
VMEM(A, int32_t, snapImageLp);
VMEM(A, int32_t, snapTileWidth);
VMEM(A, int32_t, snapTileHeight);
VMEM(A, int32_t, hasLastTileDoppler);
VMEM(A, int32_t, lastTileDopplerBins);
VMEM(A, int32_t, lastTileWidthDoppler);
VMEM(A, int32_t, hasLastTileRoi);
VMEM(A, int32_t, lastTileRoiHeight);

VMEM(A, int32_t, interGroupStride);
VMEM(A, int32_t, peakPowerRowStride);
VMEM(A, int32_t, azBinRowStride);
VMEM(A, int32_t, elBinRowStride);

VMEM_POINTER(A, snapshotsExtPtr);
VMEM_POINTER(A, interRealExtPtr);
VMEM_POINTER(A, interImagExtPtr);
VMEM_POINTER(A, peakPowerExtPtr);
VMEM_POINTER(A, azBinExtPtr);
VMEM_POINTER(A, elBinExtPtr);

VMEM(A, PVADoaCaponROIParams, GP);
VMEM(A, Para_capon_roi_part2_config, paraPart2);

#if defined(__cplusplus)
extern "C" {
#endif

__attribute__((noinline)) void capon_roi_part1_phase(void);
__attribute__((noinline)) void capon_roi_part2_phase(void);

#if defined(__cplusplus)
}
#endif

CUPVA_VPU_MAIN()
{
    capon_roi_part1_phase();
    capon_roi_part2_phase();
    return 0;
}

#if defined(__cplusplus)
extern "C" {
#endif

__attribute__((noinline)) void capon_roi_part1_phase(void)
{
    int32_t *snapshotPingBuffer = arenaBPtr;
    int32_t *snapshotPongBuffer = arenaCPtr;
    float *part1LinvImagBuffer  = (float *)(arenaBPtr + BUF_SZ_SNAPSHOT);
    float *covImagBuffer        = (float *)(arenaCPtr + BUF_SZ_SNAPSHOT);
#if CUPVA_PVA_GEN_NUMBER >= 3
    Para_capon_roi_part1_config *paraPart1 = &paraPart1Storage;
#else
    Para_capon_roi_part1_config *paraPart1 = (Para_capon_roi_part1_config *)(part1LinvImagBuffer + BUF_SZ_LINV * 2);
#endif

    cupvaSQDFOpen(snapshotSQDFHdl);
    cupvaSQDFTrig(snapshotSQDFHdl);

#if CUPVA_PVA_GEN_NUMBER < 3
    // Fill the Part 1 hot path while the first snapshot tile is being read.
    cupvaICachePrefetch((uintptr_t)update_snapshot_agen_addr, CAPON_PART1_HOT_PREFETCH_BYTES);
#endif

    cupvaSQDFOpen(interOutSQDFHdl);

    init_capon_roi_part1_config(paraPart1, covRealBuffer, covImagBuffer, GP.adpDiagConst, dopplerBatchSize,
                                frcp((float)GP.numDopplerBinsCovMat),
#if !CAPON_ROI_USE_FP_COVMAT
                                numDopplerBinsCovMatBits,
#endif
                                (int32_t)GP.inputQBits, lastTileDopplerBins);

    uint64_t snapshotsBase = snapshotsExtPtr.base + snapshotsExtPtr.offset;
    uint64_t interRealBase = interRealExtPtr.base + interRealExtPtr.offset;
    uint64_t interImagBase = interImagExtPtr.base + interImagExtPtr.offset;
    int32_t pingPong       = 0;
    int32_t linvPingPong   = 0;

    for (int32_t roiTile = 0; roiTile < numTilesROIdv; ++roiTile)
    {
        for (int32_t dopplerTile = 0; dopplerTile < numTilesDoppler; ++dopplerTile)
        {
            cupvaSQDFSync(snapshotSQDFHdl);

            // prime the done wait for the first tile
            if (roiTile == 0 && dopplerTile == 0)
            {
                cupvaPrefetchDoneWait();
            }

            int32_t isLastDopplerTile = hasLastTileDoppler && dopplerTile == numTilesDoppler - 1;
            update_snapshot_agen_addr(pingPong, snapshotPingBuffer, snapshotPongBuffer, isLastDopplerTile, roiTile,
                                      dopplerTile, numTilesROIdv, numTilesDoppler, snapTileHeight, snapTileWidth,
                                      snapImageLp, lastTileRoiHeight, lastTileWidthDoppler, hasLastTileRoi,
                                      hasLastTileDoppler, snapshotsBase, snapshotSQDFHdl, paraPart1);
            pingPong ^= 1;

            COMPUTE_COVMAT_DIAG_IMPL(paraPart1, dopplerTile, isLastDopplerTile);
            COMPUTE_COVMAT_OFF_DIAG_IMPL(paraPart1, dopplerTile, isLastDopplerTile);
        }

        add_diagonal_loading_knl_fp_impl(paraPart1);

        float *interReal             = LinvRealBuffer + linvPingPong * BUF_SZ_LINV;
        float *interImag             = part1LinvImagBuffer + linvPingPong * BUF_SZ_LINV;
        paraPart1->Linv_real_buf_ptr = interReal;
        paraPart1->Linv_imag_buf_ptr = interImag;

        block_cholesky_opt_full_driver_bs4(paraPart1);
        linv_opt_full_driver_bs4(paraPart1);

        if (roiTile > 0)
        {
            cupvaSQDFSync(interOutSQDFHdl);
        }
        int32_t interOffset = roiTile * interGroupStride;
        cupvaSQDFUpdateAddr(interOutSQDFHdl, 0, cupvaGetVmemAddress(interReal), MEMTYPE_VMEM,
                            interRealBase + interOffset, MEMTYPE_DRAM);
        cupvaSQDFUpdateAddr(interOutSQDFHdl, 1, cupvaGetVmemAddress(interImag), MEMTYPE_VMEM,
                            interImagBase + interOffset, MEMTYPE_DRAM);
        cupvaSQDFFlushAndTrig(interOutSQDFHdl);
        linvPingPong ^= 1;
    }

    // Fill Part 2 control and initialization while draining the final Part 1 write.
    cupvaICachePrefetch((uintptr_t)capon_roi_part2_phase, CAPON_PART2_INIT_PREFETCH_BYTES);

    cupvaSQDFSync(interOutSQDFHdl);

    cupvaSQDFClose(snapshotSQDFHdl);
    cupvaSQDFClose(interOutSQDFHdl);
}

__attribute__((noinline)) void capon_roi_part2_phase(void)
{
    uint64_t interRealBase = interRealExtPtr.base + interRealExtPtr.offset;
    uint64_t interImagBase = interImagExtPtr.base + interImagExtPtr.offset;

    char *svCh0PingBase = (char *)arenaBPtr;
    char *svCh1PingBase = (char *)arenaCPtr;
    char *svCh0PongBase = (char *)arenaBPtr + BUF_SV_SINGLE_TILE_BYTES;
    char *svCh1PongBase = (char *)arenaCPtr + BUF_SV_SINGLE_TILE_BYTES;

    uint16_t *fixedPtSpectrum  = (uint16_t *)(svCh0PongBase + BUF_PART2_SV0_FIXED_SPEC_OFST);
    float *floatPtSpectrum     = (float *)(svCh0PongBase + BUF_PART2_SV0_FLOAT_SPEC_OFST);
    uint16_t *noiseFloorBuffer = (uint16_t *)(svCh0PongBase + BUF_PART2_SV0_NOISE_FLOOR_OFST);
    uint8_t *noiseCumulBuffer  = (uint8_t *)(svCh1PongBase + BUF_PART2_SV1_NOISE_CUMUL_OFST);
    uint16_t *localSumD8Buffer = (uint16_t *)(svCh1PongBase + BUF_PART2_SV1_LOCAL_SUM_OFST);

    cupvaSQDFOpen(interInSQDFHdl);
    cupvaSQDFTrig(interInSQDFHdl);

    cupvaSQDFOpen(svCh0SQDFHdl);
    cupvaSQDFOpen(svCh1SQDFHdl);
    cupvaSQDFTrig(svCh0SQDFHdl);
    cupvaSQDFTrig(svCh1SQDFHdl);
    cupvaSQDFOpen(outSQDFHdl);

#if CUPVA_PVA_GEN_NUMBER < 3
    // Complete the init fill before consuming it.
    cupvaPrefetchDoneWait();
#endif

    init_capon_roi_part2_config(GP, scaleBuffer, LinvFixedPtBuffer, rcpSpectrumBuffer, spectrumMinBuffer,
                                fixedPtSpectrum, floatPtSpectrum, noiseHistBuffer, noiseCumulBuffer, noiseFloorBuffer,
                                localMaxBuffer, localSumD8Buffer, candidateBuffer, paraPart2);

#if CUPVA_PVA_GEN_NUMBER < 3
    // Fill the hot path after the one-time initialization code has been consumed.
    cupvaICachePrefetch((uintptr_t)obtain_fixed_pt_L_inv_stage0, CAPON_PART2_HOT_PREFETCH_BYTES);
#endif

    cupvaSQDFSync(interInSQDFHdl);
    cupvaSQDFSync(svCh0SQDFHdl);
    cupvaSQDFSync(svCh1SQDFHdl);

    int32_t *svCh0Ping   = arenaBPtr;
    int32_t *svCh0Pong   = svCh0Ping + BUF_SV_SINGLE_TILE_PIX;
    int32_t *svCh1Ping   = arenaCPtr;
    int32_t *svCh1Pong   = svCh1Ping + BUF_SV_SINGLE_TILE_PIX;
    int32_t steeringTile = 0;
    int32_t outPingPong  = 0;
    int32_t pingPong     = 0;
    int32_t linvPingPong = 0;
    int32_t numRoiGroups = numTilesROIdv;

    uint64_t peakPowerBase = peakPowerExtPtr.base + peakPowerExtPtr.offset;
    uint64_t azBinBase     = azBinExtPtr.base + azBinExtPtr.offset;
    uint64_t elBinBase     = elBinExtPtr.base + elBinExtPtr.offset;

    // Guarantee the active request before entering the repeated compute path.
    cupvaPrefetchDoneWait();

    for (int32_t roiGroup = 0; roiGroup < numRoiGroups; ++roiGroup)
    {
        // Consume the current L_inv slot while the opposite slot is filled for the next group.
        if (roiGroup != 0)
        {
            cupvaSQDFSync(interInSQDFHdl);
        }

        const float *LinvReal = LinvRealBuffer + linvPingPong * BUF_SZ_LINV;
        const float *LinvImag = LinvImagBuffer + linvPingPong * BUF_SZ_LINV;

        // update interIn ping-pong address for the next group
        if (roiGroup + 1 < numRoiGroups)
        {
            int32_t nextPingPong    = linvPingPong ^ 1;
            int32_t nextInterOffset = (roiGroup + 1) * interGroupStride;
            float *nextLinvReal     = LinvRealBuffer + nextPingPong * BUF_SZ_LINV;
            float *nextLinvImag     = LinvImagBuffer + nextPingPong * BUF_SZ_LINV;
            cupvaSQDFUpdateAddr(interInSQDFHdl, 0, interRealBase + nextInterOffset, MEMTYPE_DRAM,
                                cupvaGetVmemAddress(nextLinvReal), MEMTYPE_VMEM);
            cupvaSQDFUpdateAddr(interInSQDFHdl, 1, interImagBase + nextInterOffset, MEMTYPE_DRAM,
                                cupvaGetVmemAddress(nextLinvImag), MEMTYPE_VMEM);
            cupvaSQDFFlushAndTrig(interInSQDFHdl);
        }

        obtain_fixed_pt_L_inv(LinvReal, LinvImag, paraPart2);
        linvPingPong ^= 1;

        for (int32_t tileInGroup = 0; tileInGroup < 2 && steeringTile < numTilesROIsv; ++tileInGroup, ++steeringTile)
        {
            int32_t *svCh0 = pingPong == 0 ? svCh0Ping : svCh0Pong;
            int32_t *svCh1 = pingPong == 0 ? svCh1Ping : svCh1Pong;

            if (steeringTile != 0)
            {
                cupvaSQDFSync(svCh0SQDFHdl);
                cupvaSQDFSync(svCh1SQDFHdl);
            }
            if (steeringTile < numTilesROIsv - 1)
            {
                cupvaSQDFTrig(svCh0SQDFHdl);
                cupvaSQDFTrig(svCh1SQDFHdl);
            }

            update_part2_sv_agen_base(paraPart2, svCh0, svCh1, pingPong);
            pingPong ^= 1;

            if (paraPart2.shift_bits != 0)
            {
                shift_fixed_pt_steering_matrix(paraPart2);
            }
            compute_L_inv_times_steering_matrix(paraPart2);
            compute_fp_rcp_spectrum(paraPart2);
        }

        obtain_spectrum(paraPart2);
        estimate_noise_floor(paraPart2);
        extract_3x3filtered_spectrum_in16az8el_out14az6el(paraPart2);
        mark_peak_candidates(paraPart2);

        float *dstPeakPower = peakPowerBuffer + outPingPong * BUF_SZ_OUTPUT;
        float *dstAzBin     = azBinBuffer + outPingPong * BUF_SZ_OUTPUT;
        float *dstElBin     = elBinBuffer + outPingPong * BUF_SZ_OUTPUT;
        cupvaModifyAgenCfgBase(&paraPart2.cfg_power, dstPeakPower);
        cupvaModifyAgenCfgBase(&paraPart2.cfg_az, dstAzBin);
        cupvaModifyAgenCfgBase(&paraPart2.cfg_el, dstElBin);
        get_top3_peaks(paraPart2);

        if (roiGroup > 0)
        {
            cupvaSQDFSync(outSQDFHdl);
        }

        int32_t roiOffset  = roiGroup * DV_ROI_NUM;
        int32_t tileHeight = min(numROIs - roiOffset, DV_ROI_NUM);

        // update ping-pong output address
        cupvaSQDFUpdateAddr(outSQDFHdl, 0, cupvaGetVmemAddress(dstPeakPower), MEMTYPE_VMEM,
                            peakPowerBase + roiOffset * peakPowerRowStride, MEMTYPE_DRAM);
        cupvaSQDFUpdateAddr(outSQDFHdl, 1, cupvaGetVmemAddress(dstAzBin), MEMTYPE_VMEM,
                            azBinBase + roiOffset * azBinRowStride, MEMTYPE_DRAM);
        cupvaSQDFUpdateAddr(outSQDFHdl, 2, cupvaGetVmemAddress(dstElBin), MEMTYPE_VMEM,
                            elBinBase + roiOffset * elBinRowStride, MEMTYPE_DRAM);
        cupvaSQDFUpdateTileSize(outSQDFHdl, 0, NUM_PEAKS_PER_ROI * sizeof(float), tileHeight);
        cupvaSQDFUpdateTileSize(outSQDFHdl, 1, NUM_PEAKS_PER_ROI * sizeof(float), tileHeight);
        cupvaSQDFUpdateTileSize(outSQDFHdl, 2, NUM_PEAKS_PER_ROI * sizeof(float), tileHeight);
        cupvaSQDFFlushAndTrig(outSQDFHdl);
        outPingPong ^= 1;
    }

    cupvaSQDFSync(outSQDFHdl);
    cupvaSQDFClose(interInSQDFHdl);
    cupvaSQDFClose(svCh0SQDFHdl);
    cupvaSQDFClose(svCh1SQDFHdl);
    cupvaSQDFClose(outSQDFHdl);
}

#if defined(__cplusplus)
}
#endif
