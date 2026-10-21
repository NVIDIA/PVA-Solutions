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

#include "peak_detection_knls.h"

#include <cupva_device.h>

void peakDetectionNmsInit(int32_t inPitch, int32_t tileX, int32_t tileY, int32_t cbSizeBytes, intptr_t cbStartAddr,
                          NmsAgenParam *paramPtr)
{
    const int32_t DVECW = chess_elementsof(dvintx);
    paramPtr->niterX    = tileX / DVECW;
    paramPtr->niterY    = tileY;
    paramPtr->niter     = (tileX / DVECW) * tileY;

    {
        AgenWrapper wrapper;
        wrapper.size   = sizeof(uint32_t);
        agen streamIn0 = init(NULL);
        wrapper.n1     = 2;
        wrapper.n2     = tileX / DVECW;
        wrapper.n3     = tileY;
        wrapper.s1     = 2 * inPitch;
        wrapper.s2     = DVECW;
        wrapper.s3     = inPitch;
        INIT_AGEN3(streamIn0, wrapper);
        streamIn0           = update_agen_cb_start(streamIn0, cbStartAddr);
        streamIn0           = update_agen_cb_size(streamIn0, cbSizeBytes);
        paramPtr->streamIn0 = extract_agen_cfg(streamIn0);
    }
    {
        AgenWrapper wrapper;
        wrapper.size   = sizeof(uint32_t);
        agen streamIn1 = init(NULL);
        wrapper.n1     = 2;
        wrapper.n2     = tileX / DVECW;
        wrapper.n3     = tileY;
        wrapper.s1     = 2;
        wrapper.s2     = DVECW;
        wrapper.s3     = inPitch;
        INIT_AGEN3(streamIn1, wrapper);
        streamIn1           = update_agen_cb_start(streamIn1, cbStartAddr);
        streamIn1           = update_agen_cb_size(streamIn1, cbSizeBytes);
        paramPtr->streamIn1 = extract_agen_cfg(streamIn1);
    }
    {
        AgenWrapper wrapper;
        wrapper.size      = sizeof(uint32_t);
        agen streamCenter = init(NULL);
        wrapper.n1        = tileX / DVECW;
        wrapper.n2        = tileY;
        wrapper.s1        = DVECW;
        wrapper.s2        = inPitch;
        INIT_AGEN2(streamCenter, wrapper);
        streamCenter           = update_agen_cb_start(streamCenter, cbStartAddr);
        streamCenter           = update_agen_cb_size(streamCenter, cbSizeBytes);
        paramPtr->streamCenter = extract_agen_cfg(streamCenter);
    }
}

int32_t peakDetectionNms(uint32_t *in0Ptr, uint32_t *in1Ptr, uint32_t *centerPtr, const uint32_t *noiseEstPtr,
                         const uint32_t *laneIdPtr, NmsAgenParam *paramPtr, int32_t tilePosY, uint32_t *rangeIdxPtr,
                         uint32_t *dopplerFoldIdxPtr)
{
    cupvaModifyAgenCfgBase(&paramPtr->streamIn0, in0Ptr);
    cupvaModifyAgenCfgBase(&paramPtr->streamIn1, in1Ptr);
    cupvaModifyAgenCfgBase(&paramPtr->streamCenter, centerPtr);

    agen streamIn0    = init_agen_from_cfg(paramPtr->streamIn0);
    agen streamIn1    = init_agen_from_cfg(paramPtr->streamIn1);
    agen streamCenter = init_agen_from_cfg(paramPtr->streamCenter);
    dvintx dv_lane_id = zero_extend(*(dvuint *)laneIdPtr);

    chess_separator_scheduler();
    dvintx dv_pos_y;
    dv_pos_y.lo = replicatew(tilePosY);
    dv_pos_y.hi = dv_pos_y.lo;

    int32_t count       = 0;
    const int32_t DVECW = chess_elementsof(dvintx);

    dvintx dv_pos_x_start = dv_lane_id;
    dvintx dv_pos_x_end   = dv_lane_id + (paramPtr->niterX - 1) * DVECW;
    dvintx dv_pos_x       = dv_pos_x_start;

    const int32_t niter = paramPtr->niter;
    for (int32_t i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvintx dv_p0     = dvuint_load(streamIn0);
        dvintx dv_p2     = dvuint_load(streamIn0);
        dvintx dv_p1     = dvuint_load(streamIn1);
        dvintx dv_p3     = dvuint_load(streamIn1);
        dvintx dv_center = dvuint_load(streamCenter);

        // If MSB of noiseEstPtr[i / (TILE_X / DVECW)] is guaranteed to be 0, then the following comparison is equivalent to the original one.
        //dvintx dv_flag = dv_center > noiseEstPtr[i / (TILE_X / DVECW)];
        dvintx dv_noiseEst;
        dv_noiseEst.lo = replicatew(noiseEstPtr[i / (TILE_X / DVECW)]);
        dv_noiseEst.hi = dv_noiseEst.lo;
        dvintx dv_flag = dv_center > dv_noiseEst;

        dvintx dv_max012  = dvmax3(dv_p0, dv_p1, dv_p2);
        dvintx dv_max0123 = dvmax(dv_max012, dv_p3);

        dvintx dv_peak = dvCmpLT_andL(dv_max0123, dv_center, dv_flag);

        // Pack nonzero values and form they into x axis and y axis.
        int32_t bitsNzIdx;
        vintx v_idx_lo;
        vcollate_idx_bits(dv_peak.lo, v_idx_lo, bitsNzIdx);
        int32_t count_lo = bitcount(bitsNzIdx);

        vintx v_idx_hi;
        vcollate_idx_bits(dv_peak.hi, v_idx_hi, bitsNzIdx);
        int32_t count_hi = bitcount(bitsNzIdx);

        vintx v_pos_x_lo;
        v_pos_x_lo                 = vpermute(dv_pos_x.lo, v_idx_lo);
        *(vint *)dopplerFoldIdxPtr = extract(v_pos_x_lo);
        dopplerFoldIdxPtr += count_lo;

        vintx v_pos_x_hi;
        v_pos_x_hi                 = vpermute(dv_pos_x.hi, v_idx_hi);
        *(vint *)dopplerFoldIdxPtr = extract(v_pos_x_hi);
        dopplerFoldIdxPtr += count_hi;

        *(dvint *)rangeIdxPtr = extract(dv_pos_y);
        rangeIdxPtr += count_lo + count_hi;

        count += count_lo + count_hi;

        dvintx dv_flag_start_new_row = dv_pos_x == dv_pos_x_end;
        dv_pos_x                     = dvmux(dv_flag_start_new_row, dv_pos_x_start, dv_pos_x + DVECW);
        dv_pos_y                     = dvmux(dv_flag_start_new_row, dv_pos_y + 1, dv_pos_y);
    }

    return count;
}

// The most time consuming part of this function is to calculate the sum of
// inPtr[ddmSpace[0]%REPEAT_FOLD] + inPtr[ddmSpace[1]%REPEAT_FOLD] + inPtr[ddmSpace[2]%REPEAT_FOLD] + inPtr[ddmSpace[3]%REPEAT_FOLD]
// This is achieved by the following steps:
// 1. Unpack the ddmSpace into the following form, (use ddmSpace={0, 7, 6, 4} as an example)
//    0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0
//    where 0 means the corresponding value in inPtr is used, 1 means the corresponding value in inPtr is not used.
//    Note that the unpacked sequence is of 2 * REPEAT_FOLD to avoid the periodicity access of ddmSpace.
// 2. Then take a position shift to get the bitmask position, e.g. 0, 1, 1, 1, 0, 1, 0, 0 mean position #0, #4, #6 and #7
//    Note that the above sequence is packed into a 32-bit integer, so that can leverage vbitunpackw to generate the per lane mask.
// 3. Use vmux to get the values to be accumulated, and then use vsumr_s to get the sum of the 4 values.
int32_t computeNciRxSumFindMax(uint32_t nciRxGatheredPtr[REPEAT_FOLD], uint32_t ddmSpaceBitMask)
{
    // Stores the sum of the 4 values in the DDMspace
    uint32_t sumBuf[REPEAT_FOLD];
    vintx v0 = zero_extend(*(vuint *)nciRxGatheredPtr);

    for (int32_t i = 0, j = REPEAT_FOLD - 1; i < REPEAT_FOLD; i++, j--)
    {
        vintx v_mask    = vbitunpackw(ddmSpaceBitMask);
        ddmSpaceBitMask = ddmSpaceBitMask >> 1;

        vintx v1  = vmux(v_mask, 0, v0);
        sumBuf[j] = vsumr_s(v1);
    }

    // Find index of the max value in the sumBuf.
    vintx v2 = zero_extend(*(vuint *)sumBuf);
    int32_t index, placeholder;
    vmaxrid_s(v2, placeholder, index);

    return index;
}

void computeFT2DGatherIndex(int32_t rangeIdx, int32_t dopplerIdx, const int32_t *tbl1024Ptr,
                            const int32_t *ddmSpaceTblPtr, int32_t *offsetFT2DPtr)
{
    dvintx dv0 = zero_extend(*(dvuint *)ddmSpaceTblPtr);
    dvintx dv1 = dv0 + dopplerIdx;
#if (NCIRX_WIDTH & (NCIRX_WIDTH - 1)) != 0
#    error NCIRX_WIDTH must be a power of 2 for modulo optimization to work correctly
#endif
    dvintx dv2 = dv1 & (NCIRX_WIDTH - 1); // %NCIRX_WIDTH
    dvintx dv3 = dv2 << 1;                // *NUM_INT_PER_CMPLX

    int32_t rowAddr = rangeIdx * (RX_COUNT * NCIRX_WIDTH) * NUM_INT_PER_CMPLX;
    dvintx dv4      = dv3 + rowAddr;

    dvintx dv5 = zero_extend(*(dvuint *)tbl1024Ptr);

    dvintx dv6 = dv5 + dv4;

    *(dvint *)offsetFT2DPtr = extract(dv6);
}

void peakDetectionNmsBorderInit(int32_t inPitch, int32_t tileY, int32_t cbSizeBytes, intptr_t cbStartAddr,
                                NmsBorderAgenParam *paramPtr)
{
    // The input tile height is tileY+2, but needs to skip the first and last row.
    paramPtr->niter = tileY;

    {
        AgenWrapper wrapper;
        wrapper.size      = sizeof(uint32_t);
        agen streamStart0 = init(NULL);
        wrapper.n1        = tileY;
        wrapper.s1        = inPitch;
        INIT_AGEN1(streamStart0, wrapper);
        streamStart0           = update_agen_cb_start(streamStart0, cbStartAddr);
        streamStart0           = update_agen_cb_size(streamStart0, cbSizeBytes);
        paramPtr->streamStart0 = extract_agen_cfg(streamStart0);
    }
    {
        AgenWrapper wrapper;
        wrapper.size      = sizeof(uint32_t);
        agen streamStart1 = init(NULL);
        wrapper.n1        = tileY;
        wrapper.s1        = inPitch;
        INIT_AGEN1(streamStart1, wrapper);
        streamStart1           = update_agen_cb_start(streamStart1, cbStartAddr);
        streamStart1           = update_agen_cb_size(streamStart1, cbSizeBytes);
        paramPtr->streamStart1 = extract_agen_cfg(streamStart1);
    }
    {
        AgenWrapper wrapper;
        wrapper.size       = sizeof(uint32_t);
        agen streamEndSub1 = init(NULL);
        wrapper.n1         = tileY;
        wrapper.s1         = inPitch;
        INIT_AGEN1(streamEndSub1, wrapper);
        streamEndSub1           = update_agen_cb_start(streamEndSub1, cbStartAddr);
        streamEndSub1           = update_agen_cb_size(streamEndSub1, cbSizeBytes);
        paramPtr->streamEndSub1 = extract_agen_cfg(streamEndSub1);
    }
    {
        AgenWrapper wrapper;
        wrapper.size       = sizeof(uint32_t);
        agen streamEndSub2 = init(NULL);
        wrapper.n1         = tileY;
        wrapper.s1         = inPitch;
        INIT_AGEN1(streamEndSub2, wrapper);
        streamEndSub2           = update_agen_cb_start(streamEndSub2, cbStartAddr);
        streamEndSub2           = update_agen_cb_size(streamEndSub2, cbSizeBytes);
        paramPtr->streamEndSub2 = extract_agen_cfg(streamEndSub2);
    }
}

// Fill the nciFinalBuf's left and right border with values from the opposite side.
void peakDetectionFillBorder(uint32_t *start0Ptr, uint32_t *start1Ptr, uint32_t *endSub1Ptr, uint32_t *endSub2Ptr,
                             NmsBorderAgenParam *paramPtr)
{
    cupvaModifyAgenCfgBase(&paramPtr->streamStart0, start0Ptr);
    cupvaModifyAgenCfgBase(&paramPtr->streamStart1, start1Ptr);
    cupvaModifyAgenCfgBase(&paramPtr->streamEndSub1, endSub1Ptr);
    cupvaModifyAgenCfgBase(&paramPtr->streamEndSub2, endSub2Ptr);

    agen streamStart0  = init_agen_from_cfg(paramPtr->streamStart0);
    agen streamStart1  = init_agen_from_cfg(paramPtr->streamStart1);
    agen streamEndSub1 = init_agen_from_cfg(paramPtr->streamEndSub1);
    agen streamEndSub2 = init_agen_from_cfg(paramPtr->streamEndSub2);

    const int32_t niter = paramPtr->niter;

    chess_separator_scheduler();

    for (int32_t i = 0; i < niter; i++) chess_loop_range(16, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        uint32_t endSub2 = uint_load(streamEndSub2);
        uint32_t start1  = uint_load(streamStart1);

        uint_store(endSub2, streamStart0);
        uint_store(start1, streamEndSub1);
    }
}

uint32_t gatherNciRxTileComputeDopplerIdx(SequenceDataFlowHandler &nciRxHdl, uint64_t nciRxBaseExtMemBase,
                                          uint32_t *nciRxTilePtr, uint32_t rangeIdx, uint32_t dopplerFoldIdx,
                                          uint32_t ddmSpaceBitMask)
{
    uint64_t src = nciRxBaseExtMemBase + rangeIdx * NCIRX_WIDTH * sizeof(uint32_t);
    cupvaSQDFUpdateAddr(nciRxHdl, 0, src, MEMTYPE_DRAM, cupvaGetVmemAddress(nciRxTilePtr), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(nciRxHdl);
    cupvaSQDFSync(nciRxHdl);

    uint32_t nciRxTileBufGather[REPEAT_FOLD];
    for (int32_t i = 0; i < REPEAT_FOLD; i++)
    {
        nciRxTileBufGather[i] = nciRxTilePtr[dopplerFoldIdx + NCIRX_WIDTH / REPEAT_FOLD * i];
    }
    int32_t index       = computeNciRxSumFindMax(nciRxTileBufGather, ddmSpaceBitMask);
    uint32_t dopplerIdx = dopplerFoldIdx + index * (NCIRX_WIDTH / REPEAT_FOLD);

    return dopplerIdx;
}

inline void updateIdxSnapAddrAndSize(SequenceDataFlowHandler &idxSnapHdl, uint64_t rangeIdxExtMemBase,
                                     uint64_t dopplerFoldIdxExtMemBase, uint64_t dopplerIdxExtMemBase,
                                     uint64_t snapExtMemBase, int32_t &extPeakBufLoc, int32_t index,
                                     PvaPeakTypeDevice *peakTilePtr)
{
    {
        uint64_t dst = rangeIdxExtMemBase + extPeakBufLoc * sizeof(uint32_t);
        cupvaSQDFUpdateAddr(idxSnapHdl, 0, cupvaGetVmemAddress(&peakTilePtr->rangeIdx[index]), MEMTYPE_VMEM, dst,
                            MEMTYPE_DRAM);
        cupvaSQDFUpdateTileSize(idxSnapHdl, 0, 1 * sizeof(uint32_t), 1);
    }
    {
        uint64_t dst = dopplerFoldIdxExtMemBase + extPeakBufLoc * sizeof(uint32_t);
        cupvaSQDFUpdateAddr(idxSnapHdl, 1, cupvaGetVmemAddress(&peakTilePtr->dopplerFoldIdx[index]), MEMTYPE_VMEM, dst,
                            MEMTYPE_DRAM);
        cupvaSQDFUpdateTileSize(idxSnapHdl, 1, 1 * sizeof(uint32_t), 1);
    }
    {
        uint64_t dst = dopplerIdxExtMemBase + extPeakBufLoc * sizeof(uint32_t);
        cupvaSQDFUpdateAddr(idxSnapHdl, 2, cupvaGetVmemAddress(&peakTilePtr->dopplerIdx[index]), MEMTYPE_VMEM, dst,
                            MEMTYPE_DRAM);
        cupvaSQDFUpdateTileSize(idxSnapHdl, 2, 1 * sizeof(uint32_t), 1);
    }
    {
        uint64_t dst = snapExtMemBase + extPeakBufLoc * TX_COUNT * RX_COUNT * NUM_INT_PER_CMPLX * sizeof(int32_t);
        cupvaSQDFUpdateAddr(idxSnapHdl, 3, cupvaGetVmemAddress(peakTilePtr->snap[index]), MEMTYPE_VMEM, dst,
                            MEMTYPE_DRAM);
        cupvaSQDFUpdateTileSize(idxSnapHdl, 3, TX_COUNT * RX_COUNT * NUM_INT_PER_CMPLX * sizeof(int32_t), 1);
    }
    extPeakBufLoc++;
}

void peakDetectionProcessPeaks(SequenceDataFlowHandler &idxSnapHdl, SequenceDataFlowHandler &nciRxHdl,
                               GSDFHandler *fT2DHdl, uint64_t nciRxBaseExtMemBase, uint64_t peakIndexExtMemBase,
                               uint64_t snapExtMemBase, int32_t &extPeakBufLoc, PvaPeakTypeDevice *peakTilePtr,
                               int32_t peakCount, uint32_t *nciRxTilePtr,
                               uint32_t (*fT2DTilePtr)[TX_COUNT * RX_COUNT * NUM_INT_PER_CMPLX],
                               const PeakDetectionHost2DevParam *vpuParam, int32_t &pingpongGSDF)
{
    if (peakCount == 0)
    {
        return;
    }

    peakTilePtr->dopplerIdx[0] =
        gatherNciRxTileComputeDopplerIdx(nciRxHdl, nciRxBaseExtMemBase, nciRxTilePtr, peakTilePtr->rangeIdx[0],
                                         peakTilePtr->dopplerFoldIdx[0], vpuParam->ddmSpaceBitMask);

    const uint64_t rangeIdxExtMemBase       = peakIndexExtMemBase;
    const uint64_t dopplerFoldIdxExtMemBase = peakIndexExtMemBase + PEAKDET_MAX_PEAK_COUNT * sizeof(uint32_t);
    const uint64_t dopplerIdxExtMemBase     = peakIndexExtMemBase + PEAKDET_MAX_PEAK_COUNT * 2 * sizeof(uint32_t);

    // Conditionally fetch the bins from FT2DBuf per range bin. Then rearrange the bins to the final output.
    int32_t offsetFT2D[TX_COUNT * RX_COUNT];
    computeFT2DGatherIndex(peakTilePtr->rangeIdx[0], peakTilePtr->dopplerIdx[0], vpuParam->tbl1024,
                           vpuParam->ddmSpaceTbl, offsetFT2D);

    cupvaGSDFUpdateOffsets1D(fT2DHdl, offsetFT2D);
    cupvaGSDFTrig(fT2DHdl);
    for (int32_t i = 0; i < peakCount; i++)
    {
        if (i > 0) // Only calcuate the i - 1 case, so need to skip when i == 0.
        {
            if (extPeakBufLoc > 0)
            {
                cupvaSQDFSync(idxSnapHdl);
            }

            updateIdxSnapAddrAndSize(idxSnapHdl, rangeIdxExtMemBase, dopplerFoldIdxExtMemBase, dopplerIdxExtMemBase,
                                     snapExtMemBase, extPeakBufLoc, i - 1, peakTilePtr);
            cupvaSQDFFlushAndTrig(idxSnapHdl);
        }

        if (i != (peakCount - 1))
        {
            peakTilePtr->dopplerIdx[i + 1] = gatherNciRxTileComputeDopplerIdx(
                nciRxHdl, nciRxBaseExtMemBase, nciRxTilePtr, peakTilePtr->rangeIdx[i + 1],
                peakTilePtr->dopplerFoldIdx[i + 1], vpuParam->ddmSpaceBitMask);
        }

        cupvaGSDFSync(fT2DHdl);
        if (i != (peakCount - 1))
        {
            computeFT2DGatherIndex(peakTilePtr->rangeIdx[i + 1], peakTilePtr->dopplerIdx[i + 1], vpuParam->tbl1024,
                                   vpuParam->ddmSpaceTbl, offsetFT2D);
            cupvaGSDFUpdateOffsets1D(fT2DHdl, offsetFT2D);
            cupvaGSDFTrig(fT2DHdl);
        }

        {
            const int32_t DVECW = chess_elementsof(dvint);
            dvint *srcPtr       = (dvint *)&fT2DTilePtr[pingpongGSDF][0];
            dvint *dstPtr       = (dvint *)&peakTilePtr->snap[i][0];
            for (int32_t j = 0; j < NUM_INT_PER_CMPLX * TX_COUNT * RX_COUNT / DVECW; j++)
            {
                dvintx dv0 = sign_extend(*srcPtr++);
                *dstPtr++  = extract(dv0);
            }
        }

        pingpongGSDF ^= 1;
    }

    // Wait for the previous peak's transfer, as in the loop. With a single peak in the first batch nothing has
    // been triggered yet, and syncing an idle handler never returns.
    if (extPeakBufLoc > 0)
    {
        cupvaSQDFSync(idxSnapHdl);
    }
    updateIdxSnapAddrAndSize(idxSnapHdl, rangeIdxExtMemBase, dopplerFoldIdxExtMemBase, dopplerIdxExtMemBase,
                             snapExtMemBase, extPeakBufLoc, peakCount - 1, peakTilePtr);
    cupvaSQDFFlushAndTrig(idxSnapHdl);
}
