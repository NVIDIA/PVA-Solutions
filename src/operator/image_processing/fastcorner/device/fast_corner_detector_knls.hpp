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

#include "fast_corner_detector_param.hpp"

#include <cupva_device.h>

typedef struct
{
    AgenCFG streamCircleSeg0;
    AgenCFG streamCircleSeg1;
    AgenCFG streamCircleCenter;
    int32_t niter;
} QuickRejAgenParam;

typedef struct
{
    AgenCFG streamCircleSeg0;
    AgenCFG streamCircleSeg1;
    AgenCFG streamCircleSeg2;
    AgenCFG streamCircleCenter;
    AgenCFG streamDst;
    AgenCFG streamZero;
    int32_t niter;
    int32_t niterNonRejVec;
} ScoreAgenParam;

typedef struct
{
    AgenCFG streamIn;
    AgenCFG streamCenter;
    int32_t niterX;
    int32_t niter;
    uint16_t *laneId;
} NmsPackAgenParam;

#ifdef TOPK_ON
typedef struct
{
    AgenCFG scoreAgen;
    AgenCFG locAgen;
} FilterAgenParam;

typedef struct
{
    AgenCFG histVectorAgen;
} HistReduceAgenParam;
#endif // TOPK_ON

void fastCornerDetectorQuickRejAgenInit(int32_t inPitch, int32_t tileX, int32_t tileY, int32_t cbSizeBytes,
                                        intptr_t cb0StartAddr, intptr_t cb1StartAddr, intptr_t cb2StartAddr,
                                        QuickRejAgenParam *params);

uint64_t fastCornerDetectorQuickRej(BASE_TYPE *seg0Ptr, BASE_TYPE *seg1Ptr, BASE_TYPE *centerPtr, int32_t threshold,
                                    QuickRejAgenParam *params);

void fastCornerDetectorScoreNoRejAgenInit(int32_t inPitch, int32_t outPitch, int32_t tileX, int32_t tileY,
                                          int32_t cbSizeBytes, intptr_t cb0StartAddr, intptr_t cb1StartAddr,
                                          intptr_t cb2StartAddr, ScoreAgenParam *params);

void fastCornerDetectorScorePartialRejAgenInit(int32_t inPitch, int32_t outPitch, int32_t tileX, int32_t tileY,
                                               int32_t cbSizeBytes, intptr_t cb0StartAddr, intptr_t cb1StartAddr,
                                               intptr_t cb2StartAddr, ScoreAgenParam *params);

void fastCornerDetectorScoreNoRej(BASE_TYPE *seg0Ptr, BASE_TYPE *seg1Ptr, BASE_TYPE *seg2Ptr, BASE_TYPE *centerPtr,
                                  BASE_TYPE *outPtr, int32_t threshold, ScoreAgenParam *params);

int32_t fastCornerDetectorCalVecBaseAdv(uint64_t rejBitMask, int32_t inputLinePitch,
                                        const int32_t inputVecOffsetInTile[], const int32_t outputVecOffsetInTile[],
                                        int32_t inputBaseAdv[NUM_DVEC_PER_TILE],
                                        int32_t outputBaseAdv[NUM_DVEC_PER_TILE]);

void fastCornerDetectorScorePartialRej(BASE_TYPE *seg0Ptr, BASE_TYPE *seg1Ptr, BASE_TYPE *seg2Ptr, BASE_TYPE *centerPtr,
                                       BASE_TYPE *outPtr, int32_t threshold, ScoreAgenParam *params,
                                       int32_t inputBaseAdv[NUM_DVEC_PER_TILE],
                                       int32_t outputBaseAdv[NUM_DVEC_PER_TILE]);

void convertPixelToF32(uint32_t *src, int32_t size, float *dst);

void writeExtBuf(SequenceDataFlowHandler &sqdfHdl, bool isFirstWrite, float *src, int32_t size, uint64_t dst);

void convertPixelToF32WriteExtBufWrapper(SequenceDataFlowHandler &locSQDFHdl, float locBuf[], int32_t &extBufLocCnt,
                                         int32_t localBufLocCnt, int32_t locCapacity, int32_t pingpong,
                                         uint32_t *packBuf, const ExtMemPointer &locExtPtr);

#ifdef TOPK_ON
// TopK histogram and filtering functions
void InitHistograms(int16_t *histVectorBuf, int32_t *histScalarBuf, HistReduceAgenParam *params);
void ReduceVectorToScalar(int32_t *histScalarBuf, HistReduceAgenParam *params);
uint16_t ComputeThreshold(const int32_t *histScalarBuf, const int32_t topK);
void filterAgenInit(int32_t maxBatchSize, uint16_t *packScoreBuf, uint32_t *packBuf, FilterAgenParam *params);
int32_t filterPackedFeaturesByThresholdVPU(uint32_t *packedLocs, uint16_t *packedScores, int32_t numPacked,
                                           uint16_t threshold, uint32_t *outputLocs, FilterAgenParam *params);

// TopK SQDF helpers
void writePackedDataSQDF(SequenceDataFlowHandler &handler, uint32_t *packBuf, uint16_t *packScoreBuf, int32_t count,
                         int32_t offset, int32_t pingpong, const ExtMemPointer &packedLocsExtPtr,
                         const ExtMemPointer &packedScoresExtPtr, bool isFirstWrite);

void readPackedDataSQDF(SequenceDataFlowHandler &handler, uint32_t *packBuf, uint16_t *packScoreBuf, int32_t count,
                        int32_t offset, int32_t readBufOffset, const ExtMemPointer &packedLocsExtPtr,
                        const ExtMemPointer &packedScoresExtPtr);
#endif // TOPK_ON

#ifdef NMS_ON
void fastCornerDetectorNmsPackAgenInit(int32_t pitch, int32_t tileX, int32_t tileY, int32_t cbSizeBytes,
                                       intptr_t cbStartAddr, NmsPackAgenParam *params, uint16_t *laneId);

int32_t fastCornerDetectorNmsPack(BASE_TYPE *tilePtr, BASE_TYPE *centerPtr, NmsPackAgenParam *params, int32_t tilePosX,
                                  int32_t tilePosY, uint32_t chess_storage(RAM_Bb) *restrict packBufPtr
#    ifdef TOPK_ON
                                  ,
                                  uint16_t *restrict packScoreBufPtr, int16_t *histVectorBuf
#    endif // TOPK_ON
);
#else
void fastCornerDetectorPackAgenInit(BASE_TYPE *tilePtr, int32_t pitch, int32_t tileX, int32_t tileY,
                                    NmsPackAgenParam *params, uint16_t *laneId);

// Pack detected corners from score buffer (NMS OFF mode).
// hasLastTileX: when true, performs width boundary checking for partial last column tiles.
template<bool hasLastTileX>
int32_t fastCornerDetectorPack(NmsPackAgenParam *params, int32_t tilePosX, int32_t tilePosY, int32_t width,
                               int32_t lastTileY, uint32_t *restrict packBufPtr
#    ifdef TOPK_ON
                               ,
                               uint16_t *restrict packScoreBufPtr, int16_t *histVectorBuf
#    endif // TOPK_ON
)
{
    agen streamCenter = init_agen_from_cfg(params->streamCenter);

    chess_separator_scheduler();

    const int32_t niter     = lastTileY * params->niterX;
    dvshortx dv_lane_id     = zero_extend(*(dvushort *)params->laneId);
    dvshortx dv_pos_x_start = dv_lane_id + tilePosX;
    dvshortx dv_pos_x_end   = dv_pos_x_start + (params->niterX - 1) * LANE_ID_TBL_SZ;
    dvshortx dv_pos_x       = dv_pos_x_start;
    vshortx v_pos_y         = replicateh(tilePosY);

    dvshortx dv_width;
    dvshortx dv_zero;
#    if defined(TOPK_ON) && (defined(U16_IO) || defined(S16_IO))
    dvshortx dv_hist_zero = chess_dont_care(dvshortx) & 0;
#    endif
    if constexpr (hasLastTileX)
    {
        dv_zero     = chess_dont_care(dvshortx) & 0;
        dv_width.lo = replicateh(width);
        dv_width.hi = replicateh(width);
    }

#    ifdef TOPK_ON
    dvshortx dv_weight;
    dv_weight.lo = replicateh(1);
    dv_weight.hi = replicateh(1);
#    endif // TOPK_ON

    int32_t numLoc = 0;

    for (int32_t i = 0; i < niter; i++) chess_loop_range(1, )
    chess_prepare_for_pipelining
    {
        // Input is always unsigned.
#    if defined(U16_IO) || defined(S16_IO)
        DVEC_TYPE dv_score = dvushort_load(streamCenter);
#    elif defined(U8_IO) || defined(S8_IO)
        // Read 32 uint8_t per iteration, since vcollate_idx_bits does not have char version and need to promote to short.
        dvshortx dv_score = vuchar_dvshortx_load(streamCenter);
#    endif

        if constexpr (hasLastTileX)
        {
            dvshortx dv_larger_width_flag;

            dv_larger_width_flag = dv_pos_x < dv_width;
            dv_score             = dvmux(dv_larger_width_flag, dv_score, dv_zero);
        }

        // Inline histogram accumulation (TopK only)
#    ifdef TOPK_ON
#        if defined(U8_IO) || defined(S8_IO)
        vhist_simple_32h(histVectorBuf, dv_score, dv_weight);
#        elif defined(U16_IO) || defined(S16_IO)
        dvshortx dv_bins    = dv_score >> 8;
        dvshortx dv_nonzero = dv_score > dv_hist_zero;
        dvshortx dv_hist_weight;
        dv_hist_weight.lo = dv_weight.lo & dv_nonzero.lo;
        dv_hist_weight.hi = dv_weight.hi & dv_nonzero.hi;
        vhist_simple_32h(histVectorBuf, dv_bins, dv_hist_weight);
#        endif
#    endif // TOPK_ON

        // The following is to pack nonzero values within dv_score.
        int bitsNzIdx;
        vshortx v_idx_lo;
        vcollate_idx_bits(dv_score.lo, v_idx_lo, bitsNzIdx);
        int32_t count_lo = bitcount(bitsNzIdx);

        vshortx v_idx_hi;
        vcollate_idx_bits(dv_score.hi, v_idx_hi, bitsNzIdx);
        int32_t count_hi = bitcount(bitsNzIdx);

        vshortx v_x_lo = vpermute(dv_pos_x.lo, v_idx_lo);
        vshortx v_x_hi = vpermute(dv_pos_x.hi, v_idx_hi);

        dvintx dv_x_int_lo, dv_x_int_hi;
        vpromote(v_x_lo, dv_x_int_lo.lo, dv_x_int_lo.hi);
        vpromote(v_x_hi, dv_x_int_hi.lo, dv_x_int_hi.hi);

        dvintx dv_y_int;
        vpromote_di(v_pos_y, dv_y_int.lo, dv_y_int.hi);

        dvintx dv_loc;
        dv_loc                                     = dvshiftadd(dv_y_int, 16, dv_x_int_lo);
        *(dvint chess_storage(RAM_Bb) *)packBufPtr = extract(dv_loc);
        packBufPtr                                 = (uint32_t *)packBufPtr + count_lo;

        dv_loc                                     = dvshiftadd(dv_y_int, 16, dv_x_int_hi);
        *(dvint chess_storage(RAM_Bb) *)packBufPtr = extract(dv_loc);
        packBufPtr                                 = (uint32_t *)packBufPtr + count_hi;

        // Pack scores alongside locations (TopK only)
#    ifdef TOPK_ON
        vshortx v_score_lo = vpermute(dv_score.lo, v_idx_lo);
        vshortx v_score_hi = vpermute(dv_score.hi, v_idx_hi);

        *(vshort chess_storage(RAM_Bb) *)packScoreBufPtr = extract(v_score_lo);
        packScoreBufPtr                                  = (uint16_t *)packScoreBufPtr + count_lo;
        *(vshort chess_storage(RAM_Bb) *)packScoreBufPtr = extract(v_score_hi);
        packScoreBufPtr                                  = (uint16_t *)packScoreBufPtr + count_hi;
#    endif // TOPK_ON

        numLoc += count_lo + count_hi;

        dvshortx dv_flag_start_new_row = dv_pos_x == dv_pos_x_end;
        dv_pos_x                       = dvmux(dv_flag_start_new_row, dv_pos_x_start, dv_pos_x + LANE_ID_TBL_SZ);
        v_pos_y                        = vmux(dv_flag_start_new_row.lo, v_pos_y + 1, v_pos_y);
    }

    return numLoc;
}
#endif     // NMS_ON
