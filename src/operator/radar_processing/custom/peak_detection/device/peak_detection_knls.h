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

#include "peak_detection_param.h"

#include <cupva_device.h>

typedef struct
{
    AgenCFG streamIn0;
    AgenCFG streamIn1;
    AgenCFG streamCenter;
    int32_t niter;
    int32_t niterX;
    int32_t niterY;
} NmsAgenParam;

typedef struct
{
    AgenCFG streamStart0;
    AgenCFG streamStart1;
    AgenCFG streamEndSub1;
    AgenCFG streamEndSub2;
    int32_t niter;
} NmsBorderAgenParam;

void peakDetectionNmsBorderInit(int32_t inPitch, int32_t tileY, int32_t cbSizeBytes, intptr_t cbStartAddr,
                                NmsBorderAgenParam *paramPtr);

void peakDetectionFillBorder(uint32_t *start0Ptr, uint32_t *start1Ptr, uint32_t *endSub1Ptr, uint32_t *endSub2Ptr,
                             NmsBorderAgenParam *paramPtr);

void peakDetectionNmsInit(int32_t inPitch, int32_t tileX, int32_t tileY, int32_t cbSizeBytes, intptr_t cbStartAddr,
                          NmsAgenParam *paramPtr);

int32_t peakDetectionNms(uint32_t *in0Ptr, uint32_t *in1Ptr, uint32_t *centerPtr, const uint32_t *noiseEstPtr,
                         const uint32_t *laneIdPtr, NmsAgenParam *paramPtr, int32_t tilePosY, uint32_t *rangeIdxPtr,
                         uint32_t *dopplerFoldIdxPtr);

int32_t computeNciRxSumFindMax(uint32_t nciRxGatheredPtr[REPEAT_FOLD], uint32_t ddmSpaceBitMask);

void computeFT2DGatherIndex(int32_t rangeIdx, int32_t dopplerIdx, const int32_t *tbl1024Ptr,
                            const int32_t *ddmSpaceTblPtr, int32_t *offsetFT2DPtr);

void peakDetectionProcessPeaks(SequenceDataFlowHandler &idxSnapHdl, SequenceDataFlowHandler &nciRxHdl,
                               GSDFHandler *fT2DHdl, uint64_t nciRxBaseExtMemBase, uint64_t peakIndexExtMemBase,
                               uint64_t snapExtMemBase, int32_t &extPeakBufLoc, PvaPeakTypeDevice *peakTilePtr,
                               int32_t peakCount, uint32_t *nciRxTilePtr,
                               uint32_t (*fT2DTilePtr)[TX_COUNT * RX_COUNT * NUM_INT_PER_CMPLX],
                               const PeakDetectionHost2DevParam *vpuParam, int32_t &pingpongGSDF);
