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
#include "peak_detection_param.h"

#include <cupva_device.h>

// Dataflow handlers.
VMEM_RDF_UNIFIED(A, nciFinalHdl);
VMEM_RDF_UNIFIED(B, noiseEstHdl);

VMEM_SEQDF_HANDLER(A, nciRxHdl);
VMEM_POINTER(A, nciRxBaseExtMemPtr);

VMEM_GSDF_HANDLER(B, fT2DHdl, TX_COUNT *RX_COUNT);

VMEM_SEQDF_HANDLER(A, peakCountHdl);
VMEM_SEQDF_HANDLER(A, idxSnapHdl);

VMEM_POINTER(A, peakIndexExtMemPtr);
VMEM_POINTER(A, snapExtMemPtr);

// Input buffers.
VMEM(A, uint32_t, nciFinalBuf, RDF_CIRCULAR(uint32_t, TILE_X, TILE_Y, HALO_X, HALO_Y));
VMEM(B, uint32_t, noiseEstBuf, TILE_Y);
VMEM(A, uint32_t, nciRxTileBuf, NCIRX_WIDTH);
VMEM(A, uint32_t, fT2DTileBuf, 2, TX_COUNT *RX_COUNT *NUM_INT_PER_CMPLX); // The first 2 is for ping-pong.

VMEM(B, PeakDetectionHost2DevParam, vpuParam);

// Output buffers.
VMEM(A, int32_t, peakCountVem2Dram);
VMEM(C, PvaPeakTypeDevice, peakBuf);
CUPVA_EXPORT(peakBuf.rangeIdx, rangeIdxExport);
CUPVA_EXPORT(peakBuf.dopplerFoldIdx, dopplerFoldIdxExport);
CUPVA_EXPORT(peakBuf.dopplerIdx, dopplerIdxExport);
CUPVA_EXPORT(peakBuf.snap, snapExport);

// Device side variables.
VMEM(A, NmsBorderAgenParam, nmsBorderAgenParam);

VMEM(A, NmsAgenParam, nmsAgenParam);

VMEM(A, uint32_t, laneIdU32Buf, pva_elementsof(dvint));

CUPVA_VPU_MAIN()
{
    // Init the parameters.
    const int32_t tileX  = TILE_X;
    const int32_t tileY  = TILE_Y;
    const int32_t niterY = vpuParam.niterY;

    for (uint32_t i = 0; i < pva_elementsof(dvint); i++)
    {
        laneIdU32Buf[i] = i;
    }

    cupvaRasterDataFlowOpen(nciFinalHdl, nciFinalBuf);
    cupvaRasterDataFlowOpen(noiseEstHdl, noiseEstBuf);

    int32_t nciFinalInCbLen = cupvaRasterDataFlowGetCbLen(nciFinalHdl);
    int32_t nciFinalInPitch = cupvaRasterDataFlowGetLinePitch(nciFinalHdl);

    peakDetectionNmsBorderInit(nciFinalInPitch, tileY, nciFinalInCbLen * sizeof(uint32_t), (uintptr_t)nciFinalBuf,
                               &nmsBorderAgenParam);

    peakDetectionNmsInit(nciFinalInPitch, tileX, tileY, nciFinalInCbLen * sizeof(uint32_t), (uintptr_t)nciFinalBuf,
                         &nmsAgenParam);

    // Calculate the NMS on the nciFinalBuf.
    uint32_t *rangeIdxWrPtr       = peakBuf.rangeIdx;
    uint32_t *dopplerFoldIdxWrPtr = peakBuf.dopplerFoldIdx;
    int32_t peakCount             = 0; // Locally accumulated peak count.
    int32_t extPeakBufLoc         = 0;
    int32_t pingpongGSDF          = 0;
    cupvaSQDFOpen(nciRxHdl);
    cupvaGSDFOpen(fT2DHdl, fT2DTileBuf[0], 2, 2);
    cupvaGSDFCycleBuffers(fT2DHdl);
    cupvaGSDFUpdateVmemAddr(fT2DHdl, fT2DTileBuf[1], 2, 2);
    cupvaGSDFCycleBuffers(fT2DHdl);
    cupvaSQDFOpen(idxSnapHdl);
    const uint64_t peakIndexExtMemBase = peakIndexExtMemPtr.base + peakIndexExtMemPtr.offset;
    const uint64_t snapExtMemBase      = snapExtMemPtr.base + snapExtMemPtr.offset;
    const uint64_t nciRxBaseExtMemBase = nciRxBaseExtMemPtr.base + nciRxBaseExtMemPtr.offset;
    for (int32_t i = 0, tilePosY = 0; i < niterY; i++, tilePosY += tileY)
    {
        ptrdiff_t offset      = (uint32_t *)cupvaRasterDataFlowAcquire(nciFinalHdl) - nciFinalBuf;
        uint32_t *noiseEstPtr = (uint32_t *)cupvaRasterDataFlowAcquire(noiseEstHdl);

        cupvaCircularBufferMemcpy(nciFinalBuf, nciFinalInCbLen * sizeof(uint32_t));

        ptrdiff_t start0Offset = offset + (1 * nciFinalInPitch + 0);
        uint32_t *start0Ptr =
            &nciFinalBuf[start0Offset >= nciFinalInCbLen ? (start0Offset - nciFinalInCbLen) : start0Offset];
        ptrdiff_t start1Offset = offset + (1 * nciFinalInPitch + 1);
        uint32_t *start1Ptr =
            &nciFinalBuf[start1Offset >= nciFinalInCbLen ? (start1Offset - nciFinalInCbLen) : start1Offset];
        ptrdiff_t endSub1Offset = offset + (1 * nciFinalInPitch + nciFinalInPitch - 1);
        uint32_t *endSub1Ptr =
            &nciFinalBuf[endSub1Offset >= nciFinalInCbLen ? (endSub1Offset - nciFinalInCbLen) : endSub1Offset];
        ptrdiff_t endSub2Offset = offset + (1 * nciFinalInPitch + nciFinalInPitch - 2);
        uint32_t *endSub2Ptr =
            &nciFinalBuf[endSub2Offset >= nciFinalInCbLen ? (endSub2Offset - nciFinalInCbLen) : endSub2Offset];

        peakDetectionFillBorder(start0Ptr, start1Ptr, endSub1Ptr, endSub2Ptr, &nmsBorderAgenParam);

        uint32_t *nciFinalSeg1Ptr    = start0Ptr;
        uint32_t *nciFinalCenterPtr  = start1Ptr;
        ptrdiff_t nciFinalSeg0Offset = offset + (0 * nciFinalInPitch + 1);
        uint32_t *nciFinalSeg0Ptr =
            &nciFinalBuf[nciFinalSeg0Offset >= nciFinalInCbLen ? (nciFinalSeg0Offset - nciFinalInCbLen)
                                                               : nciFinalSeg0Offset];
        // NMS operation and save the rangeIdx and dopplerFoldIdx
        int32_t peakCountPerTile =
            peakDetectionNms(nciFinalSeg0Ptr, nciFinalSeg1Ptr, nciFinalCenterPtr, noiseEstPtr, laneIdU32Buf,
                             &nmsAgenParam, tilePosY, rangeIdxWrPtr, dopplerFoldIdxWrPtr);

        // If after adding the current tile's peak count, the total peak count exceeds the half of the output tile,
        // then needs to process the peaks. After processing the peaks, the local buffer "peakBuf" and its attributes
        // are reset.
        if (peakCount + peakCountPerTile > OUTPUT_TILE_Y / 2)
        {
            peakCount = min(peakCount + peakCountPerTile, PEAKDET_MAX_PEAK_COUNT - extPeakBufLoc);
            peakDetectionProcessPeaks(idxSnapHdl, nciRxHdl, fT2DHdl, nciRxBaseExtMemBase, peakIndexExtMemBase,
                                      snapExtMemBase, extPeakBufLoc, &peakBuf, peakCount, nciRxTileBuf, fT2DTileBuf,
                                      &vpuParam, pingpongGSDF);

            peakCount           = 0;
            rangeIdxWrPtr       = peakBuf.rangeIdx;
            dopplerFoldIdxWrPtr = peakBuf.dopplerFoldIdx;
        }
        else
        {
            // peakCount == 0 case is also handled here.
            peakCount += peakCountPerTile;
            rangeIdxWrPtr += peakCountPerTile;
            dopplerFoldIdxWrPtr += peakCountPerTile;
        }

        cupvaRasterDataFlowRelease(noiseEstHdl);
        cupvaRasterDataFlowRelease(nciFinalHdl);
    }

    cupvaRasterDataFlowClose(nciFinalHdl);
    cupvaRasterDataFlowClose(noiseEstHdl);

    // If there are still some peaks not being transferred to DRAM, i.e. accumulated peakCount is larger than 0,
    // but less than the threshold within the kernel loop, then process them.
    if (peakCount > 0 && PEAKDET_MAX_PEAK_COUNT - extPeakBufLoc > 0)
    {
        peakCount = min(peakCount, PEAKDET_MAX_PEAK_COUNT - extPeakBufLoc);
        peakDetectionProcessPeaks(idxSnapHdl, nciRxHdl, fT2DHdl, nciRxBaseExtMemBase, peakIndexExtMemBase,
                                  snapExtMemBase, extPeakBufLoc, &peakBuf, peakCount, nciRxTileBuf, fT2DTileBuf,
                                  &vpuParam, pingpongGSDF);
    }

    cupvaSQDFClose(nciRxHdl);
    cupvaGSDFClose(fT2DHdl);

    peakCountVem2Dram = extPeakBufLoc;
    cupvaSQDFOpen(peakCountHdl);
    cupvaSQDFTrig(peakCountHdl);
    cupvaSQDFSync(peakCountHdl);
    cupvaSQDFClose(peakCountHdl);

    if (extPeakBufLoc > 0)
    {
        cupvaSQDFSync(idxSnapHdl);
    }
    cupvaSQDFClose(idxSnapHdl);

    return 0;
}