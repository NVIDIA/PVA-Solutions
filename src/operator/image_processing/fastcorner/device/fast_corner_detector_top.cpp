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

#include "fast_corner_detector_knls.hpp"
#include "fast_corner_detector_param.hpp"

#include <cupva_device.h>

VMEM_RDF_UNIFIED(A, srcHdl);
#ifdef NMS_ON
VMEM_RDF_UNIFIED(A, scoreHdl);
VMEM_RDF_UNIFIED(A, nmsInTrig);
#endif // NMS_ON

VMEM_SEQDF_HANDLER(A, locCntSQDFHdl);
VMEM_SEQDF_HANDLER(A, locSQDFHdl);

#ifdef TOPK_ON
// TopK SQDF handlers (2 transfers each: packed locations and scores)
VMEM_SEQDF_HANDLER(A, packedDataWriteHdl);
VMEM_SEQDF_HANDLER(A, packedDataReadHdl);
#endif // TOPK_ON

VMEM(A, BASE_TYPE, scoreInBuf0, RDF_CIRCULAR(BASE_TYPE, TILE_X, TILE_Y, HALO_X, HALO_Y));
VMEM(B, BASE_TYPE, scoreInBuf1, RDF_CIRCULAR(BASE_TYPE, TILE_X, TILE_Y, HALO_X, HALO_Y));
VMEM(C, BASE_TYPE, scoreInBuf2, RDF_CIRCULAR(BASE_TYPE, TILE_X, TILE_Y, HALO_X, HALO_Y));
// PINGPONG buffer used for the calculated score
VMEM(B, BASE_TYPE, scoreBuf, RDF_DOUBLE(BASE_TYPE, TILE_X, TILE_Y));

#ifdef NMS_ON
VMEM(A, BASE_TYPE, nmsInBuf, RDF_CIRCULAR(BASE_TYPE, TILE_X, TILE_Y, 1, 1));
#endif // NMS_ON

VMEM(A, int32_t, locCntBuf, 1);

// external memory pointer
VMEM_POINTER(A, locExtPtr);

#ifdef TOPK_ON
// TopK external memory pointers for packed data
VMEM_POINTER(A, packedLocsExtPtr);
VMEM_POINTER(A, packedScoresExtPtr);
#endif // TOPK_ON

// The first 2 is for pingpong, the second 2 is for two float.
VMEM(C, float, locBuf, 2 * 2 * LOC_BUF_SIZE);

#ifdef TOPK_ON
// TopK mode: pingpong buffers sized per tile
VMEM(B, uint32_t, packBuf, 2 * TOPK_LOC_BUF_SIZE);
VMEM(A, uint16_t, packScoreBuf, 2 * TOPK_LOC_BUF_SIZE);
// Histogram buffers
VMEM(B, int16_t, histVectorBuf, HISTOGRAM_VECTOR_WIDTH *HIST_BINS);
VMEM(C, int32_t, histScalarBuf, HIST_BINS);
#else
// Non-TopK mode: larger buffer for batched writes
VMEM(B, uint32_t, packBuf, LOC_BUF_SIZE);
#endif // TOPK_ON

// laneId is always of type uint16_t [32]
uint16_t chess_storage(DMb % 2 : chess_segment(A)) laneId[LANE_ID_TBL_SZ];

// Offsets within a tile, which are used to update the agen's base address, always to be int32_t type.
int32_t chess_storage(DMb % 4 : chess_segment(A)) inputVecOffsetInTile[pva_elementsof(vintx)];
int32_t chess_storage(DMb % 4 : chess_segment(A)) outputVecOffsetInTile[pva_elementsof(vintx)];

#ifdef NMS_ON
// One byte for one tile. Enlarge this buffer size if the number of tiles exceeds the current buffer size.
VMEM(C, uint8_t, tileRejFlagBuf, 1024);
#endif // NMS_ON

// threshold needs to be specified during the submit phase, so it is a separate VMEM variable.
VMEM(A, int32_t, threshold);

// Set during the submission phase.
VMEM(A, int32_t, locCapacity);

VMEM(A, ScoreAgenParam, scoreNoQuickRejAgen);
VMEM(A, ScoreAgenParam, scoreAgen);

// Agen parameters used for either NMS is ON or OFF.
VMEM(A, NmsPackAgenParam, nmsPackAgen);

VMEM(A, QuickRejAgenParam, scoreQuickRejAgen);

#ifdef TOPK_ON
// TopK-specific AGEN parameters
VMEM(A, FilterAgenParam, filterAgenParam);
VMEM(A, HistReduceAgenParam, histReduceAgenParam);
#endif // TOPK_ON

VMEM(B, FastcornerHost2DevParam, vpuParam);

CUPVA_VPU_MAIN()
{
    const int32_t width  = vpuParam.width;
    const int32_t height = vpuParam.height;
    const int32_t tileX  = TILE_X;
    const int32_t tileY  = TILE_Y;
    const int32_t niterX = vpuParam.niterX;
    const int32_t niterY = vpuParam.niterY;

#ifdef TOPK_ON
    const int32_t packedCapacity = MAX_LOC_CAPACITY;
    const int32_t maxFeatures    = vpuParam.numTopKFeatures;
#endif // TOPK_ON

    int32_t scoreInCbLen  = cupvaRasterDataFlowGetCbLen(srcHdl);
    int32_t scoreInPitch  = cupvaRasterDataFlowGetLinePitch(srcHdl);
    int32_t scoreOutPitch = tileX;

    fastCornerDetectorQuickRejAgenInit(scoreInPitch, tileX, tileY, scoreInCbLen * sizeof(BASE_TYPE),
                                       (uintptr_t)scoreInBuf0, (uintptr_t)scoreInBuf1, (uintptr_t)scoreInBuf2,
                                       &scoreQuickRejAgen);

    fastCornerDetectorScoreNoRejAgenInit(scoreInPitch, scoreOutPitch, tileX, tileY, scoreInCbLen * sizeof(BASE_TYPE),
                                         (uintptr_t)scoreInBuf0, (uintptr_t)scoreInBuf1, (uintptr_t)scoreInBuf2,
                                         &scoreNoQuickRejAgen);

    fastCornerDetectorScorePartialRejAgenInit(scoreInPitch, scoreOutPitch, tileX, tileY,
                                              scoreInCbLen * sizeof(BASE_TYPE), (uintptr_t)scoreInBuf0,
                                              (uintptr_t)scoreInBuf1, (uintptr_t)scoreInBuf2, &scoreAgen);

    // pack
    int32_t extBufLocCnt   = 0;
    int32_t localBufLocCnt = 0;
    int32_t pingpong       = 0;
    cupvaSQDFOpen(locCntSQDFHdl);
    cupvaSQDFOpen(locSQDFHdl);

#ifdef TOPK_ON
    int32_t totalPackedCount = 0;
    int32_t tilesProcessed   = 0;
    bool isFirstWrite        = true;
    int32_t pingpongWrite    = 0;

    InitHistograms(histVectorBuf, histScalarBuf, &histReduceAgenParam);

    cupvaSQDFOpen(packedDataWriteHdl);
    cupvaSQDFOpen(packedDataReadHdl);
#endif // TOPK_ON

    cupvaRasterDataFlowOpen(srcHdl, scoreInBuf0);
#ifdef NMS_ON
    cupvaRasterDataFlowOpen(scoreHdl, scoreBuf);
#else
    fastCornerDetectorPackAgenInit(scoreBuf, scoreOutPitch, tileX, tileY, &nmsPackAgen, laneId);
#endif // NMS_ON

    // Initiate a set of offsets within the tile, used for calculating the actual offsets of non rejected vectors.
    const int32_t numVecPerRow = TILE_X / DVECW;
    for (int32_t i = 0; i < pva_elementsof(vintx); i++)
    {
        int32_t y = i / numVecPerRow;
        int32_t x = i % numVecPerRow;
        // Both offsets are in bytes unit and output offset is continuous so that line pitch can be skipped.
        inputVecOffsetInTile[i]  = (y * scoreInPitch + x * DVECW) * sizeof(BASE_TYPE);
        outputVecOffsetInTile[i] = i * DVECW * sizeof(BASE_TYPE);
    }

    int32_t inputBaseAdv[NUM_DVEC_PER_TILE], outputBaseAdv[NUM_DVEC_PER_TILE];
    /*
    * Need to calculate a tile's starting x and y axis, so cannot merge the two loops into one, otherwise i%niterX
    * is needed, which is quite costly.
    */
    int32_t tilePosY  = 0;
    int32_t lastTileY = height % tileY;
    bool hasLastTileX = width % tileX != 0;
    bool hasLastTileY = height % tileY != 0;
    for (int32_t i = 0; i < niterY; i++)
    {
        int32_t tilePosX   = 0;
        int32_t _lastTileY = (hasLastTileY && (i == (niterY - 1))) ? lastTileY : tileY;
        for (int32_t j = 0; j < niterX; j++)
        {
            // scoreInBuf0 ~ scoreInBuf2 have the same geometry, so that can use the same offset.
            ptrdiff_t offset = (BASE_TYPE *)cupvaRasterDataFlowAcquire(srcHdl) - scoreInBuf0;

            cupvaCircularBufferMemcpy(scoreInBuf1, scoreInCbLen * sizeof(BASE_TYPE));
            cupvaCircularBufferMemcpy(scoreInBuf2, scoreInCbLen * sizeof(BASE_TYPE));

#ifdef NMS_ON
            BASE_TYPE *scoreOutPtr = (BASE_TYPE *)cupvaRasterDataFlowAcquire(scoreHdl);
#else
            // If NMS is off, then use only the PING part of scoreBuf as the scratch buffer.
            BASE_TYPE *scoreOutPtr = scoreBuf;
#endif // NMS_ON

            BASE_TYPE *rejSeg0Ptr   = &scoreInBuf0[(offset + (0 * scoreInPitch + 3)) % scoreInCbLen];
            BASE_TYPE *rejSeg1Ptr   = &scoreInBuf1[(offset + (3 * scoreInPitch + 0)) % scoreInCbLen];
            BASE_TYPE *rejCenterPtr = &scoreInBuf2[(offset + (3 * scoreInPitch + 3)) % scoreInCbLen];

            uint64_t rejBitMask =
                fastCornerDetectorQuickRej(rejSeg0Ptr, rejSeg1Ptr, rejCenterPtr, threshold, &scoreQuickRejAgen);

            BASE_TYPE *scoreSeg0Ptr   = &scoreInBuf0[(offset + (0 * scoreInPitch + 2)) % scoreInCbLen];
            BASE_TYPE *scoreSeg1Ptr   = &scoreInBuf1[(offset + (1 * scoreInPitch + 1)) % scoreInCbLen];
            BASE_TYPE *scoreSeg2Ptr   = &scoreInBuf2[(offset + (2 * scoreInPitch + 0)) % scoreInCbLen];
            BASE_TYPE *scoreCenterPtr = &scoreInBuf1[(offset + (3 * scoreInPitch + 3)) % scoreInCbLen];

            // Arrange the if else conditions in a probability descending order.
            if (rejBitMask == ~0x0)
            {
                // The entire tile can be skipped.
                ;
            }
            else if (rejBitMask != 0x0)
            {
                // Some double vectors in the tile have been rejected.
                scoreAgen.niterNonRejVec = fastCornerDetectorCalVecBaseAdv(
                    rejBitMask, scoreInPitch, inputVecOffsetInTile, outputVecOffsetInTile, inputBaseAdv, outputBaseAdv);

                fastCornerDetectorScorePartialRej(scoreSeg0Ptr, scoreSeg1Ptr, scoreSeg2Ptr, scoreCenterPtr, scoreOutPtr,
                                                  threshold, &scoreAgen, inputBaseAdv, outputBaseAdv);
            }
            else
            {
                // Not a single double vector in the tile has been rejected.
                fastCornerDetectorScoreNoRej(scoreSeg0Ptr, scoreSeg1Ptr, scoreSeg2Ptr, scoreCenterPtr, scoreOutPtr,
                                             threshold, &scoreNoQuickRejAgen);
            }

#ifdef NMS_ON
            tileRejFlagBuf[i * niterX + j] = rejBitMask == ~0x0;
#else // !NMS_ON - Pack immediately during score computation \
      // Determine whether to pack this tile
            bool shouldPack;
#    ifdef TOPK_ON
            shouldPack = (rejBitMask != ~0x0) && (totalPackedCount < packedCapacity);
#    else
            shouldPack = (rejBitMask != ~0x0);
#    endif

            if (shouldPack)
            {
#    ifdef TOPK_ON
                int32_t packOffset     = pingpongWrite * TOPK_LOC_BUF_SIZE + localBufLocCnt;
                uint32_t *packPtr      = &packBuf[packOffset];
                uint16_t *scorePackPtr = &packScoreBuf[packOffset];
#    else
                uint32_t *packPtr = &packBuf[localBufLocCnt];
#    endif

                int32_t packed;
                if (hasLastTileX && (j == (niterX - 1)))
                {
                    packed = fastCornerDetectorPack<true>(&nmsPackAgen, tilePosX, tilePosY, width, _lastTileY, packPtr
#    ifdef TOPK_ON
                                                          ,
                                                          scorePackPtr, histVectorBuf
#    endif
                    );
                }
                else
                {
                    packed = fastCornerDetectorPack<false>(&nmsPackAgen, tilePosX, tilePosY, width, _lastTileY, packPtr
#    ifdef TOPK_ON
                                                           ,
                                                           scorePackPtr, histVectorBuf
#    endif
                    );
                }
                localBufLocCnt += packed;

#    ifdef TOPK_ON
                // Clamp write count to respect packedCapacity
                int32_t writeCount = localBufLocCnt;
                if (totalPackedCount + writeCount > packedCapacity)
                    writeCount = packedCapacity - totalPackedCount;

                if (writeCount > 0)
                {
                    writePackedDataSQDF(packedDataWriteHdl, packBuf, packScoreBuf, writeCount, totalPackedCount,
                                        pingpongWrite, packedLocsExtPtr, packedScoresExtPtr, isFirstWrite);
                    isFirstWrite = false;
                    totalPackedCount += writeCount;
                    pingpongWrite ^= 1;
                }
                localBufLocCnt = 0;
                tilesProcessed++;

                if (tilesProcessed % TILES_PER_REDUCTION == 0)
                    ReduceVectorToScalar(histScalarBuf, &histReduceAgenParam);
#    else  // !TOPK_ON
                /*
                * Write to the external buffer in DRAM if there is not enough space within the local buffer
                * so that next tile can possibly write to.
                */
                if (LOC_BUF_SIZE - localBufLocCnt < tileX * tileY)
                {
                    if (extBufLocCnt < locCapacity)
                    {
                        convertPixelToF32WriteExtBufWrapper(locSQDFHdl, locBuf, extBufLocCnt, localBufLocCnt,
                                                            locCapacity, pingpong, packBuf, locExtPtr);

                        // The pingpong flag is only for the SQDF write and toggles after write occurs.
                        pingpong ^= 1;
                    }
                    localBufLocCnt = 0;
                }
#    endif // TOPK_ON
            }
#endif     // NMS_ON
            tilePosX += tileX;

            cupvaRasterDataFlowRelease(srcHdl);
#ifdef NMS_ON
            cupvaRasterDataFlowRelease(scoreHdl);
#endif // NMS_ON
        }
        tilePosY += tileY;
    }
    cupvaRasterDataFlowClose(srcHdl);
#ifdef NMS_ON
    cupvaRasterDataFlowClose(scoreHdl);
#endif // NMS_ON

// NMS and pack
#ifdef NMS_ON
    cupvaRasterDataFlowOpen(nmsInTrig, nmsInBuf);

    int32_t nmsInLp    = cupvaRasterDataFlowGetLinePitch(nmsInTrig);
    int32_t scoreCbLen = cupvaRasterDataFlowGetCbLen(nmsInTrig);

    fastCornerDetectorNmsPackAgenInit(nmsInLp, tileX, tileY, scoreCbLen * sizeof(BASE_TYPE), (uintptr_t)nmsInBuf,
                                      &nmsPackAgen, laneId);

    /*
     * Need to calculate a tile's starting x and y axis, so cannot merge the two loops into one, otherwise i%niterX
     * is needed, which is quite costly.
     */
    tilePosY = 0;
    for (int32_t i = 0; i < niterY; i++)
    {
        int32_t tilePosX = 0;
        for (int32_t j = 0; j < niterX; j++)
        {
            BASE_TYPE *nmsInPtr = (BASE_TYPE *)cupvaRasterDataFlowAcquire(nmsInTrig);

            bool isTileNotRej = tileRejFlagBuf[i * niterX + j] != 1;

            // Determine whether to pack this tile
            bool shouldPackNms;
#    ifdef TOPK_ON
            shouldPackNms = isTileNotRej && (totalPackedCount < packedCapacity);
#    else
            shouldPackNms = isTileNotRej;
#    endif

            if (shouldPackNms)
            {
                BASE_TYPE *centerPtr = &nmsInBuf[(nmsInPtr + 1 * nmsInLp - nmsInBuf) % scoreCbLen];

#    ifdef TOPK_ON
                int32_t packOffset = pingpongWrite * TOPK_LOC_BUF_SIZE + localBufLocCnt;
                localBufLocCnt += fastCornerDetectorNmsPack(nmsInPtr, centerPtr, &nmsPackAgen, tilePosX, tilePosY,
                                                            (uint32_t chess_storage(RAM_Bb) *)&packBuf[packOffset],
                                                            &packScoreBuf[packOffset], histVectorBuf);

                int32_t writeCount = localBufLocCnt;
                if (totalPackedCount + writeCount > packedCapacity)
                    writeCount = packedCapacity - totalPackedCount;

                if (writeCount > 0)
                {
                    writePackedDataSQDF(packedDataWriteHdl, packBuf, packScoreBuf, writeCount, totalPackedCount,
                                        pingpongWrite, packedLocsExtPtr, packedScoresExtPtr, isFirstWrite);
                    isFirstWrite = false;
                    totalPackedCount += writeCount;
                    pingpongWrite ^= 1;
                }
                localBufLocCnt = 0;
                tilesProcessed++;

                if (tilesProcessed % TILES_PER_REDUCTION == 0)
                    ReduceVectorToScalar(histScalarBuf, &histReduceAgenParam);
#    else  // !TOPK_ON
                localBufLocCnt += fastCornerDetectorNmsPack(nmsInPtr, centerPtr, &nmsPackAgen, tilePosX, tilePosY,
                                                            (uint32_t chess_storage(RAM_Bb) *)&packBuf[localBufLocCnt]);

                /*
                 * Write to the external buffer in DRAM if there is not enough space within the local buffer
                 * so that next tile can possibly write to.
                 */
                if (LOC_BUF_SIZE - localBufLocCnt < tileX * tileY)
                {
                    if (extBufLocCnt < locCapacity)
                    {
                        convertPixelToF32WriteExtBufWrapper(locSQDFHdl, locBuf, extBufLocCnt, localBufLocCnt,
                                                            locCapacity, pingpong, packBuf, locExtPtr);
                        pingpong ^= 1;
                    }
                    localBufLocCnt = 0;
                }
#    endif // TOPK_ON
            }
            tilePosX += tileX;
            cupvaRasterDataFlowRelease(nmsInTrig);
        }
        tilePosY += tileY;
    }
    cupvaRasterDataFlowClose(nmsInTrig);
#endif // NMS_ON

#ifdef TOPK_ON
    {
        // Final packed data write to DRAM
        if (localBufLocCnt > 0)
        {
            bool isFinalWriteFirst = (totalPackedCount == 0);
            writePackedDataSQDF(packedDataWriteHdl, packBuf, packScoreBuf, localBufLocCnt, totalPackedCount,
                                pingpongWrite, packedLocsExtPtr, packedScoresExtPtr, isFinalWriteFirst);
            totalPackedCount += localBufLocCnt;
        }

        // Final histogram reduction
        if (tilesProcessed % TILES_PER_REDUCTION != 0)
            ReduceVectorToScalar(histScalarBuf, &histReduceAgenParam);

        // Sync final write before closing
        if (totalPackedCount > 0)
            cupvaSQDFSync(packedDataWriteHdl);

        // Compute threshold from histogram
        uint16_t topKThreshold = ComputeThreshold(histScalarBuf, maxFeatures);

        // FILTER STAGE: Double-buffered pipelined reads with filtering
        // Reuse scoreBuf as filter output buffer
        uint32_t *filterBuf    = (uint32_t *)scoreBuf;
        int32_t filterBufCnt   = 0;
        int32_t filteredCount  = 0;
        int32_t pingpongFilter = 0;

        int32_t batchSize  = FILTER_BUF_CAPACITY;
        int32_t numBatches = (totalPackedCount + batchSize - 1) / batchSize;
        if (totalPackedCount > 0 && numBatches > 0)
        {
            filterAgenInit(batchSize, packScoreBuf, packBuf, &filterAgenParam);

            int32_t batch0Count     = (numBatches == 1) ? totalPackedCount : batchSize;
            int32_t batch0Offset    = 0;
            int32_t batch0BufOffset = 0;

            readPackedDataSQDF(packedDataReadHdl, packBuf, packScoreBuf, batch0Count, batch0Offset, batch0BufOffset,
                               packedLocsExtPtr, packedScoresExtPtr);

            for (int32_t batch = 1; batch < numBatches; batch++)
            {
                int32_t prevBatch = batch - 1;
                int32_t prevBatchCount =
                    (prevBatch == numBatches - 1) ? (totalPackedCount - prevBatch * batchSize) : batchSize;
                int32_t prevBufOffset = (prevBatch % 2 == 0) ? 0 : TOPK_LOC_BUF_SIZE;

                cupvaSQDFSync(packedDataReadHdl);

                uint32_t *prevPackBuf      = &packBuf[prevBufOffset];
                uint16_t *prevPackScoreBuf = &packScoreBuf[prevBufOffset];

                // Flush accumulated features if filterBuf might overflow
                if (filterBufCnt > 0 && (FILTER_BUF_CAPACITY - filterBufCnt) < prevBatchCount)
                {
                    if (filteredCount < locCapacity)
                    {
                        convertPixelToF32WriteExtBufWrapper(locSQDFHdl, locBuf, filteredCount, filterBufCnt,
                                                            locCapacity, pingpongFilter, filterBuf, locExtPtr);
                        pingpongFilter ^= 1;
                    }
                    filterBufCnt = 0;
                }

                int32_t localFiltered =
                    filterPackedFeaturesByThresholdVPU(prevPackBuf, prevPackScoreBuf, prevBatchCount, topKThreshold,
                                                       &filterBuf[filterBufCnt], &filterAgenParam);
                filterBufCnt += localFiltered;

                // Trigger next batch read
                int32_t nextBatchCount = (batch == numBatches - 1) ? (totalPackedCount - batch * batchSize) : batchSize;
                int32_t nextBatchOffset = batch * batchSize;
                int32_t nextBufOffset   = (batch % 2 == 0) ? 0 : TOPK_LOC_BUF_SIZE;

                readPackedDataSQDF(packedDataReadHdl, packBuf, packScoreBuf, nextBatchCount, nextBatchOffset,
                                   nextBufOffset, packedLocsExtPtr, packedScoresExtPtr);
            }

            // Process the final batch
            int32_t finalBatch      = numBatches - 1;
            int32_t finalBatchCount = totalPackedCount - finalBatch * batchSize;
            int32_t finalBufOffset  = (finalBatch % 2 == 0) ? 0 : TOPK_LOC_BUF_SIZE;

            cupvaSQDFSync(packedDataReadHdl);

            uint32_t *finalPackBuf      = &packBuf[finalBufOffset];
            uint16_t *finalPackScoreBuf = &packScoreBuf[finalBufOffset];

            int32_t localFiltered =
                filterPackedFeaturesByThresholdVPU(finalPackBuf, finalPackScoreBuf, finalBatchCount, topKThreshold,
                                                   &filterBuf[filterBufCnt], &filterAgenParam);
            filterBufCnt += localFiltered;

            // Write remaining filtered data
            if (filterBufCnt > 0 && filteredCount < locCapacity)
            {
                convertPixelToF32WriteExtBufWrapper(locSQDFHdl, locBuf, filteredCount, filterBufCnt, locCapacity,
                                                    pingpongFilter, filterBuf, locExtPtr);
            }
        }

        extBufLocCnt = filteredCount;
    }
#else  // !TOPK_ON
    // Non-TopK flow: Write remaining data in location buffers
    if ((localBufLocCnt > 0) && (extBufLocCnt < locCapacity))
    {
        convertPixelToF32WriteExtBufWrapper(locSQDFHdl, locBuf, extBufLocCnt, localBufLocCnt, locCapacity, pingpong,
                                            packBuf, locExtPtr);
    }
#endif // TOPK_ON

    // Avoid the SQDF's hang by judging whether there are previous SQDF triggers, since it is possible of no detected corners in an image.
    if (extBufLocCnt > 0)
    {
        cupvaSQDFSync(locSQDFHdl);
    }
    cupvaSQDFClose(locSQDFHdl);

    locCntBuf[0] = extBufLocCnt;
    cupvaSQDFTrig(locCntSQDFHdl);
    cupvaSQDFSync(locCntSQDFHdl);
    cupvaSQDFClose(locCntSQDFHdl);

#ifdef TOPK_ON
    cupvaSQDFClose(packedDataWriteHdl);
    cupvaSQDFClose(packedDataReadHdl);
#endif // TOPK_ON

    return 0;
}
