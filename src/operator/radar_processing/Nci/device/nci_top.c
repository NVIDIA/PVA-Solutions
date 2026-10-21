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

#include <cupva_device.h>

VMEM(B, Q16_15, ft2d, RDF_DOUBLE(Q16_15, MaxNumRangeBinsPerTile, MaxSamplesSizePerTile, TRANS_MODE_2));
VMEM(C, UQ16_16, nciRx, RDF_DOUBLE(UQ16_16, 1, MaxNciRxSize));
VMEM(C, UQ16_16, nciFinal, RDF_DOUBLE(UQ16_16, 1, MaxNciFinalSize));
VMEM(C, UQ16_16, noiseEstimate, RDF_SINGLE(UQ16_16, 1, MaxNoiseEstimateSize));
VMEM(A, UQ16_16, nciRxTemp, MaxNciRxSize + MaxSamplesSizePerTile);

VMEM(C, uint32_t, Nr);
VMEM(C, uint32_t, Ns);
VMEM(C, uint32_t, NofRx);
VMEM(C, uint32_t, numDopplerFolds);
VMEM(C, uint32_t, NsPerFold);
VMEM(C, uint32_t, NrPerTile);
VMEM(C, uint32_t, NsPerTile);
VMEM(C, uint32_t, numTiles);
VMEM(C, uint32_t, tileSize);
VMEM(C, uint32_t, nciRxShiftValue);
VMEM(C, uint32_t, nciFinalShiftValue);
VMEM(C, uint32_t, inputTileSize);
VMEM(C, uint32_t, outputNciRxTileSize);
VMEM(C, uint32_t, outputNciFinalTileSize);
VMEM(C, uint32_t, outputNoiseEstimateTileSize);
VMEM(C, uint32_t, noiseEstimateEnabled);
VMEM(C, uint32_t, isRangeDopplerRxLayout);

VMEM(A, RasterDataFlowHandler, srcDataFlowHandler);
VMEM(A, RasterDataFlowHandler, dstDataFlowHandler);
VMEM(A, RasterDataFlowHandler, dstNoiseEstimateDataFlowHandler);

VMEM(A, param_nci_t, params_nci, 1);

CUPVA_VPU_MAIN()
{
    uint8_t srcPingPong = 0u;
    uint8_t dstPingPong = 0u;
    uint32_t currNs     = 0u;

    int32_t dstOffset = 0;
    uint32_t isFirstNciRxIter;
    uint32_t isFirstNciFinalIter;
    uint32_t inputTileSizeRDRx;

    uint32_t isNciRxReleased = 1u;

    cupvaRasterDataFlowTrig(srcDataFlowHandler);

    NsPerFold = Ns / numDopplerFolds;

    // Calculate shift values as ceil(log2(N)) for proper scaling
    // Each comparison adds 1 if N exceeds the power of 2
    nciRxShiftValue    = (NofRx > 1) + (NofRx > 2) + (NofRx > 4) + (NofRx > 8) + (NofRx > 16) + (NofRx > 32);
    nciFinalShiftValue = (numDopplerFolds > 1) + (numDopplerFolds > 2) + (numDopplerFolds > 4) + (numDopplerFolds > 8) +
                         (numDopplerFolds > 16) + (numDopplerFolds > 32);

    inputTileSize               = NrPerTile * NofRx * NsPerTile * 2;
    inputTileSizeRDRx           = NrPerTile * NofRx * (NsPerTile + 1) * 2; // Adding 2 for transpose operations
    outputNciRxTileSize         = NrPerTile * Ns;
    outputNciFinalTileSize      = NrPerTile * NsPerFold;
    outputNoiseEstimateTileSize = NrPerTile;

    numTiles = (Ns * Nr) / (NsPerTile * NrPerTile);

    uint32_t rowIndex = 0u;

    UQ16_16 *nciRxTileBuffer;

    nciRx_params_init(&params_nci[0], NrPerTile, Ns, NsPerTile, NofRx);
    nciFinal_params_init(&params_nci[0], NrPerTile, NsPerFold, Ns);

    for (uint32_t tileIndex = 0; tileIndex < numTiles; tileIndex++)
    {
        cupvaRasterDataFlowSync(srcDataFlowHandler);
        cupvaRasterDataFlowTrig(srcDataFlowHandler);
        Q16_15 *inputTileBuffer     = ft2d + (srcPingPong * inputTileSize);
        Q16_15 *inputTileBufferRDRx = ft2d + (srcPingPong * inputTileSizeRDRx);
        nciRxTileBuffer             = nciRx + (dstPingPong * outputNciRxTileSize);

        isFirstNciRxIter    = 1u;
        isFirstNciFinalIter = 1u;

        if (1u == isRangeDopplerRxLayout)
        {
            compute_nciRx_RangeDopplerRx(&params_nci[0], inputTileBufferRDRx, NrPerTile, NsPerTile, NofRx, tileIndex,
                                         nciRxTemp, Ns);
        }
        else
        {
            /* Compute NCI Rx intermediate sum for all Rx channels except the last two */
            /* Compute the average across all Rx channels at the end */
            for (uint32_t RxIndex = 0; RxIndex < NofRx; RxIndex += 2u)
            {
                if (1u == isFirstNciRxIter)
                {
                    compute_nciRx_first(&params_nci[0], inputTileBuffer + (RxIndex * (NsPerTile * 2)),
                                        nciRxTileBuffer + currNs, NrPerTile, NsPerTile,
                                        (2u == NofRx) ? nciRxShiftValue : 0u);
                    isFirstNciRxIter = 0u;
                }
                else
                {
                    compute_nciRx(&params_nci[0], inputTileBuffer + (RxIndex * (NsPerTile * 2)),
                                  nciRxTileBuffer + currNs, (RxIndex == (NofRx - 2u)) ? lastRx : notLastRx, NrPerTile,
                                  NsPerTile, nciRxShiftValue);
                }
            }
        }

        srcPingPong ^= 1u;

        currNs += NsPerTile;

        if (currNs == Ns)
        {
            if (1u == isRangeDopplerRxLayout)
            {
                //Copy to the NCI Rx Buffer from the Transposed Buffer if Range Doppler Rx Layout
                copy_nciRxTemp_to_nciRx(&params_nci[0], Ns, NrPerTile, nciRxTileBuffer, nciRxTemp, nciRxShiftValue);
            }
            UQ16_16 *nciFinalTileBuffer = nciFinal + (dstPingPong * outputNciFinalTileSize);

            for (uint32_t dopplerFoldIndex = 0; dopplerFoldIndex < numDopplerFolds; dopplerFoldIndex += 4u)
            {
                compute_nciFinal(&params_nci[0], nciRxTileBuffer + (NsPerFold * dopplerFoldIndex),
                                 nciRxTileBuffer + (NsPerFold * (dopplerFoldIndex + 1)),
                                 nciRxTileBuffer + (NsPerFold * (dopplerFoldIndex + 2)),
                                 nciRxTileBuffer + (NsPerFold * (dopplerFoldIndex + 3)), nciFinalTileBuffer,
                                 noiseEstimate + rowIndex, isFirstNciFinalIter,
                                 (dopplerFoldIndex == (numDopplerFolds - 4u)) ? lastDopplerFold : notLastDopplerFold,
                                 NrPerTile, NsPerFold, nciFinalShiftValue);
                isFirstNciFinalIter = 0u;
            }

            rowIndex += NrPerTile;

            cupvaRasterDataFlowSync(dstDataFlowHandler);
            cupvaRasterDataFlowTrig(dstDataFlowHandler);
            dstPingPong ^= 1u;

            currNs = 0u;
        }
    }

    cupvaRasterDataFlowSync(dstDataFlowHandler);

    if (1u == noiseEstimateEnabled)
    {
        cupvaRasterDataFlowTrig(dstNoiseEstimateDataFlowHandler);
        cupvaRasterDataFlowSync(dstNoiseEstimateDataFlowHandler);
    }

    return 0;
}