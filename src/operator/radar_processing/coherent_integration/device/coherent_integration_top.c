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

#include "coherent_integration_full_grid.h"
#include "coherent_integration_knl.h"

#include <cupva_device.h>

/* VMEM working buffers. */
VMEM(A, uint8_t, rawInput, CI_RAW_INPUT_BYTES);
VMEM(A, int8_t, cellShift, CI_CELL_SHIFT_BYTES);
VMEM(A, int32_t, peakPower, CI_PEAK_POWER_BYTES / sizeof(int32_t));
VMEM(A, float, foldAccumulator, CI_FOLD_ACCUMULATOR_BYTES / sizeof(float));
VMEM(B, int16_t, highLimbs, CI_LIMB_BYTES / sizeof(int16_t));
VMEM(B, uint8_t, steering, CI_STEERING_CAPACITY_BYTES);
VMEM(C, int16_t, residualLimbs, CI_LIMB_BYTES / sizeof(int16_t));
VMEM(C, uint8_t, work, CI_WORK_BYTES);

VMEM(A, CoherentIntegrationParams, GP);
VMEM_SEQDF_HANDLER(A, rawInputHdl);
VMEM_SEQDF_HANDLER(A, steeringHdl);
VMEM_SEQDF_HANDLER(A, coherentIntegrationRxOutputHdl);
VMEM_SEQDF_HANDLER(A, coherentIntegrationFinalOutputHdl);
VMEM_POINTER(A, inputExtPtr);
VMEM_POINTER(A, steeringExtPtr);
VMEM_POINTER(A, coherentIntegrationRxExtPtr);
VMEM_POINTER(A, coherentIntegrationFinalExtPtr);

static uint64_t externalAddress(ExtMemPointer const *pointer, uint64_t byteOffset)
{
    return pointer->base + pointer->offset + byteOffset;
}

/* Trigger one complete folded row into fixed Rx planes in VMEM. */
static void triggerRawTile(uint32_t globalRange, uint32_t globalDoppler)
{
    const uint64_t sourceOffset =
        (uint64_t)globalRange * GP.inputRangeStrideBytes + (uint64_t)globalDoppler * GP.inputDopplerStrideBytes;

    cupvaSQDFUpdateAddr(rawInputHdl, 0, externalAddress(&inputExtPtr, sourceOffset), MEMTYPE_DRAM,
                        cupvaGetVmemAddress(rawInput), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(rawInputHdl);
}

static void waitRawTile(void)
{
    cupvaSQDFSync(rawInputHdl);
}

static void triggerSteeringTile(uint8_t *destination, uint32_t beamStart, uint32_t beamCount)
{
    const uint32_t beamBytes = GP.numRx * CI_STEERING_COMPLEX_BYTES;

    cupvaSQDFUpdateTileSize(steeringHdl, 0, (int32_t)beamBytes, (int32_t)beamCount);
    cupvaSQDFUpdateAddr(steeringHdl, 0, externalAddress(&steeringExtPtr, (uint64_t)beamStart * beamBytes), MEMTYPE_DRAM,
                        cupvaGetVmemAddress(destination), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(steeringHdl);
}

static void waitSteeringTile(void)
{
    cupvaSQDFSync(steeringHdl);
}

static void triggerOutput(SequenceDataFlowHandler *handler, uint8_t *source, ExtMemPointer const *destination,
                          uint32_t rangeStart, uint32_t dopplerStart, uint32_t destinationRangeStride)
{
    const uint64_t destinationOffset =
        (uint64_t)rangeStart * destinationRangeStride + (uint64_t)dopplerStart * sizeof(uint32_t);

    cupvaSQDFUpdateAddr(*handler, 0, cupvaGetVmemAddress(source), MEMTYPE_VMEM,
                        externalAddress(destination, destinationOffset), MEMTYPE_DRAM);
    cupvaSQDFFlushAndTrig(*handler);
}

static void waitOutput(SequenceDataFlowHandler *handler)
{
    cupvaSQDFSync(*handler);
}

CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(rawInputHdl);
    cupvaSQDFOpen(steeringHdl);
    cupvaSQDFOpen(coherentIntegrationRxOutputHdl);
    cupvaSQDFOpen(coherentIntegrationFinalOutputHdl);

    uint8_t *const coherentIntegrationRxPing     = work + CI_COHERENT_INTEGRATION_RX_PING_OFFSET;
    uint8_t *const coherentIntegrationRxPong     = work + CI_COHERENT_INTEGRATION_RX_PONG_OFFSET;
    uint8_t *const coherentIntegrationFinalStage = work + CI_COHERENT_INTEGRATION_FINAL_STAGE_OFFSET;
    /* Keep the complete host-packed steering tensor resident in VMEM. */
    triggerSteeringTile(steering, 0U, GP.numBeams);
    waitSteeringTile();
    cupvaSQDFClose(steeringHdl);

    const uint32_t rangeTileCount = GP.numRangeBins / GP.processingRangeTile;
    const uint32_t totalRDTiles   = rangeTileCount * GP.repeatFold;
    const uint32_t cellCount      = GP.processingRangeTile * GP.foldedWidth;
    CoherentIntegrationPreprocessAgenConfig preprocessAgenConfig;
    coherentInitializePreprocessAgenConfig(&preprocessAgenConfig, GP.rawCellCapacity);
    CoherentIntegrationFullGridAgenConfig fullGridAgenConfig = {0};
    coherentInitializeFullGridAgenConfig(&fullGridAgenConfig, cellCount);
    uint32_t rangeStart                        = 0U;
    uint32_t foldIndex                         = 0U;
    bool coherentIntegrationRxOutputPending    = false;
    bool coherentIntegrationFinalOutputPending = false;
    /* Prime raw tile zero after the one-time resident steering transfer. */
    triggerRawTile(0U, 0U);
    /* One iteration processes one range section, one complete fold, and all Rx. */
    for (uint32_t rdTileIndex = 0U; rdTileIndex < totalRDTiles; rdTileIndex++)
    {
        const bool firstFold         = foldIndex == 0U;
        const bool lastFold          = foldIndex + 1U == GP.repeatFold;
        const uint32_t globalDoppler = foldIndex * GP.foldedWidth;

        /* The descriptor includes all Rx. Preprocessing releases rawInput. */
        waitRawTile();
        /* VPU: preprocess the raw Rx values for beam search. */
        coherentPreprocessRawTile(&preprocessAgenConfig, rawInput, cellShift, highLimbs, residualLimbs, cellCount);

        /* DMA: overlap the next raw tile with current search and output work. */
        if (rdTileIndex + 1U < totalRDTiles)
        {
            if (lastFold)
            {
                triggerRawTile(rangeStart + GP.processingRangeTile, 0U);
            }
            else
            {
                triggerRawTile(rangeStart, globalDoppler + GP.foldedWidth);
            }
        }

        /* Search every resident azimuth/elevation beam for each cell. */
        coherentSearchFullGridResident(&GP, &fullGridAgenConfig, cellShift, peakPower, highLimbs, residualLimbs,
                                       (int16_t const *)steering, cellCount);

        uint8_t *const coherentIntegrationRxStage =
            ((rdTileIndex & 1U) == 0U) ? coherentIntegrationRxPing : coherentIntegrationRxPong;
        /* VPU: reconstruct this fold, initializing or updating the fold accumulator as required. */
        coherentFinishMagnitudes(&GP, cellShift, peakPower, foldAccumulator, coherentIntegrationRxStage, cellCount,
                                 firstFold);

        /* Keep the previous output active through search/finish, then reuse its handler. */
        if (coherentIntegrationRxOutputPending)
        {
            waitOutput(&coherentIntegrationRxOutputHdl);
        }
        triggerOutput(&coherentIntegrationRxOutputHdl, coherentIntegrationRxStage, &coherentIntegrationRxExtPtr,
                      rangeStart, globalDoppler, GP.coherentIntegrationRxRangeStrideBytes);
        coherentIntegrationRxOutputPending = true;

        if (lastFold)
        {
            if (coherentIntegrationFinalOutputPending)
            {
                waitOutput(&coherentIntegrationFinalOutputHdl);
            }
            /* VPU: encode into a slot distinct from the active coherentIntegrationRx DMA. */
            coherentEncodeFolded(&GP, foldAccumulator, coherentIntegrationFinalStage, cellCount);
            triggerOutput(&coherentIntegrationFinalOutputHdl, coherentIntegrationFinalStage,
                          &coherentIntegrationFinalExtPtr, rangeStart, 0U, GP.coherentIntegrationFinalRangeStrideBytes);
            coherentIntegrationFinalOutputPending = true;
        }

        if (lastFold)
        {
            foldIndex = 0U;
            rangeStart += GP.processingRangeTile;
        }
        else
        {
            foldIndex++;
        }
    }

    if (coherentIntegrationRxOutputPending)
    {
        waitOutput(&coherentIntegrationRxOutputHdl);
    }
    if (coherentIntegrationFinalOutputPending)
    {
        waitOutput(&coherentIntegrationFinalOutputHdl);
    }

    cupvaSQDFClose(coherentIntegrationFinalOutputHdl);
    cupvaSQDFClose(coherentIntegrationRxOutputHdl);
    cupvaSQDFClose(rawInputHdl);
    return 0;
}
