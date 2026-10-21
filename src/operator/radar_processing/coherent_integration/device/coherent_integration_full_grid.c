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

#include <cupva_device.h>

#ifndef PVASOL_COHERENT_INTEGRATION_RX
#    error "PVASOL_COHERENT_INTEGRATION_RX must select 4, 8, 12, or 16 receivers"
#endif

#if (PVASOL_COHERENT_INTEGRATION_RX != CI_RX4) && (PVASOL_COHERENT_INTEGRATION_RX != CI_RX8) && \
    (PVASOL_COHERENT_INTEGRATION_RX != CI_RX12) && (PVASOL_COHERENT_INTEGRATION_RX != CI_RX16)
#    error "PVASOL_COHERENT_INTEGRATION_RX must be 4, 8, 12, or 16"
#endif

#define CI_FULL_GRID_LANES (CI_CELL_VECTOR_LANES)
#define CI_FULL_GRID_RX (PVASOL_COHERENT_INTEGRATION_RX)

typedef char coherentIntegrationFullGridHasSixteenLanes[(CI_FULL_GRID_LANES == 16U) ? 1 : -1];
typedef char coherentIntegrationFullGridHasPairedRx[((CI_FULL_GRID_RX & 1U) == 0U) ? 1 : -1];

/* Derive the worst-case post-initialization loop bounds from the 48 KiB steering capacity. */
#define CI_MAX_PAIRED_PANEL_ITERATIONS ((CI_STEERING_CAPACITY_BYTES / (CI_RX4 * CI_STEERING_COMPLEX_BYTES) - 2U) / 2U)
#define CI_MAX_SINGLE_PANEL_ITERATIONS (CI_STEERING_CAPACITY_BYTES / (CI_RX12 * CI_STEERING_COMPLEX_BYTES) - 1U)

static inline dvshortx makeCoefficient(int32_t real0, int32_t imag0, int32_t real1, int32_t imag1, vshortx role)
{
    /*
     * Scalar S16 reads promote before negation. This preserves the legal
     * signed-17 coefficient -(-32768) = +32768 in the WRF.
     */
    dvshortx coefficient;
    coefficient.lo = replicateh(-imag1);
    coefficient.lo = vmux(role == 2, real1, coefficient.lo);
    coefficient.lo = vmux(role == 1, -imag0, coefficient.lo);
    coefficient.lo = vmux(role == 0, real0, coefficient.lo);
    coefficient.hi = replicateh(real1);
    coefficient.hi = vmux(role == 2, imag1, coefficient.hi);
    coefficient.hi = vmux(role == 1, real0, coefficient.hi);
    coefficient.hi = vmux(role == 0, imag0, coefficient.hi);
    return coefficient;
}

static inline uint32_t coefficientOffset(uint32_t firstBeam, uint32_t ordinal, uint32_t pair)
{
    return (((firstBeam + ordinal) * CI_FULL_GRID_RX + 2U * pair) * 2U);
}

static inline dvintx reconstructPower(dvintx highReal, dvintx highImag, dvintx residualReal, dvintx residualImag,
                                      dvintx reconstructionShift, uint32_t termShift)
{
    const dvintx real = dvround(dvshiftadd(residualReal, reconstructionShift, highReal), termShift - 1U);
    const dvintx imag = dvround(dvshiftadd(residualImag, reconstructionShift, highImag), termShift - 1U);
    return dvsumsq_t16(real, imag);
}

enum
{
    CI_RX_PAIR_COUNT = CI_FULL_GRID_RX / 2U,
#if (PVASOL_COHERENT_INTEGRATION_RX == CI_RX4) || (PVASOL_COHERENT_INTEGRATION_RX == CI_RX8)
    CI_MAX_PANEL_BEAMS = 2U
#else
    CI_MAX_PANEL_BEAMS = 1U
#endif
};

/* Constant propagation of panelSize and accumulates is required to keep the hot loop specialized. */
[[gnu::always_inline]] static inline void consumeFlatPanel(int16_t const *highLimbsB, int16_t const *residualLimbsC,
                                                           int16_t const *residentSteering, int8_t const *cellShiftA,
                                                           int32_t *peakPowerA,
                                                           CoherentIntegrationFullGridAgenConfig const *agenConfig,
                                                           uint32_t termShift, uint32_t blockCount, uint32_t firstBeam,
                                                           uint32_t panelSize, uint32_t accumulates)
{
    const vshortx role = vcollate_idx(replicateh(0)) & 3;
    dvshortx coefficients[CI_RX_PAIR_COUNT][CI_MAX_PANEL_BEAMS];
    for (uint32_t pair = 0U; pair < CI_RX_PAIR_COUNT; ++pair) chess_unroll_loop(*)
    {
        const uint32_t offset0 = coefficientOffset(firstBeam, 0U, pair);
        coefficients[pair][0]  = makeCoefficient(residentSteering[offset0], residentSteering[offset0 + 1U],
                                                 residentSteering[offset0 + 2U], residentSteering[offset0 + 3U], role);
#if (PVASOL_COHERENT_INTEGRATION_RX == CI_RX4) || (PVASOL_COHERENT_INTEGRATION_RX == CI_RX8)
        if (panelSize > 1U)
        {
            const uint32_t offset1 = coefficientOffset(firstBeam, 1U, pair);
            coefficients[pair][1] =
                makeCoefficient(residentSteering[offset1], residentSteering[offset1 + 1U],
                                residentSteering[offset1 + 2U], residentSteering[offset1 + 3U], role);
        }
#endif
    }

    agen_B highInput     = init_B((dvshort *)highLimbsB);
    agen_C residualInput = init_C((dvshort *)residualLimbsC);
    INIT_AGEN2(highInput, agenConfig->limb);
    INIT_AGEN2(residualInput, agenConfig->limb);

    agen_A shiftInput = init_A((vchar *)cellShiftA);
    INIT_AGEN1(shiftInput, agenConfig->shift);

    agen_A peakInput;
    if (accumulates != 0U)
    {
        peakInput = init_A((dvint *)peakPowerA);
        INIT_AGEN2(peakInput, agenConfig->peak);
    }
    agen_A peakOutput = init_A((dvint *)peakPowerA);
    INIT_AGEN2(peakOutput, agenConfig->peak);

    for (uint32_t block = 0U; block < blockCount; ++block) chess_prepare_for_pipelining
    chess_loop_range(1, CI_MAX_PROCESSING_CELL_BLOCKS)
    {
        dvintx highReal[CI_MAX_PANEL_BEAMS];
        dvintx highImag[CI_MAX_PANEL_BEAMS];
        dvintx residualReal[CI_MAX_PANEL_BEAMS];
        dvintx residualImag[CI_MAX_PANEL_BEAMS];

        dvshortx high0     = dvshort_load_di(highInput);
        dvshortx high1     = dvshort_load_di(highInput);
        dvshortx residual0 = dvshort_load_di(residualInput);
        dvshortx residual1 = dvshort_load_di(residualInput);
        vdotp4x2_hhw(high0, high1, coefficients[0][0], highReal[0], highImag[0], 0, highReal[0], highImag[0]);
        vdotp4x2_hhw(residual0, residual1, coefficients[0][0], residualReal[0], residualImag[0], 0, residualReal[0],
                     residualImag[0]);
#if (PVASOL_COHERENT_INTEGRATION_RX == CI_RX4) || (PVASOL_COHERENT_INTEGRATION_RX == CI_RX8)
        if (panelSize > 1U)
        {
            vdotp4x2_hhw(high0, high1, coefficients[0][1], highReal[1], highImag[1], 0, highReal[1], highImag[1]);
            vdotp4x2_hhw(residual0, residual1, coefficients[0][1], residualReal[1], residualImag[1], 0, residualReal[1],
                         residualImag[1]);
        }
#endif

        for (uint32_t pair = 1U; pair < CI_RX_PAIR_COUNT; ++pair) chess_unroll_loop(*)
        {
            high0     = dvshort_load_di(highInput);
            high1     = dvshort_load_di(highInput);
            residual0 = dvshort_load_di(residualInput);
            residual1 = dvshort_load_di(residualInput);
            vdotp4x2_hhw(high0, high1, coefficients[pair][0], highReal[0], highImag[0], -1, highReal[0], highImag[0]);
            vdotp4x2_hhw(residual0, residual1, coefficients[pair][0], residualReal[0], residualImag[0], -1,
                         residualReal[0], residualImag[0]);
#if (PVASOL_COHERENT_INTEGRATION_RX == CI_RX4) || (PVASOL_COHERENT_INTEGRATION_RX == CI_RX8)
            if (panelSize > 1U)
            {
                vdotp4x2_hhw(high0, high1, coefficients[pair][1], highReal[1], highImag[1], -1, highReal[1],
                             highImag[1]);
                vdotp4x2_hhw(residual0, residual1, coefficients[pair][1], residualReal[1], residualImag[1], -1,
                             residualReal[1], residualImag[1]);
            }
#endif
        }

        const dvintx reconstructionShift = vchar_dvintx_load_di(shiftInput);
        dvintx newPeak                   = reconstructPower(highReal[0], highImag[0], residualReal[0], residualImag[0],
                                                            reconstructionShift, termShift);
#if (PVASOL_COHERENT_INTEGRATION_RX == CI_RX4) || (PVASOL_COHERENT_INTEGRATION_RX == CI_RX8)
        if (panelSize > 1U)
        {
            newPeak = dvmax(newPeak, reconstructPower(highReal[1], highImag[1], residualReal[1], residualImag[1],
                                                      reconstructionShift, termShift));
        }
#endif

        if (accumulates != 0U)
        {
            /* Unsigned low-word load is required when reconstructing signed-48 power. */
            const dvintx oldLow  = dvuint_load_di(peakInput);
            const dvintx oldHigh = dvint_load_di(peakInput);
            const dvintx best    = dvmax(dvshiftadd(oldHigh, 32, oldLow), newPeak);
            vstore_i(best, peakOutput);
            vstore_i(dvshift(best, -32), peakOutput);
        }
        else
        {
            vstore_i(newPeak, peakOutput);
            vstore_i(dvshift(newPeak, -32), peakOutput);
        }
    }
}

static inline void consumeFlatPanel1Initialize(int16_t const *highLimbsB, int16_t const *residualLimbsC,
                                               int16_t const *residentSteering, int8_t const *cellShiftA,
                                               int32_t *peakPowerA,
                                               CoherentIntegrationFullGridAgenConfig const *agenConfig,
                                               uint32_t termShift, uint32_t blockCount, uint32_t firstBeam)
{
    consumeFlatPanel(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig, termShift,
                     blockCount, firstBeam, 1U, 0U);
}

static inline void consumeFlatPanel1Accumulate(int16_t const *highLimbsB, int16_t const *residualLimbsC,
                                               int16_t const *residentSteering, int8_t const *cellShiftA,
                                               int32_t *peakPowerA,
                                               CoherentIntegrationFullGridAgenConfig const *agenConfig,
                                               uint32_t termShift, uint32_t blockCount, uint32_t firstBeam)
{
    consumeFlatPanel(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig, termShift,
                     blockCount, firstBeam, 1U, 1U);
}

#if (PVASOL_COHERENT_INTEGRATION_RX == CI_RX4) || (PVASOL_COHERENT_INTEGRATION_RX == CI_RX8)
static inline void consumeFlatPanel2Initialize(int16_t const *highLimbsB, int16_t const *residualLimbsC,
                                               int16_t const *residentSteering, int8_t const *cellShiftA,
                                               int32_t *peakPowerA,
                                               CoherentIntegrationFullGridAgenConfig const *agenConfig,
                                               uint32_t termShift, uint32_t blockCount, uint32_t firstBeam)
{
    consumeFlatPanel(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig, termShift,
                     blockCount, firstBeam, 2U, 0U);
}

static inline void consumeFlatPanel2Accumulate(int16_t const *highLimbsB, int16_t const *residualLimbsC,
                                               int16_t const *residentSteering, int8_t const *cellShiftA,
                                               int32_t *peakPowerA,
                                               CoherentIntegrationFullGridAgenConfig const *agenConfig,
                                               uint32_t termShift, uint32_t blockCount, uint32_t firstBeam)
{
    consumeFlatPanel(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig, termShift,
                     blockCount, firstBeam, 2U, 1U);
}
#endif

void coherentInitializeFullGridAgenConfig(CoherentIntegrationFullGridAgenConfig *config, uint32_t cellCount)
{
    const uint32_t blockCount = cellCount / CI_FULL_GRID_LANES;

    config->limb.size = sizeof(int16_t);
    config->limb.n1   = CI_FULL_GRID_RX;
    config->limb.s1   = 2U * CI_FULL_GRID_LANES;
    config->limb.n2   = blockCount;
    config->limb.s2   = CI_FULL_GRID_RX * 2U * CI_FULL_GRID_LANES;

    config->shift.size = sizeof(int8_t);
    config->shift.n1   = blockCount;
    config->shift.s1   = CI_FULL_GRID_LANES;

    config->peak.size = sizeof(int32_t);
    config->peak.n1   = 2U;
    config->peak.s1   = CI_FULL_GRID_LANES;
    config->peak.n2   = blockCount;
    config->peak.s2   = 2U * CI_FULL_GRID_LANES;
}

void coherentSearchFullGridResident(CoherentIntegrationParams const *params,
                                    CoherentIntegrationFullGridAgenConfig const *agenConfig, int8_t const *cellShiftA,
                                    int32_t *peakPowerA, int16_t const *highLimbsB, int16_t const *residualLimbsC,
                                    int16_t const *residentSteering, uint32_t cellCount)
{
    const uint32_t blockCount = cellCount / CI_FULL_GRID_LANES;

    /* Process two beams at a time for Rx4/Rx8 and one beam at a time for Rx12/Rx16. */
#if (PVASOL_COHERENT_INTEGRATION_RX == CI_RX4) || (PVASOL_COHERENT_INTEGRATION_RX == CI_RX8)
    if (params->numBeams == 1U)
    {
        consumeFlatPanel1Initialize(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig,
                                    params->termShift, blockCount, 0U);
        return;
    }

    consumeFlatPanel2Initialize(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig,
                                params->termShift, blockCount, 0U);
    uint32_t beam = 2U;
    if (beam + 2U <= params->numBeams)
    {
        for (; beam + 2U <= params->numBeams; beam += 2U) chess_loop_range(1, CI_MAX_PAIRED_PANEL_ITERATIONS)
        {
            consumeFlatPanel2Accumulate(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA,
                                        agenConfig, params->termShift, blockCount, beam);
        }
    }
    if (beam < params->numBeams)
    {
        consumeFlatPanel1Accumulate(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig,
                                    params->termShift, blockCount, beam);
    }
#else
    consumeFlatPanel1Initialize(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA, agenConfig,
                                params->termShift, blockCount, 0U);
    if (params->numBeams > 1U)
    {
        for (uint32_t beam = 1U; beam < params->numBeams; ++beam) chess_loop_range(1, CI_MAX_SINGLE_PANEL_ITERATIONS)
        {
            consumeFlatPanel1Accumulate(highLimbsB, residualLimbsC, residentSteering, cellShiftA, peakPowerA,
                                        agenConfig, params->termShift, blockCount, beam);
        }
    }
#endif
}
