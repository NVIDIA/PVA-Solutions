/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TEMPLATE_MATCHING_KNL_H
#define TEMPLATE_MATCHING_KNL_H

#include "template_matching_param.h"

#include <cupva_device.h>
#include <stdint.h>

// Agen configurations, built once per submit and rebased per tile. Bank map:
//   A: imgA, permute table, algoParams, template stage
//   B: imgB, V ring, correlation scratch
//   C: outTile, coefficients
// Grouped by stage below, in the order the stages run.
typedef struct TemplateMatchingConfig
{
    // box sum: squared-window horizontal sum + vertical prefix sum
    AgenCFG sqEvenTapCfg; /* A: image, taps 0,2,..            */
    AgenCFG sqOddTapCfg;  /* B: image, taps 1,3,..            */
    AgenCFG sqLastACfg;   /* A: image, last even tap          */
    AgenCFG sqLastBCfg;   /* B: image, last odd tap (masked)  */
    AgenCFG vPrevCfg;     /* B: V(r-1) de-interleaving load   */
    AgenCFG vStoreCfg;    /* B: V(r)   interleaving store     */

    // correlation: 2D VFilt4x2x2BBH
    AgenCFG corrEvenRowCfg; /* A: src1a, even image rows       */
    AgenCFG corrOddRowCfg;  /* B: src1b, odd  image rows       */
    AgenCFG corrCoefCfg;    /* C: 16-byte coefficient blocks   */
    AgenCFG corrStoreCfg;   /* B: correlation scratch          */

    // score epilogue
    AgenCFG epiCorrCfg; /* B: correlation scratch           */
    AgenCFG epiVNewCfg; /* B: V(y+h-1)                      */
    AgenCFG epiVOldCfg; /* B: V(y-1)                        */
    AgenCFG epiOutCfg;  /* C: F32 output tile               */

    // zero fill of the V(-1) row
    AgenCFG vZeroCfg; /* B: V ring row                    */

    // shared: DVLDPermHBU pattern vector
    AgenCFG permCfg; /* A: 32-entry permute table        */

    int32_t nChunk;         /* 32-column chunks evaluated per tile  */
    int32_t groupTaps;      /* taps accumulated in one 24-bit group */
    int32_t numGroups;      /* 24-bit accumulator groups per chunk  */
    int32_t sqLoopPairs;    /* sqPairs - 1 pairs inside the loop    */
    int32_t templateEnergy; /* ET = sum of squared template samples */
    int32_t lastTapMask;    /* 255 keeps, 0 zeroes the peeled odd tap (odd w) */
} TemplateMatchingConfig;

// imgCbStart* / imgCbLen are the RDF ring bounds. RDF may lay the halo'd tile
// out circularly, so every image-reading agen wraps against them; the bounds
// are inert when RDF picks its two-buffer layout instead.
void template_matching_init_agens(const TemplateMatchingParams *params, int32_t imgLinePitch, int32_t outLinePitch,
                                  const uint8_t *imgCbStartA, const uint8_t *imgCbStartB, int32_t imgCbLen,
                                  TemplateMatchingConfig *cfg);

int32_t template_matching_template_energy(const uint8_t *stage);

void template_matching_pack_coefficients(const uint8_t *stage, uint8_t *coef, const TemplateMatchingParams *params);

void template_matching_zero_v_row(int32_t *vRow, TemplateMatchingConfig *cfg);

void template_matching_box_row(const uint8_t *imgARow, const uint8_t *imgBRow, const uint8_t *lastARow,
                               const uint8_t *lastBRow, const int32_t *vPrevRow, int32_t *vCurRow,
                               TemplateMatchingConfig *cfg);

void template_matching_correlate(const uint8_t *imgARow, const uint8_t *imgBRow, const uint8_t *coef,
                                 const uint8_t *permTable, int32_t *corrScratch, TemplateMatchingConfig *cfg);

void template_matching_score(const int32_t *corrScratch, const int32_t *vNewRow0, const int32_t *vOldRow0,
                             const int32_t *vNewRow1, const int32_t *vOldRow1, float *outTile,
                             TemplateMatchingConfig *cfg);

#endif /* TEMPLATE_MATCHING_KNL_H */
