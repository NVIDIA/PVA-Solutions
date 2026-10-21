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

#ifndef COHERENT_INTEGRATION_KNL_H
#define COHERENT_INTEGRATION_KNL_H

#include "coherent_integration_param.h"

#include <cupva_device.h>

typedef struct CoherentIntegrationPreprocessAgenConfigRec
{
    AgenCFG raw;
    AgenCFG limb;
    AgenCFG shift;
} CoherentIntegrationPreprocessAgenConfig;

void coherentInitializePreprocessAgenConfig(CoherentIntegrationPreprocessAgenConfig *config, uint32_t rawCellCapacity);

void coherentPreprocessRawTile(CoherentIntegrationPreprocessAgenConfig const *agenConfig, uint8_t *rawInput,
                               int8_t *cellShift, int16_t *highLimbs, int16_t *residualLimbs, uint32_t validCellCount);

void coherentPreprocessRawTileImpl(CoherentIntegrationPreprocessAgenConfig const *agenConfig, uint8_t *rawInput,
                                   int8_t *cellShift, int16_t *highLimbs, int16_t *residualLimbs,
                                   uint32_t validCellCount);

void coherentFinishMagnitudes(CoherentIntegrationParams const *params, int8_t const *cellShift, int32_t *peakPower,
                              float *foldAccumulator, uint8_t *outputStage, uint32_t validCellCount, bool firstFold);

void coherentEncodeFolded(CoherentIntegrationParams const *params, float const *foldAccumulator, uint8_t *outputStage,
                          uint32_t cellCount);

#endif /* COHERENT_INTEGRATION_KNL_H */
