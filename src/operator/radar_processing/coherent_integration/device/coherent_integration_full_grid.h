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

#ifndef COHERENT_INTEGRATION_FULL_GRID_H
#define COHERENT_INTEGRATION_FULL_GRID_H

#include "coherent_integration_param.h"

#include <cupva_device.h>
#include <stdint.h>

typedef struct CoherentIntegrationFullGridAgenConfigRec
{
    AgenWrapper limb;
    AgenWrapper shift;
    AgenWrapper peak;
} CoherentIntegrationFullGridAgenConfig;

void coherentInitializeFullGridAgenConfig(CoherentIntegrationFullGridAgenConfig *config, uint32_t cellCount);

/*
 * Search every beam in the complete resident Q15 tensor.  The tensor is
 * packed [Az][El][Rx][real,imag], so its flattened beam order is
 * beam = azimuth * numElevationBins + elevation.
 *
 * The selected VPU executable fixes Rx at compile time. Runtime Az and El
 * establish params->numBeams; every flattened beam is evaluated.
 */
void coherentSearchFullGridResident(CoherentIntegrationParams const *params,
                                    CoherentIntegrationFullGridAgenConfig const *agenConfig, int8_t const *cellShiftA,
                                    int32_t *peakPowerA, int16_t const *highLimbsB, int16_t const *residualLimbsC,
                                    int16_t const *residentSteering, uint32_t cellCount);

#endif /* COHERENT_INTEGRATION_FULL_GRID_H */
