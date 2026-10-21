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

#ifndef NCI_KNL_H
#define NCI_KNL_H

#include "nci_param.h"

#include <cupva_device.h>

typedef struct
{
    AgenWrapper nciRxWrappers[8];
    AgenWrapper nciFinalWrappers[6];
    AgenWrapper nciNoiseEstimateWrappers[1];
} param_nci_t;

void nciRx_params_init(param_nci_t *params, uint32_t NrPerTile, uint32_t Ns, uint32_t NsPerTile, uint32_t NofRx);
void nciFinal_params_init(param_nci_t *params, uint32_t NrPerTile, uint32_t NsPerFold, uint32_t Ns);
void noiseEstimate_params_init(param_nci_t *params, uint32_t NrPerTile, uint32_t NsPerFold);

int32_t compute_nciRx_first(param_nci_t *params, Q16_15 *ft2d, UQ16_16 *nciRx, uint32_t NrPerTile, uint32_t Ns,
                            uint32_t nciRxShiftValue);
int32_t compute_nciRx(param_nci_t *params, Q16_15 *ft2d, UQ16_16 *nciRx, uint32_t isLast, uint32_t NrPerTile,
                      uint32_t Ns, uint32_t nciRxShiftValue);
int32_t compute_nciRx_RangeDopplerRx(param_nci_t *params, Q16_15 *ft2d, uint32_t NrPerTile, uint32_t NsPerTile,
                                     uint32_t NofRx, uint32_t tileIndex, uint32_t *nciRxTemp, uint32_t Ns);
int32_t compute_nciFinal(param_nci_t *params, UQ16_16 *nciRxFold0, UQ16_16 *nciRxFold1, UQ16_16 *nciRxFold2,
                         UQ16_16 *nciRxFold3, UQ16_16 *nciFinal, UQ16_16 *noiseEstimate, uint32_t isFirstIter,
                         uint32_t isLastDopplerFold, uint32_t NrPerTile, uint32_t NsPerFold,
                         uint32_t nciFinalShiftValue);
int32_t compute_noiseEstimate(param_nci_t *params, UQ16_16 *nciFinal, UQ16_16 *noiseEstimate, uint32_t NrPerTile,
                              uint32_t NsPerFold);
int32_t copy_nciRxTemp_to_nciRx(param_nci_t *params, uint32_t Ns, uint32_t NrPerTile, UQ16_16 *nciRx,
                                uint32_t *nciRxTemp, uint32_t nciRxShiftValue);

#endif /* NCI_KNL_H */