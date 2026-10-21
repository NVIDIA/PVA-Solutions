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

#ifndef DEPTHTOSPACE_RDF_KNL_H
#define DEPTHTOSPACE_RDF_KNL_H

#include "depthtospace_rdf_param.h"

#include <cupva_device.h>

/*
 * Agen configuration for one output row offset; all are transposed-access agens.
 *
 * One DESTINATION agen per bx, not one shared, because an agen advances by s1 after EVERY
 * access -- BLK_SZ stores through one agen would step BLK_SZ*BLK_SZ bytes per iteration.
 */
typedef struct DepthToSpaceConfig
{
    AgenCFG srcAgenCfg[BLK_SZ];
    AgenCFG dstAgenCfg[BLK_SZ];
} DepthToSpaceConfig;

void depthtospace_agen_init(int32_t srcLinePitch, int32_t dstLinePitch, DepthToSpaceConfig *cfg);

void depthtospace_agen_update(uint8_t *const *srcTiles, uint8_t *dstTile, DepthToSpaceConfig *cfg);

void depthtospace_exec(int32_t niter, DepthToSpaceConfig *cfg);

#endif /* DEPTHTOSPACE_RDF_KNL_H */
