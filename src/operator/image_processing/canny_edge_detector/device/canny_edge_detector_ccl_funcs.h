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

#ifndef CANNY_EDGE_DETECTOR_CCL_FUNCS_H
#define CANNY_EDGE_DETECTOR_CCL_FUNCS_H

#include "canny_edge_detector_ccl_knls.h"

#include <cupva_device.h>

struct Stage3BufPtrs
{
    uint8_t *inBuffer;
    uint16_t *inIntermediateBuffer;
    uint16_t *equivalentPairs;
    uint16_t *previousRowLabels;
    uint16_t *reformattedLabels;
    uint16_t *outIntermediateBuffer;
    uint8_t *outEdgeMap;
    uint8_t *reformatInput;
    uint16_t *tempCollapseBuffer;
    uint16_t *outLabels;
    uint16_t *labelLut;
};

int32_t canny_edge_detector_ccl_exec(struct Stage3BufPtrs *bufPtrs, int32_t nIterX, int32_t nIterY);

#endif // CANNY_EDGE_DETECTOR_CCL_FUNCS_H