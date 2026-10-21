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
#ifndef CANNY_EDGE_DETECTOR_CCL_KNLS_H
#define CANNY_EDGE_DETECTOR_CCL_KNLS_H

#include "canny_edge_detector_params.h"

#include <cupva_device.h>
#include <stdint.h>

// Pass 1 - Reformat the input so we can load in parallelogram pattern
// Actually, it is possible for the DMA engine to lay the input out for us in this way
// But there are no APIs in CUPVA to achieve it with RDF! This could be improved in future.
// Fortunately the reformat is cheap, ~15us or so for 1920x1080 input
struct param_reformat
{
    AgenCFG input;
    AgenCFG inputTail;
    AgenCFG output;
    AgenCFG outputTail;
};

void init_reformat(param_reformat *param, uint32_t inputLp, void *inputBase, uint32_t inputCbSize, void *output);
void exec_reformat(param_reformat *param, void *inTile, bool isTail);

// Pass 2 - Record initial labels and equivalent pairs
// We use scratch buffers to pass current label values between tiles
struct param_label
{
    AgenCFG input;
    AgenCFG outputLabels;
    // laid out as upright, up, left, upleft
    uint16_t runningLabels[pva_elementsof(dvshortx) * 4];
    // current idx to use for labeling, monotonically increasing, 0 is no label
    uint16_t currentIdx;
    // scratch buffer to hold labels from previous tilerow
    uint16_t *previousRowLabels;
    // current offset where we are reading from in the previousRowLabels buffer
    uint32_t previousRowOffset;
};

inline void reset_label_for_row(param_label *param)
{
    vshortx vzero = replicateh(0);
    vshort *out   = (vshort *)&param->runningLabels[0];

    //(*out++) = extract(vzero); // Upright
    //(*out++) = extract(vzero); // Up
    //(*out++) = extract(vzero); // Upleft
    //(*out)   = extract(vzero); // Left
    // LLVM workaround: dvshort extract(dvshortx)
    for (int32_t i = 0; i < 8; i++) chess_unroll_loop(*)
    {
        *out++ = extract(vzero);
    }

    // First idx of upright must come from the previous row
    param->runningLabels[0] = param->previousRowLabels[CCL_HALOX + 1];
    // First idx of up must come from the previous row
    param->runningLabels[pva_elementsof(dvshortx)] = param->previousRowLabels[CCL_HALOX];
    // Reset the row offset
    param->previousRowOffset = CCL_HALOX + 2;
}

void init_label(param_label *param, void *input, uint32_t outputLaneOffset, uint16_t *previousRowLabels,
                uint16_t *labelLut);
int32_t exec_label(param_label *param, uint16_t *equivalentPairs, uint16_t *labelLut, void *intermediateOutput,
                   uint32_t intermediateLp, bool isTail);

// Pass 3 - Collapse the label tree
void exec_tree_collapse(uint16_t *labelLut, uint16_t *tempBuffer, param_label *param);

// Pass 4 - Reformat for relabelling - remove parallelogram skew
struct param_relabel_reformat
{
    AgenCFG input;
    AgenCFG output;
    uint32_t niter;
    uint32_t labelOverflow;
};

void init_relabel_reformat(param_relabel_reformat *param, uint32_t inputLp, void *inputBase, uint32_t inputCbSize,
                           void *output);
void exec_relabel_reformat(param_relabel_reformat *param, void *input);

void init_relabel_sampler(CupvaSampler *sampler, uint16_t *input, uint16_t *labelsLUT, uint16_t *output);

void init_edge_map(param_relabel_reformat *param, uint16_t *input, uint32_t outputLp);
void exec_edge_map(param_relabel_reformat *param, void *output);

#endif // CANNY_EDGE_DETECTOR_CCL_KNLS_H