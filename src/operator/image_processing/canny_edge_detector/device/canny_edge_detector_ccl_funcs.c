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

#include "canny_edge_detector_ccl_funcs.h"

#include "canny_edge_detector_params.h"

#include <cupva_device.h>
#include <cupva_device_debug.h>

VMEM_RDF_UNIFIED(A, inEdgeTypeHdl);
VMEM_RDF_UNIFIED(A, outIntermediateHdl);
VMEM_RDF_UNIFIED(A, inIntermediateHdl);
VMEM_RDF_UNIFIED(A, outHdl);

int32_t canny_edge_detector_ccl_exec(struct Stage3BufPtrs *bufPtrs, int32_t nIterX, int32_t nIterY)
{
    int32_t ret = 0;

    uint8_t *inBuffer               = bufPtrs->inBuffer;
    uint16_t *inIntermediateBuffer  = bufPtrs->inIntermediateBuffer;
    uint16_t *equivalentPairs       = bufPtrs->equivalentPairs;
    uint16_t *previousRowLabels     = bufPtrs->previousRowLabels;
    uint16_t *reformattedLabels     = bufPtrs->reformattedLabels;
    uint16_t *outIntermediateBuffer = bufPtrs->outIntermediateBuffer;
    uint8_t *outEdgeMap             = bufPtrs->outEdgeMap;
    uint8_t *reformatInput          = bufPtrs->reformatInput;
    uint16_t *tempCollapseBuffer    = bufPtrs->tempCollapseBuffer;
    uint16_t *outLabels             = bufPtrs->outLabels;
    uint16_t *labelLut              = bufPtrs->labelLut;

    // Start DMA phase 1
    cupvaRasterDataFlowOpen(inEdgeTypeHdl, inBuffer);
    cupvaRasterDataFlowOpen(outIntermediateHdl, outIntermediateBuffer);

    uint32_t const inputLp = cupvaRasterDataFlowGetLinePitch(inEdgeTypeHdl);
    uint32_t const outIntermediateLaneOffset =
        cupvaTransposeLaneOfst(outIntermediateHdl, TRANS_MODE_1, sizeof(uint16_t));
    uint32_t const outIntermediateLp = cupvaRasterDataFlowGetLinePitch(outIntermediateHdl);

    param_reformat reformatParam;
    init_reformat(&reformatParam, inputLp, inBuffer, cupvaRasterDataFlowGetCbLen(inEdgeTypeHdl), reformatInput);

    param_label labelParam;
    init_label(&labelParam, reformatInput, outIntermediateLaneOffset, previousRowLabels, labelLut);

    chess_separator_scheduler();

    for (int32_t i = 0; i < nIterY; i++)
    {
        reset_label_for_row(&labelParam);

        for (int32_t j = 0; j < nIterX; j++)
        {
            void *inTile              = cupvaRasterDataFlowAcquire(inEdgeTypeHdl);
            void *outIntermediateTile = cupvaRasterDataFlowAcquire(outIntermediateHdl);

            exec_reformat(&reformatParam, inTile, j == nIterX - 1);
            ret |= exec_label(&labelParam, equivalentPairs, labelLut, outIntermediateTile, outIntermediateLp, false);

            cupvaRasterDataFlowRelease(inEdgeTypeHdl);
            cupvaRasterDataFlowRelease(outIntermediateHdl);
        }

        void *outIntermediateTileTail = cupvaRasterDataFlowAcquire(outIntermediateHdl);
        ret |= exec_label(&labelParam, equivalentPairs, labelLut, outIntermediateTileTail, outIntermediateLp, true);
        cupvaRasterDataFlowRelease(outIntermediateHdl);
    }

    // Close DMA for phase 1
    cupvaRasterDataFlowClose(inEdgeTypeHdl);
    cupvaRasterDataFlowClose(outIntermediateHdl);

    // Phase 2 - collapse the label map with DLUT
    // We want it so that for each i in LUT, LUT[LUT[i]] == LUT[i]
    // In other words, each label points directly to a root
    // We will solve this via repeated iterations of DLUT i.e.
    // assign LUT[i] = LUT[LUT[i]] until this is a no-op

    // Start DMA phase 3
    cupvaRasterDataFlowOpen(inIntermediateHdl, inIntermediateBuffer);
    cupvaRasterDataFlowOpen(outHdl, outEdgeMap);
    if (!ret)
    {
        exec_tree_collapse(labelLut, tempCollapseBuffer, &labelParam);
    }

    // Phase 3 - relabel the outputs
    param_relabel_reformat relabelReformatParam, edgeMapParam;
    init_relabel_reformat(&relabelReformatParam, cupvaRasterDataFlowGetLinePitch(inIntermediateHdl),
                          inIntermediateBuffer, cupvaRasterDataFlowGetCbLen(inIntermediateHdl), reformattedLabels);

    init_edge_map(&edgeMapParam, outLabels, cupvaRasterDataFlowGetLinePitch(outHdl));
    edgeMapParam.labelOverflow = ret != 0;

    CupvaSampler relabelSampler;
    init_relabel_sampler(&relabelSampler, reformattedLabels, labelLut, outLabels);

    for (int32_t i = 0; i < nIterY * nIterX; i++)
    {
        void *inTile  = cupvaRasterDataFlowAcquire(inIntermediateHdl);
        void *outTile = cupvaRasterDataFlowAcquire(outHdl);

        if (!ret)
        {
            exec_relabel_reformat(&relabelReformatParam, inTile);

            cupvaSamplerStart(&relabelSampler);
            cupvaSamplerWait();
        }
        exec_edge_map(&edgeMapParam, outTile);
        cupvaRasterDataFlowRelease(inIntermediateHdl);
        cupvaRasterDataFlowRelease(outHdl);
    }

    cupvaRasterDataFlowClose(inIntermediateHdl);
    cupvaRasterDataFlowClose(outHdl);
    return 0;
}
