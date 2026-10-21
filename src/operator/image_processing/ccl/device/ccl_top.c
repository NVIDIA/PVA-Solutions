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

#include "ccl_knls.h"
#include "ccl_params.h"

#include <cupva_device.h>

// DMA buffers
VMEM(A, uint8_t, inBuffer, RDF_CIRCULAR(uint8_t, CCL_TILEW, CCL_TILEH, CCL_HALOX, 0));
VMEM(B, uint16_t, outIntermediateBuffer, RDF_DOUBLE(uint16_t, CCL_TILEW, CCL_TILEH, 0, 0, TRANS_MODE_1));
VMEM(A, uint16_t, inIntermediateBuffer, RDF_CIRCULAR(uint16_t, CCL_TILEW, CCL_TILEH, CCL_HALOX, 0));
VMEM(B, uint16_t, outLabels, RDF_DOUBLE(uint16_t, CCL_TILEW, CCL_TILEH));

// DMA handles
VMEM_RDF_UNIFIED(A, inHdl);
VMEM_RDF_UNIFIED(A, outIntermediateHdl);
VMEM_RDF_UNIFIED(A, inIntermediateHdl);
VMEM_RDF_UNIFIED(A, outLabelsHdl);
VMEM_SEQDF_HANDLER(A, numLabelsHdl);

// Params
VMEM(A, param_ccl, algoParams);

VMEM(A, int32_t, numLabels);

// Private buffers
// Reformatted input to support transposed parallelogram access with a skew of -2 per row
VMEM(B, uint8_t, reformatInput, CCL_REFORMAT_TILE_LP *CCL_REFORMAT_TILE_H);

// Buffer to hold equivalent pairs for a single tile. Its sized for a tile which has an equivalent
// pair every single pixel, even though that is not possible.
// Also is double size to allow for collapsing with DLUT
VMEM(A, uint16_t, equivalentPairs, CCL_TILEW *CCL_TILEH * 4);

// Look-up table for label remapping - take up one full bank
VMEM(C, uint16_t, labelLut, CCL_MAX_LABELS);

// Buffer for tracking labels between tile rows
VMEM(A, uint16_t, previousRowLabels, CCL_MAX_IMG_WIDTH + 2 * CCL_HALOX);

// Temporary buffer for collapsing the label tree
VMEM(B, uint16_t, tempCollapseBuffer, CCL_COLLAPSE_TILESIZE);

// Reformatted labels - i.e. skew removed
VMEM(A, uint16_t, reformattedLabels, CCL_TILEW *CCL_TILEH);

CUPVA_VPU_MAIN()
{
    int32_t ret = 0;
    // Start DMA phase 1
    cupvaRasterDataFlowOpen(inHdl, &inBuffer[0]);
    cupvaRasterDataFlowOpen(outIntermediateHdl, &outIntermediateBuffer[0]);

    uint32_t const inputLp = cupvaRasterDataFlowGetLinePitch(inHdl);
    uint32_t const outIntermediateLaneOffset =
        cupvaTransposeLaneOfst(outIntermediateHdl, TRANS_MODE_1, sizeof(uint16_t));
    uint32_t const outIntermediateLp = cupvaRasterDataFlowGetLinePitch(outIntermediateHdl);

    param_reformat reformatParam;
    init_reformat(&reformatParam, inputLp, &inBuffer[0], cupvaRasterDataFlowGetCbLen(inHdl), &reformatInput[0]);

    param_label labelParam;
    init_label(&labelParam, &reformatInput[0], outIntermediateLaneOffset, &previousRowLabels[0], &labelLut[0],
               algoParams.isFourWay, algoParams.threshold);

    chess_separator_scheduler();

    int32_t const nIterX = algoParams.nIterX;
    int32_t const nIterY = algoParams.nIterY;

    for (int32_t i = 0; i < nIterY; i++)
    {
        reset_label_for_row(&labelParam);

        for (int32_t j = 0; j < nIterX; j++)
        {
            void *inTile              = cupvaRasterDataFlowAcquire(inHdl);
            void *outIntermediateTile = cupvaRasterDataFlowAcquire(outIntermediateHdl);

            exec_reformat(&reformatParam, inTile, j == nIterX - 1);
            ret |= exec_label(&labelParam, &equivalentPairs[0], &labelLut[0], outIntermediateTile, outIntermediateLp,
                              false);

            cupvaRasterDataFlowRelease(inHdl);
            cupvaRasterDataFlowRelease(outIntermediateHdl);
        }

        void *outIntermediateTileTail = cupvaRasterDataFlowAcquire(outIntermediateHdl);
        ret |= exec_label(&labelParam, &equivalentPairs[0], &labelLut[0], outIntermediateTileTail, outIntermediateLp,
                          true);
        cupvaRasterDataFlowRelease(outIntermediateHdl);
    }

    // Close DMA for phase 1
    cupvaRasterDataFlowClose(inHdl);
    cupvaRasterDataFlowClose(outIntermediateHdl);

    // Phase 2 - collapse the label map with DLUT
    // We want it so that for each i in LUT, LUT[LUT[i]] == LUT[i]
    // In other words, each label points directly to a root
    // We will solve this via repeated iterations of DLUT i.e.
    // assign LUT[i] = LUT[LUT[i]] until this is a no-op
    cupvaSQDFOpen(numLabelsHdl);
    if (!ret)
    {
        exec_tree_collapse(&labelLut[0], &tempCollapseBuffer[0], &labelParam);
        count_labels(&labelLut[0], &numLabels, &labelParam);
    }
    else
    {
        numLabels = -1;
    }
    cupvaSQDFTrig(numLabelsHdl);

    // Phase 3 - relabel the outputs
    param_relabel_reformat relabelReformatParam;
    init_relabel_reformat(&relabelReformatParam, cupvaRasterDataFlowGetLinePitch(inIntermediateHdl),
                          &inIntermediateBuffer[0], cupvaRasterDataFlowGetCbLen(inIntermediateHdl),
                          &reformattedLabels[0]);

    cupvaSQDFSync(numLabelsHdl);
    cupvaSQDFClose(numLabelsHdl);

    cupvaRasterDataFlowOpen(inIntermediateHdl, &inIntermediateBuffer[0]);
    cupvaRasterDataFlowOpen(outLabelsHdl, &outLabels[0]);

    CupvaSampler relabelSampler;
    init_relabel_sampler(&relabelSampler, &reformattedLabels[0], &labelLut[0],
                         cupvaRasterDataFlowGetLinePitch(outLabelsHdl));

    for (int32_t i = 0; i < nIterY * nIterX; i++)
    {

        void *inTile  = cupvaRasterDataFlowAcquire(inIntermediateHdl);
        void *outTile = cupvaRasterDataFlowAcquire(outLabelsHdl);

        if (!ret)
        {
            exec_relabel_reformat(&relabelReformatParam, inTile);
            cupvaSamplerUpdateAddrs(&relabelSampler, &labelLut[0], &reformattedLabels[0], outTile);
            cupvaSamplerStart(&relabelSampler);
            cupvaSamplerWait();
        }

        cupvaRasterDataFlowRelease(inIntermediateHdl);
        cupvaRasterDataFlowRelease(outLabelsHdl);
    }

    cupvaRasterDataFlowClose(inIntermediateHdl);
    cupvaRasterDataFlowClose(outLabelsHdl);

    return 0;
}
