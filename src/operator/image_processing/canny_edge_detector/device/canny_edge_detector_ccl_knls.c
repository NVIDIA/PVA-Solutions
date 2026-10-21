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
#include "canny_edge_detector_ccl_knls.h"

#include "canny_edge_detector_params.h"

void init_reformat(param_reformat *param, uint32_t inputLp, void *inputBase, uint32_t inputCbSize, void *output)
{
    // agen for the input
    AgenWrapper input_wrapper;
    input_wrapper.size = sizeof(uint8_t);
    // Iterate inner dim (cols, loaded in dvecs)
    input_wrapper.n1 = (CCL_TILEW - 1) / pva_elementsof(dvcharx) + 1;
    input_wrapper.s1 = pva_elementsof(dvcharx); // jump to next vector
    // Outer dim (rows with -2 offset per row)
    input_wrapper.n2 = CCL_TILEH;
    input_wrapper.s2 = inputLp - 2;
    agen input_agen  = init((vint *)NULL); // placeholder
    INIT_AGEN2(input_agen, input_wrapper);
    input_agen   = update_agen_cb_start(input_agen, (intptr_t)inputBase);
    input_agen   = update_agen_cb_size(input_agen, inputCbSize);
    param->input = extract_agen_cfg(input_agen);

    // agen for the output
    AgenWrapper output_wrapper;
    output_wrapper.size = sizeof(uint8_t);
    output_wrapper.n1   = input_wrapper.n1;
    output_wrapper.s1   = input_wrapper.s1;
    output_wrapper.n2   = input_wrapper.n2;
    output_wrapper.s2   = CCL_REFORMAT_TILE_LP;
    agen output_agen    = init((vint *)output);
    INIT_AGEN2(output_agen, output_wrapper);
    param->output = extract_agen_cfg(output_agen);

    // We slightly modify these for tail - add right edge halo
    input_wrapper.n1     = (CCL_TILEW + CCL_HALOX - 1) / pva_elementsof(dvcharx) + 1;
    agen input_agen_tail = init((vint *)NULL); // placeholder
    INIT_AGEN2(input_agen_tail, input_wrapper);
    input_agen_tail  = update_agen_cb_start(input_agen_tail, (intptr_t)inputBase);
    input_agen_tail  = update_agen_cb_size(input_agen_tail, inputCbSize);
    param->inputTail = extract_agen_cfg(input_agen_tail);

    output_wrapper.n1     = input_wrapper.n1;
    agen output_agen_tail = init((vint *)output);
    INIT_AGEN2(output_agen_tail, output_wrapper);
    param->outputTail = extract_agen_cfg(output_agen_tail);
}

void exec_reformat(param_reformat *param, void *inTile, bool isTail)
{
    AgenCFG *inputCfg = isTail ? &param->inputTail : &param->input;
    agen_A inputAgen  = init_agen_A_from_cfg(*inputCfg);
    inputAgen.a       = (void chess_storage(RAM_Ab) *)inTile;
    adv_agen_base_A(inputAgen, CCL_HALOX);
    AgenCFG *outputCfg = isTail ? &param->outputTail : &param->output;
    agen_B outputAgen  = init_agen_B_from_cfg(*outputCfg);

    const uint32_t tileW = isTail ? CCL_TILEW + CCL_HALOX : CCL_TILEW;
    const uint32_t niter = ((tileW - 1) / pva_elementsof(dvcharx) + 1) * CCL_TILEH;

    for (uint32_t i = 0; i < niter; i++) chess_prepare_for_pipelining
    chess_unroll_loop(4)
    chess_loop_range(8, )
    {
        dvcharx input = dvuchar_load(inputAgen);
        vstore(input, outputAgen);
    }
}

static inline void get_rep(uint16_t labelA, uint16_t labelB, uint16_t *rootA, uint16_t *rootB, uint16_t *labelLut)
{
    uint16_t newLabelA = labelA;
    uint16_t newLabelB = labelB;

    while (newLabelA != labelLut[newLabelA] || newLabelB != labelLut[newLabelB])
    {
        newLabelA = labelLut[newLabelA];
        newLabelB = labelLut[newLabelB];
    }
    *rootA = newLabelA;
    *rootB = newLabelB;
}

static inline void writeEquivPairs(uint16_t *equivPairs1, uint16_t *equivPairs2, uint16_t *labelLut,
                                   int32_t equivPairsCount)
{
    for (int32_t i = 0; i < equivPairsCount; i++)
    {
        uint16_t const labelA = *equivPairs1++;
        uint16_t const labelB = *equivPairs2++;

        uint16_t repA, repB;
        get_rep(labelA, labelB, &repA, &repB, labelLut);

        uint16_t const newRoot = repA < repB ? repA : repB;
        uint16_t const oldRoot = repA + repB - newRoot;
        labelLut[oldRoot]      = newRoot;
    }
}

void init_label(param_label *param, void *input, uint32_t outputLaneOffset, uint16_t *previousRowLabels,
                uint16_t *labelLut)
{
    // agen for the input - transposed load benefiting from the earlier reformat task
    AgenWrapper input_wrapper;
    input_wrapper.size = sizeof(uint8_t);
    input_wrapper.n1   = CCL_TILEW;
    input_wrapper.s1   = 1;
    agen input_agen    = init((vint *)input);
    INIT_AGEN1(input_agen, input_wrapper);
    input_agen.lane_ofst = CCL_REFORMAT_TILE_LANE_OFFST;
    param->input         = extract_agen_cfg(input_agen);

    // agen for output intermediate labels
    AgenWrapper output_wrapper;
    output_wrapper.size = sizeof(uint16_t);
    output_wrapper.n1   = CCL_TILEW;
    output_wrapper.s1   = 1;
    agen output_agen    = init((vint *)NULL); // placeholder
    INIT_AGEN1(output_agen, output_wrapper);
    output_agen.lane_ofst = outputLaneOffset;
    param->outputLabels   = extract_agen_cfg(output_agen);

    // Start labeling from 2, as 1 is reserved for strong edges
    param->currentIdx = 2;

    // Intermediate buffer for tracking labels between rows
    param->previousRowLabels = previousRowLabels;

    // Initialize the label LUT
    uint32_t const nIter = (CCL_MAX_LABELS - 1) / pva_elementsof(dvushort) + 1;
    dvshortx vRamp;
    vRamp.lo        = vcollate_idx(replicateh(0));
    vRamp.hi        = vcollate_idx(replicateh(0)) + pva_elementsof(vushort);
    dvshort *output = (dvshort *)labelLut;
    for (uint32_t i = 0; i < nIter; i++) chess_prepare_for_pipelining
    {
        (*output++) = extract(vRamp);
        vRamp += pva_elementsof(dvushort);
    }
}

int32_t exec_label(param_label *param, uint16_t *equivalentPairs, uint16_t *labelLut, void *intermediateOutput,
                   uint32_t intermediateLp, bool isTail)
{
    // Load the running labels vectors
    dvushort *inLabels     = (dvushort *)&param->runningLabels[0];
    dvshortx vUpRightLabel = zero_extend(*inLabels++);
    dvshortx vUpLabel      = zero_extend(*inLabels++);
    dvshortx vLeftLabel    = zero_extend(*inLabels++);
    dvshortx vUpLeftLabel  = zero_extend(*inLabels);

    agen_A input = init_agen_A_from_cfg(param->input);
    if (isTail)
    {
        adv_agen_base_A(input, CCL_TILEW);
    }
    uint32_t const iters = isTail ? CCL_HALOX : CCL_TILEW;

    agen_B output = init_agen_B_from_cfg(param->outputLabels);
    output.a      = (void chess_storage(RAM_Bb) *)intermediateOutput;

    int32_t currentIdx = param->currentIdx;

    dvshortx vZero;
    vZero.lo     = replicateh(0);
    vZero.hi     = replicateh(0);
    vshortx vOne = replicateh(1);

    uint16_t *equivalentPairsPtr1 = equivalentPairs;
    uint16_t *equivalentPairsPtr2 = equivalentPairs + CCL_TILEW * CCL_TILEH * 2;

    // Temp buffer to use for permuting the vLabel result
    uint16_t tempBuffer[pva_elementsof(dvshort) + 1];
    // LLVM workaround single-vector store
    vshort *tempWriteLo = (vshort *)&tempBuffer[1];
    vshort *tempWriteHi = (vshort *)&tempBuffer[1 + pva_elementsof(vshort)];
    dvushort *tempRead  = (dvushort *)&tempBuffer[0];

    // Pre-fill label buffer with previous row's labels
    uint16_t *inputPrevRow = &param->previousRowLabels[param->previousRowOffset];

    for (uint32_t i = 0; i < iters; i++) chess_prepare_for_pipelining
    chess_loop_range(32, )
    {
        dvshortx vInData    = vuchar_dvshortx_load_transp(input);
        dvshortx vConnected = dvor3(vUpRightLabel, vUpLabel, vUpLeftLabel) || vLeftLabel;
        dvshortx vStrong    = vInData == EDGE_STRONG;
        dvshortx vWeak      = vInData == EDGE_WEAK;

        dvshortx vNeedsNewLabel = vWeak & (!vConnected);
        int32_t loCount         = vsumr_s(vNeedsNewLabel.lo);
        int32_t hiCount         = vsumr_s(vNeedsNewLabel.hi);
        dvshortx vNewLabel;
        vNewLabel.lo = vexpand_idx(vNeedsNewLabel.lo) + currentIdx;
        vNewLabel.hi = vexpand_idx(vNeedsNewLabel.hi) + (currentIdx + loCount);
        // Increment for the next iteration
        currentIdx += loCount + hiCount;

        dvshortx vLabel = dvmux(vUpRightLabel, vUpRightLabel, vNewLabel);
        vLabel          = dvmux(vUpLeftLabel, vUpLeftLabel, vLabel);
        vLabel          = dvmux(vLeftLabel, vLeftLabel, vLabel);
        vLabel          = dvmux(vUpLabel, vUpLabel, vLabel);
        vLabel          = dvmux(vInData, vLabel, vZero);

        dvshortx vResolve = vLeftLabel || vUpLeftLabel;
        vResolve          = vUpRightLabel && vResolve;
        vResolve          = (!vUpLabel) & vResolve;
        vResolve          = (vLabel != vUpRightLabel) & vResolve;
        vResolve          = vLabel && vResolve;

        // 1. If the pixel is a strong edge
        dvshortx vPLabel = dvmux(vStrong, 1, vLabel);
        // 2. If the pixel is connected
        // add equivalent pairs vLabel and 1
        dvshortx vPred = vConnected & vStrong;
        vPred          = vPred & (vLabel != 1);
        vPred          = vPred & (vUpRightLabel != 1); // TO TRY

        // Store output labels twice, which allows us to read shifted version of vLabel
        *tempWriteLo  = extract(vPLabel.lo);
        *tempWriteHi  = extract(vPLabel.hi);
        tempBuffer[0] = *inputPrevRow++;
        vstore_transp(vPLabel, output);

        uint32_t adv1 = vsumr_s(vResolve.lo);
        uint32_t adv2 = vsumr_s(vResolve.hi);

        dvshortx vEquivIdx;
        dvshortx vEquivLabels1, vEquivLabels2;
        vEquivIdx.lo     = vcollate_idx(vResolve.lo);
        vEquivLabels1.lo = vpermute(vLabel.lo, vEquivIdx.lo);
        vEquivLabels2.lo = vpermute(vUpRightLabel.lo, vEquivIdx.lo);

        vEquivIdx.hi     = vcollate_idx(vResolve.hi);
        vEquivLabels1.hi = vpermute(vLabel.hi, vEquivIdx.hi);
        vEquivLabels2.hi = vpermute(vUpRightLabel.hi, vEquivIdx.hi);

        *((vshort *)equivalentPairsPtr1) = extract(vEquivLabels1.lo);
        *((vshort *)equivalentPairsPtr2) = extract(vEquivLabels2.lo);
        equivalentPairsPtr1 += adv1;
        equivalentPairsPtr2 += adv1;
        *((vshort *)equivalentPairsPtr1) = extract(vEquivLabels1.hi);
        *((vshort *)equivalentPairsPtr2) = extract(vEquivLabels2.hi);
        equivalentPairsPtr1 += adv2;
        equivalentPairsPtr2 += adv2;

        uint32_t adv3 = vsumr_s(vPred.lo);
        uint32_t adv4 = vsumr_s(vPred.hi);

        vEquivIdx.lo     = vcollate_idx(vPred.lo);
        vEquivIdx.hi     = vcollate_idx(vPred.hi);
        vEquivLabels1.lo = vpermute(vLabel.lo, vEquivIdx.lo);
        vEquivLabels1.hi = vpermute(vLabel.hi, vEquivIdx.hi);

        *((vshort *)equivalentPairsPtr1) = extract(vEquivLabels1.lo);
        *((vshort *)equivalentPairsPtr2) = extract(vOne);
        equivalentPairsPtr1 += adv3;
        equivalentPairsPtr2 += adv3;
        *((vshort *)equivalentPairsPtr1) = extract(vEquivLabels1.hi);
        *((vshort *)equivalentPairsPtr2) = extract(vOne);
        equivalentPairsPtr1 += adv4;
        equivalentPairsPtr2 += adv4;

        // Shuffle over the labels
        vUpLeftLabel  = vUpLabel;
        vUpLabel      = vUpRightLabel;
        vLeftLabel    = vPLabel;
        vUpRightLabel = zero_extend(*tempRead);
    }

    // Store the running labels vectors...
    dvshort *outLabels = (dvshort *)&param->runningLabels[0];
    *outLabels++       = extract(vUpRightLabel);
    *outLabels++       = extract(vUpLabel);
    *outLabels++       = extract(vLeftLabel);
    *outLabels++       = extract(vUpLeftLabel);

    // Save the last row labels from this tile for the next tile row
    dvshort *outputLastRow     = (dvshort *)&param->previousRowLabels[param->previousRowOffset - CCL_HALOX];
    uint16_t *lastRowInTile    = (uint16_t *)intermediateOutput + (CCL_TILEH - 1) * intermediateLp;
    dvushort *lastRowInTileVec = (dvushort *)lastRowInTile;
    for (uint32_t i = 0; i < CCL_TILEW / pva_elementsof(dvshortx); i++) chess_loop_range(4, 4)
    chess_unroll_loop(*)
    {
        dvshortx vLastRow = zero_extend(*lastRowInTileVec++);
        *outputLastRow++  = extract(vLastRow);
    }

    // Update the state for the next tile...
    param->currentIdx = currentIdx;
    param->previousRowOffset += CCL_TILEW;

    if (currentIdx >= CCL_MAX_LABELS)
    {
        return 1;
    }
    chess_separator_scheduler();

    int32_t const equivPairsCount = equivalentPairsPtr1 - equivalentPairs;
    writeEquivPairs(equivalentPairs, equivalentPairs + CCL_TILEW * CCL_TILEH * 2, labelLut, equivPairsCount);
    return 0;
}

void exec_tree_collapse(uint16_t *labelLut, uint16_t *tempBuffer, param_label *param)
{
    AgenCFG checkLut;
    CupvaSampler samplers[CCL_COLLAPSE_ITERATIONS];
    uint32_t const checkIters = (CCL_COLLAPSE_TILESIZE - 1) / pva_elementsof(dvshortx) + 1;
    uint32_t const nIter      = (param->currentIdx - 1) / CCL_COLLAPSE_TILESIZE + 1;

    // agen we will use to compare the n-1 iteration to the n iteration
    // if these match, we are done with lookups
    AgenWrapper input_wrapper;
    input_wrapper.size = sizeof(uint16_t);
    input_wrapper.n1   = checkIters;
    input_wrapper.s1   = pva_elementsof(dvshortx); // jump to next vector
    agen input_agen    = init(labelLut);           // placeholder
    INIT_AGEN1(input_agen, input_wrapper);
    checkLut = extract_agen_cfg(input_agen);

    // Setup the samplers
    CupvaSamplerIndices1D indicesBuffer = {
        .data               = NULL,
        .type               = SAMPLER_INDEX_TYPE_U16,
        .width              = CCL_COLLAPSE_TILESIZE,
        .height             = 1,
        .fractionalHandling = SAMPLER_FRAC_HANDLING_TRUNCATE,
    };
    CupvaSamplerInput1D lutBuffer = {
        .data           = NULL,
        .type           = SAMPLER_INPUT_TYPE_U16,
        .length         = CCL_MAX_LABELS,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT,
    };
    CupvaSamplerOutput outputBuffer = {
        .data      = NULL,
        .transMode = TRANS_MODE_NONE,
    };

    // These samplers are doing the same thing, but we need a temp buffer so we can check
    // for the convergence condition
    cupvaSamplerSetup(&samplers[0], &lutBuffer, &indicesBuffer, &outputBuffer);
    cupvaSamplerSetup(&samplers[1], &lutBuffer, &indicesBuffer, &outputBuffer);
    cupvaSamplerLink(&samplers[0], &samplers[1]);

    chess_memory_fence();

    for (uint32_t i = 0; i < nIter; i++)
    {
        cupvaSamplerUpdateAddrs(&samplers[0], &labelLut[0], &labelLut[i * CCL_COLLAPSE_TILESIZE], tempBuffer);
        cupvaSamplerUpdateAddrs(&samplers[1], &labelLut[0], tempBuffer, &labelLut[i * CCL_COLLAPSE_TILESIZE]);

        bool converged = false;

        while (!converged)
        {
            cupvaSamplerStart(&samplers[0]);

            agen_B tempCheck = init_agen_B_from_cfg(checkLut);
            tempCheck.a      = (void chess_storage(RAM_Bb) *)&tempBuffer[0];
            agen_C lutCheck  = init_agen_C_from_cfg(checkLut);
            lutCheck.a       = (void chess_storage(RAM_Cb) *)&labelLut[i * CCL_COLLAPSE_TILESIZE];

            dvshortx vCheck;
            vCheck.lo = replicateh(1);
            vCheck.hi = replicateh(1);

            cupvaSamplerWait();

            chess_separator_scheduler();

            for (uint32_t j = 0; j < checkIters; j++) chess_prepare_for_pipelining
            chess_loop_range(24, )
            chess_unroll_loop(8)
            {
                dvshortx vTemp = dvushort_load(tempCheck);
                dvshortx vLut  = dvushort_load(lutCheck);
                vCheck &= (vTemp == vLut);
            }

            converged = vandr_s(vCheck.lo) && vandr_s(vCheck.hi);
        }
    }
}

void init_relabel_reformat(param_relabel_reformat *param, uint32_t inputLp, void *inputBase, uint32_t inputCbSize,
                           void *output)
{
    // agen for the input
    AgenWrapper input_wrapper;
    input_wrapper.size = sizeof(uint16_t);
    // Iterate inner dim (cols, loaded in dvecs)
    input_wrapper.n1 = (CCL_TILEW - 1) / pva_elementsof(dvshortx) + 1;
    input_wrapper.s1 = pva_elementsof(dvshortx); // jump to next vector
    // Outer dim (rows with +2 offset per row)
    input_wrapper.n2 = CCL_TILEH;
    input_wrapper.s2 = inputLp + 2;
    agen input_agen  = init((vint *)NULL); // placeholder
    INIT_AGEN2(input_agen, input_wrapper);
    input_agen   = update_agen_cb_start(input_agen, (intptr_t)inputBase);
    input_agen   = update_agen_cb_size(input_agen, inputCbSize * sizeof(uint16_t));
    param->input = extract_agen_cfg(input_agen);

    // agen for the output
    AgenWrapper output_wrapper;
    output_wrapper.size = sizeof(uint16_t);
    output_wrapper.n1   = input_wrapper.n1;
    output_wrapper.s1   = input_wrapper.s1;
    output_wrapper.n2   = input_wrapper.n2;
    output_wrapper.s2   = CCL_TILEW;
    agen output_agen    = init((vint *)output);
    INIT_AGEN2(output_agen, output_wrapper);
    param->output = extract_agen_cfg(output_agen);
    param->niter  = ((CCL_TILEW - 1) / pva_elementsof(dvshortx) + 1) * CCL_TILEH;
}

void exec_relabel_reformat(param_relabel_reformat *param, void *input)
{
    agen_A inputAgen  = init_agen_A_from_cfg(param->input);
    inputAgen.a       = (void chess_storage(RAM_Ab) *)input;
    agen_A outputAgen = init_agen_A_from_cfg(param->output);

    const uint32_t niter = param->niter;

    for (uint32_t i = 0; i < niter; i++) chess_prepare_for_pipelining
    chess_loop_range(16, )
    chess_unroll_loop(8)
    {
        dvshortx input = dvushort_load(inputAgen);
        vstore(input, outputAgen);
    }
}

void init_relabel_sampler(CupvaSampler *sampler, uint16_t *input, uint16_t *labelsLUT, uint16_t *output)
{
    CupvaSamplerIndices1D indicesBuffer = {
        .data               = input,
        .type               = SAMPLER_INDEX_TYPE_U16,
        .width              = CCL_TILEW,
        .height             = CCL_TILEH,
        .fractionalHandling = SAMPLER_FRAC_HANDLING_TRUNCATE,
    };

    CupvaSamplerInput1D inputBuffer = {
        .data           = labelsLUT,
        .type           = SAMPLER_INPUT_TYPE_U16,
        .length         = CCL_MAX_LABELS,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT,
    };

    CupvaSamplerOutput outputBuffer = {
        .data      = output,
        .pitch     = CCL_TILEW,
        .transMode = TRANS_MODE_NONE,
    };

    cupvaSamplerSetup(sampler, &inputBuffer, &indicesBuffer, &outputBuffer);
}

void init_edge_map(param_relabel_reformat *param, uint16_t *input, uint32_t outputLp)
{
    AgenWrapper input_wrapper;
    input_wrapper.size = sizeof(uint16_t);
    input_wrapper.n1   = (CCL_TILEW - 1) / pva_elementsof(dvshortx) + 1;
    input_wrapper.s1   = pva_elementsof(dvshortx);
    input_wrapper.n2   = CCL_TILEH;
    input_wrapper.s2   = CCL_TILEW;
    agen input_agen    = init((vint *)input);
    INIT_AGEN2(input_agen, input_wrapper);
    param->input = extract_agen_cfg(input_agen);

    AgenWrapper output_agenwrapper;
    output_agenwrapper.size = sizeof(uint8_t);
    output_agenwrapper.n1   = input_wrapper.n1;
    output_agenwrapper.s1   = input_wrapper.s1;
    output_agenwrapper.n2   = CCL_TILEH;
    output_agenwrapper.s2   = outputLp;
    agen output_agen        = init((vint *)NULL); // placeholder
    INIT_AGEN2(output_agen, output_agenwrapper);
    param->output        = extract_agen_cfg(output_agen);
    param->niter         = ((CCL_TILEW - 1) / pva_elementsof(dvshortx) + 1) * CCL_TILEH;
    param->labelOverflow = 0;
}

void exec_edge_map(param_relabel_reformat *param, void *output)
{
    agen_A inputAgen  = init_agen_A_from_cfg(param->input);
    agen_B outputAgen = init_agen_B_from_cfg(param->output);
    outputAgen.a      = (void chess_storage(RAM_Bb) *)output;

    if (param->labelOverflow)
    {
        dvshortx vOverflow;
        vOverflow.lo = replicateh(1);
        vOverflow.hi = replicateh(1);

        for (uint32_t i = 0; i < param->niter; i++) chess_prepare_for_pipelining
        chess_loop_range(16, )
        chess_unroll_loop(8)
        {
            vstore_hb(vOverflow, outputAgen);
        }
        return;
    }

    dvshortx vZero;
    vZero.lo = replicateh(0);
    vZero.hi = replicateh(0);

    const uint32_t niter = param->niter;

    for (uint32_t i = 0; i < niter; i++) chess_prepare_for_pipelining
    chess_loop_range(16, )
    chess_unroll_loop(8)
    {
        dvshortx vLabel = dvushort_load(inputAgen);
        dvshortx vOut   = dvmux(vLabel == 1, 255, vZero);
        vstore_hb(vOut, outputAgen);
    }
}
