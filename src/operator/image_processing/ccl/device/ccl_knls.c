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
    inputAgen.a       = (void chess_storage(RAM_Ab) *restrict)inTile;
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

void init_label(param_label *param, void *input, uint32_t outputLaneOffset, uint16_t *previousRowLabels,
                uint16_t *labelLut, uint8_t isFourWay, uint8_t threshold)
{
    // agen for the input - transposed load benefiting from the earlier reformat task
    AgenWrapper agenWrapperIn;
    agenWrapperIn.size = sizeof(uint8_t);
    agenWrapperIn.n1   = CCL_TILEW;
    agenWrapperIn.s1   = 1;
    agen inAgen        = init((vint *)input);
    INIT_AGEN1(inAgen, agenWrapperIn);
    inAgen.lane_ofst = CCL_REFORMAT_TILE_LANE_OFFST;
    param->input     = extract_agen_cfg(inAgen);

    // agen for output intermediate labels
    AgenWrapper agenWrapperOut;
    agenWrapperOut.size = sizeof(uint16_t);
    agenWrapperOut.n1   = CCL_TILEW;
    agenWrapperOut.s1   = 1;
    agen outAgen        = init((vint *)input);
    INIT_AGEN1(outAgen, agenWrapperOut);
    outAgen.lane_ofst   = outputLaneOffset;
    param->outputLabels = extract_agen_cfg(outAgen);

    // Start labeling from 1
    param->currentIdx = 1;

    // Intermediate buffer for tracking labels between rows
    param->previousRowLabels = previousRowLabels;

    param->isFourWay = isFourWay;
    param->threshold = threshold;

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

static inline void writeEquivPairs(uint16_t *equivPairs, uint16_t *labelLut, int32_t equivPairsCount)
{
    // Record the equivalent pairs by making the root of one of the trees the parent of the other
    for (int32_t i = 0; i < equivPairsCount; i++)
    {
        uint16_t const labelA = *equivPairs++;
        uint16_t const labelB = *equivPairs++;

        uint16_t repA, repB;
        get_rep(labelA, labelB, &repA, &repB, labelLut);

        uint16_t const newRoot = repA < repB ? repA : repB;
        uint16_t const oldRoot = repA + repB - newRoot;
        labelLut[oldRoot]      = newRoot;
    }
}

static int32_t exec_label_4way(param_label *param, uint16_t *equivalentPairs, uint16_t *labelLut,
                               void *intermediateOutput, uint32_t intermediateLp, bool isTail)
{
    // Load the running labels vectors
    dvushort *inLabels = (dvushort *)&param->runningLabels[0];
    // Technically 'upright' is not needed for 4-way, but we keep it here to make this similar between
    // 4-way and 8-way
    dvshortx vUpRightLabel = zero_extend(*inLabels++);
    dvshortx vUpLabel      = zero_extend(*inLabels++);
    dvshortx vLeftLabel    = zero_extend(*inLabels);

    agen_A input = init_agen_A_from_cfg(param->input);
    if (isTail)
    {
        adv_agen_base_A(input, CCL_TILEW);
    }
    uint32_t const iters = isTail ? CCL_HALOX : CCL_TILEW;
    agen_B output        = init_agen_B_from_cfg(param->outputLabels);
    output.a             = (void chess_storage(RAM_Bb) *restrict)intermediateOutput;

    int32_t currentIdx = param->currentIdx;

    dvshortx vZero;
    vZero.lo = replicateh(0);
    vZero.hi = replicateh(0);

    uint16_t *equivalentPairsPtr = equivalentPairs;

    // Temp buffer to use for permuting the vLabel result
    uint16_t tempBuffer[pva_elementsof(dvshort) + 1];
    vshort *tempWriteLo = (vshort *)&tempBuffer[1];
    vshort *tempWriteHi = (vshort *)&tempBuffer[1 + pva_elementsof(vshort)];
    dvushort *tempRead  = (dvushort *)&tempBuffer[0];

    uint8_t threshold = param->threshold;

    // Pre-fill label buffer with previous row's labels
    uint16_t *inputPrevRow = &param->previousRowLabels[param->previousRowOffset];

    for (uint32_t i = 0; i < iters; i++) chess_loop_range(32, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        dvshortx vConnected = vUpLabel || vLeftLabel;

        dvshortx vInData        = vuchar_dvshortx_load_transp(input);
        dvshortx vIsValid       = vInData > threshold;
        dvshortx vNeedsNewLabel = vIsValid && !vConnected;

        // Need to expand with single vectors, so do it twice here to fill the new labels vector
        int32_t loCount = vsumr_s(vNeedsNewLabel.lo);
        int32_t hiCount = vsumr_s(vNeedsNewLabel.hi);
        dvshortx vNewLabels;
        vNewLabels.lo = vexpand_idx(vNeedsNewLabel.lo) + currentIdx;
        vNewLabels.hi = vexpand_idx(vNeedsNewLabel.hi) + currentIdx + loCount;
        // Increment for the next iteration...
        currentIdx += loCount + hiCount;

        // Assign label OR zero
        dvshortx vLabel = dvmux(vUpLabel, vUpLabel, vLeftLabel);
        vLabel          = dvmux(vNeedsNewLabel, vNewLabels, vLabel);
        vLabel          = dvmux(vIsValid, vLabel, vZero);

        // Store output labels twice, which allows us to read shifted version of vLabel
        *tempWriteLo  = extract(vLabel.lo);
        *tempWriteHi  = extract(vLabel.hi);
        tempBuffer[0] = *inputPrevRow++;
        vstore_transp(vLabel, output);

        // Record equivalent pairs
        // Requires both up and left to be set AND for them to not match each other
        dvshortx vResolve = vLeftLabel && vUpLabel;               // 1. Must be connected to the left and up
        vResolve          = (vLeftLabel != vUpLabel) && vResolve; // 2. Left and up must not be equal
        vResolve          = vLabel && vResolve;                   // 3. Must have a label AKA this pixel is connected

        uint32_t adv1 = vsumr_s(vResolve.lo) * 2;
        // Agen for storing out equivalent pairs with interleaving...
        agen equivOut = init(equivalentPairsPtr);
        equivOut.n1   = 2;
        equivOut.mod1 = adv1 * sizeof(uint16_t);

        // Collate equivalent pairs 2x, due to missing DV pseudos...
        vshortx vEquivIdx = vcollate_idx(vResolve.lo);
        dvshortx vEquivLabels;
        vEquivLabels.lo = vpermute(vLabel.lo, vEquivIdx);
        vEquivLabels.hi = vpermute(vLeftLabel.lo, vEquivIdx);
        vstore_i(vEquivLabels, equivOut);

        vEquivIdx       = vcollate_idx(vResolve.hi);
        vEquivLabels.lo = vpermute(vLabel.hi, vEquivIdx);
        vEquivLabels.hi = vpermute(vLeftLabel.hi, vEquivIdx);
        vstore_i(vEquivLabels, equivOut);

        equivalentPairsPtr += adv1 + vsumr_s(vResolve.hi) * 2;

        // Shuffle over the labels
        vUpLabel      = vUpRightLabel;
        vLeftLabel    = vLabel;
        vUpRightLabel = zero_extend(*tempRead);
    }

    // Store the running labels vectors...
    dvshort *outLabels = (dvshort *)&param->runningLabels[0];
    *outLabels++       = extract(vUpRightLabel);
    *outLabels++       = extract(vUpLabel);
    *outLabels++       = extract(vLeftLabel);

    // Save the last row labels from this tile for the next tile row
    dvshort *outputLastRow     = (dvshort *)&param->previousRowLabels[param->previousRowOffset - CCL_HALOX];
    uint16_t *lastRowInTile    = (uint16_t *)intermediateOutput + (CCL_TILEH - 1) * intermediateLp;
    dvushort *lastRowInTileVec = (dvushort *)lastRowInTile;
    for (uint32_t i = 0; i < CCL_TILEW / pva_elementsof(dvshortx); i++) chess_prepare_for_pipelining
    {
        dvshortx vLastRow = zero_extend(*lastRowInTileVec++);
        *outputLastRow++  = extract(vLastRow);
    }

    // Update the state for the next tile...
    param->currentIdx = currentIdx;
    param->previousRowOffset += CCL_TILEW;
    if (currentIdx > CCL_MAX_LABELS)
    {
        return 1;
    }

    chess_separator_scheduler();

    int32_t const equivPairsCount = (equivalentPairsPtr - equivalentPairs) / 2;
    writeEquivPairs(equivalentPairs, labelLut, equivPairsCount);
    return 0;
}

static int32_t exec_label_8way(param_label *param, uint16_t *equivalentPairs, uint16_t *labelLut,
                               void *intermediateOutput, uint32_t intermediateLp, bool isTail)
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
    output.a      = (void chess_storage(RAM_Bb) *restrict)intermediateOutput;

    int32_t currentIdx = param->currentIdx;

    dvshortx vZero;
    vZero.lo = replicateh(0);
    vZero.hi = replicateh(0);

    uint16_t *equivalentPairsPtr = equivalentPairs;

    // Temp buffer to use for permuting the vLabel result
    uint16_t tempBuffer[pva_elementsof(dvshort) + 1];
    vshort *tempWriteLo = (vshort *)&tempBuffer[1];
    vshort *tempWriteHi = (vshort *)&tempBuffer[1 + pva_elementsof(vshort)];
    dvushort *tempRead  = (dvushort *)&tempBuffer[0];

    // Pre-fill label buffer with previous row's labels
    uint16_t *inputPrevRow = &param->previousRowLabels[param->previousRowOffset];

    uint8_t threshold = param->threshold;

    for (uint32_t i = 0; i < iters; i++) chess_prepare_for_pipelining
    chess_unroll_loop(2)
    chess_loop_range(32, )
    {
        dvshortx vConnected = dvor3(vUpRightLabel, vUpLabel, vUpLeftLabel) || vLeftLabel;

        dvshortx vInData        = vuchar_dvshortx_load_transp(input);
        dvshortx vIsValid       = vInData > threshold;
        dvshortx vNeedsNewLabel = vIsValid && !vConnected;

        // Need to expand with single vectors, so do it twice here to fill the new labels vector
        int32_t loCount = vsumr_s(vNeedsNewLabel.lo);
        int32_t hiCount = vsumr_s(vNeedsNewLabel.hi);
        dvshortx vNewLabels;
        vNewLabels.lo = vexpand_idx(vNeedsNewLabel.lo) + currentIdx;
        vNewLabels.hi = vexpand_idx(vNeedsNewLabel.hi) + currentIdx + loCount;
        // Increment for the next iteration...
        currentIdx += loCount + hiCount;

        // Assign label OR zero
        dvshortx vLabelBase = dvmux(vUpLeftLabel, vUpLeftLabel, vLeftLabel);
        dvshortx vLabel     = dvmux(vUpLabel, vUpLabel, vLabelBase);
        // By preferring upright we reduce the number of duplicate equivalent pairs
        vLabel = dvmux(vUpRightLabel, vUpRightLabel, vLabel);
        vLabel = dvmux(vNeedsNewLabel, vNewLabels, vLabel);
        vLabel = dvmux(vIsValid, vLabel, vZero);

        // Store output labels twice, which allows us to read shifted version of vLabel
        *tempWriteLo  = extract(vLabel.lo);
        *tempWriteHi  = extract(vLabel.hi);
        tempBuffer[0] = *inputPrevRow++;
        vstore_transp(vLabel, output);

        // Record equivalent pairs
        // If you work through the 16 possible combinations of the 4 connected pixels, you will see that
        // there can only be 1 equivalent pair per pixel, and this requires a few conditions to be met:
        dvshortx vResolve = vUpRightLabel && vLabelBase; // 1. Must be connected to upright && (upleft || left)
        vResolve          = (!vUpLabel) &&
                   vResolve; // 2. Must not be connected to up - such scenarios will not result in conflicting labels
        vResolve = (vLabelBase != vUpRightLabel) && vResolve; // 3. Must not be the same label as upright
        vResolve = vLabel && vResolve;                        // 4. Must have a label

        uint32_t adv1 = vsumr_s(vResolve.lo) * 2;
        // Agen for storing out equivalent pairs with interleaving...
        agen equivOut = init(equivalentPairsPtr);
        equivOut.n1   = 2;
        equivOut.mod1 = adv1 * sizeof(uint16_t);

        // Collate equivalent pairs 2x, due to missing DV pseudos...
        vshortx vEquivIdx = vcollate_idx(vResolve.lo);
        dvshortx vEquivLabels;
        vEquivLabels.lo = vpermute(vLabelBase.lo, vEquivIdx);
        vEquivLabels.hi = vpermute(vUpRightLabel.lo, vEquivIdx);
        vstore_i(vEquivLabels, equivOut);

        vEquivIdx       = vcollate_idx(vResolve.hi);
        vEquivLabels.lo = vpermute(vLabelBase.hi, vEquivIdx);
        vEquivLabels.hi = vpermute(vUpRightLabel.hi, vEquivIdx);
        vstore_i(vEquivLabels, equivOut);

        equivalentPairsPtr += adv1 + vsumr_s(vResolve.hi) * 2;

        // Shuffle over the labels
        vUpLeftLabel  = vUpLabel;
        vUpLabel      = vUpRightLabel;
        vLeftLabel    = vLabel;
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
    for (uint32_t i = 0; i < CCL_TILEW / pva_elementsof(dvshortx); i++) chess_prepare_for_pipelining
    {
        dvshortx vLastRow = zero_extend(*lastRowInTileVec++);
        *outputLastRow++  = extract(vLastRow);
    }

    // Update the state for the next tile...
    param->currentIdx = currentIdx;
    param->previousRowOffset += CCL_TILEW;

    if (currentIdx > CCL_MAX_LABELS)
    {
        return 1;
    }

    chess_separator_scheduler();

    int32_t const equivPairsCount = (equivalentPairsPtr - equivalentPairs) / 2;
    writeEquivPairs(equivalentPairs, labelLut, equivPairsCount);
    return 0;
}

int32_t exec_label(param_label *param, uint16_t *equivalentPairs, uint16_t *labelLut, void *intermediateOutput,
                   uint32_t intermediateLp, bool isTail)
{
    if (param->isFourWay)
    {
        return exec_label_4way(param, equivalentPairs, labelLut, intermediateOutput, intermediateLp, isTail);
    }
    else
    {
        return exec_label_8way(param, equivalentPairs, labelLut, intermediateOutput, intermediateLp, isTail);
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
}

void exec_relabel_reformat(param_relabel_reformat *param, void *input)
{
    agen_A inputAgen  = init_agen_A_from_cfg(param->input);
    inputAgen.a       = (void chess_storage(RAM_Ab) *restrict)input;
    agen_A outputAgen = init_agen_A_from_cfg(param->output);

    const uint32_t niter = ((CCL_TILEW - 1) / pva_elementsof(dvshortx) + 1) * CCL_TILEH;

    for (uint32_t i = 0; i < niter; i++) chess_prepare_for_pipelining
    chess_loop_range(16, )
    chess_unroll_loop(8)
    {
        dvshortx input = dvushort_load(inputAgen);
        vstore(input, outputAgen);
    }
}

void init_relabel_sampler(CupvaSampler *sampler, uint16_t *input, uint16_t *labelsLUT, uint32_t outputLp)
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
        .data      = NULL,
        .pitch     = (uint16_t)outputLp,
        .transMode = TRANS_MODE_NONE,
    };

    cupvaSamplerSetup(sampler, &inputBuffer, &indicesBuffer, &outputBuffer);
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
    // Iterate inner dim (cols, loaded in dvecs)
    input_wrapper.n1 = checkIters;
    input_wrapper.s1 = pva_elementsof(dvshortx); // jump to next vector
    agen input_agen  = init(labelLut);           // placeholder
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
            tempCheck.a      = (void chess_storage(RAM_Bb) *restrict)&tempBuffer[0];
            agen_C lutCheck  = init_agen_C_from_cfg(checkLut);
            lutCheck.a       = (void chess_storage(RAM_Cb) *restrict)&labelLut[i * CCL_COLLAPSE_TILESIZE];

            dvshortx vCheck = (chess_dont_care(dvshortx) & 0) + 1;

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

void count_labels(uint16_t *labelLut, int32_t *numLabels, param_label *param)
{
    int32_t vecw  = pva_elementsof(dvushort);
    int32_t niter = (param->currentIdx - 1) / vecw;

    // 0 is not a valid label
    dvshortx vRamp;
    vRamp.lo = vcollate_idx(replicateh(0)) + 1;
    vRamp.hi = vcollate_idx(replicateh(0)) + pva_elementsof(vushort) + 1;

    dvushort *lut = (dvushort *)(labelLut + 1);
    int32_t count = 0;

    for (int32_t i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        dvshortx vLabel = zero_extend(*lut++);

        dvshortx vEqual = vLabel == vRamp;
        count += vsumr_s(vEqual.lo);
        count += vsumr_s(vEqual.hi);
        vRamp += vecw;
    }

    chess_separator_scheduler();

    for (int32_t i = niter * vecw + 1; i < param->currentIdx; i++)
    {
        if (labelLut[i] == i)
        {
            count++;
        }
    }

    *numLabels = count;
}