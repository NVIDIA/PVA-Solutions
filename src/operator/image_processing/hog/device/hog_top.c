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

#include "hog_knl.h"
#include "hog_tdata.h"

VMEM(C, uint8_t, input_r_v, RGB_BUF_SIZE * 2);
// buffer overlay: consolidated hist with magnitudes/bins results
VMEM(C, SubHistOutput, subHistOutput);
// overlapped write need 64 more bytes
VMEM(C, OUT_TYPE, output_v, (FEATURE_SIZE * ORIENTATIONS_NUM + 64) * 2);
VMEM(C, uint8_t, input_b_v, RGB_BUF_SIZE * 2);

VMEM(B, uint8_t, input_g_v, RGB_BUF_SIZE * 2);
VMEM(B, float, norm, HIST_SIZE_P);
// divide the hist into 2 parts evenly to feed into each superbank, last row has partial results: [0, (HIST_HEIGHT >> 1) - 1]
// one simd(32 lanes) will cover 8 cells, each cell have 18 bins, for 8way histogram, we need (18 * 8 /4*64=2304 bytes padding)
VMEM(B, uint16_t, hist0, (HIST_SIZE * ORIENTATIONS_NUM * HIST_PARALLISM) / 2 + 1152);

// trigger and params
VMEM_SEQDF_HANDLER(A, inRHdl);
VMEM_SEQDF_HANDLER(A, inGHdl);
VMEM_SEQDF_HANDLER(A, inBHdl);
VMEM_SEQDF_HANDLER(A, outHdl);
VMEM(A, TORgbU8Param, rgbParam);
VMEM(A, BpeParam, bpeParam, 3);
VMEM(A, CalcMagBinParam, magBinParam);
VMEM(A, CalcHistParam, histParam);
VMEM(A, ConsHistParam, consParam, 2);
VMEM(A, NormParam, normParam);
VMEM(A, CoeffParam, coeffParam);
VMEM(A, HogOutParam, hogOutParam);

// n1~n4 to produce HOG features
VMEM(A, float, coeff, COEF_SIZE_PER_PLANE * 4);
// divide the hist into 2 parts evenly to feed into each superbank, first row has partial results: [(HIST_HEIGHT >> 1), HIST_HEIGHT - 1]
VMEM(A, uint16_t, hist1, (HIST_SIZE * ORIENTATIONS_NUM * HIST_PARALLISM) / 2 + 1152);

CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(inRHdl);
    cupvaSQDFOpen(inGHdl);
    cupvaSQDFOpen(inBHdl);
    cupvaSQDFOpen(outHdl);

    cupvaSQDFTrig(inRHdl);
    cupvaSQDFTrig(inGHdl);
    cupvaSQDFTrig(inBHdl);
    const int32_t batchNum = B_WIDTH * B_HEIGHT;

    // agen setup for gardient pre-processing
    const int32_t rgbOfst = CELL_SIZE * RGB_LP + CELL_SIZE;
    // manually pad surrounding pixels, only support pad size == CELL_SIZE for BEPExec
    produceBPEAgenInit(CELL_SIZE, CELL_SIZE, INPUT_WIDTH, INPUT_HEIGHT, RGB_LP, &bpeParam[0]);
    produceBPEAgenInit(CELL_SIZE, CELL_SIZE, INPUT_WIDTH, INPUT_HEIGHT, RGB_LP, &bpeParam[1]);
    produceBPEAgenInit(CELL_SIZE, CELL_SIZE, INPUT_WIDTH, INPUT_HEIGHT, RGB_LP, &bpeParam[2]);

    // agen setup for gradient/histogram processing
    calcMagBinAgenInit(RGB_WIDTH, RGB_HEIGHT, RGB_LP, &subHistOutput.magBin.mag[0], &subHistOutput.magBin.bin[0],
                       BIN_LP, &magBinParam);
    calcHistAgenInit(&subHistOutput.magBin.mag[0], &subHistOutput.magBin.bin[0], &hist0[0], &hist1[0], BIN_LP,
                     (RGB_HEIGHT - 2), BIN_LP, ORIENTATIONS_NUM, &histParam);

    // agen setup for 8-way histogram consolidation, each plane will padded to be multiple of 64-byte
    consolidateHistAgenInit(&hist0[0], &subHistOutput.hist[0], ORIENTATIONS_NUM, HIST_WIDTH, (HIST_HEIGHT >> 1),
                            HIST_SIZE_P, &consParam[0]);
    consolidateHistAgenInit(&hist1[0], &subHistOutput.hist[HIST_SIZE >> 1], ORIENTATIONS_NUM, HIST_WIDTH,
                            (HIST_HEIGHT >> 1), HIST_SIZE_P, &consParam[1]);

    // agen setup for Q7 normalized hist in fp32
    normAgenInit(&subHistOutput.hist[0], &norm[0], ORIENTATIONS_NUM, HIST_SIZE_P, &normParam);

    // agen setup for n1~n4 to produce HOG features
    genCoeffAgenInit(&norm[0], &coeff[0], FEATURE_WIDTH, FEATURE_HEIGHT, HIST_WIDTH, COEF_SIZE_PER_PLANE, &coeffParam);

    // agen setup for the final output HOG features
    genHogFeaturesAgenInit(&subHistOutput.hist[0], &coeff[0], &output_v[0], ORIENTATIONS_NUM, HIST_SIZE_P, HIST_WIDTH,
                           FEATURE_WIDTH, FEATURE_HEIGHT, COEF_SIZE_PER_PLANE, &hogOutParam);

    int32_t fg = 0;

    for (int i = 0; i < batchNum; i++)
    {

        memset_opt(&hist0[0], 0, ((HIST_SIZE * ORIENTATIONS_NUM * HIST_PARALLISM) / 2) * sizeof(uint16_t));
        memset_opt(&hist1[0], 0, ((HIST_SIZE * ORIENTATIONS_NUM * HIST_PARALLISM) / 2) * sizeof(uint16_t));

        const int32_t curRGBOfst = fg * RGB_BUF_SIZE + rgbOfst;
        cupvaModifyAgenCfgBase(&bpeParam[0].cfgs[0], (void *)&input_r_v[curRGBOfst]);
        cupvaModifyAgenCfgBase(&bpeParam[0].cfgs[1], (void *)&input_r_v[curRGBOfst - RGB_LP]);
        cupvaModifyAgenCfgBase(&bpeParam[0].cfgs[2], (void *)&input_r_v[curRGBOfst - CELL_SIZE * RGB_LP]);
        cupvaModifyAgenCfgBase(&bpeParam[0].cfgs[3], (void *)&input_r_v[curRGBOfst - CELL_SIZE * RGB_LP - CELL_SIZE]);
        cupvaModifyAgenCfgBase(&bpeParam[1].cfgs[0], (void *)&input_g_v[curRGBOfst]);
        cupvaModifyAgenCfgBase(&bpeParam[1].cfgs[1], (void *)&input_g_v[curRGBOfst - RGB_LP]);
        cupvaModifyAgenCfgBase(&bpeParam[1].cfgs[2], (void *)&input_g_v[curRGBOfst - CELL_SIZE * RGB_LP]);
        cupvaModifyAgenCfgBase(&bpeParam[1].cfgs[3], (void *)&input_g_v[curRGBOfst - CELL_SIZE * RGB_LP - CELL_SIZE]);
        cupvaModifyAgenCfgBase(&bpeParam[2].cfgs[0], (void *)&input_b_v[curRGBOfst]);
        cupvaModifyAgenCfgBase(&bpeParam[2].cfgs[1], (void *)&input_b_v[curRGBOfst - RGB_LP]);
        cupvaModifyAgenCfgBase(&bpeParam[2].cfgs[2], (void *)&input_b_v[curRGBOfst - CELL_SIZE * RGB_LP]);
        cupvaModifyAgenCfgBase(&bpeParam[2].cfgs[3], (void *)&input_b_v[curRGBOfst - CELL_SIZE * RGB_LP - CELL_SIZE]);

        cupvaSQDFSync(inRHdl);
        cupvaSQDFSync(inGHdl);
        cupvaSQDFSync(inBHdl);
        if (i != batchNum - 1)
        {
            cupvaSQDFTrig(inRHdl);
            cupvaSQDFTrig(inGHdl);
            cupvaSQDFTrig(inBHdl);
        }

        // do the BPE for RGB plane, pre-processing for gradient calculating
        produceBPEExec(&bpeParam[0]);
        produceBPEExec(&bpeParam[1]);
        produceBPEExec(&bpeParam[2]);

        const int32_t curHistOfst = fg * RGB_BUF_SIZE;
        cupvaModifyAgenCfgBase(&magBinParam.cfgs[0], (void *)&input_r_v[curHistOfst + RGB_LP]);
        cupvaModifyAgenCfgBase(&magBinParam.cfgs[1], (void *)&input_g_v[curHistOfst + RGB_LP]);
        cupvaModifyAgenCfgBase(&magBinParam.cfgs[2], (void *)&input_b_v[curHistOfst + RGB_LP]);
        cupvaModifyAgenCfgBase(&magBinParam.cfgs[3], (void *)&input_r_v[curHistOfst + 1]);
        cupvaModifyAgenCfgBase(&magBinParam.cfgs[4], (void *)&input_g_v[curHistOfst + 1]);
        cupvaModifyAgenCfgBase(&magBinParam.cfgs[5], (void *)&input_b_v[curHistOfst + 1]);

        // calculate the gradient and histogram
        calcMagBinExec(&magBinParam);
        calcHistExec(&histParam);

        // consolidate the 8 way hist in CHW format
        consolidateHistExec(&consParam[0]);
        consolidateHistExec(&consParam[1]);

        // normalize
        normExec(&normParam);

        // generate HOG coeff n1~n4
        genCoeffExec(&coeffParam);

        // generate HOG features in CHW format
        genHogFeaturesExec(&hogOutParam);

        if (i != 0)
        {
            cupvaSQDFSync(outHdl);
        }
        cupvaSQDFTrig(outHdl);

        fg ^= 0x1;
        const int32_t outOfst = fg * (FEATURE_SIZE * ORIENTATIONS_NUM + 64);
        cupvaModifyAgenCfgBase(&hogOutParam.cfgs[2], (void *)&output_v[outOfst]);
    }
    cupvaSQDFSync(outHdl);

    cupvaSQDFClose(inRHdl);
    cupvaSQDFClose(inGHdl);
    cupvaSQDFClose(inBHdl);
    cupvaSQDFClose(outHdl);

    return 0;
}