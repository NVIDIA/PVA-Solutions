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

#ifndef HOG_KNL_H
#define HOG_KNL_H

#include "hog_tdata.h"

#include <cupva_device.h>
#include <stdint.h>

typedef struct
{
    uint16_t bin[BIN_LP * (RGB_HEIGHT - 2)];
    uint16_t mag[BIN_LP * (RGB_HEIGHT - 2)];
} MagBin;

union SubHistOutput
{
    MagBin magBin;
    // consolidated version hist, may larger than UINT16_MAX
    uint32_t hist[HIST_SIZE_P * ORIENTATIONS_NUM];
};

typedef struct
{
    AgenCFG cfgs[5];
    int32_t niter;
} TORgbU8Param;

typedef struct
{
    AgenCFG cfgs[4];
    int32_t niter1;
    int32_t niter2;
    int32_t pred_x;
    int32_t pred_y;
} BpeParam;

typedef struct
{
    AgenCFG cfgs[8];
    int32_t niter;
} CalcMagBinParam;

typedef struct
{
    AgenCFG cfgs[2];
    int32_t niter;
    int32_t niter1;
    uint16_t *hist0;
    uint16_t *hist1;
    int32_t oriNum;
    int32_t histW;
} CalcHistParam;

typedef struct
{
    AgenCFG cfgs[6];
    int32_t niter1;
    int32_t niter2;
} ConsHistParam;

typedef struct
{
    AgenCFG cfgs[2];
    int32_t niter;
    int32_t oriNum;
} NormParam;

typedef struct
{
    AgenCFG cfgs[2];
    int32_t niter;
} CoeffParam;

typedef struct
{
    AgenCFG cfgs[3];
    int32_t niter;
} HogOutParam;

void memset_opt(void *mem, char data, int32_t size);

void toRgbAgenInitU8(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_r, uint8_t *out_g, uint8_t *out_b, const int32_t tw,
                     const int32_t th, const int32_t lp, TORgbU8Param *param);
void toRgbExec(TORgbU8Param *param);

// base here is the effective data address(regardless of paddings)
void produceBPEAgenInit(const int32_t padding_x, const int32_t padding_y, const int32_t tw, const int32_t th,
                        const int32_t lp, BpeParam *param);
void produceBPEExec(BpeParam *param);

void calcMagBinAgenInit(const int32_t twPad, const int32_t thPad, const int32_t lp, uint16_t *mag, uint16_t *bin,
                        const int32_t lp_out, CalcMagBinParam *param);
void calcMagBinExec(CalcMagBinParam *param);
void calcHistAgenInit(uint16_t *mag, uint16_t *bin, uint16_t *hist0, uint16_t *hist1, const int32_t tw,
                      const int32_t th, const int32_t lp, const int32_t oriNum, CalcHistParam *param);
void calcHistExec(CalcHistParam *param);

void consolidateHistAgenInit(uint16_t *histSrc, uint32_t *histDst, const int32_t oriNum, const int32_t tw,
                             const int32_t th, const int32_t histSizePadded, ConsHistParam *param);
void consolidateHistExec(ConsHistParam *param);

void normAgenInit(uint32_t *hist, float *norm, const int32_t oriNum, const int32_t histSizePadded, NormParam *param);
void normExec(NormParam *param);

// produce HOG coeff n1~n4
void genCoeffAgenInit(float *norm, float *coeff, const int32_t featureW, const int32_t featureH, const int32_t histW,
                      const int32_t coefSizePerPlane, CoeffParam *param);
void genCoeffExec(CoeffParam *param);

void genHogFeaturesAgenInit(uint32_t *hist, float *coeff, uint8_t *output, const int32_t oriNum,
                            const int32_t histSizePadded, const int32_t histW, const int32_t featureW,
                            const int32_t featureH, const int32_t coefSizePerPlane, HogOutParam *param);
void genHogFeaturesExec(HogOutParam *param);

#endif