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

#ifndef BACK_SUB_KNL_H
#define BACK_SUB_KNL_H

#include <background_subtractor_params.h>
#include <cupva_device.h>

const int TILE_P_PIX    = 66;
const int TILE_P_PIXF   = 33;
const int TILE_P_MODEL  = 161; // TILE_W * NUM_MODELS + 1
const int TILE_P_WTNMOD = 193;
const int TILE_P_INT    = 33;

#ifdef FP16
typedef hfloat gmm_t;
typedef short gmm_int_t;
#else
typedef float gmm_t;
typedef int gmm_int_t;
#endif
typedef uint8_t pixel_t;

const float TOL_F = 0.0001;
// Ratio to classify as background
const float BG_RATIO_F = 0.9f;
// Threshold for matching model
const float VAR_GEN_F = 3.0f * 3.0f;
// complex reduction threshold, used to calculate prune
const float CT_F = 0.05;
// Init variance for new model
const float VAR_INIT_F = 15.0f;
// Max allowed variance for model
const float VAR_MAX_F = 5 * VAR_INIT_F;
// Min allowed variance for model
const float VAR_MIN_F = 4.0f;
const float EPSILON_F = 0.000001f;
const float TAU_F     = 0.5;
const float ZERO_F    = 0.0f;

#ifdef FP16
const gmm_t TOL      = as_hfloat(0x068e); // 1e-4
const gmm_t BG_RATIO = as_hfloat(0x3B33); // 0.9
const gmm_t VAR_GEN  = as_hfloat(0x4880); // 9.0
const gmm_t CT       = as_hfloat(0x2A66); // 0.05
const gmm_t VAR_INIT = as_hfloat(0x4B80); // 15.0
const gmm_t VAR_MAX  = as_hfloat(0x54B0); // 75.0
const gmm_t VAR_MIN  = as_hfloat(0x4400); // 4.0
const gmm_t EPSILON  = as_hfloat(0x068e); // 1e-4
const gmm_t TAU      = as_hfloat(0x3800); // 0.5
const gmm_t ZERO     = as_hfloat(0x0000); // 0.0
#else
const gmm_t TOL      = gmm_t(TOL_F);
const gmm_t BG_RATIO = gmm_t(BG_RATIO_F);
const gmm_t VAR_GEN  = gmm_t(VAR_GEN_F);
const gmm_t CT       = gmm_t(CT_F);
const gmm_t VAR_INIT = gmm_t(VAR_INIT_F);
const gmm_t VAR_MAX  = gmm_t(VAR_MAX_F);
const gmm_t VAR_MIN  = gmm_t(VAR_MIN_F);
const gmm_t EPSILON  = gmm_t(EPSILON_F);
const gmm_t TAU      = gmm_t(TAU_F);
const gmm_t ZERO     = gmm_t(ZERO_F);
#endif

#ifdef FP16
const int VECW  = 16;
const int DVECW = 32;
#else
const int VECW       = 8;
const int DVECW      = 16;
#endif

#ifdef FP16
typedef vhfloatx gmm_vec_t;
typedef dvhfloatx gmm_dvec_t;
typedef vshortx gmm_int_vec_t;
typedef dvshortx gmm_int_dvec_t;
typedef vhfloat gmm_vec_mem_t;
#else
typedef vfloatx gmm_vec_t;
typedef dvfloatx gmm_dvec_t;
typedef vintx gmm_int_vec_t;
typedef dvintx gmm_int_dvec_t;
typedef vfloat gmm_vec_mem_t;
#endif

inline float pixelToGmm(pixel_t pixel)
{
    return (float)pixel;
}

inline gmm_t max(gmm_t a, gmm_t b)
{
    return a > b ? a : b;
}

inline dvfloatx max(dvfloatx a, float b)
{
    return dvmux(a < b, b, a);
}

inline vfloatx max(vfloatx a, float b)
{
    return vmux(a < b, b, a);
}

inline vhfloatx max(vhfloatx a, hfloat b)
{
    return vmux(a < b, b, a);
}

inline gmm_t min(gmm_t a, gmm_t b)
{
    return a < b ? a : b;
}

inline dvfloatx min(dvfloatx a, float b)
{
    return dvmux(a > b, b, a);
}

inline vhfloatx min(vhfloatx a, hfloat b)
{
    return vmux(a > b, b, a);
}

inline vfloatx min(vfloatx a, float b)
{
    return vmux(a > b, b, a);
}

inline gmm_t abs1(gmm_t a)
{
    return (a > gmm_t(0)) ? a : -a;
}

inline dvfloatx abs1(dvfloatx a)
{
    return dvmux(a < 0, -a, a);
}

inline vfloatx abs1(vfloatx a)
{
    return vmux(a < 0, -a, a);
}

inline vhfloatx abs1(vhfloatx a)
{
    return vmux(a < hfloat(0), -a, a);
}

inline float reciprocal(float a)
{
    return 1.0f / a;
}

inline hfloat reciprocal(hfloat a)
{
    float r = (1.0 / float(a));
    return hfloat(r);
}

inline vfloatx reciprocal(vfloatx a)
{
    return vfrcp(a);
}

inline vhfloatx reciprocal(vhfloatx a)
{
    dvfloatx tmp = vfp16_dvfp(a);
    dvfloatx r   = dvfrcp(tmp);
    return dvfp_vfp16(r);
}

inline vintx replicate(int a)
{
    return replicatew(a);
}

inline vshortx replicate(short a)
{
    return replicateh(a);
}

inline vfloatx replicate(float a)
{
    return replicatef(a);
}

inline vhfloatx replicate(hfloat a)
{
    return replicatehf(a);
}

inline dvfloatx vlookup(float *table, dvintx index)
{
    return vlookup_16f(table, index);
}

inline dvhfloatx vlookup(hfloat *table, dvshortx index)
{
    return vlookup_32hf(table, index);
}

inline void vast(float *table, dvintx index, dvfloatx value)
{
    vast_16f(table, index, value, 0xffff);
}

inline void vast(hfloat *table, dvshortx index, dvhfloatx value)
{
    vast_32hf(table, index, value, 0xffffffff);
}

typedef struct
{
    AgenCFG agenMeanIn, agenMeanOut, agenVarIn, agenVarOut, agenWeightNModIn, agenWeightNModOut, agenPix, agenMask,
        agenMaskIn;
} BGSubConfig;

typedef struct
{
    AgenCFG agenMeanIn, agenVarIn, agenWeightNModIn, agenPix, agenMaskIn, agenMaskOut, agenBGImgOut;
} ShadowAndBGDetectConfig;

void BGSubInit(pixel_t *imgIn, gmm_t *gmmMeanIn, gmm_t *gmmMeanOut, gmm_t *gmmVarIn, gmm_t *gmmVarOut,
               gmm_t *gmmWeightIn, gmm_t *gmmWeightOut, uint8_t *maskOut, uint8_t *bgImgOut, BGSubConfig *config,
               ShadowAndBGDetectConfig *shadowDetConfig);

void BGSub(int frameNum, int tileNum, BGSubConfig *config, gmm_t varThreshold, gmm_t learningRate);

void doShadowDetection(ShadowAndBGDetectConfig *config, uint8_t shadowPixelValue);

void computeBgImg(ShadowAndBGDetectConfig *config);

#endif