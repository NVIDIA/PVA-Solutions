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

// clang-format off
#include <cupva_device.h>
#include <background_subtractor_knl.h>
// clang-format on

VMEM(B, pixel_t, img_in, RDF_DOUBLE(pixel_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(A, gmm_t, gmm_mean_in, RDF_DOUBLE(gmm_t, TILE_W *NUM_MODELS, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(A, gmm_t, gmm_mean_out, RDF_DOUBLE(gmm_t, TILE_W *NUM_MODELS, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, pixel_t, bg_img_out, RDF_DOUBLE(pixel_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, gmm_t, gmm_var_in, RDF_DOUBLE(gmm_t, TILE_W *NUM_MODELS, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(B, gmm_t, gmm_var_out, RDF_DOUBLE(gmm_t, TILE_W *NUM_MODELS, TILE_H, 0, 0, TRANS_MODE_1));
VMEM(B, gmm_t, gmm_weight_nmod_in, RDF_DOUBLE(gmm_t, TILE_W *(NUM_MODELS + 1), TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, gmm_t, gmm_weight_nmod_out, RDF_DOUBLE(gmm_t, TILE_W *(NUM_MODELS + 1), TILE_H, 0, 0, TRANS_MODE_1));
VMEM(C, uint8_t, FGMask, RDF_DOUBLE(uint8_t, TILE_W, TILE_H, 0, 0, TRANS_MODE_1));
VMEM_RDF_UNIFIED(A, imgInTrig);
VMEM_RDF_UNIFIED(A, outMaskTrig);
VMEM_RDF_UNIFIED(A, gmmInTrig);
VMEM_RDF_UNIFIED(A, gmmOutTrig);
VMEM_RDF_UNIFIED(A, gmmWtNModInTrig);
VMEM_RDF_UNIFIED(A, gmmWtNModOutTrig);
VMEM(A, int, frameNum);
VMEM(A, int, niter);
VMEM(A, float, varThreshold);
VMEM(A, uint8_t, detectShadow);
VMEM(A, uint8_t, shadowPixelValue);
VMEM(A, float, learningRate);
VMEM(A, int, getBGImg);

CUPVA_VPU_MAIN()
{
    cupvaRasterDataFlowOpen(imgInTrig, img_in);
    cupvaRasterDataFlowOpen(outMaskTrig, FGMask);
    cupvaRasterDataFlowOpen(gmmInTrig, gmm_mean_in);
    cupvaRasterDataFlowOpen(gmmOutTrig, gmm_mean_out);
    cupvaRasterDataFlowOpen(gmmWtNModInTrig, gmm_weight_nmod_in);
    cupvaRasterDataFlowOpen(gmmWtNModOutTrig, gmm_weight_nmod_out);

    BGSubConfig config;
    ShadowAndBGDetectConfig shadowBGDetectConfig;
    BGSubInit(img_in, gmm_mean_in, gmm_mean_out, gmm_var_in, gmm_var_out, gmm_weight_nmod_in, gmm_weight_nmod_out,
              FGMask, bg_img_out, &config, &shadowBGDetectConfig);
    gmm_t iVarThreshold = gmm_t(varThreshold);
    gmm_t iLearningRate = gmm_t(learningRate);

    for (int i = 0; i < niter; i++)
    {
        void *imgInTile            = cupvaRasterDataFlowAcquire(imgInTrig);
        uint8_t *outMaskTile       = (uint8_t *)cupvaRasterDataFlowAcquire(outMaskTrig);
        gmm_t *gmmMeanInTile       = (gmm_t *)cupvaRasterDataFlowAcquire(gmmInTrig);
        gmm_t *gmmMeanOutTile      = (gmm_t *)cupvaRasterDataFlowAcquire(gmmOutTrig);
        void *gmmWeightNModInTile  = cupvaRasterDataFlowAcquire(gmmWtNModInTrig);
        void *gmmWeightNModOutTile = cupvaRasterDataFlowAcquire(gmmWtNModOutTrig);

        ptrdiff_t offsetIn      = gmmMeanInTile - &gmm_mean_in[0];
        ptrdiff_t offsetOut     = gmmMeanOutTile - &gmm_mean_out[0];
        ptrdiff_t offsetMaskOut = outMaskTile - &FGMask[0];

        cupvaModifyAgenCfgBase(&config.agenPix, imgInTile);
        cupvaModifyAgenCfgBase(&config.agenMask, outMaskTile);
        cupvaModifyAgenCfgBase(&config.agenMeanIn, gmmMeanInTile);
        cupvaModifyAgenCfgBase(&config.agenVarIn, &gmm_var_in[0] + offsetIn);
        cupvaModifyAgenCfgBase(&config.agenWeightNModIn, gmmWeightNModInTile);
        cupvaModifyAgenCfgBase(&config.agenMeanOut, gmmMeanOutTile);
        cupvaModifyAgenCfgBase(&config.agenVarOut, &gmm_var_out[0] + offsetOut);
        cupvaModifyAgenCfgBase(&config.agenWeightNModOut, gmmWeightNModOutTile);

        BGSub(frameNum, i, &config, iVarThreshold, iLearningRate);

        if (detectShadow || getBGImg)
        {
            cupvaModifyAgenCfgBase(&shadowBGDetectConfig.agenMeanIn, gmmMeanOutTile);
            cupvaModifyAgenCfgBase(&shadowBGDetectConfig.agenVarIn, &gmm_var_out[0] + offsetOut);
            cupvaModifyAgenCfgBase(&shadowBGDetectConfig.agenWeightNModIn, gmmWeightNModOutTile);
            cupvaModifyAgenCfgBase(&shadowBGDetectConfig.agenPix, imgInTile);
            cupvaModifyAgenCfgBase(&shadowBGDetectConfig.agenMaskIn, outMaskTile);
            cupvaModifyAgenCfgBase(&shadowBGDetectConfig.agenMaskOut, outMaskTile);
            cupvaModifyAgenCfgBase(&shadowBGDetectConfig.agenBGImgOut, &bg_img_out[0] + offsetMaskOut);
        }

        if (detectShadow)
        {
            doShadowDetection(&shadowBGDetectConfig, shadowPixelValue);
        }

        if (getBGImg)
        {
            computeBgImg(&shadowBGDetectConfig);
        }

        cupvaRasterDataFlowRelease(imgInTrig);
        cupvaRasterDataFlowRelease(outMaskTrig);
        cupvaRasterDataFlowRelease(gmmInTrig);
        cupvaRasterDataFlowRelease(gmmOutTrig);
        cupvaRasterDataFlowRelease(gmmWtNModInTrig);
        cupvaRasterDataFlowRelease(gmmWtNModOutTrig);
    }

    cupvaRasterDataFlowClose(imgInTrig);
    cupvaRasterDataFlowClose(outMaskTrig);
    cupvaRasterDataFlowClose(gmmInTrig);
    cupvaRasterDataFlowClose(gmmOutTrig);
    cupvaRasterDataFlowClose(gmmWtNModInTrig);
    cupvaRasterDataFlowClose(gmmWtNModOutTrig);

    return 0;
}