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

/*
 * Gaussian Mixture Model (GMM) Background Subtraction Kernel Implementation
 *
 * This file implements a vectorized GMM-based background subtraction algorithm
 * optimized for PVA (Programmable Vision Accelerator) hardware. The algorithm
 * maintains multiple Gaussian models per pixel to model the background and
 * detect foreground objects.
 *
 * Key Algorithm Components:
 * - Each pixel maintains up to 5 Gaussian models (NUM_MODELS)
 * - Models are characterized by mean, variance, and weight
 * - Background/foreground classification based on pixel-model matching
 * - Adaptive learning with configurable learning rate
 * - Shadow detection capabilities
 * - Model sorting by weight for better performance
 */

// clang-format off
#include <cupva_device.h>
#include <background_subtractor_knl.h>
// clang-format on

VMEM(A, gmm_t, gmmMeanMem, DVECW * 5);
VMEM(B, gmm_t, gmmVarMem, DVECW * 5);
VMEM(C, gmm_t, gmmWeightMem, DVECW * 5);

/*
 * Initialize address generators for background subtraction
 * Sets up memory access patterns for efficient vectorized processing
 */
void BGSubInit(pixel_t *imgIn, gmm_t *gmmMeanIn, gmm_t *gmmMeanOut, gmm_t *gmmVarIn, gmm_t *gmmVarOut,
               gmm_t *gmmWeightNModIn, gmm_t *gmmWeightNModOut, uint8_t *maskOut, uint8_t *bgImgOut,
               BGSubConfig *config, ShadowAndBGDetectConfig *shadowBGDetConfig)
{
    // Common agen wrapper for gmm mean, variance
    AgenWrapper agenWrapperGMMNew;
    agenWrapperGMMNew.size = sizeof(gmm_t);       // Element size
    agenWrapperGMMNew.n1   = NUM_MODELS;          // models per pixel
    agenWrapperGMMNew.n2   = TILE_H / VECW;       // Tile height in vectors
    agenWrapperGMMNew.n3   = TILE_W;              // Tile width
    agenWrapperGMMNew.s1   = 1;                   // Stride between models
    agenWrapperGMMNew.s2   = TILE_P_MODEL * VECW; // Stride between rows
    agenWrapperGMMNew.s3   = NUM_MODELS;          // Stride between columns

    AgenWrapper agenWrapperPix, agenWrapperN, agenWrapperMask, agenWrapperGMMWtNMod;
    agen agenMeanIn, agenMeanOut, agenVarIn, agenVarOut, agenWeightNModIn, agenWeightNModOut, agenPix, agenMask;

    // Initialize address generators for GMM mean parameters (input and output)
    agenMeanIn = init(gmmMeanIn);
    INIT_AGEN3(agenMeanIn, agenWrapperGMMNew);
    agenMeanIn.lane_ofst = TILE_P_MODEL / (2 * VECW); // Lane offset for vectorization

    agenMeanOut = init(gmmMeanOut);
    INIT_AGEN3(agenMeanOut, agenWrapperGMMNew);
    agenMeanOut.lane_ofst = TILE_P_MODEL / (2 * VECW);

    // Initialize address generators for GMM variance parameters
    agenVarIn = init(gmmVarIn);
    INIT_AGEN3(agenVarIn, agenWrapperGMMNew);
    agenVarIn.lane_ofst = TILE_P_MODEL / (2 * VECW);

    agenVarOut = init(gmmVarOut);
    INIT_AGEN3(agenVarOut, agenWrapperGMMNew);
    agenVarOut.lane_ofst = TILE_P_MODEL / (2 * VECW);

    // Configure wrapper for weight and number of models (NUM_MODELS + 1 to include count)
    // The number of models is interleaved with the weights to avoid extra dma read/write channels
    agenWrapperGMMWtNMod.size = sizeof(gmm_t);
    agenWrapperGMMWtNMod.n1   = NUM_MODELS + 1; // 5 weights + 1 count
    agenWrapperGMMWtNMod.n2   = TILE_H / VECW;
    agenWrapperGMMWtNMod.n3   = TILE_W;
    agenWrapperGMMWtNMod.s1   = 1;
    agenWrapperGMMWtNMod.s2   = TILE_P_WTNMOD * VECW;
    agenWrapperGMMWtNMod.s3   = NUM_MODELS + 1;

    // Initialize address generators for weights and model counts
    agenWeightNModIn = init(gmmWeightNModIn);
    INIT_AGEN3(agenWeightNModIn, agenWrapperGMMWtNMod);
    agenWeightNModIn.lane_ofst = TILE_P_WTNMOD / (2 * VECW);

    agenWeightNModOut = init(gmmWeightNModOut);
    INIT_AGEN3(agenWeightNModOut, agenWrapperGMMWtNMod);
    agenWeightNModOut.lane_ofst = TILE_P_WTNMOD / (2 * VECW);

    // Configure address generator for input pixels
    agenPix             = init(imgIn);
    agenWrapperPix.size = sizeof(pixel_t);
    agenWrapperPix.n1   = TILE_H / VECW; // Tile dimensions
    agenWrapperPix.n2   = TILE_W;
    agenWrapperPix.s1   = TILE_P_PIX * VECW; // Row stride
    agenWrapperPix.s2   = 1;                 // Column stride
    INIT_AGEN2(agenPix, agenWrapperPix);
    agenPix.lane_ofst = 1;

    // Configure address generator for output mask
    agenMask             = init(maskOut);
    agenWrapperMask.size = sizeof(pixel_t);
    agenWrapperMask.n1   = TILE_H / VECW;
    agenWrapperMask.n2   = TILE_W;
    agenWrapperMask.s1   = TILE_P_PIX * VECW;
    agenWrapperMask.s2   = 1;
    INIT_AGEN2(agenMask, agenWrapperMask);
    agenMask.lane_ofst = 1;

    // Configure address generator for background image output
    agen agenBGOut = init(bgImgOut);
    INIT_AGEN2(agenBGOut, agenWrapperPix);
    agenBGOut.lane_ofst = 1;

    // Store address generator configurations for main processing functions
    config->agenMask          = extract_agen_cfg(agenMask);
    config->agenPix           = extract_agen_cfg(agenPix);
    config->agenMeanIn        = extract_agen_cfg(agenMeanIn);
    config->agenMeanOut       = extract_agen_cfg(agenMeanOut);
    config->agenVarIn         = extract_agen_cfg(agenVarIn);
    config->agenVarOut        = extract_agen_cfg(agenVarOut);
    config->agenWeightNModIn  = extract_agen_cfg(agenWeightNModIn);
    config->agenWeightNModOut = extract_agen_cfg(agenWeightNModOut);

    // Store configurations for shadow detection functions
    shadowBGDetConfig->agenMeanIn       = extract_agen_cfg(agenMeanIn);
    shadowBGDetConfig->agenVarIn        = extract_agen_cfg(agenVarIn);
    shadowBGDetConfig->agenWeightNModIn = extract_agen_cfg(agenWeightNModIn);
    shadowBGDetConfig->agenPix          = extract_agen_cfg(agenPix);
    shadowBGDetConfig->agenMaskIn       = extract_agen_cfg(agenMask);
    shadowBGDetConfig->agenMaskOut      = extract_agen_cfg(agenMask);
    shadowBGDetConfig->agenBGImgOut     = extract_agen_cfg(agenBGOut);
}

/*
 * Load GMM parameters (mean, variance, weight) from memory using vectorized operations
 */
inline void load_gmm_params(gmm_vec_t &gmmMeanV, gmm_vec_t &gmmVarV, gmm_vec_t &gmmWeightV, agen &agenMean,
                            agen &agenVar, agen &agenWeight)
{
#ifdef FP16
    // Load using 16-bit floating point format for memory efficiency
    gmmMeanV   = vhfloat_load_transp(agenMean);
    gmmVarV    = vhfloat_load_transp(agenVar);
    gmmWeightV = vhfloat_load_transp(agenWeight);
#else
    // Load using 32-bit floating point format for higher precision
    gmmMeanV   = vfloat_load_transp(agenMean);
    gmmVarV    = vfloat_load_transp(agenVar);
    gmmWeightV = vfloat_load_transp(agenWeight);
#endif
}

/*
 * Find distance between pixel and GMM model and determine matches
 * Computes Mahalanobis distance and checks thresholds for foreground/background detection
 */
inline void findDistAndMatch(gmm_vec_t &pixV, gmm_vec_t &gmmMeanV, gmm_vec_t gmmVarV, gmm_int_vec_t &numModV,
                             gmm_int_vec_t &matchFoundV, gmm_int_vec_t &matchFoundBV, gmm_int_vec_t &matchingModelIndV,
                             gmm_int_vec_t &matchingModelIndBV, gmm_t varThreshold, int m)
{
    // Compute squared Mahalanobis distance: (pixel - mean)^2
    gmm_vec_t dist  = (pixV - gmmMeanV);
    gmm_vec_t dist2 = dist * dist;

    // Check if this model index is valid (within active model count)
    gmm_int_vec_t tmp3 = (numModV > m);

    // Foreground matching: distance < variance * VAR_GEN and no previous match
    gmm_int_vec_t tmp1 = (dist2 < (gmmVarV * VAR_GEN) && !matchFoundV);
    gmm_int_vec_t tmp2 = tmp1 && tmp3;
    matchingModelIndV  = vmux(tmp2, m, matchingModelIndV); // Set matching model index
    matchFoundV        = vmux(tmp2, 1, matchFoundV);       // Mark match found

    // Background matching: stricter threshold for background detection
    gmm_int_vec_t tmp4 = (dist2 < (gmmVarV * varThreshold) && !matchFoundBV);
    gmm_int_vec_t tmp5 = tmp4 && tmp3;
    matchingModelIndBV = vmux(tmp5, m, matchingModelIndBV);
    matchFoundBV       = vmux(tmp5, 1, matchFoundBV);
}

/*
 * Apply temporal decay to model weights to remove outdated models
 */
inline void pruneWeights(gmm_vec_t &gmmWeightV, gmm_int_vec_t &numModV, gmm_t alpha1, gmm_t prune, int m)
{
    gmm_int_vec_t tmp1 = (numModV > m); // Check if model is active
    // Apply weight decay: weight = weight * alpha1 + prune
    gmmWeightV = vmux(tmp1, gmmWeightV * alpha1 + prune, gmmWeightV);
}

/*
 * Update GMM model parameters when a match is found
 * Implements online learning: weight, mean, and variance updates
 */
inline void loop3_2(gmm_vec_t &pixV, gmm_vec_t &gmmWeightV, gmm_vec_t &gmmVarV, gmm_vec_t &gmmMeanV,
                    gmm_int_vec_t &matchFoundV, gmm_int_vec_t &matchingModelIndV, gmm_t alphaT, int m)
{
    // Check if this model is the matching one
    gmm_int_vec_t tmp1 = matchFoundV && (matchingModelIndV == m);

    // Update weight with learning rate
    gmm_vec_t weight = gmmWeightV + alphaT;

    // Compute adaptive learning coefficient k = alphaT / new_weight
    gmm_vec_t k = reciprocal(weight) * alphaT;

    // Compute pixel-mean difference and squared distance
    gmm_vec_t dist  = pixV - gmmMeanV;
    gmm_vec_t dist2 = dist * dist;

    // Update variance with bounds checking
    gmm_vec_t varnew = gmmVarV + k * (dist2 - gmmVarV);
    varnew           = max(varnew, VAR_MIN); // Prevent variance collapse
    varnew           = min(varnew, VAR_MAX); // Prevent variance explosion

    // Update mean toward current pixel
    gmm_vec_t meanNew = gmmMeanV + k * dist;

    // Apply updates only if this model matched
    gmmMeanV   = vmux(tmp1, meanNew, gmmMeanV);
    gmmVarV    = vmux(tmp1, varnew, gmmVarV);
    gmmWeightV = vmux(tmp1, weight, gmmWeightV);
}

/*
 * Compute total weight and update model counts
 * Accumulates weights and removes models with very low weights
 */
inline void computeTotalWeight(gmm_vec_t &gmmWeightV, gmm_int_vec_t &numModV, gmm_int_vec_t &newNumModV,
                               gmm_vec_t &totalWeightV, gmm_vec_t &totalWeightBV, gmm_int_vec_t &matchingModelIndBV,
                               gmm_t prune, int m)
{
    gmm_int_vec_t tmp1 = (numModV > m);         // Model is active
    gmm_int_vec_t tmp2 = (gmmWeightV < -prune); // Weight too low, remove model

    // Set weight to 0 for models that should be pruned
    gmmWeightV = vmux(tmp1 && tmp2, gmm_t(0), gmmWeightV);

    // Add to total weight only for active models
    totalWeightV = totalWeightV + vmux(!tmp1, gmm_t(0), gmmWeightV);

    // Decrement model count if model was pruned
    newNumModV = vmux(tmp1 && tmp2, newNumModV - 1, newNumModV);

    // Add to background weight if this model had a background match
    gmm_int_vec_t tmp3 = (matchingModelIndBV > m);
    totalWeightBV      = vmux(tmp3, totalWeightBV + gmmWeightV, totalWeightBV);
}

/*
 * Normalize model weights so they sum to 1
 */
inline void mulWithInvWeight(gmm_vec_t &gmmWeightV, gmm_int_vec_t &matchFoundV, gmm_int_vec_t &numModV,
                             gmm_vec_t &invWeightV)
{
    // Skip normalization if no match found and only one model exists
    gmmWeightV = vmux(!(!matchFoundV && (numModV <= 1)), gmmWeightV * invWeightV, gmmWeightV);
}

/*
 * Extract model parameters at a specific source index during sorting
 */
inline void getModelAtSrcIndex(gmm_vec_t &gmmWeightV, gmm_vec_t &gmmVarV, gmm_vec_t &gmmMeanV,
                               gmm_vec_t &gmmWeightSrcIndex, gmm_vec_t &gmmVarSrcIndex, gmm_vec_t &gmmMeanSrcIndex,
                               gmm_int_vec_t &srcIndexV, int m)
{
    // Copy parameters if this is the source index
    gmmWeightSrcIndex = vmux(srcIndexV == m, gmmWeightV, gmmWeightSrcIndex);
    gmmVarSrcIndex    = vmux(srcIndexV == m, gmmVarV, gmmVarSrcIndex);
    gmmMeanSrcIndex   = vmux(srcIndexV == m, gmmMeanV, gmmMeanSrcIndex);
}

/*
 * Find destination position for model sorting (models sorted by weight descending)
 */
inline void getDstIndex(gmm_vec_t &gmmWeightV, gmm_vec_t &gmmWeightSrcIndex, gmm_int_vec_t &destIndexV,
                        gmm_int_vec_t &srcIndexV, int m)
{
    // If source model weight > current model weight, insert here
    destIndexV = vmux((srcIndexV > m) && (gmmWeightV < gmmWeightSrcIndex), m, destIndexV);
}

/*
 * Swap model parameters during bubble sort
 */
inline void swapModelsConditionally(gmm_vec_t &gmmWeightV, gmm_vec_t &gmmMeanV, gmm_vec_t &gmmVarV,
                                    gmm_vec_t &gmmWeightVp, gmm_vec_t &gmmMeanVp, gmm_vec_t &gmmVarVp,
                                    gmm_int_vec_t &srcIndexV, gmm_int_vec_t &destIndexV, int m)
{
    // Swap if source index >= m and destination < m
    gmm_int_vec_t cond = (srcIndexV >= m) && (destIndexV < m);
    gmmWeightV         = vmux(cond, gmmWeightVp, gmmWeightV);
    gmmMeanV           = vmux(cond, gmmMeanVp, gmmMeanV);
    gmmVarV            = vmux(cond, gmmVarVp, gmmVarV);
}

/*
 * Move model to its final sorted position
 */
inline void moveNewModel(gmm_vec_t &gmmWeightV, gmm_vec_t &gmmVarV, gmm_vec_t &gmmMeanV, gmm_vec_t &gmmWeightSrcIndex,
                         gmm_vec_t &gmmVarSrcIndex, gmm_vec_t &gmmMeanSrcIndex, gmm_int_vec_t &destIndexV, int m)
{
    // Place source model at destination if this is the target position
    gmmWeightV = vmux(destIndexV == m, gmmWeightSrcIndex, gmmWeightV);
    gmmVarV    = vmux(destIndexV == m, gmmVarSrcIndex, gmmVarV);
    gmmMeanV   = vmux(destIndexV == m, gmmMeanSrcIndex, gmmMeanV);
}

/*
 * Store updated model parameters back to memory using vectorized operations
 */
inline void storeModels(gmm_vec_t &gmmMeanV, gmm_vec_t &gmmVarV, gmm_vec_t &gmmWeightV, agen &agenMean, agen &agenVar,
                        agen &agenWeight, int m)
{
    vstore_transp(gmmMeanV, agenMean);
    vstore_transp(gmmVarV, agenVar);
    vstore_transp(gmmWeightV, agenWeight);
}

/*
 * Detect shadow pixels using chromaticity test
 * Checks if pixel is darker than background but same color direction
 */
inline void detectShadow(gmm_vec_t &pixV, gmm_vec_t &gmmMeanV, gmm_vec_t &gmmVarV, gmm_vec_t &gmmWeightV,
                         gmm_int_vec_t &isShadow, gmm_int_vec_t &isShadowValid, gmm_vec_t &totalWeight)
{
    // Compute dot products for chromaticity test
    gmm_vec_t numerator   = pixV * gmmMeanV;     // pixel · mean
    gmm_vec_t denominator = gmmMeanV * gmmMeanV; // mean · mean

    // Handle division by zero case
    gmm_int_vec_t denIsZero = (denominator == gmm_t(0));
    denominator             = vmux(denIsZero, gmm_t(1), denominator);

    // Shadow condition 1: TAU < ratio < 1 (darker but same color direction)
    gmm_int_vec_t condition1 = (numerator < denominator) && (numerator > (denominator * TAU));

    // Compute brightness ratio and check distance constraint
    gmm_vec_t a      = numerator * reciprocal(denominator);
    gmm_vec_t a2     = a * a;
    gmm_vec_t dD     = a * gmmMeanV - pixV; // Scaled difference
    gmm_vec_t dist2a = dD * dD;             // Squared distance

    // Shadow condition 2: Distance constraint with scaled variance
    gmm_int_vec_t condition2 = (dist2a < (gmmVarV * VAR_GEN * a2));

    // Update cumulative background weight
    totalWeight += gmmWeightV;

    // Background condition: Enough background weight accumulated
    gmm_int_vec_t condition3 = (totalWeight > BG_RATIO);

    // Determine shadow classification: shadow if conditions 1&2, background if condition 3
    gmm_int_vec_t isShadowNew = vmux(denIsZero, 0, vmux(condition1 && condition2, 1, !condition3));

    // Update shadow result only if not already determined
    isShadow      = vmux(isShadowValid, isShadow, isShadowNew);
    isShadowValid = isShadowValid || (denIsZero || (condition1 && condition2) || condition3);
}

/*
 * Main background subtraction processing function
 * Implements complete GMM algorithm: model matching, parameter updates,
 * model sorting, and foreground/background classification
 */
void BGSub(int frameNum, int tileNum, BGSubConfig *config, gmm_t varThreshold, gmm_t learningRate)
{
    // Compute learning parameters
    gmm_t alphaT = learningRate;      // Learning rate
    gmm_t alpha1 = gmm_t(1) - alphaT; // Decay factor for weights
    gmm_t prune  = -alphaT * CT;      // Pruning threshold

    // Initialize address generators from configuration
    agen agenMeanIn        = init_agen_from_cfg(config->agenMeanIn);
    agen agenMeanOut       = init_agen_from_cfg(config->agenMeanOut);
    agen agenVarIn         = init_agen_from_cfg(config->agenVarIn);
    agen agenVarOut        = init_agen_from_cfg(config->agenVarOut);
    agen agenWeightNModIn  = init_agen_from_cfg(config->agenWeightNModIn);
    agen agenWeightNModOut = init_agen_from_cfg(config->agenWeightNModOut);
    agen agenPix           = init_agen_from_cfg(config->agenPix);
    agen agenMask          = init_agen_from_cfg(config->agenMask);

    // Main processing loop - process pixels in vectorized tiles
    for (int i = 0; i < TILE_W * TILE_H / VECW; i++) chess_loop_range(10, )
    chess_prepare_for_pipelining
    {
        // Initialize match tracking variables
        gmm_int_vec_t matchingModelIndV; // Index of matching model for foreground
        matchingModelIndV = replicate(gmm_int_t(0));
        gmm_int_vec_t matchingModelIndBV; // Index of matching model for background
        matchingModelIndBV = replicate(gmm_int_t(0));
        gmm_int_vec_t matchFoundV; // Foreground match found flag
        matchFoundV = replicate(gmm_int_t(0));
        gmm_int_vec_t matchFoundBV; // Background match found flag
        matchFoundBV = replicate(gmm_int_t(0));

        // Model parameter vectors for all 5 models
        gmm_vec_t gmmMeanV_0_, gmmMeanV_1_, gmmMeanV_2_, gmmMeanV_3_, gmmMeanV_4_;
        gmm_vec_t gmmVarV_0_, gmmVarV_1_, gmmVarV_2_, gmmVarV_3_, gmmVarV_4_;
        gmm_vec_t gmmWeightV_0_, gmmWeightV_1_, gmmWeightV_2_, gmmWeightV_3_, gmmWeightV_4_;

        // Load GMM model parameters for all 5 models
        load_gmm_params(gmmMeanV_0_, gmmVarV_0_, gmmWeightV_0_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_1_, gmmVarV_1_, gmmWeightV_1_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_2_, gmmVarV_2_, gmmWeightV_2_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_3_, gmmVarV_3_, gmmWeightV_3_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_4_, gmmVarV_4_, gmmWeightV_4_, agenMeanIn, agenVarIn, agenWeightNModIn);

        // Load current pixel values and convert to floating point
#ifdef FP16
        dvshortx pixiV        = vuchar_dvshortx_load_transp(agenPix); // Load as 16-bit integers
        gmm_vec_t pixV        = vint24_vfp16(pixiV.lo, 0);            // Convert to FP16
        gmm_int_vec_t numModV = vshort_load_transp(agenWeightNModIn); // Number of models per pixel
#else
        dvintx pixiV          = vuchar_dvintx_load_transp(agenPix); // Load as 32-bit integers
        gmm_vec_t pixV        = vint_vfp(pixiV.lo);                 // Convert to FP32
        gmm_int_vec_t numModV = vint_load_transp(agenWeightNModIn); // Number of models per pixel
#endif

        // Find distance and matches for all 5 models
        findDistAndMatch(pixV, gmmMeanV_0_, gmmVarV_0_, numModV, matchFoundV, matchFoundBV, matchingModelIndV,
                         matchingModelIndBV, varThreshold, 0);
        findDistAndMatch(pixV, gmmMeanV_1_, gmmVarV_1_, numModV, matchFoundV, matchFoundBV, matchingModelIndV,
                         matchingModelIndBV, varThreshold, 1);
        findDistAndMatch(pixV, gmmMeanV_2_, gmmVarV_2_, numModV, matchFoundV, matchFoundBV, matchingModelIndV,
                         matchingModelIndBV, varThreshold, 2);
        findDistAndMatch(pixV, gmmMeanV_3_, gmmVarV_3_, numModV, matchFoundV, matchFoundBV, matchingModelIndV,
                         matchingModelIndBV, varThreshold, 3);
        findDistAndMatch(pixV, gmmMeanV_4_, gmmVarV_4_, numModV, matchFoundV, matchFoundBV, matchingModelIndV,
                         matchingModelIndBV, varThreshold, 4);

        // Apply weight pruning (temporal decay) to all models
        pruneWeights(gmmWeightV_0_, numModV, alpha1, prune, 0);
        pruneWeights(gmmWeightV_1_, numModV, alpha1, prune, 1);
        pruneWeights(gmmWeightV_2_, numModV, alpha1, prune, 2);
        pruneWeights(gmmWeightV_3_, numModV, alpha1, prune, 3);
        pruneWeights(gmmWeightV_4_, numModV, alpha1, prune, 4);

        // Store model parameters in vector memory for efficient lookup operations
        gmm_dvec_t gmmWeightM, gmmVarM, gmmMeanM;
        *((gmm_vec_mem_t *)(&gmmMeanMem[0 * DVECW]))   = extract(gmmMeanV_0_);
        *((gmm_vec_mem_t *)(&gmmMeanMem[1 * DVECW]))   = extract(gmmMeanV_1_);
        *((gmm_vec_mem_t *)(&gmmMeanMem[2 * DVECW]))   = extract(gmmMeanV_2_);
        *((gmm_vec_mem_t *)(&gmmMeanMem[3 * DVECW]))   = extract(gmmMeanV_3_);
        *((gmm_vec_mem_t *)(&gmmMeanMem[4 * DVECW]))   = extract(gmmMeanV_4_);
        *((gmm_vec_mem_t *)(&gmmVarMem[0 * DVECW]))    = extract(gmmVarV_0_);
        *((gmm_vec_mem_t *)(&gmmVarMem[1 * DVECW]))    = extract(gmmVarV_1_);
        *((gmm_vec_mem_t *)(&gmmVarMem[2 * DVECW]))    = extract(gmmVarV_2_);
        *((gmm_vec_mem_t *)(&gmmVarMem[3 * DVECW]))    = extract(gmmVarV_3_);
        *((gmm_vec_mem_t *)(&gmmVarMem[4 * DVECW]))    = extract(gmmVarV_4_);
        *((gmm_vec_mem_t *)(&gmmWeightMem[0 * DVECW])) = extract(gmmWeightV_0_);
        *((gmm_vec_mem_t *)(&gmmWeightMem[1 * DVECW])) = extract(gmmWeightV_1_);
        *((gmm_vec_mem_t *)(&gmmWeightMem[2 * DVECW])) = extract(gmmWeightV_2_);
        *((gmm_vec_mem_t *)(&gmmWeightMem[3 * DVECW])) = extract(gmmWeightV_3_);
        *((gmm_vec_mem_t *)(&gmmWeightMem[4 * DVECW])) = extract(gmmWeightV_4_);

#ifdef __nvvpu__
        mem_fence(); // vlookup is non-coherent w.r.t. preceding stores
#endif

        // Use vector lookup to efficiently update matched models
        gmm_int_dvec_t index;
        index.lo   = matchingModelIndV; // Model indices for lookup
        index.hi   = replicate(gmm_int_t(0));
        gmmMeanM   = vlookup(gmmMeanMem, index);   // Lookup mean values
        gmmVarM    = vlookup(gmmVarMem, index);    // Lookup variance values
        gmmWeightM = vlookup(gmmWeightMem, index); // Lookup weight values

        // Update matched model parameters using efficient vectorized operations
        gmm_vec_t weightNew = gmmWeightM.lo + alphaT;                // Update weight
        gmm_vec_t k         = reciprocal(weightNew) * alphaT;        // Adaptive learning rate
        gmm_vec_t dist      = pixV - gmmMeanM.lo;                    // Pixel-mean difference
        gmm_vec_t dist2     = dist * dist;                           // Squared distance
        gmm_vec_t meanNew   = gmmMeanM.lo + dist * k;                // Update mean
        gmm_vec_t varNew    = gmmVarM.lo + k * (dist2 - gmmVarM.lo); // Update variance
        varNew              = max(varNew, VAR_MIN);                  // Clamp variance
        varNew              = min(varNew, VAR_MAX);

        // Apply updates only to matched models
        gmmMeanM.lo   = vmux(matchFoundV, meanNew, gmmMeanM.lo);
        gmmVarM.lo    = vmux(matchFoundV, varNew, gmmVarM.lo);
        gmmWeightM.lo = vmux(matchFoundV, weightNew, gmmWeightM.lo);

        // Store updated parameters back to vector memory
        vast(gmmMeanMem, index, gmmMeanM);
        vast(gmmVarMem, index, gmmVarM);
        vast(gmmWeightMem, index, gmmWeightM);

        chess_memory_fence(); // Ensure memory operations complete
#ifdef __nvvpu__
        mem_fence(); // sign_extend reads same scratch as vast() above
#endif

        // Load updated model parameters back to individual vectors
        gmmMeanV_0_   = sign_extend(*(gmm_vec_mem_t *)(&gmmMeanMem[0 * DVECW]));
        gmmMeanV_1_   = sign_extend(*(gmm_vec_mem_t *)(&gmmMeanMem[1 * DVECW]));
        gmmMeanV_2_   = sign_extend(*(gmm_vec_mem_t *)(&gmmMeanMem[2 * DVECW]));
        gmmMeanV_3_   = sign_extend(*(gmm_vec_mem_t *)(&gmmMeanMem[3 * DVECW]));
        gmmMeanV_4_   = sign_extend(*(gmm_vec_mem_t *)(&gmmMeanMem[4 * DVECW]));
        gmmVarV_0_    = sign_extend(*(gmm_vec_mem_t *)(&gmmVarMem[0 * DVECW]));
        gmmVarV_1_    = sign_extend(*(gmm_vec_mem_t *)(&gmmVarMem[1 * DVECW]));
        gmmVarV_2_    = sign_extend(*(gmm_vec_mem_t *)(&gmmVarMem[2 * DVECW]));
        gmmVarV_3_    = sign_extend(*(gmm_vec_mem_t *)(&gmmVarMem[3 * DVECW]));
        gmmVarV_4_    = sign_extend(*(gmm_vec_mem_t *)(&gmmVarMem[4 * DVECW]));
        gmmWeightV_0_ = sign_extend(*(gmm_vec_mem_t *)(&gmmWeightMem[0 * DVECW]));
        gmmWeightV_1_ = sign_extend(*(gmm_vec_mem_t *)(&gmmWeightMem[1 * DVECW]));
        gmmWeightV_2_ = sign_extend(*(gmm_vec_mem_t *)(&gmmWeightMem[2 * DVECW]));
        gmmWeightV_3_ = sign_extend(*(gmm_vec_mem_t *)(&gmmWeightMem[3 * DVECW]));
        gmmWeightV_4_ = sign_extend(*(gmm_vec_mem_t *)(&gmmWeightMem[4 * DVECW]));

        // Compute total weights and handle model pruning
        gmm_vec_t totalWeightV;
        totalWeightV = replicate(gmm_t(0));
        gmm_vec_t totalWeightBV; // Background weight sum
        totalWeightBV            = replicate(gmm_t(0));
        gmm_int_vec_t newNumModV = numModV; // Updated model count

        // Accumulate weights and prune low-weight models
        computeTotalWeight(gmmWeightV_0_, numModV, newNumModV, totalWeightV, totalWeightBV, matchingModelIndBV, prune,
                           0);
        computeTotalWeight(gmmWeightV_1_, numModV, newNumModV, totalWeightV, totalWeightBV, matchingModelIndBV, prune,
                           1);
        computeTotalWeight(gmmWeightV_2_, numModV, newNumModV, totalWeightV, totalWeightBV, matchingModelIndBV, prune,
                           2);
        computeTotalWeight(gmmWeightV_3_, numModV, newNumModV, totalWeightV, totalWeightBV, matchingModelIndBV, prune,
                           3);
        computeTotalWeight(gmmWeightV_4_, numModV, newNumModV, totalWeightV, totalWeightBV, matchingModelIndBV, prune,
                           4);

        numModV                  = newNumModV;
        gmm_vec_t totalWeightM1V = totalWeightV - gmmWeightV_4_; // Total weight excluding last model

        // Generate foreground/background mask
        gmm_int_vec_t maskV;
        maskV = replicate(gmm_int_t(255)); // Default to foreground (white)

        // Classify as background if background match found with sufficient weight
        gmm_int_vec_t notBGV =
            matchFoundBV && (totalWeightBV < BG_RATIO) && ((matchingModelIndBV <= matchingModelIndV) || (!matchFoundV));

        maskV = vmux(notBGV, 0, maskV); // Set background pixels to black (0)

        // Handle case where no match was found - add new model
        gmm_vec_t alphaTV;
        alphaTV = replicate(alphaT);
        gmm_vec_t normalizationFactorV;
        gmm_int_vec_t matchNotFoundV = !matchFoundV && (alphaT > gmm_t(0));
        gmm_int_vec_t maxModelsUsedV = (numModV == NUM_MODELS);
        gmm_int_vec_t numModTmpV     = vmux(maxModelsUsedV, NUM_MODELS, numModV + 1);

        // Initialize new model parameters for unmatched pixels
        gmmMeanV_4_   = vmux(matchNotFoundV, pixV, gmmMeanV_4_);    // Set mean to current pixel
        gmmVarV_4_    = vmux(matchNotFoundV, VAR_INIT, gmmVarV_4_); // Initialize variance
        gmmWeightV_4_ = vmux(matchNotFoundV, vmux(numModTmpV == 1, gmm_t(1), alphaTV), gmmWeightV_4_);

        // Compute normalization factor for weight update
        normalizationFactorV =
            vmux(matchNotFoundV, vmux(numModTmpV == 1, gmm_t(1), totalWeightM1V * reciprocal(-alphaTV + gmm_t(1))),
                 totalWeightV);
        numModV = vmux(matchNotFoundV, numModTmpV, numModV);

        // Normalize weights to sum to 1
        gmm_vec_t invWeightV;
        invWeightV = replicate(gmm_t(0));
        invWeightV = vmux(abs1(normalizationFactorV) > EPSILON, reciprocal(normalizationFactorV), invWeightV);

        // Apply weight normalization to all models
        mulWithInvWeight(gmmWeightV_0_, matchFoundV, numModV, invWeightV);
        mulWithInvWeight(gmmWeightV_1_, matchFoundV, numModV, invWeightV);
        mulWithInvWeight(gmmWeightV_2_, matchFoundV, numModV, invWeightV);
        mulWithInvWeight(gmmWeightV_3_, matchFoundV, numModV, invWeightV);

        gmmWeightV_4_ = vmux(matchFoundV, gmmWeightV_4_ * invWeightV, gmmWeightV_4_);

        // Sort models by weight for better performance (highest weight first)
        gmm_int_vec_t srcIndexV  = vmux(!matchFoundV, NUM_MODELS - 1, matchingModelIndV);
        gmm_int_vec_t destIndexV = srcIndexV;
        gmm_vec_t gmmWeightSrcIndex;
        gmm_vec_t gmmVarSrcIndex;
        gmm_vec_t gmmMeanSrcIndex;

        // Extract source model parameters
        getModelAtSrcIndex(gmmWeightV_0_, gmmVarV_0_, gmmMeanV_0_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                           srcIndexV, 0);
        getModelAtSrcIndex(gmmWeightV_1_, gmmVarV_1_, gmmMeanV_1_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                           srcIndexV, 1);
        getModelAtSrcIndex(gmmWeightV_2_, gmmVarV_2_, gmmMeanV_2_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                           srcIndexV, 2);
        getModelAtSrcIndex(gmmWeightV_3_, gmmVarV_3_, gmmMeanV_3_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                           srcIndexV, 3);
        getModelAtSrcIndex(gmmWeightV_4_, gmmVarV_4_, gmmMeanV_4_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                           srcIndexV, 4);

        // Find destination position for insertion sort
        getDstIndex(gmmWeightV_4_, gmmWeightSrcIndex, destIndexV, srcIndexV, 4);
        getDstIndex(gmmWeightV_3_, gmmWeightSrcIndex, destIndexV, srcIndexV, 3);
        getDstIndex(gmmWeightV_2_, gmmWeightSrcIndex, destIndexV, srcIndexV, 2);
        getDstIndex(gmmWeightV_1_, gmmWeightSrcIndex, destIndexV, srcIndexV, 1);
        getDstIndex(gmmWeightV_0_, gmmWeightSrcIndex, destIndexV, srcIndexV, 0);

        // Perform bubble sort swaps to maintain weight order
        swapModelsConditionally(gmmWeightV_4_, gmmMeanV_4_, gmmVarV_4_, gmmWeightV_3_, gmmMeanV_3_, gmmVarV_3_,
                                srcIndexV, destIndexV, 4);
        swapModelsConditionally(gmmWeightV_3_, gmmMeanV_3_, gmmVarV_3_, gmmWeightV_2_, gmmMeanV_2_, gmmVarV_2_,
                                srcIndexV, destIndexV, 3);
        swapModelsConditionally(gmmWeightV_2_, gmmMeanV_2_, gmmVarV_2_, gmmWeightV_1_, gmmMeanV_1_, gmmVarV_1_,
                                srcIndexV, destIndexV, 2);
        swapModelsConditionally(gmmWeightV_1_, gmmMeanV_1_, gmmVarV_1_, gmmWeightV_0_, gmmMeanV_0_, gmmVarV_0_,
                                srcIndexV, destIndexV, 1);

        // Place source model at final destination
        moveNewModel(gmmWeightV_0_, gmmVarV_0_, gmmMeanV_0_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                     destIndexV, 0);
        moveNewModel(gmmWeightV_1_, gmmVarV_1_, gmmMeanV_1_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                     destIndexV, 1);
        moveNewModel(gmmWeightV_2_, gmmVarV_2_, gmmMeanV_2_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                     destIndexV, 2);
        moveNewModel(gmmWeightV_3_, gmmVarV_3_, gmmMeanV_3_, gmmWeightSrcIndex, gmmVarSrcIndex, gmmMeanSrcIndex,
                     destIndexV, 3);

        // Store all updated model parameters to output memory
        storeModels(gmmMeanV_0_, gmmVarV_0_, gmmWeightV_0_, agenMeanOut, agenVarOut, agenWeightNModOut, 0);
        storeModels(gmmMeanV_1_, gmmVarV_1_, gmmWeightV_1_, agenMeanOut, agenVarOut, agenWeightNModOut, 1);
        storeModels(gmmMeanV_2_, gmmVarV_2_, gmmWeightV_2_, agenMeanOut, agenVarOut, agenWeightNModOut, 2);
        storeModels(gmmMeanV_3_, gmmVarV_3_, gmmWeightV_3_, agenMeanOut, agenVarOut, agenWeightNModOut, 3);
        storeModels(gmmMeanV_4_, gmmVarV_4_, gmmWeightV_4_, agenMeanOut, agenVarOut, agenWeightNModOut, 4);
        vstore_transp(numModV, agenWeightNModOut); // Store updated model count

        // Convert and store the final mask
#ifdef FP16
        dvshortx maskShortV;
        maskShortV.lo    = maskV;
        maskShortV.hi    = replicate(gmm_int_t(0));
        vcharx maskCharV = vdemote(maskShortV);     // Convert to 8-bit
        vstore_transp(maskCharV, agenMask, 0xFFFF); // Store mask
#else
        dvshortx maskShortV;
        maskShortV.lo = replicateh(0);
        maskShortV.hi = replicateh(0);
        dvintx maskDV;
        maskDV.lo = maskV;
        maskDV.hi = replicatew(0);
        vdemote(maskDV.lo, maskDV.hi, maskShortV.lo);
        vcharx maskCharV = vdemote(maskShortV);
        vstore_transp(maskCharV, agenMask, 0xFF);
#endif
        chess_memory_fence(); // Ensure all stores complete
#ifdef __nvvpu__
        mem_fence(); // tile stores must complete before next iteration acquires
#endif
    }
}

/*
 * Perform shadow detection on foreground pixels
 * Uses chromaticity test to distinguish shadows from true foreground objects
 */
void doShadowDetection(ShadowAndBGDetectConfig *config, uint8_t shadowPixelValue)
{
    // Initialize address generators for shadow detection
    agen agenMeanIn       = init_agen_from_cfg(config->agenMeanIn);
    agen agenVarIn        = init_agen_from_cfg(config->agenVarIn);
    agen agenWeightNModIn = init_agen_from_cfg(config->agenWeightNModIn);
    agen agenPix          = init_agen_from_cfg(config->agenPix);
    agen agenMaskIn       = init_agen_from_cfg(config->agenMaskIn);
    agen agenMaskOut      = init_agen_from_cfg(config->agenMaskOut);

    // Process pixels in vectorized tiles
    for (int i = 0; i < TILE_W * TILE_H / VECW; i++) chess_loop_range(10, )
    chess_prepare_for_pipelining
    {
        // Load GMM model parameters for all 5 models
        gmm_vec_t gmmMeanV_0_, gmmMeanV_1_, gmmMeanV_2_, gmmMeanV_3_, gmmMeanV_4_;
        gmm_vec_t gmmVarV_0_, gmmVarV_1_, gmmVarV_2_, gmmVarV_3_, gmmVarV_4_;
        gmm_vec_t gmmWeightV_0_, gmmWeightV_1_, gmmWeightV_2_, gmmWeightV_3_, gmmWeightV_4_;

        load_gmm_params(gmmMeanV_0_, gmmVarV_0_, gmmWeightV_0_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_1_, gmmVarV_1_, gmmWeightV_1_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_2_, gmmVarV_2_, gmmWeightV_2_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_3_, gmmVarV_3_, gmmWeightV_3_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_4_, gmmVarV_4_, gmmWeightV_4_, agenMeanIn, agenVarIn, agenWeightNModIn);

        // Load current pixels and input mask
#ifdef FP16
        dvshortx pixiV        = vuchar_dvshortx_load_transp(agenPix);
        gmm_vec_t pixV        = vint24_vfp16(pixiV.lo, 0); // Convert pixels to FP16
        gmm_int_vec_t numModV = vshort_load_transp(agenWeightNModIn);

        dvshortx maskInTmp = vuchar_dvshortx_load_transp(agenMaskIn);
        vshortx maskV      = maskInTmp.lo; // Load input mask
#else
        dvintx pixiV          = vuchar_dvintx_load_transp(agenPix);
        gmm_vec_t pixV        = vint_vfp(pixiV.lo); // Convert pixels to FP32
        gmm_int_vec_t numModV = vint_load_transp(agenWeightNModIn);

        dvintx maskInTmp = vuchar_dvintx_load_transp(agenMaskIn);
        vintx maskV      = maskInTmp.lo; // Load input mask
#endif

        // Initialize shadow detection variables
        gmm_int_vec_t isShadow      = replicate(gmm_int_t(0)); // Shadow detection result
        gmm_int_vec_t isShadowValid = replicate(gmm_int_t(0)); // Whether result is valid
        gmm_vec_t totalWeightS      = replicate(gmm_t(0));     // Background weight accumulator

        // Test each model for shadow detection
        detectShadow(pixV, gmmMeanV_0_, gmmVarV_0_, gmmWeightV_0_, isShadow, isShadowValid, totalWeightS);
        detectShadow(pixV, gmmMeanV_1_, gmmVarV_1_, gmmWeightV_1_, isShadow, isShadowValid, totalWeightS);
        detectShadow(pixV, gmmMeanV_2_, gmmVarV_2_, gmmWeightV_2_, isShadow, isShadowValid, totalWeightS);
        detectShadow(pixV, gmmMeanV_3_, gmmVarV_3_, gmmWeightV_3_, isShadow, isShadowValid, totalWeightS);
        detectShadow(pixV, gmmMeanV_4_, gmmVarV_4_, gmmWeightV_4_, isShadow, isShadowValid, totalWeightS);

        // Update mask: mark foreground pixels as shadows if detected
        maskV = vmux(isShadow && isShadowValid && (maskV == 255), shadowPixelValue, maskV);

        // Convert and store the updated mask
#ifdef FP16
        dvshortx maskShortV;
        maskShortV.lo    = maskV;
        maskShortV.hi    = replicate(gmm_int_t(0));
        vcharx maskCharV = vdemote(maskShortV);
        vstore_transp(maskCharV, agenMaskOut, 0xFFFF);
#else
        dvshortx maskShortV;
        maskShortV.lo = replicateh(0);
        maskShortV.hi = replicateh(0);
        dvintx maskDV;
        maskDV.lo = maskV;
        maskDV.hi = replicatew(0);
        vdemote(maskDV.lo, maskDV.hi, maskShortV.lo);
        vcharx maskCharV = vdemote(maskShortV);
        vstore_transp(maskCharV, agenMaskOut, 0xFF);
#endif
    }
}

/*
 * Helper function for background image computation
 * Accumulates weighted background models up to BG_RATIO threshold
 */
inline void getBgLoop(gmm_vec_t &meanValV, gmm_vec_t &totalWeightV, gmm_vec_t &gmmMeanV, gmm_vec_t &gmmWeightV,
                      gmm_int_vec_t &numModV, int m)
{
    // Include model in background if: total weight <= threshold AND model is active
    gmm_int_vec_t cond = (totalWeightV <= BG_RATIO) && (numModV > m);
    meanValV           = vmux(cond, meanValV + gmmMeanV * gmmWeightV, meanValV); // Weight * mean
    totalWeightV       = vmux(cond, totalWeightV + gmmWeightV, totalWeightV);    // Accumulate weight
}

/*
 * Generate clean background image from GMM models
 * Computes weighted average of background models for each pixel
 */
void computeBgImg(ShadowAndBGDetectConfig *config)
{
    // Initialize address generators
    agen agenMeanIn       = init_agen_from_cfg(config->agenMeanIn);
    agen agenVarIn        = init_agen_from_cfg(config->agenVarIn);
    agen agenWeightNModIn = init_agen_from_cfg(config->agenWeightNModIn);
    agen agenBGImgOut     = init_agen_from_cfg(config->agenBGImgOut);

    // Process pixels in vectorized tiles
    for (int i = 0; i < TILE_W * TILE_H / VECW; i++) chess_loop_range(10, )
    chess_prepare_for_pipelining
    {
        // Load GMM model parameters
        gmm_vec_t gmmMeanV_0_, gmmMeanV_1_, gmmMeanV_2_, gmmMeanV_3_, gmmMeanV_4_;
        gmm_vec_t gmmVarV_0_, gmmVarV_1_, gmmVarV_2_, gmmVarV_3_, gmmVarV_4_;
        gmm_vec_t gmmWeightV_0_, gmmWeightV_1_, gmmWeightV_2_, gmmWeightV_3_, gmmWeightV_4_;

        load_gmm_params(gmmMeanV_0_, gmmVarV_0_, gmmWeightV_0_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_1_, gmmVarV_1_, gmmWeightV_1_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_2_, gmmVarV_2_, gmmWeightV_2_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_3_, gmmVarV_3_, gmmWeightV_3_, agenMeanIn, agenVarIn, agenWeightNModIn);
        load_gmm_params(gmmMeanV_4_, gmmVarV_4_, gmmWeightV_4_, agenMeanIn, agenVarIn, agenWeightNModIn);

        // Load number of active models per pixel
#ifdef FP16
        gmm_int_vec_t numModV = vshort_load_transp(agenWeightNModIn);
#else
        gmm_int_vec_t numModV = vint_load_transp(agenWeightNModIn);
#endif

        // Initialize accumulators for background computation
        gmm_vec_t totalWeightV = replicate(gmm_t(0)); // Total weight of background models
        gmm_vec_t meanValV     = replicate(gmm_t(0)); // Weighted sum of background means

        // Accumulate background models (models are sorted by weight, so first models are most likely background)
        getBgLoop(meanValV, totalWeightV, gmmMeanV_0_, gmmWeightV_0_, numModV, 0);
        getBgLoop(meanValV, totalWeightV, gmmMeanV_1_, gmmWeightV_1_, numModV, 1);
        getBgLoop(meanValV, totalWeightV, gmmMeanV_2_, gmmWeightV_2_, numModV, 2);
        getBgLoop(meanValV, totalWeightV, gmmMeanV_3_, gmmWeightV_3_, numModV, 3);
        getBgLoop(meanValV, totalWeightV, gmmMeanV_4_, gmmWeightV_4_, numModV, 4);

        // Normalize by total weight to get final background pixel value
        gmm_vec_t invWeightV = replicate(gmm_t(0));
        gmm_int_vec_t cond1  = (totalWeightV > EPSILON); // Avoid division by zero
        invWeightV           = vmux(cond1, reciprocal(totalWeightV), invWeightV);
        meanValV             = meanValV * invWeightV; // Final background value

        // Convert floating point background to integer and store
#ifdef FP16
        dvshortx bgPixShortV;
        bgPixShortV.lo    = vfp16_vint24_round(meanValV, 0); // Round to nearest integer
        bgPixShortV.hi    = replicateh(0);
        vcharx bgPixCharV = vdemote(bgPixShortV);        // Convert to 8-bit
        vstore_transp(bgPixCharV, agenBGImgOut, 0xFFFF); // Store background image
#else
        dvintx bgPixIntDV;
        dvshortx bgPixShortDV;
        bgPixIntDV.lo = vfp_vintx_trunc(meanValV); // Truncate to integer
        bgPixIntDV.hi = replicatew(0);
        vdemote(bgPixIntDV.lo, bgPixIntDV.hi, bgPixShortDV.lo);
        bgPixShortDV.hi   = replicateh(0);
        vcharx bgPixCharV = vdemote(bgPixShortDV);
        vstore_transp(bgPixCharV, agenBGImgOut, 0xFF);
#endif
    }
}