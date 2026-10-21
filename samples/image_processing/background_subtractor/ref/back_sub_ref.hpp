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

#include <algorithm>
#include <cstdint>
#include <limits>

// shadow threshold. The shadow is detected if the pixel is darker
// version of the background. Tau is a threshold on how much darker the shadow can be.
// Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
constexpr float TAU = 0.5;

// Threshold for matching model
constexpr float VAR_GEN = 3.0f * 3.0f;

// Ratio to classify as background
constexpr float BG_RATIO = 0.9f;

// Init variance for new model
constexpr float VAR_INIT = 15.0f;

// Max allowed variance for model
constexpr float VAR_MAX = 5 * VAR_INIT;

// Min allowed variance for model
constexpr float VAR_MIN = 4.0f;

// complex reduction threshold, used to calculate prune
constexpr float CT = 0.05;

/**
 * Structure that defines the parameters for \ref vpiCreateBackgroundSubtractor
 */
typedef struct
{
    /** Threshold on the squared Mahalanobis distance between the pixel and the model
      * to decide whether a pixel is well described by the background model.
      * This parameter does not affect the background update. */
    float varThreshold = 100;

    /** The algorithm will detect shadows and mark them if this is set to non-zero value */
    uint8_t detectShadow = 0;

    /** Pixel value representing shadow. */
    uint8_t shadowPixelValue = 0;

    /** Learning rate that indicates how fast the background model is learnt.
     *  + This value must be between 0 and 1.
     *    - Minimum of 0 means that the background model is not updated at all
     *    - Maximum of 1 means that the background model is completely reinitialized from the last frame.
     */

    float learningRate = 0.5;
} BackgroundSubtractorRefParams;

template<class T, class U>
struct GMMS
{
    static constexpr int MAX_NUM_CHANNELS = 4;
    static constexpr int MAX_NUM_MODELS   = 5;

    struct GMM
    {
        U weight;
        U mean;
        U variance;
    };

    GMM gmm[MAX_NUM_MODELS];
    uint8_t numModelsInUse;
};

template<class T, class U>
class BGSubRef
{
public:
    BGSubRef(int w, int h);
    ~BGSubRef();
    int w, h;
    void reinitialize();
    void runGMMS(BackgroundSubtractorRefParams &params, T *img, T *fgMask);
    void getBackgroundImage(T *img);
    bool detectShadow(GMMS<T, U> &gmms, T data);
    uint8_t proces_pixel(int index, U alpha1, U alphaT, U prune, T data, U varThreshold, bool bDetectShadow,
                         uint8_t shadowPixelValue);
    void printWeights(int r, int c);

    GMMS<T, U> *m_gmms;
};