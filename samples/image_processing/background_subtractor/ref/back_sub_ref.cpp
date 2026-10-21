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

#include <back_sub_ref.hpp>

#include <cmath>
#include <iostream>

using std::cout;

template<class T, class U>
void BGSubRef<T, U>::printWeights(int r, int c)
{
    int ind = r * w + c;
    for (int i = 0; i < m_gmms[ind].MAX_NUM_MODELS; i++)
    {
        cout << m_gmms[ind].gmm[i].weight << " ";
    }
    cout << "\n";
}

template<class T, class U>
void BGSubRef<T, U>::getBackgroundImage(T *BGImg)
{
    for (int row = 0; row < h; row++)
    {
        for (int col = 0; col < w; col++)
        {

            int index        = row * w + col;
            GMMS<T, U> &gmms = m_gmms[index];

            U totalWeight(0);
            U meanVal(0);

            // loop over of the models
            for (int model = 0; model < gmms.numModelsInUse; model++)
            {
                typename GMMS<T, U>::GMM &gmm = gmms.gmm[model];

                meanVal += gmm.weight * gmm.mean;
                totalWeight += gmm.weight;

                // Enough background model
                if (totalWeight > BG_RATIO)
                {
                    break;
                }
            }

            U invWeight(0);
            if (std::fabs(totalWeight) > std::numeric_limits<float>::epsilon())
            {
                invWeight = 1.f / totalWeight;
            }

            meanVal *= invWeight;

            BGImg[index] = meanVal;
        }
    }
}

template<class T, class U>
BGSubRef<T, U>::BGSubRef(int w, int h)
    : w{w}
    , h{h}
{
    m_gmms = new GMMS<T, U>[w * h];
    reinitialize();
}

template<class T, class U>
BGSubRef<T, U>::~BGSubRef()
{
    delete[] m_gmms;
}

template<class T, class U>
void BGSubRef<T, U>::reinitialize()
{
    for (int i = 0; i < w * h; i++)
    {
        m_gmms[i].numModelsInUse = 0;
        for (int j = 0; j < GMMS<T, U>::MAX_NUM_MODELS; j++)
        {
            m_gmms[i].gmm[j].weight   = 0;
            m_gmms[i].gmm[j].mean     = 0;
            m_gmms[i].gmm[j].variance = 0;
        }
    }
}

template<class T, class U>
U to_U(T in)
{
    return static_cast<U>(in);
}

template<class U>
U dot(U in1, U in2)
{
    return in1 * in2;
}

template<class T, class U>
bool BGSubRef<T, U>::detectShadow(GMMS<T, U> &gmms, T data)
{
    U totalWeight;
    totalWeight = 0;
    for (int model = 0; model < gmms.numModelsInUse; model++)
    {
        U numerator(0.0);
        U denominator(0.0);

        numerator   = dot<U>(static_cast<U>(data), static_cast<U>(gmms.gmm[model].mean));
        denominator = dot<U>(static_cast<U>(gmms.gmm[model].mean), static_cast<U>(gmms.gmm[model].mean));

        // no division by zero allowed
        if (denominator == 0)
        {
            return false;
        }

        // if tau < a < 1 then also check the color distortion
        if (numerator < denominator && numerator > TAU * denominator)
        {
            U a  = numerator / denominator;
            U a2 = a * a;
            U dist2a(0.0);

            T dD   = a * gmms.gmm[model].mean - data;
            dist2a = dot<U>(static_cast<U>(dD), static_cast<U>(dD));

            if (dist2a < VAR_GEN * gmms.gmm[model].variance * a2)
            {
                return true;
            }
        }

        totalWeight += gmms.gmm[model].weight;

        if (totalWeight > BG_RATIO)
        {
            return false;
        }
    }

    return false;
}

template<class T, class U>
uint8_t BGSubRef<T, U>::proces_pixel(int index, U alpha1, U alphaT, U prune, T data, U varThreshold, bool bDetectShadow,
                                     uint8_t shadowPixelValue)
{
    GMMS<T, U> &gmms = m_gmms[index];
    bool background  = false;
    U totalWeight(0.0);
    uint8_t numModelsInUse = gmms.numModelsInUse;
    bool fitsPDF           = false;

    // loop over the models
    for (int model = 0; model < gmms.numModelsInUse; model++)
    {
        int newModelIndex = model;

        typename GMMS<T, U>::GMM &gmm = gmms.gmm[model];

        U weight = alpha1 * static_cast<U>(gmm.weight) + prune;

        // fit not found yet
        if (!fitsPDF)
        {
            // check if it belongs to some of the remaining modes
            U var = static_cast<U>(gmm.variance);

            // calculate difference and distance
            U dist2(0.0);
            U dData;
            dData = to_U<T, U>(data) - (gmm.mean);
            dist2 = dot<U>(dData, dData);

            // background?
            if (totalWeight < BG_RATIO && dist2 < varThreshold * var)
            {
                background = true;
            }

            // Check whether belong to model or not
            if (dist2 < VAR_GEN * var)
            {
                // belongs to the model
                fitsPDF = true;

                // update weight
                weight += alphaT;
                U k = alphaT / weight;

                // update mean
                gmm.mean += k * dData;

                // update variance
                U varnew = var + k * (dist2 - var);

                // limit the variance
                varnew       = std::max(varnew, static_cast<U>(VAR_MIN));
                varnew       = std::min(varnew, static_cast<U>(VAR_MAX));
                gmm.variance = varnew;

                // sort, bubble up
                // all other weights are at the same place and only the matched (iModes) is higher -> just find the new place for it
                for (int i = model; i > 0; i--)
                {
                    // check one up
                    if (weight < gmms.gmm[i - 1].weight)
                    {
                        break;
                    }

                    // swap one up
                    std::swap(gmms.gmm[i], gmms.gmm[i - 1]);
                    newModelIndex = i - 1;
                }
            }
        }

        // check prune
        if (weight < -prune)
        {
            weight = 0.0;
            numModelsInUse--;
        }

        gmms.gmm[newModelIndex].weight = weight; // update weight by the calculated value
        totalWeight += weight;
    }

    // Renormalize weights. In the special case that the pixel does not agree with any models, set weights to zero (a new model will be added below).
    float invWeight = 0.f;
    if (std::fabs(totalWeight) > std::numeric_limits<float>::epsilon())
    {
        invWeight = 1.f / totalWeight;
    }

    for (int model = 0; model < numModelsInUse; model++)
    {
        gmms.gmm[model].weight *= invWeight;
    }

    // make a new model if needed and exit
    // alphaT > 0 -> learning rate is not zero, otherwise it will not update
    if (!fitsPDF && alphaT > 0.f)
    {
        // replace the weakest or add a new one
        int model = numModelsInUse == GMMS<T, U>::MAX_NUM_MODELS ? GMMS<T, U>::MAX_NUM_MODELS - 1 : numModelsInUse++;

        if (numModelsInUse == 1)
        {
            // previous no model exists
            gmms.gmm[model].weight = 1.f;
        }
        else
        {
            gmms.gmm[model].weight = alphaT;

            // renormalize all other weights
            for (int i = 0; i < numModelsInUse - 1; i++)
            {
                gmms.gmm[i].weight *= alpha1;
            }
        }

        // init
        gmms.gmm[model].mean     = data;
        gmms.gmm[model].variance = VAR_INIT;

        // sort
        // find the new place for it
        for (int i = model - 1; i > 0; i--)
        {
            // check one up
            if (alphaT < gmms.gmm[i].weight)
            {
                break;
            }

            // swap one up
            std::swap(gmms.gmm[i], gmms.gmm[i + 1]);
        }
    }

    // set the number of models
    gmms.numModelsInUse = numModelsInUse;

    uint8_t fgMask;
    if (background)
    {
        fgMask = 0;
    }
    else
    {
        if (bDetectShadow && detectShadow(gmms, data))
        {
            fgMask = shadowPixelValue;
        }
        else
        {
            fgMask = 255;
        }
    }

    return fgMask;
}

template<class T, class U>
void BGSubRef<T, U>::runGMMS(BackgroundSubtractorRefParams &params, T *img, T *fgMask)
{
    U varThreshold;
    varThreshold = params.varThreshold;
    U alphaT;
    alphaT = params.learningRate;
    U alpha1;
    alpha1 = 1 - alphaT;
    U prune;
    prune = -alphaT * CT;

    if (alphaT == 1.0)
    {
        reinitialize();
        return;
    }

    bool bDetectShadow       = params.detectShadow;
    uint8_t shadowPixelValue = params.shadowPixelValue;

    if (bDetectShadow == true && shadowPixelValue == 0)
    {
        // setting shadow pixel to zero is equivalent to turn of shadow detection
        bDetectShadow = false;
    }

    for (int row = 0; row < h; row++)
    {
        for (int col = 0; col < w; col++)
        {
            T data    = img[row * w + col];
            int index = row * w + col;
            fgMask[row * w + col] =
                proces_pixel(index, alpha1, alphaT, prune, data, varThreshold, bDetectShadow, shadowPixelValue);
        }
    }
}

template class BGSubRef<uint8_t, float>;