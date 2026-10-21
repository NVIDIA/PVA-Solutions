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

#ifndef BARTLETT_COMMON_H
#define BARTLETT_COMMON_H

#include "bartlett_knl.h"

#include <cupva_device.h>

#define COMPLEX_PER_VEC_LOAD 8

#define DB_SCALE_LOG2_TO_DB10 6.0206f
#define EPSILON_AVOID_LOG_ZERO 1e-10f

// Reduce complex accumulator to power (real^2 + imag^2) as scalar float
static inline float reducePower(vintx realAcc, vintx imagAcc)
{
    vintx rSum = vsumr(realAcc);
    vintx iSum = vsumr(imagAcc);
    vfloatx rF = vintx_vfp(rSum);
    vfloatx iF = vintx_vfp(iSum);
    return smovf(vmaddf(rF, rF, vmulf(iF, iF)));
}

// Quadratic peak interpolation: delta = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
// Processes Az (lo) and El (hi) simultaneously using double-vectors
static inline dvfloatx dvQuadraticPeakOffset(dvfloatx y0, dvfloatx y1, dvfloatx y2)
{
    dvfloatx num = y0 - y2;

    dvfloatx den = (y0 + y2) - (y1 + y1);
    den          = den + den;

    vfloatx zero = replicatef(0.0f);
    vfloatx one  = replicatef(1.0f);

    vintx denZeroMaskLo = (den.lo == zero);
    vintx denZeroMaskHi = (den.hi == zero);

    // Replace zero denominators with 1.0 BEFORE computing reciprocal to prevent
    // vfrcp(0)=Inf and subsequent 0*Inf=NaN which triggers the VPU NaN trap.
    vfloatx safeDenLo = vmux(denZeroMaskLo, one, den.lo);
    vfloatx safeDenHi = vmux(denZeroMaskHi, one, den.hi);

    dvfloatx delta;
    delta.lo = vmux(denZeroMaskLo, zero, num.lo * vfrcp(safeDenLo));
    delta.hi = vmux(denZeroMaskHi, zero, num.hi * vfrcp(safeDenHi));

    return delta;
}

// Vectorized peak search using vmaxrid_s with scalar remainder
static inline int32_t peakSearchVec(AgenCFG *agenCfg, float *buf, int32_t count, float *outMaxPower)
{
    const int32_t vecw    = 8;
    const int32_t numVecs = count / vecw;
    float maxPower        = 0.0f;
    int32_t peakIdx       = 0;

    agen a = init_agen_from_cfg(*agenCfg);

    for (int32_t i = 0; i < numVecs; i++) chess_prepare_for_pipelining
    {
        vfloatx vPow = vfloat_load(a);
        int32_t maxInt, maxI;
        vmaxrid_s(as_vintx(vPow), maxInt, maxI);
        float currMax = as_float(maxInt);
        maxI += i * vecw;
        int32_t cond = (currMax > maxPower);
        maxPower     = mux(cond, currMax, maxPower);
        peakIdx      = mux(cond, maxI, peakIdx);
    }

    for (int32_t i = numVecs * vecw; i < count; i++)
    {
        int32_t cond = (buf[i] > maxPower);
        maxPower     = mux(cond, buf[i], maxPower);
        peakIdx      = mux(cond, i, peakIdx);
    }

    *outMaxPower = maxPower;
    return peakIdx;
}

#endif // BARTLETT_COMMON_H
