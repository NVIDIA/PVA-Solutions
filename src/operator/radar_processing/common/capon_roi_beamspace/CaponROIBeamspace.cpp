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

#include "CaponROIBeamspace.hpp"

#include <cmath>

namespace pva_solutions::capon_roi {

/// asin(v) in degrees, with v clamped to [-1, 1].
double asind(double v)
{
    if (v > 1.0)
        v = 1.0;
    if (v < -1.0)
        v = -1.0;
    return std::asin(v) * 180.0 / kPi;
}

/// Element-space steering vector (indexed by vc0 = rx*numTx + tx):
/// phase = x*cos(el)*sin(az) + y*sin(el); a = exp(-j 2pi phase).
void steeringVector(const std::vector<double> &vcPosX, const std::vector<double> &vcPosY, double azDeg, double elDeg,
                    std::vector<cd> &a)
{
    const double raz = azDeg * kPi / 180.0;
    const double rel = elDeg * kPi / 180.0;
    const size_t nvc = vcPosX.size();
    a.resize(nvc);
    for (size_t i = 0; i < nvc; ++i)
    {
        const double phase = vcPosX[i] * std::cos(rel) * std::sin(raz) + vcPosY[i] * std::sin(rel);
        a[i]               = std::polar(1.0, -2.0 * kPi * phase);
    }
}

/// Validate the config and fill @p g. Returns 0 on success or a negative error code matching the
/// caponROIPreprocess() contract (-1 bad counts, -2/-3 RX/virtual-array mismatch, -6 svDim mismatch,
/// -8 outside the operator's limits).
int buildGeom(const CaponROIPreprocConfig *cfg, PreprocGeom &g)
{
    // The operator's limits (OpCaponROIPreprocess.h, Limitations) for the parameters both share, so the
    // reference rejects the configurations the operator rejects. Each factor is range-checked before any
    // product of them is formed.
    if (cfg->numTx <= 0 || cfg->numTx > 8 || cfg->numRx <= 0 || cfg->numRx > 8 || cfg->numChirps <= 0 ||
        cfg->numChirps > 512)
        return -1;
    for (int tx = 0; tx < cfg->numTx; ++tx)
        if (!std::isfinite(cfg->ddmaWeightRe[tx]) || !std::isfinite(cfg->ddmaWeightIm[tx]))
            return -8;
    const bool grid = cfg->numAzVirt != 0; // explicit virtual grid (CaponROIPreprocConfig::numAzVirt)
    if ((grid ? (cfg->numAzVirt < 1 || cfg->numAzVirt > 69)
              : (cfg->numAzimuthElements < 1 || cfg->numAzimuthElements > 69)) ||
        cfg->numElevationElements < 1 || cfg->numElevationElements > 35 || cfg->roiAzSize < 1 || cfg->roiAzSize > 128 ||
        cfg->roiElSize < 1 || cfg->roiElSize > 64)
        return -8;

    g.numTx      = cfg->numTx;
    g.numRx      = cfg->numRx;
    g.numAzElem  = grid ? 0 : cfg->numAzimuthElements;
    g.numAzVirt  = grid ? cfg->numAzVirt : g.numAzElem * g.numTx;
    g.numElVirt  = cfg->numElevationElements;
    g.nvc        = g.numAzVirt * g.numElVirt;
    g.svDim      = cfg->roiAzSize * cfg->roiElSize;
    g.numStVecAz = cfg->numAzBins;
    g.numStVecEl = cfg->numElBins;
    g.H          = cfg->numDopplerBinsCovMat;

    if (!grid && g.numAzElem * cfg->numElevationElements != g.numRx)
        return -2;
    if (grid ? (g.numTx * g.numRx > g.nvc) : (g.nvc != g.numTx * g.numRx))
        return -3;

    // Beamspace 2D DFT sizes (elevation zero-padded to the azimuth size if smaller).
    g.beamAz = g.numAzVirt;
    g.beamEl = g.numElVirt > g.numAzVirt ? g.numElVirt : g.numAzVirt;
    g.roiAz  = cfg->roiAzSize;
    g.roiEl  = cfg->roiElSize;
    if (g.roiAz * g.roiEl != g.svDim)
        return -6;

    if (g.numAzVirt > 69 || g.numElVirt > 35 || g.nvc > 69 || g.beamAz > 128 || g.beamEl > 64)
        return -8;
    if (g.roiAz < 1 || g.roiAz > g.beamAz || g.roiEl < 1 || g.roiEl > g.beamEl || g.svDim > 16)
        return -8;
    if (g.numStVecAz < 1 || g.numStVecAz > 16 || g.numStVecEl < 1 || g.numStVecEl > 8)
        return -8;
    if (g.H < 8 || g.H > 64 || g.H % 8 != 0 || g.roiAz * g.numElVirt * g.H > 1024)
        return -8;
    if (cfg->inputQBits < 0 || cfg->inputQBits > 30 || cfg->steeringQBits < 0 || cfg->steeringQBits > 30)
        return -8;

    g.numRoiAzPos = g.beamAz - g.roiAz + 1;
    g.numRoiElPos = g.beamEl - g.roiEl + 1;

    // Beamspace bin -> angle (degrees) tables.
    g.azBinsBeam.resize(g.beamAz);
    g.elBinsBeam.resize(g.beamEl);
    for (int k = 0; k < g.beamAz; ++k) g.azBinsBeam[k] = asind(2.0 * k / g.beamAz - 1.0);
    for (int k = 0; k < g.beamEl; ++k) g.elBinsBeam[k] = asind(2.0 * k / g.beamEl - 1.0);

    // Virtual element positions, indexed by vc0 = rx*numTx + tx. RX layout is elevation-outer /
    // azimuth-inner (rx = el_r*numAzElem + az_r); TX is an azimuth ULA. An explicit grid places vc0 at
    // its grid position instead.
    g.vcPosX.assign(g.nvc, 0.0);
    g.vcPosY.assign(g.nvc, 0.0);
    if (grid)
    {
        for (int vc0 = 0; vc0 < g.nvc; ++vc0)
        {
            g.vcPosX[vc0] = (vc0 % g.numAzVirt) * cfg->txSpacing;
            g.vcPosY[vc0] = (vc0 / g.numAzVirt) * cfg->rxElSpacing;
        }
        return 0;
    }
    for (int rx = 0; rx < g.numRx; ++rx)
    {
        const int azR    = rx % g.numAzElem;
        const int elR    = rx / g.numAzElem;
        const double rxX = azR * cfg->rxAzSpacing;
        const double rxY = elR * cfg->rxElSpacing;
        for (int tx = 0; tx < g.numTx; ++tx)
        {
            const int vc0 = rx * g.numTx + tx;
            g.vcPosX[vc0] = rxX + tx * cfg->txSpacing;
            g.vcPosY[vc0] = rxY; // TX elevation offset is 0
        }
    }
    return 0;
}

/// Steering block (stage 10): fill one quantized steering block for ROI start position
/// (roiAzStart, roiElStart) into @p blk (NCHW [svDim, numElBins, numAzBins] complex 2S32).
/// Shared by the LUT builder and the per-ROI steering cache in caponROIPreprocess().
void buildSteeringBlock(const CaponROIPreprocConfig *cfg, const PreprocGeom &g, Beamspace2D &beamspace, int roiAzStart,
                        int roiElStart, int32_t *blk)
{
    std::vector<cd> aElem, patch;
    const double azLo = g.azBinsBeam[roiAzStart];
    const double azHi = g.azBinsBeam[roiAzStart + g.roiAz - 1];
    const double elLo = g.elBinsBeam[roiElStart];
    const double elHi = g.elBinsBeam[roiElStart + g.roiEl - 1];
    for (int azIdx = 0; azIdx < g.numStVecAz; ++azIdx)
    {
        const double azDeg = (g.numStVecAz == 1) ? azLo : azLo + (azHi - azLo) * azIdx / (g.numStVecAz - 1);
        for (int elIdx = 0; elIdx < g.numStVecEl; ++elIdx)
        {
            const double elDeg = (g.numStVecEl == 1) ? elLo : elLo + (elHi - elLo) * elIdx / (g.numStVecEl - 1);
            steeringVector(g.vcPosX, g.vcPosY, azDeg, elDeg, aElem);
            beamspace.patchROI(aElem, roiAzStart, roiElStart, patch);
            for (int c = 0; c < g.svDim; ++c)
            {
                // patch[c] == full beamspace at (roiElStart+eLoc, roiAzStart+aLoc), c = eLoc*roiAz + aLoc.
                const cd v = patch[c];
                // block[c][elIdx][azIdx]
                const size_t o = ((static_cast<size_t>(c) * g.numStVecEl + elIdx) * g.numStVecAz + azIdx) * 2u;
                blk[o]         = quantizeQ(v.real(), cfg->steeringQBits);
                blk[o + 1]     = quantizeQ(v.imag(), cfg->steeringQBits);
            }
        }
    }
}

} // namespace pva_solutions::capon_roi

using namespace pva_solutions::capon_roi;

extern "C" int caponROIPreprocessSteeringLut(const CaponROIPreprocConfig *cfg, int32_t *lut)
{
    if (!cfg || !lut)
        return -4;
    PreprocGeom g;
    int rc = buildGeom(cfg, g);
    if (rc != 0)
        return rc;

    Beamspace2D beamspace(g.numAzVirt, g.numElVirt, g.beamAz, g.beamEl, g.roiAz, g.roiEl);
    const size_t svBlockInts = static_cast<size_t>(g.svDim) * g.numStVecEl * g.numStVecAz * 2u;
    for (int rEl = 0; rEl < g.numRoiElPos; ++rEl)
        for (int rAz = 0; rAz < g.numRoiAzPos; ++rAz)
        {
            int32_t *blk = lut + (static_cast<size_t>(rEl) * g.numRoiAzPos + rAz) * svBlockInts;
            buildSteeringBlock(cfg, g, beamspace, rAz, rEl, blk);
        }
    return 0;
}
