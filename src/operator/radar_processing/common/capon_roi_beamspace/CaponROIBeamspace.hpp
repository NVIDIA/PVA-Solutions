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

/**
 * @file CaponROIBeamspace.hpp
 *
 * @brief Internal beamspace helpers shared by the steering-LUT builder and the CPU reference
 *        (caponROIPreprocess): the 2D beamspace transform, the derived geometry and the steering block.
 */

#ifndef CAPON_ROI_BEAMSPACE_HPP
#define CAPON_ROI_BEAMSPACE_HPP

#include "capon_roi_beamspace.h"

#include <complex>
#include <cstdint>
#include <vector>

namespace pva_solutions::capon_roi {

using cd = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;

/**
 * @brief conj(x) -> zero-padded 2D DFT (dim0=az size beamAz, dim1=el size beamEl)
 *        -> shift both dims so output bin b is DFT bin (b + beam/2) % beam.
 *
 * The DFT twiddle tables and stage scratch depend only on the (fixed) transform sizes, so they
 * are built once in the constructor and reused across every call.
 */
class Beamspace2D
{
public:
    Beamspace2D(int numAzVirt, int numElVirt, int beamAz, int beamEl, int roiAz, int roiEl)
        : m_numAzVirt(numAzVirt)
        , m_numElVirt(numElVirt)
        , m_beamAz(beamAz)
        , m_beamEl(beamEl)
        , m_roiAz(roiAz)
        , m_roiEl(roiEl)
        , m_Waz(static_cast<size_t>(beamAz) * numAzVirt)
        , m_Wel(static_cast<size_t>(beamEl) * numElVirt)
        , m_conjX(static_cast<size_t>(numAzVirt) * numElVirt)
        , m_tmp(static_cast<size_t>(beamAz) * numElVirt)
        , m_full(static_cast<size_t>(beamAz) * beamEl)
        , m_tmpRoi(static_cast<size_t>(roiAz) * numElVirt)
    {
        // W_az[ka][a] = exp(-j 2pi ka a / beamAz), W_el[ke][e] = exp(-j 2pi ke e / beamEl)
        for (int ka = 0; ka < beamAz; ++ka)
            for (int a = 0; a < numAzVirt; ++a)
                m_Waz[static_cast<size_t>(ka) * numAzVirt + a] = std::polar(1.0, -2.0 * kPi * ka * a / beamAz);
        for (int ke = 0; ke < beamEl; ++ke)
            for (int e = 0; e < numElVirt; ++e)
                m_Wel[static_cast<size_t>(ke) * numElVirt + e] = std::polar(1.0, -2.0 * kPi * ke * e / beamEl);
    }

    /**
     * @param x   Element-space array, column-major [numAzVirt, numElVirt] (x[e*numAzVirt + a]).
     * @param out Result, column-major [beamAz, beamEl] (out[ke*beamAz + ka]), already shifted.
     */
    void operator()(const std::vector<cd> &x, std::vector<cd> &out)
    {
        // conj(x) once, outside the per-bin DFT loops.
        for (size_t i = 0; i < m_conjX.size(); ++i) m_conjX[i] = std::conj(x[i]);

        // Stage 1: DFT along azimuth (dim0). tmp[ka][e], ka in [0,beamAz), e in [0,numElVirt).
        for (int e = 0; e < m_numElVirt; ++e)
        {
            const cd *xcol = &m_conjX[static_cast<size_t>(e) * m_numAzVirt];
            for (int ka = 0; ka < m_beamAz; ++ka)
            {
                cd acc(0.0, 0.0);
                const cd *wrow = &m_Waz[static_cast<size_t>(ka) * m_numAzVirt];
                for (int a = 0; a < m_numAzVirt; ++a)
                {
                    acc += xcol[a] * wrow[a];
                }
                m_tmp[static_cast<size_t>(ka) * m_numElVirt + e] = acc;
            }
        }

        // Stage 2: DFT along elevation (dim1). full[ka][ke].
        for (int ka = 0; ka < m_beamAz; ++ka)
        {
            const cd *trow = &m_tmp[static_cast<size_t>(ka) * m_numElVirt];
            for (int ke = 0; ke < m_beamEl; ++ke)
            {
                cd acc(0.0, 0.0);
                const cd *wrow = &m_Wel[static_cast<size_t>(ke) * m_numElVirt];
                for (int e = 0; e < m_numElVirt; ++e)
                {
                    acc += trow[e] * wrow[e];
                }
                m_full[static_cast<size_t>(ka) * m_beamEl + ke] = acc;
            }
        }

        // Shift both dims and store column-major [beamAz, beamEl]: out[ke*beamAz + ka].
        const int shAz = m_beamAz / 2;
        const int shEl = m_beamEl / 2;
        out.assign(static_cast<size_t>(m_beamAz) * m_beamEl, cd(0.0, 0.0));
        for (int ka = 0; ka < m_beamAz; ++ka)
        {
            const int kaS = (ka + shAz) % m_beamAz;
            for (int ke = 0; ke < m_beamEl; ++ke)
            {
                const int keS                                  = (ke + shEl) % m_beamEl;
                out[static_cast<size_t>(keS) * m_beamAz + kaS] = m_full[static_cast<size_t>(ka) * m_beamEl + ke];
            }
        }
    }

    /**
     * @brief Azimuth and elevation projection (stages 7-8): compute ONLY the roiAz x roiEl ROI
     *        patch of the shifted beamspace via a partial 2D DFT. Fills
     *        outPatch[eLoc*roiAz + aLoc] = full-transform bin at shifted position
     *        (roiElStart+eLoc, roiAzStart+aLoc). Byte-identical to operator()(x,out) followed by
     *        extracting that patch (same twiddles, same accumulation order).
     *
     * @param roiAzStart / roiElStart  0-based ROI start in the shifted beamspace,
     *                                 in [0, beamAz-roiAz] / [0, beamEl-roiEl].
     */
    void patchROI(const std::vector<cd> &x, int roiAzStart, int roiElStart, std::vector<cd> &outPatch)
    {
        for (size_t i = 0; i < m_conjX.size(); ++i) m_conjX[i] = std::conj(x[i]);

        const int shAz = m_beamAz / 2;
        const int shEl = m_beamEl / 2;

        // Azimuth projection: az DFT, only for the roiAz unshifted az bins that map to the ROI's
        // shifted range. Shifted bin kaS -> unshifted ka = (kaS - shAz) mod beamAz.
        for (int aLoc = 0; aLoc < m_roiAz; ++aLoc)
        {
            int ka = (roiAzStart + aLoc - shAz) % m_beamAz;
            if (ka < 0)
                ka += m_beamAz;
            const cd *wrow = &m_Waz[static_cast<size_t>(ka) * m_numAzVirt];
            for (int e = 0; e < m_numElVirt; ++e)
            {
                const cd *xcol = &m_conjX[static_cast<size_t>(e) * m_numAzVirt];
                cd acc(0.0, 0.0);
                for (int a = 0; a < m_numAzVirt; ++a)
                {
                    acc += xcol[a] * wrow[a];
                }
                m_tmpRoi[static_cast<size_t>(aLoc) * m_numElVirt + e] = acc;
            }
        }

        // Elevation projection: el DFT, only for the roiEl bins of the ROI, into the roiAz x roiEl patch.
        outPatch.assign(static_cast<size_t>(m_roiAz) * m_roiEl, cd(0.0, 0.0));
        for (int eLoc = 0; eLoc < m_roiEl; ++eLoc)
        {
            int ke = (roiElStart + eLoc - shEl) % m_beamEl;
            if (ke < 0)
                ke += m_beamEl;
            const cd *wrow = &m_Wel[static_cast<size_t>(ke) * m_numElVirt];
            for (int aLoc = 0; aLoc < m_roiAz; ++aLoc)
            {
                const cd *trow = &m_tmpRoi[static_cast<size_t>(aLoc) * m_numElVirt];
                cd acc(0.0, 0.0);
                for (int e = 0; e < m_numElVirt; ++e)
                {
                    acc += trow[e] * wrow[e];
                }
                outPatch[static_cast<size_t>(eLoc) * m_roiAz + aLoc] = acc;
            }
        }
    }

private:
    int m_numAzVirt, m_numElVirt, m_beamAz, m_beamEl, m_roiAz, m_roiEl;
    std::vector<cd> m_Waz, m_Wel;  // twiddle tables (constant across calls)
    std::vector<cd> m_conjX;       // conj(x) scratch
    std::vector<cd> m_tmp, m_full; // full-transform stage scratch
    std::vector<cd> m_tmpRoi;      // partial (ROI) stage-1 scratch
};

/// asin(v) in degrees, with v clamped to [-1, 1].
double asind(double v);

/// Element-space steering vector (indexed by vc0 = rx*numTx + tx):
/// phase = x*cos(el)*sin(az) + y*sin(el); a = exp(-j 2pi phase).
void steeringVector(const std::vector<double> &vcPosX, const std::vector<double> &vcPosY, double azDeg, double elDeg,
                    std::vector<cd> &a);

/// Saturating quantize a double to int32 Q(qbits).
inline int32_t quantizeQ(double v, int qbits)
{
    const double scale  = static_cast<double>(static_cast<int64_t>(1) << qbits);
    const double satMax = 2147483520.0;
    const double satMin = -2147483648.0;
    double s            = std::round(v * scale);
    if (s > satMax)
        s = satMax;
    if (s < satMin)
        s = satMin;
    return static_cast<int32_t>(s);
}

/// Derived geometry + precomputed angle/position tables, built once from the config and shared by
/// caponROIPreprocess() and caponROIPreprocessSteeringLut().
struct PreprocGeom
{
    int numTx, numRx, numAzElem, numAzVirt, numElVirt, nvc;
    int beamAz, beamEl, roiAz, roiEl, svDim, numStVecAz, numStVecEl;
    int numRoiAzPos, numRoiElPos, H;
    std::vector<double> vcPosX, vcPosY; // virtual element positions (indexed by vc0 = rx*numTx + tx)
    std::vector<double> azBinsBeam, elBinsBeam;
};

/// Validate the config and fill @p g. Returns 0 on success or a negative error code matching the
/// caponROIPreprocess() contract (-1 bad counts, -2/-3 RX/virtual-array mismatch, -6 svDim mismatch,
/// -8 outside the operator's limits).
int buildGeom(const CaponROIPreprocConfig *cfg, PreprocGeom &g);

/// Steering block (stage 10): fill one quantized steering block for ROI start position
/// (roiAzStart, roiElStart) into @p blk (NCHW [svDim, numElBins, numAzBins] complex 2S32).
/// Shared by the LUT builder and the per-ROI steering cache in caponROIPreprocess().
void buildSteeringBlock(const CaponROIPreprocConfig *cfg, const PreprocGeom &g, Beamspace2D &beamspace, int roiAzStart,
                        int roiElStart, int32_t *blk);

} // namespace pva_solutions::capon_roi

#endif // CAPON_ROI_BEAMSPACE_HPP
