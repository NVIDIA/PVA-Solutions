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

#include "capon_roi_preprocess_ref.h"

#include "CaponROIBeamspace.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>
// The beamspace transform, geometry and steering block are shared with the steering-LUT builder
// (pva_capon_roi_beamspace).
using namespace pva_solutions::capon_roi;

// ============================================================================
// Entry point
// ============================================================================

int caponROIPreprocess(const CaponROIPreprocConfig *cfg, const CaponROIPreprocInput *in, CaponROIPreprocOutput *out)
{
    if (!cfg || !in || !out)
        return -4;
    if (in->numDetections < 0 || in->numDetections > 1024)
        return -1;

    PreprocGeom g;
    int rc = buildGeom(cfg, g);
    if (rc != 0)
        return rc;

    const int numTx     = g.numTx;
    const int numRx     = g.numRx;
    const int numChirps = cfg->numChirps;
    const int numAzVirt = g.numAzVirt;
    const int numElVirt = g.numElVirt;
    const int nvc       = g.nvc;
    const int beamAz    = g.beamAz;
    const int beamEl    = g.beamEl;
    const int roiAz     = g.roiAz;
    const int roiEl     = g.roiEl;
    const int svDim     = g.svDim;
    const int H         = g.H;

    if (!in->snapshots || !in->rangeDopplerCube || !in->detectionList || !out->snapshotsROI || !out->steeringVectorsROI)
        return -4;
    // DDMA decode terms, as the operator takes them.
    std::vector<cd> ddmaWeights(numTx);
    for (int tx = 0; tx < numTx; ++tx)
    {
        ddmaWeights[tx] = cd(cfg->ddmaWeightRe[tx], cfg->ddmaWeightIm[tx]);
    }

    // Range-Doppler cube stride helper: layout [numRangeBins, numChirps, numRx], complex interleaved.
    auto rdCube = [&](int rangeIdx, int chirp, int rx) -> cd {
        const size_t base = ((static_cast<size_t>(rangeIdx) * numChirps + chirp) * numRx + rx) * 2u;
        return cd(in->rangeDopplerCube[base], in->rangeDopplerCube[base + 1]);
    };
    // Per-detection element snapshot: layout [numDetections, numTx, numRx], complex interleaved.
    auto snap = [&](int det, int tx, int rx) -> cd {
        const size_t base = ((static_cast<size_t>(det) * numTx + tx) * numRx + rx) * 2u;
        return cd(in->snapshots[base], in->snapshots[base + 1]);
    };

    // Output strides.
    const size_t snapRoiStride = static_cast<size_t>(H) * svDim * 2u; // per ROI (int32 count)
    const size_t svRoiStride   = static_cast<size_t>(svDim) * g.numStVecEl * g.numStVecAz * 2u;

    // Doppler selection window, as in the operator: 2*floor(N/2) bins around the detection, at least
    // one, and (doubled by FBA) at most H rows; the rows up to H are zero.
    // Checked in 64 bits: doubling a window near INT_MAX must not overflow.
    const int64_t rows = 2 * static_cast<int64_t>(cfg->dopplerNeighborhood / 2) * (cfg->enableFBA ? 2 : 1);
    if (cfg->dopplerNeighborhood < 2 || rows > H)
    {
        std::fprintf(stderr,
                     "[capon_preprocess] dopplerNeighborhood %d gives %lld snapshot rows; needs >= 2 and rows in "
                     "[1, numDopplerBinsCovMat = %d] (FBA=%d)\n",
                     cfg->dopplerNeighborhood, static_cast<long long>(rows), H, cfg->enableFBA ? 1 : 0);
        return -7;
    }
    const int halfW        = cfg->dopplerNeighborhood / 2;
    const int numSel       = 2 * halfW;
    const int numSnapshots = static_cast<int>(rows);

    std::vector<cd> beam; // scratch full beamspace [beamAz, beamEl] (coarse DoA only)
    std::vector<cd> elemVec(nvc);

    // 2D-FFT engine with twiddle tables built once for the fixed transform sizes.
    Beamspace2D beamspace(numAzVirt, numElVirt, beamAz, beamEl, roiAz, roiEl);

    // Steering-vector lookup cache. The quantized steering block depends ONLY on the ROI start
    // position (roiAzStart, roiElStart), so detections sharing a coarse ROI share a block. Compute
    // each distinct block once (via the shared buildSteeringBlock) and reuse it.
    const int numRoiAzPos    = g.numRoiAzPos;
    const size_t svBlockInts = svRoiStride;
    std::vector<std::vector<int32_t>> steeringLut(static_cast<size_t>(numRoiAzPos) * g.numRoiElPos);
    auto steeringBlockFor = [&](int roiAzStart, int roiElStart) -> const std::vector<int32_t> & {
        std::vector<int32_t> &blk = steeringLut[static_cast<size_t>(roiElStart) * numRoiAzPos + roiAzStart];
        if (!blk.empty())
            return blk; // cache hit
        blk.resize(svBlockInts);
        buildSteeringBlock(cfg, g, beamspace, roiAzStart, roiElStart, blk.data());
        return blk;
    };

    for (int det = 0; det < in->numDetections; ++det)
    {
        const int rangeIdx   = in->detectionList[static_cast<size_t>(det) * 2 + 0];
        const int detDoppler = in->detectionList[static_cast<size_t>(det) * 2 + 1];

        int32_t *snapOut = out->snapshotsROI + static_cast<size_t>(det) * snapRoiStride;
        int32_t *svOut   = out->steeringVectorsROI + static_cast<size_t>(det) * svRoiStride;

        // --- Coarse DoA (stage 5) from the per-detection element snapshot.
        for (int rx = 0; rx < numRx; ++rx)
            for (int tx = 0; tx < numTx; ++tx) elemVec[static_cast<size_t>(rx) * numTx + tx] = snap(det, tx, rx);
        beamspace(elemVec, beam);
        // With one elevation element every elevation row is the same, so search only the 0-degree row
        // (shifted index beamEl/2), as the operator does.
        const int keLo = (numElVirt == 1) ? beamEl / 2 : 0;
        const int keHi = (numElVirt == 1) ? beamEl / 2 + 1 : beamEl;
        int peakAz = 0, peakEl = keLo;
        double best = -1.0;
        for (int ke = keLo; ke < keHi; ++ke)
            for (int ka = 0; ka < beamAz; ++ka)
            {
                const double p = std::norm(beam[static_cast<size_t>(ke) * beamAz + ka]);
                if (p > best)
                {
                    best   = p;
                    peakAz = ka;
                    peakEl = ke;
                }
            }

        // --- ROI selection (stage 6): the ROI is centered on the peak bin (0-based start
        // peak - (roiSize-1)/2), clamped.
        int roiAzStart = peakAz - (roiAz - 1) / 2;
        int roiElStart = peakEl - (roiEl - 1) / 2;
        if (roiAzStart > beamAz - roiAz)
            roiAzStart = beamAz - roiAz;
        if (roiAzStart < 0)
            roiAzStart = 0;
        if (roiElStart > beamEl - roiEl)
            roiElStart = beamEl - roiEl;
        if (roiElStart < 0)
            roiElStart = 0;

        // --- Doppler window and DDM decode (stages 1-2): per-Doppler-bin virtual-array snapshots.
        std::vector<int> selDoppler;
        selDoppler.reserve(numSel);
        for (int off = -halfW; off < halfW; ++off)
        {
            int d = (detDoppler + off) % numChirps;
            if (d < 0)
                d += numChirps;
            selDoppler.push_back(d);
        }
        std::fill(snapOut + static_cast<size_t>(numSnapshots) * svDim * 2u, snapOut + snapRoiStride, 0);

        // Build virtualData[nvc][numSel].
        std::vector<cd> virtualData(static_cast<size_t>(nvc) * numSel);
        for (int di = 0; di < numSel; ++di)
        {
            const int d = selDoppler[di];
            for (int tx = 0; tx < numTx; ++tx)
            {
                // 64-bit: relDopplerOffset is any int32.
                int dBin = static_cast<int>((static_cast<int64_t>(d) + cfg->relDopplerOffset[tx]) % numChirps);
                if (dBin < 0)
                    dBin += numChirps;
                for (int rx = 0; rx < numRx; ++rx)
                {
                    const int vc0                                       = rx * numTx + tx;
                    virtualData[static_cast<size_t>(vc0) * numSel + di] = rdCube(rangeIdx, dBin, rx) * ddmaWeights[tx];
                }
            }
        }

        // --- Azimuth and elevation projection (stages 7-8) of each snapshot onto the ROI beamspace,
        //     then output quantization (stage 9) into snapshotsROI.
        // snapshotsROI layout NHW [roiNum, H, svDim]; write row h (snapshot), col c (svDim).
        std::vector<cd> patch;
        auto writeSnapshotColumn = [&](int h, const std::vector<cd> &col) {
            beamspace.patchROI(col, roiAzStart, roiElStart, patch);
            for (int c = 0; c < svDim; ++c)
            {
                // patch[c] == full beamspace at (roiElStart+eLoc, roiAzStart+aLoc), c = eLoc*roiAz + aLoc.
                const cd v     = patch[c];
                const size_t o = (static_cast<size_t>(h) * svDim + c) * 2u;
                snapOut[o]     = quantizeQ(v.real(), cfg->inputQBits);
                snapOut[o + 1] = quantizeQ(v.imag(), cfg->inputQBits);
            }
        };

        std::vector<cd> col(nvc);
        for (int di = 0; di < numSel; ++di)
        {
            for (int i = 0; i < nvc; ++i) col[i] = virtualData[static_cast<size_t>(i) * numSel + di];
            writeSnapshotColumn(di, col);
            if (cfg->enableFBA)
            {
                // Forward-backward averaging (stage 3): element i = conj(virtualData[nvc-1-i][di]).
                std::vector<cd> fba(nvc);
                for (int i = 0; i < nvc; ++i)
                    fba[i] = std::conj(virtualData[static_cast<size_t>(nvc - 1 - i) * numSel + di]);
                writeSnapshotColumn(numSel + di, fba);
            }
        }

        // --- Steering block (stage 10), computed once per ROI position and reused.
        const std::vector<int32_t> &svBlock = steeringBlockFor(roiAzStart, roiElStart);
        std::memcpy(svOut, svBlock.data(), svBlock.size() * sizeof(int32_t));
    }

    return 0;
}
