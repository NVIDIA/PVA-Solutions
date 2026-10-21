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

#include "OpCaponROIPreprocess.hpp"

#include "device/capon_roi_preprocess_params.h"

#include <IOperator.hpp>
#include <RadarUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpCaponROIPreprocess.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_nonsafety.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <cupva_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_capon_roi_preprocess_dft_dev);
CUPVA_DECLARE_EXECUTABLE(pva_capon_roi_preprocess_dftgen_dev);
CUPVA_DECLARE_EXECUTABLE(pva_capon_roi_preprocess_fft_dev);

namespace pva_solutions::priv {

namespace {

/* Coarse-DoA kernel selection: the batched FFT-64 kernel when it is requested and the geometry is the
 * one it transforms (one elevation row, 64 azimuth taps, 64 azimuth beams); otherwise the direct-DFT
 * kernel. */
bool UseFftCoarseDoA(const PVACaponROIPreprocessParams &params)
{
    return params.coarseDoAUseFft && params.numElVirt == 1 && params.beamAz == 64 && params.numAzVirt == 64;
}

bool UseDotProductProjection(const PVACaponROIPreprocessParams &params)
{
    const int32_t numSel = 2 * (params.dopplerNeighborhood / 2);
    const bool fbaSym    = params.enableFBA && !params.useCustomTwiddles;
    /* Elevation: two element rows, and tmpRoi holds the forward and (FBA by symmetry) backward Doppler blocks. */
    const int32_t blocks = fbaSym ? 2 * ((numSel + 7) / 8) * 8 : params.numDopplerBinsCovMat;
    const bool elOk      = params.numElVirt == 1 ||
                      (params.numElVirt == 2 && params.roiAz * params.numElVirt * blocks <= CRP_MAX_SVDIM * CRP_MAX_H);
    return elOk && params.numAzVirt % 2 == 0 && (!params.enableFBA || !params.useCustomTwiddles || numSel % 8 == 0);
}

bool CanFoldDdmDecode(const PVACaponROIPreprocessParams &params, bool useDotProjection)
{
    bool unit = true;
    for (int32_t tx = 0; tx < params.numTx; tx++)
    {
        const double wr = params.ddmaWeightRe[tx], wi = params.ddmaWeightIm[tx];
        unit = unit && (wr * wr + wi * wi <= 1.0 + 1e-5);
    }
    /* With elevation rows the channel vc = e*numAzVirt + a must keep the TX of its azimuth index a. */
    const bool txByAz = (params.numElVirt == 1) || (params.numAzVirt % params.numTx == 0);
    return useDotProjection && unit && txByAz && (!params.enableFBA || !params.useCustomTwiddles);
}

std::array<int32_t, CRP_MAX_TX> VdTxBase(const PVACaponROIPreprocessParams &params)
{
    const int32_t halfW = params.dopplerNeighborhood / 2;
    std::array<int32_t, CRP_MAX_TX> txBase{};
    for (int32_t tx = 0; tx < params.numTx; tx++)
    {
        const int64_t b = (static_cast<int64_t>(params.relDopplerOffset[tx]) - halfW) % params.numChirps;
        txBase[tx]      = static_cast<int32_t>(b < 0 ? b + params.numChirps : b);
    }
    return txBase;
}

void BuildFbaRotation(int32_t numVirt, int32_t beam, int16_t *rot)
{
    constexpr double kPi = 3.14159265358979323846;
    for (int32_t k = 0; k < beam; k++)
    {
        const double th   = -2.0 * kPi * ((k * (numVirt - 1)) % beam) / beam;
        const double c[2] = {std::cos(th), std::sin(th)};
        for (int32_t p = 0; p < 2; p++)
        {
            const double h     = std::nearbyint(c[p] * 16384.0);
            rot[4 * k + p]     = static_cast<int16_t>(h);
            rot[4 * k + 2 + p] = static_cast<int16_t>(std::nearbyint((c[p] * 16384.0 - h) * 32768.0));
        }
    }
}

/* Every DMA pitch is derived from the shapes, so a tensor must have no padding between rows. */
void CheckDenselyPacked(const NVCVTensorRequirements &r, const char *name)
{
    int32_t elemBytes = 0;
    if (nvcvDataTypeGetStrideBytes(r.dtype, &elemBytes) != NVCV_SUCCESS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s has an invalid data type", name);
    }
    int64_t expected = elemBytes;
    for (int32_t d = r.rank - 1; d >= 0; --d)
    {
        if (r.shape[d] != 1 && r.strides[d] != expected)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "%s must be densely packed: dimension %d has a stride of %lld bytes, expected %lld",
                                  name, d, static_cast<long long>(r.strides[d]), static_cast<long long>(expected));
        }
        expected *= r.shape[d];
    }
}

/* A Submit tensor must be the one described to Create: same layout, type, shape and strides. */
NVCVTensorData ExportMatching(NVCVTensorHandle handle, const NVCVTensorRequirements &r, const char *name)
{
    if (handle == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor must not be NULL", name);
    }
    NVCVTensorData d{};
    if (nvcvTensorExportData(handle, &d) != NVCV_SUCCESS || d.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be a strided tensor", name);
    }
    bool same = d.dtype == r.dtype && d.layout == r.layout && d.rank == r.rank;
    for (int32_t i = 0; same && i < r.rank; ++i)
    {
        same = d.shape[i] == r.shape[i] && d.buffer.strided.strides[i] == r.strides[i];
    }
    if (!same)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s does not match the tensor requirements passed to Create", name);
    }
    return d;
}

/* Host view of a tensor the CPU part of Submit reads. */
template<typename T>
T *HostPointer(const NVCVTensorData &d, const char *name)
{
    void *p = nullptr;
    if (CupvaMemGetHostPointer(&p, static_cast<void *>(d.buffer.strided.basePtr)) != CUPVA_ERROR_NONE || p == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be host-accessible cuPVA memory", name);
    }
    return static_cast<T *>(p);
}

/* RasterDataFlow allows at most 256 tiles per dimension, so maxDet > 256 is reshaped into a
 * [tileCols x tileRows] grid; row-major acquire order still yields det = row*tileCols + col. Returns the
 * smallest tileCols >= ceil(maxDet/256) that divides maxDet into at most 256 rows, or 0 if there is none. */
constexpr int32_t kMaxTilesPerDim = 256;
int32_t DetTileColumns(int32_t maxDet)
{
    if (maxDet <= kMaxTilesPerDim)
    {
        return 1;
    }
    for (int32_t x = (maxDet + kMaxTilesPerDim - 1) / kMaxTilesPerDim; x <= kMaxTilesPerDim; ++x)
    {
        if (maxDet % x == 0 && (maxDet / x) <= kMaxTilesPerDim)
        {
            return x;
        }
    }
    return 0;
}

} // namespace

std::mutex CaponROIPreprocess::s_execMapMutex;
std::map<CaponROIPreprocess::ExecKey, std::weak_ptr<cupva::Executable>> CaponROIPreprocess::s_execs;

std::shared_ptr<cupva::Executable> CaponROIPreprocess::GetExec(Kernel kernel)
{
    IOperator::ExecutableData execData = {CUPVA_EXECUTABLE_DATA(pva_capon_roi_preprocess_dft_dev),
                                          CUPVA_EXECUTABLE_SIZE(pva_capon_roi_preprocess_dft_dev)};
    if (kernel == Kernel::kDftGeneral)
    {
        execData = {CUPVA_EXECUTABLE_DATA(pva_capon_roi_preprocess_dftgen_dev),
                    CUPVA_EXECUTABLE_SIZE(pva_capon_roi_preprocess_dftgen_dev)};
    }
    else if (kernel == Kernel::kFft)
    {
        execData = {CUPVA_EXECUTABLE_DATA(pva_capon_roi_preprocess_fft_dev),
                    CUPVA_EXECUTABLE_SIZE(pva_capon_roi_preprocess_fft_dev)};
    }
    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent(), kernel);
    return IOperator::GetExecutable<ExecKey>(execData, execKey, s_execs, s_execMapMutex);
}

/* Rejects every parameter value the operator cannot run, in the order of the header's Limitations,
 * and fills in the defaults (coarseElVirt, coarse-DoA field of view). Runs before any allocation. */
void CaponROIPreprocess::ValidateParams()
{
    const int32_t numTx     = m_params.numTx;
    const int32_t numRx     = m_params.numRx;
    const int32_t numAzVirt = m_params.numAzVirt;
    const int32_t numElVirt = m_params.numElVirt;
    const int32_t beamAz    = m_params.beamAz;
    const int32_t beamEl    = m_params.beamEl;
    const int32_t svDim     = m_params.svDim;
    const int32_t H         = m_params.numDopplerBinsCovMat;
    const int32_t numChirps = m_params.numChirps;
    const int32_t maxDet    = m_params.maxDet;
    const int32_t roiAz     = m_params.roiAz;
    const int32_t roiEl     = m_params.roiEl;
    const int32_t numAzBins = m_params.numAzBins;
    const int32_t numElBins = m_params.numElBins;

    /* The FFT coarse DoA is a plain DFT, so it would search different beams than the projection uses
     * with custom tables. It transforms exactly 64 taps (numAzVirt == 64, no zero padding). */
    if (m_params.coarseDoAUseFft && m_params.useCustomTwiddles)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "coarseDoAUseFft requires the generated DFT twiddles (useCustomTwiddles == false)");
    }
    if (m_params.coarseDoAUseFft && (numElVirt != 1 || beamAz != 64 || numAzVirt != 64))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "coarseDoAUseFft requires numElVirt == 1, beamAz == 64 and numAzVirt == 64; got %d, %d and %d", numElVirt,
            beamAz, numAzVirt);
    }
    /* Every VMEM buffer is sized from the CRP_MAX_* limits in capon_roi_preprocess_params.h, and the
     * device has no way to report an overrun -- it would silently corrupt neighbouring buffers. Check
     * them here, where a bad configuration can still be rejected. */
    if (numTx <= 0 || numTx > CRP_MAX_TX || numRx <= 0 || numRx > CRP_MAX_RX)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numTx and numRx must be in [1, %d]; got %d and %d",
                              CRP_MAX_TX, numTx, numRx);
    }
    /* The device decodes with these weights; a NaN or Inf one would give unspecified snapshots. */
    for (int32_t tx = 0; tx < numTx; tx++)
    {
        if (!std::isfinite(m_params.ddmaWeightRe[tx]) || !std::isfinite(m_params.ddmaWeightIm[tx]))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "ddmaWeightRe[%d] and ddmaWeightIm[%d] must be finite", tx, tx);
        }
    }
    /* Coarse DoA is bounded by the virtual-array extents, the projection by the physical channel
     * count; they differ when the coarse-DoA array is a zero-filled position grid. */
    if (numAzVirt <= 0 || numAzVirt > CRP_MAX_AZVIRT || numElVirt <= 0 || numElVirt > CRP_MAX_ELVIRT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numAzVirt must be in [1, %d] and numElVirt in [1, %d]; got %d and %d", CRP_MAX_AZVIRT,
                              CRP_MAX_ELVIRT, numAzVirt, numElVirt);
    }
    /* Products and differences of parameters are formed only after their factors are range-checked. */
    const int32_t nvc = numAzVirt * numElVirt;
    if (nvc > CRP_MAX_NVC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numAzVirt*numElVirt must be <= %d; got %d * %d = %d", CRP_MAX_NVC, numAzVirt, numElVirt,
                              nvc);
    }
    /* The device VD build writes physical channel rx*numTx+tx into a VD row of nvc channels. */
    if (numTx * numRx > nvc)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numTx*numRx must be <= numAzVirt*numElVirt; got %d * %d > %d * %d", numTx, numRx,
                              numAzVirt, numElVirt);
    }
    if (beamAz <= 0 || beamAz > CRP_MAX_BEAM_AZ || beamEl <= 0 || beamEl > CRP_MAX_BEAM_EL)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "beamAz must be in [1, %d] and beamEl in [1, %d]; got %d and %d", CRP_MAX_BEAM_AZ,
                              CRP_MAX_BEAM_EL, beamAz, beamEl);
    }

    /* Coarse-DoA position grid. 0 = no separate grid. */
    if (m_params.coarseElVirt == 0)
    {
        m_params.coarseElVirt = numElVirt;
    }
    const int32_t coarseElVirt = m_params.coarseElVirt;
    if (coarseElVirt < numElVirt || coarseElVirt > CRP_MAX_ELVIRT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "coarseElVirt must be 0 or in [numElVirt, %d] = [%d, %d]; got %d", CRP_MAX_ELVIRT,
                              numElVirt, CRP_MAX_ELVIRT, coarseElVirt);
    }
    if (coarseElVirt != numElVirt)
    {
        /* The grid shares the az twiddle table with the projection, and reshapes the el one. Keeping
         * the elevation projection off is what makes that safe. */
        if (numElVirt != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "a coarse-DoA grid (coarseElVirt = %d != numElVirt) requires numElVirt == 1; got %d",
                                  coarseElVirt, numElVirt);
        }
        if (m_params.coarseDoAUseFft)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "a coarse-DoA grid (coarseElVirt != numElVirt) requires coarseDoAUseFft == false");
        }
    }
    if (beamAz * numAzVirt * 2 > BUF_SZ_WAZ && coarseElVirt == numElVirt)
    {
        /* Without a grid the coarse peak search may fall to the KC=4 hybrid, whose fp32 refine reads
         * the full az table out of VMEM; there is no room for one this large. */
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "beamAz*numAzVirt must be <= %d without a coarse-DoA grid; got %d * %d = %d",
                              BUF_SZ_WAZ / 2, beamAz, numAzVirt, beamAz * numAzVirt);
    }

    /* Coarse-DoA field of view: all four zero means the full beamspace. */
    if (m_params.coarseAzBinLo == 0 && m_params.coarseAzBinHi == 0 && m_params.coarseElBinLo == 0 &&
        m_params.coarseElBinHi == 0)
    {
        m_params.coarseAzBinHi = beamAz - 1;
        m_params.coarseElBinHi = beamEl - 1;
    }
    const bool fullFov = (m_params.coarseAzBinLo == 0 && m_params.coarseAzBinHi == beamAz - 1 &&
                          m_params.coarseElBinLo == 0 && m_params.coarseElBinHi == beamEl - 1);
    if (m_params.coarseAzBinLo < 0 || m_params.coarseAzBinHi >= beamAz ||
        m_params.coarseAzBinLo > m_params.coarseAzBinHi || m_params.coarseElBinLo < 0 ||
        m_params.coarseElBinHi >= beamEl || m_params.coarseElBinLo > m_params.coarseElBinHi)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "the coarse-DoA field of view must satisfy 0 <= coarseAzBinLo <= coarseAzBinHi < beamAz "
                              "and 0 <= coarseElBinLo <= coarseElBinHi < beamEl; got az [%d, %d], el [%d, %d] "
                              "with beamAz %d, beamEl %d",
                              m_params.coarseAzBinLo, m_params.coarseAzBinHi, m_params.coarseElBinLo,
                              m_params.coarseElBinHi, beamAz, beamEl);
    }
    if (!fullFov && m_params.coarseDoAUseFft)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "coarseDoAUseFft requires the full coarse-DoA field of view (all four coarse*Bin* zero, "
                              "or az [0, beamAz-1] and el [0, beamEl-1])");
    }
    if (roiAz <= 0 || roiEl <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "roiAz and roiEl must be >= 1; got %d and %d",
                              roiAz, roiEl);
    }
    if (roiAz > beamAz || roiEl > beamEl)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "roiAz must be <= beamAz and roiEl <= beamEl; got %d > %d or %d > %d", roiAz, beamAz,
                              roiEl, beamEl);
    }
    if (svDim <= 0 || svDim > CRP_MAX_SVDIM)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "svDim must be in [1, %d]; got %d", CRP_MAX_SVDIM,
                              svDim);
    }
    if (maxDet <= 0 || maxDet > CRP_MAX_DET)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxDet must be in [1, %d]; got %d", CRP_MAX_DET,
                              maxDet);
    }
    if (numChirps <= 0 || numChirps > CRP_MAX_CHIRPS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numChirps must be in [1, %d]; got %d",
                              CRP_MAX_CHIRPS, numChirps);
    }
    if (numAzBins <= 0 || numAzBins > CRP_MAX_AZBINS || numElBins <= 0 || numElBins > CRP_MAX_ELBINS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numAzBins must be in [1, %d] and numElBins in [1, %d]; got %d and %d", CRP_MAX_AZBINS,
                              CRP_MAX_ELBINS, numAzBins, numElBins);
    }
    /* H is the vector axis of every projection and quantize loop, all of which step in whole CRP_VECW(=8)
     * blocks and would silently drop the remainder rows of snapshotsROI. */
    if (H <= 0 || H > CRP_MAX_H || (H % 8) != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numDopplerBinsCovMat must be a multiple of 8 in [8, %d]; got %d", CRP_MAX_H, H);
    }
    /* The device VD build writes 2*floor(dopplerNeighborhood/2) Doppler rows, doubled by FBA, per channel; the
     * rows up to H stay zero. Fewer than 2 would give all-zero snapshots. */
    if (m_params.dopplerNeighborhood < 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "dopplerNeighborhood must be >= 2; got %d",
                              m_params.dopplerNeighborhood);
    }
    /* 64-bit: doubling the window of a dopplerNeighborhood near INT_MAX must not overflow. */
    const int64_t numRows = 2 * static_cast<int64_t>(m_params.dopplerNeighborhood / 2) * (m_params.enableFBA ? 2 : 1);
    if (numRows > H)
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "dopplerNeighborhood = %d gives %lld Doppler rows (%s), more than numDopplerBinsCovMat = %d",
            m_params.dopplerNeighborhood, static_cast<long long>(numRows),
            m_params.enableFBA ? "2*floor(N/2), doubled by FBA" : "2*floor(N/2)", H);
    }
    /* The device output shift and the Capon DoA operator that consumes these tensors both assume a
     * Q format in [0, 30]. steeringQBits is the steering LUT's format; the operator only gathers it. */
    if (m_params.inputQBits < 0 || m_params.inputQBits > 30 || m_params.steeringQBits < 0 ||
        m_params.steeringQBits > 30)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "inputQBits and steeringQBits must be in [0, 30]; got %d and %d", m_params.inputQBits,
                              m_params.steeringQBits);
    }
    if (svDim != roiAz * roiEl)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "svDim must equal roiAz*roiEl; got %d != %d * %d",
                              svDim, roiAz, roiEl);
    }
    /* The az projection's partial sums are [roiAz][numElVirt][H] in a [CRP_MAX_SVDIM][CRP_MAX_H] buffer. */
    if (roiAz * numElVirt * H > BUF_SZ_PATCHMAT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "roiAz*numElVirt*numDopplerBinsCovMat must be <= %d; got %d * %d * %d = %d",
                              BUF_SZ_PATCHMAT, roiAz, numElVirt, H, roiAz * numElVirt * H);
    }

    /* The per-detection input RasterDataFlows (elementSnapshots, rangeDopplerPlanes) hold tileCols detections per
     * line; a line pitch is at most 65535. The outputs are not tiled this way: snapshotsROI lines are one
     * detection's rows, and the steering blocks are copied DRAM to DRAM per detection. */
    const int32_t minCols  = (maxDet + kMaxTilesPerDim - 1) / kMaxTilesPerDim;
    const int32_t tileCols = DetTileColumns(maxDet);
    if (tileCols == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxDet = %d above 256 needs a divisor in [ceil(maxDet/256), 256] = [%d, 256]", maxDet,
                              minCols);
    }
    constexpr int32_t kMaxLinePitch = 65535;
    const int32_t elemRowLen        = numAzVirt * m_params.coarseElVirt * 2;
    const int32_t planeRowLen       = numChirps * numRx * 2;
    const int32_t maxRowLen         = std::max(elemRowLen, planeRowLen);
    if (tileCols * maxRowLen > kMaxLinePitch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "maxDet = %d is tiled %d detections per DMA line of %d elements each (the larger of "
                              "2*numAzVirt*coarseElVirt and 2*numChirps*numRx); %d * %d exceeds 65535. Choose a "
                              "maxDet whose smallest divisor >= ceil(maxDet/256) keeps the product within 65535.",
                              maxDet, tileCols, maxRowLen, tileCols, maxRowLen);
    }
}

/* Checks every tensor against the (validated) params and records the requirements Submit matches. */
void CaponROIPreprocess::ValidateRequirements(const NVCVTensorRequirements *const reqs[kNumTensors])
{
    for (int32_t t = 0; t < kNumTensors; ++t)
    {
        const bool optional = (t == kWaz || t == kWel) && !m_params.useCustomTwiddles;
        if (reqs[t] == nullptr && !optional)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s requirements must not be NULL",
                                  kTensorNames[t]);
        }
    }
    const NVCVTensorRequirements *wazReqs                = reqs[kWaz];
    const NVCVTensorRequirements *welReqs                = reqs[kWel];
    const NVCVTensorRequirements *detectionCountReqs     = reqs[kCount];
    const NVCVTensorRequirements *detDopplersReqs        = reqs[kDetDop];
    const NVCVTensorRequirements *elementSnapshotsReqs   = reqs[kElem];
    const NVCVTensorRequirements *rangeDopplerPlanesReqs = reqs[kPlane];
    const NVCVTensorRequirements *steeringLutReqs        = reqs[kLut];
    const NVCVTensorRequirements *snapshotsROIReqs       = reqs[kSnap];
    const NVCVTensorRequirements *steeringVectorsROIReqs = reqs[kSv];
    const int32_t numRx                                  = m_params.numRx;
    const int32_t numAzVirt                              = m_params.numAzVirt;
    const int32_t beamAz                                 = m_params.beamAz;
    const int32_t beamEl                                 = m_params.beamEl;
    const int32_t svDim                                  = m_params.svDim;
    const int32_t H                                      = m_params.numDopplerBinsCovMat;
    const int32_t numChirps                              = m_params.numChirps;
    const int32_t maxDet                                 = m_params.maxDet;
    const int32_t numAzBins                              = m_params.numAzBins;
    const int32_t numElBins                              = m_params.numElBins;
    const int32_t numRoiPos   = (beamAz - m_params.roiAz + 1) * (beamEl - m_params.roiEl + 1);
    const int32_t svBlockCplx = svDim * numElBins * numAzBins;

    if (snapshotsROIReqs->layout != nvcv::TENSOR_NHW || snapshotsROIReqs->dtype != NVCV_DATA_TYPE_2S32 ||
        snapshotsROIReqs->shape[0] != maxDet || snapshotsROIReqs->shape[1] != H || snapshotsROIReqs->shape[2] != svDim)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "snapshotsROI must be an NHW 2S32 tensor of shape [%d, %d, %d]", maxDet, H, svDim);
    }
    /* Steering LUT: HW [numRoiPos, svDim*numElBins*numAzBins] complex 2S32, row-contiguous. */
    if (steeringLutReqs->layout != nvcv::TENSOR_HW || steeringLutReqs->dtype != NVCV_DATA_TYPE_2S32 ||
        steeringLutReqs->shape[0] != numRoiPos || steeringLutReqs->shape[1] != svBlockCplx)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "steeringLut must be an HW 2S32 tensor of shape [%d, %d]", numRoiPos, svBlockCplx);
    }
    /* Steering output: NCHW [maxDet, svDim, numElBins, numAzBins] complex 2S32. */
    if (steeringVectorsROIReqs->layout != nvcv::TENSOR_NCHW || steeringVectorsROIReqs->dtype != NVCV_DATA_TYPE_2S32 ||
        steeringVectorsROIReqs->shape[0] != maxDet || steeringVectorsROIReqs->shape[1] != svDim ||
        steeringVectorsROIReqs->shape[2] != numElBins || steeringVectorsROIReqs->shape[3] != numAzBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "steeringVectorsROI must be an NCHW 2S32 tensor of shape [%d, %d, %d, %d]", maxDet, svDim,
                              numElBins, numAzBins);
    }
    /* Inputs: HW tensors, read at exactly these sizes by the DMA (waz and wel also by the host, which builds the
     * twiddle tables from them when they are given). */
    auto checkHW = [](const NVCVTensorRequirements *r, NVCVDataType dtype, int64_t h, int64_t w, const char *name) {
        if (r->layout != nvcv::TENSOR_HW || r->dtype != dtype || r->shape[0] != h || r->shape[1] != w)
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be an HW %s tensor of shape [%lld, %lld]", name,
                dtype == NVCV_DATA_TYPE_S32 ? "S32" : "2F32", static_cast<long long>(h), static_cast<long long>(w));
        }
    };
    if (m_params.useCustomTwiddles)
    {
        checkHW(wazReqs, NVCV_DATA_TYPE_2F32, 1, static_cast<int64_t>(beamAz) * numAzVirt, "waz");
        checkHW(welReqs, NVCV_DATA_TYPE_2F32, 1, static_cast<int64_t>(beamEl) * m_params.coarseElVirt, "wel");
    }
    if (detectionCountReqs->layout != nvcv::TENSOR_W || detectionCountReqs->dtype != NVCV_DATA_TYPE_S32 ||
        detectionCountReqs->shape[0] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "detectionCount must be a W S32 tensor of shape [1]");
    }
    checkHW(detDopplersReqs, NVCV_DATA_TYPE_S32, 1, maxDet, "detDopplers");
    checkHW(elementSnapshotsReqs, NVCV_DATA_TYPE_2F32, maxDet, static_cast<int64_t>(numAzVirt) * m_params.coarseElVirt,
            "elementSnapshots");
    checkHW(rangeDopplerPlanesReqs, NVCV_DATA_TYPE_2F32, maxDet, static_cast<int64_t>(numChirps) * numRx,
            "rangeDopplerPlanes");

    for (int32_t t = 0; t < kNumTensors; ++t)
    {
        if ((t == kWaz || t == kWel) && !m_params.useCustomTwiddles)
        {
            continue;
        }
        CheckDenselyPacked(*reqs[t], kTensorNames[t]);
        m_reqs[t] = *reqs[t];
    }
}

CaponROIPreprocess::CaponROIPreprocess(
    const NVCVTensorRequirements *wazReqs, const NVCVTensorRequirements *welReqs,
    const NVCVTensorRequirements *detectionCountReqs, const NVCVTensorRequirements *detDopplersReqs,
    const NVCVTensorRequirements *elementSnapshotsReqs, const NVCVTensorRequirements *rangeDopplerPlanesReqs,
    const NVCVTensorRequirements *steeringLutReqs, const NVCVTensorRequirements *snapshotsROIReqs,
    const NVCVTensorRequirements *steeringVectorsROIReqs, const PVACaponROIPreprocessParams *params)
    : m_wazOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_elemOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_lutOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_snapOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_snapOp1(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_snapOp2(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_snapOp3(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_svOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_planeOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_detDopOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
    , m_params{*params}
{
    ValidateParams();

    const int32_t numTx     = m_params.numTx;
    const int32_t numRx     = m_params.numRx;
    const int32_t numAzVirt = m_params.numAzVirt;
    const int32_t numElVirt = m_params.numElVirt;
    const int32_t beamAz    = m_params.beamAz;
    const int32_t beamEl    = m_params.beamEl;
    const int32_t svDim     = m_params.svDim;
    const int32_t H         = m_params.numDopplerBinsCovMat;
    const int32_t numChirps = m_params.numChirps;
    const int32_t maxDet    = m_params.maxDet;
    const int32_t roiAz     = m_params.roiAz;
    const int32_t roiEl     = m_params.roiEl;
    const int32_t numAzBins = m_params.numAzBins;
    const int32_t numElBins = m_params.numElBins;
    const int32_t nvc       = numAzVirt * numElVirt;
    /* Steering LUT: one block per ROI start position, roiPos = roiElStart*numRoiAzPos + roiAzStart. */
    const int32_t numRoiAzPos = beamAz - roiAz + 1;
    const int32_t numRoiElPos = beamEl - roiEl + 1;
    const int32_t numRoiPos   = numRoiAzPos * numRoiElPos;
    const int32_t svBlockCplx = svDim * numElBins * numAzBins; /* complex entries per block */

    FFTFactorization fftFactorization{priv::FactorizeFFTSize(64, false, false, CRP_FFT_GROUPS)};
    std::vector<int32_t> fftTwiddleQ30(128); /* radix-4 intra-stage twiddles, int32 Q30 interleaved re/im */
    if (m_params.coarseDoAUseFft)
    {
        if (!fftFactorization.isValid)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "FFT-64 factorization for the coarse DoA is invalid");
        }
        std::vector<int32_t> twiddleInterleaved(2 * 64);
        for (int32_t i = 0; i < 64; i++)
        {
            const double angle            = -2.0 * 3.14159265358979323846 * (double)i / 64.0;
            twiddleInterleaved[2 * i]     = (int32_t)std::lround(std::cos(angle) * (double)(1UL << TWIDDLE_QBITS));
            twiddleInterleaved[2 * i + 1] = (int32_t)std::lround(std::sin(angle) * (double)(1UL << TWIDDLE_QBITS));
        }
        priv::generateComplexFFTTwiddleIntraStage(fftTwiddleQ30.data(), twiddleInterleaved, fftFactorization,
                                                  RADIX4_NUM);
    }

    const NVCVTensorRequirements *const reqs[kNumTensors] = {wazReqs,
                                                             welReqs,
                                                             detectionCountReqs,
                                                             detDopplersReqs,
                                                             elementSnapshotsReqs,
                                                             rangeDopplerPlanesReqs,
                                                             steeringLutReqs,
                                                             snapshotsROIReqs,
                                                             steeringVectorsROIReqs};
    ValidateRequirements(reqs);

    m_useFftCoarseDoA  = UseFftCoarseDoA(m_params);
    m_useDotProjection = UseDotProductProjection(m_params);
    m_kernel           = m_useFftCoarseDoA                                 ? Kernel::kFft
                         : (m_useDotProjection && m_params.numElVirt == 1) ? Kernel::kDft
                                                                           : Kernel::kDftGeneral;
    m_vdFold           = CanFoldDdmDecode(m_params, m_useDotProjection);
    m_exec             = GetExec(m_kernel);
    auto prog          = CmdProgram::Create(*m_exec.get());

    prog["GP"].set(&m_params, 1);
    prog["crpNumDet"]     = maxDet; /* set per Submit from detectionCount */
    prog["crpVdFold"]     = m_vdFold ? 1 : 0;
    prog["crpUseDotProj"] = m_useDotProjection ? 1 : 0;

    const std::array<int32_t, CRP_MAX_TX> txBase = VdTxBase(m_params);
    prog["crpVdTxBase"].set(txBase.data(), CRP_MAX_TX);
    int16_t fbaRot[CRP_MAX_BEAM_AZ * 4]   = {};
    int16_t fbaRotEl[CRP_MAX_BEAM_EL * 4] = {};
    BuildFbaRotation(numAzVirt, beamAz, fbaRot);
    BuildFbaRotation(numElVirt, beamEl, fbaRotEl);
    prog["crpFbaRot"].set(fbaRot, CRP_MAX_BEAM_AZ * 4);
    prog["crpFbaRotEl"].set(fbaRotEl, CRP_MAX_BEAM_EL * 4);
    if (m_useFftCoarseDoA)
    {
        prog["crpFftFactorization"].set(&fftFactorization, 1);
        prog["crpFftTwiddleQ30"].set(fftTwiddleQ30.data(), static_cast<int32_t>(fftTwiddleQ30.size()));
    }

    float *wazBuf   = prog["wazBuf"].ptr<float>();
    float *elemBuf  = prog["elemBuf"].ptr<float>();
    int32_t *outBuf = prog["outBuf"].ptr<int32_t>();

    float *planeBuf    = prog["planeBuf"].ptr<float>();
    int32_t *detDopBuf = prog["detDopBuf"].ptr<int32_t>();

    /* The output DMA does the [svDim][H] -> [H][svDim] transpose; it needs a 1D grid of at most
     * CRP_OUT_BATCH tiles, so a larger maxDet is split into batches. */
    const bool batchTranspose = (maxDet > CRP_OUT_BATCH);

    /* Row lengths in scalar lanes (complex = 2 lanes). */
    const int32_t wazLen       = beamAz * numAzVirt * 2;
    const int32_t detDopLen    = maxDet;
    const int32_t elemRowLen   = numAzVirt * m_params.coarseElVirt * 2; /* coarse-DoA grid, not the projection array */
    const int32_t planeRowLen  = numChirps * numRx * 2;
    const int32_t svRowLen     = svBlockCplx * 2; /* int32 lanes per steering block */
    const int32_t svBlockBytes = svRowLen * static_cast<int32_t>(sizeof(int32_t));

    const int32_t tileCols = DetTileColumns(maxDet); /* ValidateParams checked it is nonzero */
    const int32_t tileRows = maxDet / tileCols;
    int32_t phase          = 0;

    /* A throwing constructor never runs the destructor, so free the scratch buffers if anything from
     * here on throws. */
    struct ScratchGuard
    {
        CaponROIPreprocess *op;
        ~ScratchGuard()
        {
            if (op != nullptr)
                op->FreeScratch();
        }
    } scratchGuard{this};

    /* Allocate before the dataflows below: .src() records the pointer when it is called. */
    m_waz16Scratch = static_cast<int16_t *>(
        mem::Alloc(sizeof(int16_t) * static_cast<size_t>(beamAz) * static_cast<size_t>(numAzVirt) * 2));
    if (!m_useFftCoarseDoA) /* the transposed table only feeds the direct-DFT kernel's stage-1 DFT */
    {
        m_wazT16Scratch = static_cast<int16_t *>(
            mem::Alloc(sizeof(int16_t) * static_cast<size_t>(beamAz) * static_cast<size_t>(numAzVirt) * 2));
    }
    m_wel16Scratch = static_cast<int16_t *>(
        mem::Alloc(sizeof(int16_t) * static_cast<size_t>(beamEl) * static_cast<size_t>(m_params.coarseElVirt) * 2));

    if (!m_params.useCustomTwiddles)
    {
        /* DFT twiddles: the fp32 az table stays resident for the coarse-DoA refine; both are quantized
         * to the device's int16 tables once, here. */
        constexpr double kPi = 3.14159265358979323846;
        auto dft             = [&](float *t, int32_t beam, int32_t n) {
            for (int32_t k = 0; k < beam; ++k)
            {
                for (int32_t a = 0; a < n; ++a)
                {
                    const double th        = -2.0 * kPi * ((k * a) % beam) / beam; /* reduced: exact phase */
                    t[(k * n + a) * 2 + 0] = static_cast<float>(std::cos(th));
                    t[(k * n + a) * 2 + 1] = static_cast<float>(std::sin(th));
                }
            }
        };
        m_wazGenScratch = static_cast<float *>(mem::Alloc(sizeof(float) * static_cast<size_t>(wazLen)));
        float *wazHost  = static_cast<float *>(mem::GetHostPointer(m_wazGenScratch));
        dft(wazHost, beamAz, numAzVirt);
        std::vector<float> welHost(static_cast<size_t>(beamEl) * m_params.coarseElVirt * 2);
        dft(welHost.data(), beamEl, m_params.coarseElVirt);
        BuildHostTwiddle16(wazHost, welHost.data());
        m_wazOp.setBase(m_wazGenScratch);
    }

    /* Host-quantized int16 Q15 twiddles -> device VMEM: one SQDF of static transfers (the transposed
     * coarse-DoA halves only for the direct-DFT kernel). The addresses never vary, so the device just
     * triggers and syncs once before the detection loop. */
    int16_t *waz16_v        = prog["waz16Buf"].ptr<int16_t>();
    const size_t waz16Bytes = static_cast<size_t>(beamAz) * numAzVirt * 2 * sizeof(int16_t);

    /* Phase 0: the one-shot inputs, independent transfers into separate buffers, so they run together before
     * the detection loop: detDopplers, the fp32 az table and the int16 twiddles. */

    /* Per-detection Doppler bins (one-shot: the whole [maxDet] array in one tile); detDopBuf keeps them for the
     * detection loop. */
    RasterDataFlow &detDopDF = prog.addDataFlowHead<RasterDataFlow>(phase);
    detDopDF.handler(prog["detDopRDFHdl"])
        .src(m_detDopOp, detDopLen, 1, detDopLen)
        .tileBuffer(detDopBuf)
        .roi(0, 0, detDopLen, 1)
        .tile(detDopLen, 1);

    /* The fp32 az table feeds only the azimuth-only coarse-DoA refine; a larger table is only valid with
     * a coarse-DoA grid, which does not read it (checked above). */
    if (wazLen <= BUF_SZ_WAZ)
    {
        RasterDataFlow &wazDF = prog.addDataFlowHead<RasterDataFlow>(phase);
        wazDF.handler(prog["wazRDFHdl"])
            .src(m_wazOp, wazLen, 1, wazLen)
            .tileBuffer(wazBuf)
            .roi(0, 0, wazLen, 1)
            .tile(wazLen, 1);
    }

    SequenceDataFlow &twidDF = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(prog["twiddle16Hdl"]);
    twidDF.addTransfer()
        .src(m_waz16Scratch, waz16Bytes)
        .dst(waz16_v, waz16Bytes)
        .tile(waz16Bytes, 1)
        .mode(TransferModeType::CONTINUOUS);
    if (!m_useFftCoarseDoA)
    {
        int16_t *wazT_v        = prog["wazTBuf"].ptr<int16_t>();
        const size_t wazTBytes = static_cast<size_t>(beamAz) * numAzVirt * 2 * sizeof(int16_t);
        twidDF.addTransfer()
            .src(m_wazT16Scratch, wazTBytes)
            .dst(wazT_v, wazTBytes)
            .tile(wazTBytes, 1)
            .mode(TransferModeType::CONTINUOUS);
    }
    /* El twiddles: int16 only. */
    int16_t *wel16_v        = prog["wel16Buf"].ptr<int16_t>();
    const size_t wel16Bytes = static_cast<size_t>(beamEl) * m_params.coarseElVirt * 2 * sizeof(int16_t);
    twidDF.addTransfer()
        .src(m_wel16Scratch, wel16Bytes)
        .dst(wel16_v, wel16Bytes)
        .tile(wel16Bytes, 1)
        .mode(TransferModeType::CONTINUOUS);
    phase++;

    /* Per-detection element snapshot (coarse DoA): one tile (row) per detection, over the 2D grid. */
    RasterDataFlow &elemDF = prog.addDataFlowHead<RasterDataFlow>(phase);
    elemDF.handler(prog["elemRDFHdl"])
        .src(m_elemOp, tileCols * elemRowLen, tileRows, tileCols * elemRowLen)
        .tileBuffer(elemBuf)
        .roi(0, 0, tileCols * elemRowLen, tileRows)
        .tile(elemRowLen, 1);

    /* Per-detection range-Doppler plane row, decoded into the VD matrix on the device. */
    RasterDataFlow &planeDF = prog.addDataFlowHead<RasterDataFlow>(phase);
    planeDF.handler(prog["planeRDFHdl"])
        .src(m_planeOp, tileCols * planeRowLen, tileRows, tileCols * planeRowLen)
        .tileBuffer(planeBuf)
        .roi(0, 0, tileCols * planeRowLen, tileRows)
        .tile(planeRowLen, 1);

    /* Per-detection snapshotsROI output. */
    if (!batchTranspose)
    {
        /* DRAM [maxDet, H, svDim] complex is the image [maxDet*H][svDim*2]; each detection's
         * [H][svDim*2] tile is transposed from VMEM [svDim][H] by TRANS_MODE_2. */
        RasterDataFlow &outDF = prog.addDataFlowHead<RasterDataFlow>(phase);
        outDF.handler(prog["outRDFHdl"])
            .tileBuffer(outBuf)
            .dst(m_snapOp, svDim * 2, maxDet * H, svDim * 2)
            .tile(svDim * 2, H)
            .transpose(TRANS_MODE_2)
            .scanOrder(cupva::COLUMN_MAJOR);
    }
    else
    {
        /* One TRANS_MODE_2 RDF per batch, each with its own VMEM buffer, so the device keeps them all
         * open. */
        static_assert(CRP_MAX_DET / CRP_OUT_BATCH == 4, "one out RDF handle per batch");
        const int32_t nBatch                             = (maxDet + CRP_OUT_BATCH - 1) / CRP_OUT_BATCH;
        const char *const hdlNames[4]                    = {"outRDFHdl", "outRDFHdl1", "outRDFHdl2", "outRDFHdl3"};
        const char *const bufNames[4]                    = {"outBuf", "outBuf1", "outBuf2", "outBuf3"};
        cupva::OffsetPointer<int32_t> *const batchOps[4] = {&m_snapOp, &m_snapOp1, &m_snapOp2, &m_snapOp3};
        for (int32_t bi = 0; bi < nBatch; bi++)
        {
            const int32_t detLo    = bi * CRP_OUT_BATCH;
            const int32_t batchLen = (detLo + CRP_OUT_BATCH < maxDet) ? CRP_OUT_BATCH : (maxDet - detLo);
            RasterDataFlow &bDF    = prog.addDataFlowHead<RasterDataFlow>(phase);
            bDF.handler(prog[hdlNames[bi]])
                .tileBuffer(prog[bufNames[bi]].ptr<int32_t>())
                .dst(*batchOps[bi], svDim * 2, batchLen * H, svDim * 2)
                .tile(svDim * 2, H)
                .transpose(TRANS_MODE_2)
                .scanOrder(cupva::COLUMN_MAJOR);
        }
    }

    /* Steering block: one device-addressed DRAM -> DRAM transfer per detection. The device sets src (the LUT
     * block of the computed ROI) and dst (steeringVectorsROI row) each time; the host provides placeholders. Rows
     * at or beyond the detection count are not written. */
    SequenceDataFlow &lutDF = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(prog["lutGatherHdl"]);
    lutDF.addTransfer()
        .src(m_lutOp, svBlockBytes)
        .dst(m_svOp, svBlockBytes)
        .tile(svBlockBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
    scratchGuard.op = nullptr;
}

CaponROIPreprocess::~CaponROIPreprocess()
{
    CupvaCmdDestroy(&m_prog);
    FreeScratch();
}

void CaponROIPreprocess::FreeScratch() noexcept
{
    if (m_wazGenScratch != nullptr)
    {
        mem::Free(m_wazGenScratch);
        m_wazGenScratch = nullptr;
    }
    if (m_waz16Scratch != nullptr)
    {
        mem::Free(m_waz16Scratch);
        m_waz16Scratch = nullptr;
    }
    if (m_wazT16Scratch != nullptr)
    {
        mem::Free(m_wazT16Scratch);
        m_wazT16Scratch = nullptr;
    }
    if (m_wel16Scratch != nullptr)
    {
        mem::Free(m_wel16Scratch);
        m_wel16Scratch = nullptr;
    }
}

/* Quantize the twiddles to the int16 Q15 layouts the device consumes. */
void CaponROIPreprocess::BuildHostTwiddle16(const float *wazHost, const float *welHost)
{
    int16_t *waz16 = static_cast<int16_t *>(mem::GetHostPointer(m_waz16Scratch));
    int16_t *wel16 = static_cast<int16_t *>(mem::GetHostPointer(m_wel16Scratch));

    const int32_t beamAz    = m_params.beamAz;
    const int32_t numAzVirt = m_params.numAzVirt;

    /* Checked before any float-to-int conversion: every entry must be finite with complex magnitude at most 1
     * (1e-5 of slack for float cos/sin rounding; the generated tables are unit-modulus). That keeps the
     * coarse-DoA rank power, accumulated in 32 bits, within the elementSnapshots bound in the header, which
     * entries of magnitude up to sqrt(2) would lower from about 181 to about 128. */
    auto checkTable = [](const float *t, int32_t entries) {
        for (int32_t i = 0; i < entries; i++)
        {
            const float re = t[2 * i], im = t[2 * i + 1];
            if (!(re * re + im * im <= 1.00001f))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "waz and wel entries must be finite with complex magnitude at most 1");
            }
        }
    };
    checkTable(wazHost, beamAz * numAzVirt);
    checkTable(welHost, m_params.beamEl * m_params.coarseElVirt);

    auto qToShort = [](float v) -> int16_t {
        float q = v * 32768.0f; /* Q15 */
        q       = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
        if (q > 32767.0f)
            q = 32767.0f;
        if (q < -32768.0f)
            q = -32768.0f;
        return static_cast<int16_t>(static_cast<int32_t>(q));
    };

    /* Projection layout: [ka][a] complex interleaved, as wazBuf. With m_vdFold the device VD holds conj(plane)
     * without the DDM decode, so the decode moves here: W[k][a] * conj(w_tx) with tx = a % numTx (vc = a). */
    for (int32_t i = 0; i < beamAz * numAzVirt; i++)
    {
        float re = wazHost[2 * i], im = wazHost[2 * i + 1];
        if (m_vdFold)
        {
            const int32_t tx = (i % numAzVirt) % m_params.numTx;
            const float wr = m_params.ddmaWeightRe[tx], wi = m_params.ddmaWeightIm[tx];
            const float fr = re * wr + im * wi;
            im             = im * wr - re * wi;
            re             = fr;
        }
        waz16[2 * i]     = qToShort(re);
        waz16[2 * i + 1] = qToShort(im);
    }
    /* El twiddles: same straight quantize, [ke][e] complex interleaved. */
    for (int32_t i = 0; i < m_params.beamEl * m_params.coarseElVirt * 2; i++)
    {
        wel16[i] = qToShort(welHost[i]);
    }
    if (m_wazT16Scratch == nullptr) /* FFT kernel: no direct-DFT stage-1 */
    {
        return;
    }
    /* Coarse-DoA layout: transposed, complex interleaved, wazT16[a][ka] = (re, im), so stage-1 loads the
     * (re, im) of 16 consecutive beam bins for a fixed element with one deinterleaving load. */
    int16_t *wazT16 = static_cast<int16_t *>(mem::GetHostPointer(m_wazT16Scratch));
    for (int32_t ka = 0; ka < beamAz; ka++)
    {
        const float *wrow = wazHost + static_cast<ptrdiff_t>(ka) * numAzVirt * 2;
        for (int32_t a = 0; a < numAzVirt; a++)
        {
            wazT16[(a * beamAz + ka) * 2 + 0] = qToShort(wrow[a * 2 + 0]);
            wazT16[(a * beamAz + ka) * 2 + 1] = qToShort(wrow[a * 2 + 1]);
        }
    }
}

template<typename S>
void CaponROIPreprocess::operator()(S stream, NVCVTensorHandle wazHandle, NVCVTensorHandle welHandle,
                                    NVCVTensorHandle detectionCountHandle, NVCVTensorHandle detDopplersHandle,
                                    NVCVTensorHandle elementSnapshotsHandle, NVCVTensorHandle rangeDopplerPlanesHandle,
                                    NVCVTensorHandle steeringLutHandle, NVCVTensorHandle snapshotsROIHandle,
                                    NVCVTensorHandle steeringVectorsROIHandle)
{
    const NVCVTensorData countData  = ExportMatching(detectionCountHandle, m_reqs[kCount], kTensorNames[kCount]);
    const NVCVTensorData detDopData = ExportMatching(detDopplersHandle, m_reqs[kDetDop], kTensorNames[kDetDop]);
    const NVCVTensorData elemData   = ExportMatching(elementSnapshotsHandle, m_reqs[kElem], kTensorNames[kElem]);
    const NVCVTensorData planeData  = ExportMatching(rangeDopplerPlanesHandle, m_reqs[kPlane], kTensorNames[kPlane]);
    const NVCVTensorData lutData    = ExportMatching(steeringLutHandle, m_reqs[kLut], kTensorNames[kLut]);
    const NVCVTensorData snapData   = ExportMatching(snapshotsROIHandle, m_reqs[kSnap], kTensorNames[kSnap]);
    const NVCVTensorData svData     = ExportMatching(steeringVectorsROIHandle, m_reqs[kSv], kTensorNames[kSv]);

    auto *lutBase = static_cast<int32_t *>(static_cast<void *>(lutData.buffer.strided.basePtr));
    if (m_params.useCustomTwiddles)
    {
        /* Caller-supplied tables: stream the fp32 az table from the tensor and requantize both. */
        const NVCVTensorData wazData = ExportMatching(wazHandle, m_reqs[kWaz], kTensorNames[kWaz]);
        const NVCVTensorData welData = ExportMatching(welHandle, m_reqs[kWel], kTensorNames[kWel]);
        m_wazOp.setBase(static_cast<float *>(static_cast<void *>(wazData.buffer.strided.basePtr)));
        BuildHostTwiddle16(HostPointer<float>(wazData, kTensorNames[kWaz]),
                           HostPointer<float>(welData, kTensorNames[kWel]));
    }
    m_elemOp.setBase(static_cast<float *>(static_cast<void *>(elemData.buffer.strided.basePtr)));
    m_planeOp.setBase(static_cast<float *>(static_cast<void *>(planeData.buffer.strided.basePtr)));
    m_detDopOp.setBase(static_cast<int32_t *>(static_cast<void *>(detDopData.buffer.strided.basePtr)));
    m_lutOp.setBase(lutBase);
    int32_t *const snapBase = static_cast<int32_t *>(static_cast<void *>(snapData.buffer.strided.basePtr));
    m_snapOp.setBase(snapBase);
    m_svOp.setBase(static_cast<int32_t *>(static_cast<void *>(svData.buffer.strided.basePtr)));
    /* Point each batch's snapshotsROI RDF at its detection slice. */
    if (m_params.maxDet > CRP_OUT_BATCH)
    {
        const int32_t batchStrideInts = CRP_OUT_BATCH * m_params.numDopplerBinsCovMat * m_params.svDim * 2;
        const int32_t nBatch          = (m_params.maxDet + CRP_OUT_BATCH - 1) / CRP_OUT_BATCH;
        cupva::OffsetPointer<int32_t> *const batchOps[3] = {&m_snapOp1, &m_snapOp2, &m_snapOp3};
        /* Only batches that have an RDF: pointing an unused one past the tensor fails setBase. */
        for (int32_t bi = 1; bi < nBatch; bi++) batchOps[bi - 1]->setBase(snapBase + bi * batchStrideInts);
    }

    /* Only the first detectionCount detections are processed; the other output rows are unspecified. */
    const int32_t numDet = *HostPointer<int32_t>(countData, kTensorNames[kCount]);
    if (numDet < 0 || numDet > m_params.maxDet)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "detectionCount must be in [0, maxDet = %d]; got %d", m_params.maxDet, numDet);
    }
    cupvaParameter_t numDetParam;
    if (CupvaCmdProgramGetParameter(&m_prog, &numDetParam, "crpNumDet") != CUPVA_ERROR_NONE ||
        CupvaParameterSetValueScalar(&numDetParam, &numDet, sizeof(numDet)) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to set the detection count");
    }

    /* The device computes the ROI, so it re-addresses the LUT gather at runtime from this base. */
    cupvaParameter_t lutExtPtr;
    if (CupvaCmdProgramGetParameter(&m_prog, &lutExtPtr, "lutExtPtr") != CUPVA_ERROR_NONE ||
        CupvaParameterSetValuePointer(&lutExtPtr, lutBase) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to set steering LUT external pointer");
    }
    cupvaParameter_t svExtPtr;
    if (CupvaCmdProgramGetParameter(&m_prog, &svExtPtr, "svExtPtr") != CUPVA_ERROR_NONE ||
        CupvaParameterSetValuePointer(&svExtPtr, svData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to set steeringVectorsROI external pointer");
    }

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for Capon ROI preprocess operator");
    }

    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "PVA Cmd Submit failed");
    }
}

template void CaponROIPreprocess::operator()(cupvaStream_t, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                             NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                             NVCVTensorHandle, NVCVTensorHandle);
template void CaponROIPreprocess::operator()(cudaStream_t, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                             NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                             NVCVTensorHandle, NVCVTensorHandle);

} // namespace pva_solutions::priv

/* ========================================================================= */

extern "C" {

NVCVStatus pvaCaponROIPreprocessCreate(
    NVCVOperatorHandle *handle, NVCVTensorRequirements const *const wazReqs,
    NVCVTensorRequirements const *const welReqs, NVCVTensorRequirements const *const detectionCountReqs,
    NVCVTensorRequirements const *const detDopplersReqs, NVCVTensorRequirements const *const elementSnapshotsReqs,
    NVCVTensorRequirements const *const rangeDopplerPlanesReqs, NVCVTensorRequirements const *const steeringLutReqs,
    NVCVTensorRequirements const *const snapshotsROIReqs, NVCVTensorRequirements const *const steeringVectorsROIReqs,
    PVACaponROIPreprocessParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to handle must not be NULL");
        if (params == nullptr)
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "params must not be NULL");
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::CaponROIPreprocess(
            wazReqs, welReqs, detectionCountReqs, detDopplersReqs, elementSnapshotsReqs, rangeDopplerPlanesReqs,
            steeringLutReqs, snapshotsROIReqs, steeringVectorsROIReqs, params));
    });
}

} // extern "C"

template<typename StreamType>
NVCVStatus CaponROIPreprocessSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle waz,
                                        NVCVTensorHandle wel, NVCVTensorHandle detectionCount,
                                        NVCVTensorHandle detDopplers, NVCVTensorHandle elementSnapshots,
                                        NVCVTensorHandle rangeDopplerPlanes, NVCVTensorHandle steeringLut,
                                        NVCVTensorHandle snapshotsROI, NVCVTensorHandle steeringVectorsROI)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Operator handle must not be NULL");
        auto *op = reinterpret_cast<pva_solutions::priv::CaponROIPreprocess *>(handle);
        (*op)(stream, waz, wel, detectionCount, detDopplers, elementSnapshots, rangeDopplerPlanes, steeringLut,
              snapshotsROI, steeringVectorsROI);
    });
}

// Explicit instantiations so the cuPVA/CUDA C-callable submit wrappers generated in
// PvaSubmitDefinitions.cpp (pvaCaponROIPreprocessSubmit__cupva / __cuda) resolve at link time
// (an implicit instantiation from the overloads below can be discarded by the linker).
template NVCVStatus CaponROIPreprocessSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                                NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                                NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus CaponROIPreprocessSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                               NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                               NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                               NVCVTensorHandle, NVCVTensorHandle);

NVCVStatus pvaCaponROIPreprocessSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const waz,
                                       NVCVTensorHandle const wel, NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const detDopplers, NVCVTensorHandle const elementSnapshots,
                                       NVCVTensorHandle const rangeDopplerPlanes, NVCVTensorHandle const steeringLut,
                                       NVCVTensorHandle snapshotsROI, NVCVTensorHandle steeringVectorsROI)
{
    return CaponROIPreprocessSubmitImpl(handle, stream, waz, wel, detectionCount, detDopplers, elementSnapshots,
                                        rangeDopplerPlanes, steeringLut, snapshotsROI, steeringVectorsROI);
}

NVCVStatus pvaCaponROIPreprocessSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const waz,
                                       NVCVTensorHandle const wel, NVCVTensorHandle const detectionCount,
                                       NVCVTensorHandle const detDopplers, NVCVTensorHandle const elementSnapshots,
                                       NVCVTensorHandle const rangeDopplerPlanes, NVCVTensorHandle const steeringLut,
                                       NVCVTensorHandle snapshotsROI, NVCVTensorHandle steeringVectorsROI)
{
    return CaponROIPreprocessSubmitImpl(handle, stream, waz, wel, detectionCount, detDopplers, elementSnapshots,
                                        rangeDopplerPlanes, steeringLut, snapshotsROI, steeringVectorsROI);
}
