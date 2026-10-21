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

#include "radar_pipeline_config.hpp"
#include "radar_pipeline_context.hpp"
#include "radar_pipeline_iface.hpp"
#include "radar_pipeline_pva.hpp"
#include "radar_pipeline_reference.hpp"
#include "radar_pipeline_tensors.hpp"

#include <TensorUtils.hpp>

#include <ErrorCheckMacros.h>
#include <OpDoaAngleFFT.h>
#include <OpDoaBartlettBeamforming.h>
#include <OpDopplerFFT.h>
#include <OpNci.h>
#include <OpRadarCFAR.h>
#include <OpRangeFFT.h>
#include <OpSnapshotExtraction.h>
#include <PvaOperatorTypes.h>
#include <cupva_host.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace pva_solutions;

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

/* Capon operator settings the pipeline fixes (not part of RadarPipelineCaponConfig): the steering grid the
 * Capon operator supports, and the Q formats of the ROI tensors. Q20 fills the S32 covariance kernel; the
 * cube is max-abs normalized, so the ROI snapshots stay below about 2^27 at Q20. */
constexpr int32_t kCaponNumAzBins     = 16;
constexpr int32_t kCaponNumElBins     = 8;
constexpr int32_t kCaponInputQBits    = 20;
constexpr int32_t kCaponSteeringQBits = 12;

/* Fills each Capon config field left at 0 from radarPipelineCaponConfigDefaults(); negative values stay, for
 * validateCaponConfig() to reject. */
RadarPipelineCaponConfig withCaponDefaults(RadarPipelineCaponConfig c)
{
    const RadarPipelineCaponConfig d = radarPipelineCaponConfigDefaults();
    auto def                         = [](auto &v, auto dv) {
        if (v == 0)
            v = dv;
    };
    def(c.roiAzSize, d.roiAzSize);
    def(c.roiElSize, d.roiElSize);
    def(c.numDopplerBinsCovMat, d.numDopplerBinsCovMat);
    def(c.dopplerNeighborhood, d.dopplerNeighborhood);
    def(c.adpDiagConst, d.adpDiagConst);
    def(c.noiseFloorRatio, d.noiseFloorRatio);
    def(c.maxRoi, d.maxRoi);
    return c;
}

/* Derives the RX (az, el) split from the RX positions: they must place the numRx elements at distinct
 * positions on a full nAz x nEl grid, in the channel order rx = el*nAz + az. Returns false otherwise (for
 * example when the positions are left at zero). */
bool deriveRxSplit(const RadarPipelineGP &gp, int32_t numRx, int32_t *nAzOut, int32_t *nElOut)
{
    if (gp.numRx != numRx || numRx > RADAR_PIPELINE_MAX_NUM_RX)
        return false;
    std::vector<float> az, el;
    auto rankOf = [](std::vector<float> &v, float x) {
        for (size_t k = 0; k < v.size(); ++k)
            if (std::fabs(v[k] - x) < 1e-4f)
                return static_cast<int32_t>(k);
        v.push_back(x);
        return static_cast<int32_t>(v.size() - 1);
    };
    for (int32_t r = 0; r < numRx; ++r)
    {
        rankOf(az, gp.rxPositionsAzimuth[r]);
        rankOf(el, gp.rxPositionsElevation[r]);
    }
    std::sort(az.begin(), az.end());
    std::sort(el.begin(), el.end());
    const int32_t nAz = static_cast<int32_t>(az.size()), nEl = static_cast<int32_t>(el.size());
    if (nAz * nEl != numRx)
        return false;
    for (int32_t r = 0; r < numRx; ++r)
    {
        const int32_t a = rankOf(az, gp.rxPositionsAzimuth[r]), e = rankOf(el, gp.rxPositionsElevation[r]);
        if (r != e * nAz + a)
            return false;
    }
    *nAzOut = nAz;
    *nElOut = nEl;
    return true;
}

int32_t validateCaponConfig(const RadarPipelineCaponConfig &c, const RadarPipelineGP &gp);

/* The caller's Capon config with its defaults and RX split filled in, validated; returns 0 or a
 * RadarPipelineError. Runs before init() allocates anything. */
int32_t prepareCaponConfig(const RadarPipelineCaponConfig &in, const RadarPipelineGP &gp, RadarPipelineCaponConfig *out)
{
    RadarPipelineCaponConfig c = withCaponDefaults(in);
    // RX split: as given, or derived from the RX positions (which must resolve a full grid).
    if (c.rxAzimuthElements == 0 && c.rxElevationElements == 0 &&
        !deriveRxSplit(gp, gp.NofRx, &c.rxAzimuthElements, &c.rxElevationElements))
    {
        c.rxAzimuthElements = c.rxElevationElements = 0; // rejected by validateCaponConfig()
    }
    *out = c;
    return validateCaponConfig(c, gp);
}

/* Checks a defaulted Capon config (with its RX split) against what the pipeline supports; returns 0 or a
 * RadarPipelineError. The Capon path gathers the range-Doppler cube as [range][chirp][rx], the layout the
 * pipeline uses only for 8 TX x 8 RX. The operators' own limits are checked when they are created
 * (radarCreatePvaWorkloads reports a rejection as RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG). */
int32_t validateCaponConfig(const RadarPipelineCaponConfig &c, const RadarPipelineGP &gp)
{
    const char *why = nullptr;
    const int32_t H = c.numDopplerBinsCovMat;
    if (CUPVA_LLVM_SUPPORTED != 1)
    {
        std::fprintf(stderr, "radar pipeline: Capon DoA needs an LLVM-capable PVA SDK\n");
        return RADAR_PIPELINE_CREATE_ERR_DOA_METHOD;
    }
    if (gp.NofTx != 8 || gp.NofRx != 8)
        why = "Capon DoA needs 8 TX and 8 RX";
    else if (gp.ddm_doppler_offsets_count < gp.NofTx)
    {
        std::fprintf(stderr, "radar pipeline: Capon DoA needs one DDM offset per TX; got %d for %d TX\n",
                     gp.ddm_doppler_offsets_count, gp.NofTx);
        return RADAR_PIPELINE_CREATE_ERR_DDM_OFFSETS;
    }
    else if (c.roiAzSize < 0 || c.roiElSize < 0 || c.numDopplerBinsCovMat < 0 || c.dopplerNeighborhood < 0 ||
             c.maxRoi < 0 || c.rxAzimuthElements < 0 || c.rxElevationElements < 0 ||
             !(c.adpDiagConst > 0.0f && c.adpDiagConst < INFINITY) ||
             !(c.noiseFloorRatio > 0.0f && c.noiseFloorRatio < INFINITY))
        why = "Capon config values must be 0 (default) or positive, and finite";
    else if (static_cast<int64_t>(c.roiAzSize) * c.roiElSize != 16)
        why = "Capon DoA needs roiAzSize*roiElSize == 16";
    else if (H < 16 || H > 64 || H % 8 != 0 || H == 40)
        why = "numDopplerBinsCovMat must be 16, 24, 32, 48, 56 or 64";
    /* 64-bit: doubling the window of a dopplerNeighborhood near INT_MAX must not overflow. */
    else if (c.dopplerNeighborhood < 2 ||
             2 * static_cast<int64_t>(c.dopplerNeighborhood / 2) * (c.enableFBA ? 2 : 1) > H)
        why = "dopplerNeighborhood must be >= 2 with 2*floor(N/2) rows (doubled by FBA) <= numDopplerBinsCovMat";
    else if (c.maxRoi > 1024)
        why = "maxRoi must be <= 1024";
    else if (c.rxAzimuthElements <= 0 || c.rxElevationElements <= 0 ||
             static_cast<int64_t>(c.rxAzimuthElements) * c.rxElevationElements != gp.NofRx)
        why =
            "rxAzimuthElements*rxElevationElements must equal NofRx (set both, or give RX positions on a full "
            "az x el grid in the order rx = el*rxAzimuthElements + az)";
    if (why == nullptr)
        return 0;
    std::fprintf(stderr, "radar pipeline: %s\n", why);
    return RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG;
}

// Read int32 from a 1-element PVA tensor; returns true on success.
bool readInt32FromTensor(NVCVTensorHandle handle, int32_t *out)
{
    if (!handle || !out)
        return false;
    NVCVTensorData data;
    if (nvcvTensorExportData(handle, &data) != NVCV_SUCCESS)
        return false;
    int32_t *ptr = nullptr;
    if (CupvaMemGetHostPointer((void **)&ptr, (void *)data.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
        return false;
    *out = *ptr;
    return true;
}

} // namespace

// ============================================================================
// RadarPipeline8t8r — concrete IRadarPipeline for the 8T8R variant
// ============================================================================

class RadarPipeline8t8r final : public IRadarPipeline
{
public:
    RadarPipeline8t8r() = default;

    ~RadarPipeline8t8r() override
    {
        delete ctx_;
    }

    int32_t init(const RadarPipelineConfig *config, RadarPipelineAdcDimensions *dims_out) override
    {
        if (!config || !config->gp)
            return RADAR_PIPELINE_CREATE_ERR_NULL_ARGS;

        RadarPipelineGP *gp = config->gp;

        if (config->doa_params.doa_method != RADAR_PIPELINE_DOA_ANGLE_FFT &&
            config->doa_params.doa_method != RADAR_PIPELINE_DOA_BARTLETT &&
            config->doa_params.doa_method != RADAR_PIPELINE_DOA_CAPON_ROI)
        {
            return RADAR_PIPELINE_CREATE_ERR_DOA_METHOD;
        }

        // Validate the Capon config before allocating anything, so a rejection leaks nothing.
        RadarPipelineCaponConfig capon = {};
        if (config->doa_params.doa_method == RADAR_PIPELINE_DOA_CAPON_ROI)
        {
            const int32_t caponStatus = prepareCaponConfig(config->doa_params.capon_config, *gp, &capon);
            if (caponStatus != 0)
                return caponStatus;
        }

        try
        {
            ctx_ = new RadarPipelineContext();

            ctx_->numSamples      = gp->NofSamples;
            ctx_->numRx           = gp->NofRx;
            ctx_->numChirps       = gp->NofRamps;
            ctx_->numDopplerFolds = gp->RDFold;
            ctx_->numTx           = gp->NofTx;
            ctx_->numRangeBins    = gp->NofRangeBins > 0 ? gp->NofRangeBins : (ctx_->numSamples / 2);
            ctx_->targetCount     = 0;
            switch (config->doa_params.doa_method)
            {
            case RADAR_PIPELINE_DOA_BARTLETT:
                ctx_->doaMethod = RadarDoaMethod::BARTLETT;
                break;
            case RADAR_PIPELINE_DOA_CAPON_ROI:
                ctx_->doaMethod = RadarDoaMethod::CAPON_ROI;
                break;
            default:
                ctx_->doaMethod = RadarDoaMethod::ANGLE_FFT;
                break;
            }

            ctx_->ddmDopplerOffsets.assign(gp->ddm_doppler_offsets,
                                           gp->ddm_doppler_offsets + gp->ddm_doppler_offsets_count);
            ctx_->gp = gp;
            // Only the Bartlett path reads bartlett_config (it sizes the Bartlett tensors); other methods leave
            // it zero, whatever the caller's struct holds.
            ctx_->bartlettConfig = (ctx_->doaMethod == RadarDoaMethod::BARTLETT) ? config->doa_params.bartlett_config
                                                                                 : RadarPipelineBartlettConfig{};

            // Derive CFAR threshold factors from probability of false alarm
            const RadarPipelineCfarParams *cfgCfar = &config->cfar_params;
            int32_t numHRef                        = 2 * cfgCfar->numHorTrain;
            int32_t numVRef                        = 2 * cfgCfar->numVerTrain;
            ctx_->cfarHorizontalThresholdFactors.resize(1);
            ctx_->cfarHorizontalThresholdFactors[0] =
                numHRef * (std::pow(cfgCfar->probabilityFA, -1.0f / numHRef) - 1.0f);
            ctx_->cfarParams = {
                .numHorTrain               = cfgCfar->numHorTrain,
                .numHorGuard               = cfgCfar->numHorGuard,
                .numVerTrain               = cfgCfar->numVerTrain,
                .numVerGuard               = cfgCfar->numVerGuard,
                .horizontalThreshold       = {.type  = PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM,
                                              .value = {.uniform = ctx_->cfarHorizontalThresholdFactors[0]}},
                .verticalThresholdFactor   = numVRef * (std::pow(cfgCfar->probabilityFA, -1.0f / numVRef) - 1.0f),
                .isHorizontalCyclicPadding = (cfgCfar->isHorizontalCyclicPadding != 0),
                .isVerticalCyclicPadding   = (cfgCfar->isVerticalCyclicPadding != 0),
                .enablePeakGrouping        = (cfgCfar->enablePeakGrouping != 0),
            };

            ctx_->pvaParams = std::make_unique<RadarPipelinePVAWorkloadParams>();

            const RadarPipelineDOAParams *cfgDoa          = &config->doa_params;
            const RadarPipelineTargetProcessingParams *tp = &cfgDoa->targetProcessingParams;
            ctx_->targetProcessingParams                  = {
                                 .wavelength                   = static_cast<float>(ctx_->gp->wavelength),
                                 .sweepTime                    = static_cast<float>(ctx_->gp->sweepTime),
                                 .rangeResolution              = static_cast<float>(ctx_->gp->rangeResolution),
                                 .numChirps                    = ctx_->gp->NofRamps,
                                 .fftShiftedDoppler            = (tp->fftShiftedDoppler != 0),
                                 .enableRvDecoupling           = (tp->enableRvDecoupling != 0),
                                 .enableQuadraticInterpolation = (tp->enableQuadraticInterpolation != 0),
                                 .fs                           = static_cast<float>(ctx_->gp->fs),
                                 .nofSamples                   = ctx_->gp->NofSamples,
                                 .PRI                          = static_cast<float>(ctx_->gp->PRI),
                                 .deltaV                       = static_cast<float>(ctx_->gp->DeltaV),
                                 .contR                        = ctx_->gp->ContR,
                                 .contVFast                    = ctx_->gp->ContV_fast,
                                 .contVSlow                    = ctx_->gp->ContV_slow,
                                 .longitudinalAxis             = 1, /* Y_FORWARD (automotive convention) */
                                 .groundProjection             = 1, /* GROUND_PLANE */
            };

            if (ctx_->doaMethod == RadarDoaMethod::BARTLETT)
            {
                ctx_->bartlettParams = {
                    .enableInterpolation = (cfgDoa->enableInterpolation != 0),
                    .enablePowerOutput   = (cfgDoa->enablePowerOutput != 0),
                    .separableParams =
                        {
                            .enableSeparable = (cfgDoa->separableParams.enableSeparable != 0),
                            .azNeighborhood =
                                static_cast<PVABartlettAzNeighborhood>(cfgDoa->separableParams.azNeighborhood),
                            .azSource      = static_cast<PVABartlettAzSource>(cfgDoa->separableParams.azSource),
                            .azChannelMask = gp->txGroups[0],
                        },
                };
            }
            else
            {
                ctx_->doaAngleFFTParams = {
                    .numVirtualAzimuthElements   = ctx_->numTx * ctx_->numRx,
                    .numVirtualElevationElements = 1,
                    .angleEstimationPrecision    = PVA_ANGLE_ESTIMATION_PRECISION_DEFAULT,
                    .enableLocalPeakDetection    = false,
                };
            }

            if (ctx_->doaMethod == RadarDoaMethod::CAPON_ROI)
            {
                ctx_->caponConfig = capon; // defaulted and validated above

                const int32_t svDim = capon.roiAzSize * capon.roiElSize;
                ctx_->caponParams   = {
                      .numAzBins            = static_cast<uint16_t>(kCaponNumAzBins),
                      .numElBins            = static_cast<uint16_t>(kCaponNumElBins),
                      .svDim                = static_cast<uint16_t>(svDim),
                      .numPeaksPerROI       = RADAR_PIPELINE_CAPON_NUM_PEAKS,
                      .inputQBits           = static_cast<uint32_t>(kCaponInputQBits),
                      .steeringQBits        = static_cast<uint32_t>(kCaponSteeringQBits),
                      .adpDiagConst         = capon.adpDiagConst,
                      .noiseFloorRatio      = capon.noiseFloorRatio,
                      .numDopplerBinsCovMat = capon.numDopplerBinsCovMat,
                      .useFpCovMatKnl       = false, // the S32 covariance kernel
                      .enableSubBinInterp   = (capon.enableSubBinInterp != 0),
                };

                // ROI preprocessing config, shared by the operator params and the CPU reference (POD; zero
                // then set every field).
                CaponROIPreprocConfig &pp = ctx_->caponPreprocConfig;
                pp                        = CaponROIPreprocConfig{};
                pp.numTx                  = ctx_->numTx;
                pp.numRx                  = ctx_->numRx;
                pp.numAzimuthElements     = capon.rxAzimuthElements;
                pp.numElevationElements   = capon.rxElevationElements;
                pp.numChirps              = ctx_->numChirps;
                pp.roiAzSize              = capon.roiAzSize;
                pp.roiElSize              = capon.roiElSize;
                pp.numAzBins              = kCaponNumAzBins;
                pp.numElBins              = kCaponNumElBins;
                pp.enableFBA              = (capon.enableFBA != 0);
                pp.dopplerNeighborhood    = capon.dopplerNeighborhood;
                pp.numDopplerBinsCovMat   = capon.numDopplerBinsCovMat;
                pp.txSpacing              = (gp->Dis > 0.0) ? gp->Dis : 0.5;
                // RX azimuth spacing spans the whole TX aperture so the virtual array is a filled ULA
                // (numTx * txSpacing; 4.0 for 8 TX at 0.5), as in the MATLAB and fixed-point references.
                pp.rxAzSpacing   = ctx_->numTx * pp.txSpacing;
                pp.rxElSpacing   = (gp->DisE > 0.0) ? gp->DisE : 0.5;
                pp.inputQBits    = kCaponInputQBits;
                pp.steeringQBits = kCaponSteeringQBits;
                // DDM decode terms from the Doppler offsets (validateCaponConfig checked there is one per TX):
                // relDopplerOffset + ddmaWeight = exp(+j2pi ddm[tx]), referenced to ddm[0]. The operator
                // and the reference both take them from here.
                const double kPi = 3.14159265358979323846;
                for (int32_t tx = 0; tx < ctx_->numTx && tx < CAPON_ROI_PREPROC_MAX_TX; ++tx)
                {
                    const double ddm0       = gp->ddm_doppler_offsets[0];
                    const double ddm        = gp->ddm_doppler_offsets[tx];
                    pp.relDopplerOffset[tx] = static_cast<int32_t>(std::lround((ddm0 - ddm) * ctx_->numChirps));
                    pp.ddmaWeightRe[tx]     = static_cast<float>(std::cos(2.0 * kPi * ddm));
                    pp.ddmaWeightIm[tx]     = static_cast<float>(std::sin(2.0 * kPi * ddm));
                }
            }

            PVAFFTLayout dopplerFFTOutputLayout =
                (ctx_->numRx >= 8) ? PVA_FFT_LAYOUT_RANGE_DOPPLER_RX : PVA_FFT_LAYOUT_RANGE_RX_DOPPLER;

            ctx_->calib_data = config->calib_data;
            // Sanitize calib_data so uninitialized config never leads to DMA from garbage pointer (HW hang)
            if (ctx_->calib_data.steeringVectorsPtr == nullptr)
            {
                ctx_->calib_data.steeringVectorsSize = 0;
            }
            else if (ctx_->calib_data.steeringVectorsSize <= 0 ||
                     ctx_->calib_data.steeringVectorsSize > 8 * 1024 * 1024)
            {
                ctx_->calib_data.steeringVectorsPtr  = nullptr;
                ctx_->calib_data.steeringVectorsSize = 0;
            }

            ctx_->tensorReqs = std::make_unique<RadarPipelineTensorRequirements>(
                ctx_->numSamples, ctx_->numRx, ctx_->numChirps, ctx_->numDopplerFolds, ctx_->numTx, ctx_->numRangeBins,
                ctx_->bartlettConfig,
                (ctx_->doaMethod == RadarDoaMethod::CAPON_ROI) ? &ctx_->caponPreprocConfig : nullptr,
                ctx_->caponConfig.maxRoi);

            ctx_->operators = std::make_unique<RadarPipelineOperators>();

            ctx_->rangeParams   = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                   .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                   .outputLayout = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE};
            ctx_->dopplerParams = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                   .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                   .outputLayout = dopplerFFTOutputLayout};
            ctx_->nciParams     = {
                    .repeatFold             = ctx_->numDopplerFolds,
                    .noiseEstimationEnabled = false,
                    .inputLayout            = dopplerFFTOutputLayout,
            };
            PVARadarCFARType cfarType      = PVA_CFAR_CA;
            ctx_->snapshotExtractionParams = {
                .numDopplerFolds    = ctx_->numDopplerFolds,
                .rangeDopplerFormat = dopplerFFTOutputLayout,
            };

            const PVADoaAngleFFTParams *pDoaAngleFFT =
                (ctx_->doaMethod == RadarDoaMethod::ANGLE_FFT) ? &ctx_->doaAngleFFTParams : nullptr;
            const PVADoaBartlettBeamformingParams *pBartlett =
                (ctx_->doaMethod == RadarDoaMethod::BARTLETT) ? &ctx_->bartlettParams : nullptr;
            const PVADoaCaponROIParams *pCapon =
                (ctx_->doaMethod == RadarDoaMethod::CAPON_ROI) ? &ctx_->caponParams : nullptr;
            const CaponROIPreprocConfig *pCaponPreproc =
                (ctx_->doaMethod == RadarDoaMethod::CAPON_ROI) ? &ctx_->caponPreprocConfig : nullptr;

            int32_t err = radarCreatePvaWorkloads(*ctx_->operators, *ctx_->tensorReqs, ctx_->rangeParams,
                                                  ctx_->dopplerParams, ctx_->nciParams, cfarType, &ctx_->cfarParams,
                                                  ctx_->snapshotExtractionParams, ctx_->doaMethod, pDoaAngleFFT,
                                                  pBartlett, &ctx_->targetProcessingParams, pCapon, pCaponPreproc);
            if (err != 0)
            {
                delete ctx_;
                ctx_ = nullptr;
                return err;
            }

            ctx_->tensors = std::make_unique<RadarPipelineTensors>(ctx_);

            if (dims_out)
            {
                dims_out->NofSamples = ctx_->numSamples;
                dims_out->NofRx      = ctx_->numRx;
                dims_out->NofRamps   = ctx_->numChirps;
            }

            return 0;
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "RadarPipeline8t8r::init exception: %s\n", e.what());
            delete ctx_;
            ctx_ = nullptr;
            return RADAR_PIPELINE_CREATE_ERR_EXCEPTION;
        }
        catch (...)
        {
            fprintf(stderr, "RadarPipeline8t8r::init unknown exception\n");
            delete ctx_;
            ctx_ = nullptr;
            return RADAR_PIPELINE_CREATE_ERR_EXCEPTION;
        }
    }

    int32_t process(NVCVTensorHandle input_tensor, NVCVTensorHandle output_tensor) override
    {
        if (!input_tensor)
            return -1;

        try
        {
            NVCVTensorHandle originalOutputTensor = ctx_->tensors->targetListTensorHandle;

            ctx_->tensors->inRangeFFTTensorHandle = input_tensor;
            if (output_tensor)
                ctx_->tensors->targetListTensorHandle = output_tensor;

            const CaponROIPreprocConfig *pCaponPreproc =
                (ctx_->doaMethod == RadarDoaMethod::CAPON_ROI) ? &ctx_->caponPreprocConfig : nullptr;
            int32_t err = radarSubmitPvaWorkloads(*ctx_->operators, *ctx_->tensors, *ctx_->pvaParams, ctx_->doaMethod,
                                                  pCaponPreproc);

            ctx_->tensors->inRangeFFTTensorHandle = nullptr;
            ctx_->tensors->targetListTensorHandle = originalOutputTensor;

            if (err != 0)
                return err;

            // Read target count from tensor
            int32_t readTargetCount = 0;
            bool targetCountReadOk  = readInt32FromTensor(ctx_->tensors->targetCountTensorHandle, &readTargetCount);
            if (targetCountReadOk)
                ctx_->targetCount = readTargetCount;

            // Read CFAR detection count for debugging
            int32_t detectionCount    = -1;
            bool detectionCountReadOk = readInt32FromTensor(ctx_->tensors->detectionCountTensorHandle, &detectionCount);

            if (targetCountReadOk && detectionCountReadOk)
            {
#ifdef RADAR_PIPELINE_DEBUG
                fprintf(stderr, "[radar_pipeline] CFAR detection count = %d, target count = %d\n", detectionCount,
                        ctx_->targetCount);
#endif
            }
            else if (!targetCountReadOk)
            {
                fprintf(stderr, "[radar_pipeline] Failed to read target count tensor\n");
            }
            if (!detectionCountReadOk)
            {
                fprintf(stderr, "[radar_pipeline] Failed to read detection count tensor\n");
            }

            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }

    int32_t get_target_count(int32_t *count) override
    {
        if (!count)
            return -1;
        *count = ctx_->targetCount;
        return 0;
    }

    int32_t validate(void *input_pva_ptr) override
    {
        if (!input_pva_ptr)
            return -1;

        try
        {
            NVCVTensorHandle externalInputTensor = nullptr;
            NVCVStatus status                    = utils::createTensorFromPvaMemory(
                &externalInputTensor, ctx_->tensorReqs->rawAdcDataTensorRequirements, input_pva_ptr);
            if (status != NVCV_SUCCESS)
                return static_cast<int32_t>(status);

            NVCVTensorHandle originalInputTensor  = ctx_->tensors->inRangeFFTTensorHandle;
            ctx_->tensors->inRangeFFTTensorHandle = externalInputTensor;

            const PVADoaAngleFFTParams *pDoaAngleFFT =
                (ctx_->doaMethod == RadarDoaMethod::ANGLE_FFT) ? &ctx_->doaAngleFFTParams : nullptr;
            const PVADoaBartlettBeamformingParams *pBartlett =
                (ctx_->doaMethod == RadarDoaMethod::BARTLETT) ? &ctx_->bartlettParams : nullptr;
            const CaponROIPreprocConfig *pCaponPreproc =
                (ctx_->doaMethod == RadarDoaMethod::CAPON_ROI) ? &ctx_->caponPreprocConfig : nullptr;

            int32_t err = radarExecuteRefWorkloads(
                *ctx_->tensors, ctx_->rangeParams, ctx_->dopplerParams, ctx_->nciParams, ctx_->cfarParams, PVA_CFAR_CA,
                ctx_->snapshotExtractionParams, ctx_->doaMethod, pDoaAngleFFT, pBartlett, &ctx_->targetProcessingParams,
                pCaponPreproc, (ctx_->doaMethod == RadarDoaMethod::CAPON_ROI) ? &ctx_->caponParams : nullptr);

            ctx_->tensors->inRangeFFTTensorHandle = originalInputTensor;
            nvcvTensorDecRef(externalInputTensor, NULL);

            int32_t refDet = -1, refTgt = -1;
            if (readInt32FromTensor(ctx_->tensors->refDetectionCountTensorHandle, &refDet) &&
                readInt32FromTensor(ctx_->tensors->refTargetCountTensorHandle, &refTgt))
            {
                fprintf(stderr, "[radar_pipeline] Reference pipeline: CFAR detection count = %d, target count = %d\n",
                        refDet, refTgt);
            }

            return err;
        }
        catch (...)
        {
            return -1;
        }
    }

    int32_t get_intermediate_tensor(RadarPipelineTensorType type, RadarPipelineTensorSource src,
                                    RadarPipelineTensorData *data) override
    {
        if (!data || type < 0 || type >= RADAR_PIPELINE_TENSOR_COUNT)
            return -1;

        try
        {
            NVCVTensorHandle tensorHandle = nullptr;
            RadarPipelineDataType dtype   = RADAR_PIPELINE_DTYPE_FLOAT32;
            const bool usePva             = (src == RADAR_PIPELINE_SOURCE_PVA);

            switch (type)
            {
            case RADAR_PIPELINE_TENSOR_RANGE_PROFILE:
                tensorHandle =
                    usePva ? ctx_->tensors->rangeProfileTensorHandle : ctx_->tensors->refRangeProfileTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_COMPLEX_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_DOPPLER_MAP:
                tensorHandle =
                    usePva ? ctx_->tensors->rangeDopplerMapTensorHandle : ctx_->tensors->refRangeDopplerMapTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_COMPLEX_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_NCI_RX:
                tensorHandle = usePva ? ctx_->tensors->nciRxTensorHandle : ctx_->tensors->refNciRxTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_UINT32;
                break;
            case RADAR_PIPELINE_TENSOR_NCI_FINAL:
                tensorHandle = usePva ? ctx_->tensors->nciFinalTensorHandle : ctx_->tensors->refNciFinalTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_UINT32;
                break;
            case RADAR_PIPELINE_TENSOR_CFAR_DETECTIONS:
                tensorHandle = usePva ? ctx_->tensors->foldedDetectionListTensorHandle
                                      : ctx_->tensors->refFoldedDetectionListTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_DETECTION_LIST:
                tensorHandle =
                    usePva ? ctx_->tensors->detectionListTensorHandle : ctx_->tensors->refDetectionListTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_SNAPSHOTS:
                tensorHandle = usePva ? ctx_->tensors->snapshotsTensorHandle : ctx_->tensors->refSnapshotsTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_COMPLEX_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_TARGET_LIST:
                tensorHandle =
                    usePva ? ctx_->tensors->targetListTensorHandle : ctx_->tensors->refTargetListTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_FLOAT32;
                break;
            case RADAR_PIPELINE_TENSOR_DETECTION_COUNT:
                tensorHandle =
                    usePva ? ctx_->tensors->detectionCountTensorHandle : ctx_->tensors->refDetectionCountTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_TARGET_COUNT:
                tensorHandle =
                    usePva ? ctx_->tensors->targetCountTensorHandle : ctx_->tensors->refTargetCountTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_INT32;
                break;
            // Capon outputs: the CPU reference runs no Capon chain of its own, so these have no REFERENCE source.
            case RADAR_PIPELINE_TENSOR_CAPON_PEAK_POWER:
                tensorHandle = usePva ? ctx_->tensors->caponPeakPowerTensorHandle : nullptr;
                dtype        = RADAR_PIPELINE_DTYPE_FLOAT32;
                break;
            case RADAR_PIPELINE_TENSOR_CAPON_AZ_BIN:
                tensorHandle = usePva ? ctx_->tensors->caponAzBinTensorHandle : nullptr;
                dtype        = RADAR_PIPELINE_DTYPE_FLOAT32;
                break;
            case RADAR_PIPELINE_TENSOR_CAPON_EL_BIN:
                tensorHandle = usePva ? ctx_->tensors->caponElBinTensorHandle : nullptr;
                dtype        = RADAR_PIPELINE_DTYPE_FLOAT32;
                break;
            case RADAR_PIPELINE_TENSOR_CAPON_ROI_COUNT:
                tensorHandle = usePva ? ctx_->tensors->caponROICountTensorHandle : nullptr;
                dtype        = RADAR_PIPELINE_DTYPE_INT32;
                break;
            default:
                return -1;
            }

            if (!tensorHandle)
                return -2;

            NVCVTensorData nvcvData;
            if (nvcvTensorExportData(tensorHandle, &nvcvData) != NVCV_SUCCESS)
                return -3;

            void *hostPtr = nullptr;
            if (CupvaMemGetHostPointer(&hostPtr, (void *)nvcvData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
                return -3;

            data->data  = hostPtr;
            data->ndim  = nvcvData.rank;
            data->dtype = dtype;

            int32_t elementSize = 4;
            switch (dtype)
            {
            case RADAR_PIPELINE_DTYPE_INT16:
                elementSize = 2;
                break;
            case RADAR_PIPELINE_DTYPE_COMPLEX_INT16:
                elementSize = 4;
                break;
            case RADAR_PIPELINE_DTYPE_COMPLEX_INT32:
                elementSize = 8;
                break;
            default:
                elementSize = 4;
                break;
            }

            int64_t totalBytes = 1;
            for (int32_t i = 0; i < nvcvData.rank && i < 4; i++)
            {
                data->shape[i]   = nvcvData.shape[i];
                data->strides[i] = nvcvData.buffer.strided.strides[i];
                totalBytes *= nvcvData.shape[i];
            }
            for (int32_t i = nvcvData.rank; i < 4; i++)
            {
                data->shape[i]   = 0;
                data->strides[i] = 0;
            }
            data->total_bytes = totalBytes * elementSize;

            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }

    const char *version() const override
    {
        return RADARLIB_VERSION_STRING;
    }

private:
    RadarPipelineContext *ctx_ = nullptr;
};

// ============================================================================
// Factory
// ============================================================================

extern "C" __attribute__((visibility("default"))) IRadarPipeline *radar_pipeline_8t8r()
{
    return new RadarPipeline8t8r();
}
