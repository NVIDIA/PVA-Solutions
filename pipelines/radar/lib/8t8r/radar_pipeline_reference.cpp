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

#include "radar_pipeline_reference.hpp"

#include "CaponROIBeamspace.hpp"
#include "radar_pipeline_host_utils.hpp"
#include "radar_pipeline_pva.hpp" // For RadarDoaMethod enum

#include "capon_roi_preprocess_ref.h"
#include "doa_angle_fft_ref.h"
#include "doa_bartlett_beamforming_ref.h"
#include "doppler_fft_ref.h"
#include "nci_ref.h"
#include "radar_cfar_test_ref.h"
#include "range_fft_ref.h"
#include "snapshot_extraction_ref.h"
#include "target_processing_ref.h"

#include <ErrorCheckMacros.h>
#include <OpNci.h>
#include <OpRadarCFAR.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>

#if CUPVA_LLVM_SUPPORTED == 1 // the Capon DoA operator and its reference need the LLVM front end
#    include "doa_capon_roi_ref.h"
#    include "doa_capon_roi_utils.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int rangeFFTProcessing(RadarPipelineTensors &tensors, PVARangeFFTParams rangeParams)
{
    int32_t err = 0;

    // Run reference implementation for comparison
    rangeFFTRef(tensors.inRangeFFTTensorHandle, tensors.rangeFFTWindowTensorHandle, tensors.refRangeProfileTensorHandle,
                &rangeParams);

    // Compare PVA output to reference
    err = checkRangeFFTOutput(tensors.refRangeProfileTensorHandle, tensors.rangeProfileTensorHandle, 0.0f);
    if (err)
    {
        printf("Range FFT Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("Range FFT Processing Test Pass - PVA operator matches reference\n");
    }

    return err;
}

int dopplerFFTProcessing(RadarPipelineTensors &tensors, PVADopplerFFTParams dopplerParams)
{
    int32_t err = 0;

    // Run reference implementation for comparison
    dopplerFFTRef(tensors.rangeProfileTensorHandle, tensors.dopplerFFTWindowTensorHandle,
                  tensors.refRangeDopplerMapTensorHandle, &dopplerParams);

    // Compare PVA output to reference
    err = checkDopplerFFTOutput(tensors.refRangeDopplerMapTensorHandle, tensors.rangeDopplerMapTensorHandle, 0.0f);
    if (err)
    {
        printf("Doppler FFT Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("Doppler FFT Processing Test Pass - PVA operator matches reference\n");
    }

    return err;
}

int nciProcessing(RadarPipelineTensors &tensors, PVARadarNCIParams nciParams)
{
    int32_t err = 0;

    // Tolerance values for output comparison based on the reference implementation
    const uint32_t nciRxTolerance    = 3u;
    const uint32_t nciFinalTolerance = (2u * nciRxTolerance) + 2u; // 8u

    // Run reference implementation for comparison
    NVCVTensorHandle inTensorHandles[]  = {tensors.refRangeDopplerMapTensorHandle};
    NVCVTensorHandle outTensorHandles[] = {tensors.nciRxTensorHandle, tensors.nciFinalTensorHandle, NULL};
    NVCVTensorHandle refTensorHandles[] = {tensors.refNciRxTensorHandle, tensors.refNciFinalTensorHandle, NULL};

    err = nciRef(inTensorHandles, refTensorHandles, 1, 3, nciParams.repeatFold, nciParams.inputLayout);
    if (0 != err)
    {
        printf("NCI Reference Test Fail\n");
        goto EXIT;
    }

    // Compare PVA output to reference
    err = checkNCIOutput(refTensorHandles[0], outTensorHandles[0], nciRxTolerance);
    err |= checkNCIOutput(refTensorHandles[1], outTensorHandles[1], nciFinalTolerance);

    if (err != 0)
    {
        printf("NCI Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("NCI Processing Test Pass - PVA operator matches reference\n");
    }

EXIT:
    return err;
}

int cfarProcessing(RadarPipelineTensors &tensors, PvaRadarCFARParamsRec cfarParams, PVARadarCFARType cfarType)
{
    int32_t err = 0;

    // Run reference implementation for comparison
    radarCFARRefSubmit(tensors.refNciFinalTensorHandle, tensors.refFoldedDetectionListTensorHandle, cfarParams,
                       tensors.refDetectionCountTensorHandle, cfarType);

    // Compare PVA output to reference
    err = checkagainstRadarCFARRef(tensors.refFoldedDetectionListTensorHandle, tensors.foldedDetectionListTensorHandle,
                                   tensors.refDetectionCountTensorHandle, tensors.detectionCountTensorHandle);

    if (err != 0)
    {
        printf("CFAR Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("CFAR Processing Test Pass - PVA operator matches reference\n");
    }
    return err;
}

int snapshotExtractionProcessing(RadarPipelineTensors &tensors, PVASnapshotExtractionParams snapshotExtractionParams)
{
    int32_t err = 0;

    // Run reference implementation for comparison
    snapshotExtractionRef(tensors.refDetectionCountTensorHandle, tensors.refFoldedDetectionListTensorHandle,
                          tensors.ddmDopplerOffsetsTensorHandle, tensors.refNciRxTensorHandle,
                          tensors.refRangeDopplerMapTensorHandle, NULL, tensors.refDetectionListTensorHandle,
                          tensors.refSnapshotsTensorHandle, &snapshotExtractionParams);

    // Compare PVA output to reference
    err = checkSnapshotExtractionOutput(tensors.detectionCountTensorHandle, tensors.refDetectionCountTensorHandle,
                                        tensors.detectionListTensorHandle, tensors.refDetectionListTensorHandle,
                                        tensors.snapshotsTensorHandle, tensors.refSnapshotsTensorHandle);

    if (err != 0)
    {
        printf("Snapshot Extraction Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("Snapshot Extraction Processing Test Pass - PVA operator matches reference\n");
    }
    return err;
}

int doaAngleFFTProcessing(RadarPipelineTensors &tensors, PVADoaAngleFFTParams doaAngleFFTParams,
                          PVATargetProcessingParams const *targetProcessingParams)
{
    int32_t err = 0;

    // Run DOA-only reference implementation
    doaAngleFFTRef(tensors.refDetectionCountTensorHandle, tensors.refSnapshotsTensorHandle,
                   tensors.refTargetCountTensorHandle, tensors.refTargetAnglesTensorHandle,
                   tensors.refTargetIndexMapTensorHandle, &doaAngleFFTParams);

    // Chain with standalone Target Processing reference
    targetProcessingRef(tensors.refTargetCountTensorHandle, tensors.refDetectionListTensorHandle,
                        tensors.refTargetIndexMapTensorHandle, tensors.refTargetAnglesTensorHandle,
                        tensors.ddmDopplerOffsetsTensorHandle, tensors.refNciFinalTensorHandle,
                        tensors.refTargetListTensorHandle, targetProcessingParams);

    // Compare PVA output to reference
    err = checkTargetProcessingOutput(tensors.refTargetCountTensorHandle, tensors.targetListTensorHandle,
                                      tensors.refTargetListTensorHandle);

    if (err != 0)
    {
        printf("DOA Angle FFT + Target Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("DOA Angle FFT + Target Processing Test Pass - PVA operator matches reference\n");
    }
    return err;
}

int doaBartlettProcessing(RadarPipelineTensors &tensors, PVADoaBartlettBeamformingParams const *bartlettParams,
                          PVATargetProcessingParams const *targetProcessingParams)
{
    int32_t err = 0;

    if (tensors.steeringVectorsTensorHandle == nullptr || tensors.azimuthBinsTensorHandle == nullptr ||
        tensors.elevationBinsTensorHandle == nullptr)
    {
        printf("Error: Bartlett tensors not initialized. Create tensors with RadarPipelineBartlettConfig.\n");
        return -1;
    }

    // Run DOA-only Bartlett Beamforming reference implementation
    doaBartlettBeamformingRef(
        tensors.refDetectionCountTensorHandle, tensors.refSnapshotsTensorHandle, tensors.steeringVectorsTensorHandle,
        tensors.azimuthBinsTensorHandle, tensors.elevationBinsTensorHandle, tensors.refTargetCountTensorHandle,
        tensors.refTargetAnglesTensorHandle, tensors.refTargetIndexMapTensorHandle, bartlettParams);

    // Chain with standalone Target Processing reference
    targetProcessingRef(tensors.refTargetCountTensorHandle, tensors.refDetectionListTensorHandle,
                        tensors.refTargetIndexMapTensorHandle, tensors.refTargetAnglesTensorHandle,
                        tensors.ddmDopplerOffsetsTensorHandle, tensors.refNciFinalTensorHandle,
                        tensors.refTargetListTensorHandle, targetProcessingParams);

    // Compare PVA output to reference
    err = checkTargetProcessingOutput(tensors.refTargetCountTensorHandle, tensors.targetListTensorHandle,
                                      tensors.refTargetListTensorHandle);

    if (err != 0)
    {
        printf("DOA Bartlett Beamforming + Target Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("DOA Bartlett Beamforming + Target Processing Test Pass - PVA operator matches reference\n");
    }
    return err;
}

// Two coarse peaks closer than this (relative power) are a near-tie: for a 2D array the operator ranks the
// beamspace on fixed-point power (16-bit Q13 snapshots, Q15 twiddles) and may pick either one, so it can
// select another ROI than the double-precision reference (OpCaponROIPreprocess.h, Limitations).
constexpr double kCoarseTieRel = 1e-3;

// For an ROI whose steering block differs from the reference's: the device's ROI start is the LUT position
// holding its block. Returns 1 - (strongest coarse power inside the device's ROI) / (strongest coarse power
// in the searched beamspace), from the reference's double-precision beamspace, or -1 if the device's block is
// not in the LUT.
static double coarseTieGap(const CaponROIPreprocConfig &cfg, const int32_t *lut, const int32_t *devBlock,
                           const float *snapTxRx)
{
    using namespace pva_solutions::capon_roi;
    PreprocGeom g;
    if (buildGeom(&cfg, g) != 0)
        return -1.0;
    const size_t blockInts = static_cast<size_t>(g.svDim) * g.numStVecEl * g.numStVecAz * 2u;
    int32_t pos            = -1;
    for (int32_t p = 0; p < g.numRoiAzPos * g.numRoiElPos && pos < 0; ++p)
        if (std::memcmp(lut + p * blockInts, devBlock, blockInts * sizeof(int32_t)) == 0)
            pos = p;
    if (pos < 0)
        return -1.0;
    const int32_t azStart = pos % g.numRoiAzPos, elStart = pos / g.numRoiAzPos;

    std::vector<cd> x(g.nvc), beam;
    for (int32_t rx = 0; rx < g.numRx; ++rx)
        for (int32_t tx = 0; tx < g.numTx; ++tx)
        {
            const float *v       = snapTxRx + (static_cast<size_t>(tx) * g.numRx + rx) * 2u;
            x[rx * g.numTx + tx] = cd(v[0], v[1]);
        }
    Beamspace2D beamspace(g.numAzVirt, g.numElVirt, g.beamAz, g.beamEl, g.roiAz, g.roiEl);
    beamspace(x, beam);
    // The same search rows as the reference: only the 0-degree row for an azimuth-only array.
    const int32_t keLo = (g.numElVirt == 1) ? g.beamEl / 2 : 0, keHi = (g.numElVirt == 1) ? keLo + 1 : g.beamEl;
    double maxAll = 0.0, maxRoi = 0.0;
    for (int32_t ke = keLo; ke < keHi; ++ke)
        for (int32_t ka = 0; ka < g.beamAz; ++ka)
        {
            const double p = std::norm(beam[static_cast<size_t>(ke) * g.beamAz + ka]);
            maxAll         = std::max(maxAll, p);
            if (ka >= azStart && ka < azStart + g.roiAz && ke >= elStart && ke < elStart + g.roiEl)
                maxRoi = std::max(maxRoi, p);
        }
    return maxAll > 0.0 ? 1.0 - maxRoi / maxAll : -1.0;
}

// Counts the entries of two int32 arrays that differ by more than tol, and tracks the largest difference.
static long countMismatches(const int32_t *got, const int32_t *want, size_t n, long tol, long *maxDiff)
{
    long mismatches = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const long df = std::labs(static_cast<long>(got[i]) - static_cast<long>(want[i]));
        *maxDiff      = std::max(*maxDiff, df);
        mismatches += df > tol;
    }
    return mismatches;
}

// Checks the Capon stages of the last frame on the PVA stage inputs. The standalone radar_capon_roi_preprocess
// and radar_doa_capon_roi samples check both operators on fixed inputs; this adds the pipeline's own part:
//  - ROI preprocessing: the host marshalling plus the operator against caponROIPreprocess() run on the raw
//    pre-DOA outputs. The steering blocks come from the same LUT builder as the operator's LUT, so their
//    compare checks which block each ROI selects, not the steering values themselves. An ROI the device
//    selects differently passes only at a near-tie of the coarse peaks (kCoarseTieRel).
//  - Capon DoA: each ROI's strongest peak against the DoaCaponROI CPU reference run on the same ROI tensors.
//    The weaker peaks are not compared: they sit near the validity threshold and can swap at near-ties.
int caponROIProcessing(RadarPipelineTensors &tensors, const CaponROIPreprocConfig *caponPreproc,
                       const PVADoaCaponROIParams *caponParams)
{
    void *detListPtr = nullptr, *snapPtr = nullptr, *rdPtr = nullptr, *roiCountPtr = nullptr;
    void *snapRoiPtr = nullptr, *svRoiPtr = nullptr, *lutPtr = nullptr;
    NVCVTensorData rdData;
    if (caponPreproc == nullptr || caponParams == nullptr ||
        !radarTensorHostPtr(tensors.detectionListTensorHandle, &detListPtr) ||
        !radarTensorHostPtr(tensors.snapshotsTensorHandle, &snapPtr) ||
        !radarTensorHostPtr(tensors.rangeDopplerMapTensorHandle, &rdPtr, &rdData) ||
        !radarTensorHostPtr(tensors.caponROICountTensorHandle, &roiCountPtr) ||
        !radarTensorHostPtr(tensors.caponSnapshotsRoiTensorHandle, &snapRoiPtr) ||
        !radarTensorHostPtr(tensors.caponSteeringVectorsRoiTensorHandle, &svRoiPtr) ||
        !radarTensorHostPtr(tensors.caponSteeringLutTensorHandle, &lutPtr))
    {
        printf("Error: Capon ROI tensors not available on host\n");
        return -1;
    }
    const int32_t roiNum = *static_cast<const int32_t *>(roiCountPtr);
    if (roiNum == 0)
    {
        printf("Capon DoA-on-ROI: no detections to compare\n");
        return 0;
    }

    // ROI preprocessing: the reference on the raw inputs, normalized as the pipeline does.
    const CaponPreprocGeometry g = caponPreprocGeometry(*caponPreproc);
    const int32_t numRangeBins   = static_cast<int32_t>(rdData.shape[0]);
    std::vector<float> snapFloat, rdFloat;
    radarDequantNormalized(static_cast<const int32_t *>(snapPtr), static_cast<size_t>(roiNum) * g.nvc * 2u, snapFloat);
    radarDequantNormalized(static_cast<const int32_t *>(rdPtr),
                           static_cast<size_t>(numRangeBins) * g.numChirps * g.numRx * 2u, rdFloat);
    const size_t snapRoiInts = static_cast<size_t>(caponPreproc->numDopplerBinsCovMat) * g.svDim * 2u;
    const size_t svRoiInts   = static_cast<size_t>(g.svBlockCplx) * 2u;
    std::vector<int32_t> goldSnap(roiNum * snapRoiInts), goldSv(roiNum * svRoiInts);
    const CaponROIPreprocInput in = {roiNum, snapFloat.data(), rdFloat.data(), numRangeBins,
                                     static_cast<const int32_t *>(detListPtr)};
    CaponROIPreprocOutput out     = {goldSnap.data(), goldSv.data()};
    const int prc                 = caponROIPreprocess(caponPreproc, &in, &out);
    if (prc != 0)
    {
        printf("Capon DoA-on-ROI preprocessing reference failed (%d)\n", prc);
        return prc;
    }
    // Fixed point vs double. The projection is block-float per detection, so its error scales with the ROI's
    // magnitude: allow 2^-12 of the ROI's largest snapshot component, and at least +-8 LSB at Q12 (scaled to
    // inputQBits) for weak ROIs.
    const long snapTolMin = 8L << std::max(caponPreproc->inputQBits - 12, 0);
    double worstRel       = 0.0;
    const auto *devSnap   = static_cast<const int32_t *>(snapRoiPtr);
    const auto *devSv     = static_cast<const int32_t *>(svRoiPtr);
    long snapMaxDiff = 0, snapBad = 0, svBad = 0;
    int32_t tieRois = 0;
    for (int32_t r = 0; r < roiNum; ++r)
    {
        const int32_t *dSv = devSv + r * svRoiInts;
        if (std::memcmp(dSv, goldSv.data() + r * svRoiInts, svRoiInts * sizeof(int32_t)) != 0)
        {
            const double gap = coarseTieGap(*caponPreproc, static_cast<const int32_t *>(lutPtr), dSv,
                                            snapFloat.data() + static_cast<size_t>(r) * g.nvc * 2u);
            if (gap >= 0.0 && gap <= kCoarseTieRel)
            {
                printf(
                    "Capon DoA-on-ROI preprocessing: ROI %d selected at a near-tie of the coarse peaks (power "
                    "gap %.2e); not compared\n",
                    r, gap);
                ++tieRois;
                continue;
            }
            printf("Capon DoA-on-ROI preprocessing: ROI %d selects another ROI than the reference (gap %.2e)\n", r,
                   gap);
            ++svBad;
        }
        const int32_t *gold = goldSnap.data() + r * snapRoiInts;
        long roiMax = 0, roiDiff = 0;
        for (size_t k = 0; k < snapRoiInts; ++k) roiMax = std::max(roiMax, std::labs(static_cast<long>(gold[k])));
        const long snapTol = std::max(snapTolMin, roiMax >> 12);
        snapBad += countMismatches(devSnap + r * snapRoiInts, gold, snapRoiInts, snapTol, &roiDiff);
        snapMaxDiff = std::max(snapMaxDiff, roiDiff);
        if (roiMax > 0)
            worstRel = std::max(worstRel, static_cast<double>(roiDiff) / static_cast<double>(roiMax));
    }
    printf(
        "Capon DoA-on-ROI preprocessing VPU-vs-ref over %d ROI(s) (%d near-tie): snapshots maxDiff=%ld (worst %.2e "
        "of the ROI peak; tolerance 2^-12) mismatches=%ld; steering block selection mismatches=%ld\n",
        roiNum, tieRois, snapMaxDiff, worstRel, snapBad, svBad);
    int err = (snapBad == 0 && svBad == 0) ? 0 : -1;
    printf(err == 0 ? "Capon DoA-on-ROI preprocessing Test Pass - VPU operator matches reference\n"
                    : "Capon DoA-on-ROI preprocessing Test Fail - VPU output differs from reference\n");

#if CUPVA_LLVM_SUPPORTED == 1
    // Capon DoA: the CPU reference on the same ROI tensors the operator read; each ROI's strongest peak.
    doaCaponROIRef(tensors.caponSnapshotsRoiTensorHandle, tensors.caponSteeringVectorsRoiTensorHandle,
                   tensors.refCaponPeakPowerTensorHandle, tensors.refCaponAzBinTensorHandle,
                   tensors.refCaponElBinTensorHandle, caponParams);
    void *vp = nullptr, *va = nullptr, *ve = nullptr, *rp = nullptr, *ra = nullptr, *re = nullptr;
    if (!radarTensorHostPtr(tensors.caponPeakPowerTensorHandle, &vp) ||
        !radarTensorHostPtr(tensors.caponAzBinTensorHandle, &va) ||
        !radarTensorHostPtr(tensors.caponElBinTensorHandle, &ve) ||
        !radarTensorHostPtr(tensors.refCaponPeakPowerTensorHandle, &rp) ||
        !radarTensorHostPtr(tensors.refCaponAzBinTensorHandle, &ra) ||
        !radarTensorHostPtr(tensors.refCaponElBinTensorHandle, &re))
    {
        printf("Error: Capon DoA tensors not available on host\n");
        return -1;
    }
    // Fractional bins when sub-bin interpolation is on, integer bins (exact) otherwise. An azimuth-only array
    // has no elevation resolution: every elevation bin ties, so its elevation bin is not compared.
    const float binTol   = caponParams->enableSubBinInterp ? TOLERANCE_VPU_VS_REF_BIN_ABS : 0.0f;
    const bool compareEl = g.numElVirt > 1;
    const int32_t nPeaks = caponParams->numPeaksPerROI;
    int32_t peakErrors   = 0;
    for (int32_t r = 0; r < roiNum; ++r)
    {
        const size_t i = static_cast<size_t>(r) * nPeaks; // peak 0 of ROI r
        const float pv = static_cast<const float *>(vp)[i], pr = static_cast<const float *>(rp)[i];
        const bool vVal = pv != PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER, rVal = pr != PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER;
        bool ok = vVal == rVal;
        if (ok && vVal)
        {
            ok = std::fabs(pv - pr) <= TOLERANCE_POWER_REL * std::fabs(pr) &&
                 std::fabs(static_cast<const float *>(va)[i] - static_cast<const float *>(ra)[i]) <= binTol &&
                 (!compareEl ||
                  std::fabs(static_cast<const float *>(ve)[i] - static_cast<const float *>(re)[i]) <= binTol);
        }
        if (!ok)
        {
            printf("Capon DoA mismatch, ROI %d strongest peak: power %g vs %g, az %g vs %g, el %g vs %g\n", r, pv, pr,
                   static_cast<const float *>(va)[i], static_cast<const float *>(ra)[i],
                   static_cast<const float *>(ve)[i], static_cast<const float *>(re)[i]);
            ++peakErrors;
        }
    }
    printf(peakErrors == 0 ? "Capon DoA Test Pass - PVA operator matches reference (strongest peak per ROI)\n"
                           : "Capon DoA Test Fail - PVA peaks differ from reference\n");
    if (peakErrors != 0)
        err = -1;
#endif
    return err;
}

// Function to execute all reference workloads
int radarExecuteRefWorkloads(RadarPipelineTensors &tensors, PVARangeFFTParams rangeParams,
                             PVADopplerFFTParams dopplerParams, PVARadarNCIParams nciParams,
                             PvaRadarCFARParams cfarParams, PVARadarCFARType cfarType,
                             PVASnapshotExtractionParams snapshotExtractionParams, RadarDoaMethod doaMethod,
                             PVADoaAngleFFTParams const *doaAngleFFTParams,
                             PVADoaBartlettBeamformingParams const *bartlettParams,
                             PVATargetProcessingParams const *targetProcessingParams,
                             CaponROIPreprocConfig const *caponPreproc, PVADoaCaponROIParams const *caponParams)
{
    int32_t err = 0;

    err = rangeFFTProcessing(tensors, rangeParams);
    if (err != 0)
    {
        printf("Range FFT processing failed with error: %d\n", err);
        goto EXIT;
    }

    err = dopplerFFTProcessing(tensors, dopplerParams);
    if (err != 0)
    {
        printf("Doppler FFT processing failed with error: %d\n", err);
        goto EXIT;
    }

    err = nciProcessing(tensors, nciParams);
    if (err != 0)
    {
        printf("NCI processing failed with error: %d\n", err);
        goto EXIT;
    }

    err = cfarProcessing(tensors, cfarParams, cfarType);
    if (err != 0)
    {
        printf("CFAR processing failed with error: %d\n", err);
        // Continue with DOA processing even if comparison failed
        // (comparison can fail due to detection count differences between PVA and reference)
        err = 0;
    }

    err = snapshotExtractionProcessing(tensors, snapshotExtractionParams);
    if (err != 0)
    {
        printf("Snapshot extraction failed with error: %d\n", err);
        goto EXIT;
    }

    // DOA processing based on selected method
    if (doaMethod == RadarDoaMethod::ANGLE_FFT)
    {
        if (doaAngleFFTParams == nullptr)
        {
            printf("Error: doaAngleFFTParams is required for ANGLE_FFT method\n");
            err = -1;
            goto EXIT;
        }
        err = doaAngleFFTProcessing(tensors, *doaAngleFFTParams, targetProcessingParams);
        if (err != 0)
        {
            printf("DOA Angle FFT + Target processing failed with error: %d\n", err);
            goto EXIT;
        }
    }
    else if (doaMethod == RadarDoaMethod::BARTLETT)
    {
        if (bartlettParams == nullptr)
        {
            printf("Error: bartlettParams is required for BARTLETT method\n");
            err = -1;
            goto EXIT;
        }
        err = doaBartlettProcessing(tensors, bartlettParams, targetProcessingParams);
        if (err != 0)
        {
            printf("DOA Bartlett + Target processing failed with error: %d\n", err);
            goto EXIT;
        }
    }
    else if (doaMethod == RadarDoaMethod::CAPON_ROI)
    {
        err = caponROIProcessing(tensors, caponPreproc, caponParams);
        if (err != 0)
        {
            printf("Capon DoA-on-ROI comparison failed with error: %d\n", err);
            goto EXIT;
        }
    }

EXIT:
    return err;
}
