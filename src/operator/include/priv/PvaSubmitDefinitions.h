/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file PvaSubmitDefinitions.h
 *
 * @brief Declarations and C dispatch macros for PVA operator submit functions.
 *        This file contains the actual function declarations and C _Generic dispatchers.
 */

#ifndef PVA_SOLUTIONS_PVASUBMITDEFINITIONS_H
#define PVA_SOLUTIONS_PVASUBMITDEFINITIONS_H

#include "PvaSubmitMacros.h"

#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>

// Forward declarations for types used in submit function declarations
typedef struct PvaDLInferenceSubmitParamRec PvaDLInferenceSubmitParams;
typedef float WarpMatrixType[3][3];
typedef struct PvaRadarCFARParamsRec PvaRadarCFARParams;
typedef struct PvaLidarHistogramParamsRec PvaLidarHistogramParams;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Internal implementation: Executes morphology operation with cuPVA/CUDA stream.
 * Users should call pvaMorphologySubmit() instead.
 */
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMorphologySubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageHistogramSubmit, (NVCVTensorHandle const, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCornerSubPixSubmit, (NVCVTensorHandle, inCorners), (NVCVTensorHandle, outCorners),
                             (NVCVTensorHandle, image), (int32_t const, maxIters), (float const, eps),
                             (int32_t const, numCorners));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDepthToSpaceSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMinMaxLocSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, minVal),
                             (NVCVTensorHandle, minLoc), (NVCVTensorHandle, numMin), (NVCVTensorHandle, maxVal),
                             (NVCVTensorHandle, maxLoc), (NVCVTensorHandle, numMax));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaTemplateMatchingSubmit, (NVCVTensorHandle, inImage), (NVCVTensorHandle, inTemplate),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMixChannelsSubmit, (NVCVTensorHandle *, inHandles), (NVCVTensorHandle *, outHandles),
                             (int, inTensorCount), (int, outTensorCount));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBlToPlSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaConv2dSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaGaussianFilterSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaGaussianPyramidSubmit, (NVCVImageHandle, inImageHandle),
                             (NVCVImageHandle *, outImageHandles));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBoxFilterSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaConvertImageFormatSubmit, (NVCVImageHandle const, in), (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaFastCornerDetectorSubmit, (NVCVTensorHandle, in), (int32_t, intensityThreshold),
                             (NVCVTensorHandle, loc), (NVCVTensorHandle, numLoc));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageBlendSubmit, (NVCVImageHandle const, in0), (NVCVImageHandle const, in1),
                             (float const, alpha), (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBatchSVDSubmit, (NVCVTensorHandle const, src), (NVCVTensorHandle const, U),
                             (NVCVTensorHandle const, S), (NVCVTensorHandle const, V), (int32_t const, maxIters),
                             (float const, tol));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageResizeSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBlurFilterROISubmit, (NVCVImageHandle const, in), (NVCVImageHandle, out),
                             (NVCVTensorHandle const, rect), (size_t const, numRects));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDistanceTransformSubmit, (NVCVTensorHandle const, inImage), (uint16_t, maxDistance),
                             (NVCVTensorHandle, outDistance), (NVCVTensorHandle, outLabel));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBilateralFilterSubmit, (NVCVTensorHandle, in), (float const, sigmaRange),
                             (float const, sigmaSpace), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBruteForceMatcherSubmit, (PVABriefDescriptor *, query), (int32_t, queryCount),
                             (PVABriefDescriptor *, reference), (int32_t, referenceCount),
                             (int32_t, maxMatchesPerQuery), (PVABFMatchesType *, matches), (uint8_t, enableCrossCheck),
                             (uint8_t, enableDistanceRatioTest), (int32_t, lowesTestThresholdNumerator),
                             (int32_t, lowesTestThresholdDenominator));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageStatsSubmit, (NVCVImageHandle, in), (NVCVImageHandle, mask),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCannyEdgeDetectorSubmit, (cupvaCmdStatus_t *, cmdStatus), (NVCVTensorHandle, image),
                             (NVCVTensorHandle, edgeMap), (int32_t const, thresholdStrong),
                             (int32_t const, thresholdWeak));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMedianFilterSubmit, (NVCVImageHandle, in), (NVCVTensorHandle, kernel),
                             (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaHistogramEqualizationSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDLInferenceSubmit, (PvaDLInferenceSubmitParams *, params));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaWarpPerspectiveSubmit, (NVCVImageHandle, in), (WarpMatrixType, invWarpMatrix),
                             (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBackgroundSubtractorSubmit, (NVCVImageHandle, in), (NVCVImageHandle, outMask),
                             (NVCVImageHandle, outBackgroundImage), (float, learningRate), (float, varThreshold),
                             (bool, enableShadowDetection), (uint8_t, shadowPixelValue));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaRangeFFTSubmit, (NVCVTensorHandle const, in), (NVCVTensorHandle const, win),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDopplerFFTSubmit, (NVCVTensorHandle const, in), (NVCVTensorHandle const, win),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCoherentIntegrationSubmit, (NVCVTensorHandle const, rangeDopplerMap),
                             (NVCVTensorHandle const, steeringVectors), (NVCVTensorHandle, coherentIntegrationRx),
                             (NVCVTensorHandle, coherentIntegrationFinal));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaRadarTransposeSubmit, (NVCVTensorHandle const, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaNciSubmit, (NVCVTensorHandle const *, inHandles), (NVCVTensorHandle *, outHandles));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaRemapSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out),
                             (NVCVTensorHandle, warpMap));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaORBDescExtractorSubmit, (NVCVImageHandle, in),
                             (NVCVTensorHandle, inCornersTensorHandle), (NVCVTensorHandle, outDescriptorsTensorHandle));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageFlipSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out),
                             (PVAFlipDirection, flipDirection));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaRadarCFARSubmit, (NVCVTensorHandle const, inputTensor),
                             (NVCVTensorHandle, outDetectionListTensor), (NVCVTensorHandle, outDetectionCountTensor));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaFloydWarshallSubmit, (NVCVTensorHandle, lenTensor), (NVCVTensorHandle, nextTensor));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCCLSubmit, (NVCVTensorHandle, image), (NVCVTensorHandle, labels),
                             (NVCVTensorHandle, numLabels));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaHOGSubmit, (NVCVImageHandle const, in), (NVCVTensorHandle const, out));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaSnapshotExtractionSubmit, (NVCVTensorHandle const, detectionCount),
                             (NVCVTensorHandle const, foldedDetectionList), (NVCVTensorHandle const, ddmDopplerOffsets),
                             (NVCVTensorHandle const, nciRx), (NVCVTensorHandle const, rangeDopplerMap),
                             (NVCVTensorHandle const, calibrationWeights), (NVCVTensorHandle, detectionList),
                             (NVCVTensorHandle, snapshots));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDoaBartlettBeamformingSubmit, (NVCVTensorHandle const, detectionCount),
                             (NVCVTensorHandle const, snapshots), (NVCVTensorHandle const, steeringVectors),
                             (NVCVTensorHandle const, azimuthBins), (NVCVTensorHandle const, elevationBins),
                             (NVCVTensorHandle const, detectionList), (NVCVTensorHandle const, ddmDopplerOffsets),
                             (NVCVTensorHandle, targetCount), (NVCVTensorHandle, targetAngles),
                             (NVCVTensorHandle, targetIndexMap));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDoaCaponROISubmit, (NVCVTensorHandle const, snapshotsROI),
                             (NVCVTensorHandle const, steeringVectorsROI), (NVCVTensorHandle, peakPower),
                             (NVCVTensorHandle, azimuthBinIndices), (NVCVTensorHandle, elevationBinIndices));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCaponROIPreprocessSubmit, (NVCVTensorHandle const, waz), (NVCVTensorHandle const, wel),
                             (NVCVTensorHandle const, detectionCount), (NVCVTensorHandle const, detDopplers),
                             (NVCVTensorHandle const, elementSnapshots), (NVCVTensorHandle const, rangeDopplerPlanes),
                             (NVCVTensorHandle const, steeringLut), (NVCVTensorHandle, snapshotsROI),
                             (NVCVTensorHandle, steeringVectorsROI));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDoaDAFFTSubmit, (NVCVTensorHandle *, inHandles), (NVCVTensorHandle, outHandle),
                             (int, inTensorCount));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaPeakDetectionSubmit, (NVCVTensorHandle *, inHandles), (NVCVTensorHandle *, outHandles),
                             (int, inTensorCount), (int, outTensorCount));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDoaAngleFFTSubmit, (NVCVTensorHandle const, inDetectionCountTensor),
                             (NVCVTensorHandle const, inSnapshotsTensor),
                             (NVCVTensorHandle const, inDetectionListTensor),
                             (NVCVTensorHandle const, inDdmDopplerOffsetsTensor),
                             (NVCVTensorHandle, outTargetCountTensor), (NVCVTensorHandle, outTargetAnglesTensor),
                             (NVCVTensorHandle, outTargetIndexMapTensor));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaTargetProcessingSubmit, (NVCVTensorHandle const, targetCount),
                             (NVCVTensorHandle const, detectionCount), (NVCVTensorHandle const, detectionList),
                             (NVCVTensorHandle const, targetIndexMap), (NVCVTensorHandle const, targetAngles),
                             (NVCVTensorHandle const, ddmDopplerOffsets), (NVCVTensorHandle const, nci),
                             (NVCVTensorHandle, targetList));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaLidarHistogramConversionSubmit, (PvaLidarHistogramParams const *, params),
                             (NVCVTensorHandle const, histogram), (NVCVTensorHandle const, rangeBias),
                             (NVCVTensorHandle const, xyzCalibration), (NVCVTensorHandle, range),
                             (NVCVTensorHandle, xyz), (NVCVTensorHandle, reflectance));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaContourTracingSubmit, (NVCVTensorHandle, inTensor), (NVCVTensorHandle, rectTensor),
                             (NVCVTensorHandle, outTensor), (NVCVTensorHandle, lenTensor), (int32_t, numRects));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaNhwcToNchwLetterboxNormalizeSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaYoloV5DecodeSubmit, (NVCVTensorHandle, predictions), (NVCVTensorHandle, anchors),
                             (NVCVTensorHandle, boxes), (NVCVTensorHandle, scores), (NVCVTensorHandle, classes),
                             (NVCVTensorHandle, count));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBatchedNMSSubmit, (NVCVTensorHandle, inBoxes), (NVCVTensorHandle, inScores),
                             (NVCVTensorHandle, inClasses), (NVCVTensorHandle, inCount), (NVCVTensorHandle, outBoxes),
                             (NVCVTensorHandle, outScores), (NVCVTensorHandle, outClasses),
                             (NVCVTensorHandle, outCount));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaNmsSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBundleAdjustmentSubmit, (NVCVTensorHandle, rigPoses), (NVCVTensorHandle, points3d),
                             (NVCVTensorHandle, measurements), (NVCVTensorHandle, cameraBitmask),
                             (NVCVTensorHandle, rig2cam), (NVCVTensorHandle, updatedRigPoses),
                             (NVCVTensorHandle, updatedPoints3d), (NVCVTensorHandle, finalCost),
                             (NVCVTensorHandle, iterationsUsed), (PvaBundleAdjustmentParams const *, params));
#ifdef __cplusplus
}
#endif

// C-only: _Generic dispatch for type-based function selection
#ifndef __cplusplus
#    define pvaMorphologySubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMorphologySubmit, handle, stream, in, out)
#    define pvaImageHistogramSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageHistogramSubmit, handle, stream, in, out)
#    define pvaCornerSubPixSubmit(handle, stream, inCorners, outCorners, image, maxIters, eps, numCorners)             \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCornerSubPixSubmit, handle, stream, inCorners, outCorners, image, maxIters, eps, \
                                   numCorners)
#    define pvaDepthToSpaceSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDepthToSpaceSubmit, handle, stream, in, out)
#    define pvaMinMaxLocSubmit(handle, stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax)                 \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMinMaxLocSubmit, handle, stream, in, minVal, minLoc, numMin, maxVal, maxLoc, \
                                   numMax)
#    define pvaTemplateMatchingSubmit(handle, stream, inImage, inTemplate, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaTemplateMatchingSubmit, handle, stream, inImage, inTemplate, out)
#    define pvaMixChannelsSubmit(handle, stream, inHandles, outHandles, inTensorCount, outTensorCount)         \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMixChannelsSubmit, handle, stream, inHandles, outHandles, inTensorCount, \
                                   outTensorCount)
#    define pvaBlToPlSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBlToPlSubmit, handle, stream, in, out)
#    define pvaConv2dSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaConv2dSubmit, handle, stream, in, out)
#    define pvaGaussianFilterSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaGaussianFilterSubmit, handle, stream, in, out)
#    define pvaGaussianPyramidSubmit(handle, stream, inImageHandle, outImageHandles) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaGaussianPyramidSubmit, handle, stream, inImageHandle, outImageHandles)
#    define pvaBoxFilterSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBoxFilterSubmit, handle, stream, in, out)
#    define pvaConvertImageFormatSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaConvertImageFormatSubmit, handle, stream, in, out)
#    define pvaFastCornerDetectorSubmit(handle, stream, in, intensityThreshold, loc, numLoc) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaFastCornerDetectorSubmit, handle, stream, in, intensityThreshold, loc, numLoc)
#    define pvaImageBlendSubmit(handle, stream, in0, in1, alpha, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageBlendSubmit, handle, stream, in0, in1, alpha, out)
#    define pvaBatchSVDSubmit(handle, stream, src, U, S, V, maxIters, tol) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBatchSVDSubmit, handle, stream, src, U, S, V, maxIters, tol)
#    define pvaImageResizeSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageResizeSubmit, handle, stream, in, out)
#    define pvaBlurFilterROISubmit(handle, stream, in, out, rect, numRects) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBlurFilterROISubmit, handle, stream, in, out, rect, numRects)
#    define pvaDistanceTransformSubmit(handle, stream, inImage, maxDistance, outDistance, outLabel)               \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDistanceTransformSubmit, handle, stream, inImage, maxDistance, outDistance, \
                                   outLabel)
#    define pvaBilateralFilterSubmit(handle, stream, in, sigmaRange, sigmaSpace, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBilateralFilterSubmit, handle, stream, in, sigmaRange, sigmaSpace, out)
#    define pvaBruteForceMatcherSubmit(handle, stream, query, queryCount, reference, referenceCount,           \
                                       maxMatchesPerQuery, matches, enableCrossCheck, enableDistanceRatioTest, \
                                       lowesTestThresholdNumerator, lowesTestThresholdDenominator)             \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBruteForceMatcherSubmit, handle, stream, query, queryCount, reference,   \
                                   referenceCount, maxMatchesPerQuery, matches, enableCrossCheck,              \
                                   enableDistanceRatioTest, lowesTestThresholdNumerator,                       \
                                   lowesTestThresholdDenominator)
#    define pvaImageStatsSubmit(handle, stream, in, mask, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageStatsSubmit, handle, stream, in, mask, out)
#    define pvaCannyEdgeDetectorSubmit(handle, stream, cmdStatus, image, edgeMap, thresholdStrong, thresholdWeak) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCannyEdgeDetectorSubmit, handle, stream, cmdStatus, image, edgeMap,         \
                                   thresholdStrong, thresholdWeak)
#    define pvaMedianFilterSubmit(handle, stream, in, kernel, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMedianFilterSubmit, handle, stream, in, kernel, out)
#    define pvaHistogramEqualizationSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaHistogramEqualizationSubmit, handle, stream, in, out)
#    define pvaDLInferenceSubmit(handle, stream, params) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDLInferenceSubmit, handle, stream, params)
#    define pvaWarpPerspectiveSubmit(handle, stream, in, invWarpMatrix, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaWarpPerspectiveSubmit, handle, stream, in, invWarpMatrix, out)
#    define pvaBackgroundSubtractorSubmit(handle, stream, in, outMask, outBackgroundImage, learningRate, varThreshold, \
                                          enableShadowDetection, shadowPixelValue)                                     \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBackgroundSubtractorSubmit, handle, stream, in, outMask, outBackgroundImage,     \
                                   learningRate, varThreshold, enableShadowDetection, shadowPixelValue)
#    define pvaRangeFFTSubmit(handle, stream, in, win, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaRangeFFTSubmit, handle, stream, in, win, out)
#    define pvaDopplerFFTSubmit(handle, stream, in, win, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDopplerFFTSubmit, handle, stream, in, win, out)
#    define pvaCoherentIntegrationSubmit(handle, stream, rangeDopplerMap, steeringVectors, coherentIntegrationRx,  \
                                         coherentIntegrationFinal)                                                 \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCoherentIntegrationSubmit, handle, stream, rangeDopplerMap, steeringVectors, \
                                   coherentIntegrationRx, coherentIntegrationFinal)
#    define pvaRadarTransposeSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaRadarTransposeSubmit, handle, stream, in, out)
#    define pvaNciSubmit(handle, stream, inHandles, outHandles) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaNciSubmit, handle, stream, inHandles, outHandles)
#    define pvaRemapSubmit(handle, stream, in, out, warpMap) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaRemapSubmit, handle, stream, in, out, warpMap)
#    define pvaORBDescExtractorSubmit(handle, stream, in, inCornersTensorHandle, outDescriptorsTensorHandle) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaORBDescExtractorSubmit, handle, stream, in, inCornersTensorHandle,     \
                                   outDescriptorsTensorHandle)
#    define pvaImageFlipSubmit(handle, stream, in, out, flipDirection) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageFlipSubmit, handle, stream, in, out, flipDirection)
#    define pvaRadarCFARSubmit(handle, stream, inputTensor, outDetectionListTensor, outDetectionCountTensor) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaRadarCFARSubmit, handle, stream, inputTensor, outDetectionListTensor,  \
                                   outDetectionCountTensor)
#    define pvaFloydWarshallSubmit(handle, stream, lenTensor, nextTensor) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaFloydWarshallSubmit, handle, stream, lenTensor, nextTensor)
#    define pvaCCLSubmit(handle, stream, image, labels, numLabels) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCCLSubmit, handle, stream, image, labels, numLabels)
#    define pvaHOGSubmit(handle, stream, in, out) DEFINE_PVA_SUBMIT_DISPATCH(pvaHOGSubmit, handle, stream, in, out)
#    define pvaSnapshotExtractionSubmit(handle, stream, detectionCount, foldedDetectionList, ddmDopplerOffsets, nciRx, \
                                        rangeDopplerMap, calibrationWeights, detectionList, snapshots)                 \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaSnapshotExtractionSubmit, handle, stream, detectionCount, foldedDetectionList,   \
                                   ddmDopplerOffsets, nciRx, rangeDopplerMap, calibrationWeights, detectionList,       \
                                   snapshots)
#    define pvaDoaBartlettBeamformingSubmit(handle, stream, detectionCount, snapshots, steeringVectors, azimuthBins, \
                                            elevationBins, detectionList, ddmDopplerOffsets, targetCount,            \
                                            targetAngles, targetIndexMap)                                            \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDoaBartlettBeamformingSubmit, handle, stream, detectionCount, snapshots,       \
                                   steeringVectors, azimuthBins, elevationBins, detectionList, ddmDopplerOffsets,    \
                                   targetCount, targetAngles, targetIndexMap)
#    define pvaDoaCaponROISubmit(handle, stream, snapshotsROI, steeringVectorsROI, peakPower, azimuthBinIndices,      \
                                 elevationBinIndices)                                                                 \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDoaCaponROISubmit, handle, stream, snapshotsROI, steeringVectorsROI, peakPower, \
                                   azimuthBinIndices, elevationBinIndices)
#    define pvaCaponROIPreprocessSubmit(handle, stream, waz, wel, detectionCount, detDopplers, elementSnapshots,       \
                                        rangeDopplerPlanes, steeringLut, snapshotsROI, steeringVectorsROI)             \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCaponROIPreprocessSubmit, handle, stream, waz, wel, detectionCount, detDopplers, \
                                   elementSnapshots, rangeDopplerPlanes, steeringLut, snapshotsROI,                    \
                                   steeringVectorsROI)
#    define pvaDoaDAFFTSubmit(handle, stream, inHandles, outHandle, inTensorCount) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDoaDAFFTSubmit, handle, stream, inHandles, outHandle, inTensorCount)
#    define pvaPeakDetectionSubmit(handle, stream, inHandles, outHandles, inTensorCount, outTensorCount)         \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaPeakDetectionSubmit, handle, stream, inHandles, outHandles, inTensorCount, \
                                   outTensorCount)
#    define pvaDoaAngleFFTSubmit(handle, stream, inDetectionCountTensor, inSnapshotsTensor, inDetectionListTensor,  \
                                 inDdmDopplerOffsetsTensor, outTargetCountTensor, outTargetAnglesTensor,            \
                                 outTargetIndexMapTensor)                                                           \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDoaAngleFFTSubmit, handle, stream, inDetectionCountTensor, inSnapshotsTensor, \
                                   inDetectionListTensor, inDdmDopplerOffsetsTensor, outTargetCountTensor,          \
                                   outTargetAnglesTensor, outTargetIndexMapTensor)
#    define pvaTargetProcessingSubmit(handle, stream, targetCount, detectionCount, detectionList, targetIndexMap, \
                                      targetAngles, ddmDopplerOffsets, nci, targetList)                           \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaTargetProcessingSubmit, handle, stream, targetCount, detectionCount,        \
                                   detectionList, targetIndexMap, targetAngles, ddmDopplerOffsets, nci, targetList)
#    define pvaLidarHistogramConversionSubmit(handle, stream, params, histogram, rangeBias, xyzCalibration, range,  \
                                              xyz, reflectance)                                                     \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaLidarHistogramConversionSubmit, handle, stream, params, histogram, rangeBias, \
                                   xyzCalibration, range, xyz, reflectance)
#    define pvaContourTracingSubmit(handle, stream, inTensor, rectTensor, outTensor, lenTensor, numRects)    \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaContourTracingSubmit, handle, stream, inTensor, rectTensor, outTensor, \
                                   lenTensor, numRects)
#    define pvaNhwcToNchwLetterboxNormalizeSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaNhwcToNchwLetterboxNormalizeSubmit, handle, stream, in, out)
#    define pvaYoloV5DecodeSubmit(handle, stream, predictions, anchors, boxes, scores, classes, count)         \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaYoloV5DecodeSubmit, handle, stream, predictions, anchors, boxes, scores, \
                                   classes, count)
#    define pvaBatchedNMSSubmit(handle, stream, inBoxes, inScores, inClasses, inCount, outBoxes, outScores,    \
                                outClasses, outCount)                                                          \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBatchedNMSSubmit, handle, stream, inBoxes, inScores, inClasses, inCount, \
                                   outBoxes, outScores, outClasses, outCount)
#    define pvaNmsSubmit(handle, stream, in, out) DEFINE_PVA_SUBMIT_DISPATCH(pvaNmsSubmit, handle, stream, in, out)
#    define pvaBundleAdjustmentSubmit(handle, stream, rigPoses, points3d, measurements, cameraBitmask, rig2cam, \
                                      updatedRigPoses, updatedPoints3d, finalCost, iterationsUsed, params)      \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBundleAdjustmentSubmit, handle, stream, rigPoses, points3d, measurements, \
                                   cameraBitmask, rig2cam, updatedRigPoses, updatedPoints3d, finalCost,         \
                                   iterationsUsed, params)
#endif

#endif /* PVA_SOLUTIONS_PVASUBMITDEFINITIONS_H */
