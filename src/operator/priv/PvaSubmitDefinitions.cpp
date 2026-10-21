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

#include <priv/PvaSubmitDefinitions.h>
#include <priv/PvaSubmitMacros.h>

// When PVA_FILTER_OP is defined (single-op build), only the guarded op's wrappers are compiled.
// Each guard is: PVA_OP_FILTER_<UPPER_OP_SHORT_NAME>  e.g. PVA_OP_FILTER_MORPHOLOGY
// This prevents undefined-reference linker errors from operators not included in the filtered build.
#define PVA_SUBMIT_IF(guard) !defined(PVA_FILTER_OP) || defined(PVA_OP_FILTER_##guard)

#ifdef __cplusplus
extern "C" {
#endif

// Generate __cupva/__cuda wrappers that call the internal *SubmitImpl functions
// Only include operators that have been updated with template SubmitImpl functions

#if PVA_SUBMIT_IF(MORPHOLOGY)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaMorphologySubmit, MorphologySubmitImpl, (NVCVTensorHandle, in), (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(IMAGE_HISTOGRAM)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaImageHistogramSubmit, ImageHistogramSubmitImpl, (NVCVTensorHandle const, in),
                            (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(CORNER_SUBPIX)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaCornerSubPixSubmit, CornerSubPixSubmitImpl, (NVCVTensorHandle, inCorners),
                            (NVCVTensorHandle, outCorners), (NVCVTensorHandle, image), (int32_t const, maxIters),
                            (float const, eps), (int32_t const, numCorners))
#endif

#if PVA_SUBMIT_IF(DEPTHTOSPACE)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDepthToSpaceSubmit, DepthToSpaceSubmitImpl, (NVCVTensorHandle, in),
                            (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(MINMAXLOC)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaMinMaxLocSubmit, MinMaxLocSubmitImpl, (NVCVTensorHandle, in), (NVCVTensorHandle, minVal),
                            (NVCVTensorHandle, minLoc), (NVCVTensorHandle, numMin), (NVCVTensorHandle, maxVal),
                            (NVCVTensorHandle, maxLoc), (NVCVTensorHandle, numMax))
#endif

#if PVA_SUBMIT_IF(TEMPLATE_MATCHING)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaTemplateMatchingSubmit, TemplateMatchingSubmitImpl, (NVCVTensorHandle, inImage),
                            (NVCVTensorHandle, inTemplate), (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(MIX_CHANNELS)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaMixChannelsSubmit, MixChannelsSubmitImpl, (NVCVTensorHandle *, inHandles),
                            (NVCVTensorHandle *, outHandles), (int, inTensorCount), (int, outTensorCount))
#endif

#if PVA_SUBMIT_IF(BLTOPL)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBlToPlSubmit, BlToPlSubmitImpl, (NVCVImageHandle, in), (NVCVImageHandle, out))
#endif

#if PVA_SUBMIT_IF(GAUSSIAN_PYRAMID)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaGaussianPyramidSubmit, GaussianPyramidSubmitImpl, (NVCVImageHandle, inImageHandle),
                            (NVCVImageHandle *, outImageHandles))
#endif

#if PVA_SUBMIT_IF(RADAR_CFAR)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaRadarCFARSubmit, RadarCFARSubmitImpl, (NVCVTensorHandle const, inputTensor),
                            (NVCVTensorHandle, outDetectionListTensor), (NVCVTensorHandle, outDetectionCountTensor))
#endif

#if PVA_SUBMIT_IF(BACKGROUND_SUBTRACTOR)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBackgroundSubtractorSubmit, BackgroundSubtractorSubmitImpl, (NVCVImageHandle, in),
                            (NVCVImageHandle, outMask), (NVCVImageHandle, outBackgroundImage), (float, learningRate),
                            (float, varThreshold), (bool, enableShadowDetection), (uint8_t, shadowPixelValue))
#endif

#if PVA_SUBMIT_IF(RANGE_FFT)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaRangeFFTSubmit, RangeFFTSubmitImpl, (NVCVTensorHandle const, in),
                            (NVCVTensorHandle const, win), (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(DOPPLER_FFT)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDopplerFFTSubmit, DopplerFFTSubmitImpl, (NVCVTensorHandle const, in),
                            (NVCVTensorHandle const, win), (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(COHERENT_INTEGRATION)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaCoherentIntegrationSubmit, CoherentIntegrationSubmitImpl,
                            (NVCVTensorHandle const, rangeDopplerMap), (NVCVTensorHandle const, steeringVectors),
                            (NVCVTensorHandle, coherentIntegrationRx), (NVCVTensorHandle, coherentIntegrationFinal))
#endif

#if PVA_SUBMIT_IF(RADAR_TRANSPOSE)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaRadarTransposeSubmit, RadarTransposeSubmitImpl, (NVCVTensorHandle const, in),
                            (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(NCI)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaNciSubmit, NciSubmitImpl, (NVCVTensorHandle const *, inHandles),
                            (NVCVTensorHandle *, outHandles))
#endif

#if PVA_SUBMIT_IF(BILATERAL_FILTER)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBilateralFilterSubmit, BilateralFilterSubmitImpl, (NVCVTensorHandle, in),
                            (float const, sigmaRange), (float const, sigmaSpace), (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(BLUR_FILTER_ROI)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBlurFilterROISubmit, BlurFilterROISubmitImpl, (NVCVImageHandle const, in),
                            (NVCVImageHandle, out), (NVCVTensorHandle const, rect), (size_t const, numRects))
#endif

#if PVA_SUBMIT_IF(BRUTE_FORCE_MATCHER)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBruteForceMatcherSubmit, BruteForceMatcherSubmitImpl, (PVABriefDescriptor *, query),
                            (int32_t, queryCount), (PVABriefDescriptor *, reference), (int32_t, referenceCount),
                            (int32_t, maxMatchesPerQuery), (PVABFMatchesType *, matches), (uint8_t, enableCrossCheck),
                            (uint8_t, enableDistanceRatioTest), (int32_t, lowesTestThresholdNumerator),
                            (int32_t, lowesTestThresholdDenominator))
#endif

#if PVA_SUBMIT_IF(CANNY_EDGE_DETECTOR)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaCannyEdgeDetectorSubmit, CannyEdgeDetectorSubmitImpl, (cupvaCmdStatus_t *, cmdStatus),
                            (NVCVTensorHandle, image), (NVCVTensorHandle, edgeMap), (int32_t const, thresholdStrong),
                            (int32_t const, thresholdWeak))
#endif

#if PVA_SUBMIT_IF(CONVERT_IMAGE_FORMAT)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaConvertImageFormatSubmit, ConvertImageFormatSubmitImpl, (NVCVImageHandle const, in),
                            (NVCVImageHandle, out))
#endif

#if PVA_SUBMIT_IF(DISTANCE_TRANSFORM)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDistanceTransformSubmit, DistanceTransformSubmitImpl, (NVCVTensorHandle const, inImage),
                            (uint16_t, maxDistance), (NVCVTensorHandle, outDistance), (NVCVTensorHandle, outLabel))
#endif

#if PVA_SUBMIT_IF(HISTOGRAM_EQUALIZATION)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaHistogramEqualizationSubmit, HistogramEqualizationSubmitImpl, (NVCVTensorHandle, in),
                            (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(IMAGE_BLEND)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaImageBlendSubmit, ImageBlendSubmitImpl, (NVCVImageHandle const, in0),
                            (NVCVImageHandle const, in1), (float const, alpha), (NVCVImageHandle, out))
#endif

#if PVA_SUBMIT_IF(BATCH_SVD)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBatchSVDSubmit, BatchSVDSubmitImpl, (NVCVTensorHandle const, src),
                            (NVCVTensorHandle const, U), (NVCVTensorHandle const, S), (NVCVTensorHandle const, V),
                            (int32_t const, maxIters), (float const, tol))
#endif

#if PVA_SUBMIT_IF(IMAGE_RESIZE)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaImageResizeSubmit, ImageResizeSubmitImpl, (NVCVImageHandle, in), (NVCVImageHandle, out))
#endif

#if PVA_SUBMIT_IF(IMAGE_STATS)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaImageStatsSubmit, ImageStatsSubmitImpl, (NVCVImageHandle, in), (NVCVImageHandle, mask),
                            (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(MEDIAN_FILTER)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaMedianFilterSubmit, MedianFilterSubmitImpl, (NVCVImageHandle, in),
                            (NVCVTensorHandle, kernel), (NVCVImageHandle, out))
#endif

#if PVA_SUBMIT_IF(ORB_DESC_EXTRACTOR)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaORBDescExtractorSubmit, ORBDescExtractorSubmitImpl, (NVCVImageHandle, in),
                            (NVCVTensorHandle, inCornersTensorHandle), (NVCVTensorHandle, outDescriptorsTensorHandle))
#endif

#if PVA_SUBMIT_IF(IMAGE_FLIP)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaImageFlipSubmit, ImageFlipSubmitImpl, (NVCVImageHandle, in), (NVCVImageHandle, out),
                            (PVAFlipDirection, flipDirection))
#endif

#if PVA_SUBMIT_IF(REMAP)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaRemapSubmit, RemapSubmitImpl, (NVCVImageHandle, in), (NVCVImageHandle, out),
                            (NVCVTensorHandle, warpMap))
#endif

#if PVA_SUBMIT_IF(WARP_PERSPECTIVE)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaWarpPerspectiveSubmit, WarpPerspectiveSubmitImpl, (NVCVImageHandle, in),
                            (WarpMatrixType, invWarpMatrix), (NVCVImageHandle, out))
#endif

#if PVA_SUBMIT_IF(DL_INFERENCE)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDLInferenceSubmit, DLInferenceSubmitImpl, (PvaDLInferenceSubmitParams *, params))
#endif

#if PVA_SUBMIT_IF(FLOYD_WARSHALL)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaFloydWarshallSubmit, FloydWarshallSubmitImpl, (NVCVTensorHandle, lenTensor),
                            (NVCVTensorHandle, nextTensor))
#endif

#if PVA_SUBMIT_IF(CCL)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaCCLSubmit, CCLSubmitImpl, (NVCVTensorHandle, image), (NVCVTensorHandle, labels),
                            (NVCVTensorHandle, numLabels))
#endif

#if PVA_SUBMIT_IF(HOG)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaHOGSubmit, HOGSubmitImpl, (NVCVImageHandle const, in), (NVCVTensorHandle const, out))
#endif

#if PVA_SUBMIT_IF(NHWC_TO_NCHW_LETTERBOX_NORMALIZE)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaNhwcToNchwLetterboxNormalizeSubmit, NhwcToNchwLetterboxNormalizeSubmitImpl,
                            (NVCVTensorHandle, in), (NVCVTensorHandle, out))
#endif

#if PVA_SUBMIT_IF(YOLOV5_DECODE)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaYoloV5DecodeSubmit, YoloV5DecodeSubmitImpl, (NVCVTensorHandle, predictions),
                            (NVCVTensorHandle, anchors), (NVCVTensorHandle, boxes), (NVCVTensorHandle, scores),
                            (NVCVTensorHandle, classes), (NVCVTensorHandle, count))
#endif

#if PVA_SUBMIT_IF(SNAPSHOT_EXTRACTION)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaSnapshotExtractionSubmit, SnapshotExtractionSubmitImpl,
                            (NVCVTensorHandle const, detectionCount), (NVCVTensorHandle const, foldedDetectionList),
                            (NVCVTensorHandle const, ddmDopplerOffsets), (NVCVTensorHandle const, nciRx),
                            (NVCVTensorHandle const, rangeDopplerMap), (NVCVTensorHandle const, calibrationWeights),
                            (NVCVTensorHandle, detectionList), (NVCVTensorHandle, snapshots))
#endif

#if PVA_SUBMIT_IF(BARTLETT)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDoaBartlettBeamformingSubmit, DoaBartlettBeamformingSubmitImpl,
                            (NVCVTensorHandle const, detectionCount), (NVCVTensorHandle const, snapshots),
                            (NVCVTensorHandle const, steeringVectors), (NVCVTensorHandle const, azimuthBins),
                            (NVCVTensorHandle const, elevationBins), (NVCVTensorHandle const, detectionList),
                            (NVCVTensorHandle const, ddmDopplerOffsets), (NVCVTensorHandle, targetCount),
                            (NVCVTensorHandle, targetAngles), (NVCVTensorHandle, targetIndexMap))
#endif

#if PVA_SUBMIT_IF(CAPON_ROI_PREPROCESS)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaCaponROIPreprocessSubmit, CaponROIPreprocessSubmitImpl, (NVCVTensorHandle const, waz),
                            (NVCVTensorHandle const, wel), (NVCVTensorHandle const, detectionCount),
                            (NVCVTensorHandle const, detDopplers), (NVCVTensorHandle const, elementSnapshots),
                            (NVCVTensorHandle const, rangeDopplerPlanes), (NVCVTensorHandle const, steeringLut),
                            (NVCVTensorHandle, snapshotsROI), (NVCVTensorHandle, steeringVectorsROI))
#endif

#if PVA_SUBMIT_IF(DOA_DA_FFT)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDoaDAFFTSubmit, DoaDAFFTSubmitImpl, (NVCVTensorHandle *, inHandles),
                            (NVCVTensorHandle, outHandle), (int, inTensorCount))
#endif

#if PVA_SUBMIT_IF(PEAK_DETECTION)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaPeakDetectionSubmit, PeakDetectionSubmitImpl, (NVCVTensorHandle *, inHandles),
                            (NVCVTensorHandle *, outHandles), (int, inTensorCount), (int, outTensorCount))
#endif

#if PVA_SUBMIT_IF(DOA_ANGLE_FFT)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDoaAngleFFTSubmit, DoaAngleFFTSubmitImpl,
                            (NVCVTensorHandle const, inDetectionCountTensor),
                            (NVCVTensorHandle const, inSnapshotsTensor),
                            (NVCVTensorHandle const, inDetectionListTensor),
                            (NVCVTensorHandle const, inDdmDopplerOffsetsTensor),
                            (NVCVTensorHandle, outTargetCountTensor), (NVCVTensorHandle, outTargetAnglesTensor),
                            (NVCVTensorHandle, outTargetIndexMapTensor))
#endif

#if PVA_SUBMIT_IF(TARGET_PROCESSING)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaTargetProcessingSubmit, TargetProcessingSubmitImpl,
                            (NVCVTensorHandle const, targetCount), (NVCVTensorHandle const, detectionCount),
                            (NVCVTensorHandle const, detectionList), (NVCVTensorHandle const, targetIndexMap),
                            (NVCVTensorHandle const, targetAngles), (NVCVTensorHandle const, ddmDopplerOffsets),
                            (NVCVTensorHandle const, nci), (NVCVTensorHandle, targetList))
#endif

#if PVA_SUBMIT_IF(BUNDLE_ADJUSTMENT)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBundleAdjustmentSubmit, BundleAdjustmentSubmitImpl, (NVCVTensorHandle, rigPoses),
                            (NVCVTensorHandle, points3d), (NVCVTensorHandle, measurements),
                            (NVCVTensorHandle, cameraBitmask), (NVCVTensorHandle, rig2cam),
                            (NVCVTensorHandle, updatedRigPoses), (NVCVTensorHandle, updatedPoints3d),
                            (NVCVTensorHandle, finalCost), (NVCVTensorHandle, iterationsUsed),
                            (PvaBundleAdjustmentParams const *, params))
#endif

#if CUPVA_LLVM_SUPPORTED == 1
// LLVM Operators

#    if PVA_SUBMIT_IF(CAPON_ROI)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaDoaCaponROISubmit, DoaCaponROISubmitImpl, (NVCVTensorHandle const, snapshotsROI),
                            (NVCVTensorHandle const, steeringVectorsROI), (NVCVTensorHandle, peakPower),
                            (NVCVTensorHandle, azimuthBinIndices), (NVCVTensorHandle, elevationBinIndices))
#    endif

#    if PVA_SUBMIT_IF(CONV2D)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaConv2dSubmit, Conv2dSubmitImpl, (NVCVTensorHandle, in), (NVCVTensorHandle, out))
#    endif

#    if PVA_SUBMIT_IF(GAUSSIAN_FILTER)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaGaussianFilterSubmit, GaussianFilterSubmitImpl, (NVCVTensorHandle, in),
                            (NVCVTensorHandle, out))
#    endif

#    if PVA_SUBMIT_IF(BOX_FILTER)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBoxFilterSubmit, BoxFilterSubmitImpl, (NVCVTensorHandle, in), (NVCVTensorHandle, out))
#    endif

#    if PVA_SUBMIT_IF(FASTCORNER)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaFastCornerDetectorSubmit, FastCornerDetectorSubmitImpl, (NVCVTensorHandle, in),
                            (int32_t, intensityThreshold), (NVCVTensorHandle, loc), (NVCVTensorHandle, numLoc))
#    endif

#    if PVA_SUBMIT_IF(NMS)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaNmsSubmit, NmsSubmitImpl, (NVCVTensorHandle, in), (NVCVTensorHandle, out))
#    endif

// contour_tracing requires cuPVA >= 2.7.0 (runtime version 2007)
#    if CUPVA_RUNTIME_VERSION >= 2007 && (PVA_SUBMIT_IF(CONTOUR_TRACING))
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaContourTracingSubmit, ContourTracingSubmitImpl, (NVCVTensorHandle, inTensor),
                            (NVCVTensorHandle, rectTensor), (NVCVTensorHandle, outTensor),
                            (NVCVTensorHandle, lenTensor), (int32_t, numRects))
#    endif

/// NB: lidar doesn't depend on LLVM exactly, but it needs CUPVA 2.6+ which happens to mean the same thing
#    if PVA_SUBMIT_IF(LIDAR_HISTOGRAM_CONVERSION)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaLidarHistogramConversionSubmit, LidarHistogramConversionSubmitImpl,
                            (PvaLidarHistogramParams const *, params), (NVCVTensorHandle const, histogram),
                            (NVCVTensorHandle const, rangeBias), (NVCVTensorHandle const, xyzCalibration),
                            (NVCVTensorHandle, range), (NVCVTensorHandle, xyz), (NVCVTensorHandle, reflectance))
#    endif

#    if PVA_SUBMIT_IF(BATCHED_NMS)
DEFINE_PVA_SUBMIT_FUNCTIONS(pvaBatchedNMSSubmit, BatchedNMSSubmitImpl, (NVCVTensorHandle, inBoxes),
                            (NVCVTensorHandle, inScores), (NVCVTensorHandle, inClasses), (NVCVTensorHandle, inCount),
                            (NVCVTensorHandle, outBoxes), (NVCVTensorHandle, outScores), (NVCVTensorHandle, outClasses),
                            (NVCVTensorHandle, outCount))
#    endif

#endif // CUPVA_LLVM_SUPPORTED

#ifdef __cplusplus
} // extern "C"
#endif
