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

#ifndef DOA_TARGET_PROCESSING_REF_H
#define DOA_TARGET_PROCESSING_REF_H

#include <PvaOperatorTypes.h>
#include <nvcv/Tensor.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Populates the GP structure with the default values.
 *
 * @param GP Pointer to the GP structure to populate.
 * @param calibVectorQbits Number of Q bits for the calib vector.
 * @param fftQbits Number of Q bits for the FFT tensor.
 */
void populateGP(PVARadarGP *GP, int32_t calibVectorQbits, int32_t fftQbits);

/**
 * @brief Populates the calib vector with the default values.
 *
 * @param calibVector Pointer to the calib vector to populate.
 * @param calibVectorQbits Number of Q bits for the calib vector.
 */
void populateCalibVector(int32_t *calibVector, int32_t calibVectorQbits);

/**
 * @brief Populates the calib vector with the identity (1 + 0j on every channel), for ideal
 *        channels such as synthetic captures. Same layout as populateCalibVector().
 *
 * @param calibVector Pointer to the calib vector to populate.
 * @param calibVectorQbits Number of Q bits for the calib vector.
 */
void populateIdentityCalibVector(int32_t *calibVector, int32_t calibVectorQbits);

/**
 * @brief Loads the peak count to a tensor.
 *
 * @param peakCountHandle The handle to the peak count tensor.
 * @param numPeaks The number of peaks.
 * @return Returns 0 on success, -1 on failure.
 */
int32_t loadPeakCountToTensor(NVCVTensorHandle peakCountHandle, int32_t numPeaks);

/**
 * @brief Loads peak data in Peak_t_cpp format from a raw file.
 *
 * This function reads a binary file specified by filename and populates a
 * Peak_t_cpp structure. The file is expected to be in the format generated
 * by the corresponding MATLAB script (write_peak_data.m).
 *
 * @param[in] peakIndicesHandle The handle to the peak indices tensor.
 * @param[in] peakSnapHandle The handle to the peak snap tensor.
 * @param[in] assetsDirectory The directory containing the input raw file.
 * @param[in] filename The name of the input raw file.
 * @param[in] numPeaks The number of peaks to load.
 * @return Returns 0 on successful loading, -1 otherwise.
 */
int32_t loadPeakDataFromRawFile(NVCVTensorHandle peakIndicesHandle, NVCVTensorHandle peakSnapHandle,
                                const char *assetsDirectory, const char *filename, int32_t numPeaks);

/**
 * @brief Loads the calibration vector data into a tensor.
 *
 * @param[in] calibTensorHandle The handle to the calibration vector tensor.
 * @param[in] calibVector The calibration vector data.
 * @param[in] numElements The number of complex elements in the calibration vector.
 * @return Returns 0 on success, -1 otherwise.
 */
int32_t loadCalibVectorToTensorFixedPoint(NVCVTensorHandle calibTensorHandle, const int32_t calibVector[],
                                          int32_t numElements);

/**
 * @brief Loads the NCI FINAL data from a raw file into a tensor.
 *
 * @param[in] nciTensorHandle The handle to the NCI FINAL tensor.
 * @param[in] assetsDirectory The directory containing the input raw file.
 * @param[in] filename The name of the input raw file.
 * @return Returns 0 on success, -1 otherwise.
 */
int32_t loadNciFinalData(NVCVTensorHandle nciTensorHandle, const char *assetsDirectory, const char *filename);

/**
 * @brief Main reference implementation for Direction of Arrival (DOA) target processing.
 *
 * This function performs the complete DOA target processing pipeline using the reference
 * implementation. It processes radar data to detect and localize targets by calculating
 * their azimuth and elevation angles, range, velocity, and 3D coordinates.
 *
 * The function expects input tensors in the following order:
 * - Tensor 0: Calibration vector (std::pair<int16_t, int16_t>)
 * - Tensor 1: Peak indices (uint32_t)
 * - Tensor 2: Peak snapshots (std::pair<int32_t, int32_t>)
 * - Tensor 3: NCI final data (int32_t)
 *
 * The output tensor contains target detection results with the following layout:
 * - Velocity values
 * - Range values
 * - Azimuth angles
 * - Elevation angles
 * - X coordinates
 * - Y coordinates
 * - Z coordinates
 *
 * @param[in] inTensorHandles Array of input tensor handles containing radar data.
 * @param[in] outTensorHandle Output tensor handle for storing target detection results.
 * @param[in] GP Pointer to the radar processing parameters structure.
 */
void doaTargetProcessingRef(NVCVTensorHandle *inTensorHandles, NVCVTensorHandle outTensorHandle, PVARadarGP *GP);

/**
 * @brief Writes TAR results to a raw file.
 *
 * @param outTensorHandle The handle to the output tensor containing TAR results.
 * @param numPeaks The number of peaks.
 * @param outputFile The name of the output file.
 * @return Returns 0 on success, -1 on failure.
 */
int32_t writeTarResultsToFile(NVCVTensorHandle outTensorHandle, int32_t numPeaks, const char *outputFile);

/**
 * @brief Compares the output tensor with the reference tensor.
 *
 * @param[in] outTensorHandle The handle to the output tensor.
 * @param[in] refTensorHandle The handle to the reference tensor.
 * @param[in] GP Pointer to the GP parameters structure.
 * @param[in] numPeaks The number of peaks.
 * @param[in] tolerance The tolerance for the comparison.
 * @return Returns 0 on success, -1 on failure.
 */
int32_t compareOutputWithReference(NVCVTensorHandle outTensorHandle, NVCVTensorHandle refTensorHandle,
                                   PVARadarGP const *GP, int32_t numPeaks, float tolerance);

/**
 * @brief Compares the output tensor with the reference tensor.
 * This function is a simplified version of compareOutputWithReference that only checks the range and azimuth values.
 *
 * @param[in] outTensorHandle The handle to the output tensor.
 * @param[in] refTensorHandle The handle to the reference tensor.
 * @param[in] GP Pointer to the GP parameters structure.
 * @param[in] numPeaks The number of peaks.
 * @param[in] tolerance The tolerance for the comparison.
 * @return Returns 0 on success, -1 on failure.
 */
int32_t quickCompareOutputWithReference(NVCVTensorHandle outTensorHandle, NVCVTensorHandle refTensorHandle,
                                        PVARadarGP const *GP, int32_t numPeaks, float tolerance);
#ifdef __cplusplus
}
#endif

#endif
