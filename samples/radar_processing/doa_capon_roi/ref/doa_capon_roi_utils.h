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
 * @file doa_capon_roi_utils.h
 * @brief Host utilities for the Capon DoA on ROI samples and tests.
 */

#pragma once

#include <PvaOperatorTypes.h>
#include <nvcv/Tensor.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Defaults shared by the samples and system tests. */
#define ADP_DIAG_CONST 1e-2f
#define NOISE_FLOOR_RATIO 1.58489323f /* 10^(2.0/10) */

/* Output comparison tolerances. */
#define TOLERANCE_POWER_REL 1e-4f
#define TOLERANCE_VPU_VS_REF_BIN_ABS 0.05f
#define TOLERANCE_REF_VS_GOLD_OFFSET_ABS 0.15f

typedef struct
{
    NVCVAllocatorHandle allocatorHandle;

    NVCVTensorHandle snapshotsROITensor;
    NVCVTensorHandle svTensor;

    NVCVTensorHandle peakPowerTensor;
    NVCVTensorHandle azBinTensor;
    NVCVTensorHandle elBinTensor;

    NVCVTensorHandle refPeakPowerTensor;
    NVCVTensorHandle refAzBinTensor;
    NVCVTensorHandle refElBinTensor;

    NVCVTensorRequirements snapshotsReq;
    NVCVTensorRequirements svReq;
    NVCVTensorRequirements peakPowerReq;
    NVCVTensorRequirements azBinReq;
    NVCVTensorRequirements elBinReq;
} CaponROITensors;

/* Every Capon ROI dataset (capon_roi/<name>/) uses the same layout and fixed-point formats. */
#define CAPON_ASSET_NUM_SAMPLES 64 /**< snapshots per ROI (Doppler bins) */
#define CAPON_ASSET_SV_DIM 16      /**< beamspace channels per ROI (4 x 4) */
#define CAPON_ASSET_NUM_AZ_BINS 16 /**< steering-grid azimuth directions */
#define CAPON_ASSET_NUM_EL_BINS 8  /**< steering-grid elevation directions */
#define CAPON_ASSET_Q_BITS 20      /**< inputQBits and steeringQBits of the quantized inputs */

/**
 * @brief Set the operator and dataset values for a Capon ROI dataset from its inputs.bin.
 *
 * inputs.bin is float32, C order, ROI outermost, re/im innermost:
 *  - snapshots [roiNum, numSamples, svDim, 2] (svDim = az * roiElSize + el),
 *  - steering  [roiNum, svDim, numElBins * numAzBins, 2] (elevation-major),
 *  - one float holding roiNum.
 * outputs.bin (golden peaks) is float32 [roiNum, 3 slots, 5 fields] (power, az argmax, el argmax,
 * az offset, el offset; NaN power = no peak) followed by one float holding the peak count.
 *
 * roiNum is derived from the size of inputs.bin and checked against its trailing count; the other
 * values are the CAPON_ASSET_* constants. @p roiNum and @p numSamples may be NULL.
 *
 * @return 0 on success, -1 on failure.
 */
int caponLoadDatasetConfig(const char *inputsBinPath, PVADoaCaponROIParams *params, int32_t *roiNum,
                           int32_t *numSamples);

/**
 * @brief Allocate input, intermediate, device-output, and reference-output tensors.
 *
 * The snapshot tensor height is @p params->numDopplerBinsCovMat. Call
 * caponROIFreeTensors even when allocation fails, because the structure may contain partial results.
 *
 * @return 0 on success, otherwise an NVCVStatus error code.
 */
int caponROIAllocateTensors(CaponROITensors *t, const PVADoaCaponROIParams *params, int32_t roiNum);

/** @brief Release every handle in @p t. */
void caponROIFreeTensors(CaponROITensors *t);

/**
 * @brief Load and quantize snapshots and steering vectors from inputs.bin.
 *
 * The file contains contiguous float32 snapshots
 * [assetRoiNum, numSamples, svDim, 2], steering vectors
 * [assetRoiNum, svDim, numElBins, numAzBins, 2], and one ignored detection
 * count. The stored Doppler rows are center-sliced when the snapshot tensor is
 * shorter and repeated cyclically when it is taller. Requests larger than
 * @p assetRoiNum repeat stored ROIs cyclically.
 *
 * @return 0 on success, -1 on failure.
 */
int caponLoadInputsFromFlatBin(const char *inputsBinPath, PVADoaCaponROIParams *params, int32_t numSamples,
                               int32_t assetRoiNum, int32_t roiNum, NVCVTensorHandle snapshotsHandle,
                               NVCVTensorHandle svHandle);

/**
 * @brief Compare VPU outputs against reference outputs.
 *
 * Power uses relative tolerance. Fractional bins use @p binAbsTol when
 * interpolation is enabled; integer bins must match exactly otherwise.
 *
 * @return Number of mismatched output values.
 */
int caponCompareVpuVsRef(NVCVTensorHandle vpuPower, NVCVTensorHandle refPower, NVCVTensorHandle vpuAz,
                         NVCVTensorHandle refAz, NVCVTensorHandle vpuEl, NVCVTensorHandle refEl, int32_t roiNum,
                         int32_t numPeaksPerROI, float powerRelTol, float binAbsTol, bool enableSubBinInterp);

#ifdef __cplusplus
} /* extern "C" */
#endif
