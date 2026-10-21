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

/**
 * @file OpNci.h
 *
 * @brief Defines types and functions to handle the NCI operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_NCI RadarNCI (Non-Coherent Integration)
 * @{
 */

#ifndef PVA_SOLUTIONS_OPNCI_H
#define PVA_SOLUTIONS_OPNCI_H
#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Parameters for the NCI operator.
 */
typedef struct
{
    /// Number of Doppler folds. Limitations: Must be at least 4 and divisible by 4. Number of Doppler bins in a fold must be at least 32 and divisible by 32.
    int32_t repeatFold;
    /// Boolean flag to enable the noise estimate. If disabled, noise estimation will not be performed.
    bool noiseEstimationEnabled;
    /// Input tensor layout: distinguishes between [Nb][NofRx][Nr] and [Nb][Nr][NofRx]. Limitations: must be
    /// PVA_FFT_LAYOUT_RANGE_RX_DOPPLER or PVA_FFT_LAYOUT_RANGE_DOPPLER_RX.
    PVAFFTLayout inputLayout;
} PVARadarNCIParams;

/**
  * Constructs an instance of the NCI operator.
  *
  * @param [out] handle Pointer to where the operator instance handle will be written.
  *                     Must not be NULL.
  *
  * @param [in] inTensorRequirements Pointer to a NVCVTensorRequirements structure containing tensor rank, shape, layout, and data type information.
  *                                 + The inTensorRequirements corresponds to the dimensions of the FT2D tensor
  *
  * The tensor has the following dimensions:
  * Nb - number of range bins (samples), Nr - number of Doppler bins (ramps), NofRx - number of RX antennas,
  * Limitations: NofRx must be a power of 2 with a maximum of 64.
  *              Nr - number of Doppler bins (ramps) should be a multiple of NofRx
  *
  * \b FT2D: Data type: 2S32
  *      Data Layout:    [HCW, NHCW]
  *      If Data Layout is PVA_FFT_LAYOUT_RANGE_RX_DOPPLER:
  *          H: Number of range bins.
  *          C: Number of RX channels.
  *          W: Number of Doppler bins.
  *      If Data Layout is PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
  *          H: Number of range bins.
  *          C: Number of Doppler bins.
  *          W: Number of RX channels.
  *
  * @param [in] outTensorRequirements Pointer to 3 NVCVTensorRequirements structure pointers containing tensor rank, shape, layout, and data type information.
  *
  + Tensor 0: NciRx
  * The tensor has the following dimensions:
  * Nb - number of range bins (samples), Nr - number of Doppler bins (ramps)
  * NciRx: uint32_t [Nb][Nr]
  * Data type: U32
  * Data layout: HW, H – number of range bins, W – number of Doppler bins
  *
  * Tensor 1: NciFinal
  * The tensor has the following dimensions:
  * Nb - number of range bins (samples), Nr - number of Doppler bins (ramps), repeatFold - number of times the Doppler bins are repeated
  * NciFinal: uint32_t [Nb][Nr/repeatFold]
  * Data type: U32
  * Data layout: HW, H – number of range bins, W – number of Doppler bins divided by repeatFold
  *
  * Tensor 2: NoiseEstimate (only present if noiseEstimationEnabled is true, set to NULL if noiseEstimationEnabled is false)
  * The tensor has the following dimensions:
  * Nb - number of range bins (samples)
  * NoiseEstimate: uint32_t [Nb]
  * Data type: U32
  * Data layout: W, W – number of range bins
  *
  * @param [in] params Pointer to a PVARadarNCIParams structure containing the NCI parameters. See PVARadarNCIParams for more details.
  *
  *
  * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
  * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
  * @retval NVCV_SUCCESS                    Operation executed successfully.
  */
NVCVStatus pvaNciCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements *inTensorRequirements,
                        const NVCVTensorRequirements **outTensorRequirements, PVARadarNCIParams *params);

#ifdef __cplusplus
}

/**
  * Submits the NCI operator to a cuPVA stream.
  *
  * @param [in] handle Handle to the operator.
  *                    + Must not be NULL.
  *
  * @param [in] stream Handle to a valid cuPVA stream.
  *
  * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
  *                       + inHandles[0] is the FT2D tensor.
  *
  * @param [out] outHandles Pointer to array of output NVCVTensorHandle.
  *                        + outHandles[0] is the NciRx tensor.
  *                        + outHandles[1] is the NciFinal tensor.
  *                        + outHandles[2] is the NoiseEstimate tensor. This is only present if noiseEstimationEnabled is true.
  *
  *
  * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
  * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
  * @retval NVCV_SUCCESS                Operation executed successfully.
  */
NVCVStatus pvaNciSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const *inHandles,
                        NVCVTensorHandle *outHandles);

/**
 * Submits the NCI operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
 * @param [out] outHandles Pointer to array of output NVCVTensorHandle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaNciSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const *inHandles,
                        NVCVTensorHandle *outHandles);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPNCI_H */