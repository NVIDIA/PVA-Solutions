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
 * @file OpImageStats.h
 *
 * @brief Defines types and functions to handle the image statistics operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_IMAGESTATS ImageStats
 * @{
 */

#ifndef PVA_SOLUTIONS_OPIMAGESTATS_H
#define PVA_SOLUTIONS_OPIMAGESTATS_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the ImageStats operator.
 *
 * \b Limitations:
 *      1. For input image formats NV12 and NV12_ER, the image width and height must be even.
 *      2. Product of image width and height must be less than 1<<24 (4096 * 4096).
 *      3. The value in the mask image must be either 0 or 1.
 *
 * \b Input:
 *      Input image sizes and format are specified in imageRequirements.
 *
 *      Image Format   | Allowed
 *      -------------- | -------------
 *      U8             | Yes
 *      RGB8           | Yes
 *      BGR8           | Yes
 *      NV12           | Yes
 *      NV12_ER        | Yes
 *
 * \b Mask:
 *      Mask image sizes are specified in maskRequirements.
 *      Mask must be binary and of image format U8.
 *
 * \b Input/Mask \b Dependency:
 *      Property    | Input == Mask
 *     ------------ | -------------
 *      Width       | Yes
 *      Height      | Yes
 *
 * \b Parameters
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] imageRequirements Pointer to the NVCVImageRequirements structure which contains image shape,
 *                                 strides, and format information.
 *
 * @param [in] maskRequirements Pointer to the NVCVImageRequirements structure for the mask image.
 *                                If without mask, the pointer is set to NULL.
 *
 * @param [in] flags Flags to control which statistics are computed.
 *                     Accepted flags are listed in \ref PVAImageStatFlag.
 *                     flags can be a combination of the accepted flags.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
*/
NVCVStatus pvaImageStatsCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements,
                               NVCVImageRequirements *maskRequirements, uint32_t flags);

#ifdef __cplusplus
}

/**
 * Submits the ImageStats operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [in] mask Mask image handle.
 *
 * @param [out] out Output tensor handle.
 *                    1D tensor to store the output struct \ref PVAImageStatOutput.
 *
 *      Data Layout:    [W]
 *      Length:         sizeof(PVAImageStatOutput)
 *      Data Type:      NVCV_DATA_TYPE_U8
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageStatsSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                               NVCVImageHandle mask, NVCVTensorHandle out);

/**
 * Submits the ImageStats operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [in] mask Mask image handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageStatsSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle mask,
                               NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPIMAGESTATS_H */