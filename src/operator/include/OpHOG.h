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
 * @file OpHOG.h
 *
 * @brief Defines types and functions to handle the HOG operation.
 *        HOG (Histogram of Oriented Gradients) extracts local gradient
 *        orientation histograms to capture shape/edge information for
 *        detection or description tasks. This operator computes per-pixel
 *        gradients, accumulates orientation histograms over spatial cells,
 *        normalizes them across overlapping blocks, and outputs the CHW
 *        formatted feature tensor from an RGB8p input image.
 * @defgroup PVA_OPERATOR_ALGORITHM_HOG HOG
 * @{
 */

#ifndef PVA_SOLUTIONS_OPHOG_H
#define PVA_SOLUTIONS_OPHOG_H

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

/** Constructs and an instance of the HOG operator.
 *
 * Limitations:
 *
 * Input image:
 *      Image format | Allowed
 *      ------------ | -------
 *      RGB8p        | Yes
 *
 *      Image size constraints:
 *      - 576x576 is the only supported image size.
 *
 * Output tensor:
 *     Data Layout:    [CHW]
 *
 *      Tensor size constraints:
 *      - 18x144x144 is the only supported tensor size.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] imageRequirements Pointer to the NVCVImageRequirements structure which contains input image width, height and format information.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains output tensor rank, shape, layout and data type information.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaHOGCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements,
                        NVCVTensorRequirements *tensorRequirements);

#ifdef __cplusplus
}

/**
 * Submits the HOG operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaHOGSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in,
                        NVCVTensorHandle const out);

/**
 * Submits the HOG operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaHOGSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in,
                        NVCVTensorHandle const out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPHOG_H */