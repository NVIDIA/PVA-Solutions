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
 * @file OpBatchSVD.h
 *
 * @brief Defines types and functions to handle the batch SVD operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_BATCHSVD BatchSVD
 * @{
 */

#ifndef PVA_SOLUTIONS_BATCH_SVD_H
#define PVA_SOLUTIONS_BATCH_SVD_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the BatchSVD operator.
 *
 * \b Limitations:
 *
 *     Data Layout: [HWC]
 *         C: Number of matrices
 *         H: Matrix height
 *         W: Matrix width
 *         Due to VMEM capacity, max matrix width = 31
 *         The formula for memory constraint on matrix width and height is:
 *         width * height * PITCH * 2 + (width + 1) * BATCH) > 1024 * 128 (128KB one VMEM superbank)
 *         PITCH and BATCH are set to 16, as defined in batch_svd_params.h
 *     Per matrix SVD: A = USV^T, economy SVD
 *         A and U size: H x W, H >= W >= 2
 *         S size: 1 x W, W >= 2, H = 1
 *         V size: W x W, W >= 2, H = W
 *         Please note that the S output is not in a diagonal matrix format.
 *         Instead, it is a vector containing W singular values.
 *
 *     Data Type      | Allowed
 *     -------------- | -------------
 *     8bit  Unsigned | No
 *     8bit  Signed   | No
 *     16bit Unsigned | No
 *     16bit Signed   | No
 *     32bit Unsigned | No
 *     32bit Signed   | No
 *     32bit Float    | Yes
 *     64bit Float    | No
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] srcParams Pointer to the NVCVTensorRequirements structure which contains input matrices tensor rank, shape, layout and data type information.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBatchSVDCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements *srcParams);

#ifdef __cplusplus
}

/**
 * Submits the BatchSVD operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] src Input matrix tensor handle.
 *                 Data Type: NVCV_DATA_TYPE_F32
 *                 Data Layout: [HWC]
 *
 * @param [out] U Output U matrix tensor handle.
 *                Data Type: NVCV_DATA_TYPE_F32
 *                Data Layout: [HWC]
 *
 * @param [out] S Output S matrix tensor handle.
 *                Data Type: NVCV_DATA_TYPE_F32
 *                Data Layout: [HWC]
 *
 * @param [out] V Output V matrix tensor handle.
 *                Data Type: NVCV_DATA_TYPE_F32
 *                Data Layout: [HWC]
 *
 * @param [in] maxIters The process of Jacobi rotations stops after maxIters sweeps.
 *
 * @param [in] tol Converge criterion to check if the off-diagonal elements are sufficiently small compared to the diagonal elements.
 *                 Note that if the value is less than 1e-7f, convergence may not occur until the maximum number of iterations is reached.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBatchSVDSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, const NVCVTensorHandle src,
                             const NVCVTensorHandle U, const NVCVTensorHandle S, const NVCVTensorHandle V,
                             const int32_t maxIters, const float tol);

/**
 * Submits the BatchSVD operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] src Input matrix tensor handle.
 * @param [out] U Output U matrix tensor handle.
 * @param [out] S Output S matrix tensor handle.
 * @param [out] V Output V matrix tensor handle.
 * @param [in] maxIters The process of Jacobi rotations stops after maxIters sweeps.
 * @param [in] tol Converge criterion to check if the off-diagonal elements are sufficiently small.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBatchSVDSubmit(NVCVOperatorHandle handle, cudaStream_t stream, const NVCVTensorHandle src,
                             const NVCVTensorHandle U, const NVCVTensorHandle S, const NVCVTensorHandle V,
                             const int32_t maxIters, const float tol);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_BATCH_SVD_H */