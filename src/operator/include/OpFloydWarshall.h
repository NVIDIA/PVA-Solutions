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
 * @file OpFloydWarshall.h
 *
 * @brief Defines types and functions to perform the Floyd-Warshall algorithm.
 * @defgroup PVA_OPERATOR_ALGORITHM_FLOYD_WARSHALL FloydWarshall
 * @{
 */

#ifndef PVA_SOLUTIONS_OPFLOYDWARSHALL_H
#define PVA_SOLUTIONS_OPFLOYDWARSHALL_H

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
 * Constructs an instance of the Floyd-Warshall operator.
 *
 * The Floyd-Warshall algorithm finds the shortest path between each pair of nodes in a weighted graph.
 * There are two tensors involved in the algorithm:
 * - length tensor: represents the length of the shortest path between each pair of nodes.
 *   For a graph with N nodes, this should be an N*N tensor initialized as:
 *    - length[i][i] = 0
 *    - length[i][j] = weight of the edge, if a direct edge exists from node i to j
 *    - length[i][j] = positive infinity (0xFFFF), if no direct edge exists
 * - next tensor: represents the next node in the shortest path between each pair of nodes.
 *   For a graph with N nodes, this should be an N*N tensor initialized as:
 *    - next[i][i] = i
 *    - next[i][j] = j, if a direct edge exists from node i to j
 *    - next[i][j] = NIL (0xFFFF), if no direct edge exists
 * The next tensor is used to reconstruct the shortest path. Both tensors are updated in place.
 * @note The current implementation supports graphs with up to 240 nodes (N <= 240) due to VMEM constraints.
 *
 * \b Length tensor:
 *      Data Layout:    [HWC]
 *      Channels:       [1]
 *      Data Type:      [16bit Unsigned]
 * \b Next tensor:
 *      Data Layout:    [HWC]
 *      Channels:       [1]
 *      Data Type:      [16bit Unsigned]
 * @note The length tensor and the next tensor must have the same dimensions.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaFloydWarshallCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements);

#ifdef __cplusplus
}

/**
 * Submits the Floyd-Warshall operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] lenTensor Length tensor handle.
 *
 * @param [in] nextTensor Next tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaFloydWarshallSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle lenTensor,
                                  NVCVTensorHandle nextTensor);

/**
 * Submits the Floyd-Warshall operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] lenTensor Length tensor handle.
 * @param [in] nextTensor Next tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaFloydWarshallSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle lenTensor,
                                  NVCVTensorHandle nextTensor);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPFLOYDWARSHALL_H */