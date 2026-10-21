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
 * @file OpDepthToSpace.h
 *
 * @brief Defines types and functions to handle the Depth-to-Space operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_DEPTHTOSPACE DepthToSpace
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDEPTHTOSPACE_H
#define PVA_SOLUTIONS_OPDEPTHTOSPACE_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the DepthToSpace operator.
 *
 * Rearranges data from the depth (channel) dimension into blocks of spatial data,
 * following the ONNX DepthToSpace operator in DCR (depth-column-row) ordering, see
 * https://onnx.ai/onnx/operators/onnx__DepthToSpace.html. The reverse operation is
 * called SpaceToDepth. Given
 * an input tensor of shape [C, H, W] and block size B, the output tensor has shape
 * [C / (B * B), H * B, W * B] and every output element satisfies
 *
 *     out[c, h * B + by, w * B + bx] = in[(by * B + bx) * (C / (B * B)) + c, h, w]
 *
 * for 0 <= by, bx < B. This is a pure element rearrangement: values are copied
 * unmodified, without conversion, rounding or saturation, and every input element
 * appears exactly once in the output.
 *
 * \b Limitations:
 *      The number of input channels must be an exact multiple of blkSz * blkSz.
 *
 *      The tensor is rank 3 in CHW layout: shape[0..2] are the channel count, height and
 *      width, and strides[0..1] the channel and row strides.
 *
 *      The input tensor may carry padding, but its regions must not overlap:
 *
 *      - strides[2] must be 1. Elements within a row are read contiguously and an element
 *        stride that skips over bytes cannot be expressed.
 *      - strides[1] must be at least shape[2]. A row may be followed by padding; it may
 *        not run into the next one.
 *      - strides[0] must be at least shape[1] * strides[1]. A channel may be followed by
 *        padding; it may not run into the next one.
 *
 *      The extents and the row pitch are also bounded by the widths of the DMA fields
 *      they are programmed into:
 *
 *      - strides[1] must not exceed 65535, the 16-bit surface line pitch.
 *      - shape[2] * blkSz must not exceed 65535, the output surface row pitch.
 *      - shape[1] * blkSz must not exceed 65535, the output surface height.
 *      - shape[0] * strides[0] must be below 2147483648, the range of the 32-bit offsets the
 *        transfers address the tensor by. This bounds the output too: it is tightly packed,
 *        and the no-overlap rules above hold shape[0] * shape[1] * shape[2] at or below it.
 *
 *      Every shape and stride is itself read as a 32-bit value, so each must be positive
 *      and below 2147483648. A field outside that range is rejected rather than truncated,
 *      which would describe a different tensor than the one passed.
 *
 *      The output tensor is derived from the input tensor and the block size, and is
 *      tightly packed.
 *
 * \b Input:
 *      Data Layout:    [CHW]
 *      Channels:       [C, must be a multiple of blkSz * blkSz]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Output:
 *      Data Layout:    [CHW]
 *      Channels:       [C / (blkSz * blkSz)]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Input/Output \b Dependency:
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | No, Output Channels = (Input Channels) / (blkSz^2)
 *      Width         | No, Output Width = (Input Width) * (blkSz)
 *      Height        | No, Output Height = (Input Height) * (blkSz)
 *
 * \b Parameters
 *
 *      blkSz = 2, 3 or 4
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains the input Tensor
 *                                rank, shape, layout and data type information. The output tensor requirements are
 *                                derived from it and the block size.
 *                                + Must not be NULL.
 *
 * @param [in] blkSz The spatial block size. Supported block sizes: 2, 3 and 4.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle or tensorRequirements is NULL, or points to an invalid address.
 *                                      2) Block size is not 2, 3 or 4.
 *                                      3) The channel count is not a multiple of (blkSz)^2.
 *                                      4) The tensor does not have rank 3, is not in CHW layout, has a
 *                                         non-positive extent, or its data type is not 8bit signed or
 *                                         unsigned.
 *                                      5) The tensor violates a documented limitation above:
 *                                         overlapping regions, a DMA field exceeded, or a shape
 *                                         or stride outside 32-bit range.
 *                                      6) The output tensor requirements could not be derived from the input
 *                                         tensor and the block size.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDepthToSpaceCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const tensorRequirements,
                                 int32_t blkSz);

#ifdef __cplusplus
}

/**
 * Submits the DepthToSpace operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle. Must match the tensor requirements passed to
 *                pvaDepthToSpaceCreate().
 *                + Must not be NULL.
 *
 * @param [out] out Output tensor handle. Must match the output tensor requirements derived by
 *                  pvaDepthToSpaceCreate().
 *                  + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT The possible cases might be the following:
 *                                      1) handle, stream, in or out is NULL, or points to an invalid address.
 *                                      2) The input or output tensor does not meet the requirements used to
 *                                         create the operator handle.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDepthToSpaceSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                 NVCVTensorHandle out);

/**
 * Submits the DepthToSpace operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDepthToSpaceSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                 NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDEPTHTOSPACE_H */
