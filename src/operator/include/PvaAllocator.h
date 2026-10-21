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
 * @file PvaAllocator.h
 *
 * @brief Defines types and functions to handle PVA accessible Tensor Allocator
 * @defgroup PVA_OPERATOR_CORE_PVA_ALLOCATOR PvaAllocator
 * @{
 */

#ifndef PVA_ALLOCATOR_H
#define PVA_ALLOCATOR_H

#include <nvcv/DataType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <nvcv/TensorLayout.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Calculates the resource requirements needed to create a tensor with given shape.
 *
 * @param [in] rank Rank of the tensor (its number of dimensions).
 *
 * @param [in] shape Pointer to array with tensor shape.
 *                   It must contain at least 'rank' elements.
 *
 * @param [in] dtype Type of tensor's elements.
 *
 * @param [in] layout Tensor layout.
 *                    Pass NVCV_TENSOR_NONE is layout is not available.
 *                    + Layout rank must be @p rank.
 *
 * @param [in] baseAddrAlignment Alignment, in bytes, of the requested memory buffer.
 *                               Currently, only a value of 0 is supported, which means buffers will be allocated using
 *                               the default PVA memory alignment.
 *
 * @param [in] rowAddrAlignment Alignment, in bytes, of the start of each second-to-last dimension address with respect
 *                              to the base address.
 *                              In other words, (rowAddr - baseAddr) % rowAddrAlignment == 0.
 *                              If 0, use a default suitable for optimized memory access.
 *                              The used alignment is at least the given value.
 *                              Pass 1 for creation of fully packed tensors, i.e., no padding between dimensions.
 *                              + If different from 0, it must be a power-of-two.
 *
 * @param [out] reqs  Where the tensor requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus nvcvTensorCalcRequirementsPva(int32_t rank, const int64_t *shape, NVCVDataType dtype,
                                         NVCVTensorLayout layout, int32_t baseAddrAlignment, int32_t rowAddrAlignment,
                                         NVCVTensorRequirements *reqs);

/** Calculates the resource requirements needed to create a tensor with given shape.
 *
 * @param [in] width,height Image dimensions.
 *                          + Width and height must be > 0.
 *
 * @param [in] format       Image format.
 *                          + Must not be NVCV_IMAGE_FORMAT_NONE.
 *
 * @param [in] baseAddrAlignment Alignment, in bytes, of the requested memory buffer.
 *                               Currently, only a value of 0 is supported, which means buffers will be allocated using
 *                               the default PVA memory alignment.
 *
 * @param [in] rowAddrAlignment Alignment, in bytes, of each image's row address.
 *                              In other words, (rowAddr - baseAddr) % rowAddrAlignment == 0.
 *                              If 0, use a default suitable for optimized memory access.
 *                              The used alignment is at least the given value.
 *                              Pass 1 for fully packed rows, i.e., no padding at the end of each row.
 *                              + If different from 0, it must be a power-of-two.
 *
 * @param [out] reqs    Where the image requirements will be written to.
 *                      + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */

NVCVStatus nvcvImageCalcRequirementsPva(int32_t width, int32_t height, NVCVImageFormat format,
                                        int32_t baseAddrAlignment, int32_t rowAddrAlignment,
                                        NVCVImageRequirements *reqs);

/** Constructs an allocator instance in the given storage.
 *
 * When not needed anymore, the allocator instance must be destroyed by
 * nvcvAllocatorDecRef function.
 *
 * @param [out] handle Where new instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the allocator.
 * @retval NVCV_SUCCESS                Allocator created successfully.
 */
NVCVStatus nvcvAllocatorConstructPva(NVCVAllocatorHandle *handle);

#ifdef __cplusplus
}
#endif

/** @} */
#endif /* PVA_ALLOCATOR_H */
