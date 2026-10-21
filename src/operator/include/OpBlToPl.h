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
 * @file OpBlToPl.h
 *
 * @brief Defines types and functions to handle Block Linear (BL) layout to Pitch Linear (PL) layout conversion operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_BLTOPL BlToPl
 * @{
 */

#ifndef PVA_SOLUTIONS_BLTOPL_H
#define PVA_SOLUTIONS_BLTOPL_H

#ifdef __cplusplus
#    include <cuda_runtime.h>
#endif

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an operator instance for converting an image from Block Linear (BL) layout to Pitch Linear (PL) layout.
 * This operator is specifically designed to work with images in the NV12 format only.
 *
 * PL (Pitch Linear) is the layout of most ordinary images, where elements are arranged row by row and elements within
 * the same row are consecutive in memory. BL (Block Linear)
 * <a href="https://docs.nvidia.com/drive/drive_os_5.1.6.1L/nvvib_docs/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide/NvMedia/nvmedia_concept_surface.html">block linear</a>
 * is a proprietary layout. Compared to PL layout, on a 32 bytes x 2 rows granularity (referred to as 32x2), BL layout means the elements are
 * permuted as follows:
 * + Bytes within the 32x2 are permuted.
 * + Different 32x2 blocks within the image are also permuted.
 *
 * NV12 is a YUV semi-planar format that has two planes: one for the Y plane and the other for the interleaved UV plane.
 * The UV plane is subsampled, with its image width and height being half of that of the Y plane's. For both PL and BL,
 * the layout refers to a per-plane basis; there is no permutation of blocks across planes.
 *
 * \b Input:
 *      Data Layout:    [BL]
 *
 *      Image Format:   NVCV_IMAGE_FORMAT_NV12 or NVCV_IMAGE_FORMAT_NV12_ER
 *                      Note that NVCV APIs do not support NVCV_IMAGE_FORMAT_NV12_BL or NVCV_IMAGE_FORMAT_NV12_ER_BL,
 *                    otherwise they will be more accurate.
 *
 *
 * \b Output:
 *      Data Layout:    [PL]
 *
 *      Image Format:   NVCV_IMAGE_FORMAT_NV12 or NVCV_IMAGE_FORMAT_NV12_ER
 *
 *
 * \b Input/Output \b Dependency:
 *      Property    | Input == Output
 *     -------------| -------------
 *      Data Layout | No
 *      Image Format| Yes
 *
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] imageRequirements Pointer to the NVCVImageRequirements structure which contains image width, height,
 *                               data type information, and output image's line pitch.
 *                               + Note that the line pitch for BL layout is implicitly determined by the image width,
 *                                 so cannot be explicitly specified.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Possible cases include:
 *                                            1) The handle or imageRequirements are either NULL or point to an invalid address.
 *                                            2) The NV12 format itself is incorrect, e.g., having an odd number of rows or columns.
 *                                            3) The image size is smaller than the minimum required size of 32x4.
 *                                            4) Incorrect tile size for the given image size, as the tile size is currently hardcoded.
 *
 * @retval NVCV_ERROR_OUT_OF_MEMORY         Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                     Operation executed successfully.
 */
NVCVStatus pvaBlToPlCreate(NVCVOperatorHandle *handle, const NVCVImageRequirements *imageRequirements);

#ifdef __cplusplus
}

/**
 * Submits the BlToPl operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle, points to the image with BL layout.
 *
 * @param [out] out Output image handle, points to the image with PL layout.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Possible cases include:
 *                                            1) The handle, stream, input, or output is either NULL or points to an invalid address.
 *                                            2) The input or output image does not meet the requirements used to create the operator handle.
 *                                            3) The metadata of the input does not correspond to the correct NV12 format (e.g., number of planes not equal to 2).
 *
 * @retval NVCV_ERROR_INTERNAL               Internal error in the operator; invalid types passed in.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaBlToPlSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in, NVCVImageHandle out);

/**
 * Submits the BlToPl operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle, points to the image with BL layout.
 * @param [out] out Output image handle, points to the image with PL layout.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       Possible cases include invalid handle, stream, input, or output.
 * @retval NVCV_ERROR_INTERNAL               Internal error in the operator; invalid types passed in.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaBlToPlSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_BLTOPL_H */