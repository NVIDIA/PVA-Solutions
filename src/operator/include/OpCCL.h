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
 * @file OpCCL.h
 *
 * @brief Defines types and functions to handle the Connected Component Labeling operation.
 *        Computes the labeled image for connected components in the input image using either 4- or 8-connectivity.
 *        Produces an output where each unique connected component in the input image is assigned a distinct label,
 *        yielding a total of N labels.
 *        All pixels labeled '0' indicate background. Labels from 1 correspond to the unique connected foreground
 *        components.
 *        Note that labels in the output are not required to be sequential.
 *
 * @defgroup PVA_OPERATOR_ALGORITHM_CCL CCL
 * @{
 */

#ifndef PVA_SOLUTIONS_CCL_H
#define PVA_SOLUTIONS_CCL_H

#ifdef __cplusplus
#    include <cuda_runtime.h>
#endif

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the CCL operator.
    *
    * \b Limitations:
    *
    * \b Input imageParams:
    *    Data Layout:    [CHW], [NCHW] only when C = 1 and N = 1
    *
    *    Data Type      | Allowed
    *    -------------- | --------------
    *    8bit  Unsigned | Yes
    *    8bit  Signed   | No
    *    16bit Unsigned | No
    *    16bit Signed   | No
    *    32bit Unsigned | No
    *    32bit Signed   | No
    *    32bit Float    | No
    *    64bit Float    | No
    *
    * \b Output labelParams:
    *    Data Layout:    [CHW], [NCHW] only when C = 1 and N = 1
    *
    *    Data Type      | Allowed
    *    -------------- | --------------
    *    8bit  Unsigned | No
    *    8bit  Signed   | No
    *    16bit Unsigned | Yes
    *    16bit Signed   | No
    *    32bit Unsigned | No
    *    32bit Signed   | No
    *    32bit Float    | No
    *    64bit Float    | No
    *
    * \b Input/Output Dependency:
    *    Property       | Input == Output
    *    -------------- | --------------
    *    Data Layout    | Yes
    *    Data Type      | No
    *    Number         | Yes
    *    Channels       | Yes
    *    Width          | Yes
    *    Height         | Yes
    *
    *    Value 0 in the output labeled image is background.
    *    Maximum number of labels: 65535 excluding label 0 which is reserved for background.
    *
    * @param [out] handle Where the operator instance handle will be written to.
    *                     + Must not be NULL.
    *
    * @param [in] imageParams Pointer to the NVCVTensorRequirements structure which contains input tensor rank,
    *                         shape, layout and data type information.
    *
    * @param [in] labelParams Pointer to the NVCVTensorRequirements structure which contains output tensor rank,
    *                         shape, layout and data type information.
    *
    * @param [in] connectivity Must be PVA_4_CONNECTIVITY or PVA_8_CONNECTIVITY.
    *                          PVA_4_CONNECTIVITY for 4-connectivity and PVA_8_CONNECTIVITY for 8-connectivity.
    *
    * @param [in] threshold Only pixels with values greater than this threshold are treated as foreground when
    *                       running connected components. If input image is already binary, set it to 0.
    *
    * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
    * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
    * @retval NVCV_SUCCESS                Operation executed successfully.
    */
NVCVStatus pvaCCLCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements *imageParams,
                        const NVCVTensorRequirements *labelParams, const PVAConnectivityType connectivity,
                        const uint8_t threshold);
#ifdef __cplusplus
}

/**
    * Submits the CCL operator to a cuPVA stream.
    *
    * @param [in] handle Handle to the operator.
    *                    + Must not be NULL.
    * @param [in] stream Handle to a valid cuPVA stream.
    *
    * @param [in] image Input tensor handle of input image.
    *
    * @param [out] labels Output tensor handle of labeled image. Note that labels in the output are not required
    *                     to be sequential
    *
    * @param [out] numLabels Output tensor handle of number of labels excluding label 0 which is reserved for background.
    *                        Data Layout:    [W]
    *                        Width:          1
    *                        Data Type:     NVCV_DATA_TYPE_S32
    *                        If the runtime labels exceed the limit (65536 labels including label 0 which is
    *                        reserved for background), numLabels will return -1.
    *
    * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
    * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
    * @retval NVCV_SUCCESS                Operation executed successfully.
    */
NVCVStatus pvaCCLSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle image,
                        NVCVTensorHandle labels, NVCVTensorHandle numLabels);

/**
    * Submits the CCL operator to a CUDA stream.
    *
    * @param [in] handle Handle to the operator.
    *                    + Must not be NULL.
    * @param [in] stream Handle to a valid CUDA stream.
    *
    * @param [in] image Input tensor handle of input image.
    *
    * @param [out] labels Output tensor handle of labeled image. Note that labels in the output are not required
    *                     to be sequential.
    *
    * @param [out] numLabels Output tensor handle of number of labels excluding label 0 which is reserved for background.
    *                        Data Layout:    [W]
    *                        Width:          1
    *                        Data Type:     NVCV_DATA_TYPE_S32
    *                        If the runtime labels exceed the limit (65536 labels including label 0 which is
    *                        reserved for background), numLabels will return -1.
    *
    * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
    * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
    * @retval NVCV_SUCCESS                Operation executed successfully.
    */
NVCVStatus pvaCCLSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle image, NVCVTensorHandle labels,
                        NVCVTensorHandle numLabels);

#endif // __cplusplus
/** @} */
#endif /* PVA_SOLUTIONS_CCL_H */
