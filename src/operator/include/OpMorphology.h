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
 * @file OpMorphology.h
 *
 * @brief Defines types and functions to handle the morphology operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_MORPHOLOGY Morphology
 * @{
 */

#ifndef PVA_SOLUTIONS_OPMORPHOLOGY_H
#define PVA_SOLUTIONS_OPMORPHOLOGY_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @cond SKIP_THIS */
#define MAX_MASK_MORPHOLOGY_MASK_WIDTH (5)
#define MAX_MASK_MORPHOLOGY_MASK_HEIGHT (5)
#define MAX_MASK_MORPHOLOGY_MASK_ARRAY_SIZE (MAX_MASK_MORPHOLOGY_MASK_WIDTH * MAX_MASK_MORPHOLOGY_MASK_HEIGHT)
/** @endcond */

/**
 * Morphology mask shape types
 */

typedef enum
{
    /// Rectange shaped mask
    RECTANGLE_MASK = 0,
    /// Cross shaped mask
    CROSS_MASK = 1
} MorphologyMaskType;

/**
 * Morphology mask size and shape parameters
 */
typedef struct PvaMorphologyMaskParamsRec
{
    /// Width of the mask to use (default 5).
    int32_t maskWidth;
    /// Width of the mask to use (default 5).
    int32_t maskHeight;
    /// Shape of the mask to use (default rectangle).
    MorphologyMaskType maskShape;
} PvaMorphologyMaskParams;

/** Constructs and an instance of the Morphology operator.
 *
 * \b Limitations:
 *
 * \b Input:
 *      + Data Layout:    [kHWC]
 *      + Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Output:
 *      + Data Layout:    [kHWC]
 *      + Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * \b Input/Output \b Dependency:
 *      Property    | Input == Output
 *     -------------| -------------
 *      Data Layout | Yes
 *      Data Type   | Yes
 *      Number      | Yes
 *      Channels    | Yes
 *      Width       | Yes
 *      Height      | Yes
 *
 * \b Parameters
 *
 *      maskParams:
 *      - maskWidth = maskHeight = 3 or 5
 *      - maskShape = RECTANGLE_MASK if maskWidth == 5
 *      morphType = PVA_ERODE, PVA_DILATE
 *      NVCVBorderType = NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE
 *
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains Tensor rank, shape, layout and data type information.
 *
 * @param [in] morphType Type of operation to performs Erode/Dilate. \ref PVAMorphologyType.
 *
 * @param [in] maskParams Pointer to the Morphology mask parameters. \ref PvaMorphologyMaskParams.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image.
 *
 * @param [in] borderValue Constant border value to be used when borderMode is NVCV_BORDER_CONSTANT. Ignored otherwise.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMorphologyCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                               PVAMorphologyType morphType, PvaMorphologyMaskParams *maskParams,
                               const NVCVBorderType borderMode, int32_t borderValue);

#ifdef __cplusplus
}

/**
 * Submits the Morphology operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMorphologySubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                               NVCVTensorHandle out);

/**
 * Submits the Morphology operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMorphologySubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                               NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPMORPHOLOGY_H */