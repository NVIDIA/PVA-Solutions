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
 * @file PvaOperator.h
 *
 * @brief Defines types and functions to handle operators
 * @defgroup PVA_OPERATOR_CORE_PVA_OPERATOR PvaOperator
 * @{
 */

#ifndef PVA_OPERATOR_H
#define PVA_OPERATOR_H

// Include submit function helper declarations for C _Generic dispatch
#include <priv/PvaSubmitDefinitions.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @cond SKIP_THIS */
typedef struct NVCVOperator *NVCVOperatorHandle;
/** @endcond */

/**
 * Destroys the operator instance.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 */
void nvcvOperatorDestroy(NVCVOperatorHandle handle);

#ifdef __cplusplus
}
#endif

/**
 * @page PVA_CUDA_STREAM_REQUIREMENTS CUDA Stream Support Requirements
 *
 * @note CUDA stream support requirements:
 *       - PVA SDK 2.7.0 or later
 *       - Jetpack 7 or later
 *       - DriveOS 7 or later
 *       - x86 Emulator is not supported
 */

/** @} */
#endif /* PVA_OPERATOR_H */
