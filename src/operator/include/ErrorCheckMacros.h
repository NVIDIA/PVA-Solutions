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
 * @file ErrorCheckMacros.h
 *
 * @brief Defines macros for checking return values of CUPVA and NVCV Api calls
 * @defgroup PVA_OPERATOR_CORE_ERROR_CHECK_MACROS ErrorCheckMacros
 * @{
 */

#ifndef ERROR_CHECK_MACROS_H
#define ERROR_CHECK_MACROS_H

#include <cupva_host.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
#    include <cstdio>
#    include <cstring>
#    include <stdexcept>
#else
#    include <stdio.h>
#endif

/**
 * @brief Macro to check the return value of a CUPVA API call and jump to a label if an error is detected.
 */
#define CUPVA_CHECK_ERROR_GOTO(__v, __e, __l)                                              \
    __e = __v;                                                                             \
    if (__e != CUPVA_ERROR_NONE)                                                           \
    {                                                                                      \
        const char *msgBuffer;                                                             \
        CupvaGetLastError(&msgBuffer);                                                     \
        printf("CUPVA error: %d at line %d in function %s\n", (int)(__e), __LINE__, #__v); \
        printf("Error message: %s\n", msgBuffer);                                          \
        goto __l;                                                                          \
    }

/**
 * @brief Macro to check the return value of a NVCV API call and jump to a label if an error is detected.
 */
#define NVCV_CHECK_ERROR_GOTO(__v, __e, __l)                                              \
    __e = __v;                                                                            \
    if (__e != NVCV_SUCCESS)                                                              \
    {                                                                                     \
        char msgBuffer[1024];                                                             \
        nvcvPeekAtLastErrorMessage(msgBuffer, sizeof(msgBuffer));                         \
        printf("NVCV error: %d at line %d in function %s\n", (int)(__e), __LINE__, #__v); \
        printf("Error message: %s\n", msgBuffer);                                         \
        goto __l;                                                                         \
    }

/**
 * @brief Macro to check the return value of a CUDA API call and jump to a label if an error is detected.
 * @note To use this macro, you need to include <cuda_runtime.h> and link to the CUDA runtime library.
 */
#define CUDA_CHECK_ERROR_GOTO(__v, __e, __l)                                              \
    __e = __v;                                                                            \
    if (__e != cudaSuccess)                                                               \
    {                                                                                     \
        printf("CUDA error: %d at line %d in function %s\n", (int)(__e), __LINE__, #__v); \
        printf("Error message: %s\n", cudaGetErrorString(__e));                           \
        goto __l;                                                                         \
    }

#ifdef __cplusplus

/**
 * @brief Function to check the return value of a CUPVA API call and throw an exception if an error is detected.
 */
inline void CupvaCheckError(cupvaError_t err, const char *file, const int line, void (*cleanup_callback)() = nullptr)
{
    if (err != CUPVA_ERROR_NONE)
    {
        if (cleanup_callback != nullptr)
        {
            cleanup_callback();
        }

        const char *msgBuffer;
        CupvaGetLastError(&msgBuffer);

        char message[512];
        snprintf(message, sizeof(message), "CUPVA error returned at %s:%d, Error code: %d (%s)", file, line, err,
                 msgBuffer);

        throw std::runtime_error(message);
    }
}

/**
 * @brief Function to check the return value of a NVCV API call and throw an exception if an error is detected.
 */
inline void NvcvCheckError(NVCVStatus status, const char *file, const int line, void (*cleanup_callback)() = nullptr)
{
    if (status != NVCV_SUCCESS)
    {
        if (cleanup_callback != nullptr)
        {
            cleanup_callback();
        }

        char message[512];
        snprintf(message, sizeof(message), "NVCV error returned at %s:%d, Error code: %d", file, line, status);

        throw std::runtime_error(message);
    }
}

/**
 * @brief Macro to check the return value of a CUPVA API call and throw an exception if an error is detected.
 */
#    define CUPVA_CHECK_ERROR(val, ...) CupvaCheckError((val), __FILE__, __LINE__, ##__VA_ARGS__)

/**
 * @brief Macro to check the return value of a NVCV API call and throw an exception if an error is detected.
 */
#    define NVCV_CHECK_ERROR(val, ...) NvcvCheckError((val), __FILE__, __LINE__, ##__VA_ARGS__)

#endif

/** @} */
#endif /* ERROR_CHECK_MACROS_H */
