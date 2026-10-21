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
 * @file PvaSubmitMacros.h
 *
 * @brief Helper macros for PVA operator submit functions.
 *        Provides utilities for parameter parsing, name extraction, and function generation.
 *        This file contains the low-level macros used by PvaSubmitDefinitions.h.
 *
 * @section usage Usage Guide
 *
 * To add a new submit function API that works with both PVA and CUDA streams:
 *
 * 1. In PvaSubmitDefinitions.h, add a declaration:
 *    @code
 *    DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMyOperatorSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
 *    @endcode
 *
 * 2. In PvaSubmitDefinitions.h (in the C-only section), add a _Generic dispatcher:
 *    @code
 *    #ifndef __cplusplus
 *    #define pvaMyOperatorSubmit(handle, stream, in, out) \
 *        DEFINE_PVA_SUBMIT_GENERIC(pvaMyOperatorSubmit, handle, stream, in, out)
 *    #endif
 *    @endcode
 *
 * 3. In PvaSubmitDefinitions.cpp, add the implementation:
 *    @code
 *    DEFINE_PVA_SUBMIT_FUNCTIONS(pvaMyOperatorSubmit, MorphologySubmitImpl, (NVCVTensorHandle, in), (NVCVTensorHandle, out))
 *    @endcode
 *
 * This will automatically generate:
 *   - pvaMyOperatorSubmit__cupva(handle, cupvaStream_t stream, in, out)
 *   - pvaMyOperatorSubmit__cuda(handle, cudaStream_t stream, in, out)
 *   - A _Generic macro that dispatches based on stream type (C only)
 *
 * @note Each parameter is wrapped: (type, name), (type, name), ...
 * @note Same syntax used in both DECLARE and DEFINE macros - no redundancy!
 * @note Supports up to 10 parameters (can be extended by adding more PVA_DECL_N/PVA_NAMES_N macros).
 */

#ifndef PVA_SOLUTIONS_PVASUBMITMACROS_H
#define PVA_SOLUTIONS_PVASUBMITMACROS_H

#include <cupva_host.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

// Forward declaration of cudaStream_t
typedef struct CUstream_st *cudaStream_t;

// Forward declaration of NVCVOperatorHandle
typedef struct NVCVOperator *NVCVOperatorHandle;

/**
 * @brief Declares Pva and Cuda variants of a submit function.
 *
 * @param name Base function name (e.g., pvaMorphologySubmit)
 * @param ... Wrapped parameter pairs (type, name), (type, name), ...
 *
 * Usage:
 *   DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMorphologySubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out))
 *
 * Generates:
 *   NVCVStatus pvaMorphologySubmit__cupva(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out);
 *   NVCVStatus pvaMorphologySubmit__cuda(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out);
 */
#define DECLARE_PVA_SUBMIT_FUNCTIONS(name, ...)                                                       \
    NVCVStatus name##__cupva(NVCVOperatorHandle handle, cupvaStream_t stream, PVA_DECL(__VA_ARGS__)); \
    NVCVStatus name##__cuda(NVCVOperatorHandle handle, cudaStream_t stream, PVA_DECL(__VA_ARGS__))

/**
 * @brief Defines the _Generic macro for type-based dispatch of submit functions (C only).
 *
 * @param name Base function name (e.g., pvaMorphologySubmit)
 * @param _handle Handle parameter name
 * @param _stream Stream parameter name
 * @param ... Additional parameter names to forward (e.g., in, out)
 *
 * Usage:
 *   DEFINE_PVA_SUBMIT_GENERIC(pvaMorphologySubmit, handle, stream, in, out)
 *
 * Generates a _Generic macro that dispatches based on stream type.
 */
#define DEFINE_PVA_SUBMIT_GENERIC(name, _handle, _stream, ...) \
    _Generic((_stream), cupvaStream_t: name##__cupva, cudaStream_t: name##__cuda)(_handle, _stream, __VA_ARGS__)

/**
 * @brief Simplified macro for C dispatch - just calls DEFINE_PVA_SUBMIT_GENERIC.
 *
 * This is an alias for DEFINE_PVA_SUBMIT_GENERIC with a clearer name for dispatch definitions.
 * Use this within #ifndef __cplusplus blocks.
 *
 * Usage in C-only section:
 *   #ifndef __cplusplus
 *   #define pvaMorphologySubmit(handle, stream, in, out) \
 *       DEFINE_PVA_SUBMIT_DISPATCH(pvaMorphologySubmit, handle, stream, in, out)
 *   #endif
 */
#define DEFINE_PVA_SUBMIT_DISPATCH(name, _handle, _stream, ...) \
    DEFINE_PVA_SUBMIT_GENERIC(name, _handle, _stream, __VA_ARGS__)

/**
 * @brief Helper macros to generate internal PVA/CUDA function names.
 *
 * These are useful for API overrides and testing frameworks that need to
 * reference the internal implementation functions.
 *
 * Usage:
 *   PVA_INTERNAL_CUPVA_SUBMIT_NAME(pvaMorphologySubmit)   -> pvaMorphologySubmit__cupva
 *   PVA_INTERNAL_CUDA_SUBMIT_NAME(pvaMorphologySubmit)  -> pvaMorphologySubmit__cuda
 *   PVA_INTERNAL_CUPVA_SUBMIT_STR(pvaMorphologySubmit)    -> "pvaMorphologySubmit__cupva"
 *   PVA_INTERNAL_CUDA_SUBMIT_STR(pvaMorphologySubmit)   -> "pvaMorphologySubmit__cuda"
 */
#define PVA_INTERNAL_CUPVA_SUBMIT_NAME(name) name##__cupva
#define PVA_INTERNAL_CUDA_SUBMIT_NAME(name) name##__cuda
#define PVA_INTERNAL_CUPVA_SUBMIT_STR(name) #name "__cupva"
#define PVA_INTERNAL_CUDA_SUBMIT_STR(name) #name "__cuda"

// Helper macros to extract type and name from a wrapped parameter (type, name)
#define PVA_EXTRACT_DECL(pair) PVA_EXTRACT_DECL_IMPL pair
#define PVA_EXTRACT_DECL_IMPL(type, name) type name

#define PVA_EXTRACT_NAME(pair) PVA_EXTRACT_NAME_IMPL pair
#define PVA_EXTRACT_NAME_IMPL(type, name) name

// Helper to process wrapped parameter pairs for declarations
#define PVA_DECL_1(p1) PVA_EXTRACT_DECL(p1)
#define PVA_DECL_2(p1, p2) PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2)
#define PVA_DECL_3(p1, p2, p3) PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3)
#define PVA_DECL_4(p1, p2, p3, p4) \
    PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3), PVA_EXTRACT_DECL(p4)
#define PVA_DECL_5(p1, p2, p3, p4, p5) \
    PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3), PVA_EXTRACT_DECL(p4), PVA_EXTRACT_DECL(p5)
#define PVA_DECL_6(p1, p2, p3, p4, p5, p6)                                                                        \
    PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3), PVA_EXTRACT_DECL(p4), PVA_EXTRACT_DECL(p5), \
        PVA_EXTRACT_DECL(p6)
#define PVA_DECL_7(p1, p2, p3, p4, p5, p6, p7)                                                                    \
    PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3), PVA_EXTRACT_DECL(p4), PVA_EXTRACT_DECL(p5), \
        PVA_EXTRACT_DECL(p6), PVA_EXTRACT_DECL(p7)
#define PVA_DECL_8(p1, p2, p3, p4, p5, p6, p7, p8)                                                                \
    PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3), PVA_EXTRACT_DECL(p4), PVA_EXTRACT_DECL(p5), \
        PVA_EXTRACT_DECL(p6), PVA_EXTRACT_DECL(p7), PVA_EXTRACT_DECL(p8)
#define PVA_DECL_9(p1, p2, p3, p4, p5, p6, p7, p8, p9)                                                            \
    PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3), PVA_EXTRACT_DECL(p4), PVA_EXTRACT_DECL(p5), \
        PVA_EXTRACT_DECL(p6), PVA_EXTRACT_DECL(p7), PVA_EXTRACT_DECL(p8), PVA_EXTRACT_DECL(p9)
#define PVA_DECL_10(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)                                                      \
    PVA_EXTRACT_DECL(p1), PVA_EXTRACT_DECL(p2), PVA_EXTRACT_DECL(p3), PVA_EXTRACT_DECL(p4), PVA_EXTRACT_DECL(p5), \
        PVA_EXTRACT_DECL(p6), PVA_EXTRACT_DECL(p7), PVA_EXTRACT_DECL(p8), PVA_EXTRACT_DECL(p9), PVA_EXTRACT_DECL(p10)

// Helper to process wrapped parameter pairs for names only
#define PVA_NAMES_1(p1) PVA_EXTRACT_NAME(p1)
#define PVA_NAMES_2(p1, p2) PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2)
#define PVA_NAMES_3(p1, p2, p3) PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3)
#define PVA_NAMES_4(p1, p2, p3, p4) \
    PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3), PVA_EXTRACT_NAME(p4)
#define PVA_NAMES_5(p1, p2, p3, p4, p5) \
    PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3), PVA_EXTRACT_NAME(p4), PVA_EXTRACT_NAME(p5)
#define PVA_NAMES_6(p1, p2, p3, p4, p5, p6)                                                                       \
    PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3), PVA_EXTRACT_NAME(p4), PVA_EXTRACT_NAME(p5), \
        PVA_EXTRACT_NAME(p6)
#define PVA_NAMES_7(p1, p2, p3, p4, p5, p6, p7)                                                                   \
    PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3), PVA_EXTRACT_NAME(p4), PVA_EXTRACT_NAME(p5), \
        PVA_EXTRACT_NAME(p6), PVA_EXTRACT_NAME(p7)
#define PVA_NAMES_8(p1, p2, p3, p4, p5, p6, p7, p8)                                                               \
    PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3), PVA_EXTRACT_NAME(p4), PVA_EXTRACT_NAME(p5), \
        PVA_EXTRACT_NAME(p6), PVA_EXTRACT_NAME(p7), PVA_EXTRACT_NAME(p8)
#define PVA_NAMES_9(p1, p2, p3, p4, p5, p6, p7, p8, p9)                                                           \
    PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3), PVA_EXTRACT_NAME(p4), PVA_EXTRACT_NAME(p5), \
        PVA_EXTRACT_NAME(p6), PVA_EXTRACT_NAME(p7), PVA_EXTRACT_NAME(p8), PVA_EXTRACT_NAME(p9)
#define PVA_NAMES_10(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)                                                     \
    PVA_EXTRACT_NAME(p1), PVA_EXTRACT_NAME(p2), PVA_EXTRACT_NAME(p3), PVA_EXTRACT_NAME(p4), PVA_EXTRACT_NAME(p5), \
        PVA_EXTRACT_NAME(p6), PVA_EXTRACT_NAME(p7), PVA_EXTRACT_NAME(p8), PVA_EXTRACT_NAME(p9), PVA_EXTRACT_NAME(p10)

// Count the number of wrapped parameter pairs
// This uses a common preprocessor trick to count arguments
#define PVA_ARG_COUNT(...) PVA_ARG_COUNT_IMPL(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define PVA_ARG_COUNT_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

// Concatenation helpers
#define PVA_CAT(a, b) PVA_CAT_IMPL(a, b)
#define PVA_CAT_IMPL(a, b) a##b

// Dispatcher macros
#define PVA_DECL(...) PVA_CAT(PVA_DECL_, PVA_ARG_COUNT(__VA_ARGS__))(__VA_ARGS__)
#define PVA_NAMES(...) PVA_CAT(PVA_NAMES_, PVA_ARG_COUNT(__VA_ARGS__))(__VA_ARGS__)

/**
 * @brief Defines implementations for cuPVA and Cuda variants of a submit function.
 *
 * @param name Base function name (e.g., pvaMorphologySubmit)
 * @param implFunc Internal implementation function name (e.g., MorphologySubmitImpl)
 * @param ... Wrapped parameter pairs (type, name), (type, name), ...
 *
 * Usage:
 *   DEFINE_PVA_SUBMIT_FUNCTIONS(pvaMorphologySubmit, MorphologySubmitImpl,
 *                                (NVCVTensorHandle, in),
 *                                (NVCVTensorHandle, out))
 *
 * Generates both cuPVA and Cuda function implementations that call the internal *SubmitImpl function.
 * Includes forward declaration of the template function (closes/reopens extern "C" as needed).
 *
 * Supports up to 10 parameters. Uses same syntax as DECLARE_PVA_SUBMIT_FUNCTIONS!
 */
#define DEFINE_PVA_SUBMIT_FUNCTIONS(name, implFunc, ...)                                             \
    }                                                                                                \
    template<typename StreamType>                                                                    \
    NVCVStatus implFunc(NVCVOperatorHandle, StreamType, PVA_DECL(__VA_ARGS__));                      \
    extern "C" {                                                                                     \
                                                                                                     \
    NVCVStatus name##__cupva(NVCVOperatorHandle handle, cupvaStream_t stream, PVA_DECL(__VA_ARGS__)) \
    {                                                                                                \
        return implFunc(handle, stream, PVA_NAMES(__VA_ARGS__));                                     \
    }                                                                                                \
                                                                                                     \
    NVCVStatus name##__cuda(NVCVOperatorHandle handle, cudaStream_t stream, PVA_DECL(__VA_ARGS__))   \
    {                                                                                                \
        return implFunc(handle, stream, PVA_NAMES(__VA_ARGS__));                                     \
    }

#endif /* PVA_SOLUTIONS_PVASUBMITMACROS_H */
