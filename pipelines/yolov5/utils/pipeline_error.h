/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef YOLOV5_PIPELINE_ERROR_H
#define YOLOV5_PIPELINE_ERROR_H

#include <cuda_runtime.h>
#include <nvcv/Status.h>

#include <cstdio>

#define YOLOV5_CUDA_CHECK(call)                                                                            \
    do                                                                                                     \
    {                                                                                                      \
        cudaError_t status_ = (call);                                                                      \
        if (status_ != cudaSuccess)                                                                        \
        {                                                                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            return -1;                                                                                     \
        }                                                                                                  \
    } while (0)

#define YOLOV5_NVCV_CHECK(call)                                                        \
    do                                                                                 \
    {                                                                                  \
        NVCVStatus status_ = (call);                                                   \
        if (status_ != NVCV_SUCCESS)                                                   \
        {                                                                              \
            fprintf(stderr, "NVCV error at %s:%d: %d\n", __FILE__, __LINE__, status_); \
            return -1;                                                                 \
        }                                                                              \
    } while (0)

#endif // YOLOV5_PIPELINE_ERROR_H
