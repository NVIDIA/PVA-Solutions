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

/*
 * @file PvaAllocator.hpp
 *
 * @brief Defines types and functions to handle operators
 */

#ifndef PVA_ALLOCATOR_PRIV_HPP
#define PVA_ALLOCATOR_PRIV_HPP

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Allocator.hpp>

#include <nvcv/Tensor.h>

namespace pva_solutions::priv {

class PvaCustomAllocator
{
public:
    // Delete copy constructor and assignment operator
    PvaCustomAllocator(PvaCustomAllocator &other) = delete;
    void operator=(const PvaCustomAllocator &)    = delete;

    static NVCVResourceAllocator *Allocator()
    {
        static NVCVResourceAllocator allocator;
        static bool initialized = false;

        if (!initialized)
        {
            allocator.resType         = NVCV_RESOURCE_MEM_CUDA;
            allocator.res.mem.fnAlloc = [](void *ctx, int64_t size, int32_t align) {
                void *mem = cupva::mem::Alloc(size);
                return mem;
            };
            allocator.res.mem.fnFree = [](void *ctx, void *ptr, int64_t size, int32_t align) {
                cupva::mem::Free(ptr);
            };
            allocator.cleanup = [](void *ctx, NVCVResourceAllocator *alloc) {
            };

            initialized = true;
        }

        return &allocator;
    };

    static NVCVTensorRequirements CalcRequirements(int32_t rank, const int64_t *shape, const nvcv::DataType &dtype,
                                                   NVCVTensorLayout layout, int32_t userBaseAlign,
                                                   int32_t userRowAlign);
    static NVCVImageRequirements CalcRequirements(int32_t width, int32_t height, NVCVImageFormat format,
                                                  int32_t userBaseAlign, int32_t userRowAlign);

private:
};

} // namespace pva_solutions::priv

#endif /* PVA_ALLOCATOR_PRIV_HPP */