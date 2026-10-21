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

#include <PvaAllocator.hpp>
#include <nvcv/Exception.hpp>

#include <PvaAllocator.h>
#include <nvcv/Status.h>

NVCVStatus nvcvTensorCalcRequirementsPva(int32_t rank, const int64_t *shape, NVCVDataType dtype,
                                         NVCVTensorLayout layout, int32_t baseAlign, int32_t rowAlign,
                                         NVCVTensorRequirements *reqs)
{
    return nvcv::ProtectCall([&] {
        if (reqs == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output requirements must not be NULL");
        }

        nvcv::DataType type{dtype};

        *reqs =
            pva_solutions::priv::PvaCustomAllocator::CalcRequirements(rank, shape, type, layout, baseAlign, rowAlign);
    });
}

NVCVStatus nvcvImageCalcRequirementsPva(int32_t width, int32_t height, NVCVImageFormat format, int32_t baseAlign,
                                        int32_t rowAlign, NVCVImageRequirements *reqs)
{
    return nvcv::ProtectCall([&] {
        if (reqs == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output requirements must not be NULL");
        }

        *reqs = pva_solutions::priv::PvaCustomAllocator::CalcRequirements(width, height, format, baseAlign, rowAlign);
    });
}

NVCVStatus nvcvAllocatorConstructPva(NVCVAllocatorHandle *handle)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
        }

        nvcvAllocatorConstructCustom(pva_solutions::priv::PvaCustomAllocator::Allocator(), 1, handle);
    });
}