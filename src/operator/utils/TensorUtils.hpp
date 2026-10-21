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
 * @file TensorUtils.hpp
 *
 * @brief Defines utility functions for tensor type's use.
 */

#ifndef TENSOR_UTILS_PRIV_HPP
#define TENSOR_UTILS_PRIV_HPP

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace pva_solutions::utils {

// A PyTorch like contiguous function. More can referred to PyTorch's documentation.
// Both "input" and "output" point to a contiguous C array of type T.
// The "input" is thought of holding of contiguous multi-dimensional array. By saying "contiguous",
// its implied shape and stride should satisfy if shape = [n0, n1, n2, n3, n4], then
// stride = [n1*n2*n3*n4,n2*n3*n4,n3*n4,n4,1]*sizeof(T)
// After introducing stride, an element having a coordinate of [x0,x1,x2,x3,x4] can be mapped into
// its actual position by x0*stride[0]+x1*stride[1]+x2*stride[2]+x3*stride[3]+x4*stride[4]
// The input "shape" and "stride" can be any permuted pattern from the above shape and stride,
// e.g. possible input: "shape"=[n0, n1, n4, n3, n2], "stride"=[n1*n2*n3*n4,1,n3*n4,n4,n2*n3*n4]*sizeof(T)
// This pattern is switching the last dimension with the last third dimension.
// One of the benefits of this permuted "shape" and "stride" is to reuse the same physical layout "input"
// to represent non-contiguous layout, e.g. operations like permutation. In other words, if traversing
// elements from a view of input "shape" and "stride", it is actually accessing the physical element in a
// non-contiguous pattern. Then this MakeContiguous function is to permute the elements, so that after call,
// "output" points to a contiguous layout, which is "shape"=[n0, n1, n4, n3, n2] and
// "stride" = [n1*n2*n3*n4,n2*n3*n4,n2*n3,n2,1]*sizeof(T)
// Note that: 1) The "output" is pointing to a pre-allocated memory, which should hold the same amount of "input" elements of type T
//            2) This function support arbitrary number of dimensions
template<typename T>
void MakeContiguous(const T *input, const std::vector<int64_t> &shape, const std::vector<int64_t> &stride,
                    T *const output)
{
    assert(shape.size() > 1 && shape.size() == stride.size());

    // a shape of {2,2,2,2,2} will generate a dims of {16,8,4,2}. Note that the dims has 1 less
    // dimension than that of shape's, whose last dimension is implied to be 1.
    std::vector<int> dims(shape.size() - 1);
    std::partial_sum(shape.crbegin(), shape.crend() - 1, dims.begin(), std::multiplies<int>());
    std::reverse(dims.begin(), dims.end());

    // decompose a value into coordinates within the dims, e.g. value = 24 and dims={16, 8, 4, 2}
    // will output {1, 1, 0, 0, 0}
    auto decompose = [](int val, std::vector<int> &dims) {
        std::vector<int> output;
        for (auto elem : dims)
        {
            auto dv = std::div(val, elem);
            output.push_back(dv.quot);
            val = dv.rem;
        }
        output.push_back(val);

        return output;
    };

    const int numel = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
    for (int i = 0; i < numel; i++)
    {
        std::vector<int> logical_index = decompose(i, dims);
        int physical_index = std::inner_product(logical_index.begin(), logical_index.end(), stride.begin(), 0);
        output[i]          = input[physical_index];
    }
}

constexpr bool AreTensorRequirementsEqual(const NVCVTensorRequirements &req, const NVCVTensorData &data)
{
    if ((req.dtype != data.dtype) || (req.layout != data.layout) || (req.rank != data.rank))
    {
        return false;
    }

    // Compare shape
    for (int i = 0; i < req.rank; i++)
    {
        if (req.shape[i] != data.shape[i])
            return false;
    }

    // Compare strides
    for (int i = 0; i < req.rank; i++)
    {
        if (req.strides[i] != data.buffer.strided.strides[i])
            return false;
    }

    return true;
}

// Helper function to create tensor from external PVA memory (zero-copy)
inline NVCVStatus createTensorFromPvaMemory(NVCVTensorHandle *tensorHandle, const NVCVTensorRequirements &requirements,
                                            void *pvaPtr)
{
    if (!tensorHandle || !pvaPtr)
    {
        return NVCV_ERROR_INVALID_ARGUMENT;
    }

    // Prepare tensor data structure pointing to external PVA memory
    NVCVTensorData tensorData;
    tensorData.dtype  = requirements.dtype;
    tensorData.layout = requirements.layout;
    tensorData.rank   = requirements.rank;
    memcpy(tensorData.shape, requirements.shape, sizeof(tensorData.shape));
    tensorData.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    memcpy(&tensorData.buffer.strided.strides, &requirements.strides, sizeof(requirements.strides));

    // Point directly to external PVA memory (zero-copy!)
    tensorData.buffer.strided.basePtr = (NVCVByte *)pvaPtr;

    // Wrap the data as a tensor (no cleanup function since we don't own the memory)
    return nvcvTensorWrapDataConstruct(&tensorData, nullptr, nullptr, tensorHandle);
}

// Throws unless every stride of `req` equals the element size times the product of the inner dimensions.
inline void CheckDenselyPacked(const NVCVTensorRequirements &req, const char *name)
{
    int32_t elemBytes = 0;
    nvcvDataTypeGetStrideBytes(req.dtype, &elemBytes);
    int64_t packedStride = elemBytes;
    for (int32_t i = req.rank - 1; i >= 0; i--)
    {
        if (req.strides[i] != packedStride)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "%s tensor must be densely packed; its %c stride is %ld bytes, expected %ld", name,
                                  req.layout.data[i], static_cast<long>(req.strides[i]),
                                  static_cast<long>(packedStride));
        }
        packedStride *= req.shape[i];
    }
}

} // namespace pva_solutions::utils

#endif