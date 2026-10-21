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

#include "OpBoxFilter.hpp"

#include "primitive_utils.hpp"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBoxFilter.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_conv_K3_U8_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K3_S8_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K5_U8_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K5_S8_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K7_U8_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K7_S8_dev)

CUPVA_DECLARE_EXECUTABLE(pva_conv_K3_U16_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K3_S16_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K5_U16_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K5_S16_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K7_U16_dev)
CUPVA_DECLARE_EXECUTABLE(pva_conv_K7_S16_dev)

namespace pva_solutions::priv {

template<typename T>
std::mutex BoxFilter<T>::s_execMapMutex;

template<typename T>
std::map<typename BoxFilter<T>::ExecKey, std::weak_ptr<cupva::Executable>> BoxFilter<T>::s_boxFilterExecutables;

template<typename T>
std::map<typename BoxFilter<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP(TYPE, TYPE_NAME)                                                            \
    template<>                                                                                           \
    std::map<typename BoxFilter<TYPE>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<TYPE>() \
    {                                                                                                    \
        return {                                                                                         \
            {{3},                                                                                        \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K3_##TYPE_NAME##_dev),                                      \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K3_##TYPE_NAME##_dev)}},                                    \
            {{5},                                                                                        \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K5_##TYPE_NAME##_dev),                                      \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K5_##TYPE_NAME##_dev)}},                                    \
            {{7},                                                                                        \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K7_##TYPE_NAME##_dev),                                      \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K7_##TYPE_NAME##_dev)}},                                    \
        };                                                                                               \
    }

CREATE_EXEC_DATA_MAP(uint8_t, U8)
CREATE_EXEC_DATA_MAP(int8_t, S8)
CREATE_EXEC_DATA_MAP(uint16_t, U16)
CREATE_EXEC_DATA_MAP(int16_t, S16)

template<typename T>
std::shared_ptr<cupva::Executable> BoxFilter<T>::GetExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return IOperator::GetExecutable<ExecKey>(execDataIt->second, execKey, s_boxFilterExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

template<typename T>
static void Float2Fixed(T *dst, const float *src, const int32_t size, const int qbits)
{
    // normalize values to range [-scale,scale] to preserve dynamic range before clamping to avoid overflow.
    int64_t sum        = 0;
    float multiplier   = static_cast<float>(1U << qbits) / 1.0;
    int64_t upperBound = static_cast<int64_t>(std::numeric_limits<T>::max());
    int64_t lowerBound = static_cast<int64_t>(std::numeric_limits<T>::min());
    for (int32_t i = 0; i < size; ++i)
    {
        int64_t val = static_cast<int64_t>(src[i] * multiplier);
        val         = std::min(val, upperBound);
        val         = std::max(val, lowerBound);
        dst[i]      = static_cast<T>(val);
        sum += dst[i];
    }

    // For box filter, ensure sum equals (1 << qbits)
    int64_t expected_sum = static_cast<int64_t>(1U << qbits);
    int64_t diff         = expected_sum - sum;

    // Distribute the difference among the kernel elements
    if (diff != 0)
    {
        int32_t elements_per_adjust = size / std::abs(diff);
        if (elements_per_adjust == 0)
            elements_per_adjust = 1;

        for (int32_t i = 0; i < size && diff != 0; i += elements_per_adjust)
        {
            if (diff > 0)
            {
                dst[i] = std::min(static_cast<int64_t>(dst[i]) + 1, upperBound);
                diff--;
            }
            else
            {
                dst[i] = std::max(static_cast<int64_t>(dst[i]) - 1, lowerBound);
                diff++;
            }
        }
    }
}

template<typename T>
void get_2D_box_kernel(T *outKernel, uint32_t kernelSize, int qbits)
{
    float *kernelFloat = new float[kernelSize * kernelSize];
    float kernelValue  = 1.0f / (kernelSize * kernelSize);

    // Fill kernel with uniform values
    for (size_t i = 0; i < kernelSize * kernelSize; ++i)
    {
        kernelFloat[i] = kernelValue;
    }

    Float2Fixed<T>(outKernel, kernelFloat, kernelSize * kernelSize, qbits);

    delete[] kernelFloat;
}

template<typename T>
BoxFilter<T>::BoxFilter(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize, const NVCVBorderType borderMode,
                        int32_t borderValue)
    : BaseConv2d<T, BoxFilter<T>>(tensorRequirements, kernelSize, borderMode, borderValue)
{
    // Initialize the program after construction is complete
    this->initializeProgram();
}

template<typename T>
void BoxFilter<T>::prepareKernel(void *outKernel, uint32_t kernelSize, int qbits)
{
    get_2D_box_kernel<T>(static_cast<T *>(outKernel), kernelSize, qbits);
}

template<typename T>
BoxFilter<T>::~BoxFilter()
{
    // Access base class member variable through this pointer
    CupvaCmdDestroy(&this->m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class BoxFilter<uint8_t>;
template class BoxFilter<int8_t>;
template class BoxFilter<uint16_t>;
template class BoxFilter<int16_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaBoxFilterCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                              int32_t kernelSize, const NVCVBorderType borderMode, int32_t borderValue)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (tensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to TensorRequirements must not be NULL");
        }
        switch (tensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::BoxFilter<uint8_t>(tensorRequirements, kernelSize, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::BoxFilter<int8_t>(tensorRequirements, kernelSize, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::BoxFilter<uint16_t>(tensorRequirements, kernelSize, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::BoxFilter<int16_t>(tensorRequirements, kernelSize, borderMode, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus BoxFilterSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);

        NVCVDataType inDataType, outDataType;
        nvcvTensorGetDataType(in, &inDataType);
        nvcvTensorGetDataType(out, &outDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BoxFilter<uint8_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BoxFilter<int8_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BoxFilter<uint16_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BoxFilter<int16_t>>(handle)(stream, input, output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BoxFilterSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                       NVCVTensorHandle);
template NVCVStatus BoxFilterSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                      NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaBoxFilterSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                              NVCVTensorHandle out)
{
    return BoxFilterSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaBoxFilterSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return BoxFilterSubmitImpl(handle, stream, in, out);
}
