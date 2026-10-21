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

#include "OpGaussianFilter.hpp"

#include "primitive_utils.hpp"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpGaussianFilter.h>
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
std::mutex GaussianFilter<T>::s_execMapMutex;

template<typename T>
std::map<typename GaussianFilter<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    GaussianFilter<T>::s_gaussianFilterExecutables;

template<typename T>
std::map<typename GaussianFilter<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP(TYPE, TYPE_NAME)                                                                 \
    template<>                                                                                                \
    std::map<typename GaussianFilter<TYPE>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<TYPE>() \
    {                                                                                                         \
        return {                                                                                              \
            {{3},                                                                                             \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K3_##TYPE_NAME##_dev),                                           \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K3_##TYPE_NAME##_dev)}},                                         \
            {{5},                                                                                             \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K5_##TYPE_NAME##_dev),                                           \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K5_##TYPE_NAME##_dev)}},                                         \
            {{7},                                                                                             \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K7_##TYPE_NAME##_dev),                                           \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K7_##TYPE_NAME##_dev)}},                                         \
        };                                                                                                    \
    }

CREATE_EXEC_DATA_MAP(uint8_t, U8)
CREATE_EXEC_DATA_MAP(int8_t, S8)
CREATE_EXEC_DATA_MAP(uint16_t, U16)
CREATE_EXEC_DATA_MAP(int16_t, S16)

// should make this static, otherwise same name function in conv2d will link to this
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

    dst[size / 2] += (1 << qbits) - sum;
}

void get_1D_gaussian_kernel(float *outKernel, uint32_t kernelSize, float sigma)
{
    int len        = kernelSize / 2;
    const double s = 2.0 * sigma * sigma;
    double sum     = 0.0;

    for (int x = -len; x <= len; x++)
    {
        double r       = x * x;
        uint32_t ind   = x + len;
        outKernel[ind] = std::exp(-r / s) / std::sqrt(M_PI * s);
        sum += outKernel[ind];
    }

    for (uint32_t i = 0; i < kernelSize; i++)
    {
        outKernel[i] /= sum;
        // For PVA coefficients must be strictly < 1. So for contrived
        // cases, let's enforce that.
        if (outKernel[i] == 1)
        {
            outKernel[i] -= 1e-5;
        }
    }
}

template<typename T>
void get_2D_gaussian_kernel(T *outKernel, uint32_t kernelSize, float sigmaX, float sigmaY, int qbits)
{
    float *kernelX = new float[kernelSize];
    float *kernelY = new float[kernelSize];

    float *kernelFloat = new float[kernelSize * kernelSize];

    get_1D_gaussian_kernel(kernelX, kernelSize, sigmaX);
    get_1D_gaussian_kernel(kernelY, kernelSize, sigmaY);

    for (size_t i = 0; i < kernelSize; ++i)
    {
        for (size_t j = 0; j < kernelSize; ++j)
        {
            kernelFloat[i * kernelSize + j] = kernelY[i] * kernelX[j];
        }
    }

    Float2Fixed<T>(outKernel, kernelFloat, kernelSize * kernelSize, qbits);
}
template<typename T>
GaussianFilter<T>::GaussianFilter(NVCVTensorRequirements *tensorRequirements, float sigmaX, float sigmaY,
                                  int32_t kernelSize, const NVCVBorderType borderMode, int32_t borderValue)
    : BaseConv2d<T, GaussianFilter<T>>(tensorRequirements, kernelSize, borderMode, borderValue)
    , m_sigmaX(sigmaX)
    , m_sigmaY(sigmaY)
{
    // Initialize the program after construction is complete
    this->initializeProgram();
}

template<typename T>
void GaussianFilter<T>::prepareKernel(void *outKernel, uint32_t kernelSize, int qbits)
{
    get_2D_gaussian_kernel<T>(static_cast<T *>(outKernel), kernelSize, this->m_sigmaX, this->m_sigmaY, qbits);
}

template<typename T>
GaussianFilter<T>::~GaussianFilter()
{
    // Access base class member variable through this pointer
    CupvaCmdDestroy(&this->m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class GaussianFilter<uint8_t>;
template class GaussianFilter<int8_t>;
template class GaussianFilter<uint16_t>;
template class GaussianFilter<int16_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaGaussianFilterCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements, float sigmaX,
                                   float sigmaY, int32_t kernelSize, const NVCVBorderType borderMode,
                                   int32_t borderValue)
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
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianFilter<uint8_t>(
                tensorRequirements, sigmaX, sigmaY, kernelSize, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianFilter<int8_t>(
                tensorRequirements, sigmaX, sigmaY, kernelSize, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianFilter<uint16_t>(
                tensorRequirements, sigmaX, sigmaY, kernelSize, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::GaussianFilter<int16_t>(
                tensorRequirements, sigmaX, sigmaY, kernelSize, borderMode, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus GaussianFilterSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                    NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);

        NVCVDataType inDataType, outDataType;
        nvcvTensorGetDataType(in, &inDataType);
        nvcvTensorGetDataType(out, &outDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianFilter<uint8_t>>(handle)(stream, input,
                                                                                                    output);
            break;
        case NVCV_DATA_TYPE_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianFilter<int8_t>>(handle)(stream, input,
                                                                                                   output);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianFilter<uint16_t>>(handle)(stream, input,
                                                                                                     output);
            break;
        case NVCV_DATA_TYPE_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::GaussianFilter<int16_t>>(handle)(stream, input,
                                                                                                    output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus GaussianFilterSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                            NVCVTensorHandle);
template NVCVStatus GaussianFilterSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                           NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaGaussianFilterSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                   NVCVTensorHandle out)
{
    return GaussianFilterSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaGaussianFilterSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                   NVCVTensorHandle out)
{
    return GaussianFilterSubmitImpl(handle, stream, in, out);
}
