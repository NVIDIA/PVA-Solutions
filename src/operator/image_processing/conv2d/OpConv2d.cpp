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

#include "OpConv2d.hpp"

#include "../box_filter/OpBoxFilter.hpp"
#include "../gaussian_filter/OpGaussianFilter.hpp"
#include "primitive_utils.hpp"

#include "device/conv2d_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpGaussianFilter.h>
#include <cupva_host.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/Tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

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
std::mutex Conv2d<T>::s_execMapMutex;

template<typename T>
std::map<typename Conv2d<T>::ExecKey, std::weak_ptr<cupva::Executable>> Conv2d<T>::s_conv2dExecutables;

template<typename T>
std::map<typename Conv2d<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP(TYPE, TYPE_NAME)                                                         \
    template<>                                                                                        \
    std::map<typename Conv2d<TYPE>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<TYPE>() \
    {                                                                                                 \
        return {                                                                                      \
            {{3},                                                                                     \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K3_##TYPE_NAME##_dev),                                   \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K3_##TYPE_NAME##_dev)}},                                 \
            {{5},                                                                                     \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K5_##TYPE_NAME##_dev),                                   \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K5_##TYPE_NAME##_dev)}},                                 \
            {{7},                                                                                     \
             {CUPVA_EXECUTABLE_DATA(pva_conv_K7_##TYPE_NAME##_dev),                                   \
              CUPVA_EXECUTABLE_SIZE(pva_conv_K7_##TYPE_NAME##_dev)}},                                 \
        };                                                                                            \
    }

CREATE_EXEC_DATA_MAP(uint8_t, U8)
CREATE_EXEC_DATA_MAP(int8_t, S8)
CREATE_EXEC_DATA_MAP(uint16_t, U16)
CREATE_EXEC_DATA_MAP(int16_t, S16)

template<typename T, typename Derived>
std::shared_ptr<cupva::Executable> BaseConv2d<T, Derived>::GetExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, Derived::GetExecMap(), Derived::GetExecMutex());
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

template<typename T, typename Derived>
BaseConv2d<T, Derived>::BaseConv2d(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize,
                                   NVCVBorderType borderMode, int32_t borderValue)
    : m_tensorRequirements(*tensorRequirements)
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
    , m_kernelSize(kernelSize)
    , m_borderMode(borderMode)
    , m_borderValue(borderValue)
{
    NVCVTensorLayout refLayout;
    nvcvTensorLayoutMake("HWC", &refLayout);

    // Common validation
    if (nvcvTensorLayoutCompare(m_tensorRequirements.layout, refLayout) != 0)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have HWC layout");
    if (!(m_tensorRequirements.shape[2] == 1))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have 1 channel");
    if (!(kernelSize == 3 || kernelSize == 5 || kernelSize == 7))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Filter kernel size must 3x3 or 5x5 or 7x7");
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only NVCV_BORDER_CONSTANT and NVCV_BORDER_REPLICATE are supported");

    m_exec = GetExec(std::make_tuple(kernelSize));
}

template<typename T, typename Derived>
void BaseConv2d<T, Derived>::initializeProgram()
{
    auto prog    = CmdProgram::Create(*m_exec.get());
    const int iw = m_tensorRequirements.shape[1];
    const int ih = m_tensorRequirements.shape[0];
    const int lp = m_tensorRequirements.strides[0] / sizeof(T);

    const int kw = m_kernelSize, kh = m_kernelSize;
    const int kwa = ((kw - 1) / 4 + 1) * 4;
    const int kha = ((kh - 1) / 2 + 1) * 2;
    const int kwr = kw >> 1;
    const int khr = kh >> 1;

    int tha = ((((TH / 2) - 1) / 3) + 1) * 3;

    T *src_even_v = prog["src_even_v"].ptr<T>();
    T *src_odd_v  = prog["src_odd_v"].ptr<T>();
    T *dst_v      = prog["dst_v"].ptr<T>();

    auto srcEvenHdl = prog["srcEvenHdl"];
    auto srcOddHdl  = prog["srcOddHdl"];
    auto dstHdl     = prog["dstHdl"];

    using KT = std::conditional_t<sizeof(T) == 1, uint8_t, uint16_t>;
    std::vector<KT> knl_ref(kw * kh);
    std::vector<KT> knl_d(kwa * kha * 2);
    int qbits = sizeof(T) == 1 ? QBITS_8 : QBITS_16;

    static_cast<Derived *>(this)->prepareKernel(knl_ref.data(), m_kernelSize, qbits);
    pva_solutions::utils::prepareKernelForConv2dVpu(knl_ref.data(), knl_d.data(), kw);
    prog["knl_v"].set(knl_d.data(), kwa * kha * 2);

    prog["iw"]         = iw;
    prog["ih"]         = ih;
    prog["borderMode"] = m_borderMode == NVCV_BORDER_REPLICATE ? 1 : 0;

    // Data flow setup, common
    RasterDataFlow &srcEvenDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcEvenDataFlow.handler(srcEvenHdl)
        .src(m_srcOp, iw, ih / 2, lp * 2)
        .tileBuffer(src_even_v)
        .roi(0, 0, iw, ih / 2)
        .tile(TW, TH / 2);

    RasterDataFlow &srcOddDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcOddDataFlow.handler(srcOddHdl)
        .src((m_srcOp + lp), iw, ih / 2, lp * 2)
        .tileBuffer(src_odd_v)
        .roi(0, 0, iw, ih / 2)
        .tile(TW, TH / 2);

    if (m_borderMode == NVCV_BORDER_REPLICATE)
    {
        srcEvenDataFlow.halo(kwr, (khr + 1) / 2, PAD_BPE);
        srcOddDataFlow.halo(kwr, (khr + 1) / 2, PAD_BPE);
    }
    else
    {
        srcEvenDataFlow.halo(kwr, (khr + 1) / 2, PAD_CONST, m_borderValue);
        srcOddDataFlow.halo(kwr, (khr + 1) / 2, PAD_CONST, m_borderValue);
    }

    RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    dstDataFlow.handler(dstHdl).tileBuffer(dst_v).tile(TW, TH).tileArena(TW, tha * 2).dst(m_dstOp, iw, ih, lp);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T, typename Derived>
template<typename S>
void BaseConv2d<T, Derived>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
{
    auto inTensorData  = in.exportData<nvcv::TensorDataStrided>();
    auto outTensorData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, outTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    // Access base class member variables directly (they are protected and accessible to derived classes)
    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inTensorData->basePtr())));
    m_dstOp.setBase(static_cast<T *>(static_cast<void *>(outTensorData->basePtr())));
    CUPVA_CHECK_ERROR(CupvaCmdProgramUpdateDataFlows(&m_prog));

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        CUPVA_CHECK_ERROR(CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1));
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            CUPVA_CHECK_ERROR(CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1));
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for Conv2d operator");
    }
}

template<typename T>
Conv2d<T>::Conv2d(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize, const NVCVBorderType borderMode,
                  int32_t borderValue, NVCVTensorHandle kernelCoefficients)
    : BaseConv2d<T, Conv2d<T>>(tensorRequirements, kernelSize, borderMode, borderValue)
    , m_kernelCoefficients(kernelCoefficients)
{
    // Initialize the program after construction is complete
    this->initializeProgram();
}

template<typename T>
void Conv2d<T>::prepareKernel(void *outKernel, uint32_t kernelSize, int qbits)
{
    // Get kernel data
    NVCVTensorData kernelData;
    nvcvTensorExportData(m_kernelCoefficients, &kernelData);

    // Get kernel pointer and copy to outKernel
    T *kernelPtr = NULL;
    CupvaMemGetHostPointer((void **)&kernelPtr, (void *)kernelData.buffer.strided.basePtr);

    // Copy kernel data to outKernel buffer
    std::memcpy(outKernel, kernelPtr, kernelSize * kernelSize * sizeof(T));
}

template<typename T>
template<typename S>
void Conv2d<T>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
{
    // Call the base class operator() implementation
    this->BaseConv2d<T, Conv2d<T>>::operator()(stream, in, out);
}

template<typename T>
Conv2d<T>::~Conv2d()
{
    // Access base class member variable through this pointer
    CupvaCmdDestroy(&this->m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class Conv2d<uint8_t>;
template class Conv2d<int8_t>;
template class Conv2d<uint16_t>;
template class Conv2d<int16_t>;

// Explicit instantiation of BaseConv2d with GaussianFilter for the types we care about, b/c GaussianFilter and BaseConv2d
// are in different compilation units
template class BaseConv2d<uint8_t, pva_solutions::priv::GaussianFilter<uint8_t>>;
template class BaseConv2d<int8_t, pva_solutions::priv::GaussianFilter<int8_t>>;
template class BaseConv2d<uint16_t, pva_solutions::priv::GaussianFilter<uint16_t>>;
template class BaseConv2d<int16_t, pva_solutions::priv::GaussianFilter<int16_t>>;
// Explicit instantiation of BaseConv2d with BoxFilter for the types we care about, b/c BoxFilter and BaseConv2d
// are in different compilation units
template class BaseConv2d<uint8_t, pva_solutions::priv::BoxFilter<uint8_t>>;
template class BaseConv2d<int8_t, pva_solutions::priv::BoxFilter<int8_t>>;
template class BaseConv2d<uint16_t, pva_solutions::priv::BoxFilter<uint16_t>>;
template class BaseConv2d<int16_t, pva_solutions::priv::BoxFilter<int16_t>>;

// Explicit instantiation of BaseConv2d::operator() for both stream types
template void BaseConv2d<uint8_t, Conv2d<uint8_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                              const nvcv::Tensor &);
template void BaseConv2d<uint8_t, Conv2d<uint8_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                             const nvcv::Tensor &);
template void BaseConv2d<int8_t, Conv2d<int8_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                            const nvcv::Tensor &);
template void BaseConv2d<int8_t, Conv2d<int8_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                           const nvcv::Tensor &);
template void BaseConv2d<uint16_t, Conv2d<uint16_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                                const nvcv::Tensor &);
template void BaseConv2d<uint16_t, Conv2d<uint16_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                               const nvcv::Tensor &);
template void BaseConv2d<int16_t, Conv2d<int16_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                              const nvcv::Tensor &);
template void BaseConv2d<int16_t, Conv2d<int16_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                             const nvcv::Tensor &);

template void BaseConv2d<uint8_t, GaussianFilter<uint8_t>>::operator()<cupvaStream_t>(cupvaStream_t,
                                                                                      const nvcv::Tensor &,
                                                                                      const nvcv::Tensor &);
template void BaseConv2d<uint8_t, GaussianFilter<uint8_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                                     const nvcv::Tensor &);
template void BaseConv2d<int8_t, GaussianFilter<int8_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                                    const nvcv::Tensor &);
template void BaseConv2d<int8_t, GaussianFilter<int8_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                                   const nvcv::Tensor &);
template void BaseConv2d<uint16_t, GaussianFilter<uint16_t>>::operator()<cupvaStream_t>(cupvaStream_t,
                                                                                        const nvcv::Tensor &,
                                                                                        const nvcv::Tensor &);
template void BaseConv2d<uint16_t, GaussianFilter<uint16_t>>::operator()<cudaStream_t>(cudaStream_t,
                                                                                       const nvcv::Tensor &,
                                                                                       const nvcv::Tensor &);
template void BaseConv2d<int16_t, GaussianFilter<int16_t>>::operator()<cupvaStream_t>(cupvaStream_t,
                                                                                      const nvcv::Tensor &,
                                                                                      const nvcv::Tensor &);
template void BaseConv2d<int16_t, GaussianFilter<int16_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                                     const nvcv::Tensor &);

template void BaseConv2d<uint8_t, BoxFilter<uint8_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                                 const nvcv::Tensor &);
template void BaseConv2d<uint8_t, BoxFilter<uint8_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                                const nvcv::Tensor &);
template void BaseConv2d<int8_t, BoxFilter<int8_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                               const nvcv::Tensor &);
template void BaseConv2d<int8_t, BoxFilter<int8_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                              const nvcv::Tensor &);
template void BaseConv2d<uint16_t, BoxFilter<uint16_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                                   const nvcv::Tensor &);
template void BaseConv2d<uint16_t, BoxFilter<uint16_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                                  const nvcv::Tensor &);
template void BaseConv2d<int16_t, BoxFilter<int16_t>>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                                 const nvcv::Tensor &);
template void BaseConv2d<int16_t, BoxFilter<int16_t>>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                                const nvcv::Tensor &);

} // namespace pva_solutions::priv

NVCVStatus pvaConv2dCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                           const NVCVBorderType borderMode, int32_t borderValue, NVCVTensorHandle kernelCoefficients)
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
        if (kernelCoefficients == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to kernelCoefficients must not be NULL");
        }

        // Extract kernel size from kernelCoefficients tensor
        NVCVTensorData kernelData;
        NVCVStatus status = nvcvTensorExportData(kernelCoefficients, &kernelData);
        if (status != NVCV_SUCCESS)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Failed to export kernel tensor data");
        }

        int32_t kernelSize = static_cast<int32_t>(kernelData.shape[0]);

        // Validate kernel layout and dimensions
        NVCVTensorLayout refLayout;
        nvcvTensorLayoutMake("HWC", &refLayout);

        if (nvcvTensorLayoutCompare(kernelData.layout, refLayout) != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Kernel must have HWC layout");
        }

        // Check kernel channel count (C == 1)
        if (kernelData.shape[2] != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Kernel must have 1 channel");
        }

        // Validate kernel dimensions match kernel size
        if (kernelData.shape[0] != kernelData.shape[1])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Kernel must be square (height equals width)");
        }

        // Check datatype width matches input tensor
        int32_t tensorBitsPerPixel, kernelBitsPerPixel;
        if (nvcvDataTypeGetBitsPerPixel(tensorRequirements->dtype, &tensorBitsPerPixel) != NVCV_SUCCESS ||
            nvcvDataTypeGetBitsPerPixel(kernelData.dtype, &kernelBitsPerPixel) != NVCV_SUCCESS)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Failed to get data type information");
        }

        if (tensorBitsPerPixel != kernelBitsPerPixel)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Kernel data type width must match input tensor data type width");
        }

        switch (tensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Conv2d<uint8_t>(
                tensorRequirements, kernelSize, borderMode, borderValue, kernelCoefficients));
            break;
        case NVCV_DATA_TYPE_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Conv2d<int8_t>(
                tensorRequirements, kernelSize, borderMode, borderValue, kernelCoefficients));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Conv2d<uint16_t>(
                tensorRequirements, kernelSize, borderMode, borderValue, kernelCoefficients));
            break;
        case NVCV_DATA_TYPE_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Conv2d<int16_t>(
                tensorRequirements, kernelSize, borderMode, borderValue, kernelCoefficients));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus Conv2dSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);

        NVCVDataType inDataType, outDataType;
        nvcvTensorGetDataType(in, &inDataType);
        nvcvTensorGetDataType(out, &outDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Conv2d<uint8_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Conv2d<int8_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Conv2d<uint16_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Conv2d<int16_t>>(handle)(stream, input, output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus Conv2dSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                    NVCVTensorHandle);
template NVCVStatus Conv2dSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                   NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaConv2dSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return Conv2dSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaConv2dSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out)
{
    return Conv2dSubmitImpl(handle, stream, in, out);
}
