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

#include "OpBilateralFilter.hpp"

#include "device/bilateral_filter_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpBilateralFilter.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_bilateral_filter_K3_U8_dev)
CUPVA_DECLARE_EXECUTABLE(pva_bilateral_filter_K5_U8_dev)
CUPVA_DECLARE_EXECUTABLE(pva_bilateral_filter_K7_U8_dev)

namespace pva_solutions::priv {

template<typename T>
std::mutex BilateralFilter<T>::s_execMapMutex;

template<typename T>
std::map<typename BilateralFilter<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    BilateralFilter<T>::s_bilateralFilterExecutables;

template<typename T>
std::map<typename BilateralFilter<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP(TYPE, TYPE_NAME)                                                                  \
    template<>                                                                                                 \
    std::map<typename BilateralFilter<TYPE>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<TYPE>() \
    {                                                                                                          \
        return {                                                                                               \
            {{3},                                                                                              \
             {CUPVA_EXECUTABLE_DATA(pva_bilateral_filter_K3_##TYPE_NAME##_dev),                                \
              CUPVA_EXECUTABLE_SIZE(pva_bilateral_filter_K3_##TYPE_NAME##_dev)}},                              \
            {{5},                                                                                              \
             {CUPVA_EXECUTABLE_DATA(pva_bilateral_filter_K5_##TYPE_NAME##_dev),                                \
              CUPVA_EXECUTABLE_SIZE(pva_bilateral_filter_K5_##TYPE_NAME##_dev)}},                              \
            {{7},                                                                                              \
             {CUPVA_EXECUTABLE_DATA(pva_bilateral_filter_K7_##TYPE_NAME##_dev),                                \
              CUPVA_EXECUTABLE_SIZE(pva_bilateral_filter_K7_##TYPE_NAME##_dev)}},                              \
        };                                                                                                     \
    }

CREATE_EXEC_DATA_MAP(uint8_t, U8)

template<typename T>
std::shared_ptr<cupva::Executable> BilateralFilter<T>::GetBilateralFilterExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_bilateralFilterExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
void float2Fixed(T *dst, const float *src, const int32_t size, const int32_t qbits)
{
    // normalize values to range [-scale,scale] to preserve dynamic range before clamping to avoid overflow.
    float multiplier   = static_cast<float>(1U << qbits);
    int64_t upperBound = static_cast<int64_t>(std::numeric_limits<T>::max());
    int64_t lowerBound = static_cast<int64_t>(std::numeric_limits<T>::min());
    for (int32_t i = 0; i < size; ++i)
    {
        int64_t val = static_cast<int64_t>(src[i] * multiplier + 0.5f);
        if (val < lowerBound)
            val = lowerBound;
        else if (val > upperBound)
            val = upperBound;
        dst[i] = static_cast<T>(val);
    }
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, uint8_t>>>
void getFixedFusedWeight(T *fixedFusedWeight, const int32_t space2ParamsSize, const int32_t kernelSize,
                         const float sigmaRange, const float sigmaSpace)
{
    /**
     * @brief Precomputed squared distances from each kernel position to the center.
     *
     * This array stores L2^2 distances (dx^2 + dy^2) for each kernel position. Positions
     * where distance^2 exceeds radius^2 are marked as skipped (`/`) with weight=0.
     * Central symmetry allows compact storage of unique values.
     *
     * Storage pattern for different kernel sizes:
     *
     * 3x3 kernel (radius=1, radius^2=1)
     * ┌──────┐
     * │/ 1 / │
     * │1 0 1 │ → space2Param[2] = {1, 0}
     * │/ 1 / │
     * └──────┘
     *
     * 5x5 kernel (radius=2, radius^2=4)
     * ┌──────────┐
     * │/ / 4 / / │
     * │/ 2 1 2 / │
     * │4 1 0 1 4 │ → space2Param[4] = {4, 2, 1, 0}
     * │/ 2 1 2 / │
     * │/ / 4 / / │
     * └──────────┘
     *
     * 7x7 kernel (radius=3, radius^2=9)
     * ┌──────────────┐
     * │/ / / 9 / / / │
     * │/ 8 5 4 5 8 / │
     * │/ 5 2 1 2 5 / │
     * │9 4 1 0 1 4 9 │ → space2Param[7] = {9, 8, 5, 4, 2, 1, 0}
     * │/ 5 2 1 2 5 / │
     * │/ 8 5 4 5 8 / │
     * │/ / / 9 / / / │
     * └──────────────┘
     *
     */
    std::vector<std::vector<int32_t>> space2Params = {{1, 0, 0, 0, 0, 0, 0},
                                                      {4, 2, 1, 0, 0, 0, 0},
                                                      {9, 8, 5, 4, 2, 1, 0}};

    float rangeCoeff = -0.5 / (sigmaRange * sigmaRange);
    float spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace);

    int32_t kernelSizeIndex = (kernelSize >> 1) - 1; // when kernel size = 3/5/7, index = 0/1/2
    std::vector<float> fusedWeight;
    fusedWeight.resize(COLOR_8BIT * space2ParamsSize);

    for (int32_t i = 0; i < space2ParamsSize; i++)
    {
        for (int32_t j = 0; j < COLOR_8BIT; j++)
        {
            fusedWeight[i * COLOR_8BIT + j] =
                expf(space2Params[kernelSizeIndex][i] * spaceCoeff) * expf(j * j * rangeCoeff);
        }
    }

    float2Fixed<uint8_t>(fixedFusedWeight, fusedWeight.data(), COLOR_8BIT * space2ParamsSize, QBITS);
}

template<typename T>
BilateralFilter<T>::BilateralFilter(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize,
                                    const NVCVBorderType borderMode, int32_t borderValue)
    : m_tensorRequirements{*tensorRequirements}
    , m_kernelSize{kernelSize}
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    // Get dimensions for input image
    int64_t imageShapeNCHW[4];
    int64_t imageStridesNCHW[4];
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->shape, NVCV_TENSOR_NCHW, imageShapeNCHW);
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->strides, NVCV_TENSOR_NCHW, imageStridesNCHW);

    int32_t imageBatch     = imageShapeNCHW[0];
    int32_t imageChannels  = imageShapeNCHW[1];
    int32_t imageHeight    = imageShapeNCHW[2];
    int32_t imageWidth     = imageShapeNCHW[3];
    int32_t imageLinePitch = imageStridesNCHW[2] / (nvcv::DataType(tensorRequirements->dtype).bitsPerPixel() / 8);

    // Validate dimensions
    if (tensorRequirements->layout != nvcv::TENSOR_NHWC && tensorRequirements->layout != nvcv::TENSOR_NCHW &&
        tensorRequirements->layout != nvcv::TENSOR_HWC && tensorRequirements->layout != nvcv::TENSOR_CHW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image tensor must have NHWC, NCHW, HWC or CHW layout");
    }

    if (!(kernelSize == 3 || kernelSize == 5 || kernelSize == 7))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Filter kernel size must 3x3 or 5x5 or 7x7");
    }
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only NVCV_BORDER_CONSTANT and NVCV_BORDER_REPLICATE are supported");
    }

    m_exec    = GetBilateralFilterExec(std::make_tuple(kernelSize));
    auto prog = CmdProgram::Create(*m_exec.get());

    const int32_t radius = (kernelSize >> 1);

    prog["niterX"]      = (imageWidth + TILE_WIDTH - 1) / TILE_WIDTH;
    prog["niterY"]      = (imageHeight + TILE_HEIGHT - 1) / TILE_HEIGHT;
    prog["imageHeight"] = imageHeight;

    int32_t roiWidth      = TILE_WIDTH > imageWidth ? TILE_WIDTH : imageWidth;
    int32_t roiHeight     = TILE_HEIGHT > imageHeight ? TILE_HEIGHT : imageHeight;
    int32_t dstTileWidth  = TILE_WIDTH < imageWidth ? TILE_WIDTH : imageWidth;
    int32_t dstTileHeight = TILE_HEIGHT < imageHeight ? TILE_HEIGHT : imageHeight;

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcDataFlow.handler(prog["srcHandler"])
        .src(m_srcOp, imageWidth, imageHeight, imageLinePitch)
        .tileBuffer(prog["srcBuffer"].ptr<T>())
        .tile(TILE_WIDTH, TILE_HEIGHT)
        .roi(0, 0, roiWidth, roiHeight);

    RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    dstDataFlow.handler(prog["dstHandler"])
        .tileBuffer(prog["dstBuffer"].ptr<T>())
        .dst(m_dstOp, imageWidth, imageHeight, imageLinePitch)
        .tile(dstTileWidth, dstTileHeight)
        .tileArena(TILE_WIDTH, TILE_HEIGHT);

    if (borderMode == NVCV_BORDER_CONSTANT)
    {
        srcDataFlow.halo(radius, radius, PAD_CONST, borderValue);
    }
    else if (borderMode == NVCV_BORDER_REPLICATE)
    {
        srcDataFlow.halo(radius, radius, PAD_BPE);
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void BilateralFilter<T>::operator()(S stream, const nvcv::Tensor &in, const float sigmaRange, const float sigmaSpace,
                                    const nvcv::Tensor &out)
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
    if (sigmaRange < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "sigmaRange must be >= 0");
    }
    if (sigmaSpace < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "sigmaSpace must be >= 0");
    }

    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inTensorData->basePtr())));
    m_dstOp.setBase(static_cast<T *>(static_cast<void *>(outTensorData->basePtr())));

    int32_t space2ParamsSize;
    if (m_kernelSize == 3)
    {
        space2ParamsSize = 2;
    }
    else if (m_kernelSize == 5)
    {
        space2ParamsSize = 4;
    }
    else if (m_kernelSize == 7)
    {
        space2ParamsSize = 7;
    }

    std::vector<T> fixedFusedWeight;
    fixedFusedWeight.resize(COLOR_8BIT * space2ParamsSize);
    getFixedFusedWeight(fixedFusedWeight.data(), space2ParamsSize, m_kernelSize, sigmaRange, sigmaSpace);

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "weightTable");
    CupvaParameterSetValueArray(&param, fixedFusedWeight.data(), COLOR_8BIT * space2ParamsSize);

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for BilateralFilter operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
BilateralFilter<T>::~BilateralFilter()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class BilateralFilter<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaBilateralFilterCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                                    const int32_t kernelSize, const NVCVBorderType borderMode,
                                    const int32_t borderValue)
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
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::BilateralFilter<uint8_t>(
                tensorRequirements, kernelSize, borderMode, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus BilateralFilterSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                     const float sigmaRange, const float sigmaSpace, NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);

        NVCVDataType inDataType, outDataType;
        nvcvTensorGetDataType(in, &inDataType);
        nvcvTensorGetDataType(out, &outDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BilateralFilter<uint8_t>>(handle)(
                stream, input, sigmaRange, sigmaSpace, output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BilateralFilterSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                             const float, const float, NVCVTensorHandle);
template NVCVStatus BilateralFilterSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                            const float, const float, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaBilateralFilterSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                    const float sigmaRange, const float sigmaSpace, NVCVTensorHandle out)
{
    return BilateralFilterSubmitImpl(handle, stream, in, sigmaRange, sigmaSpace, out);
}

NVCVStatus pvaBilateralFilterSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                    const float sigmaRange, const float sigmaSpace, NVCVTensorHandle out)
{
    return BilateralFilterSubmitImpl(handle, stream, in, sigmaRange, sigmaSpace, out);
}
