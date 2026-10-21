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

#include "OpImageHistogram.hpp"

#include "device/image_histogram_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpImageHistogram.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>
#include <limits>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_image_histogram_u8_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_histogram_u16_16w_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_histogram_u16_8w_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_histogram_u16_4w_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_histogram_u16_2w_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex ImageHistogram<T>::s_execMapMutex;

template<typename T>
std::map<typename ImageHistogram<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    ImageHistogram<T>::s_imageHistogramExecutables;

/*
 * A wider histogram retires more pixels per issue but spends a counter per way on every bin, so the
 * width is the widest one whose table still holds numBins.
 */
static int32_t waysForNumBins(int32_t const numBins)
{
    if (numBins <= NUM_BINS_MAX_16WAY)
    {
        return 16;
    }
    if (numBins <= NUM_BINS_MAX_8WAY)
    {
        return 8;
    }
    if (numBins <= NUM_BINS_MAX_4WAY)
    {
        return 4;
    }
    return 2;
}

template<typename T>
std::shared_ptr<cupva::Executable> ImageHistogram<T>::GetImageHistogramExec(int32_t const ways)
{
    IOperator::ExecutableData execData{};

    if constexpr (std::is_same<T, uint8_t>::value)
    {
        execData = {CUPVA_EXECUTABLE_DATA(pva_image_histogram_u8_dev),
                    CUPVA_EXECUTABLE_SIZE(pva_image_histogram_u8_dev)};
    }
    else if constexpr (std::is_same<T, uint16_t>::value)
    {
        switch (ways)
        {
        case 16:
            execData = {CUPVA_EXECUTABLE_DATA(pva_image_histogram_u16_16w_dev),
                        CUPVA_EXECUTABLE_SIZE(pva_image_histogram_u16_16w_dev)};
            break;
        case 8:
            execData = {CUPVA_EXECUTABLE_DATA(pva_image_histogram_u16_8w_dev),
                        CUPVA_EXECUTABLE_SIZE(pva_image_histogram_u16_8w_dev)};
            break;
        case 4:
            execData = {CUPVA_EXECUTABLE_DATA(pva_image_histogram_u16_4w_dev),
                        CUPVA_EXECUTABLE_SIZE(pva_image_histogram_u16_4w_dev)};
            break;
        default:
            execData = {CUPVA_EXECUTABLE_DATA(pva_image_histogram_u16_2w_dev),
                        CUPVA_EXECUTABLE_SIZE(pva_image_histogram_u16_2w_dev)};
            break;
        }
    }

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent(), ways);
    return IOperator::GetExecutable<ExecKey>(execData, execKey, s_imageHistogramExecutables, s_execMapMutex);
}

template<typename T>
ImageHistogram<T>::ImageHistogram(NVCVTensorRequirements const *const inTensorRequirements,
                                  NVCVTensorRequirements const *const outTensorRequirements,
                                  PvaImageHistogramParams const *const imageHistogramParams)
    : m_inTensorRequirements(*inTensorRequirements)
    , m_outTensorRequirements(*outTensorRequirements)
    , m_start(imageHistogramParams->start)
    , m_end(imageHistogramParams->end)
    , m_numBins(imageHistogramParams->numBins)
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<uint32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    // Get dimensions for input image
    int64_t imageShapeNCHW[4];
    int64_t imageStridesNCHW[4];
    nvcvTensorShapePermute(inTensorRequirements->layout, inTensorRequirements->shape, NVCV_TENSOR_NCHW, imageShapeNCHW);
    nvcvTensorShapePermute(inTensorRequirements->layout, inTensorRequirements->strides, NVCV_TENSOR_NCHW,
                           imageStridesNCHW);

    int32_t imageBatch    = imageShapeNCHW[0];
    int32_t imageChannels = imageShapeNCHW[1];
    m_height              = imageShapeNCHW[2];
    m_width               = imageShapeNCHW[3];
    m_inputLinePitch      = imageStridesNCHW[2] / (nvcv::DataType(inTensorRequirements->dtype).bitsPerPixel() / 8);

    if ((inTensorRequirements->layout != nvcv::TENSOR_HWC) && (inTensorRequirements->layout != nvcv::TENSOR_NHWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor must have either HWC or NHWC layout");
    }
    if ((imageBatch != 0 && imageBatch != 1) || (imageChannels != 0 && imageChannels != 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image tensor must have N=1, C=1");
    }

    if ((m_start < 0) || (m_start > (std::numeric_limits<T>::max() + 1)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Start is out of range");
    }
    if ((m_end < 0) || (m_end > (std::numeric_limits<T>::max() + 1)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "End is out of range");
    }
    if (m_start >= m_end)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Start must be smaller than end");
    }
    if ((m_numBins <= 0) || (std::is_same<T, uint8_t>::value && (m_numBins > (std::numeric_limits<T>::max() + 1))) ||
        (std::is_same<T, uint16_t>::value && (m_numBins > NUM_BINS_MAX)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "NumBins is out of range");
    }

    m_exec    = GetImageHistogramExec(std::is_same<T, uint8_t>::value ? 1 : waysForNumBins(m_numBins));
    auto prog = CmdProgram::Create(*m_exec.get());

    const int32_t niterX = (m_width + TILE_W - 1) / TILE_W;
    const int32_t niterY = (m_height + TILE_H - 1) / TILE_H;

    prog["niterX"]      = niterX;
    prog["niterY"]      = niterY;
    prog["imageWidth"]  = m_width;
    prog["imageHeight"] = m_height;

    prog["start"]   = static_cast<int32_t>(m_start);
    prog["end"]     = static_cast<int32_t>(m_end);
    prog["numBins"] = m_numBins;

    /*
     * Index scale. The device computes (v - start) * a >> qbits, so a is the bins-per-raw-value
     * slope in qbits fractional bits. Truncating a costs accuracy in proportion to how far along
     * the range a sample sits, so take the largest qbits whose multiplier still fits the int32
     * parameter and whose product stays inside the multiply's 48-bit lane.
     */
    int32_t scaleQ = 0;
    int32_t scaleA = 0;
    {
        double const slope    = static_cast<double>(m_numBins) / (static_cast<double>(m_end) - m_start);
        int64_t const maxSpan = static_cast<int64_t>(std::numeric_limits<T>::max());
        for (int32_t q = 30; q >= 1; q--)
        {
            double const scaled = std::ldexp(slope, q);
            if ((scaled < 1.0) || (scaled > static_cast<double>(std::numeric_limits<int32_t>::max())))
            {
                continue;
            }
            int64_t const a = std::llround(scaled);
            if ((maxSpan * a) >= (static_cast<int64_t>(1) << 47))
            {
                continue;
            }
            scaleQ = q;
            scaleA = static_cast<int32_t>(a);
            break;
        }
    }
    prog["scaleA"] = scaleA;
    prog["scaleQ"] = scaleQ;

    auto dType = []() {
        if (std::is_same<T, uint8_t>::value)
        {
            return static_cast<int32_t>(IMAGE_FORMAT_U8);
        }
        else if (std::is_same<T, uint16_t>::value)
        {
            return static_cast<int32_t>(IMAGE_FORMAT_U16);
        }
    };

    prog["format"] = dType();

    T *src_v        = prog["src_v"].ptr<T>();
    uint32_t *dst_v = prog["dst_v"].ptr<uint32_t>();

    auto srcHdl = prog["srcHdl"];
    auto dstHdl = prog["dstHdl"];

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcDataFlow.handler(srcHdl)
        .src(m_srcOp, m_width, m_height, m_inputLinePitch)
        .tileBuffer(src_v)
        .tile(TILE_W, TILE_H);

    SequenceDataFlow &dstDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    dstDataFlow.handler(dstHdl)
        .addTransfer()
        .src(dst_v, sizeof(uint32_t))
        .dst(m_dstOp, sizeof(uint32_t))
        .tile(sizeof(uint32_t), m_numBins)
        .mode(TransferModeType::CONTINUOUS);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void ImageHistogram<T>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
{
    auto inTensorData  = in.exportData<nvcv::TensorDataStrided>();
    auto outTensorData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outTensorRequirements, outTensorData->cdata()))
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

    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inTensorData->basePtr())));
    m_dstOp.setBase(static_cast<uint32_t *>(static_cast<void *>(outTensorData->basePtr())));
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
        static_assert(!sizeof(S), "Unsupported stream type for ImageHistogram operator");
    }
}

// Explicit template instantiation
template void ImageHistogram<uint8_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                 const nvcv::Tensor &);
template void ImageHistogram<uint8_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                const nvcv::Tensor &);
template void ImageHistogram<uint16_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                  const nvcv::Tensor &);
template void ImageHistogram<uint16_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                 const nvcv::Tensor &);

template<typename T>
ImageHistogram<T>::~ImageHistogram()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class ImageHistogram<uint8_t>;
template class ImageHistogram<uint16_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaImageHistogramCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                   NVCVTensorRequirements const *const outTensorRequirements,
                                   PvaImageHistogramParams const *const imageHistogramParams)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to Input TensorRequirements must not be NULL");
        }
        if (outTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to Output TensorRequirements must not be NULL");
        }
        if (imageHistogramParams == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to ImageHistogramParams must not be NULL");
        }
        if ((outTensorRequirements->dtype != NVCV_DATA_TYPE_U32) &&
            (outTensorRequirements->dtype != NVCV_DATA_TYPE_S32))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported output data type");
        }
        switch (inTensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ImageHistogram<uint8_t>(
                inTensorRequirements, outTensorRequirements, imageHistogramParams));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ImageHistogram<uint16_t>(
                inTensorRequirements, outTensorRequirements, imageHistogramParams));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input data type");
        }
    });
}

template<typename StreamType>
NVCVStatus ImageHistogramSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const in,
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
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ImageHistogram<uint8_t>>(handle)(stream, input,
                                                                                                    output);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ImageHistogram<uint16_t>>(handle)(stream, input,
                                                                                                     output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ImageHistogramSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                            NVCVTensorHandle);
template NVCVStatus ImageHistogramSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                           NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaImageHistogramSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out)
{
    return ImageHistogramSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaImageHistogramSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out)
{
    return ImageHistogramSubmitImpl(handle, stream, in, out);
}
