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

#include "OpORBDescExtractor.hpp"

#include "device/orb_desc_extr_params.h"
#include "device/orb_desc_extr_patterns.h"
#include "device/weights_interleaved.h"

#include <ImageUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpORBDescExtractor.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>

#include <array>
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

using namespace cupva;

// BRIEF descriptor extractor
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_u8_brief_dev);
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_u16_brief_dev);
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_s8_brief_dev);
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_s16_brief_dev);

// rBRIEF descriptor extractor, concern the orientation of each keypoint
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_u8_rbrief_dev);
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_u16_rbrief_dev);
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_s8_rbrief_dev);
CUPVA_DECLARE_EXECUTABLE(pva_orb_desc_extractor_s16_rbrief_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex ORBDescExtractor<T>::s_execMapMutex;

template<typename T>
std::map<typename ORBDescExtractor<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    ORBDescExtractor<T>::s_ORBDescExtractorExecutables;

// define executable data map entry for each image format
template<typename T>
std::map<typename ORBDescExtractor<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP_ENTRY(FORMAT, TYPE_SUFFIX)                                  \
    {{FORMAT, true},                                                                     \
     {CUPVA_EXECUTABLE_DATA(pva_orb_desc_extractor_##TYPE_SUFFIX##_brief_dev),           \
      CUPVA_EXECUTABLE_SIZE(pva_orb_desc_extractor_##TYPE_SUFFIX##_brief_dev)}},         \
    {                                                                                    \
        {FORMAT, false},                                                                 \
        {                                                                                \
            CUPVA_EXECUTABLE_DATA(pva_orb_desc_extractor_##TYPE_SUFFIX##_rbrief_dev),    \
                CUPVA_EXECUTABLE_SIZE(pva_orb_desc_extractor_##TYPE_SUFFIX##_rbrief_dev) \
        }                                                                                \
    }

// executable data map for data type of a single image format
#define CREATE_EXEC_DATA_MAP_SINGLE_FORMAT(TYPE, FORMAT, TYPE_SUFFIX)                                           \
    template<>                                                                                                  \
    std::map<typename ORBDescExtractor<TYPE>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<TYPE>() \
    {                                                                                                           \
        return {CREATE_EXEC_DATA_MAP_ENTRY(FORMAT, TYPE_SUFFIX)};                                               \
    }

CREATE_EXEC_DATA_MAP_SINGLE_FORMAT(int8_t, NVCV_IMAGE_FORMAT_S8, s8)
CREATE_EXEC_DATA_MAP_SINGLE_FORMAT(int16_t, NVCV_IMAGE_FORMAT_S16, s16)

// executable data map for data type shared by multiple image formats
template<>
std::map<typename ORBDescExtractor<uint8_t>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<uint8_t>()
{
    return {CREATE_EXEC_DATA_MAP_ENTRY(NVCV_IMAGE_FORMAT_U8, u8), CREATE_EXEC_DATA_MAP_ENTRY(NVCV_IMAGE_FORMAT_Y8, u8),
            CREATE_EXEC_DATA_MAP_ENTRY(NVCV_IMAGE_FORMAT_Y8_ER, u8)};
}

template<>
std::map<typename ORBDescExtractor<uint16_t>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<uint16_t>()
{
    return {CREATE_EXEC_DATA_MAP_ENTRY(NVCV_IMAGE_FORMAT_U16, u16),
            CREATE_EXEC_DATA_MAP_ENTRY(NVCV_IMAGE_FORMAT_Y16, u16),
            CREATE_EXEC_DATA_MAP_ENTRY(NVCV_IMAGE_FORMAT_Y16_ER, u16)};
}

template<typename T>
std::shared_ptr<cupva::Executable> ORBDescExtractor<T>::GetORBDescExtractorExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_ORBDescExtractorExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
};

template<typename T>
void ORBDescExtractor<T>::checkImageFormat(NVCVImageFormat format)
{
    // all supported image formats
    std::set<NVCVImageFormat> formatSet = {NVCV_IMAGE_FORMAT_U8,     NVCV_IMAGE_FORMAT_Y8,  NVCV_IMAGE_FORMAT_Y8_ER,
                                           NVCV_IMAGE_FORMAT_S8,     NVCV_IMAGE_FORMAT_U16, NVCV_IMAGE_FORMAT_Y16,
                                           NVCV_IMAGE_FORMAT_Y16_ER, NVCV_IMAGE_FORMAT_S16};
    auto formatIt                       = formatSet.find(format);
    if (formatIt == formatSet.end())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }
};

template<typename T>

ORBDescExtractor<T>::ORBDescExtractor(const NVCVImageRequirements *inImageRequirements,
                                      const NVCVTensorRequirements *inCornersTensorRequirements,
                                      const NVCVTensorRequirements *outDescriptorsTensorRequirements,
                                      uint32_t disableRBRIEFFlag, NVCVBorderType borderType, int32_t borderValue)
    : m_inImageRequirements(*inImageRequirements)
    , m_inCornersTensorRequirements(*inCornersTensorRequirements)
    , m_outDescriptorsTensorRequirements(*outDescriptorsTensorRequirements)
    , m_disableRBRIEFFlag(disableRBRIEFFlag)
    , m_borderType(borderType)
    , m_borderValue(borderValue)
    , m_inImageOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_inCornersOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_outDescriptorsOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    // check whether the image format is supported
    m_imageFormat = inImageRequirements->format;
    checkImageFormat(m_imageFormat);

    m_exec    = GetORBDescExtractorExec(std::make_tuple(m_imageFormat, m_disableRBRIEFFlag));
    auto prog = CmdProgram::Create(*m_exec.get());

    // get image sizes
    uint16_t imageWidth     = inImageRequirements->width;
    uint16_t imageHeight    = inImageRequirements->height;
    uint16_t imageLinePitch = inImageRequirements->planeRowStride[0] / sizeof(T);

    // get padding mode
    cupva::PadModeType padMode = (borderType == NVCV_BORDER_REPLICATE) ? PAD_BPE : PAD_CONST;

    // get batch size
    int32_t batch_size;
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
    {
        batch_size = BATCH_SIZE_8_BIT;
    }
    else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>)
    {
        batch_size = BATCH_SIZE_16_BIT;
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
    }

    // get input feature tensor shape
    if (inCornersTensorRequirements->layout != nvcv::TENSOR_HW ||
        inCornersTensorRequirements->dtype != NVCV_DATA_TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input corners tensor must be of layout HW and data type f32");
    }
    int32_t numFeatures  = inCornersTensorRequirements->shape[0];
    int32_t keypointSize = inCornersTensorRequirements->shape[1];
    if (keypointSize != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Keypoint size must be 2");
    }

    // check output descriptors tensor configuration
    if (outDescriptorsTensorRequirements->layout != nvcv::TENSOR_HW ||
        outDescriptorsTensorRequirements->dtype != NVCV_DATA_TYPE_U8 ||
        outDescriptorsTensorRequirements->shape[0] != numFeatures ||
        outDescriptorsTensorRequirements->shape[1] != PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output descriptors tensor must be of layout HW and data type u8, and shape must be "
                              "[numFeatures, PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH]");
    }

    uint8_t *out_descriptor = (uint8_t *)prog["g_opt_descriptor"].ptr<uint8_t>();
    float *feature_coords   = (float *)prog["g_feature_coords"].ptr<float>();

    int32_t n_iterations = (numFeatures + batch_size - 1) / batch_size;

    // last batch size
    int32_t has_last_batch  = (numFeatures % batch_size == 0) ? 0 : 1;
    int32_t last_batch_size = (int32_t)(numFeatures - (n_iterations - 1) * batch_size);
    prog["last_batch_size"] = last_batch_size;
    prog["has_last_batch"]  = has_last_batch;
    prog["n_iterations"]    = n_iterations;
    prog["keypointSize"]    = (int32_t)(keypointSize);

    // input data flow for image patches
    auto gsdf_handler                  = prog["gsdf_handler"];
    GatherScatterDataFlow &imgDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>();
    imgDataFlow.handler(gsdf_handler)
        .numTilesPerTrigger(batch_size)
        .src(m_inImageOp, imageWidth, imageHeight, imageLinePitch)
        .tile(PATCH_WIDTH, PATCH_HEIGHT)
        .padVal(padMode, m_borderValue);

    // input data flow for keypoint coordinates
    auto sqdf_handler_in = prog["sqdf_handler_in"];

    // first tile batch size
    batch_size = std::min(batch_size, numFeatures);

    int32_t kpt_tile_size_bytes    = keypointSize * batch_size * sizeof(float);
    SequenceDataFlow &patchaddFlow = prog.addDataFlowHead<cupva::SequenceDataFlow>().handler(sqdf_handler_in);
    patchaddFlow.addTransfer()
        .src(m_inCornersOp, kpt_tile_size_bytes)
        .dst(feature_coords, kpt_tile_size_bytes)
        .tile(kpt_tile_size_bytes, 1);

    // output data flow for descriptors
    int descriptor_tile_size_bytes = DESCRIPTOR_ARRAY_NUM_BYTES * batch_size;
    auto sqdf_handler_out          = prog["sqdf_handler_out"];
    SequenceDataFlow &outDataFlow  = prog.addDataFlowHead<cupva::SequenceDataFlow>().handler(sqdf_handler_out);
    outDataFlow.addTransfer()
        .src(out_descriptor, descriptor_tile_size_bytes)
        .dst(m_outDescriptorsOp, descriptor_tile_size_bytes)
        .tile(descriptor_tile_size_bytes, 1);

    // load constants from header files
    prog["weights"].set(weightsInterleaved, WEIGHTS_ARRAY_SIZE);
    prog["orb_bit_pattern"].set(orbBitPatterns, 2 * DESCRIPTOR_PATTERN_SIZE);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void ORBDescExtractor<T>::operator()(S stream, const nvcv::Image &inImage, const nvcv::Tensor &inCorners,
                                     const nvcv::Tensor &outDescriptors)
{
    // check input image data
    auto inImageData = inImage.exportData<nvcv::ImageDataStrided>();
    if (!pva_solutions::utils::AreImageRequirementsEqual(m_inImageRequirements, inImageData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input image tensor must have the same requirements that was used to create the operator");
    }

    // check input corners tensor data
    auto inCornersTensorData = inCorners.exportData<nvcv::TensorDataStrided>();
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inCornersTensorRequirements, inCornersTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input corners tensor must have the same requirements that was used to create the operator");
    }

    // check output descriptors tensor data
    auto outDescriptorsTensorData = outDescriptors.exportData<nvcv::TensorDataStrided>();
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_outDescriptorsTensorRequirements,
                                                          outDescriptorsTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Output descriptors tensor must have the same requirements that was used to create the operator");
    }

    // check base pointers
    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(inCornersTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outDescriptorsTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image or tensor base pointers must not be NULL");
    }
    m_inImageOp.setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(0).basePtr)));
    m_inCornersOp.setBase(static_cast<float *>(static_cast<void *>(inCornersTensorData->basePtr())));
    m_outDescriptorsOp.setBase(static_cast<uint8_t *>(static_cast<void *>(outDescriptorsTensorData->basePtr())));

    // set external memory pointers for SQDF
    cupvaParameter_t inCornersExtPtr, outDescriptorsExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &inCornersExtPtr, "feature_list_dram_addr");
    CupvaCmdProgramGetParameter(&m_prog, &outDescriptorsExtPtr, "descriptor_dram_addr");

    CupvaParameterSetValuePointer(&inCornersExtPtr, static_cast<void *>(inCornersTensorData->basePtr()));
    CupvaParameterSetValuePointer(&outDescriptorsExtPtr, static_cast<void *>(outDescriptorsTensorData->basePtr()));

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Program Update Data Flows failed");
    }

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
        static_assert(!sizeof(S), "Unsupported stream type for ORBDescExtractor operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
ORBDescExtractor<T>::~ORBDescExtractor()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class ORBDescExtractor<uint8_t>;
template class ORBDescExtractor<int8_t>;
template class ORBDescExtractor<uint16_t>;
template class ORBDescExtractor<int16_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaORBDescExtractorCreate(NVCVOperatorHandle *handle, const NVCVImageRequirements *inImageRequirements,
                                     const NVCVTensorRequirements *inCornersTensorRequirements,
                                     const NVCVTensorRequirements *outDescriptorsTensorRequirements,
                                     uint32_t disableRBRIEFFlag, NVCVBorderType borderType, int32_t borderValue)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inImageRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVImageRequirements must not be NULL");
        }

        switch (inImageRequirements->format)
        {
        case NVCV_IMAGE_FORMAT_U8:
        case NVCV_IMAGE_FORMAT_Y8:
        case NVCV_IMAGE_FORMAT_Y8_ER:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ORBDescExtractor<uint8_t>(
                inImageRequirements, inCornersTensorRequirements, outDescriptorsTensorRequirements, disableRBRIEFFlag,
                borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ORBDescExtractor<int8_t>(
                inImageRequirements, inCornersTensorRequirements, outDescriptorsTensorRequirements, disableRBRIEFFlag,
                borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_U16:
        case NVCV_IMAGE_FORMAT_Y16:
        case NVCV_IMAGE_FORMAT_Y16_ER:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ORBDescExtractor<uint16_t>(
                inImageRequirements, inCornersTensorRequirements, outDescriptorsTensorRequirements, disableRBRIEFFlag,
                borderType, borderValue));
            break;
        case NVCV_IMAGE_FORMAT_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::ORBDescExtractor<int16_t>(
                inImageRequirements, inCornersTensorRequirements, outDescriptorsTensorRequirements, disableRBRIEFFlag,
                borderType, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format!");
        }
    });
}

template<typename StreamType>
NVCVStatus ORBDescExtractorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle inImageHandle,
                                      NVCVTensorHandle inCornersTensorHandle,
                                      NVCVTensorHandle outDescriptorsTensorHandle)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle inImage(inImageHandle);
        nvcv::TensorWrapHandle inCorners(inCornersTensorHandle);
        nvcv::TensorWrapHandle outDescriptors(outDescriptorsTensorHandle);

        NVCVImageFormat inFormat;
        nvcvImageGetFormat(inImageHandle, &inFormat);

        switch (inFormat)
        {
        case NVCV_IMAGE_FORMAT_U8:
        case NVCV_IMAGE_FORMAT_Y8:
        case NVCV_IMAGE_FORMAT_Y8_ER:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ORBDescExtractor<uint8_t>>(handle)(
                stream, inImage, inCorners, outDescriptors);
            break;
        case NVCV_IMAGE_FORMAT_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ORBDescExtractor<int8_t>>(handle)(
                stream, inImage, inCorners, outDescriptors);
            break;
        case NVCV_IMAGE_FORMAT_U16:
        case NVCV_IMAGE_FORMAT_Y16:
        case NVCV_IMAGE_FORMAT_Y16_ER:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ORBDescExtractor<uint16_t>>(handle)(
                stream, inImage, inCorners, outDescriptors);
            break;
        case NVCV_IMAGE_FORMAT_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ORBDescExtractor<int16_t>>(handle)(
                stream, inImage, inCorners, outDescriptors);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format!");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ORBDescExtractorSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                              NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus ORBDescExtractorSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                             NVCVTensorHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaORBDescExtractorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle inImageHandle,
                                     NVCVTensorHandle inCornersTensorHandle,
                                     NVCVTensorHandle outDescriptorsTensorHandle)
{
    return ORBDescExtractorSubmitImpl(handle, stream, inImageHandle, inCornersTensorHandle, outDescriptorsTensorHandle);
}

NVCVStatus pvaORBDescExtractorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle inImageHandle,
                                     NVCVTensorHandle inCornersTensorHandle,
                                     NVCVTensorHandle outDescriptorsTensorHandle)
{
    return ORBDescExtractorSubmitImpl(handle, stream, inImageHandle, inCornersTensorHandle, outDescriptorsTensorHandle);
}
