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

#include "OpImageStats.hpp"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>

#include <OpImageStats.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <set>

using namespace cupva;

// Define macro for image stats executable declarations
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_u8_cnt_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_u8_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_u8_var_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_u8_mask_cnt_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_u8_mask_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_u8_mask_cnt_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_u8_mask_var_dev);

CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_cnt_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_var_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_cov_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_mask_cnt_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_mask_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_mask_cnt_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_mask_var_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_rgb_mask_cov_dev);

CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_cnt_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_var_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_cov_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_mask_cnt_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_mask_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_mask_cnt_sum_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_mask_var_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_stats_nv12_mask_cov_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex ImageStats<T>::s_execMapMutex;

template<typename T>
std::map<typename ImageStats<T>::ExecKey, std::weak_ptr<cupva::Executable>> ImageStats<T>::s_imageStatsExecutables;

template<typename T>
std::map<typename ImageStats<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP_ENTRY_NO_MASK(FORMAT, NAME_SUFFIX, OP_TYPE, OP_SUFFIX)      \
    {                                                                                    \
        {FORMAT, false, OP_TYPE},                                                        \
        {                                                                                \
            CUPVA_EXECUTABLE_DATA(pva_image_stats_##NAME_SUFFIX##_##OP_SUFFIX##_dev),    \
                CUPVA_EXECUTABLE_SIZE(pva_image_stats_##NAME_SUFFIX##_##OP_SUFFIX##_dev) \
        }                                                                                \
    }

#define CREATE_EXEC_DATA_MAP_ENTRY_WITH_MASK(FORMAT, NAME_SUFFIX, OP_TYPE, OP_SUFFIX)         \
    {                                                                                         \
        {FORMAT, true, OP_TYPE},                                                              \
        {                                                                                     \
            CUPVA_EXECUTABLE_DATA(pva_image_stats_##NAME_SUFFIX##_mask_##OP_SUFFIX##_dev),    \
                CUPVA_EXECUTABLE_SIZE(pva_image_stats_##NAME_SUFFIX##_mask_##OP_SUFFIX##_dev) \
        }                                                                                     \
    }

// mutual executables for all image formats
#define CREATE_EXEC_DATA_MAP_FORMAT_MUTUAL(FORMAT, NAME_SUFFIX)                                      \
    CREATE_EXEC_DATA_MAP_ENTRY_NO_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_CNT, cnt),               \
        CREATE_EXEC_DATA_MAP_ENTRY_NO_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_SUM, sum),           \
        CREATE_EXEC_DATA_MAP_ENTRY_NO_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_VAR, var),           \
        CREATE_EXEC_DATA_MAP_ENTRY_WITH_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_CNT, cnt),         \
        CREATE_EXEC_DATA_MAP_ENTRY_WITH_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_SUM, sum),         \
        CREATE_EXEC_DATA_MAP_ENTRY_WITH_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_CNT_SUM, cnt_sum), \
        CREATE_EXEC_DATA_MAP_ENTRY_WITH_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_VAR, var)

// multi-channel executables: covariance
#define CREATE_EXEC_DATA_MAP_FORMAT_MULTI_CHANNEL(FORMAT, NAME_SUFFIX)                 \
    CREATE_EXEC_DATA_MAP_ENTRY_NO_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_COV, cov), \
        CREATE_EXEC_DATA_MAP_ENTRY_WITH_MASK(FORMAT, NAME_SUFFIX, IMAGE_STAT_EXEC_COV, cov)

template<>
std::map<typename ImageStats<uint8_t>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<uint8_t>()
{
    return {
        // mutual executables
        CREATE_EXEC_DATA_MAP_FORMAT_MUTUAL(NVCV_IMAGE_FORMAT_U8, u8),
        CREATE_EXEC_DATA_MAP_FORMAT_MUTUAL(NVCV_IMAGE_FORMAT_RGB8, rgb),
        CREATE_EXEC_DATA_MAP_FORMAT_MUTUAL(NVCV_IMAGE_FORMAT_BGR8, rgb),
        CREATE_EXEC_DATA_MAP_FORMAT_MUTUAL(NVCV_IMAGE_FORMAT_NV12, nv12),
        CREATE_EXEC_DATA_MAP_FORMAT_MUTUAL(NVCV_IMAGE_FORMAT_NV12_ER, nv12),
        // multi-channel executables
        CREATE_EXEC_DATA_MAP_FORMAT_MULTI_CHANNEL(NVCV_IMAGE_FORMAT_RGB8, rgb),
        CREATE_EXEC_DATA_MAP_FORMAT_MULTI_CHANNEL(NVCV_IMAGE_FORMAT_BGR8, rgb),
        CREATE_EXEC_DATA_MAP_FORMAT_MULTI_CHANNEL(NVCV_IMAGE_FORMAT_NV12, nv12),
        CREATE_EXEC_DATA_MAP_FORMAT_MULTI_CHANNEL(NVCV_IMAGE_FORMAT_NV12_ER, nv12),
    };
}

template<typename T>
std::shared_ptr<cupva::Executable> ImageStats<T>::GetImageStatsExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_imageStatsExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
};

template<typename T>
void ImageStats<T>::unpackFlagsToOpType(uint32_t flags, bool hasMask, NVCVImageFormat format)
{
    m_pixelCountFlag = (flags & PVA_IMAGE_STAT_FLAG_PIXEL_COUNT) == PVA_IMAGE_STAT_FLAG_PIXEL_COUNT;
    m_sumFlag        = (flags & PVA_IMAGE_STAT_FLAG_SUM) == PVA_IMAGE_STAT_FLAG_SUM;
    m_meanFlag       = (flags & PVA_IMAGE_STAT_FLAG_MEAN) == PVA_IMAGE_STAT_FLAG_MEAN;
    m_varianceFlag   = (flags & PVA_IMAGE_STAT_FLAG_VARIANCE) == PVA_IMAGE_STAT_FLAG_VARIANCE;
    m_covarianceFlag = (flags & PVA_IMAGE_STAT_FLAG_COVARIANCE) == PVA_IMAGE_STAT_FLAG_COVARIANCE;

    // get op type from flags and hasMask
    // covariance device executable only for multi-channel formats
    if (m_covarianceFlag && format != NVCV_IMAGE_FORMAT_U8)
    {
        m_opType = IMAGE_STAT_EXEC_COV;
    }
    else if (m_varianceFlag)
    {
        m_opType = IMAGE_STAT_EXEC_VAR;
    }
    else if (m_sumFlag && m_pixelCountFlag && hasMask)
    {
        m_opType = IMAGE_STAT_EXEC_CNT_SUM;
    }
    else if (m_sumFlag)
    {
        m_opType = IMAGE_STAT_EXEC_SUM;
    }
    else
    {
        m_opType = IMAGE_STAT_EXEC_CNT;
    }
}

static std::set<NVCVImageFormat> formatSet = {NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_BGR8,
                                              NVCV_IMAGE_FORMAT_NV12, NVCV_IMAGE_FORMAT_NV12_ER};

template<typename T>
void ImageStats<T>::checkImageFormat(NVCVImageFormat format)
{
    auto formatIt = formatSet.find(format);
    if (formatIt == formatSet.end())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }
};

template<typename T>
ImageStats<T>::ImageStats(NVCVImageRequirements *imageRequirements, NVCVImageRequirements *maskRequirements,
                          uint32_t flags)
    : m_imageRequirements(*imageRequirements)
    , m_srcImageHeight(imageRequirements->height)
    , m_srcImageWidth(imageRequirements->width)
    , m_srcOp{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_maskOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_outOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    m_imageFormat = imageRequirements->format;
    checkImageFormat(m_imageFormat);

    // for NV12 and NV12_ER, image sizes must be even
    if ((m_imageFormat == NVCV_IMAGE_FORMAT_NV12 || m_imageFormat == NVCV_IMAGE_FORMAT_NV12_ER) &&
        (m_srcImageWidth % 2 != 0 || m_srcImageHeight % 2 != 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "For image format NV12 and NV12_ER, image sizes must be even");
    }

    // check mask image
    m_hasMask = (maskRequirements != nullptr);
    if (m_hasMask)
    {
        m_maskRequirements = *maskRequirements;
        if (maskRequirements->format != NVCV_IMAGE_FORMAT_U8)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask image must be of format U8");
        }
        if (maskRequirements->width != m_srcImageWidth || maskRequirements->height != m_srcImageHeight)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask image must have the same size as input image");
        }
    }

    // get op type from flags
    unpackFlagsToOpType(flags, m_hasMask, m_imageFormat);

    m_exec    = GetImageStatsExec(std::make_tuple(m_imageFormat, m_hasMask, m_opType));
    auto prog = CmdProgram::Create(*m_exec.get());

    nvcvImageFormatGetNumPlanes(m_imageFormat, &m_planeCount);

    // get plane dimensions
    uint8_t planeChannelNum[MAX_PLANE_COUNT];
    uint16_t planeWidth[MAX_PLANE_COUNT];
    uint16_t planeHeight[MAX_PLANE_COUNT];
    uint16_t planeLinePitch[MAX_PLANE_COUNT];
    uint8_t xSampleRatio[MAX_PLANE_COUNT];
    uint8_t ySampleRatio[MAX_PLANE_COUNT];

    for (int p = 0; p < m_planeCount; p++)
    {
        // subsampling ratio for NV12
        xSampleRatio[p] = (p > 0) ? 2 : 1;
        ySampleRatio[p] = (p > 0) ? 2 : 1;
        planeWidth[p]   = (m_srcImageWidth + xSampleRatio[p] - 1) / xSampleRatio[p];
        planeHeight[p]  = (m_srcImageHeight + ySampleRatio[p] - 1) / ySampleRatio[p];

        int32_t numChannels;
        nvcvImageFormatGetPlaneNumChannels(m_imageFormat, p, &numChannels);
        planeChannelNum[p] = numChannels;

        planeLinePitch[p] = m_imageRequirements.planeRowStride[p] / sizeof(T);
    }

    prog["planeCount"] = m_planeCount;
    prog["planeChannelNum"].set(planeChannelNum, MAX_PLANE_COUNT);
    prog["planeWidth"].set(planeWidth, MAX_PLANE_COUNT);
    prog["planeHeight"].set(planeHeight, MAX_PLANE_COUNT);
    prog["xSampleRatio"].set(xSampleRatio, MAX_PLANE_COUNT);
    prog["ySampleRatio"].set(ySampleRatio, MAX_PLANE_COUNT);
    prog["planeLinePitch"].set(planeLinePitch, MAX_PLANE_COUNT);

    prog["pixelCountFlag"] = (uint32_t)m_pixelCountFlag;
    prog["sumFlag"]        = (uint32_t)m_sumFlag;
    prog["meanFlag"]       = (uint32_t)m_meanFlag;
    prog["varianceFlag"]   = (uint32_t)m_varianceFlag;
    prog["covarianceFlag"] = (uint32_t)m_covarianceFlag;

    // get tile info
    uint16_t tileWidth[MAX_PLANE_COUNT];
    uint16_t tileHeight[MAX_PLANE_COUNT];

    uint16_t tileW = TILE_WIDTH;
    uint16_t tileH = TILE_HEIGHT;

    // check RDF padding: must be less than 256
    uint16_t padX = ((m_srcImageWidth + tileW - 1) / tileW * tileW - m_srcImageWidth) * planeChannelNum[0];
    if (padX > 255)
    {
        tileW /= 2;
    }

    uint8_t ntileX = (m_srcImageWidth + tileW - 1) / tileW;
    uint8_t ntileY = (m_srcImageHeight + tileH - 1) / tileH;

    bool transposeFlag = (m_imageFormat == NVCV_IMAGE_FORMAT_RGB8 || m_imageFormat == NVCV_IMAGE_FORMAT_BGR8);

    // for cnt only op without mask, no need to load the src image
    if (m_opType != IMAGE_STAT_EXEC_CNT || m_hasMask)
    {
        // init data flows
        for (int p = 0; p < m_planeCount; p++)
        {
            // get tile sizes
            tileWidth[p]  = tileW / xSampleRatio[p];
            tileHeight[p] = tileH / ySampleRatio[p];

            // RDF for input data flow
            std::string srcBufName = "srcBuf" + std::to_string(p);
            T *srcBuf              = prog[srcBufName.c_str()].ptr<T>();

            std::string srcHdlName = "srcRDFHdl" + std::to_string(p);
            auto srcRDFHdl         = prog[srcHdlName.c_str()];

            RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>(0);
            srcDataFlow.handler(srcRDFHdl)
                .src(m_srcOp[p], planeWidth[p] * planeChannelNum[p], planeHeight[p], planeLinePitch[p])
                .tileBuffer(srcBuf)
                .tile(tileWidth[p] * planeChannelNum[p], tileHeight[p])
                .roi(0, 0, tileWidth[p] * planeChannelNum[p] * ntileX, tileHeight[p] * ntileY);

            // transpose load for RGB8/BGR8
            if (transposeFlag)
            {
                srcDataFlow.transpose(TRANS_MODE_1);
            }
        }
    }

    if (m_hasMask)
    {
        // RDF for mask data flow
        uint16_t maskLinePitch = m_maskRequirements.planeRowStride[0] / sizeof(uint8_t);

        uint8_t *maskBuf             = prog["maskBuf"].ptr<uint8_t>();
        auto maskRDFHdl              = prog["maskRDFHdl"];
        RasterDataFlow &maskDataFlow = prog.addDataFlowHead<RasterDataFlow>(0);
        maskDataFlow.handler(maskRDFHdl)
            .src(m_maskOp, m_srcImageWidth, m_srcImageHeight, maskLinePitch)
            .tileBuffer(maskBuf)
            .tile(tileWidth[0], tileHeight[0])
            .roi(0, 0, tileWidth[0] * ntileX, tileHeight[0] * ntileY)
            .halo(0, PAD_CONST, 0);

        // transpose load for RGB8/BGR8
        if (transposeFlag)
        {
            maskDataFlow.transpose(TRANS_MODE_1);
        }
    }

    prog["ntileX"] = ntileX;
    prog["ntileY"] = ntileY;
    prog["tileWidth"].set(tileWidth, MAX_PLANE_COUNT);
    prog["tileHeight"].set(tileHeight, MAX_PLANE_COUNT);

    uint8_t *outBuf = prog["outBuf"].ptr<uint8_t>();

    int outBufSize                = sizeof(PVAImageStatOutput);
    auto dstSQDFHdl               = prog["dstSQDFHdl"];
    SequenceDataFlow &dstDataFlow = prog.addDataFlowHead<SequenceDataFlow>().handler(dstSQDFHdl);
    dstDataFlow.addTransfer().src(outBuf, outBufSize).dst(m_outOp, outBufSize).tile(outBufSize, 1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void ImageStats<T>::operator()(S stream, const nvcv::Image &in, const nvcv::Image &mask, const nvcv::Tensor &out_t)
{
    auto inImageData   = in.exportData<nvcv::ImageDataStrided>();
    auto outTensorData = out_t.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_imageRequirements, inImageData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image must have the same requirement that was used to create the operator");
    }
    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image or tensor base pointers must not be NULL");
    }

    for (int32_t i = 0; i < m_planeCount; i++)
    {
        m_srcOp[i].setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(i).basePtr)));
    }
    m_outOp.setBase(static_cast<uint8_t *>(static_cast<void *>(outTensorData->basePtr())));

    if (m_hasMask)
    {
        auto maskImageData = mask.exportData<nvcv::ImageDataStrided>();
        if (!pva_solutions::utils::AreImageRequirementsEqual(m_maskRequirements, maskImageData->cdata()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Mask image must have the same requirement that was used to create the operator");
        }
        if (static_cast<void *>(maskImageData->plane(0).basePtr) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask image base pointer must not be NULL");
        }
        m_maskOp.setBase(static_cast<uint8_t *>(static_cast<void *>(maskImageData->plane(0).basePtr)));
    }

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
        static_assert(!sizeof(S), "Unsupported stream type for ImageStats operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
ImageStats<T>::~ImageStats()
{
    CupvaCmdDestroy(&m_prog);
}

template class ImageStats<uint8_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaImageStatsCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements,
                               NVCVImageRequirements *maskRequirements, uint32_t flags)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (imageRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to input ImageRequirements must not be NULL");
        }

        NVCVDataType dataType;
        nvcvImageFormatGetPlaneDataType(imageRequirements->format, 0, &dataType);
        switch (dataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_3U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::ImageStats<uint8_t>(imageRequirements, maskRequirements, flags));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus ImageStatsSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle mask,
                                NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle in_t(in);
        nvcv::ImageWrapHandle mask_t(mask);
        nvcv::TensorWrapHandle out_t(out);

        NVCVImageFormat inFormat;
        nvcvImageGetFormat(in, &inFormat);

        NVCVDataType inDataType;
        nvcvImageFormatGetPlaneDataType(inFormat, 0, &inDataType);
        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_3U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ImageStats<uint8_t>>(handle)(stream, in_t, mask_t,
                                                                                                out_t);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ImageStatsSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                        NVCVImageHandle, NVCVTensorHandle);
template NVCVStatus ImageStatsSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                       NVCVImageHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaImageStatsSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                               NVCVImageHandle mask, NVCVTensorHandle out)
{
    return ImageStatsSubmitImpl(handle, stream, in, mask, out);
}

NVCVStatus pvaImageStatsSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle mask,
                               NVCVTensorHandle out)
{
    return ImageStatsSubmitImpl(handle, stream, in, mask, out);
}
