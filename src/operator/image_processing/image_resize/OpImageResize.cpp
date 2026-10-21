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

#include "OpImageResize.hpp"

#include "device/image_resize_param.h"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>

#include <OpImageResize.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

using namespace cupva;

// nearest neighbor
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_U8_nn_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_RGB8_nn_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_RGBA8_nn_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_NV12_nn_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_U16_nn_dev);
// bilinear interpolation
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_U8_bl_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_RGB8_bl_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_RGBA8_bl_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_NV12_bl_dev);
CUPVA_DECLARE_EXECUTABLE(pva_image_resize_U16_bl_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex ImageResize<T>::s_execMapMutex;

template<typename T>
std::map<typename ImageResize<T>::ExecKey, std::weak_ptr<cupva::Executable>> ImageResize<T>::s_imageResizeExecutables;

template<typename T>
std::map<typename ImageResize<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

template<>
std::map<typename ImageResize<uint8_t>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<uint8_t>()
{
    return {
        {{NVCV_IMAGE_FORMAT_U8, PVA_INTERPOLATION_NN},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_U8_nn_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_U8_nn_dev)}},
        {{NVCV_IMAGE_FORMAT_U8, PVA_INTERPOLATION_LINEAR},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_U8_bl_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_U8_bl_dev)}},
        {{NVCV_IMAGE_FORMAT_RGB8, PVA_INTERPOLATION_NN},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_RGB8_nn_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_RGB8_nn_dev)}},
        {{NVCV_IMAGE_FORMAT_RGB8, PVA_INTERPOLATION_LINEAR},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_RGB8_bl_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_RGB8_bl_dev)}},
        {{NVCV_IMAGE_FORMAT_RGBA8, PVA_INTERPOLATION_NN},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_RGBA8_nn_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_RGBA8_nn_dev)}},
        {{NVCV_IMAGE_FORMAT_RGBA8, PVA_INTERPOLATION_LINEAR},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_RGBA8_bl_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_RGBA8_bl_dev)}},
        {{NVCV_IMAGE_FORMAT_NV12, PVA_INTERPOLATION_NN},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_NV12_nn_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_NV12_nn_dev)}},
        {{NVCV_IMAGE_FORMAT_NV12, PVA_INTERPOLATION_LINEAR},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_NV12_bl_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_NV12_bl_dev)}},
        /* NV12_ER reuses the NV12 kernel binary — resize is range-agnostic
         * (only the YUV→RGB color matrix differs between studio and ER, and
         * resize never converts to RGB). */
        {{NVCV_IMAGE_FORMAT_NV12_ER, PVA_INTERPOLATION_NN},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_NV12_nn_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_NV12_nn_dev)}},
        {{NVCV_IMAGE_FORMAT_NV12_ER, PVA_INTERPOLATION_LINEAR},
         {CUPVA_EXECUTABLE_DATA(pva_image_resize_NV12_bl_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_NV12_bl_dev)}}};
}

template<>
std::map<typename ImageResize<uint16_t>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<uint16_t>()
{
    return {{{NVCV_IMAGE_FORMAT_U16, PVA_INTERPOLATION_NN},
             {CUPVA_EXECUTABLE_DATA(pva_image_resize_U16_nn_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_U16_nn_dev)}},
            {{NVCV_IMAGE_FORMAT_U16, PVA_INTERPOLATION_LINEAR},
             {CUPVA_EXECUTABLE_DATA(pva_image_resize_U16_bl_dev), CUPVA_EXECUTABLE_SIZE(pva_image_resize_U16_bl_dev)}}};
}

template<typename T>
std::shared_ptr<cupva::Executable> ImageResize<T>::GetImageResizeExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_imageResizeExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
};

static std::set<NVCVImageFormat> formatSet = {NVCV_IMAGE_FORMAT_U8,      NVCV_IMAGE_FORMAT_RGB8,
                                              NVCV_IMAGE_FORMAT_RGBA8,   NVCV_IMAGE_FORMAT_NV12,
                                              NVCV_IMAGE_FORMAT_NV12_ER, NVCV_IMAGE_FORMAT_U16};

template<typename T>
void ImageResize<T>::checkImageFormat(NVCVImageFormat format)
{
    auto formatIt = formatSet.find(format);
    if (formatIt == formatSet.end())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported image format");
    }
};

template<typename T>
std::tuple<int, int, int, int> ImageResize<T>::findBestTileSize(int srcWidth, int srcHeight, int dstWidth,
                                                                int dstHeight, int channelNum,
                                                                NVCVImageFormat imageFormat,
                                                                PVAInterpolationType resizeType)
{
    int maxSrcBufSize, maxDstBufSize;
    int maxPreBufSize, maxPostBufSize;
    if (resizeType == PVA_INTERPOLATION_NN)
    {
        switch (imageFormat)
        {
        case NVCV_IMAGE_FORMAT_U8:
            maxSrcBufSize = MAX_NN_U8_SRC_BUF_SIZE;
            maxDstBufSize = MAX_NN_U8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_NV12:
        case NVCV_IMAGE_FORMAT_NV12_ER:
            maxSrcBufSize  = MAX_NN_NV12_SRC_BUF_SIZE;
            maxDstBufSize  = MAX_NN_NV12_DST_BUF_SIZE;
            maxPreBufSize  = MAX_NN_NV12_SRC_BUF_SIZE / 4;
            maxPostBufSize = MAX_NN_NV12_DST_BUF_SIZE / 4;
            break;
        case NVCV_IMAGE_FORMAT_RGB8:
            maxSrcBufSize  = MAX_NN_RGB8_SRC_BUF_SIZE * 3;
            maxDstBufSize  = MAX_NN_RGB8_DST_BUF_SIZE * 3;
            maxPreBufSize  = MAX_NN_RGB8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_NN_RGB8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_RGBA8:
            maxSrcBufSize  = MAX_NN_RGBA8_SRC_BUF_SIZE * 4;
            maxDstBufSize  = MAX_NN_RGBA8_DST_BUF_SIZE * 4;
            maxPreBufSize  = MAX_NN_RGBA8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_NN_RGBA8_DST_BUF_SIZE;
            break;
        case NVCV_DATA_TYPE_U16:
            maxSrcBufSize = MAX_NN_U16_SRC_BUF_SIZE;
            maxDstBufSize = MAX_NN_U16_DST_BUF_SIZE;
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    }
    else
    {
        switch (imageFormat)
        {
        case NVCV_IMAGE_FORMAT_U8:
        case NVCV_IMAGE_FORMAT_NV12:
        case NVCV_IMAGE_FORMAT_NV12_ER:
            maxSrcBufSize  = MAX_BL_U8_SRC_BUF_SIZE;
            maxDstBufSize  = MAX_BL_U8_DST_BUF_SIZE;
            maxPreBufSize  = MAX_BL_U8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_BL_U8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_RGB8:
            maxSrcBufSize  = MAX_BL_RGB8_SRC_BUF_SIZE * 3;
            maxDstBufSize  = MAX_BL_RGB8_DST_BUF_SIZE * 3;
            maxPreBufSize  = MAX_BL_RGB8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_BL_RGB8_DST_BUF_SIZE;
            break;
        case NVCV_IMAGE_FORMAT_RGBA8:
            maxSrcBufSize  = MAX_BL_RGBA8_SRC_BUF_SIZE * 4;
            maxDstBufSize  = MAX_BL_RGBA8_DST_BUF_SIZE * 4;
            maxPreBufSize  = MAX_BL_RGBA8_SRC_BUF_SIZE;
            maxPostBufSize = MAX_BL_RGBA8_DST_BUF_SIZE;
            break;
        case NVCV_DATA_TYPE_U16:
            maxSrcBufSize = MAX_BL_U16_SRC_BUF_SIZE;
            maxDstBufSize = MAX_BL_U16_DST_BUF_SIZE;
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    }

    // for dlut auto-idx, patch size must be U8, thus less than 256
    // tile width is better 64 aligned
    std::vector<int> dstTileWList = {192, 128, 64, 32};

    // tile height is better 32 aligned
    std::vector<int> dstTileHList = {224, 192, 160, 128, 96, 64, 32};

    float scaleX        = srcWidth * 1.0f / dstWidth;
    float scaleY        = srcHeight * 1.0f / dstHeight;
    bool transpose_flag = channelNum > 1;

    int minTileCnt = -1;
    int optW0      = -1;
    int optH0      = -1;
    int optW1      = -1;
    int optH1      = -1;

    for (auto w1 : dstTileWList)
    {
        for (auto h1 : dstTileHList)
        {
            if (w1 >= dstWidth || h1 >= dstHeight)
            {
                continue;
            }

            int w0 = std::ceil(scaleX * w1) + 2;
            int h0 = std::ceil(scaleY * h1) + 2;

            // src with transpose load, align to 64k + 2
            int lp0 = transpose_flag ? (w0 * channelNum - 2 + 63) / 64 * 64 + 2 : w0;
            // check src buf size: align buffer size to 64B for GSDF
            int srcTileBufSize = (lp0 * h0 + 63) / 64 * 64;
            if (srcTileBufSize > maxSrcBufSize)
            {
                continue;
            }

            // dst with transpose store, align to 64k + 2
            int lp1 = transpose_flag ? (w1 * channelNum - 2 + 63) / 64 * 64 + 2 : w1;
            // check dst buf size
            if (lp1 * h1 > maxDstBufSize)
            {
                continue;
            }

            // if has pre/post processing
            if constexpr (std::is_same<T, uint8_t>::value)
            {
                if (resizeType == PVA_INTERPOLATION_LINEAR || transpose_flag)
                {
                    // check pre buf size
                    int inputW = transpose_flag ? h0 : w0;
                    int inputH = transpose_flag ? w0 : h0;
                    // align table line pitch to 32k+2 for bilinear interpolation
                    int tableLp = (resizeType == PVA_INTERPOLATION_LINEAR) ? (inputW - 2 + 31) / 32 * 32 + 2 : inputW;

                    // pre-processing kernel writes to buffer by multiples of 32
                    int preBufSize = (tableLp * inputH + 31) / 32 * 32;
                    if (preBufSize > maxPreBufSize)
                    {
                        continue;
                    }

                    // check post buf size
                    int outputW  = transpose_flag ? h1 : w1;
                    int outputH  = transpose_flag ? w1 : h1;
                    int outputLp = outputW;

                    if (outputLp * outputH > maxPostBufSize)
                    {
                        continue;
                    }
                }
            }

            // check RDF padding size: < 256
            int ntile_w = (dstWidth + w1 - 1) / w1;
            int pad_w   = (ntile_w * w1 - dstWidth) * channelNum;
            if (pad_w >= 256)
            {
                continue;
            }

            int ntile_h = (dstHeight + h1 - 1) / h1;
            int ntile   = ntile_w * ntile_h;

            // update the optimal solution: minimal tile counts
            if (minTileCnt < 0 || ntile < minTileCnt)
            {
                // for U8 data types, prefer 64 aligned tile width
                if constexpr (std::is_same<T, uint8_t>::value)
                {
                    if (w1 == 32 && optW1 > 32)
                    {
                        continue;
                    }
                }

                minTileCnt = ntile;
                optW0      = w0;
                optH0      = h0;
                optW1      = w1;
                optH1      = h1;
            }
        }
    }
    if (optW0 > 0)
    {
        return std::make_tuple(optW0, optH0, optW1, optH1);
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Could not find valid tile size");
    }
}

template<typename T>
ImageResize<T>::ImageResize(NVCVImageRequirements *inImageRequirements, NVCVImageRequirements *outImageRequirements,
                            PVAInterpolationType resizeType)
    : m_inImageRequirements(*inImageRequirements)
    , m_outImageRequirements(*outImageRequirements)
    , m_srcImageHeight(inImageRequirements->height)
    , m_srcImageWidth(inImageRequirements->width)
    , m_dstImageHeight(outImageRequirements->height)
    , m_dstImageWidth(outImageRequirements->width)
    , m_srcOp{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_dstOp{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr)}
    , m_resizeType(resizeType)
    , m_prog{}
{
    if (inImageRequirements->format != outImageRequirements->format)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image and output image must have the same format");
    }
    m_imageFormat = inImageRequirements->format;
    checkImageFormat(m_imageFormat);

    double scaleX = (double)m_srcImageWidth / m_dstImageWidth;
    double scaleY = (double)m_srcImageHeight / m_dstImageHeight;

    // for downsampling, maximum ratio supported is 3
    if (scaleX > 3 || scaleY > 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Downsampling with ratio larger than 3 is not supported");
    }
    if (m_srcImageWidth < 64 || m_srcImageHeight < 64 || m_dstImageWidth < 64 || m_dstImageHeight < 64)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output image sizes must be larger than 64x64.");
    }

    m_exec    = GetImageResizeExec(std::make_tuple(m_imageFormat, m_resizeType));
    auto prog = CmdProgram::Create(*m_exec.get());

    nvcvImageFormatGetNumPlanes(m_imageFormat, &m_planeCount);

    // How to get w/h for different planes from requirements?
    for (int p = 0; p < m_planeCount; p++)
    {
        // channel count used in VPU kernels represents the number of channels per pixel
        int32_t bitsPerPixel;
        nvcvImageFormatGetPlaneBitsPerPixel(m_imageFormat, p, &bitsPerPixel);
        m_channelCount[p] = (bitsPerPixel / CHAR_BIT) / sizeof(T);

        m_srcLinePitch[p] = m_inImageRequirements.planeRowStride[p] / sizeof(T);
        m_dstLinePitch[p] = m_outImageRequirements.planeRowStride[p] / sizeof(T);
    }

    int32_t scaleX_q = std::floor(scaleX * (1lu << QBIT));
    int32_t scaleY_q = std::floor(scaleY * (1lu << QBIT));

    int64_t scaleX_q_high = std::floor(scaleX * (1llu << QBIT_HIGH));
    int64_t scaleY_q_high = std::floor(scaleY * (1llu << QBIT_HIGH));

    prog["scaleX_q"] = scaleX_q;
    prog["scaleY_q"] = scaleY_q;

    prog["scaleX_q_high"] = scaleX_q_high;
    prog["scaleY_q_high"] = scaleY_q_high;

    prog["planeCount"] = m_planeCount;

    uint16_t srcImageWidth[2];
    uint16_t srcImageHeight[2];
    uint16_t dstImageWidth[2];
    uint16_t dstImageHeight[2];
    uint16_t srcTileWidth[2];
    uint16_t srcTileHeight[2];
    uint16_t srcTileLp[2];
    uint16_t dstTileWidth[2];
    uint16_t dstTileHeight[2];
    int nTileX[2];
    int nTileY[2];

    T *dstBuf = prog["dstBuf"].ptr<T>();

    // [NOTE] for image formats with 2 planes, only NV12 / NV12_ER are supported.
    if (m_planeCount == 2 && m_imageFormat != NVCV_IMAGE_FORMAT_NV12 && m_imageFormat != NVCV_IMAGE_FORMAT_NV12_ER)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported 2-plane color format: not NV12 / NV12_ER");
    }
    for (int p = 0; p < m_planeCount; p++)
    {
        int xSampleRatio = (p > 0) ? 2 : 1;
        int ySampleRatio = (p > 0) ? 2 : 1;

        srcImageWidth[p]  = (m_srcImageWidth + xSampleRatio - 1) / xSampleRatio;
        srcImageHeight[p] = (m_srcImageHeight + ySampleRatio - 1) / ySampleRatio;
        dstImageWidth[p]  = (m_dstImageWidth + xSampleRatio - 1) / xSampleRatio;
        dstImageHeight[p] = (m_dstImageHeight + ySampleRatio - 1) / ySampleRatio;

        std::tie(srcTileWidth[p], srcTileHeight[p], dstTileWidth[p], dstTileHeight[p]) =
            findBestTileSize(srcImageWidth[p], srcImageHeight[p], dstImageWidth[p], dstImageHeight[p],
                             m_channelCount[p], m_imageFormat, m_resizeType);

        nTileX[p] = (dstImageWidth[p] + dstTileWidth[p] - 1) / dstTileWidth[p];
        nTileY[p] = (dstImageHeight[p] + dstTileHeight[p] - 1) / dstTileHeight[p];

        // RDF for output data flow
        std::string dstHdlName = "dstRDFHdl" + std::to_string(p);
        auto dstRDFHdl         = prog[dstHdlName.c_str()];

        RasterDataFlow &dstDataFlow = prog.addDataFlowHead<RasterDataFlow>(p);
        dstDataFlow.handler(dstRDFHdl)
            .tileBuffer(dstBuf)
            .tile(dstTileWidth[p] * m_channelCount[p], dstTileHeight[p])
            .dst(m_dstOp[p], dstImageWidth[p] * m_channelCount[p], dstImageHeight[p], m_dstLinePitch[p]);

        // for multi-channel u8 output, transpose store is used in the dst tile buffer
        if (m_channelCount[p] > 1)
        {
            dstDataFlow.transpose(TRANS_MODE_1);
        }

        // GSDF for input data flow
        std::string srcHdlName = "srcGSDFHdl" + std::to_string(p);
        auto srcGSDFHdl        = prog[srcHdlName.c_str()];

        GatherScatterDataFlow &srcDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>(p);
        srcDataFlow.handler(srcGSDFHdl)
            .numTilesPerTrigger(1)
            .src(m_srcOp[p], srcImageWidth[p] * m_channelCount[p], srcImageHeight[p], m_srcLinePitch[p])
            .tile(srcTileWidth[p] * m_channelCount[p], srcTileHeight[p])
            .padVal(PAD_BPE);

        // for multi-channel u8 input, tile linePitch is aligned to 64k + 2 for transpose load/store
        srcTileLp[p] =
            (m_channelCount[p] > 1) ? (srcTileWidth[p] * m_channelCount[p] + 63) / 64 * 64 + 2 : srcTileWidth[p];
    }

    prog["channelCount"].set(m_channelCount, 2);
    prog["srcImageWidth"].set(srcImageWidth, 2);
    prog["srcImageHeight"].set(srcImageHeight, 2);
    prog["dstImageWidth"].set(dstImageWidth, 2);

    prog["dstImageHeight"].set(dstImageHeight, 2);
    prog["srcTileWidth"].set(srcTileWidth, 2);
    prog["srcTileHeight"].set(srcTileHeight, 2);
    prog["srcTileLp"].set(srcTileLp, 2);
    prog["dstTileWidth"].set(dstTileWidth, 2);
    prog["dstTileHeight"].set(dstTileHeight, 2);
    prog["nTileX"].set(nTileX, 2);
    prog["nTileY"].set(nTileY, 2);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void ImageResize<T>::operator()(S stream, const nvcv::Image &in, const nvcv::Image &out)
{
    auto inImageData  = in.exportData<nvcv::ImageDataStrided>();
    auto outImageData = out.exportData<nvcv::ImageDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_inImageRequirements, inImageData->cdata()) ||
        !pva_solutions::utils::AreImageRequirementsEqual(m_outImageRequirements, outImageData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output images must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outImageData->plane(0).basePtr) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
    }

    for (int32_t i = 0; i < m_planeCount; i++)
    {
        m_srcOp[i].setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(i).basePtr)));
        m_dstOp[i].setBase(static_cast<T *>(static_cast<void *>(outImageData->plane(i).basePtr)));
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
        static_assert(!sizeof(S), "Unsupported stream type for ImageResize operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
ImageResize<T>::~ImageResize()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class ImageResize<uint8_t>;
template class ImageResize<uint16_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaImageResizeCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *inImageRequirements,
                                NVCVImageRequirements *outImageRequirements, PVAInterpolationType resizeType)
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
                                  "Pointer to input ImageRequirements must not be NULL");
        }
        if (outImageRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output ImageRequirements must not be NULL");
        }
        NVCVDataType dataType;
        nvcvImageFormatGetPlaneDataType(inImageRequirements->format, 0, &dataType);
        switch (dataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::ImageResize<uint8_t>(inImageRequirements, outImageRequirements, resizeType));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::ImageResize<uint16_t>(inImageRequirements, outImageRequirements, resizeType));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus ImageResizeSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle input(in), output(out);

        NVCVImageFormat inFormat;
        nvcvImageGetFormat(in, &inFormat);

        NVCVDataType inDataType;
        nvcvImageFormatGetPlaneDataType(inFormat, 0, &inDataType);
        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_2U8:
        case NVCV_DATA_TYPE_3U8:
        case NVCV_DATA_TYPE_4U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ImageResize<uint8_t>>(handle)(stream, input, output);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::ImageResize<uint16_t>>(handle)(stream, input,
                                                                                                  output);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus ImageResizeSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                         NVCVImageHandle);
template NVCVStatus ImageResizeSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                        NVCVImageHandle);

// C++ overload implementations
NVCVStatus pvaImageResizeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                NVCVImageHandle out)
{
    return ImageResizeSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaImageResizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle out)
{
    return ImageResizeSubmitImpl(handle, stream, in, out);
}
