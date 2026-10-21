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

#include "OpRemap.hpp"

#include "device/remap_param.h"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpRemap.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <iostream>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(remap_u8_dev);
CUPVA_DECLARE_EXECUTABLE(remap_nv12_dev);

namespace pva_solutions::priv {

std::mutex Remap::s_execMapMutex;

std::map<typename Remap::ExecKey, std::weak_ptr<cupva::Executable>> Remap::s_remapExecutables;

static std::map<typename Remap::ExecDataKey, IOperator::ExecutableData> createExecDataMap()
{
    return {
        {{NVCV_IMAGE_FORMAT_U8}, {CUPVA_EXECUTABLE_DATA(remap_u8_dev), CUPVA_EXECUTABLE_SIZE(remap_u8_dev)}},
        {{NVCV_IMAGE_FORMAT_NV12_ER}, {CUPVA_EXECUTABLE_DATA(remap_nv12_dev), CUPVA_EXECUTABLE_SIZE(remap_nv12_dev)}},
        {{NVCV_IMAGE_FORMAT_NV12}, {CUPVA_EXECUTABLE_DATA(remap_nv12_dev), CUPVA_EXECUTABLE_SIZE(remap_nv12_dev)}}};
}

std::shared_ptr<cupva::Executable> Remap::GetRemapExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_remapExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

Remap::Remap(NVCVImageRequirements const *const inRequirements, NVCVImageRequirements const *const outRequirements,
             NVCVTensorRequirements const *const warpMapRequirements, PVAInterpolationType const interpType,
             NVCVBorderType const borderMode)
    : m_inRequirements{*inRequirements}
    , m_outRequirements{*outRequirements}
    , m_warpMapRequirements{*warpMapRequirements}
    , m_srcOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr)}
    , m_dstOp{cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr),
              cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr)}
    , m_warpMapOp{cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr)}
    , m_exec{}
    , m_prog{}
{
    if (!(inRequirements->format == NVCV_IMAGE_FORMAT_U8 || inRequirements->format == NVCV_IMAGE_FORMAT_NV12_ER ||
          inRequirements->format == NVCV_IMAGE_FORMAT_NV12))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image must have either NVCV_IMAGE_FORMAT_U8 or NVCV_IMAGE_FORMAT_NV12_ER, "
                              "NVCV_IMAGE_FORMAT_NV12 format");
    }
    if (outRequirements->format != inRequirements->format)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output image must have the same format as input image");
    }
    if (outRequirements->width != warpMapRequirements->shape[1] ||
        outRequirements->height != warpMapRequirements->shape[0])
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output image and warp map must have the same width and height");
    }
    if (warpMapRequirements->dtype != NVCV_DATA_TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Warp map must have NVCV_DATA_TYPE_F32 data type");
    }
    if (warpMapRequirements->shape[2] != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Warp map must have 2 channels (for x and y coordinates)");
    }
    if (!(interpType == PVA_INTERPOLATION_NN || interpType == PVA_INTERPOLATION_LINEAR))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Only PVA_INTERPOLATION_NN and PVA_INTERPOLATION_LINEAR interpolation modes are supported");
    }
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only NVCV_BORDER_CONSTANT and NVCV_BORDER_REPLICATE border modes are supported");
    }

    const int32_t inImageWidth  = m_inRequirements.width;
    const int32_t inImageHeight = m_inRequirements.height;
    const int32_t inLinePitch   = m_inRequirements.planeRowStride[0] / sizeof(uint8_t);

    const int32_t outImageWidth  = m_outRequirements.width;
    const int32_t outImageHeight = m_outRequirements.height;
    const int32_t outLinePitch   = m_outRequirements.planeRowStride[0] / sizeof(uint8_t);

    m_exec    = GetRemapExec(std::make_tuple(m_inRequirements.format));
    auto prog = CmdProgram::Create(*m_exec.get());

    // dataflow constructions
    auto srcHdl     = prog["srcHdl"];
    auto dstHdl     = prog["dstHdl"];
    auto warpMapHdl = prog["warpMapHdl"];

    prog["inImageWidth"]    = inImageWidth;
    prog["inImageHeight"]   = inImageHeight;
    prog["outImageWidth"]   = outImageWidth;
    prog["outImageHeight"]  = outImageHeight;
    prog["inLinePitch"]     = inLinePitch;
    InterpTypeDev interpDev = (interpType == PVA_INTERPOLATION_NN) ? NN : BILINEAR;
    prog["interpType_v"]    = static_cast<uint32_t>(interpDev);
    BorderModeDev borderDev = (borderMode == NVCV_BORDER_REPLICATE) ? BPE : CONSTANT;
    prog["borderMode_v"]    = static_cast<uint32_t>(borderDev);

    float *warpMapBuf = prog["warpMap_v"].ptr<float>();
    uint8_t *srcBuf   = prog["src_v"].ptr<uint8_t>();
    uint8_t *dstBuf   = prog["dst_v"].ptr<uint8_t>();

    const int32_t warpMapWidth      = m_warpMapRequirements.shape[1] * 2;
    const int32_t warpMapHeight     = m_warpMapRequirements.shape[0];
    const int32_t warpMapLinePitch  = m_warpMapRequirements.strides[0] / sizeof(float);
    RasterDataFlow &warpMapDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    warpMapDataFlow.handler(warpMapHdl)
        .src(m_warpMapOp, warpMapWidth, warpMapHeight, warpMapLinePitch)
        .tileBuffer(warpMapBuf)
        .tile(TILE_W * 2, TILE_H)
        .halo(0, PAD_BPE); // using BPE padding for tail case (when image size is not multiple of tile size)

#if USE_GSDF == 1
    GatherScatterDataFlow &imgInDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>();
    imgInDataFlow.handler(srcHdl)
        .numTilesPerTrigger(1)
        .tile(TILE_W, TILE_H) // will configure the exact tile size at VPU side
        .src(m_srcOp[0], inImageWidth, inImageHeight, inLinePitch);
#else
    SequenceDataFlow &imgInDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    imgInDataFlow.handler(srcHdl).addTransfer().src(m_srcOp[0], 0).dst(srcBuf, 0).tile(MAX_TABLE_SIZE / 2, 1);
#endif

    RasterDataFlow &imgOutDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    imgOutDataFlow.handler(dstHdl)
        .tileBuffer(dstBuf)
        .tile(TILE_W, TILE_H)
        .dst(m_dstOp[0], outImageWidth, outImageHeight, outLinePitch);

    // additional dataflow for UV plane
    if (m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12_ER || m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12)
    {
        auto srcUVHdl = prog["srcUVHdl"];
        auto dstUVHdl = prog["dstUVHdl"];

        const int32_t inLinePitchUV  = m_inRequirements.planeRowStride[1] / sizeof(uint8_t);
        const int32_t outLinePitchUV = m_outRequirements.planeRowStride[1] / sizeof(uint8_t);
        prog["inLinePitchUV"]        = inLinePitchUV;

        uint8_t *srcUVBuf = prog["srcUV_v"].ptr<uint8_t>();
        uint8_t *dstUVBuf = prog["dstUV_v"].ptr<uint8_t>();

#if USE_GSDF == 1
        GatherScatterDataFlow &imgUVInDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>();
        imgUVInDataFlow.handler(srcUVHdl)
            .numTilesPerTrigger(1)
            .tile(TILE_W, TILE_H / 2) // will configure the exact tile size at VPU side
            .src(m_srcOp[1], inImageWidth, inImageHeight / 2, inLinePitchUV);
#else
        SequenceDataFlow &imgUVInDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
        imgUVInDataFlow.handler(srcUVHdl)
            .addTransfer()
            .src(m_srcOp[1], 0)
            .dst(srcUVBuf, 0)
            .tile(MAX_TABLE_UV_SIZE / 2, 1);
#endif

        RasterDataFlow &imgOutUVDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        imgOutUVDataFlow.handler(dstUVHdl)
            .tileBuffer(dstUVBuf)
            .tile(TILE_W, TILE_H / 2)
            .dst(m_dstOp[1], outImageWidth, outImageHeight / 2, outLinePitchUV);
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void Remap::operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, const nvcv::Tensor &warpMap)
{
    auto inImageData  = in.exportData<nvcv::ImageDataStrided>();
    auto outImageData = out.exportData<nvcv::ImageDataStrided>();
    auto warpMapData  = warpMap.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_inRequirements, inImageData->cdata()) ||
        !pva_solutions::utils::AreImageRequirementsEqual(m_outRequirements, outImageData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_warpMapRequirements, warpMapData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input image, output image, and warp map must have the same requirements that were used "
                              "to create the operator");
    }
    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(warpMapData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image and warp map base pointers must not be NULL");
    }
    if (m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12_ER || m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12)
    {
        if ((static_cast<void *>(inImageData->plane(1).basePtr) == nullptr) ||
            (static_cast<void *>(outImageData->plane(1).basePtr) == nullptr))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Image UV plane base pointers must not be NULL");
        }
    }

#if USE_GSDF == 0
    // Set external memory pointers for SQDF.
    cupvaParameter_t srcExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &srcExtPtr, "srcExtPtr");
    CupvaParameterSetValuePointer(&srcExtPtr, static_cast<void *>(inImageData->plane(0).basePtr));
#endif

    m_srcOp[0].setBase(static_cast<uint8_t *>(static_cast<void *>(inImageData->plane(0).basePtr)));
    m_dstOp[0].setBase(static_cast<uint8_t *>(static_cast<void *>(outImageData->plane(0).basePtr)));
    m_warpMapOp.setBase(static_cast<float *>(static_cast<void *>(warpMapData->basePtr())));

    if (m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12_ER || m_inRequirements.format == NVCV_IMAGE_FORMAT_NV12)
    {

#if USE_GSDF == 0
        cupvaParameter_t srcUVExtPtr;
        CupvaCmdProgramGetParameter(&m_prog, &srcUVExtPtr, "srcUVExtPtr");
        CupvaParameterSetValuePointer(&srcUVExtPtr, static_cast<void *>(inImageData->plane(1).basePtr));
#endif

        m_srcOp[1].setBase(static_cast<uint8_t *>(static_cast<void *>(inImageData->plane(1).basePtr)));
        m_dstOp[1].setBase(static_cast<uint8_t *>(static_cast<void *>(outImageData->plane(1).basePtr)));
    }

    cupvaError_t err                = CupvaCmdProgramUpdateDataFlows(&m_prog);
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
        static_assert(!sizeof(S), "Unsupported stream type for Remap operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

Remap::~Remap()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaRemapCreate(NVCVOperatorHandle *handle, NVCVImageRequirements const *const inRequirements,
                          NVCVImageRequirements const *const outRequirements,
                          NVCVTensorRequirements const *const warpMapRequirements,
                          PVAInterpolationType const interpType, NVCVBorderType const borderMode)

{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to inRequirements must not be NULL");
        }
        if (outRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to outRequirements must not be NULL");
        }
        if (warpMapRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to warpMapRequirements must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Remap(
            inRequirements, outRequirements, warpMapRequirements, interpType, borderMode));
    });
}

template<typename StreamType>
NVCVStatus RemapSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle out,
                           NVCVTensorHandle warpMap)
{
    return nvcv::ProtectCall([&] {
        nvcv::ImageWrapHandle input(in), output(out);
        nvcv::TensorWrapHandle warp(warpMap);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Remap>(handle)(stream, input, output, warp);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus RemapSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle, NVCVImageHandle,
                                                   NVCVTensorHandle);
template NVCVStatus RemapSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle, NVCVImageHandle,
                                                  NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaRemapSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                          NVCVTensorHandle warpMap)
{
    return RemapSubmitImpl(handle, stream, in, out, warpMap);
}

NVCVStatus pvaRemapSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in, NVCVImageHandle out,
                          NVCVTensorHandle warpMap)
{
    return RemapSubmitImpl(handle, stream, in, out, warpMap);
}

NVCVStatus pvaRemapValidate(NVCVTensorHandle warpMapHandle, int32_t inWidth, int32_t inHeight, int32_t outWidth,
                            int32_t outHeight)
{
    return nvcv::ProtectCall([&] {
        NVCVTensorData warpMapData;
        nvcvTensorExportData(warpMapHandle, &warpMapData);
        float *warpMapPtr = NULL;
        CupvaMemGetHostPointer((void **)&warpMapPtr, (void *)warpMapData.buffer.strided.basePtr);

        const int32_t nTileX  = (outWidth + TILE_W - 1) / TILE_W;
        const int32_t nTileY  = (outHeight + TILE_H - 1) / TILE_H;
        int32_t warpLinePitch = warpMapData.buffer.strided.strides[0] / sizeof(float);

        for (int32_t i = 0; i < nTileY; i++)
        {
            for (int32_t j = 0; j < nTileX; j++)
            {
                int32_t xStart = j * TILE_W;
                int32_t xEnd   = std::min(xStart + TILE_W, outWidth);
                int32_t yStart = i * TILE_H;
                int32_t yEnd   = std::min(yStart + TILE_H, outHeight);

                int32_t minX, maxX, minY, maxY;
                minX = minY = INT32_MAX;
                maxX = maxY = INT32_MIN;

                for (int32_t y = yStart; y < yEnd; y++)
                {
                    for (int32_t x = xStart; x < xEnd; x++)
                    {
                        int32_t index = y * warpLinePitch + x * 2;
                        int32_t inX   = static_cast<int32_t>(warpMapPtr[index]);
                        int32_t inY   = static_cast<int32_t>(warpMapPtr[index + 1]);

                        minX = std::min(minX, inX);
                        maxX = std::max(maxX, inX);
                        minY = std::min(minY, inY);
                        maxY = std::max(maxY, inY);
                    }
                }

                // VPI treat [0, width/height) as valid range
                minX = std::min(std::max(0, minX - 1), inWidth - 1);
                minY = std::min(std::max(0, minY - 1), inHeight - 1);
                maxX = std::min(std::max(0, maxX + 1), inWidth - 1);
                maxY = std::min(std::max(0, maxY + 1), inHeight - 1);

                // VPU implementation details for calculating the tile size
                uint16_t tx = maxX - minX + 1;
                uint16_t ty = maxY - minY + 1;

                if (tx * ty > MAX_TABLE_SIZE)
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                          "Invalid warp map, PVA can only handle tile size smaller than 32KB!");
                }
            }
        }
    });
}