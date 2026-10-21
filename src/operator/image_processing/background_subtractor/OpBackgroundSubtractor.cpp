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

#include "OpBackgroundSubtractor.hpp"

#include <IOperator.hpp>
#include <ImageUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Image.hpp>

#include <OpBackgroundSubtractor.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <device/background_subtractor_params.h>

#include <cstdint>
#include <iostream>

using namespace cupva;
using namespace cupva_utils;
typedef uint16_t half; // using uint16_t in place of fp16 since it consumes the same number of bytes.
using std::cout;
using std::endl;

CUPVA_DECLARE_EXECUTABLE(back_sub_U8_FP16_dev)
CUPVA_DECLARE_EXECUTABLE(back_sub_U8_FP32_dev)

namespace pva_solutions::priv {

BackgroundSubtractorBase::BackgroundSubtractorBase(PVAGMMDataType gmmDataType)
    : m_gmmDataType{gmmDataType}
{
}

PVAGMMDataType BackgroundSubtractorBase::getGMMDataType()
{
    return m_gmmDataType;
}

template<typename T, typename GMMModelType>
std::mutex BackgroundSubtractor<T, GMMModelType>::s_execMapMutex;

template<typename T, typename GMMModelType>
std::map<typename BackgroundSubtractor<T, GMMModelType>::ExecKey, std::weak_ptr<cupva::Executable>>
    BackgroundSubtractor<T, GMMModelType>::s_backgroundSubtractorExecutables;

template<typename T, typename GMMModelType>
std::shared_ptr<cupva::Executable> BackgroundSubtractor<T, GMMModelType>::GetBackgroundSubtractorExec()
{
    // Use static data instead of static executable
    static IOperator::ExecutableData sFP16ExecData = {CUPVA_EXECUTABLE_DATA(back_sub_U8_FP16_dev),
                                                      CUPVA_EXECUTABLE_SIZE(back_sub_U8_FP16_dev)};

    static IOperator::ExecutableData sFP32ExecData = {CUPVA_EXECUTABLE_DATA(back_sub_U8_FP32_dev),
                                                      CUPVA_EXECUTABLE_SIZE(back_sub_U8_FP32_dev)};

    // Choose the right executable data based on GMMModelType
    const IOperator::ExecutableData &execData = (sizeof(GMMModelType) == 4) ? sFP32ExecData : sFP16ExecData;

    // Create key that includes both context and data type
    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent(), sizeof(GMMModelType));

    // Use the proper shared infrastructure
    return IOperator::GetExecutable<ExecKey>(execData, execKey, s_backgroundSubtractorExecutables, s_execMapMutex);
}

template<typename T, typename GMMModelType>
BackgroundSubtractor<T, GMMModelType>::BackgroundSubtractor(NVCVImageRequirements *imageRequirements,
                                                            NVCVImageRequirements *maskRequirements,
                                                            PVAGMMDataType gmmDataType)
    : BackgroundSubtractorBase{gmmDataType}
    , m_imageRequirements{*imageRequirements}
    , m_maskRequirements{*maskRequirements}
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstBGOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_progWithBG{}
    , m_progWOBG{}
    , m_frameNum{0}
{
    if (imageRequirements == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Pointer to input image requirements must not be NULL");
    }
    if (maskRequirements == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Pointer to output image requirements must not be NULL");
    }

    if (imageRequirements->format != NVCV_IMAGE_FORMAT_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input image must have U8 format");
    }
    if (maskRequirements->format != NVCV_IMAGE_FORMAT_U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output image must have U8 format");
    }

    int imgWidth     = imageRequirements->width;
    int imgHeight    = imageRequirements->height;
    int imgLinePitch = imageRequirements->planeRowStride[0] / sizeof(T);

    int maskWidth     = maskRequirements->width;
    int maskHeight    = maskRequirements->height;
    int maskLinePitch = maskRequirements->planeRowStride[0] / sizeof(T);

    if (imgWidth != maskWidth || imgHeight != maskHeight)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output images must have the same width and height");
    }

    m_execShared = GetBackgroundSubtractorExec();

    const int niterX = (imgWidth + TILE_W - 1) / TILE_W;
    const int niterY = (imgHeight + TILE_H - 1) / TILE_H;
    const int niter  = niterX * niterY;
    int imgSize      = imgWidth * imgHeight;

    // Allocate memory for GMM parameters
    m_gmmWeightsNMod_d = (GMMModelType *)mem::Alloc(imgSize * (NUM_MODELS + 1) * sizeof(GMMModelType));
    m_gmmVariances_d   = (GMMModelType *)mem::Alloc(imgSize * NUM_MODELS * sizeof(GMMModelType));
    m_gmmMeans_d       = (GMMModelType *)mem::Alloc(imgSize * NUM_MODELS * sizeof(GMMModelType));

    m_gmmWeightsNMod_h = (GMMModelType *)mem::GetHostPointer(m_gmmWeightsNMod_d);
    m_gmmVariances_h   = (GMMModelType *)mem::GetHostPointer(m_gmmVariances_d);
    m_gmmMeans_h       = (GMMModelType *)mem::GetHostPointer(m_gmmMeans_d);

    // Initialize GMM parameters
    for (int i = 0; i < imgSize * (NUM_MODELS + 1); i++)
    {
        m_gmmWeightsNMod_h[i] = 0.0f;
    }
    for (int i = 0; i < imgSize * NUM_MODELS; i++)
    {
        m_gmmVariances_h[i] = 0.0f;
        m_gmmMeans_h[i]     = 0.0f;
    }

    // Create 2 programs - one with and one without dataflow for background image.
    // Depending on whether the background image is requested by the submit command
    //  the appropriate program will be selected.
    for (int i = 0; i < 2; i++)
    {
        auto prog = CmdProgram::Create(*m_execShared.get());

        prog["niter"] = niter;

        // Set up data flows for input image
        RasterDataFlow &inStream = prog.addDataFlowHead<RasterDataFlow>();
        inStream.handler(prog["imgInTrig"])
            .tileBuffer(prog["img_in"].ptr<T>())
            .src(m_srcOp, imgWidth, imgHeight, imgLinePitch)
            .tile(TILE_W, TILE_H)
            .transpose();

        // Set up data flows for GMM parameters
        RasterDataFlow &meanStream = prog.addDataFlowHead<RasterDataFlow>();
        meanStream.handler(prog["gmmInTrig"])
            .tileBuffer(prog["gmm_mean_in"].ptr<GMMModelType>())
            .src(m_gmmMeans_d, imgWidth * NUM_MODELS, imgHeight, imgWidth * NUM_MODELS)
            .tile(TILE_W * NUM_MODELS, TILE_H)
            .transpose();

        RasterDataFlow &varStream = prog.addDataFlowHead<RasterDataFlow>();
        varStream.handler(prog["gmmInTrig"])
            .tileBuffer(prog["gmm_var_in"].ptr<GMMModelType>())
            .src(m_gmmVariances_d, imgWidth * NUM_MODELS, imgHeight, imgWidth * NUM_MODELS)
            .tile(TILE_W * NUM_MODELS, TILE_H)
            .transpose();

        RasterDataFlow &weightStream = prog.addDataFlowHead<RasterDataFlow>();
        weightStream.handler(prog["gmmWtNModInTrig"])
            .tileBuffer(prog["gmm_weight_nmod_in"].ptr<GMMModelType>())
            .src(m_gmmWeightsNMod_d, imgWidth * (NUM_MODELS + 1), imgHeight, imgWidth * (NUM_MODELS + 1))
            .tile(TILE_W * (NUM_MODELS + 1), TILE_H)
            .transpose();

        // Set up data flow for output mask
        RasterDataFlow &outStream = prog.addDataFlowHead<RasterDataFlow>();
        outStream.handler(prog["outMaskTrig"])
            .tileBuffer(prog["FGMask"].ptr<T>())
            .dst(m_dstOp, imgWidth, imgHeight, maskLinePitch)
            .tile(TILE_W, TILE_H)
            .transpose();

        // When 0, we are creating a program without dataflow for background image.
        // When 1, we are creating a program with dataflow for background image.
        if (i == 1)
        {
            RasterDataFlow &outBGStream = prog.addDataFlowHead<RasterDataFlow>();
            outBGStream.handler(prog["outMaskTrig"])
                .tileBuffer(prog["bg_img_out"].ptr<T>())
                .dst(m_dstBGOp, imgWidth, imgHeight, maskLinePitch)
                .tile(TILE_W, TILE_H)
                .transpose();
        }

        // Set up data flows for updated GMM parameters
        RasterDataFlow &meanOutStream = prog.addDataFlowHead<RasterDataFlow>();
        meanOutStream.handler(prog["gmmOutTrig"])
            .tileBuffer(prog["gmm_mean_out"].ptr<GMMModelType>())
            .dst(m_gmmMeans_d, imgWidth * NUM_MODELS, imgHeight, imgWidth * NUM_MODELS)
            .tile(TILE_W * NUM_MODELS, TILE_H)
            .transpose();

        RasterDataFlow &varOutStream = prog.addDataFlowHead<RasterDataFlow>();
        varOutStream.handler(prog["gmmOutTrig"])
            .tileBuffer(prog["gmm_var_out"].ptr<GMMModelType>())
            .dst(m_gmmVariances_d, imgWidth * NUM_MODELS, imgHeight, imgWidth * NUM_MODELS)
            .tile(TILE_W * NUM_MODELS, TILE_H)
            .transpose();

        RasterDataFlow &weightOutStream = prog.addDataFlowHead<RasterDataFlow>();
        weightOutStream.handler(prog["gmmWtNModOutTrig"])
            .tileBuffer(prog["gmm_weight_nmod_out"].ptr<GMMModelType>())
            .dst(m_gmmWeightsNMod_d, imgWidth * (NUM_MODELS + 1), imgHeight, imgWidth * (NUM_MODELS + 1))
            .tile(TILE_W * (NUM_MODELS + 1), TILE_H)
            .transpose();

        prog.compileDataFlows();

        if (i == 0)
        {
            cupva::ConvertTo(std::move(prog), m_progWOBG);
        }
        else
        {
            cupva::ConvertTo(std::move(prog), m_progWithBG);
        }
    }
}

template<typename T, typename GMMModelType>
void BackgroundSubtractor<T, GMMModelType>::setParameters(float learningRate, float varThreshold,
                                                          bool enableShadowDetection, uint8_t shadowPixelValue,
                                                          bool getBackgroundImage)
{
    cupvaCmd_t *cmdProg = getBackgroundImage ? &m_progWithBG : &m_progWOBG;
    if (learningRate < 0.0f || learningRate > 1.0f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Learning rate must be in range [0.0, 1.0]");
    }
    if (varThreshold < 0.0f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Threshold must be non-negative");
    }

    {
        cupvaParameter_t param;
        cupvaError_t err = CupvaCmdProgramGetParameter(cmdProg, &param, "varThreshold");
        CupvaParameterSetValueScalar(&param, &varThreshold, sizeof(float));
    }

    {
        cupvaParameter_t param;
        uint8_t enableShadowDetectionInt = enableShadowDetection ? 1 : 0;
        CupvaCmdProgramGetParameter(cmdProg, &param, "detectShadow");
        CupvaParameterSetValueScalar(&param, &enableShadowDetectionInt, sizeof(uint8_t));
    }

    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "shadowPixelValue");
        CupvaParameterSetValueScalar(&param, &shadowPixelValue, sizeof(uint8_t));
    }

    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "learningRate");
        CupvaParameterSetValueScalar(&param, &learningRate, sizeof(float));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "getBGImg");
        int32_t getBackgroundImageInt = getBackgroundImage ? 1 : 0;
        CupvaParameterSetValueScalar(&param, &getBackgroundImageInt, sizeof(int32_t));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "frameNum");
        CupvaParameterSetValueScalar(&param, &m_frameNum, sizeof(int));
    }
}

template<typename T, typename GMMModelType>
template<typename S>
void BackgroundSubtractor<T, GMMModelType>::operator()(S stream, const nvcv::Image &in, const nvcv::Image &out,
                                                       const nvcv::Image *outBGPtr, float learningRate,
                                                       float varThreshold, bool enableShadowDetection,
                                                       uint8_t shadowPixelValue)
{
    bool getBackgroundImage = (outBGPtr != nullptr);
    auto inImageData        = in.exportData<nvcv::ImageDataStrided>();
    auto outMaskData        = out.exportData<nvcv::ImageDataStrided>();
    nvcv::Optional<nvcv::ImageDataStrided> outBGImageData;

    if (getBackgroundImage)
    {
        outBGImageData = outBGPtr->exportData<nvcv::ImageDataStrided>();
    }

    if (!pva_solutions::utils::AreImageRequirementsEqual(m_imageRequirements, inImageData->cdata()) ||
        !pva_solutions::utils::AreImageRequirementsEqual(m_maskRequirements, outMaskData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output images must have the same requirements that was used to create the operator");
    }

    if (getBackgroundImage)
    {
        if (!pva_solutions::utils::AreImageRequirementsEqual(m_imageRequirements, outBGImageData->cdata()))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Output background image must have the same image requirements that was used to create the operator");
        }
        if (static_cast<void *>(outBGImageData->plane(0).basePtr) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output background image must not be NULL");
        }
    }

    if ((static_cast<void *>(inImageData->plane(0).basePtr) == nullptr) ||
        (static_cast<void *>(outMaskData->plane(0).basePtr) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image base pointers must not be NULL");
    }

    setParameters(learningRate, varThreshold, enableShadowDetection, shadowPixelValue, getBackgroundImage);
    m_frameNum++;

    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(inImageData->plane(0).basePtr)));
    m_dstOp.setBase(static_cast<T *>(static_cast<void *>(outMaskData->plane(0).basePtr)));

    cupvaError_t err;
    cupvaCmd_t const *submitCmds[1]; //  = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1] = {nullptr};
    if (getBackgroundImage)
    {
        m_dstBGOp.setBase(static_cast<T *>(static_cast<void *>(outBGImageData->plane(0).basePtr)));
        err           = CupvaCmdProgramUpdateDataFlows(&m_progWithBG);
        submitCmds[0] = &m_progWithBG;
    }
    else
    {
        err           = CupvaCmdProgramUpdateDataFlows(&m_progWOBG);
        submitCmds[0] = &m_progWOBG;
    }
    assert(err == CUPVA_ERROR_NONE);

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
        static_assert(!sizeof(S), "Unsupported stream type for BackgroundSubtractor operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T, typename GMMModelType>
BackgroundSubtractor<T, GMMModelType>::~BackgroundSubtractor()
{
    CupvaCmdDestroy(&m_progWOBG);
    CupvaCmdDestroy(&m_progWithBG);
    mem::Free(m_gmmWeightsNMod_d);
    mem::Free(m_gmmVariances_d);
    mem::Free(m_gmmMeans_d);
}

// Explicit instantiation of the template to support the types we care about
template class BackgroundSubtractor<uint8_t, half>;
template class BackgroundSubtractor<uint8_t, float>;

} // namespace pva_solutions::priv

NVCVStatus pvaBackgroundSubtractorCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *imageRequirements,
                                         NVCVImageRequirements *maskRequirements, PVAGMMDataType gmmDataType)
{
    return nvcv::ProtectCall([&] {
        NVCVImageFormat imageFormat, maskFormat;
        imageFormat = imageRequirements->format;
        maskFormat  = maskRequirements->format;

        if (imageFormat != NVCV_IMAGE_FORMAT_U8 && imageFormat != NVCV_IMAGE_FORMAT_Y8 &&
            imageFormat != NVCV_IMAGE_FORMAT_Y8_ER)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input image and output background image must have U8/Y8/Y8_ER image format");
        }

        if (maskFormat != NVCV_IMAGE_FORMAT_U8)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output mask image must have U8 image format");
        }

        if (gmmDataType == PVA_GMM_DATA_TYPE_FP16)
        {
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::BackgroundSubtractor<uint8_t, half>(
                imageRequirements, maskRequirements, gmmDataType));
        }
        else if (gmmDataType == PVA_GMM_DATA_TYPE_FP32)
        {
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::BackgroundSubtractor<uint8_t, float>(
                    imageRequirements, maskRequirements, gmmDataType));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported gmm data type");
        }
    });
}

template<typename StreamType>
NVCVStatus BackgroundSubtractorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in,
                                          NVCVImageHandle outMask, NVCVImageHandle outBGImage, float learningRate,
                                          float varThreshold, bool enableShadowDetection, uint8_t shadowPixelValue)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (in == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to input image must not be NULL");
        }
        if (outMask == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to output mask must not be NULL");
        }
        bool getBackgroundImage = (outBGImage != nullptr);

        nvcv::ImageWrapHandle input(in), output(outMask);

        pva_solutions::priv::BackgroundSubtractorBase &base =
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BackgroundSubtractorBase>(handle);
        PVAGMMDataType gmmDataType = base.getGMMDataType();

        nvcv::Image outputBGImage;
        nvcv::Image *outputBGImagePtr = nullptr;
        if (getBackgroundImage)
        {
            outputBGImage    = nvcv::Image::FromHandle(outBGImage, true);
            outputBGImagePtr = &outputBGImage;
        }

        if (gmmDataType == PVA_GMM_DATA_TYPE_FP32)
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BackgroundSubtractor<uint8_t, float>>(handle)(
                stream, input, output, outputBGImagePtr, learningRate, varThreshold, enableShadowDetection,
                shadowPixelValue);
        }
        else if (gmmDataType == PVA_GMM_DATA_TYPE_FP16)
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::BackgroundSubtractor<uint8_t, half>>(handle)(
                stream, input, output, outputBGImagePtr, learningRate, varThreshold, enableShadowDetection,
                shadowPixelValue);
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus BackgroundSubtractorSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVImageHandle,
                                                                  NVCVImageHandle, NVCVImageHandle, float, float, bool,
                                                                  uint8_t);
template NVCVStatus BackgroundSubtractorSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVImageHandle,
                                                                 NVCVImageHandle, NVCVImageHandle, float, float, bool,
                                                                 uint8_t);

// C++ overload implementations
NVCVStatus pvaBackgroundSubtractorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                         NVCVImageHandle outMask, NVCVImageHandle outBGImage, float learningRate,
                                         float varThreshold, bool enableShadowDetection, uint8_t shadowPixelValue)
{
    return BackgroundSubtractorSubmitImpl(handle, stream, in, outMask, outBGImage, learningRate, varThreshold,
                                          enableShadowDetection, shadowPixelValue);
}

NVCVStatus pvaBackgroundSubtractorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                         NVCVImageHandle outMask, NVCVImageHandle outBGImage, float learningRate,
                                         float varThreshold, bool enableShadowDetection, uint8_t shadowPixelValue)
{
    return BackgroundSubtractorSubmitImpl(handle, stream, in, outMask, outBGImage, learningRate, varThreshold,
                                          enableShadowDetection, shadowPixelValue);
}