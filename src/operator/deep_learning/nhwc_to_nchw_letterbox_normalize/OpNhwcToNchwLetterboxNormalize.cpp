/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpNhwcToNchwLetterboxNormalize.hpp"

#include "device/nhwc_to_nchw_letterbox_normalize_param.h"

#include <TensorUtils.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpNhwcToNchwLetterboxNormalize.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>
#include <nvcv/TensorData.h>
#include <nvcv/TensorShape.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_nhwc_to_nchw_letterbox_normalize_dev);

namespace pva_solutions::priv {

namespace {

struct SyncObjDeleter
{
    void operator()(cupvaSyncObj_t sync) const
    {
        CupvaSyncObjDestroy(sync);
    }
};

using SyncObjPtr = std::unique_ptr<std::remove_pointer_t<cupvaSyncObj_t>, SyncObjDeleter>;

struct PaddingFillJob
{
    float *output;
    int32_t outW;
    int32_t outH;
    int32_t imgW;
    int32_t imgH;
    int32_t padTop;
    int32_t padLeft;
    int32_t channels;
    float value;
};

void FillPadding(PaddingFillJob const &job)
{
    const int32_t padBottom = job.outH - job.imgH - job.padTop;
    const int32_t padRight  = job.outW - job.imgW - job.padLeft;
    const size_t planeSize  = static_cast<size_t>(job.outW) * job.outH;

    // Vertical-only letterboxing is the common YOLO path. Preserve the image
    // region and touch only the two contiguous padding bands. For side padding,
    // a full-plane fill is faster than thousands of tiny row-strip fills; the
    // PVA command that follows restores the complete image region.
    if (job.padLeft == 0 && padRight == 0)
    {
        for (int32_t ch = 0; ch < job.channels; ++ch)
        {
            float *plane = job.output + static_cast<size_t>(ch) * planeSize;
            if (job.padTop > 0)
                std::fill_n(plane, static_cast<size_t>(job.padTop) * job.outW, job.value);
            if (padBottom > 0)
                std::fill_n(plane + static_cast<size_t>(job.padTop + job.imgH) * job.outW,
                            static_cast<size_t>(padBottom) * job.outW, job.value);
        }
    }
    else
    {
        std::fill_n(job.output, static_cast<size_t>(job.channels) * planeSize, job.value);
    }
}

void CUDART_CB FillPaddingOnCudaStream(void *userData)
{
    auto *job = static_cast<PaddingFillJob *>(userData);
    FillPadding(*job);
    delete job;
}

} // namespace

std::mutex NhwcToNchwLetterboxNormalize::s_execMapMutex;

std::map<NhwcToNchwLetterboxNormalize::ExecKey, std::weak_ptr<cupva::Executable>>
    NhwcToNchwLetterboxNormalize::s_executables;

std::shared_ptr<cupva::Executable> NhwcToNchwLetterboxNormalize::GetExec()
{
    static IOperator::ExecutableData sExecData = {CUPVA_EXECUTABLE_DATA(pva_nhwc_to_nchw_letterbox_normalize_dev),
                                                  CUPVA_EXECUTABLE_SIZE(pva_nhwc_to_nchw_letterbox_normalize_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sExecData, execKey, s_executables, s_execMapMutex);
}

NhwcToNchwLetterboxNormalize::NhwcToNchwLetterboxNormalize(NVCVTensorRequirements const *inTensorReqs,
                                                           NVCVTensorRequirements const *outTensorReqs, float scale,
                                                           float padValue,
                                                           PVANhwcToNchwLetterboxNormalizePaddingPolicy paddingPolicy)
    : m_inTensorRequirements(inTensorReqs ? *inTensorReqs : NVCVTensorRequirements{})
    , m_outTensorRequirements(outTensorReqs ? *outTensorReqs : NVCVTensorRequirements{})
    , m_srcOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM))
    , m_dstC0Op(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM))
    , m_dstC1Op(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM))
    , m_dstC2Op(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM))
    , m_paddingPolicy(paddingPolicy)
    , m_prog{}
{
    if (!inTensorReqs || !outTensorReqs)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor requirements cannot be null");
    if (paddingPolicy != PVA_NHWC_TO_NCHW_PADDING_RESTORE_ON_SUBMIT &&
        paddingPolicy != PVA_NHWC_TO_NCHW_PADDING_PREFILLED)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid padding policy");

    /* ---- Validate input tensor (HWC U8, imgH x imgW x 3) ---- */
    if (inTensorReqs->rank != 3)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have rank 3 (HWC)");
    if (inTensorReqs->dtype != NVCV_DATA_TYPE_U8)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must be NVCV_DATA_TYPE_U8");

    NVCVTensorLayout hwcLayout;
    nvcvTensorLayoutMake("HWC", &hwcLayout);
    if (nvcvTensorLayoutCompare(inTensorReqs->layout, hwcLayout) != 0)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have HWC layout");

    /* ---- Validate output tensor (CHW F32, 3 x outH x outW) ---- */
    if (outTensorReqs->rank != 3)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor must have rank 3 (CHW)");
    if (outTensorReqs->dtype != NVCV_DATA_TYPE_F32)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor must be NVCV_DATA_TYPE_F32");

    NVCVTensorLayout chwLayout;
    nvcvTensorLayoutMake("CHW", &chwLayout);
    if (nvcvTensorLayoutCompare(outTensorReqs->layout, chwLayout) != 0)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor must have CHW layout");

    /* ---- Extract dimensions ---- */
    const int32_t imgH     = static_cast<int32_t>(inTensorReqs->shape[0]);
    const int32_t imgW     = static_cast<int32_t>(inTensorReqs->shape[1]);
    const int32_t channels = static_cast<int32_t>(inTensorReqs->shape[2]);

    const int32_t outC = static_cast<int32_t>(outTensorReqs->shape[0]);
    const int32_t outH = static_cast<int32_t>(outTensorReqs->shape[1]);
    const int32_t outW = static_cast<int32_t>(outTensorReqs->shape[2]);

    const int32_t inPitch = static_cast<int32_t>(inTensorReqs->strides[0]);

    if (inTensorReqs->strides[2] != sizeof(uint8_t) || inTensorReqs->strides[1] != 3 * sizeof(uint8_t))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input HWC pixels must be tightly packed within each row");
    if (outTensorReqs->strides[2] != sizeof(float) ||
        outTensorReqs->strides[1] != outW * static_cast<int64_t>(sizeof(float)) ||
        outTensorReqs->strides[0] != outH * outTensorReqs->strides[1])
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output CHW tensor must be tightly packed");

    if (imgH <= 0 || imgW <= 0 || channels <= 0 || outH <= 0 || outW <= 0)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    if (inTensorReqs->strides[0] < static_cast<int64_t>(imgW) * 3)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input strides[0] must be at least imgWidth * 3");
    if (channels != 3 || outC != 3)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Only 3-channel RGB is supported");
    if (!std::isfinite(scale) || !std::isfinite(padValue))
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Scale and padding value must be finite");
    if (imgW > outW || imgH > outH)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image dimensions must not exceed output dimensions");

    const int32_t padTop  = (outH - imgH) / 2;
    const int32_t padLeft = (outW - imgW) / 2;

    /* Cache for Submit() */
    m_imgW     = imgW;
    m_imgH     = imgH;
    m_outW     = outW;
    m_outH     = outH;
    m_padTop   = padTop;
    m_padLeft  = padLeft;
    m_channels = channels;
    m_padValue = padValue;

    /* ---- Build VPU program ---- */
    m_exec    = GetExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    /* Pack parameters */
    int8_t param_buf[sizeof(nhwc_to_nchw_letterbox_normalize_host2dev_param_t)];
    auto *vpu_param = reinterpret_cast<nhwc_to_nchw_letterbox_normalize_host2dev_param_t *>(param_buf);

    const int32_t tile_w  = NCHW_LN_TILE_W;
    const int32_t tile_h  = NCHW_LN_TILE_H;
    const int32_t niter_x = (imgW + tile_w - 1) / tile_w;
    const int32_t niter_y = (imgH + tile_h - 1) / tile_h;

    vpu_param->img_width   = imgW;
    vpu_param->img_height  = imgH;
    vpu_param->scale       = scale;
    vpu_param->tile_width  = tile_w;
    vpu_param->tile_height = tile_h;
    vpu_param->niter_x     = niter_x;
    vpu_param->niter_y     = niter_y;

    prog["vpu_param"].set(param_buf, sizeof(nhwc_to_nchw_letterbox_normalize_host2dev_param_t));

    /* VMEM buffer pointers */
    uint8_t *srcBuf = prog["srcBuf"].ptr<uint8_t>();
    float *dstC0Buf = prog["dstC0Buf"].ptr<float>();
    float *dstC1Buf = prog["dstC1Buf"].ptr<float>();
    float *dstC2Buf = prog["dstC2Buf"].ptr<float>();

    auto srcHdl = prog["srcHdl"];
    auto dstHdl = prog["dstHdl"];

    /*
     * INPUT RDF: uint8 HWC with TRANS_MODE_1 for transposed byte access.
     * Tiles over the image region (imgH x imgW x channels).
     */
    prog.addDataFlowHead<RasterDataFlow>()
        .handler(srcHdl)
        .src(m_srcOp, imgW * channels, imgH, inPitch)
        .tileBuffer(srcBuf)
        .tile(tile_w * channels, tile_h)
        .transpose(TRANS_MODE_1);

    /*
     * OUTPUT RDFs: 3x float32 planar with TRANS_MODE_1 for transposed float stores.
     * Each RDF covers one channel plane within the letterboxed output:
     *   DRAM width  = imgW (image region only)
     *   DRAM height = imgH
     *   DRAM pitch  = outW (full letterbox row width)
     * Base pointer (set in Submit) = outBase + ch * planeSize + (padTop * outW + padLeft)
     */
    prog.addDataFlowHead<RasterDataFlow>()
        .handler(dstHdl)
        .dst(m_dstC0Op, imgW, imgH, outW)
        .tileBuffer(dstC0Buf)
        .tile(tile_w, tile_h)
        .transpose(TRANS_MODE_1);

    prog.addDataFlowHead<RasterDataFlow>()
        .handler(dstHdl)
        .dst(m_dstC1Op, imgW, imgH, outW)
        .tileBuffer(dstC1Buf)
        .tile(tile_w, tile_h)
        .transpose(TRANS_MODE_1);

    prog.addDataFlowHead<RasterDataFlow>()
        .handler(dstHdl)
        .dst(m_dstC2Op, imgW, imgH, outW)
        .tileBuffer(dstC2Buf)
        .tile(tile_w, tile_h)
        .transpose(TRANS_MODE_1);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void NhwcToNchwLetterboxNormalize::operator()(S stream, nvcv::Tensor const &in, nvcv::Tensor const &out)
{
    auto inData  = in.exportData<nvcv::TensorDataStrided>();
    auto outData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outTensorRequirements, outData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor requirements must match those used at creation");
    }

    if (inData->basePtr() == nullptr || outData->basePtr() == nullptr)
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");

    float *outBase          = reinterpret_cast<float *>(outData->basePtr());
    const int32_t planeSize = m_outW * m_outH;

    /*
     * Restore padding before the PVA image-region write. Vertical padding is
     * filled as contiguous bands; side-padding cases use a full-plane fill to
     * avoid thousands of tiny row-strip operations. Every submission is
     * independent of the allocation's prior contents.
     */
    const bool hasPadding = m_imgW != m_outW || m_imgH != m_outH;
    if (hasPadding && m_paddingPolicy == PVA_NHWC_TO_NCHW_PADDING_RESTORE_ON_SUBMIT)
    {
        float *outHostPtr = nullptr;
        CUPVA_CHECK_ERROR(CupvaMemGetHostPointer(reinterpret_cast<void **>(&outHostPtr), outBase));
        if (outHostPtr == nullptr)
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Output tensor has no host-accessible PVA mapping");
        PaddingFillJob fillJob{outHostPtr, m_outW, m_outH, m_imgW, m_imgH, m_padTop, m_padLeft, m_channels, m_padValue};
        if constexpr (std::is_same_v<S, cudaStream_t>)
        {
            if constexpr (kCupvaHasCudaStreamSupport)
            {
                // The host function runs after prior CUDA/PVA work on this
                // stream and before the PVA command submitted below.
                auto *job      = new PaddingFillJob(fillJob);
                cudaError_t st = cudaLaunchHostFunc(stream, FillPaddingOnCudaStream, job);
                if (st != cudaSuccess)
                {
                    delete job;
                    throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL,
                                          "Failed to enqueue stream-ordered padding fill");
                }
            }
            else
            {
                ThrowCudaStreamNotSupported();
            }
        }
        else
        {
            // cuPVA has no host-callback command. Fence earlier work before
            // touching the host mapping, then enqueue the PVA write below.
            cupvaSyncObj_t rawSync = nullptr;
            cupvaFence_t fence;
            cupvaCmd_t request;
            CUPVA_CHECK_ERROR(CupvaSyncObjCreate(&rawSync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD));
            SyncObjPtr sync(rawSync);
            CUPVA_CHECK_ERROR(CupvaFenceInit(&fence, sync.get()));
            CUPVA_CHECK_ERROR(CupvaCmdRequestFencesInit(&request, &fence, 1));
            cupvaCmd_t const *commands[1] = {&request};
            cupvaCmdStatus_t status[1]    = {nullptr};
            CUPVA_CHECK_ERROR(CupvaStreamSubmit(stream, commands, status, 1, CUPVA_IN_ORDER, -1, -1));
            bool completed = false;
            CUPVA_CHECK_ERROR(CupvaFenceWait(&fence, -1, &completed));
            sync.reset();
            if (!completed)
                throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to order padding fill on cuPVA stream");
            FillPadding(fillJob);
        }
    }

    /* Set input base (start of image) */
    m_srcOp.setBase(reinterpret_cast<uint8_t *>(inData->basePtr()));

    /* Set output bases: each channel plane + letterbox offset within the plane */
    const int32_t padOffset = m_padTop * m_outW + m_padLeft;
    m_dstC0Op.setBase(outBase + 0 * planeSize + padOffset);
    m_dstC1Op.setBase(outBase + 1 * planeSize + padOffset);
    m_dstC2Op.setBase(outBase + 2 * planeSize + padOffset);

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
        static_assert(!sizeof(S), "Unsupported stream type for NhwcToNchwLetterboxNormalize operator");
    }
}

/* Explicit instantiations */
template void NhwcToNchwLetterboxNormalize::operator()<cupvaStream_t>(cupvaStream_t, nvcv::Tensor const &,
                                                                      nvcv::Tensor const &);
template void NhwcToNchwLetterboxNormalize::operator()<cudaStream_t>(cudaStream_t, nvcv::Tensor const &,
                                                                     nvcv::Tensor const &);

NhwcToNchwLetterboxNormalize::~NhwcToNchwLetterboxNormalize()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

/* =========================================================================
 * C API
 * ========================================================================= */

NVCVStatus pvaNhwcToNchwLetterboxNormalizeCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *inTensorReqs,
                                                 NVCVTensorRequirements const *outTensorReqs, float scale,
                                                 float padValue,
                                                 PVANhwcToNchwLetterboxNormalizePaddingPolicy paddingPolicy)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperatorHandle must not be NULL");

        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::NhwcToNchwLetterboxNormalize(
            inTensorReqs, outTensorReqs, scale, padValue, paddingPolicy));
    });
}

template<typename StreamType>
NVCVStatus NhwcToNchwLetterboxNormalizeSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                                  NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::NhwcToNchwLetterboxNormalize>(handle)(stream, input,
                                                                                                     output);
    });
}

template NVCVStatus NhwcToNchwLetterboxNormalizeSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t,
                                                                          NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus NhwcToNchwLetterboxNormalizeSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t,
                                                                         NVCVTensorHandle, NVCVTensorHandle);

// SubmitImpl owns ProtectCall because the generated plain-C wrappers in
// PvaSubmitDefinitions.cpp call it directly as well as these C++ overloads.
NVCVStatus pvaNhwcToNchwLetterboxNormalizeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out)
{
    return NhwcToNchwLetterboxNormalizeSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaNhwcToNchwLetterboxNormalizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out)
{
    return NhwcToNchwLetterboxNormalizeSubmitImpl(handle, stream, in, out);
}
