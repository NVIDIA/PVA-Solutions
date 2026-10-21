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

#include "OpYoloV5Decode.hpp"

#include "device/yolov5_decode_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <cupva_host_utils.hpp>
#include <nvcv/Tensor.hpp>

#include <OpYoloV5Decode.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_yolov5_decode_dev);

namespace pva_solutions::priv {

namespace {
[[noreturn]] void ThrowCupvaFailure(const char *operation, cupvaError_t error)
{
    const char *detail = nullptr;
    CupvaGetLastError(&detail);
    throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "%s failed: CUPVA error %d%s%s", operation, error,
                          detail ? ": " : "", detail ? detail : "");
}
} // namespace

std::mutex YoloV5Decode::s_execMapMutex;
std::map<YoloV5Decode::ExecKey, std::weak_ptr<cupva::Executable>> YoloV5Decode::s_executables;

std::shared_ptr<cupva::Executable> YoloV5Decode::GetExec()
{
    static IOperator::ExecutableData execData = {CUPVA_EXECUTABLE_DATA(pva_yolov5_decode_dev),
                                                 CUPVA_EXECUTABLE_SIZE(pva_yolov5_decode_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(execData, execKey, s_executables, s_execMapMutex);
}

YoloV5Decode::YoloV5Decode(const NVCVTensorRequirements *inPredReqs, const NVCVTensorRequirements *inAnchorReqs,
                           const NVCVTensorRequirements *outBoxesReqs, const NVCVTensorRequirements *outScoresReqs,
                           const NVCVTensorRequirements *outClassesReqs, const NVCVTensorRequirements *outCountReqs,
                           int numClasses, float confThreshold, int imgHeight, int imgWidth,
                           PVAYoloV5DecodeInputLayout inputLayout)
    : m_hwcInput(inputLayout == PVA_YOLOV5_DECODE_INPUT_HWC)
    , m_numClasses(numClasses)
    , m_confThreshold(confThreshold)
    , m_imgHeight(imgHeight)
    , m_imgWidth(imgWidth)
    , m_inPredTensorRequirements(inPredReqs ? *inPredReqs : NVCVTensorRequirements{})
    , m_inAnchorTensorRequirements(inAnchorReqs ? *inAnchorReqs : NVCVTensorRequirements{})
    , m_outBoxesTensorRequirements(outBoxesReqs ? *outBoxesReqs : NVCVTensorRequirements{})
    , m_outScoresTensorRequirements(outScoresReqs ? *outScoresReqs : NVCVTensorRequirements{})
    , m_outClassesTensorRequirements(outClassesReqs ? *outClassesReqs : NVCVTensorRequirements{})
    , m_outCountTensorRequirements(outCountReqs ? *outCountReqs : NVCVTensorRequirements{})
    , m_inputOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_anchorOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_boxesOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_scoresOp(cupva::OffsetPointer<int16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_classesOp(cupva::OffsetPointer<int16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_countOp(cupva::OffsetPointer<int>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    if (!inPredReqs || !inAnchorReqs || !outBoxesReqs || !outScoresReqs || !outClassesReqs || !outCountReqs)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor requirements cannot be null");
    }

    if (numClasses != 80)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "YoloV5Decode currently supports exactly 80 classes");
    }
    if (!(confThreshold > 0.0f && confThreshold < 1.0f) || imgHeight <= 0 || imgWidth <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Confidence threshold and image dimensions are invalid");
    }
    if (inputLayout != PVA_YOLOV5_DECODE_INPUT_CHW && inputLayout != PVA_YOLOV5_DECODE_INPUT_HWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported prediction layout");
    }
    if (inPredReqs->rank != 3 || inPredReqs->dtype != NVCV_DATA_TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Predictions must be rank-3 F32");
    }

    const int numPredictionChannels = 3 * (numClasses + 5);

    if (m_hwcInput)
    {
        if (inPredReqs->shape[2] != numPredictionChannels)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "HWC prediction channel count does not match numClasses");
        }
        if (inPredReqs->strides[2] != sizeof(float) ||
            inPredReqs->strides[1] != inPredReqs->shape[2] * static_cast<int64_t>(sizeof(float)) ||
            inPredReqs->strides[0] != inPredReqs->shape[1] * inPredReqs->strides[1])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "HWC predictions must be tightly packed");
        }
        m_predHW = static_cast<int>(inPredReqs->shape[0] * inPredReqs->shape[1]);
    }
    else
    {
        if (inPredReqs->shape[0] != numPredictionChannels)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "CHW prediction channel count does not match numClasses");
        }
        if (inPredReqs->strides[2] != sizeof(float) ||
            inPredReqs->strides[1] != inPredReqs->shape[2] * static_cast<int64_t>(sizeof(float)) ||
            inPredReqs->strides[0] != inPredReqs->shape[1] * inPredReqs->strides[1])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "CHW predictions must be tightly packed");
        }
        m_predHW = static_cast<int>(inPredReqs->shape[1] * inPredReqs->shape[2]);
    }
    m_numAnchors = m_predHW * 3;
    if (inAnchorReqs->rank != 2 || inAnchorReqs->dtype != NVCV_DATA_TYPE_F32 ||
        inAnchorReqs->shape[0] != m_numAnchors || inAnchorReqs->shape[1] != 5)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Anchors must be F32 [numAnchors, 5]");
    }
    if (inAnchorReqs->strides[1] != sizeof(float) || inAnchorReqs->strides[0] != 5 * sizeof(float))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Anchors must be tightly packed");
    }
    if (m_predHW <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Predictions H and W dimensions must be positive");
    }
    if (outBoxesReqs->rank != 2 || outBoxesReqs->dtype != NVCV_DATA_TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output boxes must be rank-2 F32");
    }
    if (outBoxesReqs->shape[0] != 4 || outBoxesReqs->shape[1] <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output boxes must be F32 SoA [4, maxOutputBoxes]");
    }
    m_maxOutputBoxes = static_cast<int>(outBoxesReqs->shape[1]);
    if (m_maxOutputBoxes > VPU_MAX_OBJECTS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Max output boxes exceeds VPU limit");
    }
    if (outScoresReqs->rank != 1 || outScoresReqs->dtype != NVCV_DATA_TYPE_F16 ||
        outScoresReqs->shape[0] != m_maxOutputBoxes || outClassesReqs->rank != 1 ||
        outClassesReqs->dtype != NVCV_DATA_TYPE_S16 || outClassesReqs->shape[0] != m_maxOutputBoxes ||
        outCountReqs->rank != 1 || outCountReqs->dtype != NVCV_DATA_TYPE_S32 || outCountReqs->shape[0] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Decode output tensor types or extents are inconsistent");
    }
    if (outBoxesReqs->strides[1] != sizeof(float) ||
        outBoxesReqs->strides[0] < m_maxOutputBoxes * static_cast<int64_t>(sizeof(float)) ||
        outScoresReqs->strides[0] != sizeof(uint16_t) || outClassesReqs->strides[0] != sizeof(int16_t) ||
        outCountReqs->strides[0] != sizeof(int32_t))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Decode outputs must use contiguous elements; box row padding is allowed");
    }

    m_exec    = GetExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    // Tiling: spatial_tile_size = 49 = 16*3+1 for bank-conflict-free vfloat_load_transp
    const int anchor_item_size  = 5;
    const int spatial_tile_size = 49;
    const int tile_size         = spatial_tile_size * 3; // 147 anchors per tile
    const int num_tiles         = (m_predHW + spatial_tile_size - 1) / spatial_tile_size;
    const int last_spatial_tile = m_predHW - spatial_tile_size * (num_tiles - 1);
    const int last_tile_size    = last_spatial_tile * 3;

    if (num_tiles <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid tiling: num_tiles must be positive (predHW=%d)", m_predHW);
    }

    // VPU parameters
    char vpu_param_a[sizeof(yolov5_decode_host2dev_param_t)];
    yolov5_decode_host2dev_param_t *vpu_param = (yolov5_decode_host2dev_param_t *)vpu_param_a;
    vpu_param->tile_size                      = tile_size;
    vpu_param->num_tiles                      = num_tiles;
    vpu_param->anchor_item_size               = anchor_item_size;
    vpu_param->num_classes                    = numClasses;
    vpu_param->conf_threshold                 = confThreshold;
    vpu_param->deconf_threshold               = -std::log(1.0f / confThreshold - 1.0f);
    vpu_param->image_height                   = imgHeight;
    vpu_param->image_width                    = imgWidth;
    vpu_param->max_objects                    = m_maxOutputBoxes;
    vpu_param->spatial_tile_size              = spatial_tile_size;
    vpu_param->last_spatial_tile              = last_spatial_tile;
    vpu_param->hwc_input                      = m_hwcInput ? 1 : 0;
    prog["vpu_param"].set(vpu_param_a, sizeof(yolov5_decode_host2dev_param_t));

    float *inputBuf_v       = prog["inputBuf"].ptr<float>();
    RasterDataFlow &inputDf = prog.addDataFlowHead<RasterDataFlow>();
    if (m_hwcInput)
    {
        // HWC: DRAM [predHW, 255] -> VMEM C (double-buffered [spatial_tile, 255] tiles)
        inputDf.handler(prog["inputRdfHdl"])
            .src(m_inputOp, numPredictionChannels, m_predHW, numPredictionChannels)
            .tileBuffer(inputBuf_v)
            .tile(numPredictionChannels, spatial_tile_size);
    }
    else
    {
        // CHW: DRAM [255, predHW] -> VMEM C (double-buffered [255, spatial_tile] tiles)
        inputDf.handler(prog["inputRdfHdl"])
            .src(m_inputOp, m_predHW, numPredictionChannels, m_predHW)
            .tileBuffer(inputBuf_v)
            .tile(spatial_tile_size, numPredictionChannels)
            .tileArena(spatial_tile_size, numPredictionChannels + 1);
    }

    // Anchor SQDF: DRAM [numAnchors, 5] -> VMEM B anchorBuf (double-buffered)
    SequenceDataFlow &anchorDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["anchorTrig"]);
    int8_t *anchor_v           = prog["anchorBuf"].ptr<int8_t>();
    int anchor_tile_bytes      = tile_size * anchor_item_size * sizeof(float);
    // OffsetPointer arithmetic counts elements of the pointee type (float).
    int anchor_tile_elems      = tile_size * anchor_item_size;
    int anchor_last_tile_bytes = last_tile_size * anchor_item_size * sizeof(float);

    if (num_tiles % 2 == 0 && last_tile_size == tile_size)
    {
        // When every tile is full and the tile count is even, one transfer
        // covers the double-buffered anchor traversal.
        anchorDf.addTransfer()
            .tile(anchor_tile_bytes)
            .src(m_anchorOp, anchor_tile_bytes)
            .srcDim1(num_tiles, anchor_tile_bytes)
            .dst(anchor_v, anchor_tile_bytes)
            .dstDim1(2, anchor_tile_bytes)
            .dstDim2(num_tiles / 2, 0)
            .mode(TransferModeType::TILE);
    }
    else if (num_tiles % 2)
    {
        if (num_tiles > 1)
        {
            anchorDf.addTransfer()
                .tile(anchor_tile_bytes)
                .src(m_anchorOp, anchor_tile_bytes)
                .srcDim1(num_tiles - 1, anchor_tile_bytes)
                .dst(anchor_v, anchor_tile_bytes)
                .dstDim1(2, anchor_tile_bytes)
                .dstDim2((num_tiles - 1) / 2, 0)
                .mode(TransferModeType::TILE);
        }
        anchorDf.addTransfer()
            .tile(anchor_last_tile_bytes)
            .src(m_anchorOp + (num_tiles - 1) * anchor_tile_elems, 0)
            .srcDim1(1, 0)
            .dst(anchor_v, 0)
            .dstDim1(1, 0)
            .mode(TransferModeType::TILE);
    }
    else
    {
        if (num_tiles > 2)
        {
            anchorDf.addTransfer()
                .tile(anchor_tile_bytes)
                .src(m_anchorOp, anchor_tile_bytes)
                .srcDim1(num_tiles - 2, anchor_tile_bytes)
                .dst(anchor_v, anchor_tile_bytes)
                .dstDim1(2, anchor_tile_bytes)
                .dstDim2((num_tiles - 2) / 2, 0)
                .mode(TransferModeType::TILE);
        }
        anchorDf.addTransfer()
            .tile(anchor_tile_bytes)
            .src(m_anchorOp + (num_tiles - 2) * anchor_tile_elems, 0)
            .srcDim1(1, 0)
            .dst(anchor_v, 0)
            .dstDim1(1, 0)
            .mode(TransferModeType::TILE);
        anchorDf.addTransfer()
            .tile(anchor_last_tile_bytes)
            .src(m_anchorOp + (num_tiles - 1) * anchor_tile_elems, 0)
            .srcDim1(1, 0)
            .dst(anchor_v + anchor_tile_bytes, 0)
            .dstDim1(1, 0)
            .mode(TransferModeType::TILE);
    }

    // Output boxes (SoA: left, top, right, bottom)
    SequenceDataFlow &outputDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["outputTrig"]);
    outputDf.addTransfer()
        .tile(sizeof(float), 4)
        .src(prog["outputBuf"].ptr<float>(), VPU_MAX_OBJECTS * sizeof(float))
        .dst(m_boxesOp, outBoxesReqs->strides[0])
        .mode(TransferModeType::TILE);

    // Output confidence scores
    SequenceDataFlow &outputConfDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["outputConfTrig"]);
    outputConfDf.addTransfer()
        .tile(sizeof(short))
        .src(prog["outputConfBuf"].ptr<short>(), 0)
        .dst(m_scoresOp, 0)
        .mode(TransferModeType::TILE);

    // Output class labels
    SequenceDataFlow &outputClsDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["outputClsTrig"]);
    outputClsDf.addTransfer()
        .tile(sizeof(short))
        .src(prog["outputClsBuf"].ptr<short>(), 0)
        .dst(m_classesOp, 0)
        .mode(TransferModeType::TILE);

    // Output count
    SequenceDataFlow &outputNumDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["outputNumTrig"]);
    outputNumDf.addTransfer()
        .tile(sizeof(int))
        .src(prog["numSurvivors"].ptr<int>(), 0)
        .srcDim1(1, 0)
        .dst(m_countOp, 0)
        .dstDim1(1, 0);

    // Input count (read DRAM append cursor into VMEM at kernel start)
    SequenceDataFlow &inputNumDf = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["inputNumTrig"]);
    inputNumDf.addTransfer()
        .tile(sizeof(int))
        .src(m_countOp, 0)
        .srcDim1(1, 0)
        .dst(prog["numSurvivors"].ptr<int>(), 0)
        .dstDim1(1, 0);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);

    CupvaCmdProgramGetParameter(&m_prog, &m_outputPtrParam, "outputPtr");
    CupvaCmdProgramGetParameter(&m_prog, &m_outputConfPtrParam, "outputConfPtr");
    CupvaCmdProgramGetParameter(&m_prog, &m_outputClsPtrParam, "outputClsPtr");
}

template<typename StreamType>
void YoloV5Decode::operator()(StreamType stream, const nvcv::Tensor &predictions, const nvcv::Tensor &anchors,
                              const nvcv::Tensor &boxes, const nvcv::Tensor &scores, const nvcv::Tensor &classes,
                              const nvcv::Tensor &count)
{
    auto predData    = predictions.exportData<nvcv::TensorDataStrided>();
    auto anchorData  = anchors.exportData<nvcv::TensorDataStrided>();
    auto boxesData   = boxes.exportData<nvcv::TensorDataStrided>();
    auto scoresData  = scores.exportData<nvcv::TensorDataStrided>();
    auto classesData = classes.exportData<nvcv::TensorDataStrided>();
    auto countData   = count.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inPredTensorRequirements, predData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_inAnchorTensorRequirements, anchorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outBoxesTensorRequirements, boxesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outScoresTensorRequirements, scoresData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outClassesTensorRequirements, classesData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outCountTensorRequirements, countData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that were used to create the operator");
    }

    if ((static_cast<void *>(predData->basePtr()) == nullptr) ||
        (static_cast<void *>(anchorData->basePtr()) == nullptr) ||
        (static_cast<void *>(boxesData->basePtr()) == nullptr) ||
        (static_cast<void *>(scoresData->basePtr()) == nullptr) ||
        (static_cast<void *>(classesData->basePtr()) == nullptr) ||
        (static_cast<void *>(countData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    void *predBase    = predData->basePtr();
    void *anchorBase  = anchorData->basePtr();
    void *boxesBase   = boxesData->basePtr();
    void *scoresBase  = scoresData->basePtr();
    void *classesBase = classesData->basePtr();
    void *countBase   = countData->basePtr();

    m_inputOp.setBase(static_cast<float *>(predBase));
    m_anchorOp.setBase(static_cast<float *>(anchorBase));
    m_boxesOp.setBase(static_cast<float *>(boxesBase));
    m_scoresOp.setBase(static_cast<int16_t *>(scoresBase));
    m_classesOp.setBase(static_cast<int16_t *>(classesBase));
    m_countOp.setBase(static_cast<int *>(countBase));

    CupvaParameterSetValuePointer(&m_outputPtrParam, boxesBase);
    CupvaParameterSetValuePointer(&m_outputConfPtrParam, scoresBase);
    CupvaParameterSetValuePointer(&m_outputClsPtrParam, classesBase);

    const cupvaError_t updateStatus = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (updateStatus != CUPVA_ERROR_NONE)
    {
        ThrowCupvaFailure("PVA Update DataFlow", updateStatus);
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {NULL};

    if constexpr (std::is_same_v<StreamType, cupvaStream_t>)
    {
        const cupvaError_t submitStatus = CupvaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1);
        if (submitStatus != CUPVA_ERROR_NONE)
        {
            ThrowCupvaFailure("PVA Cmd Submit", submitStatus);
        }
    }
    else if constexpr (std::is_same_v<StreamType, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            const cupvaError_t submitStatus =
                CupvaCudaStreamSubmit(stream, submitCmds, cmdStatus, 1, CUPVA_IN_ORDER, -1, -1);
            if (submitStatus != CUPVA_ERROR_NONE)
            {
                ThrowCupvaFailure("PVA CUDA-stream Cmd Submit", submitStatus);
            }
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(StreamType), "Unsupported stream type for YoloV5Decode operator");
    }
}

YoloV5Decode::~YoloV5Decode()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaYoloV5DecodeCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *inPredReqs,
                                 NVCVTensorRequirements const *inAnchorReqs, NVCVTensorRequirements const *outBoxesReqs,
                                 NVCVTensorRequirements const *outScoresReqs,
                                 NVCVTensorRequirements const *outClassesReqs,
                                 NVCVTensorRequirements const *outCountReqs, int numClasses, float confThreshold,
                                 int imgHeight, int imgWidth, PVAYoloV5DecodeInputLayout inputLayout)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (!inPredReqs || !inAnchorReqs || !outBoxesReqs || !outScoresReqs || !outClassesReqs || !outCountReqs)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor requirements cannot be null");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::YoloV5Decode(
            inPredReqs, inAnchorReqs, outBoxesReqs, outScoresReqs, outClassesReqs, outCountReqs, numClasses,
            confThreshold, imgHeight, imgWidth, inputLayout));
    });
}

template<typename StreamType>
NVCVStatus YoloV5DecodeSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle predictions,
                                  NVCVTensorHandle anchors, NVCVTensorHandle boxes, NVCVTensorHandle scores,
                                  NVCVTensorHandle classes, NVCVTensorHandle count)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle pred(predictions), anch(anchors), box(boxes), score(scores), cls(classes), cnt(count);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::YoloV5Decode>(handle)(stream, pred, anch, box, score,
                                                                                     cls, cnt);
    });
}

template NVCVStatus YoloV5DecodeSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                          NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                          NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus YoloV5DecodeSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                         NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                         NVCVTensorHandle, NVCVTensorHandle);

// SubmitImpl owns ProtectCall because the generated plain-C wrappers in
// PvaSubmitDefinitions.cpp call it directly as well as these C++ overloads.
NVCVStatus pvaYoloV5DecodeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle predictions,
                                 NVCVTensorHandle anchors, NVCVTensorHandle boxes, NVCVTensorHandle scores,
                                 NVCVTensorHandle classes, NVCVTensorHandle count)
{
    return YoloV5DecodeSubmitImpl(handle, stream, predictions, anchors, boxes, scores, classes, count);
}

NVCVStatus pvaYoloV5DecodeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle predictions,
                                 NVCVTensorHandle anchors, NVCVTensorHandle boxes, NVCVTensorHandle scores,
                                 NVCVTensorHandle classes, NVCVTensorHandle count)
{
    return YoloV5DecodeSubmitImpl(handle, stream, predictions, anchors, boxes, scores, classes, count);
}
