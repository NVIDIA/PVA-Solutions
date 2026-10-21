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

#include "OpFastCornerDetector.hpp"

#include "device/fast_corner_detector_param.hpp"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Image.hpp>

#include <OpFastCorner.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>

using namespace cupva;

// 16 executables: 4 data types x 2 NMS options x 2 TopK options
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U8_NMS_ON_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U8_NMS_ON_TOPK_OFF_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U8_NMS_OFF_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U8_NMS_OFF_TOPK_OFF_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U16_NMS_ON_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U16_NMS_ON_TOPK_OFF_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U16_NMS_OFF_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_U16_NMS_OFF_TOPK_OFF_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S8_NMS_ON_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S8_NMS_ON_TOPK_OFF_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S8_NMS_OFF_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S8_NMS_OFF_TOPK_OFF_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S16_NMS_ON_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S16_NMS_ON_TOPK_OFF_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S16_NMS_OFF_TOPK_ON_dev);
CUPVA_DECLARE_EXECUTABLE(pva_fastcorner_detector_S16_NMS_OFF_TOPK_OFF_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex FastCornerDetector<T>::s_execMapMutex;

template<typename T>
std::map<typename FastCornerDetector<T>::ExecKey, std::weak_ptr<cupva::Executable>>
    FastCornerDetector<T>::s_fastCornerExecutables;

// Add anonymous namespace to avoid duplicate with other operator's createExecDataMap functions.
namespace {

// Lambda to check if type is valid and get its corresponding NVCV data type
constexpr auto getNVCVType = []<typename T>() {
    struct Result
    {
        bool is_valid_type;
        NVCVDataType dtype;
    };

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        return Result{true, NVCV_DATA_TYPE_U8};
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        return Result{true, NVCV_DATA_TYPE_U16};
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return Result{true, NVCV_DATA_TYPE_S8};
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
        return Result{true, NVCV_DATA_TYPE_S16};
    }
    else
    {
        return Result{false, NVCV_DATA_TYPE_NONE};
    }
};

template<typename T>
std::map<typename FastCornerDetector<T>::ExecDataKey, IOperator::ExecutableData> createExecDataMap();

#define CREATE_EXEC_DATA_MAP(TYPE, TYPE_SUFFIX)                                                                   \
    template<>                                                                                                    \
    std::map<typename FastCornerDetector<TYPE>::ExecDataKey, IOperator::ExecutableData> createExecDataMap<TYPE>() \
    {                                                                                                             \
        return {                                                                                                  \
            {{0, 0},                                                                                              \
             {CUPVA_EXECUTABLE_DATA(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_OFF_TOPK_OFF_dev),                \
              CUPVA_EXECUTABLE_SIZE(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_OFF_TOPK_OFF_dev)}},              \
            {{1, 0},                                                                                              \
             {CUPVA_EXECUTABLE_DATA(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_ON_TOPK_OFF_dev),                 \
              CUPVA_EXECUTABLE_SIZE(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_ON_TOPK_OFF_dev)}},               \
            {{0, 1},                                                                                              \
             {CUPVA_EXECUTABLE_DATA(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_OFF_TOPK_ON_dev),                 \
              CUPVA_EXECUTABLE_SIZE(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_OFF_TOPK_ON_dev)}},               \
            {{1, 1},                                                                                              \
             {CUPVA_EXECUTABLE_DATA(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_ON_TOPK_ON_dev),                  \
              CUPVA_EXECUTABLE_SIZE(pva_fastcorner_detector_##TYPE_SUFFIX##_NMS_ON_TOPK_ON_dev)}},                \
        };                                                                                                        \
    }

CREATE_EXEC_DATA_MAP(uint8_t, U8)
CREATE_EXEC_DATA_MAP(uint16_t, U16)
CREATE_EXEC_DATA_MAP(int8_t, S8)
CREATE_EXEC_DATA_MAP(int16_t, S16)

} // namespace

template<typename T>
std::shared_ptr<cupva::Executable> FastCornerDetector<T>::GetFastCornerExec(ExecDataKey execDataKey)
{
    static auto execDataMap = createExecDataMap<T>();

    auto execDataIt = execDataMap.find(execDataKey);
    if (execDataIt != execDataMap.end())
    {
        // Create execKey by combining execDataKey and context
        ExecKey execKey = std::tuple_cat(execDataKey, std::make_tuple(cupva::Context::GetCurrent()));
        return GetExecutable(execDataIt->second, execKey, s_fastCornerExecutables, s_execMapMutex);
    }
    else
    {
        throw std::invalid_argument("Unsupported type or parameters");
    }
}

template<typename T>
FastCornerDetector<T>::FastCornerDetector(NVCVTensorRequirements const *const tensorRequirements,
                                          PvaFastCornerDetectorParams const *const params, NVCVBorderType borderMode,
                                          int32_t borderValue)
    : m_inTensorRequirements(*tensorRequirements)
    , m_srcOp{cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM)}
    , m_locOp{cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM)}
    , m_numLocOp{cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM)}
    , m_enableTopK{params->numTopKFeatures > 0}
{
    if (!(strcmp(nvcvTensorLayoutGetName(&tensorRequirements->layout), "HW") == 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input tensor layout!");
    }

    static constexpr auto typeInfo = getNVCVType.operator()<T>();
    static_assert(typeInfo.is_valid_type, "Unsupported template type T");
    if (!(tensorRequirements->dtype == typeInfo.dtype))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type!");
    }
    if (!(tensorRequirements->shape[0] <= kMaxImageWidthHeight && tensorRequirements->shape[1] <= kMaxImageWidthHeight))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image width and height should be less than %d",
                              kMaxImageWidthHeight);
    }
    if (!(params->circleRadius == 3 && params->arcLength == 9))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only supports circleRadius == 3 and arcLength == 9");
    }
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only NVCV_BORDER_CONSTANT and NVCV_BORDER_REPLICATE are supported");
    }
    if (params->numTopKFeatures < 0 || params->numTopKFeatures > MAX_LOC_CAPACITY)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numTopKFeatures must be 0 (disabled) or in range [1, %d]", MAX_LOC_CAPACITY);
    }

    const int32_t height{static_cast<int32_t>(tensorRequirements->shape[0])};
    const int32_t width{static_cast<int32_t>(tensorRequirements->shape[1])};
    const int32_t linePitch{static_cast<int32_t>(tensorRequirements->strides[0] / sizeof(T))};

    // Obtain the tile size from the template function defined in fast_corner_detector_param.hpp.
    const auto [tileX, tileY] = getTileSz<T>();

    if (params->nonMaxSuppression == 1)
    {
        m_scoreBufDevPtr = static_cast<T *>(cupva::mem::Alloc(height * width * sizeof(T)));
        if (m_scoreBufDevPtr.value() == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY,
                                  "Cannot allocate the memory using cupva::mem::Alloc()");
        }
    }

    // Allocate TopK intermediate DRAM buffers
    if (m_enableTopK)
    {
        constexpr int32_t maxPacked = MAX_LOC_CAPACITY;
        m_packedLocsDevPtr          = static_cast<uint32_t *>(cupva::mem::Alloc(maxPacked * sizeof(uint32_t)));
        m_packedScoresDevPtr        = static_cast<uint16_t *>(cupva::mem::Alloc(maxPacked * sizeof(uint16_t)));

        if (m_packedLocsDevPtr.value() == nullptr || m_packedScoresDevPtr.value() == nullptr)
        {
            // Clean up partial allocation
            if (m_packedLocsDevPtr.has_value() && m_packedLocsDevPtr.value() != nullptr)
            {
                cupva::mem::Free(m_packedLocsDevPtr.value());
            }
            if (m_packedScoresDevPtr.has_value() && m_packedScoresDevPtr.value() != nullptr)
            {
                cupva::mem::Free(m_packedScoresDevPtr.value());
            }
            throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Cannot allocate TopK intermediate buffers");
        }
    }

    m_exec = GetFastCornerExec(std::make_tuple(params->nonMaxSuppression, static_cast<uint8_t>(m_enableTopK ? 1 : 0)));
    auto prog = CmdProgram::Create(*m_exec.get());

    const int32_t niterX = (width + tileX - 1) / tileX;
    const int32_t niterY = (height + tileY - 1) / tileY;

    int8_t vpu_param_a[sizeof(FastcornerHost2DevParam)];
    FastcornerHost2DevParam *vpuParam = (FastcornerHost2DevParam *)vpu_param_a;
    vpuParam->width                   = width;
    vpuParam->height                  = height;
    vpuParam->niterX                  = niterX;
    vpuParam->niterY                  = niterY;
    vpuParam->numTopKFeatures         = params->numTopKFeatures;

    prog["vpuParam"].set(vpu_param_a, sizeof(FastcornerHost2DevParam));
    auto srcHdl = prog["srcHdl"];

    T *scoreInBuf0Ptr = (T *)prog["scoreInBuf0"].ptr<T>();
    T *scoreInBuf1Ptr = (T *)prog["scoreInBuf1"].ptr<T>();
    T *scoreInBuf2Ptr = (T *)prog["scoreInBuf2"].ptr<T>();

    RasterDataFlow &imgCp0DataFlow = prog.addDataFlowHead<RasterDataFlow>();
    imgCp0DataFlow.handler(srcHdl).tile(tileX, tileY).src(m_srcOp, width, height, linePitch).tileBuffer(scoreInBuf0Ptr);
    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        imgCp0DataFlow.halo(params->circleRadius, PAD_BPE);
    }
    else
    {
        imgCp0DataFlow.halo(params->circleRadius, PAD_CONST, borderValue);
    }

    RasterDataFlow &imgCp1DataFlow = prog.addDataFlowHead<RasterDataFlow>();
    // tileBuffer() accepts a second parameter for passing the duplicated buffer, which can save one dataflow.
    imgCp1DataFlow.handler(srcHdl)
        .tile(tileX, tileY)
        .src(m_srcOp, width, height, linePitch)
        .tileBuffer(scoreInBuf1Ptr, scoreInBuf2Ptr);
    if (borderMode == NVCV_BORDER_REPLICATE)
    {
        imgCp1DataFlow.halo(params->circleRadius, PAD_BPE);
    }
    else
    {
        imgCp1DataFlow.halo(params->circleRadius, PAD_CONST, borderValue);
    }

    if (params->nonMaxSuppression == 1)
    {
        auto scoreHdl  = prog["scoreHdl"];
        auto nmsInTrig = prog["nmsInTrig"];

        T *scoreBufPtr = (T *)prog["scoreBuf"].ptr<T>();
        T *nmsInBufPtr = (T *)prog["nmsInBuf"].ptr<T>();

        RasterDataFlow &scoreDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        scoreDataFlow.handler(scoreHdl)
            .tile(tileX, tileY)
            .tileBuffer(scoreBufPtr)
            .dst(m_scoreBufDevPtr.value(), width, height, width); // intermediate output's line pitch equals width

        RasterDataFlow &scoreInDataFlow = prog.addDataFlowHead<RasterDataFlow>();
        scoreInDataFlow.handler(nmsInTrig)
            .tile(tileX, tileY)
            .src(m_scoreBufDevPtr.value(), width, height, width)
            .tileBuffer(nmsInBufPtr)
            .halo(1, PAD_CONST);
    }

    auto locCntSQDFHdl               = prog["locCntSQDFHdl"];
    int32_t *locCntBuf               = prog["locCntBuf"].ptr<int32_t>();
    SequenceDataFlow &locCntDataFlow = prog.addDataFlowHead<SequenceDataFlow>(1);
    locCntDataFlow.handler(locCntSQDFHdl)
        .addTransfer()
        .src(locCntBuf, sizeof(int32_t))
        .dst(m_numLocOp, sizeof(int32_t))
        .tile(sizeof(int32_t), 1);

    int32_t *locBuf = prog["locBuf"].ptr<int32_t>();

    auto locSQDFHdl               = prog["locSQDFHdl"];
    SequenceDataFlow &locDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    locDataFlow.handler(locSQDFHdl)
        .addTransfer()
        .src(locBuf, LOC_BUF_SIZE * sizeof(float) * 2)
        .dst(m_locOp, sizeof(float))
        .tile(sizeof(float), 1);

    // Configure SQDF for TopK packed data write/read
    if (m_enableTopK)
    {
        uint32_t *packBufPtr      = prog["packBuf"].ptr<uint32_t>();
        uint16_t *packScoreBufPtr = prog["packScoreBuf"].ptr<uint16_t>();

        // SQDF for WRITE: 2 transfers (locations + scores) from VMEM to DRAM
        SequenceDataFlow &writeFlow = prog.addDataFlowHead<SequenceDataFlow>();
        writeFlow.handler(prog["packedDataWriteHdl"]);

        writeFlow.addTransfer()
            .src(packBufPtr, sizeof(uint32_t))
            .dst(m_packedLocsDevPtr.value(), sizeof(uint32_t))
            .tile(sizeof(uint32_t), 1);

        writeFlow.addTransfer()
            .src(packScoreBufPtr, sizeof(uint16_t))
            .dst(m_packedScoresDevPtr.value(), sizeof(uint16_t))
            .tile(sizeof(uint16_t), 1);

        // SQDF for READ: 2 transfers (locations + scores) from DRAM to VMEM
        SequenceDataFlow &readFlow = prog.addDataFlowHead<SequenceDataFlow>();
        readFlow.handler(prog["packedDataReadHdl"]);

        readFlow.addTransfer()
            .src(m_packedLocsDevPtr.value(), sizeof(uint32_t))
            .dst(packBufPtr, sizeof(uint32_t))
            .tile(sizeof(uint32_t), 1);

        readFlow.addTransfer()
            .src(m_packedScoresDevPtr.value(), sizeof(uint16_t))
            .dst(packScoreBufPtr, sizeof(uint16_t))
            .tile(sizeof(uint16_t), 1);
    }

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void FastCornerDetector<T>::operator()(S stream, nvcv::Tensor const &in, int32_t intensityThreshold,
                                       nvcv::Tensor const &loc, nvcv::Tensor const &numLoc)
{
    auto inTensorData     = in.exportData<nvcv::TensorDataStrided>();
    auto outTensorData    = loc.exportData<nvcv::TensorDataStrided>();
    auto numLocTensorData = numLoc.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensors must have the same requirements that was used to create the operator");
    }

    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(numLocTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be nullptr");
    }

    // Location tensors are required to have rank 1 and layout W.
    if (loc.rank() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output location tensors must have rank 1!");
    }

    if (intensityThreshold < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "intensityThreshold must be non negative!");
    }

    // Obtain the location buffer's capacity from the tensor size.
    int32_t locCapacity = loc.shape()[0];

    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "locCapacity");
        CupvaParameterSetValueScalar(&param, &locCapacity, sizeof(int32_t));
    }

    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "threshold");
        CupvaParameterSetValueScalar(&param, &intensityThreshold, sizeof(int32_t));
    }

    m_srcOp.setBase(reinterpret_cast<T *>(inTensorData->basePtr()));
    m_locOp.setBase(reinterpret_cast<float *>(outTensorData->basePtr()));
    m_numLocOp.setBase(reinterpret_cast<int32_t *>(numLocTensorData->basePtr()));

    // Set external memory pointers for SQDF.
    cupvaParameter_t locExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &locExtPtr, "locExtPtr");
    CupvaParameterSetValuePointer(&locExtPtr, static_cast<void *>(outTensorData->basePtr()));

    // Set TopK external memory pointers
    if (m_enableTopK)
    {
        cupvaParameter_t packedLocsPtr;
        CupvaCmdProgramGetParameter(&m_prog, &packedLocsPtr, "packedLocsExtPtr");
        CupvaParameterSetValuePointer(&packedLocsPtr, static_cast<void *>(m_packedLocsDevPtr.value()));

        cupvaParameter_t packedScoresPtr;
        CupvaCmdProgramGetParameter(&m_prog, &packedScoresPtr, "packedScoresExtPtr");
        CupvaParameterSetValuePointer(&packedScoresPtr, static_cast<void *>(m_packedScoresDevPtr.value()));
    }

    cupvaError_t err                = CupvaCmdProgramUpdateDataFlows(&m_prog);
    cupvaCmd_t const *submitCmds[1] = {&m_prog};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, submitCmds, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for FastCornerDetector operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename T>
FastCornerDetector<T>::~FastCornerDetector()
{
    CupvaCmdDestroy(&m_prog);
    if (m_scoreBufDevPtr != std::nullopt)
    {
        cupva::mem::Free(m_scoreBufDevPtr.value());
    }
    if (m_packedLocsDevPtr.has_value() && m_packedLocsDevPtr.value() != nullptr)
    {
        cupva::mem::Free(m_packedLocsDevPtr.value());
    }
    if (m_packedScoresDevPtr.has_value() && m_packedScoresDevPtr.value() != nullptr)
    {
        cupva::mem::Free(m_packedScoresDevPtr.value());
    }
}

// Explicit instantiation of the template to support the types we care about.
template class FastCornerDetector<uint16_t>;
template class FastCornerDetector<uint8_t>;
template class FastCornerDetector<int16_t>;
template class FastCornerDetector<int8_t>;
} // namespace pva_solutions::priv

NVCVStatus pvaFastCornerDetectorCreate(NVCVOperatorHandle *handle,
                                       NVCVTensorRequirements const *const tensorRequirements,
                                       PvaFastCornerDetectorParams const *const params, NVCVBorderType borderMode,
                                       int32_t borderValue)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be nullptr");
        }
        switch (tensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::FastCornerDetector<uint16_t>(
                tensorRequirements, params, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_U8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::FastCornerDetector<uint8_t>(
                tensorRequirements, params, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::FastCornerDetector<int16_t>(
                tensorRequirements, params, borderMode, borderValue));
            break;
        case NVCV_DATA_TYPE_S8:
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::FastCornerDetector<int8_t>(
                tensorRequirements, params, borderMode, borderValue));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus FastCornerDetectorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle in,
                                        int32_t intensityThreshold, NVCVTensorHandle loc, NVCVTensorHandle numLoc)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), _loc(loc), _numLoc(numLoc);

        NVCVDataType inDataType;
        nvcvTensorGetDataType(in, &inDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::FastCornerDetector<uint16_t>>(handle)(
                stream, input, intensityThreshold, _loc, _numLoc);
            break;
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::FastCornerDetector<uint8_t>>(handle)(
                stream, input, intensityThreshold, _loc, _numLoc);
            break;
        case NVCV_DATA_TYPE_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::FastCornerDetector<int16_t>>(handle)(
                stream, input, intensityThreshold, _loc, _numLoc);
            break;
        case NVCV_DATA_TYPE_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::FastCornerDetector<int8_t>>(handle)(
                stream, input, intensityThreshold, _loc, _numLoc);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus FastCornerDetectorSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                                int32_t, NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus FastCornerDetectorSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                               int32_t, NVCVTensorHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaFastCornerDetectorSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                       int32_t intensityThreshold, NVCVTensorHandle loc, NVCVTensorHandle numLoc)
{
    return FastCornerDetectorSubmitImpl(handle, stream, in, intensityThreshold, loc, numLoc);
}

NVCVStatus pvaFastCornerDetectorSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                       int32_t intensityThreshold, NVCVTensorHandle loc, NVCVTensorHandle numLoc)
{
    return FastCornerDetectorSubmitImpl(handle, stream, in, intensityThreshold, loc, numLoc);
}
