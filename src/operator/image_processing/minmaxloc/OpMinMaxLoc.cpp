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

#include "OpMinMaxLoc.h"

#include "OpMinMaxLoc.hpp"

#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_utils.hpp>

#include <ErrorCheckMacros.h>
#include <cupva_host_cuda.h>

#include <sstream>
#include <typeinfo>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_minmaxloc_u8_dev);
CUPVA_DECLARE_EXECUTABLE(pva_minmaxloc_s8_dev);
CUPVA_DECLARE_EXECUTABLE(pva_minmaxloc_u16_dev);
CUPVA_DECLARE_EXECUTABLE(pva_minmaxloc_s16_dev);
CUPVA_DECLARE_EXECUTABLE(pva_minmaxloc_u32_dev);
CUPVA_DECLARE_EXECUTABLE(pva_minmaxloc_s32_dev);
CUPVA_DECLARE_EXECUTABLE(pva_minmaxloc_f32_dev);

namespace pva_solutions::priv {

template<typename T>
std::mutex MinMaxLoc<T>::s_execMapMutex;

template<typename T>
std::map<typename MinMaxLoc<T>::ExecKey, std::weak_ptr<cupva::Executable>> MinMaxLoc<T>::s_minMaxLocExecutables;

template<typename T>
IOperator::ExecutableData createExecData();

#define CREATE_EXEC_DATA_MAP(TYPE, TYPE_SUFFIX)                            \
    template<>                                                             \
    IOperator::ExecutableData createExecData<TYPE>()                       \
    {                                                                      \
        return {CUPVA_EXECUTABLE_DATA(pva_minmaxloc_##TYPE_SUFFIX##_dev),  \
                CUPVA_EXECUTABLE_SIZE(pva_minmaxloc_##TYPE_SUFFIX##_dev)}; \
    }

CREATE_EXEC_DATA_MAP(uint8_t, u8)
CREATE_EXEC_DATA_MAP(int8_t, s8)
CREATE_EXEC_DATA_MAP(uint16_t, u16)
CREATE_EXEC_DATA_MAP(int16_t, s16)
CREATE_EXEC_DATA_MAP(uint32_t, u32)
CREATE_EXEC_DATA_MAP(int32_t, s32)
CREATE_EXEC_DATA_MAP(float, f32)

template<typename T>
std::shared_ptr<cupva::Executable> MinMaxLoc<T>::GetMinMaxLocExec()
{
    static IOperator::ExecutableData sMinMaxLocExecData = createExecData<T>();

    // Create execKey by combining execDataKey and context
    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return GetExecutable(sMinMaxLocExecData, execKey, s_minMaxLocExecutables, s_execMapMutex);
}

template<typename>
inline constexpr bool dependent_false_v = false;

// agen stride to use in location kernels
template<typename T>
constexpr int getDataTypeLocStride()
{
    if constexpr (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value)
    {
        return 32;
    }
    else if constexpr (std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value)
    {
        return 32;
    }
    else if constexpr (std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value ||
                       std::is_same<T, float>::value)
    {
        return 16;
    }
    else
    {
        static_assert(dependent_false_v<T>, "Unsupported type");
        return -1;
    }
}

template<typename T>
MinMaxLoc<T>::MinMaxLoc(NVCVTensorRequirements *tensorRequirements)
    : m_tensorRequirements{*tensorRequirements}
    , m_dataType(tensorRequirements->dtype)
    , m_srcOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstMinValOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstMaxValOp(cupva::OffsetPointer<T>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstMinCntOp(cupva::OffsetPointer<int>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstMaxCntOp(cupva::OffsetPointer<int>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstMinLocOp(cupva::OffsetPointer<unsigned short>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstMaxLocOp(cupva::OffsetPointer<unsigned short>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    // Get dimensions for input tensor
    int64_t tensorShapeNCHW[4];
    int64_t tensorStridesNCHW[4];
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->shape, NVCV_TENSOR_NCHW, tensorShapeNCHW);
    nvcvTensorShapePermute(tensorRequirements->layout, tensorRequirements->strides, NVCV_TENSOR_NCHW,
                           tensorStridesNCHW);

    int32_t batch     = tensorShapeNCHW[0];
    int32_t channels  = tensorShapeNCHW[1];
    int32_t height    = tensorShapeNCHW[2];
    int32_t width     = tensorShapeNCHW[3];
    int32_t linePitch = tensorStridesNCHW[2] / (nvcv::DataType(tensorRequirements->dtype).bitsPerPixel() / 8);

    // Validate layout
    if (tensorRequirements->layout != nvcv::TENSOR_NHWC && tensorRequirements->layout != nvcv::TENSOR_NCHW &&
        tensorRequirements->layout != nvcv::TENSOR_HWC && tensorRequirements->layout != nvcv::TENSOR_CHW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor must have NHWC, NCHW, HWC or CHW layout");
    }

    // Validate dimensions
    if ((batch != 0 && batch != 1) || (channels != 0 && channels != 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have N=1, C=1");
    }

    // Set image dimensions and line pitch
    m_imageHeight = height;
    m_imageWidth  = width;

    m_exec    = GetMinMaxLocExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    unsigned short tile_width  = MM_TILE_WIDTH;
    unsigned short tile_height = MM_TILE_HEIGHT;
    // last column: RDF BPE padding
    int has_last_column = (m_imageWidth % tile_width) > 0;
    int ntile_w         = (m_imageWidth + tile_width - 1) / tile_width;

    // last row: height direction use shorter height, do not unroll if niter not qualified
    int has_last_row = (m_imageHeight % tile_height) > 0;
    int ntile_h      = (m_imageHeight + tile_height - 1) / tile_height;

    prog["ntile_w"]         = ntile_w;
    prog["ntile_h"]         = ntile_h;
    prog["tile_width"]      = tile_width;
    prog["tile_height"]     = tile_height;
    prog["image_width"]     = m_imageWidth;
    prog["has_last_column"] = has_last_column;
    prog["has_last_row"]    = has_last_row;

    unsigned short last_row_tile_height = m_imageHeight % tile_height;
    prog["last_row_tile_height"]        = last_row_tile_height;

    // if last row has shorter tile height, disable unroll in location kernels if agen niters fail to satisfy unroll conditions
    int loc_stride                 = getDataTypeLocStride<T>();
    int loc_agen_niter             = tile_width * last_row_tile_height / loc_stride;
    int last_row_loc_no_unroll     = has_last_row && (loc_agen_niter % 4 || loc_agen_niter < 24);
    prog["last_row_loc_no_unroll"] = last_row_loc_no_unroll;

    // src data flow
    auto srcRDFHdl = prog["srcRDFHdl"];
    T *srcBuf      = prog["srcBuf"].ptr<T>();

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>();
    srcDataFlow.handler(srcRDFHdl)
        .src(m_srcOp, m_imageWidth, m_imageHeight, linePitch)
        .tileBuffer(srcBuf)
        .tile(tile_width, tile_height)
        .roi(0, 0, ntile_w * tile_width, m_imageHeight)
        .halo(0, PAD_BPE);

    // dst values and counts data flow
    T *dstMinValBuf   = prog["dstMinValBuf"].ptr<T>();
    T *dstMaxValBuf   = prog["dstMaxValBuf"].ptr<T>();
    int *dstMinCntBuf = prog["dstMinCntBuf"].ptr<int>();
    int *dstMaxCntBuf = prog["dstMaxCntBuf"].ptr<int>();

    auto dstValCntSQDFHdl               = prog["dstValCntSQDFHdl"];
    SequenceDataFlow &dstValCntDataFlow = prog.addDataFlowHead<SequenceDataFlow>(1).handler(dstValCntSQDFHdl);
    dstValCntDataFlow.addTransfer().src(dstMinValBuf, sizeof(T)).dst(m_dstMinValOp, sizeof(T)).tile(sizeof(T), 1);
    dstValCntDataFlow.addTransfer().src(dstMaxValBuf, sizeof(T)).dst(m_dstMaxValOp, sizeof(T)).tile(sizeof(T), 1);
    dstValCntDataFlow.addTransfer().src(dstMinCntBuf, sizeof(int)).dst(m_dstMinCntOp, sizeof(int)).tile(sizeof(int), 1);
    dstValCntDataFlow.addTransfer().src(dstMaxCntBuf, sizeof(int)).dst(m_dstMaxCntOp, sizeof(int)).tile(sizeof(int), 1);

    // dst locations data flow
    int pos_tile_size     = USHORT_LOC_BUF_SIZE * 2;
    prog["pos_tile_size"] = pos_tile_size;

    unsigned short *dstMaxLocBuf = prog["dstMaxLocBuf"].ptr<unsigned short>();
    unsigned short *dstMinLocBuf = prog["dstMinLocBuf"].ptr<unsigned short>();

    int pos_tile_bytes = pos_tile_size * sizeof(uint16_t);
    // capacity is set in submit stage, init to 1
    // SQDF tile size is updated in each trigger, init to 1 for bounds checking
    int dst_bytes = sizeof(uint16_t);

    auto dstMinLocSQDFHdl               = prog["dstMinLocSQDFHdl"];
    SequenceDataFlow &dstMinLocDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    dstMinLocDataFlow.handler(dstMinLocSQDFHdl)
        .addTransfer()
        .src(dstMinLocBuf, pos_tile_bytes)
        .dst(m_dstMinLocOp, dst_bytes)
        .tile(dst_bytes, 1);

    auto dstMaxLocSQDFHdl               = prog["dstMaxLocSQDFHdl"];
    SequenceDataFlow &dstMaxLocDataFlow = prog.addDataFlowHead<SequenceDataFlow>();
    dstMaxLocDataFlow.handler(dstMaxLocSQDFHdl)
        .addTransfer()
        .src(dstMaxLocBuf, pos_tile_bytes)
        .dst(m_dstMaxLocOp, dst_bytes)
        .tile(dst_bytes, 1);

    prog.compileDataFlows();

    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename T>
template<typename S>
void MinMaxLoc<T>::operator()(S stream, const nvcv::Tensor &src_t, const nvcv::Tensor &dstMinVal_t,
                              const nvcv::Tensor &dstMinLoc_t, const nvcv::Tensor &dstMinCnt_t,
                              const nvcv::Tensor &dstMaxVal_t, const nvcv::Tensor &dstMaxLoc_t,
                              const nvcv::Tensor &dstMaxCnt_t)
{
    auto srcTensorData       = src_t.exportData<nvcv::TensorDataStrided>();
    auto dstMinValTensorData = dstMinVal_t.exportData<nvcv::TensorDataStrided>();
    auto dstMaxValTensorData = dstMaxVal_t.exportData<nvcv::TensorDataStrided>();
    auto dstMinCntTensorData = dstMinCnt_t.exportData<nvcv::TensorDataStrided>();
    auto dstMaxCntTensorData = dstMaxCnt_t.exportData<nvcv::TensorDataStrided>();
    auto dstMinLocTensorData = dstMinLoc_t.exportData<nvcv::TensorDataStrided>();
    auto dstMaxLocTensorData = dstMaxLoc_t.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, srcTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input image tensor must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(srcTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(dstMinValTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(dstMaxValTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(dstMinCntTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(dstMaxCntTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(dstMinLocTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(dstMaxLocTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    // location tensors are required to have rank 1 and layout W
    if (dstMinLoc_t.rank() != 1 || dstMaxLoc_t.rank() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output location tensors must have rank 1!");
    }

    // set location capacity parameter by location tensor sizes
    // obtain tensor sizes of location buffers, and use the smaller one as capacity
    int32_t loc_capacity = std::min(dstMinLoc_t.shape()[0], dstMaxLoc_t.shape()[0]);
    // location capacity should be no more than image pixel number
    loc_capacity = std::min(loc_capacity, m_imageWidth * m_imageHeight);

    cupvaParameter_t param;
    CupvaCmdProgramGetParameter(&m_prog, &param, "loc_capacity");
    CupvaParameterSetValueScalar(&param, &loc_capacity, sizeof(int32_t));

    m_srcOp.setBase(static_cast<T *>(static_cast<void *>(srcTensorData->basePtr())));
    m_dstMinValOp.setBase(static_cast<T *>(static_cast<void *>(dstMinValTensorData->basePtr())));
    m_dstMaxValOp.setBase(static_cast<T *>(static_cast<void *>(dstMaxValTensorData->basePtr())));
    m_dstMinCntOp.setBase(static_cast<int *>(static_cast<void *>(dstMinCntTensorData->basePtr())));
    m_dstMaxCntOp.setBase(static_cast<int *>(static_cast<void *>(dstMaxCntTensorData->basePtr())));
    m_dstMinLocOp.setBase(static_cast<unsigned short *>(static_cast<void *>(dstMinLocTensorData->basePtr())));
    m_dstMaxLocOp.setBase(static_cast<unsigned short *>(static_cast<void *>(dstMaxLocTensorData->basePtr())));

    // set external memory pointers for SQDF
    cupvaParameter_t dstMinLocExtPtr, dstMaxLocExtPtr;
    CupvaCmdProgramGetParameter(&m_prog, &dstMinLocExtPtr, "dstMinLocExtPtr");
    CupvaCmdProgramGetParameter(&m_prog, &dstMaxLocExtPtr, "dstMaxLocExtPtr");

    CupvaParameterSetValuePointer(&dstMinLocExtPtr, static_cast<void *>(dstMinLocTensorData->basePtr()));
    CupvaParameterSetValuePointer(&dstMaxLocExtPtr, static_cast<void *>(dstMaxLocTensorData->basePtr()));

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
        static_assert(!sizeof(S), "Unsupported stream type for MinMaxLoc operator");
    }
}

// Explicit template instantiation
template void MinMaxLoc<uint8_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &);
template void MinMaxLoc<uint8_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &);
template void MinMaxLoc<int8_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &);
template void MinMaxLoc<int8_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                          const nvcv::Tensor &, const nvcv::Tensor &,
                                                          const nvcv::Tensor &, const nvcv::Tensor &,
                                                          const nvcv::Tensor &);
template void MinMaxLoc<uint16_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                             const nvcv::Tensor &, const nvcv::Tensor &,
                                                             const nvcv::Tensor &, const nvcv::Tensor &,
                                                             const nvcv::Tensor &);
template void MinMaxLoc<uint16_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &);
template void MinMaxLoc<int16_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &);
template void MinMaxLoc<int16_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &);
template void MinMaxLoc<uint32_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                             const nvcv::Tensor &, const nvcv::Tensor &,
                                                             const nvcv::Tensor &, const nvcv::Tensor &,
                                                             const nvcv::Tensor &);
template void MinMaxLoc<uint32_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &);
template void MinMaxLoc<int32_t>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &, const nvcv::Tensor &,
                                                            const nvcv::Tensor &);
template void MinMaxLoc<int32_t>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &, const nvcv::Tensor &,
                                                           const nvcv::Tensor &);
template void MinMaxLoc<float>::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                          const nvcv::Tensor &, const nvcv::Tensor &,
                                                          const nvcv::Tensor &, const nvcv::Tensor &,
                                                          const nvcv::Tensor &);
template void MinMaxLoc<float>::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &,
                                                         const nvcv::Tensor &, const nvcv::Tensor &,
                                                         const nvcv::Tensor &, const nvcv::Tensor &,
                                                         const nvcv::Tensor &);

template<typename T>
MinMaxLoc<T>::~MinMaxLoc()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class MinMaxLoc<uint8_t>;
template class MinMaxLoc<int8_t>;
template class MinMaxLoc<uint16_t>;
template class MinMaxLoc<int16_t>;
template class MinMaxLoc<uint32_t>;
template class MinMaxLoc<int32_t>;
template class MinMaxLoc<float>;

} // namespace pva_solutions::priv

NVCVStatus pvaMinMaxLocCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        switch (tensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U8:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MinMaxLoc<uint8_t>(tensorRequirements));
            break;
        case NVCV_DATA_TYPE_S8:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MinMaxLoc<int8_t>(tensorRequirements));
            break;
        case NVCV_DATA_TYPE_U16:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MinMaxLoc<uint16_t>(tensorRequirements));
            break;
        case NVCV_DATA_TYPE_S16:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MinMaxLoc<int16_t>(tensorRequirements));
            break;
        case NVCV_DATA_TYPE_U32:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MinMaxLoc<uint32_t>(tensorRequirements));
            break;
        case NVCV_DATA_TYPE_S32:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MinMaxLoc<int32_t>(tensorRequirements));
            break;
        case NVCV_DATA_TYPE_F32:
            *handle =
                reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::MinMaxLoc<float>(tensorRequirements));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type!");
        }
    });
}

template<typename StreamType>
NVCVStatus MinMaxLocSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle src_t,
                               NVCVTensorHandle dstMinVal_t, NVCVTensorHandle dstMinLoc_t, NVCVTensorHandle dstMinCnt_t,
                               NVCVTensorHandle dstMaxVal_t, NVCVTensorHandle dstMaxLoc_t, NVCVTensorHandle dstMaxCnt_t)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle src(src_t);
        nvcv::TensorWrapHandle dstMinVal(dstMinVal_t), dstMinLoc(dstMinLoc_t), dstMinCnt(dstMinCnt_t);
        nvcv::TensorWrapHandle dstMaxVal(dstMaxVal_t), dstMaxLoc(dstMaxLoc_t), dstMaxCnt(dstMaxCnt_t);

        NVCVDataType inDataType;
        nvcvTensorGetDataType(src_t, &inDataType);

        switch (inDataType)
        {
        case NVCV_DATA_TYPE_U8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MinMaxLoc<uint8_t>>(handle)(
                stream, src, dstMinVal, dstMinLoc, dstMinCnt, dstMaxVal, dstMaxLoc, dstMaxCnt);
            break;
        case NVCV_DATA_TYPE_S8:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MinMaxLoc<int8_t>>(handle)(
                stream, src, dstMinVal, dstMinLoc, dstMinCnt, dstMaxVal, dstMaxLoc, dstMaxCnt);
            break;
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MinMaxLoc<uint16_t>>(handle)(
                stream, src, dstMinVal, dstMinLoc, dstMinCnt, dstMaxVal, dstMaxLoc, dstMaxCnt);
            break;
        case NVCV_DATA_TYPE_S16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MinMaxLoc<int16_t>>(handle)(
                stream, src, dstMinVal, dstMinLoc, dstMinCnt, dstMaxVal, dstMaxLoc, dstMaxCnt);
            break;
        case NVCV_DATA_TYPE_U32:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MinMaxLoc<uint32_t>>(handle)(
                stream, src, dstMinVal, dstMinLoc, dstMinCnt, dstMaxVal, dstMaxLoc, dstMaxCnt);
            break;
        case NVCV_DATA_TYPE_S32:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MinMaxLoc<int32_t>>(handle)(
                stream, src, dstMinVal, dstMinLoc, dstMinCnt, dstMaxVal, dstMaxLoc, dstMaxCnt);
            break;
        case NVCV_DATA_TYPE_F32:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::MinMaxLoc<float>>(handle)(
                stream, src, dstMinVal, dstMinLoc, dstMinCnt, dstMaxVal, dstMaxLoc, dstMaxCnt);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type!");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus MinMaxLocSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                       NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                       NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus MinMaxLocSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                      NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle,
                                                      NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaMinMaxLocSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle src_t,
                              NVCVTensorHandle dstMinVal_t, NVCVTensorHandle dstMinLoc_t, NVCVTensorHandle dstMinCnt_t,
                              NVCVTensorHandle dstMaxVal_t, NVCVTensorHandle dstMaxLoc_t, NVCVTensorHandle dstMaxCnt_t)
{
    return MinMaxLocSubmitImpl(handle, stream, src_t, dstMinVal_t, dstMinLoc_t, dstMinCnt_t, dstMaxVal_t, dstMaxLoc_t,
                               dstMaxCnt_t);
}

NVCVStatus pvaMinMaxLocSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle src_t,
                              NVCVTensorHandle dstMinVal_t, NVCVTensorHandle dstMinLoc_t, NVCVTensorHandle dstMinCnt_t,
                              NVCVTensorHandle dstMaxVal_t, NVCVTensorHandle dstMaxLoc_t, NVCVTensorHandle dstMaxCnt_t)
{
    return MinMaxLocSubmitImpl(handle, stream, src_t, dstMinVal_t, dstMinLoc_t, dstMinCnt_t, dstMaxVal_t, dstMaxLoc_t,
                               dstMaxCnt_t);
}
