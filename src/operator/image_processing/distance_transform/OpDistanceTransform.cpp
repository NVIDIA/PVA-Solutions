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

#include "OpDistanceTransform.hpp"

#include "device/distance_transform_dev_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_common.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDistanceTransform.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>
#include <cstdint>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_distance_transform_U16_dev)

namespace pva_solutions::priv {

std::mutex DistanceTransform::s_execMapMutex;
std::map<DistanceTransform::ExecKey, std::weak_ptr<cupva::Executable>>
    DistanceTransform::s_distanceTransformExecutables;

std::shared_ptr<cupva::Executable> DistanceTransform::GetDistanceTransformExec()
{
    static IOperator::ExecutableData sDistanceTransformExecData = {
        CUPVA_EXECUTABLE_DATA(pva_distance_transform_U16_dev), CUPVA_EXECUTABLE_SIZE(pva_distance_transform_U16_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return GetExecutable(sDistanceTransformExecData, execKey, s_distanceTransformExecutables, s_execMapMutex);
}

DistanceTransform::DistanceTransform(NVCVTensorRequirements *tensorRequirements, PVADistanceType distanceType)
    : m_tensorRequirements{*tensorRequirements}
    , m_distType(distanceType)
    , m_shape(tensorRequirements->shape, tensorRequirements->rank, tensorRequirements->layout)
    , m_imgOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_distOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_labelOp(cupva::OffsetPointer<uint16_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    if (m_distType != PVADistanceType::PVA_DIST_L1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_IMPLEMENTED, "Only supports L1 distance");
    }
    if (!(m_shape.layout() == nvcv::TENSOR_HWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT, "Tensor must have either HWC layout");
    }
    if (!(m_shape[2] == 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT, "Tensor must have 1 channel");
    }
    // Make sure the image size is not over the max size supported by this op due to the vmem capacity limitation
    // and internal uint16_t-typed-math dynamic range
    if (!(m_shape[0] <= kMaxImageHeight && m_shape[1] <= kMaxImageWidth))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Tensor size is too large to be supported");
    }

    if (!(m_shape[0] >= kMinImageHeight && m_shape[1] >= kMinImageWidth))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor size is too small to be supported");
    }

    m_height = m_shape[0];
    m_width  = m_shape[1];

    // The default max distance is the diagonal distance of the image in UQ13.3 format
    m_maxDist = (m_height + m_width) << 3;

    // The horizontal distance is a global scratch pad buffer needed by distance transform operator
    // The buffer is owned by the operator and will be freed when the operator is destroyed
    m_hd = reinterpret_cast<uint16_t *>(mem::Alloc(m_height * m_width * sizeof(uint16_t)));

    m_exec    = GetDistanceTransformExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    auto readHdl  = prog["read_hdl"];
    auto writeHdl = prog["write_hdl"];

    auto read_trigger   = prog["read_trigger"];
    auto write_trigger0 = prog["write_trigger0"];
    auto write_trigger1 = prog["write_trigger1"];

    uint16_t *evenTile = prog["buf_b"].ptr<uint16_t>();
    uint16_t *oddTile  = prog["buf_c"].ptr<uint16_t>();

    uint16_t *hd_in = prog["hd_in"].ptr<uint16_t>();
    uint16_t *dist  = prog["dist"].ptr<uint16_t>();
    uint16_t *label = prog["label"].ptr<uint16_t>();

    prog["iw"] = m_width;
    prog["ih"] = m_height;

    // the second stage splits the process into 2 even halves
    // round up the height to even during the processing and
    // permute off the latest row written into dram
    int32_t heightRoundToEven = (m_height + 1) & ~1;
    prog["height_rounded"]    = heightRoundToEven;

    // the first stage loop needs to be unroll by 4
    int32_t widthRoundTo4 = ((m_width - 1) / 4 + 1) * 4;
    prog["width_rounded"] = widthRoundTo4;

    // round up the m_width to 32k + 1 for the transposed vector load/store
    prog["transpose_lp"] = ((widthRoundTo4 - 1) / TW + 1) * TW + 1;

    // round up the image width and height to multiple of TW and TH
    // to get the number of iterations
    prog["n_tile_row"] = ((m_height - 1) / TH) + 1;
    prog["n_tile_col"] = ((m_width - 1) / TW) + 1;

#include "device/distance_transform_sqrt_table.h"
    prog["sqrt_tbl"].set(sqrt_tbl, 65 * 32);

    // Stage 1 dataflows
    prog.addDataFlowHead<GatherScatterDataFlow>(1)
        .handler(readHdl)
        .numTilesPerTrigger(1)
        .src(m_imgOp, m_width, m_height, m_width)
        .tile(widthRoundTo4, TH)
        .padVal(PAD_CONST, 0xffff);

    prog.addDataFlowHead<GatherScatterDataFlow>(1)
        .handler(writeHdl)
        .numTilesPerTrigger(1)
        .dst(m_hd, m_width, m_height, m_width)
        .tile(m_width, TH);

    // Stage 2 dataflows
    prog.addDataFlowHead<RasterDataFlow>(3)
        .handler(read_trigger)
        .src(m_hd, m_width, m_height, m_width)
        .tileBuffer(hd_in)
        .tile(TW, m_height)
        .halo(0, PAD_CONST, m_width);

    prog.addDataFlowHead<RasterDataFlow>(3)
        .handler(write_trigger0)
        .dst(m_distOp, m_width, m_height, m_width)
        .tileBuffer(dist)
        .tile(TW, heightRoundToEven / 2)
        .scanOrder(COLUMN_MAJOR);

    prog.addDataFlowHead<RasterDataFlow>(3)
        .handler(write_trigger1)
        .dst(m_labelOp, m_width, m_height, m_width)
        .tileBuffer(label)
        .tile(TW, heightRoundToEven / 2)
        .scanOrder(COLUMN_MAJOR);

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void DistanceTransform::operator()(S stream, const nvcv::Tensor &inImage, uint16_t maxDistance,
                                   const nvcv::Tensor &outDistance, const nvcv::Tensor &outLabel)
{
    auto imgTensorData   = inImage.exportData<nvcv::TensorDataStrided>();
    auto distTensorData  = outDistance.exportData<nvcv::TensorDataStrided>();
    auto labelTensorData = outLabel.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, imgTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, distTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_tensorRequirements, labelTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(imgTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(distTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(labelTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_imgOp.setBase(static_cast<uint16_t *>(static_cast<void *>(imgTensorData->basePtr())));
    m_distOp.setBase(static_cast<uint16_t *>(static_cast<void *>(distTensorData->basePtr())));
    m_labelOp.setBase(static_cast<uint16_t *>(static_cast<void *>(labelTensorData->basePtr())));

    cupvaError_t err = CUPVA_ERROR_NONE;

    // Update max distance parameter in program if the max distance value is non-zero
    if (maxDistance != 0)
    {
        cupvaParameter_t param;
        err = CupvaCmdProgramGetParameter(&m_prog, &param, "max_dist");
        if (err != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to get maxDistance parameter");
        }
        err = CupvaParameterSetValueScalar(&param, &maxDistance, sizeof(maxDistance));
        if (err != CUPVA_ERROR_NONE)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to set maxDistance parameter value");
        }
    }

    // Update the offset pointers in the program
    err = CupvaCmdProgramUpdateDataFlows(&m_prog);

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
        static_assert(!sizeof(S), "Unsupported stream type for DistanceTransform operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

DistanceTransform::~DistanceTransform()
{
    mem::Free(m_hd);
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaDistanceTransformCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements *tensorRequirements,
                                      PVADistanceType distanceType)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }

        if (tensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to TensorRequirements must not be NULL");
        }

        switch (tensorRequirements->dtype)
        {
        case NVCV_DATA_TYPE_U16:
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::DistanceTransform(tensorRequirements, distanceType));
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_NOT_IMPLEMENTED, "Unsupported data type");
        }
    });
}

template<typename StreamType>
NVCVStatus DistanceTransformSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle inImage,
                                       uint16_t maxDistance, NVCVTensorHandle outDistance, NVCVTensorHandle outLabel)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle imgHandle(inImage), distHandle(outDistance), labelHandle(outLabel);

        NVCVDataType imgDataType, distDataType, labelDataType;
        nvcvTensorGetDataType(inImage, &imgDataType);
        nvcvTensorGetDataType(outDistance, &distDataType);
        nvcvTensorGetDataType(outLabel, &labelDataType);

        if (!(imgDataType == distDataType && imgDataType == labelDataType))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and outputs data type mismatch");
        }

        switch (imgDataType)
        {
        case NVCV_DATA_TYPE_U16:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DistanceTransform>(handle)(
                stream, imgHandle, maxDistance, distHandle, labelHandle);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_NOT_IMPLEMENTED, "Unsupported data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus DistanceTransformSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                               uint16_t, NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus DistanceTransformSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                              uint16_t, NVCVTensorHandle, NVCVTensorHandle);

// C++ overload implementations
NVCVStatus pvaDistanceTransformSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle inImage,
                                      uint16_t maxDistance, NVCVTensorHandle outDistance, NVCVTensorHandle outLabel)
{
    return DistanceTransformSubmitImpl(handle, stream, inImage, maxDistance, outDistance, outLabel);
}

NVCVStatus pvaDistanceTransformSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle inImage,
                                      uint16_t maxDistance, NVCVTensorHandle outDistance, NVCVTensorHandle outLabel)
{
    return DistanceTransformSubmitImpl(handle, stream, inImage, maxDistance, outDistance, outLabel);
}