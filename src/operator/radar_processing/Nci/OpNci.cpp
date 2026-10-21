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

#include "OpNci.hpp"

#include "device/nci_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpNci.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cmath>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_nci_dev);

namespace pva_solutions::priv {

template<typename TI, typename TO>
std::mutex Nci<TI, TO>::s_execMapMutex;

template<typename TI, typename TO>
std::map<typename Nci<TI, TO>::ExecKey, std::weak_ptr<cupva::Executable>> Nci<TI, TO>::s_nciExecutables;

template<typename TI, typename TO>
std::shared_ptr<cupva::Executable> Nci<TI, TO>::GetNciExec()
{
    static IOperator::ExecutableData sNciExecData;

    sNciExecData = {CUPVA_EXECUTABLE_DATA(pva_nci_dev), CUPVA_EXECUTABLE_SIZE(pva_nci_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sNciExecData, execKey, s_nciExecutables, s_execMapMutex);
}

template<typename TI, typename TO>
bool Nci<TI, TO>::isNumChannelsValid(int32_t numChannels) const
{
    return ((2 == numChannels) || (4 == numChannels) || (8 == numChannels) || (16 == numChannels) ||
            (32 == numChannels) || (64 == numChannels));
}

template<typename TI, typename TO>
bool Nci<TI, TO>::isNumSamplesValid(int32_t numSamples, int32_t numChannels, int32_t numRangeBins,
                                    int32_t NrPerTile) const
{
    return ((0 == (numSamples % 8)) && ((numSamples * NrPerTile) <= MaxNciRxSize));
}

template<typename TI, typename TO>
bool Nci<TI, TO>::isNumRangeBinsValid(int32_t numRangeBins, int32_t NrPerTile) const
{
    return ((numRangeBins >= NrPerTile) && (numRangeBins <= MaxNoiseEstimateSize));
}

template<typename TI, typename TO>
bool Nci<TI, TO>::isNumDopplerFoldsValid(int32_t numSamples) const
{
    return ((m_numDopplerFolds >= 4) && (0 == (m_numDopplerFolds % 4)) &&
            (((numSamples / m_numDopplerFolds) % 32) == 0) && ((numSamples / m_numDopplerFolds) >= 32));
}

template<typename TI, typename TO>
std::pair<int32_t, int32_t> Nci<TI, TO>::getTileSize(int32_t numChannels, int32_t inWidth, int32_t inHeight) const
{
    const int32_t targetTileSize = MaxInputTileSize * sizeof(TI);
    const int32_t perRowSize     = numChannels * inWidth * 2 * sizeof(TI);
    int32_t NrPerTile            = targetTileSize / perRowSize;

    if (0 == NrPerTile)
    {
        NrPerTile = 1;
    }
    if ((inHeight / NrPerTile) > 256)
    {
        NrPerTile = inHeight / 256;
    }

    while (inHeight % NrPerTile != 0)
    {
        ++NrPerTile;
    }

    int32_t NsPerTile = targetTileSize / (numChannels * 2 * sizeof(TI) * NrPerTile);

    if (NsPerTile > 0)
    {
        while (inWidth % NsPerTile != 0 && NsPerTile > 1)
        {
            --NsPerTile;
        }
    }

    return {NrPerTile, NsPerTile};
}

template<typename TI, typename TO>
Nci<TI, TO>::Nci(NVCVTensorRequirements const *const inTensorRequirements,
                 NVCVTensorRequirements const *const outNciRxTensorRequirements,
                 NVCVTensorRequirements const *const outNciFinalTensorRequirements,
                 NVCVTensorRequirements const *const outNoiseEstimateTensorRequirements, int32_t repeatFold,
                 uint32_t noiseEstimateEnabled, PVAFFTLayout inputLayout)
    : m_inTensorRequirements{*inTensorRequirements}
    , m_outNciRxTensorRequirements{*outNciRxTensorRequirements}
    , m_outNciFinalTensorRequirements{*outNciFinalTensorRequirements}
    , m_outNoiseEstimateTensorRequirements{noiseEstimateEnabled ? *outNoiseEstimateTensorRequirements
                                                                : NVCVTensorRequirements{}}
    , m_fft2dOp(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr))
    , m_NciRxOp(cupva::OffsetPointer<TO>::Create(cupva::MemType::DRAM, nullptr))
    , m_NciFinalOp(cupva::OffsetPointer<TO>::Create(cupva::MemType::DRAM, nullptr))
    , m_noiseEstimateOp(cupva::OffsetPointer<TO>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
    , m_numDopplerFolds(repeatFold)
    , m_noiseEstimateEnabled(noiseEstimateEnabled)
    , m_inputLayout(inputLayout)
{
    if (m_inputLayout != PVA_FFT_LAYOUT_RANGE_RX_DOPPLER && m_inputLayout != PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported input layout %d; NCI supports only PVA_FFT_LAYOUT_RANGE_RX_DOPPLER (%d) "
                              "and PVA_FFT_LAYOUT_RANGE_DOPPLER_RX (%d)",
                              static_cast<int32_t>(m_inputLayout),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_RX_DOPPLER),
                              static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_DOPPLER_RX));
    }

    int64_t inTensorShapeNCHW[4];
    nvcvTensorShapePermute(inTensorRequirements->layout, inTensorRequirements->shape, NVCV_TENSOR_NCHW,
                           inTensorShapeNCHW);
    int32_t inHeight    = inTensorShapeNCHW[2];
    int32_t numChannels = 0;
    int32_t inWidth     = 0;

    if (m_inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        inWidth     = inTensorShapeNCHW[1];
        numChannels = inTensorShapeNCHW[3];
    }
    else
    {
        numChannels = inTensorShapeNCHW[1];
        inWidth     = inTensorShapeNCHW[3];
    }

    const auto [NrPerTile, NsPerTile] = getTileSize(numChannels, inWidth, inHeight);

    /* Check for supported values of Nr, NofRx and Ns and repeatFold */
    if (false == isNumChannelsValid(numChannels))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported value of NofRx: %d", numChannels);
    }
    if (false == isNumSamplesValid(inWidth, numChannels, inHeight, NrPerTile))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported value of Ns: %d", inWidth);
    }
    if (false == isNumRangeBinsValid(inHeight, NrPerTile))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported value of Nr: %d", inHeight);
    }
    if (false == isNumDopplerFoldsValid(inWidth))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported value of repeatFold: %d", repeatFold);
    }

    m_exec    = GetNciExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    int32_t phase = 0;

    RasterDataFlow &srcDataFlow = prog.addDataFlowHead<RasterDataFlow>(phase);
    auto srcDataFlowHandler     = prog["srcDataFlowHandler"];
    int32_t *ft2d               = prog["ft2d"].ptr<int32_t>();

    prog["NrPerTile"]              = NrPerTile;
    prog["NsPerTile"]              = NsPerTile;
    prog["Nr"]                     = inHeight;
    prog["Ns"]                     = inWidth;
    prog["NofRx"]                  = numChannels;
    prog["numDopplerFolds"]        = m_numDopplerFolds;
    prog["noiseEstimateEnabled"]   = (true == m_noiseEstimateEnabled) ? 1u : 0u;
    prog["isRangeDopplerRxLayout"] = m_inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX ? 1u : 0u;

    if (m_inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        srcDataFlow.handler(srcDataFlowHandler)
            .src(m_fft2dOp, inWidth * 2, numChannels * inHeight, inWidth * 2)
            .tileBuffer(ft2d)
            .tile(NsPerTile * 2, numChannels * NrPerTile)
            .transpose(TRANS_MODE_2);
    }
    else
    {
        srcDataFlow.handler(srcDataFlowHandler)
            .src(m_fft2dOp, inWidth * 2, numChannels * inHeight, inWidth * 2)
            .tileBuffer(ft2d)
            .tile(NsPerTile * 2, numChannels * NrPerTile);
    }

    RasterDataFlow &dstNciRxDataFlow    = prog.addDataFlowHead<RasterDataFlow>(phase);
    RasterDataFlow &dstNciFinalDataFlow = prog.addDataFlowHead<RasterDataFlow>(phase++);
    auto dstDataFlowHandler             = prog["dstDataFlowHandler"];
    uint32_t *nciRx                     = prog["nciRx"].ptr<uint32_t>();
    uint32_t *nciFinal                  = prog["nciFinal"].ptr<uint32_t>();

    dstNciRxDataFlow.handler(dstDataFlowHandler)
        .dst(m_NciRxOp, inWidth, inHeight, inWidth)
        .tileBuffer(nciRx)
        .tile(inWidth, NrPerTile);

    dstNciFinalDataFlow.handler(dstDataFlowHandler)
        .dst(m_NciFinalOp, inWidth / m_numDopplerFolds, inHeight, inWidth / m_numDopplerFolds)
        .tileBuffer(nciFinal)
        .tile(inWidth / m_numDopplerFolds, NrPerTile);

    if (true == m_noiseEstimateEnabled)
    {
        RasterDataFlow &dstNoiseEstimateDataFlow = prog.addDataFlowHead<RasterDataFlow>(phase);
        auto dstNoiseEstimateDataFlowHandler     = prog["dstNoiseEstimateDataFlowHandler"];
        uint32_t *noiseEstimate                  = prog["noiseEstimate"].ptr<uint32_t>();

        dstNoiseEstimateDataFlow.handler(dstNoiseEstimateDataFlowHandler)
            .dst(m_noiseEstimateOp, inHeight, 1, inHeight)
            .tileBuffer(noiseEstimate)
            .tile(inHeight, 1);
    }

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI, typename TO>
template<typename S>
void Nci<TI, TO>::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &outNciRx,
                             const nvcv::Tensor &outNciFinal, const nvcv::Tensor &outNoiseEstimate)
{
    auto inTensorData          = in.exportData<nvcv::TensorDataStrided>();
    auto outNciRxTensorData    = outNciRx.exportData<nvcv::TensorDataStrided>();
    auto outNciFinalTensorData = outNciFinal.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outNciRxTensorRequirements, outNciRxTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outNciFinalTensorRequirements,
                                                          outNciFinalTensorData->cdata()))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Input and output tensors must have the same requirements that were used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outNciRxTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outNciFinalTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_fft2dOp.setBase(static_cast<TI *>(static_cast<void *>(inTensorData->basePtr())));
    m_NciRxOp.setBase(static_cast<TO *>(static_cast<void *>(outNciRxTensorData->basePtr())));
    m_NciFinalOp.setBase(static_cast<TO *>(static_cast<void *>(outNciFinalTensorData->basePtr())));

    if (true == m_noiseEstimateEnabled)
    {
        auto outNoiseEstimateTensorData = outNoiseEstimate.exportData<nvcv::TensorDataStrided>();

        if (!pva_solutions::utils::AreTensorRequirementsEqual(m_outNoiseEstimateTensorRequirements,
                                                              outNoiseEstimateTensorData->cdata()))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Output noise estimate tensor must have the same requirements that were used to create the operator");
        }

        if (static_cast<void *>(outNoiseEstimateTensorData->basePtr()) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
        }

        m_noiseEstimateOp.setBase(static_cast<TO *>(static_cast<void *>(outNoiseEstimateTensorData->basePtr())));
    }

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
    }

    cupvaCmd_t const *submitCmds[1] = {&m_prog};
    cupvaCmdStatus_t cmdStatus[1]   = {nullptr};

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
        static_assert(!sizeof(S), "Unsupported stream type for Nci operator");
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename TI, typename TO>
bool Nci<TI, TO>::isNoiseEstimateEnabled() const
{
    return m_noiseEstimateEnabled;
}

template<typename TI, typename TO>
Nci<TI, TO>::~Nci()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class Nci<int32_t, uint32_t>;

} // namespace pva_solutions::priv

NVCVStatus pvaNciCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *inTensorRequirements,
                        NVCVTensorRequirements const **outTensorRequirements, PVARadarNCIParams *params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to input NVCVTensorRequirements must not be NULL");
        }
        if (outTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output NVCVTensorRequirements must not be NULL");
        }
        if (params == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to PVARadarNCIParams must not be NULL");
        }

        for (int i = 0; i < 2; i++)
        {
            if (outTensorRequirements[i] == nullptr)
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT,
                    "Pointer to the element %d of output NVCVTensorRequirements array must not be NULL", i);
            }
        }

        if (params->noiseEstimationEnabled && outTensorRequirements[2] == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to the noise estimate output NVCVTensorRequirements must not be NULL");
        }

        if ((inTensorRequirements->dtype == NVCV_DATA_TYPE_2S32) &&
            ((*outTensorRequirements[0]).dtype == NVCV_DATA_TYPE_U32) &&
            ((*outTensorRequirements[1]).dtype == NVCV_DATA_TYPE_U32) &&
            ((false == params->noiseEstimationEnabled) ||
             ((true == params->noiseEstimationEnabled) && ((*outTensorRequirements[2]).dtype == NVCV_DATA_TYPE_U32))))
        {
            *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::Nci<int32_t, uint32_t>(
                inTensorRequirements, outTensorRequirements[0], outTensorRequirements[1], outTensorRequirements[2],
                params->repeatFold, params->noiseEstimationEnabled, params->inputLayout));
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

template<typename StreamType>
NVCVStatus NciSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const *inHandles,
                         NVCVTensorHandle *outHandles)
{
    return nvcv::ProtectCall([&] {
        if (inHandles == nullptr || outHandles == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to input/output tensor handles must not be NULL");
        }

        nvcv::TensorWrapHandle input(inHandles[0]), outputNciRx(outHandles[0]), outputNciFinal(outHandles[1]),
            outputNoiseEstimate(NULL);

        NVCVDataType inDataType, outNciRxDataType, outNciFinalDataType;
        NVCVDataType outNoiseEstimateDataType = NVCV_DATA_TYPE_U32;

        nvcvTensorGetDataType(inHandles[0], &inDataType);
        nvcvTensorGetDataType(outHandles[0], &outNciRxDataType);
        nvcvTensorGetDataType(outHandles[1], &outNciFinalDataType);

        if (pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Nci<int32_t, uint32_t>>(handle)
                .isNoiseEstimateEnabled() == true)
        {
            outputNoiseEstimate = outHandles[2];
            nvcvTensorGetDataType(outHandles[2], &outNoiseEstimateDataType);
        }

        if ((inDataType == NVCV_DATA_TYPE_2S32) && (outNciRxDataType == NVCV_DATA_TYPE_U32) &&
            (outNciFinalDataType == NVCV_DATA_TYPE_U32) && (outNoiseEstimateDataType == NVCV_DATA_TYPE_U32))
        {
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::Nci<int32_t, uint32_t>>(handle)(
                stream, input, outputNciRx, outputNciFinal, outputNoiseEstimate);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus NciSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const *,
                                                 NVCVTensorHandle *);
template NVCVStatus NciSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const *,
                                                NVCVTensorHandle *);

// C++ overload implementations
NVCVStatus pvaNciSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const *inHandles,
                        NVCVTensorHandle *outHandles)
{
    return NciSubmitImpl(handle, stream, inHandles, outHandles);
}

NVCVStatus pvaNciSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const *inHandles,
                        NVCVTensorHandle *outHandles)
{
    return NciSubmitImpl(handle, stream, inHandles, outHandles);
}