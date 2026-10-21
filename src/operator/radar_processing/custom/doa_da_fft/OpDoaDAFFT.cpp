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

#include "OpDoaDAFFT.hpp"

#include "device/fft_params.h"
#include "doa_params.h"

#include <IOperator.hpp>
#include <RadarUtils.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpDoaDAFFT.h>
#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_doa_da_fft_dev);

namespace pva_solutions::priv {

template<typename TI, typename TO>
std::mutex DOA<TI, TO>::s_execMapMutex;

template<typename TI, typename TO>
std::map<typename DOA<TI, TO>::ExecKey, std::weak_ptr<cupva::Executable>> DOA<TI, TO>::s_DOAExecutables;

template<typename TI, typename TO>
std::shared_ptr<cupva::Executable> DOA<TI, TO>::GetDOAExec()
{
    static IOperator::ExecutableData sDOAExecData = {CUPVA_EXECUTABLE_DATA(pva_doa_da_fft_dev),
                                                     CUPVA_EXECUTABLE_SIZE(pva_doa_da_fft_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sDOAExecData, execKey, s_DOAExecutables, s_execMapMutex);
}

template<typename TI, typename TO>
DOA<TI, TO>::DOA(const NVCVTensorRequirements **inTensorRequirements,
                 const NVCVTensorRequirements *outTensorRequirements, int inTensorCount, const PVARadarGP *GP)
    : m_outTensorRequirements{*outTensorRequirements}
    , m_dstOp(cupva::OffsetPointer<TO>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
    , m_nciCpyCmd{}
{
    for (int i = 0; i < inTensorCount; i++)
    {
        m_inTensorRequirements.push_back(*inTensorRequirements[i]);
        m_srcOps.push_back(cupva::OffsetPointer<TI>::Create(cupva::MemType::DRAM, nullptr));
    }

    m_exec    = GetDOAExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    // Add SequenceDataFlow for calibvector, aindices, and peakIndices
    auto inSQDFHdl        = prog["InSqdfHdl"];
    auto inSnapIndicesHdl = prog["InSnapIndicesHdl"];
    auto inSnapDataHdl    = prog["InSnapDataHdl"];
    auto inNCIHdl         = prog["nciHdl"];
    auto targetHdl        = prog["targetHdl"];

    // FFT_NUM_POINTS is a power of two by construction; FactorizeFFTSize accepts [2, 1024].
    static_assert((FFT_NUM_POINTS >= 2) && (FFT_NUM_POINTS <= 1024), "FFT_NUM_POINTS must be in [2, 1024]");
    // Digit-reverse strides at the padded row, which keeps the transpose that reads it off one bank.
    const FFTFactorization fftFactorization =
        priv::FactorizeFFTSize(FFT_NUM_POINTS, false, false, 1, FFT_TILE_WIDTH_PAD);
    if (fftFactorization.isValid == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to factorize the %d-point angle FFT",
                              FFT_NUM_POINTS);
    }
    prog["fftFactorization"].set(&fftFactorization, 1);

    auto calibvectorV   = prog["calibvector"].ptr<int32_t>();
    auto peakSnapV      = prog["peakSnap"].ptr<int32_t>();
    auto peakIndicesV   = prog["peakIndices"].ptr<uint32_t>();
    auto nciFinalV      = prog["nciFinal"].ptr<uint32_t>();
    auto targetResultsV = prog["targetResults"].ptr<float>();
    auto numPeaksV      = prog["numPeaks"].ptr<uint32_t>();

    constexpr int32_t p0{1};
    constexpr int32_t p1{2};

    const NVCVTensorRequirements *peakIndicesTensorRequirements = inTensorRequirements[2];
    const NVCVTensorRequirements *peakSnapTensorRequirements    = inTensorRequirements[3];
    const NVCVTensorRequirements *nciFinalTensorRequirements    = inTensorRequirements[4];

    int64_t peakIndicesTensorShape[2];
    int64_t peakSnapTensorShape[2];
    int64_t nciFinalTensorShape[2];
    int64_t outTensorShape[2];

    nvcvTensorShapePermute(peakIndicesTensorRequirements->layout, peakIndicesTensorRequirements->shape, NVCV_TENSOR_HW,
                           peakIndicesTensorShape);
    nvcvTensorShapePermute(peakSnapTensorRequirements->layout, peakSnapTensorRequirements->shape, NVCV_TENSOR_HW,
                           peakSnapTensorShape);
    nvcvTensorShapePermute(nciFinalTensorRequirements->layout, nciFinalTensorRequirements->shape, NVCV_TENSOR_HW,
                           nciFinalTensorShape);
    nvcvTensorShapePermute(outTensorRequirements->layout, outTensorRequirements->shape, NVCV_TENSOR_HW, outTensorShape);

    int32_t peakIndicesWidth  = peakIndicesTensorShape[1];
    int32_t peakIndicesHeight = peakIndicesTensorShape[0];
    int32_t peakSnapWidth     = peakSnapTensorShape[1] * ELEMENT_NUM_OF_COMPLEX;
    int32_t peakSnapHeight    = peakSnapTensorShape[0];
    int32_t nciFinalWidth     = nciFinalTensorShape[1];
    int32_t nciFinalHeight    = nciFinalTensorShape[0];
    int32_t numTargets        = outTensorShape[1];
    int32_t targetHeight      = outTensorShape[0];

    assert(peakIndicesHeight == PEAK_INDICES_HEIGHT);
    assert(peakSnapWidth == PEAK_SNAP_WIDTH);
    assert(targetHeight == NUM_TARGET_ROWS);

    m_nciFinalWidth              = nciFinalWidth;
    m_nciFinalHeight             = nciFinalHeight;
    m_nciFinalWidthInBytes       = nciFinalWidth * sizeof(uint32_t);
    m_nciFinalPaddedWidth        = nciFinalWidth + 2;
    m_nciFinalPaddedHeight       = nciFinalHeight + 2;
    m_nciFinalPaddedWidthInBytes = m_nciFinalPaddedWidth * sizeof(uint32_t);

    m_nciFinalPaddedDev = (uint32_t *)cupva::mem::GetL2BaseAddress();

    SequenceDataFlow &inSQDFDataFlow = prog.addDataFlowHead<SequenceDataFlow>(p0).handler(inSQDFHdl);
    inSQDFDataFlow.addTransfer()
        .src(m_srcOps[0], sizeof(int32_t))
        .dst(numPeaksV, sizeof(uint32_t))
        .tile(sizeof(uint32_t), 1);
    inSQDFDataFlow.addTransfer()
        .src(m_srcOps[1], CALIB_VECTOR_WIDTH * sizeof(int32_t))
        .dst(calibvectorV, CALIB_VECTOR_WIDTH * sizeof(int32_t))
        .tile(CALIB_VECTOR_WIDTH * sizeof(int32_t), 1);

    SequenceDataFlow &inPeakIndicesDataFlow = prog.addDataFlowHead<SequenceDataFlow>(p1).handler(inSnapIndicesHdl);
    inPeakIndicesDataFlow.addTransfer()
        .src(m_srcOps[2], peakIndicesWidth * sizeof(uint32_t))
        .dst(peakIndicesV, PEAK_TILE_HEIGHT * sizeof(uint32_t))
        .tile(PEAK_TILE_HEIGHT * sizeof(uint32_t), PEAK_INDICES_HEIGHT);

    // Add RasterDataFlow for peak snap data transfer
    SequenceDataFlow &inPeakSnapDataFlow = prog.addDataFlowHead<SequenceDataFlow>(p1).handler(inSnapDataHdl);
    inPeakSnapDataFlow.addTransfer()
        .src(m_srcOps[3], peakSnapWidth * sizeof(int32_t))
        .dst(peakSnapV, PEAK_SNAP_WIDTH * sizeof(int32_t))
        .tile(PEAK_SNAP_WIDTH * sizeof(int32_t), PEAK_TILE_HEIGHT);

    // Add GatherScatterDataFlow for nci final data transfer
    GatherScatterDataFlow &inGSDFDataFlow = prog.addDataFlowHead<GatherScatterDataFlow>(p1);
    inGSDFDataFlow.handler(inNCIHdl)
        .numTilesPerTrigger(NUM_NCI_NODES)
        .src(m_nciFinalPaddedDev, nciFinalWidth + 2, nciFinalHeight + 2, nciFinalWidth + 2)
        .tile(NCI_NODE_WIDTH, NCI_NODE_HEIGHT);

    // Add RasterDataFlow for target results output
    SequenceDataFlow &outSQDF = prog.addDataFlowHead<SequenceDataFlow>(p1).handler(targetHdl);
    outSQDF.addTransfer()
        .src(targetResultsV, PEAK_TILE_HEIGHT * sizeof(float))
        .dst(m_dstOp, numTargets * sizeof(float))
        .tile(PEAK_TILE_HEIGHT * sizeof(float), NUM_TARGET_ROWS);

    // Validate GP parameters to prevent division by zero
    if (GP->PRI == 0.0f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "PRI cannot be zero");
    }
    if (GP->nofRamps == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "nofRamps cannot be zero");
    }

    PvaRadarGPPriv GPPriv{};
    GPPriv.nr               = GP->nr;
    GPPriv.repeatFold       = GP->repeatFold;
    GPPriv.nofSnap          = GP->nofSnap;
    GPPriv.nofAperture      = GP->nofAperture;
    GPPriv.nofAFFT          = GP->nofAFFT;
    GPPriv.dis              = GP->dis;
    GPPriv.fs               = GP->fs;
    GPPriv.nofSamples       = GP->nofSamples;
    GPPriv.PRI              = GP->PRI;
    GPPriv.nofRamps         = GP->nofRamps;
    GPPriv.deltaV           = GP->deltaV;
    GPPriv.disE             = GP->disE;
    GPPriv.contVFast        = GP->contVFast;
    GPPriv.contVSlow        = GP->contVSlow;
    GPPriv.contR            = GP->contR;
    GPPriv.numNCIRows       = nciFinalHeight;
    GPPriv.numNCICols       = nciFinalWidth;
    GPPriv.elevationFactor  = ((1 << 7) * 1.0f) / (360.0f * GP->disE);
    GPPriv.f_fast_delta     = GP->fs / GP->nofSamples;
    GPPriv.f_slow_delta     = 1.0f / (GP->PRI * GP->nofRamps);
    GPPriv.velocityAmbHigh  = -1.0f * GP->deltaV * static_cast<float>(GP->nofRamps) / 2.0f / 3.6f;
    GPPriv.velocityAmbLow   = GP->deltaV * static_cast<float>(GP->nofRamps) / 2.0f / 3.6f;
    GPPriv.calibVectorQbits = GP->calibVectorQbits;
    GPPriv.fftQbits         = GP->fftQbits;
    GPPriv.nciPaddedWidth   = m_nciFinalPaddedWidth;
    GPPriv.nciPaddedHeight  = m_nciFinalPaddedHeight;
    for (int i = 0; i < 5; i++)
    {
        GPPriv.tempVCoeff[i] = (1.0f / GP->PRI) * (static_cast<float>(i) - 3.0f) * GP->contVSlow;
    }

    for (int i = 0; i < 8; i++)
    {
        std::fill_n(&GPPriv.apUpper[i * 16], 16, GP->apUpper[i]);
        std::fill_n(&GPPriv.apLower[i * 16], 16, GP->apLower[i]);
        std::fill_n(&GPPriv.apUpperInd[i * 16], 16, GP->apUpperInd[i]);
        std::fill_n(&GPPriv.apLowerInd[i * 16], 16, GP->apLowerInd[i]);
    }

    prog["GP"].set(&GPPriv, 1);

    prog.compileDataFlows();
    prog.setL2Size(m_nciFinalPaddedWidthInBytes * m_nciFinalPaddedHeight);
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename TI, typename TO>
template<typename S>
void DOA<TI, TO>::operator()(S stream, NVCVTensorHandle *inHandles, NVCVTensorHandle outHandle, int32_t inTensorCount)
{
    if ((inTensorCount != m_inTensorRequirements.size()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The number of input tensor(s) must be equal to %d",
                              pva_solutions::priv::DOA<int32_t, float>::kDefaultTensorCount);
    }

    cupvaParameter_t indicesExtAddr;
    CupvaCmdProgramGetParameter(&m_prog, &indicesExtAddr, "indicesExtAddr");
    cupvaParameter_t snapExtAddr;
    CupvaCmdProgramGetParameter(&m_prog, &snapExtAddr, "snapExtAddr");
    void *inTensorBasePtrs[inTensorCount];

    for (int i = 0; i < inTensorCount; i++)
    {
        nvcv::Tensor input = nvcv::TensorWrapHandle(inHandles[i]);
        auto inTensorData  = input.exportData<nvcv::TensorDataStrided>();

        if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements[i], inTensorData->cdata()))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Input tensor %d must have the same requirements that was used to create the operator.", i);
        }
        if (static_cast<void *>(inTensorData->basePtr()) == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
        }

        inTensorBasePtrs[i] = inTensorData->basePtr();
        m_srcOps[i].setBase(static_cast<TI *>(static_cast<void *>(inTensorBasePtrs[i])));
    }

    CupvaParameterSetValuePointer(&indicesExtAddr, static_cast<void *>(inTensorBasePtrs[2]));
    CupvaParameterSetValuePointer(&snapExtAddr, static_cast<void *>(inTensorBasePtrs[3]));

    nvcv::Tensor output = nvcv::TensorWrapHandle(outHandle);
    auto outTensorData  = output.exportData<nvcv::TensorDataStrided>();
    cupvaParameter_t targetExtAddr;
    CupvaCmdProgramGetParameter(&m_prog, &targetExtAddr, "targetExtAddr");

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_outTensorRequirements, outTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "output tensor must have the same requirements that was used to create the operator");
    }
    if ((static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_dstOp.setBase(static_cast<TO *>(static_cast<void *>(outTensorData->basePtr())));
    CupvaParameterSetValuePointer(&targetExtAddr, static_cast<void *>(outTensorData->basePtr()));

    cupvaError_t err = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update data flows");
    }

    cupvaCmdMemcpyBufferParams_t nciCpyParams0 = {
        .ptrSrc = (uint8_t *)inTensorBasePtrs[4] + m_nciFinalWidthInBytes - 4,
        .ptrDst = m_nciFinalPaddedDev + m_nciFinalPaddedWidth,
    };
    cupvaCmdMemcpyBufferParams_t nciCpyParams1 = {
        .ptrSrc = (uint8_t *)inTensorBasePtrs[4],
        .ptrDst = m_nciFinalPaddedDev + m_nciFinalPaddedWidth + 1,
    };
    cupvaCmdMemcpyBufferParams_t nciCpyParams2 = {
        .ptrSrc = (uint8_t *)inTensorBasePtrs[4],
        .ptrDst = m_nciFinalPaddedDev + m_nciFinalPaddedWidth + m_nciFinalWidth + 1,
    };

    err = CupvaCmdMemcpyCreate2D(&m_nciCpyCmd[0], &nciCpyParams0, m_nciFinalPaddedWidthInBytes, m_nciFinalWidthInBytes,
                                 1 * sizeof(uint32_t), m_nciFinalHeight);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "PVA CopyCmd0 Submit failed");
    }
    err = CupvaCmdMemcpyCreate2D(&m_nciCpyCmd[1], &nciCpyParams1, m_nciFinalPaddedWidthInBytes, m_nciFinalWidthInBytes,
                                 m_nciFinalWidthInBytes, m_nciFinalHeight);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "PVA CopyCmd1 Submit failed");
    }
    err = CupvaCmdMemcpyCreate2D(&m_nciCpyCmd[2], &nciCpyParams2, m_nciFinalPaddedWidthInBytes, m_nciFinalWidthInBytes,
                                 1 * sizeof(uint32_t), m_nciFinalHeight);
    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "PVA CopyCmd2 Submit failed");
    }

    // Submit copy commands to copy nci final data to padded nci final data
    // and program to execute DOA
    cupvaCmd_t const *submitCmds[kNciCpyCmdCount + 1] = {&m_nciCpyCmd[0], &m_nciCpyCmd[1], &m_nciCpyCmd[2], &m_prog};

    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, submitCmds, nullptr, kNciCpyCmdCount + 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, submitCmds, nullptr, kNciCpyCmdCount + 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for DOA operator");
    }

    // Destroy CmdMemcpy commands
    for (int i = 0; i < kNciCpyCmdCount; i++)
    {
        CupvaCmdDestroy(&m_nciCpyCmd[i]);
    }

    if (!(err == CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "PVA Cmd Submit failed");
    }
}

template<typename TI, typename TO>
DOA<TI, TO>::~DOA()
{
    CupvaCmdDestroy(&m_prog);
}

// Explicit instantiation of the template to support the types we care about
template class DOA<int32_t, float>;

} // namespace pva_solutions::priv

NVCVStatus pvaDoaDAFFTCreate(NVCVOperatorHandle *handle, const NVCVTensorRequirements **inTensorRequirements,
                             const NVCVTensorRequirements *outTensorRequirements, int inTensorCount,
                             const PVARadarGP *GP)
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
                                  "Pointer to input NVCVTensorRequirements array must not be NULL");
        }

        if (inTensorCount != pva_solutions::priv::DOA<int32_t, float>::kDefaultTensorCount)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The number of input tensor(s) must be equal to 4");
        }

        for (int i = 0; i < inTensorCount; i++)
        {
            if (inTensorRequirements[i] == nullptr)
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT,
                    "Pointer to the element %d of input NVCVTensorRequirements array must not be NULL", i);
            }
        }

        if (outTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to output NVCVTensorRequirements must not be NULL");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::DOA<int32_t, float>(
            inTensorRequirements, outTensorRequirements, inTensorCount, GP));
    });
}

template<typename StreamType>
NVCVStatus DoaDAFFTSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle *inHandles,
                              NVCVTensorHandle outHandle, int inTensorCount)
{
    return nvcv::ProtectCall([&] {
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DOA<int32_t, float>>(handle)(stream, inHandles,
                                                                                            outHandle, inTensorCount);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus DoaDAFFTSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle *,
                                                      NVCVTensorHandle, int);
template NVCVStatus DoaDAFFTSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle *,
                                                     NVCVTensorHandle, int);

// C++ overload implementations
NVCVStatus pvaDoaDAFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle *inHandles,
                             NVCVTensorHandle outHandle, int inTensorCount)
{
    return DoaDAFFTSubmitImpl(handle, stream, inHandles, outHandle, inTensorCount);
}

NVCVStatus pvaDoaDAFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle *inHandles,
                             NVCVTensorHandle outHandle, int inTensorCount)
{
    return DoaDAFFTSubmitImpl(handle, stream, inHandles, outHandle, inTensorCount);
}
