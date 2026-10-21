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

#include "OpCoherentIntegration.hpp"

#include "coherent_integration_host.h"

#include <SteeringVectorGenerator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Exception.hpp>

#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_coherent_integration_rx4_dev);
CUPVA_DECLARE_EXECUTABLE(pva_coherent_integration_rx8_dev);
CUPVA_DECLARE_EXECUTABLE(pva_coherent_integration_rx12_dev);
CUPVA_DECLARE_EXECUTABLE(pva_coherent_integration_rx16_dev);

namespace {

using pva_solutions::priv::steering_vector::generateSteeringVectorsQ15;
using pva_solutions::priv::steering_vector::OutputConvention;
using pva_solutions::priv::steering_vector::StridedFloatView;
using pva_solutions::priv::steering_vector::StridedPositionView;

constexpr uint32_t kSqdfMaxDimensionAdvanceBytes = (1U << 23U) - 1U;

uint32_t steeringBeamCapacity(uint32_t numRx)
{
    return numRx == 0U ? 0U : CI_STEERING_CAPACITY_BYTES / (CI_STEERING_COMPLEX_BYTES * numRx);
}

struct SteeringDimensions
{
    uint32_t numAzimuthBins;
    uint32_t numElevationBins;
    uint32_t numBeams;
};

SteeringDimensions validateSteeringTensor(NVCVTensorData const &steering, uint32_t expectedNumRx)
{
    if ((steering.dtype != NVCV_DATA_TYPE_2S16) || (steering.layout != nvcv::TENSOR_CHW) || (steering.rank != 3) ||
        (steering.shape[0] <= 0) || (steering.shape[1] <= 0) || (steering.shape[2] != expectedNumRx))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Steering vectors must be a nonempty complex-2S16 CHW tensor with matching Rx");
    }

    uint64_t const numAzimuthBins   = static_cast<uint64_t>(steering.shape[0]);
    uint64_t const numElevationBins = static_cast<uint64_t>(steering.shape[1]);
    // The complete steering grid must fit in the device's resident steering buffer.
    uint32_t const residentBeamCapacity = steeringBeamCapacity(expectedNumRx);
    if (numAzimuthBins > residentBeamCapacity / numElevationBins)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Steering tensor exceeds the resident steering capacity");
    }

    // The device search consumes a packed [Az][El][Rx] tensor.
    uint64_t const packedElevationStride = static_cast<uint64_t>(expectedNumRx) * CI_STEERING_COMPLEX_BYTES;
    uint64_t const packedAzimuthStride   = numElevationBins * packedElevationStride;
    if ((steering.buffer.strided.strides[2] != CI_STEERING_COMPLEX_BYTES) ||
        (static_cast<uint64_t>(steering.buffer.strided.strides[1]) != packedElevationStride) ||
        (static_cast<uint64_t>(steering.buffer.strided.strides[0]) != packedAzimuthStride))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Steering vectors must be packed [Az][El][Rx]");
    }

    return {static_cast<uint32_t>(numAzimuthBins), static_cast<uint32_t>(numElevationBins),
            static_cast<uint32_t>(numAzimuthBins * numElevationBins)};
}

uint32_t largestDivisorAtMost(uint32_t value, uint32_t limit)
{
    for (uint32_t candidate = std::min(value, limit); candidate > 0U; candidate--)
    {
        if ((value % candidate) == 0U)
        {
            return candidate;
        }
    }
    return 0U;
}

std::optional<uint32_t> selectProcessingRangeTile(uint32_t numRangeBins, uint32_t foldedWidth, uint32_t numRx,
                                                  uint32_t maximumRangeTile)
{
    if ((numRangeBins == 0U) || (foldedWidth == 0U) || (maximumRangeTile == 0U))
    {
        return std::nullopt;
    }

    // One range tile must fit its folded cells in raw VMEM and divide the range axis exactly.
    uint32_t const capacity = coherentIntegrationFoldedWidthCapacity(numRx);
    if ((capacity == 0U) || (foldedWidth > capacity))
    {
        return std::nullopt;
    }

    uint32_t const rangeLimit =
        std::min({numRangeBins, capacity / foldedWidth, CI_MAX_PROCESSING_RANGE_TILE, maximumRangeTile});
    uint32_t const rangeTile = largestDivisorAtMost(numRangeBins, rangeLimit);
    return rangeTile == 0U ? std::nullopt : std::optional<uint32_t>{rangeTile};
}

uint32_t ceilLog2(uint32_t value)
{
    uint32_t result = 0U;
    while ((uint32_t{1} << result) < value)
    {
        result++;
    }
    return result;
}

bool hasLayout(NVCVTensorLayout layout, char const *description)
{
    NVCVTensorLayout expected{};
    return (nvcvTensorLayoutMake(description, &expected) == NVCV_SUCCESS) && (layout == expected);
}

void validateDimensions(uint32_t numRangeBins, uint32_t numDopplerBins, uint32_t numRx, int32_t repeatFold)
{
    if (!coherentIntegrationIsSupportedGeometry(numRangeBins, numDopplerBins, numRx, repeatFold))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "Unsupported coherent-integration geometry; got range=%u, Doppler=%u, Rx=%u, repeatFold=%d", numRangeBins,
            numDopplerBins, numRx, repeatFold);
    }
}

struct InputGeometry
{
    uint32_t range;
    uint32_t doppler;
    uint32_t rx;
    uint32_t rangeStride;
    uint32_t rxStride;
    uint32_t dopplerStride;
};

uint32_t checkedStride(int64_t value, char const *name)
{
    if ((value <= 0) || (value > std::numeric_limits<uint32_t>::max()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s is outside the supported range; got %lld", name,
                              static_cast<long long>(value));
    }
    return static_cast<uint32_t>(value);
}

uint64_t checkedLinearSpan(uint32_t count, uint32_t stride, uint64_t tailBytes, char const *name)
{
    uint64_t const intervals = count - 1U;
    if (intervals > ((std::numeric_limits<uint64_t>::max() - tailBytes) / stride))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s overflows", name);
    }
    return intervals * stride + tailBytes;
}

InputGeometry resolveInputGeometry(NVCVTensorRequirements const &requirements, PVAFFTLayout inputLayout)
{
    bool const isHCW  = hasLayout(requirements.layout, "HCW");
    bool const isNHCW = hasLayout(requirements.layout, "NHCW");
    if (!isHCW && !isNHCW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT,
                              "Range-Doppler input layout must be HCW or NHCW");
    }
    if ((isHCW && (requirements.rank != 3)) || (isNHCW && (requirements.rank != 4)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT,
                              "HCW input must have rank three and NHCW input must have rank four");
    }

    // Normalize HCW and NHCW requirements before applying layout-specific stride rules.
    int64_t shape[4]   = {};
    int64_t strides[4] = {};
    nvcvTensorShapePermute(requirements.layout, requirements.shape, NVCV_TENSOR_NCHW, shape);
    nvcvTensorShapePermute(requirements.layout, requirements.strides, NVCV_TENSOR_NCHW, strides);
    if ((shape[0] != 1) || (shape[1] <= 0) || (shape[2] <= 0) || (shape[3] <= 0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The optional batch dimension must equal one and all data dimensions must be positive");
    }

    InputGeometry result{};
    result.range       = checkedStride(shape[2], "Input range dimension");
    result.rangeStride = checkedStride(strides[2], "Input range stride");
    if (inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        result.doppler                    = checkedStride(shape[1], "Input Doppler dimension");
        result.rx                         = checkedStride(shape[3], "Input Rx dimension");
        result.dopplerStride              = checkedStride(strides[1], "Input Doppler stride");
        result.rxStride                   = checkedStride(strides[3], "Input Rx stride");
        uint64_t const packedDopplerBytes = static_cast<uint64_t>(result.rx) * CI_INPUT_COMPLEX_BYTES;
        if (result.rxStride != CI_INPUT_COMPLEX_BYTES)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RDRx input Rx samples must use an 8-byte packed stride");
        }
        if (result.dopplerStride < packedDopplerBytes)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RDRx input Doppler stride must be at least numRx * 8 bytes");
        }
        if (result.dopplerStride > std::numeric_limits<uint16_t>::max())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RDRx input Doppler stride must not exceed 65535 bytes");
        }
        uint64_t const rangeSpan =
            checkedLinearSpan(result.doppler, result.dopplerStride, packedDopplerBytes, "RDRx input range span");
        if (result.rangeStride < rangeSpan)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "RDRx input Range rows must not overlap");
        }
    }
    else if (inputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        result.rx                    = checkedStride(shape[1], "Input Rx dimension");
        result.doppler               = checkedStride(shape[3], "Input Doppler dimension");
        result.rxStride              = checkedStride(strides[1], "Input Rx stride");
        result.dopplerStride         = checkedStride(strides[3], "Input Doppler stride");
        uint64_t const packedRxBytes = static_cast<uint64_t>(result.doppler) * CI_INPUT_COMPLEX_BYTES;
        if (result.dopplerStride != CI_INPUT_COMPLEX_BYTES)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RRxD input Doppler samples must use an 8-byte packed stride");
        }
        if (result.rxStride < packedRxBytes)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RRxD input Rx stride must be at least numDopplerBins * 8 bytes");
        }
        if (result.rxStride > std::numeric_limits<uint16_t>::max())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "RRxD input Rx stride must not exceed 65535 bytes");
        }
        uint64_t const rangeSpan =
            checkedLinearSpan(result.rx, result.rxStride, packedRxBytes, "RRxD input range span");
        if (result.rangeStride < rangeSpan)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "RRxD input Range rows must not overlap");
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported inputLayout value");
    }
    return result;
}

uint32_t validateOutputRequirements(NVCVTensorRequirements const &requirements, uint32_t expectedRange,
                                    uint32_t expectedDoppler, char const *name)
{
    if ((requirements.dtype != NVCV_DATA_TYPE_U32) || (requirements.layout != nvcv::TENSOR_HW) ||
        (requirements.rank != 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT,
                              "%s must have U32 type, HW layout, and rank two", name);
    }
    int64_t shape[2]   = {};
    int64_t strides[2] = {};
    nvcvTensorShapePermute(requirements.layout, requirements.shape, NVCV_TENSOR_HW, shape);
    nvcvTensorShapePermute(requirements.layout, requirements.strides, NVCV_TENSOR_HW, strides);
    if ((shape[0] != expectedRange) || (shape[1] != expectedDoppler))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s shape must match the configured geometry",
                              name);
    }
    if (strides[1] != sizeof(uint32_t))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s W axis must be packed", name);
    }
    uint32_t const rangeStride = checkedStride(strides[0], name);
    if (rangeStride < static_cast<uint64_t>(expectedDoppler) * sizeof(uint32_t))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s rows must not overlap", name);
    }
    return rangeStride;
}

void setPointerParameter(cupvaCmd_t *program, char const *name, void *pointer)
{
    cupvaParameter_t parameter;
    cupvaError_t const getError = CupvaCmdProgramGetParameter(program, &parameter, name);
    if (getError != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to resolve device pointer parameter %s", name);
    }
    if (CupvaParameterSetValuePointer(&parameter, pointer) != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to set device pointer parameter %s", name);
    }
}

struct AddressSpan
{
    uintptr_t begin;
    uintptr_t end;
};

AddressSpan tensorSpan(NVCVTensorData const &data, uint32_t elementBytes)
{
    // Cover the complete positive-stride tensor footprint for conservative alias checks.
    uintptr_t begin = reinterpret_cast<uintptr_t>(data.buffer.strided.basePtr);
    uint64_t bytes  = elementBytes;
    for (int32_t axis = 0; axis < data.rank; axis++)
    {
        if ((data.shape[axis] <= 0) || (data.buffer.strided.strides[axis] <= 0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor shape and strides must be positive");
        }
        bytes += static_cast<uint64_t>(data.shape[axis] - 1) * data.buffer.strided.strides[axis];
    }
    if (begin > std::numeric_limits<uintptr_t>::max() - static_cast<uintptr_t>(bytes))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor address span overflows");
    }
    return {begin, begin + static_cast<uintptr_t>(bytes)};
}

bool overlaps(AddressSpan lhs, AddressSpan rhs)
{
    return (lhs.begin < rhs.end) && (rhs.begin < lhs.end);
}

void *hostPointer(NVCVTensorData const &data)
{
    void *pointer = nullptr;
    if ((data.buffer.strided.basePtr == nullptr) ||
        (CupvaMemGetHostPointer(&pointer, static_cast<void *>(data.buffer.strided.basePtr)) != CUPVA_ERROR_NONE) ||
        (pointer == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor storage must be host-mappable PVA memory");
    }
    return pointer;
}

NVCVTensorData exportTensor(NVCVTensorHandle handle, char const *name)
{
    if (handle == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor must not be NULL", name);
    }
    NVCVTensorData data{};
    if (nvcvTensorExportData(handle, &data) != NVCV_SUCCESS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Failed to export %s tensor", name);
    }
    if ((data.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA) || (data.buffer.strided.basePtr == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor must provide non-NULL strided storage",
                              name);
    }
    return data;
}

} // namespace

namespace pva_solutions::priv {

std::mutex CoherentIntegration::s_execMapMutex;
std::map<CoherentIntegration::ExecKey, std::weak_ptr<cupva::Executable>> CoherentIntegration::s_executables;

std::shared_ptr<cupva::Executable> CoherentIntegration::GetExecutable(uint32_t numRx)
{
    // Each Rx count has a compile-time-specialized VPU executable cached per cuPVA context.
    static IOperator::ExecutableData const rx4Data{CUPVA_EXECUTABLE_DATA(pva_coherent_integration_rx4_dev),
                                                   CUPVA_EXECUTABLE_SIZE(pva_coherent_integration_rx4_dev)};
    static IOperator::ExecutableData const rx8Data{CUPVA_EXECUTABLE_DATA(pva_coherent_integration_rx8_dev),
                                                   CUPVA_EXECUTABLE_SIZE(pva_coherent_integration_rx8_dev)};
    static IOperator::ExecutableData const rx12Data{CUPVA_EXECUTABLE_DATA(pva_coherent_integration_rx12_dev),
                                                    CUPVA_EXECUTABLE_SIZE(pva_coherent_integration_rx12_dev)};
    static IOperator::ExecutableData const rx16Data{CUPVA_EXECUTABLE_DATA(pva_coherent_integration_rx16_dev),
                                                    CUPVA_EXECUTABLE_SIZE(pva_coherent_integration_rx16_dev)};

    IOperator::ExecutableData const *data = nullptr;
    switch (numRx)
    {
    case CI_RX4:
        data = &rx4Data;
        break;
    case CI_RX8:
        data = &rx8Data;
        break;
    case CI_RX12:
        data = &rx12Data;
        break;
    case CI_RX16:
        data = &rx16Data;
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported number of Rx channels");
    }
    ExecKey const key = std::make_tuple(numRx, cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(*data, key, s_executables, s_execMapMutex);
}

CoherentIntegration::CoherentIntegration(NVCVTensorRequirements const *rangeDopplerMapTensorRequirements,
                                         NVCVTensorRequirements const *coherentIntegrationRxTensorRequirements,
                                         NVCVTensorRequirements const *coherentIntegrationFinalTensorRequirements,
                                         PVARadarCoherentIntegrationParams const *params)
    : m_rangeDopplerMapTensorRequirements{*rangeDopplerMapTensorRequirements}
    , m_coherentIntegrationRxTensorRequirements{*coherentIntegrationRxTensorRequirements}
    , m_coherentIntegrationFinalTensorRequirements{*coherentIntegrationFinalTensorRequirements}
    , m_globalParams{}
    , m_inputOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_steeringOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_coherentIntegrationRxOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_coherentIntegrationFinalOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_exec{}
    , m_prog{}
{
    // Resolve the immutable geometry and select the matching VPU executable.
    if (rangeDopplerMapTensorRequirements->dtype != NVCV_DATA_TYPE_2S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_IMAGE_FORMAT,
                              "Range-Doppler input type must be complex 2S32");
    }
    InputGeometry const input = resolveInputGeometry(*rangeDopplerMapTensorRequirements, params->inputLayout);
    validateDimensions(input.range, input.doppler, input.rx, params->repeatFold);
    m_exec                                          = GetExecutable(input.rx);
    uint32_t const foldedWidth                      = input.doppler / static_cast<uint32_t>(params->repeatFold);
    uint32_t const coherentIntegrationRxRangeStride = validateOutputRequirements(
        *coherentIntegrationRxTensorRequirements, input.range, input.doppler, "coherentIntegrationRx output");
    uint32_t const coherentIntegrationFinalRangeStride = validateOutputRequirements(
        *coherentIntegrationFinalTensorRequirements, input.range, foldedWidth, "coherentIntegrationFinal output");

    /* Process one range at a time when an SQDF dimension advance cannot encode an external stride. */
    bool const canAdvanceAcrossRanges = (input.rangeStride <= kSqdfMaxDimensionAdvanceBytes) &&
                                        (coherentIntegrationRxRangeStride <= kSqdfMaxDimensionAdvanceBytes) &&
                                        (coherentIntegrationFinalRangeStride <= kSqdfMaxDimensionAdvanceBytes);
    uint32_t const maximumRangeTile = canAdvanceAcrossRanges ? CI_MAX_PROCESSING_RANGE_TILE : 1U;
    auto const processingRangeTile  = selectProcessingRangeTile(input.range, foldedWidth, input.rx, maximumRangeTile);
    if (!processingRangeTile)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "No legal processing tile exists for the configured dimensions");
    }

    // Record geometry, external strides, and fixed-point scaling for the device program.
    m_globalParams.numRangeBins                             = input.range;
    m_globalParams.numRx                                    = input.rx;
    m_globalParams.repeatFold                               = static_cast<uint32_t>(params->repeatFold);
    m_globalParams.foldedWidth                              = foldedWidth;
    m_globalParams.processingRangeTile                      = *processingRangeTile;
    m_globalParams.rawCellCapacity                          = coherentIntegrationRawCellCapacity(input.rx);
    m_globalParams.inputRangeStrideBytes                    = input.rangeStride;
    m_globalParams.inputDopplerStrideBytes                  = input.dopplerStride;
    m_globalParams.coherentIntegrationRxRangeStrideBytes    = coherentIntegrationRxRangeStride;
    m_globalParams.coherentIntegrationFinalRangeStrideBytes = coherentIntegrationFinalRangeStride;
    m_globalParams.termShift                                = ceilLog2(input.rx);
    m_globalParams.coherentIntegrationRxShift               = m_globalParams.termShift + 1U;
    m_globalParams.coherentIntegrationFinalShift =
        m_globalParams.coherentIntegrationRxShift + ceilLog2(params->repeatFold);

    // Bind the VMEM symbols and SQDF handlers exposed by the selected VPU executable.
    auto program                           = CmdProgram::Create(*m_exec);
    auto rawInputHdl                       = program["rawInputHdl"];
    auto steeringHdl                       = program["steeringHdl"];
    auto coherentIntegrationRxOutputHdl    = program["coherentIntegrationRxOutputHdl"];
    auto coherentIntegrationFinalOutputHdl = program["coherentIntegrationFinalOutputHdl"];
    uint8_t *rawInputV                     = program["rawInput"].ptr<uint8_t>();
    uint8_t *steeringV                     = program["steering"].ptr<uint8_t>();
    uint8_t *workV                         = program["work"].ptr<uint8_t>();

    SequenceDataFlow &rawFlow = program.addDataFlowHead<SequenceDataFlow>().handler(rawInputHdl);
    uint32_t const inputRangeAdvance =
        (m_globalParams.processingRangeTile > 1U) ? m_globalParams.inputRangeStrideBytes : 0U;
    uint32_t const rawRxAdvance    = m_globalParams.rawCellCapacity * CI_INPUT_COMPLEX_BYTES;
    uint32_t const rawRangeAdvance = foldedWidth * CI_INPUT_COMPLEX_BYTES;
    if (params->inputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        /* Transfer each packed Doppler row contiguously for every Rx. */
        uint32_t const foldedRowBytes = foldedWidth * CI_INPUT_COMPLEX_BYTES;
        rawFlow.addTransfer()
            .src(m_inputOp, foldedRowBytes)
            .srcDim1(input.rx, input.rxStride)
            .srcDim2(m_globalParams.processingRangeTile, inputRangeAdvance)
            .dst(rawInputV, foldedRowBytes)
            .dstDim1(input.rx, rawRxAdvance)
            .dstDim2(m_globalParams.processingRangeTile, rawRangeAdvance)
            .tile(foldedRowBytes, 1U)
            .mode(TransferModeType::CONTINUOUS);
    }
    else
    {
        /* Rx is packed in RDRx. Gather each Rx across the strided Doppler rows. */
        rawFlow.addTransfer()
            .src(m_inputOp, m_globalParams.inputDopplerStrideBytes)
            .srcDim1(input.rx, CI_INPUT_COMPLEX_BYTES)
            .srcDim2(m_globalParams.processingRangeTile, inputRangeAdvance)
            .dst(rawInputV, CI_INPUT_COMPLEX_BYTES)
            .dstDim1(input.rx, rawRxAdvance)
            .dstDim2(m_globalParams.processingRangeTile, rawRangeAdvance)
            .tile(CI_INPUT_COMPLEX_BYTES, foldedWidth)
            .mode(TransferModeType::CONTINUOUS);
    }

    // Transfer the complete packed steering grid into resident VMEM before processing.
    SequenceDataFlow &steeringFlow   = program.addDataFlowHead<SequenceDataFlow>().handler(steeringHdl);
    uint32_t const steeringBeamBytes = m_globalParams.numRx * CI_STEERING_COMPLEX_BYTES;
    steeringFlow.addTransfer()
        .src(m_steeringOp, steeringBeamBytes)
        .dst(steeringV, steeringBeamBytes)
        .tile(steeringBeamBytes, 1)
        .mode(TransferModeType::CONTINUOUS);

    /* Output tiles are one row high; range strides are encoded in Dim1. */
    SequenceDataFlow &coherentIntegrationRxFlow =
        program.addDataFlowHead<SequenceDataFlow>().handler(coherentIntegrationRxOutputHdl);
    uint32_t const coherentIntegrationRxTransferWidth = foldedWidth * sizeof(uint32_t);
    coherentIntegrationRxFlow.addTransfer()
        .src(workV, coherentIntegrationRxTransferWidth)
        .srcDim1(m_globalParams.processingRangeTile, coherentIntegrationRxTransferWidth)
        .dst(m_coherentIntegrationRxOp, coherentIntegrationRxTransferWidth)
        .dstDim1(m_globalParams.processingRangeTile,
                 (m_globalParams.processingRangeTile > 1U) ? m_globalParams.coherentIntegrationRxRangeStrideBytes : 0U)
        .tile(coherentIntegrationRxTransferWidth, 1)
        .mode(TransferModeType::CONTINUOUS);

    SequenceDataFlow &coherentIntegrationFinalFlow =
        program.addDataFlowHead<SequenceDataFlow>().handler(coherentIntegrationFinalOutputHdl);
    coherentIntegrationFinalFlow.addTransfer()
        .src(workV, foldedWidth * sizeof(uint32_t))
        .srcDim1(m_globalParams.processingRangeTile, foldedWidth * sizeof(uint32_t))
        .dst(m_coherentIntegrationFinalOp, foldedWidth * sizeof(uint32_t))
        .dstDim1(m_globalParams.processingRangeTile, (m_globalParams.processingRangeTile > 1U)
                                                         ? m_globalParams.coherentIntegrationFinalRangeStrideBytes
                                                         : 0U)
        .tile(foldedWidth * sizeof(uint32_t), 1)
        .mode(TransferModeType::CONTINUOUS);

    // Compile the geometry-dependent graph once; submissions only rebind data and steering dimensions.
    program["GP"].set(&m_globalParams, 1);
    program.compileDataFlows();
    cupva::ConvertTo(std::move(program), m_prog);
}

CoherentIntegration::~CoherentIntegration()
{
    CupvaCmdDestroy(&m_prog);
}

template<typename StreamType>
void CoherentIntegration::operator()(StreamType stream, nvcv::Tensor const &rangeDopplerMap,
                                     nvcv::Tensor const &steeringVectors, nvcv::Tensor const &coherentIntegrationRx,
                                     nvcv::Tensor const &coherentIntegrationFinal)
{
    auto inputData                    = rangeDopplerMap.exportData<nvcv::TensorDataStrided>();
    auto steeringData                 = steeringVectors.exportData<nvcv::TensorDataStrided>();
    auto coherentIntegrationRxData    = coherentIntegrationRx.exportData<nvcv::TensorDataStrided>();
    auto coherentIntegrationFinalData = coherentIntegrationFinal.exportData<nvcv::TensorDataStrided>();
    if (!inputData || !steeringData || !coherentIntegrationRxData || !coherentIntegrationFinalData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All tensors must provide strided storage");
    }
    // Input and output geometry is fixed at creation; the steering grid remains submission-specific.
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_rangeDopplerMapTensorRequirements, inputData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_coherentIntegrationRxTensorRequirements,
                                                          coherentIntegrationRxData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_coherentIntegrationFinalTensorRequirements,
                                                          coherentIntegrationFinalData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output tensors must match the requirements supplied at creation");
    }

    NVCVTensorData const &steering                       = steeringData->cdata();
    SteeringDimensions const steeringDimensions          = validateSteeringTensor(steering, m_globalParams.numRx);
    NVCVTensorData const &input                          = inputData->cdata();
    NVCVTensorData const &coherentIntegrationRxTensor    = coherentIntegrationRxData->cdata();
    NVCVTensorData const &coherentIntegrationFinalTensor = coherentIntegrationFinalData->cdata();
    if ((input.buffer.strided.basePtr == nullptr) || (steering.buffer.strided.basePtr == nullptr) ||
        (coherentIntegrationRxTensor.buffer.strided.basePtr == nullptr) ||
        (coherentIntegrationFinalTensor.buffer.strided.basePtr == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }
    // Rebind the external buffers used by the precompiled data-flow graph.
    m_inputOp.setBase(static_cast<uint8_t *>(static_cast<void *>(input.buffer.strided.basePtr)));
    m_steeringOp.setBase(static_cast<uint8_t *>(static_cast<void *>(steering.buffer.strided.basePtr)));
    m_coherentIntegrationRxOp.setBase(
        static_cast<uint8_t *>(static_cast<void *>(coherentIntegrationRxTensor.buffer.strided.basePtr)));
    m_coherentIntegrationFinalOp.setBase(
        static_cast<uint8_t *>(static_cast<void *>(coherentIntegrationFinalTensor.buffer.strided.basePtr)));

    // Only steering-grid dimensions vary per submission within the resident-capacity contract.
    CoherentIntegrationParams runtimeParams = m_globalParams;
    runtimeParams.numBeams                  = steeringDimensions.numBeams;
    runtimeParams.numAzimuthBins            = steeringDimensions.numAzimuthBins;
    runtimeParams.numElevationBins          = steeringDimensions.numElevationBins;
    cupvaParameter_t gpParameter;
    if ((CupvaCmdProgramGetParameter(&m_prog, &gpParameter, "GP") != CUPVA_ERROR_NONE) ||
        (CupvaParameterSetValueArray(&gpParameter, &runtimeParams, sizeof(runtimeParams)) != CUPVA_ERROR_NONE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update coherent-integration parameters");
    }
    setPointerParameter(&m_prog, "inputExtPtr", input.buffer.strided.basePtr);
    setPointerParameter(&m_prog, "steeringExtPtr", steering.buffer.strided.basePtr);
    setPointerParameter(&m_prog, "coherentIntegrationRxExtPtr", coherentIntegrationRxTensor.buffer.strided.basePtr);
    setPointerParameter(&m_prog, "coherentIntegrationFinalExtPtr",
                        coherentIntegrationFinalTensor.buffer.strided.basePtr);

    // Apply the rebound external pointers before submitting the existing command program.
    cupvaError_t error = CupvaCmdProgramUpdateDataFlows(&m_prog);
    if (error != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Failed to update coherent-integration data flows");
    }

    // The same program supports cuPVA streams and, when available, CUDA streams.
    cupvaCmd_t const *commands[1] = {&m_prog};
    if constexpr (std::is_same_v<StreamType, cupvaStream_t>)
    {
        error = CupvaStreamSubmit(stream, commands, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<StreamType, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            error = CupvaCudaStreamSubmit(stream, commands, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(StreamType), "Unsupported Coherent Integration stream type");
    }
    if (error != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA coherent-integration submission failed");
    }
}

template void CoherentIntegration::operator()<cupvaStream_t>(cupvaStream_t, nvcv::Tensor const &, nvcv::Tensor const &,
                                                             nvcv::Tensor const &, nvcv::Tensor const &);
template void CoherentIntegration::operator()<cudaStream_t>(cudaStream_t, nvcv::Tensor const &, nvcv::Tensor const &,
                                                            nvcv::Tensor const &, nvcv::Tensor const &);

} // namespace pva_solutions::priv

extern "C" NVCVStatus pvaCoherentIntegrationComputeSteeringVectors(NVCVTensorHandle rxPositions,
                                                                   NVCVTensorHandle azimuthBins,
                                                                   NVCVTensorHandle elevationBins,
                                                                   NVCVTensorHandle steeringVectors)
{
    return nvcv::ProtectCall([&] {
        NVCVTensorData const positions = exportTensor(rxPositions, "Rx positions");
        NVCVTensorData const azimuth   = exportTensor(azimuthBins, "Azimuth bins");
        NVCVTensorData const elevation = exportTensor(elevationBins, "Elevation bins");
        NVCVTensorData const steering  = exportTensor(steeringVectors, "Steering vectors");
        if ((positions.dtype != NVCV_DATA_TYPE_2F32) || (positions.layout != nvcv::TENSOR_W) || (positions.rank != 1) ||
            (azimuth.dtype != NVCV_DATA_TYPE_F32) || (azimuth.layout != nvcv::TENSOR_W) || (azimuth.rank != 1) ||
            (elevation.dtype != NVCV_DATA_TYPE_F32) || (elevation.layout != nvcv::TENSOR_W) || (elevation.rank != 1))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Steering-vector tensors do not satisfy the documented types and layouts");
        }

        int64_t const numRx = positions.shape[0];
        int64_t const numAz = azimuth.shape[0];
        int64_t const numEl = elevation.shape[0];
        bool const invalidDimensions =
            (coherentIntegrationFoldedWidthCapacity(numRx) == 0U) || (numAz <= 0) || (numEl <= 0);
        if (invalidDimensions)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Steering-vector dimensions are outside the supported envelope");
        }

        SteeringDimensions const steeringDimensions = validateSteeringTensor(steering, static_cast<uint32_t>(numRx));
        if ((static_cast<int64_t>(steeringDimensions.numAzimuthBins) != numAz) ||
            (static_cast<int64_t>(steeringDimensions.numElevationBins) != numEl))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Steering-vector output dimensions do not match the angle-bin tensor dimensions");
        }

        if ((positions.buffer.strided.strides[0] < static_cast<int64_t>(sizeof(float) * 2U)) ||
            (azimuth.buffer.strided.strides[0] < static_cast<int64_t>(sizeof(float))) ||
            (elevation.buffer.strided.strides[0] < static_cast<int64_t>(sizeof(float))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Steering-vector inputs require positive element strides");
        }

        // Generation writes the full output, so reject overlap with every input tensor.
        AddressSpan const steeringSpan = tensorSpan(steering, CI_STEERING_COMPLEX_BYTES);
        if (overlaps(steeringSpan, tensorSpan(positions, sizeof(float) * 2U)) ||
            overlaps(steeringSpan, tensorSpan(azimuth, sizeof(float))) ||
            overlaps(steeringSpan, tensorSpan(elevation, sizeof(float))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Steering-vector output must not overlap an input tensor");
        }

        uint8_t const *positionsBase = static_cast<uint8_t const *>(hostPointer(positions));
        uint8_t const *azimuthBase   = static_cast<uint8_t const *>(hostPointer(azimuth));
        uint8_t const *elevationBase = static_cast<uint8_t const *>(hostPointer(elevation));
        int16_t *steeringBase        = static_cast<int16_t *>(hostPointer(steering));

        // The common generator assumes finite inputs; validate them at this API boundary.
        for (int64_t rx = 0; rx < numRx; rx++)
        {
            float const *position =
                reinterpret_cast<float const *>(positionsBase + rx * positions.buffer.strided.strides[0]);
            if (!std::isfinite(position[0]) || !std::isfinite(position[1]))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Rx positions must be finite");
            }
        }
        for (int64_t az = 0; az < numAz; az++)
        {
            float const azimuthDegrees =
                *reinterpret_cast<float const *>(azimuthBase + az * azimuth.buffer.strided.strides[0]);
            if (!std::isfinite(azimuthDegrees))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Azimuth bins must be finite");
            }
        }
        for (int64_t el = 0; el < numEl; el++)
        {
            float const elevationDegrees =
                *reinterpret_cast<float const *>(elevationBase + el * elevation.buffer.strided.strides[0]);
            if (!std::isfinite(elevationDegrees))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Elevation bins must be finite");
            }
        }

        // CI consumes the direct Q15 phase convention in packed [Az][El][Rx] order.
        generateSteeringVectorsQ15(
            StridedPositionView{reinterpret_cast<float const *>(positionsBase), positions.buffer.strided.strides[0]},
            StridedFloatView{reinterpret_cast<float const *>(azimuthBase), azimuth.buffer.strided.strides[0]},
            StridedFloatView{reinterpret_cast<float const *>(elevationBase), elevation.buffer.strided.strides[0]},
            steeringBase, static_cast<int32_t>(numRx), static_cast<int32_t>(numAz), static_cast<int32_t>(numEl),
            OutputConvention::Direct);
    });
}

NVCVStatus pvaCoherentIntegrationCreate(NVCVOperatorHandle *handle,
                                        NVCVTensorRequirements const *rangeDopplerMapTensorRequirements,
                                        NVCVTensorRequirements const *coherentIntegrationRxTensorRequirements,
                                        NVCVTensorRequirements const *coherentIntegrationFinalTensorRequirements,
                                        PVARadarCoherentIntegrationParams const *params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (rangeDopplerMapTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to range-Doppler map NVCVTensorRequirements must not be NULL");
        }
        if (coherentIntegrationRxTensorRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to coherent-integration Rx output NVCVTensorRequirements must not be NULL");
        }
        if (coherentIntegrationFinalTensorRequirements == nullptr)
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Pointer to coherent-integration final output NVCVTensorRequirements must not be NULL");
        }
        if (params == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to PVARadarCoherentIntegrationParams must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::CoherentIntegration(
            rangeDopplerMapTensorRequirements, coherentIntegrationRxTensorRequirements,
            coherentIntegrationFinalTensorRequirements, params));
    });
}

template<typename StreamType>
NVCVStatus CoherentIntegrationSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                         NVCVTensorHandle const rangeDopplerMap, NVCVTensorHandle const steeringVectors,
                                         NVCVTensorHandle coherentIntegrationRx,
                                         NVCVTensorHandle coherentIntegrationFinal)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (rangeDopplerMap == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Range-Doppler map tensor handle must not be NULL");
        }
        if (steeringVectors == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Steering vectors tensor handle must not be NULL");
        }
        if (coherentIntegrationRx == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Coherent-integration Rx output tensor handle must not be NULL");
        }
        if (coherentIntegrationFinal == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Coherent-integration final output tensor handle must not be NULL");
        }
        nvcv::TensorWrapHandle input(rangeDopplerMap);
        nvcv::TensorWrapHandle steering(steeringVectors);
        nvcv::TensorWrapHandle coherentIntegrationRxTensor(coherentIntegrationRx);
        nvcv::TensorWrapHandle coherentIntegrationFinalTensor(coherentIntegrationFinal);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::CoherentIntegration>(handle)(
            stream, input, steering, coherentIntegrationRxTensor, coherentIntegrationFinalTensor);
    });
}

template NVCVStatus CoherentIntegrationSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle,
                                                                 NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus CoherentIntegrationSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle,
                                                                NVCVTensorHandle, NVCVTensorHandle, NVCVTensorHandle);

NVCVStatus pvaCoherentIntegrationSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                        NVCVTensorHandle const rangeDopplerMap, NVCVTensorHandle const steeringVectors,
                                        NVCVTensorHandle coherentIntegrationRx,
                                        NVCVTensorHandle coherentIntegrationFinal)
{
    return CoherentIntegrationSubmitImpl(handle, stream, rangeDopplerMap, steeringVectors, coherentIntegrationRx,
                                         coherentIntegrationFinal);
}

NVCVStatus pvaCoherentIntegrationSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                        NVCVTensorHandle const rangeDopplerMap, NVCVTensorHandle const steeringVectors,
                                        NVCVTensorHandle coherentIntegrationRx,
                                        NVCVTensorHandle coherentIntegrationFinal)
{
    return CoherentIntegrationSubmitImpl(handle, stream, rangeDopplerMap, steeringVectors, coherentIntegrationRx,
                                         coherentIntegrationFinal);
}
