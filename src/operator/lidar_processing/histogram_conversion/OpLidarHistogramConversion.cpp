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

#include "OpLidarHistogramConversion.hpp"

#include "device/lidar_histogram_conversion_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#ifdef PVASOL_LIDAR_PROFILING
#    include <cupva_host_nonsafety.hpp>
#endif
#include <nvcv/Tensor.hpp>

#include <ErrorCheckMacros.h>
#include <OpLidarHistogramConversion.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_lidar_histogram_conversion_dev);

namespace pva_solutions::priv {

std::mutex LidarHistogramConversion::s_execMapMutex;

std::map<typename LidarHistogramConversion::ExecKey, std::weak_ptr<cupva::Executable>>
    LidarHistogramConversion::s_lidarHistogramConversionExecutables;

std::shared_ptr<cupva::Executable> LidarHistogramConversion::GetLidarHistogramConversionExec()
{
    static IOperator::ExecutableData sLidarHistogramConversionExecData;

    sLidarHistogramConversionExecData = {CUPVA_EXECUTABLE_DATA(pva_lidar_histogram_conversion_dev),
                                         CUPVA_EXECUTABLE_SIZE(pva_lidar_histogram_conversion_dev)};

    ExecKey execKey = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sLidarHistogramConversionExecData, execKey,
                                             s_lidarHistogramConversionExecutables, s_execMapMutex);
}

LidarHistogramConversion::LidarHistogramConversion(NVCVTensorRequirements const *const histogramRequirements,
                                                   PvaLidarHistogramSpec const *const histogramSpec)
    : m_histogramRequirements(*histogramRequirements)
    , m_histogramSpec(*histogramSpec)
    , m_histogramOp(cupva::OffsetPointer<uint8_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_rangeBiasOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_xyzCalibrationOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_rangeOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_xyzOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_reflectanceOp(cupva::OffsetPointer<float>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    // Validate histogram requirements
    if (histogramRequirements->rank != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Histogram tensor must have NHWC layout (rank 4)");
    }

    // Get dimensions for histogram tensor
    int64_t tensorShapeNHWC[4];
    int64_t tensorStridesNHWC[4];
    nvcvTensorShapePermute(histogramRequirements->layout, histogramRequirements->shape, NVCV_TENSOR_NHWC,
                           tensorShapeNHWC);
    nvcvTensorShapePermute(histogramRequirements->layout, histogramRequirements->strides, NVCV_TENSOR_NHWC,
                           tensorStridesNHWC);

    if (tensorShapeNHWC[0] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Batch size (N) must be 1");
    }

    m_height                          = tensorShapeNHWC[1];
    m_width                           = tensorShapeNHWC[2];
    m_channels                        = tensorShapeNHWC[3];
    m_histogramLinePitch              = tensorStridesNHWC[1];
    int32_t const histogramPixelPitch = tensorStridesNHWC[2];

    // Validate histogram specification
    if (histogramSpec->numHistPerPixel < 1 || histogramSpec->numHistPerPixel > MAX_HISTOGRAMS_PER_PIXEL)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numHistPerPixel out of range");
    }
    if (histogramSpec->numBinsPerHist < 3 || histogramSpec->numBinsPerHist > MAX_HISTOGRAM_BINS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numBinsPerHist out of range");
    }
    if (histogramSpec->numPeaksPerHist < 1 || histogramSpec->numPeaksPerHist > MAX_PEAKS_PER_HISTOGRAM)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numPeaksPerHist out of range");
    }

    // Validate tensor dimensions
    if (m_height < 1 || m_width < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor height and width must be at least 1");
    }

    // Get executable and create command program
    m_exec    = GetLidarHistogramConversionExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    // Set invariant parameters
    prog["numHistPerPixel"] = histogramSpec->numHistPerPixel;
    prog["numBinsPerHist"]  = histogramSpec->numBinsPerHist;
    prog["numPeaksPerHist"] = histogramSpec->numPeaksPerHist;
    prog["pixelHeaderSize"] = histogramSpec->pixelHeaderSize;
    prog["histHeaderSize"]  = histogramSpec->histHeaderSize;
    prog["packingType"]     = (int32_t)histogramSpec->packingType;
    prog["height"]          = m_height;
    prog["width"]           = m_width;

    // Set up the histogram input data flow as a TensorDataFlow
    // The input tensor is HWC layout, but the cuPVA API is designed around a CHW layout.
    // To work with the API, we just treat C channel as "width", W as "height", and H as "depth" in units of bytes.
    int32_t const histPixelElements =
        histogramSpec->pixelHeaderSize +
        histogramSpec->numHistPerPixel * (histogramSpec->histHeaderSize + histogramSpec->numBinsPerHist);
    int32_t const histPixelBytes = (histogramSpec->packingType == PVA_LIDAR_DATA_PACKING_TYPE_NONE)
                                       ? histPixelElements * sizeof(uint16_t)
                                       : ((histPixelElements + 1) / 2) * 3;

    if (histPixelBytes > MAX_TILE_BUFFER_SIZE_BYTES)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Total histogram data for one pixel exceeds tile buffer size limit.");
    }

    // Here we greedily pull in as much W dim as possible to fill the VMEM tile, and set tile H dim (depth) to 1 for simplicity.
    /// NOTE: This tiling scheme aims to support a wide range of tensor shapes, emphasizing flexibility over optimal
    /// utilization of VMEM. If the input tensor shape is fixed in the system, a more precise tile layout and traversal
    /// pattern may be possible by modifying both device and host code to make use of depth dimensions other than 1.
    int32_t numHistPixPerTile =
        std::min(m_width, std::min((MAX_PIXELS_PER_TILE), (MAX_TILE_BUFFER_SIZE_BYTES) / histPixelBytes));

    // The header size limits are a function of the number of pixels and histograms per tile. We have a "soft" limit
    // here that allows the documented "max" values to be exceeded as long as the total space for header data would not
    // exceed the VMEM buffers.
    if ((numHistPixPerTile * histogramSpec->numHistPerPixel * histogramSpec->pixelHeaderSize) >
        (MAX_PIXEL_HEADER_SIZE * MAX_PIXELS_PER_TILE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pixel header size exceeds the maximum allowed");
    }
    if ((numHistPixPerTile * histogramSpec->numHistPerPixel * histogramSpec->histHeaderSize) >
        (MAX_HISTOGRAM_HEADER_SIZE * MAX_HISTOGRAMS_PER_TILE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Histogram header size exceeds the maximum allowed");
    }

    // The device unpacks a whole tile into one VMEM buffer of UNPACKED_HISTOGRAM_BUFFER_SIZE elements: every histogram
    // padded to a whole number of vectors, and for RAW12 one decoded pixel of scratch after the last histogram. How
    // much of it a tile needs is not proportional to the input tile bytes, so reduce the number of pixels per tile
    // until the device's footprint fits. One pixel always fits for every shape accepted above.
    int32_t const isRAW12              = (histogramSpec->packingType == PVA_LIDAR_DATA_PACKING_TYPE_RAW12) ? 1 : 0;
    auto const unpackedElementsPerTile = [&](int32_t numPix) {
        return UnpackedHistogramElementsPerTile(numPix, histogramSpec->numHistPerPixel, histogramSpec->numBinsPerHist,
                                                histPixelBytes, isRAW12);
    };
    while (numHistPixPerTile > 1 && unpackedElementsPerTile(numHistPixPerTile) > UNPACKED_HISTOGRAM_BUFFER_SIZE)
    {
        numHistPixPerTile--;
    }
    if (unpackedElementsPerTile(numHistPixPerTile) > UNPACKED_HISTOGRAM_BUFFER_SIZE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unpacked histogram data for one pixel exceeds the VMEM buffer size limit.");
    }

    // NOTE: The tensor dimension mapping from TensorDataFlow to our HWC layout is:
    //  "Depth" -> H, "Width" -> C, "Height" -> W
    // TDF traversal order is Depth, Width, Height, so our tensor is visited in tiles of (Tw,Th) = (numHistPixPerTile, 1) pixels
    // (every pixel is a full C dim), first along the H dim, then W dim.
    TensorDataFlow &histogramDataFlow = prog.addDataFlowHead<TensorDataFlow>();
    histogramDataFlow.handler(prog["histogramInputHdl"])
        .src(m_histogramOp, histPixelBytes, m_width, m_height, histogramPixelPitch, m_histogramLinePitch)
        .tileBuffer(prog["histogramIn"].ptr<uint8_t>())
        .tile(histPixelBytes, numHistPixPerTile, 1);

    // Find total number of tiles to pull all pixels. Round W dim up to the nearest multiple of numHistPerTile (last tile
    // along W dim is auto-padded with 0s by TDF). The device code compensates for partial tiles by adjusting output transfers.
    int32_t const numTilesW   = (m_width + numHistPixPerTile - 1) / numHistPixPerTile;
    prog["numTilesW"]         = numTilesW;
    prog["numTilesH"]         = m_height;
    prog["numHistPixPerTile"] = numHistPixPerTile;

    // Set up the auxiliary input data flow (just a placeholder, fully configured on device side depending on which
    // inputs/outputs are enabled at runtime).
    SequenceDataFlow &auxInputHdl = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["auxInputHdl"]);
    auxInputHdl.addTransfer()
        .tile(numHistPixPerTile * sizeof(float))
        .src(m_rangeBiasOp, numHistPixPerTile * sizeof(float))
        .dst(prog["rangeBiasIn"].ptr<float>(), numHistPixPerTile * sizeof(float));
    auxInputHdl.addTransfer()
        .tile(numHistPixPerTile * 3 * sizeof(float))
        .src(m_xyzCalibrationOp, numHistPixPerTile * 3 * sizeof(float))
        .dst(prog["xyzCalibrationIn"].ptr<float>(), numHistPixPerTile * 3 * sizeof(float));

    // Set up the output SQDF (is just a placeholder, fully configured on device side depending on which output signals
    // have been enabled at runtime).
    int32_t const outputTileBytes = histogramSpec->numPeaksPerHist * sizeof(float) * 3 * numHistPixPerTile;
    SequenceDataFlow &outputHdl   = prog.addDataFlowHead<SequenceDataFlow>().handler(prog["outputHdl"]);
    outputHdl.addTransfer()
        .tile(outputTileBytes)
        .src(prog["xyzOut"].ptr<float>(), outputTileBytes)
        .dst(m_xyzOp, outputTileBytes);

    // Compile data flows
    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

void LidarHistogramConversion::setParameters(PvaLidarHistogramParams const *const params)
{
    cupvaCmd_t *cmdProg = &m_prog;

    // Validate parameters
    if (params == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Parameters must not be NULL");
    }

    // Set histogram parameters
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "offsetNs");
        CupvaParameterSetValueScalar(&param, &params->offsetNs, sizeof(float));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "binSizeNs");
        CupvaParameterSetValueScalar(&param, &params->binSizeNs, sizeof(float));
    }
    {
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "rangeScale");
        CupvaParameterSetValueScalar(&param, &params->rangeScale, sizeof(float));
    }
    {
        // Reflectance is defined as zero when maxIntensity is not positive. The reciprocal is
        // formed here rather than on the VPU so the guard is evaluated once, in host float
        // semantics, and the device is left with a plain multiply.
        float const invMaxIntensity = (params->maxIntensity > 0.0F) ? (1.0F / params->maxIntensity) : 0.0F;
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(cmdProg, &param, "invMaxIntensity");
        CupvaParameterSetValueScalar(&param, &invMaxIntensity, sizeof(float));
    }
}

template<typename S>
void LidarHistogramConversion::operator()(S stream, const nvcv::Tensor &histogram, const nvcv::Tensor &rangeBias,
                                          const nvcv::Tensor &xyzCalibration, const nvcv::Tensor &range,
                                          const nvcv::Tensor &xyz, const nvcv::Tensor &reflectance,
                                          PvaLidarHistogramParams const *const params)
{
    // Validate input histogram tensor
    auto histogramTensorData = histogram.exportData<nvcv::TensorDataStrided>();
    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_histogramRequirements, histogramTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Histogram tensor must match the requirements used to create the operator");
    }

    if (static_cast<void *>(histogramTensorData->basePtr()) == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Histogram tensor base pointer must not be NULL");
    }

    // At least one output must be provided
    bool const hasRangeOutput       = range.operator bool();
    bool const hasXyzOutput         = xyz.operator bool();
    bool const hasReflectanceOutput = reflectance.operator bool();

    if (!hasRangeOutput && !hasXyzOutput && !hasReflectanceOutput)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "At least one output tensor must be provided");
    }

    // make sure required optional inputs are provided if corresponding outputs are enabled
    bool const hasRangeBiasInput      = rangeBias.operator bool();
    bool const hasXyzCalibrationInput = xyzCalibration.operator bool();

    if (hasRangeOutput && !hasRangeBiasInput)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Range bias tensor must be provided if range output is enabled");
    }

    if (hasXyzOutput && (!hasXyzCalibrationInput || !hasRangeBiasInput))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "XYZ calibration tensor and range bias tensor must be provided if xyz output is enabled");
    }

    // Set base pointer for required histogram input DF (automatically reflected to RDF in UpdateDataFlows)
    m_histogramOp.setBase(static_cast<uint8_t *>(static_cast<void *>(histogramTensorData->basePtr())));

    // Set pointers and line pitches for optional inputs/outputs
    if (hasRangeBiasInput)
    {
        auto rangeBiasData = rangeBias.exportData<nvcv::TensorDataStrided>();
        if (rangeBiasData->shape(0) != 1 || rangeBiasData->shape(1) != m_height || rangeBiasData->shape(2) != m_width ||
            rangeBiasData->shape(3) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Range bias tensor must have NHWC layout (1, H, W, 1)");
        }
        if (rangeBiasData->dtype() != NVCV_DATA_TYPE_F32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Range bias tensor must have F32 data type");
        }
        m_rangeBiasOp.setBase(static_cast<float *>(static_cast<void *>(rangeBiasData->basePtr())));

        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "rangeBiasExtPtr");
        CupvaParameterSetValuePointer(&param, static_cast<void *>(rangeBiasData->basePtr()));

        CupvaCmdProgramGetParameter(&m_prog, &param, "rangeBiasLinePitch");
        CupvaParameterSetValueScalar(&param, &rangeBiasData->stride(1), sizeof(int32_t));
    }
    else
    {
        // base pointer does not matter since it won't be used, but it needs to be set to something other than nullptr
        m_rangeBiasOp.setBase(static_cast<float *>(static_cast<void *>(histogramTensorData->basePtr())));
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "rangeBiasLinePitch");
        int32_t val = 0;
        CupvaParameterSetValueScalar(&param, &val, sizeof(int32_t));
    }

    if (hasXyzCalibrationInput)
    {
        auto xyzCalibrationData = xyzCalibration.exportData<nvcv::TensorDataStrided>();
        if (xyzCalibrationData->shape(0) != 1 || xyzCalibrationData->shape(1) != m_height ||
            xyzCalibrationData->shape(2) != m_width || xyzCalibrationData->shape(3) != 3)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "XYZ calibration tensor must have NHWC layout (1, H, W, 3)");
        }
        if (xyzCalibrationData->dtype() != NVCV_DATA_TYPE_F32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "XYZ calibration tensor must have F32 data type");
        }
        m_xyzCalibrationOp.setBase(static_cast<float *>(static_cast<void *>(xyzCalibrationData->basePtr())));

        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "xyzCalibrationExtPtr");
        CupvaParameterSetValuePointer(&param, static_cast<void *>(xyzCalibrationData->basePtr()));

        CupvaCmdProgramGetParameter(&m_prog, &param, "xyzCalibrationLinePitch");
        CupvaParameterSetValueScalar(&param, &xyzCalibrationData->stride(1), sizeof(int32_t));
    }
    else
    {
        // base pointer does not matter since it won't be used, but it needs to be set to something other than nullptr
        m_xyzCalibrationOp.setBase(static_cast<float *>(static_cast<void *>(histogramTensorData->basePtr())));
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "xyzCalibrationLinePitch");
        int32_t val = 0;
        CupvaParameterSetValueScalar(&param, &val, sizeof(int32_t));
    }

    if (hasRangeOutput)
    {
        auto rangeData = range.exportData<nvcv::TensorDataStrided>();
        if (rangeData->shape(0) != 1 || rangeData->shape(1) != m_height || rangeData->shape(2) != m_width ||
            rangeData->shape(3) != m_histogramSpec.numHistPerPixel * m_histogramSpec.numPeaksPerHist)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Range tensor must have NHWC layout (1, H, W, numHistPerPixel*numPeaksPerHist)");
        }
        if (rangeData->dtype() != NVCV_DATA_TYPE_F32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Range tensor must have F32 data type");
        }
        m_rangeOp.setBase(static_cast<float *>(static_cast<void *>(rangeData->basePtr())));

        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "rangeExtPtr");
        CupvaParameterSetValuePointer(&param, static_cast<void *>(rangeData->basePtr()));

        CupvaCmdProgramGetParameter(&m_prog, &param, "rangeOutLinePitch");
        CupvaParameterSetValueScalar(&param, &rangeData->stride(1), sizeof(int32_t));
    }
    else
    {
        // base pointer does not matter since it won't be used, but it needs to be set to something other than nullptr
        m_rangeOp.setBase(static_cast<float *>(static_cast<void *>(histogramTensorData->basePtr())));
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "rangeOutLinePitch");
        int32_t val = 0;
        CupvaParameterSetValueScalar(&param, &val, sizeof(int32_t));
    }

    if (hasXyzOutput)
    {
        auto xyzData = xyz.exportData<nvcv::TensorDataStrided>();
        if (xyzData->shape(0) != 1 || xyzData->shape(1) != m_height || xyzData->shape(2) != m_width ||
            xyzData->shape(3) != 3 * m_histogramSpec.numHistPerPixel * m_histogramSpec.numPeaksPerHist)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "XYZ tensor must have NHWC layout (1, H, W, 3*numHistPerPixel*numPeaksPerHist)");
        }
        if (xyzData->dtype() != NVCV_DATA_TYPE_F32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "XYZ tensor must have F32 data type");
        }
        m_xyzOp.setBase(static_cast<float *>(static_cast<void *>(xyzData->basePtr())));

        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "xyzExtPtr");
        CupvaParameterSetValuePointer(&param, static_cast<void *>(xyzData->basePtr()));

        CupvaCmdProgramGetParameter(&m_prog, &param, "xyzOutLinePitch");
        CupvaParameterSetValueScalar(&param, &xyzData->stride(1), sizeof(int32_t));
    }
    else
    {
        // base pointer does not matter since it won't be used, but it needs to be set to something other than nullptr
        m_xyzOp.setBase(static_cast<float *>(static_cast<void *>(histogramTensorData->basePtr())));
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "xyzOutLinePitch");
        int32_t val = 0;
        CupvaParameterSetValueScalar(&param, &val, sizeof(int32_t));
    }

    if (hasReflectanceOutput)
    {
        auto reflectanceData = reflectance.exportData<nvcv::TensorDataStrided>();
        if (reflectanceData->shape(0) != 1 || reflectanceData->shape(1) != m_height ||
            reflectanceData->shape(2) != m_width ||
            reflectanceData->shape(3) != m_histogramSpec.numHistPerPixel * m_histogramSpec.numPeaksPerHist)
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "Reflectance tensor must have NHWC layout (1, H, W, numHistPerPixel*numPeaksPerHist)");
        }
        if (reflectanceData->dtype() != NVCV_DATA_TYPE_F32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Reflectance tensor must have F32 data type");
        }
        m_reflectanceOp.setBase(static_cast<float *>(static_cast<void *>(reflectanceData->basePtr())));

        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "reflectanceExtPtr");
        CupvaParameterSetValuePointer(&param, static_cast<void *>(reflectanceData->basePtr()));

        CupvaCmdProgramGetParameter(&m_prog, &param, "reflectanceOutLinePitch");
        CupvaParameterSetValueScalar(&param, &reflectanceData->stride(1), sizeof(int32_t));
    }
    else
    {
        // base pointer does not matter since it won't be used, but it needs to be set to something other than nullptr
        m_reflectanceOp.setBase(static_cast<float *>(static_cast<void *>(histogramTensorData->basePtr())));
        cupvaParameter_t param;
        CupvaCmdProgramGetParameter(&m_prog, &param, "reflectanceOutLinePitch");
        int32_t val = 0;
        CupvaParameterSetValueScalar(&param, &val, sizeof(int32_t));
    }

    CUPVA_CHECK_ERROR(CupvaCmdProgramUpdateDataFlows(&m_prog));

    setParameters(params);

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
        static_assert(!sizeof(S), "Unsupported stream type for LidarHistogramConversion operator");
    }
}

// Explicit template instantiation
template void LidarHistogramConversion::operator()<cupvaStream_t>(cupvaStream_t, const nvcv::Tensor &,
                                                                  const nvcv::Tensor &, const nvcv::Tensor &,
                                                                  const nvcv::Tensor &, const nvcv::Tensor &,
                                                                  const nvcv::Tensor &,
                                                                  PvaLidarHistogramParams const *const);
template void LidarHistogramConversion::operator()<cudaStream_t>(cudaStream_t, const nvcv::Tensor &,
                                                                 const nvcv::Tensor &, const nvcv::Tensor &,
                                                                 const nvcv::Tensor &, const nvcv::Tensor &,
                                                                 const nvcv::Tensor &,
                                                                 PvaLidarHistogramParams const *const);

LidarHistogramConversion::~LidarHistogramConversion()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaLidarHistogramConversionCreate(NVCVOperatorHandle *handle,
                                             NVCVTensorRequirements const *const histogramRequirements,
                                             PvaLidarHistogramSpec const *const histogramSpec)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (histogramRequirements == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to histogram TensorRequirements must not be NULL");
        }
        if (histogramSpec == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to histogram specification must not be NULL");
        }

#ifdef PVASOL_LIDAR_PROFILING
        cupva::SetVPUPrintBufferSize(64 * 1024);
#endif

        // Validate packing type and data type compatibility
        if (histogramSpec->packingType == PVA_LIDAR_DATA_PACKING_TYPE_NONE)
        {
            if (histogramRequirements->dtype != NVCV_DATA_TYPE_U16)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "For NONE packing type, histogram data type must be U16");
            }
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new pva_solutions::priv::LidarHistogramConversion(histogramRequirements, histogramSpec));
        }
        else if (histogramSpec->packingType == PVA_LIDAR_DATA_PACKING_TYPE_RAW12)
        {
            switch (histogramRequirements->dtype)
            {
            case NVCV_DATA_TYPE_U8:
            case NVCV_DATA_TYPE_U16:
            case NVCV_DATA_TYPE_U32:
                *handle = reinterpret_cast<NVCVOperatorHandle>(
                    new pva_solutions::priv::LidarHistogramConversion(histogramRequirements, histogramSpec));
                break;
            default:
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "For RAW12 packing type, histogram data type must be U8, U16, or U32");
            }
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid packing type");
        }
    });
}

template<typename StreamType>
NVCVStatus LidarHistogramConversionSubmitImpl(NVCVOperatorHandle handle, StreamType stream,
                                              PvaLidarHistogramParams const *params, NVCVTensorHandle const histogram,
                                              NVCVTensorHandle const rangeBias, NVCVTensorHandle const xyzCalibration,
                                              NVCVTensorHandle range, NVCVTensorHandle xyz,
                                              NVCVTensorHandle reflectance)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle histogramWrap(histogram), rangeBiasWrap(rangeBias), xyzCalibrationWrap(xyzCalibration),
            rangeWrap(range), xyzWrap(xyz), reflectanceWrap(reflectance);

        if (histogram == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Histogram tensor handle must not be NULL");
        }

        NVCVDataType histDataType;
        nvcvTensorGetDataType(histogram, &histDataType);

        switch (histDataType)
        {
        case NVCV_DATA_TYPE_U8:
        case NVCV_DATA_TYPE_U16:
        case NVCV_DATA_TYPE_U32:
            pva_solutions::priv::ToDynamicRef<pva_solutions::priv::LidarHistogramConversion>(handle)(
                stream, histogramWrap, rangeBiasWrap, xyzCalibrationWrap, rangeWrap, xyzWrap, reflectanceWrap, params);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported histogram data type");
        }
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus LidarHistogramConversionSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t,
                                                                      PvaLidarHistogramParams const *,
                                                                      NVCVTensorHandle const, NVCVTensorHandle const,
                                                                      NVCVTensorHandle const, NVCVTensorHandle,
                                                                      NVCVTensorHandle, NVCVTensorHandle);
template NVCVStatus LidarHistogramConversionSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t,
                                                                     PvaLidarHistogramParams const *,
                                                                     NVCVTensorHandle const, NVCVTensorHandle const,
                                                                     NVCVTensorHandle const, NVCVTensorHandle,
                                                                     NVCVTensorHandle, NVCVTensorHandle);

// C API overload implementations
NVCVStatus pvaLidarHistogramConversionSubmit(NVCVOperatorHandle const handle, cupvaStream_t const stream,
                                             PvaLidarHistogramParams const *const params,
                                             NVCVTensorHandle const histogram, NVCVTensorHandle const rangeBias,
                                             NVCVTensorHandle const xyzCalibration, NVCVTensorHandle range,
                                             NVCVTensorHandle xyz, NVCVTensorHandle reflectance)
{
    return LidarHistogramConversionSubmitImpl(handle, stream, params, histogram, rangeBias, xyzCalibration, range, xyz,
                                              reflectance);
}

NVCVStatus pvaLidarHistogramConversionSubmit(NVCVOperatorHandle const handle, cudaStream_t const stream,
                                             PvaLidarHistogramParams const *const params,
                                             NVCVTensorHandle const histogram, NVCVTensorHandle const rangeBias,
                                             NVCVTensorHandle const xyzCalibration, NVCVTensorHandle range,
                                             NVCVTensorHandle xyz, NVCVTensorHandle reflectance)
{
    return LidarHistogramConversionSubmitImpl(handle, stream, params, histogram, rangeBias, xyzCalibration, range, xyz,
                                              reflectance);
}
