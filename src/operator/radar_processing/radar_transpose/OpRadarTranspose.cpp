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

#include "OpRadarTranspose.hpp"

#include "device/radar_transpose_param.h"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>

#include <OpRadarTranspose.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

using namespace cupva;

CUPVA_DECLARE_EXECUTABLE(pva_radar_transpose_dev);

namespace pva_solutions::priv {

static_assert(static_cast<int32_t>(RADAR_TRANSPOSE_OUTPUT_LAYOUT_DOPPLER_RX_RANGE) ==
                  static_cast<int32_t>(PVA_FFT_LAYOUT_DOPPLER_RX_RANGE),
              "RadarTranspose and public DOPPLER_RX_RANGE layout values must match");
static_assert(static_cast<int32_t>(RADAR_TRANSPOSE_OUTPUT_LAYOUT_RANGE_RX_DOPPLER) ==
                  static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_RX_DOPPLER),
              "RadarTranspose and public RANGE_RX_DOPPLER layout values must match");
static_assert(static_cast<int32_t>(RADAR_TRANSPOSE_OUTPUT_LAYOUT_RANGE_DOPPLER_RX) ==
                  static_cast<int32_t>(PVA_FFT_LAYOUT_RANGE_DOPPLER_RX),
              "RadarTranspose and public RANGE_DOPPLER_RX layout values must match");

std::mutex RadarTranspose::s_execMapMutex;
std::map<RadarTranspose::ExecKey, std::weak_ptr<cupva::Executable>> RadarTranspose::s_radarTransposeExecutables;

std::shared_ptr<cupva::Executable> RadarTranspose::GetRadarTransposeExec()
{
    static IOperator::ExecutableData sRadarTransposeExecData;
    sRadarTransposeExecData = {CUPVA_EXECUTABLE_DATA(pva_radar_transpose_dev),
                               CUPVA_EXECUTABLE_SIZE(pva_radar_transpose_dev)};
    ExecKey execKey         = std::make_tuple(cupva::Context::GetCurrent());
    return IOperator::GetExecutable<ExecKey>(sRadarTransposeExecData, execKey, s_radarTransposeExecutables,
                                             s_execMapMutex);
}

RadarTranspose::RadarTranspose(NVCVTensorRequirements const *const inTensorRequirements,
                               NVCVTensorRequirements const *const outTensorRequirements,
                               PVARadarTransposeParams const *const params)
    : m_inTensorRequirements{*inTensorRequirements}
    , m_outTensorRequirements{*outTensorRequirements}
    , m_srcOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_dstOp(cupva::OffsetPointer<int32_t>::Create(cupva::MemType::DRAM, nullptr))
    , m_prog{}
{
    const NVCVTensorLayout layoutHCW  = NVCV_TENSOR_LAYOUT_MAKE("HCW");
    const NVCVTensorLayout layoutNHCW = NVCV_TENSOR_LAYOUT_MAKE("NHCW");

    auto validateTensorLayout = [&](NVCVTensorRequirements const *requirements, const char *tensorName) {
        const bool isHCW  = nvcvTensorLayoutCompare(requirements->layout, layoutHCW) == 0;
        const bool isNHCW = nvcvTensorLayoutCompare(requirements->layout, layoutNHCW) == 0;
        if ((!isHCW && !isNHCW) || requirements->rank != requirements->layout.rank)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor must have HCW or NHCW layout",
                                  tensorName);
        }
        if (requirements->dtype != NVCV_DATA_TYPE_2S32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor must have 2S32 data type",
                                  tensorName);
        }
    };

    validateTensorLayout(inTensorRequirements, "Input");
    validateTensorLayout(outTensorRequirements, "Output");

    switch (params->outputLayout)
    {
    case PVA_FFT_LAYOUT_RANGE_DOPPLER_RX:
    case PVA_FFT_LAYOUT_DOPPLER_RX_RANGE:
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported outputLayout %d",
                              static_cast<int>(params->outputLayout));
    }

    // Normalize HCW and NHCW shapes to NCHW for validation and dimension extraction.
    int64_t inShapeNCHW[4];
    nvcvTensorShapePermute(inTensorRequirements->layout, inTensorRequirements->shape, NVCV_TENSOR_NCHW, inShapeNCHW);

    if (inShapeNCHW[0] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor batch dimension must be 1");
    }

    const int64_t nRange64   = inShapeNCHW[2];
    const int64_t nRx64      = inShapeNCHW[1];
    const int64_t nDoppler64 = inShapeNCHW[3];

    if (nRange64 <= 0 || nRx64 <= 0 || nDoppler64 <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor dimensions must be positive");
    }
    if (nRx64 > MAX_NRX)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "nRx exceeds maximum %d", MAX_NRX);
    }
    if (nDoppler64 > MAX_NDOPPLER)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "nDoppler exceeds maximum %d", MAX_NDOPPLER);
    }
    if (nRange64 > MAX_NRANGE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "nRange exceeds maximum %d", MAX_NRANGE);
    }

    auto validatePackedTensor = [](NVCVTensorRequirements const *requirements, const char *tensorName) {
        int64_t expectedStride = COMPLEX_ELEMENT_NUM * static_cast<int64_t>(sizeof(int32_t));
        for (int32_t dimension = requirements->rank - 1; dimension >= 0; --dimension)
        {
            if (requirements->strides[dimension] != expectedStride)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "%s tensor must be fully packed; padded strides are not supported", tensorName);
            }
            expectedStride *= requirements->shape[dimension];
        }
    };

    validatePackedTensor(inTensorRequirements, "Input");

    const int32_t nRange   = static_cast<int32_t>(nRange64);
    const int32_t nRx      = static_cast<int32_t>(nRx64);
    const int32_t nDoppler = static_cast<int32_t>(nDoppler64);

    int64_t outShapeNCHW[4];
    nvcvTensorShapePermute(outTensorRequirements->layout, outTensorRequirements->shape, NVCV_TENSOR_NCHW, outShapeNCHW);
    if (outShapeNCHW[0] != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output tensor batch dimension must be 1");
    }

    const bool outputShapeMatches =
        params->outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX
            ? outShapeNCHW[1] == nDoppler64 && outShapeNCHW[2] == nRange64 && outShapeNCHW[3] == nRx64
            : outShapeNCHW[1] == nRx64 && outShapeNCHW[2] == nDoppler64 && outShapeNCHW[3] == nRange64;
    if (!outputShapeMatches)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output tensor shape does not match the requested transpose layout");
    }
    validatePackedTensor(outTensorRequirements, "Output");

    // For DOPPLER_RX_RANGE, nRange must be a multiple of CHUNK_RANGE
    if (params->outputLayout == PVA_FFT_LAYOUT_DOPPLER_RX_RANGE)
    {
        if ((nRange % CHUNK_RANGE) != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "nRange must be a multiple of %d for DOPPLER_RX_RANGE (got %d)", CHUNK_RANGE, nRange);
        }
        if (nRange > 512 && (nRange % 8) != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "nRange values greater than 512 must be multiples of 8 for DOPPLER_RX_RANGE "
                                  "(got %d)",
                                  nRange);
        }
    }

    m_exec    = GetRadarTransposeExec();
    auto prog = CmdProgram::Create(*m_exec.get());

    int32_t phase         = 0;
    bool useVectorizedRdx = false;

    auto srcHdl = prog["srcHdl"];
    auto dstHdl = prog["dstHdl"];

    int32_t *src_v = prog["srcBuf"].ptr<int32_t>();
    int32_t *dst_v = prog["dstBuf"].ptr<int32_t>();

    // Byte offset between the two VMEM double-buffer slots (srcBuf[0]/srcBuf[1], dstBuf[0]/dstBuf[1]).
    static constexpr int32_t tileSizeBytes = MAX_TILE_SIZE * static_cast<int32_t>(sizeof(int32_t));

    SequenceDataFlow &srcDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(srcHdl);
    SequenceDataFlow &dstDataFlow = prog.addDataFlowHead<SequenceDataFlow>(phase).handler(dstHdl);

    // SQDF dimension limits
    static constexpr int32_t kMaxDim1      = 256;
    static constexpr int32_t kMaxTileBytes = 65535;   // tile width must be < 65536
    static constexpr int64_t kMaxAdvBytes  = 4194304; // SQDF dim advance must be <= 4MB (hardware limit)

    if (params->outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        int32_t dopplerSlice = nDoppler;
        while (dopplerSlice > 1 &&
               nRx * dopplerSlice * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t)) > kMaxTileBytes)
        {
            dopplerSlice /= 2;
        }
        if (nDoppler % dopplerSlice != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The dimensions nRange=%d, nRx=%d and nDoppler=%d are not supported for "
                                  "RANGE_DOPPLER_RX",
                                  nRange, nRx, nDoppler);
        }
        const int32_t nDopplerSlices   = nDoppler / dopplerSlice;
        const int32_t sliceWidthBytes  = dopplerSlice * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
        const int32_t sliceTotalBytes  = nRx * sliceWidthBytes;
        const int32_t srcRowPitchBytes = nDoppler * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
        const int32_t rangeBinBytes    = nRx * srcRowPitchBytes;

        // The vectorized path supports any valid nRx when dopplerSlice is a multiple of 8 and the padded tile fits.
        static constexpr int32_t kVecw  = RADAR_TRANSPOSE_VECW_INT32;
        const int32_t outLinePitchInt32 = ((nRx * 2 + kVecw - 1) & ~(kVecw - 1)) + 2;
        // The padded tile contains dopplerSlice * outLinePitchInt32 int32 elements.
        useVectorizedRdx = ((dopplerSlice % 8) == 0) && (dopplerSlice * outLinePitchInt32 <= MAX_TILE_SIZE);

        if (nDopplerSlices == 1)
        {
            if ((nRange % 2) != 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "nRange must be even for RANGE_DOPPLER_RX (got %d)", nRange);
            }

            int32_t dim1_n = std::min(nRange, kMaxDim1);
            while (dim1_n > 1 &&
                   (nRange % dim1_n != 0 || static_cast<int64_t>(dim1_n) * sliceTotalBytes > kMaxAdvBytes))
            {
                dim1_n /= 2;
            }
            if (nRange % dim1_n != 0 || nRange / dim1_n > kMaxDim1)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The dimensions nRange=%d, nRx=%d and nDoppler=%d are not supported for "
                                      "RANGE_DOPPLER_RX",
                                      nRange, nRx, nDoppler);
            }

            if (dim1_n < 2 || (dim1_n % 2) != 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The dimensions nRange=%d, nRx=%d and nDoppler=%d are not supported for "
                                      "RANGE_DOPPLER_RX",
                                      nRange, nRx, nDoppler);
            }
            const int32_t dim2_n = nRange / dim1_n;

            auto srcXfer = srcDataFlow.addTransfer();
            srcXfer.src(m_srcOp, sliceTotalBytes)
                .dst(src_v, sliceTotalBytes)
                .srcDim1(dim1_n, sliceTotalBytes)
                .dstDim1(2, tileSizeBytes)
                .dstDim2(dim1_n / 2, 0)
                .tile(sliceTotalBytes, 1)
                .mode(TransferModeType::TILE);
            if (dim2_n > 1)
                srcXfer.srcDim2(dim2_n, dim1_n * sliceTotalBytes).dstDim3(dim2_n, 0);

            auto dstXfer = dstDataFlow.addTransfer();
            if (useVectorizedRdx)
            {
                // dstBuf rows padded to outLinePitchInt32 int32; strip via 2D tile transfer.
                const int32_t validBytesPerRow  = nRx * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
                const int32_t paddedBytesPerRow = outLinePitchInt32 * static_cast<int32_t>(sizeof(int32_t));
                dstXfer.src(dst_v, paddedBytesPerRow)
                    .dst(m_dstOp, validBytesPerRow)
                    .srcDim1(2, tileSizeBytes)
                    .srcDim2(dim1_n / 2, 0)
                    .dstDim1(dim1_n, sliceTotalBytes)
                    .tile(validBytesPerRow, dopplerSlice)
                    .mode(TransferModeType::TILE);
                if (dim2_n > 1)
                    dstXfer.srcDim3(dim2_n, 0).dstDim2(dim2_n, dim1_n * sliceTotalBytes);
            }
            else
            {
                dstXfer.src(dst_v, sliceTotalBytes)
                    .dst(m_dstOp, sliceTotalBytes)
                    .srcDim1(2, tileSizeBytes)
                    .srcDim2(dim1_n / 2, 0)
                    .dstDim1(dim1_n, sliceTotalBytes)
                    .tile(sliceTotalBytes, 1)
                    .mode(TransferModeType::TILE);
                if (dim2_n > 1)
                    dstXfer.srcDim3(dim2_n, 0).dstDim2(dim2_n, dim1_n * sliceTotalBytes);
            }

            prog["nIter"] = nRange;
        }
        else
        {
            int32_t dim2RangeBlock = std::min(nRange, kMaxDim1);
            if (nRange > kMaxDim1)
            {
                while (dim2RangeBlock > 1 && (nRange % dim2RangeBlock != 0 ||
                                              static_cast<int64_t>(dim2RangeBlock) * rangeBinBytes > kMaxAdvBytes))
                {
                    dim2RangeBlock /= 2;
                }
            }
            if (nRange % dim2RangeBlock != 0 || (nRange / dim2RangeBlock) > kMaxDim1)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The dimensions nRange=%d, nRx=%d and nDoppler=%d are not supported for "
                                      "RANGE_DOPPLER_RX",
                                      nRange, nRx, nDoppler);
            }
            const int32_t dim2RangeOuter = nRange / dim2RangeBlock;

            // The VMEM ping-pong folds nDopplerSlices/2 into vmemDim2, so an odd
            // nDopplerSlices would under-count the VMEM sequence and deadlock.
            if ((nDopplerSlices % 2) != 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "nDoppler=%d with nRx=%d is not supported for RANGE_DOPPLER_RX because it "
                                      "cannot be divided into an even number of equal Doppler partitions",
                                      nDoppler, nRx);
            }

            const int32_t vmemDim2 = (nDopplerSlices / 2) * dim2RangeBlock;
            if (vmemDim2 > kMaxDim1)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The dimensions nRange=%d, nRx=%d and nDoppler=%d are not supported for "
                                      "RANGE_DOPPLER_RX",
                                      nRange, nRx, nDoppler);
            }

            // src: 2D tile [dopplerSlice cols × nRx rows], srcLinePitch = full doppler row width.
            // VMEM (dst) uses ping-pong dim1 + folded dim2 + range-outer dim3.
            auto srcXfer = srcDataFlow.addTransfer();
            srcXfer.src(m_srcOp, srcRowPitchBytes)
                .dst(src_v, sliceWidthBytes)
                .srcDim1(nDopplerSlices, sliceWidthBytes)
                .dstDim1(2, tileSizeBytes)
                .dstDim2(vmemDim2, 0)
                .tile(sliceWidthBytes, nRx)
                .mode(TransferModeType::TILE);
            if (dim2RangeBlock > 1)
                srcXfer.srcDim2(dim2RangeBlock, rangeBinBytes); // DRAM only; range block folded into vmemDim2
            if (dim2RangeOuter > 1)
            {
                srcXfer.srcDim3(dim2RangeOuter, static_cast<int32_t>(dim2RangeBlock) * rangeBinBytes);
                srcXfer.dstDim3(dim2RangeOuter, 0); // VMEM range-outer stride=0
            }

            auto dstXfer = dstDataFlow.addTransfer();
            if (useVectorizedRdx)
            {
                // Padded dstBuf [dopplerSlice][outLinePitch] — strip padding via 2D tile
                const int32_t validBytesPerRow  = nRx * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
                const int32_t paddedBytesPerRow = outLinePitchInt32 * static_cast<int32_t>(sizeof(int32_t));
                dstXfer.src(dst_v, paddedBytesPerRow)
                    .dst(m_dstOp, validBytesPerRow)
                    .srcDim1(2, tileSizeBytes)
                    .srcDim2(vmemDim2, 0)
                    .dstDim1(nDopplerSlices, sliceTotalBytes)
                    .tile(validBytesPerRow, dopplerSlice)
                    .mode(TransferModeType::TILE);
                if (dim2RangeBlock > 1)
                    dstXfer.dstDim2(dim2RangeBlock, rangeBinBytes); // DRAM only; range block folded into vmemDim2
                if (dim2RangeOuter > 1)
                {
                    dstXfer.srcDim3(dim2RangeOuter, 0); // VMEM stride=0
                    dstXfer.dstDim3(dim2RangeOuter, static_cast<int32_t>(dim2RangeBlock) * rangeBinBytes);
                }
            }
            else
            {
                // Scalar RDX: flat [dopplerSlice*nRx*2] output per iteration
                dstXfer.src(dst_v, sliceTotalBytes)
                    .dst(m_dstOp, sliceTotalBytes)
                    .srcDim1(2, tileSizeBytes)
                    .srcDim2(vmemDim2, 0)
                    .dstDim1(nDopplerSlices, sliceTotalBytes)
                    .tile(sliceTotalBytes, 1)
                    .mode(TransferModeType::TILE);
                if (dim2RangeBlock > 1)
                    dstXfer.dstDim2(dim2RangeBlock, rangeBinBytes); // DRAM only
                if (dim2RangeOuter > 1)
                {
                    dstXfer.srcDim3(dim2RangeOuter, 0); // VMEM stride=0
                    dstXfer.dstDim3(dim2RangeOuter, static_cast<int32_t>(dim2RangeBlock) * rangeBinBytes);
                }
            }

            prog["nIter"] = nRange * nDopplerSlices;
        }

        prog["nDoppler"] = dopplerSlice;
    }
    else if (params->outputLayout == PVA_FFT_LAYOUT_DOPPLER_RX_RANGE)
    {
        int32_t effectiveChunk = CHUNK_RANGE;
        while (effectiveChunk > 1 && ((nRange % effectiveChunk) != 0 || ((nRange / effectiveChunk) % 2) != 0))
        {
            effectiveChunk /= 2;
        }

        // Reduce dopplerSlice until:
        //   (a) the input tile fits in VMEM srcBuf (MAX_TILE_SIZE int32), AND
        //   (b) the dst SQDF dim1 advance (dstSliceStride = dopplerSlice*nRx*nRange*2*4) <= kMaxAdvBytes.
        int32_t dopplerSlice = nDoppler;
        while (dopplerSlice > 1 &&
               (static_cast<int64_t>(effectiveChunk) * nRx * dopplerSlice * COMPLEX_ELEMENT_NUM > MAX_TILE_SIZE ||
                static_cast<int64_t>(dopplerSlice) * nRx * nRange * COMPLEX_ELEMENT_NUM *
                        static_cast<int32_t>(sizeof(int32_t)) >
                    kMaxAdvBytes))
        {
            dopplerSlice /= 2;
        }
        if (nDoppler % dopplerSlice != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The dimensions nRange=%d, nRx=%d and nDoppler=%d are not supported for "
                                  "DOPPLER_RX_RANGE",
                                  nRange, nRx, nDoppler);
        }

        const int32_t nDopplerSlices    = nDoppler / dopplerSlice;
        const int32_t nRangeChunks      = nRange / effectiveChunk;
        const int32_t nIter             = nRangeChunks * nDopplerSlices;
        const int32_t nRowsDopplerRx    = dopplerSlice * nRx;
        const int32_t chunkOutBytes     = effectiveChunk * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
        const int32_t dopplerSliceBytes = dopplerSlice * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
        const int32_t srcRowPitchBytes  = nDoppler * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
        const int32_t dstRowPitchBytes  = nRange * COMPLEX_ELEMENT_NUM * static_cast<int32_t>(sizeof(int32_t));
        const int32_t srcRangeChunkStride = effectiveChunk * nRx * srcRowPitchBytes;

        if (nDopplerSlices > kMaxDim1 || nRangeChunks > kMaxDim1 ||
            (nRangeChunks > 1 && srcRangeChunkStride > kMaxAdvBytes))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The dimensions nRange=%d, nRx=%d and nDoppler=%d are not supported for "
                                  "DOPPLER_RX_RANGE",
                                  nRange, nRx, nDoppler);
        }

        // Source SQDF: 2D tile [dopplerSlice cols × effectiveChunk*nRx rows].
        auto srcXfer = srcDataFlow.addTransfer();
        srcXfer.src(m_srcOp, srcRowPitchBytes)
            .dst(src_v, dopplerSliceBytes)
            .dstDim1(2, tileSizeBytes)
            .tile(dopplerSliceBytes, effectiveChunk * nRx)
            .mode(TransferModeType::TILE);

        // Destination SQDF: tight [chunkOutBytes × nRowsDopplerRx rows].
        // Both vectorized (compact to dstBuf[cur]) and scalar paths read from dstBuf.
        auto dstXfer = dstDataFlow.addTransfer();
        dstXfer.src(dst_v, chunkOutBytes)
            .srcDim1(2, tileSizeBytes)
            .dst(m_dstOp, dstRowPitchBytes)
            .tile(chunkOutBytes, nRowsDopplerRx)
            .mode(TransferModeType::TILE);

        if (nDopplerSlices == 1)
        {
            // No doppler split — iterate over range chunks only.
            const int32_t dim1_n = nRangeChunks;
            srcXfer.srcDim1(dim1_n, srcRangeChunkStride).dstDim2(dim1_n / 2, 0);
            dstXfer.srcDim2(dim1_n / 2, 0).dstDim1(dim1_n, chunkOutBytes);
        }
        else
        {
            // Doppler split: dim1=nDopplerSlices (inner), dim2=nRangeChunks (outer).
            if ((nDopplerSlices % 2) != 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "nDoppler=%d with nRx=%d is not supported for DOPPLER_RX_RANGE because it "
                                      "cannot be divided into an even number of equal Doppler partitions",
                                      nDoppler, nRx);
            }
            srcXfer.srcDim1(nDopplerSlices, dopplerSliceBytes).dstDim2(nDopplerSlices / 2, 0);
            if (nRangeChunks > 1)
                srcXfer.srcDim2(nRangeChunks, srcRangeChunkStride).dstDim3(nRangeChunks, 0);
            const int32_t dstSliceStride = nRowsDopplerRx * dstRowPitchBytes;
            dstXfer.srcDim2(nDopplerSlices / 2, 0).dstDim1(nDopplerSlices, dstSliceStride);
            if (nRangeChunks > 1)
                dstXfer.srcDim3(nRangeChunks, 0).dstDim2(nRangeChunks, chunkOutBytes);
        }

        prog["nIter"]      = nIter;
        prog["chunkRange"] = effectiveChunk;
        prog["nDoppler"]   = dopplerSlice;
    }

    // Set device parameters
    prog["nRange"]           = nRange;
    prog["nRx"]              = nRx;
    prog["outputLayout"]     = static_cast<int32_t>(params->outputLayout);
    prog["useVectorizedRdx"] = useVectorizedRdx ? 1 : 0;

    prog.compileDataFlows();
    cupva::ConvertTo(std::move(prog), m_prog);
}

template<typename S>
void RadarTranspose::operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
{
    auto inTensorData  = in.exportData<nvcv::TensorDataStrided>();
    auto outTensorData = out.exportData<nvcv::TensorDataStrided>();

    if (!pva_solutions::utils::AreTensorRequirementsEqual(m_inTensorRequirements, inTensorData->cdata()) ||
        !pva_solutions::utils::AreTensorRequirementsEqual(m_outTensorRequirements, outTensorData->cdata()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output tensors must match requirements used to create the operator");
    }
    if ((static_cast<void *>(inTensorData->basePtr()) == nullptr) ||
        (static_cast<void *>(outTensorData->basePtr()) == nullptr))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor base pointers must not be NULL");
    }

    m_srcOp.setBase(static_cast<int32_t *>(static_cast<void *>(inTensorData->basePtr())));
    m_dstOp.setBase(static_cast<int32_t *>(static_cast<void *>(outTensorData->basePtr())));

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
        static_assert(!sizeof(S), "Unsupported stream type for RadarTranspose operator");
    }

    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

RadarTranspose::~RadarTranspose()
{
    CupvaCmdDestroy(&m_prog);
}

} // namespace pva_solutions::priv

NVCVStatus pvaRadarTransposeCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                   NVCVTensorRequirements const *const outTensorRequirements,
                                   PVARadarTransposeParams const *const params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        if (inTensorRequirements == nullptr || outTensorRequirements == nullptr || params == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Tensor requirements and params must not be NULL");
        }

        *handle = reinterpret_cast<NVCVOperatorHandle>(
            new pva_solutions::priv::RadarTranspose(inTensorRequirements, outTensorRequirements, params));
    });
}

template<typename StreamType>
NVCVStatus RadarTransposeSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVTensorHandle const in,
                                    NVCVTensorHandle out)
{
    return nvcv::ProtectCall([&] {
        nvcv::TensorWrapHandle input(in), output(out);
        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::RadarTranspose>(handle)(stream, input, output);
    });
}

template NVCVStatus RadarTransposeSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t, NVCVTensorHandle const,
                                                            NVCVTensorHandle);
template NVCVStatus RadarTransposeSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, NVCVTensorHandle const,
                                                           NVCVTensorHandle);

NVCVStatus pvaRadarTransposeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out)
{
    return RadarTransposeSubmitImpl(handle, stream, in, out);
}

NVCVStatus pvaRadarTransposeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out)
{
    return RadarTransposeSubmitImpl(handle, stream, in, out);
}
