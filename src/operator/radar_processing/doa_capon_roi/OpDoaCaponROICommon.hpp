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

#ifndef PVA_SOLUTIONS_PRIV_DOA_CAPON_ROI_COMMON_HPP
#define PVA_SOLUTIONS_PRIV_DOA_CAPON_ROI_COMMON_HPP

#include "IOperator.hpp"

#include "device/capon_roi_vmem_layout.h"

#include <nvcv/Tensor.hpp>

#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace pva_solutions::priv {

inline PVADoaCaponROIParams ValidateCommonParams(const PVADoaCaponROIParams *params)
{
    // Validate fixed dimensions, Q formats, thresholds, and Doppler tiling.
    PVADoaCaponROIParams normalized = *params;

    if (normalized.svDim != 16 || normalized.numAzBins != 16 || normalized.numElBins != 8 ||
        normalized.numPeaksPerROI != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DoaCaponROI: require svDim=16, numAzBins=16, numElBins=8, numPeaksPerROI=3");
    }
    if (normalized.inputQBits > 30 || normalized.steeringQBits > 30)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DoaCaponROI: inputQBits and steeringQBits must be in [0, 30]");
    }
    if (!std::isfinite(normalized.adpDiagConst) || normalized.adpDiagConst < 0.0F)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DoaCaponROI: adpDiagConst must be finite and nonnegative");
    }
    if (!std::isfinite(normalized.noiseFloorRatio) || normalized.noiseFloorRatio <= 0.0F)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DoaCaponROI: noiseFloorRatio must be finite and positive");
    }
    if (normalized.numDopplerBinsCovMat <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numDopplerBinsCovMat must be positive");
    }
    int32_t const partialTileSize = normalized.numDopplerBinsCovMat % BUF_MAX_NUM_SAMPLES;
    if (partialTileSize != 0 && (partialTileSize < 12 || partialTileSize % 4 != 0))
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT,
            "The partial numDopplerBinsCovMat tile must contain at least 12 bins and be divisible by 4");
    }

    return normalized;
}

template<typename S>
cupvaError_t SubmitCommand(S stream, cupvaCmd_t const *command)
{
    cupvaCmd_t const *commands[1] = {command};
    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        return CupvaStreamSubmit(stream, commands, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            return CupvaCudaStreamSubmit(stream, commands, nullptr, 1, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for Capon ROI operator");
    }
}

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_DOA_CAPON_ROI_COMMON_HPP
