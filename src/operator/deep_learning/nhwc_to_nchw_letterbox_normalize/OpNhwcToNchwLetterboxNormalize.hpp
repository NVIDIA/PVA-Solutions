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

/**
 * @file OpNhwcToNchwLetterboxNormalize.hpp
 *
 * @brief Private C++ class for the fused NHWC->NCHW + letterbox + normalize operator.
 */

#pragma once

#include "IOperator.hpp"

#include "device/nhwc_to_nchw_letterbox_normalize_param.h"

#include <cupva_host.hpp>
#include <nvcv/Tensor.hpp>

#include <OpNhwcToNchwLetterboxNormalize.h>
#include <PvaOperatorTypes.h>

#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

/**
 * Fused NHWC->NCHW + letterbox + normalize PVA operator.
 *
 * Input:  uint8 HWC (imgH x imgW x 3) - aspect-ratio-resized image
 * Output: float32 CHW (3 x outH x outW) - letterboxed, normalized, transposed
 *
 * DMA configuration:
 *   Input  RDF: uint8 HWC, TRANS_MODE_1, tiles over image region
 *   Output RDFs: 3x float32 planar, TRANS_MODE_1, with letterbox offset in DRAM base
 *
 * Under RESTORE_ON_SUBMIT, padding is restored in stream order on every
 * submission before the PVA image-region write.
 */
class NhwcToNchwLetterboxNormalize final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit NhwcToNchwLetterboxNormalize(NVCVTensorRequirements const *inTensorReqs,
                                          NVCVTensorRequirements const *outTensorReqs, float scale, float padValue,
                                          PVANhwcToNchwLetterboxNormalizePaddingPolicy paddingPolicy);

    template<typename S>
    void operator()(S stream, nvcv::Tensor const &in, nvcv::Tensor const &out);

    ~NhwcToNchwLetterboxNormalize();

private:
    NVCVTensorRequirements m_inTensorRequirements;
    NVCVTensorRequirements m_outTensorRequirements;

    cupva::OffsetPointer<uint8_t> m_srcOp; /* Input: uint8 HWC image */
    cupva::OffsetPointer<float> m_dstC0Op; /* Output: channel 0 (R) plane */
    cupva::OffsetPointer<float> m_dstC1Op; /* Output: channel 1 (G) plane */
    cupva::OffsetPointer<float> m_dstC2Op; /* Output: channel 2 (B) plane */

    /* Cached geometry */
    int32_t m_imgW;
    int32_t m_imgH;
    int32_t m_outW;
    int32_t m_outH;
    int32_t m_padTop;
    int32_t m_padLeft;
    int32_t m_channels;
    float m_padValue;
    PVANhwcToNchwLetterboxNormalizePaddingPolicy m_paddingPolicy;

    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    static std::mutex s_execMapMutex;
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_executables;
    static std::shared_ptr<cupva::Executable> GetExec();
};

} // namespace pva_solutions::priv
