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

/**
 * @file OpBackgroundSubtractor.hpp
 *
 * @brief Defines the private C++ Class for the BackgroundSubtractor operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_BACKGROUNDSUBTRACTOR_HPP
#define PVA_SOLUTIONS_PRIV_BACKGROUNDSUBTRACTOR_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>

#include <OpBackgroundSubtractor.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

class BackgroundSubtractorBase : public IOperator
{
    PVAGMMDataType m_gmmDataType;

public:
    BackgroundSubtractorBase(PVAGMMDataType gmmDataType);
    PVAGMMDataType getGMMDataType();
};

template<typename T, typename GMMModelType>
class BackgroundSubtractor final : public BackgroundSubtractorBase
{
public:
    using ExecKey        = std::tuple<cupva::impl::Context *, size_t>;
    using ModelCountType = std::conditional_t<std::is_same_v<GMMModelType, float>, int32_t, int16_t>;

    explicit BackgroundSubtractor(NVCVImageRequirements *imageRequirements, NVCVImageRequirements *maskRequirements,
                                  PVAGMMDataType gmmDataType);

    ~BackgroundSubtractor();

    void setParameters(float learningRate, float varThreshold, bool enableShadowDetection, uint8_t shadowPixelValue);

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, const nvcv::Image *outBGPtr,
                    float learningRate, float varThreshold, bool enableShadowDetection, uint8_t shadowPixelValue);
    void setParameters(float learningRate, float varThreshold, bool enableShadowDetection, uint8_t shadowPixelValue,
                       bool enableBackgroundImage);

    static std::shared_ptr<cupva::Executable> GetBackgroundSubtractorExec();

private:
    NVCVImageRequirements m_imageRequirements;
    NVCVImageRequirements m_maskRequirements;
    int32_t m_width;
    int32_t m_height;
    int32_t m_inputLinePitch;
    int32_t m_outputLinePitch;
    int32_t m_frameNum;

    GMMModelType *m_gmmWeightsNMod_d;
    GMMModelType *m_gmmVariances_d;
    GMMModelType *m_gmmMeans_d;
    GMMModelType *m_gmmWeightsNMod_h;
    GMMModelType *m_gmmVariances_h;
    GMMModelType *m_gmmMeans_h;

    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<T> m_dstOp;
    cupva::OffsetPointer<T> m_dstBGOp;
    cupvaCmd_t m_progWithBG, m_progWOBG;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_backgroundSubtractorExecutables;

    std::shared_ptr<cupva::Executable> m_execShared;
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus BackgroundSubtractorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in,
                                          NVCVImageHandle outMask, NVCVImageHandle outBGImage, float learningRate,
                                          float varThreshold, bool enableShadowDetection, uint8_t shadowPixelValue);

#endif // PVA_SOLUTIONS_PRIV_BACKGROUNDSUBTRACTOR_HPP
