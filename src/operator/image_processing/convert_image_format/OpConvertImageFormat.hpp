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

/**
 * @file OpConvertImageFormat.hpp
 *
 * @brief Defines the private C++ Class for the ConvertImageFormat operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_CONVERTIMAGEFORMAT_HPP
#define PVA_SOLUTIONS_PRIV_CONVERTIMAGEFORMAT_HPP

#include "IOperator.hpp"

#include "device/convert_image_format_param.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>

#include <OpConvertImageFormat.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>

#include <cstdint>

namespace pva_solutions::priv {

template<typename T>
class ConvertImageFormat final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit ConvertImageFormat(NVCVImageRequirements const *const inImageRequirements,
                                NVCVImageRequirements const *const outImageRequirements);

    ~ConvertImageFormat();

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &out);

private:
    NVCVImageRequirements m_inImageRequirements;
    NVCVImageRequirements m_outImageRequirements;
    int32_t m_width;
    int32_t m_height;
    int32_t m_inputLinePitch;
    int32_t m_outputLinePitch;
    NVCVImageFormat m_inFormat;
    NVCVImageFormat m_outFormat;
    int32_t m_inPlaneCount;
    int32_t m_outPlaneCount;

    cupva::OffsetPointer<T> m_srcOps[MAX_CHANNEL_COUNT];
    cupva::OffsetPointer<T> m_dstOps[MAX_CHANNEL_COUNT];
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_convertImageFormatExecutables;
    static std::shared_ptr<cupva::Executable> GetConvertImageFormatExec();

    ImageFormatConversionCode getImageFormatConversionCode(NVCVImageFormat inFormat, NVCVImageFormat outFormat);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus ConvertImageFormatSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle const in,
                                        NVCVImageHandle out);

#endif // PVA_SOLUTIONS_PRIV_CONVERTIMAGEFORMAT_HPP
