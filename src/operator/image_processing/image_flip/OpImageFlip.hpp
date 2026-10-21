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
 * @file OpImageFlip.hpp
 *
 * @brief Defines the private C++ Class for the ImageFlip operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_OP_IMAGE_FLIP_HPP
#define PVA_SOLUTIONS_PRIV_OP_IMAGE_FLIP_HPP

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>

#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class ImageFlip final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *, int>;

    explicit ImageFlip(NVCVImageRequirements *imageRequirements);

    ~ImageFlip();

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Image &out, int32_t flipDirection);

    void checkImageFormat(NVCVImageFormat format);

private:
    NVCVImageRequirements m_imageRequirements;
    int32_t m_imageWidth;
    int32_t m_imageHeight;
    int32_t m_inputLinePitch;
    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<T> m_dstOp;

    // Persistent executables and programs for each flip direction
    std::shared_ptr<cupva::Executable> m_execHorizontal;
    std::shared_ptr<cupva::Executable> m_execVertical;
    std::shared_ptr<cupva::Executable> m_execCombined;
    cupvaCmd_t m_progHorizontal;
    cupvaCmd_t m_progVertical;
    cupvaCmd_t m_progCombined;

    // Data flow setup methods for different flip directions
    template<typename TriggerType>
    void setupHorizontalFlipDataFlows(cupva::CmdProgram &prog, uint8_t *input_v, uint8_t *output_v, TriggerType &srcHdl,
                                      TriggerType &dstHdl);
    template<typename TriggerType>
    void setupVerticalFlipDataFlows(cupva::CmdProgram &prog, uint8_t *input_v, uint8_t *output_v, TriggerType &srcHdl,
                                    TriggerType &dstHdl);

    template<typename TriggerType>
    void setupCombinedFlipDataFlows(cupva::CmdProgram &prog, uint8_t *input_v, uint8_t *output_v, TriggerType &srcHdl2,
                                    TriggerType &dstHdl2);

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_ImageFlipExecutables;
    static std::shared_ptr<cupva::Executable> GetImageFlipExec(PVAFlipDirection flipDirection);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus ImageFlipSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle in, NVCVImageHandle out,
                               PVAFlipDirection flipDirection);

#endif // PVA_SOLUTIONS_PRIV_OP_IMAGE_FLIP_HPP
