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
 * @file OpORBDescExtractor.hpp
 *
 * @brief Defines the private C++ Class for the ORBDescExtractor operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_ORB_DESC_EXTRACTOR_HPP
#define PVA_SOLUTIONS_PRIV_ORB_DESC_EXTRACTOR_HPP

#include "IOperator.hpp"

#include "device/orb_desc_extr_params.h"
#include "nvcv/ImageFormat.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <OpORBDescExtractor.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

namespace pva_solutions::priv {

template<typename T>
class ORBDescExtractor final : public IOperator
{
public:
    using ExecDataKey = std::tuple<NVCVImageFormat, bool>;

    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit ORBDescExtractor(const NVCVImageRequirements *inImageRequirements,
                              const NVCVTensorRequirements *inCornersTensorRequirements,
                              const NVCVTensorRequirements *outDescriptorsTensorRequirements,
                              uint32_t disableRBRIEFFlag, NVCVBorderType borderType, int32_t borderValue);

    void checkImageFormat(NVCVImageFormat format);

    template<typename S>
    void operator()(S stream, const nvcv::Image &in, const nvcv::Tensor &inCorners, const nvcv::Tensor &outDescriptors);

    ~ORBDescExtractor();

private:
    NVCVImageRequirements m_inImageRequirements;
    NVCVTensorRequirements m_inCornersTensorRequirements;
    NVCVTensorRequirements m_outDescriptorsTensorRequirements;
    uint32_t m_disableRBRIEFFlag;
    NVCVBorderType m_borderType;
    int32_t m_borderValue;

    NVCVImageFormat m_imageFormat;

    cupva::OffsetPointer<T> m_inImageOp;
    cupva::OffsetPointer<float> m_inCornersOp;
    cupva::OffsetPointer<uint8_t> m_outDescriptorsOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_ORBDescExtractorExecutables;
    static std::shared_ptr<cupva::Executable> GetORBDescExtractorExec(ExecDataKey execDataKey);
};

} // namespace pva_solutions::priv

// Internal implementation function for submit operations
// Used by macro-generated __cupva/__cuda wrappers
template<typename StreamType>
NVCVStatus ORBDescExtractorSubmitImpl(NVCVOperatorHandle handle, StreamType stream, NVCVImageHandle inImageHandle,
                                      NVCVTensorHandle inCornersTensorHandle,
                                      NVCVTensorHandle outDescriptorsTensorHandle);

#endif // PVA_SOLUTIONS_PRIV_ORB_DESC_EXTRACTOR_HPP
