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

#pragma once

#include "IOperator.hpp"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>

#include <OpBlToPl.h>
#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>

namespace pva_solutions::priv {

class Bltopl final : public IOperator

{
public:
    // The alignment value for BL's width
    static constexpr int32_t kBL_ALIGN{64};
    // The minimum image size that the device code can handle is 32x2 for either plane. Then
    // The height should be doubled when the image height represents the Y plane.
    static constexpr int32_t kMinImageWidth{32};
    static constexpr int32_t kMinImageHeight{4};

    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit Bltopl(NVCVImageRequirements const *const imageRequirements);

    template<typename S>
    void operator()(S stream, nvcv::Image const &in, nvcv::Image const &out);

    ~Bltopl();

private:
    NVCVImageRequirements m_inImageRequirements;
    NVCVImageRequirements m_outImageRequirements;
    std::array<cupva::OffsetPointer<uint8_t>, 2> m_srcOp;
    std::array<cupva::OffsetPointer<uint8_t>, 2> m_dstOp;
    cupvaCmd_t m_prog;
    std::shared_ptr<cupva::Executable> m_exec;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_bltoplExecutables;
    static std::shared_ptr<cupva::Executable> GetBltoplExec();
};
} // namespace pva_solutions::priv
