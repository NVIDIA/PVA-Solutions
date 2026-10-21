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

#ifndef PVA_SOLUTIONS_PRIV_MINMAXLOC_HPP
#define PVA_SOLUTIONS_PRIV_MINMAXLOC_HPP

#include "IOperator.hpp"

#include "device/minmaxloc_param.h"

#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>

#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T>
class MinMaxLoc final : public IOperator
{
public:
    using ExecKey = std::tuple<cupva::impl::Context *>;

    explicit MinMaxLoc(NVCVTensorRequirements *tensorRequirements);

    ~MinMaxLoc();

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &src_t, const nvcv::Tensor &dstMinVal_t,
                    const nvcv::Tensor &dstMinLoc_t, const nvcv::Tensor &dstMinCnt_t, const nvcv::Tensor &dstMaxVal_t,
                    const nvcv::Tensor &dstMaxLoc_t, const nvcv::Tensor &dstMaxCnt_t);

private:
    NVCVTensorRequirements m_tensorRequirements;
    uint16_t m_imageWidth;
    uint16_t m_imageHeight;
    NVCVDataType m_dataType;

    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<T> m_dstMinValOp;
    cupva::OffsetPointer<T> m_dstMaxValOp;
    cupva::OffsetPointer<int> m_dstMinCntOp;
    cupva::OffsetPointer<int> m_dstMaxCntOp;
    cupva::OffsetPointer<unsigned short> m_dstMinLocOp;
    cupva::OffsetPointer<unsigned short> m_dstMaxLocOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;

    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_minMaxLocExecutables;
    static std::shared_ptr<cupva::Executable> GetMinMaxLocExec();
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_MINMAXLOC_HPP
