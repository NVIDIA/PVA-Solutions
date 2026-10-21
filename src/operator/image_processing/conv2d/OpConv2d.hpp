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
 * @file OpConv2d.hpp
 *
 * @brief Defines the private C++ Class for the Conv2d operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_CONV2D_HPP
#define PVA_SOLUTIONS_PRIV_CONV2D_HPP

#include "IOperator.hpp"

#include "nvcv/Tensor.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpConv2d.h>
#include <cupva_host_scheduling.h>
#include <nvcv/BorderType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace pva_solutions::priv {

template<typename T, typename Derived>
class BaseConv2d : public IOperator
{
public:
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));
    std::shared_ptr<cupva::Executable> GetExec(ExecDataKey execDataKey);
    BaseConv2d(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize, NVCVBorderType borderMode,
               int32_t borderValue);
    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out);

    // Method to initialize the program after construction
    //
    // IMPORTANT CAVEAT: This method should only be called after the derived class
    // constructor has completed. CRTP methods (like prepareKernel) can only
    // be safely called after the entire object is fully constructed.
    //
    // Initialization order:
    // 1. Memory allocation
    // 2. Initialization list execution (base class + member variables)
    // 3. Base class constructor body
    // 4. Derived class constructor body
    // 5. NOW it's safe to call initializeProgram() and CRTP methods
    //
    // Calling CRTP methods from base class constructors is dangerous because
    // the derived class part of the object isn't ready yet, even though the
    // base class part is fully initialized.
    // Example:
    /*
        template<typename T, typename Derived>
        class BaseConv2d : public IOperator
        {
        public:
            BaseConv2d(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize,
                    NVCVBorderType borderMode, int32_t borderValue)
                : m_tensorRequirements(*tensorRequirements)
                , m_kernelSize(kernelSize)
                , m_borderMode(borderMode)
                , m_borderValue(borderValue)
            {
                // DANGEROUS: Calling CRTP method before derived class is fully constructed
                this->initializeProgram(); // This calls prepareKernel via CRTP!
            }

            void initializeProgram()
            {
                // ... setup code ...

                // This calls the derived class's prepareKernel method
                static_cast<Derived *>(this)->prepareKernel(knl_ref.data(), kernelSize, qbits);

                // ... more setup code ...
            }
    */
    void initializeProgram();

protected:
    NVCVTensorRequirements m_tensorRequirements;
    cupva::OffsetPointer<T> m_srcOp;
    cupva::OffsetPointer<T> m_dstOp;
    std::shared_ptr<cupva::Executable> m_exec;
    cupvaCmd_t m_prog;
    uint32_t m_kernelSize;
    NVCVBorderType m_borderMode;
    int32_t m_borderValue;
};

template<typename T>
class Conv2d : public BaseConv2d<T, Conv2d<T>>
{
public:
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    explicit Conv2d(NVCVTensorRequirements *tensorRequirements, int32_t kernelSize, const NVCVBorderType borderMode,
                    int32_t borderValue, NVCVTensorHandle kernelCoefficients);

    ~Conv2d();

    static auto &GetExecMap()
    {
        return s_conv2dExecutables;
    }
    static auto &GetExecMutex()
    {
        return s_execMapMutex;
    }
    void prepareKernel(void *outKernel, uint32_t kernelSize, int qbits);

    template<typename S>
    void operator()(S stream, const nvcv::Tensor &in, const nvcv::Tensor &out);

private:
    // Mutex to protect shared Executable map
    static std::mutex s_execMapMutex;
    // Map to store executables with reference counting
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_conv2dExecutables;

    // Store kernel coefficients for use in prepareKernel
    NVCVTensorHandle m_kernelCoefficients;
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_CONV2D_HPP
