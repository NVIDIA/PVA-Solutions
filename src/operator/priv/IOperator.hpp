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
 * @file IOperator.hpp
 *
 * @brief Defines the private C++ Class for the operator interface.
 */

#ifndef PVA_SOLUTIONS_PRIV_IOPERATOR_HPP
#define PVA_SOLUTIONS_PRIV_IOPERATOR_HPP

#include <cupva_host.hpp>
#include <nvcv/Exception.hpp>

#include <PvaOperator.h>

#include <map>
#include <mutex>

namespace pva_solutions::priv {

// PVA SDK extended feature support checks
// CUPVA_HAS_CUDA_STREAM_SUPPORT is defined by CMake based on PVA SDK version
// CUDA stream support (CupvaCudaStreamSubmit) requirements:
//   - PVA SDK 2.7.0+
//   - Jetpack 7+
//   - DriveOS 7+
inline constexpr bool kCupvaHasCudaStreamSupport = CUPVA_HAS_CUDA_STREAM_SUPPORT;

// Helper function to throw exception for unsupported CUPVA features
[[noreturn]] inline void ThrowCudaStreamNotSupported()
{
    throw nvcv::Exception(nvcv::Status::ERROR_NOT_IMPLEMENTED,
                          "CUDA stream support requires PVA SDK 2.7.0+ (Jetpack 7+ / DriveOS 7+)");
}

class IOperator
{
public:
    using HandleType    = NVCVOperatorHandle;
    using InterfaceType = IOperator;

    virtual ~IOperator() = default;

    HandleType handle() const
    {
        return reinterpret_cast<HandleType>(const_cast<IOperator *>(static_cast<const IOperator *>(this)));
    }

    struct ExecutableData
    {
        const void *data;
        int32_t size;
    };

    // Custom deleter to remove the Executable entry from the map
    template<typename Key>
    struct ExecutableDeleter
    {
        Key key;
        std::map<Key, std::weak_ptr<cupva::Executable>> &execMap;
        std::mutex &execMapMutex;

        ExecutableDeleter(Key k, std::map<Key, std::weak_ptr<cupva::Executable>> &map, std::mutex &mutex)
            : key(k)
            , execMap(map)
            , execMapMutex(mutex)
        {
        }

        void operator()(cupva::Executable *ptr)
        {
            std::lock_guard<std::mutex> lock(execMapMutex);
            auto it = execMap.find(key);
            if (it != execMap.end() && it->second.expired())
            {
                execMap.erase(it);
            }
            delete ptr;
        }
    };

protected:
    template<typename Key>
    static std::shared_ptr<cupva::Executable> GetExecutable(ExecutableData execData, Key execKey,
                                                            std::map<Key, std::weak_ptr<cupva::Executable>> &execMap,
                                                            std::mutex &execMapMutex)
    {
        std::lock_guard<std::mutex> lock(execMapMutex);

        auto it = execMap.find(execKey);
        if (it != execMap.end())
        {
            auto sharedExec = it->second.lock();
            if (sharedExec)
            {
                return sharedExec;
            }
        }

        auto rawExec     = cupva::Executable::Create(execData.data, execData.size);
        auto exec        = std::shared_ptr<cupva::Executable>(new cupva::Executable(std::move(rawExec)),
                                                       ExecutableDeleter<Key>(execKey, execMap, execMapMutex));
        execMap[execKey] = exec;
        return exec;
    }
};

IOperator *ToOperatorPtr(void *handle);

template<class T>
inline T *ToDynamicPtr(NVCVOperatorHandle h)
{
    return dynamic_cast<T *>(ToOperatorPtr(h));
}

template<class T>
inline T &ToDynamicRef(NVCVOperatorHandle h)
{
    if (h == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Handle cannot be NULL");
    }

    if (T *child = ToDynamicPtr<T>(h))
    {
        return *child;
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE,
                              "Handle doesn't correspond to the requested object or was already destroyed.");
    }
}

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_IOPERATOR_HPP
