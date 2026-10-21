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

#include "OpDLInference.hpp"

#include <IOperator.hpp>
#include <TensorUtils.hpp>
#include <cupva_host.hpp>
#include <cupva_host_conversions.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cupva_host_cuda.h>
#include <cupva_host_scheduling.h>
#include <cupva_platform.h>
#include <detail/cupva_host_types_detail.h>

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cupva;

namespace pva_solutions::priv {

namespace detail {
// PVA_DL_INFERENCE_MAX_NUM_INPUTS and PVA_DL_INFERENCE_MAX_NUM_OUTPUTS are from OpDLInference.h
constexpr size_t MAX_DISPATCH_ARGS = PVA_DL_INFERENCE_MAX_NUM_INPUTS + PVA_DL_INFERENCE_MAX_NUM_OUTPUTS;

using NetworkFuncTypeForConstructor = void (*)(pva_dl_lib_engine_t *, ...);

template<typename Container, size_t... Is>
void call_network_func_with_indices(NetworkFuncTypeForConstructor network_func, pva_dl_lib_engine_t *engine,
                                    const Container &ptrs, std::index_sequence<Is...>)
{
    // Assumes ptrs.size() == sizeof...(Is)
    network_func(engine, ptrs[Is]...);
}

// Specialization for zero variadic arguments (only engine)
template<typename Container>
void call_network_func_with_indices(NetworkFuncTypeForConstructor network_func, pva_dl_lib_engine_t *engine,
                                    const Container & /*ptrs*/, std::index_sequence<>)
{
    network_func(engine);
}

// Dispatcher function template that calls call_network_func_with_indices with a specific N
template<size_t N>
void dispatch_network_func_call(NetworkFuncTypeForConstructor network_func, pva_dl_lib_engine_t *engine,
                                const std::vector<void *> &ptrs)
{
    call_network_func_with_indices(network_func, engine, ptrs, std::make_index_sequence<N>{});
}

// Helper to generate the array of dispatcher function pointers for the network function
template<size_t... Ns>
constexpr std::array<void (*)(NetworkFuncTypeForConstructor, pva_dl_lib_engine_t *, const std::vector<void *> &),
                     sizeof...(Ns)>
    make_network_func_dispatcher_array(std::index_sequence<Ns...>)
{
    return {&dispatch_network_func_call<Ns>...};
}

// The actual dispatch table for the network function, generated at compile time.
// It will contain functions for 0 to MAX_DISPATCH_ARGS variadic arguments.
static const auto network_func_dispatch_table =
    make_network_func_dispatcher_array(std::make_index_sequence<MAX_DISPATCH_ARGS + 1>{});

} // namespace detail

void *DLInference::getNetworkFunction(const std::string &networkName)
{
    std::string symbol = "main_graph_" + networkName;
    void *handle       = dlopen(nullptr, RTLD_LAZY); // Use nullptr for the main program
    if (!handle)
    {
        const char *error = dlerror();
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "%s", error ? error : "Failed to open main program");
    }

    void *sym = dlsym(handle, symbol.c_str());
    if (!sym)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Network function not found");
    }

    return sym;
}

DLInference::DLInference(PvaDLInferenceCreateParams *params)
    : m_networkFunc(nullptr)
{
    if (params == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to params must not be NULL");
    }
    if (params->numInReqs < 0 || params->numInReqs > PVA_DL_INFERENCE_MAX_NUM_INPUTS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numInReqs out of range");
    }
    if (params->numOutReqs < 0 || params->numOutReqs > PVA_DL_INFERENCE_MAX_NUM_OUTPUTS)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "numOutReqs out of range");
    }

    for (int i = 0; i < params->numInReqs; i++)
    {
        m_inTensorRequirements.push_back(*params->inReqs[i]);

        // Create proxy tensors for the inputs
        int64_t tensorSize = CalcTotalSizeBytes(nvcv::Requirements{params->inReqs[i]->mem}.cudaMem());
        m_inProxyTensorSizes.push_back(tensorSize);
        m_inProxyTensorPtrs.push_back(cupva::mem::Alloc(tensorSize));
    }
    for (int i = 0; i < params->numOutReqs; i++)
    {
        m_outTensorRequirements.push_back(*params->outReqs[i]);

        // Create proxy tensors for the outputs
        int64_t tensorSize = CalcTotalSizeBytes(nvcv::Requirements{params->outReqs[i]->mem}.cudaMem());
        m_outProxyTensorSizes.push_back(tensorSize);
        m_outProxyTensorPtrs.push_back(cupva::mem::Alloc(tensorSize));
    }
    if (params->networkName == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Network name must not be NULL");
    }
    m_networkName = params->networkName;
    m_networkFunc = getNetworkFunction(m_networkName);

    // Cast m_networkFunc to a C-style variadic function pointer after the engine argument.
    // The C function will still expect a pointer to the engine.
    using NetworkFuncType = void (*)(pva_dl_lib_engine_t *, ...);
    auto func_to_call     = reinterpret_cast<NetworkFuncType>(m_networkFunc);

    // Collect all proxy tensor base pointers
    std::vector<void *> allProxyTensorBasePtrs;
    allProxyTensorBasePtrs.reserve(m_inProxyTensorPtrs.size() + m_outProxyTensorPtrs.size());
    for (void *ptr : m_inProxyTensorPtrs)
    {
        allProxyTensorBasePtrs.push_back(ptr);
    }
    for (void *ptr : m_outProxyTensorPtrs)
    {
        allProxyTensorBasePtrs.push_back(ptr);
    }

    const size_t numTotalProxyPtrs = allProxyTensorBasePtrs.size();

    if (numTotalProxyPtrs <= detail::MAX_DISPATCH_ARGS) // MAX_DISPATCH_ARGS is already defined and suitable
    {
        detail::network_func_dispatch_table[numTotalProxyPtrs](func_to_call, &m_engine, allProxyTensorBasePtrs);
    }
    else
    {
        throw nvcv::Exception(
            nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
            ("Unsupported total number of input/output proxy tensors for network function setup (got " +
             std::to_string(numTotalProxyPtrs) + ", max " + std::to_string(detail::MAX_DISPATCH_ARGS) + " supported).")
                .c_str());
    }
}

template<typename S>
void DLInference::operator()(S stream, NVCVTensorHandle *inTensors, NVCVTensorHandle *outTensors)
{
    // submit copy cmds for input tensors
    submitCopyCmds(stream, inTensors, true);
    // Use the passed engine reference for submit
    submit(stream, m_engine.num_commands);
    // submit copy cmds for output tensors
    submitCopyCmds(stream, outTensors, false);
}

template<typename S>
void DLInference::submit(S stream, int numOnnxCmds)
{
    std::vector<cupvaCmd_t *> cmd_ptrs;
    cmd_ptrs.reserve(numOnnxCmds);

    for (int i = 0; i < numOnnxCmds; i++)
    {
        cmd_ptrs.push_back(&(m_engine.cmds[i]));
    }

    cupvaError_t err;
    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, cmd_ptrs.data(), nullptr, numOnnxCmds, CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, cmd_ptrs.data(), nullptr, numOnnxCmds, CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
            return;
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for DLInference operator");
    }

    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed");
    }
}

template<typename S>
void DLInference::submitCopyCmds(S stream, NVCVTensorHandle *tensors, bool isInput)
{
    std::vector<cupvaCmd_t> cmds;
    const auto &proxyTensorPtrs  = isInput ? m_inProxyTensorPtrs : m_outProxyTensorPtrs;
    const auto &proxyTensorSizes = isInput ? m_inProxyTensorSizes : m_outProxyTensorSizes;

    cmds.reserve(proxyTensorPtrs.size()); // Pre-allocate

    for (size_t i = 0; i < proxyTensorPtrs.size(); ++i)
    {
        cupvaCmd_t cmd;
        cupvaCmdMemcpyBufferParams_t cmdMemcpyBufferParams = {0};
        nvcv::Tensor tensor                                = nvcv::TensorWrapHandle(tensors[i]);
        auto tensorData                                    = tensor.exportData<nvcv::TensorDataStrided>();

        if (isInput)
        {
            cmdMemcpyBufferParams.ptrSrc = static_cast<void *>(tensorData->basePtr());
            cmdMemcpyBufferParams.ptrDst = proxyTensorPtrs[i];
        }
        else // for output
        {
            cmdMemcpyBufferParams.ptrSrc = proxyTensorPtrs[i];
            cmdMemcpyBufferParams.ptrDst = static_cast<void *>(tensorData->basePtr());
        }
        CupvaCmdMemcpyCreate(&cmd, &cmdMemcpyBufferParams, proxyTensorSizes[i]);
        cmds.push_back(cmd);
    }

    std::vector<cupvaCmd_t *> cmd_ptrs;
    cmd_ptrs.reserve(cmds.size());
    for (size_t i = 0; i < cmds.size(); ++i)
    {
        cmd_ptrs.push_back(&cmds[i]);
    }

    cupvaError_t err;
    if constexpr (std::is_same_v<S, cupvaStream_t>)
    {
        err = CupvaStreamSubmit(stream, cmd_ptrs.data(), nullptr, cmd_ptrs.size(), CUPVA_IN_ORDER, -1, -1);
    }
    else if constexpr (std::is_same_v<S, cudaStream_t>)
    {
        if constexpr (kCupvaHasCudaStreamSupport)
        {
            err = CupvaCudaStreamSubmit(stream, cmd_ptrs.data(), nullptr, cmd_ptrs.size(), CUPVA_IN_ORDER, -1, -1);
        }
        else
        {
            ThrowCudaStreamNotSupported();
            return;
        }
    }
    else
    {
        static_assert(!sizeof(S), "Unsupported stream type for DLInference operator");
    }

    // Destroy CmdMemcpy commands
    for (int i = 0; i < cmds.size(); i++)
    {
        CupvaCmdDestroy(&cmds[i]);
    }

    if (err != CUPVA_ERROR_NONE)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "PVA Cmd Submit failed in submitCopyCmds");
    }
}

DLInference::~DLInference()
{
    cupvaError_t err;
    bool cleanup_success = true;

    // Free all allocated device memory (constants + IO buffers)
    // This is a unified list of ACTUAL allocations, no duplicates
    for (int i = 0; i < m_engine.num_allocs; i++)
    {
        err = CupvaMemFree(m_engine.allocated_ptrs[i]);
        if (err != CUPVA_ERROR_NONE)
        {
            printf("Error: CupvaMemFree failed for allocated_ptrs[%d] (ptr=%p) with error code %d\n", i,
                   m_engine.allocated_ptrs[i], err);
            cleanup_success = false;
        }
    }
    free(m_engine.allocated_ptrs);

    for (int i = 0; i < m_engine.num_buffers; i++)
    {
        err = CupvaOffsetPointerDestroy(&m_engine.offset_ptrs[i]);
        if (err != CUPVA_ERROR_NONE)
        {
            printf("Error: CupvaOffsetPointerDestroy failed for offset_ptrs[%d] with error code %d\n", i, err);
            cleanup_success = false;
        }
    }
    free(m_engine.buffer_bases);
    free(m_engine.offset_ptrs);
    free(m_engine.single_sizes);

    // Free commands and executables
    for (int i = 0; i < m_engine.num_commands; i++)
    {
        err = CupvaCmdDestroy(&(m_engine.cmds[i]));
        if (err != CUPVA_ERROR_NONE)
        {
            printf("Error: CupvaCmdDestroy failed for cmds[%d] with error code %d\n", i, err);
            cleanup_success = false;
        }

        err = CupvaExecutableDestroy(m_engine.execs[i]);
        if (err != CUPVA_ERROR_NONE)
        {
            printf("Error: CupvaExecutableDestroy failed for execs[%d] with error code %d\n", i, err);
            cleanup_success = false;
        }
    }
    free(m_engine.execs);
    free(m_engine.cmds);

    free(m_engine.task_buffer_offsets);

    if (cleanup_success)
    {
        printf("Clean up completed successfully\n");
    }
    else
    {
        printf("Clean up with errors\n");
    }

    for (int i = 0; i < m_inProxyTensorPtrs.size(); i++)
    {
        CupvaMemFree(m_inProxyTensorPtrs[i]);
    }
    for (int i = 0; i < m_outProxyTensorPtrs.size(); i++)
    {
        CupvaMemFree(m_outProxyTensorPtrs[i]);
    }
}

} // namespace pva_solutions::priv

NVCVStatus pvaDLInferenceCreate(NVCVOperatorHandle *handle, PvaDLInferenceCreateParams *params)
{
    return nvcv::ProtectCall([&] {
        if (handle == nullptr)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Pointer to NVCVOperator handle must not be NULL");
        }
        *handle = reinterpret_cast<NVCVOperatorHandle>(new pva_solutions::priv::DLInference(params));
    });
}

template<typename StreamType>
NVCVStatus DLInferenceSubmitImpl(NVCVOperatorHandle handle, StreamType stream, PvaDLInferenceSubmitParams *params)
{
    return nvcv::ProtectCall([&] {
        auto &op = pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DLInference>(handle);

        // 1. Check total number of tensors against creation requirements
        const auto &inReqs  = op.getInTensorRequirements();
        const auto &outReqs = op.getOutTensorRequirements();

        if (static_cast<size_t>(params->numInTensors) != inReqs.size())
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                ("Number of provided input tensors (" + std::to_string(params->numInTensors) +
                 ") does not match operator creation requirements (" + std::to_string(inReqs.size()) + ").")
                    .c_str());
        }
        if (static_cast<size_t>(params->numOutTensors) != outReqs.size())
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                ("Number of provided output tensors (" + std::to_string(params->numOutTensors) +
                 ") does not match operator creation requirements (" + std::to_string(outReqs.size()) + ").")
                    .c_str());
        }

        // 2. Iterate and Validate Input Tensors
        for (int i = 0; i < params->numInTensors; ++i)
        {
            if (params->inTensors[i] == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                                      ("Input NVCVTensorHandle is NULL at index " + std::to_string(i)).c_str());
            }
            nvcv::TensorWrapHandle tensor_handle(params->inTensors[i]);
            auto optional_data =
                static_cast<const nvcv::Tensor &>(tensor_handle).template exportData<nvcv::TensorDataStrided>();

            if (!optional_data)
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INTERNAL, "%s",
                    ("Failed to export tensor data for input tensor at index " + std::to_string(i)).c_str());
            }
            nvcv::TensorDataStrided &tensor_data = *optional_data;

            if (tensor_data.basePtr() == nullptr)
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                    ("Input tensor base pointer is NULL after export at index " + std::to_string(i)).c_str());
            }

            if (!pva_solutions::utils::AreTensorRequirementsEqual(inReqs[i], tensor_data.cdata()))
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                    ("Input tensor at index " + std::to_string(i) + " does not match operator creation requirements.")
                        .c_str());
            }
        }

        // 3. Iterate and Validate Output Tensors
        for (int i = 0; i < params->numOutTensors; ++i)
        {
            if (params->outTensors[i] == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                                      ("Output NVCVTensorHandle is NULL at index " + std::to_string(i)).c_str());
            }
            nvcv::TensorWrapHandle tensor_handle(params->outTensors[i]);
            auto optional_data =
                static_cast<const nvcv::Tensor &>(tensor_handle).template exportData<nvcv::TensorDataStrided>();

            if (!optional_data)
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INTERNAL, "%s",
                    ("Failed to export tensor data for output tensor at index " + std::to_string(i)).c_str());
            }
            nvcv::TensorDataStrided &tensor_data = *optional_data;

            if (tensor_data.basePtr() == nullptr)
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                    ("Output tensor base pointer is NULL after export at index " + std::to_string(i)).c_str());
            }

            if (!pva_solutions::utils::AreTensorRequirementsEqual(outReqs[i], tensor_data.cdata()))
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT, "%s",
                    ("Output tensor at index " + std::to_string(i) + " does not match operator creation requirements.")
                        .c_str());
            }
        }

        pva_solutions::priv::ToDynamicRef<pva_solutions::priv::DLInference>(handle)(stream, params->inTensors,
                                                                                    params->outTensors);
    });
}

// Explicit template instantiation for supported stream types
template NVCVStatus DLInferenceSubmitImpl<cupvaStream_t>(NVCVOperatorHandle, cupvaStream_t,
                                                         PvaDLInferenceSubmitParams *);
template NVCVStatus DLInferenceSubmitImpl<cudaStream_t>(NVCVOperatorHandle, cudaStream_t, PvaDLInferenceSubmitParams *);

// C++ overload implementations
NVCVStatus pvaDLInferenceSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, PvaDLInferenceSubmitParams *params)
{
    return DLInferenceSubmitImpl(handle, stream, params);
}

NVCVStatus pvaDLInferenceSubmit(NVCVOperatorHandle handle, cudaStream_t stream, PvaDLInferenceSubmitParams *params)
{
    return DLInferenceSubmitImpl(handle, stream, params);
}
