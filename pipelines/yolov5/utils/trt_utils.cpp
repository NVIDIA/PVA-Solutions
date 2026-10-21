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

#include "trt_utils.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

namespace yolov5_common {

void TRTLogger::log(Severity severity, const char *message) noexcept
{
    if (severity <= Severity::kWARNING)
        printf("[TensorRT] %s\n", message);
}

nvinfer1::ICudaEngine *loadEngine(const std::string &path, nvinfer1::IRuntime &runtime)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return nullptr;
    const std::streamsize size = file.tellg();
    if (size <= 0)
        return nullptr;
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size))
        return nullptr;
    return runtime.deserializeCudaEngine(buffer.data(), buffer.size());
}

bool saveEngine(nvinfer1::ICudaEngine &engine, const std::string &path)
{
    std::unique_ptr<nvinfer1::IHostMemory> serialized(engine.serialize());
    if (!serialized)
        return false;
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
        return false;
    file.write(static_cast<const char *>(serialized->data()), serialized->size());
    return file.good();
}

} // namespace yolov5_common
