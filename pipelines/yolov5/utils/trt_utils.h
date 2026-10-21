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

#ifndef YOLOV5_TRT_UTILS_H
#define YOLOV5_TRT_UTILS_H

#include <NvInfer.h>

#include <string>

namespace yolov5_common {

class TRTLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *message) noexcept override;
};

nvinfer1::ICudaEngine *loadEngine(const std::string &path, nvinfer1::IRuntime &runtime);
bool saveEngine(nvinfer1::ICudaEngine &engine, const std::string &path);

} // namespace yolov5_common

#endif // YOLOV5_TRT_UTILS_H
