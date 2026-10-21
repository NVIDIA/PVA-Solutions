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

/*
 * Common INT8 entropy calibrator for YOLOv5 pipelines.
 */

#ifndef YOLOV5_COMMON_INT8_CALIBRATOR_H
#define YOLOV5_COMMON_INT8_CALIBRATOR_H

#include "int8_preprocess.h"
#include "yolov5_common.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <string>
#include <vector>

namespace yolov5_common {

/**
 * INT8 entropy calibrator with resize-based NCHW preprocessing.
 * Supports PNG and .bin (pre-decoded RGB) files.
 */
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8Calibrator(const std::string &imageDir, int batchSize, int channels, int height, int width,
                   const std::string &cacheFile, int maxImages = 100);
    ~Int8Calibrator() override;

    int32_t getBatchSize() const noexcept override;
    bool getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept override;
    void const *readCalibrationCache(std::size_t &length) noexcept override;
    void writeCalibrationCache(void const *cache, std::size_t length) noexcept override;

private:
    int mBatchSize;
    int mChannels;
    int mHeight;
    int mWidth;
    std::string mCacheFile;
    int mCurrentIdx;
    void *mDeviceBuffer;
    std::vector<std::string> mImageFiles;
    std::vector<char> mCalibCache;
};

} // namespace yolov5_common

#endif // YOLOV5_COMMON_INT8_CALIBRATOR_H
