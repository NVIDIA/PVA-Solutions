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
 * Common INT8 calibrator for YOLOv5 pipelines.
 */

#include "int8_calibrator.h"

#include "image_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace yolov5_common {

namespace {

constexpr int64_t MAX_BIN_PIXELS = 4LL * 4096 * 4096;

} // namespace

Int8Calibrator::Int8Calibrator(const std::string &imageDir, int batchSize, int channels, int height, int width,
                               const std::string &cacheFile, int maxImages)
    : mBatchSize(batchSize)
    , mChannels(channels)
    , mHeight(height)
    , mWidth(width)
    , mCacheFile(cacheFile)
    , mCurrentIdx(0)
    , mDeviceBuffer(nullptr)
{
    mImageFiles = getImageFiles(imageDir);
    if (mImageFiles.size() > static_cast<size_t>(maxImages))
        mImageFiles.resize(maxImages);
    printf("  INT8 calibrator: %zu images from %s\n", mImageFiles.size(), imageDir.c_str());

    size_t bufferSize  = batchSize * channels * height * width * sizeof(float);
    cudaError_t status = cudaMalloc(&mDeviceBuffer, bufferSize);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Failed to allocate INT8 calibration buffer: ") +
                                 cudaGetErrorString(status));
}

Int8Calibrator::~Int8Calibrator()
{
    if (mDeviceBuffer)
        cudaFree(mDeviceBuffer);
}

int32_t Int8Calibrator::getBatchSize() const noexcept
{
    return mBatchSize;
}

bool Int8Calibrator::getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept
{
    (void)names;
    if (!bindings || nbBindings < 1)
        return false;

    while (mCurrentIdx < static_cast<int>(mImageFiles.size()))
    {
        const std::string &path = mImageFiles[mCurrentIdx];
        unsigned char *imgData  = nullptr;
        int imgW = 0, imgH = 0;

        bool isBin = false;
        if (path.size() >= 4u)
        {
            std::string suf = path.substr(path.size() - 4);
            isBin           = (suf[0] == '.' && (suf[1] == 'b' || suf[1] == 'B') && (suf[2] == 'i' || suf[2] == 'I') &&
                     (suf[3] == 'n' || suf[3] == 'N'));
        }

        if (isBin)
        {
            FILE *f = fopen(path.c_str(), "rb");
            if (!f)
            {
                fprintf(stderr, "  Calibrator: Failed to open %s\n", path.c_str());
                ++mCurrentIdx;
                continue;
            }
            int32_t w = 0, h = 0;
            if (fread(&w, sizeof(int32_t), 1, f) != 1 || fread(&h, sizeof(int32_t), 1, f) != 1 || w <= 0 || h <= 0 ||
                static_cast<int64_t>(w) * h > MAX_BIN_PIXELS)
            {
                fclose(f);
                fprintf(stderr, "  Calibrator: Invalid .bin dimensions %dx%d in %s\n", w, h, path.c_str());
                ++mCurrentIdx;
                continue;
            }
            size_t size = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
            imgData     = static_cast<unsigned char *>(malloc(size));
            if (!imgData || fread(imgData, 1, size, f) != size)
            {
                free(imgData);
                fclose(f);
                fprintf(stderr, "  Calibrator: Failed to read %s\n", path.c_str());
                ++mCurrentIdx;
                continue;
            }
            fclose(f);
            imgW = w;
            imgH = h;
        }
        else
        {
            imgData = loadPngRgb(path.c_str(), &imgW, &imgH);
            if (!imgData)
            {
                fprintf(stderr, "  Calibrator: Failed to load %s\n", path.c_str());
                ++mCurrentIdx;
                continue;
            }
        }

        std::vector<float> hostBuffer(static_cast<size_t>(mChannels) * mHeight * mWidth);
        const int resizeStatus = resizeToNCHWFloat(imgData, imgW, imgH, mWidth, mHeight, hostBuffer.data());
        freeImage(imgData);
        if (resizeStatus != 0)
        {
            fprintf(stderr, "  Calibrator: Failed to preprocess %s\n", path.c_str());
            ++mCurrentIdx;
            continue;
        }

        cudaError_t status =
            cudaMemcpy(mDeviceBuffer, hostBuffer.data(), hostBuffer.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
        {
            fprintf(stderr, "  Calibrator: CUDA copy failed for %s: %s\n", path.c_str(), cudaGetErrorString(status));
            return false;
        }
        bindings[0] = mDeviceBuffer;

        if (mCurrentIdx % 10 == 0)
            printf("  Calibrating: %d/%zu\n", mCurrentIdx + 1, mImageFiles.size());
        ++mCurrentIdx;
        return true;
    }
    return false;
}

void const *Int8Calibrator::readCalibrationCache(std::size_t &length) noexcept
{
    mCalibCache.clear();
    std::ifstream file(mCacheFile, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, std::ios::end);
        length = file.tellg();
        file.seekg(0, std::ios::beg);
        mCalibCache.resize(length);
        file.read(mCalibCache.data(), length);
        printf("  Loaded calibration cache: %s (%zu bytes)\n", mCacheFile.c_str(), length);
        return mCalibCache.data();
    }
    printf("  No calibration cache found, will run calibration passes\n");
    length = 0;
    return nullptr;
}

void Int8Calibrator::writeCalibrationCache(void const *cache, std::size_t length) noexcept
{
    std::ofstream file(mCacheFile, std::ios::binary);
    if (file.good())
    {
        file.write(static_cast<const char *>(cache), length);
        printf("  Saved calibration cache: %s (%zu bytes)\n", mCacheFile.c_str(), length);
    }
}

} // namespace yolov5_common
