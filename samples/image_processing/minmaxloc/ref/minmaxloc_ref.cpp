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

#include "minmaxloc_ref.h"

#include <nvcv/Tensor.hpp>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

using namespace std;

namespace {
template<typename T>
void cpu_minmaxloc_ref(T *src, int image_width, int image_height, int line_pitch, T &min_val, T &max_val, int &min_cnt,
                       int &max_cnt, short *p_min_loc_xy, short *p_max_loc_xy)
{
    // 1. get min/max values
    min_val = src[0], max_val = src[0];
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            int idx = i * line_pitch + j;

            if (src[idx] < min_val)
            {
                min_val = src[idx];
            }
            if (src[idx] > max_val)
            {
                max_val = src[idx];
            }
        }
    }

    // 2. get min/max locations: get all minmax locations in cpu ref, as the whole set
    min_cnt = 0, max_cnt = 0;
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            int idx = i * line_pitch + j;

            if (src[idx] == min_val)
            {
                p_min_loc_xy[2 * min_cnt]     = j;
                p_min_loc_xy[2 * min_cnt + 1] = i;
                min_cnt++;
            }
            if (src[idx] == max_val)
            {
                p_max_loc_xy[2 * max_cnt]     = j;
                p_max_loc_xy[2 * max_cnt + 1] = i;
                max_cnt++;
            }
        }
    }
}

template<typename T>
void init_rand_src(T *src, T min_val0, T max_val0, int min_cnt0, int max_cnt0, int image_height, int image_width,
                   int line_pitch)
{
    unsigned int seed = 365;
    srand(seed);

    // generate random source data
    int sample_length = image_height * image_width;
    for (int i = 0; i < sample_length; i++)
    {
        if (i < min_cnt0)
        {
            src[i] = min_val0;
        }
        else if (i < min_cnt0 + max_cnt0)
        {
            src[i] = max_val0;
        }
        else if (max_val0 > min_val0)
        {
            if (std::is_same<T, float>::value)
            {
                float offset = min(0.1, 0.1 * (max_val0 - min_val0));
                src[i]       = min_val0 + offset + (float)rand() / RAND_MAX * (max_val0 - min_val0 - 2 * offset);
            }
            else
            {
                src[i] = min_val0 + 1 + rand() % (int)(max_val0 - min_val0 - 1);
            }
        }
        else
        {
            src[i] = min_val0;
        }
    }
    shuffle(src, src + sample_length, default_random_engine(seed));

    // consider line pitch, fill data to the right position in reverse order to avoid extra memory cost
    for (int i = image_height - 1; i >= 0; i--)
    {
        for (int j = image_width - 1; j >= 0; j--)
        {
            src[i * line_pitch + j] = src[i * image_width + j];
        }
    }
}

// check tensor layout limit: currently only supporting [HWC/CHW/NCHW/NHWC] layouts with N=1 and C=1
void checkTensorLayouts(NVCVTensorRequirements *tensorRequirements)
{
    nvcv::TensorShape shape(tensorRequirements->shape, tensorRequirements->rank, tensorRequirements->layout);

    if (shape.layout() == nvcv::TENSOR_HWC)
    {
        if (shape[2] != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input tensor with layout HWC must have 1 channel!");
        }
    }
    else if (shape.layout() == nvcv::TENSOR_CHW)
    {
        if (shape[0] != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input tensor with layout CHW must have 1 channel!");
        }
    }
    else if (shape.layout() == nvcv::TENSOR_NCHW)
    {
        if (shape[0] != 1 || shape[1] != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input tensor with layout NCHW must have N=1 and C=1!");
        }
    }
    else if (shape.layout() == nvcv::TENSOR_NHWC)
    {
        if (shape[0] != 1 || shape[3] != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input tensor with layout NHWC must have N=1 and C=1!");
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input tensor layout!");
    }
}

// get image width, height and line pitch from NVCVTensorRequirements
template<typename T>
std::tuple<int, int, int> getImageShapeFromTensor(NVCVTensorRequirements *tensorRequirements,
                                                  NVCVTensorHandle tensorHandle)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    nvcv::TensorShape shape(tensorRequirements->shape, tensorRequirements->rank, tensorRequirements->layout);

    int image_width, image_height, line_pitch;
    // currently only supporting [HWC/CHW/NCHW/NHWC] layouts
    if (shape.layout() == nvcv::TENSOR_HWC)
    {
        image_height = shape[0];
        image_width  = shape[1];
        line_pitch   = tensorData.buffer.strided.strides[0] / sizeof(T);
    }
    else if (shape.layout() == nvcv::TENSOR_CHW)
    {
        image_height = shape[1];
        image_width  = shape[2];
        line_pitch   = tensorData.buffer.strided.strides[1] / sizeof(T);
    }
    else if (shape.layout() == nvcv::TENSOR_NCHW)
    {
        image_height = shape[2];
        image_width  = shape[3];
        line_pitch   = tensorData.buffer.strided.strides[2] / sizeof(T);
    }
    else if (shape.layout() == nvcv::TENSOR_NHWC)
    {
        image_height = shape[1];
        image_width  = shape[2];
        line_pitch   = tensorData.buffer.strided.strides[1] / sizeof(T);
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input tensor layout!");
    }
    return std::make_tuple(image_width, image_height, line_pitch);
}

template<typename T>
void generateRandomMinMaxLocInputImpl(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle tensorHandle,
                                      const T min_val0, const T max_val0, const int min_cnt0, const int max_cnt0)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    checkTensorLayouts(tensorRequirements);

    int image_width, image_height, line_pitch;
    std::tie(image_width, image_height, line_pitch) = getImageShapeFromTensor<T>(tensorRequirements, tensorHandle);

    init_rand_src<T>(tensorPtr_h, min_val0, max_val0, min_cnt0, max_cnt0, image_height, image_width, line_pitch);
}

template<typename T>
void generateMinMaxLocFixedImgInputImpl(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle tensorHandle,
                                        std::string assetDir)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = NULL;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    checkTensorLayouts(tensorRequirements);

    int imageWidth, imageHeight, linePitch;
    std::tie(imageWidth, imageHeight, linePitch) = getImageShapeFromTensor<T>(tensorRequirements, tensorHandle);

    // load uint8_t data from raw file using a dynamic allocation to avoid stack overflow
    int imageSize = imageHeight * imageWidth;
    std::vector<uint8_t> rawImageDataU8(imageSize);
    std::string rawImageFile = "minmaxloc/kodim08_grayscale_1080x1920.raw";

    int err = ReadImageBuffer(rawImageFile.c_str(), assetDir.c_str(), (void *)rawImageDataU8.data(), imageSize);
    if (err)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_OPERATION, "Failed to load read image buffer!\n");
    }

    // consider line pitch, fill data to the right position
    for (int i = 0; i < imageHeight; i++)
    {
        for (int j = 0; j < imageWidth; j++)
        {
            // convert uint8_t to desired input data type
            tensorPtr_h[i * linePitch + j] = (T)rawImageDataU8[i * imageWidth + j];
        }
    }
}

template<typename T>
int32_t checkMinMaxLocOutputImpl(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle srcTensorHandle,
                                 NVCVTensorHandle dstMinValTensorHandle, NVCVTensorHandle dstMinLocTensorHandle,
                                 NVCVTensorHandle dstMinCntTensorHandle, NVCVTensorHandle dstMaxValTensorHandle,
                                 NVCVTensorHandle dstMaxLocTensorHandle, NVCVTensorHandle dstMaxCntTensorHandle,
                                 int loc_capacity)
{

    int image_width, image_height, line_pitch;
    std::tie(image_width, image_height, line_pitch) = getImageShapeFromTensor<T>(tensorRequirements, srcTensorHandle);

    int max_loc_length      = 2 * image_width * image_height * sizeof(unsigned short);
    short *ref_min_loc_xy_h = (short *)malloc(max_loc_length);
    short *ref_max_loc_xy_h = (short *)malloc(max_loc_length);

    NVCVTensorData tensorData;
    nvcvTensorExportData(srcTensorHandle, &tensorData);
    T *src_h = NULL;
    CupvaMemGetHostPointer((void **)&src_h, (void *)tensorData.buffer.strided.basePtr);

    // run cref
    T ref_min_val, ref_max_val;
    int ref_min_cnt, ref_max_cnt;
    cpu_minmaxloc_ref<T>(src_h, image_width, image_height, line_pitch, ref_min_val, ref_max_val, ref_min_cnt,
                         ref_max_cnt, ref_min_loc_xy_h, ref_max_loc_xy_h);

    // create set of location coordinates
    set<pair<short, short>> ref_min_loc_set;
    for (int i = 0; i < ref_min_cnt; i++)
    {
        ref_min_loc_set.insert(make_pair(ref_min_loc_xy_h[2 * i], ref_min_loc_xy_h[2 * i + 1]));
    }
    set<pair<short, short>> ref_max_loc_set;
    for (int i = 0; i < ref_max_cnt; i++)
    {
        ref_max_loc_set.insert(make_pair(ref_max_loc_xy_h[2 * i], ref_max_loc_xy_h[2 * i + 1]));
    }

    // check results
    NVCVTensorData dstMinValTensorData, dstMaxValTensorData;
    nvcvTensorExportData(dstMinValTensorHandle, &dstMinValTensorData);
    nvcvTensorExportData(dstMaxValTensorHandle, &dstMaxValTensorData);
    void *dst_min_val_h, *dst_max_val_h;
    CupvaMemGetHostPointer((void **)&dst_min_val_h, (void *)dstMinValTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&dst_max_val_h, (void *)dstMaxValTensorData.buffer.strided.basePtr);
    T opt_min_val = *(T *)dst_min_val_h;
    T opt_max_val = *(T *)dst_max_val_h;

    NVCVTensorData dstMinCntTensorData, dstMaxCntTensorData;
    nvcvTensorExportData(dstMinCntTensorHandle, &dstMinCntTensorData);
    nvcvTensorExportData(dstMaxCntTensorHandle, &dstMaxCntTensorData);
    int *dst_min_cnt_h, *dst_max_cnt_h;
    CupvaMemGetHostPointer((void **)&dst_min_cnt_h, (void *)dstMinCntTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&dst_max_cnt_h, (void *)dstMaxCntTensorData.buffer.strided.basePtr);
    int opt_min_cnt = dst_min_cnt_h[0];
    int opt_max_cnt = dst_max_cnt_h[0];

    int32_t err = 0;
    // check values and counts:
    if (opt_min_val != ref_min_val)
    {
        printf("TEST FAIL! min value does not match: ref_min_val=%x, opt_min_val=%x\n", *(unsigned int *)&ref_min_val,
               *(unsigned int *)&opt_min_val);
        err = 1;
        goto FreeResources;
    }
    if (opt_max_val != ref_max_val)
    {
        printf("TEST FAIL! max value does not match: ref_max_val=%x, opt_max_val=%x\n", *(unsigned int *)&ref_max_val,
               *(unsigned int *)&opt_max_val);
        err = 1;
        goto FreeResources;
    }

    if (opt_min_cnt != ref_min_cnt)

    {
        printf("TEST FAIL! min count does not match: ref_min_cnt=%d, opt_min_cnt=%d\n", ref_min_cnt, opt_min_cnt);
        err = 2;
        goto FreeResources;
    }
    if (opt_max_cnt != ref_max_cnt)
    {
        printf("TEST FAIL! max count does not match: ref_max_cnt=%d, opt_max_cnt=%d\n", ref_max_cnt, opt_max_cnt);
        err = 2;
        goto FreeResources;
    }

    // check locations
    NVCVTensorData dstMinLocTensorData;
    nvcvTensorExportData(dstMinLocTensorHandle, &dstMinLocTensorData);
    unsigned short *dst_min_loc_xy_h;
    dst_min_loc_xy_h = NULL;
    CupvaMemGetHostPointer((void **)&dst_min_loc_xy_h, (void *)dstMinLocTensorData.buffer.strided.basePtr);

    NVCVTensorData dstMaxLocTensorData;
    nvcvTensorExportData(dstMaxLocTensorHandle, &dstMaxLocTensorData);
    unsigned short *dst_max_loc_xy_h;
    dst_max_loc_xy_h = NULL;
    CupvaMemGetHostPointer((void **)&dst_max_loc_xy_h, (void *)dstMaxLocTensorData.buffer.strided.basePtr);

    for (size_t i = 0; i < min(opt_min_cnt, loc_capacity); i++)
    {
        auto coord = make_pair(dst_min_loc_xy_h[2 * i], dst_min_loc_xy_h[2 * i + 1]);
        if (ref_min_loc_set.find(coord) == ref_min_loc_set.end())
        {
            printf("TEST FAIL! min location does not match: idx=%ld, opt coord=(%d, %d)\n", i, dst_min_loc_xy_h[2 * i],
                   dst_min_loc_xy_h[2 * i + 1]);
            err = 3;
            goto FreeResources;
        }
    }

    for (size_t i = 0; i < min(opt_max_cnt, loc_capacity); i++)
    {
        auto coord = make_pair(dst_max_loc_xy_h[2 * i], dst_max_loc_xy_h[2 * i + 1]);
        if (ref_max_loc_set.find(coord) == ref_max_loc_set.end())
        {
            printf("TEST FAIL! max location does not match: idx=%ld, opt coord=(%d, %d)\n", i, dst_max_loc_xy_h[2 * i],
                   dst_max_loc_xy_h[2 * i + 1]);
            err = 4;
            goto FreeResources;
        }
    }

FreeResources:
    free(ref_min_loc_xy_h);
    free(ref_max_loc_xy_h);
    return err;
}

} // namespace

// generate random input data, supporting custom min/max values and counts
void generateRandomMinMaxLocInput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                  int min_cnt0, int max_cnt0)
{

    NVCVDataType inDataType;
    nvcvTensorGetDataType(inTensorHandle, &inDataType);

    switch (inDataType)
    {
    case NVCV_DATA_TYPE_U8:
        uint8_t min_val_u8, max_val_u8;
        min_val_u8 = 15;
        max_val_u8 = 250;
        generateRandomMinMaxLocInputImpl<uint8_t>(tensorRequirements, inTensorHandle, min_val_u8, max_val_u8, min_cnt0,
                                                  max_cnt0);
        break;
    case NVCV_DATA_TYPE_S8:
        int8_t min_val_s8, max_val_s8;
        min_val_s8 = -100;
        max_val_s8 = 88;
        generateRandomMinMaxLocInputImpl<int8_t>(tensorRequirements, inTensorHandle, min_val_s8, max_val_s8, min_cnt0,
                                                 max_cnt0);
        break;
    case NVCV_DATA_TYPE_U16:
        uint16_t min_val_u16, max_val_u16;
        min_val_u16 = 21;
        max_val_u16 = 55250;
        generateRandomMinMaxLocInputImpl<uint16_t>(tensorRequirements, inTensorHandle, min_val_u16, max_val_u16,
                                                   min_cnt0, max_cnt0);
        break;
    case NVCV_DATA_TYPE_S16:
        int16_t min_val_s16, max_val_s16;
        min_val_s16 = -1921;
        max_val_s16 = 32250;
        generateRandomMinMaxLocInputImpl<int16_t>(tensorRequirements, inTensorHandle, min_val_s16, max_val_s16,
                                                  min_cnt0, max_cnt0);
        break;
    case NVCV_DATA_TYPE_U32:
        uint32_t min_val_u32, max_val_u32;
        min_val_u32 = 13;
        max_val_u32 = 4294967200;
        generateRandomMinMaxLocInputImpl<uint32_t>(tensorRequirements, inTensorHandle, min_val_u32, max_val_u32,
                                                   min_cnt0, max_cnt0);
        break;
    case NVCV_DATA_TYPE_S32:
        int32_t min_val_s32, max_val_s32;
        min_val_s32 = -124113;
        max_val_s32 = 429496720;
        generateRandomMinMaxLocInputImpl<int32_t>(tensorRequirements, inTensorHandle, min_val_s32, max_val_s32,
                                                  min_cnt0, max_cnt0);
        break;
    case NVCV_DATA_TYPE_F32:
        float min_val_f32, max_val_f32;
        min_val_f32 = -140.56;
        max_val_f32 = 11401.712;
        generateRandomMinMaxLocInputImpl<float>(tensorRequirements, inTensorHandle, min_val_f32, max_val_f32, min_cnt0,
                                                max_cnt0);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type!");
        break;
    }
}

// generate input data by loading a fixed image
void generateMinMaxLocFixedImgInput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                    std::string assetDir)
{
    NVCVDataType inDataType;
    nvcvTensorGetDataType(inTensorHandle, &inDataType);

    switch (inDataType)
    {
    case NVCV_DATA_TYPE_U8:
        generateMinMaxLocFixedImgInputImpl<uint8_t>(tensorRequirements, inTensorHandle, assetDir);
        break;
    case NVCV_DATA_TYPE_S8:
        generateMinMaxLocFixedImgInputImpl<int8_t>(tensorRequirements, inTensorHandle, assetDir);
        break;
    case NVCV_DATA_TYPE_U16:
        generateMinMaxLocFixedImgInputImpl<uint16_t>(tensorRequirements, inTensorHandle, assetDir);
        break;
    case NVCV_DATA_TYPE_S16:
        generateMinMaxLocFixedImgInputImpl<int16_t>(tensorRequirements, inTensorHandle, assetDir);
        break;
    case NVCV_DATA_TYPE_U32:
        generateMinMaxLocFixedImgInputImpl<uint32_t>(tensorRequirements, inTensorHandle, assetDir);
        break;
    case NVCV_DATA_TYPE_S32:
        generateMinMaxLocFixedImgInputImpl<int32_t>(tensorRequirements, inTensorHandle, assetDir);
        break;
    case NVCV_DATA_TYPE_F32:
        generateMinMaxLocFixedImgInputImpl<float>(tensorRequirements, inTensorHandle, assetDir);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type!");
        break;
    }
}

int32_t checkMinMaxLocOutput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle srcTensorHandle,
                             NVCVTensorHandle dstMinValTensorHandle, NVCVTensorHandle dstMinLocTensorHandle,
                             NVCVTensorHandle dstMinCntTensorHandle, NVCVTensorHandle dstMaxValTensorHandle,
                             NVCVTensorHandle dstMaxLocTensorHandle, NVCVTensorHandle dstMaxCntTensorHandle,
                             int loc_capacity)
{
    nvcv::TensorShape shape(tensorRequirements->shape, tensorRequirements->rank, tensorRequirements->layout);

    NVCVDataType inDataType;
    nvcvTensorGetDataType(srcTensorHandle, &inDataType);

    switch (inDataType)
    {
    case NVCV_DATA_TYPE_U8:
        return checkMinMaxLocOutputImpl<uint8_t>(tensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                                 dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                                 dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);
    case NVCV_DATA_TYPE_S8:
        return checkMinMaxLocOutputImpl<int8_t>(tensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                                dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                                dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);

    case NVCV_DATA_TYPE_U16:
        return checkMinMaxLocOutputImpl<uint16_t>(tensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                                  dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                                  dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);
    case NVCV_DATA_TYPE_S16:
        return checkMinMaxLocOutputImpl<int16_t>(tensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                                 dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                                 dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);

    case NVCV_DATA_TYPE_U32:
        return checkMinMaxLocOutputImpl<uint32_t>(tensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                                  dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                                  dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);
    case NVCV_DATA_TYPE_S32:
        return checkMinMaxLocOutputImpl<int32_t>(tensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                                 dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                                 dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);

    case NVCV_DATA_TYPE_F32:
        return checkMinMaxLocOutputImpl<float>(tensorRequirements, srcTensorHandle, dstMinValTensorHandle,
                                               dstMinLocTensorHandle, dstMinCntTensorHandle, dstMaxValTensorHandle,
                                               dstMaxLocTensorHandle, dstMaxCntTensorHandle, loc_capacity);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type!");
        break;
    }
    return -1;
}

// C-compatible wrapper for generateMinMaxLocFixedImgInput
void generateMinMaxLocFixedImgInputC(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                     const char *assetDir)
{
    std::string assetDirStr = (assetDir != NULL) ? std::string(assetDir) : std::string("");
    generateMinMaxLocFixedImgInput(tensorRequirements, inTensorHandle, assetDirStr);
}
