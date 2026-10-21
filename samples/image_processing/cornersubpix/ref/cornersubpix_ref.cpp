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

#include "cornersubpix_ref.h"

#include "../../../utils/Geometry.hpp"

#include <ImageIO.h>
#include <OpCornerSubPix.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define QBITS 15
#define TOL 3
#define DIFF_TOL 0.001f // threshold of CPU and VPU floating point outputs' absdiff

#define PRINT_OUTLIERS 0 // 1 to print outliers, 0 to disable

using namespace std;

int32_t loadCornerSubPixCornerInput(NVCVTensorHandle cornerTensorHandle, const char *directory, const char *filename,
                                    const int32_t count)
{
    int32_t err = 0;
    NVCVTensorData cornerTensorData;
    nvcvTensorExportData(cornerTensorHandle, &cornerTensorData);

    float *corners_h = nullptr;
    CupvaMemGetHostPointer((void **)&corners_h, (void *)cornerTensorData.buffer.strided.basePtr);

    err = ReadImageBuffer(filename, directory, corners_h, sizeof(float) * count * 2);
    return err;
}

int32_t loadCornerSubPixImageInput(NVCVTensorHandle imageTensorHandle, const char *directory, const char *filename,
                                   const int64_t size)
{
    int32_t err = 0;
    NVCVTensorData imageTensorData;
    nvcvTensorExportData(imageTensorHandle, &imageTensorData);

    uint8_t *image_h = nullptr;
    CupvaMemGetHostPointer((void **)&image_h, (void *)imageTensorData.buffer.strided.basePtr);

    err = ReadImageBuffer(filename, directory, image_h, sizeof(uint8_t) * size);
    return err;
}

int32_t checkCornerSubPixOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle, const int32_t count)
{
    int32_t err         = 0;
    float outlier_count = 0;

    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    float *ref_h = nullptr;
    CupvaMemGetHostPointer((void **)&ref_h, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);

    float *pva_h = nullptr;
    CupvaMemGetHostPointer((void **)&pva_h, (void *)pvaTensorData.buffer.strided.basePtr);

    for (int32_t i = 0; i < count; i++)
    {
        float ref_x = ref_h[i * 2];
        float pva_x = pva_h[i * 2];
        float ref_y = ref_h[i * 2 + 1];
        float pva_y = pva_h[i * 2 + 1];
        if (abs(ref_x - pva_x) > DIFF_TOL || abs(ref_y - pva_y) > DIFF_TOL)
        {
#if PRINT_OUTLIERS
            printf("Outlier[%d]: ref = (%f, %f) pva = (%f, %f)\n", i, ref_x, ref_y, pva_x, pva_y);
#endif
            outlier_count++;
        }
    }

    // PVA and CPU may exhibit minor floating-point differences after a single iteration.
    // However, these differences can accumulate over multiple iterations and become significant.
    // Outlier occurs usually due to large zeroZone.
    // If the outlier ratio is less than 1%, we consider it acceptable.
    if (outlier_count / count > 0.01)
    {
        err = -1;
    }
    return err;
}

void getRectSubPixFixedPointDlut(const uint8_t *src, NVCVSize2D src_size, int32_t line_pitch, NVCVSize2D win_size,
                                 Point2i center, int32_t *dst)
{
    Point2i start;
    start.x = center.x >> QBITS;
    start.y = center.y >> QBITS;

    start.x -= (win_size.w - 1) / 2;
    start.y -= (win_size.h - 1) / 2;

    int32_t frac = (1 << QBITS) - 1;
    int32_t a    = center.x & frac;
    int32_t b    = center.y & frac;

    for (int32_t i = 0; i < win_size.h; i++)
    {
        for (int32_t j = 0; j < win_size.w; j++)
        {
            int32_t x1 = min(max(0, (start.x + j)), (src_size.w - 1));
            int32_t y1 = min(max(0, (start.y + i)), (src_size.h - 1));
            int32_t x2 = min(max(0, (start.x + 1 + j)), (src_size.w - 1));
            int32_t y2 = min(max(0, (start.y + 1 + i)), (src_size.h - 1));

            int64_t a11 = static_cast<int64_t>(src[x1 + y1 * line_pitch]) << 8;
            int64_t a12 = static_cast<int64_t>(src[x2 + y1 * line_pitch]) << 8;
            int64_t a21 = static_cast<int64_t>(src[x1 + y2 * line_pitch]) << 8;
            int64_t a22 = static_cast<int64_t>(src[x2 + y2 * line_pitch]) << 8;

            int64_t val_top = ((a12 - a11) * a + (1 << QBITS - 1) >> QBITS) + a11;
            int64_t val_bot = ((a22 - a21) * a + (1 << QBITS - 1) >> QBITS) + a21;

            int64_t val = ((val_bot - val_top) * b + (1 << QBITS - 1) >> QBITS) + val_top;

            dst[i * win_size.w + j] = static_cast<int32_t>(val);
        }
    }
}

void cornerSubPixFixedPointRef(NVCVTensorHandle cornerTensorHandle, NVCVTensorHandle imageTensorHandle,
                               const NVCVSize2D src, const NVCVSize2D win, const NVCVSize2D zeroZone,
                               const int32_t maxIters, const float epsilon, const int32_t count)
{
    int32_t win_w = win.w * 2 + 1, win_h = win.h * 2 + 1;
    int32_t i, j, k;
    float eps = epsilon * epsilon; // use square of error in comparison operations

    NVCVTensorData cornerTensorData;
    nvcvTensorExportData(cornerTensorHandle, &cornerTensorData);

    Point2f *corners = nullptr;
    CupvaMemGetHostPointer((void **)&corners, (void *)cornerTensorData.buffer.strided.basePtr);

    NVCVTensorData imageTensorData;
    nvcvTensorExportData(imageTensorHandle, &imageTensorData);

    uint8_t *image = nullptr;
    CupvaMemGetHostPointer((void **)&image, (void *)imageTensorData.buffer.strided.basePtr);

    int32_t line_pitch;
    // image tensor layout is checked at pvaCornerSubPixCreate stage
    std::string imageLayout = std::string(nvcvTensorLayoutGetName(&imageTensorData.layout));
    if (imageLayout == "HW" || imageLayout == "HWC")
    {
        line_pitch = imageTensorData.buffer.strided.strides[0];
    }
    else if (imageLayout == "CHW" || imageLayout == "NHWC")
    {
        line_pitch = imageTensorData.buffer.strided.strides[1];
    }
    else if (imageLayout == "NCHW")
    {
        line_pitch = imageTensorData.buffer.strided.strides[2];
    }

    if (count == 0)
        return;

    vector<float> mask(win_h * win_w);
    vector<int32_t> subpix_buf((win_h + 2) * (win_w + 2));
    vector<uint16_t> mask_i(win_h * win_w);

    // make mask
    for (i = 0; i < win_h; i++)
    {
        float y  = (float)(i - win.h) / win.h;
        float vy = exp(-y * y);
        for (j = 0; j < win_w; j++)
        {
            float x             = (float)(j - win.w) / win.w;
            mask[i * win_w + j] = (float)(vy * exp(-x * x));
        }
    }

    // make zero_zone
    if (zeroZone.w >= 0 && zeroZone.h >= 0 && zeroZone.w * 2 + 1 < win_w && zeroZone.h * 2 + 1 < win_h)
    {
        for (i = win.h - zeroZone.h; i <= win.h + zeroZone.h; i++)
        {
            for (j = win.w - zeroZone.w; j <= win.w + zeroZone.w; j++)
            {
                mask[i * win_w + j] = 0;
            }
        }
    }

    // convert mask to uint16
    for (i = 0; i < win_h; i++)
    {
        for (j = 0; j < win_w; j++)
        {
            mask_i[i * win_w + j] = static_cast<uint16_t>(mask[i * win_w + j] * (1 << QBITS) + 0.5f);
        }
    }

    // do optimization loop for all the points
    for (int32_t pt_i = 0; pt_i < count; pt_i++)
    {
        Point2f cT = corners[pt_i];
        if (!Rect<float>(0, 0, src.w, src.h).contains(cT))
            continue;
        int32_t iter = 0;
        float err    = 0;

        Point2i cI;
        cI.x = static_cast<int32_t>(cT.x * (1 << QBITS));
        cI.y = static_cast<int32_t>(cT.y * (1 << QBITS));

        Point2i cI2 = cI;

        do
        {
            Point2i cI3;

            int64_t a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;

            getRectSubPixFixedPointDlut(image, src, line_pitch, {win_w + 2, win_h + 2}, cI2, subpix_buf.data());
            int32_t *subpix = &subpix_buf[(win_w + 2) + 1];

            // process gradient
            for (i = 0, k = 0; i < win_h; i++, subpix += win_w + 2)
            {
                int64_t py = i - win.h;

                for (j = 0; j < win_w; j++, k++)
                {
                    int32_t m   = mask_i[k];                                     //mask uint16 Q15
                    int32_t tgx = subpix[j + 1] - subpix[j - 1];                 //subpix uint16 Q8
                    int32_t tgy = subpix[j + win_w + 2] - subpix[j - win_w - 2]; //subpix uint16 Q8

                    int64_t tmp_xx = static_cast<int64_t>(tgx) * tgx; //vmulhw 32bit + 1signbit
                    int64_t tmp_yy = static_cast<int64_t>(tgy) * tgy;
                    int64_t tmp_xy = static_cast<int64_t>(tgx) * tgy;

                    int64_t gxx = ((tmp_xx * m >> 15) + 1) >> 1; //Q15 vmulwhw
                    int64_t gxy = ((tmp_xy * m >> 15) + 1) >> 1; //Q15
                    int64_t gyy = ((tmp_yy * m >> 15) + 1) >> 1; //Q15

                    int64_t px = j - win.w;

                    a += gxx; // Q15
                    b += gxy;
                    c += gyy;

                    bb1 += gxx * px + gxy * py; // Q15
                    bb2 += gxy * px + gyy * py;
                }
            }

            float a_f   = static_cast<float>(a);
            float b_f   = static_cast<float>(b);
            float c_f   = static_cast<float>(c);
            float bb1_f = static_cast<float>(bb1);
            float bb2_f = static_cast<float>(bb2);

            float det = a_f * c_f - b_f * b_f;

            if (fabs(det) <= FLT_EPSILON * FLT_EPSILON)
                break;

            // 2x2 matrix inversion
            float scale = 1.0f / det; // -2*QBITS

            float dx = c_f * scale * bb1_f - b_f * scale * bb2_f;
            float dy = -b_f * scale * bb1_f + a_f * scale * bb2_f;

            int32_t dx_i = static_cast<int32_t>(dx * (1 << QBITS));
            int32_t dy_i = static_cast<int32_t>(dy * (1 << QBITS));

            cI3.x = cI2.x + dx_i;
            cI3.y = cI2.y + dy_i;

            Point2f cI2_f;
            cI2_f.x = cI3.x * (1.0f / (1 << QBITS));
            cI2_f.y = cI3.y * (1.0f / (1 << QBITS));

            err = dx * dx + dy * dy;

            // if new point is out of image, leave previous point as the result
            if (!Rect<float>(0, 0, src.w, src.h).contains(cI2_f))
                break;
#if 0
            {
                cI2 = cI3;
            }
#else
            // Clamp to tile boundary
            {
                cI2.x = min(max(cI3.x, cI.x - ((win.w + TOL) << QBITS)), cI.x + ((win.w + TOL + 1) << QBITS));
                cI2.y = min(max(cI3.y, cI.y - ((win.h + TOL) << QBITS)), cI.y + ((win.h + TOL + 1) << QBITS));
            }
#endif

        } while (++iter < maxIters && err > eps);

        // if new point is too far from initial, it means poor convergence.
        // leave initial point as the result
        if (abs(cI2.x - cI.x) > (win.w << QBITS) || abs(cI2.y - cI.y) > (win.h << QBITS))
            cI2 = cI;

        corners[pt_i].x = cI2.x * (1.f / (1 << QBITS));
        corners[pt_i].y = cI2.y * (1.f / (1 << QBITS));
    }
}
