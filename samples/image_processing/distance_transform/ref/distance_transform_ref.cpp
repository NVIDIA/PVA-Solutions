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

#include "distance_transform_ref.h"

#include "../../../utils/Geometry.hpp"

#include "nvcv/Fwd.h"

#include <ImageIO.h>
#include <cupva_host.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * Detect most significant bit of unsigned input
 */
int msbd_w(long long data)
{
    int i;
    long long temp;
    if (data == 0)
        return 48;
    for (i = 0, temp = data / 2; temp > 0; i++) temp = temp / 2;
    return i;
}

/**
 * extract_bits
 *
 * Extract bit field by shift (up or down) then bit-mask between low and high bits
 * For example, extract_bits(data, 2, 4, 7) would return (data << 2) & (~0xF) & (0xFF)
 */
int extract_bits(int data, int shift, int low, int high)
{
    int temp1 = (shift >= 0) ? (data << shift) : (data >> -shift);
    int temp2 = ~((1 << low) - 1);
    int temp3 = (1 << (high + 1)) - 1;
    int temp4 = temp1 & temp2 & temp3;
    return temp4;
}

/**
 * my_round
 *
 * rounding down r bits, shift up if r < 0
 */
int my_round(long long x, int r)
{
    long long y;

    if (r < 0)
        return x << -r;
    else if (r == 0)
        return x;
    else
    {
        y = x >> (r - 1);
        y = (y + 1) >> 1;
        return y;
    }
}

/**
 *  blend
 *
 *  Perform alpha blending between x0 and x1, return (1-alpha)*x0 + alpha*x1,
 *  qbits is number of fraction bits in alpha
 */
int blend(int alpha, int x0, int x1, int qbits)
{
    int temp1 = alpha * (x1 - x0);
    int temp2 = my_round(temp1, qbits);
    return temp2 + x0;
}

/**
 * sqrt_q3_sca
 * square root with tbl_bits=6 (65-entry table) and q_bits=3
 */
int sqrt_q3_sca(int x, unsigned short *sqrt_tbl)
{
    int exp, mant;
    int idx, frac;
    int y0, y1, z;
    long long z_ll;

    if (x == 0)
        return 0;

    exp  = msbd_w(x);
    mant = x << (30 - exp); // MSB to bit 30

    // bit 31 30 29 28 27 26 25 24 .. 0, example tbl_bits=4
    //        msb<-  idx  -> <- frac ->
    idx  = extract_bits(mant, 6 - 30, 0, 6 - 1);
    frac = extract_bits(mant, 6 - 15, 0, 14); // to Q15

    /* lookup & interpolate */
    y0   = sqrt_tbl[idx];
    y1   = sqrt_tbl[idx + 1];
    z    = blend(frac, y0, y1, 15);
    z_ll = z;
    if (exp & 1)
        z = my_round(z_ll * sqrt_tbl[64], 15);
    z = my_round(z, 15 - 3 - exp / 2); // convert Q15 to Q3
    // combining with left shift (exp/2) bits
    return z;
}

/**
 * this function gives the horizontal distance from free space to obstacle
 * input_img is 0xffff means the pixel is an ON pixel (free space)
 * else means it is an OFF pixel with label value (obstacle)
 * horz_dist = output array, for horizontal distance
 * upper 4-bit: label, lower 12-bit: distance
 */
int distance_transform_stage1_ref(unsigned short *input_img, unsigned short *horz_dist, int tw, int th, int max_dist)
{
    int i, j, a;
    int prev_dist, prev_label;

    // -------------------------------------------------------------------- //
    // for each row                                                         //
    // -------------------------------------------------------------------- //
    for (i = 0; i < th; i++)
    {
        // tw + th is not enough to represent full image size
        // put max_dist here, it can be IW + IH or other value
        // but don't use the max value of an unsigned short to avoid overflow
        // prev_dist = tw + th;
        prev_dist  = max_dist;
        prev_label = 0;

        // ---------------------------------------------------------------- //
        // scan from left to right                                          //
        // ---------------------------------------------------------------- //
        for (j = 0; j < tw; j++)
        {
            a = i * tw + j;
            // if the input is an obstacle
            if (input_img[a] != 0xffff)
            {
                prev_dist  = 0;
                prev_label = input_img[a];
            }
            // else it is a free space
            else
            {
                prev_dist++;
            }
            horz_dist[a] = ((prev_label & 0xF) << 12) | (prev_dist & 0xFFF);
        }

        // prev_dist = tw + th;
        prev_dist  = max_dist;
        prev_label = 0;

        // ---------------------------------------------------------------- //
        // scan from right to left                                          //
        // ---------------------------------------------------------------- //
        for (j = tw - 1; j >= 0; j--)
        {
            a = i * tw + j;
            if ((prev_dist + 1) < (horz_dist[a] & 0xFFF))
            {
                prev_dist++;
                prev_dist    = prev_dist > tw ? tw : prev_dist;
                horz_dist[a] = ((prev_label & 0xF) << 12) | (prev_dist & 0xFFF);
            }
            else
            {
                prev_dist  = horz_dist[a] & 0xFFF;
                prev_label = (horz_dist[a] >> 12) & 0xF;
            }
        }
    }

    return 0;
}

int sum_sq(int x, int y)
{
    // Guard against overflow in multiplication and addition
    int sqrt_int_max = static_cast<int>(sqrt(INT_MAX));
    assert(x <= sqrt_int_max && y <= sqrt_int_max);

    int x_sq = x * x;
    int y_sq = y * y;
    assert(x_sq >= 0 && y_sq >= 0);

    int result = x_sq + y_sq;
    assert(result >= 0 && result <= INT_MAX);

    return result;
}

/**
 * parabola_intercept_nd
 * Find parabola intercept (or Voronoi partition) between (i1, h1)
 * and (i2, h2), i1 < i2, return numerator and denominator
 */
void parabola_intercept_nd(int i1, int i2, int h1, int h2, int *num, int *denom)
{
    *num   = sum_sq(i2, h2) - sum_sq(i1, h1);
    *denom = 2 * (i2 - i1);
}

int distance_transform_stage2_ref_nd_1(unsigned short *horz_dist, unsigned short *v_arr, unsigned short *output_dist,
                                       unsigned short *output_label, unsigned short *sqrt_tbl, int tw, int th, int iw,
                                       int ih, int max_dist)
{
    int i, j, k, q;
    int hd_q;
    int n1, d1, n2, d2;
    int done;
    int dist;

    int valid_hd;
    int km1, km2;
    int vkm1, vkm2;
    int hd_vkm1, hd_vkm2;
    int s_not_too_large, replacing_vk;
    int write_v_idx;
    int k2;
    int vk2, vk1;
    int hd_vk2, hd_vk1;

    // -------------------------------------------------------------------- //
    // for each column                                                      //
    // -------------------------------------------------------------------- //
    for (j = 0; j < tw; j++)
    {
        k = 0;
        q = 0;
        // v_arr[0] = 0;
        v_arr[0 * tw + j] = 0;
        done              = 0;

        // ---------------------------------------------------------------- //
        // scan top down                                                    //
        // ---------------------------------------------------------------- //
        while (!done)
        {
            hd_q     = horz_dist[q * tw + j] & 0xFFF;
            valid_hd = hd_q < iw;
            km1      = (k == 0) ? 0 : k - 1;
            km2      = (k <= 1) ? 0 : k - 2;
            // vkm1 = v_arr[km1];
            // vkm2 = v_arr[km2];
            vkm1 = v_arr[km1 * tw + j];
            vkm2 = v_arr[km2 * tw + j];

            hd_vkm1 = horz_dist[vkm1 * tw + j] & 0xFFF;
            hd_vkm2 = horz_dist[vkm2 * tw + j] & 0xFFF;

            // -------------------------------------------------------- //
            // compute intersection without division                    //
            // -------------------------------------------------------- //
            n1 = hd_q * hd_q - hd_vkm1 * hd_vkm1 + q * q - vkm1 * vkm1;
            d1 = 2 * (q - vkm1);

            n2 = hd_vkm1 * hd_vkm1 - hd_vkm2 * hd_vkm2 + vkm1 * vkm1 - vkm2 * vkm2;
            d2 = 2 * (vkm1 - vkm2);

            n2 = (k <= 1) ? 0 : n2;
            d2 = (k <= 1) ? 1 : d2;

            s_not_too_large = n1 <= d1 * (th - 1);
            replacing_vk    = n1 * d2 <= n2 * d1;

            // state: v[?] = q
            write_v_idx = (k == 1 && replacing_vk) ? (k - 1) : k;
            // if (valid_hd && (k == 0 || s_not_too_large)) v_arr[write_v_idx] = q;
            if (valid_hd && (k == 0 || s_not_too_large))
                v_arr[write_v_idx * tw + j] = q;

            // state: update q & done
            if (!(valid_hd && k > 1 && replacing_vk))
            {
                q++;
                done = (q >= th);
            }

            // state: update k
            if (valid_hd && (k == 0 || (!replacing_vk && s_not_too_large)))
                k++;
            else if (valid_hd && k > 1 && replacing_vk)
                k--;
        }

        // second vertical scan, v[0..k-1] capture Voronoi regions
        // v_arr[k] = v_arr[0];
        v_arr[k * tw + j] = v_arr[0 * tw + j];

        k2   = 0;
        q    = 0;
        done = 0;
        int update_k2;

        while (!done)
        {
            vk2    = v_arr[k2 * tw + j];
            hd_vk2 = horz_dist[vk2 * tw + j] & 0xFFF;
            vk1    = v_arr[(k2 + 1) * tw + j];
            hd_vk1 = horz_dist[vk1 * tw + j] & 0xFFF;
            d1     = sum_sq(q - vk2, hd_vk2);
            d2     = sum_sq(q - vk1, hd_vk1);

            update_k2 = (d2 <= d1 && k2 + 1 < k);
            k2 += update_k2;

            done = (q >= th - 1) && (!update_k2);
            if (!update_k2)
            {
                dist                    = sqrt_q3_sca(d1, sqrt_tbl);
                output_dist[q * tw + j] = (dist >= 8 * (iw + ih) || dist >= max_dist) ? max_dist : dist;
                // vk2 = v_arr[k2];
                vk2                      = v_arr[k2 * tw + j];
                output_label[q * tw + j] = (dist >= max_dist) ? 0 : ((horz_dist[vk2 * tw + j] >> 12) & 0xF);
            }
            q += (!update_k2);
        }
    }

    return 0;
}

int32_t randomizeDistanceTransformInputData(NVCVTensorHandle inputTensorHandle, int32_t width, int32_t height,
                                            int32_t numDot, uint16_t offPixel, uint16_t labelRange)
{
    int32_t err = 0;
    NVCVTensorData inputTensorData;
    nvcvTensorExportData(inputTensorHandle, &inputTensorData);

    uint16_t *input_h = nullptr;
    CupvaMemGetHostPointer((void **)&input_h, (void *)inputTensorData.buffer.strided.basePtr);

    // 1. set all pixels to off-pixel value
    for (int32_t i = 0; i < width * height; i++)
    {
        input_h[i] = offPixel;
    }

    // 2. random all dots coordinates and labels
    struct Point
    {
        int32_t x, y;
    };

    std::vector<Point> coordinates(numDot);
    std::vector<int32_t> labels(numDot);
    for (int32_t i = 0; i < numDot; i++)
    {
        coordinates[i] = {std::rand() % width, std::rand() % height};
        labels[i]      = std::rand() % labelRange;
    }

    // 3. fill in the pixels
    for (int32_t i = 0; i < numDot; i++)
    {
        auto x                 = coordinates[i].x;
        auto y                 = coordinates[i].y;
        input_h[y * width + x] = labels[i];
    }

    return 0;
}

int32_t getDistanceTransformRefData(NVCVTensorHandle inputTensorHandle, NVCVTensorHandle distTensorHandle,
                                    NVCVTensorHandle labelTensorHandle, int32_t width, int32_t height,
                                    uint16_t offPixel, uint16_t maxDist)
{
    int32_t err = 0;
    NVCVTensorData inputTensorData, distTensorData, labelTensorData;
    nvcvTensorExportData(inputTensorHandle, &inputTensorData);
    nvcvTensorExportData(distTensorHandle, &distTensorData);
    nvcvTensorExportData(labelTensorHandle, &labelTensorData);

    uint16_t *input_h{nullptr}, *dist_h{nullptr}, *label_h{nullptr};
    CupvaMemGetHostPointer((void **)&input_h, (void *)inputTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&dist_h, (void *)distTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&label_h, (void *)labelTensorData.buffer.strided.basePtr);

    std::vector<uint16_t> horDist(height * width);
    std::vector<uint16_t> vArray(height * width);
    std::vector<uint16_t> sqrtTableQ15{32768, 33023, 33276, 33527, 33776, 34024, 34270, 34514, 34756, 34996, 35235,
                                       35472, 35708, 35942, 36175, 36406, 36636, 36864, 37091, 37316, 37540, 37763,
                                       37985, 38205, 38424, 38642, 38858, 39073, 39287, 39500, 39712, 39923, 40132,
                                       40341, 40548, 40755, 40960, 41164, 41368, 41570, 41771, 41972, 42171, 42369,
                                       42567, 42763, 42959, 43154, 43348, 43541, 43733, 43925, 44115, 44305, 44494,
                                       44682, 44869, 45056, 45242, 45427, 45611, 45795, 45977, 46160, 46341};
    distance_transform_stage1_ref(input_h, horDist.data(), width, height, width);
    distance_transform_stage2_ref_nd_1(horDist.data(), vArray.data(), dist_h, label_h, sqrtTableQ15.data(), width,
                                       height, width, height, maxDist);
    return 0;
}

int32_t loadDistanceTransformData(NVCVTensorHandle tensorHandle, const char *directory, const char *filename,
                                  const int32_t tensorSizeInBytes)
{
    int32_t err = 0;
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    uint16_t *tensor_h = nullptr;
    CupvaMemGetHostPointer((void **)&tensor_h, (void *)tensorData.buffer.strided.basePtr);

    err = ReadImageBuffer(filename, directory, tensor_h, tensorSizeInBytes);
    return err;
}

int32_t checkDistanceTransformResult(NVCVTensorHandle devTensorHandle, NVCVTensorHandle refTensorHandle,
                                     const int32_t tensorSizeInBytes)
{
    NVCVTensorData devTensorData, refTensorData;
    nvcvTensorExportData(devTensorHandle, &devTensorData);
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    uint16_t *dev_h = nullptr, *ref_h = nullptr;
    CupvaMemGetHostPointer((void **)&dev_h, (void *)devTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&ref_h, (void *)refTensorData.buffer.strided.basePtr);

    return memcmp(dev_h, ref_h, tensorSizeInBytes);
}