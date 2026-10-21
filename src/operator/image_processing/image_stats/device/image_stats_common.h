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

#ifndef IMAGE_STATS_COMMON_H
#define IMAGE_STATS_COMMON_H

#include "image_stats_param.h"
#include "longlong_div.h"

#include <cupva_device.h>
typedef struct
{
    int32_t _niter;

    // agen for different planes
    AgenCFG _src0_agen_cfg;
    AgenCFG _src1_agen_cfg;
    AgenCFG _mask_agen_cfg;

    // auxiliary agens
    AgenCFG _srcC0_aux_agen_cfg;
    AgenCFG _srcC1_aux_agen_cfg;
    AgenCFG _mask_aux_agen_cfg;

    bool _pixelCountFlag;
    bool _sumFlag;
    bool _meanFlag;
    bool _varianceFlag;
    bool _covarianceFlag;

    int32_t *_planePixelCount;
    uint64_t *_perChannelSumU64Buf;
    uint64_t *_covarianceMatU64Buf;
} IMAGE_STATS_CFG;

static inline void image_stats_init_common_params(IMAGE_STATS_CFG &cfg, int32_t *planePixelCount,
                                                  uint64_t *perChannelSumU64Buf, uint64_t *covarianceMatU64Buf,
                                                  bool pixelCountFlag, bool sumFlag, bool meanFlag, bool varianceFlag,
                                                  bool covarianceFlag)
{
    cfg._planePixelCount     = planePixelCount;
    cfg._perChannelSumU64Buf = perChannelSumU64Buf;
    cfg._covarianceMatU64Buf = covarianceMatU64Buf;

    cfg._pixelCountFlag = pixelCountFlag;
    cfg._sumFlag        = sumFlag;
    cfg._meanFlag       = meanFlag;
    cfg._varianceFlag   = varianceFlag;
    cfg._covarianceFlag = covarianceFlag;
}

static inline void image_stats_normalize_exec(IMAGE_STATS_CFG &cfg, int32_t planeCount, uint8_t *planeChannelNum,
                                              uint16_t *planeWidth, uint16_t *planeHeight, int32_t *pixelCountBuf,
                                              float *perChannelSumBuf, float *perChannelMeanBuf,
                                              float *covarianceMatBuf)
{
#if !defined(MASK_IO)
    // if without mask, init pixel count for each plane
    cfg._planePixelCount[0] = planeWidth[0] * planeHeight[0];
#endif

    // the pixel count of the first plane is the total pixel count
    if (cfg._pixelCountFlag)
    {
        pixelCountBuf[0] = cfg._planePixelCount[0];
    }

#if defined(SUM_IO)
    int32_t totalChannelNum = 0;
    for (int32_t p = 0; p < planeCount; p++) chess_prepare_for_pipelining
    {
        totalChannelNum += planeChannelNum[p];
    }

    float qbit_scale = 1.0f / (1 << QBITS);
    uint32_t n       = pixelCountBuf[0];

    // calculate per channel sum & mean
    for (int32_t c = 0; c < totalChannelNum; c++)
    {
        int32_t planeIdx = c >= planeChannelNum[0] ? 1 : 0;

        // calculate per channel sum
        int64_t tmpChannelSum = cfg._perChannelSumU64Buf[c];
        perChannelSumBuf[c]   = (float)(uint32_t)(tmpChannelSum);

        // calculate per channel mean
        if (cfg._meanFlag)
        {
            uint64_t d1, r1;
            d1 = ll_udiv(tmpChannelSum, n, r1);
            // fixed-point division
            // uint64_t r1_q = (r1 << QBITS) / n;
            uint64_t r1_q, rem;
            r1_q                 = ll_udiv(r1 << QBITS, n, rem);
            perChannelMeanBuf[c] = fmadd(qbit_scale, (float)(uint32_t)r1_q, (float)(uint32_t)d1);
        }
    }
#endif

#if defined(VAR_IO) || defined(COV_IO)
    uint32_t divCnt = n > 1 ? n - 1 : 1;

    // normalize covariance matrix
    for (int32_t c0 = 0; c0 < totalChannelNum; c0++)
    {
        int32_t planeIdx0 = c0 >= planeChannelNum[0] ? 1 : 0;

#    if defined(COV_IO)
        for (int32_t c1 = c0; c1 < totalChannelNum; c1++)
#    else
        int32_t c1 = c0;
#    endif
        {
            int32_t planeIdx1 = c1 >= planeChannelNum[0] ? 1 : 0;

            uint64_t covSum   = cfg._covarianceMatU64Buf[c0 * 4 + c1];
            uint64_t sumMean0 = cfg._perChannelSumU64Buf[c0];
            uint64_t sumMean1 = cfg._perChannelSumU64Buf[c1];

            /*
            cov = (covSum - sumMean0 * sumMean1 / N) / (N-1)
                = (covSum - d1 - r1_f)/(N-1)
                = d2 + r2/(N-1) - r1_f/(N-1)
            */

            uint64_t numerator1 = sumMean0 * sumMean1;
            uint64_t d1, r1;
            d1 = ll_udiv(numerator1, n, r1);

            int64_t numerator2 = covSum - d1;
            int64_t d2, r2;
            d2 = ll_sdiv(numerator2, (int64_t)divCnt, r2);

            // fixed-point division:
            // uint64_t r1_f_q = (r1 << QBITS) / (n * divCnt);
            // int64_t r2_f_q  = (r2 << QBITS) / divCnt;
            uint64_t rem1;
            uint64_t r1_f_q = ll_udiv(r1 << QBITS, mulwl_u(n, divCnt), rem1);
            int64_t rem2;
            int64_t r2_f_q = ll_sdiv(r2 << QBITS, (int64_t)divCnt, rem2);

            float cov = fmadd(qbit_scale, -(float)(uint32_t)r1_f_q,
                              fmadd(qbit_scale, (float)(int32_t)r2_f_q, (float)(int32_t)(d2)));

            covarianceMatBuf[c0 * 4 + c1] = cov;
            // fill the lower triangular part of the covariance matrix by symmetry
            if (c1 > c0)
            {
                covarianceMatBuf[c1 * 4 + c0] = cov;
            }
        }
    }
#endif
}

#endif // IMAGE_STATS_COMMON_H
