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

#include "pvaAplConv2dVpu.hpp"

#include "conv2d_param.h"

#include <cupva_device.h>

VMEM(A, DATA_TYPE, src_even_v, RDF_CIRCULAR(DATA_TYPE, TW, TH / 2, KWR, (KHR + 1) / 2));
VMEM(B, DATA_TYPE, src_odd_v, RDF_CIRCULAR(DATA_TYPE, TW, TH / 2, KWR, (KHR + 1) / 2));
VMEM(C, DATA_TYPE, dst_v, RDF_DOUBLE(DATA_TYPE, TW, THA * 2));
VMEM(C, KERNEL_TYPE, knl_v, KWA *KHA * 2);
VMEM(C, int32_t, iw);
VMEM(C, int32_t, ih);
VMEM(C, int32_t, borderMode);

VMEM_RDF_UNIFIED(C, srcEvenHdl);
VMEM_RDF_UNIFIED(C, srcOddHdl);
VMEM_RDF_UNIFIED(C, dstHdl);

#define NVCV_BORDER_CONSTANT 0
#define NVCV_BORDER_REPLICATE 1

void copyPaddingVpu(DATA_TYPE *dst, DATA_TYPE *src, int offset, int count)
{
    int vecw  = 64;
    int niter = count * sizeof(DATA_TYPE) / vecw;

    AgenWrapper wrap;
    wrap.size = sizeof(char);
    wrap.n1   = niter;
    wrap.s1   = vecw;

    agen src_agen = init((dvchar *)(src + offset));
    INIT_AGEN1(src_agen, wrap);

    agen dst_agen = init((dvchar *)(dst + offset));
    INIT_AGEN1(dst_agen, wrap);

    for (int i = 0; i < niter; ++i) chess_loop_range(1, )
    chess_prepare_for_pipelining
    {
        dvcharx src_v = dvchar_load(src_agen);
        vstore(src_v, dst_agen);
    }

    // tail case
    for (int k = niter * vecw / sizeof(DATA_TYPE); k < count; ++k)
    {
        *(dst + offset + k) = *(src + offset + k);
    }
}

CUPVA_VPU_MAIN()
{
    int32_t niter1 = (iw + TW - 1) / TW;
    int32_t niter2 = (ih + TH - 1) / TH;

    int32_t src_even_cb_len   = cupvaRasterDataFlowGetCbLen(srcEvenHdl);
    int32_t src_even_cb_pitch = cupvaRasterDataFlowGetLinePitch(srcEvenHdl);
    int32_t src_odd_cb_len    = cupvaRasterDataFlowGetCbLen(srcOddHdl);
    int32_t src_odd_cb_pitch  = cupvaRasterDataFlowGetLinePitch(srcOddHdl);

    // odd/even cb should have the same size
    int32_t cb_len = src_even_cb_len;

    // Construct the 2D Convolution primitive handle
    pvaApl::Conv2dVpu<KW, DATA_TYPE, KERNEL_TYPE> conv2dVpuHandle;

    DATA_TYPE *src_even = &src_even_v[0];
    DATA_TYPE *src_odd  = &src_odd_v[0];
    DATA_TYPE *dst      = &dst_v[0];

    // for conv3x3 and conv7x7, src_even starts from the second row (skip top padding)
#if defined(KSIZE_3X3) || defined(KSIZE_7X7)
    src_even += src_even_cb_pitch;
#endif

    // Initialize the 2D Convolution primitive handle
    int qbits = sizeof(DATA_TYPE) == 1 ? QBITS_8 : QBITS_16;
    conv2dVpuHandle.Init(src_even, src_even_cb_pitch, src_odd, src_odd_cb_pitch, knl_v, KWA * 4, dst_v, TW, qbits, TW,
                         TH, src_even_v, src_odd_v, cb_len);

    cupvaRasterDataFlowOpen(srcEvenHdl, &src_even_v[0]);
    cupvaRasterDataFlowOpen(srcOddHdl, &src_odd_v[0]);
    cupvaRasterDataFlowOpen(dstHdl, &dst_v[0]);

    for (int32_t i = 0; i < niter2; i++)
    {
        for (int32_t j = 0; j < niter1; j++)
        {
            src_even = (DATA_TYPE *)cupvaRasterDataFlowAcquire(srcEvenHdl);
            src_odd  = (DATA_TYPE *)cupvaRasterDataFlowAcquire(srcOddHdl);
            dst      = (DATA_TYPE *)cupvaRasterDataFlowAcquire(dstHdl);

            if (borderMode == NVCV_BORDER_REPLICATE)
            {
                // copy top padding from even buffer, copy bottom padding from odd buffer
                int32_t count = TW + KWR * 2;

                if (i == 0)
                {
                    for (int k = 0; k < (KHR + 1) / 2; ++k)
                    {
                        int32_t offset  = (src_odd - src_odd_v + k * src_odd_cb_pitch) % cb_len;
                        int32_t max_cnt = cb_len - offset; // max count without exceeding cb_len

                        if (count > max_cnt)
                        {
                            copyPaddingVpu(src_odd_v, src_even_v, offset, max_cnt + 64 / sizeof(DATA_TYPE));
                            copyPaddingVpu(src_odd_v, src_even_v, 0, count - max_cnt);
                        }
                        else
                        {
                            copyPaddingVpu(src_odd_v, src_even_v, offset, count);
                        }
                    }
                }

                if (i == niter2 - 1)
                {
                    int32_t row_num  = (ih / 2) % (TH / 2) == 0 ? (TH / 2) : (ih / 2) % (TH / 2);
                    int32_t last_row = (row_num + (KHR + 1) / 2) * src_even_cb_pitch;

                    for (int k = 0; k < (KHR + 1) / 2; ++k)
                    {
                        int32_t offset  = (src_odd - src_odd_v + last_row + k * src_odd_cb_pitch) % cb_len;
                        int32_t max_cnt = cb_len - offset; // max count without exceeding cb_len

                        if (count > max_cnt)
                        {
                            copyPaddingVpu(src_even_v, src_odd_v, offset, max_cnt + 64 / sizeof(DATA_TYPE));
                            copyPaddingVpu(src_even_v, src_odd_v, 0, count - max_cnt);
                        }
                        else
                        {
                            copyPaddingVpu(src_even_v, src_odd_v, offset, count);
                        }
                    }
                }
            }

#if defined(KSIZE_3X3) || defined(KSIZE_7X7)
            int32_t srcEvenOfst  = src_even - src_even_v;
            int32_t srcEvenStart = (srcEvenOfst + src_even_cb_pitch > src_even_cb_len)
                                       ? ((srcEvenOfst + src_even_cb_pitch) - src_even_cb_len)
                                       : (srcEvenOfst + src_even_cb_pitch);
            src_even             = src_even_v + srcEvenStart;
#endif

            conv2dVpuHandle.Update(src_even, src_odd, dst);
            conv2dVpuHandle.Execute();

            cupvaRasterDataFlowRelease(srcEvenHdl);
            cupvaRasterDataFlowRelease(srcOddHdl);
            cupvaRasterDataFlowRelease(dstHdl);
        }
    }
    cupvaRasterDataFlowClose(srcEvenHdl);
    cupvaRasterDataFlowClose(srcOddHdl);
    cupvaRasterDataFlowClose(dstHdl);

    return 0;
}
