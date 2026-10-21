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

#include "fast_corner_detector_knls.hpp"

#include <cupva_device.h>

void fastCornerDetectorQuickRejAgenInit(int32_t inPitch, int32_t tileX, int32_t tileY, int32_t cbSizeBytes,
                                        intptr_t cb0StartAddr, intptr_t cb1StartAddr, intptr_t cb2StartAddr,
                                        QuickRejAgenParam *params)
{
    params->niter = (tileX / DVECW) * tileY;
    // All the following agen's base addresses are initialized with nullptr, since they will be updated when used.
    {
        // P1 and P9
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg0 = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = 2;
        wrapper.n2 = tileX / DVECW;
        wrapper.n3 = tileY;
        wrapper.s1 = 6 * inPitch;
        wrapper.s2 = DVECW;
        wrapper.s3 = inPitch;
        INIT_AGEN3(streamCircleSeg0, wrapper);
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = 2; // 2 vld
        wrapper.n2 = 2;
        wrapper.n3 = tileX / DVECW;
        wrapper.n4 = tileY;
        wrapper.s1 = VECW;
        wrapper.s2 = 6 * inPitch;
        wrapper.s3 = DVECW;
        wrapper.s4 = inPitch;
        INIT_AGEN4(streamCircleSeg0, wrapper);
#endif
        streamCircleSeg0         = update_agen_cb_start(streamCircleSeg0, cb0StartAddr);
        streamCircleSeg0         = update_agen_cb_size(streamCircleSeg0, cbSizeBytes);
        params->streamCircleSeg0 = extract_agen_cfg(streamCircleSeg0);
    }
    {
        // P5 and P13
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg1 = init(nullptr);
        wrapper.n1            = 2;
        wrapper.n2            = tileX / DVECW;
        wrapper.n3            = tileY;
        wrapper.s1            = 6;
        wrapper.s2            = DVECW;
        wrapper.s3            = inPitch;
        INIT_AGEN3(streamCircleSeg1, wrapper);
        streamCircleSeg1         = update_agen_cb_start(streamCircleSeg1, cb1StartAddr);
        streamCircleSeg1         = update_agen_cb_size(streamCircleSeg1, cbSizeBytes);
        params->streamCircleSeg1 = extract_agen_cfg(streamCircleSeg1);
    }
    {
        AgenWrapper wrapper;
        wrapper.size            = sizeof(BASE_TYPE);
        agen streamCircleCenter = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = tileX / DVECW;
        wrapper.n2 = tileY;
        wrapper.s1 = DVECW;
        wrapper.s2 = inPitch;
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = tileX / VECW;
        wrapper.n2 = tileY;
        wrapper.s1 = VECW;
        wrapper.s2 = inPitch;
#endif
        INIT_AGEN2(streamCircleCenter, wrapper);
        streamCircleCenter         = update_agen_cb_start(streamCircleCenter, cb2StartAddr);
        streamCircleCenter         = update_agen_cb_size(streamCircleCenter, cbSizeBytes);
        params->streamCircleCenter = extract_agen_cfg(streamCircleCenter);
    }
}

uint64_t fastCornerDetectorQuickRej(BASE_TYPE *seg0Ptr, BASE_TYPE *seg1Ptr, BASE_TYPE *centerPtr, int32_t threshold,
                                    QuickRejAgenParam *params)
{
    cupvaModifyAgenCfgBase(&params->streamCircleSeg0, seg0Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleSeg1, seg1Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleCenter, centerPtr);

    agen streamCircleSeg0   = init_agen_from_cfg(params->streamCircleSeg0);
    agen streamCircleSeg1   = init_agen_from_cfg(params->streamCircleSeg1);
    agen streamCircleCenter = init_agen_from_cfg(params->streamCircleCenter);

    chess_separator_scheduler();

    DVEC_TYPE dv_threshold;
    dv_threshold.lo = REPLICATE(threshold);
    dv_threshold.hi = REPLICATE(threshold);

    uint64_t outBitMask = 0; // Bitmap, value 1 means rejection for the current 1xDVECW pixels.
    for (int32_t i = 0; i < params->niter; i++) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
// P1, P9, P5, P13
#if defined(U16_IO) || defined(S16_IO)
        DVEC_TYPE dv_p1 = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p9 = DLOAD_TYPE(streamCircleSeg0);
#elif defined(U8_IO) || defined(S8_IO)
        DVEC_TYPE dv_p1;
        dv_p1.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p1.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p9;
        dv_p9.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p9.hi = LOAD_TYPE(streamCircleSeg0);
#endif

        DVEC_TYPE dv_p13 = DLOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p5  = DLOAD_TYPE(streamCircleSeg1);

#if defined(U16_IO) || defined(S16_IO)
        DVEC_TYPE dv_center = DLOAD_TYPE(streamCircleCenter);
#elif defined(U8_IO) || defined(S8_IO)
        DVEC_TYPE dv_center;
        dv_center.lo = LOAD_TYPE(streamCircleCenter);
        dv_center.hi = LOAD_TYPE(streamCircleCenter);
#endif
        // dvsort2: simultaneous min+max of each pair
        DVEC_TYPE dv_min_15, dv_max_15, dv_min_9d, dv_max_9d;
        dvsort2(dv_p1, dv_p5, dv_min_15, dv_max_15);
        dvsort2(dv_p9, dv_p13, dv_min_9d, dv_max_9d);
        DVEC_TYPE dv_min_all = dvmin(dv_min_15, dv_min_9d);
        DVEC_TYPE dv_max_all = dvmax(dv_max_15, dv_max_9d);

        DVEC_TYPE dv_center_sub_threshold = dv_center - dv_threshold;

        DVEC_TYPE dv_center_add_threshold = dv_center + dv_threshold;
        DVEC_TYPE dv_is_similar = (dv_min_all >= dv_center_sub_threshold) & (dv_max_all <= dv_center_add_threshold);

        int bitsNzIdx;
        vshortx v_placeholder;
#if defined(U16_IO) || defined(S16_IO)
        vcollate_idx_bits(dv_is_similar.lo, v_placeholder, bitsNzIdx);
        int32_t count = bitcount(bitsNzIdx);
        vcollate_idx_bits(dv_is_similar.hi, v_placeholder, bitsNzIdx);
        count += bitcount(bitsNzIdx);
#elif defined(U8_IO) || defined(S8_IO)
        dvshortx dv_is_similar_lo;
        vpromote_di(dv_is_similar.lo, dv_is_similar_lo.lo, dv_is_similar_lo.hi);
        vcollate_idx_bits(dv_is_similar_lo.lo, v_placeholder, bitsNzIdx);
        int32_t count = bitcount(bitsNzIdx);

        vcollate_idx_bits(dv_is_similar_lo.hi, v_placeholder, bitsNzIdx);
        count += bitcount(bitsNzIdx);

        dvshortx dv_is_similar_hi;
        vpromote_di(dv_is_similar.hi, dv_is_similar_hi.lo, dv_is_similar_hi.hi);
        vcollate_idx_bits(dv_is_similar_hi.lo, v_placeholder, bitsNzIdx);
        count += bitcount(bitsNzIdx);

        vcollate_idx_bits(dv_is_similar_hi.hi, v_placeholder, bitsNzIdx);
        count += bitcount(bitsNzIdx);
#endif

        outBitMask |= ((uint64_t)(count == DVECW)) << i;
    }
    return outBitMask;
}

void fastCornerDetectorScoreNoRejAgenInit(int32_t inPitch, int32_t outPitch, int32_t tileX, int32_t tileY,
                                          int32_t cbSizeBytes, intptr_t cb0StartAddr, intptr_t cb1StartAddr,
                                          intptr_t cb2StartAddr, ScoreAgenParam *params)
{
    params->niter = (tileX / DVECW) * tileY;
    // All the following agen's base address is initialized with nullptr, since they will be updated when used.
    {
        // P0~P2, P10~P8
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg0 = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = 3;
        wrapper.n2 = 2;
        wrapper.n3 = tileX / DVECW;
        wrapper.n4 = tileY;
        wrapper.s1 = 1;
        wrapper.s2 = 6 * inPitch;
        wrapper.s3 = DVECW;
        wrapper.s4 = inPitch;
        INIT_AGEN4(streamCircleSeg0, wrapper);
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = 2; // 2 vld
        wrapper.n2 = 3;
        wrapper.n3 = 2;
        wrapper.n4 = tileX / DVECW;
        wrapper.n5 = tileY;
        wrapper.s1 = VECW;
        wrapper.s2 = 1;
        wrapper.s3 = 6 * inPitch;
        wrapper.s4 = DVECW;
        wrapper.s5 = inPitch;
        INIT_AGEN5(streamCircleSeg0, wrapper);
#endif
        streamCircleSeg0         = update_agen_cb_start(streamCircleSeg0, cb0StartAddr);
        streamCircleSeg0         = update_agen_cb_size(streamCircleSeg0, cbSizeBytes);
        params->streamCircleSeg0 = extract_agen_cfg(streamCircleSeg0);
    }
    {
        // P15, P3, P11, P7
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg1 = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = 2;
        wrapper.n2 = 2;
        wrapper.n3 = tileX / DVECW;
        wrapper.n4 = tileY;
        wrapper.s1 = 4;
        wrapper.s2 = 4 * inPitch;
        wrapper.s3 = DVECW;
        wrapper.s4 = inPitch;
        INIT_AGEN4(streamCircleSeg1, wrapper);
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = 2; // 2 vld
        wrapper.n2 = 2;
        wrapper.n3 = 2;
        wrapper.n4 = tileX / DVECW;
        wrapper.n5 = tileY;
        wrapper.s1 = VECW;
        wrapper.s2 = 4;
        wrapper.s3 = 4 * inPitch;
        wrapper.s4 = DVECW;
        wrapper.s5 = inPitch;
        INIT_AGEN5(streamCircleSeg1, wrapper);
#endif
        streamCircleSeg1         = update_agen_cb_start(streamCircleSeg1, cb1StartAddr);
        streamCircleSeg1         = update_agen_cb_size(streamCircleSeg1, cbSizeBytes);
        params->streamCircleSeg1 = extract_agen_cfg(streamCircleSeg1);
    }
    {
        // P14, P4, P13, P5, P12, P6
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg2 = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = 2;
        wrapper.n2 = 3;
        wrapper.n3 = tileX / DVECW;
        wrapper.n4 = tileY;
        wrapper.s1 = 6;
        wrapper.s2 = inPitch;
        wrapper.s3 = DVECW;
        wrapper.s4 = inPitch;
        INIT_AGEN4(streamCircleSeg2, wrapper);
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = 2; // 2 vld
        wrapper.n2 = 2;
        wrapper.n3 = 3;
        wrapper.n4 = tileX / DVECW;
        wrapper.n5 = tileY;
        wrapper.s1 = VECW;
        wrapper.s2 = 6;
        wrapper.s3 = inPitch;
        wrapper.s4 = DVECW;
        wrapper.s5 = inPitch;
        INIT_AGEN5(streamCircleSeg2, wrapper);
#endif
        streamCircleSeg2         = update_agen_cb_start(streamCircleSeg2, cb2StartAddr);
        streamCircleSeg2         = update_agen_cb_size(streamCircleSeg2, cbSizeBytes);
        params->streamCircleSeg2 = extract_agen_cfg(streamCircleSeg2);
    }
    {
        AgenWrapper wrapper;
        wrapper.size = sizeof(BASE_TYPE);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = tileX / DVECW;
        wrapper.n2 = tileY;
        wrapper.s1 = DVECW;
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = tileX / VECW;
        wrapper.n2 = tileY;
        wrapper.s1 = VECW;
#endif
        wrapper.s2              = inPitch;
        agen streamCircleCenter = init(nullptr);
        INIT_AGEN2(streamCircleCenter, wrapper);
        streamCircleCenter         = update_agen_cb_start(streamCircleCenter, cb1StartAddr);
        streamCircleCenter         = update_agen_cb_size(streamCircleCenter, cbSizeBytes);
        params->streamCircleCenter = extract_agen_cfg(streamCircleCenter);
    }
    {
        AgenWrapper wrapper;
        wrapper.size   = sizeof(BASE_TYPE);
        wrapper.n1     = tileX / DVECW;
        wrapper.n2     = tileY;
        wrapper.s1     = DVECW;
        wrapper.s2     = outPitch;
        agen streamDst = init(nullptr);
        INIT_AGEN2(streamDst, wrapper);
        params->streamDst = extract_agen_cfg(streamDst);
    }
}

/*
 * This function computes the score in a tile, without considering the quick rejected vectors as contrary in
 * fastCornerDetectorScorePartialRej(). It will thus have better performance due to there being no branches within the for loop.
 */
void fastCornerDetectorScoreNoRej(BASE_TYPE *seg0Ptr, BASE_TYPE *seg1Ptr, BASE_TYPE *seg2Ptr, BASE_TYPE *centerPtr,
                                  BASE_TYPE *outPtr, int32_t threshold, ScoreAgenParam *params)
{
    cupvaModifyAgenCfgBase(&params->streamCircleSeg0, seg0Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleSeg1, seg1Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleSeg2, seg2Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleCenter, centerPtr);
    cupvaModifyAgenCfgBase(&params->streamDst, outPtr);

    agen streamCircleSeg0   = init_agen_from_cfg(params->streamCircleSeg0);
    agen streamCircleSeg1   = init_agen_from_cfg(params->streamCircleSeg1);
    agen streamCircleSeg2   = init_agen_from_cfg(params->streamCircleSeg2);
    agen streamCircleCenter = init_agen_from_cfg(params->streamCircleCenter);
    agen streamDst          = init_agen_from_cfg(params->streamDst);

    chess_separator_scheduler();

    DVEC_TYPE dv_threshold;
    dv_threshold.lo = REPLICATE(threshold);
    dv_threshold.hi = REPLICATE(threshold);

    DVEC_TYPE dv_zero = chess_dont_care(DVEC_TYPE) & 0;

    // The loop body has a SOL of 3.63 cycle/pixel.
    for (int32_t i = 0; i < params->niter; i++) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
#if defined(U16_IO) || defined(S16_IO)
        DVEC_TYPE dv_p0 = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p1 = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p2 = DLOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_p15 = DLOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p3  = DLOAD_TYPE(streamCircleSeg1);

        DVEC_TYPE dv_p14 = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p4  = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p13 = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p5  = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p12 = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p6  = DLOAD_TYPE(streamCircleSeg2);

        DVEC_TYPE dv_p11 = DLOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p7  = DLOAD_TYPE(streamCircleSeg1);

        DVEC_TYPE dv_p10 = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p9  = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p8  = DLOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_center = DLOAD_TYPE(streamCircleCenter);
#elif defined(U8_IO) || defined(S8_IO)
        DVEC_TYPE dv_p0;
        dv_p0.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p0.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p1;
        dv_p1.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p1.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p2;
        dv_p2.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p2.hi = LOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_p15;
        dv_p15.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p15.hi = LOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p3;
        dv_p3.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p3.hi = LOAD_TYPE(streamCircleSeg1);

        DVEC_TYPE dv_p14;
        dv_p14.lo = LOAD_TYPE(streamCircleSeg2);
        dv_p14.hi = LOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p4;
        dv_p4.lo = LOAD_TYPE(streamCircleSeg2);
        dv_p4.hi = LOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p13;
        dv_p13.lo = LOAD_TYPE(streamCircleSeg2);
        dv_p13.hi = LOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p5;
        dv_p5.lo = LOAD_TYPE(streamCircleSeg2);
        dv_p5.hi = LOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p12;
        dv_p12.lo = LOAD_TYPE(streamCircleSeg2);
        dv_p12.hi = LOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p6;
        dv_p6.lo = LOAD_TYPE(streamCircleSeg2);
        dv_p6.hi = LOAD_TYPE(streamCircleSeg2);

        DVEC_TYPE dv_p11;
        dv_p11.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p11.hi = LOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p7;
        dv_p7.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p7.hi = LOAD_TYPE(streamCircleSeg1);

        DVEC_TYPE dv_p10;
        dv_p10.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p10.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p9;
        dv_p9.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p9.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p8;
        dv_p8.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p8.hi = LOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_center;
        dv_center.lo = LOAD_TYPE(streamCircleCenter);
        dv_center.hi = LOAD_TYPE(streamCircleCenter);
#endif

        DVEC_TYPE dv_min2_0  = dvmin(dv_p0, dv_p1);
        DVEC_TYPE dv_min2_14 = dvmin(dv_p14, dv_p15);
        DVEC_TYPE dv_min2_2  = dvmin(dv_p2, dv_p3);
        DVEC_TYPE dv_min2_12 = dvmin(dv_p12, dv_p13);
        DVEC_TYPE dv_min2_4  = dvmin(dv_p4, dv_p5);
        DVEC_TYPE dv_min2_10 = dvmin(dv_p10, dv_p11);
        DVEC_TYPE dv_min2_6  = dvmin(dv_p6, dv_p7);
        DVEC_TYPE dv_min2_8  = dvmin(dv_p8, dv_p9);

        DVEC_TYPE dv_min4_0  = dvmin(dv_min2_0, dv_min2_2);
        DVEC_TYPE dv_min4_2  = dvmin(dv_min2_2, dv_min2_4);
        DVEC_TYPE dv_min4_4  = dvmin(dv_min2_4, dv_min2_6);
        DVEC_TYPE dv_min4_6  = dvmin(dv_min2_6, dv_min2_8);
        DVEC_TYPE dv_min4_8  = dvmin(dv_min2_8, dv_min2_10);
        DVEC_TYPE dv_min4_10 = dvmin(dv_min2_10, dv_min2_12);
        DVEC_TYPE dv_min4_12 = dvmin(dv_min2_12, dv_min2_14);
        DVEC_TYPE dv_min4_14 = dvmin(dv_min2_14, dv_min2_0);

        DVEC_TYPE dv_min8_0  = dvmin(dv_min4_0, dv_min4_4);
        DVEC_TYPE dv_min8_2  = dvmin(dv_min4_2, dv_min4_6);
        DVEC_TYPE dv_min8_4  = dvmin(dv_min4_4, dv_min4_8);
        DVEC_TYPE dv_min8_6  = dvmin(dv_min4_6, dv_min4_10);
        DVEC_TYPE dv_min8_8  = dvmin(dv_min4_8, dv_min4_12);
        DVEC_TYPE dv_min8_10 = dvmin(dv_min4_10, dv_min4_14);
        DVEC_TYPE dv_min8_12 = dvmin(dv_min4_12, dv_min4_0);
        DVEC_TYPE dv_min8_14 = dvmin(dv_min4_14, dv_min4_2);

        DVEC_TYPE dv_min9_0  = dvmin(dv_min8_0, dv_p8);
        DVEC_TYPE dv_min9_1  = dvmin(dv_min8_2, dv_p1);
        DVEC_TYPE dv_min9_2  = dvmin(dv_min8_2, dv_p10);
        DVEC_TYPE dv_min9_3  = dvmin(dv_min8_4, dv_p3);
        DVEC_TYPE dv_min9_4  = dvmin(dv_min8_4, dv_p12);
        DVEC_TYPE dv_min9_5  = dvmin(dv_min8_6, dv_p5);
        DVEC_TYPE dv_min9_6  = dvmin(dv_min8_6, dv_p14);
        DVEC_TYPE dv_min9_7  = dvmin(dv_min8_8, dv_p7);
        DVEC_TYPE dv_min9_8  = dvmin(dv_min8_8, dv_p0);
        DVEC_TYPE dv_min9_9  = dvmin(dv_min8_10, dv_p9);
        DVEC_TYPE dv_min9_10 = dvmin(dv_min8_10, dv_p2);
        DVEC_TYPE dv_min9_11 = dvmin(dv_min8_12, dv_p11);
        DVEC_TYPE dv_min9_12 = dvmin(dv_min8_12, dv_p4);
        DVEC_TYPE dv_min9_13 = dvmin(dv_min8_14, dv_p13);
        DVEC_TYPE dv_min9_14 = dvmin(dv_min8_14, dv_p6);
        DVEC_TYPE dv_min9_15 = dvmin(dv_min8_0, dv_p15);

        DVEC_TYPE dv_max_of_min9_red00 = dvmax(dv_min9_0, dv_min9_1);
        DVEC_TYPE dv_max_of_min9_red01 = dvmax(dv_min9_2, dv_min9_3);
        DVEC_TYPE dv_max_of_min9_red02 = dvmax(dv_min9_4, dv_min9_5);
        DVEC_TYPE dv_max_of_min9_red03 = dvmax(dv_min9_6, dv_min9_7);
        DVEC_TYPE dv_max_of_min9_red04 = dvmax(dv_min9_8, dv_min9_9);
        DVEC_TYPE dv_max_of_min9_red05 = dvmax(dv_min9_10, dv_min9_11);
        DVEC_TYPE dv_max_of_min9_red06 = dvmax(dv_min9_12, dv_min9_13);
        DVEC_TYPE dv_max_of_min9_red07 = dvmax(dv_min9_14, dv_min9_15);

        DVEC_TYPE dv_max_of_min9_red0001 = dvmax(dv_max_of_min9_red00, dv_max_of_min9_red01);
        DVEC_TYPE dv_max_of_min9_red0203 = dvmax(dv_max_of_min9_red02, dv_max_of_min9_red03);
        DVEC_TYPE dv_max_of_min9_red0405 = dvmax(dv_max_of_min9_red04, dv_max_of_min9_red05);
        DVEC_TYPE dv_max_of_min9_red0607 = dvmax(dv_max_of_min9_red06, dv_max_of_min9_red07);

        DVEC_TYPE dv_max_of_min9_red00010203 = dvmax(dv_max_of_min9_red0001, dv_max_of_min9_red0203);
        DVEC_TYPE dv_max_of_min9_red04050607 = dvmax(dv_max_of_min9_red0405, dv_max_of_min9_red0607);

        DVEC_TYPE dv_max_of_min9 = dvmax(dv_max_of_min9_red00010203, dv_max_of_min9_red04050607);

        DVEC_TYPE dv_diff_brighter      = dv_max_of_min9 - dv_center;
        DVEC_TYPE dv_is_brighter_corner = dv_diff_brighter > dv_threshold;
        DVEC_TYPE dv_score_brighter     = dvmux(dv_is_brighter_corner, dv_diff_brighter, dv_zero);

        DVEC_TYPE dv_max2_0  = dvmax(dv_p0, dv_p1);
        DVEC_TYPE dv_max2_14 = dvmax(dv_p14, dv_p15);
        DVEC_TYPE dv_max2_2  = dvmax(dv_p2, dv_p3);
        DVEC_TYPE dv_max2_12 = dvmax(dv_p12, dv_p13);
        DVEC_TYPE dv_max2_4  = dvmax(dv_p4, dv_p5);
        DVEC_TYPE dv_max2_10 = dvmax(dv_p10, dv_p11);
        DVEC_TYPE dv_max2_6  = dvmax(dv_p6, dv_p7);
        DVEC_TYPE dv_max2_8  = dvmax(dv_p8, dv_p9);

        DVEC_TYPE dv_max4_0  = dvmax(dv_max2_0, dv_max2_2);
        DVEC_TYPE dv_max4_2  = dvmax(dv_max2_2, dv_max2_4);
        DVEC_TYPE dv_max4_4  = dvmax(dv_max2_4, dv_max2_6);
        DVEC_TYPE dv_max4_6  = dvmax(dv_max2_6, dv_max2_8);
        DVEC_TYPE dv_max4_8  = dvmax(dv_max2_8, dv_max2_10);
        DVEC_TYPE dv_max4_10 = dvmax(dv_max2_10, dv_max2_12);
        DVEC_TYPE dv_max4_12 = dvmax(dv_max2_12, dv_max2_14);
        DVEC_TYPE dv_max4_14 = dvmax(dv_max2_14, dv_max2_0);

        DVEC_TYPE dv_max8_0  = dvmax(dv_max4_0, dv_max4_4);
        DVEC_TYPE dv_max8_2  = dvmax(dv_max4_2, dv_max4_6);
        DVEC_TYPE dv_max8_4  = dvmax(dv_max4_4, dv_max4_8);
        DVEC_TYPE dv_max8_6  = dvmax(dv_max4_6, dv_max4_10);
        DVEC_TYPE dv_max8_8  = dvmax(dv_max4_8, dv_max4_12);
        DVEC_TYPE dv_max8_10 = dvmax(dv_max4_10, dv_max4_14);
        DVEC_TYPE dv_max8_12 = dvmax(dv_max4_12, dv_max4_0);
        DVEC_TYPE dv_max8_14 = dvmax(dv_max4_14, dv_max4_2);

        DVEC_TYPE dv_max9_0  = dvmax(dv_max8_0, dv_p8);
        DVEC_TYPE dv_max9_1  = dvmax(dv_max8_2, dv_p1);
        DVEC_TYPE dv_max9_2  = dvmax(dv_max8_2, dv_p10);
        DVEC_TYPE dv_max9_3  = dvmax(dv_max8_4, dv_p3);
        DVEC_TYPE dv_max9_4  = dvmax(dv_max8_4, dv_p12);
        DVEC_TYPE dv_max9_5  = dvmax(dv_max8_6, dv_p5);
        DVEC_TYPE dv_max9_6  = dvmax(dv_max8_6, dv_p14);
        DVEC_TYPE dv_max9_7  = dvmax(dv_max8_8, dv_p7);
        DVEC_TYPE dv_max9_8  = dvmax(dv_max8_8, dv_p0);
        DVEC_TYPE dv_max9_9  = dvmax(dv_max8_10, dv_p9);
        DVEC_TYPE dv_max9_10 = dvmax(dv_max8_10, dv_p2);
        DVEC_TYPE dv_max9_11 = dvmax(dv_max8_12, dv_p11);
        DVEC_TYPE dv_max9_12 = dvmax(dv_max8_12, dv_p4);
        DVEC_TYPE dv_max9_13 = dvmax(dv_max8_14, dv_p13);
        DVEC_TYPE dv_max9_14 = dvmax(dv_max8_14, dv_p6);
        DVEC_TYPE dv_max9_15 = dvmax(dv_max8_0, dv_p15);

        DVEC_TYPE dv_min_of_max9_red00 = dvmin(dv_max9_0, dv_max9_1);
        DVEC_TYPE dv_min_of_max9_red01 = dvmin(dv_max9_2, dv_max9_3);
        DVEC_TYPE dv_min_of_max9_red02 = dvmin(dv_max9_4, dv_max9_5);
        DVEC_TYPE dv_min_of_max9_red03 = dvmin(dv_max9_6, dv_max9_7);
        DVEC_TYPE dv_min_of_max9_red04 = dvmin(dv_max9_8, dv_max9_9);
        DVEC_TYPE dv_min_of_max9_red05 = dvmin(dv_max9_10, dv_max9_11);
        DVEC_TYPE dv_min_of_max9_red06 = dvmin(dv_max9_12, dv_max9_13);
        DVEC_TYPE dv_min_of_max9_red07 = dvmin(dv_max9_14, dv_max9_15);

        DVEC_TYPE dv_min_of_max9_red0001 = dvmin(dv_min_of_max9_red00, dv_min_of_max9_red01);
        DVEC_TYPE dv_min_of_max9_red0203 = dvmin(dv_min_of_max9_red02, dv_min_of_max9_red03);
        DVEC_TYPE dv_min_of_max9_red0405 = dvmin(dv_min_of_max9_red04, dv_min_of_max9_red05);
        DVEC_TYPE dv_min_of_max9_red0607 = dvmin(dv_min_of_max9_red06, dv_min_of_max9_red07);

        DVEC_TYPE dv_min_of_max9_red00010203 = dvmin(dv_min_of_max9_red0001, dv_min_of_max9_red0203);
        DVEC_TYPE dv_min_of_max9_red04050607 = dvmin(dv_min_of_max9_red0405, dv_min_of_max9_red0607);

        DVEC_TYPE dv_min_of_max9 = dvmin(dv_min_of_max9_red00010203, dv_min_of_max9_red04050607);

        DVEC_TYPE dv_diff_darker      = dv_center - dv_min_of_max9;
        DVEC_TYPE dv_is_darker_corner = dv_diff_darker > dv_threshold;
        DVEC_TYPE dv_score            = dvmux(dv_is_darker_corner, dv_diff_darker, dv_score_brighter);

        vstore(dv_score, streamDst);
    }
}

int32_t fastCornerDetectorCalVecBaseAdv(uint64_t rejBitMask, int32_t inputLinePitch,
                                        const int32_t inputVecOffsetInTile[], const int32_t outputVecOffsetInTile[],
                                        int32_t inputBaseAdv[NUM_DVEC_PER_TILE],
                                        int32_t outputBaseAdv[NUM_DVEC_PER_TILE])
{
    int32_t niterNonRejVec    = 0;
    int32_t lastInputBaseAdv  = 0;
    int32_t lastOutputBaseAdv = 0;
    int32_t *inputBaseAdvPtr  = inputBaseAdv;
    int32_t *outputBaseAdvPtr = outputBaseAdv;

    vintx v_inputOffset                   = zero_extend(*(vuint *)inputVecOffsetInTile);
    vintx v_outputOffset                  = zero_extend(*(vuint *)outputVecOffsetInTile);
    const int32_t numElementsPerInnerLoop = pva_elementsof(vintx);
    const int32_t numVecPerRow            = TILE_X / DVECW;
    for (int32_t i = 0; i < NUM_DVEC_PER_TILE / numElementsPerInnerLoop; i++)
    {
        for (int32_t j = 0; j < numElementsPerInnerLoop; j++) chess_prepare_for_pipelining
        {
            int32_t index = i * numElementsPerInnerLoop + j;
            if ((rejBitMask & (1ULL << index)) == 0)
            {
                int32_t inputOffset = vselectlane(v_inputOffset, j);
                *inputBaseAdvPtr++  = inputOffset - lastInputBaseAdv;

                int32_t outputOffset = vselectlane(v_outputOffset, j);
                *outputBaseAdvPtr++  = outputOffset - lastOutputBaseAdv;

                niterNonRejVec++;
                lastInputBaseAdv  = inputOffset;
                lastOutputBaseAdv = outputOffset;
            }
        }
        v_inputOffset  = v_inputOffset + (numElementsPerInnerLoop / numVecPerRow) * inputLinePitch * sizeof(BASE_TYPE);
        v_outputOffset = v_outputOffset + DVECW * sizeof(BASE_TYPE) * numElementsPerInnerLoop;
    }
    return niterNonRejVec;
}

void fastCornerDetectorScorePartialRejAgenInit(int32_t inPitch, int32_t outPitch, int32_t tileX, int32_t tileY,
                                               int32_t cbSizeBytes, intptr_t cb0StartAddr, intptr_t cb1StartAddr,
                                               intptr_t cb2StartAddr, ScoreAgenParam *params)
{
    /*
     * All the following agen's base addresses are initialized with nullptr, since they will be updated when used.
     * except for streamZero, all other agen's outer most loop count is initialized with its maximum value,
     * but the actual interation count is controlled by the for loop count.
     */
    params->niter = (tileX / DVECW) * tileY;
    {
        // P0~P2, P10~P8
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg0 = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = 3;
        wrapper.n2 = 2;
        wrapper.n3 = tileX / DVECW * tileY;
        wrapper.s1 = 1;
        wrapper.s2 = 6 * inPitch;
        wrapper.s3 = 0;
        INIT_AGEN3(streamCircleSeg0, wrapper);
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1   = 2; // 2 vld
        wrapper.n2   = 3;
        wrapper.n3   = 2;
        wrapper.n4   = tileX / DVECW * tileY;
        wrapper.s1   = VECW;
        wrapper.s2   = 1;
        wrapper.s3   = 6 * inPitch;
        wrapper.s4   = 0;
        INIT_AGEN4(streamCircleSeg0, wrapper);
#endif
        streamCircleSeg0         = update_agen_cb_start(streamCircleSeg0, cb0StartAddr);
        streamCircleSeg0         = update_agen_cb_size(streamCircleSeg0, cbSizeBytes);
        params->streamCircleSeg0 = extract_agen_cfg(streamCircleSeg0);
    }
    {
        // P15, P3, P11, P7
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg1 = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = 2;
        wrapper.n2 = 2;
        wrapper.n3 = tileX / DVECW * tileY;
        wrapper.s1 = 4;
        wrapper.s2 = 4 * inPitch;
        wrapper.s3 = 0;
        INIT_AGEN3(streamCircleSeg1, wrapper);
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = 2; // 2 vld
        wrapper.n2 = 2;
        wrapper.n3 = 2;
        wrapper.n4 = tileX / DVECW * tileY;
        wrapper.s1 = VECW;
        wrapper.s2 = 4;
        wrapper.s3 = 4 * inPitch;
        wrapper.s4 = 0;
        INIT_AGEN4(streamCircleSeg1, wrapper);
#endif
        streamCircleSeg1         = update_agen_cb_start(streamCircleSeg1, cb1StartAddr);
        streamCircleSeg1         = update_agen_cb_size(streamCircleSeg1, cbSizeBytes);
        params->streamCircleSeg1 = extract_agen_cfg(streamCircleSeg1);
    }
    {
        // P14, P4, P13, P5, P12, P6
        AgenWrapper wrapper;
        wrapper.size          = sizeof(BASE_TYPE);
        agen streamCircleSeg2 = init(nullptr);
        wrapper.n1            = 2;
        wrapper.n2            = 3;
        wrapper.n3            = tileX / DVECW * tileY;
        wrapper.s1            = 6;
        wrapper.s2            = inPitch;
        wrapper.s3            = 0;
        INIT_AGEN3(streamCircleSeg2, wrapper);
        streamCircleSeg2         = update_agen_cb_start(streamCircleSeg2, cb2StartAddr);
        streamCircleSeg2         = update_agen_cb_size(streamCircleSeg2, cbSizeBytes);
        params->streamCircleSeg2 = extract_agen_cfg(streamCircleSeg2);
    }
    {
        AgenWrapper wrapper;
        wrapper.size            = sizeof(BASE_TYPE);
        agen streamCircleCenter = init(nullptr);
#if defined(U16_IO) || defined(S16_IO)
        wrapper.n1 = tileX / DVECW * tileY;
        wrapper.s1 = 0;
        INIT_AGEN1(streamCircleCenter, wrapper);
#elif defined(U8_IO) || defined(S8_IO)
        wrapper.n1 = 2; // 2 vld
        wrapper.n2 = tileX / VECW * tileY;
        wrapper.s1 = VECW;
        wrapper.s2 = 0;
        INIT_AGEN2(streamCircleCenter, wrapper);
#endif
        streamCircleCenter         = update_agen_cb_start(streamCircleCenter, cb1StartAddr);
        streamCircleCenter         = update_agen_cb_size(streamCircleCenter, cbSizeBytes);
        params->streamCircleCenter = extract_agen_cfg(streamCircleCenter);
    }
    {
        AgenWrapper wrapper;
        wrapper.size   = sizeof(BASE_TYPE);
        wrapper.n1     = tileX / DVECW * tileY;
        wrapper.s1     = 0;
        agen streamDst = init(nullptr);
        INIT_AGEN1(streamDst, wrapper);
        params->streamDst = extract_agen_cfg(streamDst);
    }
    {
        AgenWrapper wrapper;
        wrapper.size    = sizeof(BASE_TYPE);
        wrapper.n1      = tileX / DVECW;
        wrapper.n2      = tileY;
        wrapper.s1      = DVECW;
        wrapper.s2      = outPitch;
        agen streamZero = init(nullptr);
        INIT_AGEN2(streamZero, wrapper);
        params->streamZero = extract_agen_cfg(streamZero);
    }
}

void fastCornerDetectorScorePartialRej(BASE_TYPE *seg0Ptr, BASE_TYPE *seg1Ptr, BASE_TYPE *seg2Ptr, BASE_TYPE *centerPtr,
                                       BASE_TYPE *outPtr, int32_t threshold, ScoreAgenParam *params,
                                       int32_t inputBaseAdv[NUM_DVEC_PER_TILE],
                                       int32_t outputBaseAdv[NUM_DVEC_PER_TILE])
{
    cupvaModifyAgenCfgBase(&params->streamCircleSeg0, seg0Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleSeg1, seg1Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleSeg2, seg2Ptr);
    cupvaModifyAgenCfgBase(&params->streamCircleCenter, centerPtr);
    cupvaModifyAgenCfgBase(&params->streamDst, outPtr);
    cupvaModifyAgenCfgBase(&params->streamZero, outPtr);

    agen streamCircleSeg0   = init_agen_from_cfg(params->streamCircleSeg0);
    agen streamCircleSeg1   = init_agen_from_cfg(params->streamCircleSeg1);
    agen streamCircleSeg2   = init_agen_from_cfg(params->streamCircleSeg2);
    agen streamCircleCenter = init_agen_from_cfg(params->streamCircleCenter);
    agen streamDst          = init_agen_from_cfg(params->streamDst);
    agen streamZero         = init_agen_from_cfg(params->streamZero);

    chess_separator_scheduler();

    adv_agen_base(streamCircleSeg0, inputBaseAdv[0]);
    adv_agen_base(streamCircleSeg1, inputBaseAdv[0]);
    adv_agen_base(streamCircleSeg2, inputBaseAdv[0]);
    adv_agen_base(streamCircleCenter, inputBaseAdv[0]);
    adv_agen_base(streamDst, outputBaseAdv[0]);

    DVEC_TYPE dv_zero = chess_dont_care(DVEC_TYPE) & 0;
    // If a pixel is not detected as a corner, the output will be zero. Choose to stuff all the tile
    // with zeros, so that later only need to write values for detected corners.
    for (int32_t i = 0; i < params->niter; i++) chess_loop_range(4, )
    chess_unroll_loop(4)
    {
        vstore(dv_zero, streamZero);
    }

    DVEC_TYPE dv_threshold;
    dv_threshold.lo = REPLICATE(threshold);
    dv_threshold.hi = REPLICATE(threshold);

    // params->niterNonRejVec = 1~(tileX/DVECW*tileY-1)
    for (int32_t i = 0; i < params->niterNonRejVec; i++) chess_loop_range(1, )
    chess_prepare_for_pipelining
    {
#if defined(U16_IO) || defined(S16_IO)
        DVEC_TYPE dv_p0 = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p1 = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p2 = DLOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_p15 = DLOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p3  = DLOAD_TYPE(streamCircleSeg1);
#elif defined(U8_IO) || defined(S8_IO)
        DVEC_TYPE dv_p0;
        dv_p0.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p0.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p1;
        dv_p1.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p1.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p2;
        dv_p2.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p2.hi = LOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_p15;
        dv_p15.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p15.hi = LOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p3;
        dv_p3.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p3.hi = LOAD_TYPE(streamCircleSeg1);
#endif
        DVEC_TYPE dv_p14 = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p4  = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p13 = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p5  = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p12 = DLOAD_TYPE(streamCircleSeg2);
        DVEC_TYPE dv_p6  = DLOAD_TYPE(streamCircleSeg2);

#if defined(U16_IO) || defined(S16_IO)
        DVEC_TYPE dv_p11 = DLOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p7  = DLOAD_TYPE(streamCircleSeg1);

        DVEC_TYPE dv_p10 = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p9  = DLOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p8  = DLOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_center = DLOAD_TYPE(streamCircleCenter);
#elif defined(U8_IO) || defined(S8_IO)
        DVEC_TYPE dv_p11;
        dv_p11.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p11.hi = LOAD_TYPE(streamCircleSeg1);
        DVEC_TYPE dv_p7;
        dv_p7.lo = LOAD_TYPE(streamCircleSeg1);
        dv_p7.hi = LOAD_TYPE(streamCircleSeg1);

        DVEC_TYPE dv_p10;
        dv_p10.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p10.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p9;
        dv_p9.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p9.hi = LOAD_TYPE(streamCircleSeg0);
        DVEC_TYPE dv_p8;
        dv_p8.lo = LOAD_TYPE(streamCircleSeg0);
        dv_p8.hi = LOAD_TYPE(streamCircleSeg0);

        DVEC_TYPE dv_center;
        dv_center.lo = LOAD_TYPE(streamCircleCenter);
        dv_center.hi = LOAD_TYPE(streamCircleCenter);
#endif
        adv_agen_base(streamCircleSeg0, inputBaseAdv[i + 1]);
        adv_agen_base(streamCircleSeg1, inputBaseAdv[i + 1]);
        adv_agen_base(streamCircleSeg2, inputBaseAdv[i + 1]);
        adv_agen_base(streamCircleCenter, inputBaseAdv[i + 1]);

        DVEC_TYPE dv_min2_0  = dvmin(dv_p0, dv_p1);
        DVEC_TYPE dv_min2_14 = dvmin(dv_p14, dv_p15);
        DVEC_TYPE dv_min2_2  = dvmin(dv_p2, dv_p3);
        DVEC_TYPE dv_min2_12 = dvmin(dv_p12, dv_p13);
        DVEC_TYPE dv_min2_4  = dvmin(dv_p4, dv_p5);
        DVEC_TYPE dv_min2_10 = dvmin(dv_p10, dv_p11);
        DVEC_TYPE dv_min2_6  = dvmin(dv_p6, dv_p7);
        DVEC_TYPE dv_min2_8  = dvmin(dv_p8, dv_p9);

        DVEC_TYPE dv_min4_0  = dvmin(dv_min2_0, dv_min2_2);
        DVEC_TYPE dv_min4_2  = dvmin(dv_min2_2, dv_min2_4);
        DVEC_TYPE dv_min4_4  = dvmin(dv_min2_4, dv_min2_6);
        DVEC_TYPE dv_min4_6  = dvmin(dv_min2_6, dv_min2_8);
        DVEC_TYPE dv_min4_8  = dvmin(dv_min2_8, dv_min2_10);
        DVEC_TYPE dv_min4_10 = dvmin(dv_min2_10, dv_min2_12);
        DVEC_TYPE dv_min4_12 = dvmin(dv_min2_12, dv_min2_14);
        DVEC_TYPE dv_min4_14 = dvmin(dv_min2_14, dv_min2_0);

        DVEC_TYPE dv_min8_0  = dvmin(dv_min4_0, dv_min4_4);
        DVEC_TYPE dv_min8_2  = dvmin(dv_min4_2, dv_min4_6);
        DVEC_TYPE dv_min8_4  = dvmin(dv_min4_4, dv_min4_8);
        DVEC_TYPE dv_min8_6  = dvmin(dv_min4_6, dv_min4_10);
        DVEC_TYPE dv_min8_8  = dvmin(dv_min4_8, dv_min4_12);
        DVEC_TYPE dv_min8_10 = dvmin(dv_min4_10, dv_min4_14);
        DVEC_TYPE dv_min8_12 = dvmin(dv_min4_12, dv_min4_0);
        DVEC_TYPE dv_min8_14 = dvmin(dv_min4_14, dv_min4_2);

        DVEC_TYPE dv_min9_0  = dvmin(dv_min8_0, dv_p8);
        DVEC_TYPE dv_min9_1  = dvmin(dv_min8_2, dv_p1);
        DVEC_TYPE dv_min9_2  = dvmin(dv_min8_2, dv_p10);
        DVEC_TYPE dv_min9_3  = dvmin(dv_min8_4, dv_p3);
        DVEC_TYPE dv_min9_4  = dvmin(dv_min8_4, dv_p12);
        DVEC_TYPE dv_min9_5  = dvmin(dv_min8_6, dv_p5);
        DVEC_TYPE dv_min9_6  = dvmin(dv_min8_6, dv_p14);
        DVEC_TYPE dv_min9_7  = dvmin(dv_min8_8, dv_p7);
        DVEC_TYPE dv_min9_8  = dvmin(dv_min8_8, dv_p0);
        DVEC_TYPE dv_min9_9  = dvmin(dv_min8_10, dv_p9);
        DVEC_TYPE dv_min9_10 = dvmin(dv_min8_10, dv_p2);
        DVEC_TYPE dv_min9_11 = dvmin(dv_min8_12, dv_p11);
        DVEC_TYPE dv_min9_12 = dvmin(dv_min8_12, dv_p4);
        DVEC_TYPE dv_min9_13 = dvmin(dv_min8_14, dv_p13);
        DVEC_TYPE dv_min9_14 = dvmin(dv_min8_14, dv_p6);
        DVEC_TYPE dv_min9_15 = dvmin(dv_min8_0, dv_p15);

        DVEC_TYPE dv_max_of_min9_red00 = dvmax(dv_min9_0, dv_min9_1);
        DVEC_TYPE dv_max_of_min9_red01 = dvmax(dv_min9_2, dv_min9_3);
        DVEC_TYPE dv_max_of_min9_red02 = dvmax(dv_min9_4, dv_min9_5);
        DVEC_TYPE dv_max_of_min9_red03 = dvmax(dv_min9_6, dv_min9_7);
        DVEC_TYPE dv_max_of_min9_red04 = dvmax(dv_min9_8, dv_min9_9);
        DVEC_TYPE dv_max_of_min9_red05 = dvmax(dv_min9_10, dv_min9_11);
        DVEC_TYPE dv_max_of_min9_red06 = dvmax(dv_min9_12, dv_min9_13);
        DVEC_TYPE dv_max_of_min9_red07 = dvmax(dv_min9_14, dv_min9_15);

        DVEC_TYPE dv_max_of_min9_red0001 = dvmax(dv_max_of_min9_red00, dv_max_of_min9_red01);
        DVEC_TYPE dv_max_of_min9_red0203 = dvmax(dv_max_of_min9_red02, dv_max_of_min9_red03);
        DVEC_TYPE dv_max_of_min9_red0405 = dvmax(dv_max_of_min9_red04, dv_max_of_min9_red05);
        DVEC_TYPE dv_max_of_min9_red0607 = dvmax(dv_max_of_min9_red06, dv_max_of_min9_red07);

        DVEC_TYPE dv_max_of_min9_red00010203 = dvmax(dv_max_of_min9_red0001, dv_max_of_min9_red0203);
        DVEC_TYPE dv_max_of_min9_red04050607 = dvmax(dv_max_of_min9_red0405, dv_max_of_min9_red0607);

        DVEC_TYPE dv_max_of_min9 = dvmax(dv_max_of_min9_red00010203, dv_max_of_min9_red04050607);

        DVEC_TYPE dv_diff_brighter      = dv_max_of_min9 - dv_center;
        DVEC_TYPE dv_is_brighter_corner = dv_diff_brighter > dv_threshold;
        DVEC_TYPE dv_score_brighter     = dvmux(dv_is_brighter_corner, dv_diff_brighter, dv_zero);

        DVEC_TYPE dv_max2_0  = dvmax(dv_p0, dv_p1);
        DVEC_TYPE dv_max2_14 = dvmax(dv_p14, dv_p15);
        DVEC_TYPE dv_max2_2  = dvmax(dv_p2, dv_p3);
        DVEC_TYPE dv_max2_12 = dvmax(dv_p12, dv_p13);
        DVEC_TYPE dv_max2_4  = dvmax(dv_p4, dv_p5);
        DVEC_TYPE dv_max2_10 = dvmax(dv_p10, dv_p11);
        DVEC_TYPE dv_max2_6  = dvmax(dv_p6, dv_p7);
        DVEC_TYPE dv_max2_8  = dvmax(dv_p8, dv_p9);

        DVEC_TYPE dv_max4_0  = dvmax(dv_max2_0, dv_max2_2);
        DVEC_TYPE dv_max4_2  = dvmax(dv_max2_2, dv_max2_4);
        DVEC_TYPE dv_max4_4  = dvmax(dv_max2_4, dv_max2_6);
        DVEC_TYPE dv_max4_6  = dvmax(dv_max2_6, dv_max2_8);
        DVEC_TYPE dv_max4_8  = dvmax(dv_max2_8, dv_max2_10);
        DVEC_TYPE dv_max4_10 = dvmax(dv_max2_10, dv_max2_12);
        DVEC_TYPE dv_max4_12 = dvmax(dv_max2_12, dv_max2_14);
        DVEC_TYPE dv_max4_14 = dvmax(dv_max2_14, dv_max2_0);

        DVEC_TYPE dv_max8_0  = dvmax(dv_max4_0, dv_max4_4);
        DVEC_TYPE dv_max8_2  = dvmax(dv_max4_2, dv_max4_6);
        DVEC_TYPE dv_max8_4  = dvmax(dv_max4_4, dv_max4_8);
        DVEC_TYPE dv_max8_6  = dvmax(dv_max4_6, dv_max4_10);
        DVEC_TYPE dv_max8_8  = dvmax(dv_max4_8, dv_max4_12);
        DVEC_TYPE dv_max8_10 = dvmax(dv_max4_10, dv_max4_14);
        DVEC_TYPE dv_max8_12 = dvmax(dv_max4_12, dv_max4_0);
        DVEC_TYPE dv_max8_14 = dvmax(dv_max4_14, dv_max4_2);

        DVEC_TYPE dv_max9_0  = dvmax(dv_max8_0, dv_p8);
        DVEC_TYPE dv_max9_1  = dvmax(dv_max8_2, dv_p1);
        DVEC_TYPE dv_max9_2  = dvmax(dv_max8_2, dv_p10);
        DVEC_TYPE dv_max9_3  = dvmax(dv_max8_4, dv_p3);
        DVEC_TYPE dv_max9_4  = dvmax(dv_max8_4, dv_p12);
        DVEC_TYPE dv_max9_5  = dvmax(dv_max8_6, dv_p5);
        DVEC_TYPE dv_max9_6  = dvmax(dv_max8_6, dv_p14);
        DVEC_TYPE dv_max9_7  = dvmax(dv_max8_8, dv_p7);
        DVEC_TYPE dv_max9_8  = dvmax(dv_max8_8, dv_p0);
        DVEC_TYPE dv_max9_9  = dvmax(dv_max8_10, dv_p9);
        DVEC_TYPE dv_max9_10 = dvmax(dv_max8_10, dv_p2);
        DVEC_TYPE dv_max9_11 = dvmax(dv_max8_12, dv_p11);
        DVEC_TYPE dv_max9_12 = dvmax(dv_max8_12, dv_p4);
        DVEC_TYPE dv_max9_13 = dvmax(dv_max8_14, dv_p13);
        DVEC_TYPE dv_max9_14 = dvmax(dv_max8_14, dv_p6);
        DVEC_TYPE dv_max9_15 = dvmax(dv_max8_0, dv_p15);

        DVEC_TYPE dv_min_of_max9_red00 = dvmin(dv_max9_0, dv_max9_1);
        DVEC_TYPE dv_min_of_max9_red01 = dvmin(dv_max9_2, dv_max9_3);
        DVEC_TYPE dv_min_of_max9_red02 = dvmin(dv_max9_4, dv_max9_5);
        DVEC_TYPE dv_min_of_max9_red03 = dvmin(dv_max9_6, dv_max9_7);
        DVEC_TYPE dv_min_of_max9_red04 = dvmin(dv_max9_8, dv_max9_9);
        DVEC_TYPE dv_min_of_max9_red05 = dvmin(dv_max9_10, dv_max9_11);
        DVEC_TYPE dv_min_of_max9_red06 = dvmin(dv_max9_12, dv_max9_13);
        DVEC_TYPE dv_min_of_max9_red07 = dvmin(dv_max9_14, dv_max9_15);

        DVEC_TYPE dv_min_of_max9_red0001 = dvmin(dv_min_of_max9_red00, dv_min_of_max9_red01);
        DVEC_TYPE dv_min_of_max9_red0203 = dvmin(dv_min_of_max9_red02, dv_min_of_max9_red03);
        DVEC_TYPE dv_min_of_max9_red0405 = dvmin(dv_min_of_max9_red04, dv_min_of_max9_red05);
        DVEC_TYPE dv_min_of_max9_red0607 = dvmin(dv_min_of_max9_red06, dv_min_of_max9_red07);

        DVEC_TYPE dv_min_of_max9_red00010203 = dvmin(dv_min_of_max9_red0001, dv_min_of_max9_red0203);
        DVEC_TYPE dv_min_of_max9_red04050607 = dvmin(dv_min_of_max9_red0405, dv_min_of_max9_red0607);

        DVEC_TYPE dv_min_of_max9 = dvmin(dv_min_of_max9_red00010203, dv_min_of_max9_red04050607);

        DVEC_TYPE dv_diff_darker      = dv_center - dv_min_of_max9;
        DVEC_TYPE dv_is_darker_corner = dv_diff_darker > dv_threshold;
        DVEC_TYPE dv_score            = dvmux(dv_is_darker_corner, dv_diff_darker, dv_score_brighter);

        vstore(dv_score, streamDst);
        adv_agen_base(streamDst, outputBaseAdv[i + 1]);
    }
}

// Convert number of "size" pixels into float.
void convertPixelToF32(uint32_t *src, int32_t size, float *dst)
{
    int32_t niter = (size + pva_elementsof(vshortx) - 1) / pva_elementsof(vshortx);

    AgenWrapper wrapper;
    wrapper.size  = sizeof(uint16_t);
    wrapper.n1    = niter;
    wrapper.s1    = pva_elementsof(vshortx);
    agen streamIn = init(src);
    INIT_AGEN1(streamIn, wrapper);

    dvfloat *dstDvfloat = (dvfloat *)dst;
    for (int32_t i = 0; i < niter; i++) chess_loop_range(1, )
    chess_prepare_for_pipelining
    {
        dvintx v0 = vshort_dvintx_load(streamIn);

        dvfloatx v1;
        v1.lo = vint_vfp(v0.lo);
        v1.hi = vint_vfp(v0.hi);

        *dstDvfloat++ = extract(v1);
    }
}

// SQDF write from VMEM to DRAM.
void writeExtBuf(SequenceDataFlowHandler &sqdfHdl, bool isFirstWrite, float *src, int32_t size, uint64_t dst)
{
    const int32_t numAxisPerPos = 2;

    if (!isFirstWrite)
    {
        cupvaSQDFSync(sqdfHdl);
    }

    cupvaSQDFUpdateAddr(sqdfHdl, 0, cupvaGetVmemAddress(src), MEMTYPE_VMEM, dst, MEMTYPE_DRAM);

    cupvaSQDFUpdateTileSize(sqdfHdl, 0, size * sizeof(float) * numAxisPerPos, 1);

    cupvaSQDFFlushAndTrig(sqdfHdl);
}

void convertPixelToF32WriteExtBufWrapper(SequenceDataFlowHandler &locSQDFHdl, float locBuf[], int32_t &extBufLocCnt,
                                         int32_t localBufLocCnt, int32_t locCapacity, int32_t pingpong,
                                         uint32_t *packBuf, const ExtMemPointer &locExtPtr)
{
    const int32_t numAxisPerPos = 2;

    int32_t writeCnt = (localBufLocCnt > locCapacity - extBufLocCnt) ? (locCapacity - extBufLocCnt) : localBufLocCnt;
    float *locBufPtr = pingpong ? (locBuf + numAxisPerPos * LOC_BUF_SIZE) : locBuf;

    convertPixelToF32(packBuf, writeCnt * numAxisPerPos, locBufPtr);

    uint64_t dst      = locExtPtr.base + locExtPtr.offset + extBufLocCnt * sizeof(float) * numAxisPerPos;
    bool isFirstWrite = extBufLocCnt == 0;
    writeExtBuf(locSQDFHdl, isFirstWrite, locBufPtr, writeCnt, dst);
    extBufLocCnt += writeCnt;
}

#ifdef NMS_ON
void fastCornerDetectorNmsPackAgenInit(int32_t pitch, int32_t tileX, int32_t tileY, int32_t cbSizeBytes,
                                       intptr_t cbStartAddr, NmsPackAgenParam *params, uint16_t *laneId)
{
    // For U16/S16, it is using double vector to process 32 pixels.
    // For U8/S8, it is using single vector to load 32 pixels then promote to double vector.
    const uint32_t numPixelsPerIteration = 32;

    int32_t niterX = tileX / numPixelsPerIteration;
    int32_t niter  = niterX * tileY;

    params->niterX = niterX;
    params->niter  = niter;
    params->laneId = laneId;

    for (uint32_t i = 0; i < numPixelsPerIteration; i++)
    {
        params->laneId[i] = i;
    }

    // All the following agen's base addresses are initialized with nullptr, since they will be updated when used.
    {
        AgenWrapper wrapper;
        wrapper.size  = sizeof(BASE_TYPE);
        wrapper.n1    = 3;
        wrapper.n2    = 2;
        wrapper.n3    = niterX;
        wrapper.n4    = tileY;
        wrapper.s1    = 1;
        wrapper.s2    = 2 * pitch;
        wrapper.s3    = numPixelsPerIteration;
        wrapper.s4    = pitch;
        agen streamIn = init(nullptr);
        INIT_AGEN4(streamIn, wrapper);
        streamIn         = update_agen_cb_start(streamIn, cbStartAddr);
        streamIn         = update_agen_cb_size(streamIn, cbSizeBytes);
        params->streamIn = extract_agen_cfg(streamIn);
    }
    {
        AgenWrapper wrapper;
        wrapper.size      = sizeof(BASE_TYPE);
        wrapper.n1        = 3;
        wrapper.n2        = niterX;
        wrapper.n3        = tileY;
        wrapper.s1        = 1;
        wrapper.s2        = numPixelsPerIteration;
        wrapper.s3        = pitch;
        agen streamCenter = init(nullptr);
        INIT_AGEN3(streamCenter, wrapper);
        streamCenter         = update_agen_cb_start(streamCenter, cbStartAddr);
        streamCenter         = update_agen_cb_size(streamCenter, cbSizeBytes);
        params->streamCenter = extract_agen_cfg(streamCenter);
    }
}

int32_t fastCornerDetectorNmsPack(BASE_TYPE *tilePtr, BASE_TYPE *centerPtr, NmsPackAgenParam *params, int32_t tilePosX,
                                  int32_t tilePosY, uint32_t chess_storage(RAM_Bb) *restrict packBufPtr
#    ifdef TOPK_ON
                                  ,
                                  uint16_t *restrict packScoreBufPtr, int16_t *histVectorBuf
#    endif // TOPK_ON
)
{
    cupvaModifyAgenCfgBase(&params->streamIn, tilePtr);
    cupvaModifyAgenCfgBase(&params->streamCenter, centerPtr);

    agen streamIn     = init_agen_from_cfg(params->streamIn);
    agen streamCenter = init_agen_from_cfg(params->streamCenter);

    chess_separator_scheduler();

    dvshortx dv_zero = chess_dont_care(dvshortx) & 0;

    dvshortx dv_lane_id     = zero_extend(*(dvushort *)params->laneId);
    dvshortx dv_pos_x_start = dv_lane_id + tilePosX;
    dvshortx dv_pos_x_end   = dv_pos_x_start + (params->niterX - 1) * LANE_ID_TBL_SZ;
    dvshortx dv_pos_x       = dv_pos_x_start;
    vshortx v_pos_y         = replicateh(tilePosY);
    int32_t numLoc          = 0;

#    ifdef TOPK_ON
    dvshortx dv_weight;
    dv_weight.lo = replicateh(1);
    dv_weight.hi = replicateh(1);
#    endif // TOPK_ON

    for (int32_t i = 0; i < params->niter; i++) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        // Input is always unsigned.
// 4 is the center
#    if defined(U16_IO) || defined(S16_IO)
        DVEC_TYPE dv_p0 = dvushort_load(streamIn);
        DVEC_TYPE dv_p1 = dvushort_load(streamIn);
        DVEC_TYPE dv_p2 = dvushort_load(streamIn);

        DVEC_TYPE dv_p6 = dvushort_load(streamIn);
        DVEC_TYPE dv_p7 = dvushort_load(streamIn);
        DVEC_TYPE dv_p8 = dvushort_load(streamIn);

        DVEC_TYPE dv_p3 = dvushort_load(streamCenter);
        DVEC_TYPE dv_p4 = dvushort_load(streamCenter);
        DVEC_TYPE dv_p5 = dvushort_load(streamCenter);
#    elif defined(U8_IO) || defined(S8_IO)
        // Read 32 uint8_t per iteration, since vcollate_idx_bits does not have char version and need to be promoted to short.
        dvshortx dv_p0 = vuchar_dvshortx_load(streamIn);
        dvshortx dv_p1 = vuchar_dvshortx_load(streamIn);
        dvshortx dv_p2 = vuchar_dvshortx_load(streamIn);

        dvshortx dv_p6 = vuchar_dvshortx_load(streamIn);
        dvshortx dv_p7 = vuchar_dvshortx_load(streamIn);
        dvshortx dv_p8 = vuchar_dvshortx_load(streamIn);

        dvshortx dv_p3 = vuchar_dvshortx_load(streamCenter);
        dvshortx dv_p4 = vuchar_dvshortx_load(streamCenter);
        dvshortx dv_p5 = vuchar_dvshortx_load(streamCenter);
#    endif

        dvshortx dv_max0, dv_max2, dv_max5, dv_max7;
        dv_max0 = dvmax(dv_p0, dv_p1);
        dv_max2 = dvmax(dv_p2, dv_p3);
        dv_max5 = dvmax(dv_p5, dv_p6);
        dv_max7 = dvmax(dv_p7, dv_p8);

        dvshortx dv_max02, dv_max57;
        dv_max02 = dvmax(dv_max0, dv_max2);
        dv_max57 = dvmax(dv_max5, dv_max7);

        dvshortx dv_max0257;
        dv_max0257 = dvmax(dv_max02, dv_max57);

        dvshortx dv_peak;
        dv_peak = dv_p4 > dv_max0257;

        dvshortx dv_score;
        dv_score = dvmux(dv_peak, dv_p4, dv_zero);

        // Inline histogram accumulation (TopK only)
#    ifdef TOPK_ON
#        if defined(U8_IO) || defined(S8_IO)
        vhist_simple_32h(histVectorBuf, dv_score, dv_weight);
#        elif defined(U16_IO) || defined(S16_IO)
        dvshortx dv_bins    = dv_score >> 8;
        dvshortx dv_nonzero = dv_score > dv_zero;
        dvshortx dv_hist_weight;
        dv_hist_weight.lo = dv_weight.lo & dv_nonzero.lo;
        dv_hist_weight.hi = dv_weight.hi & dv_nonzero.hi;
        vhist_simple_32h(histVectorBuf, dv_bins, dv_hist_weight);
#        endif
#    endif // TOPK_ON

        // Pack nonzero values and form them into x-y axis.
        int bitsNzIdx;
        vshortx v_idx_lo;
        vcollate_idx_bits(dv_score.lo, v_idx_lo, bitsNzIdx);
        int32_t count_lo = bitcount(bitsNzIdx);

        vshortx v_idx_hi;
        vcollate_idx_bits(dv_score.hi, v_idx_hi, bitsNzIdx);
        int32_t count_hi = bitcount(bitsNzIdx);

        vshortx v_x_lo = vpermute(dv_pos_x.lo, v_idx_lo);
        vshortx v_x_hi = vpermute(dv_pos_x.hi, v_idx_hi);

        dvintx dv_x_int_lo, dv_x_int_hi;
        vpromote(v_x_lo, dv_x_int_lo.lo, dv_x_int_lo.hi);
        vpromote(v_x_hi, dv_x_int_hi.lo, dv_x_int_hi.hi);

        dvintx dv_y_int;
        vpromote_di(v_pos_y, dv_y_int.lo, dv_y_int.hi);

        dvintx dv_loc;
        dv_loc                                     = dvshiftadd(dv_y_int, 16, dv_x_int_lo);
        *(dvint chess_storage(RAM_Bb) *)packBufPtr = extract(dv_loc);
        packBufPtr                                 = (uint32_t chess_storage(RAM_Bb) *)packBufPtr + count_lo;

        dv_loc                                     = dvshiftadd(dv_y_int, 16, dv_x_int_hi);
        *(dvint chess_storage(RAM_Bb) *)packBufPtr = extract(dv_loc);
        packBufPtr                                 = (uint32_t chess_storage(RAM_Bb) *)packBufPtr + count_hi;

        // Pack scores alongside locations (TopK only)
#    ifdef TOPK_ON
        vshortx v_score_lo = vpermute(dv_score.lo, v_idx_lo);
        vshortx v_score_hi = vpermute(dv_score.hi, v_idx_hi);

        *(vshort chess_storage(RAM_Bb) *)packScoreBufPtr = extract(v_score_lo);
        packScoreBufPtr                                  = (uint16_t *)packScoreBufPtr + count_lo;
        *(vshort chess_storage(RAM_Bb) *)packScoreBufPtr = extract(v_score_hi);
        packScoreBufPtr                                  = (uint16_t *)packScoreBufPtr + count_hi;
#    endif // TOPK_ON

        numLoc += count_lo + count_hi;

        dvshortx dv_flag_start_new_row = dv_pos_x == dv_pos_x_end;
        dv_pos_x                       = dvmux(dv_flag_start_new_row, dv_pos_x_start, dv_pos_x + LANE_ID_TBL_SZ);
        v_pos_y                        = vmux(dv_flag_start_new_row.lo, v_pos_y + 1, v_pos_y);
    }

    return numLoc;
}
#else
void fastCornerDetectorPackAgenInit(BASE_TYPE *tilePtr, int32_t pitch, int32_t tileX, int32_t tileY,
                                    NmsPackAgenParam *params, uint16_t *laneId)
{
    // For U16/S16, it is using double vector to process 32 pixels.
    // For U8/S8, it is using single vector to load 32 pixels then promote to double vector.
    const uint32_t numPixelsPerIteration = 32;

    int32_t niterX = tileX / numPixelsPerIteration;
    int32_t niter  = niterX * tileY;

    params->niterX = niterX;
    params->niter  = niter;
    params->laneId = laneId;

    for (uint32_t i = 0; i < LANE_ID_TBL_SZ; i++)
    {
        params->laneId[i] = i;
    }

    {
        AgenWrapper wrapper;
        wrapper.size      = sizeof(BASE_TYPE);
        wrapper.n1        = niterX;
        wrapper.n2        = tileY;
        wrapper.s1        = numPixelsPerIteration;
        wrapper.s2        = pitch;
        agen streamCenter = init(tilePtr);
        INIT_AGEN2(streamCenter, wrapper);
        params->streamCenter = extract_agen_cfg(streamCenter);
    }
}
#endif // NMS_ON

#ifdef TOPK_ON

void InitHistograms(int16_t *histVectorBuf, int32_t *histScalarBuf, HistReduceAgenParam *params)
{
    // Zero-fill vector histogram (int16_t[256 * 32])
    {
        dvshortx zeros        = chess_dont_care(dvshortx) & 0;
        int32_t totalElements = HIST_BINS * HISTOGRAM_VECTOR_WIDTH;
        int32_t niter         = totalElements / pva_elementsof(dvshortx);
        AgenWrapper wrapper;
        wrapper.size  = sizeof(int16_t);
        wrapper.n1    = niter;
        wrapper.s1    = pva_elementsof(dvshortx);
        agen histAgen = init(histVectorBuf);
        INIT_AGEN1(histAgen, wrapper);

        for (int32_t i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(8)
        chess_prepare_for_pipelining
        {
            vstore(zeros, histAgen);
        }
    }

    // Zero-fill scalar histogram (int32_t[256])
    {
        dvintx zeros          = chess_dont_care(dvintx) & 0;
        int32_t totalElements = HIST_BINS;
        int32_t niter         = totalElements / pva_elementsof(dvintx);
        AgenWrapper wrapper;
        wrapper.size  = sizeof(int32_t);
        wrapper.n1    = niter;
        wrapper.s1    = pva_elementsof(dvintx);
        agen histAgen = init(histScalarBuf);
        INIT_AGEN1(histAgen, wrapper);

        for (int32_t i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(8)
        chess_prepare_for_pipelining
        {
            vstore(zeros, histAgen);
        }
    }

    // Init AGEN for histogram reduction
    {
        AgenWrapper histVectorWrapper;
        agen histVectorAgen    = init((dvshort *)histVectorBuf);
        histVectorWrapper.size = sizeof(short);
        histVectorWrapper.n1   = 2;
        histVectorWrapper.s1   = 0;
        histVectorWrapper.n2   = HIST_BINS;
        histVectorWrapper.s2   = pva_elementsof(dvshortx);
        INIT_AGEN2(histVectorAgen, histVectorWrapper);

        params->histVectorAgen = extract_agen_cfg(histVectorAgen);
    }
}

void ReduceVectorToScalar(int32_t *histScalarBuf, HistReduceAgenParam *params)
{
    agen_B histVectorAgen = init_agen_B_from_cfg(params->histVectorAgen);

    dvshortx vhist;
    dvshortx zeros = chess_dont_care(dvshortx) & 0;

    chess_separator_scheduler();

    for (int base = 0; base < HIST_BINS; base += 8) chess_loop_range(32, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        int32_t accum0 = histScalarBuf[base + 0];
        int32_t accum1 = histScalarBuf[base + 1];
        int32_t accum2 = histScalarBuf[base + 2];
        int32_t accum3 = histScalarBuf[base + 3];
        int32_t accum4 = histScalarBuf[base + 4];
        int32_t accum5 = histScalarBuf[base + 5];
        int32_t accum6 = histScalarBuf[base + 6];
        int32_t accum7 = histScalarBuf[base + 7];

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum0 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum1 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum2 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum3 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum4 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum5 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum6 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        vhist    = dvshort_load(histVectorAgen);
        vhist.lo = vhist.lo + vhist.hi;
        accum7 += vsumr_s(vhist.lo);
        vstore(zeros, histVectorAgen);

        histScalarBuf[base + 0] = accum0;
        histScalarBuf[base + 1] = accum1;
        histScalarBuf[base + 2] = accum2;
        histScalarBuf[base + 3] = accum3;
        histScalarBuf[base + 4] = accum4;
        histScalarBuf[base + 5] = accum5;
        histScalarBuf[base + 6] = accum6;
        histScalarBuf[base + 7] = accum7;
    }
}

uint16_t ComputeThreshold(const int32_t *histScalarBuf, const int32_t topK)
{
    int32_t cumulative = 0;

    // 4-way unrolled scan from highest bin downward
    for (int32_t bin = HIST_BINS - 1; bin >= 4; bin -= 4) chess_loop_range(1, )
    {
        int32_t val_n  = histScalarBuf[bin];
        int32_t val_n1 = histScalarBuf[bin - 1];
        int32_t val_n2 = histScalarBuf[bin - 2];
        int32_t val_n3 = histScalarBuf[bin - 3];
        int32_t val_n4 = (bin >= 5) ? histScalarBuf[bin - 4] : 0;

        cumulative += val_n;
        if (cumulative + val_n1 >= topK)
        {
#    if defined(U8_IO) || defined(S8_IO)
            return static_cast<uint16_t>(bin);
#    elif defined(U16_IO) || defined(S16_IO)
            return static_cast<uint16_t>(bin << 8);
#    endif
        }

        cumulative += val_n1;
        if (cumulative + val_n2 >= topK)
        {
#    if defined(U8_IO) || defined(S8_IO)
            return static_cast<uint16_t>(bin - 1);
#    elif defined(U16_IO) || defined(S16_IO)
            return static_cast<uint16_t>((bin - 1) << 8);
#    endif
        }

        cumulative += val_n2;
        if (cumulative + val_n3 >= topK)
        {
#    if defined(U8_IO) || defined(S8_IO)
            return static_cast<uint16_t>(bin - 2);
#    elif defined(U16_IO) || defined(S16_IO)
            return static_cast<uint16_t>((bin - 2) << 8);
#    endif
        }

        cumulative += val_n3;
        if (cumulative + val_n4 >= topK)
        {
#    if defined(U8_IO) || defined(S8_IO)
            return static_cast<uint16_t>(bin - 3);
#    elif defined(U16_IO) || defined(S16_IO)
            return static_cast<uint16_t>((bin - 3) << 8);
#    endif
        }
    }

    // Handle remaining bins (3, 2, 1)
    for (int32_t bin = 3; bin >= 1; bin--)
    {
        int32_t curr_val = histScalarBuf[bin];
        int32_t next_val = histScalarBuf[bin - 1];

        cumulative += curr_val;
        if (cumulative + next_val >= topK)
        {
#    if defined(U8_IO) || defined(S8_IO)
            return static_cast<uint16_t>(bin);
#    elif defined(U16_IO) || defined(S16_IO)
            // For 16-bit, bin 0 contains valid corners with scores 1-255.
            // If we don't yet have enough features without bin 0, include it.
            if (bin == 1 && cumulative < topK)
                return 1;
            return static_cast<uint16_t>(bin << 8);
#    endif
        }
    }

    // Total corners < topK, keep all
    return 1;
}

void writePackedDataSQDF(SequenceDataFlowHandler &handler, uint32_t *packBuf, uint16_t *packScoreBuf, int32_t count,
                         int32_t offset, int32_t pingpong, const ExtMemPointer &packedLocsExtPtr,
                         const ExtMemPointer &packedScoresExtPtr, bool isFirstWrite)
{
    if (count == 0)
        return;

    if (!isFirstWrite)
        cupvaSQDFSync(handler);

    // Transfer 0: Locations (VMEM -> DRAM)
    uint64_t vmemLocsAddr = cupvaGetVmemAddress(&packBuf[pingpong * TOPK_LOC_BUF_SIZE]);
    uint64_t dramLocsAddr = packedLocsExtPtr.base + packedLocsExtPtr.offset + offset * sizeof(uint32_t);
    int32_t locTileSize   = count * sizeof(uint32_t);

    cupvaSQDFUpdateAddr(handler, 0, vmemLocsAddr, MEMTYPE_VMEM, dramLocsAddr, MEMTYPE_DRAM);
    cupvaSQDFUpdateTileSize(handler, 0, locTileSize, 1);

    // Transfer 1: Scores (VMEM -> DRAM)
    uint64_t vmemScoresAddr = cupvaGetVmemAddress(&packScoreBuf[pingpong * TOPK_LOC_BUF_SIZE]);
    uint64_t dramScoresAddr = packedScoresExtPtr.base + packedScoresExtPtr.offset + offset * sizeof(uint16_t);
    int32_t scoreTileSize   = count * sizeof(uint16_t);

    cupvaSQDFUpdateAddr(handler, 1, vmemScoresAddr, MEMTYPE_VMEM, dramScoresAddr, MEMTYPE_DRAM);
    cupvaSQDFUpdateTileSize(handler, 1, scoreTileSize, 1);

    cupvaSQDFFlushAndTrig(handler);
}

void readPackedDataSQDF(SequenceDataFlowHandler &handler, uint32_t *packBuf, uint16_t *packScoreBuf, int32_t count,
                        int32_t offset, int32_t readBufOffset, const ExtMemPointer &packedLocsExtPtr,
                        const ExtMemPointer &packedScoresExtPtr)
{
    if (count == 0)
        return;

    // Transfer 0: Locations (DRAM -> VMEM)
    uint64_t dramLocsAddr = packedLocsExtPtr.base + packedLocsExtPtr.offset + offset * sizeof(uint32_t);
    uint64_t vmemLocsAddr = cupvaGetVmemAddress(&packBuf[readBufOffset]);
    int32_t locTileSize   = count * sizeof(uint32_t);

    cupvaSQDFUpdateAddr(handler, 0, dramLocsAddr, MEMTYPE_DRAM, vmemLocsAddr, MEMTYPE_VMEM);
    cupvaSQDFUpdateTileSize(handler, 0, locTileSize, 1);

    // Transfer 1: Scores (DRAM -> VMEM)
    uint64_t dramScoresAddr = packedScoresExtPtr.base + packedScoresExtPtr.offset + offset * sizeof(uint16_t);
    uint64_t vmemScoresAddr = cupvaGetVmemAddress(&packScoreBuf[readBufOffset]);
    int32_t scoreTileSize   = count * sizeof(uint16_t);

    cupvaSQDFUpdateAddr(handler, 1, dramScoresAddr, MEMTYPE_DRAM, vmemScoresAddr, MEMTYPE_VMEM);
    cupvaSQDFUpdateTileSize(handler, 1, scoreTileSize, 1);

    cupvaSQDFFlushAndTrig(handler);
}

void filterAgenInit(int32_t maxBatchSize, uint16_t *packScoreBuf, uint32_t *packBuf, FilterAgenParam *params)
{
    int32_t scores_dvecw   = pva_elementsof(dvshortx);
    int32_t max_niter_full = maxBatchSize / scores_dvecw;

    // AGEN for loading packed scores
    AgenWrapper scoreWrapper;
    scoreWrapper.size = sizeof(uint16_t);
    scoreWrapper.n1   = max_niter_full;
    scoreWrapper.s1   = scores_dvecw;
    agen scoreAgen    = init(packScoreBuf);
    INIT_AGEN1(scoreAgen, scoreWrapper);
    params->scoreAgen = extract_agen_cfg(scoreAgen);

    // AGEN for loading packed locations with permutation
    AgenWrapper locWrapper;
    locWrapper.size = sizeof(uint32_t);
    locWrapper.n1   = 2; // 2 loads per iteration (lo and hi halves)
    locWrapper.n2   = max_niter_full;
    locWrapper.s1   = scores_dvecw / 2;
    locWrapper.s2   = scores_dvecw;
    agen locAgen    = init(packBuf);
    INIT_AGEN2(locAgen, locWrapper);
    params->locAgen = extract_agen_cfg(locAgen);
}

int32_t filterPackedFeaturesByThresholdVPU(uint32_t *packedLocs, uint16_t *packedScores, int32_t numPacked,
                                           uint16_t threshold, uint32_t *outputLocs, FilterAgenParam *params)
{
    int32_t scores_dvecw = pva_elementsof(dvshortx);
    if (numPacked == 0)
        return 0;

    int32_t numFiltered = 0;
    int32_t niter_full  = numPacked / scores_dvecw;
    int32_t remainder   = numPacked % scores_dvecw;

    cupvaModifyAgenCfgBase(&params->scoreAgen, packedScores);
    cupvaModifyAgenCfgBase(&params->locAgen, packedLocs);

    agen scoreAgen = init_agen_from_cfg(params->scoreAgen);
    agen locAgen   = init_agen_from_cfg(params->locAgen);

    chess_separator_scheduler();

    dvshortx dv_threshold;
    dv_threshold.lo = replicateh(threshold);
    dv_threshold.hi = replicateh(threshold);

    // Vectorized loop: process full 32-element chunks
    for (int32_t iter = 0; iter < niter_full; iter++) chess_prepare_for_pipelining
    {
        dvshortx dv_scores = dvushort_load(scoreAgen);
        dvshortx dv_pass   = dv_scores >= dv_threshold;

        int32_t bitsNzIdx;
        vshortx v_idx_lo;
        vcollate_idx_bits(dv_pass.lo, v_idx_lo, bitsNzIdx);
        const int32_t count_lo = bitcount(bitsNzIdx);

        vshortx v_idx_hi;
        vcollate_idx_bits(dv_pass.hi, v_idx_hi, bitsNzIdx);
        const int32_t count_hi = bitcount(bitsNzIdx);

        dvintx dv_locs_lo                                        = dvint_load_perm(locAgen, v_idx_lo);
        *(dvint chess_storage(RAM_Bb) *)&outputLocs[numFiltered] = extract(dv_locs_lo);
        numFiltered += count_lo;

        dvintx dv_locs_hi                                        = dvint_load_perm(locAgen, v_idx_hi);
        *(dvint chess_storage(RAM_Bb) *)&outputLocs[numFiltered] = extract(dv_locs_hi);
        numFiltered += count_hi;
    }

    // Scalar tail for remaining elements
    if (remainder > 0)
    {
        int32_t baseIdx = niter_full * scores_dvecw;
        for (int32_t i = 0; i < remainder; i++) chess_loop_range(1, )
        chess_prepare_for_pipelining
        {
            uint16_t score = packedScores[baseIdx + i];
            if (score >= threshold)
            {
                outputLocs[numFiltered] = packedLocs[baseIdx + i];
                numFiltered++;
            }
        }
    }

    return numFiltered;
}

#endif // TOPK_ON
