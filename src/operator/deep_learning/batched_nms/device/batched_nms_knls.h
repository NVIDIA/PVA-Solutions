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

#include "pvaAplSortPayloadVpu.hpp"

#include "batched_nms_param.h"

#include <cupva_device.h>

// Sort storage is sized from the largest SortPayload instantiation that
// run_sort_payload() dispatches to, which covers every smaller one.
using NmsSortMax = pvaApl::SortPayloadVpu<int32_t, 8192>;

constexpr size_t NMS_SORT_CFG_COUNT = NmsSortMax::CFG_COUNT;
constexpr size_t NMS_SORT_PING_INTS = NmsSortMax::MIN_INPUT_BUFFER_SIZE / sizeof(int) + NMS_SORT_TAIL_PAD;
constexpr size_t NMS_SORT_PONG_INTS = NmsSortMax::SCRATCH_SIZE / sizeof(int) + NMS_SORT_TAIL_PAD;

// SortPayload writes its key-payload output across this whole buffer. The
// payloads are compacted into the first half before the second half is reused
// as the suppression mask, so the sort output bound below covers both halves.
constexpr size_t NMS_ORDER_MASK_INTS = MAX_BOX_NUM_PER_CLASS * 2 + NMS_SORT_TAIL_PAD;

inline int get_sort_pad_size(int arr_len)
{
    // Make sure height is at least 32
    if (arr_len <= 256)
        return 256;

    // Method from Hacker's Delight
    size_t x = arr_len - 1;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return (x + 1);
}

template<int Size>
inline void run_sort_payload(int *sortInput, int *sortOutput, int *sortScratch, AgenCFG *sortCfgs)
{
    using Sorter = pvaApl::SortPayloadVpu<int32_t, Size>;

    static_assert(Sorter::CFG_COUNT <= NMS_SORT_CFG_COUNT, "sort cfgs buffer is too small");
    static_assert(Sorter::MIN_INPUT_BUFFER_SIZE <= NMS_SORT_PING_INTS * sizeof(int), "sort ping buffer is too small");
    static_assert(Sorter::SCRATCH_SIZE <= NMS_SORT_PONG_INTS * sizeof(int), "sort pong buffer is too small");
    static_assert(Sorter::MIN_OUTPUT_BUFFER_SIZE <= NMS_ORDER_MASK_INTS * sizeof(int),
                  "sort output buffer is too small");

    Sorter sorter;
    sorter.Init(sortInput, sortOutput, sortCfgs, reinterpret_cast<uint8_t *>(sortScratch));
    sorter.Execute(sortCfgs);
}

void inline sort_with_payload_int(hfloat *scores, int *order, int *sortPing, int *sortPong, int length,
                                  AgenCFG *sortCfgs)
{
    // SortPayload operates on contiguous key-payload pairs and pads the
    // intermediate rows in its scratch buffer.
    int padded_box_count = get_sort_pad_size(length);
    int height           = padded_box_count / 8;
    // Pre-process
    {
        int vecw         = pva_elementsof(dvintx);
        int pair_per_row = 8;
        int niter        = (length + pair_per_row - 1) / pair_per_row;
        niter            = (niter + 1) / 2 * 2; // pad to multiples of 2

        agen a0 = init(scores);
        a0.n1   = niter / 2;
        a0.mod1 = vecw * sizeof(short);

        agen a2 = init(sortPing);
        a2.n1   = niter;
        a2.mod1 = vecw * sizeof(int);

        for (int i = 0; i < niter / 2; i++)
        {
            dvintx vin0     = vushort_dvintx_load(a0);
            vintx vorder_lo = vcollate_idx(replicatew(1)) + i * vecw;
            vintx vorder_hi = vorder_lo + vecw / 2;
            dvintx vout0;
            // SortPayload orders ascending. Scores must be finite and positive:
            // for such FP16 values the bit pattern grows with the value, so the
            // negated bit pattern sorts scores in descending order and places the
            // zero-filled padding after every valid detection. Zero, negative,
            // infinite or NaN scores do not have a defined position in the order.
            vout0.lo = -vin0.lo;
            vout0.hi = vorder_lo;
            dvintx vout1;
            vout1.lo = -vin0.hi;
            vout1.hi = vorder_hi;
            vstore_i(vout0, a2);
            vstore_i(vout1, a2);
        }
        // Clear padded segment
        dvintx v0;
        v0.lo       = replicatew(0);
        v0.hi       = replicatew(0);
        int offset  = length * 2;
        int niter_r = (padded_box_count * 2 - offset + vecw - 1) / vecw;
        agen a3     = init((dvuint *)(sortPing + offset));
        a3.n1       = niter_r;
        a3.mod1     = vecw * sizeof(int);
        for (int i = 0; i < niter_r; i++)
        {
            vstore(v0, a3);
        }
    }

    int padded_size = height * 16;
    if (padded_size == 512)
        run_sort_payload<512>(sortPing, order, sortPong, sortCfgs);
    else if (padded_size == 1024)
        run_sort_payload<1024>(sortPing, order, sortPong, sortCfgs);
    else if (padded_size == 2048)
        run_sort_payload<2048>(sortPing, order, sortPong, sortCfgs);
    else if (padded_size == 4096)
        run_sort_payload<4096>(sortPing, order, sortPong, sortCfgs);
    else
        run_sort_payload<8192>(sortPing, order, sortPong, sortCfgs);

    int *ptrSortOut = order;

    // Post-process
    {
        int vecw          = pva_elementsof(dvintx);
        int niter1        = height * 2 / vecw;
        int elems_per_row = height * 2;
        int niter2        = (length * 2 + elems_per_row - 1) / elems_per_row;
        int niter         = niter1 * niter2;
        int pitch         = height * 2 + 2;

        agen a0 = init((dvuint *)ptrSortOut);
        AgenWrapper w0;
        w0.size = sizeof(int);
        w0.n1   = niter1;
        w0.s1   = vecw;
        w0.n2   = niter2;
        w0.s2   = pitch;
        INIT_AGEN2(a0, w0);

        agen a2 = init((dvuint *)order);
        a2.n1   = niter / 2;
        a2.mod1 = vecw * sizeof(int);
        for (int i = 0; i < niter / 2; i++)
        {
            dvintx data0 = dvint_load_di(a0);
            dvintx data1 = dvint_load_di(a0);
            dvintx vorder;
            vorder.lo = data0.hi;
            vorder.hi = data1.hi;
            vstore(vorder, a2);
        }
    }
}

int nms_suppression_f32(short *labels, hfloat *scores, float *boxesIn, int *order, int *supp_mask,
                        hfloat *scores_per_class, float *boxes_per_class, int *sort_ping, int *sort_pong,
                        AgenCFG *sort_cfgs, float *output, int output_lp, int num_boxes, int boxes_lp, float threshold,
                        int topk)
{

    // Update label mapping (mapped labels start from 1)
    int num_labels                       = 0;
    short unique_labels[MAX_NUM_CLASSES] = {0};
    short label_mapping[MAX_NUM_CLASSES] = {0};
    for (int i = 0; i < num_boxes; i++)
    {
        short cur_label = labels[i];
        if ((unsigned short)cur_label >= MAX_NUM_CLASSES)
        {
            continue;
        }
        if (!label_mapping[cur_label])
        {
            unique_labels[num_labels] = cur_label;
            label_mapping[cur_label]  = ++num_labels;
        }
    }

    int vecw = chess_elementsof(dvfloat); // = chess_elementsof(vshort)

    // Clear tail
    int num_boxes_padded = (num_boxes + vecw - 1) / vecw * vecw;
    for (int i = num_boxes; i < num_boxes_padded; i++)
    {
        labels[i] = -1;
    }

    float *ptrX0Select     = output;
    float *ptrY0Select     = output + output_lp;
    float *ptrX1Select     = output + output_lp * 2;
    float *ptrY1Select     = output + output_lp * 3;
    float *ptrScoresSelect = output + output_lp * 4;
    int *ptrLabelSelect    = (int *)output + output_lp * 5;

    int niter_box_in = (num_boxes + vecw - 1) / vecw;
    AgenWrapper wrapper0;
    wrapper0.size = sizeof(short);
    wrapper0.n1   = niter_box_in;
    wrapper0.s1   = vecw;
    wrapper0.n2   = num_labels;
    wrapper0.s2   = 0;
    agen a_label  = init(labels);
    INIT_AGEN2(a_label, wrapper0);

    agen a_score = init(scores);
    INIT_AGEN2(a_score, wrapper0);

    wrapper0.size = sizeof(float);
    wrapper0.n1   = 4;
    wrapper0.s1   = boxes_lp;
    wrapper0.n2   = niter_box_in;
    wrapper0.s2   = vecw;
    wrapper0.n3   = num_labels;
    wrapper0.s3   = 0;
    agen a_box_in = init(boxesIn);
    INIT_AGEN3(a_box_in, wrapper0);

    // Suppression
    int num_to_keep = 0;
    for (int nl = 0; nl < num_labels; nl++)
    {

        // Gather this class's scores and boxes.
        short cur_label = unique_labels[nl];
        int cur_cnt     = 0;
        int max_cnt     = MAX_BOX_NUM_PER_CLASS;
        float *ptrX0Out = boxes_per_class;
        float *ptrY0Out = boxes_per_class + MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD;
        float *ptrX1Out = boxes_per_class + (MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD) * 2;
        float *ptrY1Out = boxes_per_class + (MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD) * 3;
        for (int i = 0; i < niter_box_in; i++)
        {
            vshortx vlabel = vshort_load(a_label);
            vshortx vcmp   = (vlabel == cur_label);
            // These agens span [box-vector, class]. Consume the complete box
            // dimension for every class even after the class buffer is full, or
            // the next class starts from a stale position in the shared agen.
            if (cur_cnt >= max_cnt)
                vcmp = replicateh(0);
            vshortx vpos;
            int mask;
            vcollate_idx_bits(vcmp, vpos, mask);
            vhfloatx vscore                            = vpermute(vhfloat_load(a_score), vpos);
            *((vhfloat *)(scores_per_class + cur_cnt)) = extract(vscore);
            *((dvfloat *)(ptrX0Out + cur_cnt))         = extract(dvfloat_load_perm(a_box_in, vpos));
            *((dvfloat *)(ptrY0Out + cur_cnt))         = extract(dvfloat_load_perm(a_box_in, vpos));
            *((dvfloat *)(ptrX1Out + cur_cnt))         = extract(dvfloat_load_perm(a_box_in, vpos));
            *((dvfloat *)(ptrY1Out + cur_cnt))         = extract(dvfloat_load_perm(a_box_in, vpos));
            cur_cnt += bitcount(mask);
            if (cur_cnt > max_cnt)
            {
                cur_cnt = max_cnt;
            }
        }

        if (cur_cnt > 1)
            sort_with_payload_int(scores_per_class, order, sort_ping, sort_pong, cur_cnt, sort_cfgs);
        else
            order[0] = 0;

        // Compute box areas and initialize the suppression mask.
        int niter_per_class = (cur_cnt + vecw - 1) / vecw;
        AgenWrapper wrapper;
        wrapper.size = sizeof(float);
        wrapper.n1   = 4;
        wrapper.s1   = MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD;
        wrapper.n2   = niter_per_class;
        wrapper.s2   = vecw;
        wrapper.n3   = cur_cnt + 1;
        wrapper.s3   = 0;
        agen a_box   = init(boxes_per_class);
        INIT_AGEN3(a_box, wrapper);

        wrapper.n1    = niter_per_class;
        wrapper.s1    = vecw;
        wrapper.n2    = cur_cnt + 1;
        wrapper.s2    = 0;
        agen a_mask_r = init(supp_mask);
        INIT_AGEN2(a_mask_r, wrapper);
        agen a_mask_w = a_mask_r;

        float *area_per_class = (float *)sort_ping;
        agen a_area           = init(area_per_class);
        INIT_AGEN2(a_area, wrapper);

        chess_separator_scheduler();

        dvfloatx vzero = as_dvfloatx(chess_dont_care(dvintx) & 0);
        for (int i = 0; i < niter_per_class; i++)
        {
            dvfloatx vx0   = dvfloat_load(a_box);
            dvfloatx vy0   = dvfloat_load(a_box);
            dvfloatx vx1   = dvfloat_load(a_box);
            dvfloatx vy1   = dvfloat_load(a_box);
            dvfloatx varea = (vx1 - vx0) * (vy1 - vy0);
            vstore(varea, a_area);
            vstore(vzero, a_mask_w);
        }

        int num_to_keep_cur_class = 0;
        for (int ii = 0; ii < cur_cnt && num_to_keep < topk; ii++)
        {
            int i = order[ii];
            if (supp_mask[i])
            {
                continue;
            }
            float iarea                  = area_per_class[i];
            float ix0                    = boxes_per_class[i];
            float iy0                    = boxes_per_class[i + MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD];
            float ix1                    = boxes_per_class[i + (MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD) * 2];
            float iy1                    = boxes_per_class[i + (MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD) * 3];
            ptrX0Select[num_to_keep]     = ix0;
            ptrY0Select[num_to_keep]     = iy0;
            ptrX1Select[num_to_keep]     = ix1;
            ptrY1Select[num_to_keep]     = iy1;
            ptrScoresSelect[num_to_keep] = fp16_fp(scores_per_class[i]);
            ptrLabelSelect[num_to_keep]  = cur_label;
            num_to_keep++;
            num_to_keep_cur_class++;

            for (int j = 0; j < niter_per_class; j++)
            {
                dvfloatx varea  = dvfloat_load(a_area);
                dvfloatx vx0    = dvfloat_load(a_box);
                dvfloatx vy0    = dvfloat_load(a_box);
                dvfloatx vx1    = dvfloat_load(a_box);
                dvfloatx vy1    = dvfloat_load(a_box);
                dvfloatx vxx0   = dvmux(vx0 < ix0, ix0, vx0); // max
                dvfloatx vyy0   = dvmux(vy0 < iy0, iy0, vy0);
                dvfloatx vxx1   = dvmux(vx1 > ix1, ix1, vx1); // min
                dvfloatx vyy1   = dvmux(vy1 > iy1, iy1, vy1);
                dvfloatx vw     = vxx1 - vxx0;
                dvfloatx vh     = vyy1 - vyy0;
                vw              = dvmux(vw < 0, 0, vw);
                vh              = dvmux(vh < 0, 0, vh);
                dvfloatx vinter = vw * vh;
                dvfloatx vuni   = varea + iarea - vinter;
                dvintx vsupp    = (vinter > (vuni * threshold));
                dvintx vmask    = dvint_load(a_mask_r);
                vstore(vmask | vsupp, a_mask_w);
            }
        }
    }
    return num_to_keep;
}
