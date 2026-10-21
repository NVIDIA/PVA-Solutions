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

/* YOLOv5 decode kernel (VPU device code). */

#include <cupva_device.h>

#define minus_log2exp -1.4426950216f
inline float sigmoid(float x)
{
    return frcp(fexp2(x * minus_log2exp) + 1.0f);
}

int decode_knl(float *predict, float *anchor, float *output_left, float *output_top, float *output_right,
               float *output_bottom, hfloat *output_conf, short *output_class, int num_bboxes, int num_classes,
               int max_survivors, float confidence_threshold, float deconfidence_threshold, float image_height,
               float image_width)
{
    int vecw = chess_elementsof(dvfloatx);
    /*
     * Create accepts exactly 80 classes. A dvfloatx contains 16 FP32 lanes,
     * so every accepted submission has five complete class vectors and no
     * inactive lanes can enter the vector FP path. Keep this guard here as a
     * device-side backstop if the host contract changes.
     */
    if ((num_classes % vecw) != 0)
        return 0;
    int niter               = num_classes / vecw;
    int total_num_survivors = 0;
    for (int i = 0; i < num_bboxes; i++)
    {
        if (total_num_survivors >= max_survivors)
            break;

        float *pitem     = predict + i * (num_classes + 5);
        float objectness = pitem[4];
        if (objectness <= deconfidence_threshold)
            continue;

        objectness  = sigmoid(objectness);
        agen a_conf = init(pitem + 5);
        a_conf.n1   = niter;
        a_conf.mod1 = vecw * sizeof(float);

        int num_survivors_i = 0;

        for (int j = 0; j < niter; j++) chess_loop_range(2, )
        chess_prepare_for_pipelining
        {
            dvfloatx vconf = dvfloat_load_di(a_conf);

            dvfloatx vconf_sigmoid = dvfrcp(dvfexp2(vconf * minus_log2exp) + 1.0f);
            dvfloatx vcls_conf     = vconf_sigmoid * objectness;

            vhfloatx vcls_conf_fp16 = dvfp_vfp16(vcls_conf); // matches load_di
            // Compact in the FP16 lane order consumed by vpermute. Include
            // the rounded threshold boundary here, then enforce the public
            // F32 threshold contract on the compacted candidates below.
            vshortx vcmp = vcls_conf_fp16 >= fp_fp16(confidence_threshold);
            vshortx vindex;
            int index_mask = 0;
            vcollate_idx_bits(vcmp, vindex, index_mask);
            vhfloatx vconf_out = vpermute(vcls_conf_fp16, vindex);

            *((vhfloat *)(output_conf + total_num_survivors)) = extract(vconf_out);
            *((vshort *)(output_class + total_num_survivors)) = extract(vindex + j * vecw);

            int num_candidates = bitcount(index_mask);
            int num_survivors  = 0;
            for (int k = 0; k < num_candidates; ++k)
            {
                short class_id       = output_class[total_num_survivors + k];
                float exact_conf     = sigmoid(pitem[5 + class_id]) * objectness;
                int compacted_offset = total_num_survivors + num_survivors;
                if (exact_conf > confidence_threshold)
                {
                    output_conf[compacted_offset]  = output_conf[total_num_survivors + k];
                    output_class[compacted_offset] = class_id;
                    ++num_survivors;
                }
            }
            num_survivors = min(num_survivors, max_survivors - total_num_survivors);
            total_num_survivors += num_survivors;
            num_survivors_i += num_survivors;

            if (total_num_survivors >= max_survivors)
                break;
        }
        for (int j = 0; j < num_survivors_i; j++)
        {
            float predict_cx = sigmoid(pitem[0]);
            float predict_cy = sigmoid(pitem[1]);
            float predict_w  = sigmoid(pitem[2]);
            float predict_h  = sigmoid(pitem[3]);

            const float *prior_ptr = anchor + i * 5;
            float stride           = prior_ptr[4];
            float cx               = fmadd(predict_cx, 2 * stride, (-0.5f + prior_ptr[0]) * stride);
            float cy               = fmadd(predict_cy, 2 * stride, (-0.5f + prior_ptr[1]) * stride);
            float double_w         = predict_w * 2;
            float double_h         = predict_h * 2;
            float width            = double_w * double_w * prior_ptr[2];
            float height           = double_h * double_h * prior_ptr[3];
            float left             = fmsub(width, 0.5f, cx);
            float top              = fmsub(height, 0.5f, cy);
            float right            = fmadd(width, 0.5f, cx);
            float bottom           = fmadd(height, 0.5f, cy);

            left   = mux(left < 0.0f, 0.0f, left);
            top    = mux(top < 0.0f, 0.0f, top);
            right  = mux(right > image_width, image_width, right);
            bottom = mux(bottom > image_height, image_height, bottom);

            *output_left++   = left;
            *output_top++    = top;
            *output_right++  = right;
            *output_bottom++ = bottom;
        }
    }
    return total_num_survivors;
}
