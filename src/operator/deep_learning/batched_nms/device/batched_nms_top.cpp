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

#include "batched_nms_knls.h"

VMEM(C, float, boxesBuf, MAX_BOX_NUM_BEFORE_NMS_F32 * 4); // 128K
VMEM(A, hfloat, scoresBuf, MAX_BOX_NUM_BEFORE_NMS_F32);   // 16K
VMEM(B, short, labelsBuf, MAX_BOX_NUM_BEFORE_NMS_F32);    // 16K

// SortPayload writes key-payload pairs here before the payloads are compacted
// into the first half. The second half is then reused as the suppression mask.
VMEM(A, int, orderMaskBuf, NMS_ORDER_MASK_INTS);                               // 32K + sort tail
VMEM(B, hfloat, scoresTmpBuf, MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD);     // 8K + tail
VMEM(B, float, boxesTmpBuf, (MAX_BOX_NUM_PER_CLASS + NMS_COMPACTION_PAD) * 4); // 64K + tails

VMEM(A, int, sortPingBuf, NMS_SORT_PING_INTS);
VMEM(B, int, sortPongBuf, NMS_SORT_PONG_INTS);

VMEM(A, float, outputBuf, MAX_BOX_NUM_AFTER_NMS * 6); // 24K
VMEM(A, AgenCFG, cfgs, NMS_SORT_CFG_COUNT);

VMEM_SEQDF_HANDLER(B, inputCountTrig);
VMEM_SEQDF_HANDLER(B, inputPayloadTrig);
VMEM_SEQDF_HANDLER(B, outputTrig);
VMEM(B, int, numBuf, 1);

VMEM(B, batched_nms_vpu_param_t, vpu_param);

CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(inputCountTrig);
    cupvaSQDFOpen(inputPayloadTrig);
    cupvaSQDFOpen(outputTrig);

    cupvaSQDFTrig(inputCountTrig);
    cupvaSQDFSync(inputCountTrig);

    int num_boxes = max(0, min(numBuf[0], vpu_param.max_input_boxes));

    cupvaSQDFUpdateTileSize(inputPayloadTrig, 0, num_boxes * sizeof(float), 4);
    cupvaSQDFUpdateTileSize(inputPayloadTrig, 1, num_boxes * sizeof(short), 1);
    cupvaSQDFUpdateTileSize(inputPayloadTrig, 2, num_boxes * sizeof(short), 1);
    cupvaSQDFFlushAndTrig(inputPayloadTrig);
    cupvaSQDFSync(inputPayloadTrig);
    float threshold = vpu_param.threshold;
    int topk        = vpu_param.topk;

    int *orderBuf   = orderMaskBuf;
    int *maskBuf    = orderMaskBuf + MAX_BOX_NUM_PER_CLASS + NMS_SORT_TAIL_PAD;
    int num_to_keep = nms_suppression_f32(labelsBuf, scoresBuf, boxesBuf, orderBuf, maskBuf, scoresTmpBuf, boxesTmpBuf,
                                          sortPingBuf, sortPongBuf, cfgs, outputBuf, MAX_BOX_NUM_AFTER_NMS, num_boxes,
                                          MAX_BOX_NUM_BEFORE_NMS_F32, threshold, topk);
    numBuf[0]       = num_to_keep;

    cupvaSQDFUpdateTileSize(outputTrig, 0, num_to_keep * sizeof(float), 4);
    cupvaSQDFUpdateTileSize(outputTrig, 1, num_to_keep * sizeof(float), 1);
    cupvaSQDFUpdateTileSize(outputTrig, 2, num_to_keep * sizeof(int), 1);
    cupvaSQDFFlushAndTrig(outputTrig);
    cupvaSQDFSync(outputTrig);

    cupvaSQDFClose(inputCountTrig);
    cupvaSQDFClose(inputPayloadTrig);
    cupvaSQDFClose(outputTrig);
    return 0;
}
