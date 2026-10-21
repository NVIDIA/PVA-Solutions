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
 * YOLOv5 decode VPU top-level kernel.
 *
 * Input: CHW [255, H*W] or HWC [H*W, 255] prediction tensor (one scale), read via RDF in
 * double-buffered tiles of spatial_tile_size grid cells in VMEM C. CHW tiles are transposed
 * by transpose_process_buf() from channel-major [255, line_pitch] to anchor-major
 * [tile_size, 85] in VMEM B before decode_knl(); HWC tiles are already anchor-major and are
 * decoded in place.
 */

#include "yolov5_decode_knls.h"
#include "yolov5_decode_param.h"

#define VEC_WIDTH 8   /* vfloatx lanes (Gen2 256-bit) */
#define VMEM_ROW 16   /* floats per 64-byte VMEM row  */
#define NUM_CH 255    /* 3 anchors x 85 attributes    */
#define C_ITERS 32    /* ceil(NUM_CH / VEC_WIDTH)      */
#define LANE_OFST_K 3 /* line_pitch 49 = VMEM_ROW*3+1  */

/* VMEM A: output buffers (~102 KB) */
VMEM(A, float, outputBuf, VPU_MAX_OBJECTS * 4);
// Compaction emits a full vector before advancing by the selected-lane count.
// Padding keeps the final capacity-bounded vector store inside VMEM.
VMEM(A, hfloat, outputConfBuf, VPU_MAX_OBJECTS + 32);
VMEM(A, short, outputClsBuf, VPU_MAX_OBJECTS + 32);

/* VMEM B: anchor ping/pong + transposed prediction + misc (~57 KB) */
VMEM(B, float, anchorBuf, 2 * 147 * 5);              /* 5,880 bytes - double-buffered anchor tiles (49*3=147) */
VMEM(B, float, transposedBuf, 147 * 85 + VEC_WIDTH); /* transposed [147, 85] tile + overflow pad */

VMEM(B, int, numSurvivors, 1);
VMEM_POINTER(B, outputPtr);
VMEM_POINTER(B, outputConfPtr);
VMEM_POINTER(B, outputClsPtr);

VMEM_SEQDF_HANDLER(B, anchorTrig);
VMEM_SEQDF_HANDLER(B, outputTrig);
VMEM_SEQDF_HANDLER(B, outputConfTrig);
VMEM_SEQDF_HANDLER(B, outputClsTrig);
VMEM_SEQDF_HANDLER(B, outputNumTrig);
VMEM_SEQDF_HANDLER(B, inputNumTrig);

VMEM(B, yolov5_decode_host2dev_param_t, vpu_param);

/* VMEM C: double-buffered RDF input (~100 KB), holding CHW [256, 49] or HWC [49, 255] tiles */
VMEM_RDF_UNIFIED(C, inputRdfHdl);
// CHW transpose performs one final 8-lane load for channels 248..255. Channel
// 255 is intentionally discarded, but its load must still land in the tile
// arena. Reserve one padding row in each double-buffered arena.
VMEM(C, float, inputBuf, RDF_DOUBLE(float, 49, 256));

/* CHW -> anchor-major transpose (vectorized) */
/*
 * Vectorized [255, line_pitch] -> [cur_spatial, 255] == [cur_spatial*3, 85].
 *
 * vfloat_load_transp reads 8 channels per iteration across VMEM rows;
 * line_pitch = 49 = 16*3+1 satisfies the bank-conflict-free constraint.
 * Each spatial group writes 32*8 = 256 floats; the 256th (junk) overlaps
 * the next group's first element and gets overwritten by its first vstore.
 *
 * n2 is set to spatial_tile_size (the max); for the last tile we simply
 * run fewer iterations - the agen addresses are correct for any prefix.
 */

static AgenWrapper s_inWrap;
static AgenWrapper s_outWrap;

static void transpose_init(int32_t line_pitch, int32_t spatial_tile_size)
{
    s_inWrap.size = sizeof(float);
    s_inWrap.n1   = C_ITERS;
    s_inWrap.n2   = spatial_tile_size;
    s_inWrap.s1   = VEC_WIDTH * line_pitch;
    s_inWrap.s2   = 1;

    AgenWrapper outWrap;
    outWrap.size = sizeof(float);
    outWrap.n1   = C_ITERS;
    outWrap.n2   = spatial_tile_size;
    outWrap.s1   = VEC_WIDTH;
    outWrap.s2   = NUM_CH;

    s_outWrap = outWrap;
}

static void transpose_process_buf(const float *src, int32_t total_iters)
{
    agen inAgen = init((vfloatx *)src);
    INIT_AGEN2(inAgen, s_inWrap);
    inAgen.lane_ofst = LANE_OFST_K;
    AgenCFG inCfg    = extract_agen_cfg(inAgen);

    /* The output agen is rebuilt per call because a stored AgenCFG does not
       carry the full agen state. */
    agen outAgen  = init((vfloatx *)transposedBuf);
    outAgen.round = 0;
    INIT_AGEN2(outAgen, s_outWrap);

    agen_A in  = init_agen_A_from_cfg(inCfg);
    agen_B out = init_agen_B_from_cfg(extract_agen_cfg(outAgen));

    const int32_t main_iters = (total_iters / 6) * 6;
    const int32_t tail_iters = total_iters - main_iters;

    for (int32_t i = 0; i < main_iters; i++) chess_unroll_loop(6)
    chess_prepare_for_pipelining
    chess_loop_range(6, )
    {
        vfloatx v = vfloat_load_transp(in);
        vstore(v, out);
    }

    for (int32_t i = 0; i < tail_iters; i++)
    {
        vfloatx v = vfloat_load_transp(in);
        vstore(v, out);
    }
}

/* Main */
CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(anchorTrig);
    cupvaSQDFOpen(outputTrig);
    cupvaSQDFOpen(outputConfTrig);
    cupvaSQDFOpen(outputClsTrig);
    cupvaSQDFOpen(outputNumTrig);
    cupvaSQDFOpen(inputNumTrig);

    /* Read the current count from DRAM so we can append after previous invocations */
    cupvaSQDFTrig(inputNumTrig);
    cupvaSQDFSync(inputNumTrig);
    int tile_size          = vpu_param.tile_size;
    int num_classes        = vpu_param.num_classes;
    float conf_threshold   = vpu_param.conf_threshold;
    float deconf_threshold = vpu_param.deconf_threshold;
    int anchor_item_size   = vpu_param.anchor_item_size;
    int anchor_tile_size   = tile_size * anchor_item_size;
    int num_tiles          = vpu_param.num_tiles;
    int max_objects        = vpu_param.max_objects;
    float image_height     = (float)(vpu_param.image_height);
    float image_width      = (float)(vpu_param.image_width);
    int spatial_tile_size  = vpu_param.spatial_tile_size;
    int last_spatial_tile  = vpu_param.last_spatial_tile;

    int hwc_input = vpu_param.hwc_input;

    int initial_count = max(0, min(numSurvivors[0], max_objects));

    int32_t line_pitch = cupvaRasterDataFlowGetLinePitch(inputRdfHdl);

    if (!hwc_input)
        transpose_init(line_pitch, spatial_tile_size);

    int num_survivors_counter = 0;
    int num_survivors_total   = initial_count;

    float *output_box         = outputBuf;
    hfloat *output_conf       = outputConfBuf;
    short *output_cls         = outputClsBuf;
    uint64_t output_box_addr  = outputPtr.base + outputPtr.offset + initial_count * sizeof(float);
    uint64_t output_conf_addr = outputConfPtr.base + outputConfPtr.offset + initial_count * sizeof(short);
    uint64_t output_cls_addr  = outputClsPtr.base + outputClsPtr.offset + initial_count * sizeof(short);

    cupvaSQDFTrig(anchorTrig);

    cupvaRasterDataFlowOpen(inputRdfHdl, &inputBuf[0]);

    for (int i = 0; i < num_tiles; i++)
    {
        int cur_spatial = (i == num_tiles - 1) ? last_spatial_tile : spatial_tile_size;
        int cur_tile    = cur_spatial * 3;

        float *inputTile = (float *)cupvaRasterDataFlowAcquire(inputRdfHdl);

        cupvaSQDFSync(anchorTrig);
        if (i < num_tiles - 1)
            cupvaSQDFTrig(anchorTrig);

        float *decodeSrc;
        if (hwc_input)
        {
            decodeSrc = inputTile;
        }
        else
        {
            transpose_process_buf(inputTile, C_ITERS * cur_spatial);
            cupvaRasterDataFlowRelease(inputRdfHdl);
            decodeSrc = transposedBuf;
        }

        int remaining_capacity = max(0, max_objects - num_survivors_total);
        int cur_num_survivors  = decode_knl(
            decodeSrc, anchorBuf + (i % 2) * anchor_tile_size, output_box, output_box + VPU_MAX_OBJECTS,
            output_box + VPU_MAX_OBJECTS * 2, output_box + VPU_MAX_OBJECTS * 3, output_conf, output_cls, cur_tile,
            num_classes, remaining_capacity, conf_threshold, deconf_threshold, image_height, image_width);

        if (hwc_input)
            cupvaRasterDataFlowRelease(inputRdfHdl);

        num_survivors_counter += cur_num_survivors;
        num_survivors_total += cur_num_survivors;

        int will_flush = (num_survivors_counter >= VPU_MAX_OBJECTS || i == num_tiles - 1);

        if (will_flush)
        {
            cupvaSQDFUpdateAddr(outputTrig, 0, cupvaGetVmemAddress((void *)outputBuf), MEMTYPE_VMEM, output_box_addr,
                                MEMTYPE_DRAM);
            cupvaSQDFUpdateTileSize(outputTrig, 0, num_survivors_counter * sizeof(float), 4);
            cupvaSQDFFlushAndTrig(outputTrig);

            cupvaSQDFUpdateAddr(outputConfTrig, 0, cupvaGetVmemAddress((void *)outputConfBuf), MEMTYPE_VMEM,
                                output_conf_addr, MEMTYPE_DRAM);
            cupvaSQDFUpdateTileSize(outputConfTrig, 0, num_survivors_counter * sizeof(short), 1);
            cupvaSQDFFlushAndTrig(outputConfTrig);

            cupvaSQDFUpdateAddr(outputClsTrig, 0, cupvaGetVmemAddress((void *)outputClsBuf), MEMTYPE_VMEM,
                                output_cls_addr, MEMTYPE_DRAM);
            cupvaSQDFUpdateTileSize(outputClsTrig, 0, num_survivors_counter * sizeof(short), 1);
            cupvaSQDFFlushAndTrig(outputClsTrig);

            output_box_addr += num_survivors_counter * sizeof(float);
            output_conf_addr += num_survivors_counter * sizeof(short);
            output_cls_addr += num_survivors_counter * sizeof(short);
            cupvaSQDFSync(outputTrig);
            cupvaSQDFSync(outputConfTrig);
            cupvaSQDFSync(outputClsTrig);
        }

        if (num_survivors_counter >= VPU_MAX_OBJECTS)
        {
            output_box            = outputBuf;
            output_conf           = outputConfBuf;
            output_cls            = outputClsBuf;
            num_survivors_counter = 0;
        }
        else
        {
            output_box += cur_num_survivors;
            output_conf += cur_num_survivors;
            output_cls += cur_num_survivors;
        }
    }

    cupvaRasterDataFlowClose(inputRdfHdl);

    numSurvivors[0] = num_survivors_total;

    cupvaSQDFTrig(outputNumTrig);
    cupvaSQDFSync(outputNumTrig);

    cupvaSQDFClose(anchorTrig);
    cupvaSQDFClose(outputTrig);
    cupvaSQDFClose(outputConfTrig);
    cupvaSQDFClose(outputClsTrig);
    cupvaSQDFClose(outputNumTrig);
    cupvaSQDFClose(inputNumTrig);
    return 0;
}
