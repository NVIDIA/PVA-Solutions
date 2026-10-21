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

#include "orb_desc_extr_common.h"
#include "orb_desc_extr_params.h"

#include <cupva_device.h>

// input image patch buffer: ping pong
VMEM(B, DATA_TYPE, g_image_patch, TABLE_BATCH_STRIDE *BATCH_SIZE);
VMEM(C, DATA_TYPE, g_image_patch2, TABLE_BATCH_STRIDE *BATCH_SIZE);

// input feature coords buffer: ping pong
VMEM(A, float, g_feature_coords, 4 * BATCH_SIZE);

// output buffer: ping pong
VMEM(C, uint8_t, g_opt_descriptor, 2 * DESCRIPTOR_ARRAY_NUM_BYTES * BATCH_SIZE);

VMEM_GSDF_HANDLER(A, gsdf_handler, BATCH_SIZE);
VMEM_SEQDF_HANDLER(A, sqdf_handler_in);
VMEM_SEQDF_HANDLER(A, sqdf_handler_out);

VMEM(A, int32_t, last_batch_size);
VMEM(A, int32_t, has_last_batch);
VMEM(A, int32_t, n_iterations);
VMEM(A, int32_t, keypointSize);

VMEM_POINTER(A, descriptor_dram_addr);
VMEM_POINTER(A, feature_list_dram_addr);

VMEM(A, ORBDescExtractorConfig, cfg);

// constants loaded from header files and set on host
VMEM(A, WTS_TYPE, weights, WEIGHTS_ARRAY_SIZE);
VMEM(C, int8_t, orb_bit_pattern, 2 * DESCRIPTOR_PATTERN_SIZE);

// intermediate buffers
VMEM(A, WTS_PERM_TYPE, weight_perm_array, 32);
VMEM(A, uint16_t, g_rotated_orb_coords, LOOKUP_IDX_LP *BATCH_SIZE);
VMEM(A, DATA_TYPE, g_lookup_pixels, DESCRIPTOR_PATTERN_SIZE *BATCH_SIZE);
VMEM(C, int32_t, keypoint_buf_int_y, BATCH_SIZE);
VMEM(C, int32_t, keypoint_buf_int_x, BATCH_SIZE);
VMEM(A, int32_t, moment_x, BATCH_SIZE);
VMEM(A, int32_t, moment_y, BATCH_SIZE);
VMEM(A, int8_t, buf_cos_sin, 2 * BATCH_SIZE);
VMEM(A, int8_t, buf_msin_cos, 2 * BATCH_SIZE);
VMEM(B, OFFSET_TYPE, offset_vec, 2 * BATCH_SIZE);

#if defined(X8_IO)
#    include "orb_desc_extr_knl_x8.h"
#    define SERIAL_RUN serial_run_x8
#elif defined(X16_IO)
#    include "orb_desc_extr_knl_x16.h"
#    define SERIAL_RUN serial_run_x16
#endif

CUPVA_VPU_MAIN()
{
    cupvaSQDFOpen(sqdf_handler_in);
    cupvaSQDFOpen(sqdf_handler_out);

    uint64_t feature_coords_base_dram_addr = feature_list_dram_addr.base + feature_list_dram_addr.offset;
    uint64_t descriptor_out_base_dram_addr = descriptor_dram_addr.base + descriptor_dram_addr.offset;

    // init common params
    CupvaSampler sampler = {};
    init_orb_desc_extr_common_cfg(&cfg, keypointSize, has_last_batch, last_batch_size, n_iterations, keypoint_buf_int_x,
                                  keypoint_buf_int_y, moment_x, moment_y, weights, buf_cos_sin, buf_msin_cos,
                                  orb_bit_pattern, g_rotated_orb_coords, g_lookup_pixels, g_feature_coords,
                                  g_image_patch, weight_perm_array, offset_vec, &sampler);

    // trig SQDF in
    cupvaSQDFTrig(sqdf_handler_in);

    cupvaGSDFOpen(gsdf_handler, &g_image_patch[0], cfg.gsdf_tile_lp, cfg.gsdf_tile_stride);

    // sync SQDF in
    cupvaSQDFSync(sqdf_handler_in);

    // trig GSDF in
    update_gsdf_params(g_feature_coords, g_image_patch, gsdf_handler, &cfg, 0);
    cupvaGSDFTrig(gsdf_handler);

    // trig SQDF in for the next tile
    if (n_iterations > 1)
    {
        uint64_t feature_coords_buf_dram_addr_next =
            feature_coords_base_dram_addr + cfg.feature_tile_size * sizeof(float);
        float *feature_coords_buffer_next = g_feature_coords + cfg.feature_tile_size;
        cupvaSQDFUpdateAddr(sqdf_handler_in, 0, feature_coords_buf_dram_addr_next, MEMTYPE_DRAM,
                            cupvaGetVmemAddress(feature_coords_buffer_next), MEMTYPE_VMEM);

        int batch_size_next = (1 == n_iterations - 1 && cfg.has_last_batch) ? cfg.last_batch_size : cfg.gsdf_batch_size;
        cupvaSQDFUpdateTileSize(sqdf_handler_in, 0, batch_size_next * keypointSize * sizeof(float), 1);
        cupvaSQDFFlushAndTrig(sqdf_handler_in);
    }

    int32_t ping_pong = 0;

    // main loop for processing
    for (int32_t i = 0; i < n_iterations; i++)
    {

        float *feature_coords_buffer  = g_feature_coords + ping_pong * cfg.feature_tile_size;
        DATA_TYPE *image_patch_buffer = ping_pong ? g_image_patch2 : g_image_patch;
        uint8_t *descriptor_buffer    = g_opt_descriptor + ping_pong * cfg.descriptor_tile_size;
        ping_pong ^= 1;

        // sync GSDF in
        cupvaGSDFSync(gsdf_handler);

        // for the next tile, sync SQDF in & trig GSDF in
        if (i < n_iterations - 1)
        {
            cupvaSQDFSync(sqdf_handler_in);

            float *feature_coords_buffer_next  = g_feature_coords + ping_pong * cfg.feature_tile_size;
            DATA_TYPE *image_patch_buffer_next = ping_pong ? g_image_patch2 : g_image_patch;

            update_gsdf_params(feature_coords_buffer_next, image_patch_buffer_next, gsdf_handler, &cfg, i + 1);
            cupvaGSDFTrig(gsdf_handler);

            // for the next next tile, trig SQDF in
            if (i < n_iterations - 2)
            {
                uint64_t feature_coords_buf_dram_addr_next_next =
                    feature_coords_base_dram_addr + (i + 2) * cfg.feature_tile_size * sizeof(float);
                cupvaSQDFUpdateAddr(sqdf_handler_in, 0, feature_coords_buf_dram_addr_next_next, MEMTYPE_DRAM,
                                    cupvaGetVmemAddress(feature_coords_buffer), MEMTYPE_VMEM);
                int batch_size_next_next =
                    (cfg.has_last_batch && i + 2 == n_iterations - 1) ? cfg.last_batch_size : cfg.gsdf_batch_size;

                cupvaSQDFUpdateTileSize(sqdf_handler_in, 0, batch_size_next_next * keypointSize * sizeof(float), 1);
                cupvaSQDFFlushAndTrig(sqdf_handler_in);
            }
        }

        // update agen base address
#if defined(KNL_RBRIEF)
        cupvaModifyAgenCfgBase(&cfg._centroid_calc_in_cfg, image_patch_buffer + cfg._centroid_calc_start_addr_ofst);
#endif

        // update 2D lookup sampler addrs
        update_2d_lookup_sampler_addrs(&cfg, (void *)image_patch_buffer);

        // computations
        SERIAL_RUN(&cfg, descriptor_buffer);

        // sync SQDF out
        if (i > 0)
        {
            cupvaSQDFSync(sqdf_handler_out);
        }
        uint64_t descriptor_out_buf_dram_addr =
            descriptor_out_base_dram_addr + i * cfg.descriptor_tile_size * sizeof(uint8_t);
        cupvaSQDFUpdateAddr(sqdf_handler_out, 0, cupvaGetVmemAddress(descriptor_buffer), MEMTYPE_VMEM,
                            descriptor_out_buf_dram_addr, MEMTYPE_DRAM);

        int batch_size = (i == n_iterations - 1 && cfg.has_last_batch) ? cfg.last_batch_size : cfg.gsdf_batch_size;
        cupvaSQDFUpdateTileSize(sqdf_handler_out, 0, batch_size * DESCRIPTOR_ARRAY_NUM_BYTES, 1);
        cupvaSQDFFlushAndTrig(sqdf_handler_out);
    }

    cupvaGSDFClose(gsdf_handler);
    cupvaSQDFClose(sqdf_handler_in);
    cupvaSQDFSync(sqdf_handler_out);
    cupvaSQDFClose(sqdf_handler_out);

    return 0;
}