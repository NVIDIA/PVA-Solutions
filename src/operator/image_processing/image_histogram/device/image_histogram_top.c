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

#include "image_histogram_param.h"

#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
#    include "image_histogram_knl_u8.h"
#    define DATA_TYPE uint8_t
#    define GENERATE_MASK generate_mask_u8
#    define IMAGE_HISTOGRAM_AGEN_INIT image_histogram_agen_init_u8
#    define IMAGE_HISTOGRAM_KNL_EXEC image_histogram_knl_exec_u8
#elif defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
#    include "image_histogram_knl_u16.h"
#    define DATA_TYPE uint16_t
#    define GENERATE_MASK generate_mask_u16
#    define IMAGE_HISTOGRAM_AGEN_INIT image_histogram_agen_init_u16
#    define IMAGE_HISTOGRAM_KNL_VALID image_histogram_knl_valid_u16
#    define IMAGE_HISTOGRAM_KNL_MASK image_histogram_knl_mask_u16
#    define IMAGE_HISTOGRAM_KNL_INDEX image_histogram_knl_index_u16
#    define IMAGE_HISTOGRAM_KNL_EXEC image_histogram_knl_exec_u16
#endif

VMEM(A, DATA_TYPE, src_v, RDF_DOUBLE(DATA_TYPE, TILE_W, TILE_H));
#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
VMEM(B, int16_t, hist, HIST_U8_BUCKETS *HIST_U8_WAYS);
#else
VMEM(B, int32_t, hist, 32 * 1024);
#endif
VMEM(C, uint32_t, dst_v, RDF_SINGLE(uint32_t, NUM_BINS_MAX, 1));
#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
VMEM(C, int32_t, bucketAcc, HIST_U8_BUCKETS);
#else
VMEM(C, DATA_TYPE, valid, TILE_W *TILE_H);
#endif
VMEM(C, DATA_TYPE, mask, TILE_W);

VMEM(A, int32_t, format);
VMEM(A, int32_t, start);
VMEM(A, int32_t, end);
VMEM(A, int32_t, numBins);
VMEM(A, int32_t, scaleA);
VMEM(A, int32_t, scaleQ);

VMEM(A, int32_t, niterX);
VMEM(A, int32_t, niterY);
VMEM(A, int32_t, imageWidth);
VMEM(A, int32_t, imageHeight);

VMEM(A, ImageHistogramConfig, cfg);

VMEM_RDF_UNIFIED(A, srcHdl);
VMEM_SEQDF_HANDLER(A, dstHdl);

CUPVA_VPU_MAIN()
{
    cupvaRasterDataFlowOpen(srcHdl, &src_v[0]);
    cupvaSQDFOpen(dstHdl);

    // check whether image width is multiple of tile width
    cfg.maskWidth = imageWidth % TILE_W;
    if (cfg.maskWidth)
    {
        // generate mask buffer to handle case when image width is not multiple of tile width
        GENERATE_MASK(mask, TILE_W, cfg.maskWidth);
    }

    // calculate height of last tile row
    const int TILE_H_LAST = (imageHeight % TILE_H ? imageHeight % TILE_H : TILE_H);
#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
    IMAGE_HISTOGRAM_AGEN_INIT(src_v, TILE_W, TILE_H, TILE_H_LAST, TILE_W, start, end, numBins, hist, dst_v, bucketAcc,
                              mask, &cfg);
#elif defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
    IMAGE_HISTOGRAM_AGEN_INIT(src_v, TILE_W, TILE_H, TILE_H_LAST, TILE_W, start, end, numBins, scaleA, scaleQ, valid,
                              hist, dst_v, mask, HIST_WAYS, &cfg);
#endif
#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
    int32_t tilesSinceDrain = 0;
#endif

    // set iteration number for normal tile row
#if defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
    cfg.validNiter = cfg.niter_valid;
    if (cfg.maskWidth)
    {
        cfg.maskNiter = cfg.niter_mask;
    }
    cfg.indexNiter = cfg.niter_index;
#endif
    cfg.execNiter = cfg.niter_exec;

    for (int32_t y = 0; y < niterY; y++)
    {
        if (y >= (niterY - 1))
        {
            // set iteration number for last tile row
#if defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
            cfg.validNiter = cfg.niter_valid_last;
            if (cfg.maskWidth)
            {
                cfg.maskNiter = cfg.niter_mask_last;
            }
            cfg.indexNiter = cfg.niter_index_last;
#endif
            cfg.execNiter = cfg.niter_exec_last;
        }

        for (int32_t x = 0; x < niterX; x++)
        {
            void *inputTile = cupvaRasterDataFlowAcquire(srcHdl);

            // update base address for agens
#if defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
            cupvaModifyAgenCfgBase(&(cfg.validAgenCfgs[0]), inputTile);
            cupvaModifyAgenCfgBase(&(cfg.indexAgenCfgs[0]), inputTile);
            cupvaModifyAgenCfgBase(&(cfg.indexAgenCfgs[1]), inputTile);
#endif
            cupvaModifyAgenCfgBase(&(cfg.execAgenCfgs[0]), inputTile);

            // execute kernels
#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
            /* No valid or index pass: samples are counted at weight one and folded at the end. */
            if (cfg.maskWidth && (x >= (niterX - 1)))
            {
                image_histogram_knl_exec_masked_u8(&cfg);
            }
            else
            {
                IMAGE_HISTOGRAM_KNL_EXEC(&cfg);
            }

            if (++tilesSinceDrain >= HIST_U8_DRAIN_TILES)
            {
                image_histogram_knl_fold_u8(&cfg);
                tilesSinceDrain = 0;
            }
#elif defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
            IMAGE_HISTOGRAM_KNL_VALID(&cfg);
            if (cfg.maskWidth && (x >= (niterX - 1)))
            {
                IMAGE_HISTOGRAM_KNL_MASK(&cfg);
            }
            IMAGE_HISTOGRAM_KNL_INDEX(&cfg);
            IMAGE_HISTOGRAM_KNL_EXEC(&cfg);
#endif

            cupvaRasterDataFlowRelease(srcHdl);
        }
    }

    cupvaRasterDataFlowClose(srcHdl);

#if defined(DATA_TYPE_U8) && (DATA_TYPE_U8 == 1)
    if (tilesSinceDrain > 0)
    {
        image_histogram_knl_fold_u8(&cfg);
    }
    image_histogram_knl_emit_u8(&cfg);
#elif defined(DATA_TYPE_U16) && (DATA_TYPE_U16 == 1)
    image_histogram_knl_reduce_u16(&cfg);
#endif

    cupvaSQDFTrig(dstHdl);
    cupvaSQDFSync(dstHdl);
    cupvaSQDFClose(dstHdl);

    return 0;
}
