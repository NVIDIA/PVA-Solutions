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

#include "capon_roi_part1_config.h"

#include "cov_mat/cov_mat_common.h"

#if defined(CUPVA_PVA_GEN_NUMBER) && CUPVA_PVA_GEN_NUMBER >= 3
VMEM_NOEXPORT(D, int32_t, capon_roi_tri_col_bias, BUF_MAX_SV_DIM) = {
#else
VMEM_NOEXPORT(A, int32_t, capon_roi_tri_col_bias, BUF_MAX_SV_DIM) = {
#endif
    0, 15, 29, 42, 54, 65, 75, 84, 92, 99, 105, 110, 114, 117, 119, 120};

void init_capon_roi_part1_config(Para_capon_roi_part1_config *cfg, float *cov_real_buf, float *cov_imag_buf,
                                 float adp_diag_const, int n_samples_per_tile, float inv_n_samples,
#if !CAPON_ROI_USE_FP_COVMAT
                                 int n_samples_bits,
#endif
                                 int input_qbits, int last_tile_doppler_bins)
{
    cfg->adp_diag_const = adp_diag_const;
    cfg->L_real_buf_ptr = cov_real_buf;
    cfg->L_imag_buf_ptr = cov_imag_buf;

    compute_covariance_matrix_knl_nhw_init(cfg, n_samples_per_tile, inv_n_samples,
#if !CAPON_ROI_USE_FP_COVMAT
                                           n_samples_bits,
#endif
                                           input_qbits, last_tile_doppler_bins);
}

void update_snapshot_agen_addr(int32_t pingPong, int32_t *snapshotPingBuffer, int32_t *snapshotPongBuffer,
                               int32_t isLastDopplerTile, int32_t roiTileIdx, int32_t dopplerTileIdx,
                               int32_t numTilesROIdv, int32_t numTilesDoppler, int32_t snapTileHeight,
                               int32_t snapTileWidth, int32_t snapImageLp, int32_t lastTileRoiHeight,
                               int32_t lastTileWidthDoppler, int32_t hasLastTileRoi, int32_t hasLastTileDoppler,
                               uint64_t snapshotsBase, SequenceDataFlowHandler &snapshotSQDFHdl,
                               Para_capon_roi_part1_config *cfg)
{
    // update agen address
    int32_t *src          = pingPong == 0 ? snapshotPingBuffer : snapshotPongBuffer;
    AgenCFG &snapshot_cfg = isLastDopplerTile ? cfg->agen_cfg_cov_x_last_tile : cfg->agen_cfg_cov_x;
    cupvaModifyAgenCfgBase(&snapshot_cfg, src);

    if (roiTileIdx == numTilesROIdv - 1 && dopplerTileIdx == numTilesDoppler - 1)
    {
        return;
    }

    // update SQDF address for the next tile
    int32_t nextRoiTileIdx     = roiTileIdx;
    int32_t nextDopplerTileIdx = dopplerTileIdx + 1;
    if (nextDopplerTileIdx == numTilesDoppler)
    {
        ++nextRoiTileIdx;
        nextDopplerTileIdx = 0;
    }

    int32_t nextSrcOffset =
        (nextRoiTileIdx * snapTileHeight * snapImageLp + nextDopplerTileIdx * snapTileWidth) * sizeof(int32_t);

    int32_t *nextSrc = pingPong == 0 ? snapshotPongBuffer : snapshotPingBuffer;
    cupvaSQDFUpdateAddr(snapshotSQDFHdl, 0, snapshotsBase + nextSrcOffset, MEMTYPE_DRAM, cupvaGetVmemAddress(nextSrc),
                        MEMTYPE_VMEM);

    int32_t lastDopplerTileIdx = numTilesDoppler - 1;
    int32_t lastRoiTileIdx     = numTilesROIdv - 1;
    int32_t updateTileWidth =
        hasLastTileDoppler && (dopplerTileIdx == lastDopplerTileIdx || nextDopplerTileIdx == lastDopplerTileIdx);
    int32_t updateTileHeight = hasLastTileRoi && roiTileIdx != nextRoiTileIdx && nextRoiTileIdx == lastRoiTileIdx;

    if (updateTileWidth || updateTileHeight)
    {
        int32_t nextTileWidth =
            hasLastTileDoppler && nextDopplerTileIdx == lastDopplerTileIdx ? lastTileWidthDoppler : snapTileWidth;
        int32_t nextTileHeight =
            hasLastTileRoi && nextRoiTileIdx == lastRoiTileIdx ? lastTileRoiHeight : snapTileHeight;
        cupvaSQDFUpdateTileSize(snapshotSQDFHdl, 0, nextTileWidth * sizeof(int32_t), nextTileHeight);
    }

    cupvaSQDFFlushAndTrig(snapshotSQDFHdl);
}
