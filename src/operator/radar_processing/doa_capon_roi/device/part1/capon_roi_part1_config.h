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

#ifndef CAPON_ROI_PART1_CONFIG_H
#define CAPON_ROI_PART1_CONFIG_H

#include "capon_roi_part1_config_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize covariance, Cholesky, and L-inverse parameters for Part 1. */
void init_capon_roi_part1_config(Para_capon_roi_part1_config *cfg, float *cov_real_buf, float *cov_imag_buf,
                                 float adp_diag_const, int n_samples_per_tile, float inv_n_samples,
#if !CAPON_ROI_USE_FP_COVMAT
                                 int n_samples_bits,
#endif
                                 int input_qbits, int last_tile_width);

void update_snapshot_agen_addr(int32_t pingPong, int32_t *snapshotPingBuffer, int32_t *snapshotPongBuffer,
                               int32_t isLastDopplerTile, int32_t roiTileIdx, int32_t dopplerTileIdx,
                               int32_t numTilesROIdv, int32_t numTilesDoppler, int32_t snapTileHeight,
                               int32_t snapTileWidth, int32_t snapImageLp, int32_t lastTileRoiHeight,
                               int32_t lastTileWidthDoppler, int32_t hasLastTileRoi, int32_t hasLastTileDoppler,
                               uint64_t snapshotsBase, SequenceDataFlowHandler &snapshotSQDFHdl,
                               Para_capon_roi_part1_config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* CAPON_ROI_PART1_CONFIG_H */
