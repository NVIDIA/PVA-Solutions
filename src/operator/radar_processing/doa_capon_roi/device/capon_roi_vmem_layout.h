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

/**
 * @file capon_roi_vmem_layout.h
 * @brief VMEM buffer sizes for Capon DoA-on-ROI.
 */

#ifndef PVA_CAPON_ROI_VMEM_LAYOUT_H
#define PVA_CAPON_ROI_VMEM_LAYOUT_H

/* Per-task-instance maxima. */
#define BUF_MAX_SV_DIM 16
#define NUM_AZ_BINS 16
#define NUM_EL_BINS 8
#define BUF_MAX_SV_NUM (NUM_AZ_BINS * NUM_EL_BINS)
#define BUF_MAX_NUM_SAMPLES 32
#define BUF_MAX_ROI_NUM 16

#define BUF_MAX_TRI_LEN (BUF_MAX_SV_DIM * (BUF_MAX_SV_DIM + 1) / 2)      /* 136 */
#define BUF_MAX_IMAG_TRI_LEN (BUF_MAX_SV_DIM * (BUF_MAX_SV_DIM - 1) / 2) /* 120 */

#define SV_ROI_NUM 8
#define DV_ROI_NUM 16
#define NUM_PEAKS_PER_ROI 3

/* part1 buffers */
/* The transpose store requires a 16k+1 line pitch. */
#define BUF_LINE_PITCH_SNAPSHOT ((((BUF_MAX_NUM_SAMPLES * BUF_MAX_SV_DIM * 2) - 1 + 15) / 16) * 16 + 1) /* 1025 */
#define BUF_SZ_SNAPSHOT BUF_LINE_PITCH_SNAPSHOT *DV_ROI_NUM
#define BUF_SZ_COV_REAL BUF_MAX_TRI_LEN *DV_ROI_NUM
#define BUF_SZ_COV_IMAG BUF_MAX_IMAG_TRI_LEN *DV_ROI_NUM
#define BUF_SZ_LINV BUF_SZ_COV_REAL /* 2176 */

/* part2 buffers */
#define BUF_SV_SINGLE_TILE_PIX (BUF_MAX_SV_DIM * BUF_MAX_SV_NUM * SV_ROI_NUM)

// phase 4: obtain_fixed_pt_L_inv in double vectors
#define BUF_LINE_PITCH_LINV_FIXED ((3 * BUF_MAX_TRI_LEN - 1 + 15) / 16 * 16 + 1) /* 417 */
#define BUF_SZ_LINV_FIXED DV_ROI_NUM *BUF_LINE_PITCH_LINV_FIXED
#define BUF_SZ_SCALE DV_ROI_NUM
#define BUF_SZ_LINV_OFST ((BUF_MAX_TRI_LEN + 15) / 16 * 16)

#define BUF_LINE_PITCH_RCP ((BUF_MAX_SV_NUM - 1 + 15) / 16 * 16 + 1) /* 129 */
#define BUF_SZ_RCP_SPECTRUM (DV_ROI_NUM * BUF_LINE_PITCH_RCP)
#define BUF_SZ_FIXED_SPEC ((DV_ROI_NUM / 16) * BUF_MAX_SV_NUM * 16) /* 2048 */
#define BUF_SZ_FLOAT_SPEC ((DV_ROI_NUM / 16) * BUF_MAX_SV_NUM * 16) /* 2048 */
#define BUF_SZ_SPEC_MIN DV_ROI_NUM
#define BUF_SZ_NOISE_HIST 2048  /* 2^5 histogram bins * sizeof(dvshort) */
#define BUF_SZ_NOISE_CUMUL 1056 /* 33 * sizeof(vshort) */
#define BUF_SZ_NOISE_FLOOR DV_ROI_NUM
#define BUF_SZ_LOCAL_MAX (16 * 6 * DV_ROI_NUM)                                /* 1536 */
#define BUF_SZ_LOCAL_SUM BUF_SZ_LOCAL_MAX                                     /* 1536 */
#define BUF_SZ_CANDIDATE ((NUM_AZ_BINS - 2) * (NUM_EL_BINS - 2) * DV_ROI_NUM) /* 1344 */
#define PEAK_POWER_LINE_PITCH ((NUM_PEAKS_PER_ROI - 1 + 15) / 16 * 16 + 1)    /* 17, 16k+1 for T1 transpose store */
#define BUF_SZ_OUTPUT (DV_ROI_NUM * PEAK_POWER_LINE_PITCH)

/* Part-2 aliases within the steering pong tiles. Offsets are in bytes. */
#define BUF_SV_SINGLE_TILE_BYTES (BUF_SV_SINGLE_TILE_PIX * sizeof(int32_t))

#define BUF_PART2_SV0_FIXED_SPEC_OFST 0
#define BUF_PART2_SV0_FLOAT_SPEC_OFST (BUF_PART2_SV0_FIXED_SPEC_OFST + BUF_SZ_FIXED_SPEC * sizeof(uint16_t))
#define BUF_PART2_SV0_NOISE_FLOOR_OFST (BUF_PART2_SV0_FLOAT_SPEC_OFST + BUF_SZ_FLOAT_SPEC * sizeof(float))

#define BUF_PART2_SV1_NOISE_CUMUL_OFST 0
#define BUF_PART2_SV1_LOCAL_SUM_OFST (BUF_PART2_SV1_NOISE_CUMUL_OFST + BUF_SZ_NOISE_CUMUL * sizeof(uint8_t))

#endif /* PVA_CAPON_ROI_VMEM_LAYOUT_H */
