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

/**
 * @file radar_cfar_param.h
 *
 * @brief Shared host/device constants and the host-to-device parameter block for RadarCFAR.
 */

#ifndef RADAR_CFAR_PARAM_H
#define RADAR_CFAR_PARAM_H

#include <stdint.h>

/** Maximum supported input width and height (elements). */
#define CFAR_MAX_DIM (1024)

/** Maximum supported number of input elements (width * height). */
#define CFAR_MAX_PIXELS (1024 * 512)

/**
 * Maximum numTrain + numGuard per direction. Validated at Create, and therefore the bound
 * on every halo in this operator.
 */
#define CFAR_MAX_WINDOW (255)

/** Maximum detection-list capacity accepted at construction. */
#define CFAR_MAX_DETECTIONS (8192)

/** Q-format of the threshold factors, matching CFAR_QBITS in the CPU reference. */
#define CFAR_QBITS (14)
#define CFAR_FIXED_POINT_SCALE (1 << CFAR_QBITS)

/** Q-format of the count reciprocals, matching the reference's 65536 / count. */
#define CFAR_RECIP_QBITS (16)
#define CFAR_RECIP_ONE (1 << CFAR_RECIP_QBITS)

/**
 * Number of lanes processed in parallel by both marching passes. One dvintx holds
 * 16 x 32-bit elements (extended to 48-bit in the register file).
 */
#define CFAR_LANES (16)

/**
 * Detections staged in VMEM before being drained to DRAM.
 *
 * The staging buffer is drained whenever it is within one strip's worst-case yield of
 * full, so it must exceed CFAR_MAX_DIM. Kept small deliberately: the drain SQDF's tile is
 * compiled at this height, and a tall compiled tile is a large descriptor for the DMA
 * engine to set up. peak_detection uses the same shape with a 100-entry hint.
 */
#define CFAR_STAGE_CAPACITY (1280)

/** Padded extent of a marching strip: the marched dimension plus both halos. */
#define CFAR_MAX_PADDED (CFAR_MAX_DIM + 2 * CFAR_MAX_WINDOW)

/**
 * Number of per-position tables staged in DRAM and DMA'd into VMEM once at startup.
 *
 * Order: horizontal total/trailing/leading reciprocals, vertical total/trailing/leading
 * reciprocals, then the per-row horizontal threshold factors. They live in one DRAM
 * allocation rather than the parameter block because the block is capped at 8 KB while
 * the tables need 7 * 1024 * 4 = 28 KB at the widest supported image.
 */
#define CFAR_NUM_TABLES (7)

/** Stride, in elements, between consecutive tables in the DRAM staging buffer. */
#define CFAR_TABLE_STRIDE (CFAR_MAX_DIM)

/** Index of each table within the staging buffer. */
#define CFAR_TBL_HOR_TOTAL (0)
#define CFAR_TBL_HOR_TRAIL (1)
#define CFAR_TBL_HOR_LEAD (2)
#define CFAR_TBL_VER_TOTAL (3)
#define CFAR_TBL_VER_TRAIL (4)
#define CFAR_TBL_VER_LEAD (5)
#define CFAR_TBL_HOR_FACTOR (6)

/**
 * @brief Host-to-device parameter block.
 *
 * Sizes and per-position tables are resolved on the host so the device inner loops
 * stay free of control flow. The reciprocal tables hold @c 65536/count for every
 * position along the marched axis; in cyclic mode every entry is identical, in
 * "full" (zero-pad) mode they vary because the divisor shrinks near the borders.
 */
typedef struct
{
    int32_t width;  /**< input width in elements */
    int32_t height; /**< input height in elements */

    int32_t numHorTrain; /**< horizontal training cells per side */
    int32_t numHorGuard; /**< horizontal guard cells per side */
    int32_t numVerTrain; /**< vertical training cells per side */
    int32_t numVerGuard; /**< vertical guard cells per side */
    int32_t horWindow;   /**< numHorTrain + numHorGuard */
    int32_t verWindow;   /**< numVerTrain + numVerGuard */

    int32_t horCyclic;    /**< 1 = wrap horizontally, 0 = zero pad */
    int32_t verCyclic;    /**< 1 = wrap vertically, 0 = zero pad */
    int32_t peakGrouping; /**< 1 = apply the immediate-neighbour NMS */

    int32_t numHorStrips; /**< number of 16-row strips in the horizontal pass */
    int32_t numVerStrips; /**< number of 16-column strips in the vertical pass */

    int32_t maxDetections; /**< detection-list capacity */

    int32_t verFactorQ14; /**< vertical threshold factor in Q14 */
} CfarHost2DevParam;

#endif /* RADAR_CFAR_PARAM_H */
