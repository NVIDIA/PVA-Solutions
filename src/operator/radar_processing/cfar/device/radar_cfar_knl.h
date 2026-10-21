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
 * @file radar_cfar_knl.h
 *
 * @brief Vector kernels for the RadarCFAR operator.
 *
 * The operator is separable: a horizontal pass marches along x with the vector lanes
 * holding CFAR_LANES image rows, and a vertical pass marches along y with the lanes
 * holding CFAR_LANES image columns. Each pass maintains the trailing and leading
 * training-cell sums as running sums, so the cost per cell is independent of numTrain.
 *
 * Running sums live only in vector registers, where a word lane is extended to 48 bits
 * (Programmer's Guide, Vector Register File). With numTrain + numGuard <= 255 a window sum
 * is bounded by
 * 510 * (2^32 - 1) < 2^41, so no multi-word accumulator is needed.
 */

#ifndef RADAR_CFAR_KNL_H
#define RADAR_CFAR_KNL_H

#include "radar_cfar_param.h"

#include <cupva_device.h>

/**
 * Line pitch, in elements, of the private padded row strip used by the horizontal pass.
 *
 * Transposed vector access requires a pitch of 64k + 4 bytes for word-sized elements
 * i.e. 16k + 1 elements, which yields the integral lane
 * offset (pitchBytes - 4) / 64 = k.
 *
 * The pitch must hold CFAR_MAX_DIM + 2 * CFAR_MAX_WINDOW elements, plus the one-element
 * read-ahead slop consumed by the leading-window entry head, plus up to CFAR_LANES - 1
 * elements of tail overrun from the vectorised halo fill (the halo width is not a
 * multiple of the vector width, so its final store spills forward). That is
 * 1024 + 510 + 1 + 15 = 1550; 1553 = 16 * 97 + 1 is the smallest valid pitch above it.
 */
#define CFAR_H_PAD_LP (1553)

/** Lane offset, in 64-byte units, matching CFAR_H_PAD_LP. */
#define CFAR_H_PAD_LANE_OFST ((CFAR_H_PAD_LP - 1) / 16)

/*
 * CFAR_H_PAD_LP must be 16k+1 elements, or the lane offset truncates silently and every
 * lane but the first reads the wrong column. Negative-array-size typedef rather than
 * static_assert, which is not available on every supported device compiler.
 */
typedef char cfar_h_pad_lp_is_16k_plus_1[(16 * CFAR_H_PAD_LANE_OFST + 1 == CFAR_H_PAD_LP) ? 1 : -1];
/* CFAR_H_PAD_LP must hold the padded row, the read-ahead slop and the halo-fill overrun. */
typedef char cfar_h_pad_lp_holds_row_slop_and_overrun
    [(CFAR_H_PAD_LP >= CFAR_MAX_DIM + 2 * CFAR_MAX_WINDOW + 1 + (CFAR_LANES - 1)) ? 1 : -1];

/**
 * Line pitch, in elements, of the horizontal mask tile.
 *
 * The mask is written with transposed stores and drained by a transposing
 * RasterDataFlow, so its pitch must satisfy the same 64k + 4 byte rule. This is
 * TRANSPOSE_LINE_PITCH_ALIGNED(CFAR_MAX_DIM * 4, TRANS_MODE_1, 4) / 4 evaluated for the
 * widest supported image; the kernel still reads the actual pitch back from the dataflow
 * so a narrower image uses the pitch the RDF chose.
 */
#define CFAR_MASK_LP (1025)

/** Elements in the shared padded/tile buffer on superbank B. */
#define CFAR_PAD_BUF_ELEMS (CFAR_LANES * CFAR_H_PAD_LP)

/** Elements in the shared mask buffer on superbank C. */
#define CFAR_MASK_BUF_ELEMS (CFAR_LANES * CFAR_MASK_LP)

/** Extra slop on the detection staging arrays: one full vector may be stored past the count. */
#define CFAR_STAGE_SLOP (CFAR_LANES)

/**
 * @brief Write the lane index vector {0, 1, ... CFAR_LANES-1}.
 *
 * Used by the vertical pass to turn a lane position into an image x.
 *
 * @param [out] laneIdx CFAR_LANES entries
 */
void cfarInitLaneIdx(int32_t *restrict laneIdx);

/**
 * @brief Fill the private padded row strip for one horizontal strip.
 *
 * Copies @p rows rows of @p width elements from the DMA tile into the centre of
 * @p pad and materialises both x halos: wrapped from the opposite edge of the same row
 * when @p cyclic is non-zero, zero otherwise (matching createPaddedDataFixedPoint in
 * the CPU reference).
 *
 * @param [in]  tile     DMA tile, @p rows rows of pitch @p tileLp elements
 * @param [in]  tileLp   line pitch of @p tile in elements
 * @param [out] pad      private padded strip, @p rows rows of pitch CFAR_H_PAD_LP
 * @param [in]  width    image width in elements
 * @param [in]  window   halo width in elements (numTrain + numGuard), < @p width
 * @param [in]  rows     number of valid rows in the strip, in [1, CFAR_LANES]
 * @param [in]  cyclic   non-zero for cyclic padding, zero for zero padding
 */
void cfarPadRowStrip(uint32_t const *restrict tile, int32_t tileLp, uint32_t *restrict pad, int32_t width,
                     int32_t window, int32_t rows, int32_t cyclic);

/**
 * @brief Fill the private padded column strip for one vertical strip.
 *
 * Mirror of cfarPadRowStrip for the other axis: copies @p height rows of CFAR_LANES
 * elements from the DMA tile into the centre of @p pad and materialises both y halos,
 * wrapped from the opposite edge when @p cyclic is non-zero and zero otherwise.
 *
 * @param [in]  tile    DMA tile, @p height rows of pitch @p tileLp elements
 * @param [in]  tileLp  line pitch of @p tile in elements
 * @param [out] pad     private padded strip, rows of pitch CFAR_LANES, halo first
 * @param [in]  height  image height in elements
 * @param [in]  window  halo height in elements (numTrain + numGuard), < @p height
 * @param [in]  cyclic  non-zero for cyclic padding, zero for zero padding
 */
void cfarPadColStrip(uint32_t const *restrict tile, int32_t tileLp, uint32_t *restrict pad, int32_t height,
                     int32_t window, int32_t cyclic);

/**
 * @brief Horizontal CFAR pass over one row strip.
 *
 * Marches x from 0 to width-1 with the lanes holding CFAR_LANES consecutive image rows,
 * and writes the per-cell horizontal decision (0 or -1) transposed into @p mask so the
 * output RasterDataFlow lands it in natural row-major order in DRAM.
 *
 * @param [in]  pad          padded row strip produced by cfarPadRowStrip
 * @param [out] mask         transposed-access mask tile
 * @param [in]  maskLp       line pitch of @p mask in elements
 * @param [in]  maskLaneOfst lane offset for @p mask, from cupvaTransposeLaneOfst
 * @param [in]  recipTotal   per-x reciprocal 65536/totalCount    (CA only)
 * @param [in]  recipTrail   per-x reciprocal 65536/trailingCount (GO and SO only)
 * @param [in]  recipLead    per-x reciprocal 65536/leadingCount  (GO and SO only)
 * @param [in]  factorQ14    per-row threshold factors for this strip, CFAR_LANES entries
 * @param [in]  p            host parameters
 */
void cfarHorizontalStrip(uint32_t const *restrict pad, int32_t *restrict mask, int32_t maskLp, int32_t maskLaneOfst,
                         int32_t const *restrict recipTotal, int32_t const *restrict recipTrail,
                         int32_t const *restrict recipLead, int32_t const *restrict factorQ14,
                         CfarHost2DevParam const *restrict p);

/**
 * @brief Vertical CFAR pass over one column strip, with detection emission.
 *
 * Marches y from y0 to y0 + rows - 1 with the lanes holding CFAR_LANES consecutive image
 * columns, combines the vertical decision with the stored horizontal decision, and
 * compacts the surviving cells into @p stageY / @p stageX.
 *
 * @param [in]  pad         padded column strip, rows of pitch CFAR_LANES, halo first
 * @param [in]  mask        horizontal mask tile, rows of pitch @p maskLp
 * @param [in]  maskLp      line pitch of @p mask in elements
 * @param [in]  laneIdx     vector of lane indices 0..CFAR_LANES-1
 * @param [in]  recipTotal  per-y reciprocal 65536/totalCount    (CA only)
 * @param [in]  recipTrail  per-y reciprocal 65536/trailingCount (GO and SO only)
 * @param [in]  recipLead   per-y reciprocal 65536/leadingCount  (GO and SO only)
 * @param [in]  x0          image x of lane 0
 * @param [in]  y0          first image row to march
 * @param [in]  rows        number of rows to march, a multiple of 8
 * @param [out] stageY      vertical indices of surviving cells
 * @param [out] stageX      horizontal indices of surviving cells
 * @param [in]  room        writable entries remaining in the staging arrays
 * @param [in]  p           host parameters
 *
 * @return number of detections written, never more than @p room
 */
int32_t cfarVerticalStrip(uint32_t const *restrict pad, int32_t const *restrict mask, int32_t maskLp,
                          int32_t const *restrict laneIdx, int32_t const *restrict recipTotal,
                          int32_t const *restrict recipTrail, int32_t const *restrict recipLead, int32_t x0, int32_t y0,
                          int32_t rows, int32_t *restrict stageY, int32_t *restrict stageX, int32_t room,
                          CfarHost2DevParam const *restrict p);

#endif /* RADAR_CFAR_KNL_H */
