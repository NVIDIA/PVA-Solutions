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
 * @file histogram_equalization_param.h
 *
 * @brief Geometry contract shared by the host program and the VPU kernels.
 *        Data contract only: tile/capacity constants and static_asserts.
 *        Contains no function that computes an output value.
 */

#ifndef HISTOGRAM_EQUALIZATION_PARAM_H
#define HISTOGRAM_EQUALIZATION_PARAM_H

/** Number of intensity bins. Fixed at 256 by the U8 input domain. */
#define HEQ_NBINS (256)

/**
 * Parallelism of the histogram and lookup units, in lanes.
 *
 * 32 is the machine maximum for both: the halfword parallel histogram reaches
 * "up to 32 histogram updates per cycle, in case of 32-way half-word case"
 * (Programmer's Guide, instruction-set-architecture), and 32H/32HU is
 * the widest parallel table lookup. Both are M0-slot-only
 * instructions, so 32 lanes per issue is the floor for each pass.
 */
#define HEQ_WAYS (32)

/*
 * Tile width is a multiple of 2 * HEQ_WAYS: the device loops consume that many pixels per issue,
 * and a tile width that is only a single-group multiple leaves half a double vector unconsumed at
 * the end of a row, which makes the histogram undercount.
 */
#define HEQ_TILE_WIDTH_GRAIN (2 * HEQ_WAYS)

/**
 * Bytes per replicated 64-byte line of a 32-way halfword object.
 *
 * For an M-byte entry type at N-way parallelism the hardware packs
 * K = 64/(M*N) entries of each sub-table into every 64-byte line. Halfword entries at 32-way give
 * K = 1, so bin b of way i sits at byte 2*i + 64*b: exactly one 64-byte line per
 * bin, holding that bin's 32 ways. Both the histogram scratch and the lookup
 * table use this layout.
 */
#define HEQ_LINE_BYTES (64)

/** Halfword elements spanned by one bin of a 32-way halfword object. */
#define HEQ_LINE_ELEMS (HEQ_LINE_BYTES / 2)

/** Halfword elements in the 32-way histogram scratch / lookup table. */
#define HEQ_TABLE_ELEMS (HEQ_NBINS * HEQ_LINE_ELEMS)

/**
 * Drain threshold, in histogram instruction issues.
 *
 * Each vhist_simple_32h adds 1 to exactly one bin per lane, so after C issues no bin of any way
 * can exceed C. The bins are signed halfwords, so the ways must be folded into 32-bit accumulators
 * and cleared before C reaches 32767. The threshold is a whole tile's worth of issues below that
 * bound, so a tile is never split mid-way.
 */
#define HEQ_DRAIN_ISSUES (30000)

/** Fixed-point fraction bits of the LUT scale factor. Matches the reference. */
#define HEQ_QBITS (15)

/** Numerator of the reference scale factor: 255 * (1 << 15). */
#define HEQ_SCALE_NUM (255 * (1 << HEQ_QBITS))

/**
 * Maximum tile width in pixels, and the VMEM byte budget of one tile.
 *
 * The host picks the actual tw/th at Create time inside these bounds; the
 * constants exist only to size the VMEM buffers. TW_MAX is a multiple of
 * HEQ_WAYS so a tile row is a whole number of vector groups, and is set to the
 * full-HD width so that a 1080p row fits one tile.
 *
 * The byte budget is set by bank B, not bank A: bank A holds one double-buffered
 * input tile (2 * 28800 = 57600 B, shared by both read passes), while bank B
 * holds the double-buffered output tile plus the 16 KB histogram and 3 KB of scan
 * buffers (2 * 28800 + 19456 = 77056 B). Both fit one 128 KB superbank, and B is
 * the tighter of the two -- that is the constraint to check when resizing.
 */
#define HEQ_TW_MAX (1920)
#define HEQ_TILE_BYTES (28800)
#define HEQ_TH_MAX (HEQ_TILE_BYTES / HEQ_TW_MAX)

/** Shortest tile row that still gives the DMA a useful contiguous run. */
#define HEQ_TW_MIN (256)

/** Largest right-hand padding a RasterDataFlow accepts on the last tile column, in pixels. */
#define HEQ_RDF_MAX_PAD (255)

/** Minimum supported image dimensions, per the public header's Limitations. */
#define HEQ_MIN_WIDTH (129)
#define HEQ_MIN_HEIGHT (17)

/** Hillis-Steele doubling passes over HEQ_NBINS bins: log2(HEQ_NBINS). */
#define HEQ_SCAN_PASSES (8)

/**
 * Zero pad, in 32-bit words, in front of the cumulative-count arrays.
 *
 * The inclusive scan is a Hillis-Steele doubling scan whose shifted operand is
 * read at an address offset (src[i - k]) rather than through a lane permute.
 * Padding the front with HEQ_NBINS/2 zero words makes every i - k read land on
 * real, zero-valued memory for all k up to the final HEQ_NBINS/2 step, so the
 * scan needs no boundary predication.
 */
#define HEQ_SCAN_PAD (HEQ_NBINS / 2)

/** Elements of one scan buffer: the zero pad plus the bins. */
#define HEQ_SCAN_ELEMS (HEQ_SCAN_PAD + HEQ_NBINS)

/*
 * chess_loop_range upper bounds.
 *
 * The hints in the kernels have to spell their bounds as literal digits, so changing a tile
 * constant can silently invalidate one. Each literal is pinned here against the geometry it came
 * from, and a mismatch fails the build:
 *
 *   900  issues per tile          = HEQ_TILE_BYTES / HEQ_WAYS   (both hot loops)
 *   256  bins                     = HEQ_NBINS                   (per-bin loops)
 *    16  bin sweeps at 16/issue   = HEQ_NBINS / 16              (delta, LUT, scan)
 *    60  mask groups per row      = HEQ_TW_MAX / 32             (mask build)
 *   256  64-byte lines per table  = HEQ_TABLE_ELEMS * 2 / 64    (table clear)
 */
#ifdef __cplusplus
#    include <cstdint>
/* Negative-array-size typedefs, not static_assert: not every supported device compiler has it. */
typedef char heq_param_contract_1[(HEQ_TILE_BYTES / HEQ_WAYS == 900)
                                      ? 1
                                      : -1]; /* chess_loop_range(8, 900) in the hot loops is stale */
typedef char
    heq_param_contract_2[(HEQ_NBINS == 256) ? 1 : -1]; /* chess_loop_range(256, 256) in the per-bin loops is stale */
typedef char heq_param_contract_3[(HEQ_NBINS / 16 == 16)
                                      ? 1
                                      : -1]; /* chess_loop_range(16, 16) in the bin-sweep loops is stale */
typedef char
    heq_param_contract_4[(HEQ_TW_MAX / 32 == 60) ? 1 : -1]; /* chess_loop_range(1, 60) in the mask build is stale */
typedef char heq_param_contract_5[(HEQ_TABLE_ELEMS * 2 / 64 == 256)
                                      ? 1
                                      : -1]; /* chess_loop_range(1, 256) in heq_bzero is stale */
typedef char heq_param_contract_6[(HEQ_TW_MAX % HEQ_WAYS == 0)
                                      ? 1
                                      : -1]; /* tile width must be a whole number of 32-lane groups */
typedef char heq_param_contract_7[(HEQ_TILE_BYTES % HEQ_TW_MAX == 0)
                                      ? 1
                                      : -1]; /* tile byte budget must divide evenly by the max tile width */
typedef char heq_param_contract_8[(HEQ_TABLE_ELEMS * 2 == 16384)
                                      ? 1
                                      : -1]; /* a 32-way halfword table for 256 bins occupies 16 KB */
typedef char heq_param_contract_9
    [(HEQ_DRAIN_ISSUES + (HEQ_TILE_BYTES / HEQ_WAYS) <= 32767)
         ? 1
         : -1]; /* one tile of histogram issues past the drain threshold must still fit a signed 16-bit bin */
/*
 * The tile search always finds a candidate, so it needs no fallback and no error path: twMax is
 * clamped to at least HEQ_TW_MIN, and a HEQ_TW_MIN-wide tile always qualifies because its padding
 * is at most HEQ_TW_MIN - 1, it holds at least one row, and the grain divides it.
 */
typedef char heq_param_contract_11[(HEQ_TW_MIN - 1 <= HEQ_RDF_MAX_PAD)
                                       ? 1
                                       : -1]; /* a minimum-width tile could exceed the RDF padding cap */
typedef char heq_param_contract_12[(HEQ_TW_MIN % HEQ_TILE_WIDTH_GRAIN == 0)
                                       ? 1
                                       : -1]; /* the tile-width scan would step past HEQ_TW_MIN */
typedef char heq_param_contract_13[(HEQ_TILE_BYTES / HEQ_TW_MIN >= 1)
                                       ? 1
                                       : -1]; /* a minimum-width tile must hold at least one row */

typedef char heq_param_contract_10[(HEQ_SCALE_NUM == 8355840)
                                       ? 1
                                       : -1]; /* scale numerator must match the CPU reference's 255 * (1 << 15) */
typedef char heq_param_contract_11[(HEQ_SCAN_PAD >= HEQ_NBINS / 2)
                                       ? 1
                                       : -1]; /* scan pad must cover the largest Hillis-Steele shift */
#endif

#endif // HISTOGRAM_EQUALIZATION_PARAM_H
