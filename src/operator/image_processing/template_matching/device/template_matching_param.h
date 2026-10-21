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

#ifndef TEMPLATE_MATCHING_PARAM_H
#define TEMPLATE_MATCHING_PARAM_H

#include <stdint.h>

// Output tile. Width is a whole number of 32-column chunks; height is even
// because one correlation pass produces two output rows.
//
// Widest tile. Sizes the VMEM buffers and scratch rows; the tile actually used
// is chosen per request by the host and arrives as params.nChunk.
#define TM_TILE_WIDTH 256
#define TM_TILE_HEIGHT 32

#define TM_MAX_TDIM 31
// Operator-wide minimum image dimension. Published, and what makes the host's
// tile search total: it pins the smallest output to the narrowest tile.
#define TM_MIN_IMAGE_DIM 94
#define TM_MAX_HALO (TM_MAX_TDIM - 1)

// Correlation loop bounds for the widest template.
#define TM_MAX_IB ((TM_MAX_TDIM / 2) + 1)
#define TM_MAX_JB ((TM_MAX_TDIM + 3) / 4)

// Worst-case samples a 24-bit VFilt4x2x2BBH lane holds, rounded down to a power
// of two. Larger templates split into groups; see the host's group search.
#define TM_ACC_MAX_SAMPLES 128

// 16 bytes per (ib, jb) pair: rows (2i, 2i+1) for coefficient set 0 and rows
// (2i-1, 2i) for set 1, four bytes each. The tail pad covers the 64-byte
// window the replicating load reads from the last block.
#define TM_COEF_BLOCK_BYTES 16
#define TM_COEF_BYTES ((TM_MAX_IB * TM_MAX_JB * TM_COEF_BLOCK_BYTES) + 64)

// Template staging buffer the device packs the coefficients from. The pitch is
// one vector and covers the widest template. Rows 0..2*IB-1 are every row the
// packing reads, at most 31 since IB <= 16; the last row is a permanent zero
// row standing in for template row -1.
#define TM_STAGE_PITCH 32
#define TM_STAGE_ROWS (TM_MAX_TDIM + 2)
#define TM_STAGE_ZERO_ROW (TM_STAGE_ROWS - 1)
#define TM_STAGE_BYTES (TM_STAGE_ROWS * TM_STAGE_PITCH)

#define TM_CHUNK 32
#define TM_CHUNKS_PER_TILE (TM_TILE_WIDTH / TM_CHUNK)

// Unroll factor of the score chunk loop. params.nChunk is a multiple of it, so
// it is also the smallest chunk count the kernel can be given.
#define TM_SCORE_UNROLL 4
#define TM_MIN_CHUNKS_PER_TILE TM_SCORE_UNROLL

// Output row y consumes V(y-1) and V(y+h-1) while V(y+h) is being produced, so
// h+2 prefix-sum rows must stay live.
#define TM_V_ROWS (TM_MAX_TDIM + 2)

// int32 values per scratch row, in the kernel's even-then-odd lane order.
#define TM_SCRATCH_ROW TM_TILE_WIDTH

// Two DVLDPermHBU patterns: [0..31] the image window, [32..63] the
// coefficients replicated across lanes.
#define TM_PERM_PATTERN 32
#define TM_PERM_ENTRIES (2 * TM_PERM_PATTERN)

// Parameters produced by the host. Every field is derived from the tensor
// geometry at create time, so none of them varies from submit to submit.
typedef struct TemplateMatchingParams
{
    int32_t imageWidth;
    int32_t imageHeight;
    int32_t templateWidth;  /* w, 1..TM_MAX_TDIM                    */
    int32_t templateHeight; /* h, 1..TM_MAX_TDIM                    */
    int32_t haloX;          /* RDF halo in x, rounded up to even    */
    int32_t haloY;          /* RDF halo in y, = h - 1               */
    int32_t outWidth;       /* imageWidth  - w + 1                  */
    int32_t outHeight;      /* imageHeight - h + 1                  */
    int32_t niterX;         /* output tiles, x then y               */
    int32_t niterY;
    int32_t numTaps;    /* tapPairsY * tapGroupsX, below        */
    int32_t tapGroupsX; /* JB = ceil(w/4)                       */
    int32_t tapPairsY;  /* IB = h/2 + 1, 2 if h == 1 and w <= 4  */
    int32_t sqPairs;    /* (w+1)/2 VSumSq pairs                 */
    int32_t groupTaps;  /* taps per 24-bit accumulator group    */
    int32_t numGroups;  /* accumulator groups per chunk         */
    int32_t vRows;      /* h + 2 live prefix-sum rows           */
    int32_t nChunk;     /* 32-column chunks evaluated per tile  */
} TemplateMatchingParams;

#endif /* TEMPLATE_MATCHING_PARAM_H */
