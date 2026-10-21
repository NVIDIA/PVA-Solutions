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
 * Data contract shared by the host operator and the device kernel: geometry #defines and
 * capacity limits only. RasterDataFlow constraint numbers below: see class RasterDataFlow
 * in cupva_host.hpp.
 */

#ifndef DEPTHTOSPACE_RDF_PARAM_H
#define DEPTHTOSPACE_RDF_PARAM_H

/*
 * Set per device executable by the build (-DBLK_SZ=2|3|4), so the agen strides, the agen
 * count and the VMEM buffer count are all compile-time constants.
 */
#ifndef BLK_SZ
#    error "BLK_SZ must be defined to 2, 3 or 4 by the device build"
#endif

#if (BLK_SZ != 2) && (BLK_SZ != 3) && (BLK_SZ != 4)
#    error "BLK_SZ must be 2, 3 or 4"
#endif

/*
 * One executable is ONE output row offset: BLK_SZ input dataflows + 1 output. The host builds
 * BLK_SZ programs and submits them together, so VMEM holds one row offset's tiles.
 */
#define NUM_SRC_BUFFERS (BLK_SZ)

/* Transposed byte access moves 32 lanes, each a different ROW, so TILE_H must be a
 * multiple of 32. */
#define TRANSP_ROWS (32)

/*
 * Output tile <= 256 columns, so a trailing partial tile can never need more than the 255
 * bytes of column padding constraint 6 allows, at any width: 128/64/64 for block size 2/3/4.
 * The host evaluates this same macro, so the two cannot disagree.
 */
#define TILE_W_FOR_BLK(b) (((256 / (b)) / 32) * 32)

#define TILE_W TILE_W_FOR_BLK(BLK_SZ)
#define TILE_H (32)

#define OUT_TILE_W (TILE_W * BLK_SZ)
#define OUT_TILE_H (TILE_H)

/* Each iteration covers one input column across TRANSP_ROWS rows. */
#define EXEC_NITER (TILE_W * (TILE_H / TRANSP_ROWS))

#endif /* DEPTHTOSPACE_RDF_PARAM_H */
