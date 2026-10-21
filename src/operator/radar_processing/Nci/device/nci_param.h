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

#ifndef NCI_PARAM_H
#define NCI_PARAM_H

#include <stdint.h>

// Fixed-point type definitions
typedef int32_t Q16_15;   // Signed 16.15 fixed-point (1 sign + 16 integer + 15 fractional bits)
typedef uint32_t UQ16_16; // Unsigned 16.16 fixed-point (16 integer + 16 fractional bits)
#define MAX_INT32 0x7FFFFFFF
#define MAX_UINT32 0xFFFFFFFF

// Fixed-point multiplication functions
static inline Q16_15 Q16_15_mult(Q16_15 a, Q16_15 b)
{
    return (Q16_15)(((int64_t)a * b) >> 15);
}

static inline UQ16_16 UQ16_16_mult(UQ16_16 a, UQ16_16 b)
{
    return (UQ16_16)(((uint64_t)a * b) >> 15);
}

// Fixed-point conversion functions
static inline float Q16_15_to_float(Q16_15 val)
{
    return (float)val / (1 << 15);
}

static inline float UQ16_16_to_float(UQ16_16 val)
{
    return (float)val / (1 << 16);
}

static inline Q16_15 float_to_Q16_15(float val)
{
    return (Q16_15)(val * (1 << 15));
}

static inline UQ16_16 float_to_UQ16_16(float val)
{
    return (UQ16_16)(val * (1 << 16));
}

const uint32_t MaxInputTileSize       = (48u * 1024u) / sizeof(Q16_15);
const uint32_t MaxNciRxSize           = (32u * 1024u) / sizeof(UQ16_16);
const uint32_t MaxNciFinalSize        = (16u * 1024u) / sizeof(UQ16_16);
const uint32_t MaxNoiseEstimateSize   = (16u * 1024u) / sizeof(UQ16_16);
const uint32_t MaxSamplesSizePerTile  = 1024u;
const uint32_t MaxNumRangeBinsPerTile = 12u;
const uint32_t threshold              = 10u;

const uint32_t agenTruncateMask   = 0x80;
const uint32_t agenRoundMask      = 0x0;
const uint32_t notLastRx          = 0u;
const uint32_t lastRx             = 1u;
const uint32_t notLastDopplerFold = 0u;
const uint32_t lastDopplerFold    = 1u;

#endif /* NCI_PARAM_H */