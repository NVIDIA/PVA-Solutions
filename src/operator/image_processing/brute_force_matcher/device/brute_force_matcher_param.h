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

#ifndef BRUTE_FORCE_MATCHER_PARAM_H
#define BRUTE_FORCE_MATCHER_PARAM_H

#include "PvaOperatorTypes.h"

/**
 * @brief Number of queries processed simultaneously per SIMD batch (dvshortx = 32 halfword lanes)
 */
#define PVA_BF_QUERY_BATCH_SIZE (32)

/**
 * @brief Number of queries processed simultaneously per SIMD batch in uint32_t ( index and distance are packed into uint32_t)
 */
#define PVA_BF_QUERY_BATCH_SIZE_INT (PVA_BF_QUERY_BATCH_SIZE / 2)

/**
 * @brief Number of halfwords per descriptor (256 bits / 16 bits)
 */
#define PVA_BF_DESCRIPTOR_HALFWORDS (PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH / 2)

/**
 * @brief Size of one transposed patch buffer in bytes (batch_size * descriptor_halfwords * sizeof(uint16_t))
 */
#define PVA_BF_PATCH_BUFFER_SIZE (PVA_BF_QUERY_BATCH_SIZE * PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH)

/**
 * @brief Number of double-buffered patch/sampler instances for DMA-compute overlap
 */
#define PVA_BF_NUM_DOUBLE_BUFFERS (2)

/**
 * @brief 12-bit mask for extracting reference index from packed distance|index values
 */
#define PVA_BF_INDEX_MASK (0x0FFF)

/**
 * @brief Sentinel value indicating an invalid/rejected match in the internal match buffers
 */
#define PVA_BF_INVALID_MATCH (0xFFFFFFFFu)

/**
 * @brief Maximum possible Hamming distance for a 256-bit BRIEF descriptor (32 bytes x 8 bits)
 */
#define PVA_BF_MAX_HAMMING_DISTANCE (256)

/**
 * @brief Bit shift for packing distance and index in Q2R format: (distance << 12) | index
 */
#define PVA_BF_Q2R_DISTANCE_SHIFT (12)

/**
 * @brief Bit shift for packing distance and index in R2Q format: (distance << 16) | query_index
 */
#define PVA_BF_R2Q_DISTANCE_SHIFT (16)

#endif
