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

#ifndef DATA_TYPE_DEF_H
#define DATA_TYPE_DEF_H

#include "image_resize_param.h"

#include <cupva_device.h>

// VMEM buffer allocations
// src and pre must be in different superbanks
// dst and post must be in different superbanks
#if defined(NN_OP)
#    if defined(U8_IO)
VMEM(A, uint8_t, srcBufPing, MAX_NN_U8_SRC_BUF_SIZE);
VMEM(B, uint8_t, srcBufPong, MAX_NN_U8_SRC_BUF_SIZE);
VMEM(B, uint8_t, dstBuf, MAX_NN_U8_DST_BUF_SIZE * 2);
VMEM(C, uint32_t, idxBuf, MAX_NN_U8_DST_BUF_SIZE * 2);
#    elif defined(NV12_IO)
VMEM(B, uint8_t, srcBufPing, MAX_NN_NV12_SRC_BUF_SIZE);
VMEM(B, uint8_t, srcBufPong, MAX_NN_NV12_SRC_BUF_SIZE);
VMEM(C, uint32_t, idxBuf, MAX_NN_NV12_DST_BUF_SIZE * 2);
VMEM(A, uint8_t, dstBuf, MAX_NN_NV12_DST_BUF_SIZE * 2);
VMEM(A, uint8_t, preBuf, MAX_NN_NV12_SRC_BUF_SIZE / 4);
VMEM(B, uint8_t, postBuf, MAX_NN_NV12_DST_BUF_SIZE / 4);
#    elif defined(RGB8_IO)
VMEM(B, uint8_t, srcBufPing, MAX_NN_RGB8_SRC_BUF_SIZE * 3);
VMEM(B, uint8_t, srcBufPong, MAX_NN_RGB8_SRC_BUF_SIZE * 3);
VMEM(A, uint8_t, dstBuf, MAX_NN_RGB8_DST_BUF_SIZE * 6);
VMEM(C, uint32_t, idxBuf, MAX_NN_RGB8_DST_BUF_SIZE * 2);
VMEM(A, uint8_t, preBuf, MAX_NN_RGB8_SRC_BUF_SIZE);
VMEM(B, uint8_t, postBuf, MAX_NN_RGB8_DST_BUF_SIZE);
#    elif defined(RGBA8_IO)
VMEM(A, uint8_t, srcBufPing, MAX_NN_RGBA8_SRC_BUF_SIZE * 4);
VMEM(A, uint8_t, srcBufPong, MAX_NN_RGBA8_SRC_BUF_SIZE * 4);
VMEM(B, uint8_t, dstBuf, MAX_NN_RGBA8_DST_BUF_SIZE * 8);
VMEM(C, uint32_t, idxBuf, MAX_NN_RGBA8_DST_BUF_SIZE * 2);
VMEM(B, uint8_t, preBuf, MAX_NN_RGBA8_SRC_BUF_SIZE);
VMEM(C, uint8_t, postBuf, MAX_NN_RGBA8_DST_BUF_SIZE);
#    elif defined(U16_IO)
VMEM(B, uint16_t, srcBufPing, MAX_NN_U16_SRC_BUF_SIZE);
VMEM(B, uint16_t, srcBufPong, MAX_NN_U16_SRC_BUF_SIZE);
VMEM(A, uint16_t, dstBuf, MAX_NN_U16_DST_BUF_SIZE * 2);
VMEM(C, uint32_t, idxBuf, MAX_NN_U16_DST_BUF_SIZE * 2);
#    endif

#elif defined(BL_OP)
// NV12 shares VMEM allocation configurations with U8, since NV12 is bounded on the Y plane
#    if defined(U8_IO) || defined(NV12_IO)
VMEM(B, uint8_t, srcBufPing, MAX_BL_U8_SRC_BUF_SIZE);
VMEM(C, uint8_t, srcBufPong, MAX_BL_U8_SRC_BUF_SIZE);
VMEM(C, uint8_t, dstBuf, MAX_BL_U8_DST_BUF_SIZE * 2);
VMEM(A, uint16_t, preBuf, MAX_BL_U8_SRC_BUF_SIZE);
VMEM(B, uint16_t, postBuf, MAX_BL_U8_DST_BUF_SIZE);
#    elif defined(RGB8_IO)
VMEM(B, uint8_t, srcBufPing, MAX_BL_RGB8_SRC_BUF_SIZE * 3);
VMEM(B, uint8_t, srcBufPong, MAX_BL_RGB8_SRC_BUF_SIZE * 3);
VMEM(C, uint8_t, dstBuf, MAX_BL_RGB8_DST_BUF_SIZE * 6);
VMEM(A, uint16_t, preBuf, MAX_BL_RGB8_SRC_BUF_SIZE);
VMEM(A, uint16_t, postBuf, MAX_BL_RGB8_DST_BUF_SIZE);
#    elif defined(RGBA8_IO)
VMEM(B, uint8_t, srcBufPing, MAX_BL_RGBA8_SRC_BUF_SIZE * 4);
VMEM(B, uint8_t, srcBufPong, MAX_BL_RGBA8_SRC_BUF_SIZE * 4);
VMEM(C, uint8_t, dstBuf, MAX_BL_RGBA8_DST_BUF_SIZE * 8);
VMEM(A, uint16_t, preBuf, MAX_BL_RGBA8_SRC_BUF_SIZE);
VMEM(A, uint16_t, postBuf, MAX_BL_RGBA8_DST_BUF_SIZE);
#    elif defined(U16_IO)
VMEM(A, uint16_t, srcBufPing, MAX_BL_U16_SRC_BUF_SIZE);
VMEM(B, uint16_t, srcBufPong, MAX_BL_U16_SRC_BUF_SIZE);
VMEM(C, uint16_t, dstBuf, MAX_BL_U16_DST_BUF_SIZE * 2);
#    endif
#endif

#if defined(U16_IO)
#    define DATA_TYPE uint16_t
#    define U8_FLAG 0
#else
#    define DATA_TYPE uint8_t
#    define U8_FLAG 1
#endif

// define auxiliary buffers
#if defined(NN_OP)
VMEM(A, uint32_t, laneIds, 16);
#    include "image_resize_opt_nn.h"
#    define INIT_RESIZE_CFG init_nn_resize_cfg
#    define INIT_SAMPLER_CFG init_nn_sampler_cfg
#    define UPDATE_SAMPLER_ADDRS update_nn_sampler_addrs
#    define UPDATE_SAMPLER_OFFSETS update_nn_sampler_offsets
#    define PRE_KNL_EXEC split_channels_u8_nn_exec
#    define POST_KNL_EXEC merge_channels_u8_nn_exec

#elif defined(BL_OP)
VMEM(A, int32_t, transformMatrix, 4);
#    include "image_resize_opt_bl.h"
#    define INIT_RESIZE_CFG init_bl_resize_cfg
#    define INIT_SAMPLER_CFG init_bl_sampler_cfg
#    define UPDATE_SAMPLER_ADDRS update_bl_sampler_addrs
#    define UPDATE_SAMPLER_OFFSETS update_bl_sampler_offsets
#    define PRE_KNL_EXEC pre_knl_bl_exec
#    define POST_KNL_EXEC post_knl_bl_exec

#endif

// define flag for compile pre/post processing buffers
#if defined(NN_OP)
#    if U8_FLAG && !defined(U8_IO)
#        define COMPILE_PREPROCESS 1
#    else
#        define COMPILE_PREPROCESS 0
#    endif
#elif defined(BL_OP)
#    define COMPILE_PREPROCESS U8_FLAG
#endif

#endif /* DATA_TYPE_DEF_H */
