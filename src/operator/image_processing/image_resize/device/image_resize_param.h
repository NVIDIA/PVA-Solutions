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

#ifndef IMAGE_RESIZE_PARAM_H
#define IMAGE_RESIZE_PARAM_H

#define MAX_NTILE (256)
#define QBIT (16)
#define QBIT_HIGH (32)

// nearest neighbor
#define MAX_NN_U8_SRC_BUF_SIZE (98304)
#define MAX_NN_U8_DST_BUF_SIZE (16384)
#define MAX_NN_NV12_SRC_BUF_SIZE (63488)
#define MAX_NN_NV12_DST_BUF_SIZE (16384)
#define MAX_NN_RGB8_SRC_BUF_SIZE (14336)
#define MAX_NN_RGB8_DST_BUF_SIZE (16384)
#define MAX_NN_RGBA8_SRC_BUF_SIZE (14080)
#define MAX_NN_RGBA8_DST_BUF_SIZE (14528)
#define MAX_NN_U16_SRC_BUF_SIZE (32768)
#define MAX_NN_U16_DST_BUF_SIZE (16384)
// bilinear interpolation
#define MAX_BL_U8_SRC_BUF_SIZE (43680)
#define MAX_BL_U8_DST_BUF_SIZE (43680)
#define MAX_BL_RGB8_SRC_BUF_SIZE (21845)
#define MAX_BL_RGB8_DST_BUF_SIZE (21845)
#define MAX_BL_RGBA8_SRC_BUF_SIZE (16384)
#define MAX_BL_RGBA8_DST_BUF_SIZE (16384)
#define MAX_BL_U16_SRC_BUF_SIZE (54000)
#define MAX_BL_U16_DST_BUF_SIZE (32768)

#endif /* IMAGE_RESIZE_PARAM_H */