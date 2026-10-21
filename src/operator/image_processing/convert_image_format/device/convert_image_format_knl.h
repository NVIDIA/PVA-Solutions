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

#ifndef CONVERT_IMAGE_FORMAT_KNL_H
#define CONVERT_IMAGE_FORMAT_KNL_H

#include "convert_image_format_param.h"

#include <cupva_device.h>

typedef enum
{
    CHANNEL_Y,
    CHANNEL_U,
    CHANNEL_V,
} ChannelYUV;

typedef enum
{
    CHANNEL_R,
    CHANNEL_G,
    CHANNEL_B,
    CHANNEL_A,
} ChannelRGBA;

typedef struct ConvertImageFormatConfig
{
    AgenCFG inAgenCfgs[MAX_CHANNEL_COUNT];
    AgenCFG outAgenCfgs[MAX_CHANNEL_COUNT];
    AgenCFG inSamplingAgenCfg;
    AgenCFG outSamplingAgenCfg;

    int niter;
    int samplingNiter;

    /* Luma offset subtracted from Y on the YUV→RGB path. COLOR_OFFSET_L (16)
     * for studio-range BT.601; 0 for NV12_ER full-range. Set per conversion
     * code in the device top-level (convert_image_format_top.c). */
    int lumaOffset;
} ConvertImageFormatConfig;

#define REPLICATE_X8(X) X, X, X, X, X, X, X, X

#endif // CONVERT_IMAGE_FORMAT_KNL_H
