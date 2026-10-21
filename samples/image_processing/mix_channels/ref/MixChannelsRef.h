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

/*
 * @file MixChannelsRef.h
 *
 * @brief Defines functions for reference model verification
 */

#ifndef MIX_CHANNELS_REF_H
#define MIX_CHANNELS_REF_H

#include <OpMixChannels.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

void generateRandomMixChannelsInput(NVCVTensorHandle *inHandles, int inTensorCount);

int32_t checkMixChannelsOutput(NVCVTensorHandle *refHandles, NVCVTensorHandle *pvaHandles, int outTensorCount);

void mixChannelsRef(NVCVTensorHandle *inHandles, NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount,
                    PVAMixChannelsCode code);

#ifdef __cplusplus
}
#endif

#endif /* MIX_CHANNELS_REF_H */
