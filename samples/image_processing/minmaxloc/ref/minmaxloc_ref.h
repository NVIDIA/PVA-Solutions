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
 * @file minmaxloc_ref.h
 *
 * @brief Defines functions for reference the minmax location model verification
 */

#ifndef MINMAXLOC_REF_H
#define MINMAXLOC_REF_H

#include <ImageIO.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
#    include <string>
extern "C" {
#endif

void generateRandomMinMaxLocInput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                  int min_cnt0, int max_cnt);

void generateMinMaxLocFixedImgInputC(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                     const char *assetDir);

int32_t checkMinMaxLocOutput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle srcTensorHandle,
                             NVCVTensorHandle dstMinValTensorHandle, NVCVTensorHandle dstMinLocTensorHandle,
                             NVCVTensorHandle dstMinCntTensorHandle, NVCVTensorHandle dstMaxValTensorHandle,
                             NVCVTensorHandle dstMaxLocTensorHandle, NVCVTensorHandle dstMaxCntTensorHandle,
                             int loc_capacity);

#ifdef __cplusplus
}

void generateMinMaxLocFixedImgInput(NVCVTensorRequirements *tensorRequirements, NVCVTensorHandle inTensorHandle,
                                    std::string assetDir);
#endif

#endif /* MINMAXLOC_REF_H */
