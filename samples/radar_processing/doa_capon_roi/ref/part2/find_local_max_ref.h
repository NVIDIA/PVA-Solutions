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

#ifndef FIND_LOCAL_MAX_REF_H
#define FIND_LOCAL_MAX_REF_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Compute the three strongest Capon peaks for one ROI. */
void findLocalMaxRef(const float *inverseReal, const float *inverseImag, const int32_t *steeringVectors,
                     int32_t matrixSize, int32_t numAzimuthBins, int32_t numElevationBins, int32_t steeringQBits,
                     float noiseFloorRatio, bool enableSubBinInterpolation, float *peakPower, float *azimuthBins,
                     float *elevationBins);

#ifdef __cplusplus
}
#endif

#endif /* FIND_LOCAL_MAX_REF_H */
