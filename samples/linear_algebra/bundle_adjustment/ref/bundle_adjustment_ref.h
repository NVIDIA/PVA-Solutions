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
 * @file bundle_adjustment_ref.h
 *
 * @brief Dependency-free CPU reference for the bundle_adjustment operator
 *        (Levenberg-Marquardt with Schur-complement point elimination).
 */

#ifndef BUNDLE_ADJUSTMENT_REF_H
#define BUNDLE_ADJUSTMENT_REF_H

#include <cstdint>

/** @brief LM and robustifier tuning knobs for BundleAdjustmentRef(). */
struct BundleAdjustmentSetting
{
    int32_t maxIterations = 10;
    float minCostChange   = 1e-7f;
    float minCost         = 1e-6f;

    float lambdaInit      = 1e-4f;
    float lambdaMin       = 1e-8f;
    float lambdaMax       = 1e8f;
    float lambdaDecFactor = 0.5f;
    float lambdaIncFactor = 100.f;

    // Points closer than this along camera +Z get residual (1, 1) and no Jacobian.
    float minDistance = 1.0f;

    // Inverse of the isotropic measurement variance.
    float rinvMeas = 1.f / (0.01f * 0.01f);

    // Cauchy robustifier scale (normalized image plane).
    float robustifierScaleMeas = 4.f / 1900.f;
};

/**
 * @brief Run CPU-reference Levenberg-Marquardt bundle adjustment.
 *
 * @param poses         cameraCount * 16 floats: row-major 4x4 SE(3) rig poses.
 * @param points        trackCount * 4 floats: homogeneous (x, y, z, w).
 * @param measurements  trackCount * cameraCount * 2 floats: normalized (u, v) of track i in
 *                       camera j at [(i * cameraCount + j) * 2]; unobserved entries are not read.
 * @param bitmask       trackCount entries: bit j set = camera j observes the track; bit 31 set =
 *                       track is active, clear = point fixed (still contributes to cost/camera Hessian).
 * @param rig2cam       16 floats: row-major 4x4 rig-to-camera transform; cameraPose = rig2cam * pose.
 * @param cameraCount   Number of cameras; must be <= 31 (bit 31 is the active flag).
 * @param trackCount    Number of tracks.
 * @param activeCameraCount  Number of optimized cameras.
 * @param activeCameraBegin  First optimized camera; other cameras are fixed but still contribute
 *                       to the point Hessian and cost.
 * @param setting       LM/robustifier tuning knobs.
 * @param updatedPoses  Output, same layout as poses.
 * @param updatedPoints Output, same layout as points (w written as 1).
 * @param finalCost     Output, total cost of the returned state (0 if no iteration ran).
 * @param iterationsUsed Output, LM iterations executed.
 *
 * @return 0 on success.
 */
int32_t BundleAdjustmentRef(const float *poses, const float *points, const float *measurements, const int32_t *bitmask,
                            const float *rig2cam, int32_t cameraCount, int32_t trackCount, int32_t activeCameraCount,
                            int32_t activeCameraBegin, const BundleAdjustmentSetting &setting, float *updatedPoses,
                            float *updatedPoints, float *finalCost, int32_t *iterationsUsed);

#endif /* BUNDLE_ADJUSTMENT_REF_H */
