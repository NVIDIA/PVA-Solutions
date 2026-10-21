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

#ifndef BUNDLE_ADJUSTMENT_PARAM_H
#define BUNDLE_ADJUSTMENT_PARAM_H

const int kCameraBlockSize      = 32;
const int kMaxCameraCount       = 21;
const int kTrackCountTile       = 128;
const int kMaxTrackCount        = 20480;
const int kActiveCameraCount    = 5;
const int kSymmetricCameraSize  = (kActiveCameraCount * (kActiveCameraCount + 1) / 2);
const int kTrackPropertiesCount = 44;
const int kWordWidth            = 16;

const int kMatAHeight = (kSymmetricCameraSize * 36 * kWordWidth + kTrackCountTile - 1) / kTrackCountTile;
const int kMatBHeight = (kActiveCameraCount * 6 * kWordWidth + kTrackCountTile - 1) / kTrackCountTile;
const int kHpcHeight  = 18 * kActiveCameraCount * 2;

// Transposing load/store line pitch must be 16k+1 elements (agen.lane_ofst = k); shared with
// bundle_adjustment_top.c (lane_ofst) and OpBundleAdjustment.cpp (SQDF VMEM-side pitch).
const int kPoints3dStagePitch     = 17; // row width 4 (x,y,z,w): 16*1+1
const int kRigPoseStagePitch      = 17; // row width 16 (4x4 pose): 16*1+1
const int kMeasurementsStagePitch = 49; // row width 2*kMaxCameraCount=42: 16*3+1

const float kPi     = 3.1415926535897932384626433832795028841971;
const float kPiNorm = 1.0f / (2 * kPi);
const float kPiHalf = 3.1415926535897932384626433832795028841971 / 2.0f;
const float kFac    = 1.0f / 6.0f;
const float kFac1   = kFac * 0.25;
const float kFac2   = kFac * 0.05;

const float kLambdaMin       = 1.0e-8f;
const float kLambdaMax       = 1.0e8f;
const float kLambdaDecFactor = 0.5f;
const float kLambdaIncFactor = 100.0f;
const float kMinCostChange   = 1.0e-7f;
const float kMinCost         = 1.0e-6f;

// I-cache prefetch span for the per-tile hot group that bundle_adjustment.bcf lays out contiguously
// (BuildReducedSystem .. ComputeResidualKnl); only meaningful together with that layout.
const int kIcachePrefetchBytes = 16384;

// Prefetch span for the solve/update code (TransformCam2rigKnl .. ComputeCostKnl), which the per-tile
// loop evicts before ComputeUpdate runs.
const int kIcachePrefetchSolveBytes = 15360;

#endif // BUNDLE_ADJUSTMENT_PARAM_H
