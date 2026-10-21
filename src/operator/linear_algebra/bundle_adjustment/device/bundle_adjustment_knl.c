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

#include "bundle_adjustment_knl.h"

#include "bundle_adjustment_param.h"

#include <cupva_device.h>

void AgenConfigInitKnl(ParamBundlerType *params)
{
    dvfloatx consts;
    consts.lo     = replicatef(0.0f);
    consts.hi     = replicatef(0.0f);
    params->zeros = consts;

    consts.lo    = replicatef(1.0f);
    consts.hi    = replicatef(1.0f);
    params->ones = consts;

    int niteri = 0, thi = 0, cfgi = 0;
    params->vecw = pva_elementsof(dvfloatx);

    int tile_iter = (kTrackCountTile + params->vecw - 1) / params->vecw;
    agen a0;
    AgenWrapper w0;

    {
        // ResetVariableKnl Hpp
        int niter2 = tile_iter;
        a0         = init((float *)params->Hpp0_ptr);
        w0.size    = sizeof(float);
        w0.n1      = 9;
        w0.s1      = kTrackCountTile;
        w0.n2      = niter2;
        w0.s2      = params->vecw;
        INIT_AGEN2(a0, w0);

        params->niters[niteri++] = 9 * niter2;
        params->cfgs[cfgi++]     = extract_agen_cfg(a0);
        params->ths[thi++]       = 0;

        // ResetVariableKnl bp
        a0      = init((float *)params->bp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = niter2;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);

        params->niters[niteri++] = 3 * niter2;
        params->cfgs[cfgi++]     = extract_agen_cfg(a0);
        params->ths[thi++]       = 0;

        // ResetVariableKnl Hpc
        a0      = init((float *)params->Hpc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 18 * kActiveCameraCount;
        w0.s1   = kTrackCountTile;
        w0.n2   = niter2;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);

        params->niters[niteri++] = 18 * kActiveCameraCount * niter2;
        params->cfgs[cfgi++]     = extract_agen_cfg(a0);
        params->ths[thi++]       = 0;
    }

    // TransformPointKnl
    {
        // cfgi = 3
        a0      = init((float *)params->trans_poses_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kCameraBlockSize;
        w0.n2   = 4;
        w0.s2   = 4 * kCameraBlockSize;
        w0.n3   = tile_iter;
        w0.s3   = 0;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->rig_poses_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kCameraBlockSize;
        w0.n2   = 4;
        w0.s2   = 4 * kCameraBlockSize;
        w0.n3   = tile_iter;
        w0.s3   = 0;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 4;
        w0.s2   = kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_ptr + 4 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 4;
        w0.s2   = kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_cam_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 4;
        w0.s2   = 0;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_rig_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 4;
        w0.s2   = 0;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        params->niters[niteri++] = 4 * tile_iter;
        params->ths[thi++]       = 3;
    }

    // ComputeResidualKnl
    {
        // cfgi = 9
        // points3d
        a0      = init((float *)params->points3d_cam_ptr + 2 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = -kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // Transposing load from the padded AoS staging buffer; lane_ofst selects column camera*2+1 (y).
        // Camera is chosen at call time via adv_agen_base_B().
        const int32_t measLaneOfst = (kMeasurementsStagePitch - 1) / 16;

        a0           = init((float *)params->measurements_ptr + 1);
        a0.lane_ofst = measLaneOfst;
        w0.size      = sizeof(float);
        w0.n1        = 2;
        w0.s1        = -1;
        w0.n2        = tile_iter;
        w0.s2        = 16 * kMeasurementsStagePitch;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0           = init((float *)params->measurements_ptr + kMeasurementsStagePitch * kTrackCountTile + 1);
        a0.lane_ofst = measLaneOfst;
        w0.size      = sizeof(float);
        w0.n1        = 2;
        w0.s1        = -1;
        w0.n2        = tile_iter;
        w0.s2        = 16 * kMeasurementsStagePitch;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->residual0_ptr + kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 2;
        w0.s1   = -kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((int *)params->infront_ptr);
        w0.size = sizeof(int);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 4
        params->niters[niteri++] = tile_iter;
        params->ths[thi++]       = 0;
    }
    // FindValidTracksKnl
    {
        // cfgi = 14
        a0      = init((int *)params->camera_bitmask_ptr);
        w0.size = sizeof(int);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((int *)params->camera_bitmask_ptr + kTrackCountTile);
        w0.size = sizeof(int);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((int *)params->infront_ptr);
        w0.size = sizeof(int);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((int *)params->valid_flag_ptr);
        w0.size = sizeof(int);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 5
        params->niters[niteri++] = tile_iter;
        params->ths[thi++]       = 0;
    }

    // weight
    {
        // cfgi = 18
        a0      = init((float *)params->residual0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 2;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->weight0_ptr);
        w0.size = sizeof(float);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->weight1_ptr);
        w0.size = sizeof(float);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 6
        params->niters[niteri++] = tile_iter;
        params->ths[thi++]       = 0;
    }

    // ComputeJacobianPoint
    {
        // cfgi = 21
        a0      = init((float *)params->trans_poses_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kCameraBlockSize;
        w0.n2   = 3;
        w0.s2   = 4 * kCameraBlockSize;
        w0.n3   = tile_iter;
        w0.s3   = 0;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_cam_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 7
        params->niters[niteri++] = tile_iter;
        params->ths[thi++]       = 0;

        a0      = init((int *)params->valid_flag_ptr);
        w0.size = sizeof(int);
        w0.n1   = 6;
        w0.s1   = 0;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jp1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 8
        params->niters[niteri++] = 6 * tile_iter;
        params->ths[thi++]       = 0;
    }

    {
        // ComputeWJpKnl
        // cfgi = 26
        a0      = init((float *)params->weight1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 2;
        w0.s2   = 0;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jp1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = 3 * kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = 3 * kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 9
        params->niters[niteri++] = 2 * tile_iter;
        params->ths[thi++]       = 0;

        // ComputeJpTJpKnl
        // cfgi = 29
        a0      = init((float *)params->Jp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 2 * kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 3 / 3;
        w0.s3   = 3 * 2 * kTrackCountTile;
        w0.n4   = 3;
        w0.s4   = 0;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jp1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 3 / 3;
        w0.s3   = 0;
        w0.n4   = 3;
        w0.s4   = 2 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->JpTJp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = 0;
        w0.n3   = 3 / 3;
        w0.s3   = 3 * kTrackCountTile;
        w0.n4   = 3;
        w0.s4   = 3 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 10
        params->niters[niteri++] = 6 * tile_iter;
        params->ths[thi++]       = 1;

        // ScalePoseDiagnalKnl
        // cfgi = 32
        a0      = init((float *)params->JpTJp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 4 * kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 11
        params->niters[niteri++] = 3 * tile_iter;
        params->ths[thi++]       = 0;

        // ScalePoseDiagnalKnl
        // cfgi = 33
        a0      = init((float *)params->Hpp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 9;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Hpp1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 9;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->JpTJp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 9;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 12
        params->niters[niteri++] = 9 * tile_iter;
        params->ths[thi++]       = 0;
    }

    {
        // ComputeJpTEKnl
        // cfgi = 36
        a0      = init((float *)params->Jp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 2 * kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 3 / 3;
        w0.s3   = 3 * 2 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 0;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->residual0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 3 / 3;
        w0.s3   = 0;
        w0.n4   = 1;
        w0.s4   = 2 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->JpTE_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = 0;
        w0.n3   = 3 / 3;
        w0.s3   = 3 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 3 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 13
        params->niters[niteri++] = 2 * tile_iter;
        params->ths[thi++]       = 1;

        // UpdateBpKnl
        // cfgi = 39
        a0      = init((float *)params->bp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->bp1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->JpTE_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 14
        params->niters[niteri++] = 3 * tile_iter;
        params->ths[thi++]       = 0;
    }

    {
        // ComputeJacobianPoseS1Knl
        // cfgi = 42
        a0      = init((float *)params->points3d_cam_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->rig2cam_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 1;
        w0.n2   = 3;
        w0.s2   = 4;
        w0.n3   = tile_iter;
        w0.s3   = 0;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jc0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // ComputeJacobianPoseS2Knl
        a0      = init((float *)params->points3d_rig_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jc0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jc0_ptr + 6 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 15
        params->niters[niteri++] = tile_iter;
        params->ths[thi++]       = 0;

        // FilterJacobianKnl
        // cfgi = 48
        a0      = init((float *)params->Jc0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 12;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->valid_flag_ptr);
        w0.size = sizeof(float);
        w0.n1   = 12;
        w0.s1   = 0;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jc1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 12;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 16
        params->niters[niteri++] = 12 * tile_iter;
        params->ths[thi++]       = 0;
    }

    {
        // ComputeWJcKnl
        // cfgi = 51
        a0      = init((float *)params->weight1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 4;
        w0.s2   = 0;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jc1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 4;
        w0.s2   = 3 * kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jc0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 4;
        w0.s2   = 3 * kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 17
        params->niters[niteri++] = 4 * tile_iter;
        params->ths[thi++]       = 0;

        // ComputeJcTJcKnl
        // cfgi = 54
        // MatrixMulKnl_323: 3x2 * 2x3 -> 3x3 inner product
        a0      = init((float *)params->Jc0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 2 * kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 2;
        w0.s3   = 6 * kTrackCountTile;
        w0.n4   = 2;
        w0.s4   = 0;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Jc1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 2 * kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 2;
        w0.s3   = 0;
        w0.n4   = 2;
        w0.s4   = 6 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->JcTJc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 6 * kTrackCountTile;
        w0.n3   = 2;
        w0.s3   = 3 * kTrackCountTile;
        w0.n4   = 2;
        w0.s4   = 18 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 18
        params->niters[niteri++] = 4 * tile_iter;
        params->ths[thi++]       = 1;

        // ScalePoseDiagnalKnl
        // cfgi = 57
        a0      = init((float *)params->JcTJc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = 7 * kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 19
        params->niters[niteri++] = 6 * tile_iter;
        params->ths[thi++]       = 0;
    }

    // update A with Hcc
    {
        // cfgi = 58
        a0      = init((float *)params->MatA0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 36;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        agen_C a0_bufC = init_C((float *)params->buf_c_ptr);
        w0.size        = sizeof(float);
        w0.n1          = 36;
        w0.s1          = params->vecw;
        INIT_AGEN1(a0_bufC, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a0_bufC);

        a0      = init((float *)params->MatA1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 36;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 20
        params->niters[niteri++] = 36;
        params->ths[thi++]       = 0;

        // CopyAKnl
        // cfgi = 61
        int curr_iter = (36 * kWordWidth + params->vecw - 1) / params->vecw;
        a0            = init((float *)params->MatA0_ptr);
        w0.size       = sizeof(float);
        w0.n1         = curr_iter;
        w0.s1         = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->MatA1_ptr);
        w0.size = sizeof(float);
        w0.n1   = curr_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 21
        params->niters[niteri++] = curr_iter;
        params->ths[thi++]       = 0;
    }

    // update JcTE
    {
        // ComputeJpTJcKnl
        // cfgi = 63
        // MatrixMulKnl_323 config for Hpc
        agen_A a0_Jp0, a1_Jc1;
        agen_D a2_Hpc;
        a0_Jp0  = init_A((float *)params->Jp0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 2 * kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 2;
        w0.s3   = 0;
        w0.n4   = tile_iter;
        w0.s4   = params->vecw;
        INIT_AGEN4(a0_Jp0, w0);
        params->cfgs[cfgi++] = extract_agen_A_cfg(a0_Jp0);

        a1_Jc1  = init_A((float *)params->Jc1_ptr);
        w0.size = sizeof(float);

        w0.n1 = 3;
        w0.s1 = 2 * kTrackCountTile;
        w0.n2 = 2;
        w0.s2 = kTrackCountTile;
        w0.n3 = 2;
        w0.s3 = 6 * kTrackCountTile;
        w0.n4 = tile_iter;
        w0.s4 = params->vecw;
        INIT_AGEN4(a1_Jc1, w0);
        params->cfgs[cfgi++] = extract_agen_A_cfg(a1_Jc1);

        a2_Hpc  = init_D((float *)params->Hpc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 3 * kTrackCountTile;
        w0.n3   = 2;
        w0.s3   = 9 * kTrackCountTile;
        w0.n4   = tile_iter;
        w0.s4   = params->vecw;
        INIT_AGEN4(a2_Hpc, w0);
        params->cfgs[cfgi++] = extract_agen_D_cfg(a2_Hpc);

        // niteri = 22
        params->niters[niteri++] = 2 * tile_iter;
        params->ths[thi++]       = 1;

        // ComputeJcTEKnl
        // cfgi = 66
        a0      = init((float *)params->Jc0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 2 * kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 6 / 3;
        w0.s3   = 3 * 2 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 0;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->residual0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 2;
        w0.s2   = kTrackCountTile;
        w0.n3   = 6 / 3;
        w0.s3   = 0;
        w0.n4   = 1;
        w0.s4   = 2 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->JcTE_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 2;
        w0.s2   = 0;
        w0.n3   = 6 / 3;
        w0.s3   = 3 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 6 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 23
        params->niters[niteri++] = 4 * tile_iter;
        params->ths[thi++]       = 1;

        // cfgi = 69
        a0      = init((float *)params->MatB0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        agen_C a0_bufC = init_C((float *)params->buf_c_ptr);
        w0.size        = sizeof(float);
        w0.n1          = 6;
        w0.s1          = params->vecw;
        INIT_AGEN1(a0_bufC, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a0_bufC);

        a0      = init((float *)params->MatB1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 24
        params->niters[niteri++] = 6;
        params->ths[thi++]       = 0;
    }

    {
        // InverseSymmetric3X3Knl
        // cfgi = 72
        a0      = init((float *)params->Hpp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 9;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Hpp_inv_ptr);
        w0.size = sizeof(float);
        w0.n1   = 9;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((int *)params->status_ptr);
        w0.size = sizeof(int);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 25
        params->niters[niteri++] = tile_iter;
        params->ths[thi++]       = 0;

        // ComputeTpKnl
        // cfgi = 75
        a0      = init((float *)params->Hpp_inv_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 3 * kTrackCountTile;
        w0.n3   = 3 / 3;
        w0.s3   = 3 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 0;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->bp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 3;
        w0.s2   = kTrackCountTile;
        w0.n3   = 3 / 3;
        w0.s3   = 0;
        w0.n4   = 1;
        w0.s4   = 3 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->tp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 0;
        w0.n3   = 3 / 3;
        w0.s3   = 3 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 3 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 26
        params->niters[niteri++] = 3 * tile_iter;
        params->ths[thi++]       = 2;
    }

    // Reduced system B
    {
        // ComputeHpcTp
        // cfgi = 78
        a0      = init((float *)params->Hpc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 3 * kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = kTrackCountTile;
        w0.n3   = 6 / 3;
        w0.s3   = 3 * 3 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 0;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->tp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 3;
        w0.s2   = kTrackCountTile;
        w0.n3   = 6 / 3;
        w0.s3   = 0;
        w0.n4   = 1;
        w0.s4   = 3 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->HpcTp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 0;
        w0.n3   = 6 / 3;
        w0.s3   = 3 * kTrackCountTile;
        w0.n4   = 1;
        w0.s4   = 6 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 27
        params->niters[niteri++] = 6 * tile_iter;
        params->ths[thi++]       = 2;

        // cfgi = 81
        a0      = init((float *)params->MatB1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        agen_C a0_bufC = init_C((float *)params->buf_c_ptr);
        w0.size        = sizeof(float);
        w0.n1          = 6;
        w0.s1          = params->vecw;
        INIT_AGEN1(a0_bufC, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a0_bufC);

        a0      = init((float *)params->MatB0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 6;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 28
        params->niters[niteri++] = 6;
        params->ths[thi++]       = 0;
    }

    // Reduced system A
    {
        // ComputeTpc
        // cfgi = 84
        agen_D a0_HpcT;
        agen_C a1_HppI;
        agen_A a2_Tpc;
        a0_HpcT = init_D((float *)params->Hpc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 2;
        w0.s1   = 3 * kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = kTrackCountTile;
        w0.n3   = 3;
        w0.s3   = 6 * kTrackCountTile;
        w0.n4   = tile_iter;
        w0.s4   = params->vecw;
        INIT_AGEN4(a0_HpcT, w0);
        params->cfgs[cfgi++] = extract_agen_D_cfg(a0_HpcT);

        a1_HppI = init_C((float *)params->Hpp_inv_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 3 * kTrackCountTile;
        w0.n3   = 3;
        w0.s3   = 0;
        w0.n4   = tile_iter;
        w0.s4   = params->vecw;

        INIT_AGEN4(a1_HppI, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a1_HppI);

        a2_Tpc  = init_A((float *)params->Tpc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 2;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 6 * kTrackCountTile;
        w0.n3   = 3;
        w0.s3   = 2 * kTrackCountTile;
        w0.n4   = tile_iter;
        w0.s4   = params->vecw;
        INIT_AGEN4(a2_Tpc, w0);
        params->cfgs[cfgi++] = extract_agen_A_cfg(a2_Tpc);

        // niteri = 29
        params->niters[niteri++] = 3 * tile_iter;
        params->ths[thi++]       = 2;

        // ComputeTpcHpc
        // cfgi = 87
        agen_A a0_Tpc;
        a0_Tpc  = init_A((float *)params->Tpc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 2;
        w0.s1   = kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = 6 * kTrackCountTile;
        w0.n3   = 6 / 2;
        w0.s3   = 2 * kTrackCountTile;
        w0.n4   = 2;
        w0.s4   = 0;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0_Tpc, w0);
        params->cfgs[cfgi++] = extract_agen_A_cfg(a0_Tpc);

        agen_C a0_Hpc;
        a0_Hpc  = init_C((float *)params->Hpc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 3 * kTrackCountTile;
        w0.n2   = 3;
        w0.s2   = kTrackCountTile;
        w0.n3   = 6 / 2;
        w0.s3   = 0;
        w0.n4   = 2;
        w0.s4   = 9 * kTrackCountTile;
        w0.n5   = tile_iter;
        w0.s5   = params->vecw;
        INIT_AGEN5(a0_Hpc, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a0_Hpc);

        agen_D a0_TpcHpc;
        a0_TpcHpc = init_D((float *)params->TpcHpc_ptr);
        w0.size   = sizeof(float);
        w0.n1     = 2;
        w0.s1     = kTrackCountTile;
        w0.n2     = 3;
        w0.s2     = 6 * kTrackCountTile;
        w0.n3     = 6 / 2;
        w0.s3     = 2 * kTrackCountTile;
        w0.n4     = 2;
        w0.s4     = 18 * kTrackCountTile;
        w0.n5     = tile_iter;
        w0.s5     = params->vecw;
        INIT_AGEN5(a0_TpcHpc, w0);
        params->cfgs[cfgi++] = extract_agen_D_cfg(a0_TpcHpc);

        // niteri = 30
        params->niters[niteri++] = 6 * tile_iter;
        params->ths[thi++]       = 2;

        // cfgi = 90
        a0      = init((float *)params->MatA1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 36;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        agen_C a0_bufC = init_C((float *)params->buf_c_ptr);
        w0.size        = sizeof(float);
        w0.n1          = 36;
        w0.s1          = params->vecw;
        INIT_AGEN1(a0_bufC, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a0_bufC);

        a0      = init((float *)params->MatA0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 36;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 31
        params->niters[niteri++] = 36;
        params->ths[thi++]       = 0;
    }

    // ApplyUpdate
    {
        // ComputePointUpdateKnl
        // cfgi = 93
        a0      = init((float *)params->Tpc_all_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 6 * kTrackCountTile;
        w0.n2   = 6;
        w0.s2   = kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->Tpc_all_ptr + 18 * kActiveCameraCount * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 6 * kTrackCountTile;
        w0.n2   = 6;
        w0.s2   = kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->dC_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 6;
        w0.s2   = 1;
        w0.n3   = tile_iter;
        w0.s3   = 0;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->dp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = 6;
        w0.s2   = 0;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 32
        params->niters[niteri++] = 6 * tile_iter;
        params->ths[thi++]       = 5;

        // UpdateDpKnl
        // cfgi = 97
        a0      = init((float *)params->tp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->dp1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->dp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 33
        params->niters[niteri++] = 3 * tile_iter;
        params->ths[thi++]       = 0;

        // cfgi = 100
        a0      = init((float *)params->pnew_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->pnew_ptr + 4 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_ptr + 4 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->camera_bitmask_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->camera_bitmask_ptr + kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 34
        params->niters[niteri++] = 3 * tile_iter;
        params->ths[thi++]       = 0;

        // cfgi = 106
        a0      = init((float *)params->points3d_ptr + 3 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_ptr + 4 * kTrackCountTile + 3 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = tile_iter;
        w0.s1   = params->vecw;
        INIT_AGEN1(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 35
        params->niters[niteri++] = tile_iter;
        params->ths[thi++]       = 0;
    }

    // Compute cost
    {
        // cfgi = 108
        a0      = init((int *)params->valid_flag_ptr);
        w0.size = sizeof(int);
        w0.n1   = 2;
        w0.s1   = 0;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->residual0_ptr);
        w0.size = sizeof(float);
        w0.n1   = 2;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->residual1_ptr);
        w0.size = sizeof(float);
        w0.n1   = 2;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        INIT_AGEN2(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 36
        params->niters[niteri++] = 2 * tile_iter;
        params->ths[thi++]       = 0;

        // cfgi = 111
        a0      = init((float *)params->points3d_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 4;
        w0.s2   = kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0      = init((float *)params->points3d_ptr + 4 * kTrackCountTile);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = 0;
        w0.n2   = 4;
        w0.s2   = kTrackCountTile;
        w0.n3   = tile_iter;
        w0.s3   = params->vecw;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // cfgi = 113
        a0      = init((float *)params->JcTJc_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        w0.n3   = 36 / 3;
        w0.s3   = 3 * kTrackCountTile;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        agen_C a0_bufC = init_C((float *)params->buf_c_ptr);
        w0.size        = sizeof(float);
        w0.n1          = 3;
        w0.s1          = kWordWidth;
        w0.n2          = tile_iter;
        w0.s2          = 0;
        w0.n3          = 36 / 3;
        w0.s3          = 3 * kWordWidth;
        INIT_AGEN3(a0_bufC, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a0_bufC);

        // niteri = 37
        params->niters[niteri++] = 36 * tile_iter / 3;
        params->ths[thi++]       = tile_iter - 1;

        // cfgi = 115
        a0      = init((float *)params->JcTE_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        w0.n3   = 6 / 3;
        w0.s3   = 3 * kTrackCountTile;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        a0_bufC = init_C((float *)params->buf_c_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kWordWidth;
        w0.n2   = tile_iter;
        w0.s2   = 0;
        w0.n3   = 6 / 3;
        w0.s3   = 3 * kWordWidth;
        INIT_AGEN3(a0_bufC, w0);
        params->cfgs[cfgi++] = extract_agen_C_cfg(a0_bufC);

        a0      = init((float *)params->HpcTp_ptr);
        w0.size = sizeof(float);
        w0.n1   = 3;
        w0.s1   = kTrackCountTile;
        w0.n2   = tile_iter;
        w0.s2   = params->vecw;
        w0.n3   = 6 / 3;
        w0.s3   = 3 * kTrackCountTile;
        INIT_AGEN3(a0, w0);
        params->cfgs[cfgi++] = extract_agen_cfg(a0);

        // niteri = 38
        params->niters[niteri++] = 6 * tile_iter / 3;
        params->ths[thi++]       = tile_iter - 1;

        agen_D a0_TpcHpc = init_D((float *)params->TpcHpc_ptr);
        w0.size          = sizeof(float);
        w0.n1            = 3;
        w0.s1            = kTrackCountTile;
        w0.n2            = tile_iter;
        w0.s2            = params->vecw;
        w0.n3            = 36 / 3;
        w0.s3            = 3 * kTrackCountTile;
        INIT_AGEN3(a0_TpcHpc, w0);
        params->cfgs[cfgi++] = extract_agen_D_cfg(a0_TpcHpc);
    }
}

void ResetVariableKnl(int cfg_index, ParamBundlerType *params)
{
    agen a0 = init_agen_from_cfg(params->cfgs[cfg_index]);

    int niter = params->niters[cfg_index];

    for (int i = 0; i < niter; i++) chess_loop_range(16, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        vstore(params->zeros, a0);
    }
}

void TransformCam2rigKnl(float *T, float *poses1, float *poses2, int lp, int n)
{
    int vecw  = pva_elementsof(dvfloatx);
    int niter = (n + vecw - 1) / vecw;

    agen a0 = init((float *)T);
    AgenWrapper w0;
    w0.size = sizeof(float);
    w0.n1   = 4;
    w0.s1   = 1;
    w0.n2   = 4;
    w0.s2   = 4;
    w0.n3   = 4;
    w0.s3   = 0;
    w0.n4   = niter;
    w0.s4   = 0;
    INIT_AGEN4(a0, w0);

    agen a1 = init((float *)poses1);
    AgenWrapper w1;
    w1.size = sizeof(float);
    w1.n1   = 4;
    w1.s1   = 0;
    w1.n2   = 4;
    w1.s2   = lp;
    w1.n3   = 4;
    w1.s3   = 4 * lp;
    w1.n4   = niter;
    w1.s4   = vecw;
    INIT_AGEN4(a1, w1);

    agen a2 = init((float *)poses2);
    AgenWrapper w2;
    w2.size = sizeof(float);
    w2.n1   = 4;
    w2.s1   = lp;
    w2.n2   = 4;
    w2.s2   = 0;
    w2.n3   = 4;
    w2.s3   = 4 * lp;
    w2.n4   = niter;
    w2.s4   = vecw;
    INIT_AGEN4(a2, w2);

    dvfloatx acc0 = chess_dont_care(dvfloatx);
    dvfloatx acc1 = chess_dont_care(dvfloatx);
    dvfloatx acc2 = chess_dont_care(dvfloatx);
    dvfloatx acc3 = chess_dont_care(dvfloatx);

    int pred_madd = 0, pred_store = 0, s_ctrl_store = 1, s_ctrl_madd = 0;
    for (int i = 0; i < niter * 16; i++) chess_loop_range(16, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        {
            const float d0    = float_load(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc0.lo = vmaddf(d1.lo, d0, acc0.lo, pred_madd);
            acc0.hi = vmaddf(d1.hi, d0, acc0.hi, pred_madd);

            vstore(acc0, a2, pred_store);
        }

        {
            const float d0    = float_load(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc1.lo = vmaddf(d1.lo, d0, acc1.lo, pred_madd);
            acc1.hi = vmaddf(d1.hi, d0, acc1.hi, pred_madd);

            vstore(acc1, a2, pred_store);
        }

        {
            const float d0    = float_load(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc2.lo = vmaddf(d1.lo, d0, acc2.lo, pred_madd);
            acc2.hi = vmaddf(d1.hi, d0, acc2.hi, pred_madd);

            vstore(acc2, a2, pred_store);
        }

        {
            const float d0    = float_load(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc3.lo = vmaddf(d1.lo, d0, acc3.lo, pred_madd);
            acc3.hi = vmaddf(d1.hi, d0, acc3.hi, pred_madd);

            vstore(acc3, a2, pred_store);
        }

        s_ctrl_madd  = mod_inc_pred_nz(s_ctrl_madd, 4 - 1, pred_madd);
        s_ctrl_store = mod_inc_pred_z(s_ctrl_store, 4 - 1, pred_store);
    }
}

void CopyDataKnl(float *d1, float *d2, int n)
{
    int vecw  = pva_elementsof(dvfloatx);
    int niter = (n + vecw - 1) / vecw;

    agen a1 = init((float *)d1);
    a1.n1   = n;
    a1.mod1 = vecw * sizeof(float);

    agen a2 = init((float *)d2);
    a2.n1   = n;
    a2.mod1 = vecw * sizeof(float);

    for (int i = 0; i < niter; ++i) chess_loop_range(2, )
    chess_prepare_for_pipelining
    {
        dvfloatx dvdata1 = dvfloat_load(a1);
        vstore(dvdata1, a2);
    }
}

void ComputeResidualKnl(const AgenCFG *a_points, const AgenCFG *a_meas, const AgenCFG *a_residual,
                        const AgenCFG *a_flag, float min_distance, int offset, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_points);
    agen_B a1 = init_agen_B_from_cfg(*a_meas);
    adv_agen_base_B(a1, offset * 4);
    agen_C a2 = init_agen_C_from_cfg(*a_residual);
    agen_D a3 = init_agen_D_from_cfg(*a_flag);

    dvfloatx ones;
    ones.lo = replicatef(1.0f);
    ones.hi = ones.lo;

    dvfloatx zeros;
    zeros.lo = replicatef(0.0f);
    zeros.hi = zeros.lo;

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        const dvfloatx dv_z = dvfloat_load(a0);
        dvintx infront;
        infront.lo = dv_z.lo >= min_distance;
        infront.hi = dv_z.hi >= min_distance;
        vstore(infront, a3);

        dvfloatx zinv = dvmux(infront, dvfrcp(dv_z), zeros);

        const dvfloatx dv_y     = dvfloat_load(a0);
        const dvfloatx dv_measy = dvfloat_load_transp(a1);

        dvfloatx resy;
        resy = dv_y * zinv;
        resy = resy - dv_measy;

        resy = dvmux(infront, resy, ones);

        vstore(resy, a2);

        const dvfloatx dv_x     = dvfloat_load(a0);
        const dvfloatx dv_measx = dvfloat_load_transp(a1);
        dvfloatx resx;
        resx = dv_x * zinv;
        resx = resx - dv_measx;

        resx = dvmux(infront, resx, ones);

        vstore(resx, a2);
    }
}

void ComputeWeightKnl(const AgenCFG *a_residual, float square_scale, AgenCFG *a_weight, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_residual);
    agen_B a1 = init_agen_B_from_cfg(*a_weight);

    dvfloatx ones;
    ones.lo = replicatef(1.0f);
    ones.hi = ones.lo;
    float c = frcp(square_scale);

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        const dvfloatx d0 = dvfloat_load(a0);
        dvfloatx square_error;
        square_error = d0 * d0;

        const dvfloatx d1 = dvfloat_load(a0);
        square_error      = dvmaddf(d1, d1, square_error, -1);

        dvfloatx sum;
        sum = dvmaddf(square_error, c, ones);

        dvfloatx weight;
        weight = dvfrcp(sum);

        vstore(weight, a1);
    }
}

void ScaleWeightKnl(const AgenCFG *w1, float rinv, const AgenCFG *w2, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*w1);
    agen_B a1 = init_agen_B_from_cfg(*w2);

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvfloatx d0 = dvfloat_load(a0);
        dvfloatx d1 = d0 * rinv;
        vstore(d1, a1);
    }
}

void ComputeJacobianPointKnl(int icam, const AgenCFG *a_pose, const AgenCFG *a_points, const AgenCFG *a_jp, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_pose);
    adv_agen_base_A(a0, icam * 4);
    agen_B a1 = init_agen_B_from_cfg(*a_points);
    agen_C a2 = init_agen_C_from_cfg(*a_jp);

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_prepare_for_pipelining
    {
        const dvfloatx chess_storage(DW) x = dvfloat_load(a1);
        const dvfloatx chess_storage(DW) y = dvfloat_load(a1);
        const dvfloatx chess_storage(DW) z = dvfloat_load(a1);

        dvfloatx chess_storage(DV) zinv;
        zinv = dvmux(z == 0.0, 0.0, dvfrcp(z));

        dvfloatx chess_storage(DW) zinv2 = zinv * zinv;

        for (int j = 0; j < 3; j++) chess_unroll_loop(*)
        {
            const float d0 = float_load(a0);
            const float d1 = float_load(a0);
            const float d2 = float_load(a0);

            dvfloatx chess_storage(DV) jp0;
            dvfloatx chess_storage(DV) jp1;
            dvfloatx chess_storage(DV) tmp0, chess_storage(DV) tmp1;

            jp0  = zinv * d0;
            tmp0 = zinv2 * d2;
            jp0  = dvmsubf(tmp0, x, jp0);

            jp1  = zinv * d1;
            tmp1 = zinv2 * d2;
            jp1  = dvmsubf(tmp1, y, jp1);

            vstore(jp0, a2);
            vstore(jp1, a2);
        }
    }
}

void FilterJacobianKnl(const AgenCFG *a_jacobian0, const AgenCFG *a_flag, const AgenCFG *a_jacobian1, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_flag);
    agen_B a1 = init_agen_B_from_cfg(*a_jacobian0);
    agen_C a2 = init_agen_C_from_cfg(*a_jacobian1);

    dvfloatx zeros;
    zeros.lo = replicatef(0.0f);
    zeros.hi = zeros.lo;

    for (int i = 0; i < niter; i++) chess_loop_range(12, )
    chess_unroll_loop(6)
    chess_prepare_for_pipelining
    {
        dvintx flag = dvint_load(a0);
        dvfloatx d0 = dvfloat_load(a1);

        dvfloatx d1 = dvmux(flag, d0, zeros);
        vstore(d1, a2);
    }
}

void ComputeJacobianPoseS1Knl(const AgenCFG *pt0, const AgenCFG *rig2cam, const AgenCFG *Jc, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*pt0);
    agen_B a1 = init_agen_B_from_cfg(*rig2cam);
    agen_C a2 = init_agen_C_from_cfg(*Jc);

    for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        dvfloatx a = dvfloat_load(a0);
        dvfloatx b = dvfloat_load(a0);
        dvfloatx c = dvfloat_load(a0);

        dvfloatx cinv;
        cinv = dvmux(c == 0.0, 0.0, dvfrcp(c));
        dvfloatx cinv2;
        cinv2 = cinv * cinv;

        for (int j = 0; j < 3; j++) chess_loop_range(3, )
        chess_unroll_loop(*)
        chess_prepare_for_pipelining
        {
            const vfloatx d0 = vfloat_load_ws(a1);
            const vfloatx d1 = vfloat_load_ws(a1);
            const vfloatx d2 = vfloat_load_ws(a1);

            dvfloatx jc0, tmp0;
            jc0.lo = d0 * cinv.lo;
            jc0.hi = d0 * cinv.hi;

            tmp0.lo = a.lo * d2;
            tmp0.hi = a.hi * d2;

            jc0 = dvmsubf(tmp0, cinv2, jc0);
            vstore(jc0, a2);

            dvfloatx jc1, tmp1;
            jc1.lo = d1 * cinv.lo;
            jc1.hi = d1 * cinv.hi;

            tmp1.lo = b.lo * d2;
            tmp1.hi = b.hi * d2;

            jc1 = dvmsubf(tmp1, cinv2, jc1);
            vstore(jc1, a2);
        }
    }
}

void ComputeJacobianPoseS2Knl(const AgenCFG *pt0, const AgenCFG *Jc0, const AgenCFG *Jc1, int niter)
{

    agen_A a0 = init_agen_A_from_cfg(*pt0);
    agen_B a1 = init_agen_B_from_cfg(*Jc0);
    agen_C a2 = init_agen_C_from_cfg(*Jc1);

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        {
            const dvfloatx d = dvfloat_load(a0);
            const dvfloatx e = dvfloat_load(a0);
            const dvfloatx f = dvfloat_load(a0);

            const dvfloatx d0 = dvfloat_load(a1);
            const dvfloatx d1 = dvfloat_load(a1);
            const dvfloatx d2 = dvfloat_load(a1);
            const dvfloatx d3 = dvfloat_load(a1);
            const dvfloatx d4 = dvfloat_load(a1);
            const dvfloatx d5 = dvfloat_load(a1);

            dvfloatx jc0;
            jc0 = e * d4;
            jc0 = dvmsubf(f, d2, jc0);
            vstore(jc0, a2);

            dvfloatx jc1;
            jc1 = e * d5;
            jc1 = dvmsubf(f, d3, jc1);
            vstore(jc1, a2);

            dvfloatx jc2;
            jc2 = f * d0;
            jc2 = dvmsubf(d, d4, jc2);
            vstore(jc2, a2);

            dvfloatx jc3;
            jc3 = f * d1;
            jc3 = dvmsubf(d, d5, jc3);
            vstore(jc3, a2);

            dvfloatx jc4;
            jc4 = d * d2;
            jc4 = dvmsubf(e, d0, jc4);
            vstore(jc4, a2);

            dvfloatx jc5;
            jc5 = d * d3;
            jc5 = dvmsubf(e, d1, jc5);
            vstore(jc5, a2);
        }
    }
}

void FindObservedTracksKnl(const AgenCFG *bitmask, const AgenCFG *valid_flag, int camind, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*bitmask);
    agen_B a1 = init_agen_B_from_cfg(*valid_flag);

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvintx mask = dvint_load(a0);

        dvintx mbit;
        mbit = mask >> camind;

        mbit = mbit & 1;

        vstore(mbit, a1);
    }
}

void FilterResidualKnl(const AgenCFG *flag, const AgenCFG *res1, const AgenCFG *res2, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*flag);
    agen_B a1 = init_agen_B_from_cfg(*res1);
    agen_C a2 = init_agen_C_from_cfg(*res2);

    dvfloatx zeros;
    zeros.lo = replicatef(0.0f);
    zeros.hi = zeros.lo;

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvintx flag = dvint_load(a0);
        dvfloatx d0 = dvfloat_load(a1);

        dvfloatx d1 = dvmux(flag, d0, zeros);
        vstore(d1, a2);
    }
}

void FindValidTracksKnl(const AgenCFG *a_bitmask, const AgenCFG *a_infront, const AgenCFG *a_valid_flag, int camind,
                        int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_bitmask);
    agen_B a1 = init_agen_B_from_cfg(*a_infront);
    agen_C a2 = init_agen_C_from_cfg(*a_valid_flag);

    dvfloatx zeros;
    zeros.lo = replicatef(0.0f);
    zeros.hi = zeros.lo;

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvintx mask    = dvint_load(a0);
        dvintx isfront = dvint_load(a1);

        dvintx mbit;
        mbit.lo = mask.lo >> camind;
        mbit.hi = mask.hi >> camind;

        mbit.lo = mbit.lo & isfront.lo;
        mbit.hi = mbit.hi & isfront.hi;

        vstore(mbit, a2);
    }
}

void ScalePoseDiagnalKnl(const AgenCFG *a_JtJ, float lambda, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_JtJ);
    agen_B a1 = init_agen_B_from_cfg(*a_JtJ);

    for (int i = 0; i < niter; i++) chess_loop_range(16, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvfloatx d0 = dvfloat_load(a0);
        d0          = dvmaddf(d0, lambda, d0);
        vstore(d0, a1);
    }
}

void ReductionKnl(const AgenCFG *delta, const AgenCFG *m, float s, int th, int niter)
{
    agen_D a0 = init_agen_D_from_cfg(*delta);
    agen_C a1 = init_agen_C_from_cfg(*m);

    dvfloatx acc0 = chess_dont_care(dvfloatx);
    dvfloatx acc1 = chess_dont_care(dvfloatx);
    dvfloatx acc2 = chess_dont_care(dvfloatx);

    int pred_madd = 0, pred_store = 0, s_ctrl_store = 0, s_ctrl_madd = 0;
    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        s_ctrl_store = mod_inc_pred_z(s_ctrl_store, th, pred_store);

        {
            const dvfloatx d0 = dvfloat_load(a0);

            acc0 = dvmaddf(d0, s, acc0, pred_madd);
            vstore(acc0, a1, pred_store);
        }

        {
            const dvfloatx d0 = dvfloat_load(a0);

            acc1 = dvmaddf(d0, s, acc1, pred_madd);
            vstore(acc1, a1, pred_store);
        }

        {
            const dvfloatx d0 = dvfloat_load(a0);

            acc2 = dvmaddf(d0, s, acc2, pred_madd);
            vstore(acc2, a1, pred_store);
        }

        s_ctrl_madd = mod_inc_pred_nz(s_ctrl_madd, th, pred_madd);
    }
}

void CopyAKnl(const AgenCFG *d0, const AgenCFG *d1, int offset, int niter)
{
    agen a0 = init_agen_from_cfg(*d0);
    adv_agen_base(a0, offset * 4);
    agen a1 = init_agen_from_cfg(*d1);
    adv_agen_base(a1, offset * 4);

    for (int i = 0; i < niter; ++i) chess_loop_range(36, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        dvfloatx dvdata1 = dvfloat_load(a0);
        vstore(dvdata1, a1);
    }
}

void InverseSymmetric3X3Knl(const AgenCFG *Hpp, const AgenCFG *Hpp_inv, const AgenCFG *status, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*Hpp);
    agen_B a1 = init_agen_B_from_cfg(*Hpp_inv);
    agen_C a2 = init_agen_C_from_cfg(*status);

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_prepare_for_pipelining
    {
        const dvfloatx d0 = dvfloat_load(a0);
        const dvfloatx d1 = dvfloat_load(a0);
        const dvfloatx d2 = dvfloat_load(a0);
        dvfloat_load(a0);
        const dvfloatx d4 = dvfloat_load(a0);
        const dvfloatx d5 = dvfloat_load(a0);
        dvfloat_load(a0);
        dvfloat_load(a0);
        const dvfloatx d8 = dvfloat_load(a0);

        // Inverse
        dvfloatx va0, va1, va2, vdet;
        va0.lo = d4.lo * d8.lo;
        va0.hi = d4.hi * d8.hi;

        va1.lo = d2.lo * d5.lo;
        va1.hi = d2.hi * d5.hi;

        va2.lo = d1.lo * d5.lo;
        va2.hi = d1.hi * d5.hi;

        va0 = dvmsubf(d5, d5, va0);

        va1 = dvmsubf(d1, d8, va1);

        va2 = dvmsubf(d2, d4, va2);

        // Determinant
        vdet.lo = d0.lo * va0.lo;
        vdet.hi = d0.hi * va0.hi;

        vdet = dvmaddf(d1, va1, vdet);

        vdet = dvmaddf(d2, va2, vdet);

        // vstatus = 0 flags a singular Hpp (|det| <= 1e-18); its inverse is zeroed below.
        dvintx vstatus;
        vstatus = (vdet > 1e-18) | (vdet < -1e-18);

        vstore(vstatus, a2);

        vdet = dvmux(vstatus == 0, 0.0f, dvfrcp(vdet));

        dvfloatx vInvA0, vInvA1, vInvA2, vInvA4, vInvA5, vInvA8;
        vInvA0.lo = va0.lo * vdet.lo;
        vInvA0.hi = va0.hi * vdet.hi;
        vstore(vInvA0, a1);

        vInvA1.lo = va1.lo * vdet.lo;
        vInvA1.hi = va1.hi * vdet.hi;
        vstore(vInvA1, a1);

        vInvA2.lo = va2.lo * vdet.lo;
        vInvA2.hi = va2.hi * vdet.hi;
        vstore(vInvA2, a1);
        // Hpp is symmetric: element 3 equals element 1.
        vstore(vInvA1, a1);

        vInvA4.lo = d0.lo * d8.lo;
        vInvA4.hi = d0.hi * d8.hi;

        vInvA4 = dvmsubf(d2, d2, vInvA4);

        vInvA4.lo = vInvA4.lo * vdet.lo;
        vInvA4.hi = vInvA4.hi * vdet.hi;

        vstore(vInvA4, a1);

        vInvA5.lo = d1.lo * d2.lo;
        vInvA5.hi = d1.hi * d2.hi;

        vInvA5 = dvmsubf(d0, d5, vInvA5);

        vInvA5.lo = vInvA5.lo * vdet.lo;
        vInvA5.hi = vInvA5.hi * vdet.hi;

        vstore(vInvA5, a1);
        vstore(vInvA2, a1);
        vstore(vInvA5, a1);

        vInvA8.lo = d0.lo * d4.lo;
        vInvA8.hi = d0.hi * d4.hi;

        vInvA8 = dvmsubf(d1, d1, vInvA8);

        vInvA8.lo = vInvA8.lo * vdet.lo;
        vInvA8.hi = vInvA8.hi * vdet.hi;

        vstore(vInvA8, a1);
    }
}

void MatrixMulKnl(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1, int offset2,
                  int th, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_m0);
    adv_agen_base_A(a0, offset0 * 4);
    agen_C a1 = init_agen_C_from_cfg(*a_m1);
    adv_agen_base_C(a1, offset1 * 4);
    agen_D a2 = init_agen_D_from_cfg(*a_m2);
    adv_agen_base_D(a2, offset2 * 4);

    dvfloatx chess_storage(DV0) acc0 = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV1) acc1 = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV2) acc2 = chess_dont_care(dvfloatx);

    int pred_madd = 0, pred_store = 0, s_ctrl_store = 0, s_ctrl_madd = 0;
    for (int i = 0; i < niter / 2; i++) chess_loop_range(3, )
    chess_prepare_for_pipelining
    {
        s_ctrl_store = mod_inc_pred_z(s_ctrl_store, th, pred_store);

        const dvfloatx chess_storage(DV3) d0 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW0) d1 = dvfloat_load(a1);

        acc0 = dvmaddf(d0, d1, acc0, pred_madd);

        vstore(acc0, a2, pred_store);

        const dvfloatx chess_storage(DV4) d2 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW1) d3 = dvfloat_load(a1);

        acc1 = dvmaddf(d2, d3, acc1, pred_madd);

        vstore(acc1, a2, pred_store);

        const dvfloatx chess_storage(DV5) d4 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW2) d5 = dvfloat_load(a1);

        acc2 = dvmaddf(d4, d5, acc2, pred_madd);

        vstore(acc2, a2, pred_store);

        s_ctrl_madd = mod_inc_pred_nz(s_ctrl_madd, th, pred_madd);

        s_ctrl_store = mod_inc_pred_z(s_ctrl_store, th, pred_store);

        const dvfloatx chess_storage(DV9) d6 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW3) d7 = dvfloat_load(a1);

        acc0 = dvmaddf(d6, d7, acc0, pred_madd);

        vstore(acc0, a2, pred_store);

        const dvfloatx chess_storage(DV10) d8 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW4) d9  = dvfloat_load(a1);

        acc1 = dvmaddf(d8, d9, acc1, pred_madd);

        vstore(acc1, a2, pred_store);

        const dvfloatx chess_storage(DV11) d10 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW5) d11  = dvfloat_load(a1);

        acc2 = dvmaddf(d10, d11, acc2, pred_madd);

        vstore(acc2, a2, pred_store);

        s_ctrl_madd = mod_inc_pred_nz(s_ctrl_madd, th, pred_madd);
    }
}

void MatrixMulKnl_332(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                      int offset2, int th, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_m0);
    adv_agen_base_A(a0, offset0 * 4);
    agen_C a1 = init_agen_C_from_cfg(*a_m1);
    adv_agen_base_C(a1, offset1 * 4);
    agen_D a2 = init_agen_D_from_cfg(*a_m2);
    adv_agen_base_D(a2, offset2 * 4);

    dvfloatx chess_storage(DV0) acc0  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV1) acc1  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV2) acc2  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV3) acc3  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV14) acc4 = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV15) acc5 = chess_dont_care(dvfloatx);

    int i;
    int pred_madd = 0, pred_store = 0, s_ctrl_store = 0, s_ctrl_madd = 0;
    for (i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {

        pred_madd = 0;

        const dvfloatx chess_storage(DV6) d0 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW0) d1 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV7) d2 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW1) d3 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV8) d4 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW2) d5 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV9) d6 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW3) d7 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV10) d8 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW4) d9  = dvfloat_load(a1);

        const dvfloatx chess_storage(DV11) d10 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW5) d11  = dvfloat_load(a1);

        const dvfloatx chess_storage(DW6) d13 = dvfloat_load(a1);

        const dvfloatx chess_storage(DW7) d15 = dvfloat_load(a1);

        const dvfloatx chess_storage(DW8) d17 = dvfloat_load(a1);

        acc0 = dvmaddf(d0, d1, acc0, pred_madd);

        acc1 = dvmaddf(d2, d1, acc1, pred_madd);

        acc2 = dvmaddf(d0, d3, acc2, pred_madd);

        acc3 = dvmaddf(d2, d3, acc3, pred_madd);

        acc4 = dvmaddf(d0, d5, acc4, pred_madd);

        acc5 = dvmaddf(d2, d5, acc5, pred_madd);

        pred_madd = 1;
        acc0      = dvmaddf(d4, d7, acc0, pred_madd);

        acc1 = dvmaddf(d6, d7, acc1, pred_madd);

        acc2 = dvmaddf(d4, d9, acc2, pred_madd);

        acc3 = dvmaddf(d6, d9, acc3, pred_madd);

        acc4 = dvmaddf(d4, d11, acc4, pred_madd);

        acc5 = dvmaddf(d6, d11, acc5, pred_madd);

        acc0 = dvmaddf(d8, d13, acc0, pred_madd);
        vstore(acc0, a2);

        acc1 = dvmaddf(d10, d13, acc1, pred_madd);
        vstore(acc1, a2);
        acc2 = dvmaddf(d8, d15, acc2, pred_madd);
        vstore(acc2, a2);
        acc3 = dvmaddf(d10, d15, acc3, pred_madd);
        vstore(acc3, a2);
        acc4 = dvmaddf(d8, d17, acc4, pred_madd);
        vstore(acc4, a2);
        acc5 = dvmaddf(d10, d17, acc5, pred_madd);
        vstore(acc5, a2);
    }
}

void MatrixMulKnl_323(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                      int offset2, int th, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_m0);
    adv_agen_base_A(a0, offset0 * 4);
    agen_A a1 = init_agen_A_from_cfg(*a_m1);
    adv_agen_base_A(a1, offset1 * 4);
    agen_D a2 = init_agen_D_from_cfg(*a_m2);
    adv_agen_base_D(a2, offset2 * 4);

    dvfloatx chess_storage(DV0) acc0  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV1) acc1  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV2) acc2  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV3) acc3  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV4) acc4  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV5) acc5  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV12) acc6 = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV13) acc7 = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV14) acc8 = chess_dont_care(dvfloatx);
    int i;
    int pred_madd = 0, pred_store = 0, s_ctrl_store = 0, s_ctrl_madd = 0;
    for (i = 0; i < niter; i++) chess_loop_range(6, )
    chess_prepare_for_pipelining
    {
        pred_madd = 0;

        const dvfloatx chess_storage(DV6) d0 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW0) d1 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV7) d2 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW1) d3 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV8) d4 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW2) d5 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV9) d6 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW3) d7 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV10) d8 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW4) d9  = dvfloat_load(a1);

        const dvfloatx chess_storage(DV11) d10 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW5) d11  = dvfloat_load(a1);

        acc0 = dvmaddf(d0, d1, acc0, pred_madd);

        acc1 = dvmaddf(d2, d1, acc1, pred_madd);

        acc2 = dvmaddf(d4, d1, acc2, pred_madd);

        acc3 = dvmaddf(d0, d3, acc3, pred_madd);

        acc4 = dvmaddf(d2, d3, acc4, pred_madd);

        acc5 = dvmaddf(d4, d3, acc5, pred_madd);

        acc6 = dvmaddf(d0, d5, acc6, pred_madd);

        acc7 = dvmaddf(d2, d5, acc7, pred_madd);

        acc8 = dvmaddf(d4, d5, acc8, pred_madd);

        pred_madd = 1;
        acc0      = dvmaddf(d6, d7, acc0, pred_madd);

        acc1 = dvmaddf(d8, d7, acc1, pred_madd);

        acc2 = dvmaddf(d10, d7, acc2, pred_madd);

        acc3 = dvmaddf(d6, d9, acc3, pred_madd);

        acc4 = dvmaddf(d8, d9, acc4, pred_madd);

        acc5 = dvmaddf(d10, d9, acc5, pred_madd);

        acc6 = dvmaddf(d6, d11, acc6, pred_madd);

        acc7 = dvmaddf(d8, d11, acc7, pred_madd);

        acc8 = dvmaddf(d10, d11, acc8, pred_madd);

        vstore(acc0, a2);

        vstore(acc1, a2);
        vstore(acc2, a2);
        vstore(acc3, a2);
        vstore(acc4, a2);
        vstore(acc5, a2);

        vstore(acc6, a2);
        vstore(acc7, a2);
        vstore(acc8, a2);
    }
}

void MatrixMulKnl_233(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                      int offset2, int th, int niter)
{
    agen_D a0 = init_agen_D_from_cfg(*a_m0);
    adv_agen_base_D(a0, offset0 * 4);
    agen_C a1 = init_agen_C_from_cfg(*a_m1);
    adv_agen_base_C(a1, offset1 * 4);
    agen_A a2 = init_agen_A_from_cfg(*a_m2);
    adv_agen_base_A(a2, offset2 * 4);

    dvfloatx chess_storage(DV0) acc0  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV1) acc1  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV2) acc2  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV3) acc3  = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV14) acc4 = chess_dont_care(dvfloatx);
    dvfloatx chess_storage(DV15) acc5 = chess_dont_care(dvfloatx);

    int i;
    int pred_madd = 0, pred_store = 0, s_ctrl_store = 0, s_ctrl_madd = 0;
    for (i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        pred_madd = 0;

        const dvfloatx chess_storage(DV4) d0 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW0) d1 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV5) d2 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW1) d3 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV6) d4 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW2) d5 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV7) d6 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW3) d7 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV8) d8 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW4) d9 = dvfloat_load(a1);

        const dvfloatx chess_storage(DV9) d10 = dvfloat_load(a0);
        const dvfloatx chess_storage(DW5) d11 = dvfloat_load(a1);

        const dvfloatx chess_storage(DW6) d13 = dvfloat_load(a1);

        const dvfloatx chess_storage(DW7) d15 = dvfloat_load(a1);

        const dvfloatx chess_storage(DW8) d17 = dvfloat_load(a1);

        acc0 = dvmaddf(d0, d1, acc0, pred_madd);

        acc1 = dvmaddf(d2, d1, acc1, pred_madd);

        acc2 = dvmaddf(d0, d3, acc2, pred_madd);

        acc3 = dvmaddf(d2, d3, acc3, pred_madd);

        acc4 = dvmaddf(d0, d5, acc4, pred_madd);

        acc5 = dvmaddf(d2, d5, acc5, pred_madd);

        pred_madd = 1;
        acc0      = dvmaddf(d4, d7, acc0, pred_madd);

        acc1 = dvmaddf(d6, d7, acc1, pred_madd);

        acc2 = dvmaddf(d4, d9, acc2, pred_madd);

        acc3 = dvmaddf(d6, d9, acc3, pred_madd);

        acc4 = dvmaddf(d4, d11, acc4, pred_madd);

        acc5 = dvmaddf(d6, d11, acc5, pred_madd);

        acc0 = dvmaddf(d8, d13, acc0, pred_madd);
        vstore(acc0, a2);

        acc1 = dvmaddf(d10, d13, acc1, pred_madd);
        vstore(acc1, a2);
        acc2 = dvmaddf(d8, d15, acc2, pred_madd);
        vstore(acc2, a2);
        acc3 = dvmaddf(d10, d15, acc3, pred_madd);
        vstore(acc3, a2);
        acc4 = dvmaddf(d8, d17, acc4, pred_madd);
        vstore(acc4, a2);
        acc5 = dvmaddf(d10, d17, acc5, pred_madd);
        vstore(acc5, a2);
    }
}

int ComputeIndexKnl(int i, int j, int n)
{
    int res = 0;
    for (int k = 0; k < j; ++k)
    {
        res += (n - k);
    }
    res += (i - j);

    return res;
}

void AccumulateMatKnl(float *d0, float *d1, int d, int lp0, int lp1, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            float val = 0.0f;
            for (int k = 0; k < 16; ++k)
            {
                val += d0[i * lp0 + j * 16 + k];
            }
            d1[i + j * lp1] = val;
        }
    }
}

void Convert2MatrixKnl(float *Am, float *Ad, int lp1, int lp2, int r, int c, int m)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = i; j < m; ++j)
        {
            // Packing width must be kActiveCameraCount (as written by BuildReducedSystemStage0/1), not the
            // runtime bound m: otherwise pairs read the wrong slot when active_camera_count < kActiveCameraCount.
            int ind = ComputeIndexKnl(j, i, kActiveCameraCount);

            for (int ci = 0; ci < c; ++ci)
            {
                for (int ri = 0; ri < r; ++ri)
                {
                    Ad[(j * c + ci) * lp2 + (i * r + ri)] = Am[ind + (ri + ci * r) * lp1];
                }
            }
        }
    }

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            for (int ci = 0; ci < c; ++ci)
            {
                for (int ri = 0; ri < r; ++ri)
                {
                    Ad[(j * c + ci) * lp2 + (i * r + ri)] = Ad[(i * c + ri) * lp2 + (j * r + ci)];
                }
            }
        }
    }
}

void Convert2MatrixBKnl(float *Bm, float *Bd, int lp1, int d, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            Bd[i * d + j] = Bm[i + j * lp1];
        }
    }
}

void CholeskyDecompositionKnl(float *A, float *L, int lp, int n)
{
    for (int i = 0; i < n; ++i) chess_loop_range(2, )
    chess_prepare_for_pipelining
    {
        for (int j = 0; j < (i + 1); ++j)
        {
            float val = A[j * lp + i];
            for (int k = 0; k < j; ++k) chess_prepare_for_pipelining
            {
                float l1 = L[k * lp + i];
                float l2 = L[k * lp + j];
                val      = fmsub(l1, l2, val);
            }

            if (i == j)
            {
                // Clamp instead of sqrt(negative) -> NaN: an unanchored (gauge-free) camera window can make A
                // near-singular.
                L[j * lp + i] = fsqrt((val > 1e-12f) ? val : 1e-12f);
            }
            else
            {
                L[j * lp + i] = frcp(L[j * lp + j]) * val;
            }
        }
    }
}

void ForwardSubstitutionKnl(float *L, float *B, float *x, int lp, int n)
{
    for (int i = 0; i < n; ++i)
    {
        float val = B[i];
        for (int j = 0; j < i; ++j) chess_prepare_for_pipelining
        {
            val = fmsub(L[j * lp + i], x[j], val);
        }

        x[i] = frcp(L[i * lp + i]);

        x[i] *= val;
    }
}

void BackwardSubstitutionKnl(float *L, float *B, float *x, int lp, int n)
{
    for (int i = n - 1; i >= 0; --i)
    {
        float val = B[i];
        for (int j = i + 1; j < n; ++j) chess_prepare_for_pipelining
        {
            val = fmsub(L[i * lp + j], x[j], val);
        }

        x[i] = frcp(L[i * lp + i]);
        x[i] *= val;
    }
}

void MatrixMulV2Knl(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                    int offset2, int th, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*a_m0);
    adv_agen_base_A(a0, offset0 * 4);
    agen_B a1 = init_agen_B_from_cfg(*a_m1);
    adv_agen_base_B(a1, offset1 * 4);
    agen_C a2 = init_agen_C_from_cfg(*a_m2);
    adv_agen_base_C(a2, offset2 * 4);

    dvfloatx acc0 = chess_dont_care(dvfloatx);
    dvfloatx acc1 = chess_dont_care(dvfloatx);
    dvfloatx acc2 = chess_dont_care(dvfloatx);

    int pred_madd = 0, pred_store = 0, s_ctrl_store = 0, s_ctrl_madd = 0;
    for (int i = 0; i < niter; i++) chess_loop_range(16, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        s_ctrl_store = mod_inc_pred_z(s_ctrl_store, th, pred_store);

        {
            const vfloatx d0  = vfloat_load_ws(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc0.lo = vmaddf(d0, d1.lo, acc0.lo, pred_madd);
            acc0.hi = vmaddf(d0, d1.hi, acc0.hi, pred_madd);

            vstore(acc0, a2, pred_store);
        }

        {
            const vfloatx d0  = vfloat_load_ws(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc1.lo = vmaddf(d0, d1.lo, acc1.lo, pred_madd);
            acc1.hi = vmaddf(d0, d1.hi, acc1.hi, pred_madd);

            vstore(acc1, a2, pred_store);
        }

        {
            const vfloatx d0  = vfloat_load_ws(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc2.lo = vmaddf(d0, d1.lo, acc2.lo, pred_madd);
            acc2.hi = vmaddf(d0, d1.hi, acc2.hi, pred_madd);

            vstore(acc2, a2, pred_store);
        }

        s_ctrl_madd = mod_inc_pred_nz(s_ctrl_madd, th, pred_madd);
    }
}

void ApplyPointUpdateKnl(const AgenCFG *pold, const AgenCFG *dp, const AgenCFG *bitmask, const AgenCFG *pnew, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*pold);
    agen_B a1 = init_agen_B_from_cfg(*dp);
    agen_C a2 = init_agen_C_from_cfg(*pnew);
    agen_D a3 = init_agen_D_from_cfg(*bitmask);

    dvfloatx zeros;
    zeros.lo = replicatef(0.0f);
    zeros.hi = zeros.lo;

    for (int i = 0; i < niter; i++) chess_loop_range(24, )
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvfloatx chess_storage(DV) d0 = dvfloat_load(a0);
        dvfloatx chess_storage(DW) d1 = dvfloat_load(a1);
        dvintx dmask                  = dvint_load(a3);

        dvintx mbit;
        mbit.lo = dmask.lo >> 31;
        mbit.hi = dmask.hi >> 31;

        mbit.lo = mbit.lo & 1;
        mbit.hi = mbit.hi & 1;

        dvfloatx chess_storage(DV) d2;

        // mbit = bitmask bit 31: 1 = active track (takes delta), 0 = fixed (takes zero).
        d2 = dvmux(mbit, d1, zeros);

        d2.lo = d0.lo + d2.lo;
        d2.hi = d0.hi + d2.hi;

        vstore(d2, a2);
    }
}

void UpdateWithDeltaKnl(const AgenCFG *m0, const AgenCFG *delta, const AgenCFG *m1, float s, int offset, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*m0);
    adv_agen_base_A(a0, offset * 4);
    agen_C a1 = init_agen_C_from_cfg(*delta);
    agen_A a2 = init_agen_A_from_cfg(*m1);
    adv_agen_base_A(a2, offset * 4);

    for (int i = 0; i < niter; i++) chess_loop_range(6, )
    chess_unroll_loop(6)
    chess_prepare_for_pipelining
    {
        const dvfloatx d0 = dvfloat_load(a0);
        const dvfloatx d1 = dvfloat_load(a1);

        const dvfloatx acc0 = dvmaddf(d1, s, d0);

        vstore(acc0, a2);
    }
}

void SetHomogeneousKnl(const AgenCFG *pnew, int niter)
{
    agen_A a0 = init_agen_A_from_cfg(*pnew);

    dvfloatx ones;
    ones.lo = replicatef(1.0f);
    ones.hi = ones.lo;
    for (int i = 0; i < niter; i++) chess_prepare_for_pipelining
    {
        vstore(ones, a0);
    }
}

void Convert2SE3Knl(float *dC, float *pose, int lp, int n)
{
    for (int i = 0; i < n; ++i)
    {
        float u0 = dC[0 + i * 6], u1 = dC[1 + i * 6], u2 = dC[2 + i * 6];
        float w0 = dC[3 + i * 6], w1 = dC[4 + i * 6], w2 = dC[5 + i * 6];

        float theta_square = w0 * w0;
        theta_square       = fmadd(w1, w1, theta_square);
        theta_square       = fmadd(w2, w2, theta_square);

        float A, B, C;
        if (theta_square < 1.0e-4f)
        {
            A = 1.0f - theta_square * kFac;
            B = 0.5f - theta_square * kFac1;
            C = kFac - theta_square * kFac2;
        }
        else
        {
            float theta        = fsqrt(theta_square);
            float theta_inv    = frcp(theta);
            float theta_inv_sq = theta_inv * theta_inv;
            float theta_norm   = theta * kPiNorm;

            A = theta_inv * fsin(theta_norm);
            B = theta < kPiHalf ? A * A * frcp(1 + fcos(theta_norm)) : theta_inv_sq * (1.0f - fcos(theta_norm));
            C = theta_inv_sq * (1.0f - A);
        }

        {
            pose[i]           = 1.0f - B * (w1 * w1 + w2 * w2);
            pose[i + 5 * lp]  = 1.0f - B * (w0 * w0 + w2 * w2);
            pose[i + 10 * lp] = 1.0f - B * (w0 * w0 + w1 * w1);

            pose[i + 4 * lp] = B * w0 * w1 - A * w2;
            pose[i + lp]     = B * w0 * w1 + A * w2;

            pose[i + 8 * lp] = B * w0 * w2 + A * w1;
            pose[i + 2 * lp] = B * w0 * w2 - A * w1;

            pose[i + 9 * lp] = B * w1 * w2 - A * w0;
            pose[i + 6 * lp] = B * w1 * w2 + A * w0;
        }

        {
            float wxu0 = w1 * u2 - w2 * u1;
            float wxu1 = w2 * u0 - w0 * u2;
            float wxu2 = w0 * u1 - w1 * u0;

            float wwxu0 = w1 * wxu2 - w2 * wxu1;
            float wwxu1 = w2 * wxu0 - w0 * wxu2;
            float wwxu2 = w0 * wxu1 - w1 * wxu0;

            wwxu0 = C * wwxu0;
            wwxu1 = C * wwxu1;
            wwxu2 = C * wwxu2;

            wxu0 = B * wxu0;
            wxu1 = B * wxu1;
            wxu2 = B * wxu2;

            pose[i + 12 * lp] = u0 + wxu0 + wwxu0;
            pose[i + 13 * lp] = u1 + wxu1 + wwxu1;
            pose[i + 14 * lp] = u2 + wxu2 + wwxu2;
            pose[i + 15 * lp] = 1.0f;
        }
    }
}

void UpdatePosesKnl(float *delta, float *old_pose, float *new_pose, int lp, int n)
{
    int vecw  = pva_elementsof(dvfloatx);
    int niter = (n + vecw - 1) / vecw;

    AgenWrapper w0, w1, w2, w3;
    agen a0, a1, a2, a3;

    a0      = init((float *)delta);
    w0.size = sizeof(float);
    w0.n1   = 3;
    w0.s1   = lp;
    w0.n2   = 4;
    w0.s2   = 4 * lp;
    w0.n3   = 4;
    w0.s3   = 0;
    w0.n4   = niter;
    w0.s4   = vecw;
    INIT_AGEN4(a0, w0);

    a1      = init((float *)old_pose);
    w1.size = sizeof(float);
    w1.n1   = 3;
    w1.s1   = 0;
    w1.n2   = 4;
    w1.s2   = lp;
    w1.n3   = 4;
    w1.s3   = 4 * lp;
    w1.n4   = niter;
    w1.s4   = vecw;
    INIT_AGEN4(a1, w1);

    a2      = init((float *)new_pose);
    w2.size = sizeof(float);
    w2.n1   = 3;
    w2.s1   = lp;
    w2.n2   = 4;
    w2.s2   = 0;
    w2.n3   = 4;
    w2.s3   = 4 * lp;
    w2.n4   = niter;
    w2.s4   = vecw;
    INIT_AGEN4(a2, w2);

    a3      = init((float *)(new_pose + 15 * lp));
    w3.size = sizeof(float);
    w3.n1   = niter;
    w3.s1   = vecw;
    INIT_AGEN1(a3, w3);

    dvfloatx acc0 = chess_dont_care(dvfloatx);
    dvfloatx acc1 = chess_dont_care(dvfloatx);
    dvfloatx acc2 = chess_dont_care(dvfloatx);

    int pred_madd = 0, pred_store = 0, s_ctrl_store = 1, s_ctrl_madd = 0;
    for (int i = 0; i < niter * 16; i++) chess_loop_range(16, )
    chess_unroll_loop(2)
    chess_prepare_for_pipelining
    {
        {
            const dvfloatx d0 = dvfloat_load(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc0 = dvmaddf(d1, d0, acc0, pred_madd);

            vstore(acc0, a2, pred_store);
        }

        {
            const dvfloatx d0 = dvfloat_load(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc1 = dvmaddf(d1, d0, acc1, pred_madd);

            vstore(acc1, a2, pred_store);
        }

        {
            const dvfloatx d0 = dvfloat_load(a0);
            const dvfloatx d1 = dvfloat_load(a1);

            acc2 = dvmaddf(d1, d0, acc2, pred_madd);

            vstore(acc2, a2, pred_store);
        }

        s_ctrl_madd  = mod_inc_pred_nz(s_ctrl_madd, 4 - 1, pred_madd);
        s_ctrl_store = mod_inc_pred_z(s_ctrl_store, 4 - 1, pred_store);
    }

    dvfloatx ones;
    ones.lo = replicatef(1.0f);
    ones.hi = ones.lo;

    for (int i = 0; i < niter; i++)
    {
        vstore(ones, a3);
    }
}

void ComputeCostKnl(float *cost_in, const AgenCFG *residual, float scale, float *cost_out, int niter)
{
    agen a0 = init_agen_from_cfg(*residual);

    dvfloatx dvcost = sign_extend(*((dvfloat *)(cost_in)));

    dvfloatx ones;
    ones.lo = replicatef(1.0f);
    ones.hi = replicatef(1.0f);

    float scale_inv = frcp(scale);
    // Reference cost is scale*ln(sum); dvflog2() is base-2, so scale by ln(2) here.
    float scale_ln2 = 0.6931471805599453f * scale;

    for (int i = 0; i < niter; i++) chess_loop_range(8, )
    chess_unroll_loop(4)
    chess_prepare_for_pipelining
    {
        const dvfloatx d0 = dvfloat_load(a0);
        dvfloatx square_error;
        square_error = d0 * d0;

        const dvfloatx d1 = dvfloat_load(a0);
        square_error      = dvmaddf(d1, d1, square_error, -1);

        dvfloatx sum;
        sum = dvmaddf(square_error, scale_inv, ones, -1);

        dvfloatx val       = dvflog2(sum);
        dvfloatx curr_cost = val * scale_ln2;
        dvcost             = dvcost + curr_cost;
    }

    *((dvfloat *restrict)cost_out) = extract(dvcost);
}
