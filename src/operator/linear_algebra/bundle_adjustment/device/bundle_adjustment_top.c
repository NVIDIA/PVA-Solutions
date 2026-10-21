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
#include <cupva_device/sequence_dataflow.h>

VMEM(A, int32_t, track_iter);
VMEM(A, int32_t, camera_count);
VMEM(A, float, min_distance);
VMEM(A, float, scale_meas);
VMEM(A, float, rinv_meas);
VMEM(A, int32_t, active_camera_count);
VMEM(A, int32_t, active_camera_begin);
VMEM(A, float, lambdaInit);
VMEM(A, int32_t, maxIterations);

// LM state; pointsCurBase/pointsTentBase swap on an accepted step.
VMEM(A, float, lambda);
VMEM(A, float, total_cost);
VMEM(A, int64_t, pointsCurBase);
VMEM(A, int64_t, pointsTentBase);
VMEM(A, int32_t, iterationsUsed);

VMEM(A, ParamBundlerType, params_bundler, 1);

// DRAM pointers bound at Submit; the four *Scratch pointers are device-owned.
VMEM_POINTER(A, rigPosesExtPtr);
VMEM_POINTER(A, rig2camExtPtr);
VMEM_POINTER(A, points3dExtPtr);
VMEM_POINTER(A, measurementsExtPtr);
VMEM_POINTER(A, cameraBitmaskExtPtr);
VMEM_POINTER(A, tpcScratchExtPtr);
VMEM_POINTER(A, tpScratchExtPtr);
VMEM_POINTER(A, points3dScratchExtPtr);
VMEM_POINTER(A, points3dScratch2ExtPtr);
VMEM_POINTER(A, updatedRigPosesExtPtr);
VMEM_POINTER(A, updatedPoints3dExtPtr);
VMEM_POINTER(A, finalCostExtPtr);
VMEM_POINTER(A, iterationsUsedExtPtr);

// One-shot input: rig poses idx 0, rig2cam idx 1 (one handler, the DMA trigger pool is small).
VMEM_SEQDF_HANDLER(A, oneShotInputHdl);
VMEM_SEQDF_HANDLER(A, points3dHdl);
// Per-tile handlers: measurements, cameraBitmask and tpcRead are ping-ponged; tp/Tpc scratch is not.
VMEM_SEQDF_HANDLER(A, measurementsHdl);
VMEM_SEQDF_HANDLER(A, cameraBitmaskHdl);
VMEM_SEQDF_HANDLER(A, tpcWriteHdl);
VMEM_SEQDF_HANDLER(A, tpcReadHdl);
// A handler has a fixed direction, so tp scratch needs separate store and load handlers.
VMEM_SEQDF_HANDLER(A, tpWriteHdl);
// tpReadHdl and pnewCommit*Hdl are in bank B: bank A is near its VmemModel capacity.
VMEM_SEQDF_HANDLER(B, tpReadHdl);
// LM point state: SoA tile store/load; pnewCommitStoreHdl drains the padded AoS stage to updatedPoints3d.
VMEM_SEQDF_HANDLER(A, pnewTentativeOutHdl);
VMEM_SEQDF_HANDLER(B, pnewCommitLoadHdl);
VMEM_SEQDF_HANDLER(B, pnewCommitStoreHdl);
// One-shot output: updated rig poses idx 0, final cost idx 1, iterations used idx 2.
VMEM_SEQDF_HANDLER(A, oneShotOutputHdl);

// AgenConfigInitKnl() (bundle_adjustment_knl.c) bakes the byte layout of these buffers: do not
// resize, move to another bank or rename them without updating it.
VMEM(A, float, Hpp0, 9 * kTrackCountTile);
VMEM(B, float, Hpp1, 9 * kTrackCountTile);
VMEM(A, float, bp0, 3 * kTrackCountTile);
VMEM(B, float, bp1, 3 * kTrackCountTile);
VMEM(D, float, Hpc, kHpcHeight *kTrackCountTile);

VMEM(B, float, trans_poses, kCameraBlockSize * 16);
VMEM(B, float, rig_poses, kCameraBlockSize * 16);
VMEM(A, float, points3d, 8 * kTrackCountTile);
VMEM(C, float, points3d_cam, 4 * kTrackCountTile);
VMEM(C, float, points3d_rig, 4 * kTrackCountTile);

VMEM(B, float, residual0, 2 * kTrackCountTile);
VMEM(B, int32_t, infront, kTrackCountTile);
VMEM(A, int32_t, camera_bitmask, 2 * kTrackCountTile);
VMEM(C, int32_t, valid_flag, kTrackCountTile);

VMEM(A, float, weight0, kTrackCountTile);
VMEM(C, float, weight1, kTrackCountTile);

VMEM(A, float, Jp0, 6 * kTrackCountTile);
VMEM(B, float, Jp1, 6 * kTrackCountTile);
VMEM(C, float, JpTJp, 9 * kTrackCountTile);
VMEM(C, float, JpTE, 3 * kTrackCountTile);

VMEM(A, float, rig2cam, 16);
VMEM(B, float, Jc0, 12 * kTrackCountTile);
VMEM(D, float, Jc1, 12 * kTrackCountTile);
VMEM(C, float, JcTJc, 36 * kTrackCountTile);

VMEM(A, float, MatA0, 36 * kWordWidth * kSymmetricCameraSize);
VMEM(C, float, JcTE, 6 * kTrackCountTile);
VMEM(A, float, MatB0, 6 * kWordWidth * kActiveCameraCount);

VMEM(C, float, Hpp_inv, 9 * kTrackCountTile);
VMEM(C, int32_t, statuses, kTrackCountTile);
VMEM(A, float, tp, 3 * kTrackCountTile);

VMEM(C, float, HpcTp, 6 * kTrackCountTile);
VMEM(A, float, Tpc, 18 * kTrackCountTile);
VMEM(D, float, TpcHpc, 36 * kTrackCountTile);

VMEM(C, float, LTx, kActiveCameraCount * 6);

VMEM(C, float, dp, 3 * kTrackCountTile);
VMEM(B, float, dp1, 3 * kTrackCountTile);
VMEM(C, float, pnew, 8 * kTrackCountTile);
// dPose is AgenConfigInitKnl()-baked pose-update scratch, not the transpose staging buffer
// (rig_pose_stage); do not resize or repurpose it.
VMEM(C, float, dPose, 16 * kCameraBlockSize);
VMEM(A, float, update_poses, 16 * kCameraBlockSize);

VMEM(A, float, residual1, kTrackCountTile * 2);
VMEM(B, float, cost1, kWordWidth);
VMEM(C, float, cost2, kWordWidth);

VMEM(C, float, buf_c, kWordWidth * 36);

// AoS staging for the transposing load/store, 16k+1 line pitch (must match the src()/dst() pitches
// in OpBundleAdjustment.cpp); not part of AgenConfigInitKnl()'s layout.
// ImportPoints3d()/ExportPoints3d() use only the first tile-sized half.
VMEM(B, float, points3d_stage, 2 * kPoints3dStagePitch * kTrackCountTile);
// Ping-ponged; ComputeResidualKnl reads it directly via a transposing load (cfgi=10/11).
VMEM(B, float, measurements_stage, 2 * kMeasurementsStagePitch * kTrackCountTile);
VMEM(B, float, rig_pose_stage, kRigPoseStagePitch *kCameraBlockSize);

void ParamInit(ParamBundlerType *params);
void BuildReducedSystem(int32_t iter);
void BuildReducedSystemStage0(int ping_pong_pt);
void BuildReducedSystemStage1(int i);
void ComputeUpdate(int32_t iter);
void ApplyUpdate(int32_t iter);
static float ComputeInitialCost(void);

// Find dC for poses and dP for points: reduced camera system A*dC = B (Aij 6x6 blocks, c cameras).
//   Aij = Hcc - Hpc^T*Hpp^-1*Hpc = Hcc - Tpc*Hpc,  Tpc = Hpc^T*Hpp^-1,  Hpp = Jp^T*Jp, Hcc = Jc^T*Jc
//   Bi  = -JcTE - Hpc^T*tp,  tp = Hpp^-1*bp,  JcTE = Jc^T*residual,  JpTE = Jp^T*residual
// Point update: dP = Hpp^-1*bP - Hpp^-1*Hpc*dC = tp - Tpc^T*dC

// Public tensors are track-major AoS; kernels want SoA (element k of track t at
// k*kTrackCountTile+t) so one vector load covers 16 tracks of one component.
//
// The transposing load/store (dvfloat_load_transp/vstore_transp) needs a word line pitch of 16k+1
// (k in agen.lane_ofst), which a tight AoS row never has. A strided DMA therefore fills a padded
// staging buffer (k*StagePitch) and the transpose happens in the vector addressing:
//   - points3d: converted once by ImportPoints3d()/ExportPoints3d(); the LM loop stays SoA.
//   - rig poses: converted once on load and once on store.
//   - measurements: read directly from staging by ComputeResidualKnl() (cfgi=10/11).
//
// The per-tile DMA helpers land tiles in the ping/pong half AgenConfigInitKnl()'s cfgs expect.

// Contiguous 16-lane stream agen over numGroups groups.
static inline agen BundleAgenDvfloat(float *base, int32_t numGroups)
{
    AgenWrapper w;
    w.size      = (int32_t)sizeof(float);
    w.n1        = numGroups;
    w.s1        = 16;
    w.n2        = 1;
    w.s2        = 0;
    agen a      = init(base);
    a.lane_ofst = 0; // must be 0 for a non-transposing ld/st
    INIT_AGEN2(a, w);
    return a;
}

// Transposing agen at column `col` of a buffer with pitch `pitchElems` (= 16k+1); each step covers
// the next 16 rows.
static inline agen BundleAgenTranspColumn(float *paddedBase, int32_t pitchElems, int32_t col, int32_t numGroups)
{
    AgenWrapper w;
    w.size      = (int32_t)sizeof(float);
    w.n1        = numGroups;
    w.s1        = 16 * pitchElems;
    w.n2        = 1;
    w.s2        = 0;
    agen a      = init(paddedBase + col);
    a.lane_ofst = (pitchElems - 1) / 16;
    INIT_AGEN2(a, w);
    return a;
}

// Gathers column `col` of numGroups*16 padded rows into the contiguous SoA plane `dst`.
static inline void TranspLoadColumn(float *paddedBase, int32_t pitchElems, int32_t col, int32_t numGroups, float *dst)
{
    agen aT   = BundleAgenTranspColumn(paddedBase, pitchElems, col, numGroups);
    agen aOut = BundleAgenDvfloat(dst, numGroups);

    for (int32_t g = 0; g < numGroups; ++g)
    {
        vstore(dvfloat_load_transp(aT), aOut);
    }
}

// Inverse of TranspLoadColumn().
static inline void TranspStoreColumn(float *src, float *paddedBase, int32_t pitchElems, int32_t col, int32_t numGroups)
{
    agen aIn = BundleAgenDvfloat(src, numGroups);
    agen aT  = BundleAgenTranspColumn(paddedBase, pitchElems, col, numGroups);

    for (int32_t g = 0; g < numGroups; ++g)
    {
        vstore_transp(dvfloat_load(aIn), aT);
    }
}

// 4 components (x,y,z,w); kTrackCountTile is a multiple of 16.
static inline void TransposePoints3dAosToSoa(float *aos_padded, float *soa, int32_t n)
{
    (void)n; // tiles are always full
    const int32_t numGroups = kTrackCountTile / 16;
    for (int32_t k = 0; k < 4; ++k)
    {
        TranspLoadColumn(aos_padded, kPoints3dStagePitch, k, numGroups, &soa[k * kTrackCountTile]);
    }
}

// Inverse of TransposePoints3dAosToSoa().
static inline void TransposePoints3dSoaToAos(float *soa, float *aos_padded, int32_t n)
{
    (void)n;
    const int32_t numGroups = kTrackCountTile / 16;
    for (int32_t k = 0; k < 4; ++k)
    {
        TranspStoreColumn(&soa[k * kTrackCountTile], aos_padded, kPoints3dStagePitch, k, numGroups);
    }
}

// destBuf: Stage0 reads `points3d` (cfgi=5/6); ApplyUpdate() loads into `pnew` because its cfgs read
// pold from pnew_ptr (cfgi=100/101) and write points3d_ptr (cfgi=102/103), swapping buffer roles.
void issuePoints3dTile(int32_t tileIdx, int32_t pingPong, float *destBuf)
{
    // SoA scratch: one contiguous line per tile, no staging or transpose.
    const int64_t stride = (int64_t)(4 * kTrackCountTile) * sizeof(float);
    float *dst           = pingPong ? &destBuf[4 * kTrackCountTile] : &destBuf[0];

    cupvaSQDFUpdateAddr(pnewCommitLoadHdl, 0, pointsCurBase + tileIdx * stride, MEMTYPE_DRAM,
                        cupvaGetVmemAddress((void *)dst), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(pnewCommitLoadHdl);
}

static inline void syncPoints3dTile(int32_t pingPong, float *destBuf)
{
    cupvaSQDFSync(pnewCommitLoadHdl);
}

// Plain ping-ponged DMA; ComputeResidualKnl transposes on read.
static inline void loadMeasurementsTile(int32_t tileIdx, int32_t pingPong)
{
    const int64_t base   = measurementsExtPtr.base + measurementsExtPtr.offset;
    const int64_t stride = (int64_t)(2 * kMaxCameraCount * kTrackCountTile) * sizeof(float);
    float *dst = pingPong ? &measurements_stage[kMeasurementsStagePitch * kTrackCountTile] : &measurements_stage[0];

    cupvaSQDFUpdateAddr(measurementsHdl, 0, base + tileIdx * stride, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)dst),
                        MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(measurementsHdl);
}

static inline void loadCameraBitmaskTile(int32_t tileIdx, int32_t pingPong)
{
    const int64_t base   = cameraBitmaskExtPtr.base + cameraBitmaskExtPtr.offset;
    const int64_t stride = (int64_t)kTrackCountTile * sizeof(int32_t);
    int32_t *dst         = pingPong ? &camera_bitmask[kTrackCountTile] : &camera_bitmask[0];

    cupvaSQDFUpdateAddr(cameraBitmaskHdl, 0, base + tileIdx * stride, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)dst),
                        MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(cameraBitmaskHdl);
}

// Async triggers; waited on by syncStage0TileLoads().
void issueStage0TileLoads(int32_t tileIdx, int32_t pingPong, int32_t iter)
{
    (void)iter;
    issuePoints3dTile(tileIdx, pingPong, &points3d[0]);
    loadMeasurementsTile(tileIdx, pingPong);
    loadCameraBitmaskTile(tileIdx, pingPong);
}

static inline void syncStage0TileLoads(int32_t pingPong)
{
    syncPoints3dTile(pingPong, &points3d[0]);
    cupvaSQDFSync(measurementsHdl);
    cupvaSQDFSync(cameraBitmaskHdl);
}

// Stores camera camIdx's 6x3 Tpc block to scratch. Single-buffered: sync the previous store before
// overwriting `Tpc`.
static inline void storeTpcCamera(int32_t tileIdx, int32_t camIdx)
{
    const int64_t base   = tpcScratchExtPtr.base + tpcScratchExtPtr.offset;
    const int64_t stride = (int64_t)(18 * kTrackCountTile) * sizeof(float);
    const int64_t off    = (int64_t)(tileIdx * active_camera_count + camIdx) * stride;

    cupvaSQDFUpdateAddr(tpcWriteHdl, 0, cupvaGetVmemAddress((void *)&Tpc[0]), MEMTYPE_VMEM, base + off, MEMTYPE_DRAM);
    cupvaSQDFFlushAndTrig(tpcWriteHdl);
}

// Single-buffered, like storeTpcCamera().
static inline void storeTpTile(int32_t tileIdx)
{
    const int64_t base   = tpScratchExtPtr.base + tpScratchExtPtr.offset;
    const int64_t stride = (int64_t)(3 * kTrackCountTile) * sizeof(float);

    cupvaSQDFUpdateAddr(tpWriteHdl, 0, cupvaGetVmemAddress((void *)&tp[0]), MEMTYPE_VMEM, base + tileIdx * stride,
                        MEMTYPE_DRAM);
    cupvaSQDFFlushAndTrig(tpWriteHdl);
}

// Bulk-loads all Tpc blocks of a tile into the ping-ponged Hpc region ApplyUpdate()'s tpc_cfg expects.
static inline void loadTpcTileBulk(int32_t tileIdx, int32_t pingPong)
{
    const int64_t base   = tpcScratchExtPtr.base + tpcScratchExtPtr.offset;
    const int64_t stride = (int64_t)(18 * active_camera_count * kTrackCountTile) * sizeof(float);
    float *dst           = pingPong ? &Hpc[18 * kActiveCameraCount * kTrackCountTile] : &Hpc[0];

    cupvaSQDFUpdateAddr(tpcReadHdl, 0, base + tileIdx * stride, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)dst),
                        MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(tpcReadHdl);
}

static inline void loadTpTile(int32_t tileIdx)
{
    const int64_t base   = tpScratchExtPtr.base + tpScratchExtPtr.offset;
    const int64_t stride = (int64_t)(3 * kTrackCountTile) * sizeof(float);

    cupvaSQDFUpdateAddr(tpReadHdl, 0, base + tileIdx * stride, MEMTYPE_DRAM, cupvaGetVmemAddress((void *)&tp[0]),
                        MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(tpReadHdl);
}

// Loads into `pnew`, not `points3d` (see issuePoints3dTile()). Excludes loadTpTile(): `tp` is
// single-buffered and still in use when tile+1 is prefetched (see ApplyUpdate()).
void issueApplyUpdateTileLoads(int32_t tileIdx, int32_t pingPong, int32_t iter)
{
    (void)iter;
    issuePoints3dTile(tileIdx, pingPong, &pnew[0]);
    loadMeasurementsTile(tileIdx, pingPong);
    loadCameraBitmaskTile(tileIdx, pingPong);
    loadTpcTileBulk(tileIdx, pingPong);
}

// tp is synced separately (see the tp prefetch in ApplyUpdate()).
static inline void syncApplyUpdateTileLoads(int32_t pingPong)
{
    syncPoints3dTile(pingPong, &pnew[0]);
    cupvaSQDFSync(measurementsHdl);
    cupvaSQDFSync(cameraBitmaskHdl);
    cupvaSQDFSync(tpcReadHdl);
}

// Sources from `points3d`, not `pnew`: ApplyPointUpdateKnl()'s cfgs write the refined point to
// points3d_ptr.
static inline void storePnewTile(int32_t tileIdx, int32_t pingPong)
{
    const int64_t base   = pointsTentBase;
    const int64_t stride = (int64_t)(4 * kTrackCountTile) * sizeof(float);
    float *src           = pingPong ? &points3d[4 * kTrackCountTile] : &points3d[0];

    cupvaSQDFUpdateAddr(pnewTentativeOutHdl, 0, cupvaGetVmemAddress((void *)src), MEMTYPE_VMEM, base + tileIdx * stride,
                        MEMTYPE_DRAM);
    cupvaSQDFFlushAndTrig(pnewTentativeOutHdl);
}

// Writes the final SoA point state to the public AoS tensor; the only SoA->AoS conversion, since an
// accepted step just swaps buffers. Reproduces the input points if no step was accepted.
static void ExportPoints3d(void)
{
    const int64_t outBase = updatedPoints3dExtPtr.base + updatedPoints3dExtPtr.offset;
    const int64_t stride  = (int64_t)(4 * kTrackCountTile) * sizeof(float);
    float *staging        = &pnew[0]; // idle here

    for (int32_t i = 0; i < track_iter; ++i)
    {
        cupvaSQDFUpdateAddr(pnewCommitLoadHdl, 0, pointsCurBase + i * stride, MEMTYPE_DRAM,
                            cupvaGetVmemAddress((void *)staging), MEMTYPE_VMEM);
        cupvaSQDFFlushAndTrig(pnewCommitLoadHdl);
        cupvaSQDFSync(pnewCommitLoadHdl);

        TransposePoints3dSoaToAos(staging, &points3d_stage[0], kTrackCountTile);

        cupvaSQDFUpdateAddr(pnewCommitStoreHdl, 0, cupvaGetVmemAddress((void *)&points3d_stage[0]), MEMTYPE_VMEM,
                            outBase + i * stride, MEMTYPE_DRAM);
        cupvaSQDFFlushAndTrig(pnewCommitStoreHdl);
        cupvaSQDFSync(pnewCommitStoreHdl);
    }
}

// One-time AoS->SoA copy of the input points into scratch buffer 0 so the LM loop stays SoA.
static void ImportPoints3d(void)
{
    const int64_t inBase = points3dExtPtr.base + points3dExtPtr.offset;
    const int64_t stride = (int64_t)(4 * kTrackCountTile) * sizeof(float);

    for (int32_t i = 0; i < track_iter; ++i)
    {
        cupvaSQDFUpdateAddr(points3dHdl, 0, inBase + i * stride, MEMTYPE_DRAM,
                            cupvaGetVmemAddress((void *)&points3d_stage[0]), MEMTYPE_VMEM);
        cupvaSQDFFlushAndTrig(points3dHdl);
        cupvaSQDFSync(points3dHdl);

        TransposePoints3dAosToSoa(&points3d_stage[0], &pnew[0], kTrackCountTile);

        // Packed SoA store; pnewCommitStoreHdl would sample at the padded 17-float row pitch.
        cupvaSQDFUpdateAddr(pnewTentativeOutHdl, 0, cupvaGetVmemAddress((void *)&pnew[0]), MEMTYPE_VMEM,
                            pointsCurBase + i * stride, MEMTYPE_DRAM);
        cupvaSQDFFlushAndTrig(pnewTentativeOutHdl);
        cupvaSQDFSync(pnewTentativeOutHdl);
    }
}

// Pose layout at the DRAM boundary: public poses are camera-major AoS, column-vector 4x4 (pose[c][k] at
// c*16+k); kernels want camera-minor SoA (k*kCameraBlockSize+c) and the transposed, row-vector matrix
// (M_knl[r][c] = M_pub[c][r]). Reuses TranspLoad/StoreColumn() with the column index transposed.
static inline void TransposeRigPosesAosToSoa(float *aos_padded, float *soa, int32_t n)
{
    const int32_t numGroups = (kCameraBlockSize + 15) / 16;
    for (int32_t r = 0; r < 4; ++r)
    {
        for (int32_t col = 0; col < 4; ++col)
        {
            const int32_t srcCol  = col * 4 + r;
            const int32_t dstComp = r * 4 + col;
            TranspLoadColumn(aos_padded, kRigPoseStagePitch, srcCol, numGroups, &soa[dstComp * kCameraBlockSize]);
        }
    }
    // Lanes past camera_count (padding or stale VMEM) can decode as NaN/Inf and flow into full-width
    // pose ops; zero them up to kCameraBlockSize (a superset of the range SoaToAos re-zeroes).
    for (int32_t comp = 0; comp < 16; ++comp)
    {
        for (int32_t c = n; c < kCameraBlockSize; ++c)
        {
            soa[comp * kCameraBlockSize + c] = 0.0f;
        }
    }
}

static inline void TransposeRigPosesSoaToAos(float *soa, float *aos_padded, int32_t n)
{
    const int32_t numGroups = (kCameraBlockSize + 15) / 16;
    for (int32_t r = 0; r < 4; ++r)
    {
        for (int32_t col = 0; col < 4; ++col)
        {
            const int32_t srcComp = r * 4 + col;
            const int32_t dstCol  = col * 4 + r;
            TranspStoreColumn(&soa[srcComp * kCameraBlockSize], aos_padded, kRigPoseStagePitch, dstCol, numGroups);
        }
    }
    // The transposing store covers all kCameraBlockSize lanes; re-zero rows past camera_count to match
    // the input convention.
    for (int32_t c = n; c < kMaxCameraCount; ++c)
    {
        for (int32_t k = 0; k < 16; ++k)
        {
            aos_padded[c * kRigPoseStagePitch + k] = 0.0f;
        }
    }
}

// Reprojection cost of the current rig_poses/points3d (trans_poses must be current). Seeds prev_cost
// so iteration 0's accept/reject test is a real comparison.
static float ComputeInitialCost(void)
{
    int ping_pong = 0, ping_pong_cost = 0;
    issueStage0TileLoads(0, ping_pong, 0);

    for (int32_t i = 0; i < track_iter; ++i)
    {
        syncStage0TileLoads(ping_pong);

        if (i != (track_iter - 1))
        {
            issueStage0TileLoads(i + 1, ping_pong ^ 1, 0);
        }

        AgenCFG *points_cfg  = ping_pong ? &params_bundler[0].cfgs[6] : &params_bundler[0].cfgs[5];
        AgenCFG *bitmask_cfg = ping_pong ? &params_bundler[0].cfgs[15] : &params_bundler[0].cfgs[14];
        AgenCFG *meas_cfg    = ping_pong ? &params_bundler[0].cfgs[11] : &params_bundler[0].cfgs[10];

        for (int32_t j = 0; j < camera_count; ++j)
        {
            MatrixMulV2Knl(&params_bundler[0].cfgs[3], points_cfg, &params_bundler[0].cfgs[7], j, 0, 0,
                           params_bundler[0].ths[3], params_bundler[0].niters[3]);

            ComputeResidualKnl(&params_bundler[0].cfgs[9], meas_cfg, &params_bundler[0].cfgs[12],
                               &params_bundler[0].cfgs[13], min_distance, j * 2, params_bundler[0].niters[4]);

            FindObservedTracksKnl(bitmask_cfg, &params_bundler[0].cfgs[17], j, params_bundler[0].niters[5]);

            FilterResidualKnl(&params_bundler[0].cfgs[108], &params_bundler[0].cfgs[109], &params_bundler[0].cfgs[110],
                              params_bundler[0].niters[36]);

            float *cost_in  = ping_pong_cost ? &cost1[0] : &cost2[0];
            float *cost_out = ping_pong_cost ? &cost2[0] : &cost1[0];

            ComputeCostKnl(cost_in, &params_bundler[0].cfgs[110], scale_meas * scale_meas, cost_out,
                           params_bundler[0].niters[35]);

            ping_pong_cost ^= 1;
        }

        ping_pong ^= 1;
    }

    float mtotal_cost = 0.0f;
    float *costf      = ping_pong_cost ? &cost1[0] : &cost2[0];
    for (int i = 0; i < 16; ++i)
    {
        mtotal_cost += costf[i];
    }
    return mtotal_cost;
}

CUPVA_VPU_MAIN()
{
    ParamInit(&params_bundler[0]);
    AgenConfigInitKnl(&params_bundler[0]);

    // One-shot loads; rigPoses land in the padded `rig_pose_stage` (not `dPose`) for the transpose.
    cupvaSQDFOpen(oneShotInputHdl);

    cupvaSQDFUpdateAddr(oneShotInputHdl, 0, rigPosesExtPtr.base + rigPosesExtPtr.offset, MEMTYPE_DRAM,
                        cupvaGetVmemAddress((void *)&rig_pose_stage[0]), MEMTYPE_VMEM);
    cupvaSQDFUpdateAddr(oneShotInputHdl, 1, rig2camExtPtr.base + rig2camExtPtr.offset, MEMTYPE_DRAM,
                        cupvaGetVmemAddress((void *)&rig2cam[0]), MEMTYPE_VMEM);
    cupvaSQDFFlushAndTrig(oneShotInputHdl);
    cupvaSQDFSync(oneShotInputHdl);
    cupvaSQDFClose(oneShotInputHdl);

    TransposeRigPosesAosToSoa(&rig_pose_stage[0], &rig_poses[0], camera_count);

    cupvaSQDFOpen(points3dHdl);
    cupvaSQDFOpen(measurementsHdl);
    cupvaSQDFOpen(cameraBitmaskHdl);
    cupvaSQDFOpen(tpcWriteHdl);
    cupvaSQDFOpen(tpcReadHdl);
    cupvaSQDFOpen(tpWriteHdl);
    cupvaSQDFOpen(tpReadHdl);
    cupvaSQDFOpen(pnewTentativeOutHdl);
    cupvaSQDFOpen(pnewCommitLoadHdl);
    cupvaSQDFOpen(pnewCommitStoreHdl);

    // TransformCam2rigKnl takes rig2cam column-major, but the tensor is row-major and DMA'd straight
    // into VMEM, so transpose it here (rig_poses is transposed on load).
    {
        float rig2camTransposed[16];
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                rig2camTransposed[c * 4 + r] = rig2cam[r * 4 + c];
            }
        }
        for (int k = 0; k < 16; ++k)
        {
            rig2cam[k] = rig2camTransposed[k];
        }
    }

    TransformCam2rigKnl(&rig2cam[0], &rig_poses[0], &trans_poses[0], kCameraBlockSize, camera_count);

    // LM loop (constants in bundle_adjustment_param.h). Point state lives in two SoA scratch buffers
    // swapped on accept; ImportPoints3d() seeds the current one.
    pointsCurBase  = points3dScratchExtPtr.base + points3dScratchExtPtr.offset;
    pointsTentBase = points3dScratch2ExtPtr.base + points3dScratch2ExtPtr.offset;
    ImportPoints3d();

    float prev_cost      = ComputeInitialCost();
    float current_lambda = lambdaInit;
    int32_t iter         = 0;
    int32_t converged    = 0;

    for (iter = 0; iter < maxIterations && !converged; iter++)
    {
        lambda = current_lambda;

        // Computes the tentative update and total_cost; committed below only if accepted.
        BuildReducedSystem(iter);

        const float cost = total_cost;

        // Reject steps that do not improve the cost; growing lambda alone can walk uphill and diverge.
        const int32_t accepted = (cost < prev_cost);

        if (accepted)
        {
            current_lambda *= kLambdaDecFactor;
            if (current_lambda < kLambdaMin)
            {
                current_lambda = kLambdaMin;
            }

            {
                const int64_t swap = pointsCurBase;
                pointsCurBase      = pointsTentBase;
                pointsTentBase     = swap;
            }
            CopyDataKnl(params_bundler[0].update_poses_ptr, params_bundler[0].rig_poses_ptr, 16 * kCameraBlockSize);
        }
        else
        {
            current_lambda *= kLambdaIncFactor;
            if (current_lambda > kLambdaMax)
            {
                current_lambda = kLambdaMax;
            }

            // Rejected: state is unchanged except trans_poses, which ApplyUpdate() overwrote from
            // update_poses; resync it.
            TransformCam2rigKnl(params_bundler[0].rig2cam_ptr, params_bundler[0].rig_poses_ptr,
                                params_bundler[0].trans_poses_ptr, kCameraBlockSize, camera_count);
        }

        // rel_change is vs the prior best cost even on rejection; minCost is checked after the commit.
        // Open-coded abs: the VPU toolchain has no <math.h>.
        const float cost_delta = prev_cost - cost;
        const float abs_delta  = (cost_delta < 0.0f) ? -cost_delta : cost_delta;
        const float rel_change = (prev_cost > 0.0f) ? (abs_delta / prev_cost) : 0.0f;

        if (accepted)
        {
            prev_cost = cost;
        }

        if (rel_change <= kMinCostChange || prev_cost <= kMinCost)
        {
            converged = 1;
        }
    }

    ExportPoints3d();

    iterationsUsed = iter;

    // total_cost holds the last (possibly rejected) tentative cost; report the last-accepted one to
    // match the exported poses.
    total_cost = prev_cost;

    cupvaSQDFClose(points3dHdl);
    cupvaSQDFClose(measurementsHdl);
    cupvaSQDFClose(cameraBitmaskHdl);
    cupvaSQDFClose(tpcWriteHdl);
    cupvaSQDFClose(tpcReadHdl);
    cupvaSQDFClose(tpWriteHdl);
    cupvaSQDFClose(tpReadHdl);
    cupvaSQDFClose(pnewTentativeOutHdl);
    cupvaSQDFClose(pnewCommitLoadHdl);
    cupvaSQDFClose(pnewCommitStoreHdl);

    // Stage rig poses as camera-major AoS in the now-idle `rig_pose_stage`.
    TransposeRigPosesSoaToAos(&rig_poses[0], &rig_pose_stage[0], camera_count);

    cupvaSQDFOpen(oneShotOutputHdl);

    cupvaSQDFUpdateAddr(oneShotOutputHdl, 0, cupvaGetVmemAddress((void *)&rig_pose_stage[0]), MEMTYPE_VMEM,
                        updatedRigPosesExtPtr.base + updatedRigPosesExtPtr.offset, MEMTYPE_DRAM);
    cupvaSQDFUpdateAddr(oneShotOutputHdl, 1, cupvaGetVmemAddress((void *)&total_cost), MEMTYPE_VMEM,
                        finalCostExtPtr.base + finalCostExtPtr.offset, MEMTYPE_DRAM);
    cupvaSQDFUpdateAddr(oneShotOutputHdl, 2, cupvaGetVmemAddress((void *)&iterationsUsed), MEMTYPE_VMEM,
                        iterationsUsedExtPtr.base + iterationsUsedExtPtr.offset, MEMTYPE_DRAM);

    cupvaSQDFFlushAndTrig(oneShotOutputHdl);
    cupvaSQDFSync(oneShotOutputHdl);
    cupvaSQDFClose(oneShotOutputHdl);

    return 0;
}

void ParamInit(ParamBundlerType *params)
{
    params->Hpp0_ptr = &Hpp0[0];
    params->Hpp1_ptr = &Hpp1[0];
    params->bp0_ptr  = &bp0[0];
    params->bp1_ptr  = &bp1[0];
    params->Hpc_ptr  = &Hpc[0];

    params->trans_poses_ptr  = &trans_poses[0];
    params->rig_poses_ptr    = &rig_poses[0];
    params->points3d_ptr     = &points3d[0];
    params->points3d_cam_ptr = &points3d_cam[0];
    params->points3d_rig_ptr = &points3d_rig[0];

    params->measurements_ptr   = &measurements_stage[0];
    params->residual0_ptr      = &residual0[0];
    params->infront_ptr        = &infront[0];
    params->camera_bitmask_ptr = &camera_bitmask[0];
    params->valid_flag_ptr     = &valid_flag[0];

    params->weight0_ptr = &weight0[0];
    params->weight1_ptr = &weight1[0];

    params->Jp0_ptr   = &Jp0[0];
    params->Jp1_ptr   = &Jp1[0];
    params->JpTJp_ptr = &JpTJp[0];
    params->JpTE_ptr  = &JpTE[0];

    params->rig2cam_ptr = &rig2cam[0];
    params->Jc0_ptr     = &Jc0[0];
    params->Jc1_ptr     = &Jc1[0];
    params->JcTJc_ptr   = &JcTJc[0];
    params->MatA0_ptr   = &MatA0[0];
    params->MatA1_ptr   = &Hpc[18 * kActiveCameraCount * kTrackCountTile];
    params->JcTE_ptr    = &JcTE[0];
    params->MatB0_ptr   = &MatB0[0];
    params->MatB1_ptr   = &Hpc[(18 * kActiveCameraCount + kMatAHeight) * kTrackCountTile];

    params->Hpp_ptr     = ((camera_count % 2) == 0) ? &Hpp0[0] : &Hpp1[0];
    params->Hpp_inv_ptr = &Hpp_inv[0];
    params->status_ptr  = &statuses[0];
    params->bp_ptr      = ((camera_count % 2) == 0) ? &bp0[0] : &bp1[0];
    params->tp_ptr      = &tp[0];

    params->HpcTp_ptr  = &HpcTp[0];
    params->Tpc_ptr    = &Tpc[0];
    params->TpcHpc_ptr = &TpcHpc[0];

    params->A0_ptr           = params->MatA0_ptr;
    params->A1_ptr           = params->MatA1_ptr;
    params->B0_ptr           = params->MatB0_ptr;
    params->B1_ptr           = params->MatB1_ptr;
    params->LTx_ptr          = &LTx[0];
    params->dC_ptr           = &MatB0[0];
    params->dPose_ptr        = &dPose[0];
    params->update_poses_ptr = &update_poses[0];

    params->Tpc_all_ptr = &Hpc[0];
    params->dp_ptr      = &dp[0];
    params->dp1_ptr     = &dp1[0];
    params->pnew_ptr    = &pnew[0];

    params->residual1_ptr = &residual1[0];
    params->buf_c_ptr     = &buf_c[0];
}

void BuildReducedSystem(int32_t iter)
{
    // P2 = T0 * P1: R2 = R0*R1, t2 = R0*t1 + t0
    TransformCam2rigKnl(&rig2cam[0], &rig_poses[0], &trans_poses[0], kCameraBlockSize, camera_count);

    // Zero MatA0/MatB0 once per LM iteration: UpdateWithDeltaKnl() only accumulates and
    // ResetVariableKnl() clears only Hpp0/bp0/Hpc.
    for (int32_t k = 0; k < 36 * kWordWidth * kSymmetricCameraSize; ++k)
    {
        MatA0[k] = 0.0f;
    }
    for (int32_t k = 0; k < 6 * kWordWidth * kActiveCameraCount; ++k)
    {
        MatB0[k] = 0.0f;
    }

    // Prefetch the per-tile hot code group (BuildReducedSystem..ComputeResidualKnl, contiguous per
    // bundle_adjustment.bcf); takes a word address, so the function address is passed unscaled.
    cupvaICachePrefetch((uintptr_t)BuildReducedSystem, kIcachePrefetchBytes);

    int ping_pong_pt = 0;

    issueStage0TileLoads(0, ping_pong_pt, iter);

    // track_iter = num_tracks / tile size.
    for (int32_t i = 0; i < track_iter; ++i)
    {
        ResetVariableKnl(0, &params_bundler[0]);
        ResetVariableKnl(1, &params_bundler[0]);
        ResetVariableKnl(2, &params_bundler[0]);

        syncStage0TileLoads(ping_pong_pt);

        if (i != (track_iter - 1))
        {
            issueStage0TileLoads(i + 1, ping_pong_pt ^ 1, iter);
        }

        BuildReducedSystemStage0(ping_pong_pt);

        BuildReducedSystemStage1(i);

        ping_pong_pt ^= 1;
    }

    // Drain the last tp/Tpc stores (callers sync only the previous trigger); otherwise
    // cupvaSQDFClose() or the next iteration's re-trigger deadlocks.
    cupvaSQDFSync(tpWriteHdl);
    cupvaSQDFSync(tpcWriteHdl);

    // Warm the solve/update code group evicted by the tile loop.
    cupvaICachePrefetch((uintptr_t)TransformCam2rigKnl, kIcachePrefetchSolveBytes);

    ComputeUpdate(iter);
}

void BuildReducedSystemStage0(int ping_pong_pt)
{
    AgenCFG *points_cfg  = ping_pong_pt ? &params_bundler[0].cfgs[6] : &params_bundler[0].cfgs[5];
    AgenCFG *bitmask_cfg = ping_pong_pt ? &params_bundler[0].cfgs[15] : &params_bundler[0].cfgs[14];
    AgenCFG *meas_cfg    = ping_pong_pt ? &params_bundler[0].cfgs[11] : &params_bundler[0].cfgs[10];

    for (int32_t j = 0; j < camera_count; ++j)
    {
        // x1 = R * x0 + t
        MatrixMulV2Knl(&params_bundler[0].cfgs[3], points_cfg, &params_bundler[0].cfgs[7], j, 0, 0,
                       params_bundler[0].ths[3], params_bundler[0].niters[3]);

        // residual = [x/z - m0, y/z - m1]; j * 2 advances meas_cfg past camera j's (x,y) columns.
        ComputeResidualKnl(&params_bundler[0].cfgs[9], meas_cfg, &params_bundler[0].cfgs[12],
                           &params_bundler[0].cfgs[13], min_distance, j * 2, params_bundler[0].niters[4]);

        FindValidTracksKnl(bitmask_cfg, &params_bundler[0].cfgs[16], &params_bundler[0].cfgs[17], j,
                           params_bundler[0].niters[5]);

        ComputeWeightKnl(&params_bundler[0].cfgs[18], scale_meas * scale_meas, &params_bundler[0].cfgs[19],
                         params_bundler[0].niters[6]);

        ScaleWeightKnl(&params_bundler[0].cfgs[19], rinv_meas, &params_bundler[0].cfgs[20],
                       params_bundler[0].niters[6]);

        ComputeJacobianPointKnl(j, &params_bundler[0].cfgs[21], &params_bundler[0].cfgs[22],
                                &params_bundler[0].cfgs[23], params_bundler[0].niters[7]);

        FilterJacobianKnl(&params_bundler[0].cfgs[23], &params_bundler[0].cfgs[24], &params_bundler[0].cfgs[25],
                          params_bundler[0].niters[8]);

        // wJp = weight * Jp
        MatrixMulKnl(&params_bundler[0].cfgs[26], &params_bundler[0].cfgs[27], &params_bundler[0].cfgs[28], 0, 0, 0,
                     params_bundler[0].ths[9], params_bundler[0].niters[9]);
        // JptJp = wJp^T * Jp
        MatrixMulKnl(&params_bundler[0].cfgs[29], &params_bundler[0].cfgs[30], &params_bundler[0].cfgs[31], 0, 0, 0,
                     params_bundler[0].ths[10], params_bundler[0].niters[10]);
        // diag(JptJp) *= 1 + lambda
        ScalePoseDiagnalKnl(&params_bundler[0].cfgs[32], lambda, params_bundler[0].niters[11]);

        // Hpp += JptJp
        AgenCFG *Hpp0_cfg = ((j % 2) == 0) ? &params_bundler[0].cfgs[33] : &params_bundler[0].cfgs[34];
        AgenCFG *Hpp1_cfg = ((j % 2) == 1) ? &params_bundler[0].cfgs[33] : &params_bundler[0].cfgs[34];
        UpdateWithDeltaKnl(Hpp0_cfg, &params_bundler[0].cfgs[35], Hpp1_cfg, 1.0f, 0, params_bundler[0].niters[12]);

        // JptE = weight * Jp^T * E
        MatrixMulKnl(&params_bundler[0].cfgs[36], &params_bundler[0].cfgs[37], &params_bundler[0].cfgs[38], 0, 0, 0,
                     params_bundler[0].ths[13], params_bundler[0].niters[13]);

        // bp -= JptE
        AgenCFG *bp0_cfg = ((j % 2) == 0) ? &params_bundler[0].cfgs[39] : &params_bundler[0].cfgs[40];
        AgenCFG *bp1_cfg = ((j % 2) == 1) ? &params_bundler[0].cfgs[39] : &params_bundler[0].cfgs[40];
        UpdateWithDeltaKnl(bp0_cfg, &params_bundler[0].cfgs[41], bp1_cfg, -1.0f, 0, params_bundler[0].niters[14]);

        // Pose blocks only for cameras in the active window [begin, begin + count); other cameras have
        // no MatA0/MatB0/Hpc storage.
        if (j >= active_camera_begin && j < active_camera_begin + active_camera_count)
        {
            MatrixMulV2Knl(&params_bundler[0].cfgs[4], points_cfg, &params_bundler[0].cfgs[8], j, 0, 0,
                           params_bundler[0].ths[3], params_bundler[0].niters[3]);

            ComputeJacobianPoseS1Knl(&params_bundler[0].cfgs[42], &params_bundler[0].cfgs[43],
                                     &params_bundler[0].cfgs[44], params_bundler[0].niters[15]);

            ComputeJacobianPoseS2Knl(&params_bundler[0].cfgs[45], &params_bundler[0].cfgs[46],
                                     &params_bundler[0].cfgs[47], params_bundler[0].niters[15]);

            FilterJacobianKnl(&params_bundler[0].cfgs[48], &params_bundler[0].cfgs[49], &params_bundler[0].cfgs[50],
                              params_bundler[0].niters[16]);

            // wJc = weight * Jc
            MatrixMulKnl(&params_bundler[0].cfgs[51], &params_bundler[0].cfgs[52], &params_bundler[0].cfgs[53], 0, 0, 0,
                         params_bundler[0].ths[17], params_bundler[0].niters[17]);
            // JctJc = wJc^T * Jc (6x6)
            MatrixMulKnl_323(&params_bundler[0].cfgs[54], &params_bundler[0].cfgs[55], &params_bundler[0].cfgs[56], 0,
                             0, 0, params_bundler[0].ths[18], params_bundler[0].niters[18]);
            // diag(JctJc) *= 1 + lambda
            ScalePoseDiagnalKnl(&params_bundler[0].cfgs[57], lambda, params_bundler[0].niters[19]);

            // Hcc += JctJc (needs reduction)
            int ind = ComputeIndexKnl(j - active_camera_begin, j - active_camera_begin, kActiveCameraCount);

            ReductionKnl(&params_bundler[0].cfgs[113], &params_bundler[0].cfgs[114], 1.0f, params_bundler[0].ths[37],
                         params_bundler[0].niters[37]);

            UpdateWithDeltaKnl(&params_bundler[0].cfgs[58], &params_bundler[0].cfgs[59], &params_bundler[0].cfgs[60],
                               1.0f, ind * 36 * kWordWidth, params_bundler[0].niters[20]);

            for (int32_t k = (j + 1); k < active_camera_begin + active_camera_count; ++k)
            {
                int curr_ind = ComputeIndexKnl(k - active_camera_begin, j - active_camera_begin, kActiveCameraCount);
                CopyAKnl(&params_bundler[0].cfgs[61], &params_bundler[0].cfgs[62], curr_ind * 36 * kWordWidth,
                         params_bundler[0].niters[21]);
            }

            // Hpc = Jp^T * Jc (3x6)
            MatrixMulKnl_323(&params_bundler[0].cfgs[63], &params_bundler[0].cfgs[64], &params_bundler[0].cfgs[65], 0,
                             0, (j - active_camera_begin) * 18 * kTrackCountTile, params_bundler[0].ths[22],
                             params_bundler[0].niters[22]);
            // JcTE = weight * Jc^T * E (6x1)
            MatrixMulKnl(&params_bundler[0].cfgs[66], &params_bundler[0].cfgs[67], &params_bundler[0].cfgs[68], 0, 0, 0,
                         params_bundler[0].ths[23], params_bundler[0].niters[23]);
            // B -= JcTE
            ReductionKnl(&params_bundler[0].cfgs[115], &params_bundler[0].cfgs[116], 1.0f, params_bundler[0].ths[38],
                         params_bundler[0].niters[38]);

            UpdateWithDeltaKnl(&params_bundler[0].cfgs[69], &params_bundler[0].cfgs[70], &params_bundler[0].cfgs[71],
                               -1.0f, (j - active_camera_begin) * 6 * kWordWidth, params_bundler[0].niters[24]);
        }
    }
}

void BuildReducedSystemStage1(int i)
{
    InverseSymmetric3X3Knl(&params_bundler[0].cfgs[72], &params_bundler[0].cfgs[73], &params_bundler[0].cfgs[74],
                           params_bundler[0].niters[25]);

    // tp is single-buffered: wait for the previous tile's DMA-out before overwriting it.
    if (i != 0)
    {
        cupvaSQDFSync(tpWriteHdl);
    }

    // tp = Hpp^-1 * bp
    MatrixMulKnl(&params_bundler[0].cfgs[75], &params_bundler[0].cfgs[76], &params_bundler[0].cfgs[77], 0, 0, 0,
                 params_bundler[0].ths[26], params_bundler[0].niters[26]);
    storeTpTile(i);

    // Schur complement update of A and B; O(n^2) in active cameras.
    for (int32_t j = 0; j < active_camera_count; ++j)
    {
        // Hpc^T * tp, then B -= it
        MatrixMulKnl(&params_bundler[0].cfgs[78], &params_bundler[0].cfgs[79], &params_bundler[0].cfgs[80],
                     j * 18 * kTrackCountTile, 0, 0, params_bundler[0].ths[27], params_bundler[0].niters[27]);
        ReductionKnl(&params_bundler[0].cfgs[117], &params_bundler[0].cfgs[116], 1.0f, params_bundler[0].ths[38],
                     params_bundler[0].niters[38]);

        UpdateWithDeltaKnl(&params_bundler[0].cfgs[81], &params_bundler[0].cfgs[82], &params_bundler[0].cfgs[83], -1.0f,
                           j * 6 * kWordWidth, params_bundler[0].niters[24]);

        // Tpc is single-buffered: wait for the previous (tile, camera) DMA-out before overwriting it.
        if (i != 0 || j != 0)
        {
            cupvaSQDFSync(tpcWriteHdl);
        }

        // Tpc = Hpc^T * Hpp^-1 (6x3)
        MatrixMulKnl_233(&params_bundler[0].cfgs[84], &params_bundler[0].cfgs[85], &params_bundler[0].cfgs[86],
                         j * 18 * kTrackCountTile, 0, 0, params_bundler[0].ths[29], params_bundler[0].niters[29]);
        // ApplyUpdate() bulk-reads these per tile.
        storeTpcCamera(i, j);

        for (int32_t k = j; k < active_camera_count; ++k)
        {
            // A_kj -= Tpc * Hpc_k (6x6)
            MatrixMulKnl_332(&params_bundler[0].cfgs[87], &params_bundler[0].cfgs[88], &params_bundler[0].cfgs[89], 0,
                             k * 18 * kTrackCountTile, 0, params_bundler[0].ths[30], params_bundler[0].niters[30]);

            int ind = ComputeIndexKnl(k, j, kActiveCameraCount);

            ReductionKnl(&params_bundler[0].cfgs[118], &params_bundler[0].cfgs[114], 1.0f, params_bundler[0].ths[37],
                         params_bundler[0].niters[37]);

            UpdateWithDeltaKnl(&params_bundler[0].cfgs[90], &params_bundler[0].cfgs[91], &params_bundler[0].cfgs[92],
                               -1.0f, ind * 36 * kWordWidth, params_bundler[0].niters[31]);
        }
    }
}

void ComputeUpdate(int32_t iter)
{
    AccumulateMatKnl(params_bundler[0].A0_ptr, params_bundler[0].A1_ptr, 36, 36 * kWordWidth, kSymmetricCameraSize,
                     kSymmetricCameraSize);

    AccumulateMatKnl(params_bundler[0].B0_ptr, params_bundler[0].B1_ptr, 6, 6 * kWordWidth, kSymmetricCameraSize,
                     active_camera_count);

    Convert2MatrixKnl(params_bundler[0].A1_ptr, params_bundler[0].A0_ptr, kSymmetricCameraSize, kActiveCameraCount * 6,
                      6, 6, active_camera_count);

    Convert2MatrixBKnl(params_bundler[0].B1_ptr, params_bundler[0].B0_ptr, kSymmetricCameraSize, 6,
                       active_camera_count);

    CholeskyDecompositionKnl(params_bundler[0].A0_ptr, params_bundler[0].A1_ptr, kActiveCameraCount * 6,
                             active_camera_count * 6);

    ForwardSubstitutionKnl(params_bundler[0].A1_ptr, params_bundler[0].B0_ptr, params_bundler[0].LTx_ptr,
                           kActiveCameraCount * 6, active_camera_count * 6);

    // L^T * dC = LTx completes the Cholesky solve (the forward pass solved L * y = B).
    BackwardSubstitutionKnl(params_bundler[0].A1_ptr, params_bundler[0].LTx_ptr, params_bundler[0].dC_ptr,
                            kActiveCameraCount * 6, active_camera_count * 6);

    ApplyUpdate(iter);
}

void ApplyUpdate(int32_t iter)
{
    for (int i = 0; i < kWordWidth; i++)
    {
        cost1[i] = 0.0f;
        cost2[i] = 0.0f;
    }

    int ping_pong = 0, ping_pong_cost = 0;

    Convert2SE3Knl(params_bundler[0].dC_ptr, params_bundler[0].dPose_ptr, kCameraBlockSize, active_camera_count);

    CopyDataKnl(params_bundler[0].rig_poses_ptr, params_bundler[0].update_poses_ptr, 16 * kCameraBlockSize);

    UpdatePosesKnl(params_bundler[0].dPose_ptr, &params_bundler[0].rig_poses_ptr[active_camera_begin],
                   &params_bundler[0].update_poses_ptr[active_camera_begin], kCameraBlockSize, active_camera_count);

    // UpdatePosesKnl stores whole vecw-lane groups, clobbering update_poses slots past the active window
    // with garbage dPose; restore them from rig_poses before anything reads update_poses.
    {
        const int32_t vecw         = params_bundler[0].vecw;
        const int32_t groupLanes   = ((active_camera_count + vecw - 1) / vecw) * vecw;
        const int32_t spilloverBeg = active_camera_begin + active_camera_count;
        int32_t spilloverEnd       = active_camera_begin + groupLanes;
        if (spilloverEnd > kCameraBlockSize)
        {
            spilloverEnd = kCameraBlockSize;
        }
        for (int32_t c = spilloverBeg; c < spilloverEnd; ++c)
        {
            for (int32_t k = 0; k < 16; ++k)
            {
                update_poses[k * kCameraBlockSize + c] = rig_poses[k * kCameraBlockSize + c];
            }
        }
    }

    TransformCam2rigKnl(params_bundler[0].rig2cam_ptr, params_bundler[0].update_poses_ptr,
                        params_bundler[0].trans_poses_ptr, kCameraBlockSize, camera_count);

    issueApplyUpdateTileLoads(0, ping_pong, iter);
    loadTpTile(0);

    for (int32_t i = 0; i < track_iter; ++i)
    {
        syncApplyUpdateTileLoads(ping_pong);
        cupvaSQDFSync(tpReadHdl);

        if (i != (track_iter - 1))
        {
            issueApplyUpdateTileLoads(i + 1, ping_pong ^ 1, iter);
        }

        // tp is single-buffered at tp[0] (loadTpTile()), so its cfg (cfgs[97]) is not ping-ponged.
        AgenCFG *dp_cfg      = &params_bundler[0].cfgs[97];
        AgenCFG *points_cfg  = ping_pong ? &params_bundler[0].cfgs[101] : &params_bundler[0].cfgs[100];
        AgenCFG *bitmask_cfg = ping_pong ? &params_bundler[0].cfgs[105] : &params_bundler[0].cfgs[104];
        AgenCFG *tpc_cfg     = ping_pong ? &params_bundler[0].cfgs[94] : &params_bundler[0].cfgs[93];

        for (int32_t j = 0; j < active_camera_count; ++j)
        {
            AgenCFG *dp1_cfg = ((j % 2) == 0) ? dp_cfg : &params_bundler[0].cfgs[98];
            AgenCFG *dp2_cfg = ((j % 2) == 1) ? dp_cfg : &params_bundler[0].cfgs[98];

            // dP = tp - Tpc^T * dC
            MatrixMulV2Knl(&params_bundler[0].cfgs[95], tpc_cfg, &params_bundler[0].cfgs[96], 6 * j,
                           18 * j * kTrackCountTile, 0, params_bundler[0].ths[32], params_bundler[0].niters[32]);

            UpdateWithDeltaKnl(dp1_cfg, &params_bundler[0].cfgs[99], dp2_cfg, -1.0f, 0, params_bundler[0].niters[33]);
        }

        AgenCFG *dp3_cfg  = ((active_camera_count % 2) == 0) ? dp_cfg : &params_bundler[0].cfgs[98];
        AgenCFG *pnew_cfg = ping_pong ? &params_bundler[0].cfgs[103] : &params_bundler[0].cfgs[102];

        ApplyPointUpdateKnl(points_cfg, dp3_cfg, bitmask_cfg, pnew_cfg, params_bundler[0].niters[34]);

        // Only now can tile i+1's tp reload start: this tile's camera loop and ApplyPointUpdateKnl still
        // use the single-buffered `tp`.
        if (i != (track_iter - 1))
        {
            loadTpTile(i + 1);
        }

        AgenCFG *homo_cfg = ping_pong ? &params_bundler[0].cfgs[107] : &params_bundler[0].cfgs[106];
        SetHomogeneousKnl(homo_cfg, params_bundler[0].niters[35]);

        AgenCFG *points_new_cfg  = ping_pong ? &params_bundler[0].cfgs[112] : &params_bundler[0].cfgs[111];
        AgenCFG *bitmask_new_cfg = ping_pong ? &params_bundler[0].cfgs[15] : &params_bundler[0].cfgs[14];
        AgenCFG *meas_cfg        = ping_pong ? &params_bundler[0].cfgs[11] : &params_bundler[0].cfgs[10];

        for (int32_t j = 0; j < camera_count; ++j)
        {
            MatrixMulV2Knl(&params_bundler[0].cfgs[3], points_new_cfg, &params_bundler[0].cfgs[7], j, 0, 0,
                           params_bundler[0].ths[3], params_bundler[0].niters[3]);

            ComputeResidualKnl(&params_bundler[0].cfgs[9], meas_cfg, &params_bundler[0].cfgs[12],
                               &params_bundler[0].cfgs[13], min_distance, j * 2, params_bundler[0].niters[4]);

            // Bitmask-only (not FindValidTracksKnl): behind-camera observations keep residual 1 and are
            // penalized in the cost.
            FindObservedTracksKnl(bitmask_new_cfg, &params_bundler[0].cfgs[17], j, params_bundler[0].niters[5]);

            FilterResidualKnl(&params_bundler[0].cfgs[108], &params_bundler[0].cfgs[109], &params_bundler[0].cfgs[110],
                              params_bundler[0].niters[36]);

            float *cost_in  = ping_pong_cost ? &cost1[0] : &cost2[0];
            float *cost_out = ping_pong_cost ? &cost2[0] : &cost1[0];

            ComputeCostKnl(cost_in, &params_bundler[0].cfgs[110], scale_meas * scale_meas, cost_out,
                           params_bundler[0].niters[35]);

            ping_pong_cost ^= 1;
        }

        if (i != 0)
        {
            cupvaSQDFSync(pnewTentativeOutHdl);
        }
        storePnewTile(i, ping_pong);

        ping_pong ^= 1;
    }

    // Drain the last store before MAIN reads total_cost and the next iteration reads the scratch.
    cupvaSQDFSync(pnewTentativeOutHdl);

    float mtotal_cost = 0.0f;
    float *costf      = ping_pong_cost ? &cost1[0] : &cost2[0];

    for (int i = 0; i < 16; ++i)
    {
        mtotal_cost += costf[i];
    }
    total_cost = mtotal_cost;

    // Commits happen in CUPVA_VPU_MAIN once the step is accepted.
}
