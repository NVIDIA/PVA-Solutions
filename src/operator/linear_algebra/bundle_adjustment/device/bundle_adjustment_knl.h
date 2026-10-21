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

#ifndef BUNDLE_ADJUSTMENT_KNL_H
#define BUNDLE_ADJUSTMENT_KNL_H
#include <cupva_device.h>

typedef struct ParamBundlerType
{
    int vecw;
    AgenCFG cfgs[128];
    int niters[128];
    int ths[128];

    dvfloatx zeros;
    dvfloatx ones;

    float *Hpp0_ptr;
    float *Hpp1_ptr;
    float *bp0_ptr;
    float *bp1_ptr;
    float *Hpc_ptr;

    float *trans_poses_ptr;
    float *rig_poses_ptr;
    float *points3d_ptr;
    float *points3d_cam_ptr;
    float *points3d_rig_ptr;

    float *measurements_ptr;
    float *residual0_ptr;
    int *infront_ptr;
    int *camera_bitmask_ptr;
    int *valid_flag_ptr;

    float *weight0_ptr;
    float *weight1_ptr;

    float *Jp0_ptr;
    float *Jp1_ptr;
    float *JpTJp_ptr;
    float *JpTE_ptr;

    float *rig2cam_ptr;
    float *Jc0_ptr;
    float *Jc1_ptr;
    float *JcTJc_ptr;
    float *MatA0_ptr;
    float *MatA1_ptr;
    float *JcTE_ptr;
    float *MatB0_ptr;
    float *MatB1_ptr;

    float *Hpp_ptr;
    float *Hpp_inv_ptr;
    int *status_ptr;
    float *bp_ptr;
    float *tp_ptr;

    float *HpcTp_ptr;
    float *Tpc_ptr;
    float *TpcHpc_ptr;

    float *A0_ptr;
    float *A1_ptr;
    float *B0_ptr;
    float *B1_ptr;
    float *LTx_ptr;
    float *dC_ptr;
    float *dPose_ptr;
    float *update_poses_ptr;

    float *Tpc_all_ptr;
    float *dp_ptr;
    float *dp1_ptr;
    float *pnew_ptr;

    float *residual1_ptr;
    float *buf_c_ptr;
} ParamBundlerType;

void AgenConfigInitKnl(ParamBundlerType *params);
void ResetVariableKnl(int cfg_index, ParamBundlerType *params);
void TransformCam2rigKnl(float *T, float *poses1, float *poses2, int lp, int n);
void CopyDataKnl(float *d1, float *d2, int n);
void FindObservedTracksKnl(const AgenCFG *bitmask, const AgenCFG *valid_flag, int camind, int niter);
void FilterResidualKnl(const AgenCFG *flag, const AgenCFG *res1, const AgenCFG *res2, int niter);
void FindValidTracksKnl(const AgenCFG *a_bitmask, const AgenCFG *a_infront, const AgenCFG *a_valid_flag, int camind,
                        int niter);
void ComputeResidualKnl(const AgenCFG *a_points, const AgenCFG *a_meas, const AgenCFG *a_residual,
                        const AgenCFG *a_flag, float min_distance, int offset, int niter);
void ComputeWeightKnl(const AgenCFG *a_residual, float square_scale, AgenCFG *a_weight, int niter);
void ScaleWeightKnl(const AgenCFG *w1, float rinv, const AgenCFG *w2, int niter);
void ComputeJacobianPointKnl(int icam, const AgenCFG *a_poses, const AgenCFG *a_points, const AgenCFG *a_jp, int niter);
void FilterJacobianKnl(const AgenCFG *a_jacobian0, const AgenCFG *a_flag, const AgenCFG *a_jacobian1, int niter);
void ComputeJacobianPoseS1Knl(const AgenCFG *pt0, const AgenCFG *rig2cam, const AgenCFG *Jc, int niter);
void ComputeJacobianPoseS2Knl(const AgenCFG *pt0, const AgenCFG *Jc0, const AgenCFG *Jc1, int niter);
void ScalePoseDiagnalKnl(const AgenCFG *a_JtJ, float lambda, int niter);
void ReductionKnl(const AgenCFG *delta, const AgenCFG *m, float s, int th, int niter);
void CopyAKnl(const AgenCFG *d0, const AgenCFG *d1, int offset, int niter);
void InverseSymmetric3X3Knl(const AgenCFG *Hpp, const AgenCFG *Hpp_inv, const AgenCFG *status, int niter);
void MatrixMulKnl(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1, int offset2,
                  int th, int niter);
void MatrixMulKnl_332(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                      int offset2, int th, int niter);
void MatrixMulKnl_323(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                      int offset2, int th, int niter);
void MatrixMulKnl_233(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                      int offset2, int th, int niter);
int ComputeIndexKnl(int i, int j, int n);
void AccumulateMatKnl(float *d0, float *d1, int d, int lp0, int lp1, int n);
void Convert2MatrixKnl(float *Am, float *Ad, int lp1, int lp2, int r, int c, int m);
void Convert2MatrixBKnl(float *Bm, float *Bd, int lp1, int d, int n);
void CholeskyDecompositionKnl(float *A, float *L, int lp, int n);
void ForwardSubstitutionKnl(float *L, float *B, float *x, int lp, int n);
void BackwardSubstitutionKnl(float *L, float *B, float *x, int lp, int n);
void MatrixMulV2Knl(const AgenCFG *a_m0, const AgenCFG *a_m1, const AgenCFG *a_m2, int offset0, int offset1,
                    int offset2, int th, int niter);
void ApplyPointUpdateKnl(const AgenCFG *pold, const AgenCFG *dp, const AgenCFG *bitmask, const AgenCFG *pnew,
                         int niter);
void UpdateWithDeltaKnl(const AgenCFG *m0, const AgenCFG *delta, const AgenCFG *m1, float s, int offset, int niter);
void SetHomogeneousKnl(const AgenCFG *pnew, int niter);
void Convert2SE3Knl(float *dC, float *pose, int lp, int n);
void UpdatePosesKnl(float *delta, float *old_pose, float *new_pose, int lp, int n);
void ComputeCostKnl(float *cost_in, const AgenCFG *residual, float scale, float *cost_out, int niter);
#endif // BUNDLE_ADJUSTMENT_KNL_H
