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

/**
 * @file OpBundleAdjustment.h
 *
 * @brief Defines types and functions to perform Levenberg-Marquardt bundle adjustment.
 * @defgroup PVA_OPERATOR_ALGORITHM_LINEAR_ALGEBRA_BUNDLE_ADJUSTMENT BundleAdjustment
 * @{
 */

#ifndef PVA_SOLUTIONS_OPBUNDLEADJUSTMENT_H
#define PVA_SOLUTIONS_OPBUNDLEADJUSTMENT_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the Bundle Adjustment operator.
 *
 * Bundle adjustment jointly refines a set of rig poses and a set of 3D track (point) positions by
 * minimizing the sum of squared reprojection residuals across every camera that observes each track.
 * The operator runs a full Levenberg-Marquardt (LM) solver on-device: it iteratively builds the
 * reduced (Schur-complement) camera normal-equations system, solves it via Cholesky decomposition,
 * applies the resulting pose/point update, and adapts the LM damping factor (lambda) based on whether
 * the reprojection cost decreased, until convergence or \p maxIterations (see
 * ::PvaBundleAdjustmentParams) is reached.
 *
 * \b Rig \b poses \b tensor (\p rigPoses / \p updatedRigPoses):
 *      Data Layout:    [N, 16]
 *      Extent N:       [kMaxCameraCount]
 *      Data Type:      [32bit Float]
 *      Each row is a row-major 4x4 SE(3) world-to-rig pose matrix.
 *      A world-frame point is taken into a camera frame as \p rig2cam * rigPose, i.e. the rig pose
 *      maps world -> rig and \p rig2cam then maps rig -> camera.
 *
 * \b 3D \b track \b positions \b tensor (\p points3d / \p updatedPoints3d):
 *      Data Layout:    [N, 4]
 *      Extent N:       [kMaxTrackCount]
 *      Data Type:      [32bit Float]
 *      Each row is a homogeneous 3D point [x, y, z, 1].
 *
 * \b Measurements \b tensor (\p measurements):
 *      Data Layout:    [N, C, 2]
 *      Extent N:       [kMaxTrackCount]
 *      Extent C:       [kMaxCameraCount]
 *      Data Type:      [32bit Float]
 *      Per-track, per-camera 2D measurement [u, v] in normalized image-plane coordinates: pixel
 *      coordinates with the principal point subtracted and divided by the focal length. No camera
 *      intrinsics or lens distortion are applied, so measurements must already be undistorted.
 *
 * \b Camera \b visibility \b bitmask \b tensor (\p cameraBitmask):
 *      Data Layout:    [N]
 *      Extent N:       [kMaxTrackCount]
 *      Data Type:      [32bit Signed]
 *      One int32 bitmask per track: bit i (for i in [0, cameraCount)) set means the track is observed
 *      by camera i; bit 31 set means the track is active (its 3D point is optimized), and bit 31 clear
 *      means the track is fixed (its point is excluded from the point update, though its observations
 *      still contribute to the cost and to the camera Hessian/gradient).
 *
 * \b Rig-to-camera \b extrinsics \b tensor (\p rig2cam):
 *      Data Layout:    [16]
 *      Data Type:      [32bit Float]
 *      Row-major 4x4 SE(3) transform from the rig frame to the camera frame.
 *
 * \b Final \b cost \b tensor (\p finalCost):
 *      Data Layout:    [1]
 *      Data Type:      [32bit Float]
 *      Robust (Cauchy) cost of the returned state, which is the last accepted LM state.
 *
 * \b Iterations \b used \b tensor (\p iterationsUsed):
 *      Data Layout:    [1]
 *      Data Type:      [32bit Signed]
 *      Number of LM iterations executed, accepted or rejected, in [1, maxIterations].
 *
 * \b Outputs:
 *      - \p updatedRigPoses: cameras outside the active window keep their input pose; rows from
 *        \c cameraCount on are zero.
 *      - \p updatedPoints3d: only the first \c trackCount rows are written; fixed tracks keep their
 *        input [x, y, z], and the homogeneous coordinate is 1.
 *
 * \b Limitations:
 *      - \c trackCount (see ::PvaBundleAdjustmentParams, set at Submit time) must be an exact multiple
 *        of \c kTrackCountTile (128). Remainder/partial tiles are not supported.
 *      - \c cameraCount (set at Submit time) must be at most 21: each track's visibility
 *        is packed into one 32-bit \c cameraBitmask entry and bit 31 is reserved for the per-track
 *        active/fixed flag.
 *      - This operator requires PVA hardware generation 3 (e.g. Thor) and is not supported on PVA
 *        GEN2 (Orin): its on-chip working set needs a 4th VMEM bank that GEN2 hardware does not
 *        provide. Attempting to create or run this operator on a GEN2 platform fails.
 *      - All tensors must be densely packed. The DMA transfers use fixed, tightly packed row pitches,
 *        and the strides are not validated: a padded tensor gives wrong results, not an error.
 *      - One camera model applies to all rig poses: a single shared \p rig2cam, no intrinsics and no
 *        lens distortion.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] rigPoses Tensor requirements for the input rig poses tensor (world-to-rig).
 *                        + Must not be NULL.
 *
 * @param [in] points3d Tensor requirements for the input 3D track positions tensor.
 *                      + Must not be NULL.
 *
 * @param [in] measurements Tensor requirements for the input per-track, per-camera 2D measurements
 *                          tensor.
 *                          + Must not be NULL.
 *
 * @param [in] cameraBitmask Tensor requirements for the input per-track camera-visibility bitmask
 *                           tensor.
 *                           + Must not be NULL.
 *
 * @param [in] rig2cam Tensor requirements for the input rig-to-camera extrinsics tensor.
 *                     + Must not be NULL.
 *
 * @param [in] updatedRigPoses Tensor requirements for the output refined rig poses tensor.
 *                                + Must not be NULL.
 *
 * @param [in] updatedPoints3d Tensor requirements for the output refined 3D track positions tensor.
 *                             + Must not be NULL.
 *
 * @param [in] finalCost Tensor requirements for the output final reprojection cost tensor.
 *                       + Must not be NULL.
 *
 * @param [in] iterationsUsed Tensor requirements for the output number-of-LM-iterations tensor.
 *                            + Must not be NULL.
 *
 * @param [in] config Create-time configuration (active-camera window). See ::PvaBundleAdjustmentConfig.
 *                    + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle, a tensor requirements pointer, or config is null, or a
 *                                    tensor/config field is outside its valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBundleAdjustmentCreate(
    NVCVOperatorHandle *handle, NVCVTensorRequirements const *const rigPoses,
    NVCVTensorRequirements const *const points3d, NVCVTensorRequirements const *const measurements,
    NVCVTensorRequirements const *const cameraBitmask, NVCVTensorRequirements const *const rig2cam,
    NVCVTensorRequirements const *const updatedRigPoses, NVCVTensorRequirements const *const updatedPoints3d,
    NVCVTensorRequirements const *const finalCost, NVCVTensorRequirements const *const iterationsUsed,
    PvaBundleAdjustmentConfig const *const config);

#ifdef __cplusplus
}

/**
 * Submits the Bundle Adjustment operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] rigPoses Input rig poses tensor handle (world-to-rig).
 * @param [in] points3d Input 3D track positions tensor handle.
 * @param [in] measurements Input per-track, per-camera 2D measurements tensor handle.
 * @param [in] cameraBitmask Input per-track camera-visibility bitmask tensor handle.
 * @param [in] rig2cam Input rig-to-camera extrinsics tensor handle.
 * @param [out] updatedRigPoses Output refined rig poses tensor handle.
 * @param [out] updatedPoints3d Output refined 3D track positions tensor handle.
 * @param [out] finalCost Output final reprojection cost tensor handle.
 * @param [out] iterationsUsed Output number-of-LM-iterations tensor handle.
 * @param [in] params Submit-time Levenberg-Marquardt tuning parameters. See
 *                    ::PvaBundleAdjustmentParams.
 *                    + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBundleAdjustmentSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle rigPoses,
                                     NVCVTensorHandle points3d, NVCVTensorHandle measurements,
                                     NVCVTensorHandle cameraBitmask, NVCVTensorHandle rig2cam,
                                     NVCVTensorHandle updatedRigPoses, NVCVTensorHandle updatedPoints3d,
                                     NVCVTensorHandle finalCost, NVCVTensorHandle iterationsUsed,
                                     PvaBundleAdjustmentParams const *params);

/**
 * Submits the Bundle Adjustment operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] rigPoses Input rig poses tensor handle (world-to-rig).
 * @param [in] points3d Input 3D track positions tensor handle.
 * @param [in] measurements Input per-track, per-camera 2D measurements tensor handle.
 * @param [in] cameraBitmask Input per-track camera-visibility bitmask tensor handle.
 * @param [in] rig2cam Input rig-to-camera extrinsics tensor handle.
 * @param [out] updatedRigPoses Output refined rig poses tensor handle.
 * @param [out] updatedPoints3d Output refined 3D track positions tensor handle.
 * @param [out] finalCost Output final reprojection cost tensor handle.
 * @param [out] iterationsUsed Output number-of-LM-iterations tensor handle.
 * @param [in] params Submit-time Levenberg-Marquardt tuning parameters. See
 *                    ::PvaBundleAdjustmentParams.
 *                    + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaBundleAdjustmentSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle rigPoses,
                                     NVCVTensorHandle points3d, NVCVTensorHandle measurements,
                                     NVCVTensorHandle cameraBitmask, NVCVTensorHandle rig2cam,
                                     NVCVTensorHandle updatedRigPoses, NVCVTensorHandle updatedPoints3d,
                                     NVCVTensorHandle finalCost, NVCVTensorHandle iterationsUsed,
                                     PvaBundleAdjustmentParams const *params);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPBUNDLEADJUSTMENT_H */
