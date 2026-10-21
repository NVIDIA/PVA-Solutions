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
 * @file doa_capon_roi_ref.h
 *
 * @brief Scalar reference for the complete Capon ROI DoA operator.
 *
 * The reference computes covariance, diagonal loading, Cholesky and Linv, the Capon spectrum,
 * and peak selection. Tensor and asset helpers are declared in doa_capon_roi_utils.h.
 */

#pragma once

#include <OpDoaCaponROI.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run the mixed-precision scalar reference for both operator stages.
 *
 * Tensors must satisfy the public Capon ROI contract. @p params->useFpCovMatKnl selects the S32 or
 * floating-point covariance path; Cholesky and Linv use floating point in both paths.
 */
void doaCaponROIRef(NVCVTensorHandle snapshotsROIHandle, NVCVTensorHandle steeringVectorsROIHandle,
                    NVCVTensorHandle peakPowerHandle, NVCVTensorHandle azimuthBinIndicesHandle,
                    NVCVTensorHandle elevationBinIndicesHandle, const PVADoaCaponROIParams *params);

#ifdef __cplusplus
}
#endif
