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

#include "doa_capon_roi_ref.h"

#include "complex_cholesky_ref.h"
#include "cov_mat_ref.h"
#include "find_local_max_ref.h"

#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <cmath>
#include <vector>

void doaCaponROIRef(NVCVTensorHandle snapshotsROIHandle, NVCVTensorHandle steeringVectorsROIHandle,
                    NVCVTensorHandle peakPowerHandle, NVCVTensorHandle azimuthBinIndicesHandle,
                    NVCVTensorHandle elevationBinIndicesHandle, const PVADoaCaponROIParams *params)
{
    int32_t numAzimuthBins   = params->numAzBins;
    int32_t numElevationBins = params->numElBins;
    int32_t matrixSize       = params->svDim;
    int32_t inputQBits       = params->inputQBits;
    int32_t steeringQBits    = params->steeringQBits;
    float snapshotScale      = std::ldexp(1.0f, -inputQBits);

    NVCVTensorData snapshotsData, steeringData;
    NVCVTensorData peakPowerData, azimuthData, elevationData;
    nvcvTensorExportData(snapshotsROIHandle, &snapshotsData);
    nvcvTensorExportData(steeringVectorsROIHandle, &steeringData);
    nvcvTensorExportData(peakPowerHandle, &peakPowerData);
    nvcvTensorExportData(azimuthBinIndicesHandle, &azimuthData);
    nvcvTensorExportData(elevationBinIndicesHandle, &elevationData);

    int32_t roiCount   = (int32_t)snapshotsData.shape[0];
    int32_t numSamples = params->numDopplerBinsCovMat;

    int32_t *snapshots   = nullptr;
    int32_t *steering    = nullptr;
    float *peakPower     = nullptr;
    float *azimuthBins   = nullptr;
    float *elevationBins = nullptr;
    CupvaMemGetHostPointer((void **)&snapshots, (void *)snapshotsData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&steering, (void *)steeringData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&peakPower, (void *)peakPowerData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&azimuthBins, (void *)azimuthData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&elevationBins, (void *)elevationData.buffer.strided.basePtr);

    size_t snapshotRoiStride  = (size_t)snapshotsData.buffer.strided.strides[0] / sizeof(*snapshots);
    size_t steeringRoiStride  = (size_t)steeringData.buffer.strided.strides[0] / sizeof(*steering);
    size_t powerRoiStride     = (size_t)peakPowerData.buffer.strided.strides[0] / sizeof(*peakPower);
    size_t azimuthRoiStride   = (size_t)azimuthData.buffer.strided.strides[0] / sizeof(*azimuthBins);
    size_t elevationRoiStride = (size_t)elevationData.buffer.strided.strides[0] / sizeof(*elevationBins);

    for (int32_t roi = 0; roi < roiCount; ++roi)
    {
        const int32_t *roiSnapshots = snapshots + (size_t)roi * snapshotRoiStride;
        std::vector<float> covarianceReal((size_t)matrixSize * (matrixSize + 1) / 2);
        std::vector<float> covarianceImag((size_t)matrixSize * (matrixSize - 1) / 2);
        if (params->useFpCovMatKnl)
        {
            std::vector<float> floatingSnapshots((size_t)numSamples * matrixSize * 2);
            for (size_t index = 0; index < floatingSnapshots.size(); ++index)
            {
                floatingSnapshots[index] = (float)roiSnapshots[index] * snapshotScale;
            }
            computeFpCovarianceRef(floatingSnapshots.data(), matrixSize, numSamples, params->adpDiagConst,
                                   covarianceReal.data(), covarianceImag.data());
        }
        else
        {
            computeS32CovarianceRef(roiSnapshots, matrixSize, numSamples, inputQBits, params->adpDiagConst,
                                    covarianceReal.data(), covarianceImag.data());
        }

        complexCholeskyRef(covarianceReal.data(), covarianceImag.data(), matrixSize);

        size_t triangleSize = (size_t)matrixSize * (matrixSize + 1) / 2;
        std::vector<float> inverseReal(triangleSize);
        std::vector<float> inverseImag(triangleSize);
        complexLinvRef(covarianceReal.data(), covarianceImag.data(), inverseReal.data(), inverseImag.data(),
                       matrixSize);

        const int32_t *roiSteering = steering + (size_t)roi * steeringRoiStride;
        float roiPeakPower[3];
        float roiAzimuthBins[3];
        float roiElevationBins[3];
        findLocalMaxRef(inverseReal.data(), inverseImag.data(), roiSteering, matrixSize, numAzimuthBins,
                        numElevationBins, steeringQBits, params->noiseFloorRatio, params->enableSubBinInterp != 0,
                        roiPeakPower, roiAzimuthBins, roiElevationBins);

        for (int32_t peak = 0; peak < params->numPeaksPerROI; ++peak)
        {
            peakPower[(size_t)roi * powerRoiStride + peak]         = roiPeakPower[peak];
            azimuthBins[(size_t)roi * azimuthRoiStride + peak]     = roiAzimuthBins[peak];
            elevationBins[(size_t)roi * elevationRoiStride + peak] = roiElevationBins[peak];
        }
    }
}
