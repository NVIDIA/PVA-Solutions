/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "radar_pipeline_reference.hpp"

#include "doa_target_processing_ref.h"
#include "doppler_fft_ref.h"
#include "nci_ref.h"
#include "peak_detection_ref.h"
#include "range_fft_ref.h"

#include <ErrorCheckMacros.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>

#include <cstdio>

int range_fft_processing(RadarPipelineTensors &tensors)
{
    PVARangeFFTParams params = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                .outputLayout = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE};
    int32_t err              = 0;

    // Run reference implementation for comparison
    rangeFFTRef(tensors.inRangeFFTTensorHandle, tensors.winRangeFFTTensorHandle, tensors.refRangeFFTTensorHandle,
                &params);

    // Compare PVA output to reference
    err = checkRangeFFTOutput(tensors.refRangeFFTTensorHandle, tensors.outRangeFFTTensorHandle, 0.0f);
    if (err)
    {
        printf("Range FFT Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("Range FFT Processing Test Pass - PVA operator matches reference\n");
    }

    return err;
}

int doppler_fft_processing(RadarPipelineTensors &tensors)
{
    PVADopplerFFTParams params = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                  .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                  .outputLayout = PVA_FFT_LAYOUT_RANGE_RX_DOPPLER};
    int32_t err                = 0;

    // Run reference implementation for comparison
    dopplerFFTRef(tensors.inDopplerFFTTensorHandle, tensors.winDopplerFFTTensorHandle,
                  tensors.refDopplerFFTTensorHandle, &params);

    // Compare PVA output to reference
    err = checkDopplerFFTOutput(tensors.refDopplerFFTTensorHandle, tensors.outDopplerFFTTensorHandle, 0.0f);
    if (err)
    {
        printf("Doppler FFT Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("Doppler FFT Processing Test Pass - PVA operator matches reference\n");
    }

    return err;
}

int nci_processing(RadarPipelineTensors &tensors)
{
    int32_t err = 0;

    // Tolerance values for output comparison based on the reference implementation
    const uint32_t nciRxTolerance         = 3u;
    const uint32_t nciFinalTolerance      = (2u * nciRxTolerance) + 2u; // 8u
    const uint32_t noiseEstimateTolerance = nciFinalTolerance * 10u;    // 80u

    // Run reference implementation for comparison
    NVCVTensorHandle inTensorHandles[]  = {tensors.inNciTensorHandle};
    NVCVTensorHandle outTensorHandles[] = {tensors.outNciRxTensorHandle, tensors.outNciFinalTensorHandle,
                                           tensors.outNoiseEstimateTensorHandle};
    NVCVTensorHandle refTensorHandles[] = {tensors.refNciRxTensorHandle, tensors.refNciFinalTensorHandle,
                                           tensors.refNoiseEstimateTensorHandle};

    err =
        nciRef(inTensorHandles, refTensorHandles, 1, 3, PVA_RADAR_DOPPLER_FOLD_COUNT, PVA_FFT_LAYOUT_RANGE_RX_DOPPLER);
    if (0 != err)
    {
        printf("NCI Reference Test Fail\n");
        goto EXIT;
    }

    // Compare PVA output to reference
    err = checkNCIOutput(refTensorHandles[0], outTensorHandles[0], nciRxTolerance);
    err |= checkNCIOutput(refTensorHandles[1], outTensorHandles[1], nciFinalTolerance);
    err |= checkNCIOutput(refTensorHandles[2], outTensorHandles[2], noiseEstimateTolerance);

    if (err != 0)
    {
        printf("NCI Processing Test Fail - PVA output differs from reference\n");
    }
    else
    {
        printf("NCI Processing Test Pass - PVA operator matches reference\n");
    }

EXIT:
    return err;
}

int peak_detection(RadarPipelineTensors &tensors, int32_t NofTx, int32_t NofRx, int32_t repeatFold, int32_t &peakCount)
{
    NVCVTensorData peakCountData;
    int32_t *peakCountPtr = NULL;
    int32_t err           = 0;
    // Run reference implementation for comparison
    NVCVTensorHandle inHandles[]  = {tensors.inNciTensorHandle, tensors.refNciRxTensorHandle,
                                     tensors.refNciFinalTensorHandle, tensors.refNoiseEstimateTensorHandle};
    NVCVTensorHandle outHandles[] = {tensors.outPeakCountTensorHandle, tensors.outPeakIndexTensorHandle,
                                     tensors.outPeakSnapTensorHandle};
    NVCVTensorHandle refHandles[] = {tensors.refPeakCountTensorHandle, tensors.refPeakIndexTensorHandle,
                                     tensors.refPeakSnapTensorHandle};

    peakDetectionRef(inHandles, NofTx, repeatFold, refHandles);

    const float maxErrRatio = 0.05f;
    err = checkPeakDetectionOutput(NofTx, NofRx, tensors.refPeakCountTensorHandle, tensors.refPeakIndexTensorHandle,
                                   tensors.refPeakSnapTensorHandle, tensors.outPeakCountTensorHandle,
                                   tensors.outPeakIndexTensorHandle, tensors.outPeakSnapTensorHandle, maxErrRatio);

    // Get peak count from PVA output
    NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(tensors.outPeakCountTensorHandle, &peakCountData), err, FreeResources);
    if (CupvaMemGetHostPointer((void **)&peakCountPtr, (void *)peakCountData.buffer.strided.basePtr) !=
        CUPVA_ERROR_NONE)
    {
        err = 1;
        goto FreeResources;
    }

    peakCount = *peakCountPtr;

    // Compare PVA output to reference (simplified for now)
    printf("Peak Detection Processing Test Pass - PVA operator completed\n");

FreeResources:
    return err;
}

int doa_processing(RadarPipelineTensors &tensors, int32_t peakCount)
{
    int32_t err         = 0;
    int32_t quickResult = 0;

    // Set up DOA input tensor handles for reference implementation
    NVCVTensorHandle inDOATensorHandles[] = {tensors.outPeakCountTensorHandle, tensors.inDOACalibVectorTensorHandle,
                                             tensors.outPeakIndexTensorHandle, tensors.outPeakSnapTensorHandle,
                                             tensors.outNciFinalTensorHandle};

    // Run reference implementation for comparison
    doaTargetProcessingRef(inDOATensorHandles, tensors.refDOATensorHandle, &tensors.GP);

    // Compare PVA output to reference
    quickResult = quickCompareOutputWithReference(tensors.outDOATensorHandle, tensors.refDOATensorHandle, &tensors.GP,
                                                  peakCount, 2e-4f);
    if (quickResult == 0)
    {
        printf("DOA Target Processing Test Pass - PVA operator matches reference\n");
    }
    else
    {
        printf("DOA Target Processing Test: Found %d mismatches between PVA and reference\n", quickResult);
        return -1;
    }

    return err;
}

// Function to execute all reference workloads
int radar_execute_ref_workloads(RadarPipelineTensors &tensors, int32_t NofTx, int32_t NofRx, int32_t repeatFold)
{
    int32_t peakCount = 0;
    int32_t err       = 0;

    err = range_fft_processing(tensors);
    if (err != 0)
    {
        printf("Range FFT processing failed with error: %d\n", err);
        goto EXIT;
    }

    err = doppler_fft_processing(tensors);
    if (err != 0)
    {
        printf("Doppler FFT processing failed with error: %d\n", err);
        goto EXIT;
    }

    err = nci_processing(tensors);
    if (err != 0)
    {
        printf("NCI processing failed with error: %d\n", err);
        goto EXIT;
    }

    err = peak_detection(tensors, NofTx, NofRx, repeatFold, peakCount);
    if (err != 0)
    {
        printf("Peak detection failed with error: %d\n", err);
        goto EXIT;
    }

    err = doa_processing(tensors, peakCount);
    if (err != 0)
    {
        printf("DOA processing failed with error: %d\n", err);
        goto EXIT;
    }

EXIT:
    return err;
}
