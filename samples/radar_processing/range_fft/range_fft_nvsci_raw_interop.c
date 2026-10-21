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

#include "ErrorCheckMacros.h"
#include "NvSciInteropUtils.h"
#include "ref/range_fft_ref.h"

#include <ImageIO.h>
#include <OpRangeFFT.h>
#include <PvaAllocator.h>
#include <PvaOperatorTypes.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

int main(int argc, char **argv)
{
    const int32_t Ns    = 1024;
    const int32_t NofRx = 8;
    const int32_t Nr    = 512;

    NvSciBufModule nvSciBufModule     = NULL;
    NVCVTensorHandle inTensorHandle   = NULL;
    NvSciBufObj inTensorNvsciBuf      = NULL;
    NVCVTensorHandle winTensorHandle  = NULL;
    NvSciBufObj winTensorNvsciBuf     = NULL;
    NVCVTensorHandle outTensorHandle  = NULL;
    NvSciBufObj outTensorNvsciBuf     = NULL;
    NVCVTensorHandle refTensorHandle  = NULL;
    NvSciBufObj refTensorNvsciBuf     = NULL;
    NVCVOperatorHandle operatorHandle = NULL;
    cupvaSyncObj_t sync               = NULL;
    cupvaStream_t stream              = NULL;
    int32_t err                       = 0;

    NVSCI_CHECK_ERROR_GOTO(NvSciBufModuleOpen(&nvSciBufModule), err, FreeResources);

    NVCVDataType inDataType  = NVCV_DATA_TYPE_S32;
    NVCVDataType winDataType = NVCV_DATA_TYPE_S32;
    NVCVDataType outDataType = NVCV_DATA_TYPE_2S32;

    int32_t inTensorRank  = 3;
    int32_t winTensorRank = 1;
    int32_t outTensorRank = 3;

    int64_t inTensorShape[]  = {Nr, NofRx, Ns};
    int64_t winTensorShape[] = {Ns};
    int64_t outTensorShape[] = {Nr, NofRx, Ns / 2 + 1};

    // Setup RangeFFT parameters
    PVARangeFFTParams params = {.windowType   = PVA_BATCH_FFT_WINDOW_HANNING,
                                .inputLayout  = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE,
                                .outputLayout = PVA_FFT_LAYOUT_DOPPLER_RX_RANGE};

    NVCVTensorLayout inTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &inTensorLayout), err, FreeResources);

    NVCVTensorRequirements inTensorRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(inTensorRank, inTensorShape, inDataType, inTensorLayout, 0, 0,
                                                        &inTensorRequirements),
                          err, FreeResources);

    int64_t inTensorSizeBytes = 0;
    NVCV_CHECK_ERROR_GOTO(nvcvMemRequirementsCalcTotalSizeBytes(&inTensorRequirements.mem.cudaMem, &inTensorSizeBytes),
                          err, FreeResources);

    NVSCI_CHECK_ERROR_GOTO(pvaCreateNvSciBufRawBuffer(&inTensorNvsciBuf, nvSciBufModule, inTensorSizeBytes), err,
                           FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapNvSciBuf(&inTensorRequirements, inTensorNvsciBuf, &inTensorHandle), err,
                          FreeResources);

    NVCVTensorLayout winTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("W", &winTensorLayout), err, FreeResources);

    NVCVTensorRequirements winTensorRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(winTensorRank, winTensorShape, winDataType, winTensorLayout, 0,
                                                        0, &winTensorRequirements),
                          err, FreeResources);

    int64_t winTensorSizeBytes = 0;
    NVCV_CHECK_ERROR_GOTO(
        nvcvMemRequirementsCalcTotalSizeBytes(&winTensorRequirements.mem.cudaMem, &winTensorSizeBytes), err,
        FreeResources);

    NVSCI_CHECK_ERROR_GOTO(pvaCreateNvSciBufRawBuffer(&winTensorNvsciBuf, nvSciBufModule, winTensorSizeBytes), err,
                           FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapNvSciBuf(&winTensorRequirements, winTensorNvsciBuf, &winTensorHandle), err,
                          FreeResources);

    NVCVTensorLayout outTensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HCW", &outTensorLayout), err, FreeResources);

    NVCVTensorRequirements outTensorRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(outTensorRank, outTensorShape, outDataType, outTensorLayout, 0,
                                                        0, &outTensorRequirements),
                          err, FreeResources);

    int64_t outTensorSizeBytes = 0;
    NVCV_CHECK_ERROR_GOTO(
        nvcvMemRequirementsCalcTotalSizeBytes(&outTensorRequirements.mem.cudaMem, &outTensorSizeBytes), err,
        FreeResources);

    NVSCI_CHECK_ERROR_GOTO(pvaCreateNvSciBufRawBuffer(&outTensorNvsciBuf, nvSciBufModule, outTensorSizeBytes), err,
                           FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapNvSciBuf(&outTensorRequirements, outTensorNvsciBuf, &outTensorHandle), err,
                          FreeResources);

    NVSCI_CHECK_ERROR_GOTO(pvaCreateNvSciBufRawBuffer(&refTensorNvsciBuf, nvSciBufModule, outTensorSizeBytes), err,
                           FreeResources);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorWrapNvSciBuf(&outTensorRequirements, refTensorNvsciBuf, &refTensorHandle), err,
                          FreeResources);

    generateRangeFFTInput(inTensorHandle);
    generateRangeFFTWindow(winTensorHandle, params.windowType);

    NVCV_CHECK_ERROR_GOTO(pvaRangeFFTCreate(&operatorHandle, &inTensorRequirements, &winTensorRequirements,
                                            &outTensorRequirements, &params),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(pvaRangeFFTSubmit(operatorHandle, stream, inTensorHandle, winTensorHandle, outTensorHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    rangeFFTRef(inTensorHandle, winTensorHandle, refTensorHandle, &params);

    err = checkRangeFFTOutput(refTensorHandle, outTensorHandle, 0.01f);
    if (err)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(inTensorHandle, NULL);
    nvcvTensorDecRef(winTensorHandle, NULL);
    nvcvTensorDecRef(outTensorHandle, NULL);
    nvcvTensorDecRef(refTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    NvSciBufObjFree(inTensorNvsciBuf);
    NvSciBufObjFree(winTensorNvsciBuf);
    NvSciBufObjFree(outTensorNvsciBuf);
    NvSciBufObjFree(refTensorNvsciBuf);
    NvSciBufModuleClose(nvSciBufModule);

    return err;
}