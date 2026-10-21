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
 * @file doa_capon_roi.c
 * @brief Run the Capon ROI DoA operator on the 16x16_ura dataset and compare it with the fixed-point C
 *        reference.
 */

#include <ErrorCheckMacros.h>
#include <ImageIO.h>
#include <OpDoaCaponROI.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <cupva_host_nonsafety.h>
#include <doa_capon_roi_ref.h>
#include <doa_capon_roi_utils.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int PrepareCaponRun(CaponROITensors *tensors, NVCVOperatorHandle *op, PVADoaCaponROIParams *params,
                           const char *inputsBinPath, int32_t numSamples, int32_t assetRoiNum, int32_t roiNum,
                           const char *streamName)
{
    int err = caponROIAllocateTensors(tensors, params, roiNum);
    if (err != 0)
    {
        fprintf(stderr, "ERROR: failed to allocate tensors for %s (status %d)\n", streamName, err);
        return err;
    }

    printf("Loading inputs for %s from: %s\n", streamName, inputsBinPath);
    if (caponLoadInputsFromFlatBin(inputsBinPath, params, numSamples, assetRoiNum, roiNum, tensors->snapshotsROITensor,
                                   tensors->svTensor) != 0)
    {
        fprintf(stderr, "ERROR: failed to load inputs for %s\n", streamName);
        return 1;
    }
    printf("  Q-bits in use: inputQBits=%u, steeringQBits=%u\n", params->inputQBits, params->steeringQBits);

    NVCVStatus nvcvStatus = pvaDoaCaponROICreate(op, &tensors->snapshotsReq, &tensors->svReq, &tensors->peakPowerReq,
                                                 &tensors->azBinReq, &tensors->elBinReq, params);
    if (nvcvStatus != NVCV_SUCCESS)
    {
        char msgBuffer[1024];
        nvcvPeekAtLastErrorMessage(msgBuffer, sizeof(msgBuffer));
        fprintf(stderr, "ERROR: failed to create the Capon operator for %s (status %d): %s\n", streamName, nvcvStatus,
                msgBuffer);
        return (int)nvcvStatus;
    }

    printf("Running fixed-point reference for %s (cov=%s, interp=%s)...\n", streamName,
           params->useFpCovMatKnl ? "fp" : "s32", params->enableSubBinInterp ? "on" : "off");
    doaCaponROIRef(tensors->snapshotsROITensor, tensors->svTensor, tensors->refPeakPowerTensor, tensors->refAzBinTensor,
                   tensors->refElBinTensor, params);
    return 0;
}

static int ValidateCaponRun(const CaponROITensors *tensors, const PVADoaCaponROIParams *params, int32_t roiNum,
                            const char *streamName)
{
    printf("%s VPU vs. reference:\n", streamName);
    int errors = caponCompareVpuVsRef(tensors->peakPowerTensor, tensors->refPeakPowerTensor, tensors->azBinTensor,
                                      tensors->refAzBinTensor, tensors->elBinTensor, tensors->refElBinTensor, roiNum,
                                      params->numPeaksPerROI, TOLERANCE_POWER_REL, TOLERANCE_VPU_VS_REF_BIN_ABS,
                                      params->enableSubBinInterp);
    printf("%s test %s\n", streamName, errors == 0 ? "passed" : "failed");
    return errors == 0 ? 0 : 1;
}

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
static int RunCaponOnCudaStream(PVADoaCaponROIParams *params, const char *inputsBinPath, int32_t numSamples,
                                int32_t assetRoiNum, int32_t roiNum)
{
    int err                 = 0;
    CaponROITensors tensors = {0};
    NVCVOperatorHandle op   = NULL;
    cudaStream_t stream     = NULL;

    err = PrepareCaponRun(&tensors, &op, params, inputsBinPath, numSamples, assetRoiNum, roiNum, "CUDA stream");
    if (err != 0)
        goto FreeResources;

    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    printf("Submitting combined Capon DoA operator to CUDA stream...\n");
    NVCV_CHECK_ERROR_GOTO(pvaDoaCaponROISubmit(op, stream, tensors.snapshotsROITensor, tensors.svTensor,
                                               tensors.peakPowerTensor, tensors.azBinTensor, tensors.elBinTensor),
                          err, FreeResources);
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    err = ValidateCaponRun(&tensors, params, roiNum, "CUDA stream");

FreeResources:
    if (stream)
        cudaStreamDestroy(stream);
    if (op)
        nvcvOperatorDestroy(op);
    caponROIFreeTensors(&tensors);
    return err == 0 ? 0 : 1;
}
#endif

static int RunCaponOnCupvaStream(PVADoaCaponROIParams *params, const char *inputsBinPath, int32_t numSamples,
                                 int32_t assetRoiNum, int32_t roiNum)
{
    int err                 = 0;
    CaponROITensors tensors = {0};
    NVCVOperatorHandle op   = NULL;
    cupvaSyncObj_t sync     = NULL;
    cupvaStream_t stream    = NULL;

    err = PrepareCaponRun(&tensors, &op, params, inputsBinPath, numSamples, assetRoiNum, roiNum, "cuPVA stream");
    if (err != 0)
        goto FreeResources;

    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t requestFence;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&requestFence, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    printf("Submitting combined Capon DoA operator to cuPVA stream...\n");
    NVCV_CHECK_ERROR_GOTO(pvaDoaCaponROISubmit(op, stream, tensors.snapshotsROITensor, tensors.svTensor,
                                               tensors.peakPowerTensor, tensors.azBinTensor, tensors.elBinTensor),
                          err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&requestFence};
    cupvaCmdStatus_t status[1] = {NULL};
    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    bool waitSuccess = false;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);
    if (!waitSuccess)
    {
        fprintf(stderr, "ERROR: cuPVA stream execution did not complete successfully\n");
        err = 1;
        goto FreeResources;
    }

    err = ValidateCaponRun(&tensors, params, roiNum, "cuPVA stream");

FreeResources:
    if (stream)
        CupvaStreamDestroy(stream);
    if (sync)
        CupvaSyncObjDestroy(sync);
    if (op)
        nvcvOperatorDestroy(op);
    caponROIFreeTensors(&tensors);
    return err == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *covMode          = "s32"; /* "s32" (fixed-point) or "fp" cov-matrix path */
    const char *interpMode       = "on";
    int32_t numDopplerBinsCovMat = -1; /* -1 = default to dataset's numSamples */
    int32_t requestedRoiNum      = -1; /* -1 = use every ROI stored in the asset */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--cov") == 0 && i + 1 < argc)
            covMode = argv[++i];
        else if (strcmp(argv[i], "--interp") == 0 && i + 1 < argc)
            interpMode = argv[++i];
        else if (strcmp(argv[i], "--num_doppler_bins_cov_mat") == 0 && i + 1 < argc)
            numDopplerBinsCovMat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num_rois") == 0 && i + 1 < argc)
            requestedRoiNum = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            printf(
                "Usage: %s --assets <sample_assets_dir> [--cov <s32|fp>] [--interp <on|off>]\n"
                "       [--num_doppler_bins_cov_mat <N>] [--num_rois <N>]\n",
                argv[0]);
            printf(
                "  --assets    : sample assets root; the sample reads capon_roi/16x16_ura/inputs.bin.\n"
                "  Inputs are always quantized to 2S32 (operator I/O contract).\n"
                "  --cov s32   : fixed-point covariance path (VPU S32 executable and S32 reference).\n"
                "  --cov fp    : floating-point covariance path (VPU FP executable and FP reference).\n"
                "  --interp on : VPU sub-bin quadratic interp on az/el (default).\n"
                "  --interp off: VPU writes integer bin indices; sub-bin interp disabled.\n"
                "  --num_doppler_bins_cov_mat N : configure the number of Doppler bins used\n"
                "                                 for cov_mat (a partial 32-bin tile must contain\n"
                "                                 at least 12 bins and be divisible by 4). Stored\n"
                "                                 bins are center-sliced or repeated cyclically to\n"
                "                                 initialize a tensor with the requested height.\n"
                "                                 Default: dataset's numSamples.\n"
                "  --num_rois N : request N ROI rows. Rows beyond the stored asset are\n"
                "                 repeated cyclically. Default: all stored rows.\n");
            return 0;
        }
    }
    if (strcmp(covMode, "fp") != 0 && strcmp(covMode, "s32") != 0)
    {
        fprintf(stderr, "ERROR: --cov must be 'fp' or 's32' (got '%s')\n", covMode);
        return 1;
    }
    if (strcmp(interpMode, "on") != 0 && strcmp(interpMode, "off") != 0)
    {
        fprintf(stderr, "ERROR: --interp must be 'on' or 'off' (got '%s')\n", interpMode);
        return 1;
    }
    bool useFpCovMatKnl     = (strcmp(covMode, "fp") == 0);
    bool enableSubBinInterp = (strcmp(interpMode, "on") == 0);

    char assetsRoot[1024], assetsDir[1100];
    if (GetAssetsDirectory(argc, argv, assetsRoot, sizeof(assetsRoot)) != 0)
        return 1;
    snprintf(assetsDir, sizeof(assetsDir), "%s/capon_roi/16x16_ura", assetsRoot);

    PVADoaCaponROIParams params;
    memset(&params, 0, sizeof(params));
    int32_t assetRoiNum = 0, numSamples = 0;
    char inputsBinPath[1200];
    snprintf(inputsBinPath, sizeof(inputsBinPath), "%s/inputs.bin", assetsDir);
    if (caponLoadDatasetConfig(inputsBinPath, &params, &assetRoiNum, &numSamples) != 0)
        return 1;

    int32_t roiNum = requestedRoiNum < 0 ? assetRoiNum : requestedRoiNum;
    if (roiNum <= 0)
    {
        fprintf(stderr, "ERROR: --num_rois must be positive (got %d).\n", roiNum);
        return 1;
    }

    printf("Capon DoA on ROI - VPU vs reference (cov=%s, interp=%s)\n", covMode, interpMode);
    printf("  Config: %d az x %d el bins, svDim=%d, roiNum=%d requested from %d stored, samples=%d, IQ=%u, SQ=%u\n",
           params.numAzBins, params.numElBins, params.svDim, roiNum, assetRoiNum, numSamples, params.inputQBits,
           params.steeringQBits);

    /* Resolve the configured cov-mat Doppler bin count. */
    if (numDopplerBinsCovMat <= 0)
    {
        numDopplerBinsCovMat = numSamples;
    }
    int32_t partialTileSize = numDopplerBinsCovMat % 32;
    if (numDopplerBinsCovMat <= 0 || (partialTileSize != 0 && (partialTileSize < 12 || partialTileSize % 4 != 0)))
    {
        fprintf(stderr,
                "ERROR: the partial --num_doppler_bins_cov_mat tile must contain at least 12 bins and be divisible "
                "by 4 (got %d).\n",
                numDopplerBinsCovMat);
        return 1;
    }

    params.numPeaksPerROI       = 3;
    params.adpDiagConst         = ADP_DIAG_CONST;
    params.noiseFloorRatio      = NOISE_FLOOR_RATIO;
    params.numDopplerBinsCovMat = numDopplerBinsCovMat;
    params.useFpCovMatKnl       = useFpCovMatKnl;
    params.enableSubBinInterp   = enableSubBinInterp;
    /* inputQBits/steeringQBits come from caponLoadDatasetConfig (CAPON_ASSET_Q_BITS). */

    int cudaStreamResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "ERROR: cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStreamResult = 1;
    }
    else
    {
        uint32_t driverVersion   = 0;
        cupvaError_t cupvaStatus = CupvaGetDriverVersion(&driverVersion);
        if (cupvaStatus != CUPVA_ERROR_NONE)
        {
            fprintf(stderr, "ERROR: CupvaGetDriverVersion failed (status %d)\n", cupvaStatus);
            cudaStreamResult = 1;
        }
        else if (driverVersion >= 2007)
        {
            cudaStreamResult = RunCaponOnCudaStream(&params, inputsBinPath, numSamples, assetRoiNum, roiNum);
        }
        else
        {
            printf("CUDA stream test skipped: cuPVA driver version %u is older than 2007.\n", driverVersion);
        }
    }
#else
    printf("CUDA stream test skipped: this cuPVA build does not support CUDA stream submission.\n");
#endif

    int cupvaStreamResult = RunCaponOnCupvaStream(&params, inputsBinPath, numSamples, assetRoiNum, roiNum);
    int failed            = cudaStreamResult || cupvaStreamResult;
    printf("Test %s\n", failed ? "Fail" : "Pass");
    return failed;
}
