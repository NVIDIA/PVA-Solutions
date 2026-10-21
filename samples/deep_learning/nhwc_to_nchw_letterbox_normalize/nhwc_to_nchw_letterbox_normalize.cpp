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
 * @file nhwc_to_nchw_letterbox_normalize.cpp
 *
 * @brief Sample demonstrating the fused NHWC->NCHW + letterbox + normalize operator.
 *
 * Compares the PVA output against a CPU reference.
 *
 * Input:  HWC U8  (imgH x imgW x 3)
 * Output: CHW F32 (3 x outH x outW)
 */

#include "ErrorCheckMacros.h"
#include "ref/nhwc_to_nchw_letterbox_normalize_ref.h"

#include <OpNhwcToNchwLetterboxNormalize.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

struct TestCase
{
    const char *name;
    int imgWidth, imgHeight; // letterboxed resize output (PVA resize result)
    int outWidth, outHeight; // model input size
};

static const TestCase kTests[] = {
    // Standard 16:9 landscape -> square model input (padding top + bottom)
    {"1280x720_to_672x672", 672, 378, 672, 672},
    // Portrait input -> square (padding left + right)
    {"378x672_to_672x672", 378, 672, 672, 672},
    // Already square, no padding needed
    {"672x672_to_672x672", 672, 672, 672, 672},
    // Both padTop AND padLeft non-zero
    {"640x427_to_672x672", 640, 427, 672, 672},
    // Reverse: wide padding left, narrow padding top
    {"427x640_to_672x672", 427, 640, 672, 672},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CPU reference test
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// PVA operator test
// ---------------------------------------------------------------------------

static int runPvaTest(const TestCase &tc, bool useCudaStream)
{
    const int channels   = 3;
    const float scale    = 1.0f / 255.0f;
    const float padValue = 114.0f / 255.0f;
    const int padTop     = (tc.outHeight - tc.imgHeight) / 2;
    const int padLeft    = (tc.outWidth - tc.imgWidth) / 2;

    printf("Test [pva] %s\n", tc.name);
    printf("  img=%dx%d  out=%dx%d  padTop=%d padLeft=%d\n", tc.imgWidth, tc.imgHeight, tc.outWidth, tc.outHeight,
           padTop, padLeft);

    int32_t err                   = 0;
    NVCVAllocatorHandle allocator = NULL;
    NVCVTensorHandle imgIn        = NULL;
    NVCVTensorHandle refOut       = NULL;
    NVCVTensorHandle pvaOut       = NULL;
    NVCVTensorRequirements inReqs, outReqs;
    NVCVTensorLayout inLayout, outLayout;
    NVCVOperatorHandle opHandle = NULL;
    cupvaStream_t stream        = NULL;
    cudaStream_t cudaStream     = NULL;
    cupvaSyncObj_t sync         = NULL;
    cupvaFence_t fence;
    cupvaCmd_t rf;
    cupvaCmd_t const *cmds[1];
    cupvaCmdStatus_t status[1];
    bool waitSuccess = false;

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocator), err, Cleanup);

    // Input tensor: HWC U8
    {
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &inLayout), err, Cleanup);
        int64_t shape[] = {tc.imgHeight, tc.imgWidth, channels};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, shape, NVCV_DATA_TYPE_U8, inLayout, 0, 0, &inReqs), err,
                              Cleanup);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&inReqs, allocator, &imgIn), err, Cleanup);
        generateNhwcToNchwLetterboxNormalizeInput(imgIn);
    }

    // Output tensors: CHW F32 (one for ref, one for PVA)
    {
        NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("CHW", &outLayout), err, Cleanup);
        int64_t shape[] = {channels, tc.outHeight, tc.outWidth};
        NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(3, shape, NVCV_DATA_TYPE_F32, outLayout, 0, 0, &outReqs),
                              err, Cleanup);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outReqs, allocator, &refOut), err, Cleanup);
        NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&outReqs, allocator, &pvaOut), err, Cleanup);
    }

    // Compute reference
    nhwcToNchwLbNormRef(imgIn, refOut, scale, padValue);

    // Create PVA operator
    NVCV_CHECK_ERROR_GOTO(pvaNhwcToNchwLetterboxNormalizeCreate(&opHandle, &inReqs, &outReqs, scale, padValue,
                                                                PVA_NHWC_TO_NCHW_PADDING_RESTORE_ON_SUBMIT),
                          err, Cleanup);

    if (useCudaStream)
    {
#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
        if (cudaError_t cudaStatus = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
            cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(cudaStatus));
            err = 1;
            goto Cleanup;
        }
#else
        fprintf(stderr, "CUDA-stream submission is unavailable in this SDK\n");
        err = 1;
        goto Cleanup;
#endif
    }
    else
    {
        CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err, Cleanup);
        CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, Cleanup);
        CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, Cleanup);
        CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, Cleanup);
    }

    // Corrupt the complete destination before submission. The operator must
    // restore padding as well as the image region and must not rely on prior contents.
    {
        NVCVTensorData pvaOutData;
        NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(pvaOut, &pvaOutData), err, Cleanup);
        float *pvaOutHost = nullptr;
        CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer(reinterpret_cast<void **>(&pvaOutHost),
                                                      reinterpret_cast<void *>(pvaOutData.buffer.strided.basePtr)),
                               err, Cleanup);

        std::fill_n(pvaOutHost, channels * tc.outHeight * tc.outWidth, -123.0f);
        if (useCudaStream)
        {
#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
            NVCV_CHECK_ERROR_GOTO(pvaNhwcToNchwLetterboxNormalizeSubmit(opHandle, cudaStream, imgIn, pvaOut), err,
                                  Cleanup);
            if (cudaError_t cudaStatus = cudaStreamSynchronize(cudaStream); cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "Failed to synchronize CUDA stream: %s\n", cudaGetErrorString(cudaStatus));
                err = 1;
                goto Cleanup;
            }
#endif
        }
        else
        {
            NVCV_CHECK_ERROR_GOTO(pvaNhwcToNchwLetterboxNormalizeSubmit(opHandle, stream, imgIn, pvaOut), err, Cleanup);
            cmds[0]   = &rf;
            status[0] = NULL;
            CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, Cleanup);
            CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, Cleanup);
            if (!waitSuccess)
            {
                fprintf(stderr, "Waiting for the cuPVA fence failed\n");
                err = 1;
                goto Cleanup;
            }
        }
    }

    // Compare PVA output against reference
    {
        int32_t mismatches = checkNhwcToNchwLetterboxNormalizeOutput(refOut, pvaOut);
        if (mismatches == 0)
            printf("  PASSED [%s stream]\n\n", useCudaStream ? "CUDA" : "cuPVA");
        else
        {
            printf("  FAILED (%d mismatches)\n\n", mismatches);
            err = 1;
        }
    }

Cleanup:
    cudaStreamDestroy(cudaStream);
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvOperatorDestroy(opHandle);
    nvcvTensorDecRef(pvaOut, NULL);
    nvcvTensorDecRef(refOut, NULL);
    nvcvTensorDecRef(imgIn, NULL);
    nvcvAllocatorDecRef(allocator, NULL);

    return err;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    printf("==============================================\n");
    printf("  NHWC->NCHW+Letterbox+Normalize Operator Tests\n");
    printf("==============================================\n\n");

    int passed = 0, failed = 0;
    const int numTests = sizeof(kTests) / sizeof(kTests[0]);

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    // cuPVA decides whether CUDA interop is available while creating the first
    // cupva context, so the CUDA context has to exist before any PVA call.
    bool cudaReady = true;
    if (cudaError_t cudaStatus = cudaSetDevice(0); cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to initialize CUDA context: %s\n", cudaGetErrorString(cudaStatus));
        cudaReady = false;
    }
#endif

    printf("--- PVA Operator ---\n\n");
    for (int i = 0; i < numTests; i++)
    {
        if (runPvaTest(kTests[i], false) == 0)
            passed++;
        else
            failed++;
    }

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1
    if (!cudaReady)
    {
        failed++;
        goto Results;
    }
    {
        uint32_t driverVersion        = 0;
        cupvaError_t driverVersionErr = CupvaGetDriverVersion(&driverVersion);
        if (driverVersionErr != CUPVA_ERROR_NONE)
        {
            fprintf(stderr, "Failed to query the PVA driver version: %d\n", driverVersionErr);
            failed++;
        }
        else if (driverVersion >= 2007)
        {
            printf("--- CUDA-stream Operator ---\n\n");
            for (int i = 0; i < numTests; i++)
            {
                if (runPvaTest(kTests[i], true) == 0)
                    passed++;
                else
                    failed++;
            }
        }
        else
        {
            printf("CUDA-stream tests skipped: PVA driver %u, need >= 2007\n", driverVersion);
        }
    }
#else
    printf("CUDA-stream tests skipped: SDK has no CUDA-stream support\n");
#endif

Results:
    printf("==============================================\n");
    printf("  Results: %d/%d passed\n", passed, passed + failed);
    printf("==============================================\n");

    if (failed == 0)
    {
        printf("Test Pass\n");
        return 0;
    }
    printf("Test Fail\n");
    return 1;
}
