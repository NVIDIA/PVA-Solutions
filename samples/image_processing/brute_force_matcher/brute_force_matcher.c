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
#include "ref/brute_force_matcher_ref.h"

#include <OpBruteForceMatcher.h>
#include <PvaAllocator.h>
#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Allocate and initialize memory
 */
static int AllocateAndInitializeMemory(int32_t numQueryDescriptors, int32_t numReferenceDescriptors,
                                       PVABriefDescriptor **referenceDescriptors, PVABriefDescriptor **queryDescriptors,
                                       PVABFMatchesType **matches, PVABFMatchesType **refMatches)
{
    int32_t err = 0;

    CUPVA_CHECK_ERROR_GOTO(
        CupvaMemAlloc((void **)referenceDescriptors, numReferenceDescriptors * sizeof(PVABriefDescriptor),
                      CUPVA_READ_WRITE, CUPVA_ALLOC_DRAM),
        err, Error);
    generateBruteForceMatcherDescriptor(*referenceDescriptors, numReferenceDescriptors);

    CUPVA_CHECK_ERROR_GOTO(CupvaMemAlloc((void **)queryDescriptors, numQueryDescriptors * sizeof(PVABriefDescriptor),
                                         CUPVA_READ_WRITE, CUPVA_ALLOC_DRAM),
                           err, Error);
    generateBruteForceMatcherDescriptor(*queryDescriptors, numQueryDescriptors);

    CUPVA_CHECK_ERROR_GOTO(CupvaMemAlloc((void **)matches, numQueryDescriptors * sizeof(PVABFMatchesType),
                                         CUPVA_READ_WRITE, CUPVA_ALLOC_DRAM),
                           err, Error);

    CUPVA_CHECK_ERROR_GOTO(CupvaMemAlloc((void **)refMatches, numQueryDescriptors * sizeof(PVABFMatchesType),
                                         CUPVA_READ_WRITE, CUPVA_ALLOC_DRAM),
                           err, Error);

    return 0;

Error:
    return err;
}

/**
 * @brief Validate output against reference
 */
static int ValidateOutput(PVABriefDescriptor *queryDescriptors, int32_t numQueryDescriptors,
                          PVABriefDescriptor *referenceDescriptors, int32_t numReferenceDescriptors,
                          int32_t maxMatchesPerQuery, PVABFMatchesType *refMatches, PVABFMatchesType *matches,
                          uint8_t enableCrossCheck, uint8_t enableDistanceRatioTest,
                          int32_t lowesTestThresholdNumerator, int32_t lowesTestThresholdDenominator,
                          const char *testName)
{
    bruteForceMatcherRef(queryDescriptors, numQueryDescriptors, referenceDescriptors, numReferenceDescriptors,
                         maxMatchesPerQuery, refMatches, enableCrossCheck, enableDistanceRatioTest,
                         lowesTestThresholdNumerator, lowesTestThresholdDenominator);

    int32_t err = checkBruteForceMatcherOutput(refMatches, numQueryDescriptors, matches);
    if (err)
    {
        printf("%s test failed\n", testName);
    }
    return err;
}

/**
 * @brief Test brute force matcher operation using CUDA stream
 */
int SubmitBruteForceMatcherToCudaStream(int32_t numQueryDescriptors, int32_t numReferenceDescriptors,
                                        uint8_t enableCrossCheck, uint8_t enableDistanceRatioTest,
                                        int32_t maxMatchesPerQuery, int32_t lowesTestThresholdNumerator,
                                        int32_t lowesTestThresholdDenominator)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVOperatorHandle operatorHandle = NULL;
    cudaStream_t stream               = NULL;
    int32_t err                       = 0;

    PVABriefDescriptor *referenceDescriptors = NULL;
    PVABriefDescriptor *queryDescriptors     = NULL;
    PVABFMatchesType *matches                = NULL;
    PVABFMatchesType *refMatches             = NULL;

    /* Allocate and initialize memory */
    err = AllocateAndInitializeMemory(numQueryDescriptors, numReferenceDescriptors, &referenceDescriptors,
                                      &queryDescriptors, &matches, &refMatches);
    if (err)
        goto FreeResources;

    /* Create CUDA stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), err, FreeResources);

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaBruteForceMatcherCreate(&operatorHandle), err, FreeResources);

    /* Submit to CUDA stream */
    NVCV_CHECK_ERROR_GOTO(
        pvaBruteForceMatcherSubmit(operatorHandle, stream, queryDescriptors, numQueryDescriptors, referenceDescriptors,
                                   numReferenceDescriptors, maxMatchesPerQuery, matches, enableCrossCheck,
                                   enableDistanceRatioTest, lowesTestThresholdNumerator, lowesTestThresholdDenominator),
        err, FreeResources);

    /* Synchronize stream */
    CUDA_CHECK_ERROR_GOTO(cudaStreamSynchronize(stream), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(queryDescriptors, numQueryDescriptors, referenceDescriptors, numReferenceDescriptors,
                         maxMatchesPerQuery, refMatches, matches, enableCrossCheck, enableDistanceRatioTest,
                         lowesTestThresholdNumerator, lowesTestThresholdDenominator, "Submit to CUDA stream");

FreeResources: /* clean up all allocated resources */
    cudaStreamDestroy(stream);
    nvcvOperatorDestroy(operatorHandle);
    CupvaMemFree(referenceDescriptors);
    CupvaMemFree(queryDescriptors);
    CupvaMemFree(matches);
    CupvaMemFree(refMatches);

    return err;
}

/**
 * @brief Test brute force matcher operation using cuPVA stream
 */
int SubmitBruteForceMatcherToCupvaStream(int32_t numQueryDescriptors, int32_t numReferenceDescriptors,
                                         uint8_t enableCrossCheck, uint8_t enableDistanceRatioTest,
                                         int32_t maxMatchesPerQuery, int32_t lowesTestThresholdNumerator,
                                         int32_t lowesTestThresholdDenominator)
{
    /* Set the handles of resources that should be freed in case of an error to NULL.
     * This is done because calling the corresponding free functions with NULL is a no-op.
     */
    NVCVOperatorHandle operatorHandle = NULL;
    cupvaSyncObj_t sync               = NULL;
    cupvaStream_t stream              = NULL;
    int32_t err                       = 0;

    PVABriefDescriptor *referenceDescriptors = NULL;
    PVABriefDescriptor *queryDescriptors     = NULL;
    PVABFMatchesType *matches                = NULL;
    PVABFMatchesType *refMatches             = NULL;

    /* Allocate and initialize memory */
    err = AllocateAndInitializeMemory(numQueryDescriptors, numReferenceDescriptors, &referenceDescriptors,
                                      &queryDescriptors, &matches, &refMatches);
    if (err)
        goto FreeResources;

    /* Create operator */
    NVCV_CHECK_ERROR_GOTO(pvaBruteForceMatcherCreate(&operatorHandle), err, FreeResources);

    /* Create cuPVA stream and synchronization objects */
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    /* Submit to cuPVA stream */
    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    NVCV_CHECK_ERROR_GOTO(
        pvaBruteForceMatcherSubmit(operatorHandle, stream, queryDescriptors, numQueryDescriptors, referenceDescriptors,
                                   numReferenceDescriptors, maxMatchesPerQuery, matches, enableCrossCheck,
                                   enableDistanceRatioTest, lowesTestThresholdNumerator, lowesTestThresholdDenominator),
        err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    /* Wait for completion */
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    /* Validate output */
    err = ValidateOutput(queryDescriptors, numQueryDescriptors, referenceDescriptors, numReferenceDescriptors,
                         maxMatchesPerQuery, refMatches, matches, enableCrossCheck, enableDistanceRatioTest,
                         lowesTestThresholdNumerator, lowesTestThresholdDenominator, "Submit to cuPVA stream");

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvOperatorDestroy(operatorHandle);
    CupvaMemFree(referenceDescriptors);
    CupvaMemFree(queryDescriptors);
    CupvaMemFree(matches);
    CupvaMemFree(refMatches);

    return err;
}

int main(int argc, char **argv)
{
    /* Parameter settings — defaults */
    int32_t numQueryDescriptors           = PVA_MAX_BF_DESCRIPTOR_COUNT;
    int32_t numReferenceDescriptors       = PVA_MAX_BF_DESCRIPTOR_COUNT;
    uint8_t enableCrossCheck              = 1;
    uint8_t enableDistanceRatioTest       = 1;
    int32_t maxMatchesPerQuery            = 1;
    int32_t lowesTestThresholdNumerator   = 4;
    int32_t lowesTestThresholdDenominator = 5;

    /* Parse CLI arguments */
    for (int32_t i = 1; i < argc; i++)
    {
        if ((0 == strcmp(argv[i], "--queryCount")) && ((i + 1) < argc))
        {
            numQueryDescriptors = (int32_t)atoi(argv[++i]);
        }
        else if ((0 == strcmp(argv[i], "--refCount")) && ((i + 1) < argc))
        {
            numReferenceDescriptors = (int32_t)atoi(argv[++i]);
        }
        else if ((0 == strcmp(argv[i], "--enableCrossCheck")) && ((i + 1) < argc))
        {
            enableCrossCheck = (uint8_t)atoi(argv[++i]);
        }
        else if ((0 == strcmp(argv[i], "--enableDistanceRatioTest")) && ((i + 1) < argc))
        {
            enableDistanceRatioTest = (uint8_t)atoi(argv[++i]);
        }
        else if ((0 == strcmp(argv[i], "--maxMatchesPerQuery")) && ((i + 1) < argc))
        {
            maxMatchesPerQuery = (int32_t)atoi(argv[++i]);
        }
        else if ((0 == strcmp(argv[i], "--lowesNumerator")) && ((i + 1) < argc))
        {
            lowesTestThresholdNumerator = (int32_t)atoi(argv[++i]);
        }
        else if ((0 == strcmp(argv[i], "--lowesDenominator")) && ((i + 1) < argc))
        {
            lowesTestThresholdDenominator = (int32_t)atoi(argv[++i]);
        }
    }

    const char *offOnStr[] = {"OFF", "ON"};
    printf(
        "Configuration: numQueryDescriptors = %d, numReferenceDescriptors = %d, enableCrossCheck = %s, "
        "enableDistanceRatioTest = %s",
        numQueryDescriptors, numReferenceDescriptors, offOnStr[enableCrossCheck], offOnStr[enableDistanceRatioTest]);
    if (enableDistanceRatioTest)
    {
        printf(", thresholdRatio = %d/%d\n", lowesTestThresholdNumerator, lowesTestThresholdDenominator);
    }
    else if (enableCrossCheck)
    {
        printf("\n");
    }
    else
    {
        printf(", maxMatchesPerQuery = %d\n", maxMatchesPerQuery);
    }

    int32_t cudaStreamSubmitTestResult  = 0;
    int32_t cupvaStreamSubmitTestResult = 0;

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

    /* Initialize CUDA context*/
    CUDA_CHECK_ERROR_GOTO(cudaSetDevice(0), cudaStreamSubmitTestResult, CheckResults);

    /* Submit to CUDA stream if driver version is greater than or equal to 2007 */
    uint32_t driverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&driverVersion), cudaStreamSubmitTestResult, CheckResults);
    if (driverVersion >= 2007)
    {
        /* Submit to CUDA stream */
        cudaStreamSubmitTestResult = SubmitBruteForceMatcherToCudaStream(
            numQueryDescriptors, numReferenceDescriptors, enableCrossCheck, enableDistanceRatioTest, maxMatchesPerQuery,
            lowesTestThresholdNumerator, lowesTestThresholdDenominator);
    }
#endif

    /* Submit to cuPVA stream */
    cupvaStreamSubmitTestResult = SubmitBruteForceMatcherToCupvaStream(
        numQueryDescriptors, numReferenceDescriptors, enableCrossCheck, enableDistanceRatioTest, maxMatchesPerQuery,
        lowesTestThresholdNumerator, lowesTestThresholdDenominator);

CheckResults:
    /* Overall result */
    if (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult)
    {
        printf("Test Fail\n");
    }
    else
    {
        printf("Test Pass\n");
    }

    return (cudaStreamSubmitTestResult || cupvaStreamSubmitTestResult);
}
