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

#include "NvSciInteropUtils.h"

#include "ErrorCheckMacros.h"
#include "InteropUtilsCommon.h"

#include <cupva_host.h>
#include <cupva_host_cuda.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Requirements.h>
#include <nvscibuf.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// Helper function for NvSciBufObj import and common resource allocation
static NVCVStatus setup_nvsci_import_resources(NvSciBufObj nvsciBufObjToImport, size_t internalDataStructSize,
                                               InteropInternalDataType dataType, InteropCleanupCtx **outCtxForCleanup,
                                               void **outInternalData, const char *callerFunctionName)
{
    NVCVStatus local_nvcv_status = NVCV_SUCCESS;
    cupvaError_t local_cupva_status;

    *outCtxForCleanup = NULL;
    *outInternalData  = NULL;

    local_nvcv_status = InteropCreateCleanupCtx(outCtxForCleanup);
    if (local_nvcv_status != NVCV_SUCCESS)
    {
        return local_nvcv_status;
    }

    *outInternalData = malloc(internalDataStructSize);
    if (*outInternalData == NULL)
    {
        printf("%s: Failed to allocate internal data structure\n", callerFunctionName);
        return NVCV_ERROR_OUT_OF_MEMORY;
    }

    if (dataType == INTEROP_DATA_TENSOR)
    {
        (*outCtxForCleanup)->tensorData = (NVCVTensorData *)*outInternalData;
    }
    else
    {
        (*outCtxForCleanup)->imageData = (NVCVImageData *)*outInternalData;
    }

    local_cupva_status = CupvaMemImport(&((*outCtxForCleanup)->pvaMemHandle), nvsciBufObjToImport, CUPVA_READ_WRITE);
    if (local_cupva_status != CUPVA_ERROR_NONE)
    {
        printf("%s: CupvaMemImport for NvSciBufObj failed (%d)\n", callerFunctionName, local_cupva_status);
        return NVCV_ERROR_DEVICE;
    }

    return NVCV_SUCCESS;
}

// Utility function to import NvSciBufObj to PVA, prepare NVCVTensorData, and wrap it into an NVCVTensorHandle
NVCVStatus nvcvTensorWrapNvSciBuf(const NVCVTensorRequirements *tensorRequirements, NvSciBufObj nvsciBufObj,
                                  NVCVTensorHandle *outTensorHandle)
{
    NVCVStatus nvcv_status = NVCV_SUCCESS;

    InteropCleanupCtx *ctxForCleanup   = NULL;
    NVCVTensorData *internalTensorData = NULL;

    nvcv_status = setup_nvsci_import_resources(nvsciBufObj, sizeof(NVCVTensorData), INTEROP_DATA_TENSOR, &ctxForCleanup,
                                               (void **)&internalTensorData, "nvcvTensorWrapNvSciBuf");
    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    // Populate NVCVTensorData (Tensor-specific part)
    nvcv_status = InteropPopulateTensorData(internalTensorData, tensorRequirements, ctxForCleanup->pvaMemHandle);
    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorWrapDataConstruct(internalTensorData, (NVCVTensorDataCleanupFunc)InteropUnifiedResourceCleanup,
                                    ctxForCleanup, outTensorHandle),
        nvcv_status, nvcv_wrap_failed);

    ctxForCleanup      = NULL;
    internalTensorData = NULL;

    return NVCV_SUCCESS;

nvcv_wrap_failed:
    return InteropPerformCleanupOnFailure(nvcv_status, ctxForCleanup, (void *)internalTensorData);
}

NVCVStatus nvcvImageWrapNvSciBuf(const NVCVImageRequirements *imageRequirements, NvSciBufObj nvsciBufObj,
                                 NVCVImageHandle *outImageHandle)
{
    NVCVStatus nvcv_status = NVCV_SUCCESS;

    InteropCleanupCtx *ctxForCleanup = NULL;
    NVCVImageData *internalImageData = NULL;

    nvcv_status = setup_nvsci_import_resources(nvsciBufObj, sizeof(NVCVImageData), INTEROP_DATA_IMAGE, &ctxForCleanup,
                                               (void **)&internalImageData, "nvcvImageWrapNvSciBuf");
    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    // Populate NVCVImageData (Image-specific part)
    nvcv_status = InteropPopulateImageData(internalImageData, imageRequirements, ctxForCleanup->pvaMemHandle);
    if (nvcv_status != NVCV_SUCCESS)
    {
        goto nvcv_wrap_failed;
    }

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageWrapDataConstruct(internalImageData, (NVCVImageDataCleanupFunc)InteropUnifiedResourceCleanup,
                                   ctxForCleanup, outImageHandle),
        nvcv_status, nvcv_wrap_failed);

    ctxForCleanup     = NULL;
    internalImageData = NULL;

    return NVCV_SUCCESS;

nvcv_wrap_failed:
    return InteropPerformCleanupOnFailure(nvcv_status, ctxForCleanup, (void *)internalImageData);
}

static NvSciError reconcileAndAllocNvSciBuf(NvSciBufObj *bufobj, NvSciBufAttrList pvaAttrList,
                                            NvSciBufAttrList cpuAttrList)
{
    NvSciError err                        = 0;
    NvSciBufAttrList unreconciledLists[2] = {NULL};
    NvSciBufAttrList reconciled_attrlist  = NULL;
    NvSciBufAttrList conflictattrlist     = NULL;

    CupvaMemFillAttributes(pvaAttrList);

    unreconciledLists[0] = pvaAttrList;
    unreconciledLists[1] = cpuAttrList;
    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListReconcile(unreconciledLists, 2U, &reconciled_attrlist, &conflictattrlist),
                           err, ReconcileFailed);
    NVSCI_CHECK_ERROR_GOTO(NvSciBufObjAlloc(reconciled_attrlist, bufobj), err, ReconcileFailed);

ReconcileFailed:
    NvSciBufAttrListFree(pvaAttrList);
    NvSciBufAttrListFree(cpuAttrList);
    NvSciBufAttrListFree(reconciled_attrlist);
    NvSciBufAttrListFree(conflictattrlist);
    return err;
}

NvSciError pvaCreateNvSciBufRawBuffer(NvSciBufObj *bufobj, NvSciBufModule sciBufModule, int64_t size)
{
    NvSciError err                        = 0;
    NvSciBufAttrList pvaAttrList          = NULL;
    NvSciBufAttrList cpuAttrList          = NULL;
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType bufTypes[]               = {NvSciBufType_RawBuffer};
    bool cpu_access                       = true;

    // Setup CPU attribute list for RawBuffer
    NvSciBufAttrKeyValuePair cpu_attr_kvp[] = {{NvSciBufGeneralAttrKey_Types, &bufTypes, sizeof(bufTypes)},
                                               {NvSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
                                               {NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm)},
                                               {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpu_access, sizeof(cpu_access)}};
    const size_t num_cpu_kvp                = 4;
    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListCreate(sciBufModule, &cpuAttrList), err, CreateNvSciBufFailed);
    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListSetAttrs(cpuAttrList, cpu_attr_kvp, num_cpu_kvp), err, CreateNvSciBufFailed);

    // Setup PVA attribute list for RawBuffer
    NvSciBufAttrKeyValuePair pva_attr_kvp[] = {{NvSciBufGeneralAttrKey_Types, (void *)&bufTypes, sizeof(bufTypes)},
                                               {NvSciBufRawBufferAttrKey_Size, (void *)&size, sizeof(size)}};
    const size_t num_pva_kvp                = 2;
    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListCreate(sciBufModule, &pvaAttrList), err, CreateNvSciBufFailed);
    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListSetAttrs(pvaAttrList, pva_attr_kvp, num_pva_kvp), err, CreateNvSciBufFailed);

    // Reconcile and allocate (helper handles cleanup of attr lists)
    return reconcileAndAllocNvSciBuf(bufobj, pvaAttrList, cpuAttrList);

CreateNvSciBufFailed:
    NvSciBufAttrListFree(pvaAttrList);
    NvSciBufAttrListFree(cpuAttrList);
    return err;
}

NvSciError pvaCreateNvSciBufTensor(NvSciBufObj *bufobj, NvSciBufModule sciBufModule, uint32_t numDims,
                                   const uint64_t *dims, NvSciBufAttrValDataType dataType,
                                   const uint32_t *alignmentPerDim, uint64_t baseAddrAlign,
                                   NvSciBufAttrValColorFmt pixelFormat)
{
    NvSciError err                        = 0;
    NvSciBufAttrList pvaAttrList          = NULL;
    NvSciBufAttrList cpuAttrList          = NULL;
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType bufTypes[]               = {NvSciBufType_Tensor};
    bool cpu_access                       = true;

    // Alignment per dimension (mandatory for Tensor-only reconciliation)
    // Values must be power of 2; use default alignment of 1 if user passes NULL
    uint32_t defaultAlignment[NV_SCI_BUF_TENSOR_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    const uint32_t *alignment = (alignmentPerDim != NULL) ? alignmentPerDim : defaultAlignment;

    // Count mandatory attributes: Types, NumDims, SizePerDim, AlignmentPerDim, DataType, RequiredPerm, NeedCpuAccess
    size_t num_cpu_kvp = 7;
    size_t num_pva_kvp = 5;

    // Add optional attributes if specified
    if (baseAddrAlign > 0)
    {
        num_cpu_kvp++;
        num_pva_kvp++;
    }
    if (pixelFormat > 0)
    {
        num_cpu_kvp++;
        num_pva_kvp++;
    }

    // Allocate attribute arrays dynamically
    NvSciBufAttrKeyValuePair *cpu_attr_kvp = malloc(num_cpu_kvp * sizeof(NvSciBufAttrKeyValuePair));
    NvSciBufAttrKeyValuePair *pva_attr_kvp = malloc(num_pva_kvp * sizeof(NvSciBufAttrKeyValuePair));
    if (cpu_attr_kvp == NULL || pva_attr_kvp == NULL)
    {
        free(cpu_attr_kvp);
        free(pva_attr_kvp);
        return NvSciError_InsufficientMemory;
    }

    // Setup CPU attribute list for Tensor - mandatory attributes
    size_t cpu_idx              = 0;
    cpu_attr_kvp[cpu_idx].key   = NvSciBufGeneralAttrKey_Types;
    cpu_attr_kvp[cpu_idx].value = &bufTypes;
    cpu_attr_kvp[cpu_idx++].len = sizeof(bufTypes);
    cpu_attr_kvp[cpu_idx].key   = NvSciBufTensorAttrKey_NumDims;
    cpu_attr_kvp[cpu_idx].value = &numDims;
    cpu_attr_kvp[cpu_idx++].len = sizeof(numDims);
    cpu_attr_kvp[cpu_idx].key   = NvSciBufTensorAttrKey_SizePerDim;
    cpu_attr_kvp[cpu_idx].value = dims;
    cpu_attr_kvp[cpu_idx++].len = numDims * sizeof(uint64_t);
    cpu_attr_kvp[cpu_idx].key   = NvSciBufTensorAttrKey_AlignmentPerDim;
    cpu_attr_kvp[cpu_idx].value = alignment;
    cpu_attr_kvp[cpu_idx++].len = numDims * sizeof(uint32_t);
    cpu_attr_kvp[cpu_idx].key   = NvSciBufTensorAttrKey_DataType;
    cpu_attr_kvp[cpu_idx].value = &dataType;
    cpu_attr_kvp[cpu_idx++].len = sizeof(dataType);
    cpu_attr_kvp[cpu_idx].key   = NvSciBufGeneralAttrKey_RequiredPerm;
    cpu_attr_kvp[cpu_idx].value = &access_perm;
    cpu_attr_kvp[cpu_idx++].len = sizeof(access_perm);
    cpu_attr_kvp[cpu_idx].key   = NvSciBufGeneralAttrKey_NeedCpuAccess;
    cpu_attr_kvp[cpu_idx].value = &cpu_access;
    cpu_attr_kvp[cpu_idx++].len = sizeof(cpu_access);

    // CPU optional attributes
    if (baseAddrAlign > 0)
    {
        cpu_attr_kvp[cpu_idx].key   = NvSciBufTensorAttrKey_BaseAddrAlign;
        cpu_attr_kvp[cpu_idx].value = &baseAddrAlign;
        cpu_attr_kvp[cpu_idx++].len = sizeof(baseAddrAlign);
    }
    if (pixelFormat > 0)
    {
        cpu_attr_kvp[cpu_idx].key   = NvSciBufTensorAttrKey_PixelFormat;
        cpu_attr_kvp[cpu_idx].value = &pixelFormat;
        cpu_attr_kvp[cpu_idx++].len = sizeof(pixelFormat);
    }

    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListCreate(sciBufModule, &cpuAttrList), err, CreateNvSciBufTensorFailed);
    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListSetAttrs(cpuAttrList, cpu_attr_kvp, num_cpu_kvp), err,
                           CreateNvSciBufTensorFailed);

    // Setup PVA attribute list for Tensor - mandatory attributes
    size_t pva_idx              = 0;
    pva_attr_kvp[pva_idx].key   = NvSciBufGeneralAttrKey_Types;
    pva_attr_kvp[pva_idx].value = (void *)&bufTypes;
    pva_attr_kvp[pva_idx++].len = sizeof(bufTypes);
    pva_attr_kvp[pva_idx].key   = NvSciBufTensorAttrKey_NumDims;
    pva_attr_kvp[pva_idx].value = (void *)&numDims;
    pva_attr_kvp[pva_idx++].len = sizeof(numDims);
    pva_attr_kvp[pva_idx].key   = NvSciBufTensorAttrKey_SizePerDim;
    pva_attr_kvp[pva_idx].value = (void *)dims;
    pva_attr_kvp[pva_idx++].len = numDims * sizeof(uint64_t);
    pva_attr_kvp[pva_idx].key   = NvSciBufTensorAttrKey_AlignmentPerDim;
    pva_attr_kvp[pva_idx].value = (void *)alignment;
    pva_attr_kvp[pva_idx++].len = numDims * sizeof(uint32_t);
    pva_attr_kvp[pva_idx].key   = NvSciBufTensorAttrKey_DataType;
    pva_attr_kvp[pva_idx].value = (void *)&dataType;
    pva_attr_kvp[pva_idx++].len = sizeof(dataType);

    // PVA optional attributes
    if (baseAddrAlign > 0)
    {
        pva_attr_kvp[pva_idx].key   = NvSciBufTensorAttrKey_BaseAddrAlign;
        pva_attr_kvp[pva_idx].value = (void *)&baseAddrAlign;
        pva_attr_kvp[pva_idx++].len = sizeof(baseAddrAlign);
    }
    if (pixelFormat > 0)
    {
        pva_attr_kvp[pva_idx].key   = NvSciBufTensorAttrKey_PixelFormat;
        pva_attr_kvp[pva_idx].value = (void *)&pixelFormat;
        pva_attr_kvp[pva_idx++].len = sizeof(pixelFormat);
    }

    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListCreate(sciBufModule, &pvaAttrList), err, CreateNvSciBufTensorFailed);
    NVSCI_CHECK_ERROR_GOTO(NvSciBufAttrListSetAttrs(pvaAttrList, pva_attr_kvp, num_pva_kvp), err,
                           CreateNvSciBufTensorFailed);

    free(cpu_attr_kvp);
    free(pva_attr_kvp);

    // Reconcile and allocate (helper handles cleanup of attr lists)
    return reconcileAndAllocNvSciBuf(bufobj, pvaAttrList, cpuAttrList);

CreateNvSciBufTensorFailed:
    free(cpu_attr_kvp);
    free(pva_attr_kvp);
    NvSciBufAttrListFree(pvaAttrList);
    NvSciBufAttrListFree(cpuAttrList);
    return err;
}

// Helper function to map NvSciBuf data type to NVCV data type
static NVCVDataType mapNvSciBufDataTypeToNVCV(NvSciBufAttrValDataType nvsciBufType)
{
    switch (nvsciBufType)
    {
    case NvSciDataType_Uint8:
        return NVCV_DATA_TYPE_U8;
    case NvSciDataType_Int8:
        return NVCV_DATA_TYPE_S8;
    case NvSciDataType_Uint16:
        return NVCV_DATA_TYPE_U16;
    case NvSciDataType_Int16:
        return NVCV_DATA_TYPE_S16;
    case NvSciDataType_Uint32:
        return NVCV_DATA_TYPE_U32;
    case NvSciDataType_Int32:
        return NVCV_DATA_TYPE_S32;
    case NvSciDataType_Float32:
        return NVCV_DATA_TYPE_F32;
    default:
        return NVCV_DATA_TYPE_NONE;
    }
}

NVCVStatus nvcvTensorImportRequirementsFromNvSci(NvSciBufObj nvsciBufObj, NVCVTensorLayout tensorLayout,
                                                 NVCVTensorRequirements *tensorRequirements)
{
    if (nvsciBufObj == NULL || tensorRequirements == NULL)
    {
        return NVCV_ERROR_INVALID_ARGUMENT;
    }

    NvSciError sciErr;
    NvSciBufAttrList attrList = NULL;

    // Get attribute list from the buffer object
    sciErr = NvSciBufObjGetAttrList(nvsciBufObj, &attrList);
    if (sciErr != NvSciError_Success)
    {
        printf("nvcvTensorImportRequirementsFromNvSci: NvSciBufObjGetAttrList failed (%d)\n", sciErr);
        return NVCV_ERROR_INTERNAL;
    }

    // Query tensor attributes from the buffer
    uint32_t numDims                                   = 0;
    uint64_t sizePerDim[NV_SCI_BUF_TENSOR_MAX_DIMS]    = {0};
    uint64_t stridesPerDim[NV_SCI_BUF_TENSOR_MAX_DIMS] = {0};
    NvSciBufAttrValDataType dataType                   = 0;
    uint64_t baseAddrAlign                             = 0;

    NvSciBufAttrKeyValuePair queryAttrs[] = {{NvSciBufTensorAttrKey_NumDims, NULL, 0},
                                             {NvSciBufTensorAttrKey_SizePerDim, NULL, 0},
                                             {NvSciBufTensorAttrKey_StridesPerDim, NULL, 0},
                                             {NvSciBufTensorAttrKey_DataType, NULL, 0},
                                             {NvSciBufTensorAttrKey_BaseAddrAlign, NULL, 0}};

    sciErr = NvSciBufAttrListGetAttrs(attrList, queryAttrs, 5);
    if (sciErr != NvSciError_Success)
    {
        printf("nvcvTensorImportRequirementsFromNvSci: NvSciBufAttrListGetAttrs failed (%d)\n", sciErr);
        return NVCV_ERROR_INTERNAL;
    }

    // Extract values from query results
    numDims = *(const uint32_t *)queryAttrs[0].value;
    memcpy(sizePerDim, queryAttrs[1].value, numDims * sizeof(uint64_t));
    memcpy(stridesPerDim, queryAttrs[2].value, numDims * sizeof(uint64_t));
    dataType      = *(const NvSciBufAttrValDataType *)queryAttrs[3].value;
    baseAddrAlign = *(const uint64_t *)queryAttrs[4].value;

    // Map NvSciBuf data type to NVCV data type
    NVCVDataType nvcvDataType = mapNvSciBufDataTypeToNVCV(dataType);
    if (nvcvDataType == NVCV_DATA_TYPE_NONE)
    {
        printf("nvcvTensorImportRequirementsFromNvSci: Unsupported data type (%d)\n", dataType);
        return NVCV_ERROR_NOT_IMPLEMENTED;
    }

    // Populate NVCVTensorRequirements
    tensorRequirements->dtype  = nvcvDataType;
    tensorRequirements->layout = tensorLayout;
    tensorRequirements->rank   = (int32_t)numDims;

    // Copy shape (size per dimension)
    for (uint32_t i = 0; i < numDims && i < NVCV_TENSOR_MAX_RANK; i++)
    {
        tensorRequirements->shape[i] = (int64_t)sizePerDim[i];
    }

    // Copy strides
    for (uint32_t i = 0; i < numDims && i < NVCV_TENSOR_MAX_RANK; i++)
    {
        tensorRequirements->strides[i] = (int64_t)stridesPerDim[i];
    }

    // Set alignment from buffer's base address alignment
    tensorRequirements->alignBytes = (int32_t)baseAddrAlign;

    // Initialize memory requirements to zero for safety
    memset(&tensorRequirements->mem, 0, sizeof(tensorRequirements->mem));

    return NVCV_SUCCESS;
}

NvSciError pvaCreateNvSciSyncObj(NvSciSyncObj *syncObj, NvSciSyncModule sciSyncModule, cupvaSyncClientType_t pvaPerm,
                                 NvSciSyncAccessPerm cpuPerm)
{
    NvSciError err                         = 0;
    NvSciSyncAttrList cpuAttrList          = NULL;
    NvSciSyncAttrList pvaAttrList          = NULL;
    NvSciSyncAttrList unreconciledLists[2] = {NULL};
    NvSciSyncAttrKeyValuePair keyValues[2];
    bool cpuAccess                    = true;
    NvSciSyncAttrList reconciledList  = NULL;
    NvSciSyncAttrList newConflictList = NULL;
    NVSCI_CHECK_ERROR_GOTO(NvSciSyncAttrListCreate(sciSyncModule, &cpuAttrList), err, CreateNvSciSyncObjFailed);
    NVSCI_CHECK_ERROR_GOTO(NvSciSyncAttrListCreate(sciSyncModule, &pvaAttrList), err, CreateNvSciSyncObjFailed);

    keyValues[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValues[0].value   = (void *)&cpuAccess;
    keyValues[0].len     = sizeof(cpuAccess);
    keyValues[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValues[1].value   = (void *)&cpuPerm;
    keyValues[1].len     = sizeof(cpuPerm);
    NVSCI_CHECK_ERROR_GOTO(NvSciSyncAttrListSetAttrs(cpuAttrList, keyValues, 2), err, CreateNvSciSyncObjFailed);

    CupvaSyncObjFillAttributes(&pvaAttrList, pvaPerm);

    unreconciledLists[0] = cpuAttrList;
    unreconciledLists[1] = pvaAttrList;
    NVSCI_CHECK_ERROR_GOTO(NvSciSyncAttrListReconcile(unreconciledLists, 2, &reconciledList, &newConflictList), err,
                           CreateNvSciSyncObjFailed);

    NVSCI_CHECK_ERROR_GOTO(NvSciSyncObjAlloc(reconciledList, syncObj), err, CreateNvSciSyncObjFailed);
CreateNvSciSyncObjFailed:
    NvSciSyncAttrListFree(pvaAttrList);
    NvSciSyncAttrListFree(cpuAttrList);
    NvSciSyncAttrListFree(reconciledList);
    NvSciSyncAttrListFree(newConflictList);
    return err;
}