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

#include "HostPointer.h"

#include <cupva_host.h>
#include <stddef.h>

void *PvaGetTensorHostPointer(NVCVTensorHandle handle)
{
    void *hostPtr = NULL;
    if (nvcvTensorGetUserPointer(handle, &hostPtr) == NVCV_SUCCESS && hostPtr != NULL)
    {
        return hostPtr;
    }

    NVCVTensorData tensorData;
    if (nvcvTensorExportData(handle, &tensorData) != NVCV_SUCCESS)
    {
        return NULL;
    }

    if (CupvaMemGetHostPointer(&hostPtr, tensorData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
    {
        return NULL;
    }

    return hostPtr;
}

void *PvaGetImageHostPointer(NVCVImageHandle handle, int32_t plane)
{
    NVCVImageData imageData;
    if (nvcvImageExportData(handle, &imageData) != NVCV_SUCCESS)
    {
        return NULL;
    }

    if (plane < 0 || plane >= imageData.buffer.strided.numPlanes)
    {
        return NULL;
    }

    NVCVByte *planePtr = imageData.buffer.strided.planes[plane].basePtr;

    void *hostPtr = NULL;
    if (nvcvImageGetUserPointer(handle, &hostPtr) == NVCV_SUCCESS && hostPtr != NULL)
    {
        // The attached pointer is the host base of the whole allocation.
        return (NVCVByte *)hostPtr + (planePtr - imageData.buffer.strided.planes[0].basePtr);
    }

    if (CupvaMemGetHostPointer(&hostPtr, planePtr) != CUPVA_ERROR_NONE)
    {
        return NULL;
    }

    return hostPtr;
}
