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

#include "radar_transpose_ref.h"

#include <cupva_host.hpp>

#include <nvcv/Tensor.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Get CPU-accessible host pointer from a PVA DRAM tensor. */
static int32_t *getHostPtr(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData data;
    nvcvTensorExportData(tensorHandle, &data);
    int32_t *ptr = nullptr;
    CupvaMemGetHostPointer(reinterpret_cast<void **>(&ptr), static_cast<void *>(data.buffer.strided.basePtr));
    return ptr;
}

/* Extract nRange/nRx/nDoppler from an HCW (or NHCW) input tensor. */
static void getInputDims(NVCVTensorHandle tensorHandle, int32_t &nRange, int32_t &nRx, int32_t &nDoppler)
{
    NVCVTensorData data;
    nvcvTensorExportData(tensorHandle, &data);

    if (data.rank == 3)
    {
        nRange   = static_cast<int32_t>(data.shape[0]);
        nRx      = static_cast<int32_t>(data.shape[1]);
        nDoppler = static_cast<int32_t>(data.shape[2]);
    }
    else // rank == 4 NHCW
    {
        nRange   = static_cast<int32_t>(data.shape[1]);
        nRx      = static_cast<int32_t>(data.shape[2]);
        nDoppler = static_cast<int32_t>(data.shape[3]);
    }
}

/* Total number of int32 values in a tensor (shape product × 2 for 2S32 complex). */
static int32_t totalInt32Elems(NVCVTensorHandle tensorHandle)
{
    NVCVTensorData data;
    nvcvTensorExportData(tensorHandle, &data);

    int32_t total = 1;
    for (int32_t i = 0; i < data.rank; i++)
    {
        total *= static_cast<int32_t>(data.shape[i]);
    }
    return total * 2; // 2S32: two int32 per element
}

void generateRadarTransposeInput(NVCVTensorHandle inTensorHandle)
{
    int32_t *base  = getHostPtr(inTensorHandle);
    int32_t nElems = totalInt32Elems(inTensorHandle);

    srand(42);
    for (int32_t i = 0; i < nElems; i++)
    {
        base[i] = (rand() % 65536) - 32768;
    }
}

void radarTransposeRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle,
                       const PVARadarTransposeParams *params)
{
    const int32_t *src = getHostPtr(inTensorHandle);
    int32_t *dst       = getHostPtr(outTensorHandle);

    int32_t nRange, nRx, nDoppler;
    getInputDims(inTensorHandle, nRange, nRx, nDoppler);

    if (params->outputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        // Identity: copy src → dst
        memcpy(dst, src, static_cast<size_t>(nRange) * nRx * nDoppler * 2 * sizeof(int32_t));
    }
    else if (params->outputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        // [nRange][nRx][nDoppler] -> [nRange][nDoppler][nRx]
        for (int32_t r = 0; r < nRange; r++)
        {
            for (int32_t rx = 0; rx < nRx; rx++)
            {
                for (int32_t d = 0; d < nDoppler; d++)
                {
                    int32_t srcOff  = (r * nRx * nDoppler + rx * nDoppler + d) * 2;
                    int32_t dstOff  = (r * nDoppler * nRx + d * nRx + rx) * 2;
                    dst[dstOff + 0] = src[srcOff + 0];
                    dst[dstOff + 1] = src[srcOff + 1];
                }
            }
        }
    }
    else // PVA_FFT_LAYOUT_DOPPLER_RX_RANGE
    {
        // [nRange][nRx][nDoppler] -> [nDoppler][nRx][nRange]
        for (int32_t r = 0; r < nRange; r++)
        {
            for (int32_t rx = 0; rx < nRx; rx++)
            {
                for (int32_t d = 0; d < nDoppler; d++)
                {
                    int32_t srcOff  = (r * nRx * nDoppler + rx * nDoppler + d) * 2;
                    int32_t dstOff  = (d * nRx * nRange + rx * nRange + r) * 2;
                    dst[dstOff + 0] = src[srcOff + 0];
                    dst[dstOff + 1] = src[srcOff + 1];
                }
            }
        }
    }
}

int32_t checkRadarTransposeOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    const int32_t *ref = getHostPtr(refTensorHandle);
    const int32_t *pva = getHostPtr(pvaTensorHandle);
    int32_t nElems     = totalInt32Elems(refTensorHandle);

    int32_t nMismatch = 0;
    for (int32_t i = 0; i < nElems; i++)
    {
        if (ref[i] != pva[i])
        {
            if (nMismatch < 10)
            {
                printf("Mismatch at element %d: ref=%d pva=%d\n", i, ref[i], pva[i]);
            }
            nMismatch++;
        }
    }

    if (nMismatch > 0)
    {
        printf("Total mismatches: %d / %d\n", nMismatch, nElems);
    }
    return nMismatch;
}
