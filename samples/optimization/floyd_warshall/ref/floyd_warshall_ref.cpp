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

#include "floyd_warshall_ref.h"

#include <nvcv/Tensor.hpp>

#include <OpFloydWarshall.h>
#include <nvcv/Status.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace {

void generateRandomFloydWarshallInputImpl(NVCVTensorHandle lenTensor, NVCVTensorHandle nextTensor, bool directed,
                                          bool connected)
{
    NVCVTensorData lenTensorData;
    nvcvTensorExportData(lenTensor, &lenTensorData);
    uint16_t *lenTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&lenTensorPtr, (void *)lenTensorData.buffer.strided.basePtr);

    NVCVTensorData nextTensorData;
    nvcvTensorExportData(nextTensor, &nextTensorData);
    uint16_t *nextTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&nextTensorPtr, (void *)nextTensorData.buffer.strided.basePtr);

    int32_t num       = lenTensorData.shape[0];
    int32_t linePitch = lenTensorData.buffer.strided.strides[0] / sizeof(uint16_t);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> distWeight(1, std::numeric_limits<uint16_t>::max() - 1);
    const uint16_t INF = std::numeric_limits<uint16_t>::max();
    const uint16_t NIL = std::numeric_limits<uint16_t>::max();

    // initialize length and next tensors
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < num; j++)
        {
            if (i == j)
            {
                lenTensorPtr[i * linePitch + j]  = 0;
                nextTensorPtr[i * linePitch + j] = i;
            }
            else
            {
                lenTensorPtr[i * linePitch + j]  = INF;
                nextTensorPtr[i * linePitch + j] = NIL;
            }
        }
    }

    // generate connected graph
    if (connected)
    {
        std::vector<int> nodes(num);
        std::iota(nodes.begin(), nodes.end(), 0);
        std::shuffle(nodes.begin(), nodes.end(), gen);

        // create a random spanning tree
        for (int i = 1; i < num; i++)
        {
            std::uniform_int_distribution<int> distPrev(0, i - 1);
            int u = nodes[i];
            int v = nodes[distPrev(gen)];

            uint16_t w = distWeight(gen);

            lenTensorPtr[u * linePitch + v]  = w;
            nextTensorPtr[u * linePitch + v] = v;

            if (!directed)
            {
                lenTensorPtr[v * linePitch + u]  = w;
                nextTensorPtr[v * linePitch + u] = u;
            }
        }
    }

    // add extra random edges
    int extraEdges = connected ? num : num - 2; // ensure disconnection
    std::uniform_int_distribution<int> distNode(0, num - 1);
    for (int k = 0; k < extraEdges; k++)
    {
        int u = distNode(gen);
        int v = distNode(gen);

        if (u != v && lenTensorPtr[u * linePitch + v] == INF)
        {
            uint16_t w = distWeight(gen);

            lenTensorPtr[u * linePitch + v]  = w;
            nextTensorPtr[u * linePitch + v] = v;

            if (!directed)
            {
                lenTensorPtr[v * linePitch + u]  = w;
                nextTensorPtr[v * linePitch + u] = u;
            }
        }
    }
}

void copyInputTensorImpl(NVCVTensorHandle srcTensor, NVCVTensorHandle dstTensor)
{
    NVCVTensorData srcTensorData;
    nvcvTensorExportData(srcTensor, &srcTensorData);
    uint16_t *srcTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&srcTensorPtr, (void *)srcTensorData.buffer.strided.basePtr);

    NVCVTensorData dstTensorData;
    nvcvTensorExportData(dstTensor, &dstTensorData);
    uint16_t *dstTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&dstTensorPtr, (void *)dstTensorData.buffer.strided.basePtr);

    int32_t height    = srcTensorData.shape[0];
    int32_t width     = srcTensorData.shape[1];
    int32_t linePitch = srcTensorData.buffer.strided.strides[0] / sizeof(uint16_t);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            dstTensorPtr[i * linePitch + j] = srcTensorPtr[i * linePitch + j];
        }
    }
}

int32_t checkFloydWarshallOutputImpl(NVCVTensorHandle refLenTensor, NVCVTensorHandle refNextTensor,
                                     NVCVTensorHandle pvaLenTensor, NVCVTensorHandle pvaNextTensor)
{
    NVCVTensorData refLenTensorData;
    nvcvTensorExportData(refLenTensor, &refLenTensorData);
    uint16_t *refLenTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&refLenTensorPtr, (void *)refLenTensorData.buffer.strided.basePtr);

    NVCVTensorData refNextTensorData;
    nvcvTensorExportData(refNextTensor, &refNextTensorData);
    uint16_t *refNextTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&refNextTensorPtr, (void *)refNextTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaLenTensorData;
    nvcvTensorExportData(pvaLenTensor, &pvaLenTensorData);
    uint16_t *pvaLenTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaLenTensorPtr, (void *)pvaLenTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaNextTensorData;
    nvcvTensorExportData(pvaNextTensor, &pvaNextTensorData);
    uint16_t *pvaNextTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&pvaNextTensorPtr, (void *)pvaNextTensorData.buffer.strided.basePtr);

    int32_t height    = refLenTensorData.shape[0];
    int32_t width     = refLenTensorData.shape[1];
    int32_t linePitch = refLenTensorData.buffer.strided.strides[0] / sizeof(uint16_t);

    int32_t errCnt = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            uint16_t refLen = refLenTensorPtr[i * linePitch + j];
            uint16_t pvaLen = pvaLenTensorPtr[i * linePitch + j];
            if (refLen != pvaLen)
            {
                printf("Mismatch on lenTensor[%d][%d]: ref = %u, pva = %u\n", i, j, refLen, pvaLen);
                errCnt++;
            }

            uint16_t refNext = refNextTensorPtr[i * linePitch + j];
            uint16_t pvaNext = pvaNextTensorPtr[i * linePitch + j];
            if (refNext != pvaNext)
            {
                printf("Mismatch on nextTensor[%d][%d]: ref = %u, pva = %u\n", i, j, refNext, pvaNext);
                errCnt++;
            }
        }
    }

    if (errCnt > 0)
    {
        printf("Found %d errors in Floyd-Warshall output. \n", errCnt);
        return 1;
    }

    return errCnt;
}

int floydWarshallRefImpl(NVCVTensorHandle lenTensor, NVCVTensorHandle nextTensor)
{
    NVCVTensorData lenTensorData;
    nvcvTensorExportData(lenTensor, &lenTensorData);
    uint16_t *lenTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&lenTensorPtr, (void *)lenTensorData.buffer.strided.basePtr);

    NVCVTensorData nextTensorData;
    nvcvTensorExportData(nextTensor, &nextTensorData);
    uint16_t *nextTensorPtr = NULL;
    CupvaMemGetHostPointer((void **)&nextTensorPtr, (void *)nextTensorData.buffer.strided.basePtr);

    int32_t height    = lenTensorData.shape[0];
    int32_t width     = lenTensorData.shape[1];
    int32_t linePitch = lenTensorData.buffer.strided.strides[0] / sizeof(uint16_t);
    int32_t num       = width;

    for (int k = 0; k < num; k++)
    {
        for (int i = 0; i < num; i++)
        {
            for (int j = 0; j < num; j++)
            {
                uint16_t len_ik = lenTensorPtr[i * linePitch + k];
                uint16_t len_kj = lenTensorPtr[k * linePitch + j];
                uint16_t len_ij = lenTensorPtr[i * linePitch + j];

                if (static_cast<uint32_t>(len_ij) > static_cast<uint32_t>(len_ik) + static_cast<uint32_t>(len_kj))
                {
                    lenTensorPtr[i * linePitch + j]  = len_ik + len_kj;
                    nextTensorPtr[i * linePitch + j] = nextTensorPtr[i * linePitch + k];
                }
            }
        }
    }

    return 0;
}

} // namespace

void generateRandomFloydWarshallInput(NVCVTensorHandle lenTensor, NVCVTensorHandle nextTensor, bool directed,
                                      bool connected)
{
    generateRandomFloydWarshallInputImpl(lenTensor, nextTensor, directed, connected);
}

void copyInputTensor(NVCVTensorHandle srcTensor, NVCVTensorHandle dstTensor)
{
    copyInputTensorImpl(srcTensor, dstTensor);
}

int32_t checkFloydWarshallOutput(NVCVTensorHandle refLenTensor, NVCVTensorHandle refNextTensor,
                                 NVCVTensorHandle pvaLenTensor, NVCVTensorHandle pvaNextTensor)
{
    return checkFloydWarshallOutputImpl(refLenTensor, refNextTensor, pvaLenTensor, pvaNextTensor);
}

void floydWarshallRef(NVCVTensorHandle refLenTensor, NVCVTensorHandle refNextTensor)
{
    floydWarshallRefImpl(refLenTensor, refNextTensor);
}
