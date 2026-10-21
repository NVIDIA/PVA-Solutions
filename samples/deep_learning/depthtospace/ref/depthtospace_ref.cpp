/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "depthtospace_ref.h"

#include <TensorUtils.hpp>

#include <OpDepthToSpace.h>
#include <nvcv/TensorShape.h>

#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
template<typename T, typename = std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value>>
void generateInputImpl(NVCVTensorHandle tensorHandle, bool isRandom)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    T *tensorPtr_h = nullptr;
    CupvaMemGetHostPointer((void **)&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    int32_t const channel = static_cast<int32_t>(tensorData.shape[0]);
    int32_t const height  = static_cast<int32_t>(tensorData.shape[1]);
    int32_t const width   = static_cast<int32_t>(tensorData.shape[2]);

    // Fill through the real pitches: a flat fill over the element count writes one contiguous
    // run and leaves every later logical element uninitialised on a padded tensor -- which the
    // operator and this reference would then both read and agree on.
    int64_t const chanPitch = tensorData.buffer.strided.strides[0] / static_cast<int64_t>(sizeof(T));
    int64_t const linePitch = tensorData.buffer.strided.strides[1] / static_cast<int64_t>(sizeof(T));

    std::mt19937 gen;
    gen.seed(1);
    std::uniform_int_distribution<> distrib(0, std::numeric_limits<T>::max());

    for (int32_t c = 0; c < channel; c++)
    {
        for (int32_t y = 0; y < height; y++)
        {
            T *row = tensorPtr_h + c * chanPitch + y * linePitch;
            for (int32_t x = 0; x < width; x++)
            {
                // Keyed on the LOGICAL linear index, so the iota sequence a small tensor is
                // debugged with does not change when the tensor carries padding.
                int64_t const logicalIndex = (static_cast<int64_t>(c) * height + y) * width + x;
                row[x]                     = isRandom ? static_cast<T>(distrib(gen)) : static_cast<T>(logicalIndex);
            }
        }
    }
}

void depthToSpaceRefImpl(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, int32_t blkSz)
{
    NVCVTensorData inTensorData;
    nvcvTensorExportData(inTensorHandle, &inTensorData);

    uint8_t *input = nullptr;
    CupvaMemGetHostPointer((void **)&input, (void *)inTensorData.buffer.strided.basePtr);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);

    uint8_t *output = nullptr;
    CupvaMemGetHostPointer((void **)&output, (void *)outTensorData.buffer.strided.basePtr);

    // The following codes mimic a NumPy implementation to generate a reference DepthToSpace output.
    // The DepthToSpace's NumPy code can be written as
    // b, c, h, w = x.shape
    // tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
    // tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    // y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
    // More can be referred in https://onnx.ai/onnx/operators/onnx__DepthToSpace.html#

    NVCVTensorLayout newTensorLayout;
    // DepthToSpace only needs a fixed permutation pattern
    // Compared with the NumPy's reference code, N = 1 results in only need to permute a CHW tensor into
    // a blocksize*blocksize*(c//*blocksize**2)*height*width tensor. Or simply neglect the dim#0 and subtract
    // the above [0, 3, 4, 1, 5, 2] by 1.
    // In NVCV layout API, each character marks a dimension, so the tensor is constructed with the layout "01234"
    // and then needs to be permuted with the layout "23041"
    nvcvTensorLayoutMake("23041", &newTensorLayout);

    std::vector<int64_t> newShape(inTensorData.rank), newStrides(inTensorData.rank);
    // there is no NVCV API to permute the shape and strides simultaneously
    nvcvTensorShapePermute(inTensorData.layout, inTensorData.shape, newTensorLayout, newShape.data());
    nvcvTensorShapePermute(inTensorData.layout, inTensorData.buffer.strided.strides, newTensorLayout,
                           newStrides.data());

    pva_solutions::utils::MakeContiguous<uint8_t>(input, newShape, newStrides, output);
}

} // namespace

void generateDepthToSpaceInput(NVCVTensorHandle tensorHandle, bool isRandom)
{
    generateInputImpl<uint8_t>(tensorHandle, isRandom);
}

int32_t checkDepthToSpaceOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle)
{
    NVCVTensorData refTensorData;
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    uint8_t *ref = nullptr;
    CupvaMemGetHostPointer((void **)&ref, (void *)refTensorData.buffer.strided.basePtr);

    NVCVTensorData pvaTensorData;
    nvcvTensorExportData(pvaTensorHandle, &pvaTensorData);

    uint8_t *pva = nullptr;
    CupvaMemGetHostPointer((void **)&pva, (void *)pvaTensorData.buffer.strided.basePtr);

    int32_t channel = refTensorData.shape[0];
    int32_t height  = refTensorData.shape[1];
    int32_t width   = refTensorData.shape[2];

    int32_t err = memcmp(ref, pva, channel * width * height * sizeof(uint8_t));
    if (err)
    {
        for (int z = 0; z < channel; z++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    uint8_t refData = ref[z * height * width + y * width + x];
                    uint8_t cmpData = pva[z * height * width + y * width + x];
                    if (refData != cmpData)
                    {
                        printf("Mismatch[%d, %d, %d]: ref = %u, pva = %u\n", z, y, x, refData, cmpData);
                        return err;
                    }
                }
            }
        }
    }
    return err;
}

void depthToSpaceRef(NVCVTensorHandle inTensorHandle, NVCVTensorHandle outTensorHandle, int32_t blkSz)
{
    depthToSpaceRefImpl(inTensorHandle, outTensorHandle, blkSz);
}