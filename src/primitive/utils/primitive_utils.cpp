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

#include "primitive_utils.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace pva_solutions { namespace utils {

template<typename T>
void prepareKernelForConv2dVpu(T *knl, T *knl_opt_out, int32_t ksize)
{
    // ------------------------------------------------ //
    // the knl preparation is more tricky here:         //
    // 1. align KW to multiple of 4 (pad 0 to left)     //
    // 2. for knl0, pad 1 zero-row to bottom            //
    //    for knl1, pad 1 zero-row to top               //
    // 3. 8-element-interleave them to a double-width   //
    //    for example, if KW = 3, the first row is:     //
    //    knl0 row = {c0, c1, c2, 0, c3, c4, c5, 0}     //
    //    knl1 row = {0, 0, 0, 0, s0, s1, s2, 0}        //
    //    knl  row = {c0, c1, c2, 0, c3, c4, c5, 0 \    //
    //                0, 0, 0, 0, s0, s1, s2, 0}        //
    // ------------------------------------------------ //
    int kw = ksize, kh = ksize;
    int kwa = ((kw - 1) / 4 + 1) * 4;
    int kha = ((kh - 1) / 2 + 1) * 2;

    std::vector<T> knl0_opt_vec(kwa * kha);
    std::vector<T> knl1_opt_vec(kwa * kha);

    T *knl0_opt_ptr = knl0_opt_vec.data();
    T *knl1_opt_ptr = knl1_opt_vec.data();

    memset(knl0_opt_ptr, 0, kwa * kha * sizeof(T));
    memset(knl1_opt_ptr, 0, kwa * kha * sizeof(T));

    for (int y = 0; y < kh; y++)
    {
        for (int x = 0; x < kw; x++)
        {
            int knl_addr  = y * kw + x;
            int knl0_addr = y * kwa + x;
            int knl1_addr = (y + 1) * kwa + x;

            T knl_data                  = *(knl + knl_addr);
            *(knl0_opt_ptr + knl0_addr) = knl_data;
            *(knl1_opt_ptr + knl1_addr) = knl_data;
        }
    }

    for (int y = 0; y < (kha / 2); y++)
    {
        for (int x = 0; x < (kwa * 2); x++)
        {
            int src_addr  = y * kwa * 2 + x;
            int dst0_addr = y * 4 * kwa + x;
            int dst1_addr = y * 4 * kwa + kwa * 2 + x;

            *(knl_opt_out + dst0_addr) = *(knl0_opt_ptr + src_addr);
            *(knl_opt_out + dst1_addr) = *(knl1_opt_ptr + src_addr);
        }
    }
}

// Explicit template instantiations
template void prepareKernelForConv2dVpu<int8_t>(int8_t *, int8_t *, int32_t);
template void prepareKernelForConv2dVpu<uint8_t>(uint8_t *, uint8_t *, int32_t);
template void prepareKernelForConv2dVpu<int16_t>(int16_t *, int16_t *, int32_t);
template void prepareKernelForConv2dVpu<uint16_t>(uint16_t *, uint16_t *, int32_t);

}} // namespace pva_solutions::utils