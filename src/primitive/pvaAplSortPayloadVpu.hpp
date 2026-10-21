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

/**
 * @file pvaAplSortPayloadVpu.hpp
 *
 * @brief Defines SortPayload primitive template class
 */

#ifndef PVA_APL_SORT_PAYLOAD_VPU_HPP
#define PVA_APL_SORT_PAYLOAD_VPU_HPP

#include <cupva_device.h>

#include <type_traits>

namespace pvaApl {

// Define a traits struct to map DataType to specific vector types
template<typename DataType>
struct SortPayloadVectorTypes;

/**
 * @brief Vector type specialization for uint32_t.
 */
template<>
struct SortPayloadVectorTypes<uint32_t>
{
    /** @brief Vector type for uint32_t. */
    using DVTYPE = dvint;
    /** @brief Extended vector type for uint32_t. */
    using DVTYPEX = dvintx;
    /** @brief Permute vector type for uint32_t. */
    using PMT_VTYPEX = vshortx;
};

/**
 * @brief Vector type specialization for int32_t.
 */
template<>
struct SortPayloadVectorTypes<int32_t>
{
    /** @brief Vector type for int32_t. */
    using DVTYPE = dvint;
    /** @brief Extended vector type for int32_t. */
    using DVTYPEX = dvintx;
    /** @brief Permute vector type for int32_t. */
    using PMT_VTYPEX = vshortx;
};

namespace detail {

/**
 * @brief AgenCFG entries consumed by one SortPayloadVpu::bitonic_merge_init() call.
 *
 * Must be kept in step with that function.
 */
constexpr size_t SortPayloadMergeCfgCount(int vDist, bool reverse)
{
    size_t count = 0;
    while (vDist > 0)
    {
        count += reverse ? 3U : 2U;
        vDist   = (vDist >= 2) ? (vDist >> 2) : (vDist >> 1);
        reverse = false;
    }
    return count;
}

/**
 * @brief AgenCFG entries consumed by one SortPayloadVpu::bitonic_merge_transpose_init() call.
 *
 * Must be kept in step with that function.
 */
constexpr size_t SortPayloadTransposeCfgCount(int hDist)
{
    size_t count = 0;
    bool reverse = true;
    while (hDist > 0)
    {
        count += reverse ? 3U : 2U;
        hDist   = (hDist > 1) ? (hDist >> 2) : (hDist >> 1);
        reverse = false;
    }
    return count;
}

/**
 * @brief Total AgenCFG entries SortPayloadVpu::Init() writes for the given geometry.
 */
constexpr size_t SortPayloadInitCfgCount(int vecw, int height)
{
    size_t count = 2U; // even_odd_merge_sort_payload_init()
    for (int vDist = 8; vDist < height; vDist *= 2)
    {
        count += SortPayloadMergeCfgCount(vDist, true);
    }
    for (int hDist = 1; hDist < vecw / 2; hDist *= 2)
    {
        count += SortPayloadTransposeCfgCount(hDist) + SortPayloadMergeCfgCount(height / 2, false);
    }
    return count;
}

} // namespace detail

/**
 * @brief Context structure for the SortPayload primitive.
 *
 * This structure holds the configuration and scratch space
 * required by the sorting with payload primitive.
 */
struct SortPayloadContext
{
    /** @brief Array of Agen configurations. */
    AgenCFG cfgs[256];
    /** @brief Scratch size for the SortPayload primitive. */
    static constexpr size_t SCRATCH_SIZE = ((16 + 2) * 512 * sizeof(uint32_t));
    /** @brief Scratch space. */
    uint8_t scratch[SCRATCH_SIZE];
};

/**
 * @defgroup PVA_PRIMITIVE_SORT_PAYLOAD SortPayload
 * @{
 */

/**
  * @brief SortPayload primitive template class
  *
  * @tparam DataType Data Type of the input array. Must be one of: uint32_t, int32_t.
  * @tparam Size Number of elements to sort.
  *              For 32-bit data type, must be one of: 512, 1024, 2048, 4096, 8192.
  */
template<typename DataType, int Size>
class SortPayloadVpu
{
private:
    using DVTYPE     = typename SortPayloadVectorTypes<DataType>::DVTYPE;
    using DVTYPEX    = typename SortPayloadVectorTypes<DataType>::DVTYPEX;
    using PMT_VTYPEX = typename SortPayloadVectorTypes<DataType>::PMT_VTYPEX;

    template<typename AgenType>
    auto DVLOAD(AgenType &ag) -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint32_t>)
        {
            return static_cast<dvintx>(dvuint_load(ag));
        }
        else if constexpr (std::is_same_v<DataType, int32_t>)
        {
            return static_cast<dvintx>(dvint_load(ag));
        }
        else
        {
            static_assert(!std::is_same_v<DataType, DataType>, "Unsupported type for dvload");
        }
    }

    template<typename AgenType, typename VecType>
    auto DVLOAD_T2(AgenType &ag, VecType vx) -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint32_t>)
        {
            return static_cast<dvintx>(dvuint_load_perm_transp2(ag, vx));
        }
        else if constexpr (std::is_same_v<DataType, int32_t>)
        {
            return static_cast<dvintx>(dvint_load_perm_transp2(ag, vx));
        }
        else
        {
            static_assert(!std::is_same_v<DataType, DataType>, "Unsupported type for dvload");
        }
    }

    inline void VSORT2PL_INPLACE(DVTYPEX &v1, DVTYPEX &v2)
    {
        vsort2pl(v1.lo, v2.lo, v1.lo, v2.lo);
        vsort2pl(v1.hi, v2.hi, v1.hi, v2.hi);
    }

    inline void MERGE_2_WAY(DVTYPEX &v1, DVTYPEX &v2)
    {
        VSORT2PL_INPLACE(v1, v2);
    }

    inline void MERGE_4_WAY(DVTYPEX &v1, DVTYPEX &v2, DVTYPEX &v3, DVTYPEX &v4)
    {
        chess_separator();
        VSORT2PL_INPLACE(v1, v3);
        chess_separator();
        VSORT2PL_INPLACE(v2, v4);
        chess_separator();
        MERGE_2_WAY(v1, v2);
        chess_separator();
        MERGE_2_WAY(v3, v4);
    }

    void even_odd_merge_sort_payload(AgenCFG **cfgs, int height)
    {
        DVTYPEX v1, v2, v3, v4, v5, v6, v7, v8;

        agen in, out;

        in  = init_agen_from_cfg(*(*cfgs)++);
        out = init_agen_from_cfg(*(*cfgs)++);

        int niter = height >> 3;

        for (int i = 0; i < niter; i++) chess_loop_range(3, )
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in);
            v2 = DVLOAD(in);
            v3 = DVLOAD(in);
            v4 = DVLOAD(in);
            v5 = DVLOAD(in);
            v6 = DVLOAD(in);
            v7 = DVLOAD(in);
            v8 = DVLOAD(in);

            VSORT2PL_INPLACE(v1, v2);
            VSORT2PL_INPLACE(v3, v4);
            VSORT2PL_INPLACE(v5, v6);
            VSORT2PL_INPLACE(v7, v8);

            VSORT2PL_INPLACE(v1, v3);
            VSORT2PL_INPLACE(v5, v7);
            VSORT2PL_INPLACE(v2, v4);
            VSORT2PL_INPLACE(v6, v8);

            VSORT2PL_INPLACE(v2, v3);
            VSORT2PL_INPLACE(v6, v7);

            VSORT2PL_INPLACE(v1, v5);
            VSORT2PL_INPLACE(v2, v6);
            VSORT2PL_INPLACE(v3, v7);
            VSORT2PL_INPLACE(v4, v8);

            VSORT2PL_INPLACE(v3, v5);
            VSORT2PL_INPLACE(v4, v6);

            VSORT2PL_INPLACE(v2, v3);
            VSORT2PL_INPLACE(v4, v5);
            VSORT2PL_INPLACE(v6, v7);

            vstore(v1, out);
            vstore(v2, out);
            vstore(v3, out);
            vstore(v4, out);
            vstore(v5, out);
            vstore(v6, out);
            vstore(v7, out);
            vstore(v8, out);
        }
    }

    void bitonic_merge_4_way_reverse(AgenCFG **cfgs, int height)
    {
        DVTYPEX v1, v2, v3, v4;

        agen_A in1, in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(*(*cfgs)++);
        in2 = init_agen_A_from_cfg(*(*cfgs)++);
        out = init_agen_B_from_cfg(*(*cfgs)++);

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in1);
            v2 = DVLOAD(in1);
            v3 = DVLOAD(in2);
            v4 = DVLOAD(in2);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore(v1, out);
            vstore(v2, out);
            vstore(v3, out);
            vstore(v4, out);
        }
    }

    void bitonic_merge_4_way_transp(AgenCFG **cfgs, int height)
    {
        DVTYPEX v1, v2, v3, v4;

        agen in, out;

        in  = init_agen_from_cfg(*(*cfgs)++);
        out = init_agen_from_cfg(*(*cfgs)++);

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in);
            v2 = DVLOAD(in);
            v3 = DVLOAD(in);
            v4 = DVLOAD(in);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore_transp2(v1, out);
            vstore_transp2(v2, out);
            vstore_transp2(v3, out);
            vstore_transp2(v4, out);
        }
    }

    void bitonic_merge_4_way(AgenCFG **cfgs, int height)
    {
        DVTYPEX v1, v2, v3, v4;

        agen in, out;

        in  = init_agen_from_cfg(*(*cfgs)++);
        out = init_agen_from_cfg(*(*cfgs)++);

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in);
            v2 = DVLOAD(in);
            v3 = DVLOAD(in);
            v4 = DVLOAD(in);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore(v1, out);
            vstore(v2, out);
            vstore(v3, out);
            vstore(v4, out);
        }
    }

    void bitonic_merge_2_way_reverse(AgenCFG **cfgs, int height)
    {
        DVTYPEX v1, v2;

        agen_A in;
        agen_B out;

        in  = init_agen_A_from_cfg(*(*cfgs)++);
        out = init_agen_B_from_cfg(*(*cfgs)++);

        int niter = height >> 1;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in);
            v2 = DVLOAD(in);

            MERGE_2_WAY(v1, v2);

            vstore(v1, out);
            vstore(v2, out);
        }
    }

    void bitonic_merge_2_way_transp(AgenCFG **cfgs, int height)
    {
        DVTYPEX v1, v2;

        agen in, out;

        in  = init_agen_from_cfg(*(*cfgs)++);
        out = init_agen_from_cfg(*(*cfgs)++);

        int niter = height >> 1;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in);
            v2 = DVLOAD(in);

            MERGE_2_WAY(v1, v2);

            vstore_transp2(v1, out);
            vstore_transp2(v2, out);
        }
    }

    void bitonic_merge_2_way(AgenCFG **cfgs, int height)
    {
        DVTYPEX v1, v2;

        agen in, out;

        in  = init_agen_from_cfg(*(*cfgs)++);
        out = init_agen_from_cfg(*(*cfgs)++);

        int niter = height >> 1;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in);
            v2 = DVLOAD(in);

            MERGE_2_WAY(v1, v2);

            vstore(v1, out);
            vstore(v2, out);
        }
    }

    void bitonic_merge_transpose_4_way_reverse(AgenCFG **cfgs, int height, vshortx *permute_short_ptr,
                                               vshortx *permute_reverse_short_ptr)
    {
        DVTYPEX v1, v2, v3, v4;

        vshortx pmt_short, pmt_reverse_short;
        agen_A in1, in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(*(*cfgs)++);
        in2 = init_agen_A_from_cfg(*(*cfgs)++);

        out = init_agen_B_from_cfg(*(*cfgs)++);

        pmt_short         = *permute_short_ptr;
        pmt_reverse_short = *permute_reverse_short_ptr;

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_T2(in1, pmt_short);
            v2 = DVLOAD_T2(in1, pmt_short);
            v3 = DVLOAD_T2(in2, pmt_reverse_short);
            v4 = DVLOAD_T2(in2, pmt_reverse_short);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore_transp2(v1, out);
            vstore_transp2(v2, out);
            vstore_transp2(v3, out);
            vstore_transp2(v4, out);
        }
    }

    void bitonic_merge_transpose_4_way(AgenCFG **cfgs, int height, vshortx *permute_short_ptr)
    {
        DVTYPEX v1, v2, v3, v4;

        vshortx pmt_short;
        agen in, out;

        in  = init_agen_from_cfg(*(*cfgs)++);
        out = init_agen_from_cfg(*(*cfgs)++);

        pmt_short = *permute_short_ptr;

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_T2(in, pmt_short);
            v2 = DVLOAD_T2(in, pmt_short);
            v3 = DVLOAD_T2(in, pmt_short);
            v4 = DVLOAD_T2(in, pmt_short);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore_transp2(v1, out);
            vstore_transp2(v2, out);
            vstore_transp2(v3, out);
            vstore_transp2(v4, out);
        }
    }

    void bitonic_merge_transpose_2_way_reverse(AgenCFG **cfgs, int height, vshortx *permute_short_ptr,
                                               vshortx *permute_reverse_short_ptr)
    {
        DVTYPEX v1, v2;

        vshortx pmt_short, pmt_reverse_short;
        agen_A in1;
        agen_A in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(*(*cfgs)++);
        in2 = init_agen_A_from_cfg(*(*cfgs)++);

        out = init_agen_B_from_cfg(*(*cfgs)++);

        pmt_short         = *permute_short_ptr;
        pmt_reverse_short = *permute_reverse_short_ptr;

        int niter = height >> 1;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_T2(in1, pmt_short);
            v2 = DVLOAD_T2(in2, pmt_reverse_short);

            VSORT2PL_INPLACE(v1, v2);

            vstore_transp2(v1, out);
            vstore_transp2(v2, out);
        }
    }

    void bitonic_merge_transpose_2_way(AgenCFG **cfgs, int height, vshortx *permute_short_ptr)
    {
        DVTYPEX v1, v2;

        vshortx pmt_short;
        agen_A in;
        agen_B out;

        in  = init_agen_A_from_cfg(*(*cfgs)++);
        out = init_agen_B_from_cfg(*(*cfgs)++);

        pmt_short = *permute_short_ptr;

        int niter = height >> 1;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_T2(in, pmt_short);
            v2 = DVLOAD_T2(in, pmt_short);

            VSORT2PL_INPLACE(v1, v2);

            vstore_transp2(v1, out);
            vstore_transp2(v2, out);
        }
    }

    void even_odd_merge_sort_payload_init(DataType *ping, DataType *pong, AgenCFG **cfgs, int lofst, int height)
    {
        AgenWrapper wrapper;
        int vecw = pva_elementsof(DVTYPEX);

        agen in      = init((DVTYPE *)ping);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = height;
        wrapper.s1   = vecw;
        INIT_AGEN1(in, wrapper);

        agen out     = init((DVTYPE *)pong);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = height;
        wrapper.s1   = lofst;
        INIT_AGEN1(out, wrapper);

        *(*cfgs)++ = extract_agen_cfg(in);
        *(*cfgs)++ = extract_agen_cfg(out);
    }

    void bitonic_merge_transpose_n_way_reverse_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst,
                                                    int height, int h_dist, int n)
    {
        agen in1, in2, out;
        int vecw = pva_elementsof(DVTYPEX);

        DataType *src = (*it) & 1 ? ping : pong;
        DataType *dst = (*it) & 1 ? pong : ping;
        (*it)++;

        int niter1 = 1;
        int niter2 = n / 2;
        int niter3 = (vecw / 2) / h_dist / n;
        int niter4 = height / (vecw / 2);
        int niter5 = h_dist;

        AgenWrapper wrapper;
        wrapper.size = sizeof(DataType);

        in1           = init((DVTYPE *)src);
        in1.lane_ofst = lofst / vecw;
        wrapper.n1    = niter1;
        wrapper.n2    = niter2;
        wrapper.n3    = niter3;
        wrapper.n4    = niter4;
        wrapper.n5    = niter5;
        wrapper.s1    = 1;
        wrapper.s2    = h_dist * 2;
        wrapper.s3    = n * h_dist * 2;
        wrapper.s4    = vecw / 2 * lofst;
        wrapper.s5    = 2;
        INIT_AGEN5(in1, wrapper);

        *(*cfgs)++ = extract_agen_cfg(in1);

        in2           = init((DVTYPE *)(src + (height - vecw / 2) * lofst + h_dist * n * 2 - 2));
        in2.lane_ofst = lofst / vecw;
        wrapper.s1    = 1;
        wrapper.s2    = -h_dist * 2;
        wrapper.s3    = n * h_dist * 2;
        wrapper.s4    = -vecw / 2 * lofst;
        wrapper.s5    = -2;
        INIT_AGEN5(in2, wrapper);

        *(*cfgs)++ = extract_agen_cfg(in2);

        niter1 = 1;
        niter2 = (vecw / 2) / h_dist;
        niter3 = height / (vecw / 2);
        niter4 = h_dist;

        out           = init((DVTYPE *)dst);
        out.lane_ofst = lofst / vecw;
        wrapper.n1    = niter1;
        wrapper.n2    = niter2;
        wrapper.n3    = niter3;
        wrapper.n4    = niter4;
        wrapper.s1    = 1;
        wrapper.s2    = h_dist * 2;
        wrapper.s3    = vecw / 2 * lofst;
        wrapper.s4    = 2;
        INIT_AGEN4(out, wrapper);

        *(*cfgs)++ = extract_agen_cfg(out);
    }

    void bitonic_merge_transpose_n_way_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst,
                                            int height, int h_dist)
    {
        agen in, out;
        int vecw = pva_elementsof(DVTYPEX);

        DataType *src = (*it) & 1 ? ping : pong;
        DataType *dst = (*it) & 1 ? pong : ping;
        (*it)++;

        int niter1 = 1;
        int niter2 = (vecw / 2) / h_dist;
        int niter3 = height / (vecw / 2);
        int niter4 = h_dist;

        AgenWrapper wrapper;
        wrapper.size = sizeof(DataType);

        in           = init((DVTYPE *)src);
        in.lane_ofst = lofst / vecw;
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.n3   = niter3;
        wrapper.n4   = niter4;
        wrapper.s1   = 1;
        wrapper.s2   = h_dist * 2;
        wrapper.s3   = vecw / 2 * lofst;
        wrapper.s4   = 2;
        INIT_AGEN4(in, wrapper);

        *(*cfgs)++ = extract_agen_cfg(in);

        out           = init((DVTYPE *)dst);
        out.lane_ofst = lofst / vecw;
        wrapper.n1    = niter1;
        wrapper.n2    = niter2;
        wrapper.n3    = niter3;
        wrapper.n4    = niter4;
        wrapper.s1    = 1;
        wrapper.s2    = h_dist * 2;
        wrapper.s3    = vecw / 2 * lofst;
        wrapper.s4    = 2;
        INIT_AGEN4(out, wrapper);

        *(*cfgs)++ = extract_agen_cfg(out);
    }

    void bitonic_merge_transpose_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst, int height,
                                      int h_dist)
    {
        int reverse = 1;
        while (h_dist > 0)
        {
            if (h_dist > 1)
            {
                if (reverse)
                {
                    bitonic_merge_transpose_n_way_reverse_init(ping, pong, it, cfgs, lofst, height, h_dist >> (2 - 1),
                                                               4);
                }
                else
                {
                    bitonic_merge_transpose_n_way_init(ping, pong, it, cfgs, lofst, height, h_dist >> (2 - 1));
                }
                h_dist = h_dist >> 2;
            }
            else
            {
                if (reverse)
                {
                    bitonic_merge_transpose_n_way_reverse_init(ping, pong, it, cfgs, lofst, height, h_dist, 2);
                }
                else
                {
                    bitonic_merge_transpose_n_way_init(ping, pong, it, cfgs, lofst, height, h_dist);
                }
                h_dist = h_dist >> 1;
            }
            reverse = 0;
        }
    }

    void bitonic_merge_n_way_reverse_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst,
                                          int height, int v_dist, int n)
    {
        agen stream;

        DataType *src = (*it) & 1 ? ping : pong;
        DataType *dst = (*it) & 1 ? pong : ping;
        (*it)++;

        int niter1 = n / 2;
        int niter2 = height / v_dist / n;
        int niter3 = v_dist;

        AgenWrapper wrapper;
        wrapper.size = sizeof(DataType);

        stream     = init((DVTYPE *)src);
        wrapper.n1 = niter1;
        wrapper.n2 = niter2;
        wrapper.n3 = niter3;
        wrapper.s1 = v_dist * lofst;
        wrapper.s2 = n * v_dist * lofst;
        wrapper.s3 = lofst;
        INIT_AGEN3(stream, wrapper);

        *(*cfgs)++ = extract_agen_cfg(stream);

        stream     = init((DVTYPE *)(src + (n * v_dist - 1) * lofst));
        wrapper.s1 = -v_dist * lofst;
        wrapper.s2 = n * v_dist * lofst;
        wrapper.s3 = -lofst;
        INIT_AGEN3(stream, wrapper);

        *(*cfgs)++ = extract_agen_cfg(stream);

        niter1 = height / v_dist;
        niter2 = v_dist;

        stream       = init((DVTYPE *)dst);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist * lofst;
        wrapper.s2   = lofst;
        INIT_AGEN2(stream, wrapper);

        *(*cfgs)++ = extract_agen_cfg(stream);
    }

    void bitonic_merge_n_way_transpose_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst,
                                            int height, int v_dist)
    {
        agen stream;

        DataType *src = ping;
        DataType *dst = pong;
        (*it)++;

        int vecw   = pva_elementsof(DVTYPEX);
        int niter1 = height / v_dist;
        int niter2 = v_dist;

        AgenWrapper wrapper;

        stream       = init((DVTYPE *)src);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist * lofst;
        wrapper.s2   = lofst;
        INIT_AGEN2(stream, wrapper);

        *(*cfgs)++ = extract_agen_cfg(stream);

        stream           = init((DVTYPE *)dst);
        wrapper.size     = sizeof(DataType);
        stream.lane_ofst = height * 2 / vecw;
        wrapper.n1       = 1;
        wrapper.n2       = niter1;
        wrapper.n3       = niter2;
        wrapper.s1       = 1;
        wrapper.s2       = 2 * v_dist;
        wrapper.s3       = 1;
        INIT_AGEN3(stream, wrapper);

        *(*cfgs)++ = extract_agen_cfg(stream);
    }

    void bitonic_merge_n_way_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst, int height,
                                  int v_dist)
    {
        agen stream;

        DataType *src = (*it) & 1 ? ping : pong;
        DataType *dst = (*it) & 1 ? pong : ping;
        (*it)++;

        int niter1 = height / v_dist;
        int niter2 = v_dist;

        AgenWrapper wrapper;

        stream       = init((DVTYPE *)src);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist * lofst;
        wrapper.s2   = lofst;
        INIT_AGEN2(stream, wrapper);

        *(*cfgs)++ = extract_agen_cfg(stream);

        stream       = init((DVTYPE *)dst);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist * lofst;
        wrapper.s2   = lofst;
        INIT_AGEN2(stream, wrapper);

        *(*cfgs)++ = extract_agen_cfg(stream);
    }

    void bitonic_merge_init(DataType *ping, DataType *pong, DataType *output, int *it, AgenCFG **cfgs, int lofst,
                            int height, int v_dist, int reverse, int last)
    {
        while (v_dist > 0)
        {
            if (v_dist >= 2)
            {
                if (reverse)
                {
                    bitonic_merge_n_way_reverse_init(ping, pong, it, cfgs, lofst, height, v_dist >> (2 - 1), 4);
                }
                else
                {
                    if (last && (v_dist >> 2) == 0)
                    {
                        DataType *src = (*it) & 1 ? ping : pong;
                        bitonic_merge_n_way_transpose_init(src, output, it, cfgs, lofst, height, v_dist >> (2 - 1));
                    }
                    else
                    {
                        bitonic_merge_n_way_init(ping, pong, it, cfgs, lofst, height, v_dist >> (2 - 1));
                    }
                }
                v_dist = v_dist >> 2;
            }
            else
            {
                if (reverse)
                {
                    bitonic_merge_n_way_reverse_init(ping, pong, it, cfgs, lofst, height, v_dist >> (1 - 1), 4);
                }
                else
                {
                    if (last && (v_dist >> 1) == 0)
                    {
                        DataType *src = (*it) & 1 ? ping : pong;
                        bitonic_merge_n_way_transpose_init(src, output, it, cfgs, lofst, height, v_dist >> (1 - 1));
                    }
                    else
                    {
                        bitonic_merge_n_way_init(ping, pong, it, cfgs, lofst, height, v_dist >> (1 - 1));
                    }
                }
                v_dist = v_dist >> 1;
            }
            reverse = 0;
        }
    }

    void sort_payload_height_512(AgenCFG *cfgs, vshortx *permute_short_ptr, vshortx *permute_reverse_short_ptr)
    {
        int height = 512;

        even_odd_merge_sort_payload(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_2_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_transpose_2_way(&cfgs, height, permute_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way_transp(&cfgs, height);
    }

    void sort_payload_height_256(AgenCFG *cfgs, vshortx *permute_short_ptr, vshortx *permute_reverse_short_ptr)
    {
        int height = 256;
        even_odd_merge_sort_payload(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_transpose_2_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_transpose_2_way(&cfgs, height, permute_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_transp(&cfgs, height);
    }

    void sort_payload_height_128(AgenCFG *cfgs, vshortx *permute_short_ptr, vshortx *permute_reverse_short_ptr)
    {
        int height = 128;
        even_odd_merge_sort_payload(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_2_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_transpose_2_way(&cfgs, height, permute_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way_transp(&cfgs, height);
    }

    void sort_payload_height_64(AgenCFG *cfgs, vshortx *permute_short_ptr, vshortx *permute_reverse_short_ptr)
    {
        int height = 64;
        even_odd_merge_sort_payload(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_transpose_2_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_transpose_2_way(&cfgs, height, permute_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_transp(&cfgs, height);
    }

    void sort_payload_height_32(AgenCFG *cfgs, vshortx *permute_short_ptr, vshortx *permute_reverse_short_ptr)
    {
        int height = 32;
        even_odd_merge_sort_payload(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way_reverse(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_2_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way(&cfgs, height);
        bitonic_merge_transpose_4_way_reverse(&cfgs, height, permute_short_ptr, permute_reverse_short_ptr);
        bitonic_merge_transpose_2_way(&cfgs, height, permute_short_ptr);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_4_way(&cfgs, height);
        bitonic_merge_2_way_transp(&cfgs, height);
    }

public:
    SortPayloadVpu()
    {
        static_assert(Size == 512 || Size == 1024 || Size == 2048 || Size == 4096 || Size == 8192,
                      "Unsupported size value for SortPayloadVpu template instantiation.");
    }

    /**
     * Initialize the SortPayload algorithm parameters, setup agens for the Execute method.
     *
     * The SortPayload primitive treats two consecutive elements as a key-payload pair and sort them based on the key value.
     * Template parameter Size is the number of elements (including key and payload) in the input data.
     * The SortPayload primitive only supports ascending sort.
     *
     * @param src Pointer to the src buffer which contains the input data.
     *            The input data should be consecutively stored in a 1D array with no pitch.
     *            Input buffer size should be at least MIN_INPUT_BUFFER_SIZE.
     * @param dst Pointer to the dst buffer where the result is written to.
     *            The output data will be stored in a 2D array with width = Size / 8, height = 8, pitch = Size / 8 + 2.
     *            Output buffer size should be at least MIN_OUTPUT_BUFFER_SIZE.
     * @param cfgs Caller-owned storage for at least CFG_COUNT address-generator configurations.
     * @param scratch Caller-owned scratch buffer of at least SCRATCH_SIZE bytes.
     *
     * Taking the two buffers separately lets the caller place them in different
     * VMEM superbanks. Use the SortPayloadContext overload when a single combined
     * allocation is acceptable.
     *
     * @return Number of configurations written, which always equals CFG_COUNT.
     */
    size_t Init(DataType *src, DataType *dst, AgenCFG *cfgs, uint8_t *scratch)
    {
        AgenCFG *const cfgsBegin = cfgs;

        int it = 0;
        int v_dist, h_dist;

        int vecw   = pva_elementsof(DVTYPEX);
        int lofst  = vecw + 2;
        int height = Size / vecw;

        DataType *tmp = reinterpret_cast<DataType *>(scratch);

        even_odd_merge_sort_payload_init(src, tmp, &cfgs, lofst, height);

        for (v_dist = 8; v_dist < height; v_dist *= 2)
        {
            bitonic_merge_init(src, tmp, dst, &it, &cfgs, lofst, height, v_dist, 1, 0);
        }

        for (h_dist = 1; h_dist < vecw / 2; h_dist *= 2)
        {
            bitonic_merge_transpose_init(src, tmp, &it, &cfgs, lofst, height, h_dist);
            bitonic_merge_init(src, tmp, dst, &it, &cfgs, lofst, height, height / 2, 0, h_dist == vecw / 4);
        }

        return static_cast<size_t>(cfgs - cfgsBegin);
    }

    /**
     * Initialize using the primitive's combined context storage.
     *
     * @param src Pointer to the src buffer. Same requirements as in the overload taking separate buffers.
     * @param dst Pointer to the dst buffer. Same requirements as in the overload taking separate buffers.
     * @param context Pointer to the SortPayloadContext that provides the configuration and scratch storage.
     */
    void Init(DataType *src, DataType *dst, SortPayloadContext *context)
    {
        (void)Init(src, dst, context->cfgs, context->scratch);
    }

    /**
     * Execute the SortPayload algorithm using caller-owned address-generator configurations.
     *
     * @param cfgs The same configuration storage that was passed to Init().
     */
    void Execute(AgenCFG *cfgs)
    {
        constexpr int height = Size / pva_elementsof(DVTYPEX);

        const short permute_index_short[] = {
            0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
        };

        const short permute_index_reverse_short[] = {
            0xe, 0xf, 0xc, 0xd, 0xa, 0xb, 0x8, 0x9, 0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1,
        };

        vshortx permute_short         = sign_extend(*((vshort *)permute_index_short));
        vshortx permute_reverse_short = sign_extend(*((vshort *)permute_index_reverse_short));

        if constexpr (height == 512)
            sort_payload_height_512(cfgs, &permute_short, &permute_reverse_short);
        else if constexpr (height == 256)
            sort_payload_height_256(cfgs, &permute_short, &permute_reverse_short);
        else if constexpr (height == 128)
            sort_payload_height_128(cfgs, &permute_short, &permute_reverse_short);
        else if constexpr (height == 64)
            sort_payload_height_64(cfgs, &permute_short, &permute_reverse_short);
        else if constexpr (height == 32)
            sort_payload_height_32(cfgs, &permute_short, &permute_reverse_short);
    }

    /**
     * Execute using the primitive's combined context storage.
     *
     * @param context The same context that was passed to Init().
     */
    void Execute(SortPayloadContext *context)
    {
        Execute(context->cfgs);
    }

    /**
     * @brief Number of address-generator configurations Init() writes for this instantiation.
     * Can be used to allocate appropriately sized configuration buffers or verify buffer sizes
     * with static_assert.
     */
    static constexpr size_t CFG_COUNT{
        detail::SortPayloadInitCfgCount(pva_elementsof(DVTYPEX), Size / pva_elementsof(DVTYPEX))};
    /**
     * @brief Minimum required size in bytes for the address-generator configuration buffer.
     */
    static constexpr size_t CFGS_SIZE{CFG_COUNT * sizeof(AgenCFG)};
    /**
     * @brief Minimum required size in bytes for the scratch buffer.
     * Can be used to allocate appropriately sized scratch buffers or verify buffer sizes with static_assert.
     */
    static constexpr size_t SCRATCH_SIZE{(pva_elementsof(DVTYPEX) + 2) * (Size / pva_elementsof(DVTYPEX)) *
                                         sizeof(DataType)};

    static_assert(CFGS_SIZE <= sizeof(SortPayloadContext::cfgs),
                  "SortPayloadContext::cfgs is too small for this SortPayloadVpu instantiation.");
    static_assert(SCRATCH_SIZE <= SortPayloadContext::SCRATCH_SIZE,
                  "SortPayloadContext::scratch is too small for this SortPayloadVpu instantiation.");

    /**
     * @brief Minimum required size in bytes for the input buffer.
     * Can be used to allocate appropriately sized input buffers or verify buffer sizes with static_assert.
     */
    static constexpr size_t MIN_INPUT_BUFFER_SIZE{(pva_elementsof(DVTYPEX) + 2) * (Size / pva_elementsof(DVTYPEX)) *
                                                  sizeof(DataType)};
    /**
     * @brief Minimum required size in bytes for the output buffer.
     * Can be used to allocate appropriately sized output buffers or verify buffer sizes with static_assert.
     */
    static constexpr size_t MIN_OUTPUT_BUFFER_SIZE{(2 * Size / pva_elementsof(DVTYPEX) + 2) *
                                                   (pva_elementsof(DVTYPEX) / 2) * sizeof(DataType)};
};
/** @} */
} // namespace pvaApl

#endif /* PVA_APL_SORT_PAYLOAD_VPU_HPP */
