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
 * @file pvaAplSortVpu.hpp
 *
 * @brief Defines Sort primitive template class
 */

#ifndef PVA_APL_SORT_VPU_HPP
#define PVA_APL_SORT_VPU_HPP

#include <cupva_device.h>

#include <type_traits>

namespace pvaApl {

// Define a traits struct to map DataType to specific vector types
template<typename DataType>
struct SortVectorTypes;

/**
 * @brief Vector type specialization for uint16_t.
 */
template<>
struct SortVectorTypes<uint16_t>
{
    /** @brief Vector type for uint16_t. */
    using DVTYPE = dvshort;
    /** @brief Extended vector type for uint16_t. */
    using DVTYPEX = dvshortx;
    /** @brief Permute vector type for uint16_t. */
    using PMT_VTYPEX = vcharx;
};

/**
 * @brief Vector type specialization for int16_t.
 */
template<>
struct SortVectorTypes<int16_t>
{
    /** @brief Vector type for int16_t. */
    using DVTYPE = dvshort;
    /** @brief Extended vector type for int16_t. */
    using DVTYPEX = dvshortx;
    /** @brief Permute vector type for int16_t. */
    using PMT_VTYPEX = vcharx;
};

/**
 * @brief Vector type specialization for uint32_t.
 */
template<>
struct SortVectorTypes<uint32_t>
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
struct SortVectorTypes<int32_t>
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
 * @brief AgenCFG entries consumed by one SortVpu::bitonic_merge_init() call.
 *
 * Must be kept in step with that function.
 */
constexpr size_t SortMergeCfgCount(int vDist, bool reverse)
{
    size_t count = 0;
    while (vDist > 0)
    {
        count += reverse ? 3U : 2U;
        vDist   = (vDist > 8 || vDist == 4) ? (vDist >> 3) : (vDist >> 2);
        reverse = false;
    }
    return count;
}

/**
 * @brief AgenCFG entries consumed by one SortVpu::bitonic_merge_transpose_init() call.
 *
 * Must be kept in step with that function.
 */
constexpr size_t SortTransposeCfgCount(int hDist)
{
    size_t count = 0;
    bool reverse = true;
    while (hDist > 0)
    {
        count += reverse ? 3U : 2U;
        if (hDist > 8 || hDist == 4)
        {
            hDist >>= 3;
        }
        else if (hDist == 8 || hDist == 2)
        {
            hDist >>= 2;
        }
        else
        {
            hDist >>= 1;
        }
        reverse = false;
    }
    return count;
}

/**
 * @brief Total AgenCFG entries SortVpu::Init() writes for the given geometry.
 */
constexpr size_t SortInitCfgCount(int vecw, int height)
{
    size_t count = 2U; // even_odd_merge_sort_init()
    for (int vDist = 8; vDist < height; vDist *= 2)
    {
        count += SortMergeCfgCount(vDist, true);
    }
    for (int hDist = 1; hDist < vecw; hDist *= 2)
    {
        count += SortTransposeCfgCount(hDist) + SortMergeCfgCount(height / 2, false);
    }
    return count;
}

} // namespace detail

/**
 * @brief Context structure for the Sort primitive.
 *
 * This structure holds the configuration and scratch space
 * required by the sorting primitive.
 */
struct SortContext
{
    /** @brief Array of Agen configurations. */
    AgenCFG cfgs[128];
    /** @brief Scratch size for the Sort primitive. */
    static constexpr size_t SCRATCH_SIZE = ((16 + 1) * 512 * sizeof(uint32_t));
    /** @brief Scratch space. */
    uint8_t scratch[SCRATCH_SIZE];
};

/**
 * @defgroup PVA_PRIMITIVE_SORT Sort
 * @{
 */

/**
  * @brief Sort primitive template class
  *
  * @tparam DataType Data Type of the input array. Must be one of: uint16_t, int16_t, uint32_t, int32_t.
  * @tparam Size Number of elements to sort.
  *              For 16-bit data type, must be one of: 1024, 2048, 4096, 8192, 16384.
  *              For 32-bit data type, must be one of: 256, 512, 1024, 2048, 4096, 8192.
  */
template<typename DataType, int Size>
class SortVpu
{
private:
    using DVTYPE     = typename SortVectorTypes<DataType>::DVTYPE;
    using DVTYPEX    = typename SortVectorTypes<DataType>::DVTYPEX;
    using PMT_VTYPEX = typename SortVectorTypes<DataType>::PMT_VTYPEX;

    template<typename AgenType>
    auto DVLOAD(AgenType &ag) -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint16_t>)
        {
            return static_cast<dvshortx>(dvushort_load(ag));
        }
        else if constexpr (std::is_same_v<DataType, int16_t>)
        {
            return static_cast<dvshortx>(dvshort_load(ag));
        }
        else if constexpr (std::is_same_v<DataType, uint32_t>)
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

    template<typename AgenType>
    auto DVLOAD_TRANSP(AgenType &ag) -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint16_t>)
        {
            return static_cast<dvshortx>(dvushort_load_transp(ag));
        }
        else if constexpr (std::is_same_v<DataType, int16_t>)
        {
            return static_cast<dvshortx>(dvshort_load_transp(ag));
        }
        else if constexpr (std::is_same_v<DataType, uint32_t>)
        {
            return static_cast<dvintx>(dvuint_load_transp(ag));
        }
        else if constexpr (std::is_same_v<DataType, int32_t>)
        {
            return static_cast<dvintx>(dvint_load_transp(ag));
        }
        else
        {
            static_assert(!std::is_same_v<DataType, DataType>, "Unsupported type for dvload");
        }
    }

    template<typename AgenType, typename VecType>
    auto DVLOAD_PERM_TRANSP(AgenType &ag, VecType vx) -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint16_t>)
        {
            return static_cast<dvshortx>(dvushort_load_perm_transp(ag, vx));
        }
        else if constexpr (std::is_same_v<DataType, int16_t>)
        {
            return static_cast<dvshortx>(dvshort_load_perm_transp(ag, vx));
        }
        else if constexpr (std::is_same_v<DataType, uint32_t>)
        {
            return static_cast<dvintx>(dvuint_load_perm_transp(ag, vx));
        }
        else if constexpr (std::is_same_v<DataType, int32_t>)
        {
            return static_cast<dvintx>(dvint_load_perm_transp(ag, vx));
        }
        else
        {
            static_assert(!std::is_same_v<DataType, DataType>, "Unsupported type for dvload_perm");
        }
    }

    auto PMT_LOAD() -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint16_t> || std::is_same_v<DataType, int16_t>)
        {
            int8_t pmtArr[] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                               15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};
            return static_cast<vcharx>(sign_extend(*((vchar *)(pmtArr))));
        }
        else if constexpr (std::is_same_v<DataType, uint32_t> || std::is_same_v<DataType, int32_t>)
        {
            int16_t pmtArr[] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
            return static_cast<vshortx>(sign_extend(*((vshort *)(pmtArr))));
        }
        else
        {
            static_assert(!std::is_same_v<DataType, DataType>, "Unsupported type for dvload_perm");
        }
    }

    inline void VSORT2_INPLACE(DVTYPEX &v1, DVTYPEX &v2)
    {
        dvsort2(v1, v2, v1, v2);
    }

    inline void MERGE_2_WAY(DVTYPEX &v1, DVTYPEX &v2)
    {
        VSORT2_INPLACE(v1, v2);
    }

    inline void MERGE_4_WAY(DVTYPEX &v1, DVTYPEX &v2, DVTYPEX &v3, DVTYPEX &v4)
    {
        VSORT2_INPLACE(v1, v3);
        VSORT2_INPLACE(v2, v4);
        MERGE_2_WAY(v1, v2);
        MERGE_2_WAY(v3, v4);
    }

    inline void MERGE_8_WAY(DVTYPEX &v1, DVTYPEX &v2, DVTYPEX &v3, DVTYPEX &v4, DVTYPEX &v5, DVTYPEX &v6, DVTYPEX &v7,
                            DVTYPEX &v8)
    {
        VSORT2_INPLACE(v1, v5);
        VSORT2_INPLACE(v2, v6);
        VSORT2_INPLACE(v3, v7);
        VSORT2_INPLACE(v4, v8);
        MERGE_4_WAY(v1, v2, v3, v4);
        MERGE_4_WAY(v5, v6, v7, v8);
    }

    inline void MERGE_16_WAY(DVTYPEX &v1, DVTYPEX &v2, DVTYPEX &v3, DVTYPEX &v4, DVTYPEX &v5, DVTYPEX &v6, DVTYPEX &v7,
                             DVTYPEX &v8, DVTYPEX &v9, DVTYPEX &v10, DVTYPEX &v11, DVTYPEX &v12, DVTYPEX &v13,
                             DVTYPEX &v14, DVTYPEX &v15, DVTYPEX &v16)
    {
        VSORT2_INPLACE(v1, v9);
        VSORT2_INPLACE(v2, v10);
        VSORT2_INPLACE(v3, v11);
        VSORT2_INPLACE(v4, v12);
        VSORT2_INPLACE(v5, v13);
        VSORT2_INPLACE(v6, v14);
        VSORT2_INPLACE(v7, v15);
        VSORT2_INPLACE(v8, v16);
        MERGE_8_WAY(v1, v2, v3, v4, v5, v6, v7, v8);
        MERGE_8_WAY(v9, v10, v11, v12, v13, v14, v15, v16);
    }

    inline void MERGE_16_WAY_R(DVTYPEX &v1, DVTYPEX &v2, DVTYPEX &v3, DVTYPEX &v4, DVTYPEX &v5, DVTYPEX &v6,
                               DVTYPEX &v7, DVTYPEX &v8, DVTYPEX &v9, DVTYPEX &v10, DVTYPEX &v11, DVTYPEX &v12,
                               DVTYPEX &v13, DVTYPEX &v14, DVTYPEX &v15, DVTYPEX &v16)
    {
        VSORT2_INPLACE(v1, v16);
        VSORT2_INPLACE(v2, v15);
        VSORT2_INPLACE(v3, v14);
        VSORT2_INPLACE(v4, v13);
        VSORT2_INPLACE(v5, v12);
        VSORT2_INPLACE(v6, v11);
        VSORT2_INPLACE(v7, v10);
        VSORT2_INPLACE(v8, v9);
        MERGE_8_WAY(v1, v2, v3, v4, v5, v6, v7, v8);
        MERGE_8_WAY(v9, v10, v11, v12, v13, v14, v15, v16);
    }

    void even_odd_merge_sort(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {

        DVTYPEX v1, v2, v3, v4, v5, v6, v7, v8;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 3;

        for (int i = 0; i < niter; i++) chess_loop_range(2, )
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

            VSORT2_INPLACE(v1, v2);
            VSORT2_INPLACE(v3, v4);
            VSORT2_INPLACE(v5, v6);
            VSORT2_INPLACE(v7, v8);

            VSORT2_INPLACE(v1, v3);
            VSORT2_INPLACE(v5, v7);
            VSORT2_INPLACE(v2, v4);
            VSORT2_INPLACE(v6, v8);

            VSORT2_INPLACE(v2, v3);
            VSORT2_INPLACE(v6, v7);

            VSORT2_INPLACE(v1, v5);
            VSORT2_INPLACE(v2, v6);
            VSORT2_INPLACE(v3, v7);
            VSORT2_INPLACE(v4, v8);

            VSORT2_INPLACE(v3, v5);
            VSORT2_INPLACE(v4, v6);

            VSORT2_INPLACE(v2, v3);
            VSORT2_INPLACE(v4, v5);
            VSORT2_INPLACE(v6, v7);

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

    void bitonic_merge_8_way_reverse(AgenCFG *cfgs_in1, AgenCFG *cfgs_in2, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4, v5, v6, v7, v8;
        agen_A in1, in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(cfgs_in1[0]);
        in2 = init_agen_A_from_cfg(cfgs_in2[0]);
        out = init_agen_B_from_cfg(cfgs_out[0]);

        int niter = height >> 3;

        for (int i = 0; i < niter; i++) chess_loop_range(2, )
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in1);
            v2 = DVLOAD(in1);
            v3 = DVLOAD(in1);
            v4 = DVLOAD(in1);

            v5 = DVLOAD(in2);
            v6 = DVLOAD(in2);
            v7 = DVLOAD(in2);
            v8 = DVLOAD(in2);

            MERGE_8_WAY(v1, v2, v3, v4, v5, v6, v7, v8);

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

    void bitonic_merge_8_way_transp(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4, v5, v6, v7, v8;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 3;

        for (int i = 0; i < niter; i++) chess_loop_range(2, )
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

            MERGE_8_WAY(v1, v2, v3, v4, v5, v6, v7, v8);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
            vstore_transp(v3, out);
            vstore_transp(v4, out);
            vstore_transp(v5, out);
            vstore_transp(v6, out);
            vstore_transp(v7, out);
            vstore_transp(v8, out);
        }
    }

    void bitonic_merge_8_way(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4, v5, v6, v7, v8;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 3;

        for (int i = 0; i < niter; i++) chess_loop_range(2, )
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

            MERGE_8_WAY(v1, v2, v3, v4, v5, v6, v7, v8);

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

    void bitonic_merge_4_way_reverse(AgenCFG *cfgs_in1, AgenCFG *cfgs_in2, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4;
        agen_A in1, in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(cfgs_in1[0]);
        in2 = init_agen_A_from_cfg(cfgs_in2[0]);
        out = init_agen_B_from_cfg(cfgs_out[0]);

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(4, )
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

    void bitonic_merge_4_way_transp(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(4, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD(in);
            v2 = DVLOAD(in);
            v3 = DVLOAD(in);
            v4 = DVLOAD(in);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
            vstore_transp(v3, out);
            vstore_transp(v4, out);
        }
    }

    void bitonic_merge_4_way(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(4, )
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

    void bitonic_merge_transpose_8_way_reverse(AgenCFG *cfgs_in1, AgenCFG *cfgs_in2, AgenCFG *cfgs_out, int height,
                                               PMT_VTYPEX *pmt_ptr)
    {
        DVTYPEX v1, v2, v3, v4, v5, v6, v7, v8;
        agen_A in1, in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(cfgs_in1[0]);
        in2 = init_agen_A_from_cfg(cfgs_in2[0]);
        out = init_agen_B_from_cfg(cfgs_out[0]);

        PMT_VTYPEX pmt = *pmt_ptr;

        int niter = height >> 3;

        for (int i = 0; i < niter; i++) chess_loop_range(2, )
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_TRANSP(in1);
            v2 = DVLOAD_TRANSP(in1);
            v3 = DVLOAD_TRANSP(in1);
            v4 = DVLOAD_TRANSP(in1);

            v5 = DVLOAD_PERM_TRANSP(in2, pmt);
            v6 = DVLOAD_PERM_TRANSP(in2, pmt);
            v7 = DVLOAD_PERM_TRANSP(in2, pmt);
            v8 = DVLOAD_PERM_TRANSP(in2, pmt);

            MERGE_8_WAY(v1, v2, v3, v4, v5, v6, v7, v8);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
            vstore_transp(v3, out);
            vstore_transp(v4, out);

            vstore_transp(v5, out);
            vstore_transp(v6, out);
            vstore_transp(v7, out);
            vstore_transp(v8, out);
        }
    }

    void bitonic_merge_transpose_8_way(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4, v5, v6, v7, v8;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 3;

        for (int i = 0; i < niter; i++) chess_loop_range(4, )
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_TRANSP(in);
            v2 = DVLOAD_TRANSP(in);
            v3 = DVLOAD_TRANSP(in);
            v4 = DVLOAD_TRANSP(in);

            v5 = DVLOAD_TRANSP(in);
            v6 = DVLOAD_TRANSP(in);
            v7 = DVLOAD_TRANSP(in);
            v8 = DVLOAD_TRANSP(in);

            MERGE_8_WAY(v1, v2, v3, v4, v5, v6, v7, v8);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
            vstore_transp(v3, out);
            vstore_transp(v4, out);

            vstore_transp(v5, out);
            vstore_transp(v6, out);
            vstore_transp(v7, out);
            vstore_transp(v8, out);
        }
    }

    void bitonic_merge_transpose_4_way_reverse(AgenCFG *cfgs_in1, AgenCFG *cfgs_in2, AgenCFG *cfgs_out, int height,
                                               PMT_VTYPEX *pmt_ptr)
    {
        DVTYPEX v1, v2, v3, v4;
        agen_A in1, in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(cfgs_in1[0]);
        in2 = init_agen_A_from_cfg(cfgs_in2[0]);
        out = init_agen_B_from_cfg(cfgs_out[0]);

        chess_separator_scheduler();

        PMT_VTYPEX pmt = *pmt_ptr;

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(4, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_TRANSP(in1);
            v2 = DVLOAD_TRANSP(in1);
            v3 = DVLOAD_PERM_TRANSP(in2, pmt);
            v4 = DVLOAD_PERM_TRANSP(in2, pmt);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
            vstore_transp(v3, out);
            vstore_transp(v4, out);
        }
    }

    void bitonic_merge_transpose_4_way(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2, v3, v4;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 2;

        for (int i = 0; i < niter; i++) chess_loop_range(4, )
        chess_unroll_loop(2)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_TRANSP(in);
            v2 = DVLOAD_TRANSP(in);
            v3 = DVLOAD_TRANSP(in);
            v4 = DVLOAD_TRANSP(in);

            MERGE_4_WAY(v1, v2, v3, v4);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
            vstore_transp(v3, out);
            vstore_transp(v4, out);
        }
    }

    void bitonic_merge_transpose_2_way_reverse(AgenCFG *cfgs_in1, AgenCFG *cfgs_in2, AgenCFG *cfgs_out, int height,
                                               PMT_VTYPEX *pmt_ptr)
    {
        DVTYPEX v1, v2;
        agen_A in1, in2;
        agen_B out;

        in1 = init_agen_A_from_cfg(cfgs_in1[0]);
        in2 = init_agen_A_from_cfg(cfgs_in2[0]);
        out = init_agen_B_from_cfg(cfgs_out[0]);

        PMT_VTYPEX pmt = *pmt_ptr;

        int niter = height >> 1;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_TRANSP(in1);
            v2 = DVLOAD_PERM_TRANSP(in2, pmt);

            VSORT2_INPLACE(v1, v2);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
        }
    }

    void bitonic_merge_transpose_2_way(AgenCFG *cfgs_in1, AgenCFG *cfgs_out, int height)
    {
        DVTYPEX v1, v2;
        agen in, out;

        in  = init_agen_from_cfg(cfgs_in1[0]);
        out = init_agen_from_cfg(cfgs_out[0]);

        int niter = height >> 1;

        for (int i = 0; i < niter; i++) chess_loop_range(8, )
        chess_unroll_loop(4)
        chess_prepare_for_pipelining
        {
            v1 = DVLOAD_TRANSP(in);
            v2 = DVLOAD_TRANSP(in);

            VSORT2_INPLACE(v1, v2);

            vstore_transp(v1, out);
            vstore_transp(v2, out);
        }
    }

    void even_odd_merge_sort_init(DataType *ping, DataType *pong, AgenCFG **cfgs, int lofst, int height)
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

        int niter1 = n / 2;
        int niter2 = vecw / h_dist / n;
        int niter3 = height / vecw;
        int niter4 = h_dist;

        AgenWrapper wrapper;
        wrapper.size = sizeof(DataType);

        in1           = init((DVTYPE *)src);
        in1.lane_ofst = lofst / vecw;
        wrapper.n1    = niter1;
        wrapper.n2    = niter2;
        wrapper.n3    = niter3;
        wrapper.n4    = niter4;
        wrapper.s1    = h_dist;
        wrapper.s2    = n * h_dist;
        wrapper.s3    = vecw * lofst;
        wrapper.s4    = 1;
        INIT_AGEN4(in1, wrapper);

        *(*cfgs)++ = extract_agen_cfg(in1);

        in2           = init((DVTYPE *)(src + (height - vecw) * lofst + h_dist * n - 1));
        in2.lane_ofst = lofst / vecw;
        wrapper.s1    = -h_dist;
        wrapper.s2    = n * h_dist;
        wrapper.s3    = -vecw * lofst;
        wrapper.s4    = -1;
        INIT_AGEN4(in2, wrapper);

        *(*cfgs)++ = extract_agen_cfg(in2);

        niter1 = vecw / h_dist;
        niter2 = height / vecw;
        niter3 = h_dist;

        out           = init((DVTYPE *)dst);
        out.lane_ofst = lofst / vecw;
        wrapper.n1    = niter1;
        wrapper.n2    = niter2;
        wrapper.n3    = niter3;
        wrapper.s1    = h_dist;
        wrapper.s2    = vecw * lofst;
        wrapper.s3    = 1;
        INIT_AGEN3(out, wrapper);

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

        int niter1 = vecw / h_dist;
        int niter2 = height / vecw;
        int niter3 = h_dist;

        AgenWrapper wrapper;
        wrapper.size = sizeof(DataType);

        in           = init((DVTYPE *)src);
        in.lane_ofst = lofst / vecw;
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.n3   = niter3;
        wrapper.s1   = h_dist;
        wrapper.s2   = vecw * lofst;
        wrapper.s3   = 1;
        INIT_AGEN3(in, wrapper);

        *(*cfgs)++ = extract_agen_cfg(in);

        out           = init((DVTYPE *)dst);
        out.lane_ofst = lofst / vecw;
        wrapper.n1    = niter1;
        wrapper.n2    = niter2;
        wrapper.n3    = niter3;
        wrapper.s1    = h_dist;
        wrapper.s2    = vecw * lofst;
        wrapper.s3    = 1;
        INIT_AGEN3(out, wrapper);

        *(*cfgs)++ = extract_agen_cfg(out);
    }

    void bitonic_merge_transpose_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst, int height,
                                      int h_dist)
    {
        int reverse = 1;
        while (h_dist > 0)
        {
            if (h_dist > 8 || h_dist == 4)
            {
                if (reverse)
                {
                    bitonic_merge_transpose_n_way_reverse_init(ping, pong, it, cfgs, lofst, height, h_dist >> (3 - 1),
                                                               8);
                }
                else
                {
                    bitonic_merge_transpose_n_way_init(ping, pong, it, cfgs, lofst, height, h_dist >> (3 - 1));
                }
                h_dist = h_dist >> 3;
            }
            else if (h_dist == 8 || h_dist == 2)
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
        agen agen;

        DataType *src = (*it) & 1 ? ping : pong;
        DataType *dst = (*it) & 1 ? pong : ping;
        (*it)++;

        int niter1 = n / 2;
        int niter2 = height / v_dist / n;
        int niter3 = v_dist;

        AgenWrapper wrapper;
        wrapper.size = sizeof(DataType);

        agen       = init((DVTYPE *)src);
        wrapper.n1 = niter1;
        wrapper.n2 = niter2;
        wrapper.n3 = niter3;
        wrapper.s1 = v_dist * lofst;
        wrapper.s2 = n * v_dist * lofst;
        wrapper.s3 = lofst;
        INIT_AGEN3(agen, wrapper);

        *(*cfgs)++ = extract_agen_cfg(agen);

        agen       = init((DVTYPE *)(src + (n * v_dist - 1) * lofst));
        wrapper.s1 = -v_dist * lofst;
        wrapper.s2 = n * v_dist * lofst;
        wrapper.s3 = -lofst;
        INIT_AGEN3(agen, wrapper);

        *(*cfgs)++ = extract_agen_cfg(agen);

        niter1 = height / v_dist;
        niter2 = v_dist;

        agen       = init((DVTYPE *)dst);
        wrapper.n1 = niter1;
        wrapper.n2 = niter2;
        wrapper.s1 = v_dist * lofst;
        wrapper.s2 = lofst;
        INIT_AGEN2(agen, wrapper);

        *(*cfgs)++ = extract_agen_cfg(agen);
    }

    void bitonic_merge_n_way_transpose_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst,
                                            int height, int v_dist)
    {
        agen agen;

        DataType *src = ping;
        DataType *dst = pong;
        (*it)++;

        int vecw   = pva_elementsof(DVTYPEX);
        int niter1 = height / v_dist;
        int niter2 = v_dist;

        AgenWrapper wrapper;

        agen         = init((DVTYPE *)src);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist * lofst;
        wrapper.s2   = lofst;
        INIT_AGEN2(agen, wrapper);

        *(*cfgs)++ = extract_agen_cfg(agen);

        agen         = init((DVTYPE *)dst);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist;
        wrapper.s2   = 1;
        INIT_AGEN2(agen, wrapper);
        agen.lane_ofst = height / vecw;

        *(*cfgs)++ = extract_agen_cfg(agen);
    }

    void bitonic_merge_n_way_init(DataType *ping, DataType *pong, int *it, AgenCFG **cfgs, int lofst, int height,
                                  int v_dist)
    {
        agen agen;

        DataType *src = (*it) & 1 ? ping : pong;
        DataType *dst = (*it) & 1 ? pong : ping;
        (*it)++;

        int niter1 = height / v_dist;
        int niter2 = v_dist;

        AgenWrapper wrapper;

        agen         = init((DVTYPE *)src);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist * lofst;
        wrapper.s2   = lofst;
        INIT_AGEN2(agen, wrapper);

        *(*cfgs)++ = extract_agen_cfg(agen);

        agen         = init((DVTYPE *)dst);
        wrapper.size = sizeof(DataType);
        wrapper.n1   = niter1;
        wrapper.n2   = niter2;
        wrapper.s1   = v_dist * lofst;
        wrapper.s2   = lofst;
        INIT_AGEN2(agen, wrapper);

        *(*cfgs)++ = extract_agen_cfg(agen);
    }

    void bitonic_merge_init(DataType *ping, DataType *pong, DataType *output, int *it, AgenCFG **cfgs, int lofst,
                            int height, int v_dist, int reverse, int last)
    {
        while (v_dist > 0)
        {
            if (v_dist > 8 || v_dist == 4)
            {
                if (reverse)
                {
                    bitonic_merge_n_way_reverse_init(ping, pong, it, cfgs, lofst, height, v_dist >> (3 - 1), 8);
                }
                else
                {
                    if (last && (v_dist >> 3) == 0)
                    {
                        DataType *src = (*it) & 1 ? ping : pong;
                        bitonic_merge_n_way_transpose_init(src, output, it, cfgs, lofst, height, v_dist >> (3 - 1));
                    }
                    else
                    {
                        bitonic_merge_n_way_init(ping, pong, it, cfgs, lofst, height, v_dist >> (3 - 1));
                    }
                }
                v_dist = v_dist >> 3;
            }
            else
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
            reverse = 0;
        }
    }

    void sort_height_512(AgenCFG *cfgs, PMT_VTYPEX *pmt_ptr)
    {
        int it     = 0;
        int height = 512;

        even_odd_merge_sort(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_2_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;

        if constexpr (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>)
        {
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
        }

        bitonic_merge_8_way_transp(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
    }

    void sort_height_256(AgenCFG *cfgs, PMT_VTYPEX *pmt_ptr)
    {
        int it     = 0;
        int height = 256;

        even_odd_merge_sort(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_2_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;

        if constexpr (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>)
        {
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
        }

        bitonic_merge_4_way_transp(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
    }

    void sort_height_128(AgenCFG *cfgs, PMT_VTYPEX *pmt_ptr)
    {
        int it     = 0;
        int height = 128;

        even_odd_merge_sort(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_2_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;

        if constexpr (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>)
        {
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
        }

        bitonic_merge_4_way_transp(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
    }

    void sort_height_64(AgenCFG *cfgs, PMT_VTYPEX *pmt_ptr)
    {
        int it     = 0;
        int height = 64;

        even_odd_merge_sort(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_2_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;

        if constexpr (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>)
        {
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
        }

        bitonic_merge_8_way_transp(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
    }

    void sort_height_32(AgenCFG *cfgs, PMT_VTYPEX *pmt_ptr)
    {
        int it     = 0;
        int height = 32;

        even_odd_merge_sort(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
        it += 3;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_2_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
        it += 3;
        bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
        bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
        it += 2;

        if constexpr (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>)
        {
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_8_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
        }

        bitonic_merge_4_way_transp(&cfgs[it], &cfgs[it + 1], height);
        it += 2;
    }

    void sort_height_16(AgenCFG *cfgs, PMT_VTYPEX *pmt_ptr)
    {
        int it     = 0;
        int height = 16;

        if constexpr (std::is_same_v<DataType, int32_t> || std::is_same_v<DataType, uint32_t>)
        {
            even_odd_merge_sort(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height);
            it += 3;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_2_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_8_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_transpose_4_way_reverse(&cfgs[it], &cfgs[it + 1], &cfgs[it + 2], height, pmt_ptr);
            it += 3;
            bitonic_merge_transpose_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_4_way(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
            bitonic_merge_4_way_transp(&cfgs[it], &cfgs[it + 1], height);
            it += 2;
        }
    }

public:
    SortVpu()
    {
        static_assert(
            Size == 256 || Size == 512 || Size == 1024 || Size == 2048 || Size == 4096 || Size == 8192 || Size == 16384,
            "Unsupported size value for SortVpu template instantiation.");

        if constexpr (Size == 256)
        {
            static_assert(std::is_same_v<DataType, int32_t> || std::is_same_v<DataType, uint32_t>,
                          "SortVpu<DataType, 256> only supports 32-bit DataType (int32_t or uint32_t).");
        }

        if constexpr (Size == 512)
        {
            static_assert(std::is_same_v<DataType, int32_t> || std::is_same_v<DataType, uint32_t>,
                          "SortVpu<DataType, 512> only supports 32-bit DataType (int32_t or uint32_t).");
        }

        if constexpr (Size == 16384)
        {
            static_assert(std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>,
                          "SortVpu<DataType, 16384> only supports 16-bit DataType (int16_t or uint16_t).");
        }
    }

    /**
      * Initialize the Sort algorithm parameters, setup agens for the Execute method.
      *
      * The Sort primitive only supports ascending sort.
      *
      * @param src Pointer to the src buffer which contains the input data.
      *            The input data should be consecutively stored in a 1D array with no pitch.
      *            Input buffer size should be at least MIN_INPUT_BUFFER_SIZE.
      * @param dst Pointer to the dst buffer where the result is written to.
      *            The output data will be stored in a 2D array with
      *             for 16-bit data type: width = Size / 32, height = 32, pitch = Size / 32 + 1
      *             for 32-bit data type: width = Size / 16, height = 16, pitch = Size / 16 + 1
      *            Output buffer size should be at least MIN_OUTPUT_BUFFER_SIZE.
      * @param cfgs Caller-owned storage for at least CFG_COUNT address-generator configurations.
      * @param scratch Caller-owned scratch buffer of at least SCRATCH_SIZE bytes.
      *
      * Taking the two buffers separately lets the caller place them in different
      * VMEM superbanks. Use the SortContext overload when a single combined
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
        int lofst  = vecw + 1;
        int height = Size / vecw;

        DataType *tmp = reinterpret_cast<DataType *>(scratch);

        even_odd_merge_sort_init(src, tmp, &cfgs, lofst, height);

        for (v_dist = 8; v_dist < height; v_dist *= 2)
        {
            bitonic_merge_init(src, tmp, dst, &it, &cfgs, lofst, height, v_dist, 1, 0);
        }

        for (h_dist = 1; h_dist < vecw; h_dist *= 2)
        {
            bitonic_merge_transpose_init(src, tmp, &it, &cfgs, lofst, height, h_dist);
            bitonic_merge_init(src, tmp, dst, &it, &cfgs, lofst, height, height / 2, 0, h_dist == vecw / 2);
        }

        return static_cast<size_t>(cfgs - cfgsBegin);
    }

    /**
      * Initialize using the primitive's combined context storage.
      *
      * @param src Pointer to the src buffer. Same requirements as in the overload taking separate buffers.
      * @param dst Pointer to the dst buffer. Same requirements as in the overload taking separate buffers.
      * @param context Pointer to the SortContext that provides the configuration and scratch storage.
      */
    void Init(DataType *src, DataType *dst, SortContext *context)
    {
        (void)Init(src, dst, context->cfgs, context->scratch);
    }

    /**
      * Execute the Sort algorithm using caller-owned address-generator configurations.
      *
      * @param cfgs The same configuration storage that was passed to Init().
      */
    void Execute(AgenCFG *cfgs)
    {
        PMT_VTYPEX pmt       = PMT_LOAD();
        constexpr int height = Size / pva_elementsof(DVTYPEX);

        if constexpr (height == 512)
            sort_height_512(cfgs, &pmt);
        else if constexpr (height == 256)
            sort_height_256(cfgs, &pmt);
        else if constexpr (height == 128)
            sort_height_128(cfgs, &pmt);
        else if constexpr (height == 64)
            sort_height_64(cfgs, &pmt);
        else if constexpr (height == 32)
            sort_height_32(cfgs, &pmt);
        else if constexpr (height == 16)
            sort_height_16(cfgs, &pmt);
    }

    /**
      * Execute using the primitive's combined context storage.
      *
      * @param context The same context that was passed to Init().
      */
    void Execute(SortContext *context)
    {
        Execute(context->cfgs);
    }

    /**
     * @brief Number of address-generator configurations Init() writes for this instantiation.
     * Can be used to allocate appropriately sized configuration buffers or verify buffer sizes
     * with static_assert.
     */
    static constexpr size_t CFG_COUNT{
        detail::SortInitCfgCount(pva_elementsof(DVTYPEX), Size / pva_elementsof(DVTYPEX))};
    /**
     * @brief Minimum required size in bytes for the address-generator configuration buffer.
     */
    static constexpr size_t CFGS_SIZE{CFG_COUNT * sizeof(AgenCFG)};
    /**
     * @brief Minimum required size in bytes for the scratch buffer.
     * Can be used to allocate appropriately sized scratch buffers or verify buffer sizes with static_assert.
     */
    static constexpr size_t SCRATCH_SIZE{(pva_elementsof(DVTYPEX) + 1) * (Size / pva_elementsof(DVTYPEX)) *
                                         sizeof(DataType)};

    static_assert(CFGS_SIZE <= sizeof(SortContext::cfgs),
                  "SortContext::cfgs is too small for this SortVpu instantiation.");
    static_assert(SCRATCH_SIZE <= SortContext::SCRATCH_SIZE,
                  "SortContext::scratch is too small for this SortVpu instantiation.");

    /**
     * @brief Minimum required size in bytes for the input buffer.
     * Can be used to allocate appropriately sized input buffers or verify buffer sizes with static_assert.
     */
    static constexpr size_t MIN_INPUT_BUFFER_SIZE{(pva_elementsof(DVTYPEX) + 1) * (Size / pva_elementsof(DVTYPEX)) *
                                                  sizeof(DataType)};
    /**
     * @brief Minimum required size in bytes for the output buffer.
     * Can be used to allocate appropriately sized output buffers or verify buffer sizes with static_assert.
     */
    static constexpr size_t MIN_OUTPUT_BUFFER_SIZE{(Size / pva_elementsof(DVTYPEX) + 1) * pva_elementsof(DVTYPEX) *
                                                   sizeof(DataType)};
};
/** @} */
} // namespace pvaApl

#endif /* PVA_APL_SORT_VPU_HPP */