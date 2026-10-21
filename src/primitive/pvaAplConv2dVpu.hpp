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

/**
 * @file pvaAplConv2dVpu.hpp
 *
 * @brief Defines 2D Convolution (Conv2d) primitive template class
 *
 * Compatibility: Requires PVA SDK 2.6.0 and later.
 */

#ifndef PVA_APL_CONV2D_VPU_HPP
#define PVA_APL_CONV2D_VPU_HPP

#include <cupva_device.h>

#include <limits>
#include <type_traits>

namespace pvaApl {

/**
 * @defgroup PVA_PRIMITIVE_CONV2D Conv2d
 * @{
 */

/**
 * @brief 2D Convolution (Conv2d) primitive template class
 *
 * @tparam  KernelSize 2D Convolution Kernel Size. (Square shape: KernelSize x KernelSize)
 * @tparam  DataType   Image Data Type
 * @tparam  KernelType Kernel Data Type
 */
template<int32_t KernelSize, typename DataType, typename KernelType>
class Conv2dVpu
{
private:
    static_assert(sizeof(DataType) == sizeof(KernelType), "DataType and KernelType must have the same size.");

    template<typename AgenType>
    auto dvload_perm(AgenType &ag, vcharx vx) -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint8_t>)
        {
            return static_cast<dvcharx>(dvuchar_load_perm(ag, vx));
        }
        else if constexpr (std::is_same_v<DataType, int8_t>)
        {
            return static_cast<dvcharx>(dvchar_load_perm(ag, vx));
        }
        else if constexpr (std::is_same_v<DataType, uint16_t>)
        {
            return static_cast<dvshortx>(dvushort_load_perm(ag, vx));
        }
        else if constexpr (std::is_same_v<DataType, int16_t>)
        {
            return static_cast<dvshortx>(dvshort_load_perm(ag, vx));
        }
        else
        {
            static_assert(!std::is_same_v<DataType, DataType>, "Unsupported type for dvload_perm");
        }
    }

    template<typename AgenType>
    auto dvload_perm_coef(AgenType &ag, vcharx vx) -> decltype(auto)
    {
        if constexpr (std::is_same_v<KernelType, uint8_t>)
        {
            return static_cast<dvcharx>(dvuchar_load_perm(ag, vx));
        }
        else if constexpr (std::is_same_v<KernelType, int8_t>)
        {
            return static_cast<dvcharx>(dvchar_load_perm(ag, vx));
        }
        else if constexpr (std::is_same_v<KernelType, uint16_t>)
        {
            return static_cast<dvshortx>(dvushort_load_perm(ag, vx));
        }
        else if constexpr (std::is_same_v<KernelType, int16_t>)
        {
            return static_cast<dvshortx>(dvshort_load_perm(ag, vx));
        }
        else
        {
            static_assert(!std::is_same_v<KernelType, KernelType>, "Unsupported type for dvload_perm_coef");
        }
    }

    template<typename AgenType>
    auto dvload(AgenType &ag) -> decltype(auto)
    {
        if constexpr (std::is_same_v<DataType, uint8_t>)
        {
            return static_cast<dvcharx>(dvuchar_load(ag));
        }
        else if constexpr (std::is_same_v<DataType, int8_t>)
        {
            return static_cast<dvcharx>(dvchar_load(ag));
        }
        else if constexpr (std::is_same_v<DataType, uint16_t>)
        {
            return static_cast<dvshortx>(dvushort_load(ag));
        }
        else if constexpr (std::is_same_v<DataType, int16_t>)
        {
            return static_cast<dvshortx>(dvshort_load(ag));
        }
        else
        {
            static_assert(!std::is_same_v<DataType, DataType>, "Unsupported type for dvload");
        }
    }

    /** @brief Array of Agen configurations. */
    AgenCFG m_cfgs[4];
    /** @brief Number of iterations. */
    int32_t m_niter;

public:
    /**
     * Update the src_even buffer, src_odd buffer, dst buffer pointers for the Conv2d algorithm.
     *
     * @param src_even  Pointer to the src_even buffer which contains even rows of the input image.
     * @param src_odd   Pointer to the src_odd buffer which contains odd rows of the input image.
     * @param dst	    Pointer to the dst buffer where the result is written to.
     */
    void Update(DataType *src_even, DataType *src_odd, DataType *dst)
    {
        cupvaModifyAgenCfgBase(&m_cfgs[0], src_even);
        cupvaModifyAgenCfgBase(&m_cfgs[1], src_odd);
        cupvaModifyAgenCfgBase(&m_cfgs[3], dst);
    }

    /**
     * Initialize the Conv2d algorithm parameters, setup agens for the Execute method.
     *
     * @param src_even_tp   Pointer to the src_even buffer which contains even rows.
     * @param src_even_lp   Line pitch of the src_even_v circular buffer.
     * @param src_odd_tp    Pointer to the src_odd buffer which contains odd rows.
     * @param src_odd_lp    Line pitch of the src_odd_v circular buffer.
     * @param knl_tp        Pointer to the knl_v buffer which contains convolution kernel coefficients.
     * @param knl_lp        Line pitch of the knl_v buffer.
     * @param dst_tp        Pointer to the dst buffer where the result is written to.
     * @param dst_lp        Line pitch of the dst_v buffer.
     * @param qbits         Right shift amount for rounding the output after convolution.
     * @param tw            Tile width.
     * @param th            Tile height.
     * @param cb_even       Pointer to the src_even_v circular buffer.
     * @param cb_odd        Pointer to the src_odd_v circular buffer.
     * @param cb_len        Length of the circular buffer.
     */
    void Init(DataType *src_even_tp, int32_t src_even_lp, DataType *src_odd_tp, int32_t src_odd_lp, KernelType *knl_tp,
              int32_t knl_lp, DataType *restrict dst_tp, int32_t dst_lp, int32_t qbits, int32_t tw, int32_t th,
              DataType *cb_even, DataType *cb_odd, int32_t cb_len)
    {
        int low_bound                     = std::numeric_limits<DataType>::min();
        int high_bound                    = std::numeric_limits<DataType>::max();
        constexpr int SIGNED_SATURATION   = 2;
        constexpr int UNSIGNED_SATURATION = 3;
        int sat_opt                       = std::is_signed_v<DataType> ? SIGNED_SATURATION : UNSIGNED_SATURATION;

        if constexpr (KernelSize == 3 && (std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>))
        {
            // 3x3 Convolution for 8-bit input
            // ------------------------------------------------ //
            // IMPORTANT:                                       //
            // it is caller's job to make the srcs and dst buf's//
            // height round up to multiple of 6                 //
            // ------------------------------------------------ //
            int niter_tw = (tw - 1) / 64 + 1;
            int niter_th = (((th / 2) - 1) / 3) + 1;
            int niter_kw = 1;
            int niter_kh = 2;

            AgenWrapper w0, w1, w2, w3;

            agen agen_ld_coef = init((dvchar *)knl_tp);
            w0.size           = sizeof(char);
            w0.n1             = niter_kh;
            w0.s1             = knl_lp;
            INIT_AGEN1(agen_ld_coef, w0);

            agen agen_ld_src_even = init((dvchar *)src_even_tp);
            w1.size               = sizeof(char);
            w1.n1                 = 2;
            w1.n2                 = 4;
            w1.n3                 = niter_tw;
            w1.n4                 = niter_th;
            w1.s1                 = 32;
            w1.s2                 = src_even_lp;
            w1.s3                 = 64;
            w1.s4                 = src_even_lp * 3;
            INIT_AGEN4(agen_ld_src_even, w1);
            agen_ld_src_even = update_agen_cb_start(agen_ld_src_even, (intptr_t)cb_even);
            agen_ld_src_even = update_agen_cb_size(agen_ld_src_even, cb_len * sizeof(char));

            agen agen_ld_src_odd = init((dvchar *)src_odd_tp);
            w2.size              = sizeof(char);
            w2.n1                = 2;
            w2.n2                = 4;
            w2.n3                = niter_tw;
            w2.n4                = niter_th;
            w2.s1                = 32;
            w2.s2                = src_odd_lp;
            w2.s3                = 64;
            w2.s4                = src_odd_lp * 3;
            INIT_AGEN4(agen_ld_src_odd, w2);
            agen_ld_src_odd = update_agen_cb_start(agen_ld_src_odd, (intptr_t)cb_odd);
            agen_ld_src_odd = update_agen_cb_size(agen_ld_src_odd, cb_len * sizeof(char));

            agen agen_st_dst = init((dvchar *)dst_tp);
            w3.size          = sizeof(char);
            w3.n1            = 6;
            w3.n2            = niter_tw;
            w3.n3            = niter_th;
            w3.s1            = dst_lp;
            w3.s2            = 64;
            w3.s3            = dst_lp * 6;
            INIT_AGEN3(agen_st_dst, w3);
            agen_st_dst.round      = qbits;
            agen_st_dst.sat_lim_lo = low_bound;
            agen_st_dst.sat_val_lo = low_bound;
            agen_st_dst.sat_lim_hi = high_bound;
            agen_st_dst.sat_val_hi = high_bound;
            agen_st_dst.sat_opt    = sat_opt;

            m_cfgs[0] = extract_agen_cfg(agen_ld_src_even);
            m_cfgs[1] = extract_agen_cfg(agen_ld_src_odd);
            m_cfgs[2] = extract_agen_cfg(agen_ld_coef);
            m_cfgs[3] = extract_agen_cfg(agen_st_dst);
            m_niter   = niter_tw * niter_th;
        }
        else if constexpr (KernelSize == 3 && (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>))
        {
            // 3x3 Convolution for 16-bit input
            int vecw     = chess_elementsof(dvshortx);
            int niter_tw = (tw - 1) / vecw + 1;
            int niter_th = (((th / 2) - 1) / 2) + 1;
            int niter_kw = 2;
            int niter_kh = 2;

            AgenWrapper w0, w1, w2, w3;

            agen agen_ld_coef = init((dvshort *)knl_tp);
            w0.size           = sizeof(short);
            w0.n1             = niter_kw;
            w0.n2             = niter_kh;
            w0.s1             = 4;
            w0.s2             = knl_lp;
            INIT_AGEN2(agen_ld_coef, w0);

            agen agen_ld_src_even = init((dvshort *)src_even_tp);
            w1.size               = sizeof(short);
            w1.n1                 = 2;
            w1.n2                 = 3;
            w1.n3                 = niter_tw;
            w1.n4                 = niter_th;
            w1.s1                 = 4;
            w1.s2                 = src_even_lp;
            w1.s3                 = vecw;
            w1.s4                 = src_even_lp * 2;
            INIT_AGEN4(agen_ld_src_even, w1);
            agen_ld_src_even = update_agen_cb_start(agen_ld_src_even, (intptr_t)cb_even);
            agen_ld_src_even = update_agen_cb_size(agen_ld_src_even, cb_len * sizeof(short));

            agen agen_ld_src_odd = init((dvshort *)src_odd_tp);
            w2.size              = sizeof(short);
            w2.n1                = 2;
            w2.n2                = 3;
            w2.n3                = niter_tw;
            w2.n4                = niter_th;
            w2.s1                = 4;
            w2.s2                = src_odd_lp;
            w2.s3                = vecw;
            w2.s4                = src_odd_lp * 2;
            INIT_AGEN4(agen_ld_src_odd, w2);
            agen_ld_src_odd = update_agen_cb_start(agen_ld_src_odd, (intptr_t)cb_odd);
            agen_ld_src_odd = update_agen_cb_size(agen_ld_src_odd, cb_len * sizeof(short));

            agen agen_st_dst = init((dvshort *restrict)dst_tp);
            w3.size          = sizeof(short);
            w3.n1            = 4;
            w3.n2            = niter_tw;
            w3.n3            = niter_th;
            w3.s1            = dst_lp;
            w3.s2            = vecw;
            w3.s3            = dst_lp * 4;
            INIT_AGEN3(agen_st_dst, w3);
            agen_st_dst.round      = qbits;
            agen_st_dst.sat_lim_lo = low_bound;
            agen_st_dst.sat_val_lo = low_bound;
            agen_st_dst.sat_lim_hi = high_bound;
            agen_st_dst.sat_val_hi = high_bound;
            agen_st_dst.sat_opt    = sat_opt;

            m_cfgs[0] = extract_agen_cfg(agen_ld_src_even);
            m_cfgs[1] = extract_agen_cfg(agen_ld_src_odd);
            m_cfgs[2] = extract_agen_cfg(agen_ld_coef);
            m_cfgs[3] = extract_agen_cfg(agen_st_dst);
            m_niter   = niter_tw * niter_th;
        }
        else if constexpr (KernelSize == 5 && (std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>))
        {
            // 5x5 Convolution for 8-bit input
            int niter_tw = (tw - 1) / 32 + 1;
            int niter_th = (th - 1) / 8 + 1;
            int niter_kw = 2;
            int niter_kh = 3;

            AgenWrapper w0, w1, w2, w3;

            agen agen_ld_coef = init((dvchar *)knl_tp);
            w0.size           = sizeof(char);
            w0.n1             = niter_kw;
            w0.n2             = niter_kh;
            w0.s1             = 4;      // go to next horizontal 4 taps
            w0.s2             = knl_lp; // lp = 4 * KWA
            INIT_AGEN2(agen_ld_coef, w0);

            agen agen_ld_src_even = init((dvchar *)src_even_tp);
            w1.size               = sizeof(char);
            w1.n1                 = 6;
            w1.n2                 = 2; // 5x5 kernel padded to 8x6. 8 / 4 = 2 when horizontally
            w1.n3                 = niter_tw;
            w1.n4                 = niter_th;
            w1.s1                 = src_even_lp;
            w1.s2                 = 4;
            w1.s3                 = 32;
            w1.s4                 = src_even_lp * 4;
            INIT_AGEN4(agen_ld_src_even, w1);
            agen_ld_src_even = update_agen_cb_start(agen_ld_src_even, (intptr_t)cb_even);
            agen_ld_src_even = update_agen_cb_size(agen_ld_src_even, cb_len * sizeof(char));

            agen agen_ld_src_odd = init((dvchar *)src_odd_tp);
            w2.size              = sizeof(char);
            w2.n1                = 6;
            w2.n2                = 2; // 5x5 kernel padded to 8x6. 8 / 4 = 2 when horizontally
            w2.n3                = niter_tw;
            w2.n4                = niter_th;
            w2.s1                = src_odd_lp;
            w2.s2                = 4;
            w2.s3                = 32;
            w2.s4                = src_odd_lp * 4;
            INIT_AGEN4(agen_ld_src_odd, w2);
            agen_ld_src_odd = update_agen_cb_start(agen_ld_src_odd, (intptr_t)cb_odd);
            agen_ld_src_odd = update_agen_cb_size(agen_ld_src_odd, cb_len * sizeof(char));

            agen agen_st_dst = init((dvchar *)dst_tp);
            w3.size          = sizeof(char);
            w3.n1            = 8;
            w3.n2            = niter_tw;
            w3.n3            = niter_th;
            w3.s1            = dst_lp;
            w3.s2            = 32;
            w3.s3            = dst_lp * 8;
            INIT_AGEN3(agen_st_dst, w3);
            agen_st_dst.round      = qbits;
            agen_st_dst.sat_lim_lo = low_bound;
            agen_st_dst.sat_val_lo = low_bound;
            agen_st_dst.sat_lim_hi = high_bound;
            agen_st_dst.sat_val_hi = high_bound;
            agen_st_dst.sat_opt    = sat_opt;

            m_cfgs[0] = extract_agen_cfg(agen_ld_src_even);
            m_cfgs[1] = extract_agen_cfg(agen_ld_src_odd);
            m_cfgs[2] = extract_agen_cfg(agen_ld_coef);
            m_cfgs[3] = extract_agen_cfg(agen_st_dst);
            m_niter   = niter_tw * niter_th;
        }
        else if constexpr (KernelSize == 5 && (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>))
        {
            // 5x5 Convolution for 16-bit input
            int vecw     = chess_elementsof(dvshortx);
            int niter_tw = (tw - 1) / vecw + 1;
            int niter_th = th / 4;
            int niter_kw = 4;
            int niter_kh = 3;

            AgenWrapper w0, w1, w2, w3;

            agen agen_ld_coef = init((dvshort *)knl_tp);
            w0.size           = sizeof(short);
            w0.n1             = niter_kw;
            w0.n2             = niter_kh;
            w0.s1             = 4;
            w0.s2             = knl_lp;
            INIT_AGEN2(agen_ld_coef, w0);

            agen agen_ld_src_even = init((dvshort *)src_even_tp);
            w1.size               = sizeof(short);
            w1.n1                 = 2; // 2 for 3x3 or 5x5, 4 pixels apart
            w1.n2                 = 4; // read 8 rows per iteration, deinterleave into 2 superbanks
            w1.n3                 = niter_tw;
            w1.n4                 = niter_th;
            w1.s1                 = 4;
            w1.s2                 = src_even_lp;
            w1.s3                 = vecw;
            w1.s4                 = src_even_lp * 2;
            INIT_AGEN4(agen_ld_src_even, w1);
            agen_ld_src_even = update_agen_cb_start(agen_ld_src_even, (intptr_t)cb_even);
            agen_ld_src_even = update_agen_cb_size(agen_ld_src_even, cb_len * sizeof(short));

            agen agen_ld_src_odd = init((dvshort *)src_odd_tp);
            w2.size              = sizeof(short);
            w2.n1                = 2;
            w2.n2                = 4;
            w2.n3                = niter_tw;
            w2.n4                = niter_th;
            w2.s1                = 4;
            w2.s2                = src_odd_lp;
            w2.s3                = vecw;
            w2.s4                = src_odd_lp * 2;
            INIT_AGEN4(agen_ld_src_odd, w2);
            agen_ld_src_odd = update_agen_cb_start(agen_ld_src_odd, (intptr_t)cb_odd);
            agen_ld_src_odd = update_agen_cb_size(agen_ld_src_odd, cb_len * sizeof(short));

            agen agen_st_dst = init((dvshort *restrict)dst_tp);
            w3.size          = sizeof(short);
            w3.n1            = 4;
            w3.n2            = niter_tw;
            w3.n3            = niter_th;
            w3.s1            = dst_lp;
            w3.s2            = vecw;
            w3.s3            = dst_lp * 4;
            INIT_AGEN3(agen_st_dst, w3);
            agen_st_dst.round      = qbits;
            agen_st_dst.sat_lim_lo = low_bound;
            agen_st_dst.sat_val_lo = low_bound;
            agen_st_dst.sat_lim_hi = high_bound;
            agen_st_dst.sat_val_hi = high_bound;
            agen_st_dst.sat_opt    = sat_opt;

            m_cfgs[0] = extract_agen_cfg(agen_ld_src_even);
            m_cfgs[1] = extract_agen_cfg(agen_ld_src_odd);
            m_cfgs[2] = extract_agen_cfg(agen_ld_coef);
            m_cfgs[3] = extract_agen_cfg(agen_st_dst);
            m_niter   = niter_tw * niter_th;
        }
        else if constexpr (KernelSize == 7 && (std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>))
        {
            // 7x7 Convolution for 8-bit input
            int niter_tw = (tw - 1) / 32 + 1;
            int niter_th = (th - 1) / 6 + 1;
            int niter_kw = 2;
            int niter_kh = 4;

            AgenWrapper w0, w1, w2, w3;

            agen agen_ld_coef = init((dvchar *)knl_tp);
            w0.size           = sizeof(char);
            w0.n1             = niter_kw;
            w0.n2             = niter_kh;
            w0.s1             = 4;      // go to next horizontal 4 taps
            w0.s2             = knl_lp; // lp = 4 * KWA
            INIT_AGEN2(agen_ld_coef, w0);

            agen agen_ld_src_even = init((dvchar *)src_even_tp);
            w1.size               = sizeof(char);
            w1.n1                 = 6;
            w1.n2                 = 2; // 7x7 kernel padded to 8x8. 8 / 4 = 2 when horizontally
            w1.n3                 = niter_tw;
            w1.n4                 = niter_th;
            w1.s1                 = src_even_lp;
            w1.s2                 = 4;
            w1.s3                 = 32;
            w1.s4                 = src_even_lp * 3;
            INIT_AGEN4(agen_ld_src_even, w1);
            agen_ld_src_even = update_agen_cb_start(agen_ld_src_even, (intptr_t)cb_even);
            agen_ld_src_even = update_agen_cb_size(agen_ld_src_even, cb_len * sizeof(char));

            agen agen_ld_src_odd = init((dvchar *)src_odd_tp);
            w2.size              = sizeof(char);
            w2.n1                = 6;
            w2.n2                = 2; // 7x7 kernel padded to 8x8. 8 / 4 = 2 when horizontally
            w2.n3                = niter_tw;
            w2.n4                = niter_th;
            w2.s1                = src_odd_lp;
            w2.s2                = 4;
            w2.s3                = 32;
            w2.s4                = src_odd_lp * 3;
            INIT_AGEN4(agen_ld_src_odd, w2);
            agen_ld_src_odd = update_agen_cb_start(agen_ld_src_odd, (intptr_t)cb_odd);
            agen_ld_src_odd = update_agen_cb_size(agen_ld_src_odd, cb_len * sizeof(char));

            agen agen_st_dst = init((dvchar *)dst_tp);
            w3.size          = sizeof(char);
            w3.n1            = 6;
            w3.n2            = niter_tw;
            w3.n3            = niter_th;
            w3.s1            = dst_lp;
            w3.s2            = 32;
            w3.s3            = dst_lp * 6;
            INIT_AGEN3(agen_st_dst, w3);
            agen_st_dst.round      = qbits;
            agen_st_dst.sat_lim_lo = low_bound;
            agen_st_dst.sat_val_lo = low_bound;
            agen_st_dst.sat_lim_hi = high_bound;
            agen_st_dst.sat_val_hi = high_bound;
            agen_st_dst.sat_opt    = sat_opt;

            m_cfgs[0] = extract_agen_cfg(agen_ld_src_even);
            m_cfgs[1] = extract_agen_cfg(agen_ld_src_odd);
            m_cfgs[2] = extract_agen_cfg(agen_ld_coef);
            m_cfgs[3] = extract_agen_cfg(agen_st_dst);
            m_niter   = niter_tw * niter_th;
        }
        else if constexpr (KernelSize == 7 && (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>))
        {
            // 7x7 Convolution for 16-bit input
            int vecw     = chess_elementsof(dvshortx);
            int niter_tw = (tw - 1) / vecw + 1;
            int niter_th = th / 4;
            int niter_kw = 4;
            int niter_kh = 4;

            AgenWrapper w0, w1, w2, w3;

            agen agen_ld_coef = init((dvshort *)knl_tp);
            w0.size           = sizeof(short);
            w0.n1             = niter_kw;
            w0.n2             = niter_kh;
            w0.s1             = 4;
            w0.s2             = knl_lp;
            INIT_AGEN2(agen_ld_coef, w0);

            agen agen_ld_src_even = init((dvshort *)src_even_tp);
            w1.size               = sizeof(short);
            w1.n1                 = 3;
            w1.n2                 = 5; // read 10 rows per iteration, deinterleave into 2 superbanks
            w1.n3                 = niter_tw;
            w1.n4                 = niter_th;
            w1.s1                 = 4;
            w1.s2                 = src_even_lp;
            w1.s3                 = vecw;
            w1.s4                 = src_even_lp * 2;
            INIT_AGEN4(agen_ld_src_even, w1);
            agen_ld_src_even = update_agen_cb_start(agen_ld_src_even, (intptr_t)cb_even);
            agen_ld_src_even = update_agen_cb_size(agen_ld_src_even, cb_len * sizeof(short));

            agen agen_ld_src_odd = init((dvshort *)src_odd_tp);
            w2.size              = sizeof(short);
            w2.n1                = 3;
            w2.n2                = 5;
            w2.n3                = niter_tw;
            w2.n4                = niter_th;
            w2.s1                = 4;
            w2.s2                = src_odd_lp;
            w2.s3                = vecw;
            w2.s4                = src_odd_lp * 2;
            INIT_AGEN4(agen_ld_src_odd, w2);
            agen_ld_src_odd = update_agen_cb_start(agen_ld_src_odd, (intptr_t)cb_odd);
            agen_ld_src_odd = update_agen_cb_size(agen_ld_src_odd, cb_len * sizeof(short));

            agen agen_st_dst = init((dvshort *restrict)dst_tp);
            w3.size          = sizeof(short);
            w3.n1            = 4;
            w3.n2            = niter_tw;
            w3.n3            = niter_th;
            w3.s1            = dst_lp;
            w3.s2            = vecw;
            w3.s3            = dst_lp * 4;
            INIT_AGEN3(agen_st_dst, w3);
            agen_st_dst.round      = qbits;
            agen_st_dst.sat_lim_lo = low_bound;
            agen_st_dst.sat_val_lo = low_bound;
            agen_st_dst.sat_lim_hi = high_bound;
            agen_st_dst.sat_val_hi = high_bound;
            agen_st_dst.sat_opt    = sat_opt;

            m_cfgs[0] = extract_agen_cfg(agen_ld_src_even);
            m_cfgs[1] = extract_agen_cfg(agen_ld_src_odd);
            m_cfgs[2] = extract_agen_cfg(agen_ld_coef);
            m_cfgs[3] = extract_agen_cfg(agen_st_dst);
            m_niter   = niter_tw * niter_th;
        }
        else
        {
            static_assert(std::is_same_v<DataType, void>, "Unsupported combination of KernelSize and DataType");
        }
    }

    /**
     * Execute the conv2d algorithm with the given kernel size and data type.
     */
    void Execute()
    {
        agen_C agen_ld_coef     = init_agen_C_from_cfg(m_cfgs[2]);
        agen_A agen_ld_src_even = init_agen_A_from_cfg(m_cfgs[0]);
        agen_B agen_ld_src_odd  = init_agen_B_from_cfg(m_cfgs[1]);
        agen_C agen_st_dst      = init_agen_C_from_cfg(m_cfgs[3]);

        int niter = m_niter;

        if constexpr (KernelSize == 3 && (std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>))
        {

            constexpr int8_t m_knl_perm[32] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                               4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
            constexpr int8_t m_src_perm[32] = {0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                               4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

            // 3x3 Convolution for 8-bit input
            vcharx knl_perm = zero_extend(*((vuchar *)(m_knl_perm)));
            vcharx src_perm = zero_extend(*((vuchar *)(m_src_perm)));

            dvcharx vcoef01, vcoef23;
            dvcharx vdata0_lo, vdata1_lo, vdata2_lo, vdata3_lo, vdata4_lo, vdata5_lo, vdata6_lo, vdata7_lo;
            dvcharx vdata0_hi, vdata1_hi, vdata2_hi, vdata3_hi, vdata4_hi, vdata5_hi, vdata6_hi, vdata7_hi;
            dvshortx vacc0_lo, vacc1_lo, vacc2_lo, vacc3_lo, vacc4_lo, vacc5_lo;
            dvshortx vacc0_hi, vacc1_hi, vacc2_hi, vacc3_hi, vacc4_hi, vacc5_hi;

            chess_separator_scheduler();

            vcoef01 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef23 = dvload_perm_coef(agen_ld_coef, knl_perm);

            // 16 loads for 8 src rows    -> 8 VLIW
            // 12 computes for 6 dst rows -> 6 VLIW -> 1.33 VLIW each 64 outputs
            // 6 stores for 6 dst rows    -> 6 VLIW
            for (int i = 0; i < niter; i++) chess_loop_range(8, )
            chess_prepare_for_pipelining
            {
                vdata0_lo = dvload_perm(agen_ld_src_odd, src_perm);
                vdata1_lo = dvload_perm(agen_ld_src_even, src_perm);
                vdata0_hi = dvload_perm(agen_ld_src_odd, src_perm);
                vdata1_hi = dvload_perm(agen_ld_src_even, src_perm);
                vdata2_lo = dvload_perm(agen_ld_src_odd, src_perm);
                vdata3_lo = dvload_perm(agen_ld_src_even, src_perm);
                vdata2_hi = dvload_perm(agen_ld_src_odd, src_perm);
                vdata3_hi = dvload_perm(agen_ld_src_even, src_perm);
                vdata4_lo = dvload_perm(agen_ld_src_odd, src_perm);
                vdata5_lo = dvload_perm(agen_ld_src_even, src_perm);
                vdata4_hi = dvload_perm(agen_ld_src_odd, src_perm);
                vdata5_hi = dvload_perm(agen_ld_src_even, src_perm);
                vdata6_lo = dvload_perm(agen_ld_src_odd, src_perm);
                vdata7_lo = dvload_perm(agen_ld_src_even, src_perm);
                vdata6_hi = dvload_perm(agen_ld_src_odd, src_perm);
                vdata7_hi = dvload_perm(agen_ld_src_even, src_perm);

                vfilt4x2x2_bbh(vdata0_lo, vdata1_lo, vcoef01, vacc0_lo, vacc1_lo, 0, vacc0_lo, vacc1_lo);
                vfilt4x2x2_bbh(vdata0_hi, vdata1_hi, vcoef01, vacc0_hi, vacc1_hi, 0, vacc0_hi, vacc1_hi);
                chess_separator();
                vfilt4x2x2_bbh(vdata2_lo, vdata3_lo, vcoef23, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2x2_bbh(vdata2_hi, vdata3_hi, vcoef23, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                chess_separator();

                vfilt4x2x2_bbh(vdata2_lo, vdata3_lo, vcoef01, vacc2_lo, vacc3_lo, 0, vacc2_lo, vacc3_lo);
                vfilt4x2x2_bbh(vdata2_hi, vdata3_hi, vcoef01, vacc2_hi, vacc3_hi, 0, vacc2_hi, vacc3_hi);
                chess_separator();
                vfilt4x2x2_bbh(vdata4_lo, vdata5_lo, vcoef23, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2x2_bbh(vdata4_hi, vdata5_hi, vcoef23, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                chess_separator();

                vfilt4x2x2_bbh(vdata4_lo, vdata5_lo, vcoef01, vacc4_lo, vacc5_lo, 0, vacc4_lo, vacc5_lo);
                vfilt4x2x2_bbh(vdata4_hi, vdata5_hi, vcoef01, vacc4_hi, vacc5_hi, 0, vacc4_hi, vacc5_hi);
                chess_separator();
                vfilt4x2x2_bbh(vdata6_lo, vdata7_lo, vcoef23, vacc4_lo, vacc5_lo, -1, vacc4_lo, vacc5_lo);
                vfilt4x2x2_bbh(vdata6_hi, vdata7_hi, vcoef23, vacc4_hi, vacc5_hi, -1, vacc4_hi, vacc5_hi);
                chess_separator();

                vstore_i2(vacc0_lo, vacc0_hi, agen_st_dst);
                vstore_i2(vacc1_lo, vacc1_hi, agen_st_dst);
                vstore_i2(vacc2_lo, vacc2_hi, agen_st_dst);
                vstore_i2(vacc3_lo, vacc3_hi, agen_st_dst);
                vstore_i2(vacc4_lo, vacc4_hi, agen_st_dst);
                vstore_i2(vacc5_lo, vacc5_hi, agen_st_dst);
            }
        }
        else if constexpr (KernelSize == 3 && (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>))
        {
            // 3x3 Convolution for 16-bit input
            constexpr int8_t m_knl_perm[32] = {0, 1, 2,  3,  0, 1, 2,  3,  0, 1, 2,  3,  0, 1, 2,  3,
                                               8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11};

            vcharx knl_perm = zero_extend(*((vuchar *)(m_knl_perm)));

            dvshortx vdata0a, vdata0b, vdata1a, vdata1b, vdata2a, vdata2b;
            dvshortx vdata3a, vdata3b, vdata4a, vdata4b, vdata5a, vdata5b;
            dvshortx vcoef0, vcoef1, vcoef2, vcoef3;
            dvintx vacc0_lo, vacc0_hi, vacc1_lo, vacc1_hi, vacc2_lo, vacc2_hi, vacc3_lo, vacc3_hi;

            chess_separator_scheduler();

            vcoef0 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef1 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef2 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef3 = dvload_perm_coef(agen_ld_coef, knl_perm);

            // 12 loads for 6 src rows    -> 6 VLIW
            // 16 computes for 4 dst rows -> 8 VLIW -> 2 VLIW each 32 outputs
            // 4 stores for 4 dst rows    -> 4 VLIW
            for (int i = 0; i < niter; i++) chess_loop_range(8, )
            chess_prepare_for_pipelining
            {
                vdata0a = dvload(agen_ld_src_odd);
                vdata0b = dvload(agen_ld_src_odd);
                vdata1a = dvload(agen_ld_src_even);
                vdata1b = dvload(agen_ld_src_even);
                vdata2a = dvload(agen_ld_src_odd);
                vdata2b = dvload(agen_ld_src_odd);
                vdata3a = dvload(agen_ld_src_even);
                vdata3b = dvload(agen_ld_src_even);
                vdata4a = dvload(agen_ld_src_odd);
                vdata4b = dvload(agen_ld_src_odd);
                vdata5a = dvload(agen_ld_src_even);
                vdata5b = dvload(agen_ld_src_even);

                vfilt4x2_hhw(vdata0a.lo, vdata0b.lo, vcoef0, vacc0_lo, vacc1_lo, 0, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata0a.hi, vdata0b.hi, vcoef0, vacc0_hi, vacc1_hi, 0, vacc0_hi, vacc1_hi);
                chess_separator();
                vfilt4x2_hhw(vdata1a.lo, vdata1b.lo, vcoef1, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata1a.hi, vdata1b.hi, vcoef1, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                chess_separator();
                vfilt4x2_hhw(vdata2a.lo, vdata2b.lo, vcoef2, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata2a.hi, vdata2b.hi, vcoef2, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                chess_separator();
                vfilt4x2_hhw(vdata3a.lo, vdata3b.lo, vcoef3, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata3a.hi, vdata3b.hi, vcoef3, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                chess_separator();

                vfilt4x2_hhw(vdata2a.lo, vdata2b.lo, vcoef0, vacc2_lo, vacc3_lo, 0, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata2a.hi, vdata2b.hi, vcoef0, vacc2_hi, vacc3_hi, 0, vacc2_hi, vacc3_hi);
                chess_separator();
                vfilt4x2_hhw(vdata3a.lo, vdata3b.lo, vcoef1, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata3a.hi, vdata3b.hi, vcoef1, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                chess_separator();
                vfilt4x2_hhw(vdata4a.lo, vdata4b.lo, vcoef2, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata4a.hi, vdata4b.hi, vcoef2, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                chess_separator();
                vfilt4x2_hhw(vdata5a.lo, vdata5b.lo, vcoef3, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata5a.hi, vdata5b.hi, vcoef3, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                chess_separator();

                vstore_i2(vacc0_lo, vacc0_hi, agen_st_dst);
                vstore_i2(vacc1_lo, vacc1_hi, agen_st_dst);
                vstore_i2(vacc2_lo, vacc2_hi, agen_st_dst);
                vstore_i2(vacc3_lo, vacc3_hi, agen_st_dst);
            }
        }
        else if constexpr (KernelSize == 5 && (std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>))
        {
            // 5x5 Convolution for 8-bit input
            constexpr int8_t m_knl_perm[32] = {0, 1, 4,  5,  0, 1, 4,  5,  0, 1, 4,  5,  0, 1, 4,  5,
                                               8, 9, 12, 13, 8, 9, 12, 13, 8, 9, 12, 13, 8, 9, 12, 13};
            constexpr int8_t m_src_perm[32] = {0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                               4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

            vcharx knl_perm = zero_extend(*((vuchar *)(m_knl_perm)));
            vcharx src_perm = zero_extend(*((vuchar *)(m_src_perm)));

            dvcharx vcoef0, vcoef1, vcoef2, vcoef3, vcoef4, vcoef5;
            dvcharx chess_storage(DV0) vdata0, chess_storage(DV1) vdata1;
            dvcharx chess_storage(DV2) vdata2, chess_storage(DV3) vdata3;
            dvcharx chess_storage(DV4) vdata4, chess_storage(DV5) vdata5;
            dvcharx chess_storage(DV6) vdata6, chess_storage(DV7) vdata7;
            dvcharx chess_storage(DV8) vdata8, chess_storage(DV9) vdata9;
            dvcharx chess_storage(DV10) vdata10, chess_storage(DV11) vdata11;
            dvcharx chess_storage(DV12) vdata12, chess_storage(DV13) vdata13;
            dvcharx chess_storage(DV14) vdata14, chess_storage(DV15) vdata15;
            dvshortx vacc0, vacc1, vacc2, vacc3;
            dvshortx vacc4, vacc5, vacc6, vacc7;
            dvshortx vacc8, vacc9, vacc10, vacc11;
            dvshortx vacc12, vacc13, vacc14, vacc15;

            chess_separator_scheduler();

            //   |   dvcoef0   |   dvcoef1   |     //
            //   |   dvcoef2   |   dvcoef3   |     //
            //   |   dvcoef4   |   dvcoef5   |     //
            vcoef0 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef1 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef2 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef3 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef4 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef5 = dvload_perm_coef(agen_ld_coef, knl_perm);

            vdata0  = dvload_perm(agen_ld_src_even, src_perm);
            vdata1  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata2  = dvload_perm(agen_ld_src_even, src_perm);
            vdata3  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata4  = dvload_perm(agen_ld_src_even, src_perm);
            vdata5  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata6  = dvload_perm(agen_ld_src_even, src_perm);
            vdata7  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata8  = dvload_perm(agen_ld_src_even, src_perm);
            vdata9  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata10 = dvload_perm(agen_ld_src_even, src_perm);
            vdata11 = dvload_perm(agen_ld_src_odd, src_perm);

            vfilt4x2x2_bbh(vdata0, vdata1, vcoef0, vacc0, vacc1, 0, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef2, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef4, vacc0, vacc1, -1, vacc0, vacc1);

            vfilt4x2x2_bbh(vdata2, vdata3, vcoef0, vacc2, vacc3, 0, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef2, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef4, vacc2, vacc3, -1, vacc2, vacc3);

            vfilt4x2x2_bbh(vdata4, vdata5, vcoef0, vacc4, vacc5, 0, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef2, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef4, vacc4, vacc5, -1, vacc4, vacc5);

            vfilt4x2x2_bbh(vdata6, vdata7, vcoef0, vacc6, vacc7, 0, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef2, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef4, vacc6, vacc7, -1, vacc6, vacc7);

            vdata12 = dvload_perm(agen_ld_src_even, src_perm);
            vdata13 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata14 = dvload_perm(agen_ld_src_even, src_perm);
            vdata15 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata0  = dvload_perm(agen_ld_src_even, src_perm);
            vdata1  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata2  = dvload_perm(agen_ld_src_even, src_perm);
            vdata3  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata4  = dvload_perm(agen_ld_src_even, src_perm);
            vdata5  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata6  = dvload_perm(agen_ld_src_even, src_perm);
            vdata7  = dvload_perm(agen_ld_src_odd, src_perm);

            vfilt4x2x2_bbh(vdata12, vdata13, vcoef1, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef3, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef5, vacc0, vacc1, -1, vacc0, vacc1);

            vfilt4x2x2_bbh(vdata14, vdata15, vcoef1, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef3, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef5, vacc2, vacc3, -1, vacc2, vacc3);

            vfilt4x2x2_bbh(vdata0, vdata1, vcoef1, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef3, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef5, vacc4, vacc5, -1, vacc4, vacc5);

            vfilt4x2x2_bbh(vdata2, vdata3, vcoef1, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef3, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef5, vacc6, vacc7, -1, vacc6, vacc7);

            vdata8  = dvload_perm(agen_ld_src_even, src_perm);
            vdata9  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata10 = dvload_perm(agen_ld_src_even, src_perm);
            vdata11 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata12 = dvload_perm(agen_ld_src_even, src_perm);
            vdata13 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata14 = dvload_perm(agen_ld_src_even, src_perm);
            vdata15 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata0  = dvload_perm(agen_ld_src_even, src_perm);
            vdata1  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata2  = dvload_perm(agen_ld_src_even, src_perm);
            vdata3  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata4  = dvload_perm(agen_ld_src_even, src_perm);
            vdata5  = dvload_perm(agen_ld_src_odd, src_perm);

            vstore_i_hb(vacc0, agen_st_dst);
            vstore_i_hb(vacc1, agen_st_dst);
            vstore_i_hb(vacc2, agen_st_dst);
            vstore_i_hb(vacc3, agen_st_dst);

            // Need to make sure niter is even because of manual unroll 2. The prediction off will prevent any extra write.
            niter = (niter % 2 == 0) ? niter : niter + 1;

            // 24 loads    -> 12 VLIW
            // 24 computes -> 12 VLIW -> 1.5 VLIW per 32 outputs
            // 8 stores    ->  8 VLIW
            for (int i = 0; i < (niter - 2) / 2; i++) chess_loop_range(8, ) //chess_prepare_for_pipelining
            {
                vstore_i_hb(vacc4, agen_st_dst);
                vstore_i_hb(vacc5, agen_st_dst);
                vstore_i_hb(vacc6, agen_st_dst);
                vstore_i_hb(vacc7, agen_st_dst);

                vfilt4x2x2_bbh(vdata8, vdata9, vcoef0, vacc8, vacc9, 0, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef2, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef4, vacc8, vacc9, -1, vacc8, vacc9);

                vfilt4x2x2_bbh(vdata10, vdata11, vcoef0, vacc10, vacc11, 0, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef2, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef4, vacc10, vacc11, -1, vacc10, vacc11);

                vfilt4x2x2_bbh(vdata12, vdata13, vcoef0, vacc12, vacc13, 0, vacc12, vacc13);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef2, vacc12, vacc13, -1, vacc12, vacc13);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef4, vacc12, vacc13, -1, vacc12, vacc13);

                vfilt4x2x2_bbh(vdata14, vdata15, vcoef0, vacc14, vacc15, 0, vacc14, vacc15);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef2, vacc14, vacc15, -1, vacc14, vacc15);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef4, vacc14, vacc15, -1, vacc14, vacc15);

                vdata6  = dvload_perm(agen_ld_src_even, src_perm);
                vdata7  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata8  = dvload_perm(agen_ld_src_even, src_perm);
                vdata9  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata10 = dvload_perm(agen_ld_src_even, src_perm);
                vdata11 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata12 = dvload_perm(agen_ld_src_even, src_perm);
                vdata13 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata14 = dvload_perm(agen_ld_src_even, src_perm);
                vdata15 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata0  = dvload_perm(agen_ld_src_even, src_perm);
                vdata1  = dvload_perm(agen_ld_src_odd, src_perm);

                vfilt4x2x2_bbh(vdata4, vdata5, vcoef1, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef3, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef5, vacc8, vacc9, -1, vacc8, vacc9);

                vfilt4x2x2_bbh(vdata6, vdata7, vcoef1, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef3, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef5, vacc10, vacc11, -1, vacc10, vacc11);

                vfilt4x2x2_bbh(vdata8, vdata9, vcoef1, vacc12, vacc13, -1, vacc12, vacc13);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef3, vacc12, vacc13, -1, vacc12, vacc13);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef5, vacc12, vacc13, -1, vacc12, vacc13);

                vfilt4x2x2_bbh(vdata10, vdata11, vcoef1, vacc14, vacc15, -1, vacc14, vacc15);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef3, vacc14, vacc15, -1, vacc14, vacc15);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef5, vacc14, vacc15, -1, vacc14, vacc15);

                vstore_i_hb(vacc8, agen_st_dst);
                vstore_i_hb(vacc9, agen_st_dst);
                vstore_i_hb(vacc10, agen_st_dst);
                vstore_i_hb(vacc11, agen_st_dst);
                vstore_i_hb(vacc12, agen_st_dst);
                vstore_i_hb(vacc13, agen_st_dst);
                vstore_i_hb(vacc14, agen_st_dst);
                vstore_i_hb(vacc15, agen_st_dst);

                vdata2  = dvload_perm(agen_ld_src_even, src_perm);
                vdata3  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata4  = dvload_perm(agen_ld_src_even, src_perm);
                vdata5  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata6  = dvload_perm(agen_ld_src_even, src_perm);
                vdata7  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata8  = dvload_perm(agen_ld_src_even, src_perm);
                vdata9  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata10 = dvload_perm(agen_ld_src_even, src_perm);
                vdata11 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata12 = dvload_perm(agen_ld_src_even, src_perm);
                vdata13 = dvload_perm(agen_ld_src_odd, src_perm);

                vfilt4x2x2_bbh(vdata0, vdata1, vcoef0, vacc0, vacc1, 0, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef2, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef4, vacc0, vacc1, -1, vacc0, vacc1);

                vfilt4x2x2_bbh(vdata2, vdata3, vcoef0, vacc2, vacc3, 0, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef2, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef4, vacc2, vacc3, -1, vacc2, vacc3);

                vfilt4x2x2_bbh(vdata4, vdata5, vcoef0, vacc4, vacc5, 0, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef2, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef4, vacc4, vacc5, -1, vacc4, vacc5);

                vfilt4x2x2_bbh(vdata6, vdata7, vcoef0, vacc6, vacc7, 0, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef2, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef4, vacc6, vacc7, -1, vacc6, vacc7);

                vdata14 = dvload_perm(agen_ld_src_even, src_perm);
                vdata15 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata0  = dvload_perm(agen_ld_src_even, src_perm);
                vdata1  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata2  = dvload_perm(agen_ld_src_even, src_perm);
                vdata3  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata4  = dvload_perm(agen_ld_src_even, src_perm);
                vdata5  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata6  = dvload_perm(agen_ld_src_even, src_perm);
                vdata7  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata8  = dvload_perm(agen_ld_src_even, src_perm);
                vdata9  = dvload_perm(agen_ld_src_odd, src_perm);

                vfilt4x2x2_bbh(vdata12, vdata13, vcoef1, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef3, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef5, vacc0, vacc1, -1, vacc0, vacc1);

                vfilt4x2x2_bbh(vdata14, vdata15, vcoef1, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef3, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef5, vacc2, vacc3, -1, vacc2, vacc3);

                vfilt4x2x2_bbh(vdata0, vdata1, vcoef1, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef3, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef5, vacc4, vacc5, -1, vacc4, vacc5);

                vfilt4x2x2_bbh(vdata2, vdata3, vcoef1, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef3, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef5, vacc6, vacc7, -1, vacc6, vacc7);

                vdata10 = dvload_perm(agen_ld_src_even, src_perm);
                vdata11 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata12 = dvload_perm(agen_ld_src_even, src_perm);
                vdata13 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata14 = dvload_perm(agen_ld_src_even, src_perm);
                vdata15 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata0  = dvload_perm(agen_ld_src_even, src_perm);
                vdata1  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata2  = dvload_perm(agen_ld_src_even, src_perm);
                vdata3  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata4  = dvload_perm(agen_ld_src_even, src_perm);
                vdata5  = dvload_perm(agen_ld_src_odd, src_perm);

                vstore_i_hb(vacc0, agen_st_dst);
                vstore_i_hb(vacc1, agen_st_dst);
                vstore_i_hb(vacc2, agen_st_dst);
                vstore_i_hb(vacc3, agen_st_dst);
            }

            vstore_i_hb(vacc4, agen_st_dst);
            vstore_i_hb(vacc5, agen_st_dst);
            vstore_i_hb(vacc6, agen_st_dst);
            vstore_i_hb(vacc7, agen_st_dst);

            vfilt4x2x2_bbh(vdata8, vdata9, vcoef0, vacc8, vacc9, 0, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef2, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef4, vacc8, vacc9, -1, vacc8, vacc9);

            vfilt4x2x2_bbh(vdata10, vdata11, vcoef0, vacc10, vacc11, 0, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef2, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef4, vacc10, vacc11, -1, vacc10, vacc11);

            vfilt4x2x2_bbh(vdata12, vdata13, vcoef0, vacc12, vacc13, 0, vacc12, vacc13);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef2, vacc12, vacc13, -1, vacc12, vacc13);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef4, vacc12, vacc13, -1, vacc12, vacc13);

            vfilt4x2x2_bbh(vdata14, vdata15, vcoef0, vacc14, vacc15, 0, vacc14, vacc15);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef2, vacc14, vacc15, -1, vacc14, vacc15);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef4, vacc14, vacc15, -1, vacc14, vacc15);

            vdata6  = dvload_perm(agen_ld_src_even, src_perm);
            vdata7  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata8  = dvload_perm(agen_ld_src_even, src_perm);
            vdata9  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata10 = dvload_perm(agen_ld_src_even, src_perm);
            vdata11 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata12 = dvload_perm(agen_ld_src_even, src_perm);
            vdata13 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata14 = dvload_perm(agen_ld_src_even, src_perm);
            vdata15 = dvload_perm(agen_ld_src_odd, src_perm);

            vfilt4x2x2_bbh(vdata4, vdata5, vcoef1, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef3, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef5, vacc8, vacc9, -1, vacc8, vacc9);

            vfilt4x2x2_bbh(vdata6, vdata7, vcoef1, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef3, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef5, vacc10, vacc11, -1, vacc10, vacc11);

            vfilt4x2x2_bbh(vdata8, vdata9, vcoef1, vacc12, vacc13, -1, vacc12, vacc13);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef3, vacc12, vacc13, -1, vacc12, vacc13);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef5, vacc12, vacc13, -1, vacc12, vacc13);

            vfilt4x2x2_bbh(vdata10, vdata11, vcoef1, vacc14, vacc15, -1, vacc14, vacc15);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef3, vacc14, vacc15, -1, vacc14, vacc15);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef5, vacc14, vacc15, -1, vacc14, vacc15);

            vstore_i_hb(vacc8, agen_st_dst);
            vstore_i_hb(vacc9, agen_st_dst);
            vstore_i_hb(vacc10, agen_st_dst);
            vstore_i_hb(vacc11, agen_st_dst);
            vstore_i_hb(vacc12, agen_st_dst);
            vstore_i_hb(vacc13, agen_st_dst);
            vstore_i_hb(vacc14, agen_st_dst);
            vstore_i_hb(vacc15, agen_st_dst);
        }
        else if constexpr (KernelSize == 5 && (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>))
        {
            // 5x5 Convolution for 16-bit input
            constexpr int8_t m_knl_perm[32] = {0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,
                                               16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19};

            vcharx knl_perm = zero_extend(*((vuchar *)(m_knl_perm)));

            dvshortx vdata0a, vdata0b, vdata1a, vdata1b, vdata2a, vdata2b;
            dvshortx vdata3a, vdata3b, vdata4a, vdata4b, vdata5a, vdata5b;
            dvshortx vdata6a, vdata6b, vdata7a, vdata7b;
            dvshortx vcoef00, vcoef01, vcoef10, vcoef11, vcoef20, vcoef21;
            dvshortx vcoef30, vcoef31, vcoef40, vcoef41, vcoef50, vcoef51;
            dvintx vacc0_lo, vacc1_lo, vacc0_hi, vacc1_hi;
            dvintx vacc2_lo, vacc3_lo, vacc2_hi, vacc3_hi;

            vcoef00 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef01 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef10 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef11 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef20 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef21 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef30 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef31 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef40 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef41 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef50 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef51 = dvload_perm_coef(agen_ld_coef, knl_perm);

            // 16 loads for 8 src rows    ->  8 VLIW
            // 48 computes for 4 dst rows -> 24 VLIW -> 6 VLIW each 32 outputs
            // 4 stores for 4 dst rows    ->  4 VLIW
            for (int i = 0; i < niter; i++) chess_loop_range(3, )
            chess_prepare_for_pipelining
            {
                vdata0a = dvload(agen_ld_src_even);
                vdata0b = dvload(agen_ld_src_even);
                vdata1a = dvload(agen_ld_src_odd);
                vdata1b = dvload(agen_ld_src_odd);
                vdata2a = dvload(agen_ld_src_even);
                vdata2b = dvload(agen_ld_src_even);
                vdata3a = dvload(agen_ld_src_odd);
                vdata3b = dvload(agen_ld_src_odd);
                vdata4a = dvload(agen_ld_src_even);
                vdata4b = dvload(agen_ld_src_even);
                vdata5a = dvload(agen_ld_src_odd);
                vdata5b = dvload(agen_ld_src_odd);
                vdata6a = dvload(agen_ld_src_even);
                vdata6b = dvload(agen_ld_src_even);
                vdata7a = dvload(agen_ld_src_odd);
                vdata7b = dvload(agen_ld_src_odd);

                vfilt4x2_hhw(vdata0a.lo, vdata0b.lo, vcoef00, vacc0_lo, vacc1_lo, 0, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata0b.lo, vdata0a.lo, vcoef01, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata1a.lo, vdata1b.lo, vcoef10, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata1b.lo, vdata1a.lo, vcoef11, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata2a.lo, vdata2b.lo, vcoef20, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata2b.lo, vdata2a.lo, vcoef21, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata3a.lo, vdata3b.lo, vcoef30, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata3b.lo, vdata3a.lo, vcoef31, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata4a.lo, vdata4b.lo, vcoef40, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata4b.lo, vdata4a.lo, vcoef41, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata5a.lo, vdata5b.lo, vcoef50, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata5b.lo, vdata5a.lo, vcoef51, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);

                vfilt4x2_hhw(vdata0a.hi, vdata0b.hi, vcoef00, vacc0_hi, vacc1_hi, 0, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata0b.hi, vdata0a.lo, vcoef01, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata1a.hi, vdata1b.hi, vcoef10, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata1b.hi, vdata1a.lo, vcoef11, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata2a.hi, vdata2b.hi, vcoef20, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata2b.hi, vdata2a.lo, vcoef21, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata3a.hi, vdata3b.hi, vcoef30, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata3b.hi, vdata3a.lo, vcoef31, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata4a.hi, vdata4b.hi, vcoef40, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata4b.hi, vdata4a.lo, vcoef41, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata5a.hi, vdata5b.hi, vcoef50, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata5b.hi, vdata5a.lo, vcoef51, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);

                vfilt4x2_hhw(vdata2a.lo, vdata2b.lo, vcoef00, vacc2_lo, vacc3_lo, 0, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata2b.lo, vdata2a.lo, vcoef01, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata3a.lo, vdata3b.lo, vcoef10, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata3b.lo, vdata3a.lo, vcoef11, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata4a.lo, vdata4b.lo, vcoef20, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata4b.lo, vdata4a.lo, vcoef21, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata5a.lo, vdata5b.lo, vcoef30, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata5b.lo, vdata5a.lo, vcoef31, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata6a.lo, vdata6b.lo, vcoef40, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata6b.lo, vdata6a.lo, vcoef41, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata7a.lo, vdata7b.lo, vcoef50, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata7b.lo, vdata7a.lo, vcoef51, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);

                vfilt4x2_hhw(vdata2a.hi, vdata2b.hi, vcoef00, vacc2_hi, vacc3_hi, 0, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata2b.hi, vdata2a.lo, vcoef01, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata3a.hi, vdata3b.hi, vcoef10, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata3b.hi, vdata3a.lo, vcoef11, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata4a.hi, vdata4b.hi, vcoef20, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata4b.hi, vdata4a.lo, vcoef21, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata5a.hi, vdata5b.hi, vcoef30, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata5b.hi, vdata5a.lo, vcoef31, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata6a.hi, vdata6b.hi, vcoef40, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata6b.hi, vdata6a.lo, vcoef41, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata7a.hi, vdata7b.hi, vcoef50, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata7b.hi, vdata7a.lo, vcoef51, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);

                vstore_i2(vacc0_lo, vacc0_hi, agen_st_dst);
                vstore_i2(vacc1_lo, vacc1_hi, agen_st_dst);
                vstore_i2(vacc2_lo, vacc2_hi, agen_st_dst);
                vstore_i2(vacc3_lo, vacc3_hi, agen_st_dst);
            }
        }
        else if constexpr (KernelSize == 7 && (std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>))
        {
            // 7x7 Convolution for 8-bit input
            constexpr int8_t m_knl_perm[32] = {0, 1, 4,  5,  0, 1, 4,  5,  0, 1, 4,  5,  0, 1, 4,  5,
                                               8, 9, 12, 13, 8, 9, 12, 13, 8, 9, 12, 13, 8, 9, 12, 13};
            constexpr int8_t m_src_perm[32] = {0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                               4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

            vcharx knl_perm = zero_extend(*((vuchar *)(m_knl_perm)));
            vcharx src_perm = zero_extend(*((vuchar *)(m_src_perm)));

            dvcharx vcoef0, vcoef1, vcoef2, vcoef3, vcoef4, vcoef5, vcoef6, vcoef7;
            dvcharx chess_storage(DV0) vdata0, chess_storage(DV1) vdata1;
            dvcharx chess_storage(DV2) vdata2, chess_storage(DV3) vdata3;
            dvcharx chess_storage(DV4) vdata4, chess_storage(DV5) vdata5;
            dvcharx chess_storage(DV6) vdata6, chess_storage(DV7) vdata7;
            dvcharx chess_storage(DV8) vdata8, chess_storage(DV9) vdata9;
            dvcharx chess_storage(DV10) vdata10, chess_storage(DV11) vdata11;
            dvcharx chess_storage(DV12) vdata12, chess_storage(DV13) vdata13;
            dvcharx chess_storage(DV14) vdata14, chess_storage(DV15) vdata15;
            dvshortx vacc0, vacc1, vacc2, vacc3;
            dvshortx vacc4, vacc5, vacc6, vacc7;
            dvshortx vacc8, vacc9, vacc10, vacc11;

            chess_separator_scheduler();

            //   |   dvcoef0   |   dvcoef1   |     //
            //   |   dvcoef2   |   dvcoef3   |     //
            //   |   dvcoef4   |   dvcoef5   |     //
            //   |   dvcoef6   |   dvcoef7   |     //
            vcoef0 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef1 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef2 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef3 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef4 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef5 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef6 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef7 = dvload_perm_coef(agen_ld_coef, knl_perm);

            vdata0  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata1  = dvload_perm(agen_ld_src_even, src_perm);
            vdata2  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata3  = dvload_perm(agen_ld_src_even, src_perm);
            vdata4  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata5  = dvload_perm(agen_ld_src_even, src_perm);
            vdata6  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata7  = dvload_perm(agen_ld_src_even, src_perm);
            vdata8  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata9  = dvload_perm(agen_ld_src_even, src_perm);
            vdata10 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata11 = dvload_perm(agen_ld_src_even, src_perm);

            vfilt4x2x2_bbh(vdata0, vdata1, vcoef0, vacc0, vacc1, 0, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef2, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef4, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef6, vacc0, vacc1, -1, vacc0, vacc1);

            vfilt4x2x2_bbh(vdata2, vdata3, vcoef0, vacc2, vacc3, 0, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef2, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef4, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef6, vacc2, vacc3, -1, vacc2, vacc3);

            vfilt4x2x2_bbh(vdata4, vdata5, vcoef0, vacc4, vacc5, 0, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef2, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef4, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef6, vacc4, vacc5, -1, vacc4, vacc5);

            vdata12 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata13 = dvload_perm(agen_ld_src_even, src_perm);
            vdata14 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata15 = dvload_perm(agen_ld_src_even, src_perm);
            vdata0  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata1  = dvload_perm(agen_ld_src_even, src_perm);
            vdata2  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata3  = dvload_perm(agen_ld_src_even, src_perm);
            vdata4  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata5  = dvload_perm(agen_ld_src_even, src_perm);
            vdata6  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata7  = dvload_perm(agen_ld_src_even, src_perm);

            vfilt4x2x2_bbh(vdata12, vdata13, vcoef1, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef3, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef5, vacc0, vacc1, -1, vacc0, vacc1);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef7, vacc0, vacc1, -1, vacc0, vacc1);

            vfilt4x2x2_bbh(vdata14, vdata15, vcoef1, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef3, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef5, vacc2, vacc3, -1, vacc2, vacc3);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef7, vacc2, vacc3, -1, vacc2, vacc3);

            vfilt4x2x2_bbh(vdata0, vdata1, vcoef1, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef3, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata4, vdata5, vcoef5, vacc4, vacc5, -1, vacc4, vacc5);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef7, vacc4, vacc5, -1, vacc4, vacc5);

            vdata8  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata9  = dvload_perm(agen_ld_src_even, src_perm);
            vdata10 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata11 = dvload_perm(agen_ld_src_even, src_perm);
            vdata12 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata13 = dvload_perm(agen_ld_src_even, src_perm);
            vdata14 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata15 = dvload_perm(agen_ld_src_even, src_perm);
            vdata0  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata1  = dvload_perm(agen_ld_src_even, src_perm);
            vdata2  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata3  = dvload_perm(agen_ld_src_even, src_perm);

            vstore_i_hb(vacc0, agen_st_dst);
            vstore_i_hb(vacc1, agen_st_dst);
            vstore_i_hb(vacc2, agen_st_dst);

            // Need to make sure niter is even because of manual unroll 2. The prediction off will prevent any extra write.
            niter = (niter % 2 == 0) ? niter : niter + 1;

            // 24 loads    -> 12 VLIW
            // 24 computes -> 12 VLIW -> 2 VLIW per 32 outputs
            //  6 stores   ->  6 VLIW
            for (int i = 0; i < (niter - 2) / 2; i++) chess_loop_range(6, ) //chess_prepare_for_pipelining
            {
                vstore_i_hb(vacc3, agen_st_dst);
                vstore_i_hb(vacc4, agen_st_dst);
                vstore_i_hb(vacc5, agen_st_dst);

                vfilt4x2x2_bbh(vdata8, vdata9, vcoef0, vacc6, vacc7, 0, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef2, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef4, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef6, vacc6, vacc7, -1, vacc6, vacc7);

                vfilt4x2x2_bbh(vdata10, vdata11, vcoef0, vacc8, vacc9, 0, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef2, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef4, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef6, vacc8, vacc9, -1, vacc8, vacc9);

                vfilt4x2x2_bbh(vdata12, vdata13, vcoef0, vacc10, vacc11, 0, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef2, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef4, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef6, vacc10, vacc11, -1, vacc10, vacc11);

                vdata4  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata5  = dvload_perm(agen_ld_src_even, src_perm);
                vdata6  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata7  = dvload_perm(agen_ld_src_even, src_perm);
                vdata8  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata9  = dvload_perm(agen_ld_src_even, src_perm);
                vdata10 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata11 = dvload_perm(agen_ld_src_even, src_perm);
                vdata12 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata13 = dvload_perm(agen_ld_src_even, src_perm);
                vdata14 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata15 = dvload_perm(agen_ld_src_even, src_perm);

                vfilt4x2x2_bbh(vdata4, vdata5, vcoef1, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef3, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef5, vacc6, vacc7, -1, vacc6, vacc7);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef7, vacc6, vacc7, -1, vacc6, vacc7);

                vfilt4x2x2_bbh(vdata6, vdata7, vcoef1, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef3, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef5, vacc8, vacc9, -1, vacc8, vacc9);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef7, vacc8, vacc9, -1, vacc8, vacc9);

                vfilt4x2x2_bbh(vdata8, vdata9, vcoef1, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef3, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata12, vdata13, vcoef5, vacc10, vacc11, -1, vacc10, vacc11);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef7, vacc10, vacc11, -1, vacc10, vacc11);

                vstore_i_hb(vacc6, agen_st_dst);
                vstore_i_hb(vacc7, agen_st_dst);
                vstore_i_hb(vacc8, agen_st_dst);
                vstore_i_hb(vacc9, agen_st_dst);
                vstore_i_hb(vacc10, agen_st_dst);
                vstore_i_hb(vacc11, agen_st_dst);

                vdata0  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata1  = dvload_perm(agen_ld_src_even, src_perm);
                vdata2  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata3  = dvload_perm(agen_ld_src_even, src_perm);
                vdata4  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata5  = dvload_perm(agen_ld_src_even, src_perm);
                vdata6  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata7  = dvload_perm(agen_ld_src_even, src_perm);
                vdata8  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata9  = dvload_perm(agen_ld_src_even, src_perm);
                vdata10 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata11 = dvload_perm(agen_ld_src_even, src_perm);

                vfilt4x2x2_bbh(vdata0, vdata1, vcoef0, vacc0, vacc1, 0, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef2, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef4, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef6, vacc0, vacc1, -1, vacc0, vacc1);

                vfilt4x2x2_bbh(vdata2, vdata3, vcoef0, vacc2, vacc3, 0, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef2, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef4, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef6, vacc2, vacc3, -1, vacc2, vacc3);

                vfilt4x2x2_bbh(vdata4, vdata5, vcoef0, vacc4, vacc5, 0, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef2, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata8, vdata9, vcoef4, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata10, vdata11, vcoef6, vacc4, vacc5, -1, vacc4, vacc5);

                vdata12 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata13 = dvload_perm(agen_ld_src_even, src_perm);
                vdata14 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata15 = dvload_perm(agen_ld_src_even, src_perm);
                vdata0  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata1  = dvload_perm(agen_ld_src_even, src_perm);
                vdata2  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata3  = dvload_perm(agen_ld_src_even, src_perm);
                vdata4  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata5  = dvload_perm(agen_ld_src_even, src_perm);
                vdata6  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata7  = dvload_perm(agen_ld_src_even, src_perm);

                vfilt4x2x2_bbh(vdata12, vdata13, vcoef1, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata14, vdata15, vcoef3, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef5, vacc0, vacc1, -1, vacc0, vacc1);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef7, vacc0, vacc1, -1, vacc0, vacc1);

                vfilt4x2x2_bbh(vdata14, vdata15, vcoef1, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata0, vdata1, vcoef3, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef5, vacc2, vacc3, -1, vacc2, vacc3);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef7, vacc2, vacc3, -1, vacc2, vacc3);

                vfilt4x2x2_bbh(vdata0, vdata1, vcoef1, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata2, vdata3, vcoef3, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata4, vdata5, vcoef5, vacc4, vacc5, -1, vacc4, vacc5);
                vfilt4x2x2_bbh(vdata6, vdata7, vcoef7, vacc4, vacc5, -1, vacc4, vacc5);

                vdata8  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata9  = dvload_perm(agen_ld_src_even, src_perm);
                vdata10 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata11 = dvload_perm(agen_ld_src_even, src_perm);
                vdata12 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata13 = dvload_perm(agen_ld_src_even, src_perm);
                vdata14 = dvload_perm(agen_ld_src_odd, src_perm);
                vdata15 = dvload_perm(agen_ld_src_even, src_perm);
                vdata0  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata1  = dvload_perm(agen_ld_src_even, src_perm);
                vdata2  = dvload_perm(agen_ld_src_odd, src_perm);
                vdata3  = dvload_perm(agen_ld_src_even, src_perm);

                vstore_i_hb(vacc0, agen_st_dst);
                vstore_i_hb(vacc1, agen_st_dst);
                vstore_i_hb(vacc2, agen_st_dst);
            }

            vstore_i_hb(vacc3, agen_st_dst);
            vstore_i_hb(vacc4, agen_st_dst);
            vstore_i_hb(vacc5, agen_st_dst);

            vfilt4x2x2_bbh(vdata8, vdata9, vcoef0, vacc6, vacc7, 0, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef2, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef4, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef6, vacc6, vacc7, -1, vacc6, vacc7);

            vfilt4x2x2_bbh(vdata10, vdata11, vcoef0, vacc8, vacc9, 0, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef2, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef4, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef6, vacc8, vacc9, -1, vacc8, vacc9);

            vfilt4x2x2_bbh(vdata12, vdata13, vcoef0, vacc10, vacc11, 0, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef2, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata0, vdata1, vcoef4, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata2, vdata3, vcoef6, vacc10, vacc11, -1, vacc10, vacc11);

            vdata4  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata5  = dvload_perm(agen_ld_src_even, src_perm);
            vdata6  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata7  = dvload_perm(agen_ld_src_even, src_perm);
            vdata8  = dvload_perm(agen_ld_src_odd, src_perm);
            vdata9  = dvload_perm(agen_ld_src_even, src_perm);
            vdata10 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata11 = dvload_perm(agen_ld_src_even, src_perm);
            vdata12 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata13 = dvload_perm(agen_ld_src_even, src_perm);
            vdata14 = dvload_perm(agen_ld_src_odd, src_perm);
            vdata15 = dvload_perm(agen_ld_src_even, src_perm);

            vfilt4x2x2_bbh(vdata4, vdata5, vcoef1, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata6, vdata7, vcoef3, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef5, vacc6, vacc7, -1, vacc6, vacc7);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef7, vacc6, vacc7, -1, vacc6, vacc7);

            vfilt4x2x2_bbh(vdata6, vdata7, vcoef1, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata8, vdata9, vcoef3, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef5, vacc8, vacc9, -1, vacc8, vacc9);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef7, vacc8, vacc9, -1, vacc8, vacc9);

            vfilt4x2x2_bbh(vdata8, vdata9, vcoef1, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata10, vdata11, vcoef3, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata12, vdata13, vcoef5, vacc10, vacc11, -1, vacc10, vacc11);
            vfilt4x2x2_bbh(vdata14, vdata15, vcoef7, vacc10, vacc11, -1, vacc10, vacc11);

            vstore_i_hb(vacc6, agen_st_dst);
            vstore_i_hb(vacc7, agen_st_dst);
            vstore_i_hb(vacc8, agen_st_dst);
            vstore_i_hb(vacc9, agen_st_dst);
            vstore_i_hb(vacc10, agen_st_dst);
            vstore_i_hb(vacc11, agen_st_dst);
        }
        else if constexpr (KernelSize == 7 && (std::is_same_v<DataType, int16_t> || std::is_same_v<DataType, uint16_t>))
        {
            // 7x7 Convolution for 16-bit input
            constexpr int8_t m_knl_perm[32] = {0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,
                                               16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19};

            vcharx knl_perm = zero_extend(*((vuchar *)(m_knl_perm)));

            dvshortx vdata0a, vdata0b, vdata0c, vdata1a, vdata1b, vdata1c;
            dvshortx vdata2a, vdata2b, vdata2c, vdata3a, vdata3b, vdata3c;
            dvshortx vdata4a, vdata4b, vdata4c, vdata5a, vdata5b, vdata5c;
            dvshortx vdata6a, vdata6b, vdata6c, vdata7a, vdata7b, vdata7c;
            dvshortx vdata8a, vdata8b, vdata8c, vdata9a, vdata9b, vdata9c;

            dvshortx vcoef00, vcoef01, vcoef10, vcoef11, vcoef20, vcoef21;
            dvshortx vcoef30, vcoef31, vcoef40, vcoef41, vcoef50, vcoef51;
            dvshortx vcoef60, vcoef61, vcoef70, vcoef71;

            dvintx vacc0_lo, vacc1_lo, vacc0_hi, vacc1_hi;
            dvintx vacc2_lo, vacc3_lo, vacc2_hi, vacc3_hi;

            vcoef00 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef01 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef10 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef11 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef20 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef21 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef30 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef31 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef40 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef41 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef50 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef51 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef60 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef61 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef70 = dvload_perm_coef(agen_ld_coef, knl_perm);
            vcoef71 = dvload_perm_coef(agen_ld_coef, knl_perm);

            // 30 loads for 10 src rows   -> 15 VLIW
            // 64 computes for 4 dst rows -> 32 VLIW -> 8 VLIW each 32 outputs
            // 4 stores for 4 dst rows    ->  4 VLIW
            for (int i = 0; i < niter; i++) chess_loop_range(3, )
            chess_prepare_for_pipelining
            {
                vdata0a = dvload(agen_ld_src_odd);
                vdata0b = dvload(agen_ld_src_odd);
                vdata0c = dvload(agen_ld_src_odd);
                vdata1a = dvload(agen_ld_src_even);
                vdata1b = dvload(agen_ld_src_even);
                vdata1c = dvload(agen_ld_src_even);
                vdata2a = dvload(agen_ld_src_odd);
                vdata2b = dvload(agen_ld_src_odd);
                vdata2c = dvload(agen_ld_src_odd);
                vdata3a = dvload(agen_ld_src_even);
                vdata3b = dvload(agen_ld_src_even);
                vdata3c = dvload(agen_ld_src_even);
                vdata4a = dvload(agen_ld_src_odd);
                vdata4b = dvload(agen_ld_src_odd);
                vdata4c = dvload(agen_ld_src_odd);
                vdata5a = dvload(agen_ld_src_even);
                vdata5b = dvload(agen_ld_src_even);
                vdata5c = dvload(agen_ld_src_even);
                vdata6a = dvload(agen_ld_src_odd);
                vdata6b = dvload(agen_ld_src_odd);
                vdata6c = dvload(agen_ld_src_odd);
                vdata7a = dvload(agen_ld_src_even);
                vdata7b = dvload(agen_ld_src_even);
                vdata7c = dvload(agen_ld_src_even);
                vdata8a = dvload(agen_ld_src_odd);
                vdata8b = dvload(agen_ld_src_odd);
                vdata8c = dvload(agen_ld_src_odd);
                vdata9a = dvload(agen_ld_src_even);
                vdata9b = dvload(agen_ld_src_even);
                vdata9c = dvload(agen_ld_src_even);

                vfilt4x2_hhw(vdata0a.lo, vdata0b.lo, vcoef00, vacc0_lo, vacc1_lo, 0, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata0b.lo, vdata0c.lo, vcoef01, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata1a.lo, vdata1b.lo, vcoef10, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata1b.lo, vdata1c.lo, vcoef11, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata2a.lo, vdata2b.lo, vcoef20, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata2b.lo, vdata2c.lo, vcoef21, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata3a.lo, vdata3b.lo, vcoef30, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata3b.lo, vdata3c.lo, vcoef31, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata4a.lo, vdata4b.lo, vcoef40, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata4b.lo, vdata4c.lo, vcoef41, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata5a.lo, vdata5b.lo, vcoef50, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata5b.lo, vdata5c.lo, vcoef51, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata6a.lo, vdata6b.lo, vcoef60, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata6b.lo, vdata6c.lo, vcoef61, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata7a.lo, vdata7b.lo, vcoef70, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);
                vfilt4x2_hhw(vdata7b.lo, vdata7c.lo, vcoef71, vacc0_lo, vacc1_lo, -1, vacc0_lo, vacc1_lo);

                vfilt4x2_hhw(vdata0a.hi, vdata0b.hi, vcoef00, vacc0_hi, vacc1_hi, 0, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata0b.hi, vdata0c.hi, vcoef01, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata1a.hi, vdata1b.hi, vcoef10, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata1b.hi, vdata1c.hi, vcoef11, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata2a.hi, vdata2b.hi, vcoef20, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata2b.hi, vdata2c.hi, vcoef21, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata3a.hi, vdata3b.hi, vcoef30, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata3b.hi, vdata3c.hi, vcoef31, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata4a.hi, vdata4b.hi, vcoef40, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata4b.hi, vdata4c.hi, vcoef41, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata5a.hi, vdata5b.hi, vcoef50, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata5b.hi, vdata5c.hi, vcoef51, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata6a.hi, vdata6b.hi, vcoef60, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata6b.hi, vdata6c.hi, vcoef61, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata7a.hi, vdata7b.hi, vcoef70, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);
                vfilt4x2_hhw(vdata7b.hi, vdata7c.hi, vcoef71, vacc0_hi, vacc1_hi, -1, vacc0_hi, vacc1_hi);

                vfilt4x2_hhw(vdata2a.lo, vdata2b.lo, vcoef00, vacc2_lo, vacc3_lo, 0, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata2b.lo, vdata2c.lo, vcoef01, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata3a.lo, vdata3b.lo, vcoef10, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata3b.lo, vdata3c.lo, vcoef11, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata4a.lo, vdata4b.lo, vcoef20, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata4b.lo, vdata4c.lo, vcoef21, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata5a.lo, vdata5b.lo, vcoef30, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata5b.lo, vdata5c.lo, vcoef31, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata6a.lo, vdata6b.lo, vcoef40, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata6b.lo, vdata6c.lo, vcoef41, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata7a.lo, vdata7b.lo, vcoef50, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata7b.lo, vdata7c.lo, vcoef51, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata8a.lo, vdata8b.lo, vcoef60, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata8b.lo, vdata8c.lo, vcoef61, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata9a.lo, vdata9b.lo, vcoef70, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);
                vfilt4x2_hhw(vdata9b.lo, vdata9c.lo, vcoef71, vacc2_lo, vacc3_lo, -1, vacc2_lo, vacc3_lo);

                vfilt4x2_hhw(vdata2a.hi, vdata2b.hi, vcoef00, vacc2_hi, vacc3_hi, 0, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata2b.hi, vdata2c.hi, vcoef01, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata3a.hi, vdata3b.hi, vcoef10, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata3b.hi, vdata3c.hi, vcoef11, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata4a.hi, vdata4b.hi, vcoef20, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata4b.hi, vdata4c.hi, vcoef21, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata5a.hi, vdata5b.hi, vcoef30, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata5b.hi, vdata5c.hi, vcoef31, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata6a.hi, vdata6b.hi, vcoef40, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata6b.hi, vdata6c.hi, vcoef41, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata7a.hi, vdata7b.hi, vcoef50, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata7b.hi, vdata7c.hi, vcoef51, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata8a.hi, vdata8b.hi, vcoef60, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata8b.hi, vdata8c.hi, vcoef61, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata9a.hi, vdata9b.hi, vcoef70, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);
                vfilt4x2_hhw(vdata9b.hi, vdata9c.hi, vcoef71, vacc2_hi, vacc3_hi, -1, vacc2_hi, vacc3_hi);

                vstore_i2(vacc0_lo, vacc0_hi, agen_st_dst);
                vstore_i2(vacc1_lo, vacc1_hi, agen_st_dst);
                vstore_i2(vacc2_lo, vacc2_hi, agen_st_dst);
                vstore_i2(vacc3_lo, vacc3_hi, agen_st_dst);
            }
        }
        else
        {
            static_assert(std::is_same_v<DataType, void>, "Unsupported combination of KernelSize and DataType");
        }
    }
};
/** @} */
} // namespace pvaApl

#endif /* PVA_APL_CONV2D_VPU_HPP */
