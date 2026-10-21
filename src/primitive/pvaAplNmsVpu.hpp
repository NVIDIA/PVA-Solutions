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

/**
 * @file pvaAplNmsVpu.hpp
 *
 * @brief Defines Non-Maximum Suppression (NMS) primitive template class
 *
 * Compatibility: Requires PVA SDK 2.6.0 and later.
 */

#ifndef PVA_APL_NMS_VPU_HPP
#define PVA_APL_NMS_VPU_HPP

#include <cupva_device.h>

#include <type_traits>

namespace pvaApl {

// Define a traits struct to map DataType to specific vector types
template<typename DataType>
struct NmsVectorTypes;

/**
 * @brief Vector type specialization for int32_t.
 */
template<>
struct NmsVectorTypes<int32_t>
{
    /** @brief Vector type for int32_t. */
    using DVTYPE = dvint;
    /** @brief Extended vector type for int32_t. */
    using DVTYPEX = dvintx;
};

/**
 * @defgroup PVA_PRIMITIVE_NMS NMS (Non-Maximum Suppression)
 * @{
 */

/**
 * @brief Non-Maximum Suppression (NMS) primitive template class
 *
 * For every pixel of the valid output region an NmsSize x NmsSize window centered on that pixel is
 * searched. If the pixel is the maximum of the window it is copied to the output, otherwise the
 * output pixel is set to zero. Ties are resolved by position: among equal maxima within the window,
 * the pixel that appears last in raster order (left to right, top to bottom) wins, that is, the one
 * with the largest row index and, if those are equal, the largest column index. A rectangular
 * plateau of equal values therefore leaves exactly one surviving pixel, at its bottom-right corner,
 * rather than keeping all of them.
 *
 * @note The center is never the bottom-right pixel of its own window, so it loses every tie and a
 *       constant tile I(y, x) = k produces an all-zero output. A gradient tile I(y, x) = x does
 *       too, because the window always reaches a larger value to the right.
 *
 * @note The tie-break is implemented by biasing each tap by its rank in the window, so the decision
 *       is exact only where the center differs from every other pixel of its window by more than
 *       B = (NmsSize * NmsSize - 1) / 2 (4 for 3x3, 12 for 5x5). Within that margin it can err in
 *       either direction: a lower neighbor can suppress the center, or a higher one can fail to.
 *
 * @note The output is NmsSize - 1 pixels smaller than the input in both dimensions and offset by
 *       NmsSize / 2, i.e. output(0, 0) corresponds to input(NmsSize / 2, NmsSize / 2). Callers
 *       wanting a full-size result must supply a halo of NmsSize / 2 around the region they want.
 *
 * @tparam NmsSize  Suppression window size. (Square shape: NmsSize x NmsSize) Must be 3 or 5.
 * @tparam DataType Input/output data type. Must be int32_t.
 */
template<int32_t NmsSize, typename DataType>
class NmsVpu
{
private:
    static_assert((NmsSize == 3) || (NmsSize == 5), "NmsVpu supports 3x3 and 5x5 windows only.");
    static_assert(std::is_same_v<DataType, int32_t>, "NmsVpu supports int32_t only.");

    using DVTYPEX = typename NmsVectorTypes<DataType>::DVTYPEX;

    /** @brief Half window size. */
    static constexpr int32_t kRadius = NmsSize >> 1;
    /** @brief Rank bias carried by the center tap: kRadius * NmsSize + kRadius. */
    static constexpr int32_t kCenterBias = (kRadius * NmsSize) + kRadius;
    /** @brief Unroll factor for the horizontal pass. The 3x3 body is half the size, so it unrolls further. */
    static constexpr int32_t kHorUnroll = (NmsSize == 3) ? 4 : 2;

    /**
     * Load one vector of DataType elements, advancing the address generator.
     */
    template<typename AgenType>
    auto dvload(AgenType &ag) -> decltype(auto)
    {
        return static_cast<dvintx>(dvuint_load(ag));
    }

    /**
     * Build an all-zero vector of the working type.
     */
    DVTYPEX dvzero()
    {
        DVTYPEX v;
        v.lo = replicatew(0);
        v.hi = replicatew(0);
        return v;
    }

    /** @brief Array of Agen configurations. 0:hor in, 1:hor out, 2:ver center, 3:ver scratch, 4:out. */
    AgenCFG m_cfgs[5];
    /** @brief Number of iterations of the vertical pass. */
    int32_t m_niter;
    /** @brief Number of iterations of the horizontal pass. */
    int32_t m_niterFirstPass;
    /** @brief Element offset from the input base to the first center tap. */
    int32_t m_verScratchOffset;
    /** @brief Base of the input circular buffer, or nullptr when the input is not circular. */
    DataType *m_cbStart;
    /** @brief Length of the input circular buffer in elements, or 0 when the input is not circular. */
    int32_t m_cbLen;

    /**
     * Wrap a pointer back into the input circular buffer.
     *
     * Only ever called with pointers at most m_verScratchOffset past a valid ring position, and that
     * offset is far smaller than the ring, so a single subtraction is enough.
     */
    DataType *WrapToCb(DataType *p) const
    {
        if ((m_cbLen > 0) && (p >= (m_cbStart + m_cbLen)))
        {
            p -= m_cbLen;
        }
        return p;
    }

    /**
     * Horizontal pass: sliding maximum of NmsSize taps with a +dx rank bias, input to scratch.
     */
    void HorExec()
    {
        int32_t niterCeil = m_niterFirstPass;
        agen_A a_in       = init_agen_A_from_cfg(m_cfgs[0]);
        agen_C a_out      = init_agen_C_from_cfg(m_cfgs[1]);

        for (int32_t i = 0; i < niterCeil; i++) chess_prepare_for_pipelining
        chess_loop_range(12, )
        chess_unroll_loop(kHorUnroll)
        {
            DVTYPEX v0 = dvload(a_in);
            DVTYPEX v1 = dvload(a_in);
            DVTYPEX v2 = dvload(a_in);

            v1 = v1 + 1;
            v2 = v2 + 2;

            DVTYPEX vmax;
            vmax.lo = vmax3(v0.lo, v1.lo, v2.lo);
            vmax.hi = vmax3(v0.hi, v1.hi, v2.hi);
            chess_separator();

            if constexpr (NmsSize > 3)
            {
                DVTYPEX v3 = dvload(a_in);
                DVTYPEX v4 = dvload(a_in);

                v3 = v3 + 3;
                v4 = v4 + 4;

                vmax.lo = vmax3(vmax.lo, v3.lo, v4.lo);
                vmax.hi = vmax3(vmax.hi, v3.hi, v4.hi);
                chess_separator();
            }

            vstore(vmax, a_out);
        }
    }

    /**
     * Vertical pass: sliding maximum of NmsSize scratch rows with a +NmsSize*dy rank bias, compared
     * against the biased center tap to produce the suppressed output.
     */
    void VerExec()
    {
        int32_t niterCeil = m_niter;
        agen_A a_in       = init_agen_A_from_cfg(m_cfgs[2]);
        agen_C a_buf      = init_agen_C_from_cfg(m_cfgs[3]);
        agen_B a_out      = init_agen_B_from_cfg(m_cfgs[4]);

        DVTYPEX vZero = dvzero();

        for (int32_t i = 0; i < niterCeil; i++) chess_prepare_for_pipelining
        chess_loop_range(12, )
        chess_unroll_loop(2)
        {
            DVTYPEX v0 = dvload(a_buf);
            DVTYPEX v1 = dvload(a_buf);
            DVTYPEX v2 = dvload(a_buf);

            DVTYPEX vcenter = dvload(a_in);

            v1 = v1 + NmsSize;
            v2 = v2 + (2 * NmsSize);

            DVTYPEX vmax;
            vmax.lo = vmax3(v0.lo, v1.lo, v2.lo);
            vmax.hi = vmax3(v0.hi, v1.hi, v2.hi);
            chess_separator();

            if constexpr (NmsSize > 3)
            {
                DVTYPEX v3 = dvload(a_buf);
                DVTYPEX v4 = dvload(a_buf);

                v3 = v3 + (3 * NmsSize);
                v4 = v4 + (4 * NmsSize);

                vmax.lo = vmax3(vmax.lo, v3.lo, v4.lo);
                vmax.hi = vmax3(vmax.hi, v3.hi, v4.hi);
                chess_separator();
            }

            DVTYPEX vpred  = ((vcenter + kCenterBias) == vmax);
            DVTYPEX output = dvmux(vpred, vcenter, vZero);
            vstore(output, a_out);
        }
    }

public:
    /** @brief Half window size, i.e. the halo the caller must supply around the valid output region. */
    static constexpr int32_t Radius = kRadius;

    /**
     * Number of DataType elements the scratch buffer must hold for a given input tile size.
     *
     * @param width  Input tile width in pixels.
     * @param height Input tile height in pixels.
     *
     * @return Required scratch element count.
     */
    static constexpr int32_t ScratchElements(int32_t width, int32_t height)
    {
        return (((width * height) + 15) / 16) * 16;
    }

    /**
     * Update the input and output buffer pointers for the Nms algorithm.
     *
     * @param input  Pointer to the input buffer. Points at the top-left of the padded region, not at
     *               the first valid output pixel.
     * @param output Pointer to the output buffer where the result is written to.
     */
    void Update(DataType *input, DataType *restrict output)
    {
        cupvaModifyAgenCfgBase(&m_cfgs[0], static_cast<void *>(input));
        cupvaModifyAgenCfgBase(&m_cfgs[2], static_cast<void *>(WrapToCb(input + m_verScratchOffset)));
        cupvaModifyAgenCfgBase(&m_cfgs[4], static_cast<void *>(output));
    }

    /**
     * Initialize the Nms algorithm parameters, setup agens for the Execute method.
     *
     * Preconditions, none of which are checked at run time:
     *   - width >= NmsSize - 1 and height >= NmsSize - 1, both <= 0xFFFF.
     *   - inputLinePitch and outputLinePitch are in elements, >= 0 and <= 0xFFFF.
     *   - outputLinePitch >= width - (NmsSize - 1).
     *   - output holds height - (NmsSize - 1) rows rounded up to a multiple of 16, because the
     *     inner loops over-run to the unroll factor and rely on agen wraparound.
     *   - scratch holds at least ScratchElements(width, height) DataType elements.
     *   - input is reachable by agen port A, scratch by port C and output by port B, so they must be
     *     placed in the matching VMEM superbanks.
     *   - when the input is a circular buffer, cbStart / cbLen describe the whole ring and input
     *     points somewhere inside it; both source agens then wrap, so a tile whose rows straddle the
     *     end of the ring is read correctly.
     *
     * @param input           Pointer to the input buffer, at the top-left of the padded region.
     * @param width           Input tile width in pixels.
     * @param height          Input tile height in pixels.
     * @param inputLinePitch  Line pitch of the input buffer in elements.
     * @param outputLinePitch Line pitch of the output buffer in elements.
     * @param output          Pointer to the output buffer where the result is written to.
     * @param cbStart         Base of the input circular buffer, or nullptr when the input is not circular.
     * @param cbLen           Length of the input circular buffer in elements, or 0 when not circular.
     * @param scratch         Pointer to the intermediate buffer holding the horizontal pass result.
     */
    void Init(DataType *input, int32_t width, int32_t height, int32_t inputLinePitch, int32_t outputLinePitch,
              DataType *restrict output, DataType *cbStart, int32_t cbLen, DataType *scratch)
    {
        m_cbStart = cbStart;
        m_cbLen   = cbLen;

        int32_t const vecw = chess_elementsof(DVTYPEX);
        int32_t const halo = kRadius * 2;

        int32_t const outWidth  = width - halo;
        int32_t const outHeight = height - halo;

        int32_t const niter1 = (outWidth + vecw - 1) / vecw;
        int32_t const niter2 = outHeight;

        // The horizontal pass also has to produce the vertical halo rows, hence height rather than
        // outHeight here.
        int32_t const niterFirstPass = niter1 * height;
        int32_t const niter          = niter1 * niter2;

        int32_t const scratchLp = outWidth;

        // horizontal pass input
        agen a0 = init(static_cast<void *>(input));
        AgenWrapper inputWrapper;
        inputWrapper.size = sizeof(DataType);
        inputWrapper.n1   = static_cast<uint16_t>(NmsSize);
        inputWrapper.s1   = 1;
        inputWrapper.n2   = static_cast<uint16_t>(niter1);
        inputWrapper.s2   = vecw;
        inputWrapper.n3   = static_cast<uint16_t>(height);
        inputWrapper.s3   = inputLinePitch;
        INIT_AGEN3(a0, inputWrapper);
        a0        = update_agen_cb_size(a0, cbLen * static_cast<int32_t>(sizeof(DataType)));
        a0        = update_agen_cb_start(a0, reinterpret_cast<intptr_t>(cbStart));
        m_cfgs[0] = extract_agen_cfg(a0);

        // horizontal pass output
        agen a1 = init(static_cast<void *>(scratch));
        AgenWrapper horizOutWrapper;
        horizOutWrapper.size = sizeof(DataType);
        horizOutWrapper.n1   = static_cast<uint16_t>(niter1);
        horizOutWrapper.s1   = vecw;
        horizOutWrapper.n2   = static_cast<uint16_t>(height);
        horizOutWrapper.s2   = scratchLp;
        INIT_AGEN2(a1, horizOutWrapper);
        m_cfgs[1] = extract_agen_cfg(a1);

        // vertical pass center value, read back from the original input rather than lane-extracted
        // from the scratch row
        int32_t const verScratchOffset = (kRadius * inputLinePitch) + kRadius;
        m_verScratchOffset             = verScratchOffset;
        agen a2                        = init(static_cast<void *>(WrapToCb(input + verScratchOffset)));
        AgenWrapper inputWrapperCenter;
        inputWrapperCenter.size = sizeof(DataType);
        inputWrapperCenter.n1   = static_cast<uint16_t>(niter1);
        inputWrapperCenter.s1   = vecw;
        inputWrapperCenter.n2   = static_cast<uint16_t>(niter2);
        inputWrapperCenter.s2   = inputLinePitch;
        INIT_AGEN2(a2, inputWrapperCenter);
        // The center tap walks the same circular input buffer as the horizontal pass, so it needs the
        // same wrap.
        a2        = update_agen_cb_size(a2, cbLen * static_cast<int32_t>(sizeof(DataType)));
        a2        = update_agen_cb_start(a2, reinterpret_cast<intptr_t>(cbStart));
        m_cfgs[2] = extract_agen_cfg(a2);

        // vertical pass scratch taps
        agen a3 = init(static_cast<void *>(scratch));
        AgenWrapper verIn;
        verIn.size = sizeof(DataType);
        verIn.n1   = static_cast<uint16_t>(NmsSize);
        verIn.s1   = scratchLp;
        verIn.n2   = static_cast<uint16_t>(niter1);
        verIn.s2   = vecw;
        verIn.n3   = static_cast<uint16_t>(niter2);
        verIn.s3   = scratchLp;
        INIT_AGEN3(a3, verIn);
        m_cfgs[3] = extract_agen_cfg(a3);

        // output
        agen a4 = init(static_cast<void *>(output));
        AgenWrapper outputWrapper;
        outputWrapper.size = sizeof(DataType);
        outputWrapper.n1   = static_cast<uint16_t>(niter1);
        outputWrapper.s1   = vecw;
        outputWrapper.n2   = static_cast<uint16_t>(niter2);
        outputWrapper.s2   = outputLinePitch;
        INIT_AGEN2(a4, outputWrapper);
        m_cfgs[4] = extract_agen_cfg(a4);

        // Both loops over-run to their unroll factor. That is safe because every agen of a given
        // pass shares the same total iteration product, so they wrap back to base together and the
        // surplus iterations simply recompute iteration 0.
        m_niter          = (niter + 1) / 2 * 2;
        m_niterFirstPass = (niterFirstPass + kHorUnroll - 1) / kHorUnroll * kHorUnroll;
    }

    /**
     * Execute the Nms algorithm: horizontal pass followed by vertical pass.
     */
    void Execute()
    {
        HorExec();
        VerExec();
    }
};
/** @} */
} // namespace pvaApl

#endif /* PVA_APL_NMS_VPU_HPP */
