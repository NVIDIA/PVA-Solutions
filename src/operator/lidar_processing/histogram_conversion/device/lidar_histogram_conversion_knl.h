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

#ifndef LIDAR_HISTOGRAM_CONVERSION_KNL_H
#define LIDAR_HISTOGRAM_CONVERSION_KNL_H

#include "lidar_histogram_conversion_param.h"

#include <cupva_device.h>
#include <stdint.h>

/* Provided by the SDK on some versions only. */
#ifndef CUPVA_VPU_ASSERT
#    if CUPVA_BUILD_MODE == CUPVA_NATIVE
#        include <assert.h>
#        define CUPVA_VPU_ASSERT(x) assert(x)
#    else
#        define CUPVA_VPU_ASSERT(x) ((void)0)
#    endif
#endif

/// Decodes the RAW12 bitstream and inflates to a double 16b vector where lo part is even lanes and hi part is odd lanes.
static inline dvshortx UnpackRAW12_vi(vshortx *rawPtr)
{
    // load RAW12 directly into 24b extended so each lane contains a 24b packed pair of 12b values
    // layout from lsb to msb per-lane is A[4:11]B[4:11]B[0:3]A[0:3]
    /// NOTE: this 'WX' style load requires 4-byte alignment
    vshortx raw24 = *rawPtr;

    // extract the individual components of A and B parts from the packed RAW12
    vshortx alsb, amsb, blsb, bmsb;
    {
        // Mask and shift in single instruction to extract A[0:3] aligned to lower 12b
        uint8_t const aLowBit = 0;
        uint8_t const aHiBit  = 3;
        int32_t const aBitPos = (int32_t)(((uint32_t)aHiBit << 8) | (uint32_t)aLowBit);
        vshortx shiftA        = replicateh(-20);
        alsb                  = vextract_bits(raw24, shiftA, aBitPos);

        // Mask and shift in single instruction to extract B[0:3] aligned to middle 12b
        uint8_t const bLowBit = 12;
        uint8_t const bHiBit  = 15;
        int32_t const bBitPos = (int32_t)(((uint32_t)bHiBit << 8) | (uint32_t)bLowBit);
        vshortx shiftB        = replicateh(-4);
        blsb                  = vextract_bits(raw24, shiftB, bBitPos);
    }
    {
        // Mask and shift in single instruction to extract A[4:11] aligned to lower 12b
        uint8_t const aLowBit = 4;
        uint8_t const aHiBit  = 11;
        int32_t const aBitPos = (int32_t)(((uint32_t)aHiBit << 8) | (uint32_t)aLowBit);
        vshortx shiftA        = replicateh(4);
        amsb                  = vextract_bits(raw24, shiftA, aBitPos);

        // Mask and shift in single instruction to extract B[4:11] aligned to middle 12b
        uint8_t const bLowBit = 16;
        uint8_t const bHiBit  = 23;
        int32_t const bBitPos = (int32_t)(((uint32_t)bHiBit << 8) | (uint32_t)bLowBit);
        vshortx shiftB        = replicateh(8);
        bmsb                  = vextract_bits(raw24, shiftB, bBitPos);
    }

    // merge the components to put the bits in order A[0:11]B[0:11]
    vshortx unpacked = vor3(alsb | blsb, amsb, bmsb);

    // we can now reinterpret as vcharx because all elements are in order
    /// NB: only catch is these 12b values are treated as signed.
    vcharx raw12 = as_vcharx(unpacked);

    // promotion to dvshortx sign-extends from 12b to 24b and deinterleaves even and odd lanes
    // we mask to undo the sign-extension and preserve the correct 12b unsigned values.
    return vpromote_di(raw12) & 0x0FFF;
}

/// simple double-vector post-increment bulk copy
static inline void copy_v(dvshort *dst, dvushort *src, int32_t n)
{
    for (int32_t i = 0; i < n; i++)
    {
        *dst++ = extract(zero_extend(*src++));
    }
}

/// fill the 3-bin halo of each histogram using reflection to prepare for smoothing
static inline void fill_halo_v(uint16_t *histogram, int32_t numHistPixPerTile, int32_t numHistPerPixel,
                               int32_t numBinsPerHist, int32_t histPitchOut)
{
    int16_t leadingHaloPerm[16]  = {6, 5, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    vshortx leadingHaloPermVec   = sign_extend(*((vshort *)(leadingHaloPerm)));
    int16_t trailingHaloPerm[16] = {0, 1, 2, 3, 2, 1, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    vshortx trailingHaloPermVec  = sign_extend(*((vshort *)(trailingHaloPerm)));
    int32_t numTotalHist         = numHistPixPerTile * numHistPerPixel;
    for (int32_t h = 0; h < numTotalHist; h++)
    {
        uint16_t *histogramPtrStartHalo  = histogram + h * histPitchOut;
        vshortx histogramStart           = sign_extend(*((vshort *)(histogramPtrStartHalo)));
        histogramStart                   = vpermute(histogramStart, leadingHaloPermVec);
        *(vshort *)histogramPtrStartHalo = extract(histogramStart);

        uint16_t *histogramPtrEndHalo  = histogramPtrStartHalo + numBinsPerHist - 1;
        vshortx histogramEnd           = sign_extend(*((vshort *)(histogramPtrEndHalo)));
        histogramEnd                   = vpermute(histogramEnd, trailingHaloPermVec);
        *(vshort *)histogramPtrEndHalo = extract(histogramEnd);
    }
}

/// unpack input tile into separate buffers for NONE packing type (uint16_t per element)
static inline void UnpackHistogramDataNone_v(uint8_t *histogramTile, int32_t pixelPitchIn, uint16_t *pixelHeader,
                                             uint16_t *histogramHeader, uint16_t *histogram, int32_t histPitchOut,
                                             int32_t numHistPixPerTile, int32_t numHistPerPixel,
                                             int32_t pixelHeaderSize, int32_t histHeaderSize, int32_t numBinsPerHist)
{
    int32_t vecw = pva_elementsof(dvshortx);

    int32_t numPixHeadVecs  = (pixelHeaderSize + vecw - 1) / vecw;
    int32_t numHistHeadVecs = (histHeaderSize + vecw - 1) / vecw;
    int32_t numHistBinsVecs = (numBinsPerHist + vecw - 1) / vecw;

    // copy pixel header data (if any)
    if (pixelHeaderSize > 0)
    {
        for (int32_t p = 0; p < numHistPixPerTile; p++)
        {
            dvushort *pixelPtrIn   = (dvushort *)(histogramTile + p * pixelPitchIn);
            dvshort *pixHeadPtrOut = (dvshort *)(pixelHeader + p * pixelHeaderSize);

            copy_v(pixHeadPtrOut, pixelPtrIn, numPixHeadVecs);
        }
    }

    // copy histogram header data (if any)
    if (histHeaderSize > 0)
    {
        for (int32_t p = 0; p < numHistPixPerTile; p++)
        {
            for (int32_t i = 0; i < numHistPerPixel; i++)
            {
                dvushort *histHeadPtrIn =
                    (dvushort *)(histogramTile + p * pixelPitchIn +
                                 (pixelHeaderSize + i * (histHeaderSize + numBinsPerHist)) * sizeof(uint16_t));
                dvshort *histHeadPtrOut =
                    (dvshort *)(histogramHeader + p * numHistPerPixel * histHeaderSize + i * histHeaderSize);
                copy_v(histHeadPtrOut, histHeadPtrIn, numHistHeadVecs);
            }
        }
    }

    // copy histogram bin data
    for (int32_t p = 0; p < numHistPixPerTile; p++)
    {
        for (int32_t i = 0; i < numHistPerPixel; i++)
        {
            dvushort *histogramPtrIn =
                (dvushort *)(histogramTile + p * pixelPitchIn +
                             (pixelHeaderSize + i * (histHeaderSize + numBinsPerHist) + histHeaderSize) *
                                 sizeof(uint16_t));
            dvshort *histogramPtrOut =
                (dvshort *)(histogram + p * numHistPerPixel * histPitchOut + i * histPitchOut + 3);
            copy_v(histogramPtrOut, histogramPtrIn, numHistBinsVecs);
        }
    }

    // fill the histogram halo with reflection
    fill_halo_v(histogram, numHistPixPerTile, numHistPerPixel, numBinsPerHist, histPitchOut);
}

/// unpack input tile into separate buffers for RAW12 packing type (24b per pair of elements)
static inline void UnpackHistogramDataRAW12_v(uint8_t *histogramTile, int32_t pixelPitchIn, uint16_t *pixelHeader,
                                              uint16_t *histogramHeader, uint16_t *histogram, int32_t histPitchOut,
                                              int32_t numHistPixPerTile, int32_t numHistPerPixel,
                                              int32_t pixelHeaderSize, int32_t histHeaderSize, int32_t numBinsPerHist)
{
    /// NOTE: there must be an even number of elements when using RAW12 packing.
    CUPVA_VPU_ASSERT(pixelHeaderSize % 2 == 0);
    CUPVA_VPU_ASSERT(histHeaderSize % 2 == 0);
    CUPVA_VPU_ASSERT(numBinsPerHist % 2 == 0);

    // RAW12 is 24b packet per 2 element, so each 24b input lane produces 2 output elements.
    int32_t vecw = pva_elementsof(dvushort);
    CUPVA_VPU_ASSERT(pva_elementsof(vshortx) == RAW12_UNPACK_PAIRS_PER_VECTOR);
    CUPVA_VPU_ASSERT(vecw == RAW12_UNPACK_ELEMENTS_PER_VECTOR);

    // WX loads require 4-byte alignment.
    CUPVA_VPU_ASSERT((uintptr_t)(histogramTile) % 4 == 0);
    CUPVA_VPU_ASSERT(pixelPitchIn % 3 == 0); // expect only pairs of elements and tightly packed data...
    // Includes the pairs skipped for alignment, rounded up to a multiple of the unroll factor so the unrolled loop needs
    // no remainder iterations. The extra vectors land in the scratch area and are never read back.
    int32_t numPixUnpackVecs = Raw12UnpackVectorsPerPixel(pixelPitchIn);

    int32_t numPixHeadVecs  = (pixelHeaderSize + vecw - 1) / vecw;
    int32_t numHistHeadVecs = (histHeaderSize + vecw - 1) / vecw;
    int32_t numHistBinsVecs = (numBinsPerHist + vecw - 1) / vecw;

    // since unpack results in de-interleaved dvshortx vector, we need an interleaved store operation.
    // interleaved store is only available with agen.
    AgenWrapper wrapper;

    // Use the extra space at the end of the histogram buffer as scratch area for unpacking a pixel.
    int32_t scratchOffsetBase = (numHistPixPerTile * numHistPerPixel) * (histPitchOut);
    CUPVA_VPU_ASSERT(scratchOffsetBase + numPixUnpackVecs * vecw <= UNPACKED_HISTOGRAM_BUFFER_SIZE);

    agen pixUnpackAgen = init(histogram + scratchOffsetBase);
    wrapper.size       = sizeof(uint16_t);
    wrapper.n2         = numHistPixPerTile;
    wrapper.s2         = 0;
    wrapper.n1         = numPixUnpackVecs;
    wrapper.s1         = vecw;
    INIT_AGEN2(pixUnpackAgen, wrapper);

    // unpack full pixel to scratch area, then copy the components to the different buffers.
    for (int32_t p = 0; p < numHistPixPerTile; p++)
    {
        // account for how many element pairs we need to backtrack to get 4*3 byte alignment.
        int32_t rawOffset      = p * pixelPitchIn;
        int32_t alignmentBytes = rawOffset % 12;
        int32_t alignedOffset  = rawOffset - alignmentBytes;
        CUPVA_VPU_ASSERT(alignmentBytes % 3 == 0);
        int32_t numPairsToSkip = alignmentBytes / 3;

        // unpack full pixel to scratch area
        vshortx *pixelRAW12Ptr = (vshortx *)(histogramTile + alignedOffset);
        for (int32_t i = 0; i < numPixUnpackVecs; i++) chess_prepare_for_pipelining
        chess_unroll_loop(4)
        // Derived from the same quantity the assert above bounds numPixUnpackVecs by, so the two
        // cannot drift apart. A literal is easy to under-declare: the largest supported shape needs
        // well over 256.
        chess_loop_range(4, UNPACKED_HISTOGRAM_BUFFER_SIZE / 16)
        {
            dvshortx pixVec_di = UnpackRAW12_vi(pixelRAW12Ptr++);
            vstore_i(pixVec_di, pixUnpackAgen);
        }

        // skip the alignment slop
        int32_t scratchOffset = scratchOffsetBase + numPairsToSkip * 2;

        // copy out the pixel header
        if (pixelHeaderSize > 0)
        {
            dvushort *pixelPtrIn   = (dvushort *)(histogram + scratchOffset);
            dvshort *pixHeadPtrOut = (dvshort *)(pixelHeader + p * pixelHeaderSize);

            copy_v(pixHeadPtrOut, pixelPtrIn, numPixHeadVecs);
        }

        // copy out the histogram headers
        if (histHeaderSize > 0)
        {
            for (int32_t i = 0; i < numHistPerPixel; i++)
            {
                dvushort *histHeadPtrIn =
                    (dvushort *)(histogram + scratchOffset + pixelHeaderSize + i * (histHeaderSize + numBinsPerHist));
                dvshort *histHeadPtrOut =
                    (dvshort *)(histogramHeader + p * numHistPerPixel * histHeaderSize + i * histHeaderSize);
                copy_v(histHeadPtrOut, histHeadPtrIn, numHistHeadVecs);
            }
        }

        // copy out the histogram bins
        for (int32_t i = 0; i < numHistPerPixel; i++)
        {
            dvushort *histBinsPtrIn = (dvushort *)(histogram + scratchOffset + pixelHeaderSize +
                                                   i * (histHeaderSize + numBinsPerHist) + histHeaderSize);
            dvshort *histBinsPtrOut =
                (dvshort *)(histogram + p * numHistPerPixel * histPitchOut + i * histPitchOut + 3);
            copy_v(histBinsPtrOut, histBinsPtrIn, numHistBinsVecs);
        }
    }

    // fill the histogram halo with reflection
    fill_halo_v(histogram, numHistPixPerTile, numHistPerPixel, numBinsPerHist, histPitchOut);
}

/**
 * @brief Apply 7-tap Gaussian filter to smooth histogram (32-lane dvshortx + dvmaddh version)
 *
 * Integer kernel [5, 55, 248, 408, 248, 55, 5], sum=1024.
 * Accumulates 7 taps with VPU_ROUND_0 (no per-term truncation) into the 24-bit accumulator.
 * For 12-bit bins the maximum accumulated value is 4095 * 1024 = 4,193,280 < 2^23 (signed 24-bit
 * max 8,388,607). NONE-packed bins can be larger, and a run of bins above 8191 overflows it.
 * Extracts result via >> 10 (divide by 1024) on the accumulated dvshortx while storing using agen truncation option.
 *
 * @param histogram Input histogram pointer; read only, not modified
 * @param smoothBuf Output buffer for the smoothed bins. Must be distinct from histogram: writing
 *                  elsewhere is what lets the loop pipeline, since the 7-tap window would
 *                  otherwise overlap the previous iteration's stores.
 * @param numBins Number of histogram bins
 * @param histMin Output minimum value of the smoothed histogram
 * @param histMax Output maximum value of the smoothed histogram
 */
static inline void SmoothHistogram_vi16_dv(uint16_t *histogram, uint16_t *smoothBuf, int32_t numBins, uint16_t &histMin,
                                           uint16_t &histMax)
{
    int32_t const kernelRadius = 3;

    int32_t const vecw     = 32; // dvshortx = 32 int16 lanes
    int32_t const vecwLog2 = 5;
    CUPVA_VPU_ASSERT(vecw > kernelRadius && (1 << vecwLog2) == vecw);
    int32_t const niter       = (numBins + vecw - 1) >> vecwLog2;
    int32_t const numTailBins = numBins - ((niter - 1) * vecw);
    // Tail lane mask: bits [numTailBins-1:0] set. Guard against UB when numTailBins==vecw.
    int32_t const tailLaneMask = (numTailBins >= vecw) ? -1 : (int32_t) ~(0xFFFFFFFFU << numTailBins);

    dvshortx v0, v1, v2, v3, v4, v5, v6, acc;

    // Input and output live in different superbanks, so the typed agens let the compiler see that
    // loads and stores cannot alias.
    agen_A histAgenIn = init_A(histogram);
    AgenWrapper wrapper;
    wrapper.size = sizeof(uint16_t);
    wrapper.n1   = 7; // sliding 7-tap window, one element apart
    wrapper.s1   = 1;
    wrapper.n2   = niter;
    wrapper.s2   = vecw;
    INIT_AGEN2(histAgenIn, wrapper);

    agen_C histAgenOut = init_C(smoothBuf);
    wrapper.size       = sizeof(uint16_t);
    wrapper.n1         = niter;
    wrapper.s1         = vecw;
    INIT_AGEN1(histAgenOut, wrapper);
    histAgenOut.minmax_opt = 3;         // unsigned min/max, tracked for the noise floor
    histAgenOut.round      = 0x80 | 10; // truncate 10 bits, i.e. divide by 1024

    int32_t const nFullSmooth = niter - 1;
    for (int32_t i = 0; i < nFullSmooth; i++) chess_prepare_for_pipelining
    {
        v0 = dvushort_load(histAgenIn);
        v1 = dvushort_load(histAgenIn);
        v2 = dvushort_load(histAgenIn);
        v3 = dvushort_load(histAgenIn);
        v4 = dvushort_load(histAgenIn);
        v5 = dvushort_load(histAgenIn);
        v6 = dvushort_load(histAgenIn);

        // VMADD_CA H-type accumulates each tap into the 24-bit per-lane ARF accumulator.
        // pred=0 on the first tap clears it, so acc starts at v0*5.
        acc = dvmaddh(v0, 5, acc, VPU_ROUND_0, 0);
        acc = dvmaddh(v1, 55, acc, VPU_ROUND_0, -1);
        acc = dvmaddh(v2, 248, acc, VPU_ROUND_0, -1);
        acc = dvmaddh(v3, 408, acc, VPU_ROUND_0, -1);
        acc = dvmaddh(v4, 248, acc, VPU_ROUND_0, -1);
        acc = dvmaddh(v5, 55, acc, VPU_ROUND_0, -1);
        acc = dvmaddh(v6, 5, acc, VPU_ROUND_0, -1);

        vstore(acc, histAgenOut);
    }

    // final (possibly partial) iteration
    v0 = dvushort_load(histAgenIn);
    v1 = dvushort_load(histAgenIn);
    v2 = dvushort_load(histAgenIn);
    v3 = dvushort_load(histAgenIn);
    v4 = dvushort_load(histAgenIn);
    v5 = dvushort_load(histAgenIn);
    v6 = dvushort_load(histAgenIn);

    acc = dvmaddh(v0, 5, acc, VPU_ROUND_0, 0);
    acc = dvmaddh(v1, 55, acc, VPU_ROUND_0, -1);
    acc = dvmaddh(v2, 248, acc, VPU_ROUND_0, -1);
    acc = dvmaddh(v3, 408, acc, VPU_ROUND_0, -1);
    acc = dvmaddh(v4, 248, acc, VPU_ROUND_0, -1);
    acc = dvmaddh(v5, 55, acc, VPU_ROUND_0, -1);
    acc = dvmaddh(v6, 5, acc, VPU_ROUND_0, -1);

    vstore(acc, histAgenOut, tailLaneMask);
    histMin = (uint16_t)histAgenOut.min_val;
    histMax = (uint16_t)histAgenOut.max_val;
}

/**
 * @brief Detect the strongest peaks in one smoothed histogram
 *
 * A peak is a maximal run of equal bins that is higher than the bin on each side of it and higher
 * than noiseThreshold. Its index is the midpoint of the run, rounded down. The numPeaksToDetect
 * highest peaks are kept, ties going to the lower index.
 *
 * @param histogram        Smoothed bins. The detector writes a zero bin at [-1] and at [numBins],
 *                         and its loads reach index roundup(numBins, 32).
 * @param peaksOutPtr      Output, two vshort vectors: [peakIdx | peakVal] then [binLeft | binRight],
 *                         MAX_PEAKS_PER_HISTOGRAM lanes per field. Unused peakIdx lanes are -1.
 * @param numBins          Number of histogram bins
 * @param numPeaksToDetect Number of peaks to keep, at most MAX_PEAKS_PER_HISTOGRAM
 * @param noiseThreshold   Noise floor; a peak must be strictly greater than it
 * @param candBuf          Scratch for the plateau-end candidates, numBins + 16 elements
 */
static inline void DetectPeaksNMS_v(uint16_t *histogram, int16_t *peaksOutPtr, int32_t const numBins,
                                    int32_t const numPeaksToDetect, int16_t const noiseThreshold, int16_t *candBuf)
{
    int32_t const vecw  = pva_elementsof(vushort); // 16 halfword lanes
    int32_t const dvecw = 2 * vecw;                // 32 in a double vector
    CUPVA_VPU_ASSERT(MAX_PEAKS_PER_HISTOGRAM <= vecw);
    CUPVA_VPU_ASSERT(numPeaksToDetect <= vecw);

    // The detector treats a missing neighbor as 0, so pad one zero bin on each side.
    histogram[-1]      = 0;
    histogram[numBins] = 0;

    // A peak is a maximal run of equal bins that beats both its neighbors and the threshold. Pass 1
    // marks only the END of such a run, which makes the test purely local:
    //     candidate(e)  <=>  S[e] > max(S[e+1], S[e-1] - 1, thr)
    // Subtracting one from the left neighbor lets "equal to the left" pass, so the last bin of a run
    // passes even though its left neighbor is equal, while every earlier bin of the run fails against
    // its equal right neighbor. No plateau state is carried from one vector to the next. Pass 2
    // walks each candidate back to the start of its run and checks the bin before it.
    int32_t const nvec  = (numBins + dvecw - 1) / dvecw;
    int32_t const nTail = numBins - (nvec - 1) * dvecw; // 1 .. dvecw

    int16_t *candPtr = candBuf;
    {
        agen aL = init(histogram - 1);
        agen aC = init(histogram);
        agen aR = init(histogram + 1);
        AgenWrapper wrapper;
        wrapper.size = sizeof(uint16_t);
        wrapper.n1   = nvec;
        wrapper.s1   = dvecw;
        INIT_AGEN1(aL, wrapper);
        INIT_AGEN1(aC, wrapper);
        INIT_AGEN1(aR, wrapper);

        dvshortx vThr;
        vThr.lo = replicateh(noiseThreshold);
        vThr.hi = vThr.lo;

        // VCOLLATEIDX_BITS returns lane indices within each 16-lane half, so what is added back is
        // the running base of that half, not a lane ramp.
        dvshortx vPos;
        vPos.lo = replicateh(0);
        vPos.hi = replicateh(vecw);

        // Past the zero bin at [numBins] the buffer holds bins no smoothing pass has written, so the
        // final double vector is masked rather than relying on that slack being zero.
        int32_t const tailLo  = (nTail >= vecw) ? vecw : nTail;
        int32_t const tailHi  = (nTail >= vecw) ? (nTail - vecw) : 0;
        vshortx const mTailLo = vbitunpackh((tailLo >= vecw) ? ~(0xFFFF << (vecw - 1) << 1) : ~(0xFFFF << tailLo));
        vshortx const mTailHi = vbitunpackh((tailHi >= vecw) ? ~(0xFFFF << (vecw - 1) << 1) : ~(0xFFFF << tailHi));

        // This is the hottest loop in the operator. Masking the partial final vector inside it
        // costs a predicate on every iteration to matter on the last one, so the final vector is
        // peeled out below and masked there.
        int32_t const nFull = nvec - 1;

        for (int32_t i = 0; i < nFull; i++) chess_prepare_for_pipelining
        {
            dvshortx c = dvushort_load(aC);
            dvshortx l = dvushort_load(aL);
            dvshortx r = dvushort_load(aR);

            dvshortx bound = dvmax(dvmax(r, dvadd(l, -1)), vThr);
            dvshortx mask  = dvCmpGT(c, bound);

            vshortx nzLo, nzHi;
            int32_t bitsLo, bitsHi;
            vcollate_idx_bits(mask.lo, nzLo, bitsLo);
            vcollate_idx_bits(mask.hi, nzHi, bitsHi);

            dvshortx nz;
            nz.lo = nzLo;
            nz.hi = nzHi;
            nz    = dvadd(nz, vPos);

            *(vshort *)candPtr = extract(nz.lo);
            candPtr += bitcount(bitsLo);
            *(vshort *)candPtr = extract(nz.hi);
            candPtr += bitcount(bitsHi);

            vPos = dvadd(vPos, dvecw);
        }

        // Final, partial vector.
        {
            dvshortx c = dvushort_load(aC);
            dvshortx l = dvushort_load(aL);
            dvshortx r = dvushort_load(aR);

            dvshortx bound = dvmax(dvmax(r, dvadd(l, -1)), vThr);
            dvshortx mask  = dvCmpGT(c, bound);
            mask.lo        = mask.lo & mTailLo;
            mask.hi        = mask.hi & mTailHi;

            vshortx nzLo, nzHi;
            int32_t bitsLo, bitsHi;
            vcollate_idx_bits(mask.lo, nzLo, bitsLo);
            vcollate_idx_bits(mask.hi, nzHi, bitsHi);

            dvshortx nz;
            nz.lo = nzLo;
            nz.hi = nzHi;
            nz    = dvadd(nz, vPos);

            *(vshort *)candPtr = extract(nz.lo);
            candPtr += bitcount(bitsLo);
            *(vshort *)candPtr = extract(nz.hi);
            candPtr += bitcount(bitsHi);
        }
    }

    // ---- pass 2: sparse plateau resolution and top-N insertion ----
    int32_t const nCand = (int32_t)(candPtr - candBuf);

    vshortx const ramp     = vcollate_idx(replicateh(0)); // {0, 1, ... 15}
    vshortx const shiftPat = vmax(vadd(ramp, -1), 0);     // {0, 0, 1, ... 14}
    vshortx const keepLane = vCmpGT(ramp, 0);             // lane 0 never takes a shifted-in flag

    vshortx topVal = replicateh(0);
    vshortx topIdx = replicateh(-1);
    // The emit stage needs the bin triple around each peak: the peak value and the bins on either
    // side of the peak index, which are plateau bins or the bins just outside the plateau.
    // Carrying them through the same insertion means emit never has to read the smoothed histogram
    // back.
    vshortx topL  = replicateh(0);
    vshortx topR  = replicateh(0);
    int32_t worst = 0;

    for (int32_t i = 0; i < nCand; i++)
    {
        int32_t const e = (int32_t)candBuf[i];
        if (e >= numBins)
        {
            continue;
        }
        int32_t const v = (int32_t)histogram[e];
        if (v <= worst)
        {
            continue; // cannot displace the current N-th best
        }

        // Walk back over the plateau. histogram[-1] is 0 and v > threshold >= 0, so the scan
        // always stops before leaving the buffer.
        int32_t sIdx = e;
        while ((int32_t)histogram[sIdx - 1] == v)
        {
            sIdx--;
        }
        if ((int32_t)histogram[sIdx - 1] > v)
        {
            continue; // the plateau is not a maximum on its left side
        }
        int32_t const mid = (sIdx + e) >> 1;

        // Neighbors of mid. When the plateau is wider than two bins mid sits strictly inside
        // it, so both neighbors are part of the plateau and equal v; only at the ends does a
        // real neighbor show through. histogram[-1] and histogram[numBins] are the zero halo.
        int32_t const lv = (mid == sIdx) ? (int32_t)histogram[sIdx - 1] : v;
        int32_t const rv = (mid == e) ? (int32_t)histogram[e + 1] : v;

        // Insert into the descending list. Candidates arrive with increasing midpoint, so
        // placing a new entry after every equal value reproduces the reference's
        // (value desc, index asc) total order without a sort.
        vshortx const vv    = replicateh(v);
        vshortx const lt    = vCmpLT(topVal, vv);
        vshortx const ltPrv = vand(vpermute(lt, shiftPat), keepLane);
        vshortx const sVal  = vpermute(topVal, shiftPat);
        vshortx const sIdxV = vpermute(topIdx, shiftPat);

        vshortx const sL = vpermute(topL, shiftPat);
        vshortx const sR = vpermute(topR, shiftPat);

        topVal = vmux(lt, vmux(ltPrv, sVal, vv), topVal);
        topIdx = vmux(lt, vmux(ltPrv, sIdxV, replicateh(mid)), topIdx);
        topL   = vmux(lt, vmux(ltPrv, sL, replicateh(lv)), topL);
        topR   = vmux(lt, vmux(ltPrv, sR, replicateh(rv)), topR);
        worst  = vselectlane(topVal, numPeaksToDetect - 1);
    }

    // Four fields to store but at most MAX_PEAKS_PER_HISTOGRAM (8) of each, so two share one
    // 16-lane vector and the peaks buffer keeps its pitch:
    //   vector 0 = [peakIdx | peakVal ]   vector 1 = [binLeft | binRight]
    CUPVA_VPU_ASSERT(numPeaksToDetect <= MAX_PEAKS_PER_HISTOGRAM);
    // Two fields share each 16-lane vector, so the pair has to fit or the halves overlap.
    CUPVA_VPU_ASSERT(2 * MAX_PEAKS_PER_HISTOGRAM <= vecw);
    vshortx const lo8    = vCmpLT(ramp, MAX_PEAKS_PER_HISTOGRAM);
    vshortx const shift8 = vadd(ramp, -MAX_PEAKS_PER_HISTOGRAM);

    *(vshort *)peaksOutPtr          = extract(vmux(lo8, topIdx, vpermute(topVal, shift8)));
    *(vshort *)(peaksOutPtr + vecw) = extract(vmux(lo8, topL, vpermute(topR, shift8)));
}

/**
 * @brief Fused emit: gather + range/XYZ + reflectance for one pixel in a single pass.
 *
 * Gathers each peak's bin triple as integers into one scratch buffer, then computes the parabola,
 * range, XYZ and reflectance in the vector unit. The scratch is written once and read once.
 *
 * gatherBuf layout, SCRATCH_PEAKS stride per field: [peakIdx][binLeft][binCenter][binRight].
 *
 * @param pixelPeakBase  Peaks for this pixel, kPeakPitch apart per histogram. Carries the bin
 *                       triple the detector recorded, so the smoothed histogram is not re-read.
 * @param numHistPerPixel Histograms per pixel (runtime)
 * @param numPeaksPerHist Peaks requested per histogram (runtime)
 * @param numBinsPerHist  Bins per histogram; bounds the left/right neighbor clamp
 * @param kPeakPitch      int16 stride between one histogram's peaks and the next
 * @param scratchStride   Per-field stride inside gatherBuf
 * @param offsetNs        Time offset of bin 0
 * @param binSizeNs       Time width of one bin
 * @param rangeScale      Scale applied to the computed range
 * @param rangeBias       Bias added to the computed range
 * @param xyzCal          Three calibration multipliers for X, Y and Z
 * @param invMaxIntensity Reciprocal of maxIntensity, or zero when maxIntensity is not
 *                        positive. Formed on the host so the device only multiplies.
 * @param emitMask        Bit 0 selects range/XYZ, bit 1 selects reflectance
 * @param rangeOut        Range output, totalPeaks floats
 * @param xyzOutX         X output; Y and Z follow at totalPeaks strides, so stores are predicated
 * @param xyzOutY         Y output
 * @param xyzOutZ         Z output
 * @param reflectanceOut  Reflectance output, totalPeaks floats
 * @param gatherBuf       Scratch for the integer gather, 4 * scratchStride int32
 */
static inline void EmitPixelBatch_v(int16_t const *pixelPeakBase, int32_t numHistPerPixel, int32_t numPeaksPerHist,
                                    int32_t numBinsPerHist, int32_t kPeakPitch, int32_t scratchStride, float offsetNs,
                                    float binSizeNs, float rangeScale, float rangeBias, float const *xyzCal,
                                    float invMaxIntensity, int32_t emitMask, float *rangeOut, float *xyzOutX,
                                    float *xyzOutY, float *xyzOutZ, float *reflectanceOut, int32_t *gatherBuf)
{
    int32_t const totalPeaks = numHistPerPixel * numPeaksPerHist;
    int32_t *const gIdx      = gatherBuf;
    int32_t *const gL        = gatherBuf + scratchStride;
    int32_t *const gC        = gatherBuf + 2 * scratchStride;
    int32_t *const gR        = gatherBuf + 3 * scratchStride;

    // ---- scalar gather: integers only, no float conversion, no parabola ----
    for (int32_t h = 0; h < numHistPerPixel; h++)
    {
        int16_t const *pkPtr = pixelPeakBase + h * kPeakPitch;
        int32_t const base   = h * numPeaksPerHist;
        int32_t const vecwPk = pva_elementsof(vshort);

        for (int32_t j = 0; j < numPeaksPerHist; j++)
        {
            int32_t const idx = pkPtr[j];
            gIdx[base + j]    = idx;
            if (idx < 0)
            {
                gL[base + j] = 0;
                gC[base + j] = 0;
                gR[base + j] = 0;
            }
            else
            {
                // Recorded by the detector, so no read back into the smoothed histogram.
                gC[base + j] = (int32_t)(uint16_t)pkPtr[MAX_PEAKS_PER_HISTOGRAM + j];
                gL[base + j] = (idx > 0) ? (int32_t)(uint16_t)pkPtr[vecwPk + j] : 0;
                gR[base + j] =
                    (idx < numBinsPerHist - 1) ? (int32_t)(uint16_t)pkPtr[vecwPk + MAX_PEAKS_PER_HISTOGRAM + j] : 0;
            }
        }
    }

    // ---- vector epilogue ----
    float const rangeMul   = rangeScale * binSizeNs * SPEED_OF_LIGHT_M_PER_NS;
    float const rangeConst = rangeBias + rangeScale * offsetNs * SPEED_OF_LIGHT_M_PER_NS;

    int32_t const nBatches = (totalPeaks + 15) >> 4;
    int32_t const kMaxB    = (MAX_HISTOGRAMS_PER_PIXEL * MAX_PEAKS_PER_HISTOGRAM) / 16;

    AgenWrapper w;
    w.size = sizeof(int32_t);
    w.n1   = nBatches;
    w.s1   = 16;

    agen_A aI = init_A(gIdx);
    INIT_AGEN1(aI, w);
    agen_A aL = init_A(gL);
    INIT_AGEN1(aL, w);
    agen_A aC = init_A(gC);
    INIT_AGEN1(aC, w);
    agen_A aR = init_A(gR);
    INIT_AGEN1(aR, w);

    w.size        = sizeof(float);
    agen_C aRange = init_C(rangeOut);
    INIT_AGEN1(aRange, w);
    agen_C aX = init_C(xyzOutX);
    INIT_AGEN1(aX, w);
    agen_C aY = init_C(xyzOutY);
    INIT_AGEN1(aY, w);
    agen_C aZ = init_C(xyzOutZ);
    INIT_AGEN1(aZ, w);
    agen_C aRefl = init_C(reflectanceOut);
    INIT_AGEN1(aRefl, w);

    dvfloatx vOne;
    vOne.lo = replicatef(1.0f);
    vOne.hi = vOne.lo;
    dvfloatx vZero;
    vZero.lo = replicatef(0.0f);
    vZero.hi = vZero.lo;

    for (int32_t b = 0; b < nBatches; b++) chess_loop_range(1, kMaxB)
    chess_prepare_for_pipelining
    {
        // Every output slot is exactly totalPeaks wide and the components of xyzOut sit back to
        // back, so a full 16-lane store would spill into the neighboring slot. Predicate to the
        // lanes that hold a real peak.
        int32_t const rem      = totalPeaks - (b << 4);
        int32_t const laneMask = (rem >= 16) ? 0xFFFF : (int32_t)(~(0xFFFFU << rem) & 0xFFFF);

        dvintx iIdx = dvint_load(aI);
        dvintx iL   = dvint_load(aL);
        dvintx iC   = dvint_load(aC);
        dvintx iR   = dvint_load(aR);

        dvfloatx fIdx = dvintx_dvfp(iIdx);
        dvfloatx fL   = dvintx_dvfp(iL);
        dvfloatx fC   = dvintx_dvfp(iC);
        dvfloatx fR   = dvintx_dvfp(iR);

        dvintx const absent = iIdx < 0;

        if ((emitMask & 1) != 0)
        {
            // Parabolic sub-bin interpolation with clamp-to-center at the histogram edges. den and
            // num are exact (inputs are integers below 2^24), so |den| > 1e-6f is just den != 0.
            dvfloatx const cL  = dvmux(iIdx > 0, fL, fC);
            dvfloatx const cR  = dvmux(iIdx < (numBinsPerHist - 1), fR, fC);
            dvfloatx const den = (cL - fC * 2.0f) + cR;
            dvfloatx const num = (cL - cR) * 0.5f;

            // dvfrcp is an approximate reciprocal, so this sub-bin offset is not bit-identical to
            // an IEEE divide. off is clamped to +/-0.5 of a bin immediately below.
            dvfloatx safe = dvmux(den < 0.0f, den, dvmux(den > 0.0f, den, vOne));
            dvfloatx off  = num * dvfrcp(safe);
            off           = dvmux(off > 0.5f, 0.5f, off);
            off           = dvmux(off < -0.5f, -0.5f, off);
            off           = dvmux(den < 0.0f, off, dvmux(den > 0.0f, off, vZero));

            dvfloatx sbi = fIdx + off;
            sbi          = dvmux(sbi < 0.0f, 0.0f, sbi);

            dvfloatx vRange = sbi * rangeMul + rangeConst;
            vRange          = dvmux(absent, 0.0f, vRange);

            // X, Y and Z are three views of one xyzOut at offsets 0, totalPeaks and 2*totalPeaks,
            // so a full 16-lane store would reach into the next component's slot. laneMask keeps
            // each store inside its own slot, which is what makes the order between them irrelevant.
            vstore(vRange, aRange, laneMask);
            vstore(vRange * xyzCal[0], aX, laneMask);
            vstore(vRange * xyzCal[1], aY, laneMask);
            vstore(vRange * xyzCal[2], aZ, laneMask);
        }

        if ((emitMask & 2) != 0)
        {
            dvfloatx vRefl = (fL + fC + fR) * invMaxIntensity;
            vRefl          = dvmux(absent, 0.0f, vRefl);
            vstore(vRefl, aRefl, laneMask);
        }
    }
}

#endif // LIDAR_HISTOGRAM_CONVERSION_KNL_H
