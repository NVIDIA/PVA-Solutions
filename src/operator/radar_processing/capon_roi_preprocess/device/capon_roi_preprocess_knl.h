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

#ifndef PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_KNL_H
#define PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_KNL_H

/* Rules for this device code, which must build with every supported VPU compiler:
 *
 * 1. Order a vector store against the (scalar or vector) read that consumes it with CRP_SEPARATOR(),
 *    not chess_separator_scheduler(): where the scheduler separator is not honoured, CRP_SEPARATOR()
 *    is a memory fence instead.
 * 2. Read an agen stored in CaponKnl only through CRP_AG_RELOAD (a volatile load) or CRP_AG_REBUILD
 *    (rebuilt from k by its crpBuild_<field>(k) builder), never directly: neighbouring stored-agen
 *    reloads can otherwise be folded into one post-incremented load chain that reads the wrong field.
 *    Both are exact; which one a call site uses is a performance choice.
 */
#if defined(__has_builtin)
#    if __has_builtin(__builtin_nvvpu_mem_fence)
#        define CRP_SEPARATOR() chess_memory_fence()
#    endif
#endif
#ifndef CRP_SEPARATOR
#    define CRP_SEPARATOR() chess_separator_scheduler()
#endif

/* Stored-agen reads (rule 2). Under chess both are the stored copy. */
#if defined(__has_builtin)
#    if __has_builtin(__builtin_nvvpu_mem_fence)
#        define CRP_AG_RELOAD(k, f)                                     \
            ({                                                          \
                __typeof__((k)->f) const *volatile crpAgPtr_ = &(k)->f; \
                *crpAgPtr_;                                             \
            })
#        define CRP_AG_REBUILD(k, f) crpBuild_##f(k)
#    endif
#endif
#ifndef CRP_AG_RELOAD
#    define CRP_AG_RELOAD(k, f) ((k)->f)
#    define CRP_AG_REBUILD(k, f) ((k)->f)
#endif

#include "capon_roi_preprocess_params.h"

#include <PvaOperatorTypes.h>
#include <cupva_device.h>
#include <cupva_device/sequence_dataflow.h>
#include <stdint.h>

#define CRP_VECW 8

/* VD transpose staging: CRP_VECW Doppler rows of int32 at a pitch of 16*K+1 words, the lane pitch of a
 * transposed load with lane_ofst K. K = 4 -> 65 >= the 64 physical channels a row holds. */
#define CRP_VD_STAGE_LANE_OFST 4
#define CRP_VD_STAGE_PITCH (16 * CRP_VD_STAGE_LANE_OFST + 1)
#define CRP_VD_STAGE_ROWS CRP_MAX_H /* staging block rows: one TX window (numSel <= H) */

/* Store predicate enabling the first n lanes of a vector store (bit i enables lane i); n in [0, 31].
 * Use -1 for a full store. A predicate is a lane mask, not a lane count. */
#define CRP_LOW_LANES(n) ((int32_t)((1u << (uint32_t)(n)) - 1u))

typedef struct
{
    int32_t numAzVirt, numElVirt, beamAz, beamEl;
    int32_t roiAz, roiEl, svDim, H, inputQBits;
    int32_t nvc, shAz, shEl;
    /* Coarse-DoA FoV window, inclusive bins in the shifted (output) bin space. Full range means
     * azLo==0 && azHi==beamAz-1 && elLo==0 && elHi==beamEl-1. */
    int32_t azLo, azHi, elLo, elHi;
    int32_t coarseElVirt; /* coarse-DoA grid height (>= numElVirt; grid is numAzVirt wide) */
    int32_t numRoiAzPos, svBlockInts;
    int64_t lutBase;
    int64_t svBase;    /* steeringVectorsROI DRAM base */
    int32_t svPending; /* 1: a steering copy is in flight */
    /* scratch + twiddle buffers (VMEM), owned by top.c */
    int16_t *conjXBuf; /* int16 Q13 conj(elem) scratch (never read as float) */
    float *coarseTmpReBuf, *coarseTmpImBuf;
    int16_t *wazTBuf; /* host-built transposed int16 Q15 az twiddles [a][ka], complex interleaved */

    float *tmpRoiReBuf, *tmpRoiImBuf, *patchReBuf, *patchImBuf;
    float *waz; /* acquired interleaved az twiddle table (fp32 KC refine only) */
    SequenceDataFlowHandler *lutGatherHdl;
    int16_t *waz16Buf;
    int16_t *wel16Buf; /* int16 Q15 el twiddles [beamEl][numElVirt] (fixed-point el projection, numElVirt>1) */
    int32_t *vdFxReBuf, *vdFxImBuf;
    int32_t vdQBits;
    int32_t useDotProj; /* 1: dot-product projection (crpProjectAzDotP4, caponProjectElDotP4Fx); the VD rows are then
                         * interleaved re/im per 8 Doppler rows. 0: per-tap MAC paths, separate re/im planes. */
    int32_t fbaSym; /* 1 (useDotProj, FBA, generated tables): backward rows from the forward projection, no mirror VD */
    int16_t *coefB; /* projection coefficients of beams 2-3 of a group (bank B) */
    const int16_t *fbaRotEl; /* the same for the elevation tables: c_k = Wel[k][numElVirt-1] per row k */
    const int16_t *fbaRot;   /* c_k = W[k][numAzVirt-1] per table row k: Q14 re, im, then Q29 remainders re, im */
    /* VD build (caponBuildVdFx): the DDM decode of one detection's range-Doppler plane row into vdBuf. */
    int32_t numTx, numRx, numChirps, numSel, enableFBA, nvcProj;
    int32_t vdTxBase[CRP_MAX_TX]; /* (relDopplerOffset[tx] - halfW) mod numChirps, host-computed */
    float ddmaRe[CRP_MAX_TX], ddmaIm[CRP_MAX_TX];
    int32_t *vdBufInt; /* vdBuf as int32: re half [nvcProj][H], then im half */
    int32_t vdFold;    /* 1: conj(DDM weight) is folded into waz16 (host); the VD build only scales conj(plane) */
    int32_t *fftBatchIn, *fftBatchOut;
    float *fftBatchElem[CRP_FFT_GROUPS]; /* staged fp32 elem, one buffer per 8-detection chunk */

    agen_A prAgD, prAgV;
    agen_B piAgD, piAgV;
    agen_A patchRdRe;
    agen_B patchRdIm;
    agen_A accAgReD, accAgImD, accAgReV, accAgImV;
    agen_A elPRAgD, elPRAgV;
    agen_B elPIAgD, elPIAgV;
    agen_A elTrAgD, elTiAgD, elTrAgV, elTiAgV;
    agen_A rankTrAgen, rankTiAgen, rankPAgen;
    /* caponCoarseDoAFftBatchRankFx's agens, built once. Bases are static VMEM (the ping/tmp
     * spectrum scratch and the coarseTmp buffers), so they are per-run-invariant; the engine-output
     * load depends on the chunk and is built per chunk. The per-lane spectrum copy is one 2D agen
     * per side rather than one per lane: the source steps lanes in its outer dimension, the
     * destination rewinds (s2 = 0) because every lane lands in the same coarseTmp buffer. They are
     * reloaded for every chunk. */
    agen_B fftDbStRe;
    agen_C fftDbStIm;
    agen_A fftCpySrcRe;
    agen_B fftCpySrcIm;
    agen_C fftCpyDstRe;
    agen_A fftCpyDstIm;
} CaponKnl;

/* The operator has three VPU programs, built from the same sources and picked by the host per operator
 * (OpCaponROIPreprocess.cpp):
 * - fft (CRP_COARSE_DOA_FFT = 1): batched FFT-64 coarse DoA (UseFftCoarseDoA). Links capon_roi_preprocess_fft.c
 *   and the shared FFT engine.
 * - dft (CRP_COARSE_DOA_FFT = 0, CRP_PROJ_GENERAL = 0): direct-DFT coarse DoA and the dot-product projection,
 *   for azimuth-only arrays (UseDotProductProjection, numElVirt == 1).
 * - dftgen (CRP_COARSE_DOA_FFT = 0, CRP_PROJ_GENERAL = 1): direct-DFT coarse DoA for every other geometry. It
 *   picks the dot-product projection or the per-tap MAC paths at run time (crpUseDotProj) and also links
 *   capon_roi_preprocess_elproj.c and capon_roi_preprocess_pertap.c.
 * capon_roi_preprocess_dft.c is in dft and dftgen; capon_roi_preprocess_init.c, top.c, knl.c, rank.c and dft2d.c
 * are in all three. */

/* The FFT kernel only runs the geometry UseFftCoarseDoA accepts: one elevation row, 64 azimuth taps
 * and beams, and the full coarse-DoA field of view. CRP_GENERAL_GEOMETRY = 0 compiles out the paths
 * only other geometries reach (elevation projection, field-of-view mask, partial-vector refine tail). */
/* CRP_PROJ_GENERAL = 1: the general direct-DFT kernel, projecting with the per-tap MAC paths
 * (capon_roi_preprocess_pertap.c); 0: the dot-product projection (azimuth-only DFT kernel and the FFT kernel). */
#ifndef CRP_PROJ_GENERAL
#    define CRP_PROJ_GENERAL 0
#endif

#if defined(CRP_COARSE_DOA_FFT) && CRP_COARSE_DOA_FFT
#    define CRP_GENERAL_GEOMETRY 0
#else
#    define CRP_GENERAL_GEOMETRY 1
#endif

/* Setup (called once, before the detection loop), both kernels. */
void caponInitProjectAgensFx(CaponKnl *k);
void caponInitRankAgensFx(CaponKnl *k);

/* Coarse DoA: conj(elem) is quantized to int16 at this fixed Q before the int16 DFT/FFT. */
#define CRP_COARSE_CONJ_QBITS 13

/* Coarse-DoA ranking (top-KC fp32 refine of coarseTmpRe/ImBuf), both kernels. */
void caponCoarseDoARankAzFx(CaponKnl *k, const float *elem, int32_t *peakAzOut, int32_t *peakElOut);

/* ROI start for a coarse peak, both kernels: centered on the peak (start = peak - (roiSize-1)/2), then
 * clamped into the beamspace. The Capon peak search needs the coarse peak inside the ROI, not on its
 * edge. */
static inline void caponCenterRoi(const CaponKnl *k, int32_t peakAz, int32_t peakEl, int32_t *roiAzStart,
                                  int32_t *roiElStart)
{
    int32_t azStart = peakAz - (k->roiAz - 1) / 2, elStart = peakEl - (k->roiEl - 1) / 2;
    if (azStart > k->beamAz - k->roiAz)
        azStart = k->beamAz - k->roiAz;
    if (azStart < 0)
        azStart = 0;
    if (elStart > k->beamEl - k->roiEl)
        elStart = k->beamEl - k->roiEl;
    if (elStart < 0)
        elStart = 0;
    *roiAzStart = azStart;
    *roiElStart = elStart;
}

#if !CRP_COARSE_DOA_FFT
/* Coarse DoA -- direct DFT (capon_roi_preprocess_dft.c), direct-DFT kernel only. */
void caponCoarseDoAFx(CaponKnl *k, const float *elem, int32_t *roiAzStart, int32_t *roiElStart);
#else
/* Coarse DoA -- batched FFT-64 (capon_roi_preprocess_fft.c), FFT kernel only. Build
 * caponCoarseDoAFftBatchRankFx's agens once; needs fftBatchOut set. */
void caponInitFftBatchAgensFx(CaponKnl *k);
void caponCoarseDoAFftBatchPrepFx(CaponKnl *k, const float *elem, int32_t lane, int32_t groupSize);
void caponCoarseDoAFftBatchRankFx(CaponKnl *k, int32_t groupSize, int32_t *roiAzStartOut /* [groupSize] */,
                                  int32_t *roiElStartOut /* [groupSize] */);
#endif

/* VD build (capon_roi_preprocess_knl.c). caponInitVdFx clears vdBuf once per run: every detection writes the
 * same cells, so the rest stay zero. caponBuildVdFx decodes one detection's plane row into vdBuf at Q(q_d) and
 * sets vdFxReBuf/vdFxImBuf/vdQBits for the projection. plane is read only, and with numRx < CRP_MAX_RX up to one
 * row past its numChirps rows (the slack in BUF_SZ_PLANE). */
void caponInitVdFx(CaponKnl *k);
void caponBuildVdFx(CaponKnl *k, float *plane, int32_t detDoppler);

/* Azimuth projection, elevation projection and quantize (capon_roi_preprocess_knl.c). */
void caponProjectAzFx(CaponKnl *k, int32_t roiAzStart);
void caponProjectAzPerTapFx(CaponKnl *k, int32_t roiAzStart);                    /* capon_roi_preprocess_pertap.c */
void caponProjectElDotP4Fx(CaponKnl *k, int32_t roiAzStart, int32_t roiElStart); /* capon_roi_preprocess_elproj.c */
void caponCoarseDoA2DFx(CaponKnl *k, int32_t *peakAzOut, int32_t *peakElOut);    /* capon_roi_preprocess_dft2d.c */
#if CRP_GENERAL_GEOMETRY
void caponProjectElPerTapFx(CaponKnl *k, int32_t roiElStart);
#endif
void caponQuantizeSnapshotsFxT(CaponKnl *k, int32_t *out, int32_t laneOfst, int32_t linePitchDst);

/* Steering block: copy the ROI's LUT block to steeringVectorsROI row det (DRAM -> DRAM). The copy runs while the
 * detection is processed; the next issue, or caponCompleteSteering at the end of the run, waits for it. */
void caponIssueSteering(CaponKnl *k, int32_t roiAzStart, int32_t roiElStart, int32_t det);
void caponCompleteSteering(CaponKnl *k);

/* 2D int32 agen builder (size in bytes; strides in elements). Shared because both
 * capon_roi_preprocess_knl.c and capon_roi_preprocess_dft.c build this same shape. */
static inline agen_A initAgenA2_i(int32_t *base, int32_t n1, int32_t s1, int32_t n2, int32_t s2)
{
    AgenWrapper w = {.size = sizeof(int32_t), .n1 = (uint16_t)n1, .n2 = (uint16_t)n2, .s1 = s1, .s2 = s2};
    agen_A a      = init_A((void *)base);
    INIT_AGEN2(a, w);
    return a;
}
static inline agen_B initAgenB2_i(int32_t *base, int32_t n1, int32_t s1, int32_t n2, int32_t s2)
{
    AgenWrapper w = {.size = sizeof(int32_t), .n1 = (uint16_t)n1, .n2 = (uint16_t)n2, .s1 = s1, .s2 = s2};
    agen_B a      = init_B((void *)base);
    INIT_AGEN2(a, w);
    return a;
}

#endif /* PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_KNL_H */
