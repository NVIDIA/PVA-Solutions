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

#include "capon_roi_preprocess_fft.h"
#include "capon_roi_preprocess_knl.h"
#include "capon_roi_preprocess_params.h"

#include <PvaOperatorTypes.h>
#include <cupva_device.h>
#include <cupva_device/device_core.h>
#include <cupva_device/sequence_dataflow.h>

/* This file builds both kernels of the operator: CRP_COARSE_DOA_FFT = 1 is the batched FFT-64 coarse-DoA
 * kernel, 0 the direct-DFT kernel. The host picks one per operator (UseFftCoarseDoA). */
#ifndef CRP_COARSE_DOA_FFT
#    error "capon_roi_preprocess: build with -DCRP_COARSE_DOA_FFT=0 (direct DFT) or 1 (FFT)"
#endif

VMEM(A, PVACaponROIPreprocessParams, GP);

VMEM(A, float, wazBuf, BUF_SZ_WAZ);
VMEM(C, int32_t, detDopBuf, BUF_SZ_DETDOP); /* per-detection Doppler bin (bank C: scalar reads only; A is full) */
VMEM(A, float, elemBuf, BUF_SZ_ELEM);
VMEM(A, int16_t, conjXBuf, BUF_SZ_CONJX);
/* Coarse-DoA az-DFT output, split re/im, ka-contiguous [numElVirt][beamAz] (vector store axis). */
VMEM(A, float, coarseTmpReBuf, BUF_SZ_COARSE_TMP);
VMEM(A, float, coarseTmpImBuf, BUF_SZ_COARSE_TMP);
VMEM(A, float, tmpRoiReBuf, BUF_SZ_PATCHMAT); /* azimuth-projection output [<=svDim][H] */
VMEM(A, float, tmpRoiImBuf, BUF_SZ_PATCHMAT);
VMEM(A, float, patchReBuf, BUF_SZ_PATCHMAT); /* elevation-projection ROI patch [svDim][H] */
VMEM(B, float, patchImBuf, BUF_SZ_PATCHMAT); /* bank B: splits the Re/Im pair across banks */

#if !CRP_COARSE_DOA_FFT
/* Transposed az twiddles [numAzVirt][beamAz], complex interleaved, so the coarse-DoA az DFT can vectorize
 * over the beam axis. */
VMEM(B, int16_t, wazTBuf, 2 * BUF_SZ_WAZT);
#endif
VMEM(B, int16_t, waz16Buf, BUF_SZ_WAZ16); /* int16 Q15 az twiddles (fixed-point projection) */
VMEM(B, int16_t, wel16Buf, BUF_SZ_WEL16); /* int16 Q15 el twiddles (fixed-point el projection, numElVirt>1) */

/* snapshotsROI tiles, one per batch of CRP_OUT_BATCH detections. */
#define CRP_OUT_NBATCH ((CRP_MAX_DET + CRP_OUT_BATCH - 1) / CRP_OUT_BATCH)
#if CRP_OUT_NBATCH != 4
#    error "capon_roi_preprocess: declare one outBuf/outRDFHdl per snapshotsROI batch"
#endif
VMEM(C, int32_t, outBuf, RDF_DOUBLE(int32_t, CRP_MAX_SVDIM * 2, CRP_MAX_H, 0, 0, TRANS_MODE_2));
VMEM(C, int32_t, outBuf1, RDF_DOUBLE(int32_t, CRP_MAX_SVDIM * 2, CRP_MAX_H, 0, 0, TRANS_MODE_2));
VMEM(C, int32_t, outBuf2, RDF_DOUBLE(int32_t, CRP_MAX_SVDIM * 2, CRP_MAX_H, 0, 0, TRANS_MODE_2));
VMEM(C, int32_t, outBuf3, RDF_DOUBLE(int32_t, CRP_MAX_SVDIM * 2, CRP_MAX_H, 0, 0, TRANS_MODE_2));

#if CRP_COARSE_DOA_FFT
VMEM(A, int32_t, crpFftTwiddleQ30, 128); /* radix-4 intra-stage twiddles, int32 Q30 interleaved re/im */
VMEM(B, int32_t, crpFftPing, CRP_FFT_BUF_ELEMS);
VMEM(B, int32_t, crpFftPong, CRP_FFT_BUF_ELEMS);
VMEM(B, int32_t, crpFftTmp, CRP_FFT_BUF_ELEMS);

VMEM(A, FFTFactorization, crpFftFactorization);
VMEM(A, AgenCFG, crpFftCfgs, CRP_FFT_MAX_CFGS);
/* Staged fp32 elem, one buffer per packed chunk; the second lives in bank C, where it fits. */
VMEM(A, float, fftBatchElemBuf0, CRP_FFT_LANES *CRP_FFT_ELEM_STRIDE);
#    if CRP_FFT_GROUPS > 1
VMEM(C, float, fftBatchElemBuf1, CRP_FFT_LANES *CRP_FFT_ELEM_STRIDE);
#    endif
#    if CRP_FFT_GROUPS > 2
#        error "capon_roi_preprocess: size a staged-elem buffer for every FFT chunk"
#    endif
#endif /* CRP_COARSE_DOA_FFT */

VMEM(C, float, vdBuf, BUF_SZ_VDBUF);    /* VD built on the device, int32 (see caponBuildVdFx) */
VMEM(B, float, planeBuf, BUF_SZ_PLANE); /* one detection's range-Doppler plane row (plane RDF tile) */
VMEM(A, int32_t, crpVdFold);            /* 1: the host folded conj(DDM weight) into waz16 (see caponBuildVdFx) */
VMEM(A, int32_t, crpUseDotProj); /* general kernel: 1 = dot-product projection (UseDotProductProjection), host-set */
VMEM(A, int32_t, crpVdTxBase, CRP_MAX_TX); /* (relDopplerOffset[tx] - halfW) mod numChirps (host-set) */
/* c_k = exp(-2 pi j k (numAzVirt-1) / beamAz), Q14 + Q29 parts, host-set */
VMEM(B, int16_t, crpFbaRot, CRP_MAX_BEAM_AZ * 4);
VMEM(B, int16_t, crpFbaRotEl, CRP_MAX_BEAM_EL * 4); /* the same for the elevation tables, host-set */
/* Projection coefficients of beams 2-3 of a group (beams 0-1 sit in coarseTmpReBuf, bank A), so the four
 * coefficient loads of a tap pair use two superbanks. */
VMEM(B, int16_t, crpCoefB, 2 * (CRP_MAX_NVC / 2) * 32);
VMEM_POINTER(A, lutExtPtr);  /* steering LUT DRAM base (host-set program parameter) */
VMEM_POINTER(A, svExtPtr);   /* steeringVectorsROI DRAM base (host-set program parameter) */
VMEM(A, int32_t, crpNumDet); /* detections to process this run (host-set per Submit, <= maxDet) */

VMEM_RDF_UNIFIED(A, wazRDFHdl);
VMEM_RDF_UNIFIED(A, elemRDFHdl);
VMEM_RDF_UNIFIED(A, planeRDFHdl);  /* rangeDopplerPlanes, one row per detection */
VMEM_RDF_UNIFIED(A, detDopRDFHdl); /* detDopplers (one-shot) */
VMEM_RDF_UNIFIED(A, outRDFHdl);
/* Extra transpose-out handles for maxDet > CRP_OUT_BATCH: each batch needs its own TRANS_MODE_2 RDF
 * (1D grid) for the transpose to map to det-major DRAM. */
VMEM_RDF_UNIFIED(A, outRDFHdl1);
VMEM_RDF_UNIFIED(A, outRDFHdl2);
VMEM_RDF_UNIFIED(A, outRDFHdl3);
VMEM_SEQDF_HANDLER(A, lutGatherHdl); /* device-addressed DRAM -> DRAM copy of one LUT block */
VMEM_SEQDF_HANDLER(A, twiddle16Hdl); /* host-quantized int16 Q15 twiddles, one-shot */

/* Per-run context handed to the per-detection stages (buffers/handlers above + geometry). */
VMEM(A, CaponKnl, knl);

#if !CRP_COARSE_DOA_FFT
/* Direct-DFT kernel: coarse DoA and projection for one detection. Kept out of line so its schedule
 * does not depend on the rest of the main function. */
[[gnu::noinline]] static void caponProcessDetectionBartlett(int32_t det, int32_t b, int32_t nvc, int32_t H,
                                                            const int32_t *detDop, UnifiedRDFHandler *const *outH,
                                                            int32_t outLaneOfst, int32_t outLinePitch)
{
    float *elem  = (float *)cupvaRasterDataFlowAcquire(elemRDFHdl);
    float *plane = (float *)cupvaRasterDataFlowAcquire(planeRDFHdl);
    caponBuildVdFx(&knl, plane, detDop[det]); /* VD + q_d for the projection */

    int32_t *out = (int32_t *)cupvaRasterDataFlowAcquire(*outH[b]);

    int32_t roiAzStart, roiElStart;
    caponCoarseDoAFx(&knl, elem, &roiAzStart, &roiElStart); /* coarse DoA + ROI select */
    caponIssueSteering(&knl, roiAzStart, roiElStart, det);  /* steering block copy: start (overlaps) */
    caponProjectAzFx(&knl, roiAzStart);                     /* azimuth projection */
#    if CRP_PROJ_GENERAL
    if (!knl.useDotProj)
        caponProjectElPerTapFx(&knl, roiElStart); /* elevation projection, per-tap MAC paths */
    else if (knl.numElVirt > 1)
        caponProjectElDotP4Fx(&knl, roiAzStart, roiElStart); /* elevation projection, dot-product path */
#    endif
    caponQuantizeSnapshotsFxT(&knl, out, outLaneOfst, outLinePitch); /* quantize (transpose store) */

    cupvaRasterDataFlowRelease(elemRDFHdl);
    cupvaRasterDataFlowRelease(planeRDFHdl);
    cupvaRasterDataFlowRelease(*outH[b]);
}

#else  /* CRP_COARSE_DOA_FFT */

/* Coarse DoA for up to CRP_FFT_DETS detections through the batched FFT-64: groupSize acquire/release
 * cycles, one engine run over all CRP_FFT_GROUPS packed 8-detection chunks, then groupSize
 * projection/quantize/steering-gather passes. In a short tail group the last real detection refills
 * the spare lanes of its chunk, and only the chunks holding real detections are quantized and ranked.
 * noinline: the FFT path schedules better with this out of line. */
[[gnu::noinline]] static void caponProcessDetectionsFftBatch(int32_t det, int32_t groupSize, int32_t batchTranspose,
                                                             int32_t nvc, int32_t H, const int32_t *detDop,
                                                             UnifiedRDFHandler *const *outH, int32_t outLaneOfst,
                                                             int32_t outLinePitch)
{
    int32_t roiAzArr[CRP_FFT_DETS], roiElArr[CRP_FFT_DETS];
    for (int32_t lane = 0; lane < groupSize; lane++)
    {
        float *elem = (float *)cupvaRasterDataFlowAcquire(elemRDFHdl);
        caponCoarseDoAFftBatchPrepFx(&knl, elem, lane, groupSize);
        cupvaRasterDataFlowRelease(elemRDFHdl);
    }
    caponCoarseDoAFftBatchRankFx(&knl, groupSize, roiAzArr, roiElArr);

    for (int32_t lane = 0; lane < groupSize; lane++)
    {
        const int32_t d = det + lane;
        const int32_t b = batchTranspose ? (d / CRP_OUT_BATCH) : 0;
        float *plane    = (float *)cupvaRasterDataFlowAcquire(planeRDFHdl);
        caponBuildVdFx(&knl, plane, detDop[d]);

        int32_t *out = (int32_t *)cupvaRasterDataFlowAcquire(*outH[b]);

        caponIssueSteering(&knl, roiAzArr[lane], roiElArr[lane], d);
        caponProjectAzFx(&knl, roiAzArr[lane]); /* one elevation row: elevation projection fused */
        caponQuantizeSnapshotsFxT(&knl, out, outLaneOfst, outLinePitch);

        cupvaRasterDataFlowRelease(planeRDFHdl);
        cupvaRasterDataFlowRelease(*outH[b]);
    }
}
#endif /* CRP_COARSE_DOA_FFT */

/* A detection at or beyond the detection count: its tiles are acquired and released (every tile of an
 * RDF must be) without processing. Its output rows are left unspecified. */
static void caponSkipDetection(UnifiedRDFHandler *outH)
{
    cupvaRasterDataFlowAcquire(elemRDFHdl);
    cupvaRasterDataFlowRelease(elemRDFHdl);
    cupvaRasterDataFlowAcquire(planeRDFHdl);
    cupvaRasterDataFlowRelease(planeRDFHdl);
    cupvaRasterDataFlowAcquire(*outH);
    cupvaRasterDataFlowRelease(*outH);
}

[[gnu::noinline]] static void crpRunDetections(int32_t numDet, int32_t batchTranspose, int32_t nvcProj, int32_t H,
                                               const int32_t *detDop, UnifiedRDFHandler *const *outH,
                                               int32_t outLaneOfst, int32_t outLinePitch)
{
#if CRP_COARSE_DOA_FFT
    /* FFT kernel: up to CRP_FFT_DETS detections per engine run. */
    for (int32_t det = 0; det < numDet; det += CRP_FFT_DETS)
    {
        const int32_t remain    = numDet - det;
        const int32_t groupSize = (remain < CRP_FFT_DETS) ? remain : CRP_FFT_DETS;
        caponProcessDetectionsFftBatch(det, groupSize, batchTranspose, nvcProj, H, detDop, outH, outLaneOfst,
                                       outLinePitch);
    }
#else
    /* Direct-DFT kernel: one detection at a time. */
    for (int32_t det = 0; det < numDet; det++)
    {
        const int32_t b = batchTranspose ? (det / CRP_OUT_BATCH) : 0; /* out RDF that owns this detection */
        caponProcessDetectionBartlett(det, b, nvcProj, H, detDop, outH, outLaneOfst, outLinePitch);
    }
#endif
}

/* int16 Q15 az twiddles are quantized on the HOST and delivered by SQDF; static addresses, so this
 * is just trigger-and-wait. noinline: runs once per run, and inlining it perturbs the per-detection
 * body's scheduling. */
[[gnu::noinline]] static void crpFetchTwiddle16(void)
{
    cupvaSQDFOpen(twiddle16Hdl);
    cupvaSQDFTrig(twiddle16Hdl);
    cupvaSQDFSync(twiddle16Hdl);
    cupvaSQDFClose(twiddle16Hdl);
}

CUPVA_VPU_MAIN()
{
    cupvaFloatingPointNANErrorEnabled(false);
    /* Not const: copying GP out of VMEM is lowered to a memory copy that needs a non-const destination. */
    PVACaponROIPreprocessParams gp = GP;
    const int32_t numAzVirt        = gp.numAzVirt;
    const int32_t numElVirt        = gp.numElVirt;
    const int32_t beamAz           = gp.beamAz;
    const int32_t beamEl           = gp.beamEl;
    const int32_t roiAz            = gp.roiAz;
    const int32_t roiEl            = gp.roiEl;
    const int32_t svDim            = gp.svDim;
    const int32_t H                = gp.numDopplerBinsCovMat;
    const int32_t inputQBits       = gp.inputQBits;
    /* The projection runs on the virtual array (numAzVirt*numElVirt). Coarse DoA may run on a
     * larger zero-filled position grid (coarseAzVirt x coarseElVirt); when the grid parameters are
     * left at 0 the two coincide. */
    const int32_t nvcProj      = numAzVirt * numElVirt;
    const int32_t coarseElVirt = gp.coarseElVirt ? gp.coarseElVirt : numElVirt;
    const int32_t nvc          = numAzVirt * coarseElVirt; /* coarse DoA */
    const int32_t shAz         = beamAz / 2;
    const int32_t shEl         = beamEl / 2;
    const int32_t nDet         = gp.maxDet;
    const int32_t numDet       = (crpNumDet < nDet) ? crpNumDet : nDet; /* detections processed */
    const int32_t numAzBins    = gp.numAzBins;
    const int32_t numElBins    = gp.numElBins;
    /* Steering LUT geometry: block per ROI position, indexed roiPos = roiElStart*numRoiAzPos + roiAzStart. */
    const int32_t numRoiAzPos = beamAz - roiAz + 1;
    const int32_t svBlockInts = svDim * numElBins * numAzBins * 2;
    const int64_t lutBase     = lutExtPtr.base + lutExtPtr.offset;

    /* Dataflow phase 0, the one-shot inputs: the twiddles, the fp32 az table and detDopplers (the whole [maxDet]
     * array in one tile, kept in detDopBuf for the detection loop). Mirrors the host: the az table is only
     * resident when it fits, which is exactly when the KC fp32 refine can run. */
    crpFetchTwiddle16();
    float *waz = wazBuf;
    if (beamAz * numAzVirt * 2 <= BUF_SZ_WAZ)
    {
        cupvaRasterDataFlowOpen(wazRDFHdl, wazBuf);
        waz = (float *)cupvaRasterDataFlowAcquire(wazRDFHdl);
        cupvaRasterDataFlowRelease(wazRDFHdl);
        cupvaRasterDataFlowClose(wazRDFHdl);
    }
    cupvaRasterDataFlowOpen(detDopRDFHdl, detDopBuf);
    cupvaRasterDataFlowAcquire(detDopRDFHdl);
    cupvaRasterDataFlowRelease(detDopRDFHdl);
    cupvaRasterDataFlowClose(detDopRDFHdl);
    /* Per-run context for the per-detection stages; they never reference the VMEM globals directly. */
    knl.numAzVirt = numAzVirt, knl.numElVirt = numElVirt;
    knl.beamAz = beamAz, knl.beamEl = beamEl, knl.roiAz = roiAz, knl.roiEl = roiEl;
    knl.svDim = svDim, knl.H = H;
    knl.inputQBits = inputQBits, knl.nvc = nvc, knl.shAz = shAz, knl.shEl = shEl;
    knl.coarseElVirt = coarseElVirt;
    knl.azLo = gp.coarseAzBinLo, knl.azHi = gp.coarseAzBinHi;
    knl.elLo = gp.coarseElBinLo, knl.elHi = gp.coarseElBinHi;
    knl.numRoiAzPos = numRoiAzPos, knl.svBlockInts = svBlockInts, knl.lutBase = lutBase;
    knl.svBase = svExtPtr.base + svExtPtr.offset, knl.svPending = 0;
    knl.conjXBuf       = conjXBuf;
    knl.coarseTmpReBuf = coarseTmpReBuf, knl.coarseTmpImBuf = coarseTmpImBuf;
#if !CRP_COARSE_DOA_FFT
    knl.wazTBuf = wazTBuf;
#endif
    knl.waz16Buf = waz16Buf, knl.wel16Buf = wel16Buf;
    knl.tmpRoiReBuf = tmpRoiReBuf, knl.tmpRoiImBuf = tmpRoiImBuf;
    knl.patchReBuf = patchReBuf, knl.patchImBuf = patchImBuf;
    knl.waz          = waz;
    knl.lutGatherHdl = &lutGatherHdl;
    knl.numTx = gp.numTx, knl.numRx = gp.numRx, knl.numChirps = gp.numChirps;
    knl.numSel = 2 * (gp.dopplerNeighborhood / 2), knl.enableFBA = gp.enableFBA ? 1 : 0, knl.nvcProj = nvcProj;
    for (int32_t tx = 0; tx < CRP_MAX_TX; tx++)
    {
        knl.vdTxBase[tx] = crpVdTxBase[tx];
        knl.ddmaRe[tx] = gp.ddmaWeightRe[tx], knl.ddmaIm[tx] = gp.ddmaWeightIm[tx];
    }
    /* Dot-product projection (crpProjectAzDotP4, not in the general kernel): azimuth-only and whole tap pairs.
     * With FBA, either the generated tables (backward rows by symmetry, fbaSym) or mirror VD rows starting a
     * Doppler block. */
    /* The azimuth-only and FFT kernels always take the dot-product path; the general kernel when the host says so. */
    knl.useDotProj = CRP_PROJ_GENERAL ? crpUseDotProj : 1;
    knl.fbaSym     = knl.useDotProj && knl.enableFBA && !gp.useCustomTwiddles;
    knl.fbaRot     = crpFbaRot;
    knl.fbaRotEl   = crpFbaRotEl;
    knl.coefB      = crpCoefB;
    knl.vdBufInt   = (int32_t *)vdBuf;
    knl.vdFold     = crpVdFold;
    caponInitVdFx(&knl); /* cells no detection writes stay zero */

    caponInitProjectAgensFx(&knl); /* projection/quantize patch agens, built once (see knl.h) */
    caponInitRankAgensFx(&knl);    /* coarse-DoA ranking agens, built once (see knl.h) */
#if CRP_COARSE_DOA_FFT
    knl.fftBatchIn = crpFftPing, knl.fftBatchOut = crpFftPong;
    knl.fftBatchElem[0] = fftBatchElemBuf0;
#    if CRP_FFT_GROUPS > 1
    knl.fftBatchElem[1] = fftBatchElemBuf1;
#    endif
    /* Build the engine's agen configs once; valid because the buffers and the twiddle table are the
     * same static addresses on every call. */
    caponInitFft64Batch(crpFftTwiddleQ30);
    caponInitFftBatchAgensFx(&knl); /* hoist RankFx's agens (see knl.h) */
#endif

    const int32_t *detDop = detDopBuf; /* per-detection Doppler bin, fetched in phase 0 */
    cupvaRasterDataFlowOpen(elemRDFHdl, elemBuf);
    cupvaRasterDataFlowOpen(planeRDFHdl, planeBuf);

    /* snapshotsROI is always written by transposing DMA: one TRANS_MODE_2 RDF per batch of
     * CRP_OUT_BATCH detections (a single batch when nDet <= CRP_OUT_BATCH). */
    const int32_t batchTranspose                  = (nDet > CRP_OUT_BATCH);
    const int32_t nBatch                          = (nDet + CRP_OUT_BATCH - 1) / CRP_OUT_BATCH;
    UnifiedRDFHandler *const outH[CRP_OUT_NBATCH] = {&outRDFHdl, &outRDFHdl1, &outRDFHdl2, &outRDFHdl3};
    void *const outBufs[CRP_OUT_NBATCH]           = {outBuf, outBuf1, outBuf2, outBuf3};

    /* Open every out RDF exactly once, never per detection/batch: they all stay open and the loop
     * acquires from the batch owning det/CRP_OUT_BATCH. All batches share one lane offset/pitch. */
    for (int32_t bo = 0; bo < nBatch; bo++) cupvaRasterDataFlowOpen(*outH[bo], outBufs[bo]);
    const int32_t outLaneOfst  = cupvaTransposeLaneOfst(outRDFHdl, TRANS_MODE_2, sizeof(int32_t));
    const int32_t outLinePitch = cupvaRasterDataFlowGetLinePitch(outRDFHdl);
    cupvaSQDFOpen(lutGatherHdl);

    crpRunDetections(numDet, batchTranspose, nvcProj, H, detDop, outH, outLaneOfst, outLinePitch);

    for (int32_t det = numDet; det < nDet; det++)
    {
        caponSkipDetection(outH[batchTranspose ? (det / CRP_OUT_BATCH) : 0]);
    }

    cupvaRasterDataFlowClose(elemRDFHdl);
    cupvaRasterDataFlowClose(planeRDFHdl);
    for (int32_t bo = 0; bo < nBatch; bo++) cupvaRasterDataFlowClose(*outH[bo]);
    caponCompleteSteering(&knl); /* the last detection's steering copy */
    cupvaSQDFClose(lutGatherHdl);

    return 0;
}
