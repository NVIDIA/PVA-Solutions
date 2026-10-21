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
 * @file capon_roi_preprocess_params.h
 * @brief Device-side configuration limits for the Capon ROI preprocessing operator.
 */

#ifndef PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_PARAMS_H
#define PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_PARAMS_H

/* Upper bounds for VMEM sizing. H (FBA-expanded Doppler) <= 64; svDim <= 16; maxDet <= 1024.
 * The host enforces all of them in CaponROIPreprocess::ValidateParams, and OpCaponROIPreprocess.h
 * lists them under Limitations; keep the three in sync.
 *
 * The coarse-DoA stage and the projection stage are bounded separately. Coarse DoA runs on the
 * numAzVirt x coarseElVirt grid (a zero-filled position grid can be far larger than the virtual array),
 * while the projection runs on the nvc = numAzVirt*numElVirt virtual array, which holds the
 * numTx*numRx physical channels. Sizing both from one bound would make vdBuf (2*nvc*H) explode. */
#define CRP_MAX_AZVIRT 69                                    /* numAzVirt (coarse-DoA az axis / grid width)  */
#define CRP_MAX_ELVIRT 35                                    /* numElVirt (coarse-DoA el axis / grid height) */
#define CRP_MAX_BEAM_AZ 128                                  /* beamAz */
#define CRP_MAX_BEAM_EL 64                                   /* beamEl */
#define CRP_MAX_NVC_COARSE (CRP_MAX_AZVIRT * CRP_MAX_ELVIRT) /* coarse-DoA grid channels */
#define CRP_MAX_NVC CRP_MAX_AZVIRT                           /* projection channels = numAzVirt*numElVirt */
#define CRP_MAX_H 64
#define CRP_MAX_SVDIM 16
#define CRP_MAX_AZBINS 16
#define CRP_MAX_ELBINS 8
#define CRP_MAX_DET 1024
#define CRP_MAX_TX 8
#define CRP_MAX_RX 8
#define CRP_MAX_CHIRPS 512

/* Batched FFT-64 coarse DoA: 8-detection chunks packed side by side into one engine run (wide-tile
 * packing). The host passes it to FactorizeFFTSize as the digit reverse's groups per tile. */
#define CRP_FFT_GROUPS 2

/* VMEM buffer sizes, in element counts. */

/* Twiddle tables (complex fp32, interleaved re/im): 2 floats per entry. */
/* fp32 az twiddles. Only the KC=4 fp32 refine reads these, and that runs only on the
 * numElVirt==1 path, which is bounded to a 64x64 geometry -- a beamAz>64 or numAzVirt>64
 * configuration must use the numElVirt>1 path (int16 tables only). */
#define CRP_MAX_BEAM_AZ_FP32 64
#define CRP_MAX_AZVIRT_FP32 64
#define BUF_SZ_WAZ (CRP_MAX_BEAM_AZ_FP32 * CRP_MAX_AZVIRT_FP32 * 2)

/* Per-detection element snapshot (coarse DoA input): [nvc] complex fp32. */
#define BUF_SZ_ELEM (CRP_MAX_NVC_COARSE * 2)

/* Coarse-DoA az-DFT output, split re/im, ka-contiguous [numElVirt][beamAz] = nvc fp32 each. */
#define BUF_SZ_COARSE_TMP (CRP_MAX_ELVIRT * CRP_MAX_BEAM_AZ)

/* Transposed az twiddles [numAzVirt][beamAz], complex interleaved int16 (re, im), for the vectorized
 * coarse-DoA az DFT. */
#define BUF_SZ_WAZT (CRP_MAX_BEAM_AZ * CRP_MAX_AZVIRT) /* complex entries; the buffer holds 2x int16 */

/* Per-detection Doppler bin (detDopBuf): [maxDet] int32, one-shot load. */
#define BUF_SZ_DETDOP (CRP_MAX_DET)

#define BUF_SZ_PLANE ((CRP_MAX_CHIRPS + 1) * CRP_MAX_RX * 2)

/* Projection matrices, h-contiguous (snapshot index is the vector-lane axis) and split re/im so a
 * fixed channel index vector-loads H consecutive values. */
#define BUF_SZ_VDMAT (CRP_MAX_NVC * CRP_MAX_H)      /* per re/im: [nvc][H] */
#define BUF_SZ_PATCHMAT (CRP_MAX_SVDIM * CRP_MAX_H) /* per re/im: [<=svDim][H] (tmpRoi + patch) */

/* Snapshot-column matrix VD, re half then im half in one buffer, built on the device per detection. */
#define BUF_SZ_VDBUF (2 * BUF_SZ_VDMAT)

/* int16 Q15 az twiddles [beamAz][numAzVirt] complex. */
#define BUF_SZ_WAZ16 (CRP_MAX_BEAM_AZ * CRP_MAX_AZVIRT * 2)

/* int16 Q15 el twiddles [beamEl][numElVirt] complex (numElVirt>1). */
#define BUF_SZ_WEL16 (CRP_MAX_BEAM_EL * CRP_MAX_ELVIRT * 2)

/* Steering block for one ROI position: [svDim, numElBins, numAzBins] complex 2S32, copied DRAM -> DRAM. */
#define BUF_SZ_SVBLK (CRP_MAX_SVDIM * CRP_MAX_ELBINS * CRP_MAX_AZBINS * 2) /* 16*8*16*2 = 4096 int32 */

/* Per-snapshot scratch. */
#define BUF_SZ_CONJX (CRP_MAX_NVC_COARSE * 2) /* int16 units */

/* Detections per snapshotsROI output RDF batch. maxDet > this needs one RDF handle per batch --
 * transpose mode caps a single handle's stride. */
#define CRP_OUT_BATCH 256

#endif /* PVA_SOLUTIONS_PRIV_CAPON_ROI_PREPROCESS_PARAMS_H */
