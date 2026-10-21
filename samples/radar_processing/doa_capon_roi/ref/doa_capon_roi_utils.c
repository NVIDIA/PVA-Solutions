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
 * @file doa_capon_roi_utils.c
 * @brief Host-side helpers for the Capon DoA on ROI sample / system test.
 */

#include "doa_capon_roi_utils.h"

#include <PvaAllocator.h>
#include <cupva_host.h>
#include <math.h>
#include <nvcv/alloc/Allocator.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RET_IF_NVCV_FAIL(call)  \
    do                          \
    {                           \
        NVCVStatus _s = (call); \
        if (_s != NVCV_SUCCESS) \
            return (int)_s;     \
    } while (0)

int caponLoadDatasetConfig(const char *inputsBinPath, PVADoaCaponROIParams *params, int32_t *roiNum,
                           int32_t *numSamples)
{
    FILE *fp = fopen(inputsBinPath, "rb");
    if (!fp)
    {
        fprintf(stderr, "  could not open %s\n", inputsBinPath);
        return -1;
    }
    const long roiFloats = (long)CAPON_ASSET_NUM_SAMPLES * CAPON_ASSET_SV_DIM * 2 +
                           (long)CAPON_ASSET_SV_DIM * CAPON_ASSET_NUM_EL_BINS * CAPON_ASSET_NUM_AZ_BINS * 2;
    float storedRoiNum = 0.0f;
    long fileFloats    = -1;
    if (fseek(fp, 0, SEEK_END) == 0)
        fileFloats = ftell(fp) / (long)sizeof(float);
    int ok = fileFloats > 1 && (fileFloats - 1) % roiFloats == 0 &&
             fseek(fp, (long)((fileFloats - 1) * (long)sizeof(float)), SEEK_SET) == 0 &&
             fread(&storedRoiNum, sizeof(float), 1, fp) == 1;
    fclose(fp);
    const long vRoiNum = ok ? (fileFloats - 1) / roiFloats : 0;
    if (!ok || vRoiNum <= 0 || (long)storedRoiNum != vRoiNum)
    {
        fprintf(stderr, "  ERROR: %s is not a Capon ROI inputs.bin (%ld floats, %ld per ROI + 1)\n", inputsBinPath,
                fileFloats, roiFloats);
        return -1;
    }

    params->numAzBins     = CAPON_ASSET_NUM_AZ_BINS;
    params->numElBins     = CAPON_ASSET_NUM_EL_BINS;
    params->svDim         = CAPON_ASSET_SV_DIM;
    params->inputQBits    = CAPON_ASSET_Q_BITS;
    params->steeringQBits = CAPON_ASSET_Q_BITS;

    if (roiNum)
        *roiNum = (int32_t)vRoiNum;
    if (numSamples)
        *numSamples = CAPON_ASSET_NUM_SAMPLES;
    return 0;
}

/* Quantize floats to a saturating signed Q format. */
static void quantizeFloatToQ(const float *src, int32_t *dst, size_t numElements, int32_t qBits)
{
    const float scale  = (float)((int64_t)1 << qBits);
    const float satMax = 2147483520.0f;
    const float satMin = -2147483648.0f;
    for (size_t i = 0; i < numElements; ++i)
    {
        float value = roundf(src[i] * scale);
        if (value > satMax)
            value = satMax;
        if (value < satMin)
            value = satMin;
        dst[i] = (int32_t)value;
    }
}

/* Quantize directly into the destination while repeating source ROIs as needed. */
static void quantizeRepeatedRois(const float *src, int32_t assetRoiNum, int32_t roiNum, size_t srcRoiStride,
                                 size_t srcOffset, size_t roiFloats, int32_t *dst, size_t dstRoiStride, int32_t qBits)
{
    for (int32_t roi = 0; roi < roiNum; ++roi)
    {
        int32_t srcRoi = roi % assetRoiNum;
        quantizeFloatToQ(src + (size_t)srcRoi * srcRoiStride + srcOffset, dst + (size_t)roi * dstRoiStride, roiFloats,
                         qBits);
    }
}

static void quantizeRepeatedSnapshots(const float *src, int32_t assetRoiNum, int32_t roiNum, int32_t storedSamples,
                                      int32_t tensorSamples, int32_t svDim, int32_t *dst, size_t dstRoiStride,
                                      int32_t qBits)
{
    size_t sampleFloats  = (size_t)svDim * 2;
    size_t srcRoiStride  = (size_t)storedSamples * sampleFloats;
    int32_t firstDoppler = tensorSamples <= storedSamples ? (storedSamples - tensorSamples) / 2 : 0;

    for (int32_t roi = 0; roi < roiNum; ++roi)
    {
        const float *srcRoi = src + (size_t)(roi % assetRoiNum) * srcRoiStride;
        int32_t *dstRoi     = dst + (size_t)roi * dstRoiStride;
        for (int32_t doppler = 0; doppler < tensorSamples; ++doppler)
        {
            int32_t srcDoppler = (firstDoppler + doppler) % storedSamples;
            quantizeFloatToQ(srcRoi + (size_t)srcDoppler * sampleFloats, dstRoi + (size_t)doppler * sampleFloats,
                             sampleFloats, qBits);
        }
    }
}

int caponROIAllocateTensors(CaponROITensors *t, const PVADoaCaponROIParams *params, int32_t roiNum)
{
    memset(t, 0, sizeof(*t));

    NVCVTensorLayout layoutNHW, layoutNCHW, layoutHW;
    RET_IF_NVCV_FAIL(nvcvTensorLayoutMake("NHW", &layoutNHW));
    RET_IF_NVCV_FAIL(nvcvTensorLayoutMake("NCHW", &layoutNCHW));
    RET_IF_NVCV_FAIL(nvcvTensorLayoutMake("HW", &layoutHW));

    RET_IF_NVCV_FAIL(nvcvAllocatorConstructPva(&t->allocatorHandle));

    /* NHW [roiNum, numDopplerBinsCovMat, svDim] 2S32. */
    int64_t shapeSnap[] = {(int64_t)roiNum, (int64_t)params->numDopplerBinsCovMat, (int64_t)params->svDim};
    RET_IF_NVCV_FAIL(
        nvcvTensorCalcRequirementsPva(3, shapeSnap, NVCV_DATA_TYPE_2S32, layoutNHW, 0, 0, &t->snapshotsReq));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->snapshotsReq, t->allocatorHandle, &t->snapshotsROITensor));

    /* NCHW [roiNum, svDim, numElBins, numAzBins] 2S32. */
    int64_t shapeSV[] = {(int64_t)roiNum, (int64_t)params->svDim, (int64_t)params->numElBins,
                         (int64_t)params->numAzBins};
    RET_IF_NVCV_FAIL(nvcvTensorCalcRequirementsPva(4, shapeSV, NVCV_DATA_TYPE_2S32, layoutNCHW, 0, 0, &t->svReq));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->svReq, t->allocatorHandle, &t->svTensor));

    /* HW [roiNum, numPeaksPerROI] F32, one reference and VPU tensor per output. */
    int64_t shapePeak[] = {(int64_t)roiNum, (int64_t)params->numPeaksPerROI};
    RET_IF_NVCV_FAIL(nvcvTensorCalcRequirementsPva(2, shapePeak, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, &t->peakPowerReq));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->peakPowerReq, t->allocatorHandle, &t->peakPowerTensor));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->peakPowerReq, t->allocatorHandle, &t->refPeakPowerTensor));

    RET_IF_NVCV_FAIL(nvcvTensorCalcRequirementsPva(2, shapePeak, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, &t->azBinReq));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->azBinReq, t->allocatorHandle, &t->azBinTensor));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->azBinReq, t->allocatorHandle, &t->refAzBinTensor));

    RET_IF_NVCV_FAIL(nvcvTensorCalcRequirementsPva(2, shapePeak, NVCV_DATA_TYPE_F32, layoutHW, 0, 0, &t->elBinReq));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->elBinReq, t->allocatorHandle, &t->elBinTensor));
    RET_IF_NVCV_FAIL(nvcvTensorConstruct(&t->elBinReq, t->allocatorHandle, &t->refElBinTensor));

    return 0;
}

void caponROIFreeTensors(CaponROITensors *t)
{
    if (!t)
        return;
#define DEC_REF(h)                     \
    do                                 \
    {                                  \
        if (h)                         \
        {                              \
            nvcvTensorDecRef(h, NULL); \
            h = NULL;                  \
        }                              \
    } while (0)
    DEC_REF(t->snapshotsROITensor);
    DEC_REF(t->svTensor);
    DEC_REF(t->peakPowerTensor);
    DEC_REF(t->azBinTensor);
    DEC_REF(t->elBinTensor);
    DEC_REF(t->refPeakPowerTensor);
    DEC_REF(t->refAzBinTensor);
    DEC_REF(t->refElBinTensor);
#undef DEC_REF
    if (t->allocatorHandle)
    {
        nvcvAllocatorDecRef(t->allocatorHandle, NULL);
        t->allocatorHandle = NULL;
    }
}

int caponLoadInputsFromFlatBin(const char *inputsBinPath, PVADoaCaponROIParams *params, int32_t numSamples,
                               int32_t assetRoiNum, int32_t roiNum, NVCVTensorHandle snapshotsHandle,
                               NVCVTensorHandle svHandle)
{
    FILE *fp = fopen(inputsBinPath, "rb");
    if (!fp)
    {
        fprintf(stderr, "  could not open %s\n", inputsBinPath);
        return -1;
    }

    int32_t svDim = (int32_t)params->svDim;
    int32_t svNum = (int32_t)params->numAzBins * (int32_t)params->numElBins;

    if (assetRoiNum <= 0 || roiNum <= 0 || numSamples <= 0)
    {
        fprintf(stderr, "  ERROR: ROI and Doppler dimensions must be positive\n");
        fclose(fp);
        return -1;
    }
    size_t assetSnapFloats = (size_t)assetRoiNum * numSamples * svDim * 2;
    size_t assetSvFloats   = (size_t)assetRoiNum * svDim * (size_t)svNum * 2;
    size_t numSvFloats     = (size_t)roiNum * svDim * (size_t)svNum * 2;

    NVCVTensorData data;
    int32_t *dstQ = NULL;

    /* Read complete file sections before repeating requested ROIs. */
    size_t scratchFloats = assetSnapFloats > assetSvFloats ? assetSnapFloats : assetSvFloats;
    float *scratch       = (float *)malloc(sizeof(float) * scratchFloats);
    if (!scratch)
    {
        fprintf(stderr, "  ERROR: out of memory for float scratch (%zu floats)\n", scratchFloats);
        fclose(fp);
        return -1;
    }

    if (fread(scratch, sizeof(float), assetSnapFloats, fp) != assetSnapFloats)
    {
        fprintf(stderr, "  ERROR: short read of snapshots in %s\n", inputsBinPath);
        free(scratch);
        fclose(fp);
        return -1;
    }

    nvcvTensorExportData(snapshotsHandle, &data);
    CupvaMemGetHostPointer((void **)&dstQ, (void *)data.buffer.strided.basePtr);
    int32_t tensorSamples      = (int32_t)data.shape[1];
    size_t snapDstRoiStride    = (size_t)data.buffer.strided.strides[0] / sizeof(*dstQ);
    size_t snapAssetRoiStride  = (size_t)numSamples * svDim * 2;
    int32_t scannedSamples     = tensorSamples < numSamples ? tensorSamples : numSamples;
    int32_t firstScannedSample = tensorSamples < numSamples ? (numSamples - tensorSamples) / 2 : 0;
    size_t scannedSnapFloats   = (size_t)scannedSamples * svDim * 2;
    size_t scannedSnapOffset   = (size_t)firstScannedSample * svDim * 2;
    size_t numSnapFloats       = (size_t)roiNum * tensorSamples * svDim * 2;
    quantizeRepeatedSnapshots(scratch, assetRoiNum, roiNum, numSamples, tensorSamples, svDim, dstQ, snapDstRoiStride,
                              (int32_t)params->inputQBits);

    /* Steering is stored contiguously in elevation-major order. */
    if (fread(scratch, sizeof(float), assetSvFloats, fp) != assetSvFloats)
    {
        fprintf(stderr, "  ERROR: short read of steering in %s\n", inputsBinPath);
        free(scratch);
        fclose(fp);
        return -1;
    }
    nvcvTensorExportData(svHandle, &data);
    CupvaMemGetHostPointer((void **)&dstQ, (void *)data.buffer.strided.basePtr);
    size_t svDstRoiStride = (size_t)data.buffer.strided.strides[0] / sizeof(*dstQ);
    size_t svRoiFloats    = (size_t)svDim * svNum * 2;
    quantizeRepeatedRois(scratch, assetRoiNum, roiNum, svRoiFloats, 0, svRoiFloats, dstQ, svDstRoiStride,
                         (int32_t)params->steeringQBits);
    free(scratch);

    fclose(fp);
    fprintf(stderr,
            "  loaded %d ROIs from %d stored rows%s: %zu snapshot floats (%d bins from %d stored%s), "
            "%zu steering floats (contiguous, elevation-major)\n",
            roiNum, assetRoiNum, roiNum > assetRoiNum ? " by cyclic repetition" : "", numSnapFloats, tensorSamples,
            numSamples, tensorSamples > numSamples ? " by cyclic repetition" : "", numSvFloats);
    return 0;
}

typedef enum
{
    COMPARE_REL_TOL,
    COMPARE_ABS_TOL
} CompareMode;

static int copyF32TensorRows(NVCVTensorHandle tensor, int32_t rows, int32_t cols, float *dst)
{
    NVCVTensorData data;
    if (nvcvTensorExportData(tensor, &data) != NVCV_SUCCESS)
        return -1;

    const float *src = NULL;
    CupvaMemGetHostPointer((void **)&src, (void *)data.buffer.strided.basePtr);
    if (src == NULL || data.buffer.strided.strides[0] < (int64_t)cols * (int64_t)sizeof(*src) ||
        data.buffer.strided.strides[0] % (int64_t)sizeof(*src) != 0)
        return -1;

    size_t srcRowStride = (size_t)data.buffer.strided.strides[0] / sizeof(*src);
    for (int32_t row = 0; row < rows; ++row)
        memcpy(dst + (size_t)row * cols, src + (size_t)row * srcRowStride, (size_t)cols * sizeof(*dst));
    return 0;
}

/* Matches the operator contract: valid peaks have positive power; NaN is also treated as invalid. */
static int isInvalidPeakPower(float power)
{
    return !(power > 0.f);
}

static int compareF32Buffers(const char *label, const float *got, const float *want, int32_t total, CompareMode mode,
                             float tol, bool wantNanIsDontCare)
{
    int32_t errors = 0;
    for (int32_t i = 0; i < total; i++)
    {
        int gotIsNan  = isnan(got[i]);
        int wantIsNan = isnan(want[i]);
        if (wantIsNan && wantNanIsDontCare)
            continue;
        if (gotIsNan && wantIsNan)
            continue;
        if (gotIsNan != wantIsNan)
        {
            if (errors < 10)
                printf("  %s NaN mismatch at [%d]: got=%f, want=%f\n", label, i, got[i], want[i]);
            errors++;
            continue;
        }
        float diff = fabsf(got[i] - want[i]);
        float err  = (mode == COMPARE_REL_TOL) ? (diff / (fabsf(want[i]) + 1e-10f)) : diff;
        if (err > tol)
        {
            if (errors < 10)
                printf("  %s mismatch at [%d]: got=%f, want=%f, %s=%f\n", label, i, got[i], want[i],
                       mode == COMPARE_REL_TOL ? "relErr" : "absErr", err);
            errors++;
        }
    }
    return errors;
}

int caponCompareVpuVsRef(NVCVTensorHandle vpuPower, NVCVTensorHandle refPower, NVCVTensorHandle vpuAz,
                         NVCVTensorHandle refAz, NVCVTensorHandle vpuEl, NVCVTensorHandle refEl, int32_t roiNum,
                         int32_t numPeaksPerROI, float powerRelTol, float binAbsTol, bool enableSubBinInterp)
{
    int32_t total  = numPeaksPerROI * roiNum;
    float *buffers = (float *)malloc((size_t)total * 6 * sizeof(*buffers));
    if (buffers == NULL)
    {
        fprintf(stderr, "  ERROR: out of memory while comparing Capon ROI outputs\n");
        return 1;
    }

    float *vpuPowerP = buffers;
    float *refPowerP = vpuPowerP + total;
    float *vpuAzP    = refPowerP + total;
    float *refAzP    = vpuAzP + total;
    float *vpuElP    = refAzP + total;
    float *refElP    = vpuElP + total;
    if (copyF32TensorRows(vpuPower, roiNum, numPeaksPerROI, vpuPowerP) != 0 ||
        copyF32TensorRows(refPower, roiNum, numPeaksPerROI, refPowerP) != 0 ||
        copyF32TensorRows(vpuAz, roiNum, numPeaksPerROI, vpuAzP) != 0 ||
        copyF32TensorRows(refAz, roiNum, numPeaksPerROI, refAzP) != 0 ||
        copyF32TensorRows(vpuEl, roiNum, numPeaksPerROI, vpuElP) != 0 ||
        copyF32TensorRows(refEl, roiNum, numPeaksPerROI, refElP) != 0)
    {
        fprintf(stderr, "  ERROR: failed to read Capon ROI output tensors\n");
        free(buffers);
        return 1;
    }

    int32_t errors = 0;
    errors += compareF32Buffers("Power", vpuPowerP, refPowerP, total, COMPARE_REL_TOL, powerRelTol, false);
    /* Bin values are undefined when either result invalidates the peak. */
    for (int32_t i = 0; i < total; i++)
    {
        int eitherIsInvalid = isInvalidPeakPower(refPowerP[i]) || isInvalidPeakPower(vpuPowerP[i]);
        if (eitherIsInvalid)
        {
            refAzP[i] = NAN;
            refElP[i] = NAN;
            vpuAzP[i] = NAN;
            vpuElP[i] = NAN;
        }
    }
    const float effBinTol = enableSubBinInterp ? binAbsTol : 0.f;
    errors += compareF32Buffers("AzBin", vpuAzP, refAzP, total, COMPARE_ABS_TOL, effBinTol, true);
    errors += compareF32Buffers("ElBin", vpuElP, refElP, total, COMPARE_ABS_TOL, effBinTol, true);

    if (errors > 0)
        printf("FAIL: %d VPU-vs-reference errors out of %d (3 fields × %d ROIs × %d peaks)\n", errors, 3 * total,
               roiNum, numPeaksPerROI);
    else
        printf("PASS: VPU matches reference (%d ROIs × %d peaks × 3 fields)\n", roiNum, numPeaksPerROI);

    free(buffers);
    return errors;
}
