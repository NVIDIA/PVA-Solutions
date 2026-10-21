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

#include "coherent_integration_ref.h"

#include <coherent_integration_host.h>
#include <cupva_host_scheduling.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int32_t kMaxRank = 4;

/*
 * Take one input headroom bit, then split each rounded S32 component into exact
 * signed-16 high and residual limbs. The completed high/residual complex dots
 * remain signed-48 and are rounded once after Rx accumulation. Magnitude and
 * fold accumulation use FP32.
 */
constexpr int32_t kSigned48InputShift         = 1;
constexpr int32_t kFixedPowerMaxInputExponent = 15;
constexpr int32_t kFixedPowerSquareShift      = 16;
constexpr int32_t kSteeringQBits              = 15;

/* Keeps the rounded S32 magnitude within the int32_t range used by ceilLog2(). */
static_assert(kSigned48InputShift >= 1);

/*
 * Rounds as floor(value / 2^shift + 0.5). Division and remainder avoid relying
 * on the implementation-defined right shift of a negative signed value in
 * C++17.
 */
constexpr int64_t roundShiftHalfUp(int64_t value, int32_t shift)
{
    if (shift <= 0)
    {
        return value;
    }

    const int64_t divisor = int64_t{1} << shift;
    int64_t quotient      = value / divisor;
    int64_t remainder     = value % divisor;
    if (remainder < 0)
    {
        quotient--;
        remainder += divisor;
    }
    return quotient + ((remainder >= (divisor / 2)) ? 1 : 0);
}

/* Portable arithmetic right shift: floor(value / 2^shift). */
constexpr int64_t floorShift(int64_t value, int32_t shift)
{
    if (shift <= 0)
    {
        return value;
    }

    const int64_t divisor = int64_t{1} << shift;
    int64_t quotient      = value / divisor;
    if ((value % divisor) < 0)
    {
        quotient--;
    }
    return quotient;
}

/*
 * Smallest float strictly greater than INT32_MAX. INT32_MAX itself is not
 * representable in FP32, so this is the correct saturation comparison point.
 */
constexpr float kInt32MaxExclusive = 2147483648.0f;

/*
 * Element addressing resolved from the exported tensor strides, so the
 * reference never assumes a packed buffer.
 */
struct TensorView
{
    uint8_t *base            = nullptr;
    int64_t shape[kMaxRank]  = {0, 0, 0, 0};
    int64_t stride[kMaxRank] = {0, 0, 0, 0};
    int32_t rank             = 0;
};

int32_t mapTensor(NVCVTensorHandle handle, NVCVDataType expectedType, TensorView *view)
{
    if ((handle == nullptr) || (view == nullptr))
    {
        return -1;
    }

    NVCVTensorData data;
    if (0 != nvcvTensorExportData(handle, &data))
    {
        return -1;
    }

    if ((data.rank < 1) || (data.rank > kMaxRank))
    {
        return -1;
    }

    NVCVDataType dataType;
    if (0 != nvcvTensorGetDataType(handle, &dataType))
    {
        return -1;
    }
    if (dataType != expectedType)
    {
        return -1;
    }

    uint8_t *host = nullptr;
    if (CUPVA_ERROR_NONE != CupvaMemGetHostPointer((void **)&host, (void *)data.buffer.strided.basePtr))
    {
        return -1;
    }

    view->base = host;
    view->rank = data.rank;
    for (int32_t i = 0; i < data.rank; i++)
    {
        view->shape[i]  = data.shape[i];
        view->stride[i] = data.buffer.strided.strides[i];
    }
    return 0;
}

/* Collapses a leading unit batch dimension so NHCW and HCW share one code path. */
int32_t dropUnitBatchDim(TensorView *view)
{
    if (view->rank != kMaxRank)
    {
        return 0;
    }
    if (view->shape[0] != 1)
    {
        return -1;
    }
    for (int32_t i = 0; i < kMaxRank - 1; i++)
    {
        view->shape[i]  = view->shape[i + 1];
        view->stride[i] = view->stride[i + 1];
    }
    view->rank = kMaxRank - 1;
    return 0;
}

template<typename T>
inline T *elementAt(const TensorView &view, int64_t i0, int64_t i1)
{
    return reinterpret_cast<T *>(view.base + (i0 * view.stride[0]) + (i1 * view.stride[1]));
}

template<typename T>
inline T *elementAt(const TensorView &view, int64_t i0, int64_t i1, int64_t i2)
{
    return reinterpret_cast<T *>(view.base + (i0 * view.stride[0]) + (i1 * view.stride[1]) + (i2 * view.stride[2]));
}

template<typename T>
inline T *elementAt(const TensorView &view, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
{
    return reinterpret_cast<T *>(view.base + (i0 * view.stride[0]) + (i1 * view.stride[1]) + (i2 * view.stride[2]) +
                                 (i3 * view.stride[3]));
}

int32_t ceilLog2(int32_t value)
{
    int32_t result = 0;
    while ((int64_t{1} << result) < value)
    {
        result++;
    }
    return result;
}

int32_t computeOutputShifts(int32_t numRx, int32_t repeatFold, int32_t *coherentIntegrationRxShift,
                            int32_t *coherentIntegrationFinalShift)
{
    if ((coherentIntegrationFoldedWidthCapacity(numRx) == 0U) || (repeatFold < CI_MIN_REPEAT_FOLD) ||
        (repeatFold > CI_MAX_REPEAT_FOLD) || ((repeatFold % CI_REPEAT_FOLD_STEP) != 0) ||
        (coherentIntegrationRxShift == nullptr) || (coherentIntegrationFinalShift == nullptr))
    {
        return -1;
    }

    *coherentIntegrationRxShift    = ceilLog2(numRx) + 1;
    *coherentIntegrationFinalShift = *coherentIntegrationRxShift + ceilLog2(repeatFold);
    return 0;
}

inline uint32_t encodeMap(float value, float inverseShiftScale)
{
    const float scaled = std::trunc(value * inverseShiftScale);
    if (!(scaled > 0.0f))
    {
        return 0u;
    }
    if (scaled >= kInt32MaxExclusive)
    {
        return static_cast<uint32_t>(INT32_MAX);
    }
    return static_cast<uint32_t>(scaled);
}

/*
 * Resolves the range-Doppler geometry and produces the per-cell Rx gather.
 * numRx is the C axis for RANGE_RX_DOPPLER and the W axis for RANGE_DOPPLER_RX.
 */
struct RangeDopplerGeometry
{
    int32_t numRangeBins   = 0;
    int32_t numRx          = 0;
    int32_t numDopplerBins = 0;
};

int32_t resolveGeometry(const TensorView &rdm, PVAFFTLayout inputLayout, RangeDopplerGeometry *geometry)
{
    if ((rdm.rank != 3) || (rdm.shape[0] <= 0) || (rdm.shape[0] > INT32_MAX) || (rdm.shape[1] <= 0) ||
        (rdm.shape[1] > INT32_MAX) || (rdm.shape[2] <= 0) || (rdm.shape[2] > INT32_MAX))
    {
        return -1;
    }

    geometry->numRangeBins = static_cast<int32_t>(rdm.shape[0]);
    if (inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        geometry->numDopplerBins = static_cast<int32_t>(rdm.shape[1]);
        geometry->numRx          = static_cast<int32_t>(rdm.shape[2]);
    }
    else if (inputLayout == PVA_FFT_LAYOUT_RANGE_RX_DOPPLER)
    {
        geometry->numRx          = static_cast<int32_t>(rdm.shape[1]);
        geometry->numDopplerBins = static_cast<int32_t>(rdm.shape[2]);
    }
    else
    {
        return -1;
    }
    return 0;
}

inline const int32_t *rangeDopplerSampleAt(const TensorView &rdm, PVAFFTLayout inputLayout, int32_t rangeBin,
                                           int32_t rxChannel, int32_t dopplerBin)
{
    if (inputLayout == PVA_FFT_LAYOUT_RANGE_DOPPLER_RX)
    {
        return elementAt<const int32_t>(rdm, rangeBin, dopplerBin, rxChannel);
    }
    return elementAt<const int32_t>(rdm, rangeBin, rxChannel, dopplerBin);
}

struct FixedPowerScratch
{
    explicit FixedPowerScratch(int32_t numRx)
        : roundedReal(static_cast<size_t>(numRx))
        , roundedImag(static_cast<size_t>(numRx))
        , highReal(static_cast<size_t>(numRx))
        , highImag(static_cast<size_t>(numRx))
        , lowReal(static_cast<size_t>(numRx))
        , lowImag(static_cast<size_t>(numRx))
    {
    }

    std::vector<int64_t> roundedReal;
    std::vector<int64_t> roundedImag;
    std::vector<int16_t> highReal;
    std::vector<int16_t> highImag;
    std::vector<int16_t> lowReal;
    std::vector<int16_t> lowImag;
};

void prepareFixedPowerCell(const int32_t *cellReal, const int32_t *cellImag, int32_t numRx, int32_t termShift,
                           FixedPowerScratch &scratch, int32_t &fixedPowerShift, int32_t &representationShift)
{
    int64_t maxInputMagnitude = 0;
    for (int32_t rx = 0; rx < numRx; rx++)
    {
        scratch.roundedReal[rx] = roundShiftHalfUp(cellReal[rx], kSigned48InputShift);
        scratch.roundedImag[rx] = roundShiftHalfUp(cellImag[rx], kSigned48InputShift);

        const int64_t absReal = (scratch.roundedReal[rx] < 0) ? -scratch.roundedReal[rx] : scratch.roundedReal[rx];
        const int64_t absImag = (scratch.roundedImag[rx] < 0) ? -scratch.roundedImag[rx] : scratch.roundedImag[rx];
        maxInputMagnitude     = std::max(maxInputMagnitude, std::max(absReal, absImag));
    }

    /* One exponent is shared by every Rx component and beam for this cell. */
    fixedPowerShift = std::max(0, ceilLog2(static_cast<int32_t>(maxInputMagnitude)) - kFixedPowerMaxInputExponent);
    const int32_t limbShift = fixedPowerShift + 1;
    representationShift     = kSigned48InputShift + fixedPowerShift + termShift;

    const int64_t limbScale = int64_t{1} << limbShift;
    for (int32_t rx = 0; rx < numRx; rx++)
    {
        const int64_t highRealValue = roundShiftHalfUp(scratch.roundedReal[rx], limbShift);
        const int64_t highImagValue = roundShiftHalfUp(scratch.roundedImag[rx], limbShift);
        const int64_t lowRealValue  = scratch.roundedReal[rx] - (highRealValue * limbScale);
        const int64_t lowImagValue  = scratch.roundedImag[rx] - (highImagValue * limbScale);
        scratch.highReal[rx]        = static_cast<int16_t>(highRealValue);
        scratch.highImag[rx]        = static_cast<int16_t>(highImagValue);
        scratch.lowReal[rx]         = static_cast<int16_t>(lowRealValue);
        scratch.lowImag[rx]         = static_cast<int16_t>(lowImagValue);
    }
}

int64_t reconstructCompletedDot(int64_t highDot, int64_t lowDot, int32_t fixedPowerShift, int32_t termShift)
{
    const int32_t limbShift = fixedPowerShift + 1;
    if (termShift == 0)
    {
        return (2 * highDot) + roundShiftHalfUp(lowDot, fixedPowerShift);
    }
    if (termShift == 1)
    {
        return highDot + roundShiftHalfUp(lowDot, limbShift);
    }

    /* floorShift is required for a negative residual before the final half-up round. */
    const int64_t combined = highDot + floorShift(lowDot, limbShift);
    return roundShiftHalfUp(combined, termShift - 1);
}

/*
 * The fixed-point reference rounds one S32 cell, splits it exactly into
 * high/residual S16 limbs, evaluates direct Q15 complex dots, and compares
 * fixed-power beams. One square root and one scale restoration are applied
 * after the maximum.
 */
float computeFixedPowerPeakMagnitude(const int32_t *cellReal, const int32_t *cellImag, const int16_t *steeringReal,
                                     const int16_t *steeringImag, int32_t numAzimuthBins, int32_t numElevationBins,
                                     int32_t numRx, FixedPowerScratch &scratch)
{
    const int32_t termShift     = ceilLog2(numRx);
    int32_t fixedPowerShift     = 0;
    int32_t representationShift = 0;
    prepareFixedPowerCell(cellReal, cellImag, numRx, termShift, scratch, fixedPowerShift, representationShift);

    const auto computeBeamPower = [&](size_t angleIndex) -> int64_t {
        const int16_t *beamReal = &steeringReal[angleIndex * numRx];
        const int16_t *beamImag = &steeringImag[angleIndex * numRx];

        int64_t highDotReal = 0;
        int64_t highDotImag = 0;
        int64_t lowDotReal  = 0;
        int64_t lowDotImag  = 0;
        for (int32_t rx = 0; rx < numRx; rx++)
        {
            /* The supplied steering value is the direct factor; do not conjugate it. */
            const int64_t steerReal = beamReal[rx];
            const int64_t steerImag = beamImag[rx];

            highDotReal += (static_cast<int64_t>(scratch.highReal[rx]) * steerReal) -
                           (static_cast<int64_t>(scratch.highImag[rx]) * steerImag);
            highDotImag += (static_cast<int64_t>(scratch.highReal[rx]) * steerImag) +
                           (static_cast<int64_t>(scratch.highImag[rx]) * steerReal);
            lowDotReal += (static_cast<int64_t>(scratch.lowReal[rx]) * steerReal) -
                          (static_cast<int64_t>(scratch.lowImag[rx]) * steerImag);
            lowDotImag += (static_cast<int64_t>(scratch.lowReal[rx]) * steerImag) +
                          (static_cast<int64_t>(scratch.lowImag[rx]) * steerReal);
        }

        const int64_t accumRealQ15 = reconstructCompletedDot(highDotReal, lowDotReal, fixedPowerShift, termShift);
        const int64_t accumImagQ15 = reconstructCompletedDot(highDotImag, lowDotImag, fixedPowerShift, termShift);

        const int64_t realPower = (accumRealQ15 * accumRealQ15) / (int64_t{1} << kFixedPowerSquareShift);
        const int64_t imagPower = (accumImagQ15 * accumImagQ15) / (int64_t{1} << kFixedPowerSquareShift);
        return realPower + imagPower;
    };

    int64_t peakPower         = 0;
    const size_t numAngleBins = static_cast<size_t>(numAzimuthBins) * numElevationBins;
    for (size_t angleIndex = 0; angleIndex < numAngleBins; angleIndex++)
    {
        const int64_t power = computeBeamPower(angleIndex);
        if (power > peakPower)
        {
            peakPower = power;
        }
    }

    const float magnitude           = std::sqrt(static_cast<float>(peakPower));
    const int32_t magnitudeExponent = representationShift - kSteeringQBits + (kFixedPowerSquareShift / 2);
    return std::ldexp(magnitude, magnitudeExponent);
}

/* Fold FP32 magnitudes, apply output scaling, and truncate to U32. */
int32_t writeOutputs(const std::vector<float> &rxIntegratedMagnitude, const TensorView &coherentIntegrationRx,
                     const TensorView &coherentIntegrationFinal, int32_t numRangeBins, int32_t numDopplerBins,
                     int32_t numFoldedBins, int32_t repeatFold, int32_t coherentIntegrationRxShift,
                     int32_t coherentIntegrationFinalShift)
{
    const float inverseCoherentIntegrationRxScale    = std::ldexp(1.0f, -coherentIntegrationRxShift);
    const float inverseCoherentIntegrationFinalScale = std::ldexp(1.0f, -coherentIntegrationFinalShift);

    for (int32_t rangeBin = 0; rangeBin < numRangeBins; rangeBin++)
    {
        for (int32_t dopplerBin = 0; dopplerBin < numDopplerBins; dopplerBin++)
        {
            const float value = rxIntegratedMagnitude[static_cast<size_t>(rangeBin) * numDopplerBins + dopplerBin];
            *elementAt<uint32_t>(coherentIntegrationRx, rangeBin, dopplerBin) =
                encodeMap(value, inverseCoherentIntegrationRxScale);
        }

        /* final[range][folded] = sum_k rxMagnitude[range][folded + k * numFoldedBins]. */
        for (int32_t foldedBin = 0; foldedBin < numFoldedBins; foldedBin++)
        {
            float sum = 0.0f;
            for (int32_t fold = 0; fold < repeatFold; fold++)
            {
                const int32_t dopplerBin = foldedBin + (fold * numFoldedBins);
                sum += rxIntegratedMagnitude[static_cast<size_t>(rangeBin) * numDopplerBins + dopplerBin];
            }

            *elementAt<uint32_t>(coherentIntegrationFinal, rangeBin, foldedBin) =
                encodeMap(sum, inverseCoherentIntegrationFinalScale);
        }
    }
    return 0;
}

/*
 * Shared tensor validation and target-arithmetic implementation.
 */
int32_t coherentIntegrationTargetImpl(NVCVTensorHandle rangeDopplerMap, NVCVTensorHandle steeringVectors,
                                      NVCVTensorHandle coherentIntegrationRx, NVCVTensorHandle coherentIntegrationFinal,
                                      int32_t repeatFold, PVAFFTLayout inputLayout)
{
    TensorView rdm;
    TensorView steering;
    TensorView coherentIntegrationRxView;
    TensorView coherentIntegrationFinalView;

    if (0 != mapTensor(rangeDopplerMap, NVCV_DATA_TYPE_2S32, &rdm))
    {
        return -1;
    }
    if (0 != dropUnitBatchDim(&rdm))
    {
        return -1;
    }
    if (0 != mapTensor(steeringVectors, NVCV_DATA_TYPE_2S16, &steering))
    {
        return -1;
    }
    if (0 != mapTensor(coherentIntegrationRx, NVCV_DATA_TYPE_U32, &coherentIntegrationRxView))
    {
        return -1;
    }
    if (0 != mapTensor(coherentIntegrationFinal, NVCV_DATA_TYPE_U32, &coherentIntegrationFinalView))
    {
        return -1;
    }

    RangeDopplerGeometry geometry;
    if (0 != resolveGeometry(rdm, inputLayout, &geometry))
    {
        return -1;
    }

    const int32_t numRangeBins   = geometry.numRangeBins;
    const int32_t numRx          = geometry.numRx;
    const int32_t numDopplerBins = geometry.numDopplerBins;

    if (!coherentIntegrationIsSupportedGeometry(geometry.numRangeBins, geometry.numDopplerBins, geometry.numRx,
                                                repeatFold))
    {
        return -1;
    }
    const int32_t numFoldedBins = numDopplerBins / repeatFold;

    if ((steering.rank != 3) || (steering.shape[0] <= 0) || (steering.shape[1] <= 0) || (steering.shape[2] != numRx) ||
        (steering.shape[0] > INT32_MAX) || (steering.shape[1] > INT32_MAX))
    {
        return -1;
    }
    const int32_t numAzimuthBins   = static_cast<int32_t>(steering.shape[0]);
    const int32_t numElevationBins = static_cast<int32_t>(steering.shape[1]);
    uint64_t const bytesPerAzimuth = static_cast<uint64_t>(numElevationBins) * numRx * sizeof(int16_t) * 2U;
    if (static_cast<uint64_t>(numAzimuthBins) > (CI_STEERING_CAPACITY_BYTES / bytesPerAzimuth))
    {
        return -1;
    }
    const size_t numAngleBins = static_cast<size_t>(numAzimuthBins) * numElevationBins;

    if ((coherentIntegrationRxView.rank != 2) || (coherentIntegrationRxView.shape[0] != numRangeBins) ||
        (coherentIntegrationRxView.shape[1] != numDopplerBins))
    {
        return -1;
    }
    if ((coherentIntegrationFinalView.rank != 2) || (coherentIntegrationFinalView.shape[0] != numRangeBins) ||
        (coherentIntegrationFinalView.shape[1] != numFoldedBins))
    {
        return -1;
    }

    int32_t coherentIntegrationRxShift    = 0;
    int32_t coherentIntegrationFinalShift = 0;
    if (0 != computeOutputShifts(numRx, repeatFold, &coherentIntegrationRxShift, &coherentIntegrationFinalShift))
    {
        return -1;
    }

    /* Copy the potentially strided Q15 tensor once; every cell reuses it. */
    std::vector<int16_t> steeringReal(numAngleBins * numRx);
    std::vector<int16_t> steeringImag(numAngleBins * numRx);
    for (int32_t azimuthBin = 0; azimuthBin < numAzimuthBins; azimuthBin++)
    {
        for (int32_t elevationBin = 0; elevationBin < numElevationBins; elevationBin++)
        {
            const size_t angleIndex = static_cast<size_t>(azimuthBin) * numElevationBins + elevationBin;
            for (int32_t rx = 0; rx < numRx; rx++)
            {
                const int16_t *sample = elementAt<const int16_t>(steering, azimuthBin, elevationBin, rx);
                steeringReal[angleIndex * numRx + rx] = sample[0];
                steeringImag[angleIndex * numRx + rx] = sample[1];
            }
        }
    }

    FixedPowerScratch fixedPowerScratch(numRx);
    std::vector<float> rxIntegratedMagnitude(static_cast<size_t>(numRangeBins) * numDopplerBins, 0.0f);
    std::vector<int32_t> cellReal(numRx, 0);
    std::vector<int32_t> cellImag(numRx, 0);

    for (int32_t rangeBin = 0; rangeBin < numRangeBins; rangeBin++)
    {
        for (int32_t dopplerBin = 0; dopplerBin < numDopplerBins; dopplerBin++)
        {
            for (int32_t rx = 0; rx < numRx; rx++)
            {
                const int32_t *sample = rangeDopplerSampleAt(rdm, inputLayout, rangeBin, rx, dopplerBin);
                cellReal[rx]          = sample[0];
                cellImag[rx]          = sample[1];
            }

            const float peak = computeFixedPowerPeakMagnitude(cellReal.data(), cellImag.data(), steeringReal.data(),
                                                              steeringImag.data(), numAzimuthBins, numElevationBins,
                                                              numRx, fixedPowerScratch);
            rxIntegratedMagnitude[static_cast<size_t>(rangeBin) * numDopplerBins + dopplerBin] = peak;
        }
    }

    return writeOutputs(rxIntegratedMagnitude, coherentIntegrationRxView, coherentIntegrationFinalView, numRangeBins,
                        numDopplerBins, numFoldedBins, repeatFold, coherentIntegrationRxShift,
                        coherentIntegrationFinalShift);
}

} // namespace

int32_t coherentIntegrationTargetRef(NVCVTensorHandle rangeDopplerMap, NVCVTensorHandle steeringVectors,
                                     NVCVTensorHandle coherentIntegrationRx, NVCVTensorHandle coherentIntegrationFinal,
                                     int32_t repeatFold, PVAFFTLayout inputLayout)
{
    return coherentIntegrationTargetImpl(rangeDopplerMap, steeringVectors, coherentIntegrationRx,
                                         coherentIntegrationFinal, repeatFold, inputLayout);
}

int32_t checkCoherentIntegrationOutput(NVCVTensorHandle refTensorHandle, NVCVTensorHandle pvaTensorHandle,
                                       uint32_t tolerance)
{
    TensorView ref;
    TensorView cmp;

    if (0 != mapTensor(refTensorHandle, NVCV_DATA_TYPE_U32, &ref))
    {
        return 1;
    }
    if (0 != mapTensor(pvaTensorHandle, NVCV_DATA_TYPE_U32, &cmp))
    {
        return 1;
    }
    if ((ref.rank != 2) || (cmp.rank != 2) || (ref.shape[0] != cmp.shape[0]) || (ref.shape[1] != cmp.shape[1]))
    {
        printf("Unsupported data layout for comparison\n");
        return 1;
    }

    uint32_t mismatchCount = 0;
    for (int64_t row = 0; row < ref.shape[0]; row++)
    {
        for (int64_t col = 0; col < ref.shape[1]; col++)
        {
            const uint32_t refData = *elementAt<const uint32_t>(ref, row, col);
            const uint32_t cmpData = *elementAt<const uint32_t>(cmp, row, col);
            const uint32_t diff    = (refData > cmpData) ? (refData - cmpData) : (cmpData - refData);
            if (diff > tolerance)
            {
                mismatchCount++;
                if (mismatchCount < 10)
                {
                    printf("Mismatch[%ld][%ld]: ref = %u, cmp = %u\n", (long)row, (long)col, refData, cmpData);
                }
            }
        }
    }

    if (mismatchCount >= 10)
    {
        printf("Mismatch count: %u\n", mismatchCount);
    }
    return (mismatchCount == 0) ? 0 : 1;
}
