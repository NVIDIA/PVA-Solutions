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

#ifndef COHERENT_INTEGRATION_HOST_H
#define COHERENT_INTEGRATION_HOST_H

#include "device/coherent_integration_param.h"

#define CI_MIN_RANGE_BINS (4)
#define CI_MAX_RANGE_BINS (4096)
#define CI_MAX_DOPPLER_BINS (4096)

#define CI_MIN_REPEAT_FOLD (4)
#define CI_MAX_REPEAT_FOLD (128)
#define CI_REPEAT_FOLD_STEP (4)
#define CI_MIN_FOLDED_WIDTH (32)
#define CI_FOLDED_WIDTH_STEP (32)
#define CI_MIN_DOPPLER_BINS (CI_MIN_REPEAT_FOLD * CI_MIN_FOLDED_WIDTH)

static inline uint32_t coherentIntegrationRawCellCapacity(int64_t numRx)
{
    if ((numRx != CI_RX4) && (numRx != CI_RX8) && (numRx != CI_RX12) && (numRx != CI_RX16))
    {
        return 0U;
    }

    const uint64_t rx              = (uint64_t)numRx;
    const uint64_t capacity        = CI_RAW_INPUT_BYTES / (CI_INPUT_COMPLEX_BYTES * rx);
    const uint64_t alignedCapacity = (capacity / CI_CELL_VECTOR_LANES) * CI_CELL_VECTOR_LANES;
    return (uint32_t)alignedCapacity;
}

static inline uint32_t coherentIntegrationFoldedWidthCapacity(int64_t numRx)
{
    const uint32_t rawCapacity = coherentIntegrationRawCellCapacity(numRx);
    if (rawCapacity == 0U)
    {
        return 0U;
    }

    const uint64_t limbCapacity = CI_LIMB_BYTES / (CI_LIMB_COMPLEX_BYTES * (uint64_t)numRx);
    const uint64_t capacity     = rawCapacity < limbCapacity ? rawCapacity : limbCapacity;
    return (uint32_t)((capacity / CI_CELL_VECTOR_LANES) * CI_CELL_VECTOR_LANES);
}

static inline int32_t coherentIntegrationIsSupportedGeometry(int64_t numRangeBins, int64_t numDopplerBins,
                                                             int64_t numRx, int64_t repeatFold)
{
    if ((numRangeBins < CI_MIN_RANGE_BINS) || (numRangeBins > CI_MAX_RANGE_BINS) ||
        (numDopplerBins < CI_MIN_DOPPLER_BINS) || (numDopplerBins > CI_MAX_DOPPLER_BINS) ||
        (coherentIntegrationFoldedWidthCapacity(numRx) == 0U) || (repeatFold < CI_MIN_REPEAT_FOLD) ||
        (repeatFold > CI_MAX_REPEAT_FOLD) || ((repeatFold % CI_REPEAT_FOLD_STEP) != 0) ||
        ((numDopplerBins % repeatFold) != 0))
    {
        return 0;
    }

    const int64_t foldedWidth = numDopplerBins / repeatFold;
    return (foldedWidth >= CI_MIN_FOLDED_WIDTH) && ((foldedWidth % CI_FOLDED_WIDTH_STEP) == 0) &&
           ((uint64_t)foldedWidth <= coherentIntegrationFoldedWidthCapacity(numRx));
}

#endif /* COHERENT_INTEGRATION_HOST_H */
