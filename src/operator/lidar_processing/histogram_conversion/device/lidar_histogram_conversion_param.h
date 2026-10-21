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

#ifndef LIDAR_HISTOGRAM_CONVERSION_PARAM_H
#define LIDAR_HISTOGRAM_CONVERSION_PARAM_H

#include <stdint.h>

/**
 * @brief LiDAR data packing type (device-side definition)
 */
typedef enum
{
    LIDAR_DATA_PACKING_TYPE_NONE  = 0,
    LIDAR_DATA_PACKING_TYPE_RAW12 = 1
} LidarDataPackingType_Dev;

// space reserved for pixel and histogram headers is arbitrary and should be fine-tuned considering the sensor specs
#define MAX_PIXELS_PER_TILE (64)
#define MAX_HISTOGRAMS_PER_PIXEL (8)
#define MAX_HISTOGRAMS_PER_TILE (MAX_HISTOGRAMS_PER_PIXEL * MAX_PIXELS_PER_TILE)
#define MAX_PEAKS_PER_HISTOGRAM (8)
#define MAX_PIXEL_HEADER_SIZE (64)
#define MAX_HISTOGRAM_HEADER_SIZE (16)
#define MAX_HISTOGRAM_BINS (2048)
// Largest bin value the smoothing filter supports: it sums up to 1024 times a bin value in a signed 24-bit
// accumulator.
#define MAX_HISTOGRAM_BIN_VALUE (8191)

// Tile layout in VMEM is always Nx1 pixels. N is determined on host side depending on the pixel size.
// Here we tune in the tile buffer sizes with a target of utilizing one full superbank for the double-buffered input data.
// As a first dimension atom size, we use the largest possible single histogram in bytes per above definitions.
#define MAX_HISTOGRAM_SIZE (MAX_PIXEL_HEADER_SIZE + MAX_HISTOGRAM_HEADER_SIZE + MAX_HISTOGRAM_BINS)
#define MAX_TILE_BUFFER_SIZE_BYTES (1024 * 64) // 1/2 of a 128kB superbank
#define MIN_MAX_HISTOGRAMS_PER_TILE (MAX_TILE_BUFFER_SIZE_BYTES / (MAX_HISTOGRAM_SIZE * 2))

// sanity checks: make sure the tile buffer size selected is not larger than min, but also not so small that it wastes
// significant space compared to the target size.
#define ACTUAL_TILE_SIZE_BYTES (MAX_HISTOGRAM_SIZE * 2 * MIN_MAX_HISTOGRAMS_PER_TILE)
#if ACTUAL_TILE_SIZE_BYTES > MAX_TILE_BUFFER_SIZE_BYTES
#    error "ACTUAL_TILE_SIZE_BYTES > MAX_TILE_BUFFER_SIZE_BYTES"
#endif
#if (MAX_TILE_BUFFER_SIZE_BYTES - ACTUAL_TILE_SIZE_BYTES) > 2048
#    error "ACTUAL_TILE_SIZE_BYTES << MAX_TILE_BUFFER_SIZE_BYTES"
#endif

// Derive some more constants to check that extreme cases will not violate size constraints.
#define MAX_PIXEL_SIZE \
    (MAX_PIXEL_HEADER_SIZE + (MAX_HISTOGRAM_HEADER_SIZE + MAX_HISTOGRAM_BINS) * MAX_HISTOGRAMS_PER_PIXEL)
#define MIN_MAX_PIXELS_PER_TILE (MAX_TILE_BUFFER_SIZE_BYTES / (MAX_PIXEL_SIZE * 2))
// sanity check: must be able to hold at least one pixel per tile when the pixel size is as large as possible
#if MIN_MAX_PIXELS_PER_TILE < 1
#    error "MIN_MAX_PIXELS_PER_TILE < 1"
#endif

// Largest number of max-sized histograms that would fit in the input tile buffer defines how large the intermediate
// histogram buffer (padded and expanded to 16-bits) must be.
#define MAX_MAX_HISTOGRAMS_PER_TILE (MAX_TILE_BUFFER_SIZE_BYTES / ((MAX_HISTOGRAM_BINS * 3) / 2))

// sanity check: must be able to fit at least one pixel per tile, in the worst-case.
#if MAX_MAX_HISTOGRAMS_PER_TILE < MAX_HISTOGRAMS_PER_PIXEL
#    error "MAX_MAX_HISTOGRAMS_PER_TILE < MAX_HISTOGRAMS_PER_PIXEL"
#endif
// sanity check: we are being too conservative with the max settings if we can fit more max-sized histograms than allowed.
#if MAX_MAX_HISTOGRAMS_PER_TILE > MAX_HISTOGRAMS_PER_TILE
#    error "MAX_MAX_HISTOGRAMS_PER_TILE > MAX_HISTOGRAMS_PER_TILE"
#endif

// Unpacked histogram buffer geometry. The device unpack routines write their output with this layout, and the host
// derives numHistPixPerTile from the same functions so that a tile never unpacks past the end of the buffer.
//
// Each unpacked histogram row is numBinsPerHist bins plus a reflected halo on both sides (for the 7-tap smoothing),
// padded to a whole number of vshort vectors.
#define UNPACKED_HISTOGRAM_HALO (3)
#define UNPACKED_HISTOGRAM_PITCH_ALIGN (16) // pva_elementsof(vshort)

// RAW12 unpacking decodes a whole pixel into scratch space that follows the unpacked rows. One vshortx load holds 16
// packed element pairs and expands to one dvushort of 32 elements. Loads start on a 12-byte boundary (4-byte aligned
// and a whole number of 3-byte pairs), so up to 3 pairs ahead of the pixel are decoded as well;
// RAW12_UNPACK_ALIGNMENT_PAIRS reserves 6. The vector count is rounded up to the loop unroll factor.
#define RAW12_UNPACK_PAIRS_PER_VECTOR (16)    // pva_elementsof(vshortx)
#define RAW12_UNPACK_ELEMENTS_PER_VECTOR (32) // pva_elementsof(dvushort)
#define RAW12_UNPACK_ALIGNMENT_PAIRS (6)
#define RAW12_UNPACK_UNROLL (4)

// Upper bound, in elements, on how far the NONE unpack writes past the last unpacked row: the bin copy stores whole
// dvshort vectors (32 elements) starting after the leading halo, and the halo fill stores one vshort ending past the
// trailing halo. Both overhangs are smaller than one dvshort.
#define NONE_UNPACK_ROW_OVERHANG (32)

// Size in elements of the unpacked histogram buffer in VMEM. Sized for the max-sized histograms that fit in one input
// tile, plus two rows of RAW12 scratch. Shapes that need more space (small histograms with a large per-row padding
// overhead, or RAW12 pixels whose scratch is larger than two rows) are given fewer pixels per tile by the host.
#define UNPACKED_HISTOGRAM_BUFFER_SIZE \
    ((MAX_HISTOGRAM_BINS + 2 * UNPACKED_HISTOGRAM_HALO) * (MAX_MAX_HISTOGRAMS_PER_TILE + 2))

/// Row pitch, in elements, of one unpacked histogram (bins, halo on both sides, padding).
static inline int32_t UnpackedHistogramPitch(int32_t numBinsPerHist)
{
    return ((numBinsPerHist + 2 * UNPACKED_HISTOGRAM_HALO + UNPACKED_HISTOGRAM_PITCH_ALIGN - 1) /
            UNPACKED_HISTOGRAM_PITCH_ALIGN) *
           UNPACKED_HISTOGRAM_PITCH_ALIGN;
}

/// Number of vectors the RAW12 unpack loop decodes per pixel, for a pixel of pixelPitchBytes packed bytes.
static inline int32_t Raw12UnpackVectorsPerPixel(int32_t pixelPitchBytes)
{
    int32_t const numElementPairs = pixelPitchBytes / 3;
    int32_t const numVectors = (numElementPairs + RAW12_UNPACK_ALIGNMENT_PAIRS + RAW12_UNPACK_PAIRS_PER_VECTOR - 1) /
                               RAW12_UNPACK_PAIRS_PER_VECTOR;
    return ((numVectors + RAW12_UNPACK_UNROLL - 1) / RAW12_UNPACK_UNROLL) * RAW12_UNPACK_UNROLL;
}

/// Number of elements of the unpacked histogram buffer that unpacking one tile writes.
static inline int32_t UnpackedHistogramElementsPerTile(int32_t numHistPixPerTile, int32_t numHistPerPixel,
                                                       int32_t numBinsPerHist, int32_t pixelPitchBytes, int32_t isRAW12)
{
    int32_t const rowElements = numHistPixPerTile * numHistPerPixel * UnpackedHistogramPitch(numBinsPerHist);
    if (isRAW12)
    {
        // the scratch area starts right after the last row and is larger than the NONE overhang
        return rowElements + Raw12UnpackVectorsPerPixel(pixelPitchBytes) * RAW12_UNPACK_ELEMENTS_PER_VECTOR;
    }
    return rowElements + NONE_UNPACK_ROW_OVERHANG;
}

// Speed of light in meters per nanosecond
static const float SPEED_OF_LIGHT_M_PER_NS = 0.299792458f;

#endif // LIDAR_HISTOGRAM_CONVERSION_PARAM_H
