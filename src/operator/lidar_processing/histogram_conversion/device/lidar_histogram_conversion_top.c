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

#include "lidar_histogram_conversion_knl.h"
#include "lidar_histogram_conversion_param.h"

#include <cupva_device.h>
#ifdef PVASOL_LIDAR_PROFILING
#    include <cupva_device_debug.h>
#    define PROFILE_INIT() enable_TSC()
#    define PROFILE_DEF(name)      \
        uint32_t name##_total = 0; \
        uint32_t _tsc_##name  = 0
#    define PROFILE_TIC(name)        \
        chess_separator_scheduler(); \
        _tsc_##name = read_TSCL()
#    define PROFILE_TOC(name)        \
        chess_separator_scheduler(); \
        name##_total += read_TSCL() - _tsc_##name
#else
#    define PROFILE_INIT()
#    define PROFILE_DEF(name)
#    define PROFILE_TIC(name)
#    define PROFILE_TOC(name)
#endif

/**
 * @file lidar_histogram_conversion_top.c
 * @brief VPU main kernel for LiDAR histogram conversion
 *
 * This is the main entry point called from the host. It orchestrates the
 * complete histogram conversion pipeline:
 * 1. Unpack input tile into separate buffers of pixel headers, histogram headers, and histogram bin data
 * 2. Smooth histogram data
 * 3. Peak detection
 * 4. Range computation from time-of-flight
 * 5. XYZ computation from range (optional)
 * 6. Reflectance computation from intensity (optional)
 */

// Parameter definitions that will be set by host
VMEM(A, int32_t, numHistPerPixel);
VMEM(A, int32_t, numBinsPerHist);
VMEM(A, int32_t, numPeaksPerHist);
VMEM(A, int32_t, pixelHeaderSize);
VMEM(A, int32_t, histHeaderSize);
VMEM(A, int32_t, packingType);
VMEM(A, float, offsetNs);
VMEM(A, float, binSizeNs);
VMEM(A, float, rangeScale);
VMEM(A, float, invMaxIntensity);
VMEM(A, int32_t, height);
VMEM(A, int32_t, width);
VMEM(A, int32_t, numTilesW);
VMEM(A, int32_t, numTilesH);
VMEM(A, int32_t, numHistPixPerTile);

// Data flow handlers (to be configured in host code)
// Histogram input data flow is fixed shape, so uses TensorDataFlow for maximum performance
VMEM_TDF_HANDLER(A, histogramInputHdl);

// Bias and XYZ calibration data flows are optionally enabled and share a single auxiliary input SQDF
VMEM_SEQDF_HANDLER(A, auxInputHdl);

// Output data flows are all optional, so share a single output SQDF
VMEM_SEQDF_HANDLER(A, outputHdl);

// input external memory pointers (linePitch > 0 indicates enabled)
VMEM_POINTER(A, rangeBiasExtPtr);
VMEM(A, int32_t, rangeBiasLinePitch);
VMEM_POINTER(A, xyzCalibrationExtPtr);
VMEM(A, int32_t, xyzCalibrationLinePitch);

// output external memory pointers (linePitch > 0 indicates enabled)
VMEM_POINTER(A, rangeExtPtr);
VMEM(A, int32_t, rangeOutLinePitch);
VMEM_POINTER(A, xyzExtPtr);
VMEM(A, int32_t, xyzOutLinePitch);
VMEM_POINTER(A, reflectanceExtPtr);
VMEM(A, int32_t, reflectanceOutLinePitch);

// input VMEM buffers allocated for the largest possible tile size while supporting double buffering.
// NOTE: using 4B element size for alignment reasons. We need each histogram to be 4B aligned for RAW12
// unpacking routine since it uses a WX vector load.
VMEM(B, uint32_t, histogramIn,
     TDF_DOUBLE(uint32_t, MAX_TILE_BUFFER_SIZE_BYTES / (MAX_PIXELS_PER_TILE * sizeof(uint32_t)), MAX_PIXELS_PER_TILE,
                1));

// input per-pixel calibration values
VMEM(A, float, rangeBiasIn, RDF_SINGLE(float, MAX_PIXELS_PER_TILE, 1));
VMEM(A, float, xyzCalibrationIn, RDF_SINGLE(float, MAX_PIXELS_PER_TILE * 3, 1));

// intermediate buffers that hold arrays of pixel headers, histogram headers, histograms, etc.
// NOTE: pixelHeader / histogramHeader are written by the unpack stage but never read by
// downstream stages. histogramHeader's VMEM is repurposed below as the reflectance output
// buffer (size matches exactly — see the reflectanceOut alias inside CUPVA_VPU_MAIN).
VMEM(C, uint16_t, pixelHeader, RDF_SINGLE(uint16_t, MAX_PIXEL_HEADER_SIZE, MAX_PIXELS_PER_TILE));
VMEM(C, uint16_t, histogramHeader, RDF_SINGLE(uint16_t, MAX_HISTOGRAM_HEADER_SIZE, MAX_HISTOGRAMS_PER_TILE));
// intermediate buffer that holds the unpacked histogram data
VMEM(A, uint16_t, histogram, UNPACKED_HISTOGRAM_BUFFER_SIZE);
// smoothPad holds the smoothed histogram with 1 element of 0 padding on each end
VMEM(C, uint16_t, smoothPad, 1 + MAX_HISTOGRAM_BINS + 1);
// peakCandidates holds intermediate peak detection results, with one extra vshort to guard against
// vector store overshoot
VMEM(A, int16_t, peakCandidates, MAX_HISTOGRAM_BINS + 16);

// tmp buffer for peak detection, two vshort vectors per histogram. Four fields are packed two to
// a vector: [peakIdx | peakVal] then [binLeft | binRight]. The detector records the bin triple so
// the emit stage never reads the smoothed histogram back.
static int32_t const kPeakPitch = 2 * pva_elementsof(vshort);
VMEM(C, int16_t, peaks, RDF_SINGLE(int16_t, kPeakPitch, MAX_HISTOGRAMS_PER_TILE));

// output buffers
VMEM(C, float, rangeOut, RDF_SINGLE(float, MAX_PEAKS_PER_HISTOGRAM, MAX_HISTOGRAMS_PER_TILE));
VMEM(C, float, xyzOut, RDF_SINGLE(float, (3 * MAX_PEAKS_PER_HISTOGRAM), MAX_HISTOGRAMS_PER_TILE));

// Integer gather scratch for the fused emit pass: [idx][left][center][right].
// Placed in A superbank (distinct from output's C bank) so the compiler can prove the inner
// batch loop's loads and stores don't alias — eliminating the [S:6,LC] constraint and letting
// the loop be modulo-scheduled. Sized for the worst case, SCRATCH_PEAKS_PER_PIXEL peaks per pixel.
// (Bank B is the natural choice but is oversubscribed by the histogramIn DMA double buffer.)
#define SCRATCH_PEAKS_PER_PIXEL (MAX_HISTOGRAMS_PER_PIXEL * MAX_PEAKS_PER_HISTOGRAM)
VMEM(A, int32_t, scratchGather, RDF_SINGLE(int32_t, 4 * SCRATCH_PEAKS_PER_PIXEL, 1));

/**
 * @brief Main processing function
 */
CUPVA_VPU_MAIN()
{
    // Open the data flows
    cupvaTensorDataFlowOpen(histogramInputHdl, histogramIn);
    int32_t linePitch = cupvaTensorDataFlowGetTileLinePitch(histogramInputHdl); // jump between pixels in bytes

    /// NOTE: could utilize tile "depth" (height) dim > 1, but for simplicity it is always 1 (configured on host side).
    /// If a fixed input tensor shape is known apriori, it can be beneficial to choose a tile shape with "depth" > 1
    /// to fine-tune the VMEM utilization of input and output buffers. This exercise is left for final overall system
    /// design after deciding sensor resolution and operating mode.
    //int32_t planeSize = cupvaTensorDataFlowGetTilePlaneSize(histogramInputHdl); // stride in "depth" (height) dimension
    CUPVA_VPU_ASSERT(cupvaTensorDataFlowGetNumTiles(histogramInputHdl) == 1);

    bool const isRAW12                = (LidarDataPackingType_Dev)packingType == LIDAR_DATA_PACKING_TYPE_RAW12;
    bool const hasRangeOutput         = rangeOutLinePitch > 0;
    bool const hasXyzOutput           = xyzOutLinePitch > 0;
    bool const hasReflectanceOutput   = reflectanceOutLinePitch > 0;
    bool const hasRangeBiasInput      = rangeBiasLinePitch > 0;
    bool const hasXyzCalibrationInput = xyzCalibrationLinePitch > 0;
    CUPVA_VPU_ASSERT(hasRangeBiasInput || !hasXyzCalibrationInput); // xyz requires range bias

    cupvaSQDFOpen(auxInputHdl);
    cupvaSQDFOpen(outputHdl);

    // number of elements in calibration data tiles (range bias and xyz calibration)
    int32_t const numCalElementsPerTile = numHistPixPerTile;
    int32_t const numCalTailElements    = width - numCalElementsPerTile * (numTilesW - 1);
    CUPVA_VPU_ASSERT(numCalTailElements > 0 && numCalTailElements <= numCalElementsPerTile);

    // number of elements in output tiles
    int32_t const rangeOutElementsPerPix  = numHistPerPixel * numPeaksPerHist;
    int32_t const rangeOutElementsPerTile = rangeOutElementsPerPix * numHistPixPerTile;
    int32_t const rangeOutTailElements    = width * rangeOutElementsPerPix - rangeOutElementsPerTile * (numTilesW - 1);
    CUPVA_VPU_ASSERT(rangeOutTailElements > 0 && rangeOutTailElements <= rangeOutElementsPerTile);

    int32_t xyzOutElementsPerPix  = 3 * rangeOutElementsPerPix;
    int32_t xyzOutElementsPerTile = xyzOutElementsPerPix * numHistPixPerTile;
    int32_t xyzOutTailElements    = width * xyzOutElementsPerPix - xyzOutElementsPerTile * (numTilesW - 1);
    CUPVA_VPU_ASSERT(xyzOutTailElements > 0 && xyzOutTailElements <= xyzOutElementsPerTile);

    int32_t const reflectanceOutElementsPerPix  = rangeOutElementsPerPix;
    int32_t const reflectanceOutElementsPerTile = rangeOutElementsPerTile;
    int32_t const reflectanceOutTailElements    = rangeOutTailElements;
    CUPVA_VPU_ASSERT(reflectanceOutTailElements > 0 && reflectanceOutTailElements <= reflectanceOutElementsPerTile);

    // Alias histogramHeader VMEM as the reflectance output buffer. The header data is
    // dead after unpacking; sizes match exactly (MAX_HISTOGRAM_HEADER_SIZE * MAX_HISTOGRAMS_PER_TILE
    // uint16_t = MAX_PEAKS_PER_HISTOGRAM * MAX_HISTOGRAMS_PER_TILE float = 16 KB).
    // This conserves VMEM space and allows range and reflectance to be computed independently.
    CUPVA_VPU_ASSERT(sizeof(histogramHeader) >= sizeof(rangeOut));
    float *reflectanceOut = (float *)histogramHeader;

    // the tmp buffer for histogram data includes halo which is filled with reflection while unpacking
    // row pitch of the unpacked histogram is a whole number of vectors
    CUPVA_VPU_ASSERT(pva_elementsof(vshort) == UNPACKED_HISTOGRAM_PITCH_ALIGN);
    uint16_t *const smoothBuf = smoothPad + 1;
    int32_t histPitchOut      = UnpackedHistogramPitch(numBinsPerHist);

    PROFILE_DEF(unpack);
    PROFILE_DEF(smooth);
    PROFILE_DEF(peaks);
    PROFILE_DEF(scalar_stuff);
    PROFILE_DEF(dma_in);
    PROFILE_DEF(dma_out);
    PROFILE_INIT();

    // Process each tile
    // NOTE: tile traversal order for TensorDataFlow is
    //       "Depth (forward) -> Width (Left to right) -> Height (Top to bottom)"
    // We have it mapped so TDF "Depth" is our tensor H dim, TDF "Width" is our "C", and TDF "Height" is our "W" dim.
    // What this means is that we visit tiles in a "strip-mining" type of pattern rather than raster scan.
    // eg. The first tile visited contains the first numHistPixPerTile pixels from row 0, the next tile
    // contains numHistPixPerTile from row 1, etc., then the next column of tiles is visited from row 0, etc.
    for (int32_t j = 0; j < numTilesW; j++)
    {
        for (int32_t i = 0; i < numTilesH; i++)
        {
            PROFILE_TIC(dma_in);
            void *histogramTile = cupvaTensorDataFlowAcquire(histogramInputHdl);
            PROFILE_TOC(dma_in);

            // unpack input tile into separate buffers of pixel headers, histogram headers, and histogram bin data
            PROFILE_TIC(unpack);
            if (isRAW12)
            {
                UnpackHistogramDataRAW12_v((uint8_t *)histogramTile, linePitch, pixelHeader, histogramHeader, histogram,
                                           histPitchOut, numHistPixPerTile, numHistPerPixel, pixelHeaderSize,
                                           histHeaderSize, numBinsPerHist);
            }
            else
            {
                UnpackHistogramDataNone_v((uint8_t *)histogramTile, linePitch, pixelHeader, histogramHeader, histogram,
                                          histPitchOut, numHistPixPerTile, numHistPerPixel, pixelHeaderSize,
                                          histHeaderSize, numBinsPerHist);
            }
            PROFILE_TOC(unpack);

            // after unpacking, the buffer contents are like so:
            // pixelHeader: 1D array of length numHistPixPerTile * pixelHeaderSize
            // histogramHeader: 1D array of length numHistPixPerTile * numHistPerPixel * histHeaderSize
            // histogram (padded): 2D array of width numBinsPerHist + halo, height numHistPixPerTile * numHistPerPixel and pitch histPitchOut
            // we can now release the input tile so DMA can pre-fetch the next one.
            PROFILE_TIC(dma_in);
            cupvaTensorDataFlowRelease(histogramInputHdl);

            // fetch the calibration data for this tile
            if (hasRangeBiasInput)
            {
                int32_t numCalElements = (j == numTilesW - 1) ? numCalTailElements : numCalElementsPerTile;
                uint64_t rangeBiasOffset =
                    rangeBiasExtPtr.offset + i * rangeBiasLinePitch + j * numCalElementsPerTile * sizeof(float);
                cupvaSQDFUpdateAddr(auxInputHdl, 0, rangeBiasExtPtr.base + rangeBiasOffset, MEMTYPE_DRAM,
                                    cupvaGetVmemAddress(rangeBiasIn), MEMTYPE_VMEM);
                cupvaSQDFUpdateTileSize(auxInputHdl, 0, numCalElements * sizeof(float), 1);

                if (hasXyzCalibrationInput)
                {
                    uint64_t xyzCalibrationOffset = xyzCalibrationExtPtr.offset + i * xyzCalibrationLinePitch +
                                                    j * numCalElementsPerTile * 3 * sizeof(float);
                    cupvaSQDFUpdateAddr(auxInputHdl, 1, xyzCalibrationExtPtr.base + xyzCalibrationOffset, MEMTYPE_DRAM,
                                        cupvaGetVmemAddress(xyzCalibrationIn), MEMTYPE_VMEM);
                    cupvaSQDFUpdateTileSize(auxInputHdl, 1, numCalElements * 3 * sizeof(float), 1);
                }
                cupvaSQDFFlushAndTrig(auxInputHdl); // sync happens before cal data is needed
            }
            PROFILE_TOC(dma_in);

            // each histogram of each pixel is processed one at a time to extract peaks
            for (int32_t p = 0; p < numHistPixPerTile; p++)
            {
                // Visit each histogram in the pixel
                for (int32_t h = 0; h < numHistPerPixel; h++)
                {
                    uint16_t *histogramPtr = histogram + (p * numHistPerPixel + h) * histPitchOut;
                    int16_t *peakPtr       = peaks + (p * numHistPerPixel + h) * kPeakPitch;

                    // smooth into smoothBuf; collect unsigned min/max for free via agen tracking
                    PROFILE_TIC(smooth);
                    uint16_t histMin, histMax;
                    SmoothHistogram_vi16_dv(histogramPtr, smoothBuf, numBinsPerHist, histMin, histMax);
                    PROFILE_TOC(smooth);

                    // Estimate noise floor as histMin + 1/8 * (histMax - histMin).
                    CUPVA_VPU_ASSERT(histMax >= histMin && histMax <= MAX_HISTOGRAM_BIN_VALUE);
                    uint16_t noiseThreshold = histMin + ((histMax - histMin) >> 3);

                    // run peak detection
                    PROFILE_TIC(peaks);
                    DetectPeaksNMS_v(smoothBuf, peakPtr, numBinsPerHist, numPeaksPerHist, noiseThreshold,
                                     peakCandidates);
                    PROFILE_TOC(peaks);
                }
            }

            // make sure calibration data is settled in the buffers before use
            PROFILE_TIC(dma_in);
            if (hasRangeBiasInput)
            {
                cupvaSQDFSync(auxInputHdl);
            }
            PROFILE_TOC(dma_in);

            PROFILE_TIC(scalar_stuff);
            if (hasRangeOutput || hasXyzOutput || hasReflectanceOutput)
            {
                int32_t totalPeaksPerPixel = numHistPerPixel * numPeaksPerHist;
                for (int32_t p = 0; p < numHistPixPerTile; p++)
                {
                    int32_t outBase = p * totalPeaksPerPixel;
                    int32_t xyzBase = p * 3 * totalPeaksPerPixel;

                    int32_t const emitMask =
                        ((hasRangeOutput || hasXyzOutput) ? 1 : 0) | (hasReflectanceOutput ? 2 : 0);

                    // One fused pass: integer gather, then range/XYZ/reflectance in the vector unit.
                    EmitPixelBatch_v(peaks + p * numHistPerPixel * kPeakPitch, numHistPerPixel, numPeaksPerHist,
                                     numBinsPerHist, kPeakPitch, SCRATCH_PEAKS_PER_PIXEL, offsetNs, binSizeNs,
                                     rangeScale, rangeBiasIn[p], xyzCalibrationIn + p * 3, invMaxIntensity, emitMask,
                                     &rangeOut[outBase], &xyzOut[xyzBase], &xyzOut[xyzBase + totalPeaksPerPixel],
                                     &xyzOut[xyzBase + 2 * totalPeaksPerPixel], &reflectanceOut[outBase],
                                     (int32_t *)scratchGather);
                }
            }
            PROFILE_TOC(scalar_stuff);

            // All three outputs live in independent VMEM buffers (rangeOut, xyzOut, histogramHeader-as-reflectanceOut),
            // so they can be DMA'd in any order without inter-output ordering hazards.
            PROFILE_TIC(dma_out);
            if (hasRangeOutput)
            {
                uint64_t rangeOutOffset =
                    rangeExtPtr.offset + i * rangeOutLinePitch + j * rangeOutElementsPerTile * sizeof(float);
                cupvaSQDFUpdateAddr(outputHdl, 0, cupvaGetVmemAddress(rangeOut), MEMTYPE_VMEM,
                                    rangeExtPtr.base + rangeOutOffset, MEMTYPE_DRAM);
                int32_t rangeTx = (j == numTilesW - 1) ? rangeOutTailElements : rangeOutElementsPerTile;
                cupvaSQDFUpdateTileSize(outputHdl, 0, rangeTx * sizeof(float), 1);
                cupvaSQDFFlushAndTrig(outputHdl);
                cupvaSQDFSync(outputHdl);
            }
            if (hasXyzOutput)
            {
                uint64_t xyzOutOffset =
                    xyzExtPtr.offset + i * xyzOutLinePitch + j * xyzOutElementsPerTile * sizeof(float);
                cupvaSQDFUpdateAddr(outputHdl, 0, cupvaGetVmemAddress(xyzOut), MEMTYPE_VMEM,
                                    xyzExtPtr.base + xyzOutOffset, MEMTYPE_DRAM);
                int32_t xyzTx = (j == numTilesW - 1) ? xyzOutTailElements : xyzOutElementsPerTile;
                cupvaSQDFUpdateTileSize(outputHdl, 0, xyzTx * sizeof(float), 1);
                cupvaSQDFFlushAndTrig(outputHdl);
                cupvaSQDFSync(outputHdl);
            }
            if (hasReflectanceOutput)
            {
                uint64_t reflectanceOutOffset = reflectanceExtPtr.offset + i * reflectanceOutLinePitch +
                                                j * reflectanceOutElementsPerTile * sizeof(float);
                // note the alias of histogramHeader to reflectanceOut
                cupvaSQDFUpdateAddr(outputHdl, 0, cupvaGetVmemAddress(histogramHeader), MEMTYPE_VMEM,
                                    reflectanceExtPtr.base + reflectanceOutOffset, MEMTYPE_DRAM);
                int32_t reflectanceTx =
                    (j == numTilesW - 1) ? reflectanceOutTailElements : reflectanceOutElementsPerTile;
                cupvaSQDFUpdateTileSize(outputHdl, 0, reflectanceTx * sizeof(float), 1);
                cupvaSQDFFlushAndTrig(outputHdl);
                cupvaSQDFSync(outputHdl);
            }
            PROFILE_TOC(dma_out);
        }
    }
#ifdef PVASOL_LIDAR_PROFILING
    printf("[LIDAR_PROF] unpack=%u smooth=%u peaks=%u scalar_stuff=%u dma_in=%u dma_out=%u cycles\n", unpack_total,
           smooth_total, peaks_total, scalar_stuff_total, dma_in_total, dma_out_total);
#endif
    // Close the data flows
    cupvaSQDFClose(outputHdl);
    cupvaSQDFClose(auxInputHdl);
    cupvaTensorDataFlowClose(histogramInputHdl);
    return 0;
}
