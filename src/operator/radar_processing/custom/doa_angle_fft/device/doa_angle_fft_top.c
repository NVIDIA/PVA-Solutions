/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file doa_angle_fft_top.c
 * @brief VPU main kernel for Direction of Arrival (DOA) Angle Estimation via 2D FFT
 *
 * This file implements the PVA (Programmable Vision Accelerator) VPU kernel for
 * estimating target azimuth and elevation angles from radar detection snapshots
 * using 2D FFT-based beamforming.
 *
 * Algorithm Overview:
 * 1. Load coefficient data (twiddle factors, window, angle bins) from DRAM
 * 2. For each batch of OUTPUT_BATCH_SIZE (8) detections:
 *    a. For each detection:
 *       - At the first detection of a group of G (G = fft_groups_for(az, el)): window and re-map
 *         each snapshot of the group into its column band of the packed grid, then run the azimuth
 *         FFT once over all G bands; the digit reverse writes the transposed grid
 *       - Run the elevation FFT over this detection's band and compute the power spectrum
 *       - Find the peak location (azimuth and elevation bins) and gather its neighbours
 *    b. Interpolate the peaks for sub-bin accuracy and convert them to angles (degrees)
 *    c. Write the batch's results (azimuth, elevation, power in dB) to output
 * 3. Write total target count
 *
 * Key Features:
 * - Double-buffered input groups and output tiles, overlapping computation and data transfer
 * - Output batched in OUTPUT_BATCH_SIZE (8) detections for vectorization
 * - Fixed-point arithmetic with configurable Q-format
 * - SQDF (Sequence Data Flow) for DRAM <-> VMEM transfers
 */

#include "doa_angle_fft_knl.h"
#include "doa_angle_fft_param.h"

/* Scalar parameters stored in VMEM bank A */
VMEM(A, int32_t, detectionCount);      // Number of detections to process
VMEM(A, int32_t, maxTargetCount);      // Maximum number of targets to process
VMEM(A, int32_t, inTileWidthBytes);    // Snapshot tile row width, for resizing the tail transfer
VMEM(A, int32_t, log2N_AzimuthSize);   // Log2 of azimuth FFT size
VMEM(A, int32_t, log2N_ElevationSize); // Log2 of elevation FFT size
/* Stage structure for each axis, factorized on the host. */
VMEM(A, FFTFactorization, factorizationAzimuth);
VMEM(A, FFTFactorization, factorizationElevation);
VMEM(A, float, epsilon);     // Small value to avoid division by zero
VMEM(A, int32_t, targetIdx); // Running count of output targets

/* Coefficient buffers for FFT operations (VMEM bank A) */
VMEM(A, int32_t, twiddleBuf, TWIDDLE_SIZE);      // FFT twiddle factors
VMEM(A, int32_t, windowBuf, WINDOW_SIZE);        // 2D Hanning window coefficients
VMEM(A, int32_t, angleBinsBuf, ANGLE_BINS_SIZE); // Pre-computed angle bins (azimuth + elevation)

/* Temporary buffers for FFT stages (VMEM banks A and B) */
VMEM(A, int32_t, tmpBuf, FFT_BUF_SIZE);                        // FFT intermediate buffer
VMEM(A, uint16_t, peakAzimuthIndexBuf, PEAK_INDEX_BUF_SIZE);   // Peak azimuth bin
VMEM(B, uint16_t, peakElevationIndexBuf, PEAK_INDEX_BUF_SIZE); // Peak elevation bin

/* Input snapshot groups (see IN_SLOT_HEIGHT and IN_GROUP_BUF_SIZE) and the FFT ping buffer. */
VMEM(B, int32_t, srcSnapshotsBuf0, IN_GROUP_BUF_SIZE);
VMEM(C, int32_t, srcSnapshotsBuf1, IN_GROUP_BUF_SIZE);
VMEM(B, int32_t, tmpPing, FFT_BUF_SIZE); // FFT ping buffer

/* Output buffers and pong buffer (VMEM bank C) */
VMEM(C, int32_t, tmpPong, FFT_BUF_SIZE); // FFT pong buffer / power spectrum
VMEM(C, float, dstTargetAnglesBuf,
     RDF_DOUBLE(float, OUT_TARGET_ANGLE_TILE_PITCH, OUT_TARGET_ANGLE_TILE_HEIGHT)); // Target angles (double-buffered)
VMEM(C, int32_t, dstTargetIndexMapBuf,
     RDF_DOUBLE(int32_t, OUT_TARGET_INDEX_MAP_TILE_PITCH,
                OUT_TARGET_INDEX_MAP_TILE_HEIGHT)); // Target index map (double-buffered)
VMEM(C, int32_t, dstTargetCountBuf,
     RDF_SINGLE(int32_t, OUT_TARGET_COUNT_TILE_PITCH, OUT_TARGET_COUNT_TILE_HEIGHT)); // Target count

/* Sequence Data Flow (SQDF) handlers for data transfers (VMEM bank A) */
VMEM_SEQDF_HANDLER(A, coeffHdl);             // Coefficient data transfer handler
VMEM_SEQDF_HANDLER(A, srcSnapshotsHdl);      // Input snapshots transfer handler
VMEM_SEQDF_HANDLER(A, dstTargetAnglesHdl);   // Output angles transfer handler
VMEM_SEQDF_HANDLER(A, dstTargetIndexMapHdl); // Output index map transfer handler
VMEM_SEQDF_HANDLER(A, dstTargetCountHdl);    // Output count transfer handler

/* External memory pointers for SQDF (VMEM bank A) */
VMEM_POINTER(A, srcSnapshotsExtPtr);      // External memory pointer to input snapshots
VMEM_POINTER(A, dstTargetAnglesExtPtr);   // External memory pointer to output angles
VMEM_POINTER(A, dstTargetIndexMapExtPtr); // External memory pointer to output index map
VMEM_POINTER(A, dstTargetCountExtPtr);    // External memory pointer to output count

/* Samplers for angle interpolation (VMEM bank A) */
VMEM(A, CupvaSampler, samplerAzimuth);   // 1D sampler for azimuth angle interpolation
VMEM(A, CupvaSampler, samplerElevation); // 1D sampler for elevation angle interpolation

/* Parameter structures for double-buffering (VMEM bank A) */
VMEM(A, doa_angle_fft_params_t, params_doa_angle_fft, 2); // Two parameter sets for ping-pong processing

/* AGEN configurations: windowing per [buffer][slot], one azimuth FFT set, and elevation FFT and peak
 * search per slot. */
VMEM(A, AgenCFG, cfgsWindowing, 2 * MAX_FFT_GROUPS * 3);
VMEM(A, AgenCFG, cfgsAzimuthFFT, FFT_CFGS_PER_SET);
VMEM(A, AgenCFG, cfgsElevationFFT, MAX_FFT_GROUPS *FFT_CFGS_PER_SET);
VMEM(A, AgenCFG, cfgsPeak, MAX_FFT_GROUPS);

/**
  * @brief Main VPU kernel entry point for DOA Angle FFT processing
  *
  * This function performs 2D FFT-based Direction of Arrival estimation:
  * 1. Applies 2D Hanning window to input snapshots
  * 2. Performs 2D FFT (azimuth then elevation)
  * 3. Finds peak locations in the power spectrum
  * 4. Performs quadratic interpolation for sub-bin accuracy
  * 5. Converts indices to angle values using lookup tables
  */
CUPVA_VPU_MAIN()
{

    /* Calculate buffer sizes for double-buffering */
    int32_t outTargetAnglesTileBufSize   = OUT_TARGET_ANGLE_TILE_HEIGHT * OUT_TARGET_ANGLE_TILE_PITCH;
    int32_t outTargetIndexMapTileBufSize = OUT_TARGET_INDEX_MAP_TILE_HEIGHT * OUT_TARGET_INDEX_MAP_TILE_PITCH;

    /* Load coefficients (twiddle factors, window, angle bins) from DRAM */
    cupvaSQDFOpen(coeffHdl);
    cupvaSQDFTrig(coeffHdl);

    /* Setup twiddle factor pointers: [0] for azimuth FFT, [1] for elevation FFT */
    int32_t *twiddleTable[2];
    twiddleTable[0] = (int32_t *)twiddleBuf;
    twiddleTable[1] = (int32_t *)twiddleBuf + DOA_TWIDDLE_SIZE_CALCULATE((1 << log2N_AzimuthSize));

    /* Initialize target output counter */
    targetIdx = 0;

    /* Initialize two parameter sets for double-buffered processing */
    for (int32_t i = 0; i < 2; i++) chess_unroll_loop(*)
    {
        /* FFT configuration */
        params_doa_angle_fft[i].log2N_azimuth_size      = log2N_AzimuthSize;
        params_doa_angle_fft[i].log2N_elevation_size    = log2N_ElevationSize;
        params_doa_angle_fft[i].factorization_azimuth   = &factorizationAzimuth;
        params_doa_angle_fft[i].factorization_elevation = &factorizationElevation;
        params_doa_angle_fft[i].window_shift_qbits      = WINDOW_QBITS - TRUNCATE_QBITS;
        params_doa_angle_fft[i].twiddle_shift_qbits     = TWIDDLE_QBITS - TRUNCATE_QBITS;

        /* Buffer pointers */
        params_doa_angle_fft[i].power_spectrum = (float *)tmpPong;
        params_doa_angle_fft[i].angle_bins     = (int32_t *)angleBinsBuf;
        params_doa_angle_fft[i].epsilon        = epsilon;

        /* Sampler handles for angle interpolation */
        params_doa_angle_fft[i].sampler_azimuth   = &samplerAzimuth;
        params_doa_angle_fft[i].sampler_elevation = &samplerElevation;

        /* Output buffers stay plain ping-pong; the input is addressed per slot below. */
        params_doa_angle_fft[i].dst_target_angles_buf =
            ((float *)dstTargetAnglesBuf) + (i * outTargetAnglesTileBufSize);
        params_doa_angle_fft[i].dst_target_index_map_buf =
            ((int32_t *)dstTargetIndexMapBuf) + (i * outTargetIndexMapTileBufSize);

        /* Shared pointers (not double-buffered) */
        params_doa_angle_fft[i].dst_target_idx               = &targetIdx;
        params_doa_angle_fft[i].src_snapshots_ext_ptr        = &srcSnapshotsExtPtr;
        params_doa_angle_fft[i].dst_target_angles_ext_ptr    = &dstTargetAnglesExtPtr;
        params_doa_angle_fft[i].dst_target_index_map_ext_ptr = &dstTargetIndexMapExtPtr;
    }

    /* Local buffers for peak detection and interpolation (stored in registers/VMEM) */
    int32_t peakAzimuth[OUTPUT_BATCH_SIZE];   // Interpolated azimuth angles (Q16 fixed-point)
    int32_t peakElevation[OUTPUT_BATCH_SIZE]; // Interpolated elevation angles (Q16 fixed-point)
    unsigned short *peakAzimuthIndex   = (unsigned short *)peakAzimuthIndexBuf;   // Peak azimuth bin indices
    unsigned short *peakElevationIndex = (unsigned short *)peakElevationIndexBuf; // Peak elevation bin indices
    int32_t interpAzimuthIndex[OUTPUT_BATCH_SIZE];   // Interpolated azimuth indices (Q16 fixed-point)
    int32_t interpElevationIndex[OUTPUT_BATCH_SIZE]; // Interpolated elevation indices (Q16 fixed-point)
    float max_val[OUTPUT_BATCH_SIZE];                // Peak power values
    float pY1[2 * OUTPUT_BATCH_SIZE];                // Left/lower neighbor values for interpolation (Az + Elev)
    float pY2[2 * OUTPUT_BATCH_SIZE];                // Center peak values for interpolation (Az + Elev)
    float pY3[2 * OUTPUT_BATCH_SIZE];                // Right/upper neighbor values for interpolation (Az + Elev)
    int32_t pshiftedIdx[2 * OUTPUT_BATCH_SIZE];      // Shifted indices for interpolation (Az + Elev)

    const int32_t inTileHeight = (1 << log2N_AzimuthSize);

    // Detections per FFT batch: the packed grid is fftGroups * el wide by az tall.
    const int32_t azSize    = (1 << log2N_AzimuthSize);
    const int32_t elSize    = (1 << log2N_ElevationSize);
    const int32_t fftGroups = fft_groups_for(azSize, elSize);
    const int32_t elevPitch = (COMPLEX_ELEMENT_NUM * azSize) + 2;

    /* Windowing: one configuration per (input buffer, slot). Each scatters its detection into its
     * own column band of the packed grid. */
    int32_t *inGroupBase[2] = {(int32_t *)srcSnapshotsBuf0, (int32_t *)srcSnapshotsBuf1};

    for (int32_t b = 0; b < 2; b++)
    {
        for (int32_t g = 0; g < fftGroups; g++)
        {
            windowing_init_azimuth(inGroupBase[b] + (g * IN_SLOT_SIZE), tmpPong, windowBuf,
                                   &cfgsWindowing[(((b * MAX_FFT_GROUPS) + g) * 3)], &params_doa_angle_fft[0],
                                   fftGroups, g);
        }
    }

    /* Azimuth radix-4 stages and digit reverse: one fused configuration for the whole packed grid */
    fft_batched_init_azimuth(tmpPing, tmpPong, tmpBuf, twiddleTable, cfgsAzimuthFFT, &params_doa_angle_fft[0],
                             fftGroups);

    /* Elevation FFT and peak search run per detection, over that slot's band of rows in the
     * transposed grid the azimuth digit reverse produced. */
    int32_t line_width_peak = (1 << log2N_AzimuthSize);
    int32_t pad_peak        = 2;
    float *slotPowerSpectrum[MAX_FFT_GROUPS];
    for (int32_t g = 0; g < fftGroups; g++)
    {
        const int32_t off = g * elSize * elevPitch;

        fft_batched_init_elevation(tmpPing + off, tmpPong + off, tmpBuf + off, &twiddleTable[1],
                                   &cfgsElevationFFT[g * FFT_CFGS_PER_SET], &params_doa_angle_fft[0]);
        find_max_val_loc_init((float *)(tmpPong + off), log2N_ElevationSize, &cfgsPeak[g], line_width_peak, pad_peak);
        slotPowerSpectrum[g] = (float *)(tmpPong + off);
    }

    /* Initialize angle computation samplers for both buffers */
    compute_angles_init(&params_doa_angle_fft[0], &params_doa_angle_fft[1], &interpAzimuthIndex[0],
                        &interpElevationIndex[0], &peakAzimuth[0], &peakElevation[0]);

    /* Wait for coefficient data to be loaded, then close the transfer */
    cupvaSQDFSync(coeffHdl);
    cupvaSQDFClose(coeffHdl);

    /* Open data flow handlers for I/O operations */
    cupvaSQDFOpen(srcSnapshotsHdl);
    cupvaSQDFOpen(dstTargetAnglesHdl);
    cupvaSQDFOpen(dstTargetIndexMapHdl);

    detectionCount = detectionCount > maxTargetCount ? maxTargetCount : detectionCount;

    if (detectionCount != 0)
    {
        /* Start loading the first group of detection snapshots */
        updateInputSQDF(srcSnapshotsHdl, 0, inGroupBase[0], &params_doa_angle_fft[0],
                        (detectionCount < fftGroups) ? detectionCount : fftGroups, fftGroups, inTileWidthBytes);

        /* Process detections in batches of OUTPUT_BATCH_SIZE (8) for vectorization */
        for (int32_t y = 0; y < detectionCount; y += OUTPUT_BATCH_SIZE)
        {
            int32_t y_iter = (detectionCount - y) > OUTPUT_BATCH_SIZE ? OUTPUT_BATCH_SIZE : (detectionCount - y);

            /* Process each detection in the current batch */
            for (int32_t z = 0; z < y_iter; z++)
            {
                int32_t i = y + z;

                /* Slot within the packed FFT batch, and which input buffer holds that batch. */
                const int32_t slot = i % fftGroups;
                const int32_t buf  = (i / fftGroups) & 1;

                if (slot == 0)
                {
                    /* Wait for the whole group of snapshots, window each detection into its band of
                     * the packed grid, then run the azimuth FFT once across all of them. */
                    cupvaSQDFSync(srcSnapshotsHdl);

                    for (int32_t g = 0; g < fftGroups; g++)
                    {
                        windowing_exec_azimuth(&cfgsWindowing[(((buf * MAX_FFT_GROUPS) + g) * 3)],
                                               &params_doa_angle_fft[0]);
                    }

                    fft_batched_exec_azimuth(cfgsAzimuthFFT, &params_doa_angle_fft[0], fftGroups);

                    /* The packed grid is consumed; the next group can load into the other buffer. */
                    const int32_t next = i + fftGroups;
                    if (next < detectionCount)
                    {
                        const int32_t remaining = detectionCount - next;
                        updateInputSQDF(srcSnapshotsHdl, next, inGroupBase[buf ^ 1], &params_doa_angle_fft[0],
                                        (remaining < fftGroups) ? remaining : fftGroups, fftGroups, inTileWidthBytes);
                    }
                }

                /* Execute elevation FFT: 1D FFT (vertical) + peak detection */
                max_val[z] =
                    fft_batched_exec_elevation(&cfgsElevationFFT[slot * FFT_CFGS_PER_SET], &params_doa_angle_fft[0]);

                /* Find peak location in power spectrum */
                int32_t n_peak    = 1 << log2N_ElevationSize;
                int32_t iter_peak = (1 << log2N_AzimuthSize) / KERNEL_BATCH;
                find_max_val_loc(&cfgsPeak[slot], n_peak, iter_peak, max_val[z], log2N_AzimuthSize, peakAzimuthIndex,
                                 peakElevationIndex);

                /* Gather 3x3 neighborhood around peak for quadratic interpolation */
                params_doa_angle_fft[0].power_spectrum = slotPowerSpectrum[slot];
                gather_neighbours(&params_doa_angle_fft[0], peakAzimuthIndex[0], peakElevationIndex[0], &pY1[z],
                                  &pY2[z], &pY3[z], &pshiftedIdx[z]);
            }
            /* Pad the batch if not full (for vectorized operations) */
            if (y_iter != OUTPUT_BATCH_SIZE)
            {
                for (int32_t j = y_iter; j < OUTPUT_BATCH_SIZE; j++)
                {
                    /* Replicate first values for azimuth data */
                    pY1[j]         = pY1[0];
                    pY2[j]         = pY2[0];
                    pY3[j]         = pY3[0];
                    pshiftedIdx[j] = pshiftedIdx[0];

                    /* Replicate first values for elevation data (offset by OUTPUT_BATCH_SIZE) */
                    pY1[j + OUTPUT_BATCH_SIZE]         = pY1[OUTPUT_BATCH_SIZE];
                    pY2[j + OUTPUT_BATCH_SIZE]         = pY2[OUTPUT_BATCH_SIZE];
                    pY3[j + OUTPUT_BATCH_SIZE]         = pY3[OUTPUT_BATCH_SIZE];
                    pshiftedIdx[j + OUTPUT_BATCH_SIZE] = pshiftedIdx[OUTPUT_BATCH_SIZE];
                }
            }

            /* Compute interpolated indices using quadratic interpolation on the batch */
            int32_t i = y / OUTPUT_BATCH_SIZE;
            compute_indices(&params_doa_angle_fft[i & 1], &pY1[0], &pY2[0], &pY3[0], &pshiftedIdx[0],
                            &interpAzimuthIndex[0], &interpElevationIndex[0]);

            /* Convert interpolated indices to angle values (degrees) and power (dB) */
            compute_angles(&params_doa_angle_fft[i & 1], &interpAzimuthIndex[0], &interpElevationIndex[0], &max_val[0],
                           &peakAzimuth[0], &peakElevation[0], y_iter);

            /* Wait for previous output transfer to complete (except first iteration) */
            if (y)
            {
                cupvaSQDFSync(dstTargetAnglesHdl);
                cupvaSQDFSync(dstTargetIndexMapHdl);
            }

            /* Start transferring computed results to DRAM */
            updateOutputSQDF(dstTargetAnglesHdl, dstTargetIndexMapHdl, i, 0, y_iter, &params_doa_angle_fft[i & 1]);
        }
        /* Wait for final output transfer to complete */
        cupvaSQDFSync(dstTargetAnglesHdl);
        cupvaSQDFSync(dstTargetIndexMapHdl);
    }

    /* Close all input/output data flow handlers */
    cupvaSQDFClose(srcSnapshotsHdl);
    cupvaSQDFClose(dstTargetAnglesHdl);
    cupvaSQDFClose(dstTargetIndexMapHdl);

    /* Write final target count to output */
    dstTargetCountBuf[0] = targetIdx; // Total number of targets processed
    cupvaSQDFOpen(dstTargetCountHdl);
    cupvaSQDFTrig(dstTargetCountHdl);
    cupvaSQDFSync(dstTargetCountHdl);
    cupvaSQDFClose(dstTargetCountHdl);

    return 0;
}
