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

#include "doa_target_processing_ref.h"

#include "doa_target_processing_impl.hpp"

#include <ImageIO.h>
//#include <PvaOperator.h>
//#include <PvaOperatorTypes.h>
#include <cupva_host.h>

#include <algorithm> // For std::max_element, std::distance
#include <cmath>     // For std::roundf, std::abs (for float), std::arg, std::exp, std::conj
#include <complex>
#include <fstream>
#include <iostream> // For std::cerr warning messages
#include <memory>
#include <numeric>   // For std::iota in some contexts if needed, though not directly here
#include <stdexcept> // For std::runtime_error (optional, for more critical errors)
#include <string>
#include <vector>

namespace {

const uint32_t aIndicesTable[AINDICES_WIDTH] = {
#include "doa_aindices_table.h"
};

// Helper function to determine the number of bits needed to represent n
int bit_length(int n)
{
    if (n <= 1)
    {
        return 0;
    }
    return (int)ceil(log2(n));
}

// Radix-4 digit reversal function
int r4_digit_reverse(int i, int log2n)
{
    int log4n = log2n / 2;
    int res   = 0;
    for (int k = 0; k < log4n; k++)
    {
        res <<= 2;
        res |= (i & 3);
        i >>= 2;
    }
    return res;
}

// Single stage of Radix-4 DIF FFT
void fft_fixedp_r4_dif_stage(int n, int k, int64_t *x_interleaved, int64_t *twiddle_interleaved, int64_t *y)
{
    int log2n   = bit_length(n);
    int ngroups = 1 << (2 * k);
    int groupsz = n / ngroups;
    int dist    = groupsz / 4;

    // Use Q30 format for twiddle factors
    int32_t qbits = (TWIDDLE_QBITS - 16);

    for (int i = 0; i < ngroups; i++)
    {
        for (int j = 0; j < dist; j++)
        {
            int didx = i * groupsz + j;

            // Twiddle indices
            int t1idx = j * (1 << (2 * k));
            int t2idx = 2 * t1idx;
            int t3idx = 3 * t1idx;

            // Read input data (Q20 format)
            int64_t x0r = x_interleaved[2 * didx];
            int64_t x0i = x_interleaved[2 * didx + 1];
            int64_t x1r = x_interleaved[2 * (didx + dist)];
            int64_t x1i = x_interleaved[2 * (didx + dist) + 1];
            int64_t x2r = x_interleaved[2 * (didx + 2 * dist)];
            int64_t x2i = x_interleaved[2 * (didx + 2 * dist) + 1];
            int64_t x3r = x_interleaved[2 * (didx + 3 * dist)];
            int64_t x3i = x_interleaved[2 * (didx + 3 * dist) + 1];

            // Fetch twiddle factors (Q30 format)
            int64_t t1r = twiddle_interleaved[2 * t1idx];
            int64_t t1i = twiddle_interleaved[2 * t1idx + 1];
            int64_t t2r = twiddle_interleaved[2 * t2idx];
            int64_t t2i = twiddle_interleaved[2 * t2idx + 1];
            int64_t t3r = twiddle_interleaved[2 * t3idx];
            int64_t t3i = twiddle_interleaved[2 * t3idx + 1];

            // Butterfly intermediate values
            int64_t z0r = x0r + x2r;
            int64_t z0i = x0i + x2i;
            int64_t z1r = x0r - x2r;
            int64_t z1i = x0i - x2i;
            int64_t z2r = x1r + x3r;
            int64_t z2i = x1i + x3i;
            int64_t z3r = x1r - x3r;
            int64_t z3i = x1i - x3i;

            // Butterfly outputs before twiddle multiplication
            int64_t w0r = z0r + z2r;
            int64_t w0i = z0i + z2i;
            int64_t w2r = z0r - z2r;
            int64_t w2i = z0i - z2i;
            int64_t w1r = z1r + z3i; // w1 = z1 - j*z3
            int64_t w1i = z1i - z3r;
            int64_t w3r = z1r - z3i; // w3 = z1 + j*z3
            int64_t w3i = z1i + z3r;

            // Apply twiddle factors and scale
            // For w0 (no twiddle factor), keep Q20 format
            y[2 * didx]     = w0r;
            y[2 * didx + 1] = w0i;

            // For w1, w2, w3: complex multiplication with Q30 twiddle factors
            y[2 * (didx + dist)]     = my_round((w1r * t1r >> 16) - (w1i * t1i >> 16), qbits);
            y[2 * (didx + dist) + 1] = my_round((w1r * t1i >> 16) + (w1i * t1r >> 16), qbits);

            y[2 * (didx + 2 * dist)]     = my_round((w2r * t2r >> 16) - (w2i * t2i >> 16), qbits);
            y[2 * (didx + 2 * dist) + 1] = my_round((w2r * t2i >> 16) + (w2i * t2r >> 16), qbits);

            y[2 * (didx + 3 * dist)]     = my_round((w3r * t3r >> 16) - (w3i * t3i >> 16), qbits);
            y[2 * (didx + 3 * dist) + 1] = my_round((w3r * t3i >> 16) + (w3i * t3r >> 16), qbits);
        }
    }
}

// Full Radix-4 DIF FFT
void fft_fixedp_r4_dif(int n, int64_t *x_interleaved)
{
    int log2n = bit_length(n);
    if (log2n % 2 != 0)
    {
        printf("Error: FFT size 'n' must be a power of 4.\n");
        return;
    }
    int log4n             = log2n / 2;
    int64_t twiddle_scale = TWIDDLE_SCALE;

    // Pre-compute twiddle factors
    std::vector<int64_t> twiddle_interleaved(2 * n);
    for (int i = 0; i < n; i++)
    {
        double angle                   = -2.0 * PI_CONST_F * i / n; // Negative sign for forward FFT
        twiddle_interleaved[2 * i]     = (int64_t)round(cos(angle) * twiddle_scale);
        twiddle_interleaved[2 * i + 1] = (int64_t)round(sin(angle) * twiddle_scale); // Positive sin for forward FFT
    }

    // Allocate working buffers
    std::vector<int64_t> stage_data(2 * n);
    std::vector<int64_t> temp_data(2 * n);

    // Copy input to working buffer (input is Q20, but we'll process as Q20)
    for (int i = 0; i < 2 * n; i++)
    {
        stage_data[i] = x_interleaved[i];
    }

    // Process FFT stages
    for (int k = 0; k < log4n; k++)
    {
        fft_fixedp_r4_dif_stage(n, k, stage_data.data(), twiddle_interleaved.data(), temp_data.data());
        // Swap buffers
        stage_data.swap(temp_data);
    }

    // Final permutation (Radix-4 digit reversal)
    for (int i = 0; i < n; i++)
    {
        int j                    = r4_digit_reverse(i, log2n);
        x_interleaved[2 * i]     = stage_data[2 * j];
        x_interleaved[2 * i + 1] = stage_data[2 * j + 1];
    }

    // std::vector objects are automatically deallocated when they go out of scope
}

// Quadratic interpolation function qint_cpp
void qint_cpp(float y_m1, float y_0, float y_p1, float &p, float &y, float &a)
{
    float den = y_m1 - 2.0f * y_0 + y_p1;
    if (std::abs(den) < 1e-9f)
    {
        // Avoid division by zero or near-zero (linear case)
        p = 0.0f;
        y = y_0;
        a = 0.0f;
    }
    else
    {
        p = (y_m1 - y_p1) / (2.0f * den);    // Fractional offset from center point (x=0) to extremum
        y = y_0 - 0.25f * (y_m1 - y_p1) * p; // Interpolated y-value at extremum
        a = 0.5f * den;                      // Related to quadratic coefficient (0.5 * d^2y/dx^2 for x steps of 1)
    }
}

inline std::vector<std::pair<int32_t, int32_t>> radix4_fft_fixed(std::vector<std::pair<int32_t, int32_t>> &input)
{
    int n = input.size();
    if (n <= 1)
    {
        return input;
    }

    // Check if n is a power of 4 (required for radix-4 FFT)
    int log2n = bit_length(n);
    if (log2n % 2 != 0)
    {
        std::cerr << "Error: FFT size must be a power of 4 for radix-4 FFT. Got: " << n << std::endl;
        return input;
    }

    // Allocate interleaved buffer for fft_fixedp_r4_dif
    std::vector<int64_t> x_interleaved(2 * n);

    // Copy Q20 input directly to int64_t arrays (matching radix2_fft_fixed behavior)
    for (int i = 0; i < n; i++)
    {
        x_interleaved[2 * i]     = static_cast<int64_t>(input[i].first);  // real part (Q20)
        x_interleaved[2 * i + 1] = static_cast<int64_t>(input[i].second); // imaginary part (Q20)
    }

    // Perform radix-4 DIF FFT (expects Q20 input, returns Q20 output)
    fft_fixedp_r4_dif(n, x_interleaved.data());

    // Convert back to Q20 format and return (matching radix2_fft_fixed behavior)
    std::vector<std::pair<int32_t, int32_t>> result(n);
    for (int i = 0; i < n; i++)
    {
        // Convert from int64_t back to int32_t (Q20 format)
        int32_t real_part = static_cast<int32_t>(std::clamp<int64_t>(x_interleaved[2 * i], INT32_MIN, INT32_MAX));
        int32_t imag_part = static_cast<int32_t>(std::clamp<int64_t>(x_interleaved[2 * i + 1], INT32_MIN, INT32_MAX));
        result[i]         = std::make_pair(real_part, imag_part);
    }

    return result;
}

TAR_t_cpp doaTargetProcessingFixedPointRef(PVARadarGP const *const GP, GP_t_cpp const *const GP_cpp,
                                           std::pair<int32_t, int32_t> const *const calibVector,
                                           PeakIndices_t const *const PeakIndices,
                                           std::pair<int32_t, int32_t> const *const PeakSnaps,
                                           uint32_t const *const NCI_FINAL)
{
    TAR_t_cpp TAR_result;
    TAR_result.N = 0;
    int NofFFT   = static_cast<int>(std::roundf(GP->nofAFFT)); // Use roundf for float
    if (NofFFT <= 0)
    {
        std::cerr << "Error: NofAFFT must be positive." << std::endl;
        return TAR_result; // Return empty TAR
    }

    if (GP->nofSamples == 0)
    {
        std::cerr << "Error: GP.NofSamples is zero, division by zero for Delta_f_fast." << std::endl;
        return TAR_result;
    }
    float Delta_f_fast = GP->fs / GP->nofSamples;
    if (GP->PRI == 0.0f || GP->nofRamps == 0)
    {
        std::cerr << "Error: GP.PRI or GP.NofRamps is zero, division by zero for Delta_f_slow." << std::endl;
        return TAR_result;
    }
    float Delta_f_slow = 1.0f / (GP->PRI * GP->nofRamps);

    float velocityAmbHigh = -1.0f * GP->deltaV * static_cast<float>(GP->nofRamps) / 2.0f / 3.6f;
    float velocityAmbLow  = GP->deltaV * static_cast<float>(GP->nofRamps) / 2.0f / 3.6f;

    size_t nci_num_rows = GP_cpp->nciFinalHeight;
    size_t nci_num_cols = 0;
    if (nci_num_rows > 0)
    {
        nci_num_cols = GP_cpp->nciFinalWidth;
    }

    float *Aindex = (float *)aIndicesTable;
    for (int ipeak_idx = 0; ipeak_idx < GP_cpp->numPeaks; ++ipeak_idx)
    {
        if (TAR_result.N >= DOA_MAX_TARGET_COUNT)
        {
            std::cerr << "Warning: Maximum number of targets (" << DOA_MAX_TARGET_COUNT
                      << ") reached. Skipping remaining peaks." << std::endl;
            break;
        }

        std::vector<std::pair<int32_t, int32_t>> Snap_processed(GP->nofSnap);
        std::pair<int32_t, int32_t> const *currSnaps = &PeakSnaps[ipeak_idx * GP->nofSnap];
        for (size_t i = 0; i < GP->nofSnap; ++i)
        {
            // Both PeakSnaps and calibVector are already in Q15 format
            auto q15_result = complex_mult_fixed(currSnaps[i], calibVector[i], GP->calibVectorQbits);
            // Take conjugate in Q15 fixed-point
            auto q15_conj_result = complex_conj_fixed(q15_result);
            Snap_processed[i]    = q15_conj_result;
        }

        std::vector<std::pair<int32_t, int32_t>> Upper(NofFFT, {0, 0});
        std::vector<std::pair<int32_t, int32_t>> Lower(NofFFT, {0, 0});

        for (size_t i = 0; i < GP->nofAperture; ++i)
        {
            int upper_idx = GP->apUpperInd[i];
            int snap_idx  = GP->apUpper[i];
            if (upper_idx >= 0 && upper_idx < NofFFT && snap_idx >= 0 &&
                static_cast<size_t>(snap_idx) < Snap_processed.size())
            {
                Upper[upper_idx] = Snap_processed[snap_idx];
            }
            else
            {
                std::cerr << "Warning (peak " << ipeak_idx
                          << "): Out of bounds access for Upper array assignment. Check AP_UPPER_IND/AP_UPPER values."
                          << std::endl;
            }
        }

        for (size_t i = 0; i < GP->nofAperture; ++i)
        {
            int lower_idx = GP->apLowerInd[i];
            int snap_idx  = GP->apLower[i];
            if (lower_idx >= 0 && lower_idx < NofFFT && snap_idx >= 0 &&
                static_cast<size_t>(snap_idx) < Snap_processed.size())
            {
                Lower[lower_idx] = Snap_processed[snap_idx];
            }
            else
            {
                std::cerr << "Warning (peak " << ipeak_idx
                          << "): Out of bounds access for Lower array assignment. Check AP_LOWER_IND/AP_LOWER values."
                          << std::endl;
            }
        }

        std::vector<std::pair<int32_t, int32_t>> AFFT_Upper, AFFT_Lower;
        AFFT_Upper = radix4_fft_fixed(Upper);
        AFFT_Lower = radix4_fft_fixed(Lower);

        if (AFFT_Upper.size() != static_cast<size_t>(NofFFT) || AFFT_Lower.size() != static_cast<size_t>(NofFFT))
        {
            std::cerr << "Error (peak " << ipeak_idx << "): FFT output size mismatch. Expected " << NofFFT
                      << ", Got Upper: " << AFFT_Upper.size() << ", Lower: " << AFFT_Lower.size() << ". Skipping peak."
                      << std::endl;
            continue;
        }

        std::vector<int32_t> NCIAFFT(NofFFT);
        for (int i = 0; i < NofFFT; ++i)
        {
            // Use 64-bit intermediate to prevent overflow/underflow
            int64_t upper_mag = complex_abs_fixed(AFFT_Upper[i]);
            int64_t lower_mag = complex_abs_fixed(AFFT_Lower[i]);
            int64_t sum       = upper_mag + lower_mag;
            // Clamp to INT32 range to handle both overflow and underflow
            NCIAFFT[i] = static_cast<int32_t>(std::clamp<int64_t>(sum, INT32_MIN, INT32_MAX));
        }

        auto max_it_nciafft = std::max_element(NCIAFFT.begin(), NCIAFFT.end());
        int mi_nciafft      = std::distance(NCIAFFT.begin(), max_it_nciafft);

        if (static_cast<size_t>(mi_nciafft) >= AFFT_Upper.size())
        {
            std::cerr << "Error (peak " << ipeak_idx
                      << "): mi_nciafft index out of bounds for AFFT arrays. Skipping peak." << std::endl;
            continue;
        }

        // Calculate phase difference using fixed-point math
        auto upper_int32         = AFFT_Upper[mi_nciafft];
        auto lower_int32         = complex_conj_fixed(AFFT_Lower[mi_nciafft]);
        auto phase_prod_int64    = complex_mult_fixed_int64(upper_int32, lower_int32, GP->fftQbits);
        int32_t phase_diff_int32 = calculate_phase_angle_fixed(phase_prod_int64, GP->fftQbits);
        float phase_diff         = q15_to_float(phase_diff_int32) * PI_CONST_F / 180.0f;

        // Calculate phase factor in Q15
        int32_t cos_phase_int32                        = float_to_int32(std::cos(phase_diff), GP->fftQbits);
        int32_t sin_phase_int32                        = float_to_int32(std::sin(phase_diff), GP->fftQbits);
        std::pair<int32_t, int32_t> phase_factor_int32 = {cos_phase_int32, sin_phase_int32};

        std::vector<int32_t> AFFT_final_abs(NofFFT);
        for (int i = 0; i < NofFFT; ++i)
        {
            // Multiply Lower by phase factor in Q15
            auto lower_phase_int32 = complex_mult_fixed(AFFT_Lower[i], phase_factor_int32, GP->fftQbits);
            // Add Upper and phase-adjusted Lower in Q15
            auto sum_int32 = std::make_pair(AFFT_Upper[i].first + lower_phase_int32.first,
                                            AFFT_Upper[i].second + lower_phase_int32.second);

            // Calculate magnitude using fixed-point math
            AFFT_final_abs[i] = complex_abs_fixed(sum_int32);
        }

        auto max_it_afft = std::max_element(AFFT_final_abs.begin(), AFFT_final_abs.end());
        int mic_afft_idx = std::distance(AFFT_final_abs.begin(), max_it_afft);

        if (static_cast<size_t>(mic_afft_idx) >= NofFFT)
        {
            std::cerr << "Error (peak " << ipeak_idx << "): mic_afft_idx out of bounds for Aindex. Skipping peak."
                      << std::endl;
            continue;
        }
        float azimuthAngle = Aindex[mic_afft_idx];

        if (GP->disE == 0.0f)
        {
            std::cerr << "Error (peak " << ipeak_idx
                      << "): GP.DisE is zero, division by zero for elevationAngle. Skipping peak." << std::endl;
            continue;
        }
        // asind_cpp and PI_CONST_F are float versions from header
        float eleval              = phase_diff / (2.0f * PI_CONST_F * GP->disE);
        int32_t eleval_q22        = (eleval * (1 << 22));
        int32_t elevationAngleInt = arcsin_lookup_fixed(eleval_q22);
        float elevationAngle      = q15_to_float(elevationAngleInt);
        int Rind_0idx             = PeakIndices->Rind[ipeak_idx];
        int DFoldind_0idx         = PeakIndices->DFoldind[ipeak_idx];

        int nci_row_pre_r_0idx  = Rind_0idx - 1;
        int nci_row_ind_r_0idx  = Rind_0idx;
        int nci_row_post_r_0idx = Rind_0idx + 1;
        int nci_col_r_0idx      = DFoldind_0idx;

        if (nci_num_cols > 0 && nci_col_r_0idx >= 0)
        {
            nci_col_r_0idx = nci_col_r_0idx % nci_num_cols;
        }

        if (nci_row_pre_r_0idx < 0 || static_cast<size_t>(nci_row_pre_r_0idx) >= nci_num_rows ||
            nci_row_ind_r_0idx < 0 || static_cast<size_t>(nci_row_ind_r_0idx) >= nci_num_rows ||
            nci_row_post_r_0idx < 0 || static_cast<size_t>(nci_row_post_r_0idx) >= nci_num_rows || nci_col_r_0idx < 0 ||
            static_cast<size_t>(nci_col_r_0idx) >= nci_num_cols)
        {
            std::cerr << "Error (peak " << ipeak_idx
                      << "): NCI_FINAL index out of bounds for range interpolation. Rind_0idx: " << Rind_0idx
                      << ", DFoldind_0idx: " << DFoldind_0idx << ". Skipping peak." << std::endl;
            continue;
        }
        uint32_t const *NCI_FINAL_ptr = NCI_FINAL + nci_row_pre_r_0idx * GP_cpp->nciFinalWidth;
        float Y_m1_r                  = (float)NCI_FINAL_ptr[nci_col_r_0idx];
        NCI_FINAL_ptr += GP_cpp->nciFinalWidth;
        float Y_0_r = (float)NCI_FINAL_ptr[nci_col_r_0idx];
        NCI_FINAL_ptr += GP_cpp->nciFinalWidth;
        float Y_p1_r = (float)NCI_FINAL_ptr[nci_col_r_0idx];

        float p_r, y_r_interp, a_r;
        qint_cpp(Y_m1_r, Y_0_r, Y_p1_r, p_r, y_r_interp, a_r);

        float X_0_r_val_for_interp = static_cast<float>(Rind_0idx);
        float iXr_0based           = X_0_r_val_for_interp + p_r;

        int DFoldind_curr_0idx = PeakIndices->DFoldind[ipeak_idx];
        int Dind_total_0idx    = PeakIndices->Dind[ipeak_idx];

        if (GP->repeatFold == 0)
        {
            std::cerr << "Error (peak " << ipeak_idx
                      << "): RepeatFold is zero, division by zero for Doppler mod. Skipping peak." << std::endl;
            continue;
        }
        // Use std::roundf for float
        int Nr_div_RepeatFold =
            static_cast<int>(std::roundf(static_cast<float>(GP->nr) / static_cast<float>(GP->repeatFold)));
        if (Nr_div_RepeatFold == 0)
        {
            std::cerr << "Error (peak " << ipeak_idx << "): Nr/RepeatFold is zero for Doppler mod. Skipping peak."
                      << std::endl;
            continue;
        }
        int N_mod = Nr_div_RepeatFold;

        int nci_col_ind_d_0idx  = DFoldind_curr_0idx;
        int nci_col_pre_d_0idx  = (DFoldind_curr_0idx - 1 + N_mod) % N_mod;
        int nci_col_post_d_0idx = (DFoldind_curr_0idx + 1) % N_mod;

        int nci_row_d_0idx = Rind_0idx;

        if (nci_row_d_0idx < 0 || static_cast<size_t>(nci_row_d_0idx) >= nci_num_rows || nci_col_pre_d_0idx < 0 ||
            static_cast<size_t>(nci_col_pre_d_0idx) >= nci_num_cols || nci_col_ind_d_0idx < 0 ||
            static_cast<size_t>(nci_col_ind_d_0idx) >= nci_num_cols || nci_col_post_d_0idx < 0 ||
            static_cast<size_t>(nci_col_post_d_0idx) >= nci_num_cols)
        {
            std::cerr << "Error (peak " << ipeak_idx
                      << "): NCI_FINAL index out of bounds for Doppler interpolation. Rind_0idx: " << Rind_0idx
                      << ", DFoldind_curr_0idx: " << DFoldind_curr_0idx << ". Skipping peak." << std::endl;
            continue;
        }
        uint32_t const *NCI_FINAL_doppler_ptr = NCI_FINAL + nci_row_d_0idx * GP_cpp->nciFinalWidth;
        float Y_m1_d                          = (float)NCI_FINAL_doppler_ptr[nci_col_pre_d_0idx];
        float Y_0_d                           = (float)NCI_FINAL_doppler_ptr[nci_col_ind_d_0idx];
        float Y_p1_d                          = (float)NCI_FINAL_doppler_ptr[nci_col_post_d_0idx];

        float p_d, y_d_interp, a_d;
        qint_cpp(Y_m1_d, Y_0_d, Y_p1_d, p_d, y_d_interp, a_d);
        float iXd_0based = static_cast<float>(Dind_total_0idx) + p_d;

        float f_fast = Delta_f_fast * iXr_0based;
        float f_slow = Delta_f_slow * iXd_0based;

        float TempV      = 0.0f;
        bool tempV_found = false;
        if (GP->PRI == 0.0f)
        {
            std::cerr << "Error (peak " << ipeak_idx
                      << "): GP.PRI is zero, division by zero for TempV calc. Skipping peak." << std::endl;
            continue;
        }
        for (int jj_loop = 0; jj_loop < 5; ++jj_loop)
        {
            TempV = f_fast * GP->contVFast -
                    (f_slow + (1.0f / GP->PRI) * (static_cast<float>(jj_loop) - 3.0f)) * GP->contVSlow;
            if (TempV >= velocityAmbHigh && TempV <= velocityAmbLow)
            {
                tempV_found = true;
                break;
            }
        }
        if (!tempV_found)
        {
            std::cerr << "Warning (peak " << ipeak_idx
                      << "): TempV not found within ambiguity range. Using last calculated value: " << TempV
                      << std::endl;
        }

        float TempR = (f_fast - f_slow) * GP->contR;

        TAR_DET_t_cpp new_target_det; // Already float members
        new_target_det.velocity = TempV * 3.6f;
        new_target_det.range    = TempR;
        new_target_det.A        = azimuthAngle;
        new_target_det.E        = elevationAngle;

        if (static_cast<size_t>(mic_afft_idx) >= NofFFT)
        {
            std::cerr << "Error (peak " << ipeak_idx
                      << "): mic_afft_idx out of bounds for Aindex for target storage. Skipping target." << std::endl;
            continue;
        }
        // sind_cpp and cosd_cpp are float versions from header
        new_target_det.X = TempR * sind_cpp(Aindex[mic_afft_idx]);
        new_target_det.Y = TempR * cosd_cpp(Aindex[mic_afft_idx]);
        // elevationAngle is in degrees as per asind_cpp. std::sinf expects radians.
        // Original: std::sin(elevationAngle) - if elevationAngle was degrees, this was wrong.
        // Assuming elevationAngle needs to be converted to radians for std::sinf, or sind_cpp used.
        // If elevationAngle from asind_cpp output is already in radians due to PI_CONST_F usage, then std::sinf is fine.
        // The helper asind_cpp converts to degrees. So to use std::sinf, it needs to be converted back to radians.
        // Or, more simply, use sind_cpp for consistency if it's intended for degree inputs.
        // Z = R * sin(Elev_degrees)
        new_target_det.Z = TempR * sind_cpp(elevationAngle); // sind_cpp(elevationAngle);

        TAR_result.DET[TAR_result.N] = new_target_det;
        TAR_result.N++;
    }

    return TAR_result;
}

void doaTargetProcessingImpl(NVCVTensorHandle *inTensorHandles, NVCVTensorHandle outTensorHandle, PVARadarGP *GP)
{
    NVCVTensorData inTensorData[5];
    void *inTensorPtr[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 5; ++i)
    {
        nvcvTensorExportData(inTensorHandles[i], &inTensorData[i]);
        CupvaMemGetHostPointer((void **)&inTensorPtr[i], (void *)inTensorData[i].buffer.strided.basePtr);
    }

    int32_t numPeaks = *static_cast<int32_t *>(inTensorPtr[0]);

    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    void *outTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);

    int32_t PeakIndicesWidth  = inTensorData[2].shape[1];
    int32_t peakIndicesHeight = inTensorData[2].shape[0];
    int32_t peakSnapWidth     = inTensorData[3].shape[1];
    int32_t peakSnapHeight    = inTensorData[3].shape[0];
    int32_t nciFinalWidth     = inTensorData[4].shape[1];
    int32_t nciFinalHeight    = inTensorData[4].shape[0];
    int32_t numTargets        = outTensorData.shape[1];
    int32_t targetHeight      = outTensorData.shape[0];

    GP_t_cpp GP_cpp{};
    GP_cpp.numPeaks          = numPeaks;
    GP_cpp.peakIndicesWidth  = PeakIndicesWidth;
    GP_cpp.peakIndicesHeight = peakIndicesHeight;
    GP_cpp.peakSnapWidth     = peakSnapWidth;
    GP_cpp.peakSnapHeight    = peakSnapHeight;
    GP_cpp.nciFinalWidth     = nciFinalWidth;
    GP_cpp.nciFinalHeight    = nciFinalHeight;
    GP_cpp.numTargets        = numTargets;
    GP_cpp.targetHeight      = targetHeight;

    TAR_t_cpp TAR_result{};
    TAR_result.N = numPeaks;

    PeakIndices_t peak_indices;
    peak_indices.Rind     = static_cast<uint32_t *>(inTensorPtr[2]);
    peak_indices.DFoldind = peak_indices.Rind + PeakIndicesWidth;
    peak_indices.Dind     = peak_indices.DFoldind + PeakIndicesWidth;

    std::vector<std::pair<int32_t, int32_t>> calibVectorPermuted(16);
    int32_t *calibTensorPtr = (int32_t *)inTensorPtr[1];
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            calibVectorPermuted[i * 8 + j].first  = static_cast<int32_t>(calibTensorPtr[i * 16 + j]);
            calibVectorPermuted[i * 8 + j].second = static_cast<int32_t>(calibTensorPtr[i * 16 + j + 8]);
        }
    }

    TAR_result = doaTargetProcessingFixedPointRef(GP, &GP_cpp, calibVectorPermuted.data(), &peak_indices,
                                                  (std::pair<int32_t, int32_t> const *)inTensorPtr[3],
                                                  (uint32_t const *)inTensorPtr[4]);

    float *outVelocityPtrFloat  = (float *)outTensorPtr;
    float *outRangePtrFloat     = (float *)outTensorPtr + numTargets;
    float *outAzimuthPtrFloat   = (float *)outTensorPtr + numTargets * 2;
    float *outElevationPtrFloat = (float *)outTensorPtr + numTargets * 3;
    float *outXPtrFloat         = (float *)outTensorPtr + numTargets * 4;
    float *outYPtrFloat         = (float *)outTensorPtr + numTargets * 5;
    float *outZPtrFloat         = (float *)outTensorPtr + numTargets * 6;
    for (int i = 0; i < numPeaks; ++i)
    {
        outVelocityPtrFloat[i]  = TAR_result.DET[i].velocity;
        outRangePtrFloat[i]     = TAR_result.DET[i].range;
        outAzimuthPtrFloat[i]   = TAR_result.DET[i].A;
        outElevationPtrFloat[i] = TAR_result.DET[i].E;
        outXPtrFloat[i]         = TAR_result.DET[i].X;
        outYPtrFloat[i]         = TAR_result.DET[i].Y;
        outZPtrFloat[i]         = TAR_result.DET[i].Z;
    }
}

} // namespace

void populateGP(PVARadarGP *GP, int32_t calibVectorQbits, int32_t fftQbits)
{
    if (GP == nullptr)
    {
        std::cout << "Error: GP is nullptr" << std::endl;
        return;
    }

    // Populate GP_params (GP_t_cpp) using pre-computed values. The range/velocity decoupling constants are those
    // of the 4T4R waveform (f0 79 GHz, B 150 MHz over Tchirp 20.48 us, deltaF 1 MHz ramp-to-ramp step) with
    // c = 299792458 m/s; denom = B * PRI - deltaF * Tchirp.
    GP->nofAFFT          = 256;
    GP->nofSnap          = 16;
    GP->dis              = 0.5f;                  // from old GP.Dis
    GP->fs               = 25000000.0f;           // from old GP.fs
    GP->nofSamples       = 512;                   // from old GP.NofSamples
    GP->PRI              = 2.798000059556216e-05; // from old GP.PRI
    GP->nofRamps         = 512;                   // from old GP.NofRamps (was in old GP struct)
    GP->deltaV           = 0.47681338;            // c/2 / f0 / PRI / nofRamps * 3.6 (km/h)
    GP->disE             = 0.75f;                 // from old GP.DisE
    GP->contR            = 2.0566188e-05;         // c/2 * Tchirp * PRI / denom
    GP->contVFast        = -9.3041991e-06;        // -c/2 / f0 * Tchirp * deltaF / denom
    GP->contVSlow        = -0.0019067248;         // -c/2 / f0 * PRI * B / denom
    GP->nofAperture      = 8;
    GP->repeatFold       = 8;
    GP->nr               = 512;
    GP->calibVectorQbits = calibVectorQbits;
    GP->fftQbits         = fftQbits;

    int32_t ap_upper_orig[]     = {0, 3, 4, 7, 8, 11, 12, 15};
    int32_t ap_lower_orig[]     = {1, 2, 5, 6, 9, 10, 13, 14};
    int32_t ap_upper_ind_orig[] = {0, 9, 4, 13, 8, 17, 12, 21};
    int32_t ap_lower_ind_orig[] = {3, 6, 7, 10, 11, 14, 15, 18};

    std::copy(ap_upper_orig, ap_upper_orig + 8, GP->apUpper);
    std::copy(ap_lower_orig, ap_lower_orig + 8, GP->apLower);
    std::copy(ap_upper_ind_orig, ap_upper_ind_orig + 8, GP->apUpperInd);
    std::copy(ap_lower_ind_orig, ap_lower_ind_orig + 8, GP->apLowerInd);
}

void populateCalibVector(int32_t *calibVector, int32_t qbits)
{
    const int32_t calibVector_orig[] = {
        // Real part
        float_to_int32(0.95419002f, qbits),
        float_to_int32(-0.34347999f, qbits),
        float_to_int32(-0.083509997f, qbits),
        float_to_int32(0.89093000f, qbits),
        float_to_int32(0.62287998f, qbits),
        float_to_int32(0.42886999f, qbits),
        float_to_int32(0.59582001f, qbits),
        float_to_int32(0.14289001f, qbits),

        // Imaginary part
        float_to_int32(-0.15831999f, qbits),
        float_to_int32(-0.82799000f, qbits),
        float_to_int32(-0.77504998f, qbits),
        float_to_int32(0.39897999f, qbits),
        float_to_int32(0.65314001f, qbits),
        float_to_int32(-0.73414999f, qbits),
        float_to_int32(-0.49675000f, qbits),
        float_to_int32(0.94439000f, qbits),

        // Real part
        float_to_int32(-0.70082003f, qbits),
        float_to_int32(0.76574999f, qbits),
        float_to_int32(0.42387000f, qbits),
        float_to_int32(-0.97081000f, qbits),
        float_to_int32(0.21018000f, qbits),
        float_to_int32(-1.0f, qbits),
        float_to_int32(-0.78645998f, qbits),
        float_to_int32(0.77932000f, qbits),

        // Imaginary part
        float_to_int32(0.57045001f, qbits),
        float_to_int32(0.52064002f, qbits),
        float_to_int32(0.64955002f, qbits),
        float_to_int32(-0.0039901999f, qbits),
        float_to_int32(-0.98920000f, qbits),
        float_to_int32(-0.029625000f, qbits),
        float_to_int32(-0.27674001f, qbits),
        float_to_int32(-0.60835999f, qbits),
    };

    memcpy(calibVector, calibVector_orig, sizeof(calibVector_orig));
}

void populateIdentityCalibVector(int32_t *calibVector, int32_t qbits)
{
    // Two blocks of {8 real, 8 imaginary} values, as populateCalibVector()
    for (int32_t i = 0; i < PVA_RADAR_NUM_TOTAL_ANTENNA_ELEMENTS * 2; i++)
    {
        calibVector[i] = ((i / 8) % 2 == 0) ? float_to_int32(1.0f, qbits) : 0;
    }
}

int32_t loadPeakCountToTensor(NVCVTensorHandle peakCountHandle, int32_t numPeaks)
{
    NVCVTensorData peakCountTensorData;
    nvcvTensorExportData(peakCountHandle, &peakCountTensorData);
    void *peakCountPtr = nullptr;
    CupvaMemGetHostPointer((void **)&peakCountPtr, (void *)peakCountTensorData.buffer.strided.basePtr);

    *static_cast<int32_t *>(peakCountPtr) = numPeaks;

    return 0;
}

/**
 * @brief Writes TAR results to a raw file.
 *
 * @param outTensorHandle The handle to the output tensor containing TAR results.
 * @param GP Pointer to the GP parameters structure containing number of peaks.
 * @param outputFile The name of the output file (default: "tar_results.dat").
 * @return Returns 0 on success, -1 on failure.
 */
int32_t writeTarResultsToFile(NVCVTensorHandle outTensorHandle, int32_t numPeaks, const char *outputFile)
{
    NVCVTensorData outTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    void *outTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);
    int32_t numTargets = outTensorData.shape[1];

    float *outVelocityPtrFloat  = (float *)outTensorPtr;
    float *outRangePtrFloat     = (float *)outTensorPtr + numTargets;
    float *outAzimuthPtrFloat   = (float *)outTensorPtr + numTargets * 2;
    float *outElevationPtrFloat = (float *)outTensorPtr + numTargets * 3;
    float *outXPtrFloat         = (float *)outTensorPtr + numTargets * 4;
    float *outYPtrFloat         = (float *)outTensorPtr + numTargets * 5;
    float *outZPtrFloat         = (float *)outTensorPtr + numTargets * 6;

    // Create TAR_t_cpp structure for writing to file
    TAR_t_cpp tar_data;
    tar_data.N = numPeaks;
    for (int i = 0; i < tar_data.N; ++i)
    {
        tar_data.DET[i].velocity = outVelocityPtrFloat[i];
        tar_data.DET[i].range    = outRangePtrFloat[i];
        tar_data.DET[i].A        = outAzimuthPtrFloat[i];
        tar_data.DET[i].E        = outElevationPtrFloat[i];
        tar_data.DET[i].X        = outXPtrFloat[i];
        tar_data.DET[i].Y        = outYPtrFloat[i];
        tar_data.DET[i].Z        = outZPtrFloat[i];

        // Print target data for debugging
        printf(
            "Target %d: Velocity=%.3f km/h, Range=%.3f m, Azimuth=%.3f°, Elevation=%.3f°, X=%.3f m, Y=%.3f m, Z=%.3f "
            "m\n",
            i, tar_data.DET[i].velocity, tar_data.DET[i].range, tar_data.DET[i].A, tar_data.DET[i].E, tar_data.DET[i].X,
            tar_data.DET[i].Y, tar_data.DET[i].Z);
    }

    // Write to raw file
    size_t data_size = sizeof(tar_data.N) + tar_data.N * sizeof(TAR_DET_t_cpp);
    int32_t err      = WriteImageBuffer(outputFile, ".", (void *)&tar_data, data_size);
    if (err != 0)
    {
        std::cerr << "Error: Failed to write TAR results to file: " << outputFile << std::endl;
        return -1;
    }
    return 0;
}

void doaTargetProcessingRef(NVCVTensorHandle *inTensorHandles, NVCVTensorHandle outTensorHandle, PVARadarGP *GP)
{
    doaTargetProcessingImpl(inTensorHandles, outTensorHandle, GP);
}

int32_t loadPeakDataFromRawFile(NVCVTensorHandle peakIndicesHandle, NVCVTensorHandle peakSnapHandle,
                                const char *assetsDirectory, const char *filename, int32_t numPeaks)
{
    int32_t err = 0;

    std::unique_ptr<Peak_t_cpp> peakData = std::make_unique<Peak_t_cpp>();
    size_t size                          = sizeof(peakData->N) + numPeaks * sizeof(PeakDet_t_cpp);
    NVCVTensorData peakIndicesTensorData;
    nvcvTensorExportData(peakIndicesHandle, &peakIndicesTensorData);
    NVCVTensorData peakSnapTensorData;
    nvcvTensorExportData(peakSnapHandle, &peakSnapTensorData);

    int32_t peakIndicesWidth  = peakIndicesTensorData.shape[1];
    int32_t peakIndicesHeight = peakIndicesTensorData.shape[0];
    int32_t peakSnapWidth     = peakSnapTensorData.shape[1];
    int32_t peakSnapHeight    = peakSnapTensorData.shape[0];

    void *peakIndicesTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&peakIndicesTensorPtr, (void *)peakIndicesTensorData.buffer.strided.basePtr);
    void *peakSnapTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&peakSnapTensorPtr, (void *)peakSnapTensorData.buffer.strided.basePtr);

    err = ReadImageBuffer(filename, assetsDirectory, (void *)peakData.get(), size);
    if (err != 0)
    {
        std::cerr << "Error: Failed to read peak data from file: " << filename << std::endl;
        return -1;
    }

    if (peakData->N != numPeaks)
    {
        std::cerr << "Error: Number of peaks mismatch in file: " << filename << std::endl;
        return -1;
    }

    int32_t *peakIndicesRindPtrInt           = (int32_t *)peakIndicesTensorPtr;
    int32_t *peakIndicesDFoldindPtrInt       = (int32_t *)peakIndicesTensorPtr + peakIndicesWidth;
    int32_t *peakIndicesDindPtrInt           = (int32_t *)peakIndicesDFoldindPtrInt + peakIndicesWidth;
    std::pair<int32_t, int32_t> *peakSnapPtr = (std::pair<int32_t, int32_t> *)peakSnapTensorPtr;

    for (int i = 0; i < peakData->N; ++i)
    {
        peakIndicesRindPtrInt[i]     = peakData->DET[i].Rind - 1;
        peakIndicesDFoldindPtrInt[i] = peakData->DET[i].DFoldind - 1;
        peakIndicesDindPtrInt[i]     = peakData->DET[i].Dind - 1;
        for (int j = 0; j < 16; ++j)
        {
            peakSnapPtr[i * 16 + j] = peakData->DET[i].Snap[j];
        }
    }

    return err;
}

int32_t loadCalibVectorToTensorFixedPoint(NVCVTensorHandle calibTensorHandle, const int32_t calibVector[],
                                          int32_t numElements)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(calibTensorHandle, &tensorData);

    void *tensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&tensorPtr, (void *)tensorData.buffer.strided.basePtr);

    if (tensorPtr == nullptr)
    {
        std::cerr << "Error: Failed to get host pointer for calibration tensor." << std::endl;
        return -1;
    }

    size_t dataSize = numElements * sizeof(int32_t);
    memcpy(tensorPtr, calibVector, dataSize);

    return 0;
}

int32_t loadNciFinalData(NVCVTensorHandle nciTensorHandle, const char *assetsDirectory, const char *filename)
{
    int32_t err = 0;

    NVCVDataType dataType;
    nvcvTensorGetDataType(nciTensorHandle, &dataType);
    int32_t bitsPerPixel;
    nvcvDataTypeGetBitsPerPixel(dataType, &bitsPerPixel);

    NVCVTensorData nciTensorData;
    nvcvTensorExportData(nciTensorHandle, &nciTensorData);

    float *nciTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&nciTensorPtr, (void *)nciTensorData.buffer.strided.basePtr);

    if (nciTensorPtr == nullptr)
    {
        std::cerr << "Error: Failed to get host pointer for NCI FINAL tensor." << std::endl;
        return -1;
    }

    // Read raw data into temporary buffer
    size_t tensorSize = nciTensorData.shape[0] * nciTensorData.shape[1];
    size_t raw_size   = tensorSize * sizeof(float);
    std::vector<float> temp_buffer(raw_size / sizeof(float));
    err = ReadImageBuffer(filename, assetsDirectory, temp_buffer.data(), raw_size);
    if (err != 0)
    {
        std::cerr << "Error: Failed to read NCI FINAL data from file." << std::endl;
        return err;
    }

    for (size_t i = 0; i < tensorSize; ++i)
    {
        // Convert float to Q15 fixed-point
        nciTensorPtr[i] = temp_buffer[i];
    }

    return 0;
}

/**
 * @brief Compares output tensor handle with reference tensor handle for validation.
 *
 * This function extracts data from both tensors and performs a comprehensive comparison
 * of the target detection results, including velocity, range, azimuth, elevation, and
 * 3D coordinates (X, Y, Z).
 *
 * @param outTensorHandle The output tensor handle from the implementation under test.
 * @param refTensorHandle The reference tensor handle from the reference implementation.
 * @param GP Pointer to the GP parameters structure containing number of peaks.
 * @param tolerance The tolerance for floating-point comparisons (default: 1e-6).
 * @return Returns 0 if tensors match within tolerance, -1 on error, number of mismatches otherwise.
 */
int32_t compareOutputWithReference(NVCVTensorHandle outTensorHandle, NVCVTensorHandle refTensorHandle,
                                   PVARadarGP const *GP, int32_t numPeaks, float tolerance)
{
    // Export tensor data
    NVCVTensorData outTensorData, refTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    // Get host pointers
    void *outTensorPtr = nullptr, *refTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&refTensorPtr, (void *)refTensorData.buffer.strided.basePtr);

    if (outTensorPtr == nullptr || refTensorPtr == nullptr)
    {
        std::cerr << "Error: Failed to get host pointers for tensor comparison." << std::endl;
        return -1;
    }

    int32_t numTargets = outTensorData.shape[1];

    // Cast to float arrays
    float *outVelocityPtr  = (float *)outTensorPtr;
    float *outRangePtr     = (float *)outTensorPtr + numTargets;
    float *outAzimuthPtr   = (float *)outTensorPtr + numTargets * 2;
    float *outElevationPtr = (float *)outTensorPtr + numTargets * 3;
    float *outXPtr         = (float *)outTensorPtr + numTargets * 4;
    float *outYPtr         = (float *)outTensorPtr + numTargets * 5;
    float *outZPtr         = (float *)outTensorPtr + numTargets * 6;

    float *refVelocityPtr  = (float *)refTensorPtr;
    float *refRangePtr     = (float *)refTensorPtr + numTargets;
    float *refAzimuthPtr   = (float *)refTensorPtr + numTargets * 2;
    float *refElevationPtr = (float *)refTensorPtr + numTargets * 3;
    float *refXPtr         = (float *)refTensorPtr + numTargets * 4;
    float *refYPtr         = (float *)refTensorPtr + numTargets * 5;
    float *refZPtr         = (float *)refTensorPtr + numTargets * 6;

    printf("=== Tensor Comparison Results ===\n");
    printf("Number of peaks: %d\n", numPeaks);
    printf("Tolerance: %.2e\n", tolerance);
    printf("Format: [Index] Output_Value vs Reference_Value (Difference)\n\n");

    int32_t mismatchCount    = 0;
    int32_t totalComparisons = 0;

    // Compare each target detection
    for (int32_t i = 0; i < numPeaks; ++i)
    {
        bool targetMismatch = false;

        printf("Target %d:\n", i);

        // Compare velocity
        float velDiff = std::abs(outVelocityPtr[i] - refVelocityPtr[i]);
        if (velDiff > tolerance)
        {
            printf("  Velocity: %.6f vs %.6f (diff: %.6f) [MISMATCH]\n", outVelocityPtr[i], refVelocityPtr[i], velDiff);
            targetMismatch = true;
        }
        else
        {
            printf("  Velocity: %.6f vs %.6f (diff: %.6f) [OK]\n", outVelocityPtr[i], refVelocityPtr[i], velDiff);
        }
        totalComparisons++;

        // Compare range
        float rangeDiff = std::abs(outRangePtr[i] - refRangePtr[i]);
        if (rangeDiff > tolerance)
        {
            printf("  Range: %.6f vs %.6f (diff: %.6f) [MISMATCH]\n", outRangePtr[i], refRangePtr[i], rangeDiff);
            targetMismatch = true;
        }
        else
        {
            printf("  Range: %.6f vs %.6f (diff: %.6f) [OK]\n", outRangePtr[i], refRangePtr[i], rangeDiff);
        }
        totalComparisons++;

        // Compare azimuth
        float azDiff = std::abs(outAzimuthPtr[i] - refAzimuthPtr[i]);
        if (azDiff > tolerance)
        {
            printf("  Azimuth: %.6f vs %.6f (diff: %.6f) [MISMATCH]\n", outAzimuthPtr[i], refAzimuthPtr[i], azDiff);
            targetMismatch = true;
        }
        else
        {
            printf("  Azimuth: %.6f vs %.6f (diff: %.6f) [OK]\n", outAzimuthPtr[i], refAzimuthPtr[i], azDiff);
        }
        totalComparisons++;

        // Compare elevation
        float elevDiff = std::abs(outElevationPtr[i] - refElevationPtr[i]);
        if (elevDiff > tolerance)
        {
            printf("  Elevation: %.6f vs %.6f (diff: %.6f) [MISMATCH]\n", outElevationPtr[i], refElevationPtr[i],
                   elevDiff);
            targetMismatch = true;
        }
        else
        {
            printf("  Elevation: %.6f vs %.6f (diff: %.6f) [OK]\n", outElevationPtr[i], refElevationPtr[i], elevDiff);
        }
        totalComparisons++;

        // Compare X coordinate
        float xDiff = std::abs(outXPtr[i] - refXPtr[i]);
        if (xDiff > tolerance)
        {
            printf("  X: %.6f vs %.6f (diff: %.6f) [MISMATCH]\n", outXPtr[i], refXPtr[i], xDiff);
            targetMismatch = true;
        }
        else
        {
            printf("  X: %.6f vs %.6f (diff: %.6f) [OK]\n", outXPtr[i], refXPtr[i], xDiff);
        }
        totalComparisons++;

        // Compare Y coordinate
        float yDiff = std::abs(outYPtr[i] - refYPtr[i]);
        if (yDiff > tolerance)
        {
            printf("  Y: %.6f vs %.6f (diff: %.6f) [MISMATCH]\n", outYPtr[i], refYPtr[i], yDiff);
            targetMismatch = true;
        }
        else
        {
            printf("  Y: %.6f vs %.6f (diff: %.6f) [OK]\n", outYPtr[i], refYPtr[i], yDiff);
        }
        totalComparisons++;

        // Compare Z coordinate
        float zDiff = std::abs(outZPtr[i] - refZPtr[i]);
        if (zDiff > tolerance)
        {
            printf("  Z: %.6f vs %.6f (diff: %.6f) [MISMATCH]\n", outZPtr[i], refZPtr[i], zDiff);
            targetMismatch = true;
        }
        else
        {
            printf("  Z: %.6f vs %.6f (diff: %.6f) [OK]\n", outZPtr[i], refZPtr[i], zDiff);
        }
        totalComparisons++;

        if (targetMismatch)
        {
            mismatchCount++;
        }
        printf("\n");
    }

    // Summary statistics
    printf("=== Comparison Summary ===\n");
    printf("Total comparisons: %d\n", totalComparisons);
    printf("Mismatches: %d\n", mismatchCount);
    printf("Match rate: %.2f%%\n", (totalComparisons - mismatchCount) * 100.0f / totalComparisons);

    if (mismatchCount == 0)
    {
        printf("RESULT: ALL TESTS PASSED ✓\n");
    }
    else
    {
        printf("RESULT: %d MISMATCHES FOUND ✗\n", mismatchCount);
    }
    printf("==========================\n");

    return mismatchCount;
}

/**
 * @brief Performs a quick comparison with reduced output for automated testing.
 *
 * This function provides a more concise comparison suitable for automated testing
 * scenarios where detailed output is not needed.
 *
 * @param outTensorHandle The output tensor handle from the implementation under test.
 * @param refTensorHandle The reference tensor handle from the reference implementation.
 * @param GP Pointer to the GP parameters structure containing number of peaks.
 * @param tolerance The tolerance for floating-point comparisons (default: 1e-6).
 * @return Returns 0 if tensors match within tolerance, number of mismatches otherwise.
 */
int32_t quickCompareOutputWithReference(NVCVTensorHandle outTensorHandle, NVCVTensorHandle refTensorHandle,
                                        PVARadarGP const *GP, int32_t numPeaks, float tolerance)
{
    // Export tensor data
    NVCVTensorData outTensorData, refTensorData;
    nvcvTensorExportData(outTensorHandle, &outTensorData);
    nvcvTensorExportData(refTensorHandle, &refTensorData);

    // Get host pointers
    void *outTensorPtr = nullptr, *refTensorPtr = nullptr;
    CupvaMemGetHostPointer((void **)&outTensorPtr, (void *)outTensorData.buffer.strided.basePtr);
    CupvaMemGetHostPointer((void **)&refTensorPtr, (void *)refTensorData.buffer.strided.basePtr);

    if (outTensorPtr == nullptr || refTensorPtr == nullptr)
    {
        return -1;
    }

    int32_t numTargets = outTensorData.shape[1];

    // Cast to float arrays
    float *outVelocityPtr  = (float *)outTensorPtr;
    float *outRangePtr     = (float *)outTensorPtr + numTargets;
    float *outAzimuthPtr   = (float *)outTensorPtr + numTargets * 2;
    float *outElevationPtr = (float *)outTensorPtr + numTargets * 3;
    float *outXPtr         = (float *)outTensorPtr + numTargets * 4;
    float *outYPtr         = (float *)outTensorPtr + numTargets * 5;
    float *outZPtr         = (float *)outTensorPtr + numTargets * 6;

    float *refVelocityPtr  = (float *)refTensorPtr;
    float *refRangePtr     = (float *)refTensorPtr + numTargets;
    float *refAzimuthPtr   = (float *)refTensorPtr + numTargets * 2;
    float *refElevationPtr = (float *)refTensorPtr + numTargets * 3;
    float *refXPtr         = (float *)refTensorPtr + numTargets * 4;
    float *refYPtr         = (float *)refTensorPtr + numTargets * 5;
    float *refZPtr         = (float *)refTensorPtr + numTargets * 6;

    int32_t mismatchCount = 0;

    // Quick comparison - only count mismatches
    for (int32_t i = 0; i < numPeaks; ++i)
    {
        if (std::abs(outVelocityPtr[i] - refVelocityPtr[i]) > tolerance)
            mismatchCount++;
        if (std::abs(outRangePtr[i] - refRangePtr[i]) > tolerance)
            mismatchCount++;
        if (std::abs(outAzimuthPtr[i] - refAzimuthPtr[i]) > tolerance)
            mismatchCount++;
        if (std::abs(outElevationPtr[i] - refElevationPtr[i]) > tolerance)
            mismatchCount++;
        if (std::abs(outXPtr[i] - refXPtr[i]) > tolerance)
            mismatchCount++;
        if (std::abs(outYPtr[i] - refYPtr[i]) > tolerance)
            mismatchCount++;
        if (std::abs(outZPtr[i] - refZPtr[i]) > tolerance)
            mismatchCount++;
    }

    return mismatchCount;
}
