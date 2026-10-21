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

#ifndef DOA_TARGET_PROCESSING_IMPL_HPP
#define DOA_TARGET_PROCESSING_IMPL_HPP

#include "doa_params.h"

#include <PvaOperatorTypes.h>
#include <math.h> // For C math functions: asin, atan, cos, sin, roundf
#include <nvcv/Tensor.h>

#include <algorithm> // For std::max_element, std::distance
#include <cmath> // For std::roundf, std::abs (for float), std::arg, std::exp, std::conj, std::asin, std::atan, cosf, sinf
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef PI_CONST_F
#    define PI_CONST_F 3.14159265358979323846f
#endif

#ifndef Q14_SCALE
#    define Q14_SCALE (1 << 14)
#endif

#ifndef Q15_SCALE
#    define Q15_SCALE (1 << 15)
#endif

#ifndef Q15_90_DEGREES
#    define Q15_90_DEGREES (int32_t)(90.0f * (1 << 15))
#endif

#ifndef Q15_180_DEGREES
#    define Q15_180_DEGREES (int32_t)(180.0f * (1 << 15))
#endif

#define TWIDDLE_QBITS (30)
#define TWIDDLE_SCALE (1 << TWIDDLE_QBITS)

#define AINDICES_WIDTH 256
#define ATAN_TABLE_SIZE 513
#define ASIN_TABLE_SIZE ATAN_TABLE_SIZE

// Helper function for rounding with scaling
static inline int64_t my_round(int64_t x, int32_t shift)
{
    int64_t half = 1 << (shift - 1);
    return (x + half) >> shift;
}

/**
 * @brief Converts a floating-point value to a Q15 fixed-point integer representation.
 *
 * @param val The floating-point value to convert.
 * @return The Q15 fixed-point representation as a 16-bit integer.
 */
static inline int16_t float_to_q15_int16(float val)
{
    return static_cast<int16_t>(std::round(val * (1 << 15)));
}

/**
 * @brief Converts a floating-point value to a Q15 fixed-point integer representation.
 *
 * @param val The floating-point value to convert.
 * @return The Q15 fixed-point representation as a 32-bit integer.
 */
static inline int32_t float_to_q15(float val)
{
    return static_cast<int32_t>(std::round(val * (1 << 15)));
}

/**
 * @brief Converts a floating-point value to a Q-bits fixed-point integer representation.
 *
 * @param val The floating-point value to convert.
 * @param qbits The number of fractional bits (Q-bits) for the fixed-point representation.
 * @return The Q-bits fixed-point representation as a 32-bit integer.
 */
static inline int32_t float_to_int32(float val, int qbits)
{
    assert(qbits <= 30);
    return static_cast<int32_t>(std::round(val * (1 << qbits)));
}

/**
 * @brief Converts a Q-bits fixed-point integer to a floating-point value.
 *
 * @param val The Q-bits fixed-point integer to convert.
 * @param qbits The number of fractional bits (Q-bits) used in the fixed-point representation.
 * @return The floating-point representation.
 */
static inline float int32_to_float(int32_t val, int qbits)
{
    return static_cast<float>(val) / (1 << qbits);
}

/**
 * @brief Converts a Q15 fixed-point integer to a floating-point value.
 *
 * @param val The Q15 fixed-point integer to convert.
 * @return The floating-point representation.
 */
static inline float q15_to_float(int32_t val)
{
    return static_cast<float>(val) / (1 << 15);
}

/**
 * @brief Converts a Q15 fixed-point integer to a floating-point value.
 *
 * @param val The Q15 fixed-point integer to convert.
 * @param qbits The number of fractional bits (Q-bits) used in the fixed-point representation.
 * @return The floating-point representation.
 */
static inline float int64_to_float(int64_t val, int qbits)
{
    return static_cast<float>(val) / (1 << qbits);
}

/**
 * @brief Converts a floating-point value to a Q14 fixed-point integer representation.
 *
 * @param val The floating-point value to convert.
 * @return The Q14 fixed-point representation as a 32-bit integer.
 */
static inline int32_t float_to_q14(float val)
{
    return static_cast<int32_t>(std::round(val * (1 << 14)));
}

/**
 * @brief Converts a Q14 fixed-point integer to a floating-point value.
 *
 * @param val The Q14 fixed-point integer to convert.
 * @return The floating-point representation.
 */
static inline float q14_to_float(int32_t val)
{
    return static_cast<float>(val) / (1 << 14);
}

/**
 * @brief Converts a complex float to a pair of Q15 fixed-point integers.
 *
 * @param c The complex float to convert.
 * @return A pair of Q15 fixed-point integers representing real and imaginary parts.
 */
static inline std::pair<int32_t, int32_t> complex_to_q15_pair(const std::complex<float> &c)
{
    return std::make_pair(float_to_q15(c.real()), float_to_q15(c.imag()));
}

/**
 * @brief Converts a pair of Q15 fixed-point integers back to a complex float.
 *
 * @param p The pair of Q15 fixed-point integers representing real and imaginary parts.
 * @return The complex float.
 */
static inline std::complex<float> q15_pair_to_complex(const std::pair<int32_t, int32_t> &p)
{
    return std::complex<float>(q15_to_float(p.first), q15_to_float(p.second));
}

/**
 * @brief Computes the complex conjugate of a Q15 fixed-point complex number.
 *
 * This function takes a pair of Q15 fixed-point integers representing a complex number
 * and returns its complex conjugate (negating the imaginary part).
 *
 * @param p The pair of Q15 fixed-point integers representing real and imaginary parts.
 * @return The complex conjugate as a pair of Q15 fixed-point integers.
 */
static inline std::pair<int32_t, int32_t> complex_conj_fixed(const std::pair<int32_t, int32_t> &p)
{
    // For a complex number a + bi, the conjugate is a - bi
    // In Q15 format, we just negate the imaginary part
    return std::make_pair(p.first, -p.second);
}

// Helper function to compute integer square root of a 64-bit integer.
// Result is floor(sqrt(n)).
static inline uint32_t isqrt64(uint64_t n)
{
    if (n == 0)
        return 0;
    uint64_t res = 0;
    uint64_t bit = (uint64_t)1 << 62;

    // "bit" starts at the highest power of 4 <= n.
    while (bit > n)
    {
        bit >>= 2;
    }

    while (bit != 0)
    {
        if (n >= res + bit)
        {
            n -= res + bit;
            res = (res >> 1) + bit;
        }
        else
        {
            res >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)res;
}

/**
 * @brief Computes the magnitude of a complex number using fixed-point arithmetic.
 *
 * This function calculates sqrt(real^2 + imag^2) for a Q15 fixed-point complex number
 * and returns the result in Q15 format.
 *
 * @param c_q15 The complex number in Q15 fixed-point format.
 * @return The magnitude as a Q15 fixed-point integer.
 */
static inline int32_t complex_abs_fixed(const std::pair<int32_t, int32_t> &c_q15)
{
    // Square the components. The result is in Q30 format.
    // Use 64-bit integers to avoid overflow.
    int64_t r_sq = (int64_t)c_q15.first * c_q15.first;
    int64_t i_sq = (int64_t)c_q15.second * c_q15.second;

    // Sum of squares is also Q30.
    uint64_t sum_sq = r_sq + i_sq;

    // The integer square root of a Q30 number results in a Q15 number.
    return isqrt64(sum_sq);
}

static inline std::pair<int32_t, int32_t> complex_mult_fixed(std::pair<int32_t, int32_t> a,
                                                             std::pair<int32_t, int32_t> b, int32_t qbits)
{
    int64_t a_real = a.first;
    int64_t a_imag = a.second;
    int64_t b_real = b.first;
    int64_t b_imag = b.second;

    int32_t temp_real = my_round((a_real * b_real - a_imag * b_imag), qbits);
    int32_t temp_imag = my_round((a_real * b_imag + a_imag * b_real), qbits);

    return std::make_pair(temp_real, temp_imag);
}

static inline std::pair<int64_t, int64_t> complex_mult_fixed_int64(std::pair<int32_t, int32_t> a,
                                                                   std::pair<int32_t, int32_t> b, int32_t qbits)
{
    int64_t a_real = a.first;
    int64_t a_imag = a.second;
    int64_t b_real = b.first;
    int64_t b_imag = b.second;

    int64_t temp_real = my_round((a_real * b_real - a_imag * b_imag), qbits);
    int64_t temp_imag = my_round((a_real * b_imag + a_imag * b_real), qbits);

    return std::make_pair(temp_real, temp_imag);
}

inline float sind_cpp(float val)
{
    return std::sin(val * PI_CONST_F / 180.0f);
}
inline float cosd_cpp(float val)
{
    return std::cos(val * PI_CONST_F / 180.0f);
}

/**
 * @brief Computes arcsin for a Q15 fixed-point value using a lookup table.
 *
 * This function uses a pre-computed lookup table and linear interpolation
 * to calculate the arcsin of a Q15 fixed-point integer. The result is
 * returned in degrees, also as a Q15 fixed-point integer.
 * The lookup part of this function does not use any floating-point operations.
 *
 * @param val The Q15 fixed-point integer representing a value between -1 and 1.
 * @return The arcsin of the value in degrees, as a Q15 fixed-point integer.
 */
static inline int32_t arcsin_lookup_fixed(int32_t val)
{
    static int32_t asin_table[ASIN_TABLE_SIZE];
    static bool table_initialized = false;

    if (!table_initialized)
    {
        for (int i = 0; i < ASIN_TABLE_SIZE; ++i)
        {
            float x         = static_cast<float>(i) / (ASIN_TABLE_SIZE - 1);
            float angle_rad = std::asin(x);
            float angle_deg = angle_rad * 180.0f / PI_CONST_F;
            asin_table[i]   = float_to_q15(angle_deg);
        }
        table_initialized = true;
    }

    // Handle sign
    int32_t sign = 1;
    if (val < 0)
    {
        sign = -1;
        val  = -val;
    }

    // Handle boundary case
    if (val >= 4194304)
    {
        return sign * asin_table[ASIN_TABLE_SIZE - 1]; // Return Q15 representation of 90 degrees
    }

    // Map Q15 input to table index.
    // The floating-point equivalent is: table_pos = val_float * 256.
    // In fixed-point, this is: (val_q15 / 2^15) * 2^8 = val_q15 / 2^7.
    int32_t index0 = val >> 13;
    // The remainder of the division gives the fractional part of the index.
    int32_t fraction_q13 = val & 0x1FFF;

    // Lookup
    int32_t y0 = asin_table[index0];
    int32_t y1 = asin_table[index0 + 1];

    // Linear interpolation: y = y0 + (y1 - y0) * fraction
    int32_t diff = y1 - y0;
    // Use 64-bit intermediate to prevent overflow (Q15 * Q8 -> Q23)
    int64_t interp_prod = (int64_t)diff * fraction_q13;
    int32_t adjustment  = (int32_t)(interp_prod >> 13); // Convert back to Q15

    int32_t result = y0 + adjustment;

    return sign * result;
}

/**
 * @brief Computes atan2(y, x) using a lookup table approach with floating point inputs.
 *
 * This function calculates the angle in degrees for the point (x, y) using a lookup table
 * for atan(x) values in the range [0, 1] (first octant only), and then adjusts for the correct quadrant.
 *
 * @param y The y coordinate as a float.
 * @param x The x coordinate as a float.
 * @return The angle in degrees as a Q15 fixed-point integer.
 */
static inline int32_t atan2_lookup_fixed(float y, float x)
{
    // LUT for atan(x) for x in [0, 1] - first octant only (0° to 45°)
    static int32_t atan_table[ATAN_TABLE_SIZE];
    static bool table_initialized = false;

    if (!table_initialized)
    {
        for (int i = 0; i < ATAN_TABLE_SIZE; ++i)
        {
            float x         = static_cast<float>(i) / (ATAN_TABLE_SIZE - 1);
            float angle_rad = std::atan(x);
            float angle_deg = angle_rad * 180.0f / PI_CONST_F;
            // Store in Q15 format directly
            atan_table[i] = float_to_q15(angle_deg);
        }
        table_initialized = true;
    }

    // Handle special cases
    if (x == 0.0f)
    {
        if (y > 0.0f)
            return Q15_90_DEGREES;
        if (y < 0.0f)
            return -Q15_90_DEGREES;
        return 0; // y == 0
    }

    // Use absolute values for lookup
    float abs_y = y > 0.0f ? y : -y;
    float abs_x = x > 0.0f ? x : -x;

    // Determine octant and calculate ratio for atan table lookup
    // The ratio must be <= 1.0 (first octant only)
    float ratio;
    bool y_is_larger = (abs_y > abs_x);
    if (y_is_larger)
    {
        if (abs_y == 0.0f)
            return 0;
        ratio = abs_x / abs_y;
    }
    else
    {
        if (abs_x == 0.0f)
            return 0;
        ratio = abs_y / abs_x;
    }

    // Convert ratio to Q22 for table lookup (higher precision)
    int32_t ratio_q22 = static_cast<int32_t>(std::round(ratio * (1 << 22)));

    // Interpolated lookup for 513 table size with Q22 precision
    // For 513 entries, we need 9 bits for index (2^9 = 512, but we have 513 entries)
    int32_t index0       = ratio_q22 >> 13;    // Shift by 13 for 9-bit index (22-13=9 bits)
    int32_t fraction_q13 = ratio_q22 & 0x1FFF; // Extract 13 bits.

    // Clamp index to valid range for 513 table size
    if (index0 >= ATAN_TABLE_SIZE - 1)
    {
        index0 = ATAN_TABLE_SIZE - 2;
    }

    int32_t y0 = atan_table[index0];
    int32_t y1 = atan_table[index0 + 1];

    int32_t diff             = y1 - y0;
    int64_t interp_prod      = (int64_t)diff * fraction_q13;
    int32_t adjustment       = (int32_t)(interp_prod >> 13);
    int32_t angle_octant_q15 = y0 + adjustment;

    // Convert from Q15 format (0-45°) to Q15 degrees format
    // angle_octant_q15 is already in Q15 format for 0-45° range
    // We need to scale it to the full 0-90° range
    int32_t angle0_45_q15;
    if (y_is_larger)
    {
        // If y > x, we need 90° - angle
        angle0_45_q15 = Q15_90_DEGREES - angle_octant_q15;
    }
    else
    {
        // If x >= y, angle is directly from table
        angle0_45_q15 = angle_octant_q15;
    }

    // Adjust for quadrant
    if (x > 0.0f && y < 0.0f) // Quadrant 4
    {
        return -angle0_45_q15;
    }
    else if (x < 0.0f)
    {
        if (y >= 0.0f) // Quadrant 2
        {
            return Q15_180_DEGREES - angle0_45_q15;
        }
        else // Quadrant 3
        {
            return -Q15_180_DEGREES + angle0_45_q15;
        }
    }

    // Quadrant 1 (x > 0, y >= 0)
    return angle0_45_q15;
}

/**
 * @brief Computes the phase angle for a complex number using fixed-point arithmetic.
 *
 * This function takes a Q15 fixed-point complex number represented as a pair of integers,
 * converts to floating point, uses a fixed-point atan2 lookup to calculate the angle in degrees, and returns
 * the result as a Q15 fixed-point integer.
 *
 * @param c_q15 The complex number in Q15 fixed-point format.
 * @return The phase angle in degrees as a Q15 fixed-point integer.
 */
static inline int32_t calculate_phase_angle_fixed(const std::pair<int32_t, int32_t> &c_q15)
{
    int32_t r_q15 = c_q15.first;
    int32_t i_q15 = c_q15.second;

    float norm_r_float = q15_to_float(r_q15);
    float norm_i_float = q15_to_float(i_q15);

    // Calculate angle in degrees using fixed-point lookup table
    int32_t angle_deg_q15 = atan2_lookup_fixed(norm_i_float, norm_r_float);

    return angle_deg_q15;
}

static inline int32_t calculate_phase_angle_fixed(const std::pair<int64_t, int64_t> &c_q15, int qbits)
{
    int64_t r_q15 = c_q15.first;
    int64_t i_q15 = c_q15.second;

    float norm_r_float = int64_to_float(r_q15, qbits);
    float norm_i_float = int64_to_float(i_q15, qbits);

    // Calculate angle in degrees using fixed-point lookup table
    int32_t angle_deg_q15 = atan2_lookup_fixed(norm_i_float, norm_r_float);

    return angle_deg_q15;
}

// Fixed-point complex multiplication (Q15 format, int64_t)
static inline void complex_mult_q14_int64(int64_t a_real, int64_t a_imag, int16_t b_real, int16_t b_imag,
                                          int64_t *out_real, int64_t *out_imag)
{
    int64_t temp_real = ((a_real * b_real - a_imag * b_imag) >> 14);
    int64_t temp_imag = ((a_real * b_imag + a_imag * b_real) >> 14);
    *out_real         = temp_real;
    *out_imag         = temp_imag;
}

// Radix-2 FFT implementation (Q14 format, int64_t) - consistent with test implementation
static inline void radix2_fft_q14_int64(int n, std::vector<int64_t> &x_real, std::vector<int64_t> &x_imag)
{
    if (n <= 1)
        return;

    std::vector<int64_t> even_real(n / 2), even_imag(n / 2);
    std::vector<int64_t> odd_real(n / 2), odd_imag(n / 2);

    for (int i = 0; i < n / 2; i++)
    {
        even_real[i] = x_real[2 * i];
        even_imag[i] = x_imag[2 * i];
        odd_real[i]  = x_real[2 * i + 1];
        odd_imag[i]  = x_imag[2 * i + 1];
    }

    radix2_fft_q14_int64(n / 2, even_real, even_imag);
    radix2_fft_q14_int64(n / 2, odd_real, odd_imag);

    for (int k = 0; k < n / 2; k++)
    {
        int16_t twiddle_real = (int16_t)(Q14_SCALE * cos(-2.0f * PI_CONST_F * k / n));
        int16_t twiddle_imag = (int16_t)(Q14_SCALE * sin(-2.0f * PI_CONST_F * k / n));

        int64_t temp_real, temp_imag;
        complex_mult_q14_int64(odd_real[k], odd_imag[k], twiddle_real, twiddle_imag, &temp_real, &temp_imag);

        int64_t sum_real  = (even_real[k] + temp_real); // Add right shift like Python
        int64_t sum_imag  = (even_imag[k] + temp_imag); // Add right shift like Python
        int64_t diff_real = (even_real[k] - temp_real); // Add right shift like Python
        int64_t diff_imag = (even_imag[k] - temp_imag); // Add right shift like Python

        x_real[k]         = sum_real;
        x_imag[k]         = sum_imag;
        x_real[k + n / 2] = diff_real;
        x_imag[k + n / 2] = diff_imag;
    }
}

static inline std::vector<std::pair<int32_t, int32_t>> radix2_fft_fixed(std::vector<std::pair<int32_t, int32_t>> &x)
{
    int n = x.size();
    std::vector<int64_t> x_real(n);
    std::vector<int64_t> x_imag(n);

    // Copy Q15 fixed-point values
    for (int i = 0; i < n; i++)
    {
        x_real[i] = x[i].first;
        x_imag[i] = x[i].second;
    }

    // Perform fixed-point FFT
    radix2_fft_q14_int64(n, x_real, x_imag);

    std::vector<std::pair<int32_t, int32_t>> x_out(n);
    // Convert back to Q15 fixed-point
    for (int i = 0; i < n; i++)
    {
        x_out[i] = {static_cast<int32_t>(x_real[i]), static_cast<int32_t>(x_imag[i])};
    }

    return x_out;
}

// Custom fftshift for 1D vector (if C++23 std::fftshift is not available/used)
template<typename T>
std::vector<T> custom_fftshift_cpp(const std::vector<T> &v)
{
    std::vector<T> shifted_v = v;
    if (v.empty())
    {
        return shifted_v;
    }
    size_t n   = v.size();
    size_t mid = (n + 1) / 2;
    std::rotate(shifted_v.begin(), shifted_v.begin() + mid, shifted_v.end());
    return shifted_v;
}

struct GP_t_cpp
{
    uint32_t numPeaks;
    int32_t peakIndicesWidth;
    int32_t peakIndicesHeight;
    int32_t peakSnapWidth;
    int32_t peakSnapHeight;
    int32_t nciFinalWidth;
    int32_t nciFinalHeight;
    int32_t numTargets;
    int32_t targetHeight;
};

struct PeakIndices_t
{
    uint32_t *Rind;
    uint32_t *DFoldind;
    uint32_t *Dind;
};

// AoS struct
struct PeakDet_t_cpp
{
    int Rind;
    int DFoldind;
    int Dind;
    std::pair<int32_t, int32_t> Snap[16]; // Q15 fixed-point complex numbers
};

struct Peak_t_cpp
{
    int N;
    PeakDet_t_cpp DET[DOA_MAX_TARGET_COUNT];
};

struct TAR_DET_t_cpp
{
    float velocity;
    float range;
    float A; // Azimuth
    float E; // Elevation
    float X;
    float Y;
    float Z;
};

struct TAR_t_cpp
{
    int N; // Number of targets
    TAR_DET_t_cpp DET[DOA_MAX_TARGET_COUNT];
};

#endif