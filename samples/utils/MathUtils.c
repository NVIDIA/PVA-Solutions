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

#include "MathUtils.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

// Function to compute GCD
int computeGCD(int a, int b)
{
    // Compute GCD on absolute values
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;

    while (b != 0)
    {
        int t = b;
        b     = a % b;
        a     = t;
    }
    return a;
}

// Converts float to integer fraction
void convertFloatToFraction(double value, int *numerator, int *denominator)
{
    // Input validation
    if (numerator == NULL || denominator == NULL)
    {
        printf("convertFloatToFraction Error: numerator or denominator is NULL\n");
        return;
    }

    // Handle NaN or Inf
    if (isnan(value) || isinf(value))
    {
        printf("convertFloatToFraction Error: value is NaN or Inf\n");
        *numerator   = 0;
        *denominator = 1;
        return;
    }

    const int precision = 100000;
    int numerator_tmp   = (int)(round(value * precision));
    int denominator_tmp = precision;

    // Handle case where rounding resulted in 0
    if (numerator_tmp == 0)
    {
        *numerator   = 0;
        *denominator = 1;
        return;
    }

    int divisor = computeGCD(numerator_tmp, denominator_tmp);
    numerator_tmp /= divisor;
    denominator_tmp /= divisor;
    *numerator   = (int)numerator_tmp;
    *denominator = (int)denominator_tmp;
}