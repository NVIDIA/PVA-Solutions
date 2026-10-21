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

/*
 * @file primitive_utils.hpp
 *
 * @brief Defines utility functions for primitives.
 */

#ifndef PVA_PRIMITIVE_UTILS_HPP
#define PVA_PRIMITIVE_UTILS_HPP

#include <cstdint>

namespace pva_solutions { namespace utils {

/**
 * @brief Prepares convolution kernel coefficients for use with Conv2dVpu primitive.
 *
 * This function takes an input kernel and transforms it into a format suitable for Conv2dVpu primitive.
 *
 * @tparam T The data type of the kernel coefficients (e.g., int8_t, int16_t, uint8_t, uint16_t).
 * @param knl Pointer to the input kernel coefficients. Assumed to be a flat array
 *            representing a `ksize x ksize` kernel in row-major order.
 * @param knl_opt_out Pointer to the output buffer where the prepared kernel data will be written.
 *                    The buffer must be large enough to hold `2 * kwa * kha` elements of type `T`.
 * @param ksize The size of one dimension of the square input kernel (e.g., 3 for a 3x3 kernel).
 */
template<typename T>
void prepareKernelForConv2dVpu(T *knl, T *knl_opt_out, int32_t ksize);

}} // namespace pva_solutions::utils

#endif // PVA_PRIMITIVE_UTILS_HPP