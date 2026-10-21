/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file TileSizeEnumeration.hpp
 *
 * @brief Defines the private functions for math calculations.
 */

#ifndef PVA_SOLUTIONS_PRIV_MATHUTILS_HPP
#define PVA_SOLUTIONS_PRIV_MATHUTILS_HPP

#include <cstdint>
#include <optional>
#include <utility>

namespace pva_solutions::priv {

// Use the simple heuristic to calculate the tile size. Output in the format of std::pair {tileX, tileY}
auto enumerateTileSz(int32_t width, int32_t height, int32_t blkSz, int32_t maxTileSz)
    -> std::optional<std::pair<int32_t, int32_t>>;

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_MATHUTILS_HPP
