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

#ifndef LIDAR_HISTOGRAM_CONVERSION_DEBUG_OPTIONS_H
#define LIDAR_HISTOGRAM_CONVERSION_DEBUG_OPTIONS_H

// Debug flags
// Overwrite the input with smoothed histogram data (useful for debugging smoothing and peak detection)
#define DEBUG_OVERWRITE_INPUT_WITH_SMOOTHED_HISTOGRAM 0
// Store the peak indices in the range tensor output instead of range values. Useful for debugging peak detection.
#define DEBUG_WRITE_PEAK_INDICES_TO_RANGE 0

#endif /* LIDAR_HISTOGRAM_CONVERSION_DEBUG_OPTIONS_H */