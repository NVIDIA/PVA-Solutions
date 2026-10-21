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

#ifndef YOLOV5_COMMON_INT8_PREPROCESS_H
#define YOLOV5_COMMON_INT8_PREPROCESS_H

namespace yolov5_common {

/**
 * Resize an RGB image to planar NCHW float32 in [0, 1] for INT8 calibration.
 * The resize intentionally fills the model input without letterbox padding to
 * avoid constant border regions dominating TensorRT entropy calibration.
 */
int resizeToNCHWFloat(const unsigned char *imgRGB, int imgW, int imgH, int modelW, int modelH, float *outNCHW);

} // namespace yolov5_common

#endif // YOLOV5_COMMON_INT8_PREPROCESS_H
