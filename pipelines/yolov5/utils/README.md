<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# YOLOv5 pipeline dependencies

The GPU-only and PVA+GPU pipelines use the repository's LodePNG library for
PNG input. They also accept pre-decoded RGB `.bin` inputs, which hold the
width and height as two native-endian 32-bit integers followed by
width x height x 3 interleaved 8-bit RGB values. In benchmark mode the input
image is decoded once before the measured iterations, so file decoding is
excluded from timing.

CV-CUDA operators are not a dependency. The GPU-only pipeline uses the local
CUDA letterbox kernel, and the PVA+GPU pipeline uses PVA resize plus fused PVA
letterbox/normalize/layout conversion.
