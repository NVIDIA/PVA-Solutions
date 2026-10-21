..
 SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
..
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
..
 http://www.apache.org/licenses/LICENSE-2.0
..
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

Classification
==============

This sample demonstrates image classification using supported models. Currently, the following models are supported:

* **Swin-Transformer (tiny)**
* **ResNet50**

Model Preparation
-----------------

Before building the sample, download the required ONNX models to the ``samples/assets/onnx`` directory using the URLs in `network.txt`.

The model preparation is automated through CMake using the ``prepare_onnx_model`` function, which executes the ``process_classification_model.sh`` script for Swin-T.
The models are quantized to int8 with float32 scales. A single image is used to calibrate the quantized model for demonstration purposes.
We remove the first QuantizeLinear node from the quantized model, so that it can be merged with preprocessing, and the quantized model expects int8 input.
The processed models are saved to the build directory.

Model Compilation
-----------------

The compilation is configured by the CMake function ``pva_compile_networks``.

The `pva-onnx-compiler <../../tools/pva-onnx-compiler>`_ converts the processed ONNX model to LLVM IR, which is then converted to assembly by ``llc``. The assembly file is linked with `classification.cpp <./classification.cpp>`_.

During the build process, the input and output tensor requirements—including their order, shapes, and data types—are displayed in the console. This information is printed by the CMake build system to help you verify that the model's inputs and outputs are correctly configured.


Limitations
-----------
The ``pva-onnx-compiler`` and kernel library are currently in early development phase with limited operator support. Key limitations include:

* Only HWC convolution kernels are supported
* Behavior not guaranteed for models using unsupported operators or configurations
* Limited support for float32 and float16 kernels
* Other unspported features: dynamic shapes, batch size > 1, etc.