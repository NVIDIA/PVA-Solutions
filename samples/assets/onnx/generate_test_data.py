#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import onnxruntime as ort
import numpy as np
import argparse
from quantize_model import imagenet_normalize, yolo_center_pad
import onnx

def quantize_linear(img, first_yscale):
    return np.clip(np.round(img / first_yscale), -128, 127).astype(np.int8)

def get_scale_value(model_path, initializer_name):
    """Get the value of a specific initializer from the ONNX model"""
    model = onnx.load(model_path)
    print(f"Getting scale value for initializer: {initializer_name}")
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            assert initializer.data_type == 1, "Expected float32 initializers"
            assert len(initializer.float_data) == 1, "Expected a single float32 value"
            return initializer.float_data[0]
    raise ValueError(f"Initializer {initializer_name} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-data-path", type=str, required=True)
    parser.add_argument("--app", type=str, required=True, choices=['classification', 'detection'], help="Application type")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image for test data generation")
    parser.add_argument('--remove-first-quantize-node', action='store_true', help='Remove the first quantize node (optional)')
    args = parser.parse_args()

    model_path = args.model_path
    output_data_path = args.output_data_path
    app = args.app
    image_path = args.image_path
    remove_first_quantize_node = args.remove_first_quantize_node

    ort_sess = ort.InferenceSession(model_path)

    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")

    assert len(input_shape) >= 3, "Model must have at least 3 dimensions"
    height = input_shape[-2]
    width = input_shape[-1]
    print(f"Processing image with height: {height}, width: {width}")

    # Generate input data based on application type
    if app == "classification":
        data = imagenet_normalize(image_path, height, width)
    elif app == "detection":
        assert height == 672 and width == 672, "yolov5s model from cudla repo has input shape 672x672"
        data = yolo_center_pad(image_path, 640, 640, 16)
    else:
        raise ValueError(f"Invalid application: {app}")

    if remove_first_quantize_node:
        scale_name = input_name.replace('QuantizeLinear_Output', 'scale')
        scale_value = get_scale_value(model_path, scale_name)
        print(f"Scale value: {scale_value}")
        data = quantize_linear(data, scale_value)

    # Convert to NCHW format and add batch dimension
    batch_data = np.expand_dims(data.transpose(2, 0, 1), axis=0)
    print(f"Input data shape: {batch_data.shape}")

    # Run inference
    outs = ort_sess.run(None, {input_name: batch_data})

    if app == "classification":
        idx = np.argsort(outs[0].flatten())[-5:][::-1]
        print("Top 5 indices:", idx)
        print("Saving output data to", output_data_path)
        idx.tofile(output_data_path)
    elif app == "detection":
        print("Saving output data to", output_data_path)
        out_reshape = []
        for out in outs:
            # out shape: NCHW
            shape = out.shape
            assert shape[0] == 1
            out = out[0].reshape(shape[1], -1)
            out_reshape.append(out)
        out_concat = np.concatenate(out_reshape, axis=1).transpose(1,0)
        out_concat.tofile(output_data_path)
    else:
        raise ValueError(f"Invalid application: {app}")

