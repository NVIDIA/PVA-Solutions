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

# run the script with OMP_NUM_THREADS=1 to silence harmless pthread_setaffinity_np warnings
import os
os.environ["OMP_NUM_THREADS"] = "1"  # limit OpenMP threads

import onnxruntime as _ort
_ort.set_default_logger_severity(4)  # suppress harmless pthread_setaffinity_np warnings (FATAL only)

import onnx
from onnxsim import simplify
from modelopt.onnx.quantization.int8 import quantize as quantize_int8
from modelopt.onnx.quantization.calib_utils import CalibrationDataProvider
import numpy as np
from PIL import Image
import argparse
import onnx_graphsurgeon as gs


def imagenet_normalize(image_path, height, width):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = Image.open(image_path).convert('RGB').resize((width, height))
    return (np.float32(img) / 255 - mean) / std

def yolo_letterbox(image_path, target_height, target_width):
    """
    Standard Ultralytics YOLO letterbox preprocessing.

    - Maintains aspect ratio
    - Uses gray padding (114, 114, 114)
    - Top-left alignment (official Ultralytics standard)

    Args:
        image_path: Path to input image
        target_height: Model input height (e.g., 640)
        target_width: Model input width (e.g., 640)

    Returns:
        Preprocessed image [H, W, C] float32, normalized to [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size

    # Compute scale to fit within target while maintaining aspect ratio
    scale = min(target_width / img_width, target_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    # Resize with aspect ratio preserved
    resized = img.resize((new_width, new_height), Image.BILINEAR)

    # Create canvas with gray padding (114/255 ≈ 0.447)
    canvas = np.full((target_height, target_width, 3), 114.0 / 255.0, dtype=np.float32)

    # Paste resized image at top-left (official Ultralytics standard)
    resized_array = np.array(resized).astype(np.float32) / 255.0
    canvas[:new_height, :new_width, :] = resized_array

    return canvas

def yolo_center_pad(image_path, height, width, pad):
    img = Image.open(image_path).convert('RGB').resize((width, height))
    img_padded = np.zeros((height + 2 * pad, width + 2 * pad, 3), dtype=np.float32)
    img_padded[pad:pad + height, pad:pad + width, :] = np.float32(img) / 255
    return img_padded

if __name__ == "__main__":
    # modelopt's int8.quantize() (imported above as quantize_int8) writes its
    # intermediate quantized model via tempfile.mkstemp(suffix=".onnx") but only
    # deletes it from the top-level modelopt.onnx.quantization.quantize wrapper,
    # which we bypass by importing int8.quantize directly. Left alone, every run
    # orphans a ~100 MB .onnx file in /tmp. Redirect all tempfiles into a private
    # dir and remove it on exit (incl. on unhandled exceptions, via atexit) so
    # nothing accumulates.
    import tempfile, atexit, shutil
    _tmpdir = tempfile.mkdtemp(prefix="quantize_model.")
    tempfile.tempdir = _tmpdir
    atexit.register(shutil.rmtree, _tmpdir, ignore_errors=True)

    parser = argparse.ArgumentParser(description='Quantize ONNX model with calibration data')
    parser.add_argument('--image-path', required=True, help='Path to the input image for calibration')
    parser.add_argument('--original-model-path', required=True, help='Path to the original ONNX model')
    parser.add_argument('--quantized-model-path', required=True, help='Path to save the quantized model')
    parser.add_argument('--app', choices=['classification', 'detection'], required=True,
                       help='Application type: classification or detection')
    parser.add_argument('--remove-first-quantize-node', action='store_true', help='Remove the first quantize node (optional)')
    parser.add_argument('--save-preprocessed-image', help='Path to save preprocessed image binary (optional)')
    parser.add_argument('--output-nodes', help='Comma-separated list of output node names to keep in the graph (optional)')

    args = parser.parse_args()

    image_path = args.image_path
    original_model_path = args.original_model_path
    quantized_model_path = args.quantized_model_path
    app = args.app

    # Write the intermediate simplified model next to the quantized output, so
    # that several quantizations of the same input model can run concurrently.
    simplified_model_path = os.path.join(
        os.path.dirname(os.path.abspath(quantized_model_path)),
        os.path.basename(original_model_path).replace(".onnx", ".simplified.onnx"),
    )

    # Parse output node names early — slicing happens PRE-quantization
    output_node_names = [n.strip() for n in args.output_nodes.split(",")] if args.output_nodes else []
    if output_node_names:
        print(f"Slicing graph outputs to: {output_node_names} (before quantization)")

    model = onnx.load(original_model_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    assert len(model_simp.graph.input) == 1, "Model must have exactly one input"
    input_dims = model_simp.graph.input[0].type.tensor_type.shape.dim
    assert len(input_dims) >= 3, "Model must have at least 3 dimensions"
    height = input_dims[-2].dim_value
    width = input_dims[-1].dim_value
    print(f"Height: {height}, Width: {width}")

    # Slice to requested outputs before quantization so calibration runs on a smaller graph
    if output_node_names:
        graph_simp = gs.import_onnx(model_simp)
        tensors_simp = graph_simp.tensors()
        graph_simp.outputs = [tensors_simp[name] for name in output_node_names]
        graph_simp.cleanup().toposort()
        model_simp = gs.export_onnx(graph_simp)
        model_simp.ir_version = model.ir_version

    onnx.save(model_simp, simplified_model_path)

    if app == "classification":
        data = imagenet_normalize(image_path, height, width)
    elif app == "detection":
        data = yolo_letterbox(image_path, height, width)
    else:
        raise ValueError(f"Invalid application: {app}")

    # convert RGB to NCHW
    batch_data = np.expand_dims(data.transpose(2,0,1), axis=0)

    provider = CalibrationDataProvider(simplified_model_path, batch_data)
    model_quant = quantize_int8(onnx_path=simplified_model_path, calibration_data_reader=provider, output_path=quantized_model_path, use_external_data_format=False, nodes_to_quantize=[], nodes_to_exclude=["TopK"])
    model_quant.ir_version = 10  # manually set ir_version since we're using an older version of onnxruntime

    # Concretize symbolic batch dimension to 1
    for inp in model_quant.graph.input:
        dim = inp.type.tensor_type.shape.dim[0]
        if dim.dim_param or dim.dim_value == 0:
            print(f"Concretizing batch size from symbolic to 1")
            dim.ClearField("dim_param")
            dim.dim_value = 1
    del model_quant.graph.value_info[:]
    model_quant = onnx.shape_inference.infer_shapes(model_quant)

    first_q_scale = None

    if args.remove_first_quantize_node:
        input_name = model_simp.graph.input[0].name
        print("Removing first quantize node ", input_name)
        quant_node = next(
            n for n in model_quant.graph.node
            if n.op_type == 'QuantizeLinear' and n.input[0] == input_name
        )
        scale_name = quant_node.input[1]
        scale_initializer = next(i for i in model_quant.graph.initializer if i.name == scale_name)
        first_q_scale = float(onnx.numpy_helper.to_array(scale_initializer).flat[0])

        graph = gs.import_onnx(model_quant)
        tensors = graph.tensors()
        graph.inputs = [tensors[quant_node.output[0]]]
        graph.cleanup().toposort()
        model_quant = gs.export_onnx(gs.Graph(graph.nodes, inputs=graph.inputs, outputs=graph.outputs, opset=20))
        model_quant.ir_version = 10

    onnx.save(model_quant, quantized_model_path)

    if args.save_preprocessed_image:
        if first_q_scale is not None:
            # Apply quantization: scale, clip, cast to int8
            # data is already in HWC format (PVA format)
            img_quant = np.clip(np.round(data / first_q_scale), -128, 127).astype(np.int8)
            img_quant.tofile(args.save_preprocessed_image)
            print(f"Saved quantized preprocessed image (HWC format) to {args.save_preprocessed_image}")
        else:
            # data is already in HWC format (PVA format)
            data.tofile(args.save_preprocessed_image)
            print(f"Saved preprocessed image (HWC format) to {args.save_preprocessed_image}")
