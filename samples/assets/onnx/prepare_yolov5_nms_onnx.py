#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Add EfficientNMS_TRT plugin to a YOLOv5 ONNX model.

Takes either a standard decoded YOLOv5 ONNX model (output: [1, N, 85]) or the
trimmed three-head model used by the PVA pipeline. For three-head input it adds
the anchor/grid decode before appending EfficientNMS, so TensorRT handles the
complete GPU-only post-processing path.

Usage:
    pip3 install -r requirements.txt
    python3 prepare_yolov5_nms_onnx.py yolov5s_trimmed_reshape_tranpose.onnx \
        yolov5s_672_nms.onnx --score-threshold 0.25 --iou-threshold 0.45

The output model has 4 outputs:
    num_detections   [1, 1]          int32
    detection_boxes  [1, maxDets, 4] float32  (x1, y1, x2, y2)
    detection_scores [1, maxDets]    float32
    detection_classes[1, maxDets]    int32
"""

import argparse
import numpy as np
import onnx
import onnx_graphsurgeon as gs


YOLOV5_ANCHORS = {
    8: ((10.0, 13.0), (16.0, 30.0), (33.0, 23.0)),
    16: ((30.0, 61.0), (62.0, 45.0), (59.0, 119.0)),
    32: ((116.0, 90.0), (156.0, 198.0), (373.0, 326.0)),
}


def _slice_last(graph, value, start: int, end: int, name: str):
    """Slice [1, N, C] along the final dimension."""
    result = gs.Variable(name, dtype=np.float32)
    graph.nodes.append(gs.Node(
        op="Slice",
        inputs=[
            value,
            gs.Constant(f"{name}_starts", np.array([start], dtype=np.int64)),
            gs.Constant(f"{name}_ends", np.array([end], dtype=np.int64)),
            gs.Constant(f"{name}_axes", np.array([2], dtype=np.int64)),
        ],
        outputs=[result],
        name=f"{name}_slice",
    ))
    return result


def decode_three_head_outputs(graph):
    """Decode three raw YOLOv5 CHW heads into boxes and class scores."""
    if len(graph.inputs) != 1 or len(graph.inputs[0].shape) != 4:
        raise ValueError("Three-head YOLOv5 input must be a static NCHW tensor")

    _, _, model_h, model_w = graph.inputs[0].shape
    if not all(isinstance(v, int) and v > 0 for v in (model_h, model_w)):
        raise ValueError(f"Three-head YOLOv5 input must be static, got {graph.inputs[0].shape}")
    if model_h != model_w:
        raise ValueError(f"Only square YOLOv5 inputs are supported, got {model_w}x{model_h}")

    heads = []
    for output in graph.outputs:
        if output.shape is None or len(output.shape) != 4:
            raise ValueError(f"Expected CHW YOLOv5 head, got {output.name}: {output.shape}")
        batch, channels, grid_h, grid_w = output.shape
        if batch != 1 or channels != 255 or not all(isinstance(v, int) and v > 0 for v in (grid_h, grid_w)):
            raise ValueError(f"Expected [1,255,H,W] YOLOv5 head, got {output.name}: {output.shape}")
        if model_h % grid_h != 0 or model_w % grid_w != 0:
            raise ValueError(f"Head {output.name} does not divide model input {model_w}x{model_h}")
        stride_h = model_h // grid_h
        stride_w = model_w // grid_w
        if stride_h != stride_w or stride_h not in YOLOV5_ANCHORS:
            raise ValueError(f"Unsupported YOLOv5 stride for {output.name}: {stride_w}x{stride_h}")
        heads.append((stride_h, output, grid_h, grid_w))

    heads.sort(key=lambda item: item[0])
    if [item[0] for item in heads] != [8, 16, 32]:
        raise ValueError(f"Expected strides 8/16/32, got {[item[0] for item in heads]}")

    decoded_boxes = []
    decoded_scores = []
    for stride, output, grid_h, grid_w in heads:
        tag = f"s{stride}"
        count = grid_h * grid_w * 3
        reshaped_chw = gs.Variable(f"{tag}_reshape_chw", dtype=np.float32)
        graph.nodes.append(gs.Node(
            op="Reshape",
            inputs=[output, gs.Constant(f"{tag}_shape_chw", np.array([1, 3, 85, grid_h, grid_w], dtype=np.int64))],
            outputs=[reshaped_chw],
            name=f"{tag}_reshape_chw_node",
        ))
        anchor_major = gs.Variable(f"{tag}_anchor_major", dtype=np.float32)
        graph.nodes.append(gs.Node(
            op="Transpose", attrs={"perm": [0, 3, 4, 1, 2]}, inputs=[reshaped_chw],
            outputs=[anchor_major], name=f"{tag}_transpose",
        ))
        flat = gs.Variable(f"{tag}_flat", dtype=np.float32, shape=[1, count, 85])
        graph.nodes.append(gs.Node(
            op="Reshape",
            inputs=[anchor_major, gs.Constant(f"{tag}_shape_flat", np.array([1, count, 85], dtype=np.int64))],
            outputs=[flat], name=f"{tag}_reshape_flat_node",
        ))

        activated = gs.Variable(f"{tag}_sigmoid", dtype=np.float32, shape=[1, count, 85])
        graph.nodes.append(gs.Node(op="Sigmoid", inputs=[flat], outputs=[activated], name=f"{tag}_sigmoid_node"))
        xy = _slice_last(graph, activated, 0, 2, f"{tag}_xy")
        wh = _slice_last(graph, activated, 2, 4, f"{tag}_wh")
        objectness = _slice_last(graph, activated, 4, 5, f"{tag}_objectness")
        classes = _slice_last(graph, activated, 5, 85, f"{tag}_classes")

        grid = np.array([[[x, y] for y in range(grid_h) for x in range(grid_w) for _ in range(3)]],
                        dtype=np.float32)
        anchor_wh = np.array([[YOLOV5_ANCHORS[stride][a]
                               for _y in range(grid_h) for _x in range(grid_w) for a in range(3)]],
                             dtype=np.float32)
        two = gs.Constant(f"{tag}_two", np.array([2.0], dtype=np.float32))
        half = gs.Constant(f"{tag}_half", np.array([0.5], dtype=np.float32))

        xy_twice = gs.Variable(f"{tag}_xy_twice", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Mul", inputs=[xy, two], outputs=[xy_twice], name=f"{tag}_xy_twice_node"))
        xy_offset = gs.Variable(f"{tag}_xy_offset", dtype=np.float32)
        graph.nodes.append(gs.Node(
            op="Sub", inputs=[xy_twice, half], outputs=[xy_offset],
            name=f"{tag}_xy_offset_node",
        ))
        xy_grid = gs.Variable(f"{tag}_xy_grid", dtype=np.float32)
        graph.nodes.append(gs.Node(
            op="Add", inputs=[xy_offset, gs.Constant(f"{tag}_grid", grid)], outputs=[xy_grid],
            name=f"{tag}_xy_grid_node",
        ))
        centers = gs.Variable(f"{tag}_centers", dtype=np.float32)
        graph.nodes.append(gs.Node(
            op="Mul", inputs=[xy_grid, gs.Constant(f"{tag}_stride", np.array([float(stride)], dtype=np.float32))],
            outputs=[centers], name=f"{tag}_centers_node",
        ))

        wh_twice = gs.Variable(f"{tag}_wh_twice", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Mul", inputs=[wh, two], outputs=[wh_twice], name=f"{tag}_wh_twice_node"))
        wh_squared = gs.Variable(f"{tag}_wh_squared", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Mul", inputs=[wh_twice, wh_twice], outputs=[wh_squared],
                                   name=f"{tag}_wh_squared_node"))
        sizes = gs.Variable(f"{tag}_sizes", dtype=np.float32)
        graph.nodes.append(gs.Node(
            op="Mul", inputs=[wh_squared, gs.Constant(f"{tag}_anchors", anchor_wh)], outputs=[sizes],
            name=f"{tag}_sizes_node",
        ))
        half_sizes = gs.Variable(f"{tag}_half_sizes", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Mul", inputs=[sizes, half], outputs=[half_sizes],
                                   name=f"{tag}_half_sizes_node"))
        xy_min = gs.Variable(f"{tag}_xy_min", dtype=np.float32)
        xy_max = gs.Variable(f"{tag}_xy_max", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Sub", inputs=[centers, half_sizes], outputs=[xy_min], name=f"{tag}_xy_min_node"))
        graph.nodes.append(gs.Node(op="Add", inputs=[centers, half_sizes], outputs=[xy_max], name=f"{tag}_xy_max_node"))
        boxes = gs.Variable(f"{tag}_boxes", dtype=np.float32, shape=[1, count, 4])
        graph.nodes.append(gs.Node(op="Concat", attrs={"axis": 2}, inputs=[xy_min, xy_max], outputs=[boxes],
                                   name=f"{tag}_boxes_node"))
        scores = gs.Variable(f"{tag}_scores", dtype=np.float32, shape=[1, count, 80])
        graph.nodes.append(gs.Node(op="Mul", inputs=[objectness, classes], outputs=[scores],
                                   name=f"{tag}_scores_node"))
        decoded_boxes.append(boxes)
        decoded_scores.append(scores)

    boxes = gs.Variable("boxes_xyxy", dtype=np.float32,
                        shape=[1, sum(v.shape[1] for v in decoded_boxes), 4])
    scores = gs.Variable("scores", dtype=np.float32,
                         shape=[1, sum(v.shape[1] for v in decoded_scores), 80])
    graph.nodes.append(gs.Node(op="Concat", attrs={"axis": 1}, inputs=decoded_boxes, outputs=[boxes],
                               name="ConcatDecodedBoxes"))
    graph.nodes.append(gs.Node(op="Concat", attrs={"axis": 1}, inputs=decoded_scores, outputs=[scores],
                               name="ConcatDecodedScores"))
    return boxes, scores


def add_nms(input_onnx: str, output_onnx: str,
            score_threshold: float = 0.001,
            iou_threshold: float = 0.65,
            max_detections: int = 300):
    """Append decode + EfficientNMS_TRT to a YOLOv5 ONNX graph."""

    graph = gs.import_onnx(onnx.load(input_onnx))
    graph.fold_constants()
    graph.cleanup().toposort()

    # A standard exported model has one decoded [1,N,85] output. The trimmed
    # model shared with the PVA+GPU pipeline has three raw [1,255,H,W] heads and
    # needs the YOLOv5 anchor/grid decode before EfficientNMS.
    if len(graph.outputs) not in (1, 3):
        raise ValueError(f"Expected one decoded output or three raw heads, got {len(graph.outputs)}")
    if len(graph.outputs) == 3:
        print("Original outputs: three raw YOLOv5 heads")
        boxes_xyxy, scores = decode_three_head_outputs(graph)
    else:
        yolo_output = graph.outputs[0]
        print(f"Original output: {yolo_output.name}, shape={yolo_output.shape}, dtype={yolo_output.dtype}")

        num_classes = 80
        # Expected shape: [1, N, 85] where N = 25200 for 640x640 input
        expected_attrs = num_classes + 5
        if yolo_output.shape and yolo_output.shape[-1] != expected_attrs:
            raise ValueError(f"Expected last dim = {expected_attrs}, got {yolo_output.shape[-1]}")

        # ── Slice boxes: output[:, :, 0:4] → [1, N, 4] as (cx, cy, w, h) ──
        boxes_cxcywh = _slice_last(graph, yolo_output, 0, 4, "boxes_cxcywh")

        # ── Convert cx,cy,w,h → x1,y1,x2,y2 ──
        half = gs.Constant("half", np.array([2.0], dtype=np.float32))

        cxy = _slice_last(graph, boxes_cxcywh, 0, 2, "cxy")
        wh = _slice_last(graph, boxes_cxcywh, 2, 4, "wh")

        half_wh = gs.Variable("half_wh", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Div", inputs=[wh, half], outputs=[half_wh]))

        xy_min = gs.Variable("xy_min", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Sub", inputs=[cxy, half_wh], outputs=[xy_min]))

        xy_max = gs.Variable("xy_max", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Add", inputs=[cxy, half_wh], outputs=[xy_max]))

        boxes_xyxy = gs.Variable("boxes_xyxy", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Concat", attrs={"axis": 2}, inputs=[xy_min, xy_max], outputs=[boxes_xyxy]))

        objectness = _slice_last(graph, yolo_output, 4, 5, "objectness")
        class_scores = _slice_last(graph, yolo_output, 5, 85, "class_scores")

        scores = gs.Variable("scores", dtype=np.float32)
        graph.nodes.append(gs.Node(op="Mul", inputs=[objectness, class_scores], outputs=[scores]))

    # ── EfficientNMS_TRT plugin ──
    nms_attrs = {
        "plugin_version": "1",
        "background_class": -1,
        "max_output_boxes": max_detections,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "score_activation": False,
        "class_agnostic": False,
        "box_coding": 0,  # 0 = [x1, y1, x2, y2]
    }

    num_detections = gs.Variable("num_detections", dtype=np.int32, shape=[1, 1])
    detection_boxes = gs.Variable("detection_boxes", dtype=np.float32, shape=[1, max_detections, 4])
    detection_scores = gs.Variable("detection_scores", dtype=np.float32, shape=[1, max_detections])
    detection_classes = gs.Variable("detection_classes", dtype=np.int32, shape=[1, max_detections])

    nms_node = gs.Node(
        op="EfficientNMS_TRT",
        attrs=nms_attrs,
        inputs=[boxes_xyxy, scores],
        outputs=[num_detections, detection_boxes, detection_scores, detection_classes],
    )
    graph.nodes.append(nms_node)

    graph.outputs = [num_detections, detection_boxes, detection_scores, detection_classes]
    graph.cleanup().toposort()

    model = gs.export_onnx(graph)
    onnx.save(model, output_onnx)

    print(f"\nSaved: {output_onnx}")
    print(f"  score_threshold: {score_threshold}")
    print(f"  iou_threshold:   {iou_threshold}")
    print(f"  max_detections:  {max_detections}")
    print(f"  Outputs:")
    for o in graph.outputs:
        print(f"    {o.name:25s} shape={o.shape}  dtype={o.dtype}")
    print(f"\nBuild engine with:")
    print(f"  trtexec --onnx={output_onnx} --fp16 --saveEngine={output_onnx.replace('.onnx', '.engine')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add EfficientNMS_TRT to YOLOv5 ONNX model")
    parser.add_argument("input", help="Input ONNX model (e.g., yolov5s.onnx)")
    parser.add_argument("output", help="Output ONNX model (e.g., yolov5s_nms.onnx)")
    parser.add_argument("--score-threshold", type=float, default=0.001,
                        help="Score threshold (default: 0.001 for COCO eval)")
    parser.add_argument("--iou-threshold", type=float, default=0.65,
                        help="NMS IoU threshold (default: 0.65)")
    parser.add_argument("--max-detections", type=int, default=300,
                        help="Max output detections per image (default: 300)")
    args = parser.parse_args()

    add_nms(args.input, args.output,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            max_detections=args.max_detections)
