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

# GPU-only YOLOv5 pipeline

## Run

Display the complete command-line reference:

```bash
./yolov5_gpu --help
```

The build generates the GPU-only model, which adds the YOLOv5 decode and
EfficientNMS to the network. To generate it by hand, for example with other
thresholds:

```bash
python3 prepare_yolov5_nms_onnx.py yolov5s_trimmed_reshape_tranpose.onnx yolov5s_672_nms.onnx \
    --score-threshold 0.25 --iou-threshold 0.45
```

Run inference on one image:

```bash
./yolov5_gpu \
    --image traffic.png \
    yolov5s_672_nms.onnx yolov5s_672_nms.engine
```

Build and use an INT8 engine:

```bash
./yolov5_gpu \
    --int8 --calib-images ./val2017_bin \
    --image traffic.png \
    yolov5s_672_nms.onnx yolov5s_672_nms.int8.engine
```

Run a steady-state benchmark with a CUDA Graph. In benchmark mode the input
image is decoded once into pinned host memory before the measured iterations,
so image decoding is not timed:

```bash
./yolov5_gpu \
    --int8 --calib-images ./val2017_bin --benchmark --cuda-graph \
    --warmup 100 --iterations 10000 \
    --image traffic.png \
    yolov5s_672_nms.onnx yolov5s_672_nms.int8.engine
```

Run COCO evaluation:

Confidence and NMS IoU thresholds are defined by the ONNX model, not by runtime
options. The model generated above uses 0.25 and 0.45 respectively; `--eval`
does not change them. For evaluation thresholds such as 0.001 and 0.65,
regenerate the model with
`prepare_yolov5_nms_onnx.py --score-threshold 0.001 --iou-threshold 0.65`
and build a new engine. Match these thresholds when comparing with PVA+GPU.
The GPU-only pipeline rejects `--conf` and `--iou`.

```bash
./yolov5_gpu \
    --int8 --calib-images ./val2017_bin \
    --eval --images-dir ./val2017 \
    --coco-output results.json \
    yolov5s_672_nms.onnx yolov5s_672_nms.int8.engine
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--image, -i <path>` | Input PNG or pre-decoded BIN image (layout in [../utils/README.md](../utils/README.md)). |
| `--fp32` | Use FP32 instead of the default FP16 mode. |
| `--int8` | Use INT8 precision. |
| `--calib-images <dir>` | Images used to create an INT8 calibration cache. |
| `--expected-detections <count>` | Fail unless the final frame has this detection count. |
| `--check-topk <n>` | Fail unless the n highest-scoring detections each match a distinct reference detection (same class, IoU >= threshold). Portable across TensorRT versions and GPUs, unlike an exact count. Traffic image only. |
| `--check-topk-iou <val>` | IoU threshold for `--check-topk` (default 0.50). |
| `--eval` | Enable evaluation mode. |
| `--images-dir <dir>` | Input directory for evaluation. |
| `--coco-output <path>` | COCO prediction JSON output. |
| `--summary-json <path>` | Write timing summary JSON to this path. |
| `--benchmark` | Repeated-image benchmark mode. |
| `--iterations <count>` | Number of measured benchmark iterations. |
| `--warmup <count>` | Number of unmeasured warm-up iterations. |
| `--preload` | Preload `.bin` evaluation inputs into pinned host memory before processing. The benchmark input is always preloaded. |
| `--cuda-graph` | Capture TensorRT inference in a CUDA Graph; requires benchmark mode. |
| `--help, -h` | Display the complete runtime help. |

The first positional argument is the ONNX model. The optional second positional
argument is the TensorRT engine cache path. The maximum number of detections is
read from the engine; set it with `prepare_yolov5_nms_onnx.py --max-detections`.
