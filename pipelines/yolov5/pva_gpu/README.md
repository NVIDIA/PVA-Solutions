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

# PVA+GPU YOLOv5 pipeline

## Run

Confirm that the installed PVA SDK and driver support CUDA-stream submission:

```bash
./yolov5_pva_gpu --check-pva-driver
```

Exit status `0` indicates support. Exit status `77` indicates that the SDK or
driver is too old for this pipeline.

Display the complete command-line reference:

```bash
./yolov5_pva_gpu --help
```

Run inference on the packaged traffic image and check the five
highest-scoring detections against its reference:

```bash
./yolov5_pva_gpu \
    --image traffic.png --check-topk 5 \
    yolov5s_trimmed_reshape_tranpose.onnx \
    yolov5s_trimmed_reshape_tranpose.engine
```

Build an INT8 engine and run a steady-state benchmark. `--int8` requires
`--calib-images`, a directory of PNG or pre-decoded BIN images used to
calibrate the engine:

```bash
./yolov5_pva_gpu \
    --image traffic.png \
    --int8 --calib-images ./calibration_images --benchmark \
    --check-topk 5 \
    --warmup 100 --iterations 10000 \
    yolov5s_trimmed_reshape_tranpose.onnx \
    yolov5s_trimmed_reshape_tranpose.int8.engine
```

Add `--cuda-graph` to capture TensorRT inference for each persistent pipeline
slot:

```bash
./yolov5_pva_gpu \
    --image traffic.png \
    --int8 --calib-images ./calibration_images --benchmark --cuda-graph \
    --check-topk 5 \
    --warmup 100 --iterations 10000 \
    yolov5s_trimmed_reshape_tranpose.onnx \
    yolov5s_trimmed_reshape_tranpose.int8.engine
```

Benchmark mode reports the steady-state pipeline interval and the PVA
preprocessing, TensorRT inference, and PVA postprocessing service times. The
three stages operate concurrently on different frames, so their service times
must not be added to obtain the pipeline interval. Image decoding, engine
loading, initialization, result materialization, and file output are excluded
from the measured interval.

Run COCO evaluation:

`--eval` lowers the confidence threshold to 0.001 and raises the NMS IoU
threshold to 0.65 unless `--conf` or `--iou` is given. The images directory
must contain PNG or pre-decoded BIN images. Match these thresholds when
comparing with the GPU-only pipeline, whose thresholds are fixed in its model.

```bash
./yolov5_pva_gpu \
    --eval --images-dir ./val2017 \
    --coco-output results.json \
    yolov5s_trimmed_reshape_tranpose.onnx \
    yolov5s_trimmed_reshape_tranpose.engine
```

When requested with `--summary-json`, the summary records the input path and
dimensions, precision, confidence and IoU thresholds, warm-up and measured
iteration counts, CUDA Graph state, detection count, and average and
percentile timing for the pipeline and, in benchmark mode, for each stage.

## Arguments

| Argument | Description |
|----------|-------------|
| `--check-pva-driver` | Check CUDA-stream submission support and exit. Must be the only argument. |
| `--image, -i <path>` | Input PNG or pre-decoded BIN image (layout in [../utils/README.md](../utils/README.md)). |
| `--fp32` | Use FP32 instead of the default FP16 mode. |
| `--int8` | Use INT8 precision. Requires `--calib-images`. |
| `--calib-images <dir>` | Images used to create an INT8 calibration cache. |
| `--dla, -d` | Run TensorRT inference on DLA with GPU fallback. |
| `--dla-core <index>` | Select DLA core 0 or 1. |
| `--conf <value>` | Confidence threshold. |
| `--iou <value>` | NMS IoU threshold. |
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
| `--preload` | Preload BIN evaluation inputs before processing. The benchmark input is always preloaded. |
| `--cuda-graph` | Capture TensorRT inference in CUDA Graphs; requires benchmark mode. |
| `--help, -h` | Display the complete runtime help. |

An input is required: `--image`, or `--eval` with `--images-dir`. The first
positional argument is the ONNX model. The optional second positional argument
is the TensorRT engine cache path.
