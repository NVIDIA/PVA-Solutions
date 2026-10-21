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

# YOLOv5 pipelines

Two pipelines run the same YOLOv5s detection workload, so the cost of moving
pre- and postprocessing onto the PVA can be measured directly:

- `gpu`: every stage on the iGPU, with decode and NMS folded into the TensorRT
  engine graph.
- `pva_gpu`: PVA resize and letterbox/normalize, TensorRT inference on the GPU
  or DLA, then PVA decode and NMS, overlapped three frames deep.

Both print their stage placement at startup, so which processor runs each stage
is visible in any run's output.

These pipelines are built for Linux targets only, because they need the AArch64
TensorRT 10 SDK (TensorRT 11 is not supported). See the repository README for prerequisites.

## Models

The build does not download the models. Prepare them once before you
configure:

1. Download `yolov5s_trimmed_reshape_tranpose.onnx` from the URL listed in
   `samples/assets/onnx/networks.txt` into `samples/assets/onnx/`. The
   PVA+GPU pipeline runs this three-head model directly.
2. The build generates the GPU-only model `yolov5s_672_nms.onnx` from it with
   `samples/assets/onnx/prepare_yolov5_nms_onnx.py` and installs it with the
   sample assets. The script needs the `onnx` and `onnx_graphsurgeon` Python
   packages. It adds YOLOv5 decode and EfficientNMS to the same weights, with
   the confidence and NMS IoU thresholds fixed in the model. The script is also
   installed next to the pipeline binaries, to generate the model by hand:

   ```bash
   python3 prepare_yolov5_nms_onnx.py yolov5s_trimmed_reshape_tranpose.onnx \
       yolov5s_672_nms.onnx --score-threshold 0.25 --iou-threshold 0.45
   ```

## Run the demo

```bash
./yolov5_gpu --image traffic.png \
    yolov5s_672_nms.onnx yolov5s_672_nms.engine

./yolov5_pva_gpu --image traffic.png \
    yolov5s_trimmed_reshape_tranpose.onnx \
    yolov5s_trimmed_reshape_tranpose.engine
```

Each takes an ONNX model followed by an optional TensorRT engine cache path,
and prints the stage placement, the configuration and the detections. Add
`--check-topk 5` to fail unless the five highest-scoring detections match the
reference for the packaged `traffic.png`. The GPU-only pipeline reports
per-stage timings in `--benchmark` and `--eval` runs; the PVA+GPU pipeline
reports them in `--benchmark` runs only.

See [`gpu/README.md`](gpu/README.md) and
[`pva_gpu/README.md`](pva_gpu/README.md) for the full argument list and for the
INT8, benchmark and COCO evaluation commands.
