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

# ONNX Model Assets

`networks.txt` lists the ONNX models used by the deep learning samples and the
YOLOv5 pipelines. Each line contains a download URL followed by the file name
to save the model as. The models are not included in this directory, and the
build does not download them.

## YOLOv5 Models

The PVA+GPU pipeline uses `yolov5s_trimmed_reshape_tranpose.onnx`, a YOLOv5s
model with three raw detection heads. The GPU-only pipeline uses
`yolov5s_672_nms.onnx`, which is generated from the same model by
`prepare_yolov5_nms_onnx.py`. The script adds the YOLOv5 anchor/grid decode and
the TensorRT EfficientNMS plugin, so TensorRT runs the complete GPU-only
post-processing.

To prepare the models:

1. Install the Python dependencies:

   ```shell
   pip3 install -r requirements.txt
   ```

2. Download `yolov5s_trimmed_reshape_tranpose.onnx` from the URL listed in
   `networks.txt` into this directory.

The L4T build then generates `yolov5s_672_nms.onnx` and installs it with the
sample assets. To generate it by hand, for example with other thresholds:

```shell
python3 prepare_yolov5_nms_onnx.py yolov5s_trimmed_reshape_tranpose.onnx \
    yolov5s_672_nms.onnx --score-threshold 0.25 --iou-threshold 0.45
```

These thresholds match the default thresholds of the PVA+GPU pipeline. Run
`python3 prepare_yolov5_nms_onnx.py --help` for all options.
