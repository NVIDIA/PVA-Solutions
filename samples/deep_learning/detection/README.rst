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

Detection
==============

This sample demonstrates object detection using YOLO models (YOLOv5, YOLO26, etc.).

Preprocessing
-----------------

The detection models use **letterbox preprocessing** to maintain the aspect ratio of input images while resizing to the model's expected input dimensions (typically 640x640).

Standard Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Resize with Aspect Ratio Preservation**

   Calculate the scale factor to fit the image within the target dimensions:

   .. code-block:: python

       scale = min(target_width / img_width, target_height / img_height)
       new_width = int(img_width * scale)
       new_height = int(img_height * scale)
       resized = image.resize((new_width, new_height), Image.BILINEAR)

2. **Padding**

   Apply gray padding (RGB: 114, 114, 114) to fill the remaining space. Standard Ultralytics YOLO uses **top-left alignment**:

   .. code-block:: python

       padded = Image.new('RGB', (target_width, target_height), (114, 114, 114))
       padded.paste(resized, (0, 0))  # Top-left alignment

3. **Normalization**

   Normalize pixel values from [0, 255] to [0, 1]:

   .. code-block:: python

       input_img = np.array(padded).astype(np.float32) / 255.0

4. **Optional: Quantize Node Fusion**

   For quantized models, the first QuantizeLinear node can be fused into preprocessing as an optimization. This converts the normalized float32 input directly to int8:

   .. code-block:: python
       # Quantize: round(input / scale) and clip to int8 range
       input_quantized = np.clip(np.round(input_img / scale_value), -128, 127).astype(np.int8)

   When quantize node fusion is enabled:

   - The preprocessing outputs **int8** instead of float32
   - The model's first QuantizeLinear operation is removed
   - The C++ sample uses this approach, reading preprocessed int8 data in **HWC format** (height, width, channels)

Postprocessing Coordinate Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After inference, detected bounding boxes are in the model's input space (e.g., 640x640). To project them back to the original image coordinates:

.. code-block:: python

    # Calculate the inverse scale factor
    inv_scale = 1.0 / scale

    # Project coordinates back to original image
    x1_original = x1_model * inv_scale
    y1_original = y1_model * inv_scale
    x2_original = x2_model * inv_scale
    y2_original = y2_model * inv_scale

For example, with a 1280x720 image resized to 640x640:

- Scale factor: ``min(640/1280, 640/720) = 0.5``
- Inverse scale: ``1 / 0.5 = 2.0``
- A detected box at [100, 50, 200, 150] in model space becomes [200, 100, 400, 300] in original image space

Visualization
-------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots()
    image = Image.open(image_path)
    ax.imshow(image)
    coordinates = [[left, top, right, bottom], ...]
    for coord in coordinates:
        l, t, r, b = coord
        ax.add_patch(patches.Rectangle((l, t), r-l, b-t, linewidth=1, edgecolor='r'))

    plt.show()
