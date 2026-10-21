<!--
 SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# LIDAR Histogram Visualization

This document describes how to use the histogram visualization feature to understand and debug
the lidar_histogram_conversion sample application.

## Overview

The lidar_histogram_conversion sample application generates simulated histogram tensor data, runs the reference
C++ implementation and PVA implementations, then compares results and dumps input and output tensors to files.
There are several compile-time options to enable dumping out intermediate results as well.  The python script
provides a helpful visualization of the histogram data, output tensors and intermediate results.

## Workflow

### Step 1: Run the Sample Application

Run the lidar_histogram_conversion sample application. It will automatically generate random histogram data and dump it
to a file named `histogram_data.bin`, and estimate range values from histogram peaks and dump them to `range_data.bin`:

```bash
./lidar_histogram_conversion
```

See [debug_options.h](./ref/debug_options.h) for compile time flags that change the behavior of the reference code and
sample application.  For example, if the corresponding compile time option is enabled, a `smooth_histogram_data.bin`
file will also be dumped by the application.

### Step 2: Generate histogram plots

Use the Python script to visualize histogram data for a specific pixel:

```bash
python visualize_histogram.py <input_file> <x> <y> [output_file] [--range-file RANGE_FILE] [--range-is-peaks]
```

**Arguments:**
 - `input_file`       : Path to the binary histogram data file (supports RAW12 packing)
 - `x`                : X coordinate of the pixel to visualize (0-based)
 -  `y`               : Y coordinate of the pixel to visualize (0-based)
 -  `output_file`     : Output PNG filename (default: <input_file_name>_<x>_<y>.png)
 - `--range-file`     : Path to the range tensor binary file (optional)
 - `--range-is-peaks` : Interpret range values as peak indices for debugging (optional)

**Example:**
```bash
python3 visualize_histogram.py histogram_data.bin 0 0 my_histogram.png
```

The script will generate a PNG image file containing:
- Bar a plot for each histogram at the specified pixel location
- Pixel header information
- Histogram header information

The script will also print statistics to the console:
```
=== Histogram Statistics ===
Pixel Header: [1234 2345 3456 4567]

Histogram 0:
  Header: [100]
  Bins: [150 200 175]
  Total counts: 525
  Max bin value: 200 (bin 1)
```

## Python Dependencies

The visualization script requires the following Python packages:
- `numpy`
- `matplotlib`

Install them using:
```bash
pip install numpy matplotlib
```

## Binary File Format

The `histogram_data.bin` file has the following structure:

### Tensor Header (16 bytes)
- `width` (int32): Size in W dim
- `height` (int32): Size in H dim
- `channels` (int32): Size in C dim
- `data_type_size` (int32): Size of a channel element in bytes (1,2, or 4)

### Histogram Specification Header (24 bytes)
- `packing_type` (int32): Data packing type (0=NONE, 1=RAW12)
- `num_hist_per_pixel` (int32): Number of histograms per pixel
- `num_bins_per_hist` (int32): Number of bins in each histogram
- `num_peaks_per_hist` (int32): Number of peaks to detect per histogram
- `pixel_header_size` (int32): Size of pixel header in elements
- `hist_header_size` (int32): Size of histogram header in elements

The structure matches `PvaLidarHistogramSpec` in the C code.

### Data
- Raw tensor data in row-major order: `[height][width][channels]`
- Layout depends on packing type

## Histogram Structure

Each pixel contains the following data structure (for the default configuration):

```
Pixel data:
  - Pixel header (`pixel_header_size` elements)
  - Histogram 0:
    - Histogram header (`hist_header_size` elements)
    - Bins (`num_bins_per_hist` elements)
  - [Histogram 1]:[Histogram `num_hist_per_pixel`-1]
```

## Customization

### Modifying Tensor Output Location

To change the output filename or add conditional dumping, modify the call in `lidar_histogram_conversion.c`:

```c
// Dump histogram tensor to file for visualization
if (dumpHistogramTensor(histogramTensorHandle, "my_custom_name.bin", &histogramSpec) != 0)
{
    printf("Warning: Failed to dump histogram tensor\n");
}
```

### Adjusting Histogram Specification

The histogram specification is automatically embedded in the binary file, so the Python script will always correctly interpret the data structure regardless of your configuration. Simply modify the `PvaLidarHistogramSpec` in the C code:

```c
PvaLidarHistogramSpec histogramSpec;
histogramSpec.packingType     = PVA_LIDAR_DATA_PACKING_TYPE_NONE;
histogramSpec.numHistPerPixel = 3;    // Change as needed
histogramSpec.numBinsPerHist  = 512;  // Change as needed
histogramSpec.numPeaksPerHist = 1;    // Change as needed
histogramSpec.pixelHeaderSize = 4;    // Change as needed
histogramSpec.histHeaderSize  = 1;    // Change as needed
```

The Python script will automatically read these values from the file and visualize accordingly.

### Visualization Adaptations

The Python script automatically adapts its visualization based on the histogram size:
- **Small histograms** (≤50 bins): Displays as bar charts with value labels
- **Large histograms** (>50 bins): Displays as line plots with filled areas

If a range tensor is provided as input, the corresponding histogram peaks are highlighted by inverting
the range value to a bin index.

## Troubleshooting

### "Error: Input file not found"
Make sure you run the sample application first to generate the `histogram_data.bin` file.

### "Pixel coordinates out of bounds"
Check the tensor shape printed when running the Python script and ensure your x,y coordinates are within the valid range.

### "No module named 'matplotlib'"
Install the required Python dependencies:
```bash
pip install numpy matplotlib
```

### Incorrect histogram structure
The histogram specification is now automatically read from the binary file. If you see structure mismatch warnings, there may be an issue with the data file generation or the file may be corrupted. Regenerate the file by running the C application again.

