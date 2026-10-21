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
# @file visualize_histogram.py
#
# @brief Visualize LIDAR histogram data from binary dump file.
#
# This script reads histogram tensor data dumped by the lidar_histogram_conversion sample
# application and generates plots of the histogram(s) for a specified pixel location.
#

"""
Visualize LIDAR histogram data from binary dump file.

This script reads histogram tensor data dumped by the lidar_histogram_conversion sample
application and generates plots of the histogram(s) for a specified pixel location.

Supports both unpacked (uint16) and RAW12-packed (uint8) histogram data formats.
For RAW12 format, the script automatically unpacks the data transparently.

When a range tensor file is provided without --range-is-peaks, the script will
automatically invert the range values to recover the approximate bin indices
using the histogram parameters (offsetNs, binSizeNs, rangeScale) from the
histogram file metadata.

Usage:
    python visualize_histogram.py <input_file> <x> <y> [output_file] [--range-file RANGE_FILE] [--range-is-peaks]

Arguments:
    input_file       : Path to the binary histogram data file (supports RAW12 packing)
    x                : X coordinate of the pixel to visualize (0-based)
    y                : Y coordinate of the pixel to visualize (0-based)
    output_file      : Output PNG filename (default: <input_file_name>_<x>_<y>.png)
    --range-file     : Path to the range tensor binary file (optional)
    --range-is-peaks : Interpret range values as peak indices for debugging (optional)

Example:
    python visualize_histogram.py histogram_data.bin 0 0 output.png
    python visualize_histogram.py histogram_data.bin 0 0 output.png --range-file range_data.bin --range-is-peaks
    python visualize_histogram.py histogram_data.bin 0 0 output.png --range-file range_data.bin
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class HistogramSpec:
    """Histogram specification matching PvaLidarHistogramSpec from C code."""
    def __init__(self):
        self.packing_type = 0
        self.num_hist_per_pixel = 1
        self.num_bins_per_hist = 3
        self.num_peaks_per_hist = 1
        self.pixel_header_size = 0
        self.hist_header_size = 0


class HistogramParams:
    """Histogram parameters matching PvaLidarHistogramParams from C code."""
    def __init__(self):
        self.offset_ns = 0.0
        self.bin_size_ns = 0.0
        self.range_scale = 1.0
        self.max_intensity = 1.0


# Packing type constants (matching C code)
PVA_LIDAR_DATA_PACKING_TYPE_NONE = 0
PVA_LIDAR_DATA_PACKING_TYPE_RAW12 = 1


def unpack_raw12(packed_bytes):
    """
    Unpack RAW12 data from 3 bytes to 2 uint16 values.

    RAW12 format packs two 12-bit values into 3 bytes:
        byte[0] = A[11:4]
        byte[1] = B[11:4]
        byte[2] = A[3:0] | B[3:0]  (upper nibble A, lower nibble B)

    Args:
        packed_bytes: 3-byte sequence

    Returns:
        tuple: (val0, val1) as uint16 values
    """
    val0 = (int(packed_bytes[0]) << 4) | ((int(packed_bytes[2]) >> 4) & 0x0F)
    val1 = (int(packed_bytes[1]) << 4) | (int(packed_bytes[2]) & 0x0F)
    return val0, val1


def unpack_pixel_data_raw12(packed_data, elements_per_pixel):
    """
    Unpack a full pixel's worth of RAW12 data to uint16 values.

    Args:
        packed_data: Byte array containing RAW12-packed data
        elements_per_pixel: Number of uint16 elements to unpack

    Returns:
        numpy array of uint16 values
    """
    output = np.zeros(elements_per_pixel, dtype=np.uint16)
    src_idx = 0

    for i in range(0, elements_per_pixel, 2):
        val0, val1 = unpack_raw12(packed_data[src_idx:src_idx+3])
        output[i] = val0
        if i + 1 < elements_per_pixel:
            output[i + 1] = val1
        src_idx += 3

    return output


def read_histogram_tensor(filename):
    """
    Read histogram tensor from binary file.
    Tensor data is returned as uint16 values, unpacked from RAW12 data if needed.

    File format:
        - 4 int32_t values: width, height, channels, data_type_size
        - 6 int32_t values: histogram spec (packingType, numHistPerPixel, numBinsPerHist,
                           numPeaksPerHist, pixelHeaderSize, histHeaderSize)
        - 4 float32 values: histogram params (offsetNs, binSizeNs, rangeScale, maxIntensity)
        - Followed by raw tensor data (row-major, not strided, shape: [height, width, channels])

    Returns:
        tuple: (tensor_data, width, height, channels, data_type_size, histogram_spec, histogram_params)
    """
    with open(filename, 'rb') as f:
        # Read tensor header
        header_bytes = f.read(4 * 4)  # 4 int32_t values
        if len(header_bytes) != 16:
            raise ValueError("Failed to read header from file")

        width, height, channels, data_type_size = struct.unpack('iiii', header_bytes)

        print(f"Tensor shape: [{height}, {width}, {channels}]")
        print(f"Data type size: {data_type_size} bytes")

        # Read histogram specification
        histspec_bytes = f.read(6 * 4)  # 6 int32_t values
        if len(histspec_bytes) != 24:
            raise ValueError("Failed to read histogram spec from file")

        packing_type, num_hist_per_pixel, num_bins_per_hist, num_peaks_per_hist, pixel_header_size, hist_header_size = struct.unpack('iiiiii', histspec_bytes)

        # Create histogram spec object
        hist_spec = HistogramSpec()
        hist_spec.packing_type = packing_type
        hist_spec.num_hist_per_pixel = num_hist_per_pixel
        hist_spec.num_bins_per_hist = num_bins_per_hist
        hist_spec.num_peaks_per_hist = num_peaks_per_hist
        hist_spec.pixel_header_size = pixel_header_size
        hist_spec.hist_header_size = hist_header_size

        print(f"\nHistogram Specification:")
        print(f"  Packing type: {packing_type}")
        print(f"  Histograms per pixel: {num_hist_per_pixel}")
        print(f"  Bins per histogram: {num_bins_per_hist}")
        print(f"  Peaks per histogram: {num_peaks_per_hist}")
        print(f"  Pixel header size: {pixel_header_size}")
        print(f"  Histogram header size: {hist_header_size}")

        # Read histogram parameters
        params_bytes = f.read(4 * 4)  # 4 float32 values
        if len(params_bytes) != 16:
            raise ValueError("Failed to read histogram params from file")

        offset_ns, bin_size_ns, range_scale, max_intensity = struct.unpack('ffff', params_bytes)

        # Create histogram params object
        hist_params = HistogramParams()
        hist_params.offset_ns = offset_ns
        hist_params.bin_size_ns = bin_size_ns
        hist_params.range_scale = range_scale
        hist_params.max_intensity = max_intensity

        print(f"\nHistogram Parameters:")
        print(f"  Offset (ns): {offset_ns}")
        print(f"  Bin size (ns): {bin_size_ns}")
        print(f"  Range scale: {range_scale}")
        print(f"  Max intensity: {max_intensity}")

        # Read tensor data
        # For RAW12 packing, we need to unpack the data after reading
        if packing_type == PVA_LIDAR_DATA_PACKING_TYPE_RAW12:
            # Calculate elements per pixel (in unpacked space)
            elements_per_pixel = (pixel_header_size +
                                 num_hist_per_pixel * (hist_header_size + num_bins_per_hist))

            bytes_per_pixel = channels * data_type_size

            # Check that the number of channels is enough to hold the unpacked data
            print(f"RAW12 packing: {elements_per_pixel} elements/pixel packed into {bytes_per_pixel} bytes/pixel")
            assert bytes_per_pixel >= elements_per_pixel * 12/8, "pixel size is smaller than expected for RAW12 data"

            expected_bytes = bytes_per_pixel * height * width

            tensor = np.zeros((height, width, elements_per_pixel), dtype=np.uint16)
            data_bytes = f.read(expected_bytes)
            if len(data_bytes) != expected_bytes:
                raise ValueError(f"Expected {expected_bytes} bytes, got {len(data_bytes)}")

            for y in range(height):
                for x in range(width):
                    pixel_start = (y * width + x) * bytes_per_pixel
                    pixel_end = pixel_start + bytes_per_pixel
                    packed_pixel = data_bytes[pixel_start:pixel_end]
                    unpacked_data = unpack_pixel_data_raw12(packed_pixel, elements_per_pixel)
                    tensor[y, x, :] = unpacked_data

        else:  # PVA_LIDAR_DATA_PACKING_TYPE_NONE
            total_elements = height * width * channels

            if data_type_size == 2:
                # uint16
                dtype = np.uint16
                expected_bytes = total_elements * 2
            elif data_type_size == 1:
                # uint8
                dtype = np.uint8
                expected_bytes = total_elements
            else:
                raise ValueError(f"Unsupported data type size: {data_type_size}")

            data_bytes = f.read(expected_bytes)
            if len(data_bytes) != expected_bytes:
                raise ValueError(f"Expected {expected_bytes} bytes, got {len(data_bytes)}")

            # Convert to numpy array and reshape
            data = np.frombuffer(data_bytes, dtype=dtype)
            tensor = data.reshape((height, width, channels))

    return tensor, width, height, channels, data_type_size, hist_spec, hist_params


def read_range_tensor(filename):
    """
    Read range tensor from binary file.

    File format:
        - 3 int32_t values: width, height, channels
        - Followed by raw float32 tensor data (row-major, shape: [height, width, channels])

    Returns:
        tuple: (tensor_data, width, height, channels)
    """
    with open(filename, 'rb') as f:
        # Read tensor header
        header_bytes = f.read(3 * 4)  # 3 int32_t values
        if len(header_bytes) != 12:
            raise ValueError("Failed to read header from range tensor file")

        width, height, channels = struct.unpack('iii', header_bytes)

        print(f"\nRange Tensor shape: [{height}, {width}, {channels}]")

        # Read tensor data (always float32)
        total_elements = height * width * channels
        expected_bytes = total_elements * 4  # 4 bytes per float32

        data_bytes = f.read(expected_bytes)
        if len(data_bytes) != expected_bytes:
            raise ValueError(f"Expected {expected_bytes} bytes, got {len(data_bytes)}")

        # Convert to numpy array and reshape
        data = np.frombuffer(data_bytes, dtype=np.float32)
        tensor = data.reshape((height, width, channels))

    return tensor, width, height, channels


def invert_range_to_bin_index(range_value, range_bias, params):
    """
    Invert the range calculation to recover the approximate bin index.

    This is the inverse of the ComputeRange function from the C++ reference code.

    Forward formula:
        range = rangeBias + rangeScale * (tof * SPEED_OF_LIGHT_M_PER_NS)
        where tof = offsetNs + (subBinIndex * binSizeNs)

    Inverse formula:
        subBinIndex = ((range - rangeBias) / (rangeScale * SPEED_OF_LIGHT_M_PER_NS) - offsetNs) / binSizeNs

    Args:
        range_value: Range value in output units (e.g., meters or centimeters)
        range_bias: Range bias for the pixel
        params: HistogramParams object containing offsetNs, binSizeNs, rangeScale

    Returns:
        float: Approximate sub-bin index (with fractional part from parabolic interpolation)
                Returns -1.0 if range_value is 0 (indicating no peak detected)
    """
    # Speed of light in meters per nanosecond
    SPEED_OF_LIGHT_M_PER_NS = 0.299792458

    # Handle zero range (no peak detected)
    if range_value == 0.0:
        return -1.0

    # Invert the range formula
    # range = rangeBias + rangeScale * (tof * SPEED_OF_LIGHT_M_PER_NS)
    # tof = (range - rangeBias) / (rangeScale * SPEED_OF_LIGHT_M_PER_NS)
    tof = (range_value - range_bias) / (params.range_scale * SPEED_OF_LIGHT_M_PER_NS)

    # tof = offsetNs + (subBinIndex * binSizeNs)
    # subBinIndex = (tof - offsetNs) / binSizeNs
    sub_bin_index = (tof - params.offset_ns) / params.bin_size_ns

    return sub_bin_index


def parse_pixel_histograms(pixel_data, spec):
    """
    Parse histogram data for a single pixel.

    Args:
        pixel_data: 1D array of histogram data for one pixel
        spec: HistogramSpec object

    Returns:
        tuple: (pixel_header, histograms)
            pixel_header: array of pixel header values
            histograms: list of dicts, each containing:
                - 'header': histogram header values
                - 'bins': histogram bin values
    """
    idx = 0

    # Extract pixel header
    pixel_header = pixel_data[idx:idx + spec.pixel_header_size]
    idx += spec.pixel_header_size

    # Extract each histogram
    histograms = []
    for hist_idx in range(spec.num_hist_per_pixel):
        # Extract histogram header
        hist_header = pixel_data[idx:idx + spec.hist_header_size]
        idx += spec.hist_header_size

        # Extract histogram bins
        bins = pixel_data[idx:idx + spec.num_bins_per_hist]
        idx += spec.num_bins_per_hist

        histograms.append({
            'header': hist_header,
            'bins': bins
        })

    return pixel_header, histograms


def find_peaks(bins, num_bins):
    """
    Find local maxima (peaks) in histogram bins.

    Args:
        bins: Array of histogram bin values
        num_bins: Number of bins

    Returns:
        list: Indices of detected peaks
    """
    peaks = []
    if num_bins > 3:
        # Find local maxima where bin value is greater than both neighbors
        for i in range(1, num_bins - 1):
            if bins[i] > bins[i-1] and bins[i] > bins[i+1] and bins[i] > 0:
                peaks.append(i)
    elif num_bins > 0:
        # For very small histograms, consider all non-zero bins as peaks
        peaks = [i for i in range(num_bins) if bins[i] > 0]

    return peaks


def get_top_n_peaks(bins, peaks, n=5):
    """
    Select the top N peaks by value.

    Args:
        bins: Array of histogram bin values
        peaks: List of peak indices
        n: Number of top peaks to return (default: 5)

    Returns:
        list: List of tuples (peak_index, peak_value) for top N peaks, sorted by value descending
    """
    if not peaks:
        return []

    peak_values = [(i, bins[i]) for i in peaks]
    peak_values.sort(key=lambda x: x[1], reverse=True)
    return peak_values[:min(n, len(peak_values))]


def annotate_peaks_on_line_plot(ax, top_peaks, color='red', label_prefix='', range_annotations=None):
    """
    Add markers and annotations for peaks on a line plot.

    Args:
        ax: Matplotlib axis object
        top_peaks: List of tuples (peak_index, peak_value)
        color: Color for the peak markers (default: 'red')
        label_prefix: Prefix for the annotation label (default: '')
        range_annotations: Optional list of range values to display instead of peak values
    """
    for i, (peak_idx, peak_val) in enumerate(top_peaks):
        ax.plot(peak_idx, peak_val, 'o', color=color, markersize=8)

        # Use range value for annotation if available, otherwise use peak value
        if range_annotations and i < len(range_annotations) and range_annotations[i] is not None:
            label_text = f'{label_prefix}{range_annotations[i]:.1f}'
        else:
            label_text = f'{label_prefix}{int(peak_val)}'

        ax.annotate(label_text, xy=(peak_idx, peak_val),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))


def annotate_peaks_on_bar_chart(ax, bars, top_peaks, color='coral', edge_color='darkred', range_annotations=None):
    """
    Add value labels and highlighting for peaks on a bar chart.

    Args:
        ax: Matplotlib axis object
        bars: Bar chart container from ax.bar()
        top_peaks: List of tuples (peak_index, peak_value)
        color: Fill color for peak bars (default: 'coral')
        edge_color: Edge color for peak bars (default: 'darkred')
        range_annotations: Optional list of range values to display instead of peak values
    """
    for i, (peak_idx, peak_val) in enumerate(top_peaks):
        bar = bars[peak_idx]
        height = bar.get_height()

        # Use range value for annotation if available, otherwise use peak value
        if range_annotations and i < len(range_annotations) and range_annotations[i] is not None:
            label_text = f'{range_annotations[i]:.1f}'
        else:
            label_text = f'{int(peak_val)}'

        ax.text(bar.get_x() + bar.get_width()/2., height,
               label_text,
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Highlight the top peak bars
        bar.set_color(color)
        bar.set_edgecolor(edge_color)
        bar.set_linewidth(1.5)


def plot_histograms(pixel_header, histograms, spec, x, y, output_file, detected_peaks=None, range_values=None):
    """
    Generate and save bar charts of the histograms.

    Args:
        pixel_header: Pixel header data
        histograms: List of histogram dicts
        spec: HistogramSpec object
        x: X coordinate of the pixel
        y: Y coordinate of the pixel
        output_file: Output PNG filename
        detected_peaks: Optional list of detected peak indices per histogram (from range tensor)
        range_values: Optional list of range values per histogram for annotation
    """
    num_histograms = len(histograms)

    # Create figure with subplots
    fig, axes = plt.subplots(num_histograms, 1, figsize=(12, 5 * num_histograms))
    if num_histograms == 1:
        axes = [axes]

    title = f'LiDAR Histograms for Pixel ({x}, {y})'
    if detected_peaks is not None:
        if range_values is not None:
            title += ' [Red = Detected Peaks, Labels = Range Values]'
        else:
            title += ' [Red = Detected Peaks from Range Tensor]'
    else:
        title += ' [Orange/Coral = Auto-detected Peaks]'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot each histogram
    for hist_idx, (hist, ax) in enumerate(zip(histograms, axes)):
        bins = hist['bins']
        bin_indices = np.arange(len(bins))
        num_bins = len(bins)

        # Get detected peaks from range tensor if available
        range_peaks = None
        range_peak_annotations = None
        if detected_peaks is not None and hist_idx < len(detected_peaks):
            # Filter out invalid peak indices (-1 or out of bounds)
            valid_peak_indices = []
            valid_peak_values = []

            for peak_idx, p in enumerate(detected_peaks[hist_idx]):
                if p >= 0 and int(p) < num_bins:
                    valid_peak_indices.append(round(p))
                    # Store corresponding range value if available
                    if range_values is not None and hist_idx < len(range_values):
                        valid_peak_values.append(range_values[hist_idx][peak_idx])
                    else:
                        valid_peak_values.append(None)

            if valid_peak_indices:
                range_peaks = [(idx, bins[idx]) for idx in valid_peak_indices]
                range_peak_annotations = valid_peak_values if any(v is not None for v in valid_peak_values) else None

        # Find peaks automatically and select top 5 (for comparison/fallback)
        auto_peaks = find_peaks(bins, num_bins)
        auto_top_peaks = get_top_n_peaks(bins, auto_peaks, spec.num_peaks_per_hist)

        # For large histograms, use line plot instead of bar chart
        if num_bins > 100:
            ax.plot(bin_indices, bins, color='steelblue', linewidth=1.5)
            ax.fill_between(bin_indices, bins, alpha=0.3, color='steelblue')

            # Annotate detected peaks from range tensor (red)
            if range_peaks:
                annotate_peaks_on_line_plot(ax, range_peaks, color='red', label_prefix='',
                                           range_annotations=range_peak_annotations)
            # Annotate auto-detected peaks (orange) if no range peaks provided
            elif auto_top_peaks:
                annotate_peaks_on_line_plot(ax, auto_top_peaks, color='orange', label_prefix='')
        else:
            # For small histograms, use bar chart
            bars = ax.bar(bin_indices, bins, width=0.8,
                         color='steelblue', edgecolor='black', linewidth=1.2)

            # Annotate detected peaks from range tensor (red)
            if range_peaks:
                annotate_peaks_on_bar_chart(ax, bars, range_peaks, color='red', edge_color='darkred',
                                           range_annotations=range_peak_annotations)
            # Annotate auto-detected peaks (coral) if no range peaks provided
            elif auto_top_peaks:
                annotate_peaks_on_bar_chart(ax, bars, auto_top_peaks, color='coral', edge_color='darkorange')

        # Formatting
        ax.set_xlabel('Bin Index', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title(f'Histogram {hist_idx} ({num_bins} bins)', fontsize=14)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set x-axis ticks appropriately
        if num_bins <= 20:
            ax.set_xticks(bin_indices)
        else:
            # Show fewer ticks for large histograms
            tick_step = max(1, num_bins // 10)
            ax.set_xticks(bin_indices[::tick_step])

        # Add histogram header info if present
        if len(hist['header']) > 0:
            header_str = ', '.join([f'{int(v)}' for v in hist['header']])
            ax.text(0.02, 0.98, f'Header: [{header_str}]',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add pixel header info as text at the bottom
    if len(pixel_header) > 0:
        header_str = ', '.join([f'{int(v)}' for v in pixel_header])
        fig.text(0.5, 0.02, f'Pixel Header: [{header_str}]',
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nHistogram visualization saved to: {output_file}")

    # Also display some statistics
    print("\n=== Histogram Statistics ===")
    print(f"Pixel Header: {pixel_header}")
    for hist_idx, hist in enumerate(histograms):
        print(f"\nHistogram {hist_idx}:")
        print(f"  Header: {hist['header']}")
        print(f"  Bins: {hist['bins']}")
        print(f"  Total counts: {np.sum(hist['bins'])}")
        print(f"  Max bin value: {np.max(hist['bins'])} (bin {np.argmax(hist['bins'])})")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    # Parse positional arguments
    input_file = sys.argv[1]
    try:
        x = int(sys.argv[2])
        y = int(sys.argv[3])
    except ValueError:
        print("Error: x and y coordinates must be integers")
        sys.exit(1)

    # Parse optional arguments
    output_file = None
    range_file = None
    range_is_peaks = False

    i = 4
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--range-file':
            if i + 1 < len(sys.argv):
                range_file = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --range-file requires a filename argument")
                sys.exit(1)
        elif arg == '--range-is-peaks':
            range_is_peaks = True
            i += 1
        elif not output_file:
            output_file = arg
            i += 1
        else:
            print(f"Warning: Unknown argument '{arg}', ignoring")
            i += 1

    input_file_name = Path(input_file).stem
    if not output_file:
        output_file = f"{input_file_name}_{x}_{y}.png"

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    print(f"Reading histogram data from: {input_file}")
    print(f"Pixel coordinates: ({x}, {y})")

    # Read tensor data
    try:
        tensor, width, height, channels, data_type_size, spec, params = read_histogram_tensor(input_file)
    except Exception as e:
        print(f"Error reading histogram file: {e}")
        sys.exit(1)

    # Validate coordinates
    if x < 0 or x >= width or y < 0 or y >= height:
        print(f"Error: Pixel coordinates ({x}, {y}) out of bounds")
        print(f"Valid range: x=[0, {width-1}], y=[0, {height-1}]")
        sys.exit(1)

    # Verify expected number of channels based on data packing type
    bytes_per_element = data_type_size if spec.packing_type == PVA_LIDAR_DATA_PACKING_TYPE_NONE else 12/8
    elements_per_pixel = (spec.pixel_header_size +
                        spec.num_hist_per_pixel * (spec.hist_header_size + spec.num_bins_per_hist))
    min_bytes_per_pixel = elements_per_pixel * bytes_per_element

    if channels * data_type_size < min_bytes_per_pixel:
        print(f"\nWarning: Expected at least {min_bytes_per_pixel} bytes per pixel based on histogram spec")
        print(f"The tensor has only {channels} channels and {data_type_size} bytes per element, therefore")
        print(f"{channels *data_type_size} bytes per pixel is less than the minimum")
        sys.exit(1)

    # Extract pixel data
    pixel_data = tensor[y, x, :]

    # Parse histograms
    try:
        pixel_header, histograms = parse_pixel_histograms(pixel_data, spec)
    except Exception as e:
        print(f"Error parsing histogram data: {e}")
        sys.exit(1)

    # Read range tensor if provided
    detected_peaks = None
    range_values_for_plot = None
    if range_file:
        if not Path(range_file).exists():
            print(f"Warning: Range file '{range_file}' not found, skipping")
        else:
            print(f"\nReading range data from: {range_file}")

            try:
                range_tensor, range_width, range_height, range_channels = read_range_tensor(range_file)

                # Validate range tensor dimensions
                if range_width != width or range_height != height:
                    print(f"Warning: Range tensor dimensions ({range_height}x{range_width}) don't match histogram ({height}x{width})")
                elif x >= range_width or y >= range_height:
                    print(f"Warning: Pixel coordinates out of range tensor bounds")
                else:
                    # Validate range tensor channels
                    expected_range_channels = spec.num_peaks_per_hist * spec.num_hist_per_pixel
                    if range_channels != expected_range_channels:
                        print(f"Warning: Range tensor channels ({range_channels}) don't match expected ({expected_range_channels} = {spec.num_peaks_per_hist} peaks/hist * {spec.num_hist_per_pixel} hists)")

                    # Extract range values for this pixel
                    range_values = range_tensor[y, x, :]

                    if range_is_peaks:
                        print("Interpreting range values as peak indices (debug mode)")
                        # Group by histogram (numPeaksPerHist values per histogram)
                        peaks_per_hist = spec.num_peaks_per_hist
                        num_hists = spec.num_hist_per_pixel

                        detected_peaks = []
                        for hist_idx in range(num_hists):
                            start_idx = hist_idx * peaks_per_hist
                            end_idx = start_idx + peaks_per_hist
                            if end_idx <= len(range_values):
                                hist_peaks = range_values[start_idx:end_idx]
                                detected_peaks.append(hist_peaks)
                                print(f"Histogram {hist_idx} peaks read from range tensor: {[int(p) if p >= 0 else -1 for p in hist_peaks]}")
                    else:
                        # Regular range values - invert to recover bin indices
                        print(f"Range values: {range_values}")

                        # Get range bias for this pixel
                        range_bias = 1.0  # NOTE: range bias is hardcoded to the same number as sample application

                        # Invert range values to bin indices
                        peaks_per_hist = spec.num_peaks_per_hist
                        num_hists = spec.num_hist_per_pixel

                        detected_peaks = []
                        range_values_for_plot = []
                        for hist_idx in range(num_hists):
                            start_idx = hist_idx * peaks_per_hist
                            end_idx = start_idx + peaks_per_hist
                            if end_idx <= len(range_values):
                                hist_range_values = range_values[start_idx:end_idx]
                                # Convert each range value to bin index
                                hist_peaks = []
                                for range_val in hist_range_values:
                                    bin_idx = invert_range_to_bin_index(range_val, range_bias, params)
                                    hist_peaks.append(bin_idx)
                                detected_peaks.append(hist_peaks)
                                # Store the original range values for annotations
                                range_values_for_plot.append(hist_range_values)
                                print(f"Histogram {hist_idx} peaks from inverted range values: {[f'{p:.2f}' if p >= 0 else '-1' for p in hist_peaks]}")

            except Exception as e:
                print(f"Warning: Error reading range file: {e}")

    # Plot and save
    try:
        plot_histograms(pixel_header, histograms, spec, x, y, output_file, detected_peaks, range_values_for_plot)
    except Exception as e:
        print(f"Error generating plot: {e}")
        sys.exit(1)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

