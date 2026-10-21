/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Common arg parsing, config, image list building, and .bin preloading for YOLOv5 pipelines.
 */

#ifndef YOLOV5_COMMON_PIPELINE_ARGS_H
#define YOLOV5_COMMON_PIPELINE_ARGS_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace yolov5_common {

struct PipelineConfig
{
    // Paths
    std::string onnxPath;
    std::string enginePath;
    std::string imagePath;
    std::string imagesDir;
    std::string cocoOutput;
    std::string calibImagesDir;
    std::string summaryJsonPath;
    // Precision
    bool useFP16 = true;
    bool useINT8 = false;

    // DLA
    bool useDLA = false;
    int dlaCore = 0;

    // Thresholds
    float confidenceThreshold = 0.25f;
    float nmsThreshold        = 0.45f;
    bool confOverridden       = false;
    bool iouOverridden        = false;

    // Modes
    bool evalMode      = false;
    bool benchmarkMode = false;
    bool preloadBins   = false;

    // Iteration control
    int warmupRuns         = 3;
    int benchmarkIter      = 100;
    int expectedDetections = -1; // Optional end-to-end correctness gate.
    int checkTopK          = 0;  // Optional top-K reference check; 0 disables.
    float checkTopKIoU     = 0.5F;

    // Derived
    bool verbose = true;
    std::vector<std::string> imageList;
    int totalIterations = 1;
};

/**
 * Parse command-line arguments common to all YOLOv5 pipelines.
 *
 * Handles: --image/-i, --fp32, --int8, --calib-images/--calib, --dla/-d,
 * --dla-core, --conf, --iou, --eval, --images-dir, --coco-output, --warmup,
 * --benchmark, --iterations, --preload, --summary-json, --expected-detections,
 * --check-topk, --check-topk-iou and --help/-h.
 *
 * Positional args (non-flag) are assigned to onnxPath then enginePath.
 * Unrecognized flags are collected into `unhandledArgs` for pipeline-specific
 * parsing. Pipeline-specific flags take no value.
 *
 * @param argc           Argument count from main().
 * @param argv           Argument values from main().
 * @param cfg            Output: filled with parsed values.
 * @param defaultOnnx    Default ONNX model filename (e.g. "yolov5s_trimmed_reshape_tranpose.onnx").
 * @param defaultEngine  Default engine filename (e.g. "yolov5s_trimmed_reshape_tranpose.engine").
 * @param unhandledArgs  Output: pairs of (flag, value) for args not handled here.
 *                       The value is always empty.
 * @param runtimeThresholds Whether confidence and NMS thresholds can be set at runtime.
 * @param dlaSupported   Whether the pipeline can run inference on DLA.
 * @return  0 = success, 1 = --help was printed (caller should return 0), -1 = error.
 */
int parseCommonArgs(int argc, char *argv[], PipelineConfig &cfg, const char *defaultOnnx, const char *defaultEngine,
                    std::vector<std::pair<std::string, std::string>> &unhandledArgs, bool runtimeThresholds = true,
                    bool dlaSupported = true);

/**
 * Post-parse setup: apply eval-mode overrides, build engine path suffixes,
 * populate imageList, compute totalIterations, set verbose.
 * Call after parseCommonArgs() and any pipeline-specific arg handling.
 *
 * @return 0 on success, -1 on error (e.g. no images found).
 */
int finalizeConfig(PipelineConfig &cfg);

/**
 * Print the common subset of the help text. Pipelines can print additional
 * options before/after calling this.
 */
void printCommonHelp(const char *programName, bool runtimeThresholds = true, bool dlaSupported = true);

// .bin preloading
struct PreloadedBin
{
    void *ptr = nullptr;
    int w = 0, h = 0;         // buffer dimensions (may be padded when padWidth/padHeight are used)
    int origW = 0, origH = 0; // actual image dimensions from .bin header
};

/**
 * Preload all .bin images from cfg.imageList into pinned RAM.
 * Only activates if cfg.preloadBins is true AND all images are .bin.
 *
 * When padWidth > 0 && padHeight > 0, every buffer is allocated at padWidth*padHeight*3
 * with row stride = padWidth*3. Image data is placed at the top-left and remaining
 * bytes are zeroed. PreloadedBin::w/h report the padded dimensions while
 * PreloadedBin::origW/origH preserve the actual image size.
 *
 * @param cfg         Pipeline config (reads imageList, preloadBins).
 * @param bins        Output: vector of preloaded buffers (caller must call freePreloadedBins).
 * @param padWidth    If >0, pad every buffer to this width (columns).
 * @param padHeight   If >0, pad every buffer to this height (rows).
 * @return true if preloading was performed, false otherwise.
 */
bool preloadBinImages(const PipelineConfig &cfg, std::vector<PreloadedBin> &bins, int padWidth = 0, int padHeight = 0);

/**
 * Preload the input images of a run into pinned RAM.
 *
 * In benchmark mode the single input image is always preloaded, so that file
 * reading and decoding stay outside the measured iterations; both PNG and .bin
 * inputs are supported. Otherwise this defers to preloadBinImages(), which only
 * preloads when --preload is given and every image is a .bin file.
 *
 * @param cfg   Pipeline config (reads benchmarkMode, imageList, preloadBins).
 * @param bins  Output: vector of preloaded buffers (caller must call freePreloadedBins).
 * @return true if images were preloaded. In benchmark mode, false means the input could not be loaded.
 */
bool preloadInputImages(const PipelineConfig &cfg, std::vector<PreloadedBin> &bins);

/**
 * Free all buffers allocated by preloadBinImages() or preloadInputImages().
 */
void freePreloadedBins(std::vector<PreloadedBin> &bins);

} // namespace yolov5_common

#endif // YOLOV5_COMMON_PIPELINE_ARGS_H
