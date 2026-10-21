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

#include "pipeline_args.h"

#include "image_io.h"
#include "yolov5_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace yolov5_common {

namespace {

constexpr int64_t MAX_BIN_PIXELS = 4 * 4096 * 4096;

bool endsWithBin(const std::string &path)
{
    if (path.size() < 4u)
        return false;
    std::string suf = path.substr(path.size() - 4);
    return (suf[0] == '.' && (suf[1] == 'b' || suf[1] == 'B') && (suf[2] == 'i' || suf[2] == 'I') &&
            (suf[3] == 'n' || suf[3] == 'N'));
}

} // namespace

// ============================================================================
// parseCommonArgs
// ============================================================================
int parseCommonArgs(int argc, char *argv[], PipelineConfig &cfg, const char *defaultOnnx, const char *defaultEngine,
                    std::vector<std::pair<std::string, std::string>> &unhandledArgs, bool runtimeThresholds,
                    bool dlaSupported)
{
    cfg.onnxPath   = defaultOnnx;
    cfg.enginePath = defaultEngine;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (!runtimeThresholds &&
            (arg == "--conf" || arg == "--iou" || arg.rfind("--conf=", 0) == 0 || arg.rfind("--iou=", 0) == 0))
        {
            fprintf(stderr,
                    "Error: --conf and --iou are not supported by the GPU-only pipeline. "
                    "Thresholds are defined by the ONNX model; regenerate the model and rebuild the engine "
                    "to change them.\n");
            return -1;
        }

        if (!dlaSupported && (arg == "--dla" || arg == "-d" || arg == "--dla-core" || arg.rfind("--dla-core=", 0) == 0))
        {
            fprintf(stderr, "Error: --dla is not supported by the GPU-only pipeline; use yolov5_pva_gpu.\n");
            return -1;
        }

        if ((arg == "--image" || arg == "-i") && i + 1 < argc)
            cfg.imagePath = argv[++i];
        else if (arg == "--fp32")
        {
            cfg.useFP16 = false;
            cfg.useINT8 = false;
        }
        else if (arg == "--int8")
            cfg.useINT8 = true;
        else if ((arg == "--calib-images" || arg == "--calib") && i + 1 < argc)
            cfg.calibImagesDir = argv[++i];
        else if (arg == "--summary-json" && i + 1 < argc)
            cfg.summaryJsonPath = argv[++i];
        else if (arg == "--dla" || arg == "-d")
            cfg.useDLA = true;
        else if (arg == "--dla-core" && i + 1 < argc)
        {
            cfg.dlaCore = std::atoi(argv[++i]);
            cfg.useDLA  = true;
        }
        else if (arg == "--conf" && i + 1 < argc)
        {
            cfg.confidenceThreshold = static_cast<float>(std::atof(argv[++i]));
            cfg.confOverridden      = true;
        }
        else if (arg == "--iou" && i + 1 < argc)
        {
            cfg.nmsThreshold  = static_cast<float>(std::atof(argv[++i]));
            cfg.iouOverridden = true;
        }
        else if (arg == "--eval")
            cfg.evalMode = true;
        else if (arg == "--images-dir" && i + 1 < argc)
        {
            cfg.imagesDir = argv[++i];
            cfg.evalMode  = true;
        }
        else if (arg == "--coco-output" && i + 1 < argc)
            cfg.cocoOutput = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc)
            cfg.warmupRuns = std::atoi(argv[++i]);
        else if (arg == "--benchmark")
            cfg.benchmarkMode = true;
        else if (arg == "--iterations" && i + 1 < argc)
        {
            cfg.benchmarkIter = std::atoi(argv[++i]);
            cfg.benchmarkMode = true;
        }
        else if (arg == "--expected-detections" && i + 1 < argc)
            cfg.expectedDetections = std::atoi(argv[++i]);
        else if (arg == "--check-topk" && i + 1 < argc)
            cfg.checkTopK = std::atoi(argv[++i]);
        else if (arg == "--check-topk-iou" && i + 1 < argc)
            cfg.checkTopKIoU = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--preload")
            cfg.preloadBins = true;
        else if (arg == "--help" || arg == "-h")
            return 1; // caller should print help and exit
        else if (arg[0] != '-')
        {
            if (cfg.onnxPath == defaultOnnx)
            {
                cfg.onnxPath   = arg;
                cfg.enginePath = cfg.onnxPath + (cfg.useDLA ? ".dla" + std::to_string(cfg.dlaCore) : "") + ".engine";
            }
            else
            {
                cfg.enginePath = arg;
            }
        }
        else
        {
            // Unrecognized flag - collect for pipeline-specific handling. It
            // takes no value, so the next argument is left for the loop.
            unhandledArgs.emplace_back(arg, std::string());
        }
    }

    return 0;
}

// ============================================================================
// finalizeConfig
// ============================================================================
int finalizeConfig(PipelineConfig &cfg)
{
    if (cfg.evalMode && cfg.benchmarkMode)
    {
        fprintf(stderr, "Error: evaluation and benchmark modes are mutually exclusive\n");
        return -1;
    }
    if (cfg.warmupRuns < 0)
    {
        fprintf(stderr, "Error: --warmup must be non-negative\n");
        return -1;
    }
    if (cfg.benchmarkMode && cfg.benchmarkIter <= 0)
    {
        fprintf(stderr, "Error: --iterations must be positive\n");
        return -1;
    }
    if (cfg.expectedDetections < -1)
    {
        fprintf(stderr, "Error: --expected-detections must be non-negative\n");
        return -1;
    }
    if (cfg.checkTopK < 0)
    {
        fprintf(stderr, "Error: --check-topk must be non-negative\n");
        return -1;
    }
    if (cfg.checkTopKIoU <= 0.0F || cfg.checkTopKIoU > 1.0F)
    {
        fprintf(stderr, "Error: --check-topk-iou must be in (0, 1]\n");
        return -1;
    }

    // Validate INT8
    if (cfg.useINT8 && cfg.calibImagesDir.empty())
    {
        fprintf(stderr, "Error: --int8 requires --calib-images <dir> for calibration data\n");
        return -1;
    }

    // DLA engine path fixup
    if (cfg.useDLA && cfg.enginePath.find(".dla") == std::string::npos)
    {
        size_t pos = cfg.enginePath.rfind(".engine");
        if (pos != std::string::npos)
            cfg.enginePath = cfg.enginePath.substr(0, pos) + ".dla" + std::to_string(cfg.dlaCore) + ".engine";
    }

    // Precision suffix
    if (!cfg.useFP16 && !cfg.useINT8 && cfg.enginePath.find(".fp32") == std::string::npos)
    {
        size_t pos = cfg.enginePath.rfind(".engine");
        if (pos != std::string::npos)
            cfg.enginePath = cfg.enginePath.substr(0, pos) + ".fp32.engine";
    }
    if (cfg.useINT8 && cfg.enginePath.find(".int8") == std::string::npos)
    {
        size_t pos = cfg.enginePath.rfind(".engine");
        if (pos != std::string::npos)
            cfg.enginePath = cfg.enginePath.substr(0, pos) + ".int8.engine";
    }

    // Eval-mode threshold overrides
    if (cfg.evalMode)
    {
        if (!cfg.confOverridden)
            cfg.confidenceThreshold = 0.001f;
        if (!cfg.iouOverridden)
            cfg.nmsThreshold = 0.65f;
    }

    cfg.verbose = !cfg.evalMode && !cfg.benchmarkMode;

    // Build image list
    if (cfg.evalMode && !cfg.imagesDir.empty())
    {
        cfg.imageList = getImageFiles(cfg.imagesDir);
        if (cfg.imageList.empty())
        {
            fprintf(stderr, "Error: No images found in %s\n", cfg.imagesDir.c_str());
            return -1;
        }
        printf("Evaluation mode: Found %zu images in %s\n\n", cfg.imageList.size(), cfg.imagesDir.c_str());
    }
    else if (!cfg.imagePath.empty())
    {
        cfg.imageList.push_back(cfg.imagePath);
    }

    // Total iterations
    cfg.totalIterations = 1;
    if (cfg.benchmarkMode && !cfg.imageList.empty())
    {
        cfg.totalIterations = cfg.benchmarkIter;
        printf("Benchmark mode: %d iterations with warmup=%d\n\n", cfg.benchmarkIter, cfg.warmupRuns);
    }
    else if (cfg.evalMode)
    {
        cfg.totalIterations = static_cast<int>(cfg.imageList.size());
    }

    return 0;
}

// ============================================================================
// printCommonHelp
// ============================================================================
void printCommonHelp(const char *programName, bool runtimeThresholds, bool dlaSupported)
{
    printf("Usage: %s [options] [onnx_model] [engine_file]\n\n", programName);
    printf("Common options:\n");
    printf("  --image, -i <path>      Load input image from file (PNG or BIN)\n");
    printf("  --fp32                  Use FP32 precision (default: FP16)\n");
    printf("  --int8                  Use INT8 precision (requires --calib-images)\n");
    printf("  --calib-images <dir>    Directory of images for INT8 calibration\n");
    if (dlaSupported)
    {
        printf("  --dla, -d               Enable DLA execution with GPU fallback\n");
        printf("  --dla-core <N>          Use DLA core N (0 or 1 on Orin, default: 0)\n");
    }
    if (runtimeThresholds)
    {
        printf("  --conf <val>            Confidence threshold (default: 0.25, eval: 0.001)\n");
        printf("  --iou <val>             NMS IoU threshold (default: 0.45, eval: 0.65)\n");
    }
    printf("  --expected-detections N Fail unless the final frame has exactly N detections\n");
    printf("  --check-topk N          Fail unless the N highest-scoring detections each match a\n");
    printf("                          distinct reference detection (traffic image only)\n");
    printf("  --check-topk-iou <val>  IoU threshold for --check-topk (default: 0.50)\n");
    printf("\nEvaluation mode:\n");
    if (runtimeThresholds)
        printf("  --eval                  Enable evaluation mode (conf=0.001, iou=0.65)\n");
    else
    {
        printf("  --eval                  Enable evaluation mode (thresholds remain defined by the ONNX model)\n");
        printf("                          To change thresholds, regenerate the model and rebuild the engine.\n");
    }
    printf("  --images-dir <path>     Directory containing images for evaluation\n");
    printf("  --coco-output <path>    Output COCO JSON file for predictions\n");
    printf("  --warmup <N>            Number of warmup runs (default: 3)\n");
    printf("\nBenchmark mode:\n");
    printf("  --benchmark             Enable benchmark mode (single image, N iterations; the image is\n");
    printf("                          decoded and preloaded once, outside the measured iterations)\n");
    printf("  --iterations <N>        Number of iterations for benchmark (default: 100)\n");
    printf("  --preload               Preload .bin evaluation images into RAM before processing\n");
    printf("  --summary-json <path>   Write timing summary JSON to this path\n");
    printf("\n  --help, -h              Show this help message\n");
}

// ============================================================================
// Preloading
// ============================================================================
bool preloadBinImages(const PipelineConfig &cfg, std::vector<PreloadedBin> &bins, int padWidth, int padHeight)
{
    bins.clear();

    if (!cfg.preloadBins)
        return false;
    if (cfg.imageList.empty())
        return false;

    for (const auto &p : cfg.imageList)
    {
        if (!endsWithBin(p))
        {
            printf("--preload: not all images are .bin, skipping preload.\n");
            return false;
        }
    }

    const bool usePadding     = (padWidth > 0 && padHeight > 0);
    const size_t padRowStride = usePadding ? static_cast<size_t>(padWidth) * 3u : 0;
    const size_t padBufSize   = usePadding ? padRowStride * padHeight : 0;

    printf("Preloading %zu .bin images into pinned (cudaMallocHost) RAM%s...\n", cfg.imageList.size(),
           usePadding ? " (with stride padding)" : "");

    bins.resize(cfg.imageList.size());
    size_t idx        = 0;
    size_t totalBytes = 0;

    for (const auto &path : cfg.imageList)
    {
        if ((idx + 1) % 500 == 0 || idx == 0 || idx + 1 == cfg.imageList.size())
            printf("  [%zu/%zu] %s\n", idx + 1, cfg.imageList.size(), path.c_str());

        FILE *f = fopen(path.c_str(), "rb");
        if (!f)
        {
            fprintf(stderr, "Preload: failed to open %s\n", path.c_str());
            freePreloadedBins(bins);
            return false;
        }

        int32_t w = 0, h = 0;
        if (fread(&w, sizeof(int32_t), 1, f) != 1 || fread(&h, sizeof(int32_t), 1, f) != 1 || w <= 0 || h <= 0 ||
            static_cast<int64_t>(w) * h > MAX_BIN_PIXELS)
        {
            fprintf(stderr, "Preload: invalid .bin header %s\n", path.c_str());
            fclose(f);
            freePreloadedBins(bins);
            return false;
        }

        if (usePadding && (w > padWidth || h > padHeight))
        {
            fprintf(stderr, "Preload: image %s (%dx%d) exceeds pad dimensions (%dx%d)\n", path.c_str(), w, h, padWidth,
                    padHeight);
            fclose(f);
            freePreloadedBins(bins);
            return false;
        }

        const size_t allocSize = usePadding ? padBufSize : static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
        void *buf              = nullptr;
        cudaError_t cerr       = cudaMallocHost(&buf, allocSize);
        if (cerr != cudaSuccess || !buf)
        {
            fprintf(stderr, "Preload: cudaMallocHost failed for %s (%zu bytes): %s\n", path.c_str(), allocSize,
                    cudaGetErrorString(cerr));
            fclose(f);
            freePreloadedBins(bins);
            return false;
        }

        if (usePadding)
        {
            memset(buf, 0, allocSize);
            const size_t srcRowBytes = static_cast<size_t>(w) * 3u;
            uint8_t *dst             = static_cast<uint8_t *>(buf);
            for (int y = 0; y < h; y++)
            {
                if (fread(dst + y * padRowStride, 1, srcRowBytes, f) != srcRowBytes)
                {
                    fprintf(stderr, "Preload: failed to read row %d of %s\n", y, path.c_str());
                    cudaFreeHost(buf);
                    fclose(f);
                    freePreloadedBins(bins);
                    return false;
                }
            }
            bins[idx].w = padWidth;
            bins[idx].h = padHeight;
        }
        else
        {
            const size_t payloadSize = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
            if (fread(buf, 1, payloadSize, f) != payloadSize)
            {
                fprintf(stderr, "Preload: failed to read payload %s\n", path.c_str());
                cudaFreeHost(buf);
                fclose(f);
                freePreloadedBins(bins);
                return false;
            }
            bins[idx].w = w;
            bins[idx].h = h;
        }
        fclose(f);

        bins[idx].ptr   = buf;
        bins[idx].origW = w;
        bins[idx].origH = h;
        totalBytes += allocSize;
        idx++;
    }

    printf("Preloaded %zu .bin images (%.1f MB) into RAM.\n", bins.size(),
           static_cast<double>(totalBytes) / (1024.0 * 1024.0));
    return true;
}

bool preloadInputImages(const PipelineConfig &cfg, std::vector<PreloadedBin> &bins)
{
    bins.clear();
    if (!cfg.benchmarkMode)
        return preloadBinImages(cfg, bins);
    if (cfg.imageList.empty())
        return false;

    const std::string &path = cfg.imageList[0];
    if (endsWithBin(path))
    {
        PipelineConfig benchCfg = cfg;
        benchCfg.preloadBins    = true;
        benchCfg.imageList      = {path};
        return preloadBinImages(benchCfg, bins);
    }

    int w = 0, h = 0;
    uint8_t *decoded = loadPngRgb(path.c_str(), &w, &h);
    if (!decoded || w <= 0 || h <= 0)
    {
        fprintf(stderr, "Preload: failed to decode %s\n", path.c_str());
        freeImage(decoded);
        return false;
    }

    const size_t bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
    void *pinned       = nullptr;
    cudaError_t cerr   = cudaMallocHost(&pinned, bytes);
    if (cerr != cudaSuccess || !pinned)
    {
        fprintf(stderr, "Preload: cudaMallocHost failed for %s (%zu bytes): %s\n", path.c_str(), bytes,
                cudaGetErrorString(cerr));
        freeImage(decoded);
        return false;
    }
    memcpy(pinned, decoded, bytes);
    freeImage(decoded);

    bins.push_back({pinned, w, h, w, h});
    printf("Benchmark input decoded and preloaded into pinned RAM: %dx%d\n", w, h);
    return true;
}

void freePreloadedBins(std::vector<PreloadedBin> &bins)
{
    for (auto &b : bins)
    {
        if (b.ptr)
            cudaFreeHost(b.ptr);
        b.ptr = nullptr;
    }
    bins.clear();
}

} // namespace yolov5_common
