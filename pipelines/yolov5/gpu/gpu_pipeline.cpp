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
 * GPU-only YOLOv5 pipeline: CUDA preprocessing and TensorRT inference, with the
 * YOLOv5 decode and EfficientNMS inside the engine graph.
 *
 *   1. Load the image and letterbox it to 672x672 RGB uint8 (CUDA kernel)
 *   2. Convert RGB NHWC uint8 -> NCHW float32 [0,1] (CUDA kernel)
 *   3. TensorRT enqueueV3(): inference + decode + NMS
 *
 * Dependencies: TensorRT + CUDA + NVCV tensor types.
 */

#include "yolov5_pipeline_iface.hpp"

#include "detection_check.h"
#include "image_io.h"
#include "int8_calibrator.h"
#include "letterbox_gpu.h"
#include "pipeline_args.h"
#include "pipeline_error.h"
#include "trt_utils.h"
#include "yolov5_common.h"

#include "preprocess_cuda.cuh"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

// ============================================================================
// Macros
// ============================================================================

#define TRT_CHECK(cond, msg)              \
    do                                    \
    {                                     \
        if (!(cond))                      \
        {                                 \
            fprintf(stderr, "%s\n", msg); \
            return -1;                    \
        }                                 \
    } while (0)

// ============================================================================
// RAII helper for GPU resources
// ============================================================================

struct GpuResources
{
    cudaStream_t stream              = nullptr;
    cudaEvent_t evStart              = nullptr;
    cudaEvent_t evStop               = nullptr;
    NVCVTensorHandle resizedU8Tensor = nullptr; // letterbox output
    float *d_inputNchw               = nullptr;
    int32_t *d_numDets               = nullptr;
    float *d_detBoxes                = nullptr;
    float *d_detScores               = nullptr;
    int32_t *d_detClasses            = nullptr;
    void *d_preloadBuf               = nullptr; // preloaded image copied to the device

    nvinfer1::IBuilder *builder          = nullptr;
    nvinfer1::IRuntime *runtime          = nullptr;
    nvinfer1::ICudaEngine *engine        = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaGraphExec_t inferenceGraphExec   = nullptr;

    ~GpuResources()
    {
        if (inferenceGraphExec)
            cudaGraphExecDestroy(inferenceGraphExec);
        delete context;
        delete engine;
        delete runtime;
        delete builder;
        if (resizedU8Tensor)
            nvcvTensorDecRef(resizedU8Tensor, nullptr);
        if (d_inputNchw)
            cudaFree(d_inputNchw);
        if (d_numDets)
            cudaFreeHost(d_numDets);
        if (d_detBoxes)
            cudaFreeHost(d_detBoxes);
        if (d_detScores)
            cudaFreeHost(d_detScores);
        if (d_detClasses)
            cudaFreeHost(d_detClasses);
        if (d_preloadBuf)
            cudaFree(d_preloadBuf);
        if (evStart)
            cudaEventDestroy(evStart);
        if (evStop)
            cudaEventDestroy(evStop);
        if (stream)
            cudaStreamDestroy(stream);
    }
};

// ============================================================================
// TensorRT logger
// ============================================================================

static yolov5_common::TRTLogger gLogger;

// ============================================================================
// TensorRT engine build / load
// ============================================================================

static nvinfer1::ICudaEngine *buildEngine(const std::string &onnxPath, nvinfer1::IBuilder *builder,
                                          nvinfer1::IRuntime *runtime, bool fp16, bool int8 = false,
                                          const std::string &calibDir = "", int modelH = 640, int modelW = 640)
{
    printf("Building engine from ONNX: %s\n", onnxPath.c_str());

    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    if (!network)
        return nullptr;

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
        return nullptr;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        fprintf(stderr, "Failed to parse ONNX\n");
        return nullptr;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    if (fp16 && builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        printf("  FP16 enabled\n");
    }

    std::unique_ptr<yolov5_common::Int8Calibrator> calibrator;
    if (int8 && builder->platformHasFastInt8())
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        if (!fp16)
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            printf("  FP16 enabled (required fallback for INT8)\n");
        }

        std::string cacheFile = "calib_cache_nms_lb.bin";
        calibrator = std::make_unique<yolov5_common::Int8Calibrator>(calibDir, 1, 3, modelH, modelW, cacheFile);
        config->setInt8Calibrator(calibrator.get());
        printf("  INT8 enabled\n");
    }
    else if (int8)
    {
        printf("  INT8 not available on this platform, falling back to FP16/FP32\n");
    }
    else if (!fp16)
    {
        printf("  FP32 mode\n");
    }

    printf("  Building (this may take a few minutes)...\n");
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    return runtime->deserializeCudaEngine(plan->data(), plan->size());
}

// ============================================================================
// Pipeline stages
// ============================================================================

static int submitGpuPreprocess(GpuResources &gpu, int modelH, int modelW, float *normalizeMs)
{
    NVCVTensorData resizedData;
    YOLOV5_NVCV_CHECK(nvcvTensorExportData(gpu.resizedU8Tensor, &resizedData));
    const uint8_t *srcRgbPtr = reinterpret_cast<const uint8_t *>(resizedData.buffer.strided.basePtr);

    YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStart, gpu.stream));
    YOLOV5_CUDA_CHECK(launchRgbNhwcToNchw(srcRgbPtr, gpu.d_inputNchw, modelH, modelW, gpu.stream));
    YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStop, gpu.stream));
    YOLOV5_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    YOLOV5_CUDA_CHECK(cudaEventElapsedTime(normalizeMs, gpu.evStart, gpu.evStop));
    return 0;
}

static int submitGpuInference(GpuResources &gpu, bool useCudaGraph, float *inferMs)
{
    YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStart, gpu.stream));
    if (useCudaGraph)
        YOLOV5_CUDA_CHECK(cudaGraphLaunch(gpu.inferenceGraphExec, gpu.stream));
    else
        TRT_CHECK(gpu.context->enqueueV3(gpu.stream), "TensorRT enqueue failed");
    YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStop, gpu.stream));
    YOLOV5_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    YOLOV5_CUDA_CHECK(cudaEventElapsedTime(inferMs, gpu.evStart, gpu.evStop));
    return 0;
}

// Not a pipeline stage: decode and NMS already ran inside the TensorRT engine
// graph as part of inference, so all that remains is reading the results out of
// the pinned output buffers.
static int readGpuDetections(const GpuResources &gpu, int maxDets, std::vector<float> &boxes,
                             std::vector<float> &scores, std::vector<int32_t> &classes)
{
    int32_t numDets = std::min(*gpu.d_numDets, maxDets);
    if (numDets > 0)
    {
        std::memcpy(boxes.data(), gpu.d_detBoxes, numDets * 4 * sizeof(float));
        std::memcpy(scores.data(), gpu.d_detScores, numDets * sizeof(float));
        std::memcpy(classes.data(), gpu.d_detClasses, numDets * sizeof(int32_t));
    }
    return numDets;
}

// ============================================================================
// Main
// ============================================================================

// Per-frame results of the load step.
struct GpuFrameInput
{
    yolov5_common::LetterboxInfo lb{};
    float loadMs        = 0.0F;
    float imageDecodeMs = 0.0F;
    float h2dCopyMs     = 0.0F;
    float resizePadMs   = 0.0F;
};

// Configuration and long-lived state are members; the work splits into
// initialize / run / report.
class GpuOnlyPipeline final
{
public:
    int initialize(int argc, char *argv[]);
    int run();
    int report();
    bool shouldRun() const
    {
        return runRequested;
    }

    int prepareFrameInput(const std::string &curPath, int imageIndex, GpuFrameInput &out);
    void reportFrameDetections(int iter, int64_t curImageId, const GpuFrameInput &frame, int numDets,
                               float normalizeMs2, float inferMs, float d2hCopyMs, float &buildBoxesMs);

private:
    int resolveEngineIO();

    yolov5_common::PipelineConfig cfg;
    GpuResources gpu;
    bool runRequested = false;

    // Copied out of cfg in initialize().
    bool useFP16       = true;
    bool useINT8       = false;
    bool evalMode      = false;
    bool benchmarkMode = false;
    int warmupRuns     = 0;
    int benchmarkIter  = 0;

    // Model geometry. The engine input must match it.
    static constexpr int modelW = 672;
    static constexpr int modelH = 672;
    // Detection capacity, read from the engine's detection_boxes shape.
    int maxDets       = 0;
    bool useCudaGraph = false;
    std::string inputName;

    std::vector<yolov5_common::PreloadedBin> preloadedBins;
    bool usePreloadedBins = false;

    std::vector<float> loadTimings, preprocTimings, inferTimings, totalTimings;
    std::vector<float> imageDecodeTimings, cpuResizeTimings, h2dCopyTimings, normalizeTimings;
    std::vector<float> d2hCopyTimings, buildBoxesTimings;
    std::vector<yolov5_common::CocoDetection> allDetections;
    int totalDetsCount = 0;
    int lastNumDets    = -1;
    std::vector<yolov5_common::DetBox> lastDetections;

    // Host-side engine outputs, filled once per frame by readGpuDetections().
    std::vector<float> hBoxes, hScores;
    std::vector<int32_t> hClasses;

    std::chrono::high_resolution_clock::time_point appStart;
};

int GpuOnlyPipeline::initialize(int argc, char *argv[])
{
    printf("YOLOv5 + EfficientNMS - GPU-only TensorRT Pipeline\n\n");
    yolov5_pipeline::printStageMap(yolov5_pipeline::GPU_ONLY_STAGE_MAP);

    // Must initialize TRT plugins so EfficientNMS_TRT is available
    initLibNvInferPlugins(&gLogger, "");

    appStart = std::chrono::high_resolution_clock::now();

    // Both pipelines share the same three-head model and 672x672 input
    // geometry. The GPU-only model adds the YOLOv5 decode and EfficientNMS graph.
    useCudaGraph = false;

    std::vector<std::pair<std::string, std::string>> extra;
    int rc = yolov5_common::parseCommonArgs(argc, argv, cfg, "yolov5s_672_nms.onnx", "yolov5s_672_nms.engine", extra,
                                            false, false);
    if (rc == 1)
    {
        yolov5_common::printCommonHelp(argv[0], false, false);
        printf("\nPipeline-specific options:\n");
        printf("  --cuda-graph            Capture TensorRT inference in a CUDA graph (benchmark only)\n");
        printf("\nExample:\n");
        printf(
            "  python3 prepare_yolov5_nms_onnx.py yolov5s_trimmed_reshape_tranpose.onnx yolov5s_672_nms.onnx "
            "--score-threshold 0.25 --iou-threshold 0.45\n");
        printf("  %s -i traffic.png yolov5s_672_nms.onnx\n", argv[0]);
        printf("  %s --int8 --calib-images ./val2017_bin -i traffic.png yolov5s_672_nms.onnx\n", argv[0]);
        printf("  %s --eval --images-dir ./val2017 --coco-output results.json yolov5s_672_nms.onnx\n", argv[0]);
        return 0; // Help only: runRequested stays false.
    }
    if (rc < 0)
        return -1;

    for (const auto &p : extra)
    {
        if (p.first == "--cuda-graph")
            useCudaGraph = true;
        else
        {
            fprintf(stderr, "Unknown option: %s\n", p.first.c_str());
            return -1;
        }
    }

    if (yolov5_common::finalizeConfig(cfg) != 0)
        return -1;
    if (useCudaGraph && !cfg.benchmarkMode)
    {
        fprintf(stderr, "--cuda-graph requires --benchmark\n");
        return -1;
    }
    runRequested = true;

    // Convenience aliases
    const std::string &onnxPath               = cfg.onnxPath;
    const std::string &enginePath             = cfg.enginePath;
    const std::string &imagePath              = cfg.imagePath;
    const std::string &imagesDir              = cfg.imagesDir;
    const std::string &cocoOutput             = cfg.cocoOutput;
    const std::string &calibImagesDir         = cfg.calibImagesDir;
    useFP16                                   = cfg.useFP16;
    useINT8                                   = cfg.useINT8;
    evalMode                                  = cfg.evalMode;
    benchmarkMode                             = cfg.benchmarkMode;
    warmupRuns                                = cfg.warmupRuns;
    benchmarkIter                             = cfg.benchmarkIter;
    const std::vector<std::string> &imageList = cfg.imageList;

    const char *precisionStr = useINT8 ? "INT8+FP16" : (useFP16 ? "FP16" : "FP32");

    // Print config
    printf("Configuration:\n");
    printf("  Model:      %dx%d, %s\n", modelW, modelH, precisionStr);
    printf("  ONNX:       %s\n", onnxPath.c_str());
    printf("  Engine:     %s\n", enginePath.c_str());
    if (useINT8)
        printf("  Calib dir:  %s\n", calibImagesDir.empty() ? "(using cache)" : calibImagesDir.c_str());
    printf("  Pipeline:   letterbox(GPU) → normalize(GPU) → TRT (inference+NMS in graph)\n\n");

    if (evalMode && !imageList.empty())
        printf("Eval mode: %zu images from %s\n\n", imageList.size(), imagesDir.c_str());

    // RAII resource management (auto-cleanup on any return)
    YOLOV5_CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
    YOLOV5_CUDA_CHECK(cudaEventCreate(&gpu.evStart));
    YOLOV5_CUDA_CHECK(cudaEventCreate(&gpu.evStop));

    // Load / build TRT engine
    gpu.runtime = nvinfer1::createInferRuntime(gLogger);
    TRT_CHECK(gpu.runtime, "Failed to create TRT runtime");

    {
        std::ifstream ef(enginePath, std::ios::binary);
        if (ef.good())
        {
            ef.close();
            printf("Loading engine: %s\n", enginePath.c_str());
            gpu.engine = yolov5_common::loadEngine(enginePath, *gpu.runtime);
        }
    }

    if (!gpu.engine)
    {
        std::ifstream of(onnxPath);
        if (!of.good())
        {
            fprintf(stderr, "ONNX not found: %s\n", onnxPath.c_str());
            fprintf(stderr,
                    "Run: python3 prepare_yolov5_nms_onnx.py yolov5s_trimmed_reshape_tranpose.onnx %s "
                    "--score-threshold 0.25 --iou-threshold 0.45\n",
                    onnxPath.c_str());
            return -1;
        }
        of.close();
        gpu.builder = nvinfer1::createInferBuilder(gLogger);
        TRT_CHECK(gpu.builder, "Failed to create builder");
        gpu.engine = buildEngine(onnxPath, gpu.builder, gpu.runtime, useFP16, useINT8, calibImagesDir, modelH, modelW);
        if (gpu.engine)
        {
            if (yolov5_common::saveEngine(*gpu.engine, enginePath))
                printf("Engine saved: %s\n", enginePath.c_str());
            else
                fprintf(stderr, "Warning: failed to save engine to %s\n", enginePath.c_str());
        }
    }
    TRT_CHECK(gpu.engine, "Failed to load/build engine");

    // Validate the engine I/O and read the detection capacity from it.
    if (resolveEngineIO() != 0)
        return -1;

    // Engine outputs, sized from the engine's static output shape.
    YOLOV5_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&gpu.d_numDets), sizeof(int32_t)));
    YOLOV5_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&gpu.d_detBoxes), maxDets * 4 * sizeof(float)));
    YOLOV5_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&gpu.d_detScores), maxDets * sizeof(float)));
    YOLOV5_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&gpu.d_detClasses), maxDets * sizeof(int32_t)));

    // Host output buffers
    hBoxes.assign(static_cast<size_t>(maxDets) * 4, 0.0F);
    hScores.assign(static_cast<size_t>(maxDets), 0.0F);
    hClasses.assign(static_cast<size_t>(maxDets), 0);

    // Letterbox output and engine input
    {
        NVCVTensorRequirements reqs;
        int64_t shape[] = {1, modelH, modelW, 3};
        YOLOV5_NVCV_CHECK(nvcvTensorCalcRequirements(4, shape, NVCV_DATA_TYPE_U8, NVCV_TENSOR_NHWC, 0, 0, &reqs));
        YOLOV5_NVCV_CHECK(nvcvTensorConstruct(&reqs, nullptr, &gpu.resizedU8Tensor));
        size_t inputElems = static_cast<size_t>(modelH) * modelW * 3;
        YOLOV5_CUDA_CHECK(cudaMalloc(&gpu.d_inputNchw, inputElems * sizeof(float)));
    }

    gpu.context = gpu.engine->createExecutionContext();
    TRT_CHECK(gpu.context, "Failed to create execution context");

    // Bind I/O tensors
    TRT_CHECK(gpu.context->setTensorAddress(inputName.c_str(), gpu.d_inputNchw), "Failed to bind engine input");
    TRT_CHECK(gpu.context->setTensorAddress("num_detections", gpu.d_numDets), "Failed to bind num_detections");
    TRT_CHECK(gpu.context->setTensorAddress("detection_boxes", gpu.d_detBoxes), "Failed to bind detection_boxes");
    TRT_CHECK(gpu.context->setTensorAddress("detection_scores", gpu.d_detScores), "Failed to bind detection_scores");
    TRT_CHECK(gpu.context->setTensorAddress("detection_classes", gpu.d_detClasses), "Failed to bind detection_classes");

    // Benchmark repeats one decoded input, so it is always preloaded and file
    // decoding stays outside the measured iteration. Evaluation keeps the
    // optional --preload behavior for .bin datasets.
    usePreloadedBins = yolov5_common::preloadInputImages(cfg, preloadedBins);
    if (benchmarkMode && !cfg.imageList.empty() && !usePreloadedBins)
    {
        fprintf(stderr, "ERROR: benchmark input could not be preloaded.\n");
        return -1;
    }

    // Allocate device buffer for raw image data (H2D target)
    // Size to largest preloaded image or engine input, whichever is bigger
    {
        size_t maxBuf = static_cast<size_t>(modelH) * static_cast<size_t>(modelW) * 3u;
        for (const auto &b : preloadedBins)
        {
            size_t sz = static_cast<size_t>(b.w) * static_cast<size_t>(b.h) * 3u;
            if (sz > maxBuf)
                maxBuf = sz;
        }
        YOLOV5_CUDA_CHECK(cudaMalloc(&gpu.d_preloadBuf, maxBuf));
    }

    if (useCudaGraph)
    {
        // Run once before capture so TensorRT performs deferred setup outside
        // the graph, following the TensorRT trtexec capture protocol.
        TRT_CHECK(gpu.context->enqueueV3(gpu.stream), "TensorRT graph priming enqueue failed");
        YOLOV5_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));

        cudaGraph_t graph = nullptr;
        YOLOV5_CUDA_CHECK(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal));
        TRT_CHECK(gpu.context->enqueueV3(gpu.stream), "TensorRT CUDA graph capture enqueue failed");
        YOLOV5_CUDA_CHECK(cudaStreamEndCapture(gpu.stream, &graph));
        YOLOV5_CUDA_CHECK(cudaGraphInstantiate(&gpu.inferenceGraphExec, graph, nullptr, nullptr, 0));
        YOLOV5_CUDA_CHECK(cudaGraphDestroy(graph));
        printf("  CUDA graph: TensorRT inference captured successfully\n");
    }

    return 0;
}

// Check the engine I/O against what this pipeline feeds and reads: one float32
// NCHW 1x3x672x672 input and the four EfficientNMS outputs. The detection
// capacity comes from the static shape of detection_boxes.
int GpuOnlyPipeline::resolveEngineIO()
{
    const nvinfer1::ICudaEngine &engine = *gpu.engine;
    const int numIO                     = engine.getNbIOTensors();

    printf("\nEngine I/O (%d tensors):\n", numIO);
    int numInputs = 0;
    for (int i = 0; i < numIO; i++)
    {
        const char *name = engine.getIOTensorName(i);
        if (name == nullptr)
        {
            fprintf(stderr, "Engine I/O tensor %d has no name\n", i);
            return -1;
        }
        const nvinfer1::Dims dims = engine.getTensorShape(name);
        const bool isInput        = engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        printf("  %s [%s]: ", name, isInput ? "input" : "output");
        for (int d = 0; d < dims.nbDims; d++) printf("%" PRId64 "%s", dims.d[d], d < dims.nbDims - 1 ? "x" : "");
        printf("\n");
        if (isInput)
        {
            inputName = name;
            numInputs++;
        }
    }
    printf("\n");

    if (numInputs != 1)
    {
        fprintf(stderr, "Engine must have exactly one input, found %d\n", numInputs);
        return -1;
    }
    const nvinfer1::Dims in = engine.getTensorShape(inputName.c_str());
    if (engine.getTensorDataType(inputName.c_str()) != nvinfer1::DataType::kFLOAT || in.nbDims != 4 || in.d[0] != 1 ||
        in.d[1] != 3 || in.d[2] != modelH || in.d[3] != modelW)
    {
        fprintf(stderr,
                "Engine input '%s' must be float32 NCHW 1x3x%dx%d. Generate the model with "
                "prepare_yolov5_nms_onnx.py and rebuild the engine.\n",
                inputName.c_str(), modelH, modelW);
        return -1;
    }

    auto isOutput = [&engine](const char *name, nvinfer1::DataType type) {
        return engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT &&
               engine.getTensorDataType(name) == type;
    };
    const nvinfer1::Dims boxes   = engine.getTensorShape("detection_boxes");
    const nvinfer1::Dims scores  = engine.getTensorShape("detection_scores");
    const nvinfer1::Dims classes = engine.getTensorShape("detection_classes");
    const bool outputTypesValid  = numIO == 5 && isOutput("num_detections", nvinfer1::DataType::kINT32) &&
                                  isOutput("detection_boxes", nvinfer1::DataType::kFLOAT) &&
                                  isOutput("detection_scores", nvinfer1::DataType::kFLOAT) &&
                                  isOutput("detection_classes", nvinfer1::DataType::kINT32);
    const bool outputShapesValid = boxes.nbDims == 3 && boxes.d[0] == 1 && boxes.d[1] > 0 && boxes.d[2] == 4 &&
                                   scores.nbDims == 2 && scores.d[1] == boxes.d[1] && classes.nbDims == 2 &&
                                   classes.d[1] == boxes.d[1];
    if (!outputTypesValid || !outputShapesValid)
    {
        fprintf(stderr,
                "Engine outputs must be the EfficientNMS outputs num_detections, detection_boxes, "
                "detection_scores and detection_classes with a static detection count. Generate the model with "
                "prepare_yolov5_nms_onnx.py and rebuild the engine.\n");
        return -1;
    }
    maxDets = static_cast<int>(boxes.d[1]);
    printf("  Max detections: %d (from engine)\n\n", maxDets);
    return 0;
}

// Get one frame onto the device: take the preloaded pointer or decode the
// file, apply letterbox geometry, and leave the result where the preprocess
// stage expects it. Returns 0 on success, 1 to skip an unusable frame, and -1
// on a CUDA or NVCV failure, which the YOLOV5_*_CHECK macros return.
int GpuOnlyPipeline::prepareFrameInput(const std::string &curPath, int imageIndex, GpuFrameInput &out)
{

    out.lb.origW  = modelW;
    out.lb.origH  = modelH;
    out.lb.scaleX = out.lb.scaleY = 1.0f;
    out.lb.padX = out.lb.padY = 0;

    yolov5_common::Timer loadTimer;
    loadTimer.start();

    if (!curPath.empty())
    {
        // CUDA GPU letterbox
        if (usePreloadedBins && imageIndex < static_cast<int>(preloadedBins.size()))
        {
            const auto &p      = preloadedBins[imageIndex];
            size_t preloadSize = static_cast<size_t>(p.w) * static_cast<size_t>(p.h) * 3u;
            YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStart, gpu.stream));
            YOLOV5_CUDA_CHECK(
                cudaMemcpyAsync(gpu.d_preloadBuf, p.ptr, preloadSize, cudaMemcpyHostToDevice, gpu.stream));
            YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStop, gpu.stream));
            YOLOV5_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
            YOLOV5_CUDA_CHECK(cudaEventElapsedTime(&out.h2dCopyMs, gpu.evStart, gpu.evStop));
            YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStart, gpu.stream));
            if (yolov5_common::letterboxFromBufferGPU(gpu.d_preloadBuf, p.w, p.h, gpu.resizedU8Tensor, modelW, modelH,
                                                      &out.lb, gpu.stream) != 0)
            {
                fprintf(stderr, "  Failed to letterbox preloaded image %d\n", imageIndex);
                return 1;
            }
            YOLOV5_CUDA_CHECK(cudaEventRecord(gpu.evStop, gpu.stream));
            YOLOV5_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
            YOLOV5_CUDA_CHECK(cudaEventElapsedTime(&out.resizePadMs, gpu.evStart, gpu.evStop));
            out.imageDecodeMs = 0.f;
        }
        else
        {
            yolov5_common::LetterboxGPUTimings lbTimings{};
            if (yolov5_common::loadImageLetterboxGPU(curPath.c_str(), gpu.resizedU8Tensor, modelW, modelH, &out.lb,
                                                     gpu.stream, &lbTimings) != 0)
            {
                fprintf(stderr, "  Skipping image\n");
                return 1;
            }
            out.imageDecodeMs = lbTimings.decodeMs;
            out.resizePadMs   = lbTimings.resizePadMs;
            out.h2dCopyMs     = lbTimings.h2dCopyMs;
        }
    }
    else
    {
        NVCVTensorData tensorData;
        YOLOV5_NVCV_CHECK(nvcvTensorExportData(gpu.resizedU8Tensor, &tensorData));
        void *devPtr      = tensorData.buffer.strided.basePtr;
        int64_t rowStride = tensorData.buffer.strided.strides[1];
        for (int y = 0; y < modelH; y++)
            YOLOV5_CUDA_CHECK(
                cudaMemsetAsync(static_cast<uint8_t *>(devPtr) + y * rowStride, 128, modelW * 3, gpu.stream));
        YOLOV5_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    }
    out.loadMs = static_cast<float>(loadTimer.stopMs());
    return 0;
}

// Print one frame's detections and accumulate its COCO output. Kept out of
// the iteration loop so the loop shows the pipeline and nothing else.
void GpuOnlyPipeline::reportFrameDetections(int iter, int64_t curImageId, const GpuFrameInput &frame, int numDets,
                                            float normalizeMs2, float inferMs, float d2hCopyMs, float &buildBoxesMs)
{
    if (!benchmarkMode || iter < warmupRuns)
    {
        printf("\n  Detections: %d (after EfficientNMS)\n", numDets);
        printf("  Load:       %.3f ms  Convert: %.3f ms  Inference: %.3f ms  D→H: %.3f ms\n", frame.loadMs,
               normalizeMs2, inferMs, d2hCopyMs);
        if (numDets > 0)
        {
            printf("\n  Detected objects:\n");
            int displayCount = std::min(numDets, static_cast<int32_t>(10));
            for (int i = 0; i < displayCount; i++)
            {
                float dispL = (hBoxes[i * 4 + 0] - frame.lb.padX) / frame.lb.scaleX;
                float dispT = (hBoxes[i * 4 + 1] - frame.lb.padY) / frame.lb.scaleY;
                float dispR = (hBoxes[i * 4 + 2] - frame.lb.padX) / frame.lb.scaleX;
                float dispB = (hBoxes[i * 4 + 3] - frame.lb.padY) / frame.lb.scaleY;
                printf("  Detection %d: left=%.1f top=%.1f right=%.1f bottom=%.1f confidence=%.3f class=%d\n", i, dispL,
                       dispT, dispR, dispB, hScores[i], hClasses[i]);
            }
            if (numDets > 10)
                printf("  ... (%d more not shown)\n", numDets - 10);

            printf("\n  Classes detected: ");
            std::set<int> uniqueClasses;
            for (int i = 0; i < numDets; i++) uniqueClasses.insert(hClasses[i]);
            for (int cls : uniqueClasses)
            {
                if (cls >= 0 && cls < 80)
                    printf("%s ", yolov5_common::COCO_CLASSES[cls]);
            }
            printf("\n\n");
        }
    }

    // Collect COCO detections (remap to original image coords)
    if (evalMode && numDets > 0)
    {
        auto tBuild0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numDets; i++)
        {
            float x1 = (hBoxes[i * 4 + 0] - frame.lb.padX) / frame.lb.scaleX;
            float y1 = (hBoxes[i * 4 + 1] - frame.lb.padY) / frame.lb.scaleY;
            float x2 = (hBoxes[i * 4 + 2] - frame.lb.padX) / frame.lb.scaleX;
            float y2 = (hBoxes[i * 4 + 3] - frame.lb.padY) / frame.lb.scaleY;

            x1 = std::max(0.f, std::min(x1, static_cast<float>(frame.lb.origW)));
            y1 = std::max(0.f, std::min(y1, static_cast<float>(frame.lb.origH)));
            x2 = std::max(0.f, std::min(x2, static_cast<float>(frame.lb.origW)));
            y2 = std::max(0.f, std::min(y2, static_cast<float>(frame.lb.origH)));

            float bw = x2 - x1, bh = y2 - y1;
            if (bw <= 0 || bh <= 0)
                continue;

            int classIdx = hClasses[i];
            int cocoId   = (classIdx >= 0 && classIdx < 80) ? yolov5_common::COCO_CATEGORY_IDS[classIdx] : classIdx + 1;

            yolov5_common::CocoDetection d;
            d.imageId    = curImageId;
            d.categoryId = cocoId;
            d.x          = x1;
            d.y          = y1;
            d.width      = bw;
            d.height     = bh;
            d.score      = hScores[i];
            allDetections.push_back(d);
        }
        buildBoxesMs =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - tBuild0).count();
    }
}

int GpuOnlyPipeline::run()
{
    // Main loop
    {
        int totalIter = 1;
        if (evalMode)
            totalIter = static_cast<int>(cfg.imageList.size());
        else if (benchmarkMode)
            totalIter = benchmarkIter + warmupRuns;

        for (int iter = 0; iter < totalIter; iter++)
        {
            float preprocMs = 0, inferMs = 0;
            float normalizeMs2 = 0;
            float d2hCopyMs = 0, buildBoxesMs = 0;

            auto iterStartTime = std::chrono::high_resolution_clock::now();

            std::string curPath = cfg.imagePath;
            int64_t curImageId  = 0;
            if (evalMode && !cfg.imageList.empty())
            {
                curPath    = cfg.imageList[iter];
                curImageId = yolov5_common::extractImageId(curPath);
                printf("[%d/%d] %s (ID: %" PRId64 ")\n", iter + 1, totalIter, curPath.c_str(), curImageId);
            }
            else if (benchmarkMode && !cfg.imageList.empty() && iter < static_cast<int>(cfg.imageList.size()))
            {
                curPath    = cfg.imageList[0];
                curImageId = yolov5_common::extractImageId(curPath);
                if (iter < warmupRuns)
                    printf("\n[Warmup %d/%d]\n", iter + 1, warmupRuns);
                else if (iter == warmupRuns)
                    printf("\n[Benchmark: %d iterations]\n", benchmarkIter);
            }
            else if (benchmarkMode && !curPath.empty())
            {
                curImageId = yolov5_common::extractImageId(curPath);
                if (iter < warmupRuns)
                    printf("\n[Warmup %d/%d]\n", iter + 1, warmupRuns);
                else if (iter == warmupRuns)
                    printf("\n[Benchmark: %d iterations]\n", benchmarkIter);
            }

            // Stage 1: Load image
            const int imageIndex = evalMode ? iter : 0;
            GpuFrameInput frame;
            const int prepStatus = prepareFrameInput(curPath, imageIndex, frame);
            if (prepStatus < 0)
                return 1;
            if (prepStatus > 0)
                continue;
            // Stage 2: Convert (NHWC uint8 -> NCHW float32)
            yolov5_common::Timer preprocTimer;
            preprocTimer.start();
            float normalizeMs = 0;
            if (submitGpuPreprocess(gpu, modelH, modelW, &normalizeMs) != 0)
                return 1;

            preprocMs    = static_cast<float>(preprocTimer.stopMs());
            normalizeMs2 = normalizeMs;

            // Stage 3: TensorRT inference + decode + NMS
            if (submitGpuInference(gpu, useCudaGraph, &inferMs) != 0)
                return 1;

            // Stage 4: Materialize results from pinned output buffers
            int32_t numDets = readGpuDetections(gpu, maxDets, hBoxes, hScores, hClasses);
            lastNumDets     = numDets;
            d2hCopyMs       = 0.f;

            if (cfg.checkTopK > 0)
            {
                lastDetections.clear();
                lastDetections.reserve(static_cast<size_t>(numDets));
                for (int32_t i = 0; i < numDets; i++)
                {
                    yolov5_common::DetBox d;
                    d.left    = (hBoxes[i * 4 + 0] - frame.lb.padX) / frame.lb.scaleX;
                    d.top     = (hBoxes[i * 4 + 1] - frame.lb.padY) / frame.lb.scaleY;
                    d.right   = (hBoxes[i * 4 + 2] - frame.lb.padX) / frame.lb.scaleX;
                    d.bottom  = (hBoxes[i * 4 + 3] - frame.lb.padY) / frame.lb.scaleY;
                    d.score   = hScores[i];
                    d.classId = hClasses[i];
                    lastDetections.push_back(d);
                }
            }

            // Report this frame's detections
            reportFrameDetections(iter, curImageId, frame, numDets, normalizeMs2, inferMs, d2hCopyMs, buildBoxesMs);

            // Whole-pipeline wall-clock time for this iteration
            auto iterEndTime       = std::chrono::high_resolution_clock::now();
            float iterPipelineTime = std::chrono::duration<float, std::milli>(iterEndTime - iterStartTime).count();

            // Collect timings
            bool skipTiming = benchmarkMode && iter < warmupRuns;
            if (!skipTiming)
            {
                loadTimings.push_back(frame.loadMs);
                imageDecodeTimings.push_back(frame.imageDecodeMs);
                cpuResizeTimings.push_back(frame.resizePadMs);
                h2dCopyTimings.push_back(frame.h2dCopyMs);
                normalizeTimings.push_back(normalizeMs2);
                preprocTimings.push_back(preprocMs);
                inferTimings.push_back(inferMs);
                d2hCopyTimings.push_back(d2hCopyMs);
                buildBoxesTimings.push_back(buildBoxesMs);
                totalTimings.push_back(iterPipelineTime);
                totalDetsCount += numDets;
            }

            if (benchmarkMode && iter >= warmupRuns)
            {
                int doneCount = iter - warmupRuns + 1;
                if (doneCount % 100 == 0 || doneCount == benchmarkIter)
                    printf("[%d/%d] Done (dets=%d, %.2f ms)\n", doneCount, benchmarkIter, numDets, iterPipelineTime);
            }
            else
                printf("\n"); // eval or warmup: newline after each iteration
        }
    }

    return 0;
}

int GpuOnlyPipeline::report()
{
    yolov5_common::freePreloadedBins(preloadedBins);

    if (cfg.expectedDetections >= 0 && lastNumDets != cfg.expectedDetections)
    {
        fprintf(stderr, "ERROR: expected %d detections, got %d\n", cfg.expectedDetections, lastNumDets);
        return 1;
    }

    if (cfg.checkTopK > 0 && !yolov5_common::checkTopKAgainstReference(lastDetections, cfg.checkTopK, cfg.checkTopKIoU))
        return 1;

    // Summary
    yolov5_common::TimingStats pipe{}, decode{}, gpuRes{}, h2d{}, norm{}, inf{}, d2h{}, buildB{};
    float otherAvg = 0.0F;
    if (!totalTimings.empty())
    {
        pipe               = yolov5_common::TimingStats::compute(totalTimings);
        decode             = yolov5_common::TimingStats::compute(imageDecodeTimings);
        gpuRes             = yolov5_common::TimingStats::compute(cpuResizeTimings);
        h2d                = yolov5_common::TimingStats::compute(h2dCopyTimings);
        norm               = yolov5_common::TimingStats::compute(normalizeTimings);
        inf                = yolov5_common::TimingStats::compute(inferTimings);
        d2h                = yolov5_common::TimingStats::compute(d2hCopyTimings);
        buildB             = yolov5_common::TimingStats::compute(buildBoxesTimings);
        float sumStagesAvg = decode.avg + gpuRes.avg + h2d.avg + norm.avg + inf.avg + d2h.avg + buildB.avg;
        otherAvg           = std::max(0.0F, pipe.avg - sumStagesAvg);
    }

    if (evalMode || benchmarkMode)
    {
        if (evalMode)
        {
            printf("Evaluation results\n");
            printf("  Images processed: %zu\n", totalTimings.size());
            printf("  Total detections: %d\n", totalDetsCount);
        }
        else if (benchmarkMode)
        {
            printf("Benchmark results\n");
            printf("  Benchmark iterations: %d (warmup: %d)\n", benchmarkIter, warmupRuns);
        }

        if (!totalTimings.empty())
        {
            printf("  Pipeline timing (ms): min=%.3f max=%.3f avg=%.3f p50=%.3f p95=%.3f p99=%.3f std=%.3f\n", pipe.min,
                   pipe.max, pipe.avg, pipe.p50, pipe.p95, pipe.p99, pipe.stddev);
            printf("  Stage breakdown (ms average):\n");
            printf("    File read [CPU]: %.3f (p50=%.3f p95=%.3f; zero with preload)\n", decode.avg, decode.p50,
                   decode.p95);
            printf("    H2D copy: %.3f (p50=%.3f p95=%.3f)\n", h2d.avg, h2d.p50, h2d.p95);
            printf("    Resize+pad [GPU]: %.3f (p50=%.3f p95=%.3f)\n", gpuRes.avg, gpuRes.p50, gpuRes.p95);
            printf("    Convert [GPU]: %.3f (p50=%.3f p95=%.3f)\n", norm.avg, norm.p50, norm.p95);
            printf("    Inference [GPU]: %.3f (p50=%.3f p95=%.3f)\n", inf.avg, inf.p50, inf.p95);
            printf("    D2H copy: %.3f (p50=%.3f p95=%.3f)\n", d2h.avg, d2h.p50, d2h.p95);
            printf("    Build boxes [CPU]: %.3f (p50=%.3f p95=%.3f)\n", buildB.avg, buildB.p50, buildB.p95);
            printf("    Other (sync/misc): %.3f\n", otherAvg);
            printf("  Average FPS: %.2f\n", 1000.0f / pipe.avg);
        }
        printf("\n");

        // Write summary JSON only when explicitly requested.
        if (!totalTimings.empty() && !cfg.summaryJsonPath.empty())
        {
            std::string device             = "gpu";
            std::string precision          = useINT8 ? "int8" : (useFP16 ? "fp16" : "fp32");
            std::string mode               = evalMode ? "eval" : "benchmark";
            const std::string &summaryPath = cfg.summaryJsonPath;

            std::ofstream sfs(summaryPath);
            if (sfs.is_open())
            {
                sfs << "{\n";
                sfs << "  \"config\": {\n";
                sfs << "    \"device\": \"" << device << "\",\n";
                sfs << "    \"precision\": \"" << precision << "\",\n";
                sfs << "    \"mode\": \"" << mode << "\",\n";
                sfs << "    \"model\": \"" << cfg.onnxPath << "\",\n";
                sfs << "    \"engine\": \"" << cfg.enginePath << "\",\n";
                sfs << "    \"input_size\": \"" << modelW << "x" << modelH << "\"\n";
                sfs << "  },\n";
                sfs << "  \"results\": {\n";
                sfs << "    \"images_processed\": " << totalTimings.size() << ",\n";
                sfs << "    \"total_detections\": " << totalDetsCount << ",\n";
                sfs << "    \"avg_fps\": " << (pipe.avg > 0 ? 1000.0f / pipe.avg : 0.0f) << "\n";
                sfs << "  },\n";
                sfs << "  \"timing_ms\": {\n";
                yolov5_common::writeTimingJson(sfs, "pipeline", pipe);
                sfs << ",\n";
                yolov5_common::writeTimingJson(sfs, "file_read", decode);
                sfs << ",\n";
                yolov5_common::writeTimingJson(sfs, "gpu_resize_pad", gpuRes);
                sfs << ",\n";
                yolov5_common::writeTimingJson(sfs, "h2d_copy", h2d);
                sfs << ",\n";
                yolov5_common::writeTimingJson(sfs, "convert", norm);
                sfs << ",\n";
                yolov5_common::writeTimingJson(sfs, "inference", inf);
                sfs << ",\n";
                yolov5_common::writeTimingJson(sfs, "d2h_copy", d2h);
                sfs << ",\n";
                yolov5_common::writeTimingJson(sfs, "build_boxes", buildB);
                sfs << ",\n";
                sfs << "    \"other\": { \"min\": 0, \"max\": 0, \"avg\": " << otherAvg
                    << ", \"stddev\": 0, \"p50\": " << otherAvg << ", \"p95\": " << otherAvg
                    << ", \"p99\": " << otherAvg << " }\n";
                sfs << "  }\n";
                sfs << "}\n";
                sfs.close();
                printf("Summary saved to: %s\n", summaryPath.c_str());
            }
        }

        if (!cfg.cocoOutput.empty() && !allDetections.empty())
        {
            if (yolov5_common::writeCocoJson(allDetections, cfg.cocoOutput))
                printf("COCO predictions saved: %s (%zu detections)\n\n", cfg.cocoOutput.c_str(), allDetections.size());
        }
    }
    else
    {
        auto appEnd = std::chrono::high_resolution_clock::now();
        float appMs = std::chrono::duration<float, std::milli>(appEnd - appStart).count();
        printf("Total app time: %.1f ms\n", appMs);
    }

    return 0;
}

int main(int argc, char *argv[])
{
    // Report a failed allocation rather than letting an exception escape main().
    try
    {
        GpuOnlyPipeline pipeline;
        if (pipeline.initialize(argc, argv) != 0)
            return 1;
        if (!pipeline.shouldRun())
            return 0;
        if (pipeline.run() != 0)
            return 1;
        return pipeline.report();
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
