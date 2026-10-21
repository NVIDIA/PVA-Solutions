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

/**
 * @file pva_gpu_pipeline.cpp
 * @brief Implementation of the 3-stage PVA pre-process, GPU inference, and PVA post-process pipeline.
 *
 * Triple-buffered pipeline (depth=3) overlapping 3 frames with dual-VPU
 * concurrency.  Two VPU-pinned CUDA streams (VPU0 for pre, VPU1 for post)
 * run simultaneously with GPU inference on a third CUDA stream:
 *
 *   VPU0 (pvaPreStream):  PVA Resize + LB+Norm   - frame N+2
 *   GPU  (cudaStream):    TensorRT inference      - frame N+1
 *   VPU1 (pvaPostStream): PVA Fused Transp+Decode (3 scales) + NMS - frame N
 *
 * Benchmark input decoding and one-time initialization happen before timing.
 * The measured iteration covers pipeline scheduling, submission and synchronization.
 *
 * The TensorRT outputs are read by PVA decode in place through CUDA-PVA interop.
 */

#include <cuda_runtime.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <nvcv/alloc/Requirements.h> // nvcvMemRequirementsCalcTotalSizeBytes

// TensorRT headers
#include <NvInfer.h>
#include <NvOnnxParser.h>

// PVA Solution operator headers
#include <cupva_host.hpp> // cupva::mem namespace

#include <ErrorCheckMacros.h> // CUPVA_CHECK_ERROR
#include <OpBatchedNMS.h>
#include <OpImageResize.h>
#include <OpNhwcToNchwLetterboxNormalize.h>
#include <OpYoloV5Decode.h>
#include <PvaAllocator.h>     // nvcvAllocatorConstructPva, nvcvTensorCalcRequirementsPva
#include <PvaOperatorTypes.h> // PVA_INTERPOLATION_LINEAR
#include <cupva_host.h>       // CupvaMemGetHostPointer, etc.
#include <cupva_host_cuda.h>  // CUDA stream submission and PVA affinity APIs

// CUDA-PVA interop utilities (nvcvImageWrapCudaBuf, nvcvTensorWrapCudaBuf)
extern "C" {
#include "CudaInteropUtils.h"
}

// Detection headers
#include "yolov5_pipeline_iface.hpp"
#include "yolov5_pipeline_interop.hpp"
#include "yolov5_pipeline_schedule.hpp"
#include "yolov5_pipeline_tensors.hpp"

#include "coco_evaluation.h" // For Box struct only
#include "detection_check.h"
#include "image_io.h"
#include "int8_calibrator.h"
#include "pipeline_args.h"
#include "pipeline_error.h"
#include "trt_utils.h"
#include "yolov5_common.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

// ============================================================================
// TensorRT Logger
// ============================================================================

static yolov5_common::TRTLogger gLogger;
static std::unique_ptr<nvinfer1::IRuntime> gRuntime;

// With some Thor GPU drivers, once CupvaCudaStreamSetAffinity enables PVA-CUDA interop,
// cudaMalloc fails with CUDA_ERROR_EXTERNAL_DEVICE for sizes of 2 MiB or more that are not
// a multiple of 4 KiB, and an allocation like that which is still live makes the affinity
// call itself fail. Round TensorRT's and the pipeline's allocations up to 4 KiB.
static size_t roundTo4K(size_t bytes)
{
    return (bytes + 4095) & ~size_t(4095);
}

class RoundTo4KAllocator : public nvinfer1::IGpuAsyncAllocator
{
public:
    void *allocateAsync(uint64_t size, uint64_t, nvinfer1::AllocatorFlags, cudaStream_t) noexcept override
    {
        void *ptr = nullptr;
        return cudaMalloc(&ptr, roundTo4K(size)) == cudaSuccess ? ptr : nullptr;
    }
    bool deallocateAsync(void *memory, cudaStream_t) noexcept override
    {
        return cudaFree(memory) == cudaSuccess;
    }
};
static RoundTo4KAllocator gAllocator;

// ============================================================================
// TensorRT Engine Build
// ============================================================================

nvinfer1::ICudaEngine *buildEngineFromOnnx(const std::string &onnxPath, nvinfer1::IBuilder *builder,
                                           bool useFP16 = true, bool useDLA = false, int dlaCore = 0,
                                           bool useINT8 = false, const std::string &calibDir = "")
{
    printf("Building TensorRT engine from ONNX: %s\n", onnxPath.c_str());

    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    if (!network)
        return nullptr;

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
        return nullptr;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        fprintf(stderr, "Failed to parse ONNX file\n");
        return nullptr;
    }
    printf("  ONNX parsed successfully\n");

    // Handle FP16 ONNX models: Set input/output to FP32 so TensorRT handles conversion
    // This allows us to pass FP32 data even if ONNX specifies FP16
    for (int i = 0; i < network->getNbInputs(); i++)
    {
        auto input = network->getInput(i);
        auto dtype = input->getType();
        if (dtype == nvinfer1::DataType::kHALF)
        {
            printf("  Input '%s' is FP16 in ONNX, setting to FP32 (TensorRT will convert)\n", input->getName());
            input->setType(nvinfer1::DataType::kFLOAT);
        }
    }
    for (int i = 0; i < network->getNbOutputs(); i++)
    {
        auto output = network->getOutput(i);
        auto dtype  = output->getType();
        if (dtype == nvinfer1::DataType::kHALF)
        {
            printf("  Output '%s' is FP16 in ONNX, setting to FP32 (TensorRT will convert)\n", output->getName());
            output->setType(nvinfer1::DataType::kFLOAT);
        }
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
        return nullptr;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Enable FP16 (required for DLA, optional for GPU)
    if ((useFP16 || useINT8) && builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        printf("  FP16 mode enabled\n");
    }

    // The calibrator must remain alive until buildSerializedNetwork returns.
    std::unique_ptr<yolov5_common::Int8Calibrator> calibrator;

    // Enable INT8 with calibrator
    if (useINT8 && builder->platformHasFastInt8())
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        calibrator =
            std::make_unique<yolov5_common::Int8Calibrator>(calibDir, 1, 3, 672, 672, "calib_cache_trimmed_lb.bin");
        config->setInt8Calibrator(calibrator.get());
        printf("  INT8 mode enabled (calibrating with images from %s)\n", calibDir.c_str());
    }
    else if (useINT8)
    {
        printf("  Warning: INT8 requested but platform does not support fast INT8\n");
        printf("  Falling back to FP16/FP32\n");
    }

    // Configure DLA if requested
    if (useDLA)
    {
        int numDLACores = builder->getNbDLACores();
        if (numDLACores > 0)
        {
            if (dlaCore >= numDLACores)
            {
                printf("  Warning: DLA core %d not available, using core 0\n", dlaCore);
                dlaCore = 0;
            }
            config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            config->setDLACore(dlaCore);
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

            // DLA requires FP16 or INT8
            if (!builder->platformHasFastFp16())
            {
                printf("  Warning: DLA requires FP16, but platform doesn't support fast FP16\n");
            }

            printf("  DLA core %d enabled with GPU fallback\n", dlaCore);
            printf("  Note: Unsupported layers will run on GPU\n");
        }
        else
        {
            printf("  Warning: DLA requested but no DLA cores available on this platform\n");
            printf("  Falling back to GPU only execution\n");
        }
    }

    printf("  Building engine (this may take a few minutes)...\n");
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine)
        return nullptr;

    if (!gRuntime)
    {
        gRuntime.reset(nvinfer1::createInferRuntime(gLogger));
        gRuntime->setGpuAllocator(&gAllocator);
    }
    nvinfer1::ICudaEngine *engine = gRuntime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());

    if (engine)
        printf("  Engine built successfully\n");
    return engine;
}

// True when a TensorRT tensor shape is exactly [n, c, h, w].
static bool hasShape(const nvinfer1::Dims &dims, int64_t n, int64_t c, int64_t h, int64_t w)
{
    return dims.nbDims == 4 && dims.d[0] == n && dims.d[1] == c && dims.d[2] == h && dims.d[3] == w;
}

using namespace yolov5_common;

// ============================================================================
// Pipeline stage submission
// ============================================================================

enum class ParseResult
{
    Ready,
    ExitSuccess,
    Error
};

static ParseResult parsePipelineConfig(int argc, char *argv[], yolov5_common::PipelineConfig &config,
                                       bool &useCudaGraph)
{
    std::vector<std::pair<std::string, std::string>> extra;
    int rc = yolov5_common::parseCommonArgs(argc, argv, config, "yolov5s_trimmed_reshape_tranpose.onnx",
                                            "yolov5s_trimmed_reshape_tranpose.engine", extra);
    if (rc == 1)
    {
        yolov5_common::printCommonHelp(argv[0]);
        printf("\nPipeline-specific options:\n");
        printf("  --cuda-graph            Capture TensorRT inference in a CUDA graph (benchmark only)\n");
        printf("  --check-pva-driver      Exit with 0 if the PVA driver supports CUDA-stream submission,\n");
        printf("                          77 if it does not. Must be the only argument.\n");
        return ParseResult::ExitSuccess;
    }
    if (rc < 0)
        return ParseResult::Error;

    for (const auto &option : extra)
    {
        if (option.first == "--cuda-graph")
            useCudaGraph = true;
        else
        {
            fprintf(stderr, "Unknown option: %s\n", option.first.c_str());
            return ParseResult::Error;
        }
    }

    if (yolov5_common::finalizeConfig(config) != 0)
        return ParseResult::Error;
    if (useCudaGraph && !config.benchmarkMode)
    {
        fprintf(stderr, "--cuda-graph requires --benchmark\n");
        return ParseResult::Error;
    }
    return ParseResult::Ready;
}

static void printPipelineConfig(const yolov5_common::PipelineConfig &config, int srcWidth, int srcHeight,
                                int modelWidth, int modelHeight)
{
    printf("Configuration:\n");
    printf("  Input size:  %dx%d (RGB)\n", srcWidth, srcHeight);
    if (config.evalMode)
        printf("  Input src:   %zu images from %s\n", config.imageList.size(), config.imagesDir.c_str());
    else if (config.benchmarkMode)
        printf("  Input src:   %s (%d iterations)\n", config.imagePath.c_str(), config.benchmarkIter);
    else
        printf("  Input src:   %s\n", config.imagePath.c_str());
    printf("  Model size:  %dx%d (RGB, float32 [0,1])\n", modelWidth, modelHeight);
    printf("  ONNX model:  %s\n", config.onnxPath.c_str());
    printf("  Engine:      %s\n", config.enginePath.c_str());
    printf("  Precision:   %s\n", config.useINT8 ? "INT8" : (config.useFP16 ? "FP16" : "FP32"));
    printf("  Execution:   %s\n",
           config.useDLA ? ("DLA core " + std::to_string(config.dlaCore) + " + GPU fallback").c_str() : "GPU only");
    printf("  Confidence:  %.2f\n", config.confidenceThreshold);
    printf("  NMS IoU:     %.2f\n", config.nmsThreshold);
    if (!config.cocoOutput.empty())
        printf("  COCO output: %s\n", config.cocoOutput.c_str());
    printf("\n");
}

#if CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

class PvaGpuPipeline final
{
public:
    ~PvaGpuPipeline();

    int initialize(int argc, char *argv[]);
    int run();
    int report();
    bool shouldRun() const;

private:
    void cleanup();
    bool prepareFrameInput(int preFrame, int sp);

    // One iteration's work: three submissions, then the barrier.
    int submitPvaPreprocess(bool enabled, int slot);
    int submitGpuInference(bool enabled, int slot);
    int submitPvaPostprocess(bool enabled, int slot);
    int awaitIteration(bool preprocessSubmitted, bool inferenceSubmitted, bool postprocessSubmitted);
    int collectFrameResults(const yolov5_pipeline::FramesInFlight &turn, bool isMeasuredPost, int pipelineFrames,
                            int &numDecodedBoxes, int &numNmsBoxes);

    static constexpr int NUM_PIPE        = 3;
    static constexpr int NUM_YOLO_SCALES = 3;

    /// Which slots currently hold a usable frame.
    bool frameValid[NUM_PIPE] = {};
    /// Letterbox size each slot's TensorRT input padding was filled for.
    int paddedLbW[NUM_PIPE] = {};
    int paddedLbH[NUM_PIPE] = {};
    /// Set when a slot's PVA resources could not be recreated; ends the run.
    bool slotSetupFailed = false;

    static constexpr float paddingValue = 114.0f / 255.0f;
    static constexpr int modelWidth     = 672;
    static constexpr int modelHeight    = 672;
    static constexpr int channels       = 3;
    static constexpr int numClasses     = 80;
    static constexpr int maxDecodeBoxes = 5120;
    static constexpr int maxNmsBoxes    = 1000;

    yolov5_common::PipelineConfig cfg;
    std::string onnxPath;
    std::string enginePath;
    std::string imagePath;
    std::string cocoOutput;
    std::string calibImagesDir;
    bool useDLA               = false;
    bool useFP16              = false;
    bool useINT8              = false;
    int dlaCore               = 0;
    bool evalMode             = false;
    bool benchmarkMode        = false;
    bool verbose              = false;
    float confidenceThreshold = 0.F;
    float nmsThreshold        = 0.F;
    std::vector<std::string> imageList;
    int totalIterations = 0;
    int warmupRuns      = 0;
    int benchmarkIter   = 0;
    bool useCudaGraph   = false;

    int srcWidth  = 1280;
    int srcHeight = 720;
    float lbScale = 1.F;
    int lbW       = 0;
    int lbH       = 0;
    int padTop    = 0;
    int padLeft   = 0;

    std::vector<yolov5_common::PreloadedBin> preloadedBins;
    bool usePreloadedBins = false;

    cudaStream_t cudaStream                      = nullptr;
    cudaGraphExec_t inferenceGraphExec[NUM_PIPE] = {};
    cudaEvent_t preprocessStart = nullptr, preprocessStop = nullptr;
    cudaEvent_t inferenceStart = nullptr, inferenceStop = nullptr;
    cudaEvent_t postprocessStart = nullptr, postprocessStop = nullptr;

    NVCVOperatorHandle pvaResizeOp                   = nullptr;
    NVCVOperatorHandle pvaPreprocessOp               = nullptr;
    NVCVOperatorHandle pvaNmsOp                      = nullptr;
    const int YOLO_STRIDES[NUM_YOLO_SCALES]          = {8, 16, 32};
    NVCVOperatorHandle pvaDecodeOps[NUM_YOLO_SCALES] = {};

    NVCVAllocatorHandle pvaAlloc = nullptr;
    cudaStream_t pvaPreStream    = nullptr;
    cudaStream_t pvaPostStream   = nullptr;

    // Source image, resize output and its HWC view, per slot.
    yolov5_pipeline::PvaPreprocessTensors preTensors;
    float *d_nhwcBuf[NUM_PIPE] = {};
    int opSrcW = 0, opSrcH = 0;
    int slotImageIndex[NUM_PIPE] = {};
    std::vector<NVCVImageHandle> wrappedPvaInImages;

    // TensorRT buffers and the PVA views that alias them.
    yolov5_pipeline::PvaTrtInterop interop;
    // Anchor boxes, one tensor per scale.
    yolov5_pipeline::PvaAnchorTensors anchorTensors;

    // Decode and NMS outputs for every slot, with their host pointers.
    yolov5_pipeline::PvaDetectionTensors detTensors;

    nvinfer1::IBuilder *trtBuilder          = nullptr;
    nvinfer1::ICudaEngine *trtEngine        = nullptr;
    nvinfer1::IExecutionContext *trtContext = nullptr;
    int numTrtOutputs                       = 0;
    std::string inputTensorName;
    std::string outputTensorNames[NUM_YOLO_SCALES];
    int totalNumAnchors = 0;

    LetterboxInfo lbInfoSlot[NUM_PIPE] = {};
    int64_t imageIdSlot[NUM_PIPE]      = {};
    std::string imagePathSlot[NUM_PIPE];
    std::vector<CocoDetection> allDetections;
    int totalDetections = 0;

    std::function<void(int64_t)> recreatePreprocessOps;
    std::function<int(int)> recreateSlotResources;
    NVCVTensorLayout hwcLayout;
    NVCVTensorRequirements lbOutReqs = {};

    int result            = 0;
    int steadyStateFrames = 0;
    int lastNumNmsBoxes   = -1;
    std::vector<yolov5_common::DetBox> lastDetections;
    int inputLoadFailures = 0;
    float totalSteadyMs   = 0.F;
    std::vector<float> pipelineTimings;
    std::vector<float> preprocessTimings;
    std::vector<float> inferenceTimings;
    std::vector<float> postprocessTimings;
    std::chrono::high_resolution_clock::time_point appStartTime;
    bool cleanupComplete = false;
    bool runRequested    = false;
};

PvaGpuPipeline::~PvaGpuPipeline()
{
    cleanup();
}

bool PvaGpuPipeline::shouldRun() const
{
    return runRequested;
}

void PvaGpuPipeline::cleanup()
{
    if (cleanupComplete)
        return;
    cleanupComplete = true;

    yolov5_common::freePreloadedBins(preloadedBins);

    for (cudaGraphExec_t graphExec : inferenceGraphExec)
        if (graphExec)
            cudaGraphExecDestroy(graphExec);
    for (cudaEvent_t event :
         {preprocessStart, preprocessStop, inferenceStart, inferenceStop, postprocessStart, postprocessStop})
        if (event)
            cudaEventDestroy(event);
    if (trtContext)
        delete trtContext;
    if (trtEngine)
        delete trtEngine;
    if (trtBuilder)
        delete trtBuilder;
    gRuntime.reset();

    interop.release();
    anchorTensors.release();
    detTensors.release();
    for (int scale = NUM_YOLO_SCALES - 1; scale >= 0; --scale)
    {
        if (pvaDecodeOps[scale])
            nvcvOperatorDestroy(pvaDecodeOps[scale]);
    }

    if (pvaNmsOp)
        nvcvOperatorDestroy(pvaNmsOp);
    if (pvaPreprocessOp)
        nvcvOperatorDestroy(pvaPreprocessOp);
    if (pvaResizeOp)
        nvcvOperatorDestroy(pvaResizeOp);

    for (NVCVImageHandle image : wrappedPvaInImages)
        if (image)
            nvcvImageDecRef(image, nullptr);
    wrappedPvaInImages.clear();

    for (int slot = 0; slot < NUM_PIPE; ++slot)
    {
        if (d_nhwcBuf[slot])
            cudaFree(d_nhwcBuf[slot]);
    }
    preTensors.release();

    if (pvaPreStream)
        cudaStreamDestroy(pvaPreStream);
    if (pvaPostStream)
        cudaStreamDestroy(pvaPostStream);
    if (pvaAlloc)
        nvcvAllocatorDecRef(pvaAlloc, nullptr);
    if (cudaStream)
        cudaStreamDestroy(cudaStream);
}

// ============================================================================
// Main Pipeline
// ============================================================================

int PvaGpuPipeline::submitPvaPreprocess(bool enabled, int slot)
{
    if (!enabled)
        return 0;

    NVCVImageHandle resizeInput =
        (slotImageIndex[slot] < static_cast<int>(wrappedPvaInImages.size()) && wrappedPvaInImages[slotImageIndex[slot]])
            ? wrappedPvaInImages[slotImageIndex[slot]]
            : preTensors.inImage[slot];
    if ((benchmarkMode ? preprocessStart : nullptr))
        YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? preprocessStart : nullptr), pvaPreStream));
    YOLOV5_NVCV_CHECK(pvaImageResizeSubmit(pvaResizeOp, pvaPreStream, resizeInput, preTensors.outImage[slot]));
    YOLOV5_NVCV_CHECK(pvaNhwcToNchwLetterboxNormalizeSubmit(pvaPreprocessOp, pvaPreStream, preTensors.lbIn[slot],
                                                            interop.lbOut[slot]));
    if ((benchmarkMode ? preprocessStop : nullptr))
        YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? preprocessStop : nullptr), pvaPreStream));
    return 0;
}

int PvaGpuPipeline::submitGpuInference(bool enabled, int slot)
{
    if (!enabled)
        return 0;

    if (useCudaGraph)
    {
        if ((benchmarkMode ? inferenceStart : nullptr))
            YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? inferenceStart : nullptr), cudaStream));
        YOLOV5_CUDA_CHECK(cudaGraphLaunch(inferenceGraphExec[slot], cudaStream));
        if ((benchmarkMode ? inferenceStop : nullptr))
            YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? inferenceStop : nullptr), cudaStream));
        return 0;
    }

    bool bindingsOk = trtContext->setTensorAddress(inputTensorName.c_str(), interop.trtInput[slot]);
    for (int output = 0; output < numTrtOutputs; output++)
        bindingsOk = bindingsOk &&
                     trtContext->setTensorAddress(outputTensorNames[output].c_str(), interop.trtOutput[slot][output]);
    if ((benchmarkMode ? inferenceStart : nullptr))
        YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? inferenceStart : nullptr), cudaStream));
    if (!bindingsOk || !trtContext->enqueueV3(cudaStream))
    {
        fprintf(stderr, "TensorRT enqueue failed for pipeline slot %d\n", slot);
        return -1;
    }
    if ((benchmarkMode ? inferenceStop : nullptr))
        YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? inferenceStop : nullptr), cudaStream));
    return 0;
}

int PvaGpuPipeline::submitPvaPostprocess(bool enabled, int slot)
{
    if (!enabled)
        return 0;

    *detTensors.decCountHost[slot] = 0;
    if ((benchmarkMode ? postprocessStart : nullptr))
        YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? postprocessStart : nullptr), pvaPostStream));
    for (int scale = 0; scale < NUM_YOLO_SCALES; scale++)
    {
        YOLOV5_NVCV_CHECK(pvaYoloV5DecodeSubmit(pvaDecodeOps[scale], pvaPostStream, interop.predictions[slot][scale],
                                                anchorTensors.anchors[scale], detTensors.decBoxes[slot],
                                                detTensors.decScores[slot], detTensors.decClasses[slot],
                                                detTensors.decCount[slot]));
    }
    YOLOV5_NVCV_CHECK(
        pvaBatchedNMSSubmit(pvaNmsOp, pvaPostStream, detTensors.decBoxes[slot], detTensors.decScores[slot],
                            detTensors.decClasses[slot], detTensors.decCount[slot], detTensors.nmsBoxes[slot],
                            detTensors.nmsScores[slot], detTensors.nmsClasses[slot], detTensors.nmsCount[slot]));
    if ((benchmarkMode ? postprocessStop : nullptr))
        YOLOV5_CUDA_CHECK(cudaEventRecord((benchmarkMode ? postprocessStop : nullptr), pvaPostStream));
    return 0;
}

int PvaGpuPipeline::awaitIteration(bool preprocessSubmitted, bool inferenceSubmitted, bool postprocessSubmitted)
{
    if (postprocessSubmitted)
        YOLOV5_CUDA_CHECK(cudaStreamSynchronize(pvaPostStream));
    if (preprocessSubmitted)
        YOLOV5_CUDA_CHECK(cudaStreamSynchronize(pvaPreStream));
    if (inferenceSubmitted)
        YOLOV5_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    return 0;
}

int PvaGpuPipeline::initialize(int argc, char *argv[])
{
    printf("3-Stage Pipelined PVA+TRT YOLOv5 (Pre || Inf || Post)\n\n");
    yolov5_pipeline::printStageMap(yolov5_pipeline::PVA_GPU_STAGE_MAP);

    appStartTime = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Configuration
    // ========================================================================

    ParseResult parseResult = parsePipelineConfig(argc, argv, cfg, useCudaGraph);
    if (parseResult == ParseResult::ExitSuccess)
        return 0;
    if (parseResult == ParseResult::Error)
        return 1;
    runRequested = true;

    // Convenience aliases
    onnxPath            = cfg.onnxPath;
    enginePath          = cfg.enginePath;
    imagePath           = cfg.imagePath;
    cocoOutput          = cfg.cocoOutput;
    calibImagesDir      = cfg.calibImagesDir;
    useDLA              = cfg.useDLA;
    useFP16             = cfg.useFP16;
    useINT8             = cfg.useINT8;
    dlaCore             = cfg.dlaCore;
    evalMode            = cfg.evalMode;
    benchmarkMode       = cfg.benchmarkMode;
    verbose             = cfg.verbose;
    confidenceThreshold = cfg.confidenceThreshold;
    nmsThreshold        = cfg.nmsThreshold;
    imageList           = cfg.imageList;
    totalIterations     = cfg.totalIterations;
    warmupRuns          = cfg.warmupRuns;
    benchmarkIter       = cfg.benchmarkIter;
    // Letterbox geometry (recomputed when srcWidth/srcHeight change)
    lbScale = std::min(static_cast<float>(modelWidth) / srcWidth, static_cast<float>(modelHeight) / srcHeight);
    lbW     = static_cast<int>(std::round(srcWidth * lbScale));
    lbH     = static_cast<int>(std::round(srcHeight * lbScale));
    padTop  = (modelHeight - lbH) / 2;
    padLeft = (modelWidth - lbW) / 2;

    printPipelineConfig(cfg, srcWidth, srcHeight, modelWidth, modelHeight);

    // Benchmark repeats one decoded input, so it is always preloaded and file
    // decoding stays outside the measured iteration. Evaluation keeps the
    // optional --preload behavior for .bin datasets.
    usePreloadedBins = yolov5_common::preloadInputImages(cfg, preloadedBins);
    if ((cfg.preloadBins || benchmarkMode) && !usePreloadedBins)
    {
        fprintf(stderr, "ERROR: benchmark/evaluation input could not be preloaded.\n");
        return 1;
    }

    // Detect actual image dimensions from first available source so that PVA
    // operators are created with the correct input size at startup.
    {
        int detectedW = 0, detectedH = 0;
        if (usePreloadedBins && !preloadedBins.empty())
        {
            detectedW = preloadedBins[0].origW > 0 ? preloadedBins[0].origW : preloadedBins[0].w;
            detectedH = preloadedBins[0].origH > 0 ? preloadedBins[0].origH : preloadedBins[0].h;
        }
        else if (!cfg.imagePath.empty())
        {
            const std::string &path = cfg.imagePath;
            bool isBin              = path.size() >= 4u && path.compare(path.size() - 4, 4, ".bin") == 0;
            if (isBin)
            {
                FILE *f = fopen(path.c_str(), "rb");
                if (f)
                {
                    int32_t bw = 0, bh = 0;
                    if (fread(&bw, sizeof(int32_t), 1, f) == 1 && fread(&bh, sizeof(int32_t), 1, f) == 1 && bw > 0 &&
                        bh > 0)
                    {
                        detectedW = bw;
                        detectedH = bh;
                    }
                    fclose(f);
                }
            }
            else
            {
                int w = 0, h = 0;
                if (yolov5_common::getPngDimensions(path.c_str(), &w, &h))
                {
                    detectedW = w;
                    detectedH = h;
                }
            }
        }
        if (detectedW > 0 && detectedH > 0 && (detectedW != srcWidth || detectedH != srcHeight))
        {
            printf("PVA dynamic sizing: input image %dx%d (overriding default %dx%d)\n", detectedW, detectedH, srcWidth,
                   srcHeight);
            srcWidth  = detectedW;
            srcHeight = detectedH;
            lbScale = std::min(static_cast<float>(modelWidth) / srcWidth, static_cast<float>(modelHeight) / srcHeight);
            lbW     = static_cast<int>(std::round(srcWidth * lbScale));
            lbH     = static_cast<int>(std::round(srcHeight * lbScale));
            padTop  = (modelHeight - lbH) / 2;
            padLeft = (modelWidth - lbW) / 2;
            printf("  Letterbox: scale=%.4f, resized=%dx%d, pad top=%d left=%d\n", lbScale, lbW, lbH, padTop, padLeft);
        }
    }

    // ========================================================================
    // Stage 0: Check the PVA driver
    // ========================================================================

    uint32_t pvaDriverVersion = 0;
    if (CupvaGetDriverVersion(&pvaDriverVersion) != CUPVA_ERROR_NONE)
    {
        fprintf(stderr, "Failed to query the PVA driver version\n");
        return 1;
    }
    printf("  PVA driver:  %u (CUDA stream submission requires >= 2007)\n", pvaDriverVersion);
    if (pvaDriverVersion < 2007)
    {
        fprintf(stderr, "The PVA+GPU pipeline requires a PVA driver with CUDA-stream submission (driver >= 2007).\n");
        return 1;
    }

    // ========================================================================
    // Stage 1: Initialize CUDA
    // ========================================================================

    YOLOV5_CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));

    // PVA allocator + two VPU-pinned streams for true 3-stage concurrency
    YOLOV5_NVCV_CHECK(nvcvAllocatorConstructPva(&pvaAlloc));
    YOLOV5_CUDA_CHECK(cudaStreamCreateWithFlags(&pvaPreStream, cudaStreamNonBlocking));
    YOLOV5_CUDA_CHECK(cudaStreamCreateWithFlags(&pvaPostStream, cudaStreamNonBlocking));
    if (benchmarkMode)
    {
        YOLOV5_CUDA_CHECK(cudaEventCreate(&preprocessStart));
        YOLOV5_CUDA_CHECK(cudaEventCreate(&preprocessStop));
        YOLOV5_CUDA_CHECK(cudaEventCreate(&inferenceStart));
        YOLOV5_CUDA_CHECK(cudaEventCreate(&inferenceStop));
        YOLOV5_CUDA_CHECK(cudaEventCreate(&postprocessStart));
        YOLOV5_CUDA_CHECK(cudaEventCreate(&postprocessStop));
    }

    // Layouts for PVA tensor requirements
    NVCVTensorLayout layout2d, layoutW;
    nvcvTensorLayoutMake("HW", &layout2d);
    nvcvTensorLayoutMake("W", &layoutW);

    // ========================================================================
    // Stage 2: Initialize TensorRT
    // ========================================================================

    printf("Stage 2: TensorRT Initialization\n");

    trtBuilder = nvinfer1::createInferBuilder(gLogger);
    if (!trtBuilder)
    {
        fprintf(stderr, "Failed to create TensorRT builder\n");
        cleanup();
        return -1;
    }
    trtBuilder->setGpuAllocator(&gAllocator);

    printf("  Platform: DLA cores=%d, FP16=%s, INT8=%s\n", trtBuilder->getNbDLACores(),
           trtBuilder->platformHasFastFp16() ? "Yes" : "No", trtBuilder->platformHasFastInt8() ? "Yes" : "No");

    // Try to load existing engine, else build from ONNX
    {
        std::ifstream engineFile(enginePath, std::ios::binary);
        if (engineFile.good())
        {
            engineFile.close();
            printf("  Loading cached engine: %s\n", enginePath.c_str());
            if (!gRuntime)
                gRuntime.reset(nvinfer1::createInferRuntime(gLogger));
            if (gRuntime)
            {
                gRuntime->setGpuAllocator(&gAllocator);
                trtEngine = yolov5_common::loadEngine(enginePath, *gRuntime);
            }
        }
    }

    if (!trtEngine)
    {
        std::ifstream onnxFile(onnxPath);
        if (!onnxFile.good())
        {
            fprintf(stderr, "ONNX model not found: %s\n", onnxPath.c_str());
            cleanup();
            return -1;
        }
        onnxFile.close();
        trtEngine = buildEngineFromOnnx(onnxPath, trtBuilder, useFP16, useDLA, dlaCore, useINT8, calibImagesDir);
        if (!trtEngine)
        {
            fprintf(stderr, "Failed to build a TensorRT engine from %s\n", onnxPath.c_str());
            cleanup();
            return -1;
        }
        printf("  Saving engine to: %s\n", enginePath.c_str());
        yolov5_common::saveEngine(*trtEngine, enginePath);
    }

    // After the engine build: the TensorRT builder makes allocations that bypass gAllocator.
    cupvaError_t affinityStatus = CupvaCudaStreamSetAffinity(pvaPreStream, CUPVA_VPU0);
    if (affinityStatus == CUPVA_ERROR_NONE)
        affinityStatus = CupvaCudaStreamSetAffinity(pvaPostStream, CUPVA_VPU1);
    if (affinityStatus != CUPVA_ERROR_NONE)
    {
        fprintf(stderr, "Failed to set CUDA-PVA stream affinity (%d)\n", static_cast<int>(affinityStatus));
        cleanup();
        return -1;
    }

    if (trtEngine)
    {
        trtContext = trtEngine->createExecutionContext();
        if (!trtContext)
        {
            fprintf(stderr, "Failed to create execution context\n");
            cleanup();
            return -1;
        }

        // Get I/O tensor info - trimmed model has 3 outputs (one per scale)
        printf("  Engine bindings:\n");
        int numIO     = trtEngine->getNbIOTensors();
        numTrtOutputs = 0;

        for (int i = 0; i < numIO; ++i)
        {
            const char *name            = trtEngine->getIOTensorName(i);
            nvinfer1::Dims dims         = trtEngine->getTensorShape(name);
            nvinfer1::TensorIOMode mode = trtEngine->getTensorIOMode(name);

            printf("    [%d] %s: ", i, name);
            size_t size = 1;
            for (int d = 0; d < dims.nbDims; ++d)
            {
                printf("%" PRId64 "%s", dims.d[d], d < dims.nbDims - 1 ? "x" : "");
                size *= dims.d[d];
            }
            printf(" (%s)\n", mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");

            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                // PVA writes one NCHW float32 image of the model size.
                if (trtEngine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
                    !hasShape(dims, 1, channels, modelHeight, modelWidth))
                {
                    fprintf(stderr, "Unsupported TensorRT input %s: expected float32 NCHW [1,%d,%d,%d]\n", name,
                            channels, modelHeight, modelWidth);
                    cleanup();
                    return -1;
                }
                inputTensorName = name;

                // Always use cudaMallocHost so PVA can write directly via
                // nvcvTensorWrapCudaBuf (CUDA-PVA interop).
                for (int s = 0; s < NUM_PIPE; s++)
                    YOLOV5_CUDA_CHECK(cudaMallocHost(&interop.trtInput[s], roundTo4K(size * sizeof(float))));
                interop.inputPinned = true;
                printf("      Allocated with cudaMallocHost (%d slots, PVA zero-copy target)\n", NUM_PIPE);

                bool ok = trtContext->setTensorAddress(name, interop.trtInput[0]);
                printf("      Address set: %s (slot0=%p), %d slots allocated\n", ok ? "OK" : "FAILED",
                       interop.trtInput[0], NUM_PIPE);
            }
            else
            {
                if (numTrtOutputs < 3)
                {
                    outputTensorNames[numTrtOutputs] = name;
                    for (int s = 0; s < NUM_PIPE; s++)
                        YOLOV5_CUDA_CHECK(
                            cudaMalloc(&interop.trtOutput[s][numTrtOutputs], roundTo4K(size * sizeof(float))));
                    bool ok = trtContext->setTensorAddress(name, interop.trtOutput[0][numTrtOutputs]);
                    printf("      Address set: %s (slot0=%p), %d slots allocated\n", ok ? "OK" : "FAILED",
                           interop.trtOutput[0][numTrtOutputs], NUM_PIPE);
                    numTrtOutputs++;
                }
                else
                {
                    fprintf(stderr, "Warning: More than 3 output tensors; ignoring %s\n", name);
                }
            }
        }

        // Compute total anchors (sum across 3 YOLO scales: stride 8, 16, 32)
        {
            totalNumAnchors = 0;
            for (int sc = 0; sc < NUM_YOLO_SCALES; sc++)
            {
                int fmH = modelHeight / YOLO_STRIDES[sc];
                int fmW = modelWidth / YOLO_STRIDES[sc];
                totalNumAnchors += fmH * fmW * 3; // 3 anchors per cell
            }
        }

        if (numTrtOutputs != 3)
        {
            fprintf(stderr, "Error: Expected 3 TRT outputs (one per YOLO scale), got %d\n", numTrtOutputs);
            fprintf(stderr, "  Use yolov5s_trimmed_reshape_tranpose.onnx (three raw detection heads).\n");
            cleanup();
            return -1;
        }
        printf("  Outputs: %d scale tensors, total anchors: %d\n", numTrtOutputs, totalNumAnchors);

        // --- Create PVA fused transpose+decode operators (one per YOLO scale) ---
        {
            NVCVTensorLayout chwLayout;
            nvcvTensorLayoutMake("CHW", &chwLayout);

            // Decode output tensors for every slot. Allocated before the
            // per-scale tensors below; see allocateNms() on why the two halves
            // stay split.
            if (detTensors.allocateDecode(pvaAlloc, maxDecodeBoxes) != 0)
                return -1;

            // Per-scale: create prediction tensor (zero-copy from TRT output), anchor tensor, operator
            for (int sc = 0; sc < NUM_YOLO_SCALES; sc++)
            {
                int stride             = YOLO_STRIDES[sc];
                int gridH              = modelHeight / stride;
                int gridW              = modelWidth / stride;
                int numAnchorsPerScale = gridH * gridW * 3;

                // Prediction tensor: [255, gridH, gridW] CHW
                NVCVTensorRequirements predReqs = {};
                int64_t predShape[]             = {255, gridH, gridW};
                YOLOV5_NVCV_CHECK(
                    nvcvTensorCalcRequirementsPva(3, predShape, NVCV_DATA_TYPE_F32, chwLayout, 0, 0, &predReqs));

                // Alias each slot's TensorRT output as this scale's prediction
                // tensor so decode reads the inference result in place.
                for (int s = 0; s < NUM_PIPE; s++)
                {
                    if (interop.wrapPrediction(s, sc, predReqs, gridH, gridW) != 0)
                    {
                        fprintf(stderr, "  Failed to import TRT output slot %d scale %d into PVA memory\n", s, sc);
                        cleanup();
                        return -1;
                    }
                }

                // Anchor tensor: [numAnchorsPerScale, 5] (shared across pipeline slots)
                NVCVTensorRequirements anchorReqs = {};
                if (anchorTensors.allocate(pvaAlloc, sc, numAnchorsPerScale, &anchorReqs) != 0)
                {
                    cleanup();
                    return -1;
                }
                // Create fused transpose+decode operator for this scale
                YOLOV5_NVCV_CHECK(pvaYoloV5DecodeCreate(
                    &pvaDecodeOps[sc], &predReqs, &anchorReqs, &detTensors.decBoxesReqs, &detTensors.decScoresReqs,
                    &detTensors.decClassesReqs, &detTensors.decCountReqs, numClasses, confidenceThreshold, modelHeight,
                    modelWidth, PVA_YOLOV5_DECODE_INPUT_CHW));
                printf(
                    "    Scale s%d: CHW [255, %d, %d] → fused transpose+decode (%d anchors) [zero-copy TRT "
                    "output]\n",
                    stride, gridH, gridW, numAnchorsPerScale);
            }

            // Initialize per-scale anchors (YOLOv5 default anchor sizes from model config)
            {
                float anchorSizes[][6] = {
                    {10, 13, 16, 30, 33, 23},     // stride 8
                    {30, 61, 62, 45, 59, 119},    // stride 16
                    {116, 90, 156, 198, 373, 326} // stride 32
                };
                for (int sc = 0; sc < NUM_YOLO_SCALES; sc++)
                {
                    int stride = YOLO_STRIDES[sc];
                    int gridH  = modelHeight / stride;
                    int gridW  = modelWidth / stride;
                    int idx    = 0;
                    for (int y = 0; y < gridH; y++)
                        for (int x = 0; x < gridW; x++)
                            for (int a = 0; a < 3; a++)
                            {
                                anchorTensors.anchorHost[sc][idx++] = static_cast<float>(x);
                                anchorTensors.anchorHost[sc][idx++] = static_cast<float>(y);
                                anchorTensors.anchorHost[sc][idx++] = anchorSizes[sc][a * 2 + 0];
                                anchorTensors.anchorHost[sc][idx++] = anchorSizes[sc][a * 2 + 1];
                                anchorTensors.anchorHost[sc][idx++] = static_cast<float>(stride);
                            }
                }
                printf("  PVA anchors initialized (%d total)\n", totalNumAnchors);
            }

            printf("  PVA fused transpose+decode ready (%d total anchors, maxOut=%d)\n", totalNumAnchors,
                   maxDecodeBoxes);

            // Per-slot NMS output tensors (NMS reads decode output directly - zero-copy)
            {
                if (detTensors.allocateNms(pvaAlloc, maxNmsBoxes) != 0)
                    return -1;

                YOLOV5_NVCV_CHECK(pvaBatchedNMSCreate(
                    &pvaNmsOp, &detTensors.decBoxesReqs, &detTensors.decScoresReqs, &detTensors.decClassesReqs,
                    &detTensors.decCountReqs, &detTensors.nmsBoxesReqs, &detTensors.nmsScoresReqs,
                    &detTensors.nmsClassesReqs, &detTensors.nmsCountReqs, nmsThreshold));
                printf("  PVA NMS: %d slots, iou=%.2f, maxOut=%d (decode→NMS zero-copy)\n", NUM_PIPE, nmsThreshold,
                       maxNmsBoxes);
            }
        }
    }
    printf("\n");

    // ========================================================================
    // Stage 3: PVA Resize + NhwcToNchwLetterboxNormalize Initialization
    // ========================================================================

    // NhwcToNchwLetterboxNormalize OUTPUT resources are model-sized (CHW) and created once.
    // INPUT resources and operators depend on source image dimensions
    // and are recreated when the image size changes (eval mode).

    nvcvTensorLayoutMake("HWC", &hwcLayout);
    NVCVTensorLayout chwLayout;
    nvcvTensorLayoutMake("CHW", &chwLayout);
    {
        int64_t lbOutShape[] = {channels, modelHeight, modelWidth};
        YOLOV5_NVCV_CHECK(
            nvcvTensorCalcRequirementsPva(3, lbOutShape, NVCV_DATA_TYPE_F32, chwLayout, 0, 0, &lbOutReqs));
    }

    // Destroy and recreate the shared PVA resize + letterbox operators
    // for the current srcWidth/srcHeight/lbW/lbH.  The imageRowStride
    // parameter must come from an actual preTensors.outImage (avoids creating a
    // throwaway PVA image just to query its stride).
    recreatePreprocessOps = [this](int64_t imageRowStride) {
        if (pvaPreprocessOp)
        {
            nvcvOperatorDestroy(pvaPreprocessOp);
            pvaPreprocessOp = nullptr;
        }
        if (pvaResizeOp)
        {
            nvcvOperatorDestroy(pvaResizeOp);
            pvaResizeOp = nullptr;
        }

        NVCVImageRequirements imgInReqs, imgOutReqs;
        NVCV_CHECK_ERROR(nvcvImageCalcRequirementsPva(srcWidth, srcHeight, NVCV_IMAGE_FORMAT_RGB8, 0, 0, &imgInReqs));
        NVCV_CHECK_ERROR(nvcvImageCalcRequirementsPva(lbW, lbH, NVCV_IMAGE_FORMAT_RGB8, 0, 0, &imgOutReqs));
        NVCV_CHECK_ERROR(pvaImageResizeCreate(&pvaResizeOp, &imgInReqs, &imgOutReqs, PVA_INTERPOLATION_LINEAR));

        NVCVTensorRequirements lbInReqs;
        int64_t lbInShape[] = {lbH, lbW, channels};
        NVCV_CHECK_ERROR(nvcvTensorCalcRequirementsPva(3, lbInShape, NVCV_DATA_TYPE_U8, hwcLayout, 0, 0, &lbInReqs));
        lbInReqs.strides[0] = imageRowStride;

        NVCV_CHECK_ERROR(pvaNhwcToNchwLetterboxNormalizeCreate(&pvaPreprocessOp, &lbInReqs, &lbOutReqs, 1.0f / 255.0f,
                                                               114.0f / 255.0f, PVA_NHWC_TO_NCHW_PADDING_PREFILLED));
        opSrcW = srcWidth;
        opSrcH = srcHeight;
    };

    // Destroy and recreate one slot's input-side PVA resources
    // (preTensors.inImage, preTensors.outImage, preTensors.lbIn + host pointers).
    recreateSlotResources = [this](int s) {
        if (preTensors.recreateSlot(pvaAlloc, s, srcWidth, srcHeight, lbW, lbH, channels) != 0)
        {
            fprintf(stderr, "ERROR: failed to recreate preprocess slot %d\n", s);
            return -1;
        }
        return 0;
    };

    {
        printf("Stage 3: PVA Resize + NhwcToNchwLetterboxNormalize Initialization (%d slots)\n", NUM_PIPE);

        // Create slot 0 first so we can query its output image row stride,
        // then pass that to recreatePreprocessOps (avoids a temp PVA image).
        if (recreateSlotResources(0) != 0)
            return -1;
        {
            int64_t outRowStride = 0;
            if (preTensors.outRowStride(0, &outRowStride) != 0)
                return -1;
            recreatePreprocessOps(outRowStride);
        }

        // PVA writes its letterbox output straight into the pinned TensorRT
        // input, so there is no host-to-device copy between the stages.
        if (interop.wrapLetterboxOutput(lbOutReqs, modelHeight, modelWidth) != 0)
        {
            fprintf(stderr, "  Failed to import PVA preprocess output into TensorRT input memory\n");
            cleanup();
            return -1;
        }
        printf("  Zero-copy: PVA NhwcToNchwLetterboxNormalize output -> TensorRT input (CUDA-PVA interop)\n");

        for (int s = 1; s < NUM_PIPE; s++)
            if (recreateSlotResources(s) != 0)
                return -1;

        // The PVA+GPU pipeline owns these persistent buffers. Initialize every slot once before
        // processing, then preserve the padding while PVA overwrites only the
        // resized image region. This keeps host work outside measured iterations.
        const size_t modelElements = static_cast<size_t>(channels) * modelHeight * modelWidth;
        for (int s = 0; s < NUM_PIPE; ++s)
        {
            std::fill_n(reinterpret_cast<float *>(interop.trtInput[s]), modelElements, paddingValue);
            paddedLbW[s] = lbW;
            paddedLbH[s] = lbH;
        }
        printf("  Letterbox padding prefilled once for all %d persistent pipeline slots\n", NUM_PIPE);

        printf("  PVA resize: %dx%d → %dx%d, NhwcToNchwLetterboxNormalize: %dx%dx%d NCHW float32\n", srcWidth,
               srcHeight, lbW, lbH, channels, modelHeight, modelWidth);
        printf("  PVA output -> TensorRT input directly (zero-copy NCHW, no H2D)\n");
    }

    // Zero-copy wrapping: wrap preloaded pinned buffers directly as PVA images
    // to skip the per-iteration memcpy into preTensors.inImage.  Only possible when the
    // PVA image row stride matches the natural stride (srcWidth * 3).
    if (usePreloadedBins && !preloadedBins.empty())
    {
        int naturalStride = srcWidth * 3;
        if (preTensors.inRowStride[0] == naturalStride)
        {
            printf("  PVA input stride (%d) matches natural stride — enabling zero-copy wrapping\n",
                   preTensors.inRowStride[0]);
            NVCVImageRequirements wrapReqs;
            YOLOV5_NVCV_CHECK(
                nvcvImageCalcRequirementsPva(srcWidth, srcHeight, NVCV_IMAGE_FORMAT_RGB8, 0, 0, &wrapReqs));

            wrappedPvaInImages.resize(preloadedBins.size(), nullptr);
            for (size_t bi = 0; bi < preloadedBins.size(); bi++)
            {
                const auto &p = preloadedBins[bi];
                if (p.w != srcWidth || p.h != srcHeight)
                    continue;
                NVCVStatus wrapSt = nvcvImageWrapCudaBuf(&wrapReqs, p.ptr, &wrappedPvaInImages[bi]);
                if (wrapSt != NVCV_SUCCESS)
                {
                    fprintf(stderr, "    [%zu] nvcvImageWrapCudaBuf failed (%d)\n", bi, static_cast<int>(wrapSt));
                    wrappedPvaInImages[bi] = nullptr;
                }
            }
            size_t wrapped = 0;
            for (auto h : wrappedPvaInImages)
                if (h)
                    ++wrapped;
            printf("  Zero-copy wrapped %zu / %zu preloaded images\n", wrapped, preloadedBins.size());
        }
        else
        {
            printf("  PVA input stride (%d) != natural stride (%d) — using row-by-row memcpy\n",
                   preTensors.inRowStride[0], naturalStride);
        }
    }

    // A compute-only benchmark must not fall back to the row-copy staging path
    // inside the measured loop. Treat zero-copy wrapping as a benchmark
    // prerequisite so the reported iteration time contains pipeline work only.
    if (benchmarkMode)
    {
        bool allInputsWrapped = wrappedPvaInImages.size() == preloadedBins.size() && !wrappedPvaInImages.empty();
        for (NVCVImageHandle image : wrappedPvaInImages) allInputsWrapped = allInputsWrapped && image != nullptr;
        if (!allInputsWrapped)
        {
            fprintf(stderr,
                    "Benchmark requires zero-copy wrapping of every preloaded input; refusing to time input staging\n");
            cleanup();
            return 1;
        }
    }

    {
        auto initEndTime = std::chrono::high_resolution_clock::now();
        float initTimeMs = std::chrono::duration<float, std::milli>(initEndTime - appStartTime).count();
        printf("Initialization complete: %.3f ms\n\n", initTimeMs);
    }

    // ========================================================================
    // 3-Stage Pipelined Processing Loop
    //
    // Each iteration processes 3 frames at different stages:
    //   preFrame  = iter       -> Pre-process (PVA resize + lb)
    //   infFrame  = iter - 1   -> Inference (TensorRT)
    //   postFrame = iter - 2   -> Post-process (PVA decode + NMS + build boxes)
    //
    // Phase A: PVA pre(preFrame) || GPU inference(infFrame)
    // Phase B: [H->D if needed](preFrame) || PVA decode+NMS(postFrame)
    // ========================================================================

    if (!trtContext)
    {
        fprintf(stderr, "ERROR: Pipelined mode requires a valid TRT engine.\n");
        cleanup();
        return -1;
    }

    if (useCudaGraph)
    {
        for (int s = 0; s < NUM_PIPE; ++s)
        {
            bool bindingsOk = trtContext->setTensorAddress(inputTensorName.c_str(), interop.trtInput[s]);
            for (int o = 0; o < numTrtOutputs; ++o)
                bindingsOk =
                    bindingsOk && trtContext->setTensorAddress(outputTensorNames[o].c_str(), interop.trtOutput[s][o]);
            if (!bindingsOk || !trtContext->enqueueV3(cudaStream))
            {
                fprintf(stderr, "TensorRT CUDA graph slot %d setup failed\n", s);
                cleanup();
                return 1;
            }
            YOLOV5_CUDA_CHECK(cudaStreamSynchronize(cudaStream));

            cudaGraph_t graph = nullptr;
            YOLOV5_CUDA_CHECK(cudaStreamBeginCapture(cudaStream, cudaStreamCaptureModeThreadLocal));
            if (!trtContext->enqueueV3(cudaStream))
            {
                cudaStreamEndCapture(cudaStream, &graph);
                if (graph)
                    cudaGraphDestroy(graph);
                fprintf(stderr, "TensorRT CUDA graph slot %d capture failed\n", s);
                cleanup();
                return 1;
            }
            YOLOV5_CUDA_CHECK(cudaStreamEndCapture(cudaStream, &graph));
            YOLOV5_CUDA_CHECK(cudaGraphInstantiate(&inferenceGraphExec[s], graph, nullptr, nullptr, 0));
            YOLOV5_CUDA_CHECK(cudaGraphDestroy(graph));
        }
        printf("CUDA graph: captured TensorRT inference for all %d pipeline slots\n", NUM_PIPE);
    }

    return 0;
}

// Load one frame into a slot's PVA input image and compute its letterbox
// geometry. Returns false when the frame should be skipped because its load
// failed. The slot is marked valid only after the frame has been loaded and
// the slot's PVA resources match its size. If those resources cannot be
// recreated, slotSetupFailed is set and the run ends.
bool PvaGpuPipeline::prepareFrameInput(int preFrame, int sp)
{
    bool frameUsable     = true;
    bool haveInput       = false;
    frameValid[sp]       = false;
    std::string &curPath = imagePathSlot[sp];
    int64_t &curId       = imageIdSlot[sp];
    LetterboxInfo &curLb = lbInfoSlot[sp];

    if (evalMode && !imageList.empty())
    {
        curPath   = imageList[preFrame];
        curId     = extractImageId(curPath);
        haveInput = true;
    }
    else if (benchmarkMode && !imageList.empty())
    {
        curPath   = imageList[0];
        curId     = extractImageId(curPath);
        haveInput = true;
    }
    else if (!imagePath.empty())
    {
        curPath   = imagePath;
        curId     = extractImageId(curPath);
        haveInput = true;
    }

    if (!haveInput)
        frameUsable = false;

    if (frameUsable)
    {
        int imageIndex           = evalMode ? preFrame : 0;
        slotImageIndex[sp]       = imageIndex;
        const uint8_t *rawPixels = nullptr;
        int imgW = 0, imgH = 0;
        bool freePixels = false;

        if (usePreloadedBins && imageIndex < static_cast<int>(preloadedBins.size()))
        {
            const auto &p = preloadedBins[imageIndex];
            rawPixels     = reinterpret_cast<const uint8_t *>(p.ptr);
            imgW          = p.w;
            imgH          = p.h;
        }
        else
        {
            const std::string &path = curPath;
            bool isBin              = path.size() >= 4u && path.compare(path.size() - 4, 4, ".bin") == 0;
            if (isBin)
            {
                FILE *f = fopen(path.c_str(), "rb");
                if (!f)
                {
                    frameValid[sp] = false;
                    frameUsable    = false;
                    inputLoadFailures++;
                }
                else
                {
                    int32_t bw = 0, bh = 0;
                    if (fread(&bw, sizeof(int32_t), 1, f) != 1 || fread(&bh, sizeof(int32_t), 1, f) != 1 || bw <= 0 ||
                        bh <= 0)
                    {
                        fclose(f);
                        frameValid[sp] = false;
                        frameUsable    = false;
                        inputLoadFailures++;
                    }
                    else
                    {
                        size_t sz = static_cast<size_t>(bw) * bh * 3u;
                        auto *buf = static_cast<uint8_t *>(malloc(sz));
                        if (!buf || fread(buf, 1, sz, f) != sz)
                        {
                            free(buf);
                            fclose(f);
                            frameValid[sp] = false;
                            frameUsable    = false;
                            inputLoadFailures++;
                        }
                        else
                        {
                            fclose(f);
                            rawPixels  = buf;
                            imgW       = bw;
                            imgH       = bh;
                            freePixels = true;
                        }
                    }
                }
            }
            else
            {
                rawPixels = yolov5_common::loadPngRgb(path.c_str(), &imgW, &imgH);
                if (!rawPixels)
                {
                    frameValid[sp] = false;
                    frameUsable    = false;
                    inputLoadFailures++;
                }
                else
                    freePixels = true;
            }
        }

        if (frameUsable && rawPixels)
        {
            // Dynamic sizing: recompute geometry and recreate PVA
            // operators/resources when image dimensions change.
            if (imgW != srcWidth || imgH != srcHeight)
            {
                srcWidth  = imgW;
                srcHeight = imgH;
                lbScale =
                    std::min(static_cast<float>(modelWidth) / srcWidth, static_cast<float>(modelHeight) / srcHeight);
                lbW     = static_cast<int>(std::round(srcWidth * lbScale));
                lbH     = static_cast<int>(std::round(srcHeight * lbScale));
                padTop  = (modelHeight - lbH) / 2;
                padLeft = (modelWidth - lbW) / 2;
                if (verbose)
                    printf("  [%d] Resize for %dx%d (lb %dx%d, pad %d/%d)\n", preFrame, imgW, imgH, lbW, lbH, padTop,
                           padLeft);
            }
            bool resourcesReady = true;
            if (imgW != preTensors.srcW[sp] || imgH != preTensors.srcH[sp])
                resourcesReady = recreateSlotResources(sp) == 0;
            if (resourcesReady && (srcWidth != opSrcW || srcHeight != opSrcH))
            {
                int64_t outRowStride = 0;
                resourcesReady       = preTensors.outRowStride(sp, &outRowStride) == 0;
                if (resourcesReady)
                    recreatePreprocessOps(outRowStride);
            }
            if (!resourcesReady)
            {
                fprintf(stderr, "ERROR: failed to prepare PVA resources for %dx%d input in slot %d\n", imgW, imgH, sp);
                if (freePixels)
                    yolov5_common::freeImage(const_cast<uint8_t *>(rawPixels));
                slotSetupFailed = true;
                return false;
            }

            // The letterbox operator writes only the resized image region, so
            // refill this slot's TensorRT input padding when the letterbox
            // geometry differs from the one it was filled for.
            if (paddedLbW[sp] != lbW || paddedLbH[sp] != lbH)
            {
                const size_t modelElements = static_cast<size_t>(channels) * modelHeight * modelWidth;
                std::fill_n(reinterpret_cast<float *>(interop.trtInput[sp]), modelElements, paddingValue);
                paddedLbW[sp] = lbW;
                paddedLbH[sp] = lbH;
            }

            bool zeroCopyInput =
                imageIndex < static_cast<int>(wrappedPvaInImages.size()) && wrappedPvaInImages[imageIndex] != nullptr;
            if (!zeroCopyInput)
            {
                int srcRowBytes = imgW * 3;
                for (int y = 0; y < imgH; y++)
                    memcpy(preTensors.inHostPtr[sp] + y * preTensors.inRowStride[sp], rawPixels + y * srcRowBytes,
                           srcRowBytes);
            }
            if (freePixels)
                yolov5_common::freeImage(const_cast<uint8_t *>(rawPixels));

            curLb.origW    = imgW;
            curLb.origH    = imgH;
            frameValid[sp] = true;
        }

        // Set per-frame letterbox info AFTER dimension check
        curLb.scaleX = lbScale;
        curLb.scaleY = lbScale;
        curLb.padX   = static_cast<float>(padLeft);
        curLb.padY   = static_cast<float>(padTop);
    }
    return frameUsable;
}

// Read one finished frame's detections back and report them. Kept out of the
// iteration loop so the loop shows the pipeline and nothing else.
int PvaGpuPipeline::collectFrameResults(const yolov5_pipeline::FramesInFlight &turn, bool isMeasuredPost,
                                        int pipelineFrames, int &numDecodedBoxes, int &numNmsBoxes)
{
    const int so        = turn.postSlot;
    const int postFrame = turn.postFrame;
    const bool doPost   = turn.postActive;

    if (doPost && frameValid[so] && totalNumAnchors > 0)
    {
        numDecodedBoxes = *detTensors.decCountHost[so];
        numNmsBoxes     = *detTensors.nmsCountHost[so];
        lastNumNmsBoxes = numNmsBoxes;
    }

    if (doPost && frameValid[so] && numNmsBoxes > 0)
    {
        const LetterboxInfo &lb = lbInfoSlot[so];
        int64_t imgId           = imageIdSlot[so];

        const float *leftArr   = detTensors.nmsBoxesHost[so];
        const float *topArr    = detTensors.nmsBoxesHost[so] + maxNmsBoxes;
        const float *rightArr  = detTensors.nmsBoxesHost[so] + 2 * maxNmsBoxes;
        const float *bottomArr = detTensors.nmsBoxesHost[so] + 3 * maxNmsBoxes;

        std::vector<Box> finalBoxes;
        finalBoxes.reserve(numNmsBoxes);
        for (int i = 0; i < numNmsBoxes; i++)
        {
            Box b;
            b.left        = leftArr[i];
            b.top         = topArr[i];
            b.right       = rightArr[i];
            b.bottom      = bottomArr[i];
            b.confidence  = detTensors.nmsScoresHost[so][i];
            b.class_label = static_cast<float>(detTensors.nmsClassesHost[so][i]);
            finalBoxes.push_back(b);
        }

        if (cfg.checkTopK > 0)
        {
            lastDetections.clear();
            lastDetections.reserve(finalBoxes.size());
            for (const Box &fb : finalBoxes)
            {
                yolov5_common::DetBox d;
                d.left    = (fb.left - lb.padX) / lb.scaleX;
                d.top     = (fb.top - lb.padY) / lb.scaleY;
                d.right   = (fb.right - lb.padX) / lb.scaleX;
                d.bottom  = (fb.bottom - lb.padY) / lb.scaleY;
                d.score   = fb.confidence;
                d.classId = static_cast<int>(fb.class_label);
                lastDetections.push_back(d);
            }
        }

        // Print detections (single-image or verbose)
        if ((!benchmarkMode && !evalMode) || verbose)
        {
            printf("  Detected objects (%d after NMS, frame %d):\n", numNmsBoxes, postFrame);
            int displayCount = std::min(static_cast<int>(finalBoxes.size()), 20);
            for (int i = 0; i < displayCount; i++)
            {
                const Box &b = finalBoxes[i];
                printf("  Detection %d: left=%.1f top=%.1f right=%.1f bottom=%.1f confidence=%.3f class=%d\n", i,
                       b.left, b.top, b.right, b.bottom, b.confidence, static_cast<int>(b.class_label));
            }

            printf("  Classes detected: ");
            std::set<int> uniqueClasses;
            for (const auto &b : finalBoxes) uniqueClasses.insert(static_cast<int>(b.class_label));
            for (int cls : uniqueClasses)
                if (cls >= 0 && cls < 80)
                    printf("%s ", COCO_CLASSES[cls]);
            printf("\n\n");
        }

        if (evalMode)
        {
            for (const auto &b : finalBoxes)
            {
                CocoDetection det;
                det.imageId    = imgId;
                int classIdx   = static_cast<int>(b.class_label);
                det.categoryId = (classIdx >= 0 && classIdx < 80) ? COCO_CATEGORY_IDS[classIdx] : classIdx + 1;
                float x1       = (b.left - lb.padX) / lb.scaleX;
                float y1       = (b.top - lb.padY) / lb.scaleY;
                float x2       = (b.right - lb.padX) / lb.scaleX;
                float y2       = (b.bottom - lb.padY) / lb.scaleY;
                x1             = std::max(0.f, std::min(x1, static_cast<float>(lb.origW)));
                y1             = std::max(0.f, std::min(y1, static_cast<float>(lb.origH)));
                x2             = std::max(0.f, std::min(x2, static_cast<float>(lb.origW)));
                y2             = std::max(0.f, std::min(y2, static_cast<float>(lb.origH)));
                det.x          = x1;
                det.y          = y1;
                det.width      = x2 - x1;
                det.height     = y2 - y1;
                if (det.width <= 0 || det.height <= 0)
                    continue;
                det.score = b.confidence;
                allDetections.push_back(det);
            }
        }
        if (!benchmarkMode || isMeasuredPost)
            totalDetections += static_cast<int>(finalBoxes.size());
    }

    if (doPost && frameValid[so] && ((postFrame + 1) % 100 == 0 || postFrame + 1 == pipelineFrames))
    {
        printf("[%d/%d] ID %ld: decoded=%d nms=%d\n", postFrame + 1, pipelineFrames, static_cast<long>(imageIdSlot[so]),
               numDecodedBoxes, numNmsBoxes);
    }

    return 0;
}

int PvaGpuPipeline::run()
{
    steadyStateFrames = 0;
    lastNumNmsBoxes   = -1;
    inputLoadFailures = 0;
    totalSteadyMs     = 0.F;
    pipelineTimings.clear();
    preprocessTimings.clear();
    inferenceTimings.clear();
    postprocessTimings.clear();

    {

        // Benchmark iterations are the fully-overlapped ones. Feed warmup plus
        // two trailing frames so neither the filling nor the draining
        // iterations enter the steady-state distribution.
        const int pipelineFrames = benchmarkMode ? warmupRuns + benchmarkIter + NUM_PIPE - 1 : totalIterations;
        printf("Pipeline: 3-stage, %d input frames\n\n", pipelineFrames);

        const int totalIter = yolov5_pipeline::iterationsFor(pipelineFrames);

        // One iteration advances three frames at once; see FramesInFlight.
        for (int iter = 0; iter < totalIter; iter++)
        {
            auto iterStartTime = std::chrono::high_resolution_clock::now();

            // VPU0: preprocess(N) || GPU: infer(N-1) || VPU1: postprocess(N-2)
            const yolov5_pipeline::FramesInFlight turn = yolov5_pipeline::FramesInFlight::at(iter, pipelineFrames);

            const int preFrame  = turn.preFrame;
            const int infFrame  = turn.infFrame;
            const int postFrame = turn.postFrame;

            bool doPre        = turn.preActive;
            const bool doInf  = turn.infActive;
            const bool doPost = turn.postActive;

            // Negative while the pipeline fills; only ever read behind the
            // matching doInf / doPost, which short-circuits.
            const int sp = turn.preSlot;
            const int si = turn.infSlot;
            const int so = turn.postSlot;

            bool isWarmup       = benchmarkMode && postFrame < warmupRuns;
            bool isMeasuredPost = !isWarmup && doPost && (!benchmarkMode || (doPre && doInf));

            // CPU: load frame preFrame into slot sp
            if (doPre)
                doPre = prepareFrameInput(preFrame, sp);
            if (slotSetupFailed)
                return 1;
            // Submit all 3 stages concurrently (VPU0 || GPU || VPU1)
            const bool preprocessSubmitted  = doPre && frameValid[sp];
            const bool inferenceSubmitted   = doInf && frameValid[si];
            const bool postprocessSubmitted = doPost && frameValid[so] && totalNumAnchors > 0;

            if (submitPvaPreprocess(preprocessSubmitted, sp) != 0)
                return 1;
            if (submitGpuInference(inferenceSubmitted, si) != 0)
                return 1;
            if (submitPvaPostprocess(postprocessSubmitted, so) != 0)
                return 1;

            int numDecodedBoxes = 0, numNmsBoxes = 0;
            if (awaitIteration(preprocessSubmitted, inferenceSubmitted, postprocessSubmitted) != 0)
                return 1;

            // Stop compute-only timing immediately after all three pipeline
            // stages have completed. CPU result materialization, evaluation,
            // printing, and file output remain outside the measured interval.
            const bool measuredStagesValid =
                doPost && frameValid[so] && (!benchmarkMode || (doPre && doInf && frameValid[sp] && frameValid[si]));
            if (isMeasuredPost && measuredStagesValid)
            {
                auto iterEndTime       = std::chrono::high_resolution_clock::now();
                float iterPipelineTime = std::chrono::duration<float, std::milli>(iterEndTime - iterStartTime).count();
                pipelineTimings.push_back(iterPipelineTime);
                totalSteadyMs += iterPipelineTime;
                steadyStateFrames++;

                if (benchmarkMode)
                {
                    float stageMs = 0.F;
                    YOLOV5_CUDA_CHECK(cudaEventElapsedTime(&stageMs, preprocessStart, preprocessStop));
                    preprocessTimings.push_back(stageMs);
                    YOLOV5_CUDA_CHECK(cudaEventElapsedTime(&stageMs, inferenceStart, inferenceStop));
                    inferenceTimings.push_back(stageMs);
                    YOLOV5_CUDA_CHECK(cudaEventElapsedTime(&stageMs, postprocessStart, postprocessStop));
                    postprocessTimings.push_back(stageMs);
                }
            }

            // Collect and report results for postFrame
            if (collectFrameResults(turn, isMeasuredPost, pipelineFrames, numDecodedBoxes, numNmsBoxes) != 0)
                return 1;
        } // end pipeline loop
    }

    return 0;
}

int PvaGpuPipeline::report()
{

    // ========================================================================
    // Evaluation/Benchmark Results
    // ========================================================================

    if (steadyStateFrames == 0)
    {
        fprintf(stderr, "ERROR: pipeline completed without processing any valid measured frames.\n");
        cleanup();
        return 1;
    }

    if (inputLoadFailures > 0)
    {
        fprintf(stderr, "ERROR: %d input frame(s) failed to load; results are incomplete.\n", inputLoadFailures);
        result = 1;
    }

    if (benchmarkMode && steadyStateFrames != benchmarkIter)
    {
        fprintf(stderr, "ERROR: benchmark requested %d valid fully-overlapped iterations but measured %d.\n",
                benchmarkIter, steadyStateFrames);
        result = 1;
    }

    if (cfg.expectedDetections >= 0 && lastNumNmsBoxes != cfg.expectedDetections)
    {
        fprintf(stderr, "ERROR: expected %d detections, got %d\n", cfg.expectedDetections, lastNumNmsBoxes);
        result = 1;
    }

    if (cfg.checkTopK > 0 && !yolov5_common::checkTopKAgainstReference(lastDetections, cfg.checkTopK, cfg.checkTopKIoU))
    {
        result = 1;
    }

    TimingStats pipeStats{}, preprocessStats{}, inferenceStats{}, postprocessStats{};
    if (steadyStateFrames > 0)
    {
        pipeStats = TimingStats::compute(pipelineTimings);
        if (benchmarkMode)
        {
            preprocessStats  = TimingStats::compute(preprocessTimings);
            inferenceStats   = TimingStats::compute(inferenceTimings);
            postprocessStats = TimingStats::compute(postprocessTimings);
        }
    }

    if (evalMode || benchmarkMode)
    {
        printf("\n");
        printf("3-stage pipeline results\n");

        if (evalMode)
        {
            printf("  Images processed: %d\n", steadyStateFrames);
            printf("  Total detections: %d\n", totalDetections);
        }
        else
        {
            printf("  Benchmark iterations: %d (warmup: %d)\n", benchmarkIter, warmupRuns);
        }

        if (steadyStateFrames > 0)
        {
            float fps = pipeStats.avg > 0 ? 1000.0f / pipeStats.avg : 0.0f;
            printf("  Pipeline timing: frames=%d total=%.1f ms\n", steadyStateFrames, totalSteadyMs);
            printf("  Pipeline latency (ms): min=%.3f max=%.3f avg=%.3f p50=%.3f p95=%.3f p99=%.3f std=%.3f\n",
                   pipeStats.min, pipeStats.max, pipeStats.avg, pipeStats.p50, pipeStats.p95, pipeStats.p99,
                   pipeStats.stddev);
            if (benchmarkMode)
            {
                printf("  Stage service time (ms, stages overlap):\n");
                printf("    PVA preprocessing: %.3f avg %.3f p95\n", preprocessStats.avg, preprocessStats.p95);
                printf("    GPU inference: %.3f avg %.3f p95\n", inferenceStats.avg, inferenceStats.p95);
                printf("    PVA postprocessing: %.3f avg %.3f p95\n", postprocessStats.avg, postprocessStats.p95);
            }
            printf("  Average FPS: %.2f\n", fps);
            printf("  Pipeline structure: 3 frames in flight, dual VPU\n");
            printf("    VPU0: PVA pre(N+2) || GPU inf(N+1) || VPU1: post(N)\n");
            printf("    GPU-to-PVA transpose+decode: zero-copy interop (no D2H copy)\n");
            printf("    PVA-to-TRT input: zero-copy (cudaMallocHost, no H2D)\n");
            printf("    Decode-to-NMS: zero-copy (shared PVA tensors)\n");
        }
        printf("\n");

        // Write COCO JSON if requested
        if (!cocoOutput.empty() && !allDetections.empty())
        {
            if (writeCocoJson(allDetections, cocoOutput))
            {
                printf("COCO predictions saved to: %s\n", cocoOutput.c_str());
                printf("  Total detections: %zu\n\n", allDetections.size());
            }
        }

        // Write summary JSON only when explicitly requested.
        if (steadyStateFrames > 0 && !cfg.summaryJsonPath.empty())
        {
            std::string device             = useDLA ? "dla" + std::to_string(dlaCore) : "gpu";
            std::string precision          = useINT8 ? "int8" : (useFP16 ? "fp16" : "fp32");
            std::string mode               = evalMode ? "eval" : "benchmark";
            const std::string &summaryPath = cfg.summaryJsonPath;
            std::ofstream sfs(summaryPath);
            if (sfs.is_open())
            {
                float fps               = pipeStats.avg > 0 ? 1000.0f / pipeStats.avg : 0.0f;
                auto writeTimingSamples = [&](const char *name, const std::vector<float> &samples) {
                    sfs << "    \"" << name << "\": [";
                    for (size_t i = 0; i < samples.size(); ++i)
                    {
                        if (i > 0)
                            sfs << ", ";
                        sfs << static_cast<uint64_t>(std::llround(samples[i] * 1000000.0));
                    }
                    sfs << "]";
                };

                sfs << "{\n";
                sfs << "  \"config\": {\n";
                sfs << "    \"pipeline\": \"3-stage-dual-vpu\",\n";
                sfs << "    \"device\": \"" << device << "\",\n";
                sfs << "    \"precision\": \"" << precision << "\",\n";
                sfs << "    \"input_image\": \"" << imagePath << "\",\n";
                sfs << "    \"input_width\": " << srcWidth << ",\n";
                sfs << "    \"input_height\": " << srcHeight << ",\n";
                sfs << "    \"confidence_threshold\": " << confidenceThreshold << ",\n";
                sfs << "    \"nms_iou_threshold\": " << nmsThreshold << ",\n";
                sfs << "    \"warmup_iterations\": " << warmupRuns << ",\n";
                sfs << "    \"measured_iterations\": " << steadyStateFrames << ",\n";
                sfs << "    \"cuda_graph\": " << (useCudaGraph ? "true" : "false") << ",\n";
                sfs << "    \"model\": \"" << onnxPath << "\",\n";
                sfs << "    \"engine\": \"" << enginePath << "\"\n";
                sfs << "  },\n";
                sfs << "  \"results\": {\n";
                sfs << "    \"images_processed\": " << steadyStateFrames << ",\n";
                sfs << "    \"total_detections\": " << totalDetections << ",\n";
                sfs << "    \"detections_per_image\": "
                    << (steadyStateFrames > 0 ? static_cast<float>(totalDetections) / steadyStateFrames : 0.0f)
                    << ",\n";
                sfs << "    \"avg_fps\": " << fps << ",\n";
                sfs << "    \"wall_clock_total_ms\": " << totalSteadyMs << ",\n";
                sfs << "    \"wall_clock_avg_ms\": " << pipeStats.avg << ",\n";
                sfs << "    \"wall_clock_fps\": " << fps << "\n";
                sfs << "  },\n";
                sfs << "  \"timing_ms\": {\n";
                writeTimingJson(sfs, "pipeline", pipeStats);
                if (benchmarkMode)
                {
                    sfs << ",\n";
                    writeTimingJson(sfs, "preprocessing", preprocessStats);
                    sfs << ",\n";
                    writeTimingJson(sfs, "inference", inferenceStats);
                    sfs << ",\n";
                    writeTimingJson(sfs, "postprocessing", postprocessStats);
                }
                sfs << "\n";
                sfs << "  }";
                if (benchmarkMode)
                {
                    sfs << ",\n";
                    sfs << "  \"stage_samples_ns\": {\n";
                    writeTimingSamples("preprocessing", preprocessTimings);
                    sfs << ",\n";
                    writeTimingSamples("inference", inferenceTimings);
                    sfs << ",\n";
                    writeTimingSamples("postprocessing", postprocessTimings);
                    sfs << "\n";
                    sfs << "  }";
                }
                sfs << "\n";
                sfs << "}\n";
                sfs.close();
                printf("Summary saved to: %s\n", summaryPath.c_str());
            }
        }
    }

    if (!evalMode && !benchmarkMode)
    {
        printf("Note: Single-image mode uses the pipeline for 1 frame (no overlap benefit).\n");
    }

    printf("========================================\n");
    printf("  3-Stage Pipeline YOLOv5 %s!\n", result == 0 ? "PASSED" : "FAILED");
    printf("========================================\n\n");

    return result;
}

#endif // CUPVA_HAS_CUDA_STREAM_SUPPORT == 1

int main(int argc, char *argv[])
{
#if CUPVA_HAS_CUDA_STREAM_SUPPORT != 1
    if (argc == 2 && std::string(argv[1]) == "--check-pva-driver")
    {
        printf("PVA SDK does not support CUDA stream submission (requires cuPVA >= 2.7.0)\n");
        return 77;
    }

    fprintf(stderr, "The PVA+GPU pipeline requires a cuPVA SDK with CUDA-stream submission (cuPVA >= 2.7.0).\n");
    return 1;
#else
    if (argc == 2 && std::string(argv[1]) == "--check-pva-driver")
    {
        uint32_t pvaDriverVersion = 0;
        if (CupvaGetDriverVersion(&pvaDriverVersion) != CUPVA_ERROR_NONE)
        {
            fprintf(stderr, "Failed to query the PVA driver version\n");
            return 1;
        }
        printf("PVA driver version: %u (CUDA stream submission requires >= 2007)\n", pvaDriverVersion);
        return pvaDriverVersion >= 2007 ? 0 : 77;
    }

    // PVA resource allocation reports failures by throwing, so catch here and
    // let the pipeline destructor release whatever was already acquired.
    try
    {
        PvaGpuPipeline pipeline;
        int status = pipeline.initialize(argc, argv);
        if (status != 0 || !pipeline.shouldRun())
            return status;
        status = pipeline.run();
        if (status != 0)
            return status;
        return pipeline.report();
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
#endif
}
