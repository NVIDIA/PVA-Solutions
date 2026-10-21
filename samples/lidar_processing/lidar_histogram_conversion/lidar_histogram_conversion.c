/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ErrorCheckMacros.h"
#include "ref/debug_options.h"
#include "ref/lidar_histogram_conversion_ref.h"

#include <OpLidarHistogramConversion.h>
#include <PvaAllocator.h>
#include <cupva_host.h>
#include <getopt.h>
#include <limits.h>
#include <nvcv/Fwd.h>
#include <nvcv/Tensor.h>
#include <nvcv/alloc/Allocator.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * @brief Reads histogram tensor metadata from a file
 *
 * Reads tensor shape (width, height, channels, data_type_size), histogram specification,
 * and histogram parameters from the file header.
 *
 * On return, the file pointer is positioned at the start of the tensor data.
 *
 * @param fp File pointer positioned at the start of metadata
 * @param width Output parameter for tensor width
 * @param height Output parameter for tensor height
 * @param channels Output parameter for tensor channels
 * @param dataTypeSize Output parameter for data type size in bytes
 * @param histogramSpec Pointer to histogram spec structure to populate
 * @param histogramParams Pointer to histogram params structure to populate
 * @return true on success, false on error
 */
static bool readHistogramTensorMetadataFromFile(FILE *fp, int32_t *width, int32_t *height, int32_t *channels,
                                                int32_t *dataTypeSize, PvaLidarHistogramSpec *histogramSpec,
                                                PvaLidarHistogramParams *histogramParams)
{
    // Read header: width, height, channels, data_type_size
    int32_t header[4];
    if (fread(header, sizeof(int32_t), 4, fp) != 4)
    {
        printf("Failed to read header from file\n");
        return false;
    }
    *width        = header[0];
    *height       = header[1];
    *channels     = header[2];
    *dataTypeSize = header[3];

    // Read histogram specification
    int32_t histSpec[6];
    if (fread(histSpec, sizeof(int32_t), 6, fp) != 6)
    {
        printf("Failed to read histogram spec from file\n");
        return false;
    }

    histogramSpec->packingType     = histSpec[0];
    histogramSpec->numHistPerPixel = histSpec[1];
    histogramSpec->numBinsPerHist  = histSpec[2];
    histogramSpec->numPeaksPerHist = histSpec[3];
    histogramSpec->pixelHeaderSize = histSpec[4];
    histogramSpec->histHeaderSize  = histSpec[5];

    // Read histogram parameters
    float params[4];
    if (fread(params, sizeof(float), 4, fp) != 4)
    {
        printf("Failed to read histogram params from file\n");
        return false;
    }

    histogramParams->offsetNs     = params[0];
    histogramParams->binSizeNs    = params[1];
    histogramParams->rangeScale   = params[2];
    histogramParams->maxIntensity = params[3];

    return true;
}

/**
 * @brief Dumps range tensor data to a binary file
 *
 * The output file contains:
 * - 3 int32_t values: width, height, channels
 * - Followed by raw float32 tensor data
 */
static bool dumpRangeTensor(NVCVTensorHandle tensorHandle, const char *filename)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    int32_t height   = tensorData.shape[1];
    int32_t width    = tensorData.shape[2];
    int32_t channels = tensorData.shape[3];

    // Verify data type is F32
    if (tensorData.dtype != NVCV_DATA_TYPE_F32)
    {
        printf("Expected F32 data type for range tensor dump\n");
        return false;
    }

    // Get host pointer
    void *tensorPtr_h = NULL;
    CupvaMemGetHostPointer(&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    if (!tensorPtr_h)
    {
        printf("Failed to get host pointer for tensor\n");
        return false;
    }

    // Open file for writing
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Failed to open file %s for writing\n", filename);
        return false;
    }

    // Write header: width, height, channels
    int32_t header[3] = {width, height, channels};
    if (fwrite(header, sizeof(int32_t), 3, fp) != 3)
    {
        printf("Failed to write header to file\n");
        fclose(fp);
        return false;
    }

    // Write tensor data
    int32_t linePitch = tensorData.buffer.strided.strides[1];
    uint8_t *dataPtr  = (uint8_t *)tensorPtr_h;

    for (int32_t y = 0; y < height; y++)
    {
        int32_t rowSize = width * channels * sizeof(float);
        if (fwrite(dataPtr + y * linePitch, 1, rowSize, fp) != (size_t)rowSize)
        {
            printf("Failed to write tensor data to file\n");
            fclose(fp);
            return false;
        }
    }

    if (fclose(fp) != 0)
    {
        printf("Failed to close file %s\n", filename);
        return false;
    }

    printf("Range tensor dumped to %s (shape: [%d, %d, %d])\n", filename, height, width, channels);
    return true;
}

/**
 * @brief Loads histogram tensor data from a binary file
 *
 * The input file must contain:
 * - 4 int32_t values: width, height, channels, data_type_size
 * - 6 int32_t values: histogram spec (packingType, numHistPerPixel, numBinsPerHist,
 *                     numPeaksPerHist, pixelHeaderSize, histHeaderSize)
 * - 4 float32 values: histogram params (offsetNs, binSizeNs, rangeScale, maxIntensity)
 * - Followed by raw tensor data
 *
 * @return true on success, false on error
 */
static bool loadHistogramTensor(NVCVTensorHandle tensorHandle, const char *filename,
                                PvaLidarHistogramSpec *histogramSpec, PvaLidarHistogramParams *histogramParams)
{
    // Open file for reading
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Failed to open file %s for reading\n", filename);
        return false;
    }

    // Read tensor metadata and histogram specification
    int32_t fileWidth, fileHeight, fileChannels, fileDataTypeSize;
    if (!readHistogramTensorMetadataFromFile(fp, &fileWidth, &fileHeight, &fileChannels, &fileDataTypeSize,
                                             histogramSpec, histogramParams))
    {
        fclose(fp);
        return false;
    }

    // Verify tensor matches file dimensions
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    int32_t height   = tensorData.shape[1];
    int32_t width    = tensorData.shape[2];
    int32_t channels = tensorData.shape[3];

    if (height != fileHeight || width != fileWidth || channels != fileChannels)
    {
        printf("Tensor shape mismatch: file=[%d,%d,%d], tensor=[%d,%d,%d]\n", fileHeight, fileWidth, fileChannels,
               height, width, channels);
        fclose(fp);
        return false;
    }

    // Determine expected data type size
    int32_t expectedDataTypeSize = 0;
    if (tensorData.dtype == NVCV_DATA_TYPE_U16)
    {
        expectedDataTypeSize = 2;
    }
    else if (tensorData.dtype == NVCV_DATA_TYPE_U8)
    {
        expectedDataTypeSize = 1;
    }

    if (expectedDataTypeSize != fileDataTypeSize)
    {
        printf("Data type size mismatch: file=%d bytes, tensor=%d bytes\n", fileDataTypeSize, expectedDataTypeSize);
        fclose(fp);
        return false;
    }

    // Get host pointer
    void *tensorPtr_h = NULL;
    CupvaMemGetHostPointer(&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    if (!tensorPtr_h)
    {
        printf("Failed to get host pointer for tensor\n");
        fclose(fp);
        return false;
    }

    // Read tensor data
    int32_t linePitch = tensorData.buffer.strided.strides[1];
    uint8_t *dataPtr  = (uint8_t *)tensorPtr_h;

    for (int32_t y = 0; y < height; y++)
    {
        int32_t rowSize = width * channels * fileDataTypeSize;
        if (fread(dataPtr + y * linePitch, 1, rowSize, fp) != (size_t)rowSize)
        {
            printf("Failed to read tensor data from file\n");
            fclose(fp);
            return false;
        }
    }

    if (fclose(fp) != 0)
    {
        printf("Failed to close file %s\n", filename);
        return false;
    }
    printf("Histogram tensor loaded from %s (shape: [%d, %d, %d], dtype_size: %d bytes)\n", filename, height, width,
           channels, fileDataTypeSize);
    printf("  Histogram spec: packingType=%d, numHistPerPixel=%d, numBinsPerHist=%d\n", histogramSpec->packingType,
           histogramSpec->numHistPerPixel, histogramSpec->numBinsPerHist);
    printf("                  numPeaksPerHist=%d, pixelHeaderSize=%d, histHeaderSize=%d\n",
           histogramSpec->numPeaksPerHist, histogramSpec->pixelHeaderSize, histogramSpec->histHeaderSize);
    return true;
}

/**
 * @brief Dumps histogram tensor data to a binary file
 *
 * The output file contains:
 * - 4 int32_t values: width, height, channels, data_type_size
 * - 6 int32_t values: histogram spec (packingType, numHistPerPixel, numBinsPerHist,
 *                     numPeaksPerHist, pixelHeaderSize, histHeaderSize)
 * - 4 float32 values: histogram params (offsetNs, binSizeNs, rangeScale, maxIntensity)
 * - Followed by raw tensor data
 */
static bool dumpHistogramTensor(NVCVTensorHandle tensorHandle, const char *filename,
                                const PvaLidarHistogramSpec *histogramSpec,
                                const PvaLidarHistogramParams *histogramParams)
{
    NVCVTensorData tensorData;
    nvcvTensorExportData(tensorHandle, &tensorData);

    int32_t height   = tensorData.shape[1];
    int32_t width    = tensorData.shape[2];
    int32_t channels = tensorData.shape[3];

    // Determine data type size
    int32_t dataTypeSize = 0;
    if (tensorData.dtype == NVCV_DATA_TYPE_U16)
    {
        dataTypeSize = 2;
    }
    else if (tensorData.dtype == NVCV_DATA_TYPE_U8)
    {
        dataTypeSize = 1;
    }
    else
    {
        printf("Unsupported data type for histogram dump\n");
        return false;
    }

    // Get host pointer
    void *tensorPtr_h = NULL;
    CupvaMemGetHostPointer(&tensorPtr_h, (void *)tensorData.buffer.strided.basePtr);

    if (!tensorPtr_h)
    {
        printf("Failed to get host pointer for tensor\n");
        return false;
    }

    // Open file for writing
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Failed to open file %s for writing\n", filename);
        return false;
    }

    // Write header: width, height, channels, data_type_size
    int32_t header[4] = {width, height, channels, dataTypeSize};
    if (fwrite(header, sizeof(int32_t), 4, fp) != 4)
    {
        printf("Failed to write header to file\n");
        fclose(fp);
        return false;
    }

    // Write histogram specification metadata
    int32_t histSpec[6] = {histogramSpec->packingType,     histogramSpec->numHistPerPixel,
                           histogramSpec->numBinsPerHist,  histogramSpec->numPeaksPerHist,
                           histogramSpec->pixelHeaderSize, histogramSpec->histHeaderSize};
    if (fwrite(histSpec, sizeof(int32_t), 6, fp) != 6)
    {
        printf("Failed to write histogram spec to file\n");
        fclose(fp);
        return false;
    }

    // Write histogram parameters
    float params[4] = {histogramParams->offsetNs, histogramParams->binSizeNs, histogramParams->rangeScale,
                       histogramParams->maxIntensity};
    if (fwrite(params, sizeof(float), 4, fp) != 4)
    {
        printf("Failed to write histogram params to file\n");
        fclose(fp);
        return false;
    }

    // Write tensor data
    int32_t linePitch = tensorData.buffer.strided.strides[1];
    uint8_t *dataPtr  = (uint8_t *)tensorPtr_h;

    for (int32_t y = 0; y < height; y++)
    {
        int32_t rowSize = width * channels * dataTypeSize;
        if (fwrite(dataPtr + y * linePitch, 1, rowSize, fp) != (size_t)rowSize)
        {
            printf("Failed to write tensor data to file\n");
            fclose(fp);
            return false;
        }
    }

    if (fclose(fp) != 0)
    {
        printf("Failed to close file %s\n", filename);
        return false;
    }
    printf("Histogram tensor dumped to %s (shape: [%d, %d, %d], dtype_size: %d bytes)\n", filename, height, width,
           channels, dataTypeSize);
    return true;
}

static bool parseInt32Option(const char *optname, const char *s, int32_t *out)
{
    char *end = NULL;
    long v    = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v > INT32_MAX || v < INT32_MIN)
    {
        printf("ERROR: Invalid integer for %s: \"%s\"\n", optname, s);
        return false;
    }
    *out = (int32_t)v;
    return true;
}

static bool parseFloatOption(const char *optname, const char *s, float *out)
{
    char *end     = NULL;
    float const v = strtof(s, &end);
    if (end == s || *end != '\0')
    {
        printf("ERROR: Invalid float for %s: \"%s\"\n", optname, s);
        return false;
    }
    *out = v;
    return true;
}

static void printUsage(const char *programName);

typedef enum LidarHistogramCliParseRcRec
{
    LIDAR_HIST_CLI_PARSE_OK    = 0,
    LIDAR_HIST_CLI_PARSE_ERROR = 1,
    LIDAR_HIST_CLI_PARSE_HELP  = 2
} LidarHistogramCliParseRc;

/**
 * Parses argv into @p img_width, @p img_height, @p histogram_spec, @p params, and @p input_histogram_file
 * (defaults + options + optional positional input path). Does not open files.
 */
static LidarHistogramCliParseRc parseLidarHistogramCliArgs(int argc, char **argv, const char **input_histogram_file,
                                                           int32_t *img_width, int32_t *img_height,
                                                           PvaLidarHistogramSpec *histogram_spec,
                                                           PvaLidarHistogramParams *params)
{
    *input_histogram_file = NULL;
    *img_width            = 64;
    *img_height           = 32;

    histogram_spec->packingType     = PVA_LIDAR_DATA_PACKING_TYPE_RAW12;
    histogram_spec->numHistPerPixel = 1;
    histogram_spec->numBinsPerHist  = 1024;
    histogram_spec->numPeaksPerHist = 3;
    histogram_spec->pixelHeaderSize = 0;
    histogram_spec->histHeaderSize  = 2;

    params->offsetNs     = 1.0f;
    params->binSizeNs    = 1.0f;
    params->rangeScale   = 100.0f;
    params->maxIntensity = 2000.0f;

    static struct option const long_options[] = {{"help", no_argument, NULL, 'h'},
                                                 {"input", required_argument, NULL, 'i'},
                                                 {"width", required_argument, NULL, 'W'},
                                                 {"height", required_argument, NULL, 'H'},
                                                 {"packing-type", required_argument, NULL, 't'},
                                                 {"num-hist-per-pixel", required_argument, NULL, 'n'},
                                                 {"num-bins-per-hist", required_argument, NULL, 'b'},
                                                 {"num-peaks-per-hist", required_argument, NULL, 'p'},
                                                 {"pixel-header-size", required_argument, NULL, 'x'},
                                                 {"hist-header-size", required_argument, NULL, 'y'},
                                                 {"offset-ns", required_argument, NULL, 'o'},
                                                 {"bin-size-ns", required_argument, NULL, 's'},
                                                 {"range-scale", required_argument, NULL, 'r'},
                                                 {"max-intensity", required_argument, NULL, 'm'},
                                                 {NULL, 0, NULL, 0}};

    char const *optstring = "hi:W:H:t:n:b:p:x:y:o:s:r:m:";

    for (;;)
    {
        int option_index = 0;
        int c            = getopt_long(argc, argv, optstring, long_options, &option_index);
        if (c == -1)
        {
            break;
        }
        switch (c)
        {
        case 'h':
            printUsage(argv[0]);
            return LIDAR_HIST_CLI_PARSE_HELP;
        case 'i':
            if (*input_histogram_file != NULL)
            {
                printf("ERROR: Multiple input files specified (-i may only appear once).\n");
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            *input_histogram_file = optarg;
            break;
        case 'W':
            if (!parseInt32Option("width", optarg, img_width))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'H':
            if (!parseInt32Option("height", optarg, img_height))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 't':
        {
            int32_t packing = 0;
            if (!parseInt32Option("packing-type", optarg, &packing))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            if (packing != (int32_t)PVA_LIDAR_DATA_PACKING_TYPE_NONE &&
                packing != (int32_t)PVA_LIDAR_DATA_PACKING_TYPE_RAW12)
            {
                printf("ERROR: packing-type must be %d (NONE) or %d (RAW12).\n", PVA_LIDAR_DATA_PACKING_TYPE_NONE,
                       PVA_LIDAR_DATA_PACKING_TYPE_RAW12);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            histogram_spec->packingType = (PvaLidarDataPackingType)packing;
            break;
        }
        case 'n':
            if (!parseInt32Option("num-hist-per-pixel", optarg, &histogram_spec->numHistPerPixel))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'b':
            if (!parseInt32Option("num-bins-per-hist", optarg, &histogram_spec->numBinsPerHist))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'p':
            if (!parseInt32Option("num-peaks-per-hist", optarg, &histogram_spec->numPeaksPerHist))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'x':
            if (!parseInt32Option("pixel-header-size", optarg, &histogram_spec->pixelHeaderSize))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'y':
            if (!parseInt32Option("hist-header-size", optarg, &histogram_spec->histHeaderSize))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'o':
            if (!parseFloatOption("offset-ns", optarg, &params->offsetNs))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 's':
            if (!parseFloatOption("bin-size-ns", optarg, &params->binSizeNs))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'r':
            if (!parseFloatOption("range-scale", optarg, &params->rangeScale))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case 'm':
            if (!parseFloatOption("max-intensity", optarg, &params->maxIntensity))
            {
                printUsage(argv[0]);
                return LIDAR_HIST_CLI_PARSE_ERROR;
            }
            break;
        case '?':
        default:
            printUsage(argv[0]);
            return LIDAR_HIST_CLI_PARSE_ERROR;
        }
    }

    if (optind < argc)
    {
        if (*input_histogram_file != NULL)
        {
            printf("ERROR: Input file given both as -i/--input and as a positional argument.\n");
            printUsage(argv[0]);
            return LIDAR_HIST_CLI_PARSE_ERROR;
        }
        *input_histogram_file = argv[optind];
        if (optind + 1 < argc)
        {
            printf("ERROR: Unexpected extra argument: \"%s\"\n", argv[optind + 1]);
            printUsage(argv[0]);
            return LIDAR_HIST_CLI_PARSE_ERROR;
        }
    }

    return LIDAR_HIST_CLI_PARSE_OK;
}

static void printUsage(const char *programName)
{
    printf("Usage: %s [options] [input_histogram_file.bin]\n", programName);
    printf("\n");
    printf("  If input_histogram_file.bin is provided (or -i/--input), metadata and tensor data are read from the\n");
    printf("  file, overriding width, height, histogram spec, and params from the command line.\n");
    printf("  Otherwise, optional arguments below override the built-in defaults; missing options keep defaults.\n");
    printf("  If input_histogram_file.bin is not provided, random histogram data is generated and saved to a file \n");
    printf("  named histogram_data.bin in the current directory.\n");
    printf("  The computed range data is saved to a file named range_data.bin in the current directory.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help                 Show this help and exit.\n");
    printf("  -i, --input <path>         Input histogram binary (same format as dumped histogram_data.bin).\n");
    printf("  -W, --width <int>          Tensor width  (W in NHWC). Default: 64.\n");
    printf("  -H, --height <int>         Tensor height (H in NHWC). Default: 32.\n");
    printf("  -t, --packing-type <int>   0 = NONE (U16), 1 = RAW12 (3 bytes per 2 elements). Default: 1 (RAW12).\n");
    printf("  -n, --num-hist-per-pixel <int>   Default: 1.\n");
    printf("  -b, --num-bins-per-hist <int>    Default: 1024.\n");
    printf("  -p, --num-peaks-per-hist <int>   Default: 3.\n");
    printf("  -x, --pixel-header-size <int>    Default: 0.\n");
    printf("  -y, --hist-header-size <int>     Default: 2.\n");
    printf("  -o, --offset-ns <float>          Default: 1.0.\n");
    printf("  -s, --bin-size-ns <float>        Default: 1.0.\n");
    printf("  -r, --range-scale <float>        Default: 100.0 (e.g. m to cm).\n");
    printf("  -m, --max-intensity <float>      Default: 2000.0.\n");
    printf("\n");
    printf("Place options before the optional positional input file; otherwise arguments after the file name may\n");
    printf("be ignored (POSIX getopt).\n");
}

int main(int argc, char **argv)
{
    int32_t err = 0;

    const char *inputHistogramFile = NULL;
    int32_t imgWidth               = 0;
    int32_t imgHeight              = 0;
    PvaLidarHistogramSpec histogramSpec;
    PvaLidarHistogramParams params;

    LidarHistogramCliParseRc const parse_rc =
        parseLidarHistogramCliArgs(argc, argv, &inputHistogramFile, &imgWidth, &imgHeight, &histogramSpec, &params);
    if (parse_rc == LIDAR_HIST_CLI_PARSE_HELP)
    {
        return 0;
    }
    if (parse_rc != LIDAR_HIST_CLI_PARSE_OK)
    {
        return 1;
    }

    const int32_t tensorRank = 4; // NHWC

    if (inputHistogramFile != NULL)
    {
        printf("Loading histogram data from file: %s\n", inputHistogramFile);
    }
    else
    {
        printf("Generating random histogram data\n");
    }

    // Read metadata from file if provided, prior to allocating tensors (overrides CLI and defaults).
    if (inputHistogramFile)
    {
        printf("Reading histogram metadata from file: %s\n", inputHistogramFile);
        FILE *fp = fopen(inputHistogramFile, "rb");
        if (!fp)
        {
            printf("ERROR: Failed to open input file.\n");
            printUsage(argv[0]);
            return 1;
        }

        // Read tensor metadata and histogram specification, overwriting default values.
        int32_t fileChannels, fileDataTypeSize;
        if (!readHistogramTensorMetadataFromFile(fp, &imgWidth, &imgHeight, &fileChannels, &fileDataTypeSize,
                                                 &histogramSpec, &params))
        {
            printf("ERROR: Failed to read metadata from input file\n");
            fclose(fp);
            return 1;
        }

        if (fclose(fp) != 0)
        {
            printf("ERROR: Failed to close input file.\n");
            return 1;
        }
        printf("Input file dimensions (WxHxC): %dx%dx%d\n", imgWidth, imgHeight, fileChannels);
        printf(
            "Input histogram spec: packingType=%d, numHistPerPixel=%d, numBinsPerHist=%d, numPeaksPerHist=%d, "
            "pixelHeaderSize=%d, histHeaderSize=%d\n",
            histogramSpec.packingType, histogramSpec.numHistPerPixel, histogramSpec.numBinsPerHist,
            histogramSpec.numPeaksPerHist, histogramSpec.pixelHeaderSize, histogramSpec.histHeaderSize);
        printf("Input histogram params: offsetNs=%f, binSizeNs=%f, rangeScale=%f, maxIntensity=%f\n", params.offsetNs,
               params.binSizeNs, params.rangeScale, params.maxIntensity);
    }

    // Total number of elements per pixel, histogram bins + headers
    int32_t const elementsPerPixel =
        histogramSpec.pixelHeaderSize +
        histogramSpec.numHistPerPixel * (histogramSpec.histHeaderSize + histogramSpec.numBinsPerHist);

    // Select histogram C channel size and data type that will fit the pixel size
    NVCVDataType histogramTensorDataType = NVCV_DATA_TYPE_NONE;
    int32_t histogramTensorChannels      = 0;
    switch (histogramSpec.packingType)
    {
    case PVA_LIDAR_DATA_PACKING_TYPE_NONE:
        histogramTensorDataType = NVCV_DATA_TYPE_U16;
        histogramTensorChannels = elementsPerPixel;
        break;
    case PVA_LIDAR_DATA_PACKING_TYPE_RAW12:
        histogramTensorDataType = NVCV_DATA_TYPE_U8;
        // round up to an integral number of RAW12 packets (3 bytes per 2 elements)
        histogramTensorChannels = ((elementsPerPixel + 1) / 2) * 3;
        break;
    }

    // Output dimensions
    int32_t const outputTensorElementsPerPixel = histogramSpec.numHistPerPixel * histogramSpec.numPeaksPerHist;
    const int32_t rangeTensorChannels          = outputTensorElementsPerPixel;
    const int32_t xyzTensorChannels            = 3 * outputTensorElementsPerPixel;
    const int32_t reflectanceTensorChannels    = outputTensorElementsPerPixel;

    // Set the handles of resources that should be freed in case of an error to NULL.
    // This is done because calling the corresponding free functions with NULL is a no-op.
    NVCVAllocatorHandle allocatorHandle         = NULL;
    NVCVTensorHandle histogramTensorHandle      = NULL;
    NVCVTensorHandle rangeBiasTensorHandle      = NULL;
    NVCVTensorHandle rangeTensorHandle          = NULL;
    NVCVTensorHandle refRangeTensorHandle       = NULL;
    NVCVTensorHandle refXyzTensorHandle         = NULL;
    NVCVTensorHandle reflectanceTensorHandle    = NULL;
    NVCVTensorHandle refReflectanceTensorHandle = NULL;
    NVCVOperatorHandle operatorHandle           = NULL;
    /// NOTE: the xyz tensors are not used in this example application (optional for the LidarHistogramConversion operator)
    NVCVTensorHandle xyzTensorHandle            = NULL;
    NVCVTensorHandle xyzCalibrationTensorHandle = NULL;
    cupvaSyncObj_t sync                         = NULL;
    cupvaStream_t stream                        = NULL;

    // Create tensor layout
    NVCVTensorLayout tensorLayout;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("NHWC", &tensorLayout), err, FreeResources);

    // Create allocator
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&allocatorHandle), err, FreeResources);

    // Create histogram tensor
    int64_t histogramTensorShape[] = {1, imgHeight, imgWidth, histogramTensorChannels};
    NVCVTensorRequirements histogramTensorRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, histogramTensorShape, histogramTensorDataType,
                                                        tensorLayout, 0, 0, &histogramTensorRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&histogramTensorRequirements, allocatorHandle, &histogramTensorHandle),
                          err, FreeResources);

    // Create range bias tensor (HxWx1, F32)
    int64_t rangeBiasShape[] = {1, imgHeight, imgWidth, 1};
    NVCVTensorRequirements rangeBiasRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, rangeBiasShape, NVCV_DATA_TYPE_F32, tensorLayout, 0,
                                                        0, &rangeBiasRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&rangeBiasRequirements, allocatorHandle, &rangeBiasTensorHandle), err,
                          FreeResources);

    // Create range output tensor
    int64_t rangeShape[] = {1, imgHeight, imgWidth, rangeTensorChannels};
    NVCVTensorRequirements rangeRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, rangeShape, NVCV_DATA_TYPE_F32, tensorLayout, 0, 0,
                                                        &rangeRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&rangeRequirements, allocatorHandle, &rangeTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&rangeRequirements, allocatorHandle, &refRangeTensorHandle), err,
                          FreeResources);

    // Create reflectance output tensor
    int64_t reflectanceShape[] = {1, imgHeight, imgWidth, reflectanceTensorChannels};
    NVCVTensorRequirements reflectanceRequirements;
    NVCV_CHECK_ERROR_GOTO(nvcvTensorCalcRequirementsPva(tensorRank, reflectanceShape, NVCV_DATA_TYPE_F32, tensorLayout,
                                                        0, 0, &reflectanceRequirements),
                          err, FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reflectanceRequirements, allocatorHandle, &reflectanceTensorHandle), err,
                          FreeResources);
    NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reflectanceRequirements, allocatorHandle, &refReflectanceTensorHandle),
                          err, FreeResources);

    uint32_t pvaDriverVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetDriverVersion(&pvaDriverVersion), err, FreeResources);
    printf("PVA driver version: %u\n", pvaDriverVersion);

    uint32_t cupvaRuntimeVersion;
    CUPVA_CHECK_ERROR_GOTO(CupvaGetRuntimeVersion(&cupvaRuntimeVersion), err, FreeResources);
    printf("cuPVA runtime version: %u\n", cupvaRuntimeVersion);

    if (pvaDriverVersion < 2006 || cupvaRuntimeVersion < 2006)
    {
        printf("WARNING: Incompatible PVA driver or runtime version. 2006+ are required for TensorDataFlow support.\n");
        err = 0;
        goto FreeResources;
    }

    // Load or generate histogram input data
    if (inputHistogramFile)
    {
        // Load histogram data from file
        if (!loadHistogramTensor(histogramTensorHandle, inputHistogramFile, &histogramSpec, &params))
        {
            printf("ERROR: Failed to load histogram data from file\n");
            err = 1;
            goto FreeResources;
        }
    }
    else
    {
        // Generate random input data
        generateRandomLidarHistogramInput(histogramTensorHandle, &histogramSpec, &params);
    }

    // Initialize rangeBias tensor with constant value
    generateRangeBiasTensor(rangeBiasTensorHandle, 1.0f);

    // Create the operator
    NVCV_CHECK_ERROR_GOTO(
        pvaLidarHistogramConversionCreate(&operatorHandle, &histogramTensorRequirements, &histogramSpec), err,
        FreeResources);

    // Create CUPVA stream and synchronization objects
    CUPVA_CHECK_ERROR_GOTO(CupvaSyncObjCreate(&sync, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD), err,
                           FreeResources);

    cupvaFence_t fence;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, sync), err, FreeResources);

    cupvaCmd_t rf;
    CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&stream, CUPVA_PVA0, CUPVA_VPU_ANY), err, FreeResources);

    cupvaCmd_t const *cmds[1]  = {&rf};
    cupvaCmdStatus_t status[1] = {NULL};

    // Submit the operator for execution
    NVCV_CHECK_ERROR_GOTO(pvaLidarHistogramConversionSubmit(
                              operatorHandle, stream, &params, histogramTensorHandle, rangeBiasTensorHandle,
                              xyzCalibrationTensorHandle, rangeTensorHandle, xyzTensorHandle, reflectanceTensorHandle),
                          err, FreeResources);

    CUPVA_CHECK_ERROR_GOTO(CupvaStreamSubmit(stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), err, FreeResources);

    // Wait for completion
    bool waitSuccess;
    CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitSuccess), err, FreeResources);

    // Dump input histogram tensor to file if it was generated
    if (!inputHistogramFile &&
        !dumpHistogramTensor(histogramTensorHandle, "histogram_data.bin", &histogramSpec, &params))
    {
        printf("Warning: Failed to dump histogram tensor\n");
    }

    // Run reference implementation
    lidarHistogramConversionRef(histogramTensorHandle, rangeBiasTensorHandle, xyzCalibrationTensorHandle,
                                refRangeTensorHandle, refXyzTensorHandle, refReflectanceTensorHandle, &params,
                                &histogramSpec);

#if DEBUG_OVERWRITE_INPUT_WITH_SMOOTHED_HISTOGRAM
    // When this flag is set, the reference code overwrites the input histogram tensor with smoothed data
    if (!dumpHistogramTensor(histogramTensorHandle, "smooth_histogram_data.bin", &histogramSpec, &params))
    {
        printf("Warning: Failed to dump smoothed histogram tensor\n");
    }
#endif

    // Dump range tensor to file for visualization/debugging
    if (!dumpRangeTensor(refRangeTensorHandle, "range_data.bin"))
    {
        printf("Warning: Failed to dump range tensor\n");
    }

    // compare reference and PVA output tensors
    const float tolerance = 1e-5f * params.rangeScale;
    int32_t rangeErr      = checkLidarHistogramConversionOutput(refRangeTensorHandle, rangeTensorHandle, tolerance);
    if (rangeErr)
    {
        printf("Range tensor check FAILED\n");
        err = 1;
    }

    int32_t reflErr =
        checkLidarHistogramConversionOutput(refReflectanceTensorHandle, reflectanceTensorHandle, tolerance);
    if (reflErr)
    {
        printf("Reflectance tensor check FAILED\n");
        err = 1;
    }

    if (!(rangeErr || reflErr))
    {
        printf("Test Pass\n");
    }

FreeResources: /* clean up all allocated resources */
    CupvaStreamDestroy(stream);
    CupvaSyncObjDestroy(sync);
    nvcvTensorDecRef(histogramTensorHandle, NULL);
    nvcvTensorDecRef(rangeBiasTensorHandle, NULL);
    nvcvTensorDecRef(xyzCalibrationTensorHandle, NULL);
    nvcvTensorDecRef(rangeTensorHandle, NULL);
    nvcvTensorDecRef(refRangeTensorHandle, NULL);
    nvcvTensorDecRef(xyzTensorHandle, NULL);
    nvcvTensorDecRef(refXyzTensorHandle, NULL);
    nvcvTensorDecRef(reflectanceTensorHandle, NULL);
    nvcvTensorDecRef(refReflectanceTensorHandle, NULL);
    nvcvOperatorDestroy(operatorHandle);
    nvcvAllocatorDecRef(allocatorHandle, NULL);

    return err;
}
