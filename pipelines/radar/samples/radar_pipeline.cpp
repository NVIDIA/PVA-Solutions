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

#include "radar_pipeline_iface.hpp"
#include "radar_pipeline_utils.hpp"

#include <cupva_host.hpp>

#include <ImageIO.h>
#include <PvaAllocator.h>
#include <PvaOperatorTypes.h> // PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define ENABLE_INTERMEDIATE_TENSORS_DEBUG 0
#define SPEED_OF_LIGHT_MPS 299792458.0f

/** Radar module (sensor position) */
typedef enum
{
    RADAR_PIPELINE_MODULE_FRONT  = 0,
    RADAR_PIPELINE_MODULE_CORNER = 1
} RadarPipelineModule;

/**
 * @brief Radar system selection (vendor, module, mode, range).
 *
 * Identifies the radar variant for configuration.
 * rangeSubMode is vendor-specific; use 0 if not applicable.
 */
typedef struct
{
    RadarPipelineModule radarModule; /**< e.g. RADAR_PIPELINE_MODULE_FRONT or _CORNER */
} RadarPipelineSystemSelection;

// ============================================================================
// Radar Pipeline Factory — C++ runtime polymorphism (no dlopen)
// ============================================================================

using RadarPipelineFactory = IRadarPipeline *(*)();

/** Print tensor descriptor and a small prefix of the buffer for debugging. */
static void print_tensor_partial(const char *name, const RadarPipelineTensorData *td, int max_elems = 24)
{
    if (!td || !td->data)
    {
        printf("  %s: (null or no data)\n", name);
        return;
    }
    printf("  %s: ndim=%d shape=[%lld,%lld,%lld,%lld] strides=[%lld,%lld,%lld,%lld] dtype=%d total_bytes=%lld\n", name,
           td->ndim, (long long)td->shape[0], (long long)td->shape[1], (long long)td->shape[2], (long long)td->shape[3],
           (long long)td->strides[0], (long long)td->strides[1], (long long)td->strides[2], (long long)td->strides[3],
           (int)td->dtype, (long long)td->total_bytes);

    const int16_t *p16 = (const int16_t *)td->data;
    const int32_t *p32 = (const int32_t *)td->data;
    const float *pf    = (const float *)td->data;

    switch (td->dtype)
    {
    case RADAR_PIPELINE_DTYPE_INT16:
        printf("    first %d elem(s): ", std::min(max_elems, (int)(td->total_bytes / sizeof(int16_t))));
        for (int i = 0; i < max_elems && i * (int)sizeof(int16_t) < (int)td->total_bytes; i++)
            printf("%d ", (int)p16[i]);
        printf("\n");
        break;
    case RADAR_PIPELINE_DTYPE_INT32:
        printf("    first %d elem(s): ", std::min(max_elems, (int)(td->total_bytes / sizeof(int32_t))));
        for (int i = 0; i < max_elems && i * (int)sizeof(int32_t) < (int)td->total_bytes; i++)
            printf("%d ", (int)p32[i]);
        printf("\n");
        break;
    case RADAR_PIPELINE_DTYPE_FLOAT32:
        printf("    first %d elem(s): ", std::min(max_elems, (int)(td->total_bytes / sizeof(float))));
        for (int i = 0; i < max_elems && i * (int)sizeof(float) < (int)td->total_bytes; i++) printf("%.4f ", pf[i]);
        printf("\n");
        break;
    case RADAR_PIPELINE_DTYPE_COMPLEX_INT16:
    {
        int n_complex = std::min(max_elems / 2, (int)(td->total_bytes / (2 * sizeof(int16_t))));
        printf("    first %d complex (re,im): ", n_complex);
        for (int i = 0; i < n_complex; i++) printf("(%d,%d) ", (int)p16[i * 2], (int)p16[i * 2 + 1]);
        printf("\n");
        break;
    }
    case RADAR_PIPELINE_DTYPE_COMPLEX_INT32:
    {
        int n_complex = std::min(max_elems / 2, (int)(td->total_bytes / (2 * sizeof(int32_t))));
        printf("    first %d complex (re,im): ", n_complex);
        for (int i = 0; i < n_complex; i++) printf("(%d,%d) ", (int)p32[i * 2], (int)p32[i * 2 + 1]);
        printf("\n");
        break;
    }
    default:
        printf("    (unknown dtype, skip print)\n");
        break;
    }
}

/**
 * @brief Initialize RadarPipelineConfig from system selection and RadarConfig (CFAR, DOA).
 * Waveform/timing/coefficients and Bartlett positions come from gp in radar_pipeline_create.
 * CFAR and DOA params are only applicable to the 8T8R antenna layout.
 */
static void InitRadarPipelineApiConfig(RadarPipelineConfig *apiConfig, bool is8t8r, uint32_t doaMethod)
{
    /* Caller sets gp before this; a full struct assign would otherwise zero gp (C++ value-init). */
    RadarPipelineGP *gp = apiConfig->gp;
    *apiConfig          = {.gp = gp};

    if (is8t8r)
    {
        apiConfig->cfar_params = {
            .numHorTrain               = 12,
            .numHorGuard               = 2,
            .numVerTrain               = 48,
            .numVerGuard               = 8,
            .probabilityFA             = 1e-1f,
            .isHorizontalCyclicPadding = 1,
            .isVerticalCyclicPadding   = 1,
            .enablePeakGrouping        = 0,
        };

        if (doaMethod == RADAR_PIPELINE_DOA_CAPON_ROI)
        {
            /* Zero every DOA field first, so the Capon path sees no stale Bartlett or target-processing values.
             * The Capon config keeps the library defaults (a zero field, see radarPipelineCaponConfigDefaults())
             * except for the overrides below. */
            apiConfig->doa_params            = {};
            apiConfig->doa_params.doa_method = RADAR_PIPELINE_DOA_CAPON_ROI;
            RadarPipelineCaponConfig &capon  = apiConfig->doa_params.capon_config;
            capon.enableFBA                  = 1; /* doubles the snapshot rows: a steadier covariance per ROI */
            capon.enableSubBinInterp         = 1; /* fractional azimuth/elevation bins */
            /* The 8T8R module's RX array (AntennaLayout.png): 2 columns (azimuth) x 4 rows (elevation), RX
             * channel rx at (rx % 2, rx / 2); with 8 TX a 16 x 4 virtual array. */
            capon.rxAzimuthElements   = 2;
            capon.rxElevationElements = 4;
        }
        else
        {
            apiConfig->doa_params = {
                .doa_method          = RADAR_PIPELINE_DOA_BARTLETT,
                .enableInterpolation = 1,
                .enablePowerOutput =
                    1, /* must match target list 8 rows so PVA and ref both output power for validation */
                .bartlett_config =
                    {
                        .numAzimuthBins   = 81,
                        .numElevationBins = 8,
                        .azimuthStart     = -60.0f,
                        .azimuthStep      = 1.5f,
                        .elevationStart   = -12.0f,
                        .elevationStep    = 6.0f,
                    },
                .separableParams =
                    {
                        .enableSeparable = 1,
                        .azNeighborhood  = RADAR_PIPELINE_BARTLETT_AZ_NEIGHBORHOOD_2,
                        .azSource        = RADAR_PIPELINE_BARTLETT_AZ_FROM_NEIGHBORHOOD,
                    },
                .targetProcessingParams =
                    {
                        .fftShiftedDoppler            = 0,
                        .enableRvDecoupling           = 1,
                        .enableQuadraticInterpolation = 1,
                    },
            };
        }
    }
}

// Copy 3D data cube [chirps][rx][samples] into flat host buffer for PVA
static void loadRawAdcToHost(int32_t *hostPtr, const std::vector<std::vector<std::vector<int32_t>>> &inBuf)
{
    const size_t numChirps  = inBuf.size();
    const size_t numRx      = inBuf[0].size();
    const size_t numSamples = inBuf[0][0].size();
    for (size_t c = 0; c < numChirps; c++)
    {
        for (size_t r = 0; r < numRx; r++)
        {
            for (size_t s = 0; s < numSamples; s++)
            {
                hostPtr[c * numRx * numSamples + r * numSamples + s] = inBuf[c][r][s];
            }
        }
    }
}

/** Per-antenna profile: assets layout + factory for the matching pipeline class. */
struct AntennaProfile
{
    const char *id;
    const char *label;
    const char *helpLine;
    const char *assetsDir;
    const char *capturePrefix;
    RadarPipelineFactory factory; /**< Returns a heap-allocated IRadarPipeline; caller owns it. */
};

static const AntennaProfile kAntennaProfiles[] = {
    {"8t8r", "8T8R", "8 TX × 8 RX front", "/8t8r/", "radar_raw_8t8r_", radar_pipeline_8t8r},
    {"4t4r", "4T4R", "4 TX × 4 RX corner", "/4t4r/", "radar_raw_4t4r_", radar_pipeline_4t4r},
};

static const AntennaProfile *LookupAntennaProfile(const char *id)
{
    const size_t n = sizeof(kAntennaProfiles) / sizeof(kAntennaProfiles[0]);
    for (size_t j = 0; j < n; ++j)
    {
        if (strcmp(id, kAntennaProfiles[j].id) == 0)
            return &kAntennaProfiles[j];
    }
    return nullptr;
}

/** Maps --antenna_config id to initSystemConfig() variant (must stay in sync with kAntennaProfiles). */
static RadarPipelineModule RadarModuleForAntennaId(const char *id)
{
    if (strcmp(id, "8t8r") == 0)
        return RADAR_PIPELINE_MODULE_FRONT;
    return RADAR_PIPELINE_MODULE_CORNER;
}

static int32_t initSystemConfig(const RadarPipelineSystemSelection *selection, RadarPipelineGP *gp)
{
    if (selection == NULL || gp == NULL)
    {
        return -1;
    }

    const RadarPipelineModule radarModule = selection->radarModule;

    fprintf(stderr, "[radar_pipeline_config] selection: module=%s\n",
            radarModule == RADAR_PIPELINE_MODULE_FRONT ? "FRONT" : "CORNER");

    memset(gp, 0, sizeof(*gp));
    gp->c = SPEED_OF_LIGHT_MPS;

    if (radarModule == RADAR_PIPELINE_MODULE_FRONT)
    {
        gp->NofSamples      = 1024;
        gp->NofRamps        = 512;
        gp->NofRangeBins    = 512;
        gp->NofRx           = 8;
        gp->NofTx           = 8;
        gp->RDFold          = 16;
        gp->NofRampsFold    = gp->NofRamps / gp->RDFold;
        gp->numTx           = 8;
        gp->numRx           = 8;
        gp->EffBw           = 77e9f;
        gp->wavelength      = gp->c / gp->EffBw;
        gp->sweepTime       = 0.000003415696334829077;
        gp->rangeResolution = 0.5f;
        gp->txGroups[0]     = 0;
        gp->txGroupsCount   = 1;
        {
            const uint32_t n              = 16;
            gp->ddm_doppler_offsets_count = n;
            for (uint32_t i = 0; i < n && i < RADAR_PIPELINE_MAX_NUM_DOPPLER_OFFSETS; i++)
            {
                gp->ddm_doppler_offsets[i] = (float)i / (float)n;
            }
        }

        gp->fs         = 10e6f;
        gp->PRI        = 100e-6f;
        gp->DeltaV     = 3.6f;
        gp->ContR      = 7.5e-6f;
        gp->ContV_fast = 7.5e-06f;
        gp->ContV_slow = 1.5e8f;
        gp->Dis        = 0.5f;
        gp->DisE       = 0.5f;
    }
    else /* RADAR_PIPELINE_MODULE_CORNER */
    {
        gp->NofSamples    = 512;
        gp->NofRamps      = 512;
        gp->NofRangeBins  = 224;
        gp->NofRx         = 4;
        gp->NofTx         = 4;
        gp->RDFold        = 8;
        gp->NofRampsFold  = gp->NofRamps / gp->RDFold;
        gp->txGroups[0]   = 0;
        gp->txGroupsCount = 1;
    }

    return 0;
}

/* Capon writes no target list: print the ROI count and each ROI's strongest peak (power and fractional
 * azimuth/elevation bin on the 16 x 8 steering grid of its region of interest). */
static void PrintCaponPeaks(IRadarPipeline &pipeline)
{
    RadarPipelineTensorData count = {}, power = {}, az = {}, el = {};
    if (pipeline.get_intermediate_tensor(RADAR_PIPELINE_TENSOR_CAPON_ROI_COUNT, RADAR_PIPELINE_SOURCE_PVA, &count) !=
            0 ||
        pipeline.get_intermediate_tensor(RADAR_PIPELINE_TENSOR_CAPON_PEAK_POWER, RADAR_PIPELINE_SOURCE_PVA, &power) !=
            0 ||
        pipeline.get_intermediate_tensor(RADAR_PIPELINE_TENSOR_CAPON_AZ_BIN, RADAR_PIPELINE_SOURCE_PVA, &az) != 0 ||
        pipeline.get_intermediate_tensor(RADAR_PIPELINE_TENSOR_CAPON_EL_BIN, RADAR_PIPELINE_SOURCE_PVA, &el) != 0)
    {
        printf("Failed to read the Capon output tensors\n");
        return;
    }
    const int32_t roiNum = *static_cast<const int32_t *>(count.data);
    printf("Capon DoA on %d ROI(s); strongest peak per ROI (power, azimuth bin, elevation bin):\n", roiNum);
    for (int32_t r = 0; r < roiNum; ++r)
    {
        /* Row r, peak 0 (the strongest), through the byte strides. */
        const float p = *reinterpret_cast<const float *>(static_cast<const char *>(power.data) + r * power.strides[0]);
        const float a = *reinterpret_cast<const float *>(static_cast<const char *>(az.data) + r * az.strides[0]);
        const float e = *reinterpret_cast<const float *>(static_cast<const char *>(el.data) + r * el.strides[0]);
        if (p == PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER)
            printf("  ROI %3d: no valid peak\n", r);
        else
            printf("  ROI %3d: power %.4g, az bin %.2f, el bin %.2f\n", r, p, a, e);
    }
}

int main(int argc, char **argv)
{
    int32_t err                 = 0;
    const int32_t maxPathLength = 256;
    char assetsDirectory[maxPathLength];
    const AntennaProfile *antennaProfile = &kAntennaProfiles[0];
    bool antennaConfigArgSet             = false;
    uint32_t doaMethod                   = RADAR_PIPELINE_DOA_BARTLETT; // default (8T8R)
    bool doaArgSet                       = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            printf("Usage: %s [assets_directory] [--antenna_config <id>] [--doa <method>]\n", argv[0]);
            printf("Options:\n");
            printf("  -h, --help                  Show this help message\n");
            printf("  --antenna_config <id>       Radar layout (default: 8t8r)\n");
            for (size_t j = 0; j < sizeof(kAntennaProfiles) / sizeof(kAntennaProfiles[0]); ++j)
            {
                printf("                                %s — %s\n", kAntennaProfiles[j].id,
                       kAntennaProfiles[j].helpLine);
            }
            printf("  --doa <method>              DOA method for 8T8R (default: bartlett)\n");
            printf("                                bartlett   — Bartlett beamforming\n");
            printf("                                capon      — Capon (MVDR) DoA-on-ROI\n");
            return 0;
        }
        if (strcmp(argv[i], "--doa") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Error: --doa requires an argument\n");
                return 1;
            }
            if (doaArgSet)
            {
                fprintf(stderr, "Error: --doa may only be specified once\n");
                return 1;
            }
            doaArgSet     = true;
            const char *m = argv[++i];
            if (strcmp(m, "bartlett") == 0)
                doaMethod = RADAR_PIPELINE_DOA_BARTLETT;
            else if (strcmp(m, "capon") == 0)
                doaMethod = RADAR_PIPELINE_DOA_CAPON_ROI; // init() reports a PVA SDK without LLVM support
            else
            {
                fprintf(stderr, "Error: unknown --doa '%s' (expected bartlett|capon)\n", m);
                return 1;
            }
            continue;
        }
        if (strcmp(argv[i], "--antenna_config") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Error: --antenna_config requires an argument\n");
                return 1;
            }
            if (antennaConfigArgSet)
            {
                fprintf(stderr, "Error: --antenna_config may only be specified once\n");
                return 1;
            }
            antennaConfigArgSet         = true;
            const char *id              = argv[++i];
            const AntennaProfile *found = LookupAntennaProfile(id);
            if (!found)
            {
                fprintf(stderr, "Error: unknown --antenna_config '%s'\n", id);
                return 1;
            }
            antennaProfile = found;
        }
    }

    if (doaArgSet && strcmp(antennaProfile->id, "8t8r") != 0)
    {
        fprintf(stderr, "Error: --doa applies only to --antenna_config 8t8r\n");
        return 1;
    }

    err = GetAssetsDirectory(argc, argv, assetsDirectory, maxPathLength);
    if (err != 0)
    {
        printf("Error: Failed to get assets directory\n");
        return 1;
    }

    std::string assetsDirString(assetsDirectory);
    assetsDirString += "/radar/";
    assetsDirString += antennaProfile->assetsDir;

    printf("Antenna configuration: %s (assets dir: %s, capture prefix: %s)\n", antennaProfile->label,
           antennaProfile->assetsDir, antennaProfile->capturePrefix);

    std::vector<std::string> files = FindFilesWithPrefix(assetsDirString, antennaProfile->capturePrefix);

    RadarPipelineSystemSelection systemSelection = {
        .radarModule = RadarModuleForAntennaId(antennaProfile->id),
    };
    RadarPipelineGP gp;
    initSystemConfig(&systemSelection, &gp);

    RadarPipelineConfig apiConfig;
    apiConfig.gp = &gp;
    InitRadarPipelineApiConfig(&apiConfig, strcmp(antennaProfile->id, "8t8r") == 0, doaMethod);

    std::unique_ptr<IRadarPipeline> pipeline(antennaProfile->factory());

    printf("\n=== Creating radar pipeline ===\n");
    RadarPipelineAdcDimensions adcDimensions{};
    err = pipeline->init(&apiConfig, &adcDimensions);
    if (err != 0)
    {
        printf("Failed to create radar pipeline: %d", err);
        if (err == RADAR_PIPELINE_CREATE_ERR_NULL_ARGS)
            printf(" (null handle or config)");
        else if (err == RADAR_PIPELINE_CREATE_ERR_DDM_OFFSETS)
            printf(" (gp.ddm_doppler_offsets_count does not match RDFold; Capon: fewer offsets than NofTx)");
        else if (err == RADAR_PIPELINE_CREATE_ERR_DOA_METHOD)
            printf(" (unsupported doa_method; Capon needs an LLVM-capable PVA SDK)");
        else if (err == RADAR_PIPELINE_CREATE_ERR_PVA_WORKLOAD)
            printf(" (PVA operator creation failed – check stderr for the reason)");
        else if (err == RADAR_PIPELINE_CREATE_ERR_EXCEPTION)
            printf(" (exception – check stderr for message)");
        else if (err == RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG)
            printf(" (Capon configuration not supported – check stderr for the reason)");
        printf("\n");
        return err;
    }
    printf("Radar pipeline variant: %s (version %s)\n", antennaProfile->label, pipeline->version());
    printf("Radar pipeline created successfully\n");

    int32_t NofSamples = adcDimensions.NofSamples;
    int32_t NofRamps   = adcDimensions.NofRamps;
    int32_t NofRx      = adcDimensions.NofRx;

    for (const auto &file : files)
    {
        // Only process raw binary data files; skip config files (e.g. .json)
        if (file.size() < 4u || file.substr(file.size() - 4u) != ".bin")
        {
            printf("Skipping non-binary file: %s\n", file.c_str());
            continue;
        }
        printf("\n=== Processing file %s ===\n", file.c_str());

        printf("Reading raw data...\n");
        std::vector<std::vector<std::vector<int32_t>>> raw_data_result;
        try
        {
            raw_data_result = ReadRawDataCube(file, NofRamps, NofRx, NofSamples, 20);
        }
        catch (const std::exception &e)
        {
            // Missing, wrong-size or not-pulled LFS capture: report it and clean up below
            printf("Failed to read %s: %s\n", file.c_str(), e.what());
            err = -1;
            break;
        }

        // Construct a PVA-aligned input tensor directly (no separate cupva::mem::Alloc needed)
        NVCVAllocatorHandle inputAllocator = nullptr;
        nvcvAllocatorConstructPva(&inputAllocator);

        NVCVTensorLayout inputLayout;
        nvcvTensorLayoutMake("HCW", &inputLayout);
        int64_t inputShape[]             = {NofRamps, NofRx, NofSamples};
        NVCVTensorRequirements inputReqs = {};
        nvcvTensorCalcRequirementsPva(3, inputShape, NVCV_DATA_TYPE_S32, inputLayout, 0, 0, &inputReqs);

        NVCVTensorHandle input_tensor = nullptr;
        nvcvTensorConstruct(&inputReqs, inputAllocator, &input_tensor);

        // Load raw ADC data via the tensor's host-accessible pointer
        NVCVTensorData inputTensorData = {};
        if (nvcvTensorExportData(input_tensor, &inputTensorData) != NVCV_SUCCESS)
        {
            printf("Failed to export input tensor data\n");
            nvcvTensorDecRef(input_tensor, nullptr);
            nvcvAllocatorDecRef(inputAllocator, nullptr);
            break;
        }
        int32_t *input_host_ptr = (int32_t *)cupva::mem::GetHostPointer(inputTensorData.buffer.strided.basePtr);
        if (input_host_ptr == nullptr)
        {
            printf("Failed to get host pointer for input tensor\n");
            nvcvTensorDecRef(input_tensor, nullptr);
            nvcvAllocatorDecRef(inputAllocator, nullptr);
            break;
        }
        loadRawAdcToHost(input_host_ptr, raw_data_result);

        // Pass nullptr for output to write to internal buffer (used by validate())
        printf("Processing through radar pipeline...\n");
        err = pipeline->process(input_tensor, nullptr);
        if (err != 0)
        {
            printf("Pipeline processing failed with error: %d\n", err);
            nvcvTensorDecRef(input_tensor, nullptr);
            nvcvAllocatorDecRef(inputAllocator, nullptr);
            break;
        }

        int32_t targetCount = 0;
        err                 = pipeline->get_target_count(&targetCount);
        if (err != 0)
        {
            printf("Failed to get target count: %d\n", err);
            nvcvTensorDecRef(input_tensor, nullptr);
            nvcvAllocatorDecRef(inputAllocator, nullptr);
            break;
        }
        if (doaMethod == RADAR_PIPELINE_DOA_CAPON_ROI)
            PrintCaponPeaks(*pipeline);
        else
            printf("Detected %d targets\n", targetCount);

        printf("Validating PVA output against CPU reference...\n");
        err = pipeline->validate(inputTensorData.buffer.strided.basePtr);
        if (err != 0)
        {
            printf("PVA vs reference validation failed: %d\n", err);
            nvcvTensorDecRef(input_tensor, nullptr);
            nvcvAllocatorDecRef(inputAllocator, nullptr);
            break;
        }
        printf("PVA output matches CPU reference\n");

#if ENABLE_INTERMEDIATE_TENSORS_DEBUG
        RadarPipelineTensorData tensor_data = {};
        printf("\n--- Intermediate tensors (PVA) ---\n");
        err = pipeline->get_intermediate_tensor(RADAR_PIPELINE_TENSOR_DOPPLER_MAP, RADAR_PIPELINE_SOURCE_PVA,
                                                &tensor_data);
        if (err == 0)
        {
            printf("RADAR_PIPELINE_TENSOR_DOPPLER_MAP:\n");
            print_tensor_partial("Doppler map", &tensor_data, 24);
        }
        else
        {
            printf("get_intermediate_tensor(DOPPLER_MAP) failed: %d\n", err);
        }

        tensor_data = {};
        err         = pipeline->get_intermediate_tensor(RADAR_PIPELINE_TENSOR_DETECTION_LIST, RADAR_PIPELINE_SOURCE_PVA,
                                                        &tensor_data);
        if (err == 0)
        {
            printf("RADAR_PIPELINE_TENSOR_DETECTION_LIST:\n");
            print_tensor_partial("Detection list", &tensor_data, 32);
        }
        else
        {
            printf("get_intermediate_tensor(DETECTION_LIST) failed: %d\n", err);
        }
        printf("----------------------------------\n\n");
#endif

        printf("Successfully processed file\n");
        nvcvTensorDecRef(input_tensor, nullptr);
        nvcvAllocatorDecRef(inputAllocator, nullptr);
    }

    printf("\n=== Cleaning up ===\n");
    pipeline.reset(); // calls ~RadarPipelineXt4r() which calls radar_pipeline_destroy internally

    if (err == 0)
    {
        printf("=== All tests passed ===\n");
    }

    return err;
}
