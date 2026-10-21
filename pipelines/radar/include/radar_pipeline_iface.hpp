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

#ifndef RADAR_PIPELINE_IFACE_HPP
#define RADAR_PIPELINE_IFACE_HPP

#include "nvcv/Tensor.h"

#include <stddef.h>
#include <stdint.h>

// ============================================================================
// Pipeline configuration types
// ============================================================================

/** DOA method: 0 = Angle FFT (requires 8 TX), 1 = Bartlett Beamforming, 2 = Capon (MVDR) DoA-on-ROI */
#define RADAR_PIPELINE_DOA_ANGLE_FFT 0
#define RADAR_PIPELINE_DOA_BARTLETT 1
#define RADAR_PIPELINE_DOA_CAPON_ROI 2

/** Peaks reported per ROI by the Capon DoA (columns of the RADAR_PIPELINE_TENSOR_CAPON_* tensors). */
#define RADAR_PIPELINE_CAPON_NUM_PEAKS 3

/** Maximum number of targets in output list */
#define RADAR_PIPELINE_MAX_TARGETS 8192

/** Number of properties per target (range, velocity, azimuth, elevation, etc.) */
#define RADAR_PIPELINE_TARGET_FEATURES 8

/** Maximum DDM space entries (e.g. Front has 8) */
#define RADAR_PIPELINE_DDM_SPACE_MAX 8

/** Maximum calibration vector length (e.g. 8*8 = 64) */
#define RADAR_PIPELINE_CALIB_VECTOR_MAX 64

/** Maximum TX group entries and string length */
#define RADAR_PIPELINE_TX_GROUPS_MAX 9
#define RADAR_PIPELINE_TX_GROUP_LEN 20

/** Maximum number of TX and RX channels */
#define RADAR_PIPELINE_MAX_NUM_TX 8
#define RADAR_PIPELINE_MAX_NUM_RX 8

/** Maximum number of DDM Doppler offsets (e.g. RDFold for Front 16, Corner 8) */
#define RADAR_PIPELINE_MAX_NUM_DOPPLER_OFFSETS 16

/** General parameters (waveform, timing, derived coefficients) */
typedef struct
{
    double c;               /**< Speed of light (m/s) */
    int32_t NofSamples;     /**< Number of ADC samples per ramp */
    int32_t NofRamps;       /**< Number of ramps */
    int32_t NofRangeBins;   /**< Number of range bins */
    int32_t NofRx;          /**< Number of RX channels */
    int32_t NofTx;          /**< Number of TX channels */
    int32_t RDFold;         /**< Number of Doppler folds */
    int32_t NofRampsFold;   /**< NofRamps / RDFold */
    double EffBw;           /**< Effective bandwidth (Hz) */
    double DeltaF;          /**< Frequency step between ramps (Hz) */
    double Fstart;          /**< Start frequency (Hz) */
    double tPayloadUs;      /**< Chirp payload duration (us) */
    double fs;              /**< Sampling frequency (Hz) */
    double tWaitUs;         /**< Wait time (us) */
    double tPrePayloadUs;   /**< Pre-payload time (us) */
    double tPostPayloadUs;  /**< Post-payload time (us) */
    double tFlybackUs;      /**< Flyback time (us) */
    double tDelayUs;        /**< Delay (us) */
    int32_t NofAFFT;        /**< Angle FFT size */
    int32_t N_SNAP;         /**< Snapshot size */
    int32_t NofBinRJ;       /**< Range bin reject count */
    double DeltaR;          /**< Range bin step */
    double DeltaV;          /**< Velocity resolution (km/h) */
    double ChirpDwellTime;  /**< Chirp dwell time (us) */
    double PRI;             /**< Pulse repetition interval (us) */
    float ContR;            /**< Range coefficient */
    float ContV_fast;       /**< Velocity coefficient (fast) */
    float ContV_slow;       /**< Velocity coefficient (slow) */
    double DisE;            /**< Elevation antenna spacing (wavelengths) */
    double Dis;             /**< Azimuth antenna spacing (wavelengths) */
    double wavelength;      /**< Wavelength in m (speed_of_light / carrier_freq_Hz) */
    double sweepTime;       /**< Sweep time in seconds */
    double rangeResolution; /**< Range resolution in m */
    int32_t numTx;          /**< Valid entries in txPositions* / rxPositions* (Bartlett) */
    int32_t numRx;
    uint64_t txGroups[RADAR_PIPELINE_TX_GROUPS_MAX]; /**< TX group channel masks */
    int32_t txGroupsCount;
    float txPositionsAzimuth[RADAR_PIPELINE_MAX_NUM_TX];
    float txPositionsElevation[RADAR_PIPELINE_MAX_NUM_TX];
    float rxPositionsAzimuth[RADAR_PIPELINE_MAX_NUM_RX];
    float rxPositionsElevation[RADAR_PIPELINE_MAX_NUM_RX];
    float ddm_doppler_offsets[RADAR_PIPELINE_MAX_NUM_DOPPLER_OFFSETS];
    uint32_t ddm_doppler_offsets_count;
} RadarPipelineGP;

/** CFAR border handling */
typedef enum
{
    RADAR_PIPELINE_CFAR_BORDER_FULL_ENUM   = 0,
    RADAR_PIPELINE_CFAR_BORDER_CYCLIC_ENUM = 1
} RadarPipelineCfarBorder;

/** CFAR parameters. Threshold factors are derived from probabilityFA inside init(). */
typedef struct
{
    int32_t numHorTrain;                /**< Horizontal training cells */
    int32_t numHorGuard;                /**< Horizontal guard cells */
    int32_t numVerTrain;                /**< Vertical training cells */
    int32_t numVerGuard;                /**< Vertical guard cells */
    float probabilityFA;                /**< Probability of false alarm */
    uint32_t isHorizontalCyclicPadding; /**< 1 = cyclic padding in horizontal dimension */
    uint32_t isVerticalCyclicPadding;   /**< 1 = cyclic padding in vertical dimension */
    uint32_t enablePeakGrouping;        /**< 1 = enable peak grouping (local max filter) */
} RadarPipelineCfarParams;

/** Bartlett azimuth source (mirrors PVABartlettAzSource). */
typedef enum
{
    RADAR_PIPELINE_BARTLETT_AZ_FROM_NEIGHBORHOOD = 0,
    RADAR_PIPELINE_BARTLETT_AZ_FROM_EL0          = 1
} RadarPipelineBartlettAzSource;

/** Bartlett azimuth neighborhood (mirrors PVABartlettAzNeighborhood). */
typedef enum
{
    RADAR_PIPELINE_BARTLETT_AZ_NEIGHBORHOOD_0 = 0,
    RADAR_PIPELINE_BARTLETT_AZ_NEIGHBORHOOD_1 = 1,
    RADAR_PIPELINE_BARTLETT_AZ_NEIGHBORHOOD_2 = 2
} RadarPipelineBartlettAzNeighborhood;

/** Separable Bartlett params (mirrors PVASeparableBartlettParams). */
typedef struct
{
    uint32_t enableSeparable;
    RadarPipelineBartlettAzNeighborhood azNeighborhood;
    RadarPipelineBartlettAzSource azSource;
} RadarPipelineSeparableBartlettParams;

/** Bartlett beamforming angle-grid config. TX/RX positions come from RadarPipelineGP. */
typedef struct
{
    int32_t numAzimuthBins;   /**< Number of azimuth bins (0 = disable) */
    int32_t numElevationBins; /**< Number of elevation bins (0 = disable) */
    float azimuthStart;       /**< Azimuth start angle (degrees) */
    float azimuthStep;        /**< Azimuth step (degrees) */
    float elevationStart;     /**< Elevation start angle (degrees) */
    float elevationStep;      /**< Elevation step (degrees) */
} RadarPipelineBartlettConfig;

/** Target processing flags. Waveform coefficients come from RadarPipelineGP. */
typedef struct
{
    uint32_t fftShiftedDoppler;
    uint32_t enableRvDecoupling;
    uint32_t enableQuadraticInterpolation;
} RadarPipelineTargetProcessingParams;

/**
 * @brief Capon (MVDR) DoA-on-ROI config.
 *
 * Selected when RadarPipelineDOAParams::doa_method == RADAR_PIPELINE_DOA_CAPON_ROI, on an 8 TX x 8 RX
 * array, with an LLVM-capable PVA SDK. Per detection, the PVA ROI-beamspace preprocessing
 * (OpCaponROIPreprocess.h) projects the snapshots onto an roiAzSize x roiElSize beamspace ROI around the
 * coarse peak, and the PVA Capon operator (OpDoaCaponROI.h) reports the RADAR_PIPELINE_CAPON_NUM_PEAKS
 * strongest peaks on a fixed 16 (az) x 8 (el) steering grid spanning the ROI. Antenna spacings and DDM
 * offsets come from RadarPipelineGP.
 *
 * A field left at 0 takes its value from radarPipelineCaponConfigDefaults(); a negative value is rejected.
 * init() returns:
 *  - RADAR_PIPELINE_CREATE_ERR_DOA_METHOD when the PVA SDK has no LLVM support;
 *  - RADAR_PIPELINE_CREATE_ERR_DDM_OFFSETS when RadarPipelineGP has fewer DDM offsets than NofTx;
 *  - RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG for any other configuration the pipeline or either Capon
 *    operator does not support (the reason is printed to stderr).
 */
typedef struct
{
    /** ROI size in beamspace bins, azimuth x elevation; roiAzSize * roiElSize must be 16 (default 4 x 4). */
    int32_t roiAzSize;
    int32_t roiElSize;
    /** Snapshot rows per ROI: 16, 24, 32, 48, 56 or 64 (default 64). */
    int32_t numDopplerBinsCovMat;
    /** Local Doppler window around the detection (default 33): 2*floor(N/2) rows, doubled by enableFBA,
     *  must not exceed numDopplerBinsCovMat. */
    int32_t dopplerNeighborhood;
    uint32_t enableFBA;          /**< nonzero: forward-backward averaging (doubles the snapshot rows) */
    float adpDiagConst;          /**< adaptive diagonal loading of the covariance (default 1e-2) */
    float noiseFloorRatio;       /**< a peak is valid above noiseFloorRatio x the noise floor (default 10^0.2) */
    uint32_t enableSubBinInterp; /**< nonzero: quadratic sub-bin interpolation of the peak az/el bins */
    /** Maximum ROIs (detections) per frame, in [1, 1024] (default 1024); above 256 it needs a divisor in
     *  [ceil(maxRoi/256), 256]. Detections beyond maxRoi are dropped. The Capon operator takes no detection
     *  count, so it processes all maxRoi rows every frame: its cost scales with maxRoi, not with the
     *  detections. */
    int32_t maxRoi;
    /** RX layout: azimuth x elevation element counts, product NofRx, with RX channel rx at
     *  (az, el) = (rx % rxAzimuthElements, rx / rxAzimuthElements). Both 0: derived from the RadarPipelineGP
     *  RX positions, which must then place the NofRx elements on a full az x el grid in that order. The
     *  virtual azimuth aperture is rxAzimuthElements * NofTx. */
    int32_t rxAzimuthElements;
    int32_t rxElevationElements;
} RadarPipelineCaponConfig;

/** The Capon config defaults: a field left at 0 in RadarPipelineCaponConfig takes the value here. enableFBA,
 *  enableSubBinInterp (0 = off) and the RX split (0 = derive) have no other default. */
inline RadarPipelineCaponConfig radarPipelineCaponConfigDefaults()
{
    RadarPipelineCaponConfig c = {};
    c.roiAzSize                = 4;
    c.roiElSize                = 4;
    c.numDopplerBinsCovMat     = 64;
    c.dopplerNeighborhood      = 33;
    c.adpDiagConst             = 1e-2f;
    c.noiseFloorRatio          = 1.58489323f; /* 10^(2/10) */
    c.maxRoi                   = 1024;
    return c;
}

/** DOA params: method selection, Bartlett grid, Capon config, and target-processing flags. */
typedef struct
{
    /** RADAR_PIPELINE_DOA_ANGLE_FFT, RADAR_PIPELINE_DOA_BARTLETT or RADAR_PIPELINE_DOA_CAPON_ROI */
    uint32_t doa_method;
    uint32_t enableInterpolation;
    uint32_t enablePowerOutput;
    RadarPipelineBartlettConfig bartlett_config;
    RadarPipelineSeparableBartlettParams separableParams;
    RadarPipelineTargetProcessingParams targetProcessingParams;
    RadarPipelineCaponConfig capon_config; /**< used when doa_method == RADAR_PIPELINE_DOA_CAPON_ROI */
} RadarPipelineDOAParams;

/** Optional calibration / steering-vector data. */
typedef struct
{
    int16_t *steeringVectorsPtr;
    int32_t steeringVectorsSize;
} RadarPipelineCalibData;

/** ADC frame dimensions returned by IRadarPipeline::init(). */
typedef struct
{
    int32_t NofSamples; /**< ADC samples per ramp */
    int32_t NofRamps;
    int32_t NofRx;
} RadarPipelineAdcDimensions;

/** Top-level pipeline configuration passed to IRadarPipeline::init(). */
typedef struct
{
    RadarPipelineGP *gp;
    RadarPipelineCfarParams cfar_params;
    RadarPipelineDOAParams doa_params;
    RadarPipelineCalibData calib_data;
} RadarPipelineConfig;

/** Error codes returned by IRadarPipeline::init(). */
typedef enum
{
    RADAR_PIPELINE_CREATE_ERR_NULL_ARGS    = -1,
    RADAR_PIPELINE_CREATE_ERR_DDM_OFFSETS  = -2, /**< DDM offsets do not match RDFold (Capon: fewer than NofTx) */
    RADAR_PIPELINE_CREATE_ERR_DOA_METHOD   = -3,
    RADAR_PIPELINE_CREATE_ERR_PVA_WORKLOAD = -4,
    RADAR_PIPELINE_CREATE_ERR_EXCEPTION    = -5,
    RADAR_PIPELINE_CREATE_ERR_CAPON_CONFIG = -6 /**< Capon config or array not supported by the Capon operators */
} RadarPipelineError;

/** Intermediate tensor selection for get_intermediate_tensor(). */
typedef enum
{
    RADAR_PIPELINE_TENSOR_RANGE_PROFILE   = 0, /**< Range FFT output, complex int32 */
    RADAR_PIPELINE_TENSOR_DOPPLER_MAP     = 1, /**< Doppler FFT output, complex int32 */
    RADAR_PIPELINE_TENSOR_NCI_RX          = 2, /**< NCI per-RX output, uint32 */
    RADAR_PIPELINE_TENSOR_NCI_FINAL       = 3, /**< NCI combined output, uint32 */
    RADAR_PIPELINE_TENSOR_CFAR_DETECTIONS = 4, /**< CFAR folded detection list, int32 */
    RADAR_PIPELINE_TENSOR_DETECTION_LIST  = 5, /**< Unfolded detection list, int32 */
    RADAR_PIPELINE_TENSOR_SNAPSHOTS       = 6, /**< Snapshot extraction output, complex int32 */
    RADAR_PIPELINE_TENSOR_TARGET_LIST     = 7, /**< Final target list [8 x maxTargets], float */
    RADAR_PIPELINE_TENSOR_DETECTION_COUNT = 8, /**< CFAR detection count [1], int32 */
    RADAR_PIPELINE_TENSOR_TARGET_COUNT    = 9, /**< Final target count [1], int32 */
    /* Capon outputs (RADAR_PIPELINE_DOA_CAPON_ROI), PVA source only. The peak tensors are HW
     * [maxRoi, RADAR_PIPELINE_CAPON_NUM_PEAKS] float; only the first CAPON_ROI_COUNT rows belong to the
     * frame. An invalid peak has power -1.0f. */
    RADAR_PIPELINE_TENSOR_CAPON_PEAK_POWER = 10, /**< Capon peak power per ROI, strongest first */
    RADAR_PIPELINE_TENSOR_CAPON_AZ_BIN     = 11, /**< Capon fractional azimuth bin per ROI and peak */
    RADAR_PIPELINE_TENSOR_CAPON_EL_BIN     = 12, /**< Capon fractional elevation bin per ROI and peak */
    RADAR_PIPELINE_TENSOR_CAPON_ROI_COUNT  = 13, /**< Capon ROIs processed in the last frame [1], int32 */
    RADAR_PIPELINE_TENSOR_COUNT            = 14
} RadarPipelineTensorType;

/** Source selection for get_intermediate_tensor(). */
typedef enum
{
    RADAR_PIPELINE_SOURCE_PVA       = 0,
    RADAR_PIPELINE_SOURCE_REFERENCE = 1
} RadarPipelineTensorSource;

/** Element data type of an intermediate tensor. */
typedef enum
{
    RADAR_PIPELINE_DTYPE_INT16         = 0,
    RADAR_PIPELINE_DTYPE_INT32         = 1,
    RADAR_PIPELINE_DTYPE_UINT32        = 2,
    RADAR_PIPELINE_DTYPE_FLOAT32       = 3,
    RADAR_PIPELINE_DTYPE_COMPLEX_INT16 = 4,
    RADAR_PIPELINE_DTYPE_COMPLEX_INT32 = 5
} RadarPipelineDataType;

/** Tensor descriptor returned by get_intermediate_tensor(). */
typedef struct
{
    void *data;   /**< Host-accessible pointer */
    int32_t ndim; /**< Number of dimensions (1-4) */
    int64_t shape[4];
    int64_t strides[4]; /**< Stride in bytes per dimension */
    RadarPipelineDataType dtype;
    int64_t total_bytes;
} RadarPipelineTensorData;

// ============================================================================
// Abstract pipeline interface
// ============================================================================

/**
 * @brief Abstract base class for the radar pipeline.
 *
 * Concrete subclasses live inside their respective shared libraries and are
 * obtained via the per-library factory functions declared below.  The caller
 * owns the returned pointer and must delete it (or use std::unique_ptr).
 */
class IRadarPipeline
{
public:
    virtual ~IRadarPipeline() = default;

    /**
     * @brief Initialize the pipeline context and allocate PVA resources.
     *
     * Must be called exactly once before process(). On failure the object may
     * be deleted immediately; the destructor handles a partially-initialised state.
     */
    virtual int32_t init(const RadarPipelineConfig *config, RadarPipelineAdcDimensions *dims_out) = 0;

    /**
     * @brief Run the pipeline on one ADC frame. output_tensor may be nullptr to use the internal buffer.
     *
     * The target list goes to output_tensor; RADAR_PIPELINE_DOA_CAPON_ROI writes the Capon tensors instead
     * and leaves output_tensor untouched.
     */
    virtual int32_t process(NVCVTensorHandle input_tensor, NVCVTensorHandle output_tensor) = 0;

    /**
     * @brief Return the number of targets detected in the last process() call.
     *
     * This is the number of entries in the target list. RADAR_PIPELINE_DOA_CAPON_ROI produces no target
     * list (process() leaves output_tensor and RADAR_PIPELINE_TENSOR_TARGET_LIST unwritten), so the count is
     * 0: read RADAR_PIPELINE_TENSOR_CAPON_ROI_COUNT and the RADAR_PIPELINE_TENSOR_CAPON_* peak tensors
     * instead.
     */
    virtual int32_t get_target_count(int32_t *count) = 0;

    /**
     * @brief Run the CPU reference pipeline and compare every stage to the last PVA result.
     *
     * With RADAR_PIPELINE_DOA_CAPON_ROI the Capon stages are compared on the PVA stage inputs: the ROI
     * preprocessing (host marshalling and operator) against its CPU reference, and the Capon peaks against
     * the DoaCaponROI CPU reference run on the PVA ROI tensors.
     */
    virtual int32_t validate(void *input_pva_ptr) = 0;

    /** @brief Retrieve an intermediate tensor for debugging / visualization. */
    virtual int32_t get_intermediate_tensor(RadarPipelineTensorType type, RadarPipelineTensorSource src,
                                            RadarPipelineTensorData *data) = 0;

    /** @brief Return the library version string. */
    virtual const char *version() const = 0;
};

// ============================================================================
// Per-variant factory functions
// ============================================================================

extern "C" {

/** Create an 8T8R pipeline instance. Caller owns the returned pointer. */
IRadarPipeline *radar_pipeline_8t8r();

/** Create a 4T4R pipeline instance. Caller owns the returned pointer. */
IRadarPipeline *radar_pipeline_4t4r();

} // extern "C"

#endif // RADAR_PIPELINE_IFACE_HPP
