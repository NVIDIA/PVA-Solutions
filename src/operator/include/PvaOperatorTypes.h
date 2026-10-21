/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file PvaOperatorTypes.h
 *
 * @brief Defines types to be used by operators.
 * @defgroup PVA_OPERATOR_CORE_PVA_OPERATOR_TYPES PvaOperatorTypes
 * @{
 */

#ifndef PVA_OPERATOR_TYPES_H
#define PVA_OPERATOR_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Morphology Operation Codes
 */
typedef enum
{
    /// Erode operation
    PVA_ERODE = 0,
    /// Dilate operation
    PVA_DILATE = 1,
} PVAMorphologyType;

/**
 * @brief MixChannels Operation Codes
 *
 * Mix channels is a copy operation from a set of input channels to a set of output channels.
 *
 * The set of inputs and outputs may be given by any number of input and output images,
 * where each image may have one or more channels.
 *
 * The channel mapping from input to output is given by two arrays, one with the indices of
 * the input channels, and the another with the corresponding channel index in the output.
 *
 * The channel indices are enumerated starting from zero and increasing monotonically across
 * all channels of all images provided in the array. For example, given an array of 3 RGB
 * images, index 5 corresponds to the B channel of the second image.
 */
typedef enum
{
    /// split 1 rgba8 image into 4 u8 images, [0, 1, 2, 3] -> [0, 1, 2, 3]
    PVA_SPLIT_RGBA8_TO_U8 = 0,

    /// merge 4 u8 images into 1 rgba8 image, [0, 1, 2, 3] -> [0, 1, 2, 3]
    PVA_MERGE_U8_TO_RGBA8 = 1
} PVAMixChannelsCode;

/**
 * @enum PVADistanceType
 * @brief Defines the types of distance metrics for the DistanceTransform operator.
 *
 * Specifies the different distance calculation methods available.
 */
typedef enum
{
    /// Manhattan distance (L1 norm).
    PVA_DIST_L1 = 0,
    /// Euclidean distance (L2 norm).
    PVA_DIST_L2,
    /// Hamming distance.
    PVA_DIST_HAMMING
} PVADistanceType;

/**
 * @brief Interpolation method types for image operations.
 */
typedef enum
{
    /// Nearest neighbor
    PVA_INTERPOLATION_NN = 0,
    /// Bilinear interpolation
    PVA_INTERPOLATION_LINEAR
} PVAInterpolationType;

/**
 * @brief Warp perspective operator's transformation types
 */
typedef enum
{
    /// Perspective transformation
    PVA_WARP_TRANSFORMATION_PERSPECTIVE = 0,
    /// Affine transformation
    PVA_WARP_TRANSFORMATION_AFFINE = 1,
} PVAWarpTransformationType;

/**
 * @brief Brute Force Matcher operator's maximum matches per descriptor
 */
#define PVA_MAX_BF_MATCHES_PER_DESCRIPTOR (3)

/**
 * @brief Brute Force Matcher operator's maximum number of descriptors for either queries or references
 */
#define PVA_MAX_BF_DESCRIPTOR_COUNT (4096)

/**
 * @brief Brute Force Matcher operator's length of Brief Descriptor Array
 */
#define PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH (32)

/**
 * @brief Brute Force Matcher operator's structure containing the BRIEF Descriptor
 */
typedef struct
{
    /** Description vector of a BRIEF descriptor */
    uint8_t data[PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH];
} PVABriefDescriptor;

/**
 * @brief Structure containing match information for a single query descriptor
 */
typedef struct PVABFMatchesTypeRec
{
    /// Reference descriptor indices (-1 indicates no match).
    int16_t refIndex[PVA_MAX_BF_MATCHES_PER_DESCRIPTOR];

    /// Hamming distances to matched reference descriptors.
    int16_t distance[PVA_MAX_BF_MATCHES_PER_DESCRIPTOR];
} PVABFMatchesType;

/**
 * @brief Image Statistics operation flags
 */
typedef enum
{
    /// Calculate pixel count
    PVA_IMAGE_STAT_FLAG_PIXEL_COUNT = (1u << 0),
    /// Calculate per-channel sum
    PVA_IMAGE_STAT_FLAG_SUM = (1u << 1),
    /// Calculate per-channel mean, triggers PVA_IMAGE_STAT_FLAG_SUM and PVA_IMAGE_STAT_FLAG_PIXEL_COUNT
    PVA_IMAGE_STAT_FLAG_MEAN = ((1u << 2) | PVA_IMAGE_STAT_FLAG_SUM | PVA_IMAGE_STAT_FLAG_PIXEL_COUNT),
    /// Calculate per-channel variance, triggers PVA_IMAGE_STAT_FLAG_MEAN
    PVA_IMAGE_STAT_FLAG_VARIANCE = ((1u << 3) | PVA_IMAGE_STAT_FLAG_MEAN),
    /// Calculate full covariance matrix, triggers PVA_IMAGE_STAT_FLAG_VARIANCE
    PVA_IMAGE_STAT_FLAG_COVARIANCE = ((1u << 4) | PVA_IMAGE_STAT_FLAG_VARIANCE)
} PVAImageStatFlag;

/**
 * @brief Image Statistics output structure
 */
typedef struct
{
    /** @brief Number of pixels processed. */
    int pixelCount;
    /** @brief Per-channel sum of pixel values. Index corresponds to channel number. */
    float sum[4];
    /** @brief Per-channel mean of pixel values. Index corresponds to channel number. */
    float mean[4];
    /** @brief Covariance matrix.
     *         covariance[c][c] is the variance of channel c.
     *         covariance[c][c'] (for c != c') is the covariance between channel c and channel c'.
     */
    float covariance[4][4];
} PVAImageStatOutput;

/**
 * @brief Data types for the Gaussian Mixture Model (GMM) used in background subtraction.
 *
 * Specifies the available formats for storing GMM parameters such as weight, variance, and mean.
 */
typedef enum
{
    PVA_GMM_DATA_TYPE_FP32,
    PVA_GMM_DATA_TYPE_FP16
} PVAGMMDataType;

/**
 * @brief Batch FFT window types
 */
typedef enum
{
    /// Hanning window
    PVA_BATCH_FFT_WINDOW_HANNING,
    /// User defined window
    PVA_BATCH_FFT_WINDOW_USER_DEFINED,
} PVABatchFFTWindowType;

/**
 * @enum PVAFFTLayout
 * @brief Input and output layouts for the Range and Doppler FFTs; each operator supports a subset.
 */
typedef enum
{
    /// Doppler bins x Rx channels x Range bins
    PVA_FFT_LAYOUT_DOPPLER_RX_RANGE = 0,
    /// Range bins x Rx channels x Doppler bins
    PVA_FFT_LAYOUT_RANGE_RX_DOPPLER = 1,
    /// Range bins x Doppler bins x Rx channels
    PVA_FFT_LAYOUT_RANGE_DOPPLER_RX = 2,
    /// Rx channels x Range bins x Doppler bins
    PVA_FFT_LAYOUT_RX_RANGE_DOPPLER = 3,
    /// Rx channels x Doppler bins x Range bins
    PVA_FFT_LAYOUT_RX_DOPPLER_RANGE = 4,
} PVAFFTLayout;

/**
 * @brief Range FFT parameters
 */
typedef struct PVARangeFFTParamsRec
{
    /// Window type
    PVABatchFFTWindowType windowType;
    /// Input layout
    PVAFFTLayout inputLayout;
    /// Output layout
    PVAFFTLayout outputLayout;
} PVARangeFFTParams;

/**
 * @brief Doppler FFT parameters
 */
typedef struct PVADopplerFFTParamsRec
{
    /// Window type
    PVABatchFFTWindowType windowType;
    /// Input layout
    PVAFFTLayout inputLayout;
    /// Output layout
    PVAFFTLayout outputLayout;
} PVADopplerFFTParams;

/**
 * Coherent Integration operator configuration.
 */
typedef struct PVARadarCoherentIntegrationParamsRec
{
    /** Number of stride-separated Doppler values accumulated per folded bin. */
    int32_t repeatFold;

    /**
     * Input range-Doppler ordering. Supported values are
     * @ref PVA_FFT_LAYOUT_RANGE_RX_DOPPLER and
     * @ref PVA_FFT_LAYOUT_RANGE_DOPPLER_RX.
     */
    PVAFFTLayout inputLayout;
} PVARadarCoherentIntegrationParams;

/**
 * @enum PVAFlipDirection
 * @brief Flip direction options for image flip operation
 */
typedef enum
{
    /// Flip image horizontally (left-right)
    PVA_FLIP_HORIZONTAL = 0,
    /// Flip image vertically (up-down)
    PVA_FLIP_VERTICAL = 1,
    /// Flip image both horizontally and vertically (180° rotation)
    PVA_FLIP_BOTH = 2
} PVAFlipDirection;

/**
 * @brief Radar CFAR algorithm types for target detection.
 *
 * This enumeration defines the available Constant False Alarm Rate (CFAR) algorithms
 * that can be used for radar target detection. Each algorithm provides different
 * approaches to estimate the noise floor and set adaptive thresholds for target detection.
 */
typedef enum
{
    /** @brief Cell Averaging (CA) algorithm - Thresholds are set by averaging the signal values from both leading and trailing training cells */
    PVA_CFAR_CA = 0,
    /** @brief Order Statistic (OS) algorithm - Thresholds are set using the median value of the ranked signal amplitudes from both leading and trailing training cells */
    PVA_CFAR_CA_OS = 1,
    /** @brief Greatest Of (GO) algorithm - Thresholds are set using the greater of the averages from the leading and trailing training cells */
    PVA_CFAR_CA_GO = 2,
    /** @brief Smallest Of (SO) algorithm - Thresholds are set using the smaller of the averages from the leading and trailing training cells */
    PVA_CFAR_CA_SO = 3
} PVARadarCFARType;

/** @brief Configuration parameters for radar CFAR target detection operator.
 *
 * This structure contains all the necessary parameters to configure the Constant False Alarm Rate (CFAR)
 * algorithm for radar target detection. The parameters control the detection sensitivity, algorithm type,
 * and processing window dimensions for optimal target detection performance.
 *
 * The CFAR algorithm uses training cells to estimate the noise floor and guard cells to protect
 * the cell under test (CUT) from target energy leakage, ensuring accurate threshold calculation.
 *
 * CFAR Cell Arrangement:
 * N = numTrain
 * K = numGuard
 *
 * @verbatim
 * | T1 | T2 | T3 |......| TN | G1 | ... | GK | CUT | G1 | ... | GK | T1 | T2 | T3 |......| TN |
 * |<-------- Trailing ------>|<--- Guard --->|     |<--- Guard --->|<-------- Leading ------->|
 * @endverbatim
 */
/**
 * @brief Selects the mode for the horizontal CFAR threshold factor.
 *
 * Passed as the @c kind field of @ref PvaRadarCFARHorizontalThreshold.
 */
typedef enum
{
    PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM = 0, /**< Single factor applied uniformly to all range rows */
    PVA_CFAR_HORIZONTAL_THRESHOLD_PER_ROW = 1, /**< One factor per range row (range-adaptive) */
} PvaRadarCFARHorizontalThresholdType;

/**
 * @brief Horizontal CFAR threshold specification.
 *
 * Set @c type to @c PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM and fill @c uniform with a single multiplier to use the
 * same detection threshold across every range row.
 *
 * Set @c type to @c PVA_CFAR_HORIZONTAL_THRESHOLD_PER_ROW and point @c perRow at an array of per-row multipliers
 * for range-adaptive false-alarm control.  The array length must equal the number of range rows in the
 * CFAR input map; the operator infers that length from its configured dimensions.
 *
 * The threshold applied to each cell during the horizontal pass is:
 * @code
 *   detectionThreshold = noiseEstimate * factor
 * @endcode
 * where @c factor is either @c uniform or @c perRow[rowIndex].  Larger factors raise the threshold and
 * reduce sensitivity (fewer detections).
 */
typedef struct PvaRadarCFARHorizontalThresholdRec
{
    PvaRadarCFARHorizontalThresholdType type; /**< Selects uniform or per-row mode */
    union
    {
        float uniform;       /**< Used when @c type == PVA_CFAR_HORIZONTAL_THRESHOLD_UNIFORM */
        const float *perRow; /**< Used when @c type == PVA_CFAR_HORIZONTAL_THRESHOLD_PER_ROW */
    } value;                 /**< Union holding either the uniform factor or the per-row factor array */
} PvaRadarCFARHorizontalThreshold;

/**
 * @brief Parameters for the PVA Radar CFAR (Constant False Alarm Rate) operator.
 *
 * Configures the 2-D CFAR detection algorithm, including training and guard cell counts,
 * detection thresholds, padding modes, and optional peak grouping.
 */
typedef struct PvaRadarCFARParamsRec
{
    /** Number of horizontal training cells for CFAR algorithm (range: 0 <= numHorGuard < numHorTrain <= (Width-1)/2)
     * Constraint: 0 <= numHorGuard < numHorTrain < 256
     * Constraint: 1 <= numHorTrain + numHorGuard < 256
     * Training cells are used to estimate the noise floor in the horizontal direction
     */
    int32_t numHorTrain;
    /** Number of horizontal guard cells for CFAR algorithm (range: 0 <= numHorGuard < numHorTrain)
     * Constraint: 0 <= numHorGuard < numHorTrain < 256
     * Guard cells protect the cell under test from target energy leakage in horizontal direction
     */
    int32_t numHorGuard;
    /** Number of vertical training cells for CFAR algorithm (range: 0 <= numVerGuard < numVerTrain <= (Height-1)/2)
     * Constraint: 0 <= numVerGuard < numVerTrain < 256
     * Constraint: 1 <= numVerTrain + numVerGuard < 256
     * Training cells are used to estimate the noise floor in the vertical direction
     */
    int32_t numVerTrain;
    /** Number of vertical guard cells for CFAR algorithm (range: 0 <= numVerGuard < numVerTrain)
     * Constraint: 0 <= numVerGuard < numVerTrain < 256
     * Guard cells protect the cell under test from target energy leakage in vertical direction
     */
    int32_t numVerGuard;
    /** Horizontal CFAR threshold specification.
     *
     * \sa PvaRadarCFARHorizontalThreshold
     */
    PvaRadarCFARHorizontalThreshold horizontalThreshold;
    /** CFAR vertical threshold multiplier
     * Controls detection sensitivity - higher values increase detection threshold
     * detectionThreshold = noiseEstimate * thresholdFactor
     * noiseEstimate is the noise estimate in the vertical direction
     */

    float verticalThresholdFactor;
    /** Controls padding mode for horizontal dimension during CFAR processing.
     *
     * When enabled (true):
     *   - Uses cyclic padding, Uses the training cells from the opposite edge for boundary training cells.
     *   - Recommended when horizontal data exhibits periodic/cyclic characteristics or when
     *     background noise and clutter properties are assumed to be consistent across edges.
     *
     * When disabled (false):
     *   - Uses available neighbouring pixels for noise estimation at boundaries.
     *   - Training cells beyond image boundaries are not included in noise estimation.
     *   - Recommended when the signal represents an event that is known to start and end within the collected
     *     data window, with a genuinely zero-level background outside that window.
     */
    bool isHorizontalCyclicPadding;
    /** Controls padding mode for vertical dimension during CFAR processing.
     *
     * When enabled (true):
     *   - Uses cyclic padding, Uses the training cells from the opposite edge for boundary training cells.
     *   - Recommended when vertical data exhibits periodic/cyclic characteristics or when
     *     background noise and clutter properties are assumed to be consistent across edges.
     *
     * When disabled (false):
     *   - Uses available neighbouring pixels for noise estimation at boundaries.
     *   - Training cells beyond image boundaries are not included in noise estimation.
     *   - Recommended when the signal represents an event that is known to start and end within the collected
     *     data window, with a genuinely zero-level background outside that window.
     */
    bool isVerticalCyclicPadding;
    /** Enables peak grouping algorithm for target detection refinement
     * When enabled, the algorithm filters detections to retain only local maxima by comparing
     * each detection against its immediate horizontal and vertical neighbors.
     */
    bool enablePeakGrouping;
} PvaRadarCFARParams;

/**
 * @brief CCL Connectivity Types
 */
typedef enum
{
    /// 4-connectivity
    PVA_4_CONNECTIVITY = 0,
    /// 8-connectivity
    PVA_8_CONNECTIVITY = 1,
} PVAConnectivityType;

/**
 * @brief Parameters for the Snapshot Extraction operator.
 */
typedef struct
{
    /// Number of Doppler folds for DDM disambiguation
    int32_t numDopplerFolds;
    /// Range-Doppler map memory layout format
    /// Use PVA_FFT_LAYOUT_RANGE_RX_DOPPLER or PVA_FFT_LAYOUT_RANGE_DOPPLER_RX
    PVAFFTLayout rangeDopplerFormat;
} PVASnapshotExtractionParams;

/**
 * @brief Azimuth source selection for Separable Bartlett beamforming.
 *
 * Controls where the final azimuth estimate comes from in the two-step search.
 */
typedef enum
{
    /// Final azimuth and elevation both come from Step 2 (elevation scan).
    /// The joint peak of the local (az neighborhood × elevation) patch is used.
    PVA_BARTLETT_AZ_FROM_NEIGHBORHOOD = 0,
    /// Azimuth comes from Step 1 (azimuth scan at el ≈ 0°); only elevation
    /// is taken from Step 2. Can give better azimuth accuracy when targets
    /// are near the horizon; may degrade at steep elevation angles.
    PVA_BARTLETT_AZ_FROM_EL0 = 1
} PVABartlettAzSource;

/**
 * @brief Azimuth neighborhood size for Separable Bartlett beamforming.
 *
 * Controls how many azimuth bins around the Step 1 peak are searched in Step 2.
 * At boundaries the window is clamped to valid indices.
 */
typedef enum
{
    /// Only peak azimuth bin, complexity O(Naz + Nel)
    PVA_BARTLETT_AZ_NEIGHBORHOOD_0 = 0,
    /// ±1 bins (3 azimuth bins), complexity O(Naz + 3*Nel)
    PVA_BARTLETT_AZ_NEIGHBORHOOD_1 = 1,
    /// ±2 bins (5 azimuth bins), complexity O(Naz + 5*Nel)
    PVA_BARTLETT_AZ_NEIGHBORHOOD_2 = 2
} PVABartlettAzNeighborhood;

/**
 * @brief Parameters for Separable Bartlett beamforming (sequential 1D DOA estimation).
 *
 * Decomposes the full 2D azimuth–elevation search into two sequential 1D scans:
 *   - Step 1 (Azimuth scan): Beamform over all azimuth bins at elevation ≈ 0°.
 *     Find the coarse azimuth peak.
 *   - Step 2 (Elevation scan): Beamform over all elevation bins using a small
 *     azimuth neighborhood (±N bins) around the Step 1 peak. The joint peak of
 *     this local 2D patch yields the final DOA estimate.
 *
 * Complexity: O(Naz × Nel) → O(Naz + (2N+1) × Nel),  where N = azNeighborhood.
 *
 * \b Constraints:
 *      - numAzimuthBins <= 256
 *      - numElevationBins <= 255
 */
typedef struct
{
    /// Enable separable (two-step) search instead of full 2D grid search.
    /// When false (default), performs exhaustive search over all Az × El bins.
    bool enableSeparable;

    /// Half-width of the azimuth neighborhood used in Step 2 (elevation scan).
    /// Step 2 evaluates elevation at (2N+1) azimuth bins centered on the Step 1
    /// peak, forming a narrow 2D patch that improves elevation accuracy.
    ///   0 → peak bin only       (1 az bin,  complexity ≈ Naz + Nel)
    ///   1 → peak ± 1 bin        (3 az bins, complexity ≈ Naz + 3·Nel)
    ///   2 → peak ± 2 bins (max) (5 az bins, complexity ≈ Naz + 5·Nel)
    PVABartlettAzNeighborhood azNeighborhood;

    /// Controls which step supplies the final azimuth estimate.
    /// See \ref PVABartlettAzSource.
    PVABartlettAzSource azSource;

    /// Channel mask for Step 1 (azimuth scan). Bitmask over virtual channels.
    /// 0 (default): mask disabled — all channels participate in the azimuth scan.
    /// Non-zero: bit i = 1 keeps virtual channel i; bit i = 0 zeroes it out.
    /// Step 2 (elevation scan) always uses all channels regardless of this mask.
    /// Example: 0xFF retains only VCs 0–7 in a 64-channel array.
    uint64_t azChannelMask;
} PVASeparableBartlettParams;

/**
 * @brief Parameters for the Bartlett Beamforming DOA + Target Processing operator.
 *
 * This structure combines parameters for DOA (Direction of Arrival) angle estimation
 * using Bartlett beamforming and target processing. The operator performs beamforming
 * to estimate azimuth/elevation angles, then converts the radar measurements into
 * physical coordinates and velocities.
 *
 * \b Important:
 *
 *      Quadratic interpolation (enableInterpolation = true) is strongly recommended
 *      for coarse bin spacing to achieve sub-degree accuracy.
 *      Without interpolation, angle accuracy degrades significantly
 */
typedef struct
{
    // Bartlett Beamforming parameters
    /// Enable quadratic interpolation for sub-bin accuracy
    bool enableInterpolation;
    /// Enable peak power (dB) output in target list
    bool enablePowerOutput;

    // Separable Bartlett parameters (optional optimization)
    /// When separableParams.enableSeparable is true, uses 2-step search for faster processing
    PVASeparableBartlettParams separableParams;
} PVADoaBartlettBeamformingParams;

/**
 * @brief Parameters for the Capon (MVDR) ROI DoA operator.
 *
 * The operator uses a 4 x 4 beamspace ROI.
 */
typedef struct
{
    /// Number of azimuth bins per ROI. Must be 16.
    uint16_t numAzBins;

    /// Number of elevation bins per ROI. Must be 8.
    uint16_t numElBins;

    /// Beamspace steering-vector dimension. Must be 16.
    uint16_t svDim;

    /// Number of peaks reported per ROI. Must be 3.
    uint8_t numPeaksPerROI;

    /// Fractional bits in the 2S32 snapshot input. Must be in [0, 30].
    uint32_t inputQBits;

    /// Fractional bits in the 2S32 steering-vector input. Must be in [0, 30].
    uint32_t steeringQBits;

    /// Finite, nonnegative adaptive diagonal-loading coefficient. A typical value is 1e-2f.
    float adpDiagConst;

    /// Finite, positive peak-validity threshold relative to the estimated noise floor.
    float noiseFloorRatio;

    /// Number of Doppler bins per ROI used for covariance matrix computation.
    /// Must equal the snapshot tensor's H dimension.
    /// A nonzero remainder after division by 32 must be divisible by 4 and at least 12.
    int32_t numDopplerBinsCovMat;

    /// Use the floating-point covariance kernel instead of the S32 kernel.
    bool useFpCovMatKnl;

    /// Enable sub-bin interpolation for azimuth and elevation bin indices.
    bool enableSubBinInterp;

} PVADoaCaponROIParams;

/**
 * @brief Peak-power value that marks an invalid peak in the Capon ROI DoA output.
 *
 * Valid peaks always have positive power, so a consumer can test `power < 0.0f`.
 */
#define PVA_DOA_CAPON_ROI_INVALID_PEAK_POWER (-1.0f)

/**
 * @brief Parameters for the Capon DoA-on-ROI beamspace-projection preprocessing operator.
 *
 * Shared host/device GP struct (passed via prog["GP"].set). Geometry is fixed for a run,
 * so the DFT sizes are precomputed on host and passed here. See OpCaponROIPreprocess.h.
 */
typedef struct
{
    /// transmit channels
    int32_t numTx;
    /// receive channels
    int32_t numRx;
    /// Doppler dimension of the range-Doppler cube
    int32_t numChirps;
    /// virtual azimuth aperture, usually numTx times the azimuth RX element count
    int32_t numAzVirt;
    /// virtual elevation aperture, usually the elevation RX element count
    int32_t numElVirt;
    /// beamspace azimuth bins (coarse-DoA DFT length); independent of numAzVirt
    int32_t beamAz;
    /// beamspace elevation bins (coarse-DoA DFT length); independent of numElVirt
    int32_t beamEl;
    /// ROI azimuth size in beamspace bins
    int32_t roiAz;
    /// ROI elevation size in beamspace bins (svDim = roiAz*roiEl)
    int32_t roiEl;
    /// beamspace channels per ROI = roiAz*roiEl (at most 16)
    int32_t svDim;
    /// steering-grid azimuth directions per ROI (1 to 16)
    int32_t numAzBins;
    /// steering-grid elevation directions per ROI (1 to 8)
    int32_t numElBins;
    /// snapshot height H (FBA-expanded Doppler count)
    int32_t numDopplerBinsCovMat;
    /// local Doppler window, >= 2 (effective = 2*floor(N/2))
    int32_t dopplerNeighborhood;
    /// true: forward-backward averaging (doubles the snapshot count)
    bool enableFBA;
    /// snapshotsROI fixed-point Q, in [0, 30]
    int32_t inputQBits;
    /// steeringLut / steeringVectorsROI fixed-point Q, in [0, 30]
    int32_t steeringQBits;
    /// max detections (ROIs) per frame: the row count of the per-detection tensors. Submit's
    /// detectionCount tensor says how many of them are processed.
    int32_t maxDet;

    /// Host-precomputed DDM Doppler shift per TX in bins, typically round((ddm[0]-ddm[tx])*numChirps); any
    /// value, applied modulo numChirps. Indexed [0, numTx).
    int32_t relDopplerOffset[8];
    /// DDM decode weight, real part: typically cos(2*pi*ddm[tx]), times any per-TX calibration. Indexed [0, numTx);
    /// must be finite.
    float ddmaWeightRe[8];
    /// DDM decode weight, imag part: typically sin(2*pi*ddm[tx]), times any per-TX calibration. Indexed [0, numTx);
    /// must be finite.
    float ddmaWeightIm[8];

    /// true: run coarse DoA as a batched FFT-64 (16 detections per run) instead of the direct DFT.
    /// Requires numElVirt == 1, beamAz == 64, numAzVirt == 64, the generated twiddles, no coarse-DoA
    /// grid and the full coarse-DoA field of view. false (default): direct DFT.
    bool coarseDoAUseFft;

    // Coarse-DoA position grid (optional; for sparse, non-rectangular virtual arrays)
    /// Height of the coarse-DoA position grid, onto which a sparse array is zero-filled to make it
    /// separable: snapshots become [numAzVirt*coarseElVirt] with position (x,y) at y*numAzVirt + x,
    /// and numElVirt must be 1; 0 (default) = use numElVirt, i.e. no separate grid.
    int32_t coarseElVirt;

    // Coarse-DoA field of view: inclusive bins in shifted output-bin space, all four zero
    // (default) = whole beamspace, narrowing requires the direct-DFT path.
    /// First azimuth bin searched.
    int32_t coarseAzBinLo;
    /// Last azimuth bin searched; beamAz-1 is the full range.
    int32_t coarseAzBinHi;
    /// First elevation bin searched.
    int32_t coarseElBinLo;
    /// Last elevation bin searched; beamEl-1 is the full range.
    int32_t coarseElBinHi;

    /// false (default): Create() generates the DFT tables waz[ka][a] = exp(-j*2*pi*ka*a/beamAz) and
    /// wel[ke][e] = exp(-j*2*pi*ke*e/beamEl), and the waz/wel tensors are ignored (may be NULL).
    /// true: the caller passes the tables (e.g. calibrated steering vectors); not with coarseDoAUseFft.
    /// Table row k is beamspace output bin (k + beam/2) % beam.
    bool useCustomTwiddles;
} PVACaponROIPreprocessParams;

/* =========================================================================
 * Custom Radar Operator Types (DOA DA FFT, Peak Detection, DOA Angle FFT,
 * Target Processing)
 * ========================================================================= */

/**
 * @brief Number of receive antennas (4T4R radar configuration)
 */
#define PVA_RADAR_RX_ANTENNA_COUNT (4)

/**
 * @brief Number of transmit antennas (4T4R radar configuration)
 */
#define PVA_RADAR_TX_ANTENNA_COUNT (4)

/**
 * @brief Number of DDM Doppler folds
 */
#define PVA_RADAR_DOPPLER_FOLD_COUNT (8)

/**
 * @brief Maximum number of targets
 */
#define PVA_RADAR_MAX_TARGET_COUNT (8192)

/**
 * @brief Peak Detection operator's number of peak indices (3 values: range bin index, folded doppler bin index, and unfolded doppler bin index)
 */
#define PVA_RADAR_PEAKDET_NUM_PEAK_INDICES (3)

/**
 * @brief Number of target detection properties (7 values: velocity, range, azimuth, elevation, x, y, and z)
 */
#define PVA_RADAR_NUM_TARGET_DETECTION_PROPERTIES (7)

/**
 * @brief Number of upper or lower antenna elements in the radar array aperture
 */
#define PVA_RADAR_NUM_ANTENNA_ELEMENTS (8)

/**
 * @brief Number of total antenna elements
 */
#define PVA_RADAR_NUM_TOTAL_ANTENNA_ELEMENTS (2 * PVA_RADAR_NUM_ANTENNA_ELEMENTS)

/**
 * @brief Radar processing parameters structure for DOA (Direction of Arrival) target processing
 *
 * This structure contains all the geometric and signal processing parameters required
 * for radar DOA target processing operations. It defines the radar system configuration,
 * array geometry, and processing parameters used in the target detection and localization
 * algorithms.
 */
typedef struct
{
    /** @brief Number of range bins in the radar system */
    int32_t nr;
    /** @brief Number of Doppler fold repetitions for velocity ambiguity resolution */
    int32_t repeatFold;
    /** @brief Number of snapshots used for array processing */
    int32_t nofSnap;
    /** @brief Number of antenna elements in the array aperture */
    int32_t nofAperture;
    /** @brief Number of FFT points for angle processing (typically 256) */
    int32_t nofAFFT;
    /** @brief Inter-element spacing in the antenna array (in wavelengths) */
    float dis;
    /** @brief Sampling frequency of the radar system (Hz) */
    float fs;
    /** @brief Number of samples per chirp/ramp */
    int32_t nofSamples;
    /** @brief Pulse Repetition Interval (seconds) */
    float PRI;
    /** @brief Number of chirp ramps for Doppler processing */
    int32_t nofRamps;
    /** @brief Velocity resolution (m/s) */
    float deltaV;
    /** @brief Array indices for upper antenna elements in DOA processing */
    int32_t apUpperInd[PVA_RADAR_NUM_ANTENNA_ELEMENTS];
    /** @brief Snapshot indices for upper antenna elements */
    int32_t apUpper[PVA_RADAR_NUM_ANTENNA_ELEMENTS];
    /** @brief Array indices for lower antenna elements in DOA processing */
    int32_t apLowerInd[PVA_RADAR_NUM_ANTENNA_ELEMENTS];
    /** @brief Snapshot indices for lower antenna elements */
    int32_t apLower[PVA_RADAR_NUM_ANTENNA_ELEMENTS];
    /** @brief Elevation inter-element spacing (in wavelengths) */
    float disE;
    /** @brief Fast velocity conversion factor for range-Doppler mapping */
    float contVFast;
    /** @brief Slow velocity conversion factor for range-Doppler mapping */
    float contVSlow;
    /** @brief Range conversion factor for range-Doppler mapping */
    float contR;
    /** @brief Number of Q bits for the snapshot tensor */
    int32_t calibVectorQbits;
    /** @brief Number of Q bits for the FFT tensor */
    int32_t fftQbits;
} PVARadarGP;

/**
 * @brief Angle estimation precision levels for DOA Angle FFT operator.
 *
 * This enum controls the zero-padding applied to the virtual antenna array
 * snapshot before performing the spatial FFT for angle estimation. The precision
 * level determines the FFT scaling factor (zero-padding factor), which affects
 * the density of the frequency grid in the FFT output.
 *
 * Zero-padding produces a denser frequency grid, which is equivalent to sinc
 * interpolation in the angle domain. This enables more precise localization of
 * spectral peaks without increasing the fundamental angular resolution (which
 * is determined solely by the physical antenna aperture).
 *
 * The FFT lengths are computed as:
 *   - Azimuth FFT length:   numVirtualAzimuthElements * scalingFactor
 *   - Elevation FFT length: numVirtualElevationElements * scalingFactor
 */
typedef enum
{
    PVA_ANGLE_ESTIMATION_PRECISION_LOW     = 0, ///< Minimal zero-padding, fastest
    PVA_ANGLE_ESTIMATION_PRECISION_DEFAULT = 1, ///< Balanced precision and performance
    PVA_ANGLE_ESTIMATION_PRECISION_HIGH    = 2  ///< Maximum precision, slowest
} PVADoaAngleFFTPrecision;

/**
 * @brief Parameters for the Target Processing operator.
 *
 * This structure contains parameters for target processing, including the standard
 * radar parameters and optional RV (Range-Velocity) decoupling parameters.
 *
 * RV Decoupling:
 * When enableRvDecoupling is true, the operator uses FMCW coupling formulas to
 * correct for the range-velocity coupling in FMCW radar systems. This provides
 * more accurate range and velocity estimates compared to the standard method.
 *
 * The RV decoupling algorithm uses a 5-hypothesis velocity disambiguation loop.
 */
typedef struct
{
    /// RF wavelength (meters)
    float wavelength;
    /// Chirp sweep time (seconds)
    float sweepTime;
    /// Range bin resolution (meters)
    float rangeResolution;
    /// Number of chirps
    int32_t numChirps;
    /// Flag to specify if Doppler FFT is shifted to make zero velocity Doppler bin the center of the FFT bins
    bool fftShiftedDoppler;
    /// Enable range-velocity decoupling
    bool enableRvDecoupling;
    /// Enable quadratic interpolation for sub-bin accuracy (requires NCI data)
    bool enableQuadraticInterpolation;

    // =========================================================================
    // RV Decoupling Parameters (only used when enableRvDecoupling is true)
    // Base FMCW radar parameters - derived values are computed internally
    // by the operator (following the DOA operator pattern).
    // =========================================================================

    /// Sampling frequency (Hz) - used to compute f_fast_delta = fs / numSamples
    float fs;
    /// Number of samples per chirp - used with fs for f_fast_delta
    int32_t nofSamples;
    /// Pulse Repetition Interval (seconds) - used for f_slow_delta and tempVCoeff
    float PRI;
    /// Velocity resolution (km/h) - used for velocityAmbHigh/Low computation (converted to m/s internally)
    float deltaV;
    /// Range coupling constant: contR = c / (2 * slope) (meters per Hz)
    float contR;
    /// Velocity fast constant for FMCW coupling (m/s per Hz)
    float contVFast;
    /// Velocity slow constant for FMCW coupling (m/s per Hz)
    float contVSlow;

    // =========================================================================
    // Coordinate System Options (for spherical-to-Cartesian conversion)
    // =========================================================================

    /// Longitudinal axis convention for Cartesian coordinates
    /// 0 = X_FORWARD: X is forward (standard spherical), Y is lateral
    /// 1 = Y_FORWARD: Y is forward (automotive radar convention), X is lateral
    int32_t longitudinalAxis;
    /// Ground projection mode
    /// 0 = SLANT_RANGE: X/Y use slant range directly (no cos(elevation) factor)
    /// 1 = GROUND_PLANE: X/Y scaled by cos(elevation) for ground-plane position
    int32_t groundProjection;
} PVATargetProcessingParams;

/**
 * @brief Parameters for the DOA Angle FFT and Target Processing operator.
 *
 * This structure combines parameters for DOA (Direction of Arrival) angle estimation
 * and target processing. The operator performs 2D Angle FFT to estimate azimuth/elevation
 * angles, then converts the radar measurements into physical coordinates and velocities.
 */
typedef struct
{
    // DoA Angle FFT parameters
    /// Number of virtual horizontal (azimuth) antenna elements (assumed half-wavelength spacing)
    int32_t numVirtualAzimuthElements;
    /// Number of virtual vertical (elevation) antenna elements (assumed half-wavelength spacing)
    int32_t numVirtualElevationElements;
    /// Angle estimation precision level.
    ///
    /// Controls the zero-padding applied to the virtual antenna array snapshot
    /// before performing the spatial FFT for angle estimation. See @ref PVADoaAngleFFTPrecision
    /// for details on precision levels and their corresponding scaling factors.
    ///
    /// The scaling factor determines the FFT lengths:
    ///   - Azimuth FFT length:   numVirtualAzimuthElements * scalingFactor
    ///   - Elevation FFT length: numVirtualElevationElements * scalingFactor
    ///
    /// Higher precision provides finer angle bin spacing, improving peak detection
    /// accuracy at the cost of increased computation.
    PVADoaAngleFFTPrecision angleEstimationPrecision;
    /// Enable local peak detection to find multiple targets per detection
    bool enableLocalPeakDetection;
} PVADoaAngleFFTParams;

/**
 * @brief Create-time configuration for the Bundle Adjustment operator.
 *
 * These values size the on-device active-camera indexing (AGEN configuration) and the internal
 * scratch DRAM buffers, so they are fixed for the lifetime of the operator instance and cannot be
 * changed at Submit time.
 */
typedef struct
{
    /// Number of cameras solved jointly for pose refinement in the reduced (Schur-complement) camera
    /// system. Must satisfy 1 <= activeCameraCount <= kActiveCameraCount (5).
    int32_t activeCameraCount;
    /// Index of the first camera included in the active-camera window. Combined with the Submit-time
    /// cameraCount, activeCameraBegin + activeCameraCount must not exceed cameraCount.
    int32_t activeCameraBegin;
} PvaBundleAdjustmentConfig;

/**
 * @brief Submit-time Levenberg-Marquardt tuning parameters for the Bundle Adjustment operator.
 */
typedef struct
{
    /// Total number of cameras in the rig (<= kMaxCameraCount, 21).
    int32_t cameraCount;
    /// Total number of tracks (3D points) being refined. Must be an exact multiple of kTrackCountTile
    /// (128); partial/remainder tiles are not supported.
    int32_t trackCount;
    /// Maximum number of Levenberg-Marquardt outer iterations to run.
    int32_t maxIterations;
    /// Initial Levenberg-Marquardt damping factor.
    float lambdaInit;
    /// Minimum admissible reprojection distance used to guard against near-zero-depth divisions.
    float minDistance;
    /// Robustifier scale applied to the measurement residual weighting.
    float robustifierScaleMeas;
    /// Inverse measurement-noise weighting applied after the robustifier scale.
    float rinvMeas;
} PvaBundleAdjustmentParams;

#ifdef __cplusplus
}
#endif

/** @} */
#endif /* PVA_OPERATOR_TYPES_H */
