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

#include "radar_pipeline_config.hpp"
#include "radar_pipeline_context.hpp"
#include "radar_pipeline_iface.hpp"
#include "radar_pipeline_pva.hpp"
#include "radar_pipeline_reference.hpp"
#include "radar_pipeline_tensors.hpp"

#include <TensorUtils.hpp>

#include <ErrorCheckMacros.h>
#include <cupva_host.h>

#include <climits>
#include <cstring>
#include <memory>
#include <string>

using namespace pva_solutions;

// ============================================================================
// RadarPipeline4t4r — concrete IRadarPipeline for the 4T4R variant
// ============================================================================

class RadarPipeline4t4r final : public IRadarPipeline
{
public:
    RadarPipeline4t4r() = default;

    ~RadarPipeline4t4r() override
    {
        delete ctx_;
    }

    int32_t init(const RadarPipelineConfig *config, RadarPipelineAdcDimensions *dims_out) override
    {
        if (!config || !config->gp)
            return RADAR_PIPELINE_CREATE_ERR_NULL_ARGS;

        try
        {
            ctx_ = new RadarPipelineContext();

            ctx_->Ns           = config->gp->NofSamples;
            ctx_->Nr           = config->gp->NofRamps;
            ctx_->NofRangeBins = config->gp->NofRangeBins;
            ctx_->NofRx        = config->gp->NofRx;
            ctx_->NofTx        = config->gp->NofTx;
            ctx_->repeatFold   = config->gp->RDFold;
            ctx_->targetCount  = 0;

            if (dims_out)
            {
                dims_out->NofSamples = config->gp->NofSamples;
                dims_out->NofRamps   = config->gp->NofRamps;
                dims_out->NofRx      = config->gp->NofRx;
            }

            ctx_->pvaParams = std::make_unique<RadarPipelinePVAWorkloadParams>();

            ctx_->tensorReqs = std::make_unique<RadarPipelineTensorRequirements>(
                ctx_->Ns, ctx_->NofRx, ctx_->Nr, ctx_->NofRangeBins, ctx_->repeatFold, ctx_->NofTx);

            ctx_->tensors = std::make_unique<RadarPipelineTensors>(ctx_);

            ctx_->operators = std::make_unique<RadarPipelineOperators>();

            int32_t err = radar_create_pva_workloads(*ctx_->operators, *ctx_->tensors, ctx_->NofTx, ctx_->repeatFold);
            if (err != 0)
            {
                delete ctx_;
                ctx_ = nullptr;
                return RADAR_PIPELINE_CREATE_ERR_PVA_WORKLOAD;
            }

            return 0;
        }
        catch (...)
        {
            delete ctx_;
            ctx_ = nullptr;
            return RADAR_PIPELINE_CREATE_ERR_EXCEPTION;
        }
    }

    int32_t process(NVCVTensorHandle input_tensor, NVCVTensorHandle output_tensor) override
    {
        if (!input_tensor)
            return -1;

        try
        {
            NVCVTensorHandle originalOutputTensor = ctx_->tensors->outDOATensorHandle;

            ctx_->tensors->inRangeFFTTensorHandle = input_tensor;
            if (output_tensor)
                ctx_->tensors->outDOATensorHandle = output_tensor;

            int32_t err = radar_submit_pva_workloads(*ctx_->operators, *ctx_->tensors, *ctx_->pvaParams);

            ctx_->tensors->inRangeFFTTensorHandle = nullptr;
            ctx_->tensors->outDOATensorHandle     = originalOutputTensor;

            if (err != 0)
                return err;

            NVCVTensorData targetCountData;
            NVCVStatus status = nvcvTensorExportData(ctx_->tensors->outPeakCountTensorHandle, &targetCountData);
            if (status == NVCV_SUCCESS)
            {
                int32_t *targetCountPtr = nullptr;
                if (CupvaMemGetHostPointer((void **)&targetCountPtr, (void *)targetCountData.buffer.strided.basePtr) ==
                    CUPVA_ERROR_NONE)
                {
                    ctx_->targetCount = *targetCountPtr;
                }
            }

            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }

    int32_t get_target_count(int32_t *count) override
    {
        if (!count)
            return -1;
        *count = ctx_->targetCount;
        return 0;
    }

    int32_t validate(void *input_pva_ptr) override
    {
        if (!input_pva_ptr)
            return -1;

        try
        {
            NVCVTensorHandle externalInputTensor = nullptr;
            NVCVStatus status                    = utils::createTensorFromPvaMemory(
                &externalInputTensor, ctx_->tensors->m_tensorReqs.inRangeFFTRequirements, input_pva_ptr);
            if (status != NVCV_SUCCESS)
                return static_cast<int32_t>(status);

            NVCVTensorHandle originalInputTensor  = ctx_->tensors->inRangeFFTTensorHandle;
            ctx_->tensors->inRangeFFTTensorHandle = externalInputTensor;

            int32_t err = radar_execute_ref_workloads(*ctx_->tensors, ctx_->NofTx, ctx_->NofRx, ctx_->repeatFold);

            ctx_->tensors->inRangeFFTTensorHandle = originalInputTensor;
            nvcvTensorDecRef(externalInputTensor, NULL);

            return err;
        }
        catch (...)
        {
            return -1;
        }
    }

    int32_t get_intermediate_tensor(RadarPipelineTensorType type, RadarPipelineTensorSource src,
                                    RadarPipelineTensorData *data) override
    {
        if (!data || type < 0 || type >= RADAR_PIPELINE_TENSOR_COUNT)
            return -1;

        try
        {
            NVCVTensorHandle tensorHandle = nullptr;
            RadarPipelineDataType dtype   = RADAR_PIPELINE_DTYPE_FLOAT32;
            const bool usePva             = (src == RADAR_PIPELINE_SOURCE_PVA);

            switch (type)
            {
            case RADAR_PIPELINE_TENSOR_RANGE_PROFILE:
                tensorHandle = usePva ? ctx_->tensors->outRangeFFTTensorHandle : ctx_->tensors->refRangeFFTTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_COMPLEX_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_DOPPLER_MAP:
                tensorHandle =
                    usePva ? ctx_->tensors->outDopplerFFTTensorHandle : ctx_->tensors->refDopplerFFTTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_COMPLEX_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_NCI_RX:
                tensorHandle = usePva ? ctx_->tensors->outNciRxTensorHandle : ctx_->tensors->refNciRxTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_UINT32;
                break;
            case RADAR_PIPELINE_TENSOR_NCI_FINAL:
                tensorHandle = usePva ? ctx_->tensors->outNciFinalTensorHandle : ctx_->tensors->refNciFinalTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_UINT32;
                break;
            case RADAR_PIPELINE_TENSOR_CFAR_DETECTIONS:
                return -2; // Not applicable to 4T4R (no CFAR stage)
            case RADAR_PIPELINE_TENSOR_DETECTION_LIST:
                tensorHandle =
                    usePva ? ctx_->tensors->outPeakIndexTensorHandle : ctx_->tensors->refPeakIndexTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_SNAPSHOTS:
                tensorHandle = usePva ? ctx_->tensors->outPeakSnapTensorHandle : ctx_->tensors->refPeakSnapTensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_COMPLEX_INT32;
                break;
            case RADAR_PIPELINE_TENSOR_TARGET_LIST:
                tensorHandle = usePva ? ctx_->tensors->outDOATensorHandle : ctx_->tensors->refDOATensorHandle;
                dtype        = RADAR_PIPELINE_DTYPE_FLOAT32;
                break;
            case RADAR_PIPELINE_TENSOR_DETECTION_COUNT:
            case RADAR_PIPELINE_TENSOR_TARGET_COUNT:
                tensorHandle =
                    usePva ? ctx_->tensors->outPeakCountTensorHandle : ctx_->tensors->refPeakCountTensorHandle;
                dtype = RADAR_PIPELINE_DTYPE_INT32;
                break;
            default:
                return -1;
            }

            if (!tensorHandle)
                return -2;

            NVCVTensorData nvcvData;
            if (nvcvTensorExportData(tensorHandle, &nvcvData) != NVCV_SUCCESS)
                return -3;

            void *hostPtr = nullptr;
            if (CupvaMemGetHostPointer(&hostPtr, (void *)nvcvData.buffer.strided.basePtr) != CUPVA_ERROR_NONE)
                return -3;

            data->data  = hostPtr;
            data->ndim  = nvcvData.rank;
            data->dtype = dtype;

            int32_t elementSize = 4;
            switch (dtype)
            {
            case RADAR_PIPELINE_DTYPE_INT16:
                elementSize = 2;
                break;
            case RADAR_PIPELINE_DTYPE_COMPLEX_INT16:
                elementSize = 4;
                break;
            case RADAR_PIPELINE_DTYPE_COMPLEX_INT32:
                elementSize = 8;
                break;
            default:
                elementSize = 4;
                break;
            }

            int64_t totalBytes = 1;
            for (int32_t i = 0; i < nvcvData.rank && i < 4; i++)
            {
                data->shape[i]   = nvcvData.shape[i];
                data->strides[i] = nvcvData.buffer.strided.strides[i];
                totalBytes *= nvcvData.shape[i];
            }
            for (int32_t i = nvcvData.rank; i < 4; i++)
            {
                data->shape[i]   = 0;
                data->strides[i] = 0;
            }
            data->total_bytes = totalBytes * elementSize;

            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }

    const char *version() const override
    {
        return RADARLIB_VERSION_STRING;
    }

private:
    RadarPipelineContext *ctx_ = nullptr;
};

// ============================================================================
// Factory
// ============================================================================

extern "C" __attribute__((visibility("default"))) IRadarPipeline *radar_pipeline_4t4r()
{
    return new RadarPipeline4t4r();
}
