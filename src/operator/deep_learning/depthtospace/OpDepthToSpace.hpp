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
 * @file OpDepthToSpace.hpp
 *
 * @brief Defines the private C++ Class for the DepthToSpace operation.
 */

#ifndef PVA_SOLUTIONS_PRIV_DEPTHTOSPACE_HPP
#define PVA_SOLUTIONS_PRIV_DEPTHTOSPACE_HPP

#include "IOperator.hpp"

/* SCRATCH_SIZE, and the two parameter structs the GSDF device kernel consumes. */
#include "device/depthtospace_gsdf_param.h"

#include <cupva_host.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include <OpDepthToSpace.h>
#include <cupva_host_scheduling.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace pva_solutions::priv {

struct RdfGeometry;

// 1) If the whole tensor fit into the VMEM, then tileX=width and tileY=height
// 2) If {width, height, blkSz} has been calculated offline, then use the pre-calculated tileX and tileY
auto preCalTileSz(int32_t width, int32_t height, int32_t blkSz, int32_t maxTileSz)
    -> std::optional<std::pair<int32_t, int32_t>>;

class DepthToSpace final : public IOperator
{
public:
    /*
     * Create picks one path per shape:
     *   Rdf  -- fast path. RasterDataFlow, interleave done in the address generators.
     *   Gsdf -- general fallback. GatherScatterDataFlow + DLUT, tiled on the prime factorization
     *           of the extents, so it serves every shape the operator accepts.
     */
    enum class DataFlowPath
    {
        Rdf,
        Gsdf
    };

    static constexpr int32_t kMaxTileSz{SCRATCH_SIZE / 2};

    /* One RDF executable per block size, so the RDF cache key is the block size. */
    using ExecDataKey = std::tuple<int32_t>;
    using ExecKey =
        decltype(std::tuple_cat(std::declval<ExecDataKey>(), std::declval<std::tuple<cupva::impl::Context *>>()));

    /* The GSDF executable takes the block size as a runtime parameter, so it needs no data key. */
    using GsdfExecKey = std::tuple<cupva::impl::Context *>;

    explicit DepthToSpace(NVCVTensorRequirements const *const tensorRequirements, int32_t blkSz);

    ~DepthToSpace();

    template<typename S>
    void operator()(S stream, NVCVTensorHandle in, NVCVTensorHandle out);

    /*
     * Which path Create selected. Both produce the same output, so the choice is not observable
     * from the result -- the system tests read it here to assert the routing itself.
     */
    DataFlowPath GetPath() const noexcept
    {
        return m_path;
    }

private:
    void CreateRdf(RdfGeometry const &geom);
    void CreateGsdf();

    template<typename S>
    void SubmitRdf(S stream);

    template<typename S>
    void SubmitGsdf(S stream);

    NVCVTensorRequirements m_inputTensorReqs;
    NVCVTensorRequirements m_outputTensorReqs;

    DataFlowPath m_path;

    int32_t m_blkSz;

    /*
     * m_srcOp is the ROOT input pointer; m_srcChanGroupOps[g] are offset copies sharing its base.
     * Submit calls setBase() on the root only, which by design leaves the derived offsets intact.
     */
    cupva::OffsetPointer<uint8_t> m_srcOp;
    std::vector<cupva::OffsetPointer<uint8_t>> m_srcChanGroupOps;

    /* m_dstOp is the ROOT output pointer; m_dstRowOfsOps[by] are offset copies on the same
     * setBase()-once contract. */
    cupva::OffsetPointer<uint8_t> m_dstOp;
    std::vector<cupva::OffsetPointer<uint8_t>> m_dstRowOfsOps;

    std::shared_ptr<cupva::Executable> m_exec;

    /* RDF path: one frozen program per output row offset, submitted together so the overhead
     * is paid once. */
    std::vector<cupvaCmd_t> m_progs;

    /* GSDF path: a single program, plus the DLUT mapping table it indexes through. */
    cupvaCmd_t m_prog;
    uint16_t *m_mappingTblDevPtr;

    // Mutex to protect shared Executable maps
    static std::mutex s_execMapMutex;
    // Maps to store executables with reference counting, one per data-flow path
    static std::map<ExecKey, std::weak_ptr<cupva::Executable>> s_rdfExecutables;
    static std::map<GsdfExecKey, std::weak_ptr<cupva::Executable>> s_gsdfExecutables;
    static std::shared_ptr<cupva::Executable> GetRdfExec(ExecDataKey execDataKey);
    static std::shared_ptr<cupva::Executable> GetGsdfExec();
};

} // namespace pva_solutions::priv

#endif // PVA_SOLUTIONS_PRIV_DEPTHTOSPACE_HPP
