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

#include "brute_force_matcher_knls.h"

#include <cupva_device.h>

VMEM(A, uint8_t, copy_patch, PVA_BF_NUM_DOUBLE_BUFFERS *PVA_BF_PATCH_BUFFER_SIZE);
VMEM(A, uint16_t, copy_patch_indices, PVA_BF_QUERY_BATCH_SIZE *PVA_BF_DESCRIPTOR_HALFWORDS *PVA_BF_NUM_DOUBLE_BUFFERS);
VMEM(A, CupvaSampler, copy_patch_sampler, PVA_BF_NUM_DOUBLE_BUFFERS);
VMEM(A, uint8_t, r2q_lane_ids, PVA_BF_QUERY_BATCH_SIZE);

enum
{
    AGEN_CFG_MATCHES_REF = 0,
    AGEN_CFG_MATCHES_SOURCE,
    AGEN_CFG_MATCHES_OUT,
    AGEN_CFG_COUNT
};

typedef struct
{
    uint16_t *source_base;
    uint16_t *reference;
    dvushort *source;
    uint32_t *matches_source;
    uint32_t *matches_reference;
    uint32_t query_count;
    AgenCFG agen_cfgs[AGEN_CFG_COUNT];
    uint32_t k;
    int matching_algo;
    int multiplier_1;
    int multiplier_2;
    uint32_t reference_count;
    uint32_t real_reference_count;
    uint32_t real_query_count;
    uint8_t *copy_patch;
    uint16_t *copy_patch_indices;
    CupvaSampler *copy_patch_samplers;
} ClosestMatchesParams;

void create_bf_agens(ClosestMatchesParams *params)
{
    const uint8_t interleaved_size_modifier = sizeof(uint32_t) / sizeof(uint16_t);
    const uint16_t num_iter_src_ref         = (uint16_t)(params->query_count / (PVA_BF_DESCRIPTOR_HALFWORDS));
    agen_A in_matches_ref                   = init_A(params->matches_reference);
    if (params->matching_algo == 0)
    {
        AgenWrapper wrapper_match_2;
        wrapper_match_2.size = sizeof(uint16_t);
        wrapper_match_2.n1   = interleaved_size_modifier;
        wrapper_match_2.n2   = PVA_MAX_BF_MATCHES_PER_DESCRIPTOR;
        wrapper_match_2.n3   = params->query_count / PVA_BF_QUERY_BATCH_SIZE;
        wrapper_match_2.s1   = PVA_BF_DESCRIPTOR_HALFWORDS * interleaved_size_modifier;
        wrapper_match_2.s2   = PVA_MAX_BF_DESCRIPTOR_COUNT * interleaved_size_modifier;
        wrapper_match_2.s3   = PVA_BF_QUERY_BATCH_SIZE * interleaved_size_modifier;
        INIT_AGEN3(in_matches_ref, wrapper_match_2);
    }
    else
    {
        AgenWrapper wrapper_match_2;
        wrapper_match_2.size = sizeof(uint16_t);
        wrapper_match_2.n1   = num_iter_src_ref;
        wrapper_match_2.s1   = PVA_BF_QUERY_BATCH_SIZE;
        INIT_AGEN1(in_matches_ref, wrapper_match_2);
    }

    AgenWrapper wrapper_match_source;
    agen_A in_matches_source  = init_A(params->matches_source);
    wrapper_match_source.size = sizeof(uint32_t);
    wrapper_match_source.n1   = interleaved_size_modifier;
    wrapper_match_source.n2   = params->reference_count / PVA_BF_QUERY_BATCH_SIZE_INT;
    wrapper_match_source.s1   = PVA_MAX_BF_DESCRIPTOR_COUNT;
    wrapper_match_source.s2   = PVA_BF_QUERY_BATCH_SIZE_INT;
    INIT_AGEN2(in_matches_source, wrapper_match_source);

    AgenWrapper wrapper_match_source_out;
    agen_A in_matches_out         = init_A(params->matches_source);
    wrapper_match_source_out.size = sizeof(uint32_t);
    wrapper_match_source_out.n1   = params->reference_count / PVA_BF_QUERY_BATCH_SIZE_INT;
    wrapper_match_source_out.s1   = PVA_BF_QUERY_BATCH_SIZE_INT;
    INIT_AGEN1(in_matches_out, wrapper_match_source_out);

    params->agen_cfgs[AGEN_CFG_MATCHES_REF]    = extract_agen_A_cfg(in_matches_ref);
    params->agen_cfgs[AGEN_CFG_MATCHES_SOURCE] = extract_agen_A_cfg(in_matches_source);
    params->agen_cfgs[AGEN_CFG_MATCHES_OUT]    = extract_agen_A_cfg(in_matches_out);
}

// Apply Lowe's ratio test to top-2 Q2R packed matches and write filtered results via AGEN.
// Rejects matches where best_distance * denominator >= second_best_distance * numerator.
// Returns updated agen_A after vstore_i advances.
static inline agen_A apply_lowes_q2r_filter_and_store(dvshortx min_0, dvshortx min_1, dvshortx signed_split,
                                                      const int multiplier_1, const int multiplier_2,
                                                      agen_A in_matches_ref)
{
    dvshortx min_0_val, min_0_idx, min_1_val, temp;
    dvsplitbits(min_0, signed_split, min_0_val, min_0_idx);
    dvsplitbits(min_1, signed_split, min_1_val, temp);
    temp              = dvmulh(min_0_val, multiplier_2, 0);
    min_1_val         = dvmulh(min_1_val, multiplier_1, 0);
    dvshortx selector = (temp >= min_1_val);

    dvshortx final_output_values, final_output_index, output_values_lo, output_values_hi;
    final_output_values = dvmux(selector, -1, min_0_val);
    final_output_index  = dvmux(selector, -1, min_0_idx);
    dvmov(final_output_values.lo, final_output_index.lo, output_values_lo.lo, output_values_lo.hi);
    dvmov(final_output_values.hi, final_output_index.hi, output_values_hi.lo, output_values_hi.hi);
    vstore_i(output_values_lo, in_matches_ref);
    vstore_i(output_values_hi, in_matches_ref);
    return in_matches_ref;
}

void compute_closest_matches_with_lowes_test_cross_check(ClosestMatchesParams *params) property(loop_levels_1)
{
    uint16_t *reference            = params->reference;
    dvushort *source               = params->source;
    uint32_t *matches_source       = params->matches_source;
    const uint32_t k               = params->k;
    const int multiplier_1         = params->multiplier_1;
    const int multiplier_2         = params->multiplier_2;
    const uint32_t reference_count = params->reference_count;
    const uint32_t real_ref_count  = params->real_reference_count;
    agen_A in_matches_ref          = init_agen_A_from_cfg(params->agen_cfgs[AGEN_CFG_MATCHES_REF]);

    //Define the minimums reference distances for 32 sources at a time
    dvshortx min_0, min_1, signed_split;
    signed_split.lo = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);
    signed_split.hi = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);
    min_0.lo = replicateh(slladd(PVA_BF_MAX_HAMMING_DISTANCE, PVA_BF_Q2R_DISTANCE_SHIFT, PVA_MAX_BF_DESCRIPTOR_COUNT));
    min_0.hi = min_0.lo;
    min_1    = min_0;

    vcharx lane_ids = zero_extend(*(vuchar *)r2q_lane_ids);

    //Loop through all references for 32 sources at a time (26 cycles for 32 sources)
    for (uint32_t ref_idx = 0; ref_idx < real_ref_count; ++ref_idx) chess_loop_range(1, PVA_MAX_BF_DESCRIPTOR_COUNT)
    chess_unroll_loop(1)
    chess_prepare_for_pipelining
    chess_modulo_scheduling_budget_ratio(200) chess_pipeline_initiation_interval(23)
    {
        //Accumulate the hamming distances between 32 sources and current reference
        dvshortx sum_hd_32, distance_candidate;
        distance_candidate.lo = replicateh(ref_idx);
        distance_candidate.hi = replicateh(ref_idx);

        chess_protect_access dvushort *source_ptr = source;

        for (uint32_t hw_idx = 0; hw_idx < PVA_BF_DESCRIPTOR_HALFWORDS; ++hw_idx)
            chess_loop_count(PVA_BF_DESCRIPTOR_HALFWORDS)
        chess_unroll_loop(PVA_BF_DESCRIPTOR_HALFWORDS)
        chess_prepare_for_pipelining
        {
            uint16_t ref_halfword = *(reference++);
            dvshortx src_vec      = zero_extend(*(source_ptr++));
            sum_hd_32             = dvSumHD_ca(src_vec, ref_halfword, sum_hd_32, hw_idx);
        }

        dvshortx sum_vrf = sum_hd_32;

        //This will give the top 2 references for 32 sources, iterating across all references (3 cycles SOL)
        distance_candidate = dvshiftadd(sum_vrf, PVA_BF_Q2R_DISTANCE_SHIFT, distance_candidate);
        dvsort2(distance_candidate, min_0, min_0, distance_candidate);
        dvsort2(distance_candidate, min_1, min_1, distance_candidate);

        //This will give the top 2 sources from 32 source batch for current reference (3 cycles SOL)
        vcharx distance_h = vdemote(sum_vrf);
        int low_idx_0, low_val_0, low_idx_1, low_val_1;
        vminrid_s(distance_h, low_val_0, low_idx_0);

        unsigned int curr_low_0 = matches_source[ref_idx];
        unsigned int curr_low_1 = matches_source[ref_idx + PVA_MAX_BF_DESCRIPTOR_COUNT];
        unsigned int out_val_idx_0 =
            ((unsigned int)low_val_0 << PVA_BF_R2Q_DISTANCE_SHIFT) | (unsigned int)(low_idx_0 + k);
        unsigned int new_curr_low_0 = min(out_val_idx_0, curr_low_0);

        distance_h = vmux(lane_ids == low_idx_0, PVA_BF_MAX_HAMMING_DISTANCE + 1, distance_h);
        vminrid_s(distance_h, low_val_1, low_idx_1);
        unsigned int out_val_idx_1 =
            ((unsigned int)low_val_1 << PVA_BF_R2Q_DISTANCE_SHIFT) | (unsigned int)(low_idx_1 + k);
        unsigned int new_curr_low_1 = min(max(out_val_idx_0, curr_low_0), min(out_val_idx_1, curr_low_1));

        matches_source[ref_idx]                               = new_curr_low_0;
        matches_source[ref_idx + PVA_MAX_BF_DESCRIPTOR_COUNT] = new_curr_low_1;
    }
    in_matches_ref =
        apply_lowes_q2r_filter_and_store(min_0, min_1, signed_split, multiplier_1, multiplier_2, in_matches_ref);
    params->agen_cfgs[AGEN_CFG_MATCHES_REF] = extract_agen_A_cfg(in_matches_ref);
}

void compute_closest_matches_with_lowes_test(ClosestMatchesParams *params)
{
    uint16_t *reference            = params->reference;
    dvushort *source               = params->source;
    uint32_t *matches_source       = params->matches_source;
    const uint32_t k               = params->k;
    const int multiplier_1         = params->multiplier_1;
    const int multiplier_2         = params->multiplier_2;
    const uint32_t reference_count = params->reference_count;
    const uint32_t real_ref_count  = params->real_reference_count;
    agen_A in_matches_ref          = init_agen_A_from_cfg(params->agen_cfgs[AGEN_CFG_MATCHES_REF]);

    //Define the minimums reference distances for 32 sources at a time
    dvshortx min_0, min_1, signed_split;
    signed_split.lo = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);
    signed_split.hi = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);
    min_0.lo = replicateh(slladd(PVA_BF_MAX_HAMMING_DISTANCE, PVA_BF_Q2R_DISTANCE_SHIFT, PVA_MAX_BF_DESCRIPTOR_COUNT));
    min_0.hi = min_0.lo;
    min_1    = min_0;

    //Loop through all references for 32 sources at a time (26 cycles for 32 sources)
    for (uint32_t ref_idx = 0; ref_idx < real_ref_count; ++ref_idx) chess_loop_range(1, PVA_MAX_BF_DESCRIPTOR_COUNT)
    chess_unroll_loop(1)
    chess_prepare_for_pipelining
    chess_modulo_scheduling_budget_ratio(200)
    {
        //Accumulate the hamming distances between 32 sources and current reference
        dvshortx sum_hd_32, min_temp_0;
        vshortx sum_hd_32_masked_lo, sum_hd_32_masked_hi;
        dvshortx distance_candidate;
        distance_candidate.lo                     = replicateh(ref_idx);
        distance_candidate.hi                     = replicateh(ref_idx);
        chess_protect_access dvushort *source_ptr = source;

        for (uint32_t hw_idx = 0; hw_idx < PVA_BF_DESCRIPTOR_HALFWORDS; ++hw_idx)
            chess_loop_count(PVA_BF_DESCRIPTOR_HALFWORDS)
        chess_unroll_loop(PVA_BF_DESCRIPTOR_HALFWORDS)
        chess_prepare_for_pipelining
        {
            uint16_t ref_halfword = *(reference++);
            dvshortx src_vec      = zero_extend(*(source_ptr++));
            sum_hd_32             = dvSumHD_ca(src_vec, ref_halfword, sum_hd_32, hw_idx);
        }

        //This will give the top 2 references for 32 sources, iterating across all references (3 cycles SOL)
        distance_candidate = dvshiftadd(sum_hd_32, PVA_BF_Q2R_DISTANCE_SHIFT, distance_candidate);
        dvsort2(distance_candidate, min_0, min_0, distance_candidate);
        dvsort2(distance_candidate, min_1, min_1, distance_candidate);
    }

    in_matches_ref =
        apply_lowes_q2r_filter_and_store(min_0, min_1, signed_split, multiplier_1, multiplier_2, in_matches_ref);
    params->agen_cfgs[AGEN_CFG_MATCHES_REF] = extract_agen_A_cfg(in_matches_ref);
}

void compute_closest_matches_with_cross_check(ClosestMatchesParams *params)
{
    uint16_t *reference            = params->reference;
    dvushort *source               = params->source;
    uint32_t *matches_source       = params->matches_source;
    const uint32_t k               = params->k;
    const uint32_t reference_count = params->reference_count;
    const uint32_t real_ref_count  = params->real_reference_count;
    agen_A in_matches_ref          = init_agen_A_from_cfg(params->agen_cfgs[AGEN_CFG_MATCHES_REF]);

    //Define the minimums reference distances for 32 sources at a time
    dvshortx min_0, signed_split;
    min_0.lo = replicateh(slladd(PVA_BF_MAX_HAMMING_DISTANCE, PVA_BF_Q2R_DISTANCE_SHIFT, PVA_MAX_BF_DESCRIPTOR_COUNT));
    min_0.hi = min_0.lo;
    signed_split.lo = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);
    signed_split.hi = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);

    //Loop through all references for 32 sources at a time (26 cycles for 32 sources)
    for (uint32_t ref_idx = 0; ref_idx < real_ref_count; ++ref_idx) chess_loop_range(1, PVA_MAX_BF_DESCRIPTOR_COUNT)
    chess_unroll_loop(1)
    chess_prepare_for_pipelining
    chess_modulo_scheduling_budget_ratio(200)
    {
        //Accumulate the hamming distances between 32 sources and current reference
        dvshortx sum_hd_32;
        dvshortx distance_candidate;
        distance_candidate.lo                     = replicateh(ref_idx);
        distance_candidate.hi                     = replicateh(ref_idx);
        chess_protect_access dvushort *source_ptr = source;

        for (uint32_t hw_idx = 0; hw_idx < PVA_BF_DESCRIPTOR_HALFWORDS; ++hw_idx) chess_loop_count(16)
        chess_unroll_loop(16)
        chess_prepare_for_pipelining
        {
            uint16_t ref_halfword = *(reference++);
            dvshortx src_vec      = zero_extend(*(source_ptr++));
            sum_hd_32             = dvSumHD_ca(src_vec, ref_halfword, sum_hd_32, hw_idx);
        }

        //This will give the top 2 references for 32 sources, iterating across all references (3 cycles SOL)
        distance_candidate = dvshiftadd(sum_hd_32, PVA_BF_Q2R_DISTANCE_SHIFT, distance_candidate);
        dvsort2(distance_candidate, min_0, min_0, distance_candidate);

        //This will give the top 2 sources from 32 source batch for current reference (3 cycles SOL)
        vcharx distance_h = vdemote(sum_hd_32);
        int low_val_0, low_idx_0;
        vminrid_s(distance_h, low_val_0, low_idx_0);
        low_idx_0 += k;

        unsigned int curr_low_0 = matches_source[ref_idx];

        low_val_0 = (int)(((unsigned int)low_val_0 << PVA_BF_R2Q_DISTANCE_SHIFT) | (unsigned int)low_idx_0);
        unsigned int new_curr_low_0 = min((unsigned int)low_val_0, curr_low_0);

        matches_source[ref_idx] = new_curr_low_0;
    }

    //Do lowe's test with the top 2 references for the 32 sources
    dvshortx final_output_values, final_output_index, output_values_lo, output_values_hi;
    dvsplitbits(min_0, signed_split, final_output_values, final_output_index);
    dvmov(final_output_values.lo, final_output_index.lo, output_values_lo.lo, output_values_lo.hi);
    dvmov(final_output_values.hi, final_output_index.hi, output_values_hi.lo, output_values_hi.hi);
    vstore_i(output_values_lo, in_matches_ref);
    vstore_i(output_values_hi, in_matches_ref);
    params->agen_cfgs[AGEN_CFG_MATCHES_REF] = extract_agen_A_cfg(in_matches_ref);
}

void compute_closest_matches(ClosestMatchesParams *params)
{
    uint16_t *reference            = params->reference;
    dvushort *source               = params->source;
    uint32_t *matches_source       = params->matches_source;
    const uint32_t k               = params->k;
    const uint32_t reference_count = params->reference_count;
    agen_A in_matches_ref          = init_agen_A_from_cfg(params->agen_cfgs[AGEN_CFG_MATCHES_REF]);

    //Define the minimums reference distances for 32 sources at a time
    dvshortx min_0, min_1, min_2, signed_split;
    min_0.lo = replicateh(slladd(PVA_BF_MAX_HAMMING_DISTANCE, PVA_BF_Q2R_DISTANCE_SHIFT, PVA_MAX_BF_DESCRIPTOR_COUNT));
    min_0.hi = min_0.lo;
    min_1    = min_0;
    min_2    = min_0;
    signed_split.lo = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);
    signed_split.hi = replicateh(PVA_BF_Q2R_DISTANCE_SHIFT);

    //Loop through all references for 32 sources at a time (26 cycles for 32 sources)
    for (uint32_t ref_idx = 0; ref_idx < reference_count; ++ref_idx) chess_loop_range(1, PVA_MAX_BF_DESCRIPTOR_COUNT)
    chess_unroll_loop(1)
    chess_prepare_for_pipelining
    chess_modulo_scheduling_budget_ratio(200)
    {
        //Accumulate the hamming distances between 32 sources and current reference
        dvshortx sum_hd_32, min_temp_0;
        vshortx sum_hd_32_masked_lo, sum_hd_32_masked_hi;
        dvshortx distance_candidate;
        distance_candidate.lo                     = replicateh(ref_idx);
        distance_candidate.hi                     = replicateh(ref_idx);
        chess_protect_access dvushort *source_ptr = source;

        for (uint32_t hw_idx = 0; hw_idx < PVA_BF_DESCRIPTOR_HALFWORDS; ++hw_idx)
            chess_loop_count(PVA_BF_DESCRIPTOR_HALFWORDS)
        chess_unroll_loop(PVA_BF_DESCRIPTOR_HALFWORDS)
        chess_prepare_for_pipelining
        {
            uint16_t ref_halfword = *(reference++);
            dvshortx src_vec      = zero_extend(*(source_ptr++));
            sum_hd_32             = dvSumHD_ca(src_vec, ref_halfword, sum_hd_32, hw_idx);
        }

        //This will give the top 2 references for 32 sources, iterating across all references (3 cycles SOL)
        distance_candidate = dvshiftadd(sum_hd_32, PVA_BF_Q2R_DISTANCE_SHIFT, distance_candidate);
        dvsort2(distance_candidate, min_0, min_0, distance_candidate);
        dvsort2(distance_candidate, min_1, min_1, distance_candidate);
        dvsort2(distance_candidate, min_2, min_2, distance_candidate);
    }

    //Do lowe's test with the top 2 references for the 32 sources
    dvshortx final_output_values, final_output_index, output_values_lo, output_values_hi;

    dvsplitbits(min_0, signed_split, final_output_values, final_output_index);
    dvmov(final_output_values.lo, final_output_index.lo, output_values_lo.lo, output_values_lo.hi);
    dvmov(final_output_values.hi, final_output_index.hi, output_values_hi.lo, output_values_hi.hi);
    vstore_i(output_values_lo, in_matches_ref);
    vstore_i(output_values_hi, in_matches_ref);

    dvsplitbits(min_1, signed_split, final_output_values, final_output_index);
    dvmov(final_output_values.lo, final_output_index.lo, output_values_lo.lo, output_values_lo.hi);
    dvmov(final_output_values.hi, final_output_index.hi, output_values_hi.lo, output_values_hi.hi);
    vstore_i(output_values_lo, in_matches_ref);
    vstore_i(output_values_hi, in_matches_ref);

    dvsplitbits(min_2, signed_split, final_output_values, final_output_index);
    dvmov(final_output_values.lo, final_output_index.lo, output_values_lo.lo, output_values_lo.hi);
    dvmov(final_output_values.hi, final_output_index.hi, output_values_hi.lo, output_values_hi.hi);
    vstore_i(output_values_lo, in_matches_ref);
    vstore_i(output_values_hi, in_matches_ref);
    params->agen_cfgs[AGEN_CFG_MATCHES_REF] = extract_agen_A_cfg(in_matches_ref);
}

void compute_lowe_test(agen_A matches_source_in, agen_A matches_source_out, const int multiplier_1,
                       const int multiplier_2, const uint32_t reference_count)
{
    const int max_val                    = -1;
    const uint32_t reference_block_count = reference_count / PVA_BF_QUERY_BATCH_SIZE_INT;
    dvintx mult_1, mult_2;
    mult_1.lo = replicatew(multiplier_1);
    mult_1.hi = replicatew(multiplier_1);
    mult_2.lo = replicatew(multiplier_2);
    mult_2.hi = replicatew(multiplier_2);

    for (uint32_t ref_blk = 0; ref_blk < reference_block_count; ++ref_blk) chess_loop_range(2, 256)
    chess_unroll_loop(8)
    chess_prepare_for_pipelining
    {
        dvintx min_0 = dvint_load(matches_source_in);
        dvintx min_1 = dvint_load(matches_source_in);

        // the current input is distance << 16 | index
        dvintx dist_0 = dvsra(min_0, PVA_BF_R2Q_DISTANCE_SHIFT);
        dvintx dist_1 = dvsra(min_1, PVA_BF_R2Q_DISTANCE_SHIFT);

        dvintx min_0_value_lowe = dvmulw(dist_0, mult_2, 0);
        dvintx min_1_value_lowe = dvmulw(dist_1, mult_1, 0);
        dvintx selector         = (min_0_value_lowe >= min_1_value_lowe);
        dvintx final_output     = dvmux(selector, max_val, min_0);
        vstore(final_output, matches_source_out);
    }
}

void cross_check(uint16_t *matches_source, uint16_t *matches_reference, uint32_t query_count)
{
    /*
     * Cross-check validation for mutual best-match:
     * - matches_source param = caller's matches_reference (Q2R buffer)
     *   Format via vstore_i with .lo=dist, .hi=idx: [dist0, idx0, dist1, idx1, ...]
     *   So [i*2] = distance, [i*2+1] = ref_index for query i
     *
     * - matches_reference param = caller's matches_source (R2Q buffer)
     *   Format via slladd: (query_idx<<16)|distance as uint32
     *   As uint16 (little-endian): [j*2] = distance, [j*2+1] = query_idx for reference j
     *
     * Cross-check: For query i with best ref j, ref j's best query must be i
     */

    uint32_t *matches_source_ptr    = (uint32_t *)matches_source;
    uint32_t *matches_reference_ptr = (uint32_t *)matches_reference;
    for (uint32_t query_idx = 0; query_idx < query_count; ++query_idx)
    {
        if (PVA_BF_INVALID_MATCH == matches_source_ptr[query_idx])
        {
            continue;
        }

        const uint32_t src_match_index = ((uint32_t)matches_source[2 * query_idx + 1]) & PVA_BF_INDEX_MASK;

        if (src_match_index > PVA_MAX_BF_DESCRIPTOR_COUNT - 1)
        {
            matches_source_ptr[query_idx] = PVA_BF_INVALID_MATCH;
            continue;
        }

        const uint32_t ref_entry = matches_reference_ptr[src_match_index];
        if (PVA_BF_INVALID_MATCH == ref_entry)
        {
            matches_source_ptr[query_idx] = PVA_BF_INVALID_MATCH;
            continue;
        }
        const uint32_t ref_match_index = (ref_entry & PVA_BF_INDEX_MASK);
        matches_source_ptr[query_idx] =
            (ref_match_index == (uint32_t)query_idx) ? matches_source_ptr[query_idx] : PVA_BF_INVALID_MATCH;
    }
}

void populate_copy_patch_indices(uint16_t *indices)
{
    for (uint32_t hw_idx = 0; hw_idx < PVA_BF_DESCRIPTOR_HALFWORDS; ++hw_idx)
    {
        for (uint32_t lane_idx = 0; lane_idx < PVA_BF_QUERY_BATCH_SIZE; ++lane_idx)
        {
            uint32_t idx     = (hw_idx * PVA_BF_QUERY_BATCH_SIZE + lane_idx) * 2;
            indices[idx]     = (uint16_t)hw_idx;
            indices[idx + 1] = (uint16_t)lane_idx;
        }
    }
}

void setup_copy_patch_samplers(ClosestMatchesParams *params, uint16_t *patch0, uint16_t *patch1)
{
    CupvaSamplerInput2D const input = {
        .data           = params->source_base,
        .type           = SAMPLER_INPUT_TYPE_U16,
        .width          = PVA_BF_DESCRIPTOR_HALFWORDS,
        .height         = PVA_MAX_BF_DESCRIPTOR_COUNT,
        .linePitch      = PVA_BF_DESCRIPTOR_HALFWORDS,
        .outOfRangeMode = SAMPLER_OUT_OF_RANGE_CONSTANT,
        .outOfRangeVal  = 0,
    };

    CupvaSamplerIndices2D const indices = {
        .data               = params->copy_patch_indices,
        .type               = SAMPLER_INDEX_TYPE_U16,
        .width              = PVA_BF_QUERY_BATCH_SIZE,
        .height             = PVA_BF_DESCRIPTOR_HALFWORDS,
        .linePitch          = PVA_BF_QUERY_BATCH_SIZE,
        .fractionalBits     = 0,
        .fractionalHandling = SAMPLER_FRAC_HANDLING_TRUNCATE,
        .offsetX            = 0,
        .offsetY            = 0,
        .interleaving       = SAMPLER_INTERLEAVING_ELEMENTS,
    };

    CupvaSamplerOutput const output0 = {
        .data      = patch0,
        .pitch     = PVA_BF_QUERY_BATCH_SIZE,
        .transMode = TRANS_MODE_NONE,
    };

    CupvaSamplerOutput const output1 = {
        .data      = patch1,
        .pitch     = PVA_BF_QUERY_BATCH_SIZE,
        .transMode = TRANS_MODE_NONE,
    };

    cupvaSamplerSetup(&params->copy_patch_samplers[0], &input, &indices, &output0);
    cupvaSamplerSetup(&params->copy_patch_samplers[1], &input, &indices, &output1);
}

void copy_to_patch(ClosestMatchesParams *params, uint16_t *patch, int start_y, CupvaSampler *sampler)
{
    uint16_t *source_offset = params->source_base + (start_y * PVA_BF_DESCRIPTOR_HALFWORDS);
    cupvaSamplerUpdateAddrs(sampler, source_offset, params->copy_patch_indices, patch);
    cupvaSamplerStart(sampler);
}

void bfm_simple(ClosestMatchesParams *params)
{
    const uint32_t query_count = params->query_count;

    if (0 == query_count)
    {
        return;
    }

    uint8_t *patch_bufs[2] = {params->copy_patch, params->copy_patch + PVA_BF_PATCH_BUFFER_SIZE};
    int buf                = 0;
    int next_buf           = 1;

    copy_to_patch(params, (uint16_t *)patch_bufs[buf], 0, &params->copy_patch_samplers[buf]);

    for (uint32_t k = 0; k < query_count; k += PVA_BF_QUERY_BATCH_SIZE)
    {
        cupvaSamplerWait();

        uint32_t next_k = k + PVA_BF_QUERY_BATCH_SIZE;
        if (next_k < query_count)
        {
            copy_to_patch(params, (uint16_t *)patch_bufs[next_buf], next_k, &params->copy_patch_samplers[next_buf]);
        }

        params->k      = k;
        params->source = (dvushort *)patch_bufs[buf];
        compute_closest_matches(params);

        int tmp  = buf;
        buf      = next_buf;
        next_buf = tmp;
    }
}

void bfm_cross_check(ClosestMatchesParams *params)
{
    uint32_t *matches_source    = params->matches_source;
    uint32_t *matches_reference = params->matches_reference;
    const uint32_t query_count  = params->query_count;

    if (0 == query_count)
    {
        return;
    }

    uint8_t *patch_bufs[2] = {params->copy_patch, params->copy_patch + PVA_BF_PATCH_BUFFER_SIZE};
    int buf                = 0;
    int next_buf           = 1;

    copy_to_patch(params, (uint16_t *)patch_bufs[buf], 0, &params->copy_patch_samplers[buf]);

    for (uint32_t k = 0; k < query_count; k += PVA_BF_QUERY_BATCH_SIZE)
    {
        cupvaSamplerWait();

        uint32_t next_k = k + PVA_BF_QUERY_BATCH_SIZE;
        if (next_k < query_count)
        {
            copy_to_patch(params, (uint16_t *)patch_bufs[next_buf], next_k, &params->copy_patch_samplers[next_buf]);
        }

        params->k      = k;
        params->source = (dvushort *)patch_bufs[buf];
        compute_closest_matches_with_cross_check(params);

        int tmp  = buf;
        buf      = next_buf;
        next_buf = tmp;
    }

    //Cross check the matches
    cross_check((uint16_t *)matches_reference, (uint16_t *)matches_source, query_count);
}

void bfm_lowe_test(ClosestMatchesParams *params)
{
    const uint32_t query_count = params->query_count;

    if (0 == query_count)
    {
        return;
    }

    uint8_t *patch_bufs[2] = {params->copy_patch, params->copy_patch + PVA_BF_PATCH_BUFFER_SIZE};
    int buf                = 0;
    int next_buf           = 1;

    copy_to_patch(params, (uint16_t *)patch_bufs[buf], 0, &params->copy_patch_samplers[buf]);

    for (uint32_t k = 0; k < query_count; k += PVA_BF_QUERY_BATCH_SIZE)
    {
        cupvaSamplerWait();

        uint32_t next_k = k + PVA_BF_QUERY_BATCH_SIZE;
        if (next_k < query_count)
        {
            copy_to_patch(params, (uint16_t *)patch_bufs[next_buf], next_k, &params->copy_patch_samplers[next_buf]);
        }

        params->k      = k;
        params->source = (dvushort *)patch_bufs[buf];
        compute_closest_matches_with_lowes_test(params);

        int tmp  = buf;
        buf      = next_buf;
        next_buf = tmp;
    }
}

void bfm_lowe_test_cross_check(ClosestMatchesParams *params)
{
    uint32_t *matches_source       = params->matches_source;
    uint32_t *matches_reference    = params->matches_reference;
    const uint32_t query_count     = params->query_count;
    const uint32_t reference_count = params->reference_count;

    if (0 == query_count)
    {
        return;
    }

    uint8_t *patch_bufs[2] = {params->copy_patch, params->copy_patch + PVA_BF_PATCH_BUFFER_SIZE};
    int buf                = 0;
    int next_buf           = 1;

    copy_to_patch(params, (uint16_t *)patch_bufs[buf], 0, &params->copy_patch_samplers[buf]);

    for (uint32_t k = 0; k < query_count; k += PVA_BF_QUERY_BATCH_SIZE)
    {
        cupvaSamplerWait();

        uint32_t next_k = k + PVA_BF_QUERY_BATCH_SIZE;
        if (next_k < query_count)
        {
            copy_to_patch(params, (uint16_t *)patch_bufs[next_buf], next_k, &params->copy_patch_samplers[next_buf]);
        }

        params->k      = k;
        params->source = (dvushort *)patch_bufs[buf];
        compute_closest_matches_with_lowes_test_cross_check(params);

        int tmp  = buf;
        buf      = next_buf;
        next_buf = tmp;
    }

    const uint32_t real_q_count = params->real_query_count;
    const uint32_t r2q_idx_mask = (1u << PVA_BF_R2Q_DISTANCE_SHIFT) - 1;
    for (uint32_t ref_idx = 0; ref_idx < reference_count; ++ref_idx)
    {
        uint32_t entry_2nd = matches_source[ref_idx + PVA_MAX_BF_DESCRIPTOR_COUNT];
        uint32_t q_idx     = entry_2nd & r2q_idx_mask;
        if (q_idx >= real_q_count)
        {
            matches_source[ref_idx + PVA_MAX_BF_DESCRIPTOR_COUNT] =
                ((uint32_t)PVA_BF_MAX_HAMMING_DISTANCE << PVA_BF_R2Q_DISTANCE_SHIFT) | (uint32_t)q_idx;
        }
    }

    agen_A input_matches_source = init_agen_A_from_cfg(params->agen_cfgs[AGEN_CFG_MATCHES_SOURCE]);
    agen_A input_matches_out    = init_agen_A_from_cfg(params->agen_cfgs[AGEN_CFG_MATCHES_OUT]);
    compute_lowe_test(input_matches_source, input_matches_out, params->multiplier_1, params->multiplier_2,
                      reference_count);
    params->agen_cfgs[AGEN_CFG_MATCHES_SOURCE] = extract_agen_A_cfg(input_matches_source);
    params->agen_cfgs[AGEN_CFG_MATCHES_OUT]    = extract_agen_A_cfg(input_matches_out);

    //Cross check the matches
    cross_check((uint16_t *)matches_reference, (uint16_t *)matches_source, query_count);
}

void brute_force_matcher(uint16_t *source, uint16_t *reference, uint32_t *matches_source, uint32_t *matches_reference,
                         const uint32_t query_count, const uint32_t reference_count, const int matching_algo,
                         const int multiplier_1, const int multiplier_2, const uint32_t real_reference_count,
                         const uint32_t real_query_count)
{

    ClosestMatchesParams params;
    params.source_base          = source;
    params.reference            = reference;
    params.source               = NULL;
    params.matches_source       = matches_source;
    params.matches_reference    = matches_reference;
    params.query_count          = query_count;
    params.k                    = 0;
    params.matching_algo        = matching_algo;
    params.multiplier_1         = multiplier_1;
    params.multiplier_2         = multiplier_2;
    params.reference_count      = reference_count;
    params.real_reference_count = real_reference_count;
    params.real_query_count     = real_query_count;
    params.copy_patch           = copy_patch;
    params.copy_patch_indices   = copy_patch_indices;
    params.copy_patch_samplers  = copy_patch_sampler;

    create_bf_agens(&params);

    for (uint32_t lane_idx = 0; lane_idx < PVA_BF_QUERY_BATCH_SIZE; ++lane_idx)
    {
        r2q_lane_ids[lane_idx] = (uint8_t)lane_idx;
    }

    populate_copy_patch_indices(params.copy_patch_indices);
    setup_copy_patch_samplers(&params, (uint16_t *)params.copy_patch,
                              (uint16_t *)(params.copy_patch + PVA_BF_PATCH_BUFFER_SIZE));

    if (matching_algo == 0)
    {
        bfm_simple(&params);
    }
    else if (matching_algo == 1)
    {
        bfm_cross_check(&params);
    }
    else if (matching_algo == 2)
    {
        bfm_lowe_test(&params);
    }
    else if (matching_algo == 3)
    {
        bfm_lowe_test_cross_check(&params);
    }
}
