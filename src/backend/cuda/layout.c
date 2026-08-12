/*
 * Build rebuildable backend layouts directly from immutable canonical MoE weights.
 *
 * The layout is a physical execution asset: canonical artifact bytes remain the trust
 * authority, while this owner controls deterministic aligned packing and exact geometry.
 */
#include <yvex/internal/moe.h>
#include "src/backend/cuda/private.h"
#include <limits.h>
#include <string.h>

typedef struct {
    unsigned long long rows_per_expert, blocks_per_row, block_count;
    unsigned long long pair_count, scale_offset, code_offset, storage_bytes;
} cuda_moe_layout_geometry;

static int layout_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "cuda.layout", reason);
    return status;
}

static int layout_geometry_build(
    const yvex_physical_execution_decision *decision,
    cuda_moe_layout_geometry *geometry, yvex_error *err)
{
    unsigned long long raw_block_bytes, raw_bytes, section;
    if (!decision || !geometry ||
        decision->layout != YVEX_EXECUTION_LAYOUT_DERIVED_BACKEND ||
        !decision->derived_asset_required || decision->expert_count <= 1ull ||
        decision->canonical_row_count % decision->expert_count ||
        decision->canonical_row_width % YVEX_CUDA_Q8_K_BLOCK ||
        (decision->consumer != YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP &&
         decision->consumer != YVEX_EXECUTION_CONSUMER_ROUTED_DOWN))
        return layout_refuse(err, YVEX_ERR_FORMAT,
                             "derived MoE layout decision is incomplete");
    memset(geometry, 0, sizeof(*geometry));
    geometry->rows_per_expert = decision->canonical_row_count / decision->expert_count;
    geometry->blocks_per_row = decision->canonical_row_width / YVEX_CUDA_Q8_K_BLOCK;
    if (!yvex_core_u64_mul(decision->canonical_row_count, geometry->blocks_per_row,
                           &geometry->block_count))
        return layout_refuse(err, YVEX_ERR_BOUNDS,
                             "derived MoE block geometry overflowed");
    if (decision->canonical_qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        raw_block_bytes = YVEX_QUANT_IQ2_XXS_BYTES;
        if (!yvex_core_u64_mul(geometry->block_count, 2ull, &section) ||
            !yvex_core_u64_add(section, 63ull, &geometry->code_offset))
            return layout_refuse(err, YVEX_ERR_BOUNDS,
                                 "derived IQ2 scale geometry overflowed");
        geometry->code_offset &= ~63ull;
        if (!yvex_core_u64_mul(geometry->block_count, 64ull, &section) ||
            !yvex_core_u64_add(geometry->code_offset, section,
                               &geometry->storage_bytes))
            return layout_refuse(err, YVEX_ERR_BOUNDS,
                                 "derived IQ2 code geometry overflowed");
    } else if (decision->canonical_qtype == YVEX_GGUF_QTYPE_Q2_K) {
        raw_block_bytes = YVEX_QUANT_Q2_K_BYTES;
        if (geometry->rows_per_expert % 2ull ||
            !yvex_core_u64_mul(decision->expert_count,
                               geometry->rows_per_expert / 2ull,
                               &geometry->pair_count) ||
            !yvex_core_u64_mul(geometry->pair_count, geometry->blocks_per_row,
                               &geometry->pair_count) ||
            !yvex_core_u64_mul(geometry->pair_count, 8ull, &section) ||
            !yvex_core_u64_add(section, 63ull, &geometry->scale_offset))
            return layout_refuse(err, YVEX_ERR_BOUNDS,
                                 "derived Q2 pair geometry overflowed");
        geometry->scale_offset &= ~63ull;
        if (!yvex_core_u64_mul(geometry->pair_count, 32ull, &section) ||
            !yvex_core_u64_add(geometry->scale_offset, section,
                               &geometry->code_offset) ||
            !yvex_core_u64_add(geometry->code_offset, 63ull,
                               &geometry->code_offset))
            return layout_refuse(err, YVEX_ERR_BOUNDS,
                                 "derived Q2 scale geometry overflowed");
        geometry->code_offset &= ~63ull;
        if (!yvex_core_u64_mul(geometry->pair_count, 128ull, &section) ||
            !yvex_core_u64_add(geometry->code_offset, section,
                               &geometry->storage_bytes))
            return layout_refuse(err, YVEX_ERR_BOUNDS,
                                 "derived Q2 code geometry overflowed");
    } else {
        return layout_refuse(err, YVEX_ERR_UNSUPPORTED,
                             "derived MoE layout qtype is not admitted");
    }
    if (!yvex_core_u64_mul(geometry->block_count, raw_block_bytes, &raw_bytes) ||
        raw_bytes != decision->encoded_bytes ||
        geometry->storage_bytes < decision->encoded_bytes)
        return layout_refuse(err, YVEX_ERR_FORMAT,
                             "derived MoE layout differs from canonical geometry");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_moe_derived_layout_plan(
    const yvex_physical_execution_decision *decision,
    unsigned long long *storage_bytes, yvex_error *err)
{
    cuda_moe_layout_geometry geometry;
    int rc;
    if (storage_bytes) *storage_bytes = 0ull;
    if (!storage_bytes)
        return layout_refuse(err, YVEX_ERR_INVALID_ARG,
                             "derived MoE storage output is required");
    rc = layout_geometry_build(decision, &geometry, err);
    if (rc == YVEX_OK) *storage_bytes = geometry.storage_bytes;
    return rc;
}

int yvex_cuda_moe_derived_layout_build(
    const yvex_physical_execution_decision *decision,
    const unsigned char *canonical, unsigned long long canonical_bytes,
    unsigned char *derived, unsigned long long storage_bytes, yvex_error *err)
{
    cuda_moe_layout_geometry geometry;
    unsigned long long expert, row, block;
    int rc = layout_geometry_build(decision, &geometry, err);
    if (rc != YVEX_OK) return rc;
    if (!canonical || !derived || canonical == derived ||
        canonical_bytes != decision->encoded_bytes ||
        storage_bytes != geometry.storage_bytes ||
        storage_bytes > (unsigned long long)SIZE_MAX)
        return layout_refuse(err, YVEX_ERR_INVALID_ARG,
                             "separate exact derived MoE storage is required");
    memset(derived, 0, (size_t)storage_bytes);
    if (decision->canonical_qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        for (block = 0ull; block < geometry.block_count; ++block) {
            const unsigned char *source = canonical + block * YVEX_QUANT_IQ2_XXS_BYTES;
            memcpy(derived + block * 2ull, source, 2u);
            memcpy(derived + geometry.code_offset + block * 64ull, source + 2u, 64u);
        }
    } else {
        for (expert = 0ull; expert < decision->expert_count; ++expert) {
            for (row = 0ull; row < geometry.rows_per_expert; ++row) {
                unsigned long long parity = row & 1ull;
                for (block = 0ull; block < geometry.blocks_per_row; ++block) {
                    unsigned long long raw_index =
                        (expert * geometry.rows_per_expert + row) *
                            geometry.blocks_per_row + block;
                    unsigned long long pair_index =
                        (expert * (geometry.rows_per_expert / 2ull) + row / 2ull) *
                            geometry.blocks_per_row + block;
                    const unsigned char *source =
                        canonical + raw_index * YVEX_QUANT_Q2_K_BYTES;
                    unsigned long long word;
                    memcpy(derived + pair_index * 8ull + parity * 4ull,
                           source + 80u, 4u);
                    for (word = 0ull; word < 4ull; ++word) {
                        unsigned long long target = pair_index * 32ull +
                            (word / 2ull) * 16ull + parity * 8ull +
                            (word & 1ull) * 4ull;
                        memcpy(derived + geometry.scale_offset + target,
                               source + word * 4ull, 4u);
                    }
                    for (word = 0ull; word < 16ull; ++word) {
                        unsigned long long target =
                            pair_index * 128ull + word * 8ull + parity * 4ull;
                        memcpy(derived + geometry.code_offset + target,
                               source + 16ull + word * 4ull, 4u);
                    }
                }
            }
        }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
