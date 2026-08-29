/*
 * Execute admitted low-bit rows by Q8_K activations through native integer Tensor Cores.
 *
 * Encoded scales remain outside MMA. Each 16-element segment is expanded directly into an
 * integer fragment, so Q8_0, Q2_K, IQ2_XXS, and MXFP4 keep their canonical block semantics
 * without materializing a dense dequantized matrix.
 */
#include "src/backend/cuda/kernel_primitives.h"
#include <yvex/internal/execution_batch.h>
#include <mma.h>

using namespace nvcuda;

static __device__ int tensorcore_row_geometry(unsigned int qtype,
                                              unsigned long long width,
                                              unsigned long long row_bytes)
{
    if (!width || width % YVEX_CUDA_Q8_K_BLOCK) return 0;
    if (qtype == YVEX_GGUF_QTYPE_Q8_0)
        return row_bytes == (width / 32ull) * 34ull;
    if (qtype == YVEX_GGUF_QTYPE_Q2_K)
        return row_bytes == (width / 256ull) * 84ull;
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS)
        return row_bytes == (width / 256ull) * 66ull;
    if (qtype == YVEX_GGUF_QTYPE_MXFP4)
        return row_bytes == (width / 32ull) * 17ull;
    return 0;
}

static __device__ __forceinline__ void tensorcore_row_sync(unsigned int warps)
{
    if (warps == 1u) __syncwarp();
    else __syncthreads();
}

static __device__ void tensorcore_canonical_weight_tile(
    const unsigned char *base, unsigned long long row_bytes, unsigned long long row_count,
    unsigned long long row_base, unsigned long long segment, unsigned int qtype, signed char *tile)
{
    unsigned int lane = threadIdx.x & 31u, tile_row = lane >> 1u, half = lane & 1u;
    unsigned int shift = 0u, grid = 0u, signs = 0u;
    unsigned long long row_index = row_base + tile_row, first = segment + half * 8ull;
    signed char *output = tile + tile_row * 16u + half * 8u;
    if (row_index >= row_count) {
        for (unsigned int index = 0u; index < 8u; ++index) output[index] = 0;
        return;
    }
    const unsigned char *row = base + row_index * row_bytes, *packed = NULL;
    if (qtype == YVEX_GGUF_QTYPE_Q8_0)
        packed = row + (first / 32ull) * 34ull + 2u + first % 32ull;
    else if (qtype == YVEX_GGUF_QTYPE_Q2_K) {
        unsigned int local_subblock = (segment % 128ull) / 16ull, block_half = (segment % 256ull) / 128ull;
        packed = row + (segment / 256ull) * 84ull + 16u + block_half * 32u +
                 (local_subblock & 1u) * 16u + half * 8u;
        shift = (local_subblock / 2u) * 2u;
    } else if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        unsigned int group = (first % 256ull) / 32ull, subgroup = (first & 31ull) / 8ull;
        const unsigned char *block = row + (first / 256ull) * 66ull;
        unsigned int grids = qtype_load_u32(block + 2u + group * 8u), scale = qtype_load_u32(block + 6u + group * 8u);
        grid = iq2_xxs_grid[(grids >> (8u * subgroup)) & 255u];
        signs = iq2_xxs_signs((scale >> (7u * subgroup)) & 127u);
    } else {
        packed = row + (first / 32ull) * 17ull + 1u + first % 16ull;
        shift = first % 32ull < 16ull ? 0u : 4u;
    }
    for (unsigned int index = 0u; index < 8u; ++index) {
        if (qtype == YVEX_GGUF_QTYPE_Q8_0)
            output[index] = (signed char)packed[index];
        else if (qtype == YVEX_GGUF_QTYPE_Q2_K)
            output[index] = (signed char)((packed[index] >> shift) & 3u);
        else if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
            unsigned int digit = (grid >> (2u * index)) & 3u;
            int level = digit == 0u ? 8 : digit == 1u ? 25 : 43;
            output[index] = (signed char)(signs & (1u << index) ? -level : level);
        } else
            output[index] =
                (signed char)mxfp4_code_to_float((packed[index] >> shift) & 15u);
    }
}

static __device__ int tensorcore_product_factor(const unsigned char *row,
                                                unsigned long long segment,
                                                unsigned int qtype)
{
    if (qtype == YVEX_GGUF_QTYPE_Q2_K) {
        const unsigned char *block = row + (segment / 256ull) * 84ull;
        unsigned int subblock = (unsigned int)((segment % 256ull) / 16ull);
        return (int)(block[subblock] & 15u);
    }
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        const unsigned char *block = row + (segment / 256ull) * 66ull;
        unsigned int group = (unsigned int)((segment % 256ull) / 32ull);
        unsigned int sign_scale = qtype_load_u32(block + 6u + group * 8u);
        return (int)(2u * (sign_scale >> 28u) + 1u);
    }
    return 1;
}

static __device__ int tensorcore_minimum_factor(const unsigned char *row,
                                                unsigned long long segment,
                                                unsigned int qtype)
{
    if (qtype != YVEX_GGUF_QTYPE_Q2_K) return 0;
    const unsigned char *block = row + (segment / 256ull) * 84ull;
    unsigned int subblock = (unsigned int)((segment % 256ull) / 16ull);
    return (int)(block[subblock] >> 4u);
}

static __device__ float tensorcore_scaled_group(const unsigned char *row,
                                                unsigned long long segment,
                                                unsigned int qtype,
                                                float activation_scale,
                                                int product, int minimum)
{
    if (qtype == YVEX_GGUF_QTYPE_Q8_0) {
        const unsigned char *block = row + (segment / 32ull) * 34ull;
        return activation_scale * f16_bits_to_float(qtype_load_u16(block)) *
               (float)product;
    }
    if (qtype == YVEX_GGUF_QTYPE_Q2_K) {
        const unsigned char *block = row + (segment / 256ull) * 84ull;
        return activation_scale * f16_bits_to_float(qtype_load_u16(block + 80u)) *
                   (float)product -
               activation_scale * f16_bits_to_float(qtype_load_u16(block + 82u)) *
                   (float)minimum;
    }
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        const unsigned char *block = row + (segment / 256ull) * 66ull;
        return 0.125f * f16_bits_to_float(qtype_load_u16(block)) *
               activation_scale * (float)product;
    }
    const unsigned char *block = row + (segment / 32ull) * 17ull;
    return 0.5f * activation_scale * e8m0_bits_to_float(block[0]) *
           (float)product;
}

typedef struct {
    const unsigned char *base;
    unsigned long long row_bytes, expert_bytes;
    unsigned long long row_width, row_count, expert_count;
    unsigned int qtype;
} tensorcore_expert_view;

extern "C" __global__ void yvex_qtype_tensorcore_rows(
    const unsigned char *encoded, unsigned long long row_bytes,
    unsigned long long row_width, unsigned long long start_row,
    unsigned long long row_count, unsigned long long input_rows,
    unsigned long long group_count, unsigned long long group_rows,
    unsigned int qtype, const unsigned char *activation, const float *additive,
    float *output, int output_bf16, int *status)
{
    __shared__ __align__(16) signed char weight_tile[16 * 16];
    __shared__ __align__(16) signed char activation_tile[4][16 * 16];
    __shared__ __align__(16) int product_tile[4][16 * 16];
    __shared__ int integer_totals[4][16 * 16];
    __shared__ int minimum_totals[4][16 * 16];
    __shared__ float totals[4][16 * 16];
    unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    unsigned int warps = blockDim.x >> 5u;
    unsigned long long input_tiles, input_groups, row_base, input_base;
    unsigned long long q8_blocks = row_width / YVEX_CUDA_Q8_K_BLOCK;
    unsigned long long row_group;
    unsigned int mma_group_segments;
    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                   signed char, wmma::row_major> weight;
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                   signed char, wmma::col_major> values;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> product;

    if (!status || *status) return;
    if (!encoded || !activation || !output || !row_count || !input_rows ||
        !group_count || !group_rows || group_count > ~0ull / group_rows ||
        row_count != group_count * group_rows ||
        input_rows > ~0ull - 15ull ||
        !warps || warps > 4u || blockDim.x != warps * 32u ||
        (group_count > 1ull && group_rows % 16ull) ||
        !tensorcore_row_geometry(qtype, row_width, row_bytes)) {
        if (!threadIdx.x) atomicCAS(status, 0, 2);
        return;
    }
    input_tiles = (input_rows + 15ull) / 16ull;
    input_groups = (input_tiles + warps - 1ull) / warps;
    row_base = ((unsigned long long)blockIdx.x / input_groups) * 16ull;
    input_base = ((unsigned long long)blockIdx.x % input_groups) *
                     warps * 16ull + warp * 16ull;
    row_group = group_rows ? row_base / group_rows : group_count;
    /* Every admitted format except Q2_K shares its integer factor across 32 weights.
     * Accumulating two K=16 MMA fragments therefore preserves exact integer arithmetic
     * while halving fragment stores and scale application. */
    mma_group_segments = qtype == YVEX_GGUF_QTYPE_Q2_K ? 1u : 2u;
    if (row_group >= group_count) {
        if (!threadIdx.x) atomicCAS(status, 0, 2);
        return;
    }
    for (unsigned int index = lane; index < 256u; index += 32u) {
        integer_totals[warp][index] = minimum_totals[warp][index] = 0;
        totals[warp][index] = 0.0f;
    }
    tensorcore_row_sync(warps);
    for (unsigned long long segment = 0ull; segment < row_width; segment += 16ull) {
        unsigned int segment_index = (unsigned int)(segment / 16ull);
        if (!warp)
            tensorcore_canonical_weight_tile(
                encoded + start_row * row_bytes, row_bytes, row_count,
                row_base, segment, qtype, weight_tile);
        for (unsigned int index = lane; index < 256u; index += 32u) {
            unsigned int tile_row = index / 16u, tile_column = index % 16u;
            unsigned long long global_input = input_base + tile_row;
            activation_tile[warp][index] = global_input < input_rows
                ? (signed char)activation[((global_input * group_count + row_group) *
                                               q8_blocks + segment / 256ull) *
                                              YVEX_CUDA_Q8_K_BYTES +
                                          4ull + segment % 256ull + tile_column]
                : 0;
        }
        tensorcore_row_sync(warps);
        wmma::load_matrix_sync(weight, weight_tile, 16u);
        wmma::load_matrix_sync(values, activation_tile[warp], 16u);
        if (segment_index % mma_group_segments == 0u)
            wmma::fill_fragment(product, 0);
        wmma::mma_sync(product, weight, values, product);
        if ((segment_index + 1u) % mma_group_segments == 0u) {
            wmma::store_matrix_sync(product_tile[warp], product, 16u,
                                    wmma::mem_row_major);
            __syncwarp();
            for (unsigned int index = lane; index < 256u; index += 32u) {
                unsigned int tile_row = index / 16u, tile_column = index % 16u;
                unsigned long long global_row = row_base + tile_row;
                unsigned long long global_input = input_base + tile_column;
                if (global_row < row_count && global_input < input_rows) {
                    const unsigned char *row = encoded + (start_row + global_row) * row_bytes;
                    const unsigned char *q8 = activation +
                        ((global_input * group_count + row_group) * q8_blocks +
                         segment / 256ull) * YVEX_CUDA_Q8_K_BYTES;
                    unsigned int subblock = (unsigned int)((segment % 256ull) / 16ull);
                    unsigned int group_segments =
                        qtype == YVEX_GGUF_QTYPE_Q2_K ||
                                qtype == YVEX_GGUF_QTYPE_IQ2_XXS ? 16u : 2u;
                    integer_totals[warp][index] += product_tile[warp][index] *
                        tensorcore_product_factor(row, segment, qtype);
                    minimum_totals[warp][index] += q8_k_sum(q8, subblock) *
                        tensorcore_minimum_factor(row, segment, qtype);
                    if ((segment_index + 1u) % group_segments == 0u) {
                        totals[warp][index] = __fadd_rn(
                            totals[warp][index], tensorcore_scaled_group(
                                row, segment, qtype,
                                __uint_as_float(qtype_load_u32(q8)),
                                integer_totals[warp][index],
                                minimum_totals[warp][index]));
                        integer_totals[warp][index] = minimum_totals[warp][index] = 0;
                    }
                }
            }
        }
        tensorcore_row_sync(warps);
    }
    for (unsigned int index = lane; index < 256u; index += 32u) {
        unsigned int tile_row = index / 16u, tile_column = index % 16u;
        unsigned long long global_row = row_base + tile_row;
        unsigned long long global_input = input_base + tile_column;
        if (global_row < row_count && global_input < input_rows) {
            unsigned long long output_index = global_input * row_count + global_row;
            float value = additive
                ? __fadd_rn(totals[warp][index], additive[output_index])
                : totals[warp][index];
            if (!isfinite(value)) atomicCAS(status, 0, 1);
            else output[output_index] = output_bf16 ? float_to_bf16_rne(value) : value;
        }
    }
}

/* One warp maps one expert bucket to MMA columns. Every active column is a real routed row;
 * unused columns are only the bounded tail of the compiler-admitted physical tile. */
static __device__ void tensorcore_expert_bucket_dot(
    const tensorcore_expert_view *view, unsigned long long expert,
    unsigned long long row_base, const unsigned char *input,
    unsigned long long input_blocks, const unsigned long long *order,
    unsigned long long bucket_offset, unsigned long long population,
    unsigned long long topk, int ordered_input, signed char *weight_tile,
    signed char *activation_tile, int *product_tile,
    float reduction[16][6], float totals[16])
{
    unsigned int lane = threadIdx.x & 31u;
    int integer_totals[16], minimum_totals[16];
#pragma unroll
    for (unsigned int column = 0u; column < 16u; ++column) {
        totals[column] = 0.0f;
        integer_totals[column] = minimum_totals[column] = 0;
        if (lane < 16u)
            for (unsigned int level = 0u; level < 6u; ++level)
                reduction[column][level] = 0.0f;
    }
    __syncwarp();
    for (unsigned int leaf = 0u; leaf < 32u; ++leaf) {
        unsigned int block = ((leaf & 1u) << 4u) | ((leaf & 2u) << 2u) |
                             (leaf & 4u) | ((leaf & 8u) >> 2u) |
                             ((leaf & 16u) >> 4u);
        for (unsigned int column = 0u; column < 16u; ++column)
            integer_totals[column] = minimum_totals[column] = 0;
        if ((unsigned long long)block < input_blocks)
            for (unsigned int local_segment = 0u; local_segment < 16u;
                 ++local_segment) {
                unsigned long long segment =
                    (unsigned long long)block * 256ull + local_segment * 16ull;
                wmma::fragment<wmma::matrix_a, 16, 16, 16,
                               signed char, wmma::row_major> weight;
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               signed char, wmma::col_major> values;
                wmma::fragment<wmma::accumulator, 16, 16, 16, int> product;
                tensorcore_canonical_weight_tile(
                    view->base + expert * view->expert_bytes,
                    view->row_bytes, view->row_count, row_base, segment,
                    view->qtype, weight_tile);
                for (unsigned int index = lane; index < 256u; index += 32u) {
                    unsigned int column = index / 16u;
                    unsigned int input_lane = index % 16u;
                    if (column < population) {
                        unsigned long long ordered_pair = bucket_offset + column;
                        unsigned long long source_pair = order[ordered_pair];
                        unsigned long long source_row = ordered_input
                            ? ordered_pair
                            : source_pair / topk;
                        const unsigned char *activation = input +
                            source_row * input_blocks * YVEX_CUDA_Q8_K_BYTES;
                        activation_tile[index] = (signed char)activation[
                            (unsigned long long)block * YVEX_CUDA_Q8_K_BYTES +
                            4ull + local_segment * 16ull + input_lane];
                    } else {
                        activation_tile[index] = 0;
                    }
                }
                __syncwarp();
                wmma::load_matrix_sync(weight, weight_tile, 16u);
                wmma::load_matrix_sync(values, activation_tile, 16u);
                wmma::fill_fragment(product, 0);
                wmma::mma_sync(product, weight, values, product);
                wmma::store_matrix_sync(
                    product_tile, product, 16u, wmma::mem_row_major);
                __syncwarp();
                if (lane < 16u && row_base + lane < view->row_count) {
                    const unsigned char *row =
                        view->base + expert * view->expert_bytes +
                        (row_base + lane) * view->row_bytes;
                    int product_factor, minimum_factor;
                    product_factor = tensorcore_product_factor(row, segment, view->qtype);
                    minimum_factor = tensorcore_minimum_factor(row, segment, view->qtype);
                    for (unsigned int column = 0u; column < population;
                         ++column) {
                        unsigned long long ordered_pair = bucket_offset + column;
                        unsigned long long source_pair = order[ordered_pair];
                        unsigned long long source_row = ordered_input
                            ? ordered_pair
                            : source_pair / topk;
                        const unsigned char *q8 = input +
                            (source_row * input_blocks + block) *
                                YVEX_CUDA_Q8_K_BYTES;
                        integer_totals[column] +=
                            product_tile[lane * 16u + column] * product_factor;
                        minimum_totals[column] +=
                            q8_k_sum(q8, local_segment) * minimum_factor;
                        if (local_segment == 15u) {
                            float activation_scale =
                                __uint_as_float(qtype_load_u32(q8));
                            totals[column] = tensorcore_scaled_group(
                                row, segment, view->qtype, activation_scale,
                                integer_totals[column], minimum_totals[column]);
                        }
                    }
                }
                __syncwarp();
            }
        if (lane < 16u && row_base + lane < view->row_count) {
            for (unsigned int column = 0u; column < population; ++column) {
                float value = (unsigned long long)block < input_blocks
                    ? totals[column]
                    : 0.0f;
                unsigned int level = 0u;
                unsigned int complete = leaf + 1u;
                while (!(complete & (1u << level))) {
                    value = __fadd_rn(reduction[column][level], value);
                    level++;
                }
                reduction[column][level] = value;
            }
        }
        __syncwarp();
    }
    if (lane < 16u && row_base + lane < view->row_count)
        for (unsigned int column = 0u; column < population; ++column)
            totals[column] = reduction[column][5];
}

extern "C" __global__ void yvex_moe_grouped_up_tensorcore(
    const unsigned char *gate, unsigned long long gate_row_bytes,
    unsigned long long gate_expert_bytes, unsigned int gate_qtype,
    const unsigned char *up, unsigned long long up_row_bytes,
    unsigned long long up_expert_bytes, unsigned int up_qtype,
    const unsigned long long *selected, const float *weights,
    const unsigned long long *order, const unsigned long long *expert_ids,
    const unsigned long long *bucket_offsets,
    const unsigned long long *bucket_populations,
    const yvex_expert_worklist_observation *summary,
    unsigned long long pair_count, unsigned long long topk,
    unsigned long long expert_count, unsigned long long tensor_core_minimum,
    const unsigned char *input, unsigned long long input_width,
    unsigned long long intermediate_width, double limit,
    float *intermediate, int *status)
{
    __shared__ __align__(16) signed char weight_tiles[4u * 256u];
    __shared__ __align__(16) signed char activation_tiles[4u * 256u];
    __shared__ __align__(16) int product_tiles[4u * 256u];
    __shared__ float reductions[4u][16u][16u][6u];
    unsigned int warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    unsigned long long tiles = (intermediate_width + 15ull) / 16ull;
    unsigned long long task = (unsigned long long)blockIdx.x * 4ull + warp;
    unsigned long long bucket = tiles ? task / tiles : pair_count;
    unsigned long long row_base = tiles ? (task % tiles) * 16ull : intermediate_width;
    if (!status || *status || warp >= 4u || !summary || !expert_ids ||
        !bucket_offsets || !bucket_populations ||
        bucket >= summary->bucket_count) return;
    unsigned long long offset = bucket_offsets[bucket];
    unsigned long long population = bucket_populations[bucket];
    unsigned long long expert = expert_ids[bucket];
    unsigned long long input_blocks = input_width / YVEX_CUDA_Q8_K_BLOCK;
    if (!gate || !up || !selected || !weights || !order || !input || !intermediate ||
        !topk || !input_blocks || input_blocks > 32ull || !intermediate_width ||
        !tensor_core_minimum || gate_qtype != YVEX_GGUF_QTYPE_IQ2_XXS ||
        up_qtype != YVEX_GGUF_QTYPE_IQ2_XXS) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (population < tensor_core_minimum) return;
    if (population > 16ull || offset > pair_count ||
        population > pair_count - offset || expert >= expert_count ||
        !isfinite(limit) || limit <= 0.0) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (!lane) for (unsigned long long column = 0ull; column < population; ++column) {
        unsigned long long source_pair = order[offset + column];
        if (source_pair >= pair_count || selected[source_pair] != expert)
            atomicCAS(status, 0, 2);
    }
    __syncwarp();
    if (*status) return;
    tensorcore_expert_view gate_view = {
        gate, gate_row_bytes, gate_expert_bytes, input_width,
        intermediate_width, expert_count, gate_qtype};
    tensorcore_expert_view up_view = {
        up, up_row_bytes, up_expert_bytes, input_width,
        intermediate_width, expert_count, up_qtype};
    int gate_geometry = tensorcore_row_geometry(gate_qtype, input_width, gate_row_bytes);
    int up_geometry = tensorcore_row_geometry(up_qtype, input_width, up_row_bytes);
    if (*status || !gate_geometry || !up_geometry ||
        gate_expert_bytes != intermediate_width * gate_row_bytes ||
        up_expert_bytes != intermediate_width * up_row_bytes) {
        if (!lane && !*status) atomicCAS(status, 0, 2);
        return;
    }
    signed char *weight_tile = weight_tiles + warp * 256u;
    signed char *activation_tile = activation_tiles + warp * 256u;
    int *product_tile = product_tiles + warp * 256u;
    float gate_totals[16], up_totals[16];
    tensorcore_expert_bucket_dot(
        &gate_view, expert, row_base, input, input_blocks, order, offset,
        population, topk, 0, weight_tile, activation_tile, product_tile,
        reductions[warp][lane & 15u], gate_totals);
    tensorcore_expert_bucket_dot(
        &up_view, expert, row_base, input, input_blocks, order, offset,
        population, topk, 0, weight_tile, activation_tile, product_tile,
        reductions[warp][lane & 15u], up_totals);
    if (lane < 16u && row_base + lane < intermediate_width && !*status)
        for (unsigned int column = 0u; column < population; ++column) {
            unsigned long long ordered_pair = offset + column;
            unsigned long long source_pair = order[ordered_pair];
            float g = fminf(gate_totals[column], (float)limit);
            float u = fmaxf((float)-limit, fminf(up_totals[column], (float)limit));
            float silu = g >= 0.0f ? g / (1.0f + expf(-g))
                                   : g * expf(g) / (1.0f + expf(g));
            float value = float_to_bf16_rne(silu * u * weights[source_pair]);
            if (!isfinite(value)) atomicCAS(status, 0, 1);
            else intermediate[ordered_pair * intermediate_width + row_base + lane] = value;
        }
}

extern "C" __global__ void yvex_moe_grouped_down_tensorcore(
    const unsigned char *down, unsigned long long row_bytes,
    unsigned long long expert_bytes, unsigned int qtype,
    const unsigned long long *selected, const unsigned long long *order,
    const unsigned long long *expert_ids,
    const unsigned long long *bucket_offsets,
    const unsigned long long *bucket_populations,
    yvex_expert_worklist_observation *summary,
    unsigned long long pair_count, unsigned long long topk,
    unsigned long long expert_count, unsigned long long tensor_core_minimum,
    const unsigned char *intermediate, unsigned long long intermediate_width,
    unsigned long long hidden, float *pair_outputs, int *status)
{
    __shared__ __align__(16) signed char weight_tiles[4u * 256u];
    __shared__ __align__(16) signed char activation_tiles[4u * 256u];
    __shared__ __align__(16) int product_tiles[4u * 256u];
    __shared__ float reductions[4u][16u][16u][6u];
    unsigned int warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    unsigned long long tiles = (hidden + 15ull) / 16ull;
    unsigned long long task = (unsigned long long)blockIdx.x * 4ull + warp;
    unsigned long long bucket = tiles ? task / tiles : pair_count;
    unsigned long long row_base = tiles ? (task % tiles) * 16ull : hidden;
    if (!status || *status || warp >= 4u || !summary || !expert_ids ||
        !bucket_offsets || !bucket_populations ||
        bucket >= summary->bucket_count) return;
    unsigned long long offset = bucket_offsets[bucket];
    unsigned long long population = bucket_populations[bucket];
    unsigned long long expert = expert_ids[bucket];
    unsigned long long input_blocks = intermediate_width / YVEX_CUDA_Q8_K_BLOCK;
    if (!down || !selected || !order || !intermediate || !pair_outputs || !topk ||
        !input_blocks || input_blocks > 32ull || !hidden || !tensor_core_minimum ||
        qtype != YVEX_GGUF_QTYPE_Q2_K) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (population < tensor_core_minimum) return;
    if (population > 16ull || offset > pair_count ||
        population > pair_count - offset || expert >= expert_count) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (!lane) for (unsigned long long column = 0ull; column < population; ++column) {
        unsigned long long source_pair = order[offset + column];
        if (source_pair >= pair_count || selected[source_pair] != expert)
            atomicCAS(status, 0, 2);
    }
    __syncwarp();
    if (*status) return;
    tensorcore_expert_view down_view = {
        down, row_bytes, expert_bytes, intermediate_width, hidden, expert_count, qtype};
    int down_geometry = tensorcore_row_geometry(qtype, intermediate_width, row_bytes);
    if (*status || !down_geometry || expert_bytes != hidden * row_bytes) {
        if (!lane && !*status) atomicCAS(status, 0, 2);
        return;
    }
    float totals[16];
    tensorcore_expert_bucket_dot(
        &down_view, expert, row_base, intermediate, input_blocks, order,
        offset, population, topk, 1, weight_tiles + warp * 256u,
        activation_tiles + warp * 256u, product_tiles + warp * 256u,
        reductions[warp][lane & 15u], totals);
    if (lane < 16u && row_base + lane < hidden && !*status)
        for (unsigned int column = 0u; column < population; ++column) {
            unsigned long long source_pair = order[offset + column];
            float value = float_to_bf16_rne(totals[column]);
            if (!isfinite(value)) atomicCAS(status, 0, 1);
            else pair_outputs[source_pair * hidden + row_base + lane] = value;
        }
    if (!lane && !row_base && !*status)
        atomicAdd(&summary->matrix_tile_executed_pairs, population);
}
