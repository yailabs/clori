/*
 * Execute admitted low-bit rows by Q8_K activations through native integer Tensor Cores.
 *
 * Encoded scales remain outside MMA. Each 16-element segment is expanded directly into an
 * integer fragment, so Q8_0, Q2_K, IQ2_XXS, and MXFP4 keep their canonical block semantics
 * without materializing a dense dequantized matrix.
 */
#include "src/backend/cuda/kernel_primitives.h"
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

static __device__ int tensorcore_weight_i8(const unsigned char *row,
                                           unsigned long long index,
                                           unsigned int qtype)
{
    if (qtype == YVEX_GGUF_QTYPE_Q8_0) {
        const unsigned char *block = row + (index / 32ull) * 34ull;
        unsigned int raw = block[2u + (unsigned int)(index % 32ull)];
        return raw <= 127u ? (int)raw : (int)raw - 256;
    }
    if (qtype == YVEX_GGUF_QTYPE_Q2_K) {
        unsigned int lane = (unsigned int)(index % 256ull);
        unsigned int subblock = lane / 16u;
        unsigned int half = lane / 128u;
        unsigned int local_subblock = subblock & 7u;
        unsigned int pair = local_subblock & 1u;
        unsigned int group = local_subblock / 2u;
        const unsigned char *block = row + (index / 256ull) * 84ull;
        unsigned int packed = block[16u + half * 32u + pair * 16u + (lane & 15u)];
        return (int)((packed >> (group * 2u)) & 3u);
    }
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        unsigned int lane = (unsigned int)(index % 256ull);
        unsigned int group = lane / 32u;
        unsigned int subgroup = (lane & 31u) / 8u;
        unsigned int local = lane & 7u;
        const unsigned char *block = row + (index / 256ull) * 66ull;
        unsigned int grids = qtype_load_u32(block + 2u + group * 8u);
        unsigned int sign_scale = qtype_load_u32(block + 6u + group * 8u);
        unsigned int grid_index = (grids >> (8u * subgroup)) & 255u;
        unsigned int signs = iq2_xxs_signs((sign_scale >> (7u * subgroup)) & 127u);
        unsigned int digit = (iq2_xxs_grid[grid_index] >> (2u * local)) & 3u;
        int level = digit == 0u ? 8 : digit == 1u ? 25 : 43;
        return signs & (1u << local) ? -level : level;
    }
    if (qtype == YVEX_GGUF_QTYPE_MXFP4) {
        unsigned int lane = (unsigned int)(index % 32ull);
        const unsigned char *block = row + (index / 32ull) * 17ull;
        unsigned int packed = block[1u + (lane & 15u)];
        unsigned int code = lane < 16u ? packed & 15u : packed >> 4u;
        int magnitude = (code & 7u) == 0u ? 0 : (code & 7u) < 5u ? (int)(code & 7u)
                          : (code & 7u) == 5u ? 6 : (code & 7u) == 6u ? 8 : 12;
        return code & 8u ? -magnitude : magnitude;
    }
    return 0;
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
        return activation_scale *
               (f16_bits_to_float(qtype_load_u16(block + 80u)) * (float)product -
                f16_bits_to_float(qtype_load_u16(block + 82u)) * (float)minimum);
    }
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        const unsigned char *block = row + (segment / 256ull) * 66ull;
        return 0.125f * activation_scale *
               f16_bits_to_float(qtype_load_u16(block)) * (float)product;
    }
    const unsigned char *block = row + (segment / 32ull) * 17ull;
    return 0.5f * activation_scale * e8m0_bits_to_float(block[0]) *
           (float)product;
}

typedef struct {
    const unsigned char *base;
    unsigned long long row_bytes, expert_bytes, storage_bytes;
    unsigned long long row_width, row_count, expert_count;
    unsigned long long blocks_per_row, block_count, scale_offset, code_offset;
    unsigned int qtype, layout;
} tensorcore_expert_view;

static __device__ unsigned long long tensorcore_align64(unsigned long long value)
{
    return (value + 63ull) & ~63ull;
}

static __device__ int tensorcore_derived_geometry(tensorcore_expert_view *view)
{
    if (!view || !view->layout ||
        !view->base || !view->row_width || view->row_width % 256ull ||
        !view->row_count || !view->expert_count) return 0;
    view->blocks_per_row = view->row_width / 256ull;
    view->block_count = view->expert_count * view->row_count * view->blocks_per_row;
    if (view->qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        view->scale_offset = 0ull;
        view->code_offset = tensorcore_align64(view->block_count * 2ull);
        return view->storage_bytes == view->code_offset + view->block_count * 64ull;
    }
    if (view->qtype == YVEX_GGUF_QTYPE_Q2_K && !(view->row_count & 1ull)) {
        unsigned long long pairs = view->expert_count * (view->row_count / 2ull) *
                                   view->blocks_per_row;
        view->scale_offset = tensorcore_align64(pairs * 8ull);
        view->code_offset = tensorcore_align64(view->scale_offset + pairs * 32ull);
        return view->storage_bytes == view->code_offset + pairs * 128ull;
    }
    return 0;
}

static __device__ const unsigned char *tensorcore_derived_q2_field(
    const tensorcore_expert_view *view, unsigned long long expert,
    unsigned long long row, unsigned long long index,
    unsigned int field)
{
    unsigned long long block = index / 256ull, lane = index % 256ull;
    unsigned long long pair, parity = row & 1ull;
    pair = (expert * (view->row_count / 2ull) + row / 2ull) *
               view->blocks_per_row + block;
    if (field == 0u) return view->base + pair * 8ull + parity * 4ull;
    if (field == 1u) {
        unsigned long long word = (lane / 16ull) / 4ull;
        unsigned long long byte = (lane / 16ull) & 3ull;
        unsigned long long target = pair * 32ull + (word / 2ull) * 16ull +
                                    parity * 8ull + (word & 1ull) * 4ull + byte;
        return view->base + view->scale_offset + target;
    }
    {
        unsigned long long raw = (lane / 128ull) * 32ull +
                                 ((lane / 16ull) & 1ull) * 16ull + (lane & 15ull);
        unsigned long long word = raw / 4ull, byte = raw & 3ull;
        return view->base + view->code_offset + pair * 128ull + word * 8ull +
               parity * 4ull + byte;
    }
}

static __device__ int tensorcore_derived_weight_i8(
    const tensorcore_expert_view *view, unsigned long long expert,
    unsigned long long row, unsigned long long index)
{
    if (view->qtype == YVEX_GGUF_QTYPE_Q2_K) {
        const unsigned char *packed = tensorcore_derived_q2_field(
            view, expert, row, index, 2u);
        unsigned int group = (unsigned int)(((index % 256ull) / 16ull) & 7ull) / 2u;
        return packed ? (int)((*packed >> (group * 2u)) & 3u) : 0;
    }
    unsigned long long block =
        (expert * view->row_count + row) * view->blocks_per_row + index / 256ull;
    unsigned int lane = (unsigned int)(index % 256ull);
    const unsigned char *codes = view->base + view->code_offset + block * 64ull;
    unsigned int group = lane / 32u, subgroup = (lane & 31u) / 8u, local = lane & 7u;
    unsigned int grids = qtype_load_u32(codes + group * 8u);
    unsigned int sign_scale = qtype_load_u32(codes + group * 8u + 4u);
    unsigned int grid_index = (grids >> (8u * subgroup)) & 255u;
    unsigned int signs = iq2_xxs_signs((sign_scale >> (7u * subgroup)) & 127u);
    unsigned int digit = (iq2_xxs_grid[grid_index] >> (2u * local)) & 3u;
    int level = digit == 0u ? 8 : digit == 1u ? 25 : 43;
    return signs & (1u << local) ? -level : level;
}

static __device__ void tensorcore_derived_factors(
    const tensorcore_expert_view *view, unsigned long long expert,
    unsigned long long row, unsigned long long segment,
    int *product, int *minimum, float *scale)
{
    if (view->qtype == YVEX_GGUF_QTYPE_Q2_K) {
        const unsigned char *factor = tensorcore_derived_q2_field(
            view, expert, row, segment, 1u);
        const unsigned char *dm = tensorcore_derived_q2_field(
            view, expert, row, segment, 0u);
        *product = factor ? (int)(*factor & 15u) : 0;
        *minimum = factor ? (int)(*factor >> 4u) : 0;
        *scale = dm ? f16_bits_to_float(qtype_load_u16(dm)) : 0.0f;
        scale[1] = dm ? f16_bits_to_float(qtype_load_u16(dm + 2u)) : 0.0f;
    } else {
        unsigned long long block =
            (expert * view->row_count + row) * view->blocks_per_row + segment / 256ull;
        const unsigned char *codes = view->base + view->code_offset + block * 64ull;
        unsigned int group = (unsigned int)((segment % 256ull) / 32ull);
        unsigned int sign_scale = qtype_load_u32(codes + group * 8u + 4u);
        *product = (int)(2u * (sign_scale >> 28u) + 1u);
        *minimum = 0;
        *scale = 0.125f * f16_bits_to_float(qtype_load_u16(view->base + block * 2ull));
    }
}

extern "C" __global__ void yvex_qtype_tensorcore_rows(
    const unsigned char *encoded, unsigned long long row_bytes,
    unsigned long long row_width, unsigned long long start_row,
    unsigned long long row_count, unsigned long long input_rows,
    unsigned int qtype, const unsigned char *activation, const float *additive,
    float *output, int output_bf16, int *status)
{
    __shared__ __align__(16) signed char weight_tile[16 * 16];
    __shared__ __align__(16) signed char activation_tile[16 * 16];
    __shared__ __align__(16) int product_tile[16 * 16];
    __shared__ int integer_totals[16 * 16];
    __shared__ int minimum_totals[16 * 16];
    __shared__ float totals[16 * 16];
    unsigned int lane = threadIdx.x;
    unsigned long long input_tiles = (input_rows + 15ull) / 16ull;
    unsigned long long row_base = input_tiles
                                      ? ((unsigned long long)blockIdx.x / input_tiles) * 16ull
                                      : row_count;
    unsigned long long input_base = input_tiles
                                        ? ((unsigned long long)blockIdx.x % input_tiles) * 16ull
                                        : input_rows;
    unsigned long long q8_blocks = row_width / YVEX_CUDA_Q8_K_BLOCK;

    if (!status || *status || lane >= 32u) return;
    if (!encoded || !activation || !output || !row_count || !input_rows ||
        !tensorcore_row_geometry(qtype, row_width, row_bytes)) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    for (unsigned int index = lane; index < 256u; index += 32u) {
        integer_totals[index] = minimum_totals[index] = 0;
        totals[index] = 0.0f;
    }
    __syncthreads();
    for (unsigned long long segment = 0ull; segment < row_width; segment += 16ull) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       signed char, wmma::row_major> weight;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       signed char, wmma::col_major> values;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> product;
        for (unsigned int index = lane; index < 256u; index += 32u) {
            unsigned int tile_row = index / 16u, tile_column = index % 16u;
            unsigned long long global_row = row_base + tile_row;
            unsigned long long global_input = input_base + tile_row;
            weight_tile[index] = global_row < row_count
                ? (signed char)tensorcore_weight_i8(
                      encoded + (start_row + global_row) * row_bytes,
                      segment + tile_column, qtype) : 0;
            activation_tile[index] = global_input < input_rows
                ? (signed char)activation[(global_input * q8_blocks + segment / 256ull) *
                                              YVEX_CUDA_Q8_K_BYTES +
                                          4ull + segment % 256ull + tile_column]
                : 0;
        }
        __syncthreads();
        wmma::load_matrix_sync(weight, weight_tile, 16u);
        wmma::load_matrix_sync(values, activation_tile, 16u);
        wmma::fill_fragment(product, 0);
        wmma::mma_sync(product, weight, values, product);
        wmma::store_matrix_sync(product_tile, product, 16u, wmma::mem_row_major);
        __syncthreads();
        for (unsigned int index = lane; index < 256u; index += 32u) {
            unsigned int tile_row = index / 16u, tile_column = index % 16u;
            unsigned long long global_row = row_base + tile_row;
            unsigned long long global_input = input_base + tile_column;
            if (global_row < row_count && global_input < input_rows) {
                const unsigned char *row = encoded + (start_row + global_row) * row_bytes;
                const unsigned char *q8 = activation +
                    (global_input * q8_blocks + segment / 256ull) * YVEX_CUDA_Q8_K_BYTES;
                unsigned int subblock = (unsigned int)((segment % 256ull) / 16ull);
                unsigned int group_segments =
                    qtype == YVEX_GGUF_QTYPE_Q2_K ||
                            qtype == YVEX_GGUF_QTYPE_IQ2_XXS ? 16u : 2u;
                integer_totals[index] += product_tile[index] *
                                         tensorcore_product_factor(row, segment, qtype);
                minimum_totals[index] += q8_k_sum(q8, subblock) *
                                         tensorcore_minimum_factor(row, segment, qtype);
                if (((segment / 16ull) + 1ull) % group_segments == 0ull) {
                    totals[index] = __fadd_rn(
                        totals[index], tensorcore_scaled_group(
                            row, segment, qtype,
                            __uint_as_float(qtype_load_u32(q8)),
                            integer_totals[index], minimum_totals[index]));
                    integer_totals[index] = minimum_totals[index] = 0;
                }
            }
        }
        __syncthreads();
    }
    for (unsigned int index = lane; index < 256u; index += 32u) {
        unsigned int tile_row = index / 16u, tile_column = index % 16u;
        unsigned long long global_row = row_base + tile_row;
        unsigned long long global_input = input_base + tile_column;
        if (global_row < row_count && global_input < input_rows) {
            unsigned long long output_index = global_input * row_count + global_row;
            float value = additive ? __fadd_rn(totals[index], additive[output_index])
                                   : totals[index];
            if (!isfinite(value)) atomicCAS(status, 0, 1);
            else output[output_index] = output_bf16 ? float_to_bf16_rne(value) : value;
        }
    }
}

/* One warp computes sixteen output rows for one selected expert and activation row. The
 * unused MMA columns remain zero; this keeps expert choice device-resident while one native
 * instruction replaces sixteen independent warp reductions. */
static __device__ float tensorcore_selected_tile_dot(
    const tensorcore_expert_view *view, unsigned long long expert,
    unsigned long long row_base,
    const unsigned char *activation, signed char *weight_tile,
    signed char *activation_tile, int *product_tile)
{
    unsigned int lane = threadIdx.x & 31u;
    int integer_total = 0, minimum_total = 0;
    float total = 0.0f;
    for (unsigned long long segment = 0ull; segment < view->row_width; segment += 16ull) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       signed char, wmma::row_major> weight;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       signed char, wmma::col_major> values;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> product;
        for (unsigned int index = lane; index < 256u; index += 32u) {
            unsigned int tile_row = index / 16u, tile_column = index % 16u;
            unsigned long long output_row = row_base + tile_row;
            weight_tile[index] = output_row < view->row_count
                ? (signed char)(view->layout
                      ? tensorcore_derived_weight_i8(
                            view, expert, output_row, segment + tile_column)
                      : tensorcore_weight_i8(
                            view->base + expert * view->expert_bytes +
                                output_row * view->row_bytes,
                            segment + tile_column, view->qtype)) : 0;
            activation_tile[index] = index < 16u
                ? (signed char)activation[(segment / 256ull) *
                                              YVEX_CUDA_Q8_K_BYTES +
                                          4ull + segment % 256ull + index]
                : 0;
        }
        __syncwarp();
        wmma::load_matrix_sync(weight, weight_tile, 16u);
        wmma::load_matrix_sync(values, activation_tile, 16u);
        wmma::fill_fragment(product, 0);
        wmma::mma_sync(product, weight, values, product);
        wmma::store_matrix_sync(product_tile, product, 16u, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16u && row_base + lane < view->row_count) {
            const unsigned char *row = view->base + expert * view->expert_bytes +
                                       (row_base + lane) * view->row_bytes;
            const unsigned char *q8 = activation +
                (segment / 256ull) * YVEX_CUDA_Q8_K_BYTES;
            unsigned int subblock = (unsigned int)((segment % 256ull) / 16ull);
            unsigned int group_segments =
                view->qtype == YVEX_GGUF_QTYPE_Q2_K ||
                        view->qtype == YVEX_GGUF_QTYPE_IQ2_XXS ? 16u : 2u;
            int product_factor, minimum_factor;
            float derived_scale[2] = {0.0f, 0.0f};
            if (view->layout)
                tensorcore_derived_factors(
                    view, expert, row_base + lane, segment, &product_factor,
                    &minimum_factor, derived_scale);
            else {
                product_factor = tensorcore_product_factor(row, segment, view->qtype);
                minimum_factor = tensorcore_minimum_factor(row, segment, view->qtype);
            }
            integer_total += product_tile[lane * 16u] * product_factor;
            minimum_total += q8_k_sum(q8, subblock) * minimum_factor;
            if (((segment / 16ull) + 1ull) % group_segments == 0ull) {
                float activation_scale = __uint_as_float(qtype_load_u32(q8));
                float value = view->layout
                    ? activation_scale *
                          (derived_scale[0] * (float)integer_total -
                           derived_scale[1] * (float)minimum_total)
                    : tensorcore_scaled_group(
                          row, segment, view->qtype, activation_scale,
                          integer_total, minimum_total);
                total = __fadd_rn(total, value);
                integer_total = minimum_total = 0;
            }
        }
        __syncwarp();
    }
    return total;
}

extern "C" __global__ void yvex_moe_grouped_up_tensorcore(
    const unsigned char *gate, unsigned long long gate_row_bytes,
    unsigned long long gate_expert_bytes, unsigned long long gate_storage_bytes,
    unsigned int gate_layout, unsigned int gate_qtype,
    const unsigned char *up, unsigned long long up_row_bytes,
    unsigned long long up_expert_bytes, unsigned long long up_storage_bytes,
    unsigned int up_layout, unsigned int up_qtype,
    const unsigned long long *selected, const unsigned long long *order,
    unsigned long long pair_count, unsigned long long topk,
    unsigned long long expert_count, const unsigned char *input,
    unsigned long long input_width, unsigned long long intermediate_width,
    double limit, float *intermediate, int *status)
{
    __shared__ __align__(16) signed char weight_tiles[8u * 256u];
    __shared__ __align__(16) signed char activation_tiles[8u * 256u];
    __shared__ __align__(16) int product_tiles[8u * 256u];
    unsigned int warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    unsigned long long tiles = (intermediate_width + 15ull) / 16ull;
    unsigned long long task = (unsigned long long)blockIdx.x * 8ull + warp;
    unsigned long long ordered_pair = tiles ? task / tiles : pair_count;
    unsigned long long row_base = tiles ? (task % tiles) * 16ull : intermediate_width;
    if (!status || *status || warp >= 8u || ordered_pair >= pair_count) return;
    unsigned long long source_pair = order ? order[ordered_pair] : ordered_pair;
    if (source_pair >= pair_count) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long source_row = topk ? source_pair / topk : pair_count;
    unsigned long long expert = selected ? selected[source_pair] : 0ull;
    unsigned long long input_blocks = input_width / YVEX_CUDA_Q8_K_BLOCK;
    tensorcore_expert_view gate_view = {
        gate, gate_row_bytes, gate_expert_bytes, gate_storage_bytes,
        input_width, intermediate_width, expert_count, 0ull, 0ull, 0ull, 0ull,
        gate_qtype, gate_layout};
    tensorcore_expert_view up_view = {
        up, up_row_bytes, up_expert_bytes, up_storage_bytes,
        input_width, intermediate_width, expert_count, 0ull, 0ull, 0ull, 0ull,
        up_qtype, up_layout};
    int gate_geometry = gate_layout
        ? tensorcore_derived_geometry(&gate_view)
        : tensorcore_row_geometry(gate_qtype, input_width, gate_row_bytes);
    int up_geometry = up_layout
        ? tensorcore_derived_geometry(&up_view)
        : tensorcore_row_geometry(up_qtype, input_width, up_row_bytes);
    if (!gate || !up || !input || !intermediate || !topk ||
        expert >= expert_count ||
        !input_blocks || !intermediate_width || !isfinite(limit) || limit <= 0.0 ||
        !gate_geometry || !up_geometry ||
        gate_expert_bytes != intermediate_width * gate_row_bytes ||
        up_expert_bytes != intermediate_width * up_row_bytes) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    const unsigned char *activation = input + source_row * input_blocks *
                                                 YVEX_CUDA_Q8_K_BYTES;
    signed char *weight_tile = weight_tiles + warp * 256u;
    signed char *activation_tile = activation_tiles + warp * 256u;
    int *product_tile = product_tiles + warp * 256u;
    float g = tensorcore_selected_tile_dot(
        &gate_view, expert, row_base, activation,
        weight_tile, activation_tile, product_tile);
    float u = tensorcore_selected_tile_dot(
        &up_view, expert, row_base, activation,
        weight_tile, activation_tile, product_tile);
    if (lane < 16u && row_base + lane < intermediate_width && !*status) {
        g = fminf(g, (float)limit);
        u = fmaxf((float)-limit, fminf(u, (float)limit));
        float silu = g >= 0.0f ? g / (1.0f + expf(-g))
                               : g * expf(g) / (1.0f + expf(g));
        float value = float_to_bf16_rne(silu * u);
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else intermediate[ordered_pair * intermediate_width + row_base + lane] = value;
    }
}

extern "C" __global__ void yvex_moe_grouped_down_tensorcore(
    const unsigned char *down, unsigned long long row_bytes,
    unsigned long long expert_bytes, unsigned long long storage_bytes,
    unsigned int layout, unsigned int qtype,
    const unsigned long long *selected, const float *weights,
    const unsigned long long *order, unsigned long long pair_count,
    unsigned long long topk, unsigned long long expert_count,
    const unsigned char *intermediate, unsigned long long intermediate_width,
    unsigned long long hidden, float *pair_outputs, int *status)
{
    __shared__ __align__(16) signed char weight_tiles[8u * 256u];
    __shared__ __align__(16) signed char activation_tiles[8u * 256u];
    __shared__ __align__(16) int product_tiles[8u * 256u];
    unsigned int warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    unsigned long long tiles = (hidden + 15ull) / 16ull;
    unsigned long long task = (unsigned long long)blockIdx.x * 8ull + warp;
    unsigned long long ordered_pair = tiles ? task / tiles : pair_count;
    unsigned long long row_base = tiles ? (task % tiles) * 16ull : hidden;
    if (!status || *status || warp >= 8u || ordered_pair >= pair_count) return;
    unsigned long long source_pair = order ? order[ordered_pair] : ordered_pair;
    if (source_pair >= pair_count) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long expert = selected ? selected[source_pair] : 0ull;
    unsigned long long intermediate_blocks =
        intermediate_width / YVEX_CUDA_Q8_K_BLOCK;
    tensorcore_expert_view down_view = {
        down, row_bytes, expert_bytes, storage_bytes, intermediate_width,
        hidden, expert_count, 0ull, 0ull, 0ull, 0ull, qtype, layout};
    int down_geometry = layout
        ? tensorcore_derived_geometry(&down_view)
        : tensorcore_row_geometry(qtype, intermediate_width, row_bytes);
    if (!down || !intermediate || !pair_outputs || !topk ||
        expert >= expert_count ||
        !intermediate_blocks || !hidden ||
        !down_geometry ||
        expert_bytes != hidden * row_bytes) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    const unsigned char *activation = intermediate + ordered_pair *
        intermediate_blocks * YVEX_CUDA_Q8_K_BYTES;
    float dot = tensorcore_selected_tile_dot(
        &down_view, expert, row_base, activation, weight_tiles + warp * 256u,
        activation_tiles + warp * 256u, product_tiles + warp * 256u);
    if (lane < 16u && row_base + lane < hidden && !*status) {
        float route_weight = weights ? weights[source_pair] : 1.0f;
        float value = __fmul_rn(float_to_bf16_rne(dot), route_weight);
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else pair_outputs[source_pair * hidden + row_base + lane] = value;
    }
}
