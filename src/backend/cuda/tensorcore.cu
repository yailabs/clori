/*
 * Execute Q8_0 by Q8_K row batches through native integer Tensor Cores.
 *
 * A warp owns one 16x16 output tile. Encoded block scales stay outside MMA so the
 * arithmetic remains the canonical per-block Q8 product rather than pretending
 * that differently scaled blocks form one integer matrix.
 */
#include "src/backend/cuda/kernel_primitives.h"
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void yvex_q8_0_tensorcore_rows(
    const unsigned char *encoded, unsigned long long row_bytes,
    unsigned long long row_width, unsigned long long start_row,
    unsigned long long row_count, unsigned long long input_rows,
    const unsigned char *activation, const float *additive,
    float *output, int *status)
{
    __shared__ __align__(16) signed char weight_tile[16 * 16];
    __shared__ __align__(16) signed char activation_tile[16 * 16];
    __shared__ __align__(16) int product_tile[16 * 16];
    __shared__ float totals[16 * 16];
    unsigned int lane = threadIdx.x;
    unsigned long long input_tiles = (input_rows + 15ull) / 16ull;
    unsigned long long row_base = input_tiles
                                      ? ((unsigned long long)blockIdx.x / input_tiles) * 16ull
                                      : row_count;
    unsigned long long input_base = input_tiles
                                        ? ((unsigned long long)blockIdx.x % input_tiles) * 16ull
                                        : input_rows;
    unsigned long long q8_0_blocks = row_width / 32ull;
    unsigned long long q8_k_blocks = row_width / YVEX_CUDA_Q8_K_BLOCK;

    if (!status || *status || lane >= 32u) return;
    if (!encoded || !activation || !output || !row_bytes || !row_width ||
        !row_count || !input_rows || row_width % YVEX_CUDA_Q8_K_BLOCK ||
        row_bytes != q8_0_blocks * 34ull) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    for (unsigned int index = lane; index < 256u; index += 32u)
        totals[index] = 0.0f;
    __syncthreads();

    for (unsigned long long block = 0ull; block < q8_0_blocks; ++block) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       signed char, wmma::row_major> weight;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       signed char, wmma::col_major> values;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> product;
        unsigned long long activation_block = block / 8ull;
        unsigned long long activation_group = block % 8ull;

        wmma::fill_fragment(product, 0);
        for (unsigned int half = 0u; half < 2u; ++half) {
            for (unsigned int index = lane; index < 256u; index += 32u) {
                unsigned int tile_row = index / 16u;
                unsigned int tile_column = index % 16u;
                unsigned long long global_row = row_base + tile_row;
                unsigned long long global_input = input_base + tile_row;
                if (global_row < row_count) {
                    const unsigned char *source =
                        encoded + (start_row + global_row) * row_bytes + block * 34ull;
                    unsigned int raw = source[2u + half * 16u + tile_column];
                    weight_tile[index] = (signed char)(raw <= 127u ? raw : raw - 256u);
                } else {
                    weight_tile[index] = 0;
                }
                if (global_input < input_rows) {
                    const unsigned char *source = activation +
                        (global_input * q8_k_blocks + activation_block) *
                            YVEX_CUDA_Q8_K_BYTES;
                    unsigned int raw = source[4u + activation_group * 32u +
                                              half * 16u + tile_column];
                    activation_tile[tile_row * 16u + tile_column] =
                        (signed char)(raw <= 127u ? raw : raw - 256u);
                } else {
                    activation_tile[tile_row * 16u + tile_column] = 0;
                }
            }
            __syncthreads();
            wmma::load_matrix_sync(weight, weight_tile, 16u);
            wmma::load_matrix_sync(values, activation_tile, 16u);
            wmma::mma_sync(product, weight, values, product);
            __syncthreads();
        }
        wmma::store_matrix_sync(product_tile, product, 16u, wmma::mem_row_major);
        __syncthreads();
        for (unsigned int index = lane; index < 256u; index += 32u) {
            unsigned int tile_row = index / 16u;
            unsigned int tile_column = index % 16u;
            unsigned long long global_row = row_base + tile_row;
            unsigned long long global_input = input_base + tile_column;
            if (global_row < row_count && global_input < input_rows) {
                const unsigned char *weight_block =
                    encoded + (start_row + global_row) * row_bytes + block * 34ull;
                const unsigned char *activation_scale = activation +
                    (global_input * q8_k_blocks + activation_block) *
                        YVEX_CUDA_Q8_K_BYTES;
                totals[index] = fmaf(
                    f16_bits_to_float(qtype_load_u16(weight_block)) *
                        __uint_as_float(qtype_load_u32(activation_scale)),
                    (float)product_tile[index], totals[index]);
            }
        }
        __syncthreads();
    }
    for (unsigned int index = lane; index < 256u; index += 32u) {
        unsigned int tile_row = index / 16u;
        unsigned int tile_column = index % 16u;
        unsigned long long global_row = row_base + tile_row;
        unsigned long long global_input = input_base + tile_column;
        if (global_row < row_count && global_input < input_rows) {
            unsigned long long output_index = global_input * row_count + global_row;
            float value = additive
                              ? __fadd_rn(totals[index], additive[output_index])
                              : totals[index];
            if (!isfinite(value)) atomicCAS(status, 0, 1);
            else output[output_index] = value;
        }
    }
}
