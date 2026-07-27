/* Owner: test-only output-head numerical reference.
 * Owns: independent scalar BF16 row decode and complete vocabulary projection.
 * Does not own: runtime plans, residency, production qtype arithmetic, CUDA, or capability.
 * Invariants: no production projection or qtype decode implementation is called.
 * Boundary: numerical test evidence only; never linked into production products.
 * Purpose: judge row orientation and every output-head vocabulary value independently.
 * Inputs: exact encoded row geometry, hidden values, and caller output.
 * Effects: writes one complete reference row or nothing useful.
 * Failure: malformed geometry, unsupported qtype, or non-finite values return false. */
#include "tests/reference/logits.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <yvex/qtype.h>

/* Purpose: decode one little-endian BF16 value without production codecs. */
static float reference_bf16(const unsigned char *bytes)
{
    uint32_t bits = ((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u)) << 16u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Purpose: independently project every encoded row in canonical vocabulary order.
 * Inputs: BF16 row-major bytes, exact shape, finite hidden row, and caller capacity.
 * Effects: fills the complete caller logits row only for valid geometry.
 * Failure: returns false for unsupported qtype, extent, capacity, or non-finite math.
 * Boundary: performs no production calls, allocation, identity, or publication. */
int yvex_test_logits_reference_project(
    unsigned int qtype, const unsigned char *encoded, size_t encoded_bytes,
    unsigned long long rows, unsigned long long width,
    unsigned long long row_bytes, const float *hidden,
    float *logits, unsigned long long logits_capacity)
{
    unsigned long long row, column;
    if (qtype != YVEX_GGUF_QTYPE_BF16 || !encoded || !rows || !width ||
        row_bytes != width * 2ull || rows > SIZE_MAX / (size_t)row_bytes ||
        encoded_bytes != (size_t)(rows * row_bytes) || !hidden || !logits ||
        logits_capacity < rows) return 0;
    for (row = 0ull; row < rows; ++row) {
        double sum = 0.0;
        for (column = 0ull; column < width; ++column) {
            float weight = reference_bf16(
                encoded + row * row_bytes + column * 2ull);
            if (!isfinite(weight) || !isfinite(hidden[column])) return 0;
            sum += (double)weight * (double)hidden[column];
        }
        if (!isfinite(sum)) return 0;
        logits[row] = (float)sum;
    }
    return 1;
}
