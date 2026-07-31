/*
 * Judge row orientation and every output-head vocabulary value independently. No production
 * projection or qtype decode implementation is called. Numerical test evidence only; never
 * linked into production products.
 */
#include "tests/reference/logits.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <yvex/qtype.h>

static float reference_bf16(const unsigned char *bytes)
{
    uint32_t bits = ((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u)) << 16u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/*
 * Independently project every encoded row in canonical vocabulary order.
 *
 * Performs no production calls, allocation, identity, or publication.
 */
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
