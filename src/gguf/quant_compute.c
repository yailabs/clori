/*
 * Accumulate independent numeric error facts and evaluate encoded row dots.
 *
 * Metrics never retain tensors; CPU compute consumes encoded blocks directly and rejects
 * non-divisible rows and malformed byte counts. A qtype row dot is bounded compute evidence, not
 * model execution.
 */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <yvex/internal/quant_numeric.h>

void yvex_quant_metrics_init(yvex_quant_metrics *metrics) {
    if (metrics)
        memset(metrics, 0, sizeof(*metrics));
}

static int quant_metrics_add(double *total, double term) {
    double next;

    if (!total || !isfinite(*total) || !isfinite(term))
        return 0;
    next = *total + term;
    if (!isfinite(next))
        return 0;
    *total = next;
    return 1;
}

/*
 * Compare a bounded reconstructed slice against an independent reference slice.
 *
 * Null inputs, count overflow, or non-finite derived arithmetic leave metrics unchanged.
 */
int yvex_quant_metrics_update(yvex_quant_metrics *metrics, const float *reference,
                              const float *reconstructed, const float *dot_vector,
                              unsigned long long count) {
    yvex_quant_metrics next;
    unsigned long long index;

    if (!metrics || !reference || !reconstructed || count == 0u ||
        ULLONG_MAX - metrics->element_count < count || ULLONG_MAX - metrics->finite_count < count ||
        ULLONG_MAX - metrics->nonfinite_count < count)
        return 0;
    next = *metrics;
    for (index = 0u; index < count; ++index) {
        double expected = reference[index];
        double observed = reconstructed[index];
        double error;
        double absolute;
        double denominator;
        double dot = dot_vector ? dot_vector[index] : (double)((int)(index % 17u) - 8) / 8.0;

        if (!isfinite(expected) || !isfinite(observed)) {
            next.nonfinite_count++;
            continue;
        }
        if (!isfinite(dot))
            return 0;
        next.finite_count++;
        error = observed - expected;
        absolute = fabs(error);
        if (!isfinite(absolute))
            return 0;
        if (absolute > next.maximum_absolute_error)
            next.maximum_absolute_error = absolute;
        if (!quant_metrics_add(&next.absolute_error_sum, absolute) ||
            !quant_metrics_add(&next.squared_error_sum, error * error))
            return 0;
        denominator = fabs(expected);
        if (denominator < 1e-12)
            denominator = 1.0;
        if (!quant_metrics_add(&next.relative_error_sum, absolute / denominator) ||
            !quant_metrics_add(&next.reference_squared_sum, expected * expected) ||
            !quant_metrics_add(&next.dot_reference, expected * dot) ||
            !quant_metrics_add(&next.dot_reconstructed, observed * dot))
            return 0;
    }
    next.element_count += count;
    *metrics = next;
    return 1;
}

double yvex_quant_metrics_rmse(const yvex_quant_metrics *metrics) {
    return metrics && metrics->finite_count
               ? sqrt(metrics->squared_error_sum / (double)metrics->finite_count)
               : 0.0;
}

static int quant_compute_fail(yvex_quant_failure *failure, yvex_quant_failure_code code,
                              unsigned int qtype, unsigned long long expected,
                              unsigned long long actual, yvex_error *err, int status,
                              const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->terminal_ordinal = ULLONG_MAX;
        failure->source_index = ULLONG_MAX;
        failure->row_index = ULLONG_MAX;
        failure->block_index = ULLONG_MAX;
        failure->expected = expected;
        failure->actual = actual;
        failure->qtype = qtype;
        failure->operation = YVEX_TRANSFORM_OP_COUNT;
    }
    yvex_error_set(err, (yvex_status)status, "quant.cpu_dot", message);
    return status;
}

/*
 * Compute one encoded qtype row dot directly from canonical blocks.
 *
 * Unsupported qtypes, malformed rows, overflow, or non-finite terms return typed refusal.
 */
int yvex_quant_cpu_dot(unsigned int qtype, const unsigned char *encoded, size_t encoded_bytes,
                       const float *vector, unsigned long long elements, float *out,
                       yvex_quant_failure *failure, yvex_error *err) {
    const yvex_quant_numeric_capability *capability = yvex_quant_numeric_capability_at(qtype);
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(qtype);
    unsigned long long blocks;
    unsigned long long block;
    double sum = 0.0;
    float values[YVEX_QUANT_Q2_K_ELEMENTS];

    if (!encoded || !vector || !out || elements == 0u) {
        return quant_compute_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, qtype, 1u, 0u, err,
                                  YVEX_ERR_INVALID_ARG,
                                  "encoded row, vector, element count, and result are required");
    }
    if (!capability || !capability->dedicated_cpu_compute_available || !geometry ||
        geometry->block_size == 0u || geometry->bytes_per_block == 0u) {
        return quant_compute_fail(failure, YVEX_QUANT_FAILURE_CPU_COMPUTE_UNAVAILABLE, qtype, 1u,
                                  0u, err, YVEX_ERR_UNSUPPORTED,
                                  "qtype has no dedicated CPU row-dot implementation");
    }
    if (elements % geometry->block_size != 0u) {
        return quant_compute_fail(failure, YVEX_QUANT_FAILURE_ROW_DIVISIBILITY, qtype,
                                  geometry->block_size, elements, err, YVEX_ERR_BOUNDS,
                                  "qtype row width is not block divisible");
    }
    blocks = elements / geometry->block_size;
    if (blocks > SIZE_MAX / geometry->bytes_per_block ||
        (size_t)blocks * geometry->bytes_per_block != encoded_bytes) {
        return quant_compute_fail(
            failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, qtype,
            blocks <= SIZE_MAX / geometry->bytes_per_block
                ? (unsigned long long)((size_t)blocks * geometry->bytes_per_block)
                : ULLONG_MAX,
            encoded_bytes, err, YVEX_ERR_BOUNDS, "encoded qtype row byte count is inconsistent");
    }
    for (block = 0u; block < blocks; ++block) {
        unsigned long long lane;
        int rc = yvex_quant_decode_block(qtype, encoded + (size_t)block * geometry->bytes_per_block,
                                         geometry->bytes_per_block, values, geometry->block_size,
                                         failure, err);
        if (rc != YVEX_OK) {
            if (failure)
                failure->block_index = block;
            return rc;
        }
        for (lane = 0u; lane < geometry->block_size; ++lane) {
            double term = (double)values[lane] * vector[block * geometry->block_size + lane];
            if (!isfinite(term) || !isfinite(sum + term))
                return quant_compute_fail(
                    failure, YVEX_QUANT_FAILURE_NONFINITE, qtype, 0u,
                    block * geometry->block_size + lane, err, YVEX_ERR_FORMAT,
                    "qtype row dot requires finite encoded values and vector");
            sum += term;
        }
    }
    *out = (float)sum;
    if (!isfinite(*out))
        return quant_compute_fail(failure, YVEX_QUANT_FAILURE_NUMERIC_BOUND, qtype, 0u, elements,
                                  err, YVEX_ERR_BOUNDS,
                                  "qtype row dot result is not representable as finite F32");
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}
