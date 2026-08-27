/* Collect lightweight graph/runtime observations without making them dispatch authority. */
#include <yvex/internal/execution_observation.h>

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <yvex/internal/core.h>

static int execution_refuse(yvex_error *err, yvex_status status,
                            const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}

int yvex_execution_memory_facts_add(
    yvex_execution_memory_facts *facts, unsigned long long active_weight_bytes,
    unsigned long long state_bytes, unsigned long long activation_bytes,
    unsigned long long temporary_bytes, unsigned long long measured_operations,
    unsigned long long missing_operations, yvex_error *err)
{
    yvex_execution_memory_facts candidate;
    if (!facts || (!measured_operations && !missing_operations))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.memory-facts",
                                "one measured or missing operation count is required");
    candidate = *facts;
    if (!yvex_core_u64_add(candidate.active_weight_bytes, active_weight_bytes,
                           &candidate.active_weight_bytes) ||
        !yvex_core_u64_add(candidate.state_bytes, state_bytes, &candidate.state_bytes) ||
        !yvex_core_u64_add(candidate.activation_bytes, activation_bytes,
                           &candidate.activation_bytes) ||
        !yvex_core_u64_add(candidate.temporary_bytes, temporary_bytes,
                           &candidate.temporary_bytes) ||
        !yvex_core_u64_add(candidate.measured_operations, measured_operations,
                           &candidate.measured_operations) ||
        !yvex_core_u64_add(candidate.missing_operations, missing_operations,
                           &candidate.missing_operations))
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.memory-facts",
                                "compulsory memory counters overflowed");
    candidate.complete = candidate.measured_operations != 0ull &&
                         candidate.missing_operations == 0ull;
    *facts = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_memory_facts_merge(
    yvex_execution_memory_facts *facts,
    const yvex_execution_memory_facts *delta, yvex_error *err)
{
    if (!delta)
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.memory-facts",
                                "memory fact delta is required");
    if (!delta->measured_operations && !delta->missing_operations) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    return yvex_execution_memory_facts_add(
        facts, delta->active_weight_bytes, delta->state_bytes,
        delta->activation_bytes, delta->temporary_bytes,
        delta->measured_operations, delta->missing_operations, err);
}

int yvex_execution_physical_facts_add(
    yvex_execution_physical_facts *facts,
    const yvex_execution_memory_facts *memory, unsigned long long h2d_bytes,
    unsigned long long d2h_bytes, unsigned long long d2d_bytes,
    unsigned long long kernel_count, unsigned long long stream_synchronization_count,
    unsigned long long device_synchronization_count, yvex_error *err)
{
    yvex_execution_physical_facts candidate;
    unsigned long long synchronization_count;
    int rc;
    if (!facts || !memory)
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.physical-facts",
                                "physical fact owners are required");
    candidate = *facts;
    rc = yvex_execution_memory_facts_merge(&candidate.memory, memory, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_add(stream_synchronization_count, device_synchronization_count,
                           &synchronization_count) ||
        !yvex_core_u64_add(candidate.h2d_bytes, h2d_bytes, &candidate.h2d_bytes) ||
        !yvex_core_u64_add(candidate.d2h_bytes, d2h_bytes, &candidate.d2h_bytes) ||
        !yvex_core_u64_add(candidate.d2d_bytes, d2d_bytes, &candidate.d2d_bytes) ||
        !yvex_core_u64_add(candidate.kernel_count, kernel_count, &candidate.kernel_count) ||
        !yvex_core_u64_add(candidate.synchronization_count, synchronization_count,
                           &candidate.synchronization_count))
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.physical-facts",
                                "physical fact accounting overflowed");
    *facts = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_f32_hash_update(
    yvex_sha256 *hash, const float *values, unsigned long long count)
{
    unsigned long long index;
    if (!hash || !values || !count) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        if (!isfinite(values[index])) return 0;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(hash, bits)) return 0;
    }
    return 1;
}

int yvex_execution_f32_digest(
    const char *domain, const float *values, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!domain || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_execution_f32_hash_update(&hash, values, count) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
