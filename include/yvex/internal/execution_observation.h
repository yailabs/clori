/* Lightweight execution counters and deterministic numeric digests exposed to evidence owners. */
#ifndef INCLUDE_YVEX_INTERNAL_EXECUTION_OBSERVATION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_EXECUTION_OBSERVATION_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned long long active_weight_bytes, state_bytes, activation_bytes, temporary_bytes;
    unsigned long long measured_operations, missing_operations;
    int complete;
} yvex_execution_memory_facts;

int yvex_execution_memory_facts_add(
    yvex_execution_memory_facts *facts, unsigned long long active_weight_bytes,
    unsigned long long state_bytes, unsigned long long activation_bytes,
    unsigned long long temporary_bytes, unsigned long long measured_operations,
    unsigned long long missing_operations, yvex_error *err);
int yvex_execution_memory_facts_merge(
    yvex_execution_memory_facts *facts,
    const yvex_execution_memory_facts *delta, yvex_error *err);

typedef struct {
    yvex_execution_memory_facts memory;
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes;
    unsigned long long kernel_count, synchronization_count;
} yvex_execution_physical_facts;

int yvex_execution_physical_facts_add(
    yvex_execution_physical_facts *facts,
    const yvex_execution_memory_facts *memory, unsigned long long h2d_bytes,
    unsigned long long d2h_bytes, unsigned long long d2d_bytes,
    unsigned long long kernel_count, unsigned long long stream_synchronization_count,
    unsigned long long device_synchronization_count, yvex_error *err);
int yvex_execution_f32_hash_update(
    yvex_sha256 *hash, const float *values, unsigned long long count);
int yvex_execution_f32_digest(
    const char *domain, const float *values, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP]);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_EXECUTION_OBSERVATION_H_INCLUDED */
