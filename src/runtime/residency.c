/*
 * Retain exact encoded model and component weights across warm executions.
 *
 * Immutable ranges are read once, identity-bound, and shared by sessions without transferring
 * model-lifetime ownership.
 */
#include "src/runtime/private.h"

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <yvex/artifact.h>
#include <yvex/internal/core.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>
typedef enum {
    RESIDENCY_BINDING_CORE = YVEX_ATTENTION_BINDING_CORE,
    RESIDENCY_BINDING_ENVELOPE = YVEX_ATTENTION_BINDING_ENVELOPE,
    RESIDENCY_BINDING_OUTPUT_HEAD = 3,
    RESIDENCY_BINDING_MODEL = 4
} residency_binding_class;
typedef struct {
    const yvex_materialized_tensor_binding *binding;
    const yvex_physical_execution_decision *decision;
    const unsigned char *canonical_data;
    residency_binding_class binding_class;
    unsigned long long arena_offset, storage_bytes;
} residency_record;
struct yvex_runtime_residency {
    yvex_materialization_session *materialization;
    unsigned char *arena;
    residency_record *records;
    unsigned long long *record_index;
    unsigned long long record_index_count;
    yvex_backend *cuda_backend;
    yvex_device_tensor *cuda_weights;
    unsigned long long cuda_addressable_device_base, backing_bytes;
    yvex_runtime_residency_summary summary;
    pthread_mutex_t access_mutex;
    int access_mutex_ready, arena_locked, arena_mapped, arena_registered, arena_managed;
    int arena_borrowed;
};
int yvex_runtime_private_weight_placement_select(
    const yvex_runtime_binding *binding, yvex_backend_kind backend_kind,
    yvex_backend *backend,
    yvex_runtime_weight_placement *placement, yvex_error *err)
{
    yvex_backend_device_info device = {0};
    unsigned long long index;
    int mapped_compatible = 1;
    int rc;
    if (!binding || !binding->physical_execution || !placement ||
        (backend_kind == YVEX_BACKEND_KIND_CUDA && !backend)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.policy",
                       "compiled binding, backend facts, and placement output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (backend_kind == YVEX_BACKEND_KIND_CPU) {
        *placement = YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (backend_kind != YVEX_BACKEND_KIND_CUDA) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.residency.policy",
                       "the admitted backend has no production weight placement");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = yvex_backend_get_device_info(backend, &device, err);
    if (rc != YVEX_OK) return rc;
    for (index = 0ull; index < binding->summary.tensor_count; ++index) {
        const yvex_physical_execution_decision *decision =
            yvex_physical_execution_ir_decision_at(
                binding->physical_execution, binding->materialized[index].tensor_id);
        if (!decision) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.residency.policy",
                           "compiled placement decision is unavailable");
            return YVEX_ERR_FORMAT;
        }
        mapped_compatible &= !decision->derived_asset_required;
    }
    if (mapped_compatible && yvex_backend_resident_map_readonly_supported(backend)) {
        *placement = YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (device.kind != YVEX_BACKEND_KIND_CUDA || !device.unified_addressing ||
        !device.managed_memory) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.residency.policy",
                       "CUDA managed residency is not supported by the admitted backend");
        return YVEX_ERR_UNSUPPORTED;
    }

    /* Derived packs cannot borrow canonical artifact bytes; canonical bytes prefer an
     * immutable registered map only when the backend can publish one stable device address. */
    *placement = YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int residency_reject(yvex_runtime_residency_failure *failure,
                            yvex_runtime_residency_failure_code code,
                            const yvex_runtime_tensor_binding *binding,
                            unsigned long long expected, unsigned long long actual,
                            const char *reason, yvex_status status, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->tensor_id = binding ? binding->tensor_id : ULLONG_MAX;
        failure->layer_index = binding ? binding->layer_index : ULLONG_MAX;
        failure->role = binding ? binding->role : YVEX_TENSOR_ROLE_UNKNOWN;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
    }
    yvex_error_set(err, status, "runtime.residency", reason);
    return status;
}

static int residency_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    return residency_reject(NULL, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                            0ull, 0ull, reason, status, err);
}
/* Resolve one exact resident record without copying or exposing arena ownership. */
static int residency_resolve(const void *context,
                             const yvex_materialized_tensor_binding *binding,
                             const unsigned char **data, unsigned long long *bytes)
{
    yvex_runtime_residency *residency = (yvex_runtime_residency *)context;
    unsigned long long slot;
    const residency_record *record;
    int invalidated;
    if (data) *data = NULL;
    if (bytes) *bytes = 0ull;
    if (!residency || !binding || !data || !bytes ||
        binding->tensor_id >= residency->record_index_count)
        return YVEX_MATERIALIZATION_READ_MISS;
    if (!residency->access_mutex_ready ||
        pthread_mutex_lock(&residency->access_mutex) != 0)
        return YVEX_MATERIALIZATION_READ_INVALID;
    invalidated = residency->summary.invalidated;
    (void)pthread_mutex_unlock(&residency->access_mutex);
    if (invalidated) return YVEX_MATERIALIZATION_READ_INVALID;
    slot = residency->record_index[binding->tensor_id];
    if (!slot) return YVEX_MATERIALIZATION_READ_MISS;
    record = &residency->records[slot - 1ull];
    if (!record->binding || record->binding->tensor_id != binding->tensor_id ||
        record->binding->qtype != binding->qtype ||
        record->binding->encoded_bytes != binding->encoded_bytes ||
        record->binding->absolute_offset != binding->absolute_offset ||
        strcmp(record->binding->name, binding->name) != 0)
        return YVEX_MATERIALIZATION_READ_INVALID;
    *data = record->canonical_data
                ? record->canonical_data
                : residency->arena + record->arena_offset;
    *bytes = record->binding->encoded_bytes;
    return YVEX_MATERIALIZATION_READ_HIT;
}

static int residency_note_access(const void *context, unsigned long long bytes)
{
    yvex_runtime_residency *residency = (yvex_runtime_residency *)context;
    unsigned long long next_calls, next_bytes;
    if (!residency || !residency->access_mutex_ready ||
        pthread_mutex_lock(&residency->access_mutex) != 0)
        return 0;
    if (!yvex_core_u64_add(residency->summary.resident_read_calls, 1ull,
                           &next_calls) ||
        !yvex_core_u64_add(residency->summary.resident_bytes_read, bytes,
                           &next_bytes)) {
        (void)pthread_mutex_unlock(&residency->access_mutex);
        return 0;
    }
    residency->summary.resident_read_calls = next_calls;
    residency->summary.resident_bytes_read = next_bytes;
    (void)pthread_mutex_unlock(&residency->access_mutex);
    return 1;
}

static void residency_detached(const void *context)
{
    yvex_runtime_residency *residency = (yvex_runtime_residency *)context;
    if (!residency) return;
    residency->materialization = NULL;
    residency->summary.attached = 0;
}

static void residency_storage_release(yvex_runtime_residency **owner)
{
    yvex_runtime_residency *residency = owner ? *owner : NULL;
    if (!residency) return;
    if (residency->arena_locked)
        (void)munlock(residency->arena, (size_t)residency->backing_bytes);
    if (residency->arena_mapped && !residency->arena_managed)
        (void)munmap(residency->arena, (size_t)residency->backing_bytes);
    free(residency->records);
    free(residency->record_index);
    if (residency->access_mutex_ready)
        (void)pthread_mutex_destroy(&residency->access_mutex);
    memset(residency, 0, sizeof(*residency));
    free(residency);
    *owner = NULL;
}
/*
 * Release the model-owned CUDA residency without touching independent host storage.
 *
 * Exclusive residency ownership.
 */
static int residency_cuda_release(yvex_runtime_residency *residency, yvex_error *err)
{
    int managed;
    int rc = YVEX_OK;
    if (!residency) return YVEX_OK;
    managed = residency->arena_managed;
    if (residency->cuda_backend) {
        rc = yvex_backend_resident_detach(residency->cuda_backend, err);
        if (rc != YVEX_OK) return rc;
        rc = yvex_backend_close_admit(residency->cuda_backend, err);
        if (rc != YVEX_OK) return rc;
    }
    if (residency->cuda_weights) {
        rc = yvex_backend_tensor_release(
            residency->cuda_backend, &residency->cuda_weights, err);
        if (rc != YVEX_OK) return rc;
    }
    if (managed) {
        residency->arena = NULL;
        residency->arena_managed = 0;
    }
    rc = yvex_backend_close_checked(&residency->cuda_backend, err);
    if (rc != YVEX_OK) return rc;
    residency->summary.cuda_ready = 0;
    residency->summary.device_resident_bytes = 0ull;
    residency->summary.cuda_addressable_bytes = 0ull;
    residency->summary.cuda_upload_bytes = 0ull;
    residency->summary.cuda_upload_count = 0ull;
    residency->summary.cuda_host_registration_count = 0ull;
    residency->summary.cuda_pageable_map_bytes = 0ull;
    residency->summary.cuda_pageable_map_count = 0ull;
    residency->summary.cuda_managed_bytes = 0ull;
    residency->summary.cuda_managed_allocation_count = 0ull;
    residency->summary.cuda_managed_prefetch_bytes = 0ull;
    residency->summary.cuda_managed_prefetch_count = 0ull;
    residency->summary.cuda_pageable_prefetch_bytes = 0ull;
    residency->summary.cuda_pageable_prefetch_count = 0ull;
    residency->cuda_addressable_device_base = 0ull;
    residency->arena_registered = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int residency_add_record(yvex_runtime_residency *residency,
                                const yvex_runtime_tensor_binding *runtime,
                                residency_binding_class binding_class,
                                unsigned long long ordinal,
                                unsigned long long *core_bytes,
                                unsigned long long core_qtypes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP],
                                yvex_runtime_residency_failure *failure, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding = runtime ? runtime->binding : NULL;
    unsigned long long next;
    if (!binding || binding->tensor_id >= residency->record_index_count)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                residency->record_index_count,
                                binding ? binding->tensor_id : ULLONG_MAX,
                                "resident binding tensor identity is outside the descriptor",
                                YVEX_ERR_FORMAT, err);
    if (residency->record_index[binding->tensor_id])
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_DUPLICATE_BINDING,
                                runtime, 1ull, 2ull,
                                "resident binding was selected more than once",
                                YVEX_ERR_FORMAT, err);
    if (!binding->encoded_bytes || binding->encoded_bytes > (unsigned long long)SIZE_MAX ||
        binding->qtype >= YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                (unsigned long long)SIZE_MAX, binding->encoded_bytes,
                                "resident encoded range or qtype is outside platform bounds",
                                YVEX_ERR_BOUNDS, err);
    if (!yvex_core_u64_add(residency->summary.encoded_bytes, binding->encoded_bytes, &next))
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                ULLONG_MAX, binding->encoded_bytes,
                                "resident encoded byte accounting overflowed",
                                YVEX_ERR_BOUNDS, err);
    residency->records[ordinal].binding = binding;
    residency->records[ordinal].binding_class = binding_class;
    residency->records[ordinal].arena_offset = residency->summary.encoded_bytes;
    residency->record_index[binding->tensor_id] = ordinal + 1ull;
    residency->summary.encoded_bytes = next;
    residency->summary.qtype_binding_counts[binding->qtype]++;
    if (!yvex_core_u64_add(residency->summary.qtype_bytes[binding->qtype],
                           binding->encoded_bytes,
                           &residency->summary.qtype_bytes[binding->qtype]))
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                ULLONG_MAX, binding->encoded_bytes,
                                "resident qtype byte accounting overflowed",
                                YVEX_ERR_BOUNDS, err);
    if (binding_class == RESIDENCY_BINDING_CORE) {
        residency->summary.core_binding_count++;
        core_qtypes[binding->qtype]++;
        if (!yvex_core_u64_add(*core_bytes, binding->encoded_bytes, core_bytes))
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                    ULLONG_MAX, binding->encoded_bytes,
                                    "resident core byte accounting overflowed",
                                    YVEX_ERR_BOUNDS, err);
    } else if (binding_class == RESIDENCY_BINDING_ENVELOPE) {
        residency->summary.envelope_binding_count++;
    } else if (binding_class == RESIDENCY_BINDING_OUTPUT_HEAD) {
        residency->summary.output_head_binding_count++;
        if (!yvex_core_u64_add(residency->summary.output_head_encoded_bytes,
                               binding->encoded_bytes,
                               &residency->summary.output_head_encoded_bytes))
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                ULLONG_MAX, binding->encoded_bytes,
                "resident output-head byte accounting overflowed",
                YVEX_ERR_BOUNDS, err);
    } else {
        residency->summary.model_binding_count++;
    }
    residency->summary.binding_count++;
    return YVEX_OK;
}

/* Classify the sealed binding inventory before any backing allocation begins. */
static int residency_records_prepare(
    yvex_runtime_residency *residency, const yvex_runtime_descriptor *descriptor,
    const yvex_runtime_descriptor_summary *descriptor_summary,
    const yvex_runtime_model_summary *model_summary,
    const yvex_attention_plan *plan, const yvex_attention_summary *attention,
    yvex_runtime_residency_failure *failure, yvex_error *err)
{
    unsigned long long core_qtypes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP] = {0};
    unsigned long long core_bytes = 0ull;
    unsigned long long index, ordinal = 0ull;
    int rc = YVEX_OK;
    residency->summary.expected_core_binding_count = attention->required_binding_count;
    residency->summary.expected_envelope_binding_count =
        attention->required_envelope_binding_count;
    for (index = 0ull; rc == YVEX_OK && index < descriptor_summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *binding =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        yvex_attention_binding_class attention_class =
            yvex_attention_plan_binding_classify(plan, binding);
        residency_binding_class binding_class;
        int selected = 1;
        if (attention_class == YVEX_ATTENTION_BINDING_CORE)
            binding_class = RESIDENCY_BINDING_CORE;
        else if (attention_class == YVEX_ATTENTION_BINDING_ENVELOPE)
            binding_class = RESIDENCY_BINDING_ENVELOPE;
        else if (model_summary->capabilities.output_head_binding_ready && binding &&
                 binding->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD &&
                 binding->scope == YVEX_TENSOR_SCOPE_GLOBAL)
            binding_class = RESIDENCY_BINDING_OUTPUT_HEAD;
        else
            selected = 0;
        if (selected)
            rc = residency_add_record(residency, binding, binding_class, ordinal++,
                                      &core_bytes, core_qtypes, failure, err);
    }
    residency->summary.accelerator_encoded_bytes = residency->summary.encoded_bytes;
    for (index = 0ull; rc == YVEX_OK && index < descriptor_summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *binding =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        yvex_attention_binding_class attention_class =
            yvex_attention_plan_binding_classify(plan, binding);
        int accelerator_binding =
            attention_class == YVEX_ATTENTION_BINDING_CORE ||
            attention_class == YVEX_ATTENTION_BINDING_ENVELOPE ||
            (model_summary->capabilities.output_head_binding_ready && binding &&
             binding->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD &&
             binding->scope == YVEX_TENSOR_SCOPE_GLOBAL);
        if (!accelerator_binding)
            rc = residency_add_record(residency, binding, RESIDENCY_BINDING_MODEL,
                                      ordinal++, &core_bytes, core_qtypes, failure, err);
    }
    residency->summary.expected_output_head_binding_count =
        model_summary->capabilities.output_head_binding_ready ? 1ull : 0ull;
    residency->summary.expected_model_binding_count =
        descriptor_summary->tensor_count - residency->summary.expected_core_binding_count -
        residency->summary.expected_envelope_binding_count -
        residency->summary.expected_output_head_binding_count;
    if (rc == YVEX_OK &&
        (residency->summary.core_binding_count != attention->required_binding_count ||
         residency->summary.envelope_binding_count != attention->required_envelope_binding_count ||
         residency->summary.output_head_binding_count !=
             residency->summary.expected_output_head_binding_count ||
         residency->summary.model_binding_count !=
             residency->summary.expected_model_binding_count ||
         residency->summary.binding_count != descriptor_summary->tensor_count ||
         residency->summary.encoded_bytes != descriptor_summary->payload_bytes ||
         core_bytes != attention->payload_bytes_bound))
        rc = residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                              descriptor_summary->tensor_count,
                              residency->summary.binding_count,
                              "full-model resident accounting differs from the descriptor",
                              YVEX_ERR_FORMAT, err);
    for (index = 0ull; rc == YVEX_OK && index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++index)
        if (core_qtypes[index] != attention->qtype_binding_counts[index])
            rc = residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                                  attention->qtype_binding_counts[index], core_qtypes[index],
                                  "resident core qtype accounting differs from the plan",
                                  YVEX_ERR_FORMAT, err);
    return rc;
}

static int residency_layout_plan(
    yvex_runtime_residency *residency,
    const yvex_physical_execution_ir *physical,
    const yvex_artifact *artifact, yvex_backend *backend,
    yvex_runtime_weight_placement placement,
    yvex_runtime_residency_failure *failure, yvex_error *err)
{
    const yvex_backend_moe_operations *operations =
        placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED
            ? yvex_backend_moe_operations_get(backend) : NULL;
    const unsigned char *mapping = artifact ? yvex_artifact_data(artifact) : NULL;
    unsigned long long artifact_bytes = artifact ? yvex_artifact_size(artifact) : 0ull;
    unsigned long long offset = 0ull, derived_bytes = 0ull, index;
    int derived = 0;

    if (!residency || !physical)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                                1ull, 0ull, "physical execution layout is unavailable",
                                YVEX_ERR_STATE, err);
    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        residency_record *record = &residency->records[index];
        const yvex_materialized_tensor_binding *binding = record->binding;
        const yvex_physical_execution_decision *decision =
            yvex_physical_execution_ir_decision_at(physical, binding->tensor_id);
        unsigned long long storage = binding->encoded_bytes;
        if (!decision || decision->terminal_tensor_id != binding->tensor_id ||
            decision->canonical_qtype != binding->qtype ||
            decision->canonical_row_width != binding->row_width ||
            decision->canonical_row_count != binding->row_count ||
            decision->encoded_bytes != binding->encoded_bytes)
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                                    binding->tensor_id, decision ? decision->terminal_tensor_id
                                                                : ULLONG_MAX,
                                    "resident layout decision is stale",
                                    YVEX_ERR_FORMAT, err);
        record->decision = decision;
        if (decision->derived_asset_required &&
            placement != YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED)
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED, placement,
                "compiled backend-derived layout requires CUDA-managed residency",
                YVEX_ERR_UNSUPPORTED, err);
        if (placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED &&
            decision->derived_asset_required) {
            if (!operations || !operations->derived_layout_plan ||
                !operations->derived_layout_build || !mapping ||
                binding->absolute_offset > artifact_bytes ||
                binding->encoded_bytes > artifact_bytes - binding->absolute_offset)
                return residency_reject(
                    failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                    binding->encoded_bytes, 0ull,
                    "required backend-derived resident layout is unavailable",
                    YVEX_ERR_UNSUPPORTED, err);
            if (operations->derived_layout_plan(decision, &storage, err) != YVEX_OK)
                return residency_reject(
                    failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, NULL,
                    binding->encoded_bytes, storage,
                    "required backend-derived resident layout is invalid",
                    (yvex_status)yvex_error_code(err), err);
            record->canonical_data = mapping + binding->absolute_offset;
            if (!yvex_core_u64_add(derived_bytes, storage, &derived_bytes))
                return residency_reject(
                    failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, NULL,
                    ULLONG_MAX, storage,
                    "derived resident layout accounting overflowed",
                    YVEX_ERR_BOUNDS, err);
            residency->summary.derived_asset_count++;
            derived = 1;
        }
        record->storage_bytes = storage;
    }
    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        residency_record *record = &residency->records[index];
        if (derived && !yvex_core_u64_add(offset, 63ull, &offset))
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY,
                                    NULL, ULLONG_MAX, record->storage_bytes,
                                    "resident layout alignment overflowed",
                                    YVEX_ERR_BOUNDS, err);
        if (derived) offset &= ~63ull;
        record->arena_offset = offset;
        if (!yvex_core_u64_add(offset, record->storage_bytes, &offset))
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY,
                                    NULL, ULLONG_MAX, record->storage_bytes,
                                    "resident layout storage overflowed",
                                    YVEX_ERR_BOUNDS, err);
    }
    if (derived) {
        if (!yvex_core_u64_add(offset, 63ull, &offset))
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY,
                                    NULL, ULLONG_MAX, 64ull,
                                    "resident layout final alignment overflowed",
                                    YVEX_ERR_BOUNDS, err);
        offset &= ~63ull;
    }
    residency->summary.derived_asset_bytes = derived_bytes;
    residency->backing_bytes = offset;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_residency_backing_bytes(
    const yvex_runtime_binding *binding, yvex_backend *backend,
    yvex_runtime_weight_placement placement, unsigned long long *bytes,
    yvex_error *err)
{
    const yvex_backend_moe_operations *operations =
        placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED
            ? yvex_backend_moe_operations_get(backend) : NULL;
    unsigned long long index, total = 0ull;
    int derived = 0;
    if (bytes) *bytes = 0ull;
    if (!binding || !binding->physical_execution || !binding->materialized ||
        !binding->summary.tensor_count || !bytes)
        return residency_refuse(err, YVEX_ERR_INVALID_ARG,
                                "compiled residency backing facts are required");
    for (index = 0ull; index < binding->summary.tensor_count; ++index) {
        const yvex_materialized_tensor_binding *materialized =
            &binding->materialized[index];
        const yvex_physical_execution_decision *decision =
            yvex_physical_execution_ir_decision_at(
                binding->physical_execution, materialized->tensor_id);
        if (!decision || decision->terminal_tensor_id != materialized->tensor_id ||
            decision->encoded_bytes != materialized->encoded_bytes)
            return residency_refuse(err, YVEX_ERR_FORMAT,
                                    "compiled residency backing decision is stale");
        if (decision->derived_asset_required &&
            placement != YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED)
            return residency_refuse(
                err, YVEX_ERR_UNSUPPORTED,
                "compiled backend-derived layout requires CUDA-managed residency");
        derived |= placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED &&
                   decision->derived_asset_required;
    }
    for (index = 0ull; index < binding->summary.tensor_count; ++index) {
        const yvex_materialized_tensor_binding *materialized =
            &binding->materialized[index];
        const yvex_physical_execution_decision *decision =
            yvex_physical_execution_ir_decision_at(
                binding->physical_execution, materialized->tensor_id);
        unsigned long long storage = materialized->encoded_bytes;
        if (placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED &&
            decision->derived_asset_required &&
            (!operations || !operations->derived_layout_plan ||
             operations->derived_layout_plan(decision, &storage, err) != YVEX_OK))
            return residency_refuse(err, YVEX_ERR_UNSUPPORTED,
                                    "required derived residency layout is unavailable");
        if (derived && !yvex_core_u64_add(total, 63ull, &total))
            return residency_refuse(err, YVEX_ERR_BOUNDS,
                                    "compiled residency backing extent overflowed");
        if (derived) total &= ~63ull;
        if (!yvex_core_u64_add(total, storage, &total))
            return residency_refuse(err, YVEX_ERR_BOUNDS,
                                    "compiled residency backing extent overflowed");
    }
    if (derived) {
        if (!yvex_core_u64_add(total, 63ull, &total))
            return residency_refuse(err, YVEX_ERR_BOUNDS,
                                    "compiled residency final alignment overflowed");
        total &= ~63ull;
    }
    *bytes = total;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Populate one arena from the already authenticated immutable artifact snapshot. */
static int residency_load(yvex_runtime_residency *residency,
                          yvex_runtime_residency_failure *failure, yvex_error *err)
{
    const yvex_backend_moe_operations *operations =
        residency && residency->cuda_backend
            ? yvex_backend_moe_operations_get(residency->cuda_backend) : NULL;
    unsigned long long index;

    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        const residency_record *record = &residency->records[index];
        const yvex_materialized_tensor_binding *binding = record->binding;
        unsigned char *destination = residency->arena + record->arena_offset;
        int rc = record->canonical_data
                     ? operations->derived_layout_build(
                           record->decision, record->canonical_data,
                           binding->encoded_bytes, destination,
                           record->storage_bytes, err)
                     : yvex_materialization_session_read(
                           residency->materialization, binding, 0ull, destination,
                           (size_t)binding->encoded_bytes, NULL, err);
        if (rc != YVEX_OK)
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_READ, NULL,
                                    binding->encoded_bytes, 0ull,
                                    "resident encoded range cold read failed",
                                    (yvex_status)rc, err);
        if (!yvex_core_u64_add(residency->summary.cold_artifact_read_calls, 1ull,
                               &residency->summary.cold_artifact_read_calls) ||
            !yvex_core_u64_add(residency->summary.cold_artifact_bytes_read,
                               binding->encoded_bytes,
                               &residency->summary.cold_artifact_bytes_read))
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE, NULL,
                ULLONG_MAX, binding->encoded_bytes,
                "resident cold-read accounting overflowed", YVEX_ERR_BOUNDS, err);
    }
    return YVEX_OK;
}

/*
 * Bind resident content to the verified artifact and exact copied byte ranges.
 *
 * Rehashing the destination would repeat the complete artifact trust pass while protecting only
 * one instant of mutable memory. The artifact identity, stable snapshot reads, materialization
 * identity, and exact source/destination ranges instead seal the same immutable content without a
 * second model-sized SHA pass.
 */
static int residency_payload_digest_build(yvex_runtime_residency *residency,
                                          const char *artifact_identity,
                                          const char *materialization_identity,
                                          yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    int derived;

    if (!residency || !yvex_sha256_hex_valid(artifact_identity) ||
        !yvex_sha256_hex_valid(materialization_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.payload",
                       "verified artifact and materialization identities are required");
        return YVEX_ERR_INVALID_ARG;
    }
    derived = residency->summary.derived_asset_count != 0ull;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, derived ? "yvex.runtime.resident.payload.v6"
                           : "yvex.runtime.resident.payload.v5") ||
        !yvex_sha256_update_text(&hash, artifact_identity) ||
        !yvex_sha256_update_text(&hash, materialization_identity) ||
        !yvex_sha256_update_u64(&hash, residency->summary.binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.encoded_bytes) ||
        (derived &&
         (!yvex_sha256_update_u64(&hash, residency->summary.derived_asset_count) ||
          !yvex_sha256_update_u64(&hash, residency->summary.derived_asset_bytes) ||
          !yvex_sha256_update_u64(&hash, residency->backing_bytes))))
        goto failed;
    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        const residency_record *record = &residency->records[index];
        const yvex_materialized_tensor_binding *binding = record->binding;

        if (!yvex_sha256_update_u64(&hash, binding->tensor_id) ||
            !yvex_sha256_update_u64(&hash, binding->qtype) ||
            !yvex_sha256_update_u64(&hash, binding->absolute_offset) ||
            !yvex_sha256_update_u64(&hash, binding->encoded_bytes) ||
            !yvex_sha256_update_u64(&hash, record->arena_offset) ||
            (derived &&
             (!record->decision ||
              !yvex_sha256_update_u64(&hash, record->storage_bytes) ||
              !yvex_sha256_update_u64(&hash, record->decision->layout) ||
              !yvex_sha256_update_text(&hash, record->decision->decision_identity))))
            goto failed;
    }
    if (!yvex_sha256_final(&hash, digest)) goto failed;
    yvex_sha256_hex(digest, residency->summary.payload_digest);
    return YVEX_OK;

failed:
    yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.payload",
                   "resident payload derivation identity failed");
    return YVEX_ERR_STATE;
}
/*
 * Derive residency identity from semantic range order and exact encoded payload.
 *
 * Writes one canonical content identity.
 */
static int residency_identity_build(yvex_runtime_residency *residency,
                                    const yvex_runtime_model_summary *model,
                                    const yvex_attention_summary *attention,
                                    yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    int derived = residency->summary.derived_asset_count != 0ull;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, derived ? "yvex.runtime.residency.v8"
                           : "yvex.runtime.residency.v7") ||
        !yvex_sha256_update_u64(
            &hash, derived ? YVEX_RUNTIME_RESIDENCY_SCHEMA_V8
                           : YVEX_RUNTIME_RESIDENCY_SCHEMA_V7) ||
        !yvex_sha256_update_text(&hash, model->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, model->artifact_identity) ||
        !yvex_sha256_update_text(&hash, model->materialization_identity) ||
        (derived &&
         !yvex_sha256_update_text(&hash, model->physical_execution_identity)) ||
        !yvex_sha256_update_text(&hash, attention->attention_plan_identity) ||
        !yvex_sha256_update_u64(&hash, residency->summary.model_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.core_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.envelope_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.output_head_binding_count) ||
        !yvex_sha256_update_u64(
            &hash, residency->summary.accelerator_encoded_bytes) ||
        !yvex_sha256_update_u64(&hash, residency->summary.encoded_bytes) ||
        (derived &&
         (!yvex_sha256_update_u64(&hash, residency->summary.derived_asset_count) ||
          !yvex_sha256_update_u64(&hash, residency->summary.derived_asset_bytes) ||
          !yvex_sha256_update_u64(&hash, residency->backing_bytes))) ||
        !yvex_sha256_update_u64(&hash, residency->summary.placement))
        goto failed;
    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        const residency_record *record = &residency->records[index];
        if (!yvex_sha256_update_u64(&hash, record->binding->tensor_id) ||
            !yvex_sha256_update_u64(&hash, record->binding_class) ||
            !yvex_sha256_update_u64(&hash, record->binding->qtype) ||
            !yvex_sha256_update_u64(&hash, record->binding->encoded_bytes) ||
            !yvex_sha256_update_u64(&hash, record->arena_offset) ||
            (derived &&
             (!record->decision ||
              !yvex_sha256_update_u64(&hash, record->storage_bytes) ||
              !yvex_sha256_update_u64(&hash, record->decision->layout) ||
              !yvex_sha256_update_text(&hash, record->decision->decision_identity))))
            goto failed;
    }
    if (!yvex_sha256_update_text(&hash, residency->summary.payload_digest) ||
        !yvex_sha256_final(&hash, digest))
        goto failed;
    yvex_sha256_hex(digest, residency->summary.residency_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.output-head.residency.v3") ||
        !yvex_sha256_update_text(&hash, model->runtime_model_identity) ||
        !yvex_sha256_update_u64(&hash, residency->summary.output_head_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.output_head_encoded_bytes) ||
        !yvex_sha256_update_text(&hash, residency->summary.payload_digest) ||
        !yvex_sha256_final(&hash, digest))
        goto failed;
    yvex_sha256_hex(digest, residency->summary.output_head_residency_identity);
    return YVEX_OK;
failed:
    yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.identity",
                   "resident identity encoding failed");
    return YVEX_ERR_STATE;
}

/* Bind one staged component identity to its exact resident payload and range order. */
static int residency_component_identity_build(
    yvex_runtime_residency *residency,
    const yvex_materialization_summary *materialization,
    const char *component_identity, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.component-residency.v2") ||
        !yvex_sha256_update_u64(&hash, 5ull) ||
        !yvex_sha256_update_text(&hash, component_identity) ||
        !yvex_sha256_update_text(&hash, materialization->artifact_identity) ||
        !yvex_sha256_update_text(&hash, materialization->plan_identity) ||
        !yvex_sha256_update_u64(&hash, residency->summary.binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.encoded_bytes))
        goto failed;
    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        const residency_record *record = &residency->records[index];
        const yvex_materialized_tensor_binding *binding = record->binding;

        if (!yvex_sha256_update_text(&hash, binding->name) ||
            !yvex_sha256_update_u64(&hash, binding->tensor_id) ||
            !yvex_sha256_update_u64(&hash, binding->qtype) ||
            !yvex_sha256_update_u64(&hash, binding->encoded_bytes) ||
            !yvex_sha256_update_u64(&hash, record->arena_offset))
            goto failed;
    }
    if (!yvex_sha256_update_text(&hash, residency->summary.payload_digest) ||
        !yvex_sha256_final(&hash, digest))
        goto failed;
    yvex_sha256_hex(digest, residency->summary.residency_identity);
    return YVEX_OK;
failed:
    yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-residency.identity",
                   "component residency identity construction failed");
    return YVEX_ERR_STATE;
}

static int residency_release(yvex_runtime_residency **owner, yvex_error *err)
{
    yvex_runtime_residency *residency = owner ? *owner : NULL;
    int rc;
    if (!residency) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (residency->summary.attached && residency->materialization)
        if ((rc = yvex_materialization_session_detach_read_provider(
                 residency->materialization, residency, NULL, err)) != YVEX_OK)
            return rc;
    rc = residency_cuda_release(residency, err);
    if (rc != YVEX_OK) return rc;
    residency_storage_release(owner);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int residency_arena_prepare(yvex_runtime_residency *residency,
                                   yvex_backend **prepared_backend,
                                   const yvex_artifact *artifact,
                                   const yvex_runtime_residency_options *options,
                                   yvex_runtime_residency_failure *failure,
                                   yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = prepared_backend ? *prepared_backend : NULL;
    int rc = YVEX_OK;
    if (!residency->backing_bytes)
        residency->backing_bytes = residency->summary.encoded_bytes;
    if (residency->backing_bytes > (unsigned long long)SIZE_MAX)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET, NULL,
                                (unsigned long long)SIZE_MAX,
                                residency->backing_bytes,
                                "resident arena exceeds platform allocation range",
                                YVEX_ERR_BOUNDS, err);
    if (options && options->maximum_host_bytes &&
        residency->backing_bytes > options->maximum_host_bytes)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET, NULL,
                                options->maximum_host_bytes,
                                residency->backing_bytes,
                                "resident arena exceeds the configured host budget",
                                YVEX_ERR_NOMEM, err);
    if (options && options->placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED) {
        unsigned long long index, artifact_bytes = yvex_artifact_size(artifact);
        const unsigned char *mapping = yvex_artifact_data(artifact);
        if (!artifact || !mapping || !yvex_artifact_is_mapped(artifact) ||
            !artifact_bytes || artifact_bytes > (unsigned long long)SIZE_MAX)
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
                1ull, 0ull, "verified artifact mapping is unavailable",
                YVEX_ERR_UNSUPPORTED, err);
        for (index = 0ull; index < residency->summary.binding_count; ++index) {
            residency_record *record = &residency->records[index];
            unsigned long long offset = record->binding->absolute_offset;
            if (offset > artifact_bytes ||
                record->binding->encoded_bytes > artifact_bytes - offset)
                return residency_reject(
                    failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, NULL,
                    artifact_bytes, offset, "artifact-backed resident range exceeds the snapshot",
                    YVEX_ERR_BOUNDS, err);
            record->arena_offset = offset;
        }
        residency->arena = (unsigned char *)(uintptr_t)mapping;
        residency->backing_bytes = artifact_bytes;
        residency->arena_borrowed = 1;
        residency->summary.placement = YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED;
        residency->summary.artifact_backed_bytes = artifact_bytes;
        return YVEX_OK;
    }
    if (options && options->placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED) {
        if (!backend || yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA)
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
                YVEX_BACKEND_KIND_CUDA, backend ? yvex_backend_kind_of(backend) : 0ull,
                "CUDA managed residency requires one prepared CUDA backend",
                YVEX_ERR_UNSUPPORTED, err);
        descriptor.name = "runtime-managed-residency";
        descriptor.dtype = YVEX_DTYPE_I8;
        descriptor.rank = 1u;
        descriptor.dims[0] = descriptor.bytes = residency->backing_bytes;
        rc = yvex_backend_resident_alloc(
            backend, &descriptor, &residency->cuda_weights, &residency->arena, err);
        if (rc != YVEX_OK)
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
                residency->backing_bytes, 0ull,
                "CUDA managed resident arena allocation failed", (yvex_status)rc, err);
        residency->cuda_backend = backend;
        residency->arena_managed = 1;
        residency->summary.placement = YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED;
        residency->summary.cuda_managed_bytes = residency->backing_bytes;
        residency->summary.cuda_managed_allocation_count = 1ull;
        *prepared_backend = NULL;
    } else {
        residency->arena = (unsigned char *)mmap(
            NULL, (size_t)residency->backing_bytes,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (residency->arena == MAP_FAILED) {
            residency->arena = NULL;
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
                                    residency->backing_bytes, 0ull,
                                    "resident encoded arena allocation failed",
                                    YVEX_ERR_NOMEM, err);
        }
        residency->arena_mapped = 1;
        residency->summary.placement = YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED;
    }
    rc = residency_load(residency, failure, err);
    if (rc != YVEX_OK) return rc;
    if (residency->arena_managed) return YVEX_OK;
    if (mlock(residency->arena, (size_t)residency->backing_bytes) != 0)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET, NULL,
                                residency->backing_bytes, 0ull,
                                "complete resident arena could not be locked in host RAM",
                                YVEX_ERR_NOMEM, err);
    residency->arena_locked = 1;
    residency->summary.host_locked = 1;
    return YVEX_OK;
}
/*
 * Register the verified locked host arena as one immutable CUDA-addressable range.
 *
 * Registers bytes without copying or identity change.
 */
static int residency_register_cuda(yvex_runtime_residency *residency,
                                   yvex_backend *backend,
                                   yvex_device_tensor **tensor,
                                   yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned char *registered;
    int rc;
    if (!residency || !backend || !tensor || !residency->arena_mapped ||
        !residency->arena_locked || residency->arena_registered ||
        residency->summary.resident_read_calls) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.register",
                       "unconsumed first-use host residency is required");
        return YVEX_ERR_STATE;
    }
    descriptor.name = "runtime-registered-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = residency->backing_bytes;
    registered = residency->arena;
    rc = yvex_backend_resident_alloc(
        backend, &descriptor, tensor, &registered, err);
    if (rc != YVEX_OK) return rc;
    if (registered != residency->arena) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.register",
                       "CUDA registration replaced the verified host address");
        return YVEX_ERR_STATE;
    }
    residency->arena_registered = 1;
    return YVEX_OK;
}

/* Transfer one pre-opened model context only after its locked arena is addressable. */
static int residency_claim_cuda(yvex_runtime_residency *residency,
                                yvex_backend **prepared_backend,
                                yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = residency && residency->arena_managed
                                ? residency->cuda_backend
                                : (prepared_backend ? *prepared_backend : NULL);
    yvex_device_tensor *weights = residency && residency->arena_managed
                                      ? residency->cuda_weights : NULL;
    unsigned long long address = 0ull, prefetched = 0ull;
    int rc;

    if (!residency || !backend || yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.cuda-claim",
                       "one pre-opened CUDA model backend is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (residency->arena_borrowed) {
        descriptor.name = "runtime-artifact-residency";
        descriptor.dtype = YVEX_DTYPE_I8;
        descriptor.rank = 1u;
        descriptor.dims[0] = descriptor.bytes = residency->backing_bytes;
        rc = yvex_backend_resident_map_readonly(
            backend, &descriptor, residency->arena, &weights, err);
        if (rc == YVEX_OK) {
            residency->cuda_backend = backend;
            residency->cuda_weights = weights;
            *prepared_backend = NULL;
        }
    } else {
        rc = residency->arena_managed
                 ? yvex_backend_resident_prefetch(backend, weights, &prefetched, err)
                 : residency_register_cuda(residency, backend, &weights, err);
    }
    if (rc == YVEX_OK && !residency->arena_managed && !residency->arena_borrowed) {
        residency->cuda_backend = backend;
        residency->cuda_weights = weights;
        *prepared_backend = NULL;
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_resident_attach(
            backend, residency->arena, residency->backing_bytes,
            weights, residency->summary.generation, err);
    if (rc == YVEX_OK &&
        yvex_backend_resident_resolve(
            backend, residency->arena, residency->backing_bytes,
            &address) != YVEX_BACKEND_RESIDENT_HIT) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "runtime.residency.cuda-claim",
                       "pre-opened CUDA model residency did not resolve");
    }
    if (rc != YVEX_OK) return rc;
    residency->cuda_addressable_device_base = address;
    residency->summary.cuda_addressable_bytes = residency->backing_bytes;
    if (residency->arena_managed) {
        residency->summary.host_resident_bytes = 0ull;
        residency->summary.device_resident_bytes = residency->backing_bytes;
        residency->summary.cuda_managed_prefetch_bytes = prefetched;
        residency->summary.cuda_managed_prefetch_count = 1ull;
    } else if (residency->arena_borrowed) {
        residency->summary.host_resident_bytes = 0ull;
        residency->summary.device_resident_bytes = 0ull;
        residency->summary.cuda_pageable_map_bytes = residency->backing_bytes;
        residency->summary.cuda_pageable_map_count = 1ull;
        residency->summary.cuda_host_registration_count = 1ull;
    } else {
        residency->summary.host_resident_bytes = residency->backing_bytes;
        residency->summary.device_resident_bytes = 0ull;
        residency->summary.cuda_host_registration_count = 1ull;
    }
    residency->summary.cuda_ready = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Build and attach one exact process-lifetime full-model residency pack. */
int yvex_runtime_residency_prepare(yvex_runtime_residency **out, yvex_runtime_model *model,
                                   const yvex_runtime_residency_options *options,
                                   yvex_runtime_residency_failure *failure, yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    yvex_runtime_model_summary model_summary;
    const yvex_runtime_descriptor *descriptor = view ? view->descriptor : NULL;
    const yvex_runtime_descriptor_summary *descriptor_summary =
        yvex_runtime_descriptor_summary_get(descriptor);
    const yvex_attention_plan *plan = view ? view->attention : NULL;
    const yvex_attention_summary *attention = yvex_attention_plan_summary(plan);
    yvex_materialization_session *materialization = view ? view->materialization : NULL;
    yvex_runtime_residency *residency = NULL;
    yvex_materialization_read_provider provider;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!out)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_INVALID_ARGUMENT,
                                NULL, 1ull, 0ull, "residency output is required",
                                YVEX_ERR_INVALID_ARG, err);
    rc = yvex_runtime_model_summary_copy(model, &model_summary, err);
    if (rc != YVEX_OK) {
        if (failure) {
            failure->code = YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE;
            failure->reason = "runtime model snapshot failed during residency preparation";
        }
        return rc;
    }
    if (!model_summary.sealed || !model_summary.valid ||
        !descriptor_summary || !plan || !attention || !materialization)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_MODEL,
                                NULL, 1ull, 0ull,
                                "sealed runtime model facts are required for residency",
                                YVEX_ERR_STATE, err);
    residency = (yvex_runtime_residency *)calloc(1u, sizeof(*residency));
    if (!residency)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION,
                                NULL, sizeof(*residency), 0ull,
                                "resident pack allocation failed", YVEX_ERR_NOMEM, err);
    if (pthread_mutex_init(&residency->access_mutex, NULL) != 0) {
        free(residency);
        return residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL, 1ull, 0ull,
            "resident access synchronization initialization failed", YVEX_ERR_STATE, err);
    }
    residency->access_mutex_ready = 1;
    residency->materialization = materialization;
    residency->record_index_count = descriptor_summary->tensor_count;
    residency->records = (residency_record *)calloc(
        (size_t)descriptor_summary->tensor_count, sizeof(*residency->records));
    residency->record_index = (unsigned long long *)calloc(
        (size_t)descriptor_summary->tensor_count, sizeof(*residency->record_index));
    if (!residency->records || !residency->record_index) {
        residency_storage_release(&residency);
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
                                descriptor_summary->tensor_count, 0ull,
                                "resident record index allocation failed",
                                YVEX_ERR_NOMEM, err);
    }
    rc = residency_records_prepare(
        residency, descriptor, descriptor_summary, &model_summary,
        plan, attention, failure, err);
    if (rc == YVEX_OK)
        rc = residency_layout_plan(
            residency, view->physical_execution, model->artifact,
            model->opening_backend,
            options ? options->placement : YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED,
            failure, err);
    if (rc == YVEX_OK) {
        residency->summary.model_complete =
            residency->summary.binding_count == descriptor_summary->tensor_count &&
            residency->summary.model_binding_count ==
                residency->summary.expected_model_binding_count;
        residency->summary.core_complete =
            residency->summary.core_binding_count ==
            residency->summary.expected_core_binding_count;
        residency->summary.envelope_complete =
            residency->summary.envelope_binding_count ==
            residency->summary.expected_envelope_binding_count;
        residency->summary.output_head_complete =
            residency->summary.output_head_binding_count ==
            residency->summary.expected_output_head_binding_count;
    }
    if (rc == YVEX_OK)
        rc = residency_arena_prepare(
            residency, &model->opening_backend, model->artifact, options, failure, err);
    if (rc == YVEX_OK)
        rc = residency_payload_digest_build(
            residency, model_summary.artifact_identity,
            model_summary.materialization_identity, err);
    if (rc == YVEX_OK)
        rc = residency_identity_build(residency, &model_summary, attention, err);
    if (rc == YVEX_OK) residency->summary.generation = 1ull;
    if (rc == YVEX_OK && (model->opening_backend || residency->cuda_backend))
        rc = residency_claim_cuda(residency, &model->opening_backend, err);
    if (rc == YVEX_OK) {
        provider.context = residency;
        provider.resolve = residency_resolve;
        provider.note_access = residency_note_access;
        provider.detached = residency_detached;
        rc = yvex_materialization_session_attach_read_provider(
            materialization, &provider, NULL, err);
        if (rc != YVEX_OK)
            rc = residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ATTACH, NULL,
                                  1ull, 0ull, "resident read provider attachment failed",
                                  (yvex_status)rc, err);
    }
    if (rc != YVEX_OK) {
        yvex_error primary = err ? *err : (yvex_error){0};
        yvex_error cleanup;
        int cleanup_rc;

        yvex_error_clear(&cleanup);
        cleanup_rc = residency_release(&residency, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            if (err) *err = cleanup;
            return cleanup_rc;
        }
        if (err) *err = primary;
        return rc;
    }
    residency->summary.schema_version = residency->summary.derived_asset_count
                                            ? YVEX_RUNTIME_RESIDENCY_SCHEMA_V8
                                            : YVEX_RUNTIME_RESIDENCY_SCHEMA_V7;
    residency->summary.generation = 1ull;
    if (!residency->arena_managed && !residency->arena_borrowed)
        residency->summary.host_resident_bytes = residency->backing_bytes;
    residency->summary.sealed = 1;
    residency->summary.attached = 1;
    residency->summary.host_ready = 1;
    *out = residency;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Build one exact staged component pack without inventing a full-model runtime descriptor. */
int yvex_runtime_component_residency_prepare(
    yvex_runtime_residency **out, yvex_materialization_session *materialization,
    const char *component_identity, const yvex_runtime_residency_options *options,
    yvex_runtime_residency_failure *failure, yvex_error *err)
{
    const yvex_materialization_summary *source =
        yvex_materialization_session_summary(materialization);
    yvex_runtime_residency *residency = NULL;
    yvex_materialization_read_provider provider;
    unsigned long long qtypes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP] = {0};
    unsigned long long ignored_core_bytes = 0ull;
    unsigned long long index;
    int rc = YVEX_OK;

    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!out || !materialization || !yvex_sha256_hex_valid(component_identity))
        return residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_INVALID_ARGUMENT, NULL,
            1ull, 0ull, "component session, identity, and output are required",
            YVEX_ERR_INVALID_ARG, err);
    if (!source || !source->committed || !source->tensor_count ||
        source->tensor_count != source->committed_bindings || !source->payload_bytes)
        return residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_MODEL, NULL,
            source ? source->tensor_count : 1ull,
            source ? source->committed_bindings : 0ull,
            "committed complete component materialization is required",
            YVEX_ERR_STATE, err);
    residency = (yvex_runtime_residency *)calloc(1u, sizeof(*residency));
    if (!residency)
        return residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
            sizeof(*residency), 0ull, "component resident pack allocation failed",
            YVEX_ERR_NOMEM, err);
    if (pthread_mutex_init(&residency->access_mutex, NULL) != 0) {
        free(residency);
        return residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL, 1ull, 0ull,
            "component residency synchronization initialization failed",
            YVEX_ERR_STATE, err);
    }
    residency->access_mutex_ready = 1;
    residency->materialization = materialization;
    residency->record_index_count = source->tensor_count;
    residency->records = (residency_record *)calloc(
        (size_t)source->tensor_count, sizeof(*residency->records));
    residency->record_index = (unsigned long long *)calloc(
        (size_t)source->tensor_count, sizeof(*residency->record_index));
    if (!residency->records || !residency->record_index) {
        residency_storage_release(&residency);
        return residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
            source->tensor_count, 0ull, "component resident index allocation failed",
            YVEX_ERR_NOMEM, err);
    }
    for (index = 0ull; rc == YVEX_OK && index < source->tensor_count; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(materialization, index);
        yvex_runtime_tensor_binding runtime = {0};

        if (!binding) {
            rc = residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_MISSING_BINDING, NULL,
                source->tensor_count, index, "component binding inventory is incomplete",
                YVEX_ERR_FORMAT, err);
            break;
        }
        runtime.binding = binding;
        runtime.role = binding->role;
        runtime.scope = binding->scope;
        runtime.layer_index = binding->layer_index;
        rc = residency_add_record(
            residency, &runtime, RESIDENCY_BINDING_MODEL, index,
            &ignored_core_bytes, qtypes, failure, err);
    }
    residency->summary.expected_model_binding_count = source->tensor_count;
    residency->summary.accelerator_encoded_bytes = residency->summary.encoded_bytes;
    for (index = 0ull; rc == YVEX_OK && index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++index)
        if (residency->summary.qtype_binding_counts[index] !=
                source->qtype_tensor_counts[index] ||
            residency->summary.qtype_bytes[index] != source->qtype_bytes[index])
            rc = residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                source->qtype_tensor_counts[index],
                residency->summary.qtype_binding_counts[index],
                "component resident qtype accounting differs from materialization",
                YVEX_ERR_FORMAT, err);
    if (rc == YVEX_OK &&
        (residency->summary.binding_count != source->tensor_count ||
         residency->summary.model_binding_count != source->tensor_count ||
         residency->summary.encoded_bytes != source->payload_bytes))
        rc = residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
            source->payload_bytes, residency->summary.encoded_bytes,
            "component resident accounting differs from materialization",
            YVEX_ERR_FORMAT, err);
    if (rc == YVEX_OK) {
        residency->summary.model_complete = 1;
        residency->summary.core_complete = 1;
        residency->summary.envelope_complete = 1;
        residency->summary.output_head_complete = 1;
        rc = residency_arena_prepare(residency, NULL, NULL, options, failure, err);
    }
    if (rc == YVEX_OK)
        rc = residency_payload_digest_build(
            residency, source->artifact_identity, source->plan_identity, err);
    if (rc == YVEX_OK)
        rc = residency_component_identity_build(residency, source, component_identity, err);
    if (rc == YVEX_OK) {
        residency->summary.generation = 1ull;
        provider.context = residency;
        provider.resolve = residency_resolve;
        provider.note_access = residency_note_access;
        provider.detached = residency_detached;
        rc = yvex_materialization_session_attach_read_provider(
            materialization, &provider, NULL, err);
        if (rc != YVEX_OK)
            rc = residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ATTACH, NULL, 1ull, 0ull,
                "component resident read provider attachment failed",
                (yvex_status)rc, err);
    }
    if (rc != YVEX_OK) {
        yvex_error primary = err ? *err : (yvex_error){0};
        yvex_error cleanup;
        int cleanup_rc = residency_release(&residency, &cleanup);

        if (cleanup_rc != YVEX_OK) {
            if (err) *err = cleanup;
            return cleanup_rc;
        }
        if (err) *err = primary;
        return rc;
    }
    residency->summary.schema_version = YVEX_RUNTIME_RESIDENCY_SCHEMA_V7;
    residency->summary.host_resident_bytes = residency->summary.encoded_bytes;
    residency->summary.sealed = 1;
    residency->summary.attached = 1;
    residency->summary.host_ready = 1;
    *out = residency;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Register one model-owned host arena and attach an isolated session backend.
 *
 * Session graph/workspace state stays private; model release owns resident bytes/context.
 */
int yvex_runtime_residency_cuda_session_attach(
    yvex_runtime_residency *residency, yvex_backend **backend,
    unsigned long long maximum_device_bytes, int *uploaded,
    yvex_runtime_residency_summary *summary, yvex_error *err)
{
    yvex_backend_options options;
    yvex_backend *candidate_backend, *session_backend = NULL;
    yvex_device_tensor *candidate_weights = NULL;
    yvex_backend_bandwidth_evidence bandwidth = {0};
    unsigned long long candidate_address = 0ull;
    yvex_error primary, cleanup;
    int rc, cleanup_rc;
    if (uploaded) *uploaded = 0;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!residency || !backend || !uploaded || !summary) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.cuda",
                       "residency and complete CUDA attachment outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    candidate_backend = *backend;
    if (candidate_backend &&
        (yvex_backend_kind_of(candidate_backend) != YVEX_BACKEND_KIND_CUDA ||
         residency->summary.cuda_ready || residency->cuda_backend || residency->cuda_weights)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.cuda",
                       "a fresh residency accepts at most one prepared CUDA backend");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!residency->access_mutex_ready) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.cuda",
                       "resident access synchronization is not initialized");
        return YVEX_ERR_STATE;
    }
    if (pthread_mutex_lock(&residency->access_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.cuda",
                       "resident access synchronization could not be acquired");
        return YVEX_ERR_STATE;
    }
    if (!residency->summary.sealed || !residency->summary.host_ready ||
        (!residency->summary.host_locked && !residency->summary.artifact_backed_bytes &&
         !(residency->summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED &&
           residency->summary.cuda_managed_allocation_count == 1ull &&
           residency->summary.cuda_managed_bytes ==
               residency->summary.cuda_addressable_bytes)) ||
        residency->summary.invalidated || !residency->arena ||
        !residency->summary.accelerator_encoded_bytes) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "runtime.residency.cuda",
                       "sealed valid host residency is required");
        goto done;
    }
    *backend = NULL;
    if ((residency->cuda_backend || residency->cuda_weights) &&
        !residency->summary.cuda_ready) {
        rc = residency_cuda_release(residency, err);
        if (rc != YVEX_OK) goto done;
    }
    if (!residency->summary.cuda_ready) {
        if (residency->summary.placement != YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED &&
            residency->summary.placement != YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(err, rc, "runtime.residency.cuda",
                           "managed model residency lost its admitted CUDA owner");
            goto done;
        }
        memset(&options, 0, sizeof(options));
        options.kind = YVEX_BACKEND_KIND_CUDA;
        options.memory_limit_bytes = maximum_device_bytes;
        rc = candidate_backend ? YVEX_OK : yvex_backend_open(&candidate_backend, &options, err);
        if (rc == YVEX_OK) {
            if (residency->summary.placement ==
                YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED) {
                yvex_backend_tensor_desc descriptor = {0};
                descriptor.name = "runtime-artifact-residency";
                descriptor.dtype = YVEX_DTYPE_I8;
                descriptor.rank = 1u;
                descriptor.dims[0] = descriptor.bytes = residency->backing_bytes;
                rc = yvex_backend_resident_map_readonly(
                    candidate_backend, &descriptor, residency->arena,
                    &candidate_weights, err);
            } else {
                rc = residency_register_cuda(
                    residency, candidate_backend, &candidate_weights, err);
            }
        }
        if (rc == YVEX_OK)
            rc = yvex_backend_resident_attach(
                candidate_backend, residency->arena, residency->backing_bytes,
                candidate_weights, residency->summary.generation, err);
        if (rc == YVEX_OK &&
            yvex_backend_resident_resolve(
                candidate_backend, residency->arena, residency->backing_bytes,
                &candidate_address) != YVEX_BACKEND_RESIDENT_HIT) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(err, rc, "runtime.residency.cuda",
                           "registered model residency did not resolve");
        }
        if (rc != YVEX_OK) {
            primary = err ? *err : (yvex_error){0};
            yvex_error_clear(&cleanup);
            cleanup_rc = yvex_backend_resident_detach(candidate_backend, &cleanup);
            if (cleanup_rc == YVEX_OK)
                cleanup_rc = candidate_weights
                             ? yvex_backend_tensor_release(
                                   candidate_backend, &candidate_weights, &cleanup)
                             : YVEX_OK;
            if (cleanup_rc == YVEX_OK) residency->arena_registered = 0;
            if (cleanup_rc == YVEX_OK)
                cleanup_rc = yvex_backend_close_checked(&candidate_backend, &cleanup);
            if (cleanup_rc != YVEX_OK) {
                residency->cuda_backend = candidate_backend;
                residency->cuda_weights = candidate_weights;
                rc = cleanup_rc;
                if (err) *err = cleanup;
            } else if (err) *err = primary;
            goto done;
        }
        residency->cuda_backend = candidate_backend;
        residency->cuda_weights = candidate_weights;
        residency->cuda_addressable_device_base = candidate_address;
        residency->summary.host_resident_bytes =
            residency->summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED
                ? 0ull : residency->summary.encoded_bytes;
        residency->summary.device_resident_bytes = 0ull;
        residency->summary.cuda_addressable_bytes = residency->backing_bytes;
        residency->summary.cuda_upload_bytes = 0ull;
        residency->summary.cuda_upload_count = 0ull;
        residency->summary.cuda_host_registration_count =
            residency->summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED ||
            residency->summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED;
        residency->summary.cuda_pageable_map_bytes =
            residency->summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED
                ? residency->backing_bytes : 0ull;
        residency->summary.cuda_pageable_map_count =
            residency->summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED;
        residency->summary.cuda_managed_bytes = 0ull;
        residency->summary.cuda_managed_allocation_count = 0ull;
        residency->summary.cuda_managed_prefetch_bytes = 0ull;
        residency->summary.cuda_managed_prefetch_count = 0ull;
        residency->summary.cuda_pageable_prefetch_bytes = 0ull;
        residency->summary.cuda_pageable_prefetch_count = 0ull;
        residency->summary.cuda_ready = 1;
        *uploaded = 1;
    }
    rc = yvex_backend_bandwidth_probe(
        residency->cuda_backend, &bandwidth, err);
    if (rc != YVEX_OK) goto done;
    rc = yvex_backend_open_shared_cuda(
        &session_backend, residency->cuda_backend, maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_resident_attach(
            session_backend, residency->arena,
            residency->backing_bytes,
            residency->cuda_weights, residency->summary.generation, err);
    if (rc != YVEX_OK) {
        primary = err ? *err : (yvex_error){0};
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_close_checked(&session_backend, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            *backend = session_backend;
            rc = cleanup_rc;
            if (err) *err = cleanup;
        } else if (err) *err = primary;
        goto done;
    }
    *backend = session_backend;
    *summary = residency->summary;
    rc = YVEX_OK;
done:
    (void)pthread_mutex_unlock(&residency->access_mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
/* Detach and release one process-lifetime resident full-model arena. */
int yvex_runtime_residency_close(yvex_runtime_residency **residency, yvex_error *err)
{
    return residency_release(residency, err);
}
/*
 * Snapshot synchronized residency facts and optionally borrow its stable host arena.
 *
 * Copies one coherent summary and optional immutable process-lifetime span. The snapshot extends
 * no lifetime and performs no artifact read or qtype decode.
 */
int yvex_runtime_residency_snapshot(const yvex_runtime_residency *residency,
                                    yvex_runtime_residency_summary *summary,
                                    const unsigned char **arena,
                                    unsigned long long *arena_bytes,
                                    yvex_error *err)
{
    yvex_runtime_residency *mutable_residency =
        (yvex_runtime_residency *)residency;
    int borrow_arena = arena || arena_bytes;
    if (arena) *arena = NULL;
    if (arena_bytes) *arena_bytes = 0ull;
    if (!residency || !summary || borrow_arena != (arena && arena_bytes)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.snapshot",
                       "residency, summary, and paired arena outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!residency->access_mutex_ready ||
        pthread_mutex_lock(&mutable_residency->access_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.snapshot",
                       "runtime residency synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    *summary = residency->summary;
    if (borrow_arena && (!summary->sealed || !summary->host_ready ||
                         summary->invalidated || !residency->arena)) {
        (void)pthread_mutex_unlock(&mutable_residency->access_mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.snapshot",
                       "sealed available host residency is required for arena access");
        return YVEX_ERR_STATE;
    }
    if (borrow_arena) {
        *arena = residency->arena;
        *arena_bytes = residency->backing_bytes;
    }
    (void)pthread_mutex_unlock(&mutable_residency->access_mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Borrow one exact immutable resident tensor without copying or reopening the artifact.
 *
 * Returns one model-lifetime encoded span and changes no counters. Missing, stale, or invalidated
 * bindings publish no pointer. Callers may decode or execute but never mutate the borrowed
 * resident bytes.
 */
int yvex_runtime_residency_binding_view(
    const yvex_runtime_residency *residency,
    const yvex_materialized_tensor_binding *binding,
    const unsigned char **data, unsigned long long *bytes,
    yvex_error *err)
{
    int resolved;
    if (data) *data = NULL;
    if (bytes) *bytes = 0ull;
    if (!residency || !binding || !data || !bytes) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.binding",
                       "residency, binding, and borrowed-span outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    resolved = residency_resolve(residency, binding, data, bytes);
    if (resolved != YVEX_MATERIALIZATION_READ_HIT) {
        *data = NULL;
        *bytes = 0ull;
        yvex_error_set(err, resolved == YVEX_MATERIALIZATION_READ_INVALID
                               ? YVEX_ERR_STATE : YVEX_ERR_FORMAT,
                       "runtime.residency.binding",
                       "requested tensor is not an available exact resident binding");
        return resolved == YVEX_MATERIALIZATION_READ_INVALID
                   ? YVEX_ERR_STATE : YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_residency_execution_view(
    const yvex_runtime_residency *residency,
    const yvex_materialized_tensor_binding *binding,
    const unsigned char **data, unsigned long long *bytes,
    yvex_execution_layout_class *layout, yvex_error *err)
{
    yvex_runtime_residency *mutable_residency =
        (yvex_runtime_residency *)residency;
    unsigned long long slot;
    const residency_record *record;
    if (data) *data = NULL;
    if (bytes) *bytes = 0ull;
    if (layout) *layout = YVEX_EXECUTION_LAYOUT_CANONICAL_ROW;
    if (!residency || !binding || !data || !bytes || !layout ||
        binding->tensor_id >= residency->record_index_count) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.execution",
                       "residency, binding, and execution-span outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!residency->access_mutex_ready ||
        pthread_mutex_lock(&mutable_residency->access_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.execution",
                       "runtime residency synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    slot = residency->record_index[binding->tensor_id];
    record = slot ? &residency->records[slot - 1ull] : NULL;
    if (!record || !record->decision || !record->binding ||
        record->binding->tensor_id != binding->tensor_id ||
        record->binding->qtype != binding->qtype ||
        record->binding->encoded_bytes != binding->encoded_bytes ||
        record->arena_offset > residency->backing_bytes ||
        record->storage_bytes > residency->backing_bytes - record->arena_offset ||
        !residency->arena || residency->summary.invalidated) {
        (void)pthread_mutex_unlock(&mutable_residency->access_mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.execution",
                       "requested tensor has no valid resident execution span");
        return YVEX_ERR_STATE;
    }
    *data = residency->arena + record->arena_offset;
    *bytes = record->storage_bytes;
    *layout = record->decision->layout;
    (void)pthread_mutex_unlock(&mutable_residency->access_mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Invalidate one shared resident generation without releasing live arena bytes.
 *
 * Process-lifetime residency owned by an invalidated runtime model. Makes every later provider
 * resolve fail closed and advances its generation. Synchronization or generation overflow leaves
 * the pack invalidated.
 */
int yvex_runtime_residency_invalidate(yvex_runtime_residency *residency,
                                      yvex_error *err)
{
    unsigned long long next;
    if (!residency) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.invalidate",
                       "runtime residency is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!residency->access_mutex_ready ||
        getenv("YVEX_TEST_RUNTIME_RESIDENCY_INVALIDATE_FAILURE") ||
        pthread_mutex_lock(&residency->access_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.invalidate",
                       "runtime residency synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    if (residency->summary.invalidated) {
        (void)pthread_mutex_unlock(&residency->access_mutex);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    residency->summary.invalidated = 1;
    residency->summary.host_ready = 0;
    residency->summary.cuda_ready = 0;
    if (!yvex_core_u64_add(residency->summary.generation, 1ull, &next)) {
        (void)pthread_mutex_unlock(&residency->access_mutex);
        yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.residency.invalidate",
                       "resident generation overflowed");
        return YVEX_ERR_BOUNDS;
    }
    residency->summary.generation = next;
    (void)pthread_mutex_unlock(&residency->access_mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_runtime_refuse(yvex_component_failure *failure,
                                    yvex_component_failure_code code,
                                    unsigned long long expected,
                                    unsigned long long actual, yvex_status status,
                                    const char *reason, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
    }
    yvex_error_set(err, status, "runtime.component", reason);
    return status;
}

static const yvex_component_binding *component_execution_binding_find(
    const char *target_id, const char *component_id)
{
    unsigned long long index;

    if (!target_id || !component_id) return NULL;
    for (index = 0ull;; ++index) {
        const yvex_component_binding *binding = yvex_component_binding_at(index);

        if (!binding) return NULL;
        if (binding->target_id && binding->component_id &&
            strcmp(binding->target_id, target_id) == 0 &&
            strcmp(binding->component_id, component_id) == 0)
            return binding;
    }
}

static int component_plan_identity(yvex_component_plan *plan)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.component.plan.v1") ||
        !yvex_sha256_update_u64(&hash, plan->binding_id) ||
        !yvex_sha256_update_u64(&hash, plan->binding_version) ||
        !yvex_sha256_update_text(&hash, plan->target_id) ||
        !yvex_sha256_update_text(&hash, plan->component_id) ||
        !yvex_sha256_update_u64(&hash, plan->backend) ||
        !yvex_sha256_update_u64(&hash, plan->batch) ||
        !yvex_sha256_update_u64(&hash, plan->geometry_rank))
        return 0;
    for (index = 0u; index < plan->geometry_rank; ++index)
        if (!yvex_sha256_update_u64(&hash, plan->geometry[index])) return 0;
    if (!yvex_sha256_update_u64(&hash, plan->input_values) ||
        !yvex_sha256_update_u64(&hash, plan->output_values) ||
        !yvex_sha256_update_u64(&hash, plan->workspace_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->output_rank))
        return 0;
    for (index = 0u; index < plan->output_rank; ++index)
        if (!yvex_sha256_update_u64(&hash, plan->output_dims[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, plan->identity);
    return 1;
}

static int component_plan_build(const yvex_component_plan_request *request,
                                yvex_component_plan *out,
                                yvex_component_failure *failure, yvex_error *err)
{
    const yvex_component_binding *binding;
    unsigned long long fixed_bytes = 0ull;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!request || !out || !request->target_id || !request->target_id[0] ||
        !request->component_id || !request->component_id[0] || !request->batch ||
        !request->geometry_rank ||
        request->geometry_rank > YVEX_COMPONENT_GEOMETRY_CAP ||
        !request->maximum_host_bytes ||
        strlen(request->target_id) >= sizeof(out->target_id) ||
        strlen(request->component_id) >= sizeof(out->component_id))
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG, "component plan requires complete typed inputs", err);
    binding = component_execution_binding_find(request->target_id,
                                               request->component_id);
    if (!binding || binding->schema_version != YVEX_COMPONENT_BINDING_SCHEMA_V1 ||
        !binding->binding_id || !binding->binding_version || !binding->plan ||
        !binding->admission_component || !binding->admission_component[0] ||
        !binding->admit || !binding->execute)
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_UNSUPPORTED, 1ull, 0ull,
            YVEX_ERR_UNSUPPORTED,
            "no admitted component execution binding matches target and component", err);
    if (binding->backend != request->backend)
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_UNSUPPORTED, binding->backend,
            request->backend, YVEX_ERR_UNSUPPORTED,
            "component execution binding does not admit the requested backend", err);
    out->schema_version = YVEX_COMPONENT_PLAN_SCHEMA_V1;
    out->binding_id = binding->binding_id;
    out->binding_version = binding->binding_version;
    out->backend = request->backend;
    out->batch = request->batch;
    out->geometry_rank = request->geometry_rank;
    memcpy(out->geometry, request->geometry, sizeof(out->geometry));
    yvex_core_text_copy(out->target_id, sizeof(out->target_id), request->target_id);
    yvex_core_text_copy(out->component_id, sizeof(out->component_id), request->component_id);
    rc = binding->plan(request, out, failure, err);
    if (rc != YVEX_OK) return rc;
    if (!out->input_values || !out->output_values || !out->output_rank ||
        out->output_rank > YVEX_COMPONENT_OUTPUT_RANK_CAP ||
        !yvex_core_u64_mul(out->input_values, sizeof(float), &out->input_bytes) ||
        !yvex_core_u64_mul(out->output_values, sizeof(float), &out->output_bytes) ||
        !yvex_core_u64_add(out->input_bytes, out->output_bytes, &fixed_bytes) ||
        fixed_bytes >= request->maximum_host_bytes)
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_BUDGET, request->maximum_host_bytes,
            fixed_bytes, YVEX_ERR_BOUNDS,
            "component input and output exceed the host-memory budget", err);
    out->workspace_bytes = request->maximum_host_bytes - fixed_bytes;
    if (!component_plan_identity(out))
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_LIFECYCLE, 1ull, 0ull,
            YVEX_ERR_STATE, "component plan identity construction failed", err);
    out->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_plan_validate(const yvex_component_plan *plan,
                                   yvex_component_failure *failure,
                                   yvex_error *err)
{
    yvex_component_plan_request request;
    yvex_component_plan canonical;
    unsigned long long maximum_host_bytes;
    int rc;

    if (!plan || plan->schema_version != YVEX_COMPONENT_PLAN_SCHEMA_V1 ||
        !plan->complete || !yvex_sha256_hex_valid(plan->identity) ||
        !yvex_core_u64_add(plan->input_bytes, plan->output_bytes,
                           &maximum_host_bytes) ||
        !yvex_core_u64_add(maximum_host_bytes, plan->workspace_bytes,
                           &maximum_host_bytes))
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG, "sealed component plan is required", err);
    memset(&request, 0, sizeof(request));
    request.target_id = plan->target_id;
    request.component_id = plan->component_id;
    request.backend = plan->backend;
    request.batch = plan->batch;
    request.geometry_rank = plan->geometry_rank;
    request.maximum_host_bytes = maximum_host_bytes;
    memcpy(request.geometry, plan->geometry, sizeof(request.geometry));
    rc = component_plan_build(&request, &canonical, failure, err);
    if (rc != YVEX_OK) return rc;
    if (canonical.binding_id != plan->binding_id ||
        canonical.binding_version != plan->binding_version ||
        canonical.input_values != plan->input_values ||
        canonical.input_bytes != plan->input_bytes ||
        canonical.output_values != plan->output_values ||
        canonical.output_bytes != plan->output_bytes ||
        canonical.workspace_bytes != plan->workspace_bytes ||
        canonical.output_rank != plan->output_rank ||
        memcmp(canonical.output_dims, plan->output_dims,
               sizeof(plan->output_dims)) != 0 ||
        strcmp(canonical.identity, plan->identity) != 0)
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_LIFECYCLE, 1ull, 0ull,
            YVEX_ERR_FORMAT, "component plan differs from canonical lowering", err);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_request_validate(
    const yvex_component_execution_request *request,
    const yvex_component_binding **binding, yvex_component_failure *failure,
    yvex_error *err)
{
    int rc;

    if (binding) *binding = NULL;
    if (!request || !request->plan || !request->input || !request->output ||
        !binding || request->output_capacity < request->plan->output_values)
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT,
            request && request->plan ? request->plan->output_values : 1ull,
            request ? request->output_capacity : 0ull, YVEX_ERR_INVALID_ARG,
            "component execution requires plan, buffers, and output capacity", err);
    rc = component_plan_validate(request->plan, failure, err);
    if (rc != YVEX_OK) return rc;
    *binding = component_execution_binding_find(request->plan->target_id,
                                                request->plan->component_id);
    if (!*binding || (*binding)->binding_id != request->plan->binding_id ||
        (*binding)->binding_version != request->plan->binding_version)
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_LIFECYCLE,
            request->plan->binding_id, *binding ? (*binding)->binding_id : 0ull,
            YVEX_ERR_FORMAT, "component binding differs from the sealed plan", err);
    return YVEX_OK;
}

static int component_resources_open(
    const yvex_component_binding *binding, const yvex_artifact *artifact,
    const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const yvex_component_plan *component_plan,
    yvex_materialization_plan **plan, yvex_materialization_session **session,
    yvex_runtime_residency **residency, yvex_component_failure *failure,
    yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options options;
    yvex_materialization_failure materialization_failure;
    yvex_runtime_residency_options residency_options = {0};
    yvex_runtime_residency_failure residency_failure;
    int rc = binding->admit(binding->admission_component, artifact, gguf, tensors, NULL,
                            &admission, NULL, &admission_failure, err);

    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (options.max_chunk_bytes > component_plan->workspace_bytes)
        options.max_chunk_bytes = component_plan->workspace_bytes;
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            plan, &admission, artifact, gguf, tensors, NULL, &options,
            &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            session, *plan, artifact, &options, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(
            *session, &materialization_failure, err);
    residency_options.maximum_host_bytes = admission.payload_bytes;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_residency_prepare(
            residency, *session, admission.logical_component_identity,
            &residency_options, &residency_failure, err);
    if (rc != YVEX_OK && failure && failure->code == YVEX_COMPONENT_FAILURE_NONE) {
        failure->code = rc == YVEX_ERR_FORMAT
                            ? YVEX_COMPONENT_FAILURE_MATERIALIZATION
                            : YVEX_COMPONENT_FAILURE_LIFECYCLE;
        failure->reason = yvex_error_message(err);
    }
    return rc;
}

static int component_execute(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, const yvex_component_execution_request *request,
    yvex_component_execution_result *result, yvex_component_failure *failure,
    yvex_error *err)
{
    const yvex_component_binding *binding = NULL;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_runtime_residency *residency = NULL;
    yvex_error primary, cleanup;
    int rc, cleanup_rc;

    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!artifact || !gguf || !tensors || !result)
        return component_runtime_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, 4ull, 0ull,
            YVEX_ERR_INVALID_ARG,
            "component execution requires admitted artifact views and output", err);
    rc = component_request_validate(request, &binding, failure, err);
    if (rc == YVEX_OK)
        rc = component_resources_open(binding, artifact, gguf, tensors,
                                      request->plan, &plan, &session, &residency,
                                      failure, err);
    if (rc == YVEX_OK)
        rc = binding->execute(session, request, result, failure, err);
    primary = err ? *err : (yvex_error){0};
    cleanup_rc = yvex_runtime_residency_close(&residency, &cleanup);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    if (cleanup_rc != YVEX_OK) {
        if (err) *err = cleanup;
        return cleanup_rc;
    }
    if (rc != YVEX_OK && err) *err = primary;
    return rc;
}

const yvex_runtime_component_api *yvex_runtime_component_api_get(void)
{
    static const yvex_runtime_component_api api = {
        component_plan_build, component_plan_validate, component_execute};

    return &api;
}
