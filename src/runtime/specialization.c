/* Seal one compact deployment implementation catalog for an opened model engine. */
#include <yvex/internal/execution.h>
#include <yvex/internal/core.h>
#include <yvex/internal/moe.h>
#include "src/runtime/private.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int specialization_refuse(yvex_error *err, yvex_status status,
                                 const char *reason)
{
    yvex_error_set(err, status, "runtime.specialization", reason);
    return status;
}

static int hash_finish(yvex_sha256 *hash, char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int implementation_valid(yvex_engine_implementation implementation)
{
    return implementation >= YVEX_ENGINE_IMPLEMENTATION_PORTABLE_F32 &&
           implementation < YVEX_ENGINE_IMPLEMENTATION_COUNT;
}

static int consumer_is_moe(yvex_execution_consumer_class consumer)
{
    return consumer == YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP ||
           consumer == YVEX_EXECUTION_CONSUMER_ROUTED_DOWN ||
           consumer == YVEX_EXECUTION_CONSUMER_SHARED_EXPERT;
}

static int consumer_is_routed(yvex_execution_consumer_class consumer)
{
    return consumer == YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP ||
           consumer == YVEX_EXECUTION_CONSUMER_ROUTED_DOWN;
}

static int encoded_qtype(unsigned int qtype, unsigned long long row_width)
{
    return row_width % 256ull == 0ull &&
           (qtype == YVEX_GGUF_QTYPE_IQ2_XXS || qtype == YVEX_GGUF_QTYPE_Q2_K ||
            qtype == YVEX_GGUF_QTYPE_Q8_0 || qtype == YVEX_GGUF_QTYPE_MXFP4);
}

static int implementation_equal(const yvex_engine_implementation_record *left,
                                const yvex_engine_implementation_record *right)
{
    return left->implementation == right->implementation &&
           left->fallback_implementation == right->fallback_implementation &&
           left->activation == right->activation &&
           left->fallback_activation == right->fallback_activation &&
           left->supported_width_mask == right->supported_width_mask &&
           left->worklist_width_mask == right->worklist_width_mask &&
           left->tensor_core_minimum == right->tensor_core_minimum;
}

static int implementation_seal(yvex_engine_implementation_record *record,
                               yvex_error *err)
{
    yvex_sha256 hash;
    if (!record || !implementation_valid(record->implementation) ||
        !implementation_valid(record->fallback_implementation) ||
        record->activation > YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED ||
        record->fallback_activation > YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED ||
        !record->supported_width_mask ||
        (record->tensor_core_minimum &&
         (!record->worklist_width_mask || record->tensor_core_minimum >= 63ull ||
          !(record->worklist_width_mask & (1ull << record->tensor_core_minimum)))))
        return specialization_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "implementation record is incomplete");
    record->schema_version = YVEX_ENGINE_SPECIALIZATION_SCHEMA_V1;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.engine.implementation.v1") ||
        !yvex_sha256_update_u64(&hash, record->schema_version) ||
        !yvex_sha256_update_u64(&hash, record->implementation) ||
        !yvex_sha256_update_u64(&hash, record->fallback_implementation) ||
        !yvex_sha256_update_u64(&hash, record->activation) ||
        !yvex_sha256_update_u64(&hash, record->fallback_activation) ||
        !yvex_sha256_update_u64(&hash, record->supported_width_mask) ||
        !yvex_sha256_update_u64(&hash, record->worklist_width_mask) ||
        !yvex_sha256_update_u64(&hash, record->tensor_core_minimum) ||
        !hash_finish(&hash, record->identity))
        return specialization_refuse(err, YVEX_ERR_STATE,
                                     "implementation identity derivation failed");
    return YVEX_OK;
}

static int implementation_add(yvex_engine_specialization *specialization,
                              yvex_engine_implementation_record *candidate,
                              unsigned int *handle, yvex_error *err)
{
    unsigned long long index;
    for (index = 0ull; index < specialization->summary.implementation_count; ++index)
        if (implementation_equal(&specialization->implementations[index], candidate)) {
            *handle = (unsigned int)index;
            return YVEX_OK;
        }
    if (specialization->summary.implementation_count >= YVEX_ENGINE_IMPLEMENTATION_CAP)
        return specialization_refuse(err, YVEX_ERR_BOUNDS,
                                     "implementation catalog capacity was exceeded");
    if (implementation_seal(candidate, err) != YVEX_OK) return yvex_error_code(err);
    index = specialization->summary.implementation_count++;
    specialization->implementations[index] = *candidate;
    *handle = (unsigned int)index;
    return YVEX_OK;
}

static int implementation_select(
    yvex_engine_specialization *specialization,
    const yvex_physical_execution_decision *package,
    const yvex_backend_device_info *device, unsigned long long width_mask,
    unsigned int *handle, yvex_error *err)
{
    yvex_engine_implementation_record candidate = {0};
    int routed = consumer_is_routed(package->consumer);
    candidate.supported_width_mask = width_mask;
    candidate.worklist_width_mask = routed ? width_mask & 0x1feull : 0ull;
    if (device->kind == YVEX_BACKEND_KIND_CPU) {
        candidate.activation = YVEX_EXECUTION_ACTIVATION_HOST_F32;
        candidate.fallback_activation = YVEX_EXECUTION_ACTIVATION_HOST_F32;
        candidate.implementation = YVEX_ENGINE_IMPLEMENTATION_PORTABLE_F32;
        candidate.fallback_implementation = YVEX_ENGINE_IMPLEMENTATION_PORTABLE_F32;
    } else if (device->kind == YVEX_BACKEND_KIND_CUDA) {
        int encoded = consumer_is_moe(package->consumer) &&
                      encoded_qtype(package->canonical_qtype,
                                    package->canonical_row_width);
        candidate.activation = encoded ? YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED
                                       : YVEX_EXECUTION_ACTIVATION_DEVICE_F32;
        candidate.fallback_activation = YVEX_EXECUTION_ACTIVATION_DEVICE_F32;
        candidate.implementation = encoded && device->compute_capability_major == 12 &&
                                           device->compute_capability_minor == 1
                                       ? YVEX_ENGINE_IMPLEMENTATION_CUDA_SM121_MOE_ROW
                                       : encoded ? YVEX_ENGINE_IMPLEMENTATION_CUDA_ENCODED_ROW
                                                 : YVEX_ENGINE_IMPLEMENTATION_CUDA_F32;
        candidate.fallback_implementation = YVEX_ENGINE_IMPLEMENTATION_CUDA_F32;
    } else {
        return specialization_refuse(err, YVEX_ERR_UNSUPPORTED,
                                     "backend has no admitted implementation class");
    }
    return implementation_add(specialization, &candidate, handle, err);
}

static int specialization_seal(yvex_engine_specialization *specialization,
                               yvex_error *err)
{
    yvex_engine_specialization_summary *summary = &specialization->summary;
    yvex_sha256 hash;
    unsigned long long index;
    if (!summary->package_decision_count || !summary->implementation_count ||
        !yvex_sha256_hex_is_valid(summary->package_execution_identity))
        return specialization_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "engine specialization is incomplete");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.engine.specialization.v1") ||
        !yvex_sha256_update_u64(&hash, summary->schema_version) ||
        !yvex_sha256_update_text(&hash, summary->package_execution_identity) ||
        !yvex_sha256_update_u64(&hash, summary->backend) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)summary->device_index) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)summary->compute_major) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)summary->compute_minor) ||
        !yvex_sha256_update_u64(&hash, summary->package_decision_count) ||
        !yvex_sha256_update_u64(&hash, summary->implementation_count))
        goto failed;
    for (index = 0ull; index < summary->implementation_count; ++index)
        if (!yvex_sha256_update_text(&hash, specialization->implementations[index].identity))
            goto failed;
    for (index = 0ull; index < summary->package_decision_count; ++index)
        if (!yvex_sha256_update_u64(&hash, specialization->decision_handles[index]))
            goto failed;
    if (hash_finish(&hash, summary->identity)) return YVEX_OK;
failed:
    return specialization_refuse(err, YVEX_ERR_STATE,
                                 "engine specialization identity derivation failed");
}

static int specialization_build(
    yvex_engine_specialization **out,
    const yvex_physical_execution_ir *package_execution,
    const yvex_model_execution_descriptor *model,
    yvex_backend_kind backend_kind, const yvex_backend *backend,
    yvex_error *err)
{
    const yvex_physical_execution_summary *package =
        yvex_physical_execution_ir_summary(package_execution);
    yvex_engine_specialization *specialization = NULL;
    yvex_backend_device_info device = {0};
    unsigned long long index, maximum_width, width_mask;
    int rc;
    if (out) *out = NULL;
    maximum_width = model && model->verification_width_maximum
                        ? model->verification_width_maximum : 1ull;
    if (!out || !package || !model || !package->decision_count || maximum_width >= 63ull ||
        model->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        (backend_kind != YVEX_BACKEND_KIND_CPU && backend_kind != YVEX_BACKEND_KIND_CUDA) ||
        (backend_kind == YVEX_BACKEND_KIND_CUDA && !backend))
        return specialization_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "package, model, and backend are required");
    device.kind = backend_kind;
    device.device_index = backend_kind == YVEX_BACKEND_KIND_CPU ? -1 : 0;
    rc = backend_kind == YVEX_BACKEND_KIND_CUDA
             ? yvex_backend_get_device_info(backend, &device, err) : YVEX_OK;
    if (rc != YVEX_OK) return rc;
    if (device.kind != backend_kind ||
        (backend && yvex_backend_kind_of(backend) != backend_kind))
        return specialization_refuse(err, YVEX_ERR_UNSUPPORTED,
                                     "backend device facts do not match");
    if (package->decision_count > SIZE_MAX / sizeof(*specialization->decision_handles))
        return specialization_refuse(err, YVEX_ERR_BOUNDS,
                                     "specialization mapping count overflowed");
    specialization = calloc(1u, sizeof(*specialization));
    if (specialization)
        specialization->decision_handles = calloc(
            (size_t)package->decision_count, sizeof(*specialization->decision_handles));
    if (!specialization || !specialization->decision_handles) {
        runtime_specialization_release(&specialization);
        return specialization_refuse(err, YVEX_ERR_NOMEM,
                                     "specialization allocation failed");
    }
    specialization->summary.schema_version = YVEX_ENGINE_SPECIALIZATION_SCHEMA_V1;
    specialization->summary.backend = device.kind;
    specialization->summary.device_index = device.device_index;
    specialization->summary.compute_major = device.compute_capability_major;
    specialization->summary.compute_minor = device.compute_capability_minor;
    specialization->summary.package_decision_count = package->decision_count;
    yvex_core_text_copy(specialization->summary.package_execution_identity,
                        sizeof(specialization->summary.package_execution_identity),
                        package->identity);
    width_mask = (1ull << (maximum_width + 1ull)) - 2ull;
    for (index = 0ull; index < package->decision_count; ++index) {
        const yvex_physical_execution_decision *decision =
            yvex_physical_execution_ir_decision_at(package_execution, index);
        rc = decision ? implementation_select(
                            specialization, decision, &device, width_mask,
                            &specialization->decision_handles[index], err)
                      : YVEX_ERR_FORMAT;
        if (rc != YVEX_OK) {
            runtime_specialization_release(&specialization);
            return rc;
        }
    }
    rc = specialization_seal(specialization, err);
    if (rc != YVEX_OK) {
        runtime_specialization_release(&specialization);
        return rc;
    }
    *out = specialization;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_model_specialization_prepare(
    yvex_model_engine *model, yvex_backend_kind backend_kind,
    yvex_backend *backend, const yvex_engine_specialization **out,
    yvex_error *err)
{
    yvex_engine_specialization *candidate = NULL;
    const yvex_runtime_descriptor_summary *descriptor;
    const yvex_engine_specialization_summary *existing, *replacement;
    yvex_backend_device_info device = {0};
    int rc;
    if (out) *out = NULL;
    if (!model || !out || backend_kind > YVEX_BACKEND_KIND_CUDA ||
        (backend_kind == YVEX_BACKEND_KIND_CUDA && !backend) ||
        (backend && yvex_backend_kind_of(backend) != backend_kind))
        return specialization_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "matching engine and backend are required");
    if (pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return specialization_refuse(err, YVEX_ERR_STATE,
                                     "specialization lock is unavailable");
    if (model->specializations[backend_kind]) {
        const yvex_engine_specialization *value = model->specializations[backend_kind];
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        existing = &value->summary;
        device.kind = backend_kind;
        device.device_index = -1;
        rc = backend_kind == YVEX_BACKEND_KIND_CUDA
                 ? yvex_backend_get_device_info(backend, &device, err) : YVEX_OK;
        if (rc == YVEX_OK &&
            (!existing || existing->backend != device.kind ||
             (backend_kind == YVEX_BACKEND_KIND_CUDA &&
              (existing->device_index != device.device_index ||
               existing->compute_major != device.compute_capability_major ||
               existing->compute_minor != device.compute_capability_minor))))
            rc = specialization_refuse(err, YVEX_ERR_UNSUPPORTED,
                                       "backend differs from cached specialization");
        if (rc == YVEX_OK) *out = value;
        return rc;
    }
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    descriptor = yvex_runtime_descriptor_summary_get(model->descriptor);
    rc = descriptor ? specialization_build(
                          &candidate, model->physical_execution,
                          &descriptor->model_execution, backend_kind, backend, err)
                    : specialization_refuse(err, YVEX_ERR_STATE,
                                            "engine descriptor is unavailable");
    if (rc != YVEX_OK) return rc;
    if (pthread_mutex_lock(&model->lifecycle_mutex) != 0) {
        runtime_specialization_release(&candidate);
        return specialization_refuse(err, YVEX_ERR_STATE,
                                     "specialization lock is unavailable");
    }
    if (model->close_requested) {
        rc = specialization_refuse(err, YVEX_ERR_STATE,
                                   "a draining engine cannot specialize");
    } else if (!model->specializations[backend_kind]) {
        model->specializations[backend_kind] = candidate;
        candidate = NULL;
        model->summary.engine_specialization_count++;
    } else {
        existing = &model->specializations[backend_kind]->summary;
        replacement = &candidate->summary;
        if (!existing || !replacement || strcmp(existing->identity, replacement->identity) != 0)
            rc = specialization_refuse(err, YVEX_ERR_STATE,
                                       "concurrent specialization disagrees");
    }
    if (rc == YVEX_OK) *out = model->specializations[backend_kind];
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    runtime_specialization_release(&candidate);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
