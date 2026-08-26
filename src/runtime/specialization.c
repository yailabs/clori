/* Seal one compact deployment implementation catalog for an opened model engine. */
#include <yvex/internal/execution.h>
#include <yvex/internal/core.h>
#include <yvex/internal/media.h>
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

typedef struct {
    yvex_transformer_linear_operation operation;
    yvex_transformer_linear_numeric_contract numeric_contract;
    yvex_dtype source_dtype;
    yvex_backend_kind backend;
    unsigned long long input_width, output_width, workspace_bytes;
    unsigned int algorithm_id, tile_rows, tile_columns, split_k;
    yvex_transformer_linear_reduction reduction;
    yvex_transformer_linear_stages stages;
    unsigned int compute_major, compute_minor;
    int bias;
} linear_implementation;

/* These records are deployment implementations of one typed exactness contract. The family
 * recipe names only source dtype, bias, operation geometry, and required numerical semantics. */
static const linear_implementation linear_implementations[] = {
    {YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_VIDEO_OUTPUT,
     YVEX_TRANSFORMER_LINEAR_NUMERIC_SOURCE_EXACT, YVEX_DTYPE_F32,
     YVEX_BACKEND_KIND_CUDA, 5376ull, 96ull, 1024ull * 1024ull,
     10u, 32u, 32u, 10u, YVEX_TRANSFORMER_LINEAR_REDUCTION_INPLACE,
     YVEX_TRANSFORMER_LINEAR_STAGES_DEFAULT, 12u, 1u, 1},
    {YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_AUDIO_OUTPUT,
     YVEX_TRANSFORMER_LINEAR_NUMERIC_SOURCE_EXACT, YVEX_DTYPE_F32,
     YVEX_BACKEND_KIND_CUDA, 5376ull, 32ull, 1024ull * 1024ull,
     20u, 128u, 32u, 3u, YVEX_TRANSFORMER_LINEAR_REDUCTION_COMPUTE_TYPE,
     YVEX_TRANSFORMER_LINEAR_STAGES_8X5, 12u, 1u, 1},
};

static int linear_specialize_one(
    const char *semantic_domain,
    const yvex_transformer_linear_requirement *requirement,
    yvex_backend_kind backend, yvex_transformer_linear_physical_plan *plan,
    yvex_error *err)
{
    const linear_implementation *selected = NULL;
    size_t domain_length, index;
    domain_length = strnlen(semantic_domain,
                            YVEX_TRANSFORMER_LINEAR_DOMAIN_CAP);
    for (index = 0u; index < sizeof(linear_implementations) /
                                  sizeof(linear_implementations[0]); ++index) {
        const linear_implementation *candidate = linear_implementations + index;
        if (candidate->operation == requirement->operation &&
            candidate->numeric_contract == requirement->publication_contract &&
            candidate->source_dtype == requirement->source_dtype &&
            candidate->bias == requirement->bias && candidate->backend == backend &&
            candidate->input_width == requirement->input_width &&
            candidate->output_width == requirement->output_width) {
            if (selected)
                return specialization_refuse(
                    err, YVEX_ERR_STATE,
                    "joint linear requirement has ambiguous implementations");
            selected = candidate;
        }
    }
    if (!selected)
        return specialization_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "joint linear requirement has no admitted deployment implementation");
    memset(plan, 0, sizeof(*plan));
    plan->schema_version = YVEX_TRANSFORMER_LINEAR_PHYSICAL_SCHEMA_V2;
    memcpy(plan->semantic_domain, semantic_domain, domain_length);
    plan->operation = selected->operation;
    plan->numeric_contract = selected->numeric_contract;
    plan->source_dtype = selected->source_dtype;
    plan->implementation = YVEX_TRANSFORMER_LINEAR_IMPLEMENTATION_CUBLAS_LT_F32_BIAS;
    plan->reduction = selected->reduction;
    plan->stages = selected->stages;
    plan->backend = selected->backend;
    plan->algorithm_id = selected->algorithm_id;
    plan->tile_rows = selected->tile_rows;
    plan->tile_columns = selected->tile_columns;
    plan->split_k = selected->split_k;
    plan->compute_capability_major = selected->compute_major;
    plan->compute_capability_minor = selected->compute_minor;
    plan->input_width = selected->input_width;
    plan->output_width = selected->output_width;
    plan->workspace_bytes = selected->workspace_bytes;
    plan->bias = selected->bias;
    plan->deterministic = 1;
    plan->exact = 1;
    return yvex_transformer_linear_physical_seal(plan, err);
}

static int media_linear_specialization_compile(
    const char *semantic_domain,
    const yvex_transformer_linear_requirement *video_requirement,
    const yvex_transformer_linear_requirement *audio_requirement,
    yvex_backend_kind backend,
    yvex_transformer_linear_physical_plan *video,
    yvex_transformer_linear_physical_plan *audio, yvex_error *err)
{
    yvex_transformer_linear_physical_plan selected_video, selected_audio;
    size_t domain_length;
    int rc;
    if (video) memset(video, 0, sizeof(*video));
    if (audio) memset(audio, 0, sizeof(*audio));
    domain_length = semantic_domain
                        ? strnlen(semantic_domain, YVEX_TRANSFORMER_LINEAR_DOMAIN_CAP)
                        : 0u;
    if (!video_requirement || !audio_requirement || video_requirement == audio_requirement ||
        !video || !audio || video == audio || !domain_length ||
        domain_length >= YVEX_TRANSFORMER_LINEAR_DOMAIN_CAP ||
        video_requirement->operation !=
            YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_VIDEO_OUTPUT ||
        audio_requirement->operation !=
            YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_AUDIO_OUTPUT)
        return specialization_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "one complete pair of linear semantic requirements is required");
    rc = linear_specialize_one(
        semantic_domain, video_requirement, backend, &selected_video, err);
    if (rc == YVEX_OK)
        rc = linear_specialize_one(
            semantic_domain, audio_requirement, backend, &selected_audio, err);
    if (rc != YVEX_OK) return rc;
    *video = selected_video;
    *audio = selected_audio;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_media_request_specialize(
    yvex_runtime_av_generation_request *request,
    const char *semantic_domain,
    const yvex_transformer_linear_requirement *video_requirement,
    const yvex_transformer_linear_requirement *audio_requirement,
    yvex_error *err)
{
    yvex_transformer_linear_physical_plan video, audio;
    int rc;
    if (!request)
        return specialization_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "media request is required for specialization");
    rc = media_linear_specialization_compile(
        semantic_domain, video_requirement, audio_requirement,
        request->component_backend, &video, &audio, err);
    if (rc != YVEX_OK) return rc;
    request->output_semantic_domain = semantic_domain;
    request->video_output_requirement = video_requirement;
    request->audio_output_requirement = audio_requirement;
    request->video_output_specialization = video;
    request->audio_output_specialization = audio;
    yvex_error_clear(err);
    return YVEX_OK;
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
