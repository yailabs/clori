/*
 * Admit immutable artifact and materialization evidence through typed gates.
 *
 * Gate checks return facts only and preserve public model/materialization gate API behavior. Cold
 * preparation may compose admitted DeepSeek compiler and artifact facts, but it owns no runtime
 * execution semantics; gate evidence is not artifact emission, generation readiness, benchmark
 * evidence, or release readiness.
 */
#include <yvex/internal/model_artifact.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/backend.h>
#include <yvex/model.h>

typedef struct {
    int value;
    const char *name;
} gate_name;

typedef enum {
    GATE_INPUT_OPEN,
    GATE_INPUT_DIGEST,
    GATE_INPUT_DIGEST_MISMATCH,
    GATE_INPUT_GGUF,
    GATE_INPUT_READY
} gate_input_stage;

typedef struct {
    const char *name;
    const char *dtype;
    unsigned int rank;
    unsigned long long dims[4];
    unsigned long long bytes;
} gate_expected_tensor;

static const yvex_complete_artifact_admission deepseek_selected_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX,
    .metadata_count = 76ull,
    .tensor_count = 1409ull,
    .payload_bytes = 108274154488ull,
    .file_bytes = YVEX_SELECTED_DEEPSEEK_FILE_BYTES,
    .source_snapshot_identity = 0x8d8da435dea23049ull,
    .mapping_identity = 0x779aa44d104fc718ull,
    .payload_identity =
        "e05ddb86f9783bf665d05395636588f4e8dbd1ee6f1ba54be4140f84369ee939",
    .transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .profile_identity =
        "a48d43c8594999a1af3a5b1f572b34a5823042cb767832d558642bb804b036c5",
    .profile_name = "deepseek-v4-flash-dspark-bootstrap-q2-v1",
    .quant_execution_identity =
        "777559149e4e8421c34299da78f63f6b0d296a91005d7670196164c3c72b62af",
    .payload_plan_identity =
        "8d1a89e794363c0aaf1c721b07c0661ea03f9680691d0113543b2540297b69e7",
    .payload_byte_identity =
        "6dce1edb82810715687d40c6d62273e992cfe9e0aa610cb9598447e06fb7099f",
    .writer_plan_identity =
        "1ba1ceaa709862145b1a145e938cf03327cd58da27bca42ade2f884e2b2fc635",
    .artifact_identity =
        "bf80bd7372e9ff754cd61d8f6e849ca8eff2177fad40840a2dad8e840b35690a",
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .tokenizer_complete = 1,
    .native_reader_accepted = 1,
    .official_reader_accepted = 1,
    .payload_integrity_accepted = 1,
    .materialization_input_ready = 1,
};

static const yvex_complete_artifact_admission deepseek_native_drafter_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX,
    .metadata_count = 76ull,
    .tensor_count = 1409ull,
    .payload_bytes = 98006498296ull,
    .file_bytes = 98018204640ull,
    .source_snapshot_identity = 0x8d8da435dea23049ull,
    .mapping_identity = 0x779aa44d104fc718ull,
    .payload_identity =
        "e05ddb86f9783bf665d05395636588f4e8dbd1ee6f1ba54be4140f84369ee939",
    .transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .profile_identity =
        "6a99e9f7c374e3f718cce705002bf2b799db9cc1b86f65091631857f52c1c587",
    .profile_name = "deepseek-v4-flash-dspark-native-drafter-candidate",
    .quant_execution_identity =
        "35002244d5854a2d51b877ea31614cd985c9795d11c7e0904ed3475fec7fcb77",
    .payload_plan_identity =
        "e83545c729b219d327d4a437d499b73407648c94748ba7fda13905baace15c3e",
    .payload_byte_identity =
        "c79712bb85e31ebdcbd71ef0256709a001ae4cc62c4150ba8726d5dc5722dcd0",
    .writer_plan_identity =
        "2d4694925c02c04811ea846f389a94dbf524d26809a292c93f2c46ca8f05a025",
    .artifact_identity =
        "59c4649b19bb9f3eb7c01559e12ae52c3d4fbd067957e35de0a1a851759c7cc1",
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .tokenizer_complete = 1,
    .native_reader_accepted = 1,
    .official_reader_accepted = 1,
    .payload_integrity_accepted = 1,
    .materialization_input_ready = 1,
};

static const yvex_complete_artifact_admission *deepseek_catalog_find(
    unsigned long long file_bytes)
{
    static const yvex_complete_artifact_admission *const rows[] = {
        &deepseek_selected_catalog,
        &deepseek_native_drafter_catalog,
    };
    size_t index;

    for (index = 0u; index < sizeof(rows) / sizeof(rows[0]); ++index)
        if (rows[index]->file_bytes == file_bytes) return rows[index];
    return NULL;
}

int yvex_artifact_admit_deepseek(
    const yvex_artifact *artifact, yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    const yvex_complete_artifact_admission *catalog =
        artifact ? deepseek_catalog_find(yvex_artifact_size(artifact)) : NULL;
    yvex_artifact_catalog_contract contract = {0};

    if (!artifact || !out)
        return yvex_artifact_admit_catalog(
            artifact, NULL, NULL, &contract, out, failure, err);
    if (!catalog) {
        memset(out, 0, sizeof(*out));
        if (failure) {
            memset(failure, 0, sizeof(*failure));
            failure->code = YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH;
            failure->actual = yvex_artifact_size(artifact);
            yvex_core_text_copy(failure->field, sizeof(failure->field), "file-bytes");
        }
        yvex_error_set(err, YVEX_ERR_FORMAT, "model.deepseek.artifact-catalog",
                       "artifact extent is not in the admitted DeepSeek physical catalog");
        return YVEX_ERR_FORMAT;
    }
    contract.catalog = catalog;
    return yvex_artifact_admit_catalog(
        artifact, NULL, NULL, &contract, out, failure, err);
}

static int runtime_descriptor_refuse(
    yvex_runtime_descriptor_failure *failure,
    yvex_runtime_descriptor_failure_code code, const char *name,
    unsigned long long index, unsigned long long expected,
    unsigned long long actual, const char *reason, yvex_error *err,
    yvex_status status)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->tensor_index = index;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
        if (name)
            yvex_core_text_copy(
                failure->tensor_name, sizeof(failure->tensor_name), name);
    }
    yvex_error_set(err, status, "yvex_runtime_descriptor_build_deepseek", reason);
    return status;
}

static void runtime_descriptor_hash_activation(
    yvex_sha256 *hash, const yvex_attention_activation_policy *policy)
{
#define HASH(MEMBER) yvex_sha256_update_u64(hash, (unsigned long long)policy->MEMBER)
    HASH(required); HASH(stage); HASH(quantization); HASH(block_axis); HASH(block_width);
    HASH(scale_format); HASH(scale_dtype); HASH(pre_transform); HASH(tail_policy);
    HASH(nonfinite_policy); HASH(fake_quant_inplace); HASH(zero_pad_hadamard_to_power_of_two);
#undef HASH
}

static void runtime_descriptor_hash_topk(
    yvex_sha256 *hash, const yvex_attention_topk_policy *policy)
{
#define HASH(MEMBER) yvex_sha256_update_u64(hash, (unsigned long long)policy->MEMBER)
    HASH(required); HASH(version); HASH(policy); HASH(k); HASH(reject_nonfinite);
    HASH(score_descending); HASH(equal_score_ordinal_ascending); HASH(plus_zero_equals_minus_zero);
    HASH(duplicate_ordinal_refused); HASH(output_ranked_order);
#undef HASH
}

/* Seal family numeric and execution facts before the generic descriptor builder consumes them. */
static int runtime_descriptor_family_facts(
    const yvex_deepseek_v4_ir *ir,
    yvex_runtime_descriptor_family_facts *facts,
    yvex_model_execution_descriptor *model_execution,
    char logical[YVEX_SHA256_HEX_CAP], char numeric[YVEX_SHA256_HEX_CAP],
    yvex_runtime_descriptor_failure *failure, yvex_error *err)
{
    const yvex_model_family_api *api = yvex_model_register_deepseek_v4();
    const yvex_deepseek_v4_model_spec *model = api->ir.model(ir);
    const yvex_deepseek_v4_layer_spec *first = api->ir.layer_at(ir, 0ull);
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long count, index;

    if (!model || !first || model->runtime_numeric_schema_version != 2u ||
        model->runtime_compute_policy_count != 1ull ||
        !model->runtime_activation_policy_count || !model->hadamard_revision[0] ||
        !api->transform.architecture_identity(ir, logical))
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek runtime numeric authority is incomplete", err,
            YVEX_ERR_FORMAT);
    yvex_sha256_init(&hash);
    yvex_sha256_update_text(&hash, "yvex.runtime.numeric.deepseek-v4.v2");
    yvex_sha256_update_text(&hash, model->hadamard_revision);
    yvex_sha256_update_text(&hash, model->sglang_revision);
    yvex_sha256_update_u64(&hash, model->runtime_numeric_schema_version);
    yvex_sha256_update_u64(&hash, model->runtime_compute_policy_count);
    yvex_sha256_update_u64(&hash, model->runtime_activation_policy_count);
    yvex_sha256_update_u64(&hash, model->runtime_sparse_topk_policy_count);
    count = api->ir.layer_count(ir);
    yvex_sha256_update_u64(&hash, count);
    for (index = 0ull; index < count; ++index) {
        const yvex_deepseek_v4_layer_spec *layer = api->ir.layer_at(ir, index);

        if (!layer || layer->moe.routed_experts != first->moe.routed_experts ||
            layer->moe.experts_per_token != first->moe.experts_per_token)
            return runtime_descriptor_refuse(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
                index, 1ull, 0ull,
                "DeepSeek runtime numeric layer is missing", err,
                YVEX_ERR_FORMAT);
        yvex_sha256_update_u64(&hash, layer->layer_index);
        yvex_sha256_update_u64(
            &hash, (unsigned long long)layer->attention_class);
        yvex_sha256_update_u64(
            &hash, (unsigned long long)layer->compute_contract);
        runtime_descriptor_hash_activation(
            &hash, &layer->attention_kv_activation);
        runtime_descriptor_hash_activation(
            &hash, &layer->compressor_activation);
        runtime_descriptor_hash_activation(
            &hash, &layer->compressor_rotated_activation);
        runtime_descriptor_hash_activation(
            &hash, &layer->indexer_query_activation);
        runtime_descriptor_hash_topk(&hash, &layer->sparse_topk);
    }
    if (!yvex_sha256_final(&hash, digest))
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek runtime numeric identity failed", err, YVEX_ERR_STATE);
    yvex_sha256_hex(digest, numeric);
    if (!api->ir.execution_descriptor ||
        api->ir.execution_descriptor(ir, logical, model_execution, err) != YVEX_OK)
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek model execution descriptor is incomplete", err,
            YVEX_ERR_FORMAT);
    *facts = (yvex_runtime_descriptor_family_facts){
        logical,
        numeric,
        model->hadamard_revision,
        model->runtime_numeric_schema_version,
        model->runtime_compute_policy_count,
        model->runtime_activation_policy_count,
        model->runtime_sparse_topk_policy_count,
        model->main_layer_count,
        model->auxiliary_layer_count,
        first->moe.routed_experts,
        first->moe.experts_per_token,
        model->vocabulary_size,
        model_execution};
    return YVEX_OK;
}

static const yvex_runtime_tensor_binding *runtime_descriptor_find_name(
    const yvex_runtime_descriptor *descriptor, unsigned long long count,
    const char *name)
{
    unsigned long long index;

    for (index = 0ull; index < count; ++index) {
        const yvex_runtime_tensor_binding *row =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        if (row && row->binding && strcmp(row->binding->name, name) == 0)
            return row;
    }
    return NULL;
}

int yvex_runtime_descriptor_build_deepseek(
    yvex_runtime_descriptor **out,
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_session *session,
    const yvex_deepseek_gguf_map *map, const yvex_deepseek_v4_ir *ir,
    yvex_runtime_descriptor_failure *failure, yvex_error *err)
{
    const yvex_model_family_api *api = yvex_model_register_deepseek_v4();
    const yvex_deepseek_gguf_map_summary *map_summary;
    yvex_runtime_descriptor_family_facts facts;
    yvex_model_execution_descriptor model_execution;
    yvex_runtime_descriptor *descriptor = NULL;
    char logical[YVEX_SHA256_HEX_CAP], numeric[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    int rc;

    if (out) *out = NULL;
    if (!out || !map || !ir)
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek runtime descriptor requires map and architecture", err,
            YVEX_ERR_INVALID_ARG);
    map_summary = api->lowering.summary(map);
    if (!map_summary || !map_summary->complete)
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek GGUF map is incomplete", err, YVEX_ERR_FORMAT);
    rc = runtime_descriptor_family_facts(
        ir, &facts, &model_execution, logical, numeric, failure, err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_runtime_descriptor_build(
        &descriptor, admission, session, &facts, failure, err);
    if (rc != YVEX_OK) return rc;
    if (map_summary->descriptor_count !=
        yvex_runtime_descriptor_summary_get(descriptor)->tensor_count) {
        rc = runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, map_summary->descriptor_count,
            yvex_runtime_descriptor_summary_get(descriptor)->tensor_count,
            "DeepSeek runtime descriptor count differs from map", err,
            YVEX_ERR_FORMAT);
        goto fail;
    }
    for (index = 0ull; index < map_summary->descriptor_count; ++index) {
        const yvex_deepseek_gguf_descriptor *map_row =
            api->lowering.at(map, index);
        const yvex_runtime_tensor_binding *row = map_row
            ? runtime_descriptor_find_name(
                  descriptor, map_summary->descriptor_count,
                  map_row->emitted_name)
            : NULL;

        if (!map_row || !row || row->role != map_row->role ||
            row->scope != map_row->scope ||
            row->layer_index != map_row->layer_index ||
            row->predictor_index != map_row->predictor_index) {
            rc = runtime_descriptor_refuse(
                failure,
                map_row && !row
                    ? YVEX_RUNTIME_DESCRIPTOR_FAILURE_MISSING_BINDING
                    : YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
                map_row ? map_row->emitted_name : NULL, index, 1ull, 0ull,
                "DeepSeek runtime binding does not match the lowering map", err,
                YVEX_ERR_FORMAT);
            goto fail;
        }
    }
    *out = descriptor;
    yvex_error_clear(err);
    return YVEX_OK;

fail:
    yvex_runtime_descriptor_close(descriptor);
    return rc;
}

static int deepseek_terminal_find(
    const void *context, const char *emitted_name,
    yvex_materialization_terminal *out)
{
    const yvex_model_family_lowering_api *lowering =
        yvex_model_deepseek_lowering_api();
    const yvex_deepseek_gguf_map *map =
        (const yvex_deepseek_gguf_map *)context;
    const yvex_deepseek_gguf_descriptor *descriptor;
    const yvex_deepseek_gguf_descriptor *first;

    if (!map || !emitted_name || !out || !lowering) return 0;
    descriptor = lowering->find_emitted(map, emitted_name);
    first = lowering->at(map, 0ull);
    if (!descriptor || !first) return 0;
    memset(out, 0, sizeof(*out));
    out->descriptor_index = (unsigned long long)(descriptor - first);
    out->role = descriptor->role;
    out->collection = descriptor->collection;
    out->scope = descriptor->scope;
    out->layer_index = descriptor->layer_index;
    out->predictor_index = descriptor->predictor_index;
    out->expert_count = descriptor->expert_count;
    return 1;
}

int yvex_deepseek_materialization_projection(
    const yvex_deepseek_gguf_map *map,
    yvex_materialization_projection *out, yvex_error *err)
{
    const yvex_model_family_lowering_api *lowering =
        yvex_model_deepseek_lowering_api();
    const yvex_deepseek_gguf_map_summary *summary =
        map && lowering ? lowering->summary(map) : NULL;

    if (!out || !summary || !summary->complete ||
        !summary->mapping_identity) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "artifact.deepseek.materialization",
            "complete family lowering is required for terminal projection");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_MATERIALIZATION_PROJECTION_SCHEMA_VERSION;
    out->mapping_identity = summary->mapping_identity;
    out->descriptor_count = summary->descriptor_count;
    out->context = map;
    out->find = deepseek_terminal_find;
    out->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

_Static_assert(sizeof(gate_expected_tensor) == sizeof(yvex_model_gate_expected_tensor),
               "model gate tensor ABI changed");
_Static_assert(sizeof(gate_expected_tensor) == sizeof(yvex_materialize_expected_tensor),
               "materialization gate tensor ABI changed");
_Static_assert(offsetof(gate_expected_tensor, bytes) ==
                   offsetof(yvex_model_gate_expected_tensor, bytes),
               "model gate tensor layout changed");
_Static_assert(offsetof(gate_expected_tensor, bytes) ==
                   offsetof(yvex_materialize_expected_tensor, bytes),
               "materialization gate tensor layout changed");

static const gate_name model_gate_names[] = {
    {YVEX_MODEL_GATE_UNKNOWN, "model-gate-unknown"},
    {YVEX_MODEL_GATE_PASS, "model-gate-pass"},
    {YVEX_MODEL_GATE_PARTIAL, "model-gate-partial"},
    {YVEX_MODEL_GATE_FAIL, "model-gate-fail"},
    {YVEX_MODEL_GATE_BLOCKED, "model-gate-blocked"}
};

static const gate_name model_support_names[] = {
    {YVEX_MODEL_SUPPORT_NONE, "none"},
    {YVEX_MODEL_SUPPORT_DESCRIPTOR_ONLY, "descriptor-only"},
    {YVEX_MODEL_SUPPORT_SELECTED_TENSOR_MATERIALIZED,
     "selected-tensor-materialized"},
    {YVEX_MODEL_SUPPORT_FULL_WEIGHTS_MATERIALIZED,
     "full-weights-materialized"},
    {YVEX_MODEL_SUPPORT_PARTIAL_GRAPH_EXECUTABLE,
     "partial-graph-executable"},
    {YVEX_MODEL_SUPPORT_PREFILL_READY, "prefill-ready"},
    {YVEX_MODEL_SUPPORT_DECODE_READY, "decode-ready"},
    {YVEX_MODEL_SUPPORT_GENERATION_READY, "generation-ready"}
};

static const gate_name model_backend_names[] = {
    {YVEX_MODEL_GATE_BACKEND_NOT_TESTED, "not-tested"},
    {YVEX_MODEL_GATE_BACKEND_PASS, "pass"},
    {YVEX_MODEL_GATE_BACKEND_FAIL, "fail"},
    {YVEX_MODEL_GATE_BACKEND_UNAVAILABLE, "unavailable"}
};

static const gate_name materialize_gate_names[] = {
    {YVEX_MATERIALIZE_GATE_UNKNOWN, "materialize-gate-unknown"},
    {YVEX_MATERIALIZE_GATE_PASS, "materialize-gate-pass"},
    {YVEX_MATERIALIZE_GATE_PARTIAL, "materialize-gate-partial"},
    {YVEX_MATERIALIZE_GATE_FAIL, "materialize-gate-fail"},
    {YVEX_MATERIALIZE_GATE_BLOCKED, "materialize-gate-blocked"}
};

static const gate_name materialize_scope_names[] = {
    {YVEX_MATERIALIZE_SCOPE_UNKNOWN, "unknown"},
    {YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR, "selected-tensor"},
    {YVEX_MATERIALIZE_SCOPE_PARTIAL_MODEL, "partial-model"},
    {YVEX_MATERIALIZE_SCOPE_FULL_MODEL, "full-model"}
};

static const gate_name materialize_backend_names[] = {
    {YVEX_MATERIALIZE_BACKEND_NOT_TESTED, "not-tested"},
    {YVEX_MATERIALIZE_BACKEND_PASS, "pass"},
    {YVEX_MATERIALIZE_BACKEND_FAIL, "fail"},
    {YVEX_MATERIALIZE_BACKEND_UNAVAILABLE, "unavailable"}
};

static const gate_name materialize_failure_names[] = {
    {YVEX_MATERIALIZE_FAILURE_NONE, "none"},
    {YVEX_MATERIALIZE_FAILURE_MISSING_FILE, "missing_file"},
    {YVEX_MATERIALIZE_FAILURE_HASH_MISMATCH, "hash_mismatch"},
    {YVEX_MATERIALIZE_FAILURE_GGUF_PARSE, "gguf_parse"},
    {YVEX_MATERIALIZE_FAILURE_TENSOR_SPEC_MISMATCH, "tensor_spec_mismatch"},
    {YVEX_MATERIALIZE_FAILURE_UNSUPPORTED_DTYPE, "unsupported_dtype"},
    {YVEX_MATERIALIZE_FAILURE_UNSUPPORTED_QTYPE, "unsupported_qtype"},
    {YVEX_MATERIALIZE_FAILURE_BACKEND_UNAVAILABLE, "backend_unavailable"},
    {YVEX_MATERIALIZE_FAILURE_BACKEND_ALLOC, "backend_alloc"},
    {YVEX_MATERIALIZE_FAILURE_BACKEND_COPY, "backend_copy"},
    {YVEX_MATERIALIZE_FAILURE_OOM, "oom"},
    {YVEX_MATERIALIZE_FAILURE_UNKNOWN, "unknown"}
};

static const yvex_model_gate_summary model_gate_initial = {
    .status = YVEX_MODEL_GATE_UNKNOWN,
    .support_level = YVEX_MODEL_SUPPORT_NONE,
    .cpu_status = YVEX_MODEL_GATE_BACKEND_NOT_TESTED,
    .cuda_status = YVEX_MODEL_GATE_BACKEND_NOT_TESTED,
};

static const yvex_materialize_gate_summary materialize_gate_initial = {
    .status = YVEX_MATERIALIZE_GATE_UNKNOWN,
    .failure_class = YVEX_MATERIALIZE_FAILURE_NONE,
    .materialization_gate = "fail",
    .materialization_phase = "preflight",
    .integrity_status = "unchecked",
    .shape_status = "unchecked",
    .range_status = "unchecked",
    .backend_status = "not-opened",
    .cleanup_status = "not-needed",
    .cpu_status = YVEX_MATERIALIZE_BACKEND_NOT_TESTED,
    .cuda_status = YVEX_MATERIALIZE_BACKEND_NOT_TESTED,
};

static const char *gate_name_find(const gate_name *names,
                                  size_t count,
                                  int value,
                                  const char *fallback)
{
    size_t index;

    for (index = 0; index < count; ++index) {
        if (names[index].value == value)
            return names[index].name;
    }
    return fallback;
}

static int gate_options_validate(const char *owner, const char *path,
                                 unsigned long long expected_count,
                                 const void *expected, yvex_error *err)
{
    if (!path || !path[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, owner, "model_path is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (expected_count && !expected) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, owner,
                       "expected_tensors is required when expected_tensor_count is nonzero");
        return YVEX_ERR_INVALID_ARG;
    }
    return YVEX_OK;
}

static int gate_inputs_open(
    const char *path, const char *expected_sha256, yvex_artifact **artifact,
    yvex_gguf **gguf, yvex_tensor_table **tensors, char actual_sha256[65],
    gate_input_stage *stage, yvex_error *err)
{
    yvex_artifact_options options = {0};
    int rc;

    *artifact = NULL;
    *gguf = NULL;
    *tensors = NULL;
    actual_sha256[0] = '\0';
    *stage = GATE_INPUT_OPEN;
    options.path = path;
    options.readonly = 1;
    options.map = 1;
    rc = yvex_artifact_open(artifact, &options, err);
    if (rc != YVEX_OK) return rc;
    if (expected_sha256 && expected_sha256[0]) {
        *stage = GATE_INPUT_DIGEST;
        rc = yvex_artifact_sha256_hex_bytes(
            yvex_artifact_data(*artifact), yvex_artifact_size(*artifact),
            actual_sha256, err);
        if (rc != YVEX_OK) return rc;
        if (strcmp(actual_sha256, expected_sha256) != 0) {
            *stage = GATE_INPUT_DIGEST_MISMATCH;
            return YVEX_ERR_STATE;
        }
    }
    *stage = GATE_INPUT_GGUF;
    rc = yvex_gguf_open(gguf, *artifact, err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(tensors, *gguf, err);
    if (rc != YVEX_OK) return rc;
    *stage = GATE_INPUT_READY;
    return YVEX_OK;
}

static void gate_inputs_close(yvex_tensor_table *tensors,
                              yvex_gguf *gguf,
                              yvex_artifact *artifact)
{
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
}

int yvex_model_artifact_gate_from_admission(
    const yvex_complete_artifact_admission *admission,
    yvex_model_complete_artifact_gate_fact *fact,
    yvex_error *err)
{
    if (fact) memset(fact, 0, sizeof(*fact));
    if (!admission || !fact) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "model_artifact.complete_gate",
                       "admission and gate fact are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!admission->complete ||
        admission->artifact_class != YVEX_ARTIFACT_CLASS_COMPLETE_YVEX ||
        !admission->materialization_input_ready ||
        admission->runtime_supported || !admission->artifact_identity[0] ||
        !admission->artifact_path[0] || !admission->profile_name[0] ||
        admission->tensor_count == 0u || admission->file_bytes == 0u) {
        fact->status = YVEX_MODEL_GATE_BLOCKED;
        fact->support_level = YVEX_MODEL_SUPPORT_NONE;
        yvex_error_set(err, YVEX_ERR_STATE,
                       "model_artifact.complete_gate",
                       "canonical complete-artifact admission is required");
        return YVEX_ERR_STATE;
    }
    fact->status = YVEX_MODEL_GATE_PASS;
    fact->support_level = YVEX_MODEL_SUPPORT_DESCRIPTOR_ONLY;
    fact->artifact_identity = admission->artifact_identity;
    fact->artifact_path = admission->artifact_path;
    fact->profile_name = admission->profile_name;
    fact->tensor_count = admission->tensor_count;
    fact->file_bytes = admission->file_bytes;
    fact->complete_artifact_admitted = 1;
    fact->materialization_input_ready = 1;
    fact->execution_ready = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int tensor_spec_matches(const char *name,
                               const char *dtype,
                               unsigned int rank,
                               const unsigned long long dims[4],
                               unsigned long long bytes,
                               const yvex_tensor_info *actual)
{
    const char *actual_dtype;
    unsigned int i;

    if (!name || !dtype || !actual || strcmp(name, actual->name) != 0 ||
        rank != actual->rank || bytes != actual->storage_bytes) {
        return 0;
    }
    actual_dtype = yvex_dtype_name(actual->dtype);
    if (!actual_dtype || strcmp(dtype, actual_dtype) != 0) return 0;
    for (i = 0; i < rank && i < 4u; ++i) {
        if (dims[i] != actual->dims[i]) return 0;
    }
    return 1;
}

static void gate_expected_tensors_count(
    const void *expected, size_t stride, unsigned long long count,
    const yvex_tensor_table *tensors, unsigned long long *matches,
    unsigned long long *mismatches)
{
    const unsigned char *cursor = (const unsigned char *)expected;
    unsigned long long index;

    for (index = 0ull; index < count; ++index) {
        gate_expected_tensor item;
        const yvex_tensor_info *actual;

        memcpy(&item, cursor + index * stride, sizeof(item));
        actual = yvex_tensor_table_find(tensors, item.name);
        if (tensor_spec_matches(item.name, item.dtype, item.rank, item.dims,
                                item.bytes, actual))
            ++*matches;
        else
            ++*mismatches;
    }
}

static int materialize_repeated(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, yvex_backend_kind kind,
    const char *backend_name, const char *owner, unsigned int repeat_count,
    int check_capabilities, int check_cleanup,
    yvex_materialize_gate_summary *gate_summary,
    yvex_materialize_backend_status *backend_status,
    unsigned long long *bytes_materialized, int *cleanup_verified,
    yvex_materialize_failure_class *failure_class, yvex_error *err);

/*
 * Execute one temporary all-tensor materialization backend probe.
 *
 * Artifact views and backend identity are borrowed; status is caller-owned.
 */
static int materialize_backend(const yvex_artifact *artifact,
                               const yvex_gguf *gguf,
                               const yvex_tensor_table *tensors,
                               yvex_backend_kind kind,
                               const char *backend_name,
                               yvex_model_gate_backend_status *status,
                               yvex_error *err)
{
    yvex_materialize_backend_status backend_status;
    unsigned long long bytes_materialized;
    int cleanup_verified;
    int rc;

    rc = materialize_repeated(
        artifact, gguf, tensors, kind, backend_name, "yvex_model_gate_check",
        1u, 0, 0, NULL, &backend_status, &bytes_materialized,
        &cleanup_verified, NULL, err);
    if (backend_status == YVEX_MATERIALIZE_BACKEND_UNAVAILABLE)
        *status = YVEX_MODEL_GATE_BACKEND_UNAVAILABLE;
    else if (backend_status == YVEX_MATERIALIZE_BACKEND_PASS)
        *status = YVEX_MODEL_GATE_BACKEND_PASS;
    else
        *status = YVEX_MODEL_GATE_BACKEND_FAIL;
    return rc;
}

static int required_backend_failed(yvex_model_gate_backend_status status)
{
    return status != YVEX_MODEL_GATE_BACKEND_PASS;
}

int yvex_model_gate_check(const yvex_model_gate_options *options,
                          yvex_model_gate_summary *summary_out,
                          yvex_error *err)
{
    struct model_backend_check {
        yvex_backend_kind kind;
        const char *name;
        int enabled;
        int required;
        yvex_model_gate_backend_status *status;
    } backend_checks[2];
    yvex_model_gate_summary summary;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    char actual_sha256[65] = {0};
    gate_input_stage input_stage;
    size_t backend_index;
    int rc;

    if (!options || !summary_out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_model_gate_check",
                       "options and summary_out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = gate_options_validate(
        "yvex_model_gate_check", options->model_path,
        options->expected_tensor_count, options->expected_tensors, err);
    if (rc != YVEX_OK) return rc;

    summary = model_gate_initial;
    summary.model_path = options->model_path;
    summary.model_label = options->model_label;
    summary.family = options->family;
    summary.expected_sha256 = options->artifact_sha256 && options->artifact_sha256[0]
        ? options->artifact_sha256 : "";
    summary.digest_status = options->artifact_sha256 && options->artifact_sha256[0]
        ? "unchecked" : "unrequested";
    summary.identity_status = options->artifact_sha256 && options->artifact_sha256[0]
        ? "unchecked" : "unrequested";
    *summary_out = summary;

    rc = gate_inputs_open(
        options->model_path, options->artifact_sha256, &artifact, &gguf,
        &tensors, actual_sha256, &input_stage, err);
    if (artifact) summary.file_bytes = yvex_artifact_size(artifact);
    yvex_core_text_copy(summary.actual_sha256, sizeof(summary.actual_sha256),
                        actual_sha256);
    if (rc != YVEX_OK) {
        summary.status = input_stage == GATE_INPUT_OPEN ||
                                 input_stage == GATE_INPUT_DIGEST_MISMATCH
                             ? YVEX_MODEL_GATE_BLOCKED
                             : YVEX_MODEL_GATE_FAIL;
        if (input_stage == GATE_INPUT_DIGEST ||
            input_stage == GATE_INPUT_DIGEST_MISMATCH) {
            summary.digest_status = "fail";
            summary.identity_status = "fail";
            if (input_stage == GATE_INPUT_DIGEST)
                yvex_error_set(err, rc, "yvex_model_gate_check",
                               "sha256 calculation failed");
            else
                yvex_error_setf(err, YVEX_ERR_STATE, "yvex_model_gate_check",
                                "sha256 mismatch: expected %s got %s",
                                options->artifact_sha256, actual_sha256);
        }
        goto done;
    }
    if (options->artifact_sha256 && options->artifact_sha256[0]) {
        summary.digest_status = "pass";
        summary.identity_status = "pass";
    }
    summary.tensor_count = yvex_tensor_table_count(tensors);
    summary.support_level = YVEX_MODEL_SUPPORT_DESCRIPTOR_ONLY;

    gate_expected_tensors_count(
        options->expected_tensors, sizeof(options->expected_tensors[0]),
        options->expected_tensor_count, tensors, &summary.expected_tensor_matches,
        &summary.expected_tensor_mismatches);

    if (summary.expected_tensor_mismatches != 0) {
        summary.status = YVEX_MODEL_GATE_FAIL;
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_model_gate_check",
                       "expected tensor specification mismatch");
        rc = YVEX_ERR_STATE;
        goto done;
    }

    backend_checks[0] = (struct model_backend_check){
        YVEX_BACKEND_KIND_CPU, "cpu", options->check_cpu, options->require_cpu,
        &summary.cpu_status};
    backend_checks[1] = (struct model_backend_check){
        YVEX_BACKEND_KIND_CUDA, "cuda", options->check_cuda, options->require_cuda,
        &summary.cuda_status};
    for (backend_index = 0u; backend_index < 2u; ++backend_index) {
        if (!backend_checks[backend_index].enabled) continue;
        rc = materialize_backend(
            artifact, gguf, tensors, backend_checks[backend_index].kind,
            backend_checks[backend_index].name, backend_checks[backend_index].status, err);
        if (rc != YVEX_OK && backend_checks[backend_index].required) {
            summary.status = *backend_checks[backend_index].status ==
                                     YVEX_MODEL_GATE_BACKEND_UNAVAILABLE
                                 ? YVEX_MODEL_GATE_BLOCKED
                                 : YVEX_MODEL_GATE_FAIL;
            goto done;
        }
    }

    if ((options->require_cpu && required_backend_failed(summary.cpu_status)) ||
        (options->require_cuda && required_backend_failed(summary.cuda_status))) {
        summary.status = YVEX_MODEL_GATE_BLOCKED;
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_model_gate_check",
                       "required materialization backend did not pass");
    } else if ((options->check_cpu && summary.cpu_status == YVEX_MODEL_GATE_BACKEND_FAIL) ||
               (options->check_cuda && summary.cuda_status == YVEX_MODEL_GATE_BACKEND_FAIL)) {
        summary.status = YVEX_MODEL_GATE_PARTIAL;
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_model_gate_check",
                       "one requested materialization backend failed");
    } else if (options->expected_tensor_count > 0 &&
               summary.expected_tensor_matches == options->expected_tensor_count &&
               ((!options->check_cpu) || summary.cpu_status == YVEX_MODEL_GATE_BACKEND_PASS) &&
               ((!options->check_cuda) || summary.cuda_status == YVEX_MODEL_GATE_BACKEND_PASS)) {
        summary.status = YVEX_MODEL_GATE_PASS;
        summary.support_level = YVEX_MODEL_SUPPORT_SELECTED_TENSOR_MATERIALIZED;
    } else {
        summary.status = YVEX_MODEL_GATE_PASS;
        summary.support_level = YVEX_MODEL_SUPPORT_DESCRIPTOR_ONLY;
    }

    summary.execution_ready = 0;
    rc = summary.status == YVEX_MODEL_GATE_PASS ? YVEX_OK : YVEX_ERR_STATE;

done:
    *summary_out = summary;
    gate_inputs_close(tensors, gguf, artifact);
    return rc;
}

const char *yvex_model_gate_status_name(yvex_model_gate_status status)
{
    return gate_name_find(model_gate_names,
                          sizeof(model_gate_names) / sizeof(model_gate_names[0]), status,
                          "model-gate-unknown");
}

const char *yvex_model_support_level_name(yvex_model_support_level level)
{
    return gate_name_find(model_support_names,
                          sizeof(model_support_names) / sizeof(model_support_names[0]), level,
                          "none");
}

const char *yvex_model_gate_backend_status_name(yvex_model_gate_backend_status status)
{
    return gate_name_find(model_backend_names,
                          sizeof(model_backend_names) / sizeof(model_backend_names[0]), status,
                          "not-tested");
}

static yvex_materialize_failure_class classify_materialize_failure(int rc,
                                                                   const yvex_error *err)
{
    const char *msg = err ? yvex_error_message(err) : "";
    if (rc == YVEX_ERR_NOMEM) return YVEX_MATERIALIZE_FAILURE_OOM;
    if (rc == YVEX_ERR_UNSUPPORTED) {
        if (msg && strstr(msg, "qtype")) return YVEX_MATERIALIZE_FAILURE_UNSUPPORTED_QTYPE;
        return YVEX_MATERIALIZE_FAILURE_UNSUPPORTED_DTYPE;
    }
    if (rc == YVEX_ERR_BACKEND) {
        if (msg && strstr(msg, "alloc")) return YVEX_MATERIALIZE_FAILURE_BACKEND_ALLOC;
        if (msg && (strstr(msg, "write") || strstr(msg, "copy"))) {
            return YVEX_MATERIALIZE_FAILURE_BACKEND_COPY;
        }
        return YVEX_MATERIALIZE_FAILURE_BACKEND_ALLOC;
    }
    return YVEX_MATERIALIZE_FAILURE_UNKNOWN;
}

/*
 * Repeat one backend materialization while checking release-to-baseline.
 *
 * Every failing iteration closes active weights and backend ownership.
 */
static int materialize_repeated(const yvex_artifact *artifact,
                                const yvex_gguf *gguf,
                                const yvex_tensor_table *tensors,
                                yvex_backend_kind kind,
                                const char *backend_name,
                                const char *owner,
                                unsigned int repeat_count,
                                int check_capabilities,
                                int check_cleanup,
                                yvex_materialize_gate_summary *gate_summary,
                                yvex_materialize_backend_status *backend_status,
                                unsigned long long *bytes_materialized,
                                int *cleanup_verified,
                                yvex_materialize_failure_class *failure_class,
                                yvex_error *err)
{
    yvex_backend *backend = NULL;
    yvex_backend_options backend_options = {.kind = kind};
    yvex_backend_memory_stats before_stats = {0};
    yvex_backend_memory_stats after_stats = {0};
    int have_before = 0;
    unsigned int i;
    int rc;

    if (repeat_count == 0) repeat_count = 1;
    *backend_status = YVEX_MATERIALIZE_BACKEND_FAIL;
    *bytes_materialized = 0;
    if (cleanup_verified) *cleanup_verified = check_cleanup ? 1 : 0;

    if (kind == YVEX_BACKEND_KIND_CPU) {
        rc = yvex_backend_open_cpu(&backend, err);
    } else {
        rc = yvex_backend_open(&backend, &backend_options, err);
    }
    if (rc == YVEX_ERR_UNSUPPORTED) {
        *backend_status = YVEX_MATERIALIZE_BACKEND_UNAVAILABLE;
        if (gate_summary) gate_summary->backend_status = "unavailable";
        if (failure_class) *failure_class = YVEX_MATERIALIZE_FAILURE_BACKEND_UNAVAILABLE;
        return YVEX_OK;
    }
    if (rc != YVEX_OK) {
        if (gate_summary) gate_summary->backend_status = "fail";
        if (failure_class) *failure_class = classify_materialize_failure(rc, err);
        return rc;
    }
    if (gate_summary) {
        gate_summary->backend_status =
            yvex_backend_status_name(yvex_backend_status_of(backend));
    }

    if (check_capabilities &&
        (!yvex_backend_supports(backend, YVEX_BACKEND_CAP_TENSOR_ALLOC) ||
         !yvex_backend_supports(backend, YVEX_BACKEND_CAP_TENSOR_READ_WRITE))) {
        *backend_status = YVEX_MATERIALIZE_BACKEND_UNAVAILABLE;
        if (gate_summary) gate_summary->backend_status = "memory-unsupported";
        if (failure_class) *failure_class = YVEX_MATERIALIZE_FAILURE_BACKEND_UNAVAILABLE;
        yvex_backend_close(backend);
        return YVEX_OK;
    }

    if (check_cleanup &&
        yvex_backend_get_memory_stats(backend, &before_stats, err) == YVEX_OK) {
        have_before = 1;
    }

    for (i = 0; i < repeat_count; ++i) {
        yvex_weight_table *weights = NULL;
        yvex_materialize_options options = {
            .backend_name = backend_name, .require_all_tensors = 1};
        yvex_materialize_summary summary = {0};

        rc = yvex_weight_table_materialize(&weights,
                                           artifact,
                                           gguf,
                                           tensors,
                                           backend,
                                           &options,
                                           err);
        if (rc == YVEX_OK) {
            rc = yvex_weight_table_get_summary(weights, &summary, err);
        }
        if (rc != YVEX_OK || summary.status != YVEX_WEIGHT_STATUS_MATERIALIZED ||
            summary.execution_ready != 0) {
            yvex_weight_table_close(weights);
            *backend_status = YVEX_MATERIALIZE_BACKEND_FAIL;
            if (gate_summary) {
                gate_summary->materialization_gate = "fail";
                gate_summary->backend_status = "fail";
                if (getenv("YVEX_TEST_FAIL_MATERIALIZE_AFTER_TRANSFER")) {
                    gate_summary->materialization_phase = "transfer";
                    gate_summary->allocation_attempted = 1;
                    gate_summary->transfer_attempted = 1;
                    gate_summary->cleanup_attempted = 1;
                    gate_summary->cleanup_status = "pass";
                    gate_summary->bytes_allocated = gate_summary->bytes_planned;
                    gate_summary->bytes_transferred = gate_summary->bytes_planned;
                } else if (getenv("YVEX_TEST_FAIL_MATERIALIZE_AFTER_ALLOC")) {
                    gate_summary->materialization_phase = "allocation";
                    gate_summary->allocation_attempted = 1;
                    gate_summary->transfer_attempted = 0;
                    gate_summary->cleanup_attempted = 1;
                    gate_summary->cleanup_status = "pass";
                    gate_summary->bytes_allocated = gate_summary->bytes_planned;
                } else {
                    gate_summary->materialization_phase = "allocation";
                    gate_summary->cleanup_attempted = check_cleanup ? 1 : 0;
                    gate_summary->cleanup_status = check_cleanup ? "pass" : "not-needed";
                }
            }
            if (rc == YVEX_OK) {
                yvex_error_set(err, YVEX_ERR_STATE, owner,
                               "materialization did not reach weights-materialized");
                rc = YVEX_ERR_STATE;
            }
            if (failure_class) *failure_class = classify_materialize_failure(rc, err);
            yvex_backend_close(backend);
            return rc;
        }
        *bytes_materialized = summary.bytes_materialized;
        if (gate_summary) {
            gate_summary->allocation_attempted = gate_summary->allocation_attempted ||
                summary.allocation_attempted;
            gate_summary->transfer_attempted = gate_summary->transfer_attempted ||
                summary.transfer_attempted;
            gate_summary->bytes_planned = summary.bytes_planned;
            gate_summary->bytes_allocated = summary.bytes_allocated;
            gate_summary->bytes_transferred = summary.bytes_transferred;
        }
        yvex_weight_table_close(weights);

        if (check_cleanup && have_before) {
            if (yvex_backend_get_memory_stats(backend, &after_stats, err) != YVEX_OK ||
                after_stats.allocated_bytes != before_stats.allocated_bytes) {
                if (cleanup_verified) *cleanup_verified = 0;
                if (gate_summary) {
                    gate_summary->cleanup_attempted = 1;
                    gate_summary->cleanup_status = "fail";
                }
                *backend_status = YVEX_MATERIALIZE_BACKEND_FAIL;
                if (failure_class) *failure_class = YVEX_MATERIALIZE_FAILURE_BACKEND_ALLOC;
                yvex_error_set(err, YVEX_ERR_STATE, owner,
                               "backend allocated bytes did not return to baseline after close");
                yvex_backend_close(backend);
                return YVEX_ERR_STATE;
            }
        } else if (check_cleanup && cleanup_verified) {
            *cleanup_verified = 0;
        }
    }

    *backend_status = YVEX_MATERIALIZE_BACKEND_PASS;
    if (gate_summary) {
        gate_summary->materialization_gate = "pass";
        gate_summary->materialization_phase = "complete";
        gate_summary->cleanup_attempted = check_cleanup ? 1 : 0;
        gate_summary->cleanup_status = check_cleanup ? "pass" : "not-needed";
        gate_summary->backend_status = "ready";
    }
    yvex_backend_close(backend);
    return YVEX_OK;
}

static int materialize_gate_expected_tensors(
    const yvex_materialize_gate_options *options,
    const yvex_tensor_table *tensors,
    yvex_materialize_gate_summary *summary,
    yvex_error *err)
{
    gate_expected_tensors_count(
        options->expected_tensors, sizeof(options->expected_tensors[0]),
        options->expected_tensor_count, tensors, &summary->expected_tensor_matches,
        &summary->expected_tensor_mismatches);
    if (summary->expected_tensor_mismatches == 0)
        return YVEX_OK;
    summary->status = YVEX_MATERIALIZE_GATE_FAIL;
    summary->failure_class = YVEX_MATERIALIZE_FAILURE_TENSOR_SPEC_MISMATCH;
    summary->shape_status = "fail";
    yvex_error_set(err, YVEX_ERR_STATE, "yvex_materialize_gate_check",
                   "expected tensor specification mismatch");
    return YVEX_ERR_STATE;
}

static int materialize_gate_backend(
    const yvex_artifact *artifact,
    const yvex_gguf *gguf,
    const yvex_tensor_table *tensors,
    const yvex_materialize_gate_options *options,
    yvex_backend_kind kind,
    const char *name,
    int enabled,
    int required,
    yvex_materialize_gate_summary *summary,
    yvex_materialize_backend_status *backend_status,
    unsigned long long *bytes,
    int *cleanup,
    yvex_error *err)
{
    int status;

    if (!enabled)
        return YVEX_OK;
    status = materialize_repeated(
        artifact, gguf, tensors, kind, name, "yvex_materialize_gate_check",
        summary->repeat_count, 1, options->check_cleanup, summary, backend_status,
        bytes, cleanup, &summary->failure_class, err);
    if (status == YVEX_OK || !required)
        return YVEX_OK;
    summary->status = *backend_status == YVEX_MATERIALIZE_BACKEND_UNAVAILABLE
        ? YVEX_MATERIALIZE_GATE_BLOCKED : YVEX_MATERIALIZE_GATE_FAIL;
    if (*backend_status == YVEX_MATERIALIZE_BACKEND_UNAVAILABLE)
        summary->failure_class = YVEX_MATERIALIZE_FAILURE_BACKEND_UNAVAILABLE;
    return status;
}

static int materialize_gate_cleanup_status(
    const yvex_materialize_gate_options *options, int cpu, int cuda)
{
    if (!options->check_cleanup)
        return 1;
    if (options->check_cpu && options->check_cuda)
        return cpu && cuda;
    if (options->check_cpu)
        return cpu;
    if (options->check_cuda)
        return cuda;
    return 0;
}

int yvex_materialize_gate_check(const yvex_materialize_gate_options *options,
                                yvex_materialize_gate_summary *summary_out,
                                yvex_error *err)
{
    yvex_materialize_gate_summary summary;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_artifact_integrity_report integrity_report;
    char actual_sha[65] = {0};
    gate_input_stage input_stage;
    int cleanup_cpu = 0;
    int cleanup_cuda = 0;
    int rc;

    if (!options || !summary_out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_materialize_gate_check",
                       "options and summary_out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = gate_options_validate(
        "yvex_materialize_gate_check", options->model_path,
        options->expected_tensor_count, options->expected_tensors, err);
    if (rc != YVEX_OK) return rc;

    summary = materialize_gate_initial;
    summary.scope = options->scope;
    summary.label = options->label;
    summary.family = options->family;
    summary.model_path = options->model_path;
    summary.expected_sha256 = options->sha256 && options->sha256[0] ? options->sha256 : "";
    summary.digest_status = options->sha256 && options->sha256[0] ? "unchecked" : "unrequested";
    summary.identity_status = options->sha256 && options->sha256[0] ? "unchecked" : "unrequested";
    summary.metadata_status = options->metadata_status && options->metadata_status[0]
                                  ? options->metadata_status
                                  : "unregistered";
    summary.repeat_count = options->repeat_count ? options->repeat_count : 1u;
    summary.cleanup_verified = options->check_cleanup ? 0 : 1;
    *summary_out = summary;

    rc = gate_inputs_open(
        options->model_path, options->sha256, &artifact, &gguf, &tensors,
        actual_sha, &input_stage, err);
    if (artifact) summary.file_bytes = yvex_artifact_size(artifact);
    yvex_core_text_copy(summary.actual_sha256, sizeof(summary.actual_sha256),
                        actual_sha);
    if (rc != YVEX_OK) {
        summary.status = input_stage == GATE_INPUT_GGUF
                             ? YVEX_MATERIALIZE_GATE_FAIL
                             : YVEX_MATERIALIZE_GATE_BLOCKED;
        summary.failure_class = input_stage == GATE_INPUT_OPEN
                                    ? YVEX_MATERIALIZE_FAILURE_MISSING_FILE
                                : input_stage == GATE_INPUT_GGUF
                                    ? YVEX_MATERIALIZE_FAILURE_GGUF_PARSE
                                    : YVEX_MATERIALIZE_FAILURE_HASH_MISMATCH;
        summary.integrity_status = "fail";
        if (input_stage == GATE_INPUT_DIGEST ||
            input_stage == GATE_INPUT_DIGEST_MISMATCH) {
            summary.digest_status = "fail";
            summary.identity_status = "fail";
            yvex_error_setf(err, YVEX_ERR_STATE, "yvex_materialize_gate_check",
                            "sha256 mismatch: expected %s got %s", options->sha256,
                            input_stage == GATE_INPUT_DIGEST ? "unavailable" : actual_sha);
            rc = YVEX_ERR_STATE;
        }
        goto done;
    }
    if (options->sha256 && options->sha256[0]) {
        summary.digest_status = "pass";
        summary.identity_status = "pass";
    }
    summary.tensor_count = yvex_tensor_table_count(tensors);

    memset(&integrity_report, 0, sizeof(integrity_report));
    rc = yvex_artifact_integrity_validate(artifact, gguf, tensors, NULL, &integrity_report, err);
    summary.integrity_status = (rc == YVEX_OK && integrity_report.passed) ? "pass" : "fail";
    summary.shape_status =
        integrity_report.tensor_shapes_invalid == 0 &&
        integrity_report.tensor_dtypes_invalid == 0 &&
        integrity_report.tensor_byte_counts_invalid == 0 ? "pass" : "fail";
    summary.range_status = integrity_report.tensor_ranges_invalid == 0 ? "pass" : "fail";
    summary.bytes_planned = integrity_report.known_tensor_bytes;
    if (rc != YVEX_OK || !integrity_report.passed) {
        summary.status = YVEX_MATERIALIZE_GATE_FAIL;
        summary.failure_class = YVEX_MATERIALIZE_FAILURE_GGUF_PARSE;
        if (rc == YVEX_OK) {
            yvex_error_set(err, YVEX_ERR_STATE, "yvex_materialize_gate_check",
                           "artifact integrity preflight failed");
        }
        rc = rc == YVEX_OK ? YVEX_ERR_STATE : rc;
        goto done;
    }

    rc = materialize_gate_expected_tensors(options, tensors, &summary, err);
    if (rc != YVEX_OK) goto done;

    rc = materialize_gate_backend(
        artifact, gguf, tensors, options, YVEX_BACKEND_KIND_CPU, "cpu",
        options->check_cpu, options->require_cpu, &summary,
        &summary.cpu_status, &summary.bytes_materialized_cpu,
        &cleanup_cpu, err);
    if (rc == YVEX_OK)
        rc = materialize_gate_backend(
            artifact, gguf, tensors, options, YVEX_BACKEND_KIND_CUDA, "cuda",
            options->check_cuda, options->require_cuda, &summary,
            &summary.cuda_status, &summary.bytes_materialized_cuda,
            &cleanup_cuda, err);
    if (rc != YVEX_OK) goto done;

    if ((options->require_cpu && summary.cpu_status != YVEX_MATERIALIZE_BACKEND_PASS) ||
        (options->require_cuda && summary.cuda_status != YVEX_MATERIALIZE_BACKEND_PASS)) {
        summary.status = YVEX_MATERIALIZE_GATE_BLOCKED;
        summary.failure_class = YVEX_MATERIALIZE_FAILURE_BACKEND_UNAVAILABLE;
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_materialize_gate_check",
                       "required backend did not pass");
    } else if ((options->check_cpu && summary.cpu_status == YVEX_MATERIALIZE_BACKEND_FAIL) ||
               (options->check_cuda && summary.cuda_status == YVEX_MATERIALIZE_BACKEND_FAIL)) {
        summary.status = YVEX_MATERIALIZE_GATE_PARTIAL;
        if (summary.failure_class == YVEX_MATERIALIZE_FAILURE_NONE) {
            summary.failure_class = YVEX_MATERIALIZE_FAILURE_UNKNOWN;
        }
    } else {
        summary.status = YVEX_MATERIALIZE_GATE_PASS;
        summary.failure_class = YVEX_MATERIALIZE_FAILURE_NONE;
    }

    summary.cleanup_verified = materialize_gate_cleanup_status(options, cleanup_cpu, cleanup_cuda);
    summary.execution_ready = 0;
    rc = summary.status == YVEX_MATERIALIZE_GATE_PASS ? YVEX_OK : YVEX_ERR_STATE;

done:
    *summary_out = summary;
    gate_inputs_close(tensors, gguf, artifact);
    return rc;
}

const char *yvex_materialize_gate_status_name(yvex_materialize_gate_status status)
{
    return gate_name_find(materialize_gate_names,
                          sizeof(materialize_gate_names) / sizeof(materialize_gate_names[0]), status,
                          "materialize-gate-unknown");
}

const char *yvex_materialize_scope_name(yvex_materialize_scope scope)
{
    return gate_name_find(materialize_scope_names,
                          sizeof(materialize_scope_names) / sizeof(materialize_scope_names[0]), scope,
                          "unknown");
}

const char *yvex_materialize_backend_status_name(yvex_materialize_backend_status status)
{
    return gate_name_find(materialize_backend_names,
                          sizeof(materialize_backend_names) /
                              sizeof(materialize_backend_names[0]), status,
                          "not-tested");
}

const char *yvex_materialize_failure_class_name(yvex_materialize_failure_class failure)
{
    return gate_name_find(materialize_failure_names,
                          sizeof(materialize_failure_names) /
                              sizeof(materialize_failure_names[0]), failure,
                          "unknown");
}

typedef struct {
    const yvex_model_family_api *model;
    const yvex_graph_compiler_api *graph;
    yvex_deepseek_payload_handoff *handoff;
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_complete_artifact_admission admission;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_deepseek_v4_ir *architecture;
    yvex_runtime_descriptor *descriptor;
    yvex_attention_plan *attention;
    yvex_attention_plan *draft_attention;
    yvex_quant_policy *quant_policy;
    yvex_imatrix_data *imatrix;
    yvex_quant_plan *quant;
    yvex_gguf_writer_plan *writer;
    yvex_artifact_physical_compatibility compatibility;
    yvex_artifact_compatibility_failure compatibility_failure;
    yvex_deepseek_payload_handoff_options payload_options;
    yvex_deepseek_payload_failure payload_failure;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_projection materialization_projection;
    yvex_materialization_failure materialization_failure;
    yvex_runtime_descriptor_failure descriptor_failure;
    yvex_deepseek_v4_ir_failure architecture_failure;
    yvex_attention_failure attention_failure;
    yvex_quant_failure quant_failure;
    yvex_gguf_writer_failure writer_failure;
} runtime_binding_compiler;

static void runtime_binding_compiler_close(runtime_binding_compiler *compiler)
{
    if (!compiler) return;
    yvex_gguf_writer_plan_release(&compiler->writer);
    yvex_quant_plan_release(&compiler->quant);
    yvex_imatrix_data_close(compiler->imatrix);
    yvex_quant_policy_close(compiler->quant_policy);
    yvex_attention_plan_close(compiler->attention);
    yvex_attention_plan_close(compiler->draft_attention);
    yvex_runtime_descriptor_close(compiler->descriptor);
    if (compiler->model) compiler->model->ir.close(compiler->architecture);
    yvex_materialization_session_close(compiler->materialization);
    yvex_materialization_plan_close(compiler->materialization_plan);
    yvex_tensor_table_close(compiler->tensors);
    yvex_gguf_close(compiler->gguf);
    yvex_artifact_close(compiler->artifact);
    if (compiler->model) compiler->model->payload.close(compiler->handoff);
    memset(compiler, 0, sizeof(*compiler));
}

static int runtime_binding_compiler_open(
    runtime_binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    yvex_artifact_options options = {0};
    int rc;

    compiler->payload_options.source_path = request->source_path;
    compiler->payload_options.models_root = request->models_root;
    compiler->payload_options.manifest_path = request->source_manifest_path;
    yvex_source_payload_budget_default(&compiler->payload_options.budget);
    compiler->payload_options.budget.maximum_open_handles = 32u;
    compiler->payload_options.budget.maximum_streams = 16u;
    compiler->payload_options.budget.maximum_inflight_host_bytes =
        compiler->payload_options.budget.chunk_bytes *
        compiler->payload_options.budget.maximum_streams;
    compiler->payload_options.chunk_bytes = compiler->payload_options.budget.chunk_bytes;
    compiler->payload_options.page_bytes = compiler->payload_options.budget.page_bytes;
    rc = compiler->model->payload.open(&compiler->handoff, &compiler->payload_options,
                                       &compiler->payload_failure, err);
    options.path = request->artifact_path;
    options.readonly = 1;
    if (rc == YVEX_OK) rc = yvex_artifact_open(&compiler->artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&compiler->gguf, compiler->artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&compiler->tensors, compiler->gguf, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admit_deepseek(
            compiler->artifact, &compiler->admission, &compiler->admission_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(
            compiler->artifact, &compiler->admission, NULL, NULL,
            &compiler->admission_failure, err);
    return rc;
}

static int runtime_binding_compiler_plan(runtime_binding_compiler *compiler,
                                         const yvex_compilation_runtime_binding_request *request,
                                         yvex_error *err)
{
    const yvex_transform_ir_summary *transform = yvex_transform_ir_summary_get(
        compiler->model->payload.transform_ir(compiler->handoff));
    yvex_imatrix_data_options imatrix_options = {0};
    yvex_imatrix_data_summary imatrix_summary = {0};
    yvex_gguf_writer_plan_options writer_options;
    yvex_gguf_writer_plan_request writer_request;
    int rc;

    yvex_materialization_options_default(&compiler->materialization_options);
    compiler->materialization_options.require_terminal_projection = 1;
    compiler->materialization_options.max_chunk_bytes = 16ull * 1024ull * 1024ull;
    compiler->materialization_options.cache_budget_bytes = 256ull * 1024ull * 1024ull;
    compiler->materialization_options.future_graph_scratch_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    compiler->materialization_options.future_kv_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    rc = yvex_deepseek_materialization_projection(
        compiler->model->payload.map(compiler->handoff),
        &compiler->materialization_projection, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &compiler->materialization_plan, &compiler->admission, compiler->artifact,
            compiler->gguf, compiler->tensors, &compiler->materialization_projection,
            &compiler->materialization_options, &compiler->materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &compiler->materialization, compiler->materialization_plan, compiler->artifact,
            &compiler->materialization_options, &compiler->materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(
            compiler->materialization, &compiler->materialization_failure, err);
    if (rc == YVEX_OK)
        rc = compiler->model->ir.build(
            &compiler->architecture,
            compiler->model->payload.verification(compiler->handoff),
            &compiler->architecture_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_descriptor_build_deepseek(
            &compiler->descriptor, &compiler->admission, compiler->materialization,
            compiler->model->payload.map(compiler->handoff), compiler->architecture,
            &compiler->descriptor_failure, err);
    if (rc == YVEX_OK)
        rc = compiler->graph->plan_build(
            &compiler->attention, compiler->architecture, compiler->materialization,
            compiler->descriptor, &compiler->attention_failure, err);
    if (rc == YVEX_OK && compiler->graph->draft_plan_build)
        rc = compiler->graph->draft_plan_build(
            &compiler->draft_attention, compiler->architecture,
            compiler->materialization, compiler->descriptor,
            &compiler->attention_failure, err);
    if (rc == YVEX_OK && request->physical_variant_plan_path) {
        if (!transform) {
            yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                           "variant preparation requires the sealed transform identity");
            rc = YVEX_ERR_STATE;
        } else if ((request->quant_policy_path != NULL) ==
            (request->quant_preset_name != NULL)) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                           "variant preparation requires exactly one quant policy or preset");
            rc = YVEX_ERR_INVALID_ARG;
        } else if (request->quant_policy_path) {
            rc = yvex_quant_policy_open(&compiler->quant_policy,
                                        request->quant_policy_path, err);
        } else {
            rc = yvex_quant_policy_preset_open(&compiler->quant_policy,
                                               request->quant_preset_name, err);
        }
        if (rc == YVEX_OK && request->imatrix_path) {
            imatrix_options.path = request->imatrix_path;
            /* This request carries a path, not independent calibration provenance. Reconstruct
             * only the admitted predecessor prior used by planning and emission; a fresh
             * calibration requires an explicit provenance contract rather than inference from
             * the policy-selection mechanism. */
            imatrix_options.source_model_identity = YVEX_QUANT_DSPARK_IMATRIX_SOURCE_IDENTITY;
            imatrix_options.calibration_dataset_identity =
                YVEX_QUANT_DSPARK_IMATRIX_DATASET_IDENTITY;
            imatrix_options.producer = "llama.cpp-imatrix";
            imatrix_options.producer_version = 1u;
            imatrix_options.maximum_mapped_bytes = 1024u * 1024u * 1024u;
            rc = yvex_imatrix_data_open(&compiler->imatrix, &imatrix_options, err);
            if (rc == YVEX_OK)
                rc = yvex_imatrix_data_get_summary(compiler->imatrix,
                                                   &imatrix_summary, err);
        }
        if (rc == YVEX_OK)
            rc = yvex_quant_plan_build_deepseek_policy(
                &compiler->quant,
                compiler->model->payload.transform_ir(compiler->handoff),
                compiler->model->payload.binding(compiler->handoff),
                compiler->model->payload.map(compiler->handoff),
                compiler->quant_policy,
                imatrix_summary.complete ? imatrix_summary.imatrix_identity : NULL,
                NULL,
                &compiler->quant_failure, err);
        if (rc == YVEX_OK)
            rc = yvex_quant_plan_file_validate(request->physical_variant_plan_path,
                                               compiler->quant, err);
    } else if (rc == YVEX_OK &&
               (request->quant_policy_path || request->quant_preset_name ||
                request->imatrix_path)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                       "policy, preset, and imatrix require a sealed physical variant plan");
        rc = YVEX_ERR_INVALID_ARG;
    } else if (rc == YVEX_OK) {
        rc = yvex_quant_plan_build_deepseek_profile(
            &compiler->quant, compiler->model->payload.transform_ir(compiler->handoff),
            compiler->model->payload.binding(compiler->handoff),
            compiler->model->payload.map(compiler->handoff),
            YVEX_QUANT_PROFILE_RELEASE_Q8_Q2, NULL, &compiler->quant_failure, err);
    }
    if (rc != YVEX_OK) return rc;
    yvex_gguf_writer_plan_options_default(&writer_options);
    writer_options.required_execution_identity =
        request->physical_variant_plan_path ? NULL : compiler->admission.quant_execution_identity;
    memset(&writer_request, 0, sizeof(writer_request));
    writer_request.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    writer_request.quant_plan = compiler->quant;
    writer_request.options = &writer_options;
    writer_request.input.complete.family_adapter = compiler->model;
    writer_request.input.complete.lowering =
        compiler->model->payload.map(compiler->handoff);
    writer_request.input.complete.verification =
        compiler->model->payload.verification(compiler->handoff);
    return yvex_gguf_writer_plan_build(
        &compiler->writer, &writer_request, &compiler->writer_failure, err);
}

/*
 * Publish the admitted DeepSeek runtime binding through its family preparation adapter.
 *
 * Resolved compiler-plane paths, exact adapter identity, and caller-owned output.
 */
static int prepare_deepseek_runtime_binding(
    const yvex_compilation_runtime_binding_request *request,
    yvex_compilation_runtime_binding_result *result, yvex_error *err)
{
    runtime_binding_compiler compiler = {0};
    yvex_runtime_binding_prepare_request prepare = {0};
    yvex_runtime_binding_prepare_result prepared = {0};
    yvex_runtime_binding_failure failure = {0};
    const yvex_family_compiler_adapter *adapter = NULL;
    const yvex_gguf_writer_plan_summary *writer = NULL;
    const yvex_transform_ir_summary *transform = NULL;
    int rc;

    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !request->source_path || !request->models_root ||
        !request->source_manifest_path || !request->artifact_path ||
        !request->directory || !request->directory[0] ||
        !request->family_adapter_id || !request->family_adapter_version) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                       "source, artifact, binding directory, and adapter identity are required");
        return YVEX_ERR_INVALID_ARG;
    }
    compiler.model = yvex_model_register_deepseek_v4();
    adapter = yvex_compiler_family_deepseek_v4();
    compiler.graph = adapter && adapter->graph ? adapter->graph() : NULL;
    if (!compiler.model || !compiler.graph || !adapter ||
        adapter->schema_version != YVEX_FAMILY_COMPILER_SCHEMA_V1 ||
        adapter->adapter_id != request->family_adapter_id ||
        adapter->adapter_version != request->family_adapter_version ||
        !adapter->execution_capabilities || !adapter->transformer_policy ||
        !adapter->logits_policy || !adapter->speculation_policy ||
        !adapter->tokenizer_policy) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                       "family preparation and compiler adapter registration disagree");
        rc = YVEX_ERR_STATE;
    } else {
        rc = runtime_binding_compiler_open(&compiler, request, err);
    }
    if (rc == YVEX_OK) rc = runtime_binding_compiler_plan(&compiler, request, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_physical_compatibility_validate(
            compiler.writer, &compiler.admission, compiler.artifact, compiler.gguf,
            &compiler.compatibility, &compiler.compatibility_failure, err);
    if (rc == YVEX_OK) writer = yvex_gguf_writer_plan_summary_get(compiler.writer);
    if (rc == YVEX_OK)
        transform = yvex_transform_ir_summary_get(
            compiler.model->payload.transform_ir(compiler.handoff));
    if (rc == YVEX_OK && (!writer || !transform ||
                          !yvex_sha256_hex_is_valid(transform->transform_identity) ||
                          !compiler.compatibility.physical_payload_compatible)) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                       "logical transform and physical compatibility proof are required");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        prepare.directory = request->directory;
        prepare.admission = &compiler.admission;
        prepare.physical_compatibility = &compiler.compatibility;
        prepare.materialization = compiler.materialization;
        prepare.runtime_descriptor = compiler.descriptor;
        prepare.attention_plan = compiler.attention;
        prepare.draft_attention_plan = compiler.draft_attention;
        prepare.graph_compiler = compiler.graph;
        prepare.family_adapter_id = request->family_adapter_id;
        prepare.family_adapter_version = request->family_adapter_version;
        prepare.artifact_format = "gguf";
        prepare.artifact_format_version = writer->gguf_version;
        prepare.logical_transform_identity = transform->transform_identity;
        if (!adapter->execution_capabilities(&prepare.capabilities) ||
            !yvex_runtime_capabilities_contract_valid(&prepare.capabilities) ||
            !adapter->transformer_policy(
                yvex_runtime_descriptor_summary_get(compiler.descriptor),
                &prepare.transformer_policy) ||
            !adapter->logits_policy(&prepare.logits_policy) ||
            !adapter->speculation_policy(
                yvex_runtime_descriptor_summary_get(compiler.descriptor),
                &prepare.speculation_policy) ||
            !adapter->tokenizer_policy(&prepare.tokenizer_policy, err)) {
            yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                           "family execution envelope compilation failed");
            rc = YVEX_ERR_STATE;
        } else {
            rc = yvex_runtime_binding_prepare(&prepare, &prepared, &failure, err);
        }
    }
    if (rc == YVEX_OK) {
        memcpy(result->path, prepared.path, sizeof(result->path));
        result->published = prepared.published;
    }
    runtime_binding_compiler_close(&compiler);
    return rc;
}

/*
 * Enumerate compiler-plane family preparation facts and their one typed publication callback.
 *
 * Returns immutable process-lifetime storage.
 */
const yvex_graph_family_preparation *
yvex_graph_family_preparation_at(unsigned long long index)
{
    static const yvex_graph_family_preparation preparation = {
        YVEX_SOURCE_RELEASE_TARGET_ID, YVEX_SOURCE_RELEASE_MANIFEST_LEAF,
        yvex_model_register_deepseek_v4, yvex_artifact_admit_deepseek,
        prepare_deepseek_runtime_binding};
    return index == 0ull ? &preparation : NULL;
}
