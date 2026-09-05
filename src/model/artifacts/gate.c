/*
 * Admit immutable artifact and materialization evidence through typed gates.
 *
 * Gate checks return facts only and preserve public model/materialization gate API behavior. Gate
 * evidence is not artifact emission, generation readiness, benchmark evidence, or release
 * readiness; family-specific compiler preparation terminates in the family projection.
 */
#include <yvex/internal/model_artifact.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/backend.h>
#include <yvex/materialization.h>
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
