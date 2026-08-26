/*
 * This runner commits materialization for binding truth and compares CUDA output with a
 * separately linked scalar oracle over identical state. Live-plan and scoped CPU proof are graph
 * evidence only.
 */
#define _POSIX_C_SOURCE 200809L

#include <yvex/internal/artifact.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/compiler_source.h>
#include "src/graph/private.h"
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/model.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/runtime_operator.h>
#include "tests/reference/deepseek_attention.h"

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#define ATTENTION_CPU_ABSOLUTE_TOLERANCE 2.0e-4
#define ATTENTION_CPU_RELATIVE_TOLERANCE 2.0e-4
#define ATTENTION_CUDA_ABSOLUTE_TOLERANCE 5.0e-4
#define ATTENTION_CUDA_RELATIVE_TOLERANCE 5.0e-4
#define ATTENTION_NATIVE_Q8_ABSOLUTE_TOLERANCE 4.0e-2
#define ATTENTION_NATIVE_Q8_RELATIVE_TOLERANCE 4.0e-2
#define ATTENTION_NATIVE_TENSORCORE_ROWS 16ull

static const char *attention_class_name(yvex_attention_class class_id)
{
    static const char *const names[] = {"swa", "csa", "hca"};

    return class_id >= YVEX_ATTENTION_CLASS_SWA && class_id <= YVEX_ATTENTION_CLASS_HCA
               ? names[class_id]
               : "unknown";
}

static const yvex_graph_execution_api *attention_execution_api(void)
{
    const yvex_graph_execution_binding *binding =
        yvex_graph_execution_find(0ull, 0ull, "deepseek4-v4-flash-dspark");
    return binding && binding->api == &yvex_attention_execution_api ? binding->api : NULL;
}

static int semantic_model_borrow(
    yvex_semantic_model_ir **out, const yvex_source_verification *verification,
    yvex_error *err)
{
    const yvex_family_compiler_adapter *adapter = yvex_compiler_family_deepseek_v4();
    const yvex_family_binding_pipeline *pipeline =
        adapter ? adapter->binding_pipeline : NULL;

    if (!pipeline || !pipeline->semantic_model_build) return YVEX_ERR_STATE;
    return pipeline->semantic_model_build(out, verification, err);
}

typedef struct {
    yvex_sha256 hash;
    yvex_sha256 oracle_trace_hash;
    yvex_sha256 oracle_output_hash;
    yvex_sha256 fixture_hash;
    yvex_sha256 history_hash;
    yvex_sha256 comparison_contract_hash;
    unsigned long long layer_count;
    unsigned long long swa_count;
    unsigned long long csa_count;
    unsigned long long hca_count;
    unsigned long long qtype_paths;
    unsigned long long cpu_reference_values;
    unsigned long long cuda_reference_values;
    unsigned long long cpu_cuda_values;
    double cpu_squared_error;
    double cuda_squared_error;
    double cpu_cuda_squared_error;
    double cpu_maximum_absolute_error;
    double cpu_maximum_relative_error;
    double cuda_maximum_absolute_error;
    double cuda_maximum_relative_error;
    double cpu_cuda_maximum_absolute_error;
    double cpu_cuda_maximum_relative_error;
    unsigned long long exact_topk_positions;
    char oracle_trace_identity[YVEX_SHA256_HEX_CAP];
    char oracle_output_identity[YVEX_SHA256_HEX_CAP];
    char fixture_identity[YVEX_SHA256_HEX_CAP];
    char history_identity[YVEX_SHA256_HEX_CAP];
    char comparison_contract_identity[YVEX_SHA256_HEX_CAP];
} attention_live_evidence;

typedef struct {
    const yvex_attention_plan *plan;
    const yvex_deepseek_v4_ir *ir;
    yvex_materialization_session *materialization;
    const yvex_runtime_descriptor *descriptor;
    const yvex_attention_history_view *history;
    yvex_sha256 output_hash;
    yvex_sha256 state_hash;
    unsigned long long publications;
    unsigned long long compared_values;
    double squared_error;
    double maximum_absolute_error;
    double maximum_relative_error;
} runtime_oracle_evidence;

typedef struct {
    runtime_oracle_evidence oracle;
    yvex_backend_cuda_attention_graph_summary graph;
    char output_identity[YVEX_SHA256_HEX_CAP];
    char state_identity[YVEX_SHA256_HEX_CAP];
} runtime_oracle_mode_result;

typedef struct {
    unsigned int opens;
    unsigned int discards;
} runtime_oracle_state_factory_control;

static float runtime_oracle_bf16_round(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int runtime_oracle_state_factory_open(
    void *context, const yvex_attention_plan *plan,
    unsigned long long maximum_host_bytes,
    yvex_attention_state_provider *out, yvex_attention_failure *failure,
    yvex_error *err)
{
    runtime_oracle_state_factory_control *control =
        (runtime_oracle_state_factory_control *)context;
    int rc;

    if (!control) return YVEX_ERR_INVALID_ARG;
    rc = yvex_attention_state_provider_open_persistent(
        plan, maximum_host_bytes, out, failure, err);
    if (rc == YVEX_OK) control->opens++;
    return rc;
}

static int runtime_oracle_state_factory_discard(
    void *context, yvex_attention_state_provider *candidate, yvex_error *err)
{
    runtime_oracle_state_factory_control *control =
        (runtime_oracle_state_factory_control *)context;
    int rc;

    if (!control || !candidate)
        return YVEX_ERR_INVALID_ARG;
    control->discards++;
    if (!candidate->context) {
        memset(candidate, 0, sizeof(*candidate));
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!candidate->release) return YVEX_ERR_INVALID_ARG;
    rc = candidate->release(&candidate->context, err);
    if (rc == YVEX_OK) memset(candidate, 0, sizeof(*candidate));
    return rc;
}

static int attention_reference_contract_init(
    yvex_test_attention_reference_contract *contract,
    const yvex_attention_summary *summary,
    const char *path_name,
    double absolute_tolerance,
    double relative_tolerance)
{
    unsigned int stage;

    if (!contract || !summary || !path_name ||
        !isfinite(absolute_tolerance) || !isfinite(relative_tolerance) ||
        absolute_tolerance < 0.0 || relative_tolerance < 0.0)
        return 0;
    memset(contract, 0, sizeof(*contract));
    contract->schema_version = YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2;
    contract->path_name = path_name;
    /* Both paths consume the same decoded qtype values. One bound covers a
     * projection; composed transforms allow additional double-accumulation
     * ordering error before each required F32 publication. */
    for (stage = 0u; stage < YVEX_TEST_ATTENTION_STAGE_COUNT; ++stage) {
        double factor = 1.0;

        if (stage == YVEX_TEST_ATTENTION_STAGE_QUERY ||
            stage == YVEX_TEST_ATTENTION_STAGE_COMPRESSED_KV ||
            stage == YVEX_TEST_ATTENTION_STAGE_INDEXER_KV ||
            stage == YVEX_TEST_ATTENTION_STAGE_INDEX_QUERY ||
            stage == YVEX_TEST_ATTENTION_STAGE_INDEX_WEIGHTS ||
            stage == YVEX_TEST_ATTENTION_STAGE_MAIN_STATE ||
            stage == YVEX_TEST_ATTENTION_STAGE_INDEXER_STATE)
            factor = 2.0;
        else if (stage == YVEX_TEST_ATTENTION_STAGE_ATTENTION)
            factor = 4.0;
        else if (stage == YVEX_TEST_ATTENTION_STAGE_OUTPUT)
            factor = 6.0;
        contract->absolute[stage] = absolute_tolerance * factor;
        contract->relative[stage] = relative_tolerance * factor;
    }
    contract->absolute[YVEX_TEST_ATTENTION_STAGE_INPUT] = 0.0;
    contract->relative[YVEX_TEST_ATTENTION_STAGE_INPUT] = 0.0;
    memcpy(contract->qtype_binding_counts, summary->qtype_binding_counts,
           sizeof(contract->qtype_binding_counts));
    contract->compressed_positions_exact = 1;
    contract->indexer_positions_exact = 1;
    contract->topk_positions_exact = 1;
    return 1;
}

static double reference_metrics_rmse(
    const yvex_test_attention_reference_metrics *metrics)
{
    if (!metrics || !metrics->compared_values) return 0.0;
    return sqrt(metrics->squared_error_sum /
                (double)metrics->compared_values);
}

static int reference_metrics_merge(
    unsigned long long *values,
    double *squared_error,
    double *maximum_absolute_error,
    double *maximum_relative_error,
    const yvex_test_attention_reference_metrics *metrics)
{
    if (!values || !squared_error || !maximum_absolute_error ||
        !maximum_relative_error || !metrics)
        return 0;
    if (!yvex_core_u64_add(
            *values, metrics->compared_values, values) ||
        !isfinite(metrics->squared_error_sum) ||
        !isfinite(*squared_error + metrics->squared_error_sum))
        return 0;
    *squared_error += metrics->squared_error_sum;
    if (metrics->maximum_absolute_error > *maximum_absolute_error)
        *maximum_absolute_error = metrics->maximum_absolute_error;
    if (metrics->maximum_relative_error > *maximum_relative_error)
        *maximum_relative_error = metrics->maximum_relative_error;
    return 1;
}

static int runtime_oracle_compare_publication(
    void *context, yvex_backend_kind backend,
    const yvex_attention_publication *production, yvex_error *err)
{
    runtime_oracle_evidence *evidence = (runtime_oracle_evidence *)context;
    const yvex_attention_summary *summary;
    const yvex_attention_layer_plan *layer;
    yvex_attention_cpu_options options;
    yvex_attention_execution_trace reference;
    yvex_test_attention_reference_contract contract;
    yvex_test_attention_reference_metrics metrics;
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP];
    int rc = YVEX_ERR_FORMAT;

    memset(&reference, 0, sizeof(reference));
    memset(&metrics, 0, sizeof(metrics));
    if (!evidence || backend != YVEX_BACKEND_KIND_CUDA || !production ||
        !production->complete || !production->input || !production->output) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                       "one complete CUDA publication is required");
        return YVEX_ERR_STATE;
    }
    {
        unsigned long long seed = production->layer_index +
                                  production->token_position + 1009ull;
        unsigned long long code = (seed * 19ull + 11ull) % 257ull;
        float raw = (float)((long long)code - 128ll) / 257.0f;
        float rounded = runtime_oracle_bf16_round(raw);

        if (memcmp(&raw, &rounded, sizeof(raw)) == 0 ||
            memcmp(production->input, &rounded, sizeof(rounded)) != 0) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "attention.runtime_oracle.bf16",
                           "canonical V2 ingress did not publish independent BF16 RNE");
            return YVEX_ERR_FORMAT;
        }
    }
    summary = yvex_attention_plan_summary(evidence->plan);
    layer = yvex_attention_plan_layer_at(evidence->plan, production->layer_index);
    if (!summary || !layer || layer->layer_index != production->layer_index ||
        !attention_reference_contract_init(
            &contract, summary, "runtime-cuda-graph-reference",
            ATTENTION_CUDA_ABSOLUTE_TOLERANCE,
            ATTENTION_CUDA_RELATIVE_TOLERANCE)) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                       "runtime publication plan facts are inconsistent");
        return YVEX_ERR_STATE;
    }
    if (!evidence->history ||
        evidence->history->token_count != production->token_position) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle.history",
                       "independent reference lost the exact immutable prior state");
        return YVEX_ERR_STATE;
    }
    attention_execution_api()->cpu_options_default(&options);
    options.operation_scope = YVEX_ATTENTION_OPERATION_CORE;
    options.layer_index = production->layer_index;
    options.token_position = production->token_position;
    options.token_count = production->token_count;
    options.input = production->input;
    options.input_stride = layer->hidden_dimension;
    options.history = evidence->history;
    if (!yvex_test_attention_reference_execute(
            layer, evidence->ir, evidence->materialization,
            evidence->descriptor, &options, &reference, reason)) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "attention.runtime_oracle",
                        "independent runtime publication reference failed: %s",
                        reason);
        goto cleanup;
    }
    if (!yvex_test_attention_reference_compare_contract(
            production, &reference, &contract, &metrics)) {
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "attention.runtime_oracle",
            "runtime CUDA graph diverged at %s[%llu]: max_abs=%.17g max_rel=%.17g",
            metrics.first_failed_stage ? metrics.first_failed_stage : "unknown",
            metrics.first_failed_index, metrics.maximum_absolute_error,
            metrics.maximum_relative_error);
        goto cleanup;
    }
    if (!yvex_attention_publication_hash_update(
            &evidence->output_hash, &evidence->state_hash, &reference) ||
        !reference_metrics_merge(
            &evidence->compared_values, &evidence->squared_error,
            &evidence->maximum_absolute_error,
            &evidence->maximum_relative_error, &metrics) ||
        !yvex_core_u64_add(
            evidence->publications, 1ull, &evidence->publications)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "attention.runtime_oracle",
                       "runtime oracle evidence accounting overflowed");
        goto cleanup;
    }
    rc = YVEX_OK;

cleanup:
    yvex_attention_execution_trace_release(&reference);
    return rc;
}

static int attention_evidence_init(
    attention_live_evidence *evidence,
    const yvex_attention_summary *summary)
{
    unsigned long long binding_total = 0ull;
    unsigned int qtype;

    if (!evidence || !summary) return 0;
    memset(evidence, 0, sizeof(*evidence));
    yvex_sha256_init(&evidence->hash);
    yvex_sha256_init(&evidence->oracle_trace_hash);
    yvex_sha256_init(&evidence->oracle_output_hash);
    yvex_sha256_init(&evidence->fixture_hash);
    yvex_sha256_init(&evidence->history_hash);
    yvex_sha256_init(&evidence->comparison_contract_hash);
    if (!yvex_sha256_update_text(
            &evidence->hash, "yvex.attention.execution.evidence.v2") ||
        !yvex_sha256_update_text(
            &evidence->oracle_trace_hash,
            "yvex.attention.oracle.trace.aggregate.v2") ||
        !yvex_sha256_update_text(
            &evidence->oracle_output_hash,
            "yvex.attention.oracle.output.aggregate.v2") ||
        !yvex_sha256_update_text(
            &evidence->fixture_hash,
            "yvex.attention.fixture.aggregate.v2") ||
        !yvex_sha256_update_text(
            &evidence->history_hash,
            "yvex.attention.history.aggregate.v2") ||
        !yvex_sha256_update_text(
            &evidence->comparison_contract_hash,
            "yvex.attention.comparison.aggregate.v2") ||
        !yvex_sha256_update_text(
            &evidence->hash, summary->attention_plan_identity) ||
        !yvex_sha256_update_text(
            &evidence->hash, summary->runtime_descriptor_identity) ||
        !yvex_sha256_update_text(
            &evidence->hash, summary->runtime_numeric_identity) ||
        !yvex_sha256_update_u64(
            &evidence->hash, summary->required_binding_count) ||
        !yvex_sha256_update_u64(&evidence->hash, summary->layer_count))
        return 0;
    for (qtype = 0u; qtype < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++qtype) {
        unsigned long long count = summary->qtype_binding_counts[qtype];

        if (!count) continue;
        if (!yvex_core_u64_add(binding_total, count, &binding_total) ||
            !yvex_core_u64_add(
                evidence->qtype_paths, 1ull, &evidence->qtype_paths))
            return 0;
        if (!yvex_sha256_update_u64(&evidence->hash, qtype) ||
            !yvex_sha256_update_u64(&evidence->hash, count))
            return 0;
    }
    return binding_total == summary->required_binding_count;
}

static int attention_evidence_add_case(
    attention_live_evidence *evidence,
    const char *case_name,
    const yvex_attention_cpu_result *cpu,
    const yvex_attention_cpu_result *cuda,
    const yvex_test_attention_reference_metrics *cpu_reference,
    const yvex_test_attention_reference_metrics *cuda_reference,
    const yvex_test_attention_reference_metrics *cpu_cuda,
    const yvex_test_attention_reference_evidence *cpu_oracle,
    const yvex_test_attention_reference_evidence *cuda_oracle)
{
    if (!evidence || !case_name || !cpu || !cuda || !cpu_reference ||
        !cuda_reference || !cpu_cuda || !cpu_oracle || !cuda_oracle ||
        cpu_oracle->schema_version != YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2 ||
        cuda_oracle->schema_version != YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2 ||
        cpu->layer_index != cuda->layer_index ||
        cpu->attention_class != cuda->attention_class ||
        strcmp(cpu_oracle->oracle_trace_identity,
               cuda_oracle->oracle_trace_identity) != 0 ||
        strcmp(cpu_oracle->oracle_output_identity,
               cuda_oracle->oracle_output_identity) != 0 ||
        strcmp(cpu_oracle->fixture_identity,
               cuda_oracle->fixture_identity) != 0 ||
        strcmp(cpu_oracle->history_identity,
               cuda_oracle->history_identity) != 0 ||
        cpu_oracle->exact_topk_positions !=
            cuda_oracle->exact_topk_positions)
        return 0;
    if (!yvex_core_u64_add(
            evidence->exact_topk_positions,
            cpu_oracle->exact_topk_positions,
            &evidence->exact_topk_positions) ||
        !reference_metrics_merge(
        &evidence->cpu_reference_values, &evidence->cpu_squared_error,
        &evidence->cpu_maximum_absolute_error,
        &evidence->cpu_maximum_relative_error, cpu_reference) ||
        !reference_metrics_merge(
        &evidence->cuda_reference_values, &evidence->cuda_squared_error,
        &evidence->cuda_maximum_absolute_error,
        &evidence->cuda_maximum_relative_error, cuda_reference) ||
        !reference_metrics_merge(
        &evidence->cpu_cuda_values, &evidence->cpu_cuda_squared_error,
        &evidence->cpu_cuda_maximum_absolute_error,
        &evidence->cpu_cuda_maximum_relative_error, cpu_cuda))
        return 0;
    return yvex_sha256_update_text(&evidence->hash, case_name) &&
           yvex_sha256_update_u64(&evidence->hash, cpu->layer_index) &&
           yvex_sha256_update_u64(
               &evidence->hash, (unsigned long long)cpu->attention_class) &&
           yvex_sha256_update_text(&evidence->hash, cpu->output_identity) &&
           yvex_sha256_update_text(&evidence->hash, cuda->output_identity) &&
           yvex_sha256_update_text(
               &evidence->hash, cpu_oracle->oracle_trace_identity) &&
           yvex_sha256_update_text(
               &evidence->hash, cpu_oracle->oracle_output_identity) &&
           yvex_sha256_update_text(
               &evidence->hash, cpu_oracle->fixture_identity) &&
           yvex_sha256_update_text(
               &evidence->hash, cpu_oracle->history_identity) &&
           yvex_sha256_update_text(
               &evidence->hash,
               cpu_oracle->comparison_contract_identity) &&
           yvex_sha256_update_text(
               &evidence->hash,
               cuda_oracle->comparison_contract_identity) &&
           yvex_sha256_update_u64(
               &evidence->hash, cpu_oracle->exact_topk_positions) &&
           yvex_sha256_update_u64(
               &evidence->hash, cpu_reference->compared_values) &&
           yvex_sha256_update_u64(
               &evidence->hash, cuda_reference->compared_values) &&
           yvex_sha256_update_u64(
               &evidence->hash, cpu_cuda->compared_values) &&
           yvex_sha256_update_text(
               &evidence->oracle_trace_hash, case_name) &&
           yvex_sha256_update_text(
               &evidence->oracle_trace_hash,
               cpu_oracle->oracle_trace_identity) &&
           yvex_sha256_update_text(
               &evidence->oracle_output_hash, case_name) &&
           yvex_sha256_update_text(
               &evidence->oracle_output_hash,
               cpu_oracle->oracle_output_identity) &&
           yvex_sha256_update_text(&evidence->fixture_hash, case_name) &&
           yvex_sha256_update_text(
               &evidence->fixture_hash, cpu_oracle->fixture_identity) &&
           yvex_sha256_update_text(&evidence->history_hash, case_name) &&
           yvex_sha256_update_text(
               &evidence->history_hash, cpu_oracle->history_identity) &&
           yvex_sha256_update_text(
               &evidence->comparison_contract_hash, case_name) &&
           yvex_sha256_update_text(
               &evidence->comparison_contract_hash,
               cpu_oracle->comparison_contract_identity) &&
           yvex_sha256_update_text(
               &evidence->comparison_contract_hash,
               cuda_oracle->comparison_contract_identity);
}

static int attention_evidence_add(
    attention_live_evidence *evidence,
    const yvex_attention_cpu_result *cpu,
    const yvex_attention_cpu_result *cuda,
    const yvex_test_attention_reference_metrics *cpu_reference,
    const yvex_test_attention_reference_metrics *cuda_reference,
    const yvex_test_attention_reference_metrics *cpu_cuda,
    const yvex_test_attention_reference_evidence *cpu_oracle,
    const yvex_test_attention_reference_evidence *cuda_oracle)
{
    if (!evidence || !cpu) return 0;
    if (!yvex_core_u64_add(
            evidence->layer_count, 1ull, &evidence->layer_count))
        return 0;
    if (cpu->attention_class == YVEX_ATTENTION_CLASS_SWA) {
        if (!yvex_core_u64_add(
                evidence->swa_count, 1ull, &evidence->swa_count))
            return 0;
    } else if (cpu->attention_class == YVEX_ATTENTION_CLASS_CSA) {
        if (!yvex_core_u64_add(
                evidence->csa_count, 1ull, &evidence->csa_count))
            return 0;
    } else if (cpu->attention_class == YVEX_ATTENTION_CLASS_HCA) {
        if (!yvex_core_u64_add(
                evidence->hca_count, 1ull, &evidence->hca_count))
            return 0;
    }
    else
        return 0;
    return attention_evidence_add_case(
        evidence, "descriptor-layer", cpu, cuda, cpu_reference,
        cuda_reference, cpu_cuda, cpu_oracle, cuda_oracle);
}

static int attention_evidence_final(
    attention_live_evidence *evidence,
    char identity[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    if (!evidence || !identity ||
        !yvex_sha256_update_u64(&evidence->hash, evidence->layer_count) ||
        !yvex_sha256_update_u64(&evidence->hash, evidence->swa_count) ||
        !yvex_sha256_update_u64(&evidence->hash, evidence->csa_count) ||
        !yvex_sha256_update_u64(&evidence->hash, evidence->hca_count) ||
        !yvex_sha256_update_u64(
            &evidence->hash, evidence->exact_topk_positions) ||
        !yvex_sha256_final(&evidence->hash, digest))
        return 0;
    yvex_sha256_hex(digest, identity);
    if (!yvex_sha256_final(&evidence->oracle_trace_hash, digest)) return 0;
    yvex_sha256_hex(digest, evidence->oracle_trace_identity);
    if (!yvex_sha256_final(&evidence->oracle_output_hash, digest)) return 0;
    yvex_sha256_hex(digest, evidence->oracle_output_identity);
    if (!yvex_sha256_final(&evidence->fixture_hash, digest)) return 0;
    yvex_sha256_hex(digest, evidence->fixture_identity);
    if (!yvex_sha256_final(&evidence->history_hash, digest)) return 0;
    yvex_sha256_hex(digest, evidence->history_identity);
    if (!yvex_sha256_final(&evidence->comparison_contract_hash, digest))
        return 0;
    yvex_sha256_hex(digest, evidence->comparison_contract_identity);
    return 1;
}

static int path_join_selected(char *out, size_t out_size,
                              const char *models_root)
{
    int written;

    if (!out || !out_size || !models_root) return 0;
    written = snprintf(out, out_size, "%s/deepseek/%s", models_root,
                       YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME);
    return written >= 0 && (size_t)written < out_size;
}

static void print_materialization_failure(
    const char *phase,
    const yvex_materialization_failure *failure,
    const yvex_error *err)
{
    fprintf(stderr,
            "%s_failure=%s tensor=%llu name=%s expected=%llu actual=%llu offset=%llu where=%s message=%s\n",
            phase, yvex_materialization_failure_name(failure->code),
            failure->tensor_index, failure->tensor_name, failure->expected,
            failure->actual, failure->offset, yvex_error_where(err),
            yvex_error_message(err));
}

static void print_descriptor_failure(
    const char *phase,
    const yvex_runtime_descriptor_failure *failure,
    const yvex_error *err)
{
    fprintf(stderr,
            "%s_failure=%s tensor=%llu name=%s expected=%llu actual=%llu where=%s message=%s\n",
            phase, yvex_runtime_descriptor_failure_name(failure->code),
            failure->tensor_index, failure->tensor_name, failure->expected,
            failure->actual, yvex_error_where(err), yvex_error_message(err));
}

static void print_attention_failure(
    const yvex_attention_failure *failure,
    const yvex_error *err)
{
    fprintf(stderr,
            "attention_failure=%s layer=%llu role=%u tensor=%s expected=%llu actual=%llu where=%s message=%s\n",
            yvex_test_attention_failure_name(failure->code),
            failure->layer_index, (unsigned int)failure->role,
            failure->tensor_name, failure->expected, failure->actual,
            yvex_error_where(err), yvex_error_message(err));
}

static void print_architecture_failure(
    const yvex_deepseek_v4_ir_failure *failure,
    const yvex_error *err)
{
    fprintf(stderr,
            "architecture_failure=%s component=%s field=%s layer=%llu expected=%llu actual=%llu where=%s message=%s\n",
            yvex_model_register_deepseek_v4()->ir.failure_name(failure->code),
            yvex_model_register_deepseek_v4()->ir.component_name(failure->component),
            failure->field ? failure->field : "", failure->layer_index,
            failure->expected, failure->actual, yvex_error_where(err),
            yvex_error_message(err));
}

static void fill_history_values(float *values,
                                unsigned long long count,
                                unsigned long long salt)
{
    unsigned long long i;

    for (i = 0ull; i < count; ++i) {
        int lane = (int)(((i + 5ull) * 23ull + salt * 41ull) % 193ull) -
                   96;
        values[i] = (float)lane / 96.0f;
    }
}

static int run_topk_contract_cases(void)
{
    float scores[] = {2.0f, 2.0f, 1.0f};
    unsigned long long positions[] = {9ull, 4ull, 7ull};
    unsigned long long duplicate_positions[] = {9ull, 9ull, 7ull};
    unsigned long long production[3] = {~0ull, ~0ull, ~0ull};
    unsigned long long reference[3] = {~0ull, ~0ull, ~0ull};
    unsigned long long production_count = ~0ull;
    unsigned long long reference_count = ~0ull;
    yvex_attention_failure failure;
    yvex_error err;
    int rc;

    memset(&failure, 0, sizeof(failure));
    yvex_error_clear(&err);
    rc = yvex_attention_topk_select(
        scores, positions, 3ull, 0ull, production, &production_count,
        NULL, &failure, &err);
    if (rc != YVEX_OK || production_count != 0ull ||
        !yvex_test_attention_reference_topk(
            scores, positions, 3ull, 0ull, reference, &reference_count) ||
        reference_count != 0ull)
        return 0;
    rc = yvex_attention_topk_select(
        scores, positions, 3ull, 4ull, production, &production_count,
        NULL, &failure, &err);
    if (rc != YVEX_OK || production_count != 3ull ||
        !yvex_test_attention_reference_topk(
            scores, positions, 3ull, 4ull, reference, &reference_count) ||
        reference_count != 3ull ||
        memcmp(production, reference, sizeof(production)) != 0 ||
        production[0] != 1ull || production[1] != 0ull ||
        production[2] != 2ull)
        return 0;
    scores[1] = INFINITY;
    rc = yvex_attention_topk_select(
        scores, positions, 3ull, 1ull, production, &production_count,
        NULL, &failure, &err);
    if (rc == YVEX_OK ||
        failure.code != YVEX_ATTENTION_FAILURE_NUMERIC ||
        yvex_test_attention_reference_topk(
            scores, positions, 3ull, 1ull, reference, &reference_count))
        return 0;
    scores[1] = 2.0f;
    memset(&failure, 0, sizeof(failure));
    yvex_error_clear(&err);
    rc = yvex_attention_topk_select(
        scores, duplicate_positions, 3ull, 1ull, production,
        &production_count, NULL, &failure, &err);
    if (rc == YVEX_OK ||
        failure.code != YVEX_ATTENTION_FAILURE_NUMERIC ||
        yvex_test_attention_reference_topk(
            scores, duplicate_positions, 3ull, 1ull, reference,
            &reference_count))
        return 0;
    printf("attention_topk_zero_exact=1\n");
    printf("attention_topk_full_count_exact=1\n");
    printf("attention_topk_nonfinite_refused=1\n");
    printf("attention_topk_duplicate_refused=1\n");
    return 1;
}

static int run_reference_compare(
    const yvex_attention_plan *plan,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options,
    yvex_attention_cpu_result *production_result,
    yvex_test_attention_reference_metrics *metrics,
    yvex_attention_execution_trace *preserved_production,
    yvex_test_attention_reference_evidence *oracle_evidence,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_attention_summary *summary;
    const yvex_attention_layer_plan *layer;
    yvex_test_attention_reference_contract contract;
    yvex_attention_cpu_options effective_options = *options;
    yvex_attention_cpu_options production_options;
    yvex_attention_execution_trace production;
    yvex_attention_execution_trace reference;
    float *owned_input = NULL;
    unsigned long long input_count = 0ull;
    unsigned long long token_count;
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP];
    int rc;

    memset(&production, 0, sizeof(production));
    memset(&reference, 0, sizeof(reference));
    memset(metrics, 0, sizeof(*metrics));
    summary = yvex_attention_plan_summary(plan);
    if (!attention_reference_contract_init(
            &contract, summary, "cpu-reference",
            ATTENTION_CPU_ABSOLUTE_TOLERANCE,
            ATTENTION_CPU_RELATIVE_TOLERANCE))
        return YVEX_ERR_STATE;
    layer = yvex_attention_plan_layer_at(
        plan, effective_options.layer_index);
    token_count = effective_options.token_count
        ? effective_options.token_count : 1ull;
    if (!effective_options.input) {
        if (!layer ||
            !yvex_core_u64_mul(token_count, layer->hidden_dimension,
                               &input_count) ||
            input_count > (unsigned long long)(SIZE_MAX / sizeof(*owned_input)))
            return YVEX_ERR_BOUNDS;
        owned_input = (float *)malloc((size_t)input_count * sizeof(*owned_input));
        if (!owned_input) return YVEX_ERR_NOMEM;
        fill_history_values(
            owned_input, input_count,
            1009ull + effective_options.layer_index * 131ull +
                effective_options.token_position);
        effective_options.input = owned_input;
        effective_options.input_stride = layer->hidden_dimension;
    }
    production_options = effective_options;
    production_options.publication = &production;
    production_options.trace = NULL;
    rc = attention_execution_api()->cpu_chunk_execute(
        plan, ir, session, descriptor, &production_options,
        production_result, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!production.owned || !production.complete ||
        (options->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL &&
         !production.input) ||
        !production.raw_kv || !production.output ||
        (effective_options.history &&
         effective_options.history->main_rolling_state.present &&
         !production.next_main_rolling_state.present)) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    if (!yvex_test_attention_reference_execute(
            layer, ir, session, descriptor, &effective_options, &reference,
            reason)) {
        fprintf(stderr, "attention_reference_failure=%s\n", reason);
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (!yvex_test_attention_reference_compare_contract(
            &production, &reference, &contract, metrics)) {
        fprintf(stderr,
                "attention_reference_mismatch layer=%llu class=%u stage=%s "
                "coordinate=%llu values=%llu max_abs=%.17g "
                "max_rel=%.17g rmse=%.17g\n",
                options->layer_index,
                (unsigned int)production.attention_class,
                metrics->first_failed_stage ? metrics->first_failed_stage : "",
                metrics->first_failed_index,
                metrics->compared_values, metrics->maximum_absolute_error,
                metrics->maximum_relative_error,
                reference_metrics_rmse(metrics));
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (oracle_evidence &&
        !yvex_test_attention_reference_evidence_build(
            &reference, effective_options.history,
            summary->attention_plan_identity,
            &contract, oracle_evidence)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (preserved_production) {
        if (preserved_production->owned) {
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        *preserved_production = production;
        memset(&production, 0, sizeof(production));
    }
    rc = YVEX_OK;

cleanup:
    yvex_attention_execution_trace_release(&production);
    yvex_attention_execution_trace_release(&reference);
    free(owned_input);
    return rc;
}

static int real_float_mutation_detected(
    yvex_attention_execution_trace *production,
    const yvex_attention_execution_trace *reference,
    const yvex_test_attention_reference_contract *contract,
    float *values,
    unsigned long long count)
{
    yvex_test_attention_reference_metrics metrics;
    float original;
    int detected;

    if (!production || !reference || !contract || !values || !count)
        return 0;
    original = values[0];
    values[0] = isfinite(original) ? original + 64.0f : 64.0f;
    detected = !yvex_test_attention_reference_compare_contract(
        production, reference, contract, &metrics);
    values[0] = original;
    return detected;
}

static int real_position_mutation_detected(
    yvex_attention_execution_trace *production,
    const yvex_attention_execution_trace *reference,
    const yvex_test_attention_reference_contract *contract,
    unsigned long long *positions,
    unsigned long long count)
{
    yvex_test_attention_reference_metrics metrics;
    unsigned long long original;
    int detected;

    if (!production || !reference || !contract || !positions || !count)
        return 0;
    original = positions[0];
    positions[0] = original == ULLONG_MAX ? 0ull : original + 1ull;
    detected = !yvex_test_attention_reference_compare_contract(
        production, reference, contract, &metrics);
    positions[0] = original;
    return detected;
}

static int run_real_mutation_proof(
    const yvex_attention_plan *plan,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options,
    unsigned long long *detected_count,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_attention_summary *summary;
    const yvex_attention_layer_plan *layer;
    yvex_test_attention_reference_contract contract;
    yvex_test_attention_reference_metrics metrics;
    yvex_attention_cpu_options production_options;
    yvex_attention_cpu_result result;
    yvex_attention_execution_trace production;
    yvex_attention_execution_trace reference;
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP];
    unsigned long long detected = 0ull;
    int rc;

    if (!options || !options->input || !detected_count) return YVEX_ERR_INVALID_ARG;
    memset(&production, 0, sizeof(production));
    memset(&reference, 0, sizeof(reference));
    production_options = *options;
    production_options.publication = &production;
    production_options.trace = NULL;
    summary = yvex_attention_plan_summary(plan);
    layer = yvex_attention_plan_layer_at(plan, options->layer_index);
    if (!layer || !attention_reference_contract_init(
            &contract, summary, "real-stage-mutation",
            ATTENTION_CPU_ABSOLUTE_TOLERANCE,
            ATTENTION_CPU_RELATIVE_TOLERANCE))
        return YVEX_ERR_STATE;
    rc = attention_execution_api()->cpu_chunk_execute(
        plan, ir, session, descriptor, &production_options, &result,
        failure, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!yvex_test_attention_reference_execute(
            layer, ir, session, descriptor, options, &reference, reason) ||
        !yvex_test_attention_reference_compare_contract(
            &production, &reference, &contract, &metrics)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
#define MUTATE_FLOAT(field, count) do {                                        \
    unsigned long long mutation_count_ = (unsigned long long)(count);           \
    if (mutation_count_ != 0ull && !real_float_mutation_detected(               \
            &production, &reference, &contract, production.field,               \
            mutation_count_)) {                                                 \
        rc = YVEX_ERR_FORMAT;                                                   \
        goto cleanup;                                                           \
    }                                                                           \
    if (mutation_count_ != 0ull) detected++;                                    \
} while (0)
    MUTATE_FLOAT(input, production.token_count * production.hidden_width);
    MUTATE_FLOAT(q_low, production.token_count * production.q_rank);
    MUTATE_FLOAT(query, production.token_count * production.query_width);
    MUTATE_FLOAT(raw_kv, production.token_count * production.kv_width);
    MUTATE_FLOAT(compressed_kv,
                 production.compressed_count * production.compressed_stride);
    MUTATE_FLOAT(indexer_kv,
                 production.indexer_count * production.indexer_stride);
    MUTATE_FLOAT(index_query,
                 production.token_count * production.index_query_stride);
    MUTATE_FLOAT(index_weights,
                 production.token_count * production.index_weight_stride);
    MUTATE_FLOAT(attention_values,
                 production.token_count * production.query_width);
    MUTATE_FLOAT(output, production.token_count * production.hidden_width);
    MUTATE_FLOAT(next_main_rolling_state.kv_state,
                 production.next_main_rolling_state.kv_state_extent);
    MUTATE_FLOAT(next_indexer_rolling_state.kv_state,
                 production.next_indexer_rolling_state.kv_state_extent);
#undef MUTATE_FLOAT
    if (production.compressed_count) {
        if (!real_position_mutation_detected(
                &production, &reference, &contract,
                production.compressed_positions,
                production.compressed_count)) {
            rc = YVEX_ERR_FORMAT;
            goto cleanup;
        }
        detected++;
    }
    if (production.topk_counts && production.topk_counts[0]) {
        if (!real_position_mutation_detected(
                &production, &reference, &contract,
                production.topk_positions, production.topk_counts[0])) {
            rc = YVEX_ERR_FORMAT;
            goto cleanup;
        }
        detected++;
    }
    *detected_count += detected;
    rc = YVEX_OK;
cleanup:
    yvex_attention_execution_trace_release(&production);
    yvex_attention_execution_trace_release(&reference);
    return rc;
}

static int run_cuda_reference_compare(
    const yvex_attention_plan *plan,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    yvex_backend *backend,
    const yvex_attention_cpu_options *options,
    yvex_attention_cpu_result *production_result,
    yvex_test_attention_reference_metrics *metrics,
    yvex_attention_execution_trace *preserved_production,
    yvex_test_attention_reference_evidence *oracle_evidence,
    double absolute_tolerance,
    double relative_tolerance,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_attention_summary *summary;
    const yvex_attention_layer_plan *layer;
    yvex_test_attention_reference_contract contract;
    yvex_attention_cpu_options production_options = *options;
    yvex_attention_execution_trace production;
    yvex_attention_execution_trace reference;
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP];
    int rc;

    memset(&production, 0, sizeof(production));
    memset(&reference, 0, sizeof(reference));
    memset(metrics, 0, sizeof(*metrics));
    summary = yvex_attention_plan_summary(plan);
    layer = yvex_attention_plan_layer_at(plan, options->layer_index);
    if (!layer || !attention_reference_contract_init(
            &contract, summary, "cuda-reference",
            absolute_tolerance, relative_tolerance))
        return YVEX_ERR_STATE;
    production_options.publication = &production;
    production_options.trace = NULL;
    rc = attention_execution_api()->cuda_token_execute(
        plan, ir, session, descriptor, backend, &production_options,
        production_result, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!production.owned || !production.complete ||
        (options->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL &&
         !production.input) ||
        !production.raw_kv || !production.output ||
        (options->history && options->history->main_rolling_state.present &&
         !production.next_main_rolling_state.present) ||
        !production_result->cuda_executed ||
        production_result->cuda_kernel_launches == 0ull) {
        fprintf(stderr,
                "attention_cuda_device_path_missing owned=%d complete=%d input=%d "
                "raw_kv=%d output=%d next_main=%d cuda=%d launches=%llu evidence=%u\n",
                production.owned, production.complete, production.input != NULL,
                production.raw_kv != NULL, production.output != NULL,
                production.next_main_rolling_state.present,
                production_result->cuda_executed,
                production_result->cuda_kernel_launches,
                (unsigned int)options->evidence_level);
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    if (!yvex_test_attention_reference_execute(
            layer, ir, session, descriptor, options, &reference, reason)) {
        fprintf(stderr, "attention_cuda_reference_failure=%s\n", reason);
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (!yvex_test_attention_reference_compare_contract(
            &production, &reference, &contract, metrics)) {
        fprintf(stderr,
                "attention_cuda_reference_mismatch layer=%llu class=%u "
                "stage=%s coordinate=%llu values=%llu max_abs=%.17g "
                "max_rel=%.17g rmse=%.17g\n",
                options->layer_index,
                (unsigned int)production.attention_class,
                metrics->first_failed_stage ? metrics->first_failed_stage : "",
                metrics->first_failed_index,
                metrics->compared_values, metrics->maximum_absolute_error,
                metrics->maximum_relative_error,
                reference_metrics_rmse(metrics));
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (oracle_evidence &&
        !yvex_test_attention_reference_evidence_build(
            &reference, options->history, summary->attention_plan_identity,
            &contract, oracle_evidence)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (preserved_production) {
        if (preserved_production->owned) {
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        *preserved_production = production;
        memset(&production, 0, sizeof(production));
    }
    rc = YVEX_OK;

cleanup:
    yvex_attention_execution_trace_release(&production);
    yvex_attention_execution_trace_release(&reference);
    return rc;
}

static int run_cpu_cuda_reference_compare(
    const yvex_attention_plan *plan,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    yvex_backend *backend,
    const yvex_attention_cpu_options *options,
    yvex_attention_cpu_result *cpu_result,
    yvex_attention_cpu_result *cuda_result,
    yvex_test_attention_reference_metrics *cpu_reference,
    yvex_test_attention_reference_metrics *cuda_reference,
    yvex_test_attention_reference_metrics *cpu_cuda,
    yvex_test_attention_reference_evidence *cpu_oracle,
    yvex_test_attention_reference_evidence *cuda_oracle,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_attention_summary *summary;
    yvex_test_attention_reference_contract contract;
    yvex_attention_execution_trace cpu_trace;
    yvex_attention_execution_trace cuda_trace;
    int rc;

    memset(&cpu_trace, 0, sizeof(cpu_trace));
    memset(&cuda_trace, 0, sizeof(cuda_trace));
    rc = run_reference_compare(
        plan, ir, session, descriptor, options, cpu_result, cpu_reference,
        &cpu_trace, cpu_oracle, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    rc = run_cuda_reference_compare(
        plan, ir, session, descriptor, backend, options, cuda_result,
        cuda_reference, &cuda_trace, cuda_oracle,
        ATTENTION_CUDA_ABSOLUTE_TOLERANCE,
        ATTENTION_CUDA_RELATIVE_TOLERANCE, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    summary = yvex_attention_plan_summary(plan);
    if (!attention_reference_contract_init(
            &contract, summary, "cpu-cuda",
            ATTENTION_CUDA_ABSOLUTE_TOLERANCE,
            ATTENTION_CUDA_RELATIVE_TOLERANCE) ||
        !yvex_test_attention_reference_compare_contract(
            &cpu_trace, &cuda_trace, &contract, cpu_cuda)) {
        fprintf(stderr,
                "attention_cpu_cuda_mismatch layer=%llu class=%u stage=%s "
                "coordinate=%llu values=%llu max_abs=%.17g "
                "max_rel=%.17g rmse=%.17g\n",
                options->layer_index,
                (unsigned int)cpu_trace.attention_class,
                cpu_cuda->first_failed_stage ?
                    cpu_cuda->first_failed_stage : "",
                cpu_cuda->first_failed_index, cpu_cuda->compared_values,
                cpu_cuda->maximum_absolute_error,
                cpu_cuda->maximum_relative_error,
                reference_metrics_rmse(cpu_cuda));
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    rc = YVEX_OK;

cleanup:
    yvex_attention_execution_trace_release(&cpu_trace);
    yvex_attention_execution_trace_release(&cuda_trace);
    return rc;
}

typedef struct {
    unsigned long long calls;
    unsigned long long cancel_on_call;
} attention_live_cancellation;

typedef struct {
    unsigned long long capacity;
    unsigned long long peak;
    unsigned long long allocation_count;
    unsigned long long cold_executions;
    unsigned long long warm_executions;
} attention_cuda_workspace_evidence;

static int attention_cuda_workspace_add(
    attention_cuda_workspace_evidence *evidence,
    const yvex_attention_cpu_result *result)
{
    if (!evidence || !result || !result->cuda_executed ||
        !result->cuda_host_workspace_capacity ||
        !result->cuda_host_workspace_used ||
        result->cuda_host_workspace_used > result->cuda_host_workspace_capacity ||
        result->cuda_host_workspace_peak < result->cuda_host_workspace_used ||
        result->cuda_host_workspace_allocations != 1ull)
        return 0;
    if (result->cuda_host_workspace_capacity > evidence->capacity)
        evidence->capacity = result->cuda_host_workspace_capacity;
    if (result->cuda_host_workspace_peak > evidence->peak)
        evidence->peak = result->cuda_host_workspace_peak;
    if (result->cuda_host_workspace_allocations > evidence->allocation_count)
        evidence->allocation_count = result->cuda_host_workspace_allocations;
    if (result->cuda_host_workspace_reused)
        evidence->warm_executions++;
    else
        evidence->cold_executions++;
    return 1;
}

static int attention_live_cancel_requested(void *context)
{
    attention_live_cancellation *cancellation =
        (attention_live_cancellation *)context;

    if (!cancellation) return 0;
    cancellation->calls++;
    return cancellation->calls >= cancellation->cancel_on_call;
}

#define ATTENTION_LIVE_MAX_STATE_POSITION 2056ull

static int attention_cuda_workspace_prepare(
    yvex_backend *backend, const yvex_attention_plan *plan, yvex_error *err)
{
    const yvex_attention_layer_plan *layer;
    yvex_attention_state_recipe_request state_request;
    yvex_attention_state_recipe state;
    yvex_attention_workspace_recipe workspace;
    yvex_attention_failure failure;
    unsigned long long count, index, required = 0ull, layer_bytes;
    int rc;

    if (!backend || !plan)
        return 0;
    count = yvex_attention_plan_layer_count(plan);
    for (index = 0ull; index < count; ++index) {
        layer = yvex_attention_plan_layer_at(plan, index);
        if (!layer) return 0;
        memset(&state_request, 0, sizeof(state_request));
        memset(&failure, 0, sizeof(failure));
        state_request.layer_ordinal = index;
        state_request.final_position = ATTENTION_LIVE_MAX_STATE_POSITION;
        state_request.attention_plan_identity =
            yvex_attention_plan_summary(plan)->attention_plan_identity;
        rc = yvex_attention_state_recipe_build(
            layer, &state_request, &state, &failure, err);
        if (rc == YVEX_OK)
            rc = yvex_attention_workspace_recipe_build(
                layer, &state, YVEX_ATTENTION_EXECUTION_EAGER,
                YVEX_ATTENTION_OPERATION_ENVELOPE,
                YVEX_ATTENTION_EVIDENCE_FULL, 4ull, &workspace,
                &failure, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_attention_workspace_required_from_recipe(
                &workspace, &layer_bytes, err);
        if (rc != YVEX_OK) return 0;
        if (layer_bytes > required) required = layer_bytes;
    }
    return required &&
           yvex_backend_host_workspace_prepare_owned(backend, required, err) ==
               YVEX_OK;
}

/* Contract: runs one failure seam on a fresh backend and proves no result or
 * trace becomes observable after rollback. */
static int run_cuda_fault_case(
    const yvex_attention_plan *plan,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *base_options,
    const char *environment,
    const char *value,
    unsigned long long cancel_on_call,
    yvex_attention_failure_code expected_failure,
    yvex_error *err)
{
    yvex_backend_options backend_options;
    yvex_backend *backend = NULL;
    yvex_attention_cpu_options options;
    yvex_attention_cpu_result result;
    yvex_attention_execution_trace trace;
    yvex_attention_failure failure;
    attention_live_cancellation cancel_state;
    yvex_attention_cancellation cancellation;
    int rc;
    int passed;

    memset(&backend_options, 0, sizeof(backend_options));
    memset(&result, 0, sizeof(result));
    memset(&trace, 0, sizeof(trace));
    memset(&failure, 0, sizeof(failure));
    memset(&cancel_state, 0, sizeof(cancel_state));
    memset(&cancellation, 0, sizeof(cancellation));
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &backend_options, err);
    if (rc != YVEX_OK) return 0;
    if (!attention_cuda_workspace_prepare(backend, plan, err)) {
        (void)yvex_backend_close_checked(&backend, err);
        return 0;
    }
    options = *base_options;
    options.trace = &trace;
    if (cancel_on_call) {
        cancel_state.cancel_on_call = cancel_on_call;
        cancellation.requested = attention_live_cancel_requested;
        cancellation.context = &cancel_state;
        options.cancellation = &cancellation;
    }
    if (environment && setenv(environment, value, 1) != 0) {
        (void)yvex_backend_close_checked(&backend, err);
        return 0;
    }
    rc = attention_execution_api()->cuda_token_execute(
        plan, ir, session, descriptor, backend, &options, &result, &failure,
        err);
    if (environment) (void)unsetenv(environment);
    passed = rc != YVEX_OK && !result.executed && !trace.owned &&
             failure.code == expected_failure;
    if (!passed)
        fprintf(stderr,
                "attention_cuda_fault_case_failed environment=%s value=%s "
                "cancel=%llu rc=%d executed=%d trace_owned=%d failure=%u "
                "expected=%u where=%s message=%s\n",
                environment ? environment : "", value ? value : "",
                cancel_on_call, rc, result.executed, trace.owned,
                (unsigned int)failure.code, (unsigned int)expected_failure,
                yvex_error_where(err), yvex_error_message(err));
    yvex_attention_execution_trace_release(&trace);
    rc = yvex_backend_close_checked(&backend, err);
    if (rc != YVEX_OK)
        fprintf(stderr,
                "attention_cuda_fault_close_failed environment=%s value=%s "
                "rc=%d where=%s message=%s\n",
                environment ? environment : "", value ? value : "", rc,
                yvex_error_where(err), yvex_error_message(err));
    return passed && rc == YVEX_OK;
}

/* Prove page-locked staging cleanup failure remains observable after ownership discharge. */
static int run_cuda_workspace_cleanup_fault(
    const yvex_attention_plan *plan, yvex_error *err)
{
    yvex_backend_options options = {0};
    yvex_backend *backend = NULL;
    int close_rc;

    options.kind = YVEX_BACKEND_KIND_CUDA;
    if (yvex_backend_open(&backend, &options, err) != YVEX_OK ||
        !attention_cuda_workspace_prepare(backend, plan, err)) {
        (void)yvex_backend_close_checked(&backend, err);
        return 0;
    }
    if (setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE", "host-workspace", 1) != 0) {
        (void)yvex_backend_close_checked(&backend, err);
        return 0;
    }
    close_rc = yvex_backend_close_checked(&backend, err);
    (void)unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE");
    if (close_rc != YVEX_ERR_BACKEND || backend)
        fprintf(stderr,
                "attention_cuda_workspace_cleanup_fault_failed rc=%d "
                "backend_live=%d where=%s message=%s\n",
                close_rc, backend != NULL, yvex_error_where(err),
                yvex_error_message(err));
    return close_rc == YVEX_ERR_BACKEND && !backend;
}

static int run_cuda_bundle_refusal(yvex_error *err)
{
    yvex_backend_options options;
    yvex_backend_capability_result capability;
    yvex_backend *backend = NULL;
    int rc;
    int passed;

    memset(&options, 0, sizeof(options));
    memset(&capability, 0, sizeof(capability));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    if (setenv("YVEX_TEST_CUDA_BUNDLE_FAILURE", "yvex_attention_topk", 1) !=
        0)
        return 0;
    rc = yvex_backend_open(&backend, &options, err);
    (void)unsetenv("YVEX_TEST_CUDA_BUNDLE_FAILURE");
    if (rc != YVEX_OK || !backend) {
        (void)yvex_backend_close_checked(&backend, err);
        return 0;
    }
    rc = yvex_backend_query_capability(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &capability, err);
    passed = rc == YVEX_OK &&
             capability.state == YVEX_BACKEND_CAPABILITY_FAILED &&
             capability.reason ==
                 YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING &&
             !capability.kernel_bundle_available &&
             !capability.function_available;
    rc = yvex_backend_close_checked(&backend, err);
    if (!passed || rc != YVEX_OK)
        fprintf(stderr,
                "attention_cuda_bundle_refusal_failed passed=%d close_rc=%d "
                "state=%u reason=%u where=%s message=%s\n",
                passed, rc, (unsigned int)capability.state,
                (unsigned int)capability.reason, yvex_error_where(err),
                yvex_error_message(err));
    return passed && rc == YVEX_OK;
}

typedef struct {
    yvex_attention_history_view view;
    float *local_kv;
    unsigned long long *local_positions;
    float *compressed_kv;
    unsigned long long *compressed_positions;
    float *indexer_kv;
    unsigned long long *indexer_positions;
    float *main_kv;
    float *main_score;
    float *index_kv;
    float *index_score;
} live_attention_history;

static void live_history_release(live_attention_history *history)
{
    if (!history) return;
    free(history->local_kv);
    free(history->local_positions);
    free(history->compressed_kv);
    free(history->compressed_positions);
    free(history->indexer_kv);
    free(history->indexer_positions);
    free(history->main_kv);
    free(history->main_score);
    free(history->index_kv);
    free(history->index_score);
    memset(history, 0, sizeof(*history));
}

static int live_rolling_init(
    yvex_attention_rolling_state_view *state,
    float **kv_out,
    float **score_out,
    const yvex_attention_layer_plan *layer,
    yvex_attention_rolling_kind kind,
    unsigned long long token_position,
    const char *plan_identity)
{
    unsigned long long head_dimension =
        kind == YVEX_ATTENTION_ROLLING_INDEXER
            ? layer->indexer_head_dimension : layer->head_dimension;
    int overlap = kind == YVEX_ATTENTION_ROLLING_INDEXER ||
                  layer->attention_class == YVEX_ATTENTION_CLASS_CSA;
    unsigned long long width = head_dimension * (overlap ? 2ull : 1ull);
    unsigned long long slots = layer->compression_ratio *
                               (overlap ? 2ull : 1ull);
    unsigned long long extent = slots * width;
    unsigned long long slot;
    unsigned long long active;

    *kv_out = (float *)calloc((size_t)extent, sizeof(**kv_out));
    *score_out = (float *)malloc((size_t)extent * sizeof(**score_out));
    if (!*kv_out || !*score_out) return 0;
    for (slot = 0ull; slot < extent; ++slot) (*score_out)[slot] = -INFINITY;
    memset(state, 0, sizeof(*state));
    state->present = 1;
    state->schema_version = YVEX_ATTENTION_ROLLING_STATE_SCHEMA_V1;
    state->kind = kind;
    state->attention_class = layer->attention_class;
    state->layer_index = layer->layer_index;
    state->next_token_position = token_position;
    state->ratio = layer->compression_ratio;
    state->head_dimension = head_dimension;
    state->state_width = width;
    state->state_slots = slots;
    state->cursor = token_position % layer->compression_ratio;
    state->current_fill = state->cursor;
    state->previous_fill = overlap && token_position >= layer->compression_ratio
        ? layer->compression_ratio : 0ull;
    state->kv_state_stride = width;
    state->score_state_stride = width;
    state->kv_state_extent = extent;
    state->score_state_extent = extent;
    state->kv_state = *kv_out;
    state->score_state = *score_out;
    state->overlap = overlap;
    state->rotated = kind == YVEX_ATTENTION_ROLLING_INDEXER;
    (void)snprintf(state->attention_plan_identity,
                   sizeof(state->attention_plan_identity), "%s",
                   plan_identity);
    if (state->previous_fill) {
        for (slot = 0ull; slot < layer->compression_ratio; ++slot) {
            unsigned long long lane;
            for (lane = 0ull; lane < width; ++lane) {
                unsigned long long offset = slot * width + lane;
                (*kv_out)[offset] =
                    (float)((int)((offset * 17ull + 3ull) % 97ull) - 48) /
                    128.0f;
                (*score_out)[offset] =
                    (float)((int)((offset * 13ull + 5ull) % 67ull) - 33) /
                    64.0f;
            }
        }
    }
    active = state->cursor;
    for (slot = 0ull; slot < active; ++slot) {
        unsigned long long target = overlap ? layer->compression_ratio + slot
                                            : slot;
        unsigned long long lane;
        for (lane = 0ull; lane < width; ++lane) {
            unsigned long long offset = target * width + lane;
            (*kv_out)[offset] =
                (float)((int)((offset * 19ull + 7ull) % 101ull) - 50) /
                128.0f;
            (*score_out)[offset] =
                (float)((int)((offset * 11ull + 9ull) % 71ull) - 35) /
                64.0f;
        }
    }
    return 1;
}

/* Contract: creates deterministic external history including real-size local,
 * compressed, indexer, and rolling-state facts. */
static int live_history_init(
    live_attention_history *history,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_summary *summary,
    unsigned long long token_position)
{
    unsigned long long local_capacity;
    unsigned long long local_count;
    unsigned long long compressed_count = 0ull;
    unsigned long long i;

    if (!history || !layer || !summary || !layer->sliding_window) return 0;
    local_capacity = layer->sliding_window - 1ull;
    local_count = token_position < local_capacity
        ? token_position : local_capacity;
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA) {
        if (!layer->compression_ratio) return 0;
        compressed_count = token_position / layer->compression_ratio;
    }
    memset(history, 0, sizeof(*history));
    history->view.immutable = 1;
    history->view.token_count = token_position;
    history->view.local_tail_count = local_count;
    history->view.compressed_entry_count = compressed_count;
    history->view.indexer_entry_count =
        layer->attention_class == YVEX_ATTENTION_CLASS_CSA
            ? compressed_count : 0ull;
    if (local_count) {
        history->local_kv = (float *)calloc(
            (size_t)(local_count * layer->head_dimension), sizeof(float));
        history->local_positions = (unsigned long long *)calloc(
            (size_t)local_count, sizeof(*history->local_positions));
        if (!history->local_kv || !history->local_positions) goto fail;
        fill_history_values(history->local_kv,
                            local_count * layer->head_dimension,
                            layer->layer_index + 101ull);
        for (i = 0ull; i < local_count; ++i)
            history->local_positions[i] = token_position - local_count + i;
        history->view.local_kv = history->local_kv;
        history->view.local_positions = history->local_positions;
        history->view.local_kv_stride = layer->head_dimension;
    }
    if (compressed_count) {
        history->compressed_kv = (float *)calloc(
            (size_t)(compressed_count * layer->head_dimension), sizeof(float));
        history->compressed_positions = (unsigned long long *)calloc(
            (size_t)compressed_count,
            sizeof(*history->compressed_positions));
        if (!history->compressed_kv || !history->compressed_positions)
            goto fail;
        fill_history_values(history->compressed_kv,
                            compressed_count * layer->head_dimension,
                            layer->layer_index + 211ull);
        for (i = 0ull; i < compressed_count; ++i)
            history->compressed_positions[i] =
                i * layer->compression_ratio;
        history->view.compressed_kv = history->compressed_kv;
        history->view.compressed_positions = history->compressed_positions;
        history->view.compressed_kv_stride = layer->head_dimension;
    }
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
        compressed_count) {
        history->indexer_kv = (float *)calloc(
            (size_t)(compressed_count * layer->indexer_head_dimension),
            sizeof(float));
        history->indexer_positions = (unsigned long long *)calloc(
            (size_t)compressed_count, sizeof(*history->indexer_positions));
        if (!history->indexer_kv || !history->indexer_positions) goto fail;
        fill_history_values(
            history->indexer_kv,
            compressed_count * layer->indexer_head_dimension,
            layer->layer_index + 307ull);
        memcpy(history->indexer_positions, history->compressed_positions,
               (size_t)compressed_count *
                   sizeof(*history->indexer_positions));
        history->view.indexer_kv = history->indexer_kv;
        history->view.indexer_positions = history->indexer_positions;
        history->view.indexer_kv_stride = layer->indexer_head_dimension;
    }
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA &&
        !live_rolling_init(&history->view.main_rolling_state,
                           &history->main_kv, &history->main_score, layer,
                           YVEX_ATTENTION_ROLLING_MAIN,
                           token_position, summary->attention_plan_identity))
        goto fail;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
        !live_rolling_init(&history->view.indexer_rolling_state,
                           &history->index_kv, &history->index_score, layer,
                           YVEX_ATTENTION_ROLLING_INDEXER,
                           token_position, summary->attention_plan_identity))
        goto fail;
    return 1;

fail:
    live_history_release(history);
    return 0;
}

static void live_history_bind_next_state(
    live_attention_history *history,
    const yvex_attention_execution_trace *trace)
{
    const yvex_attention_rolling_state_output *main_state =
        &trace->next_main_rolling_state;
    const yvex_attention_rolling_state_output *index_state =
        &trace->next_indexer_rolling_state;

    history->view.token_count = trace->token_position + trace->token_count;
    if (main_state->present) {
        history->view.main_rolling_state =
            (yvex_attention_rolling_state_view){
                .present = main_state->present,
                .schema_version = main_state->schema_version,
                .kind = main_state->kind,
                .attention_class = main_state->attention_class,
                .layer_index = main_state->layer_index,
                .next_token_position = main_state->next_token_position,
                .ratio = main_state->ratio,
                .head_dimension = main_state->head_dimension,
                .state_width = main_state->state_width,
                .state_slots = main_state->state_slots,
                .previous_fill = main_state->previous_fill,
                .current_fill = main_state->current_fill,
                .cursor = main_state->cursor,
                .kv_state_stride = main_state->kv_state_stride,
                .score_state_stride = main_state->score_state_stride,
                .kv_state_extent = main_state->kv_state_extent,
                .score_state_extent = main_state->score_state_extent,
                .kv_state = main_state->kv_state,
                .score_state = main_state->score_state,
                .overlap = main_state->overlap,
                .rotated = main_state->rotated,
            };
        memcpy(history->view.main_rolling_state.attention_plan_identity,
               main_state->attention_plan_identity,
               sizeof(main_state->attention_plan_identity));
    }
    if (index_state->present) {
        history->view.indexer_rolling_state =
            (yvex_attention_rolling_state_view){
                .present = index_state->present,
                .schema_version = index_state->schema_version,
                .kind = index_state->kind,
                .attention_class = index_state->attention_class,
                .layer_index = index_state->layer_index,
                .next_token_position = index_state->next_token_position,
                .ratio = index_state->ratio,
                .head_dimension = index_state->head_dimension,
                .state_width = index_state->state_width,
                .state_slots = index_state->state_slots,
                .previous_fill = index_state->previous_fill,
                .current_fill = index_state->current_fill,
                .cursor = index_state->cursor,
                .kv_state_stride = index_state->kv_state_stride,
                .score_state_stride = index_state->score_state_stride,
                .kv_state_extent = index_state->kv_state_extent,
                .score_state_extent = index_state->score_state_extent,
                .kv_state = index_state->kv_state,
                .score_state = index_state->score_state,
                .overlap = index_state->overlap,
                .rotated = index_state->rotated,
            };
        memcpy(history->view.indexer_rolling_state.attention_plan_identity,
               index_state->attention_plan_identity,
               sizeof(index_state->attention_plan_identity));
    }
}

/* Contract: an envelope with non-identity mHC/norm must feed its transformed input to the
 * backend core. Replaying that captured input through the same backend must reproduce every
 * core projection and state delta exactly; CPU/CUDA are not treated as mutual numeric oracles. */
static int run_cuda_core_input_regression(
    const yvex_attention_plan *plan, const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_backend *backend, const yvex_attention_summary *summary,
    unsigned long long csa_layer, yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_attention_layer_plan *layer =
        yvex_attention_plan_layer_at(plan, csa_layer);
    yvex_attention_cpu_options options;
    yvex_attention_cpu_result cpu_result, cuda_result;
    yvex_attention_execution_trace cpu_trace, cuda_trace;
    yvex_attention_execution_trace cpu_direct_trace, cuda_direct_trace;
    yvex_graph_f32_comparison input_comparison;
    yvex_test_attention_reference_contract contract;
    yvex_test_attention_reference_metrics cpu_direct, cuda_direct;
    yvex_attention_cpu_options direct_options;
    live_attention_history history;
    float *residual = NULL;
    unsigned long long lane;
    int transformed = 0;
    int rc = YVEX_ERR_STATE;

    memset(&cpu_trace, 0, sizeof(cpu_trace));
    memset(&cuda_trace, 0, sizeof(cuda_trace));
    memset(&cpu_direct_trace, 0, sizeof(cpu_direct_trace));
    memset(&cuda_direct_trace, 0, sizeof(cuda_direct_trace));
    memset(&cpu_direct, 0, sizeof(cpu_direct));
    memset(&cuda_direct, 0, sizeof(cuda_direct));
    memset(&history, 0, sizeof(history));
    if (!layer || layer->attention_class != YVEX_ATTENTION_CLASS_CSA ||
        !layer->mhc_attention_pre_and_post || !layer->residual_expanded_width ||
        layer->residual_expanded_width > SIZE_MAX / sizeof(*residual))
        goto cleanup;
    residual = (float *)malloc((size_t)layer->residual_expanded_width * sizeof(*residual));
    if (!residual || !live_history_init(&history, layer, summary, 0ull)) {
        rc = YVEX_ERR_NOMEM;
        goto cleanup;
    }
    fill_history_values(residual, layer->residual_expanded_width, 9107ull);
    attention_execution_api()->cpu_options_default(&options);
    options.operation_scope = YVEX_ATTENTION_OPERATION_ENVELOPE;
    options.layer_index = csa_layer;
    options.token_count = 1ull;
    options.input = residual;
    options.input_stride = layer->residual_expanded_width;
    options.history = &history.view;
    options.publication = &cpu_trace;
    rc = attention_execution_api()->cpu_chunk_execute(
        plan, ir, session, descriptor, &options, &cpu_result, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    options.publication = &cuda_trace;
    rc = attention_execution_api()->cuda_token_execute(
        plan, ir, session, descriptor, backend, &options, &cuda_result, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!cpu_trace.complete || !cuda_trace.complete || !cpu_trace.input || !cuda_trace.input ||
        !cpu_trace.index_weights || !cuda_trace.index_weights ||
        !cpu_trace.next_main_rolling_state.present ||
        !cuda_trace.next_main_rolling_state.present ||
        !cpu_trace.next_indexer_rolling_state.present ||
        !cuda_trace.next_indexer_rolling_state.present)
        goto mismatch;
    for (lane = 0ull; lane < layer->hidden_dimension; ++lane) {
        double scale = fmax(fabs((double)cpu_trace.input[lane]),
                            fabs((double)residual[lane]));
        double difference = fabs((double)cpu_trace.input[lane] -
                                 (double)residual[lane]);
        if (difference > ATTENTION_CUDA_ABSOLUTE_TOLERANCE +
                             ATTENTION_CUDA_RELATIVE_TOLERANCE * scale) {
            transformed = 1;
            break;
        }
    }
    if (!transformed) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "attention.cuda.core_input.transform",
                       "attention envelope did not transform its core input");
        goto mismatch;
    }
    if (yvex_graph_f32_compare(
            cpu_trace.input, cuda_trace.input, layer->hidden_dimension,
            ATTENTION_CUDA_ABSOLUTE_TOLERANCE, ATTENTION_CUDA_RELATIVE_TOLERANCE,
            &input_comparison, err) != YVEX_OK || !input_comparison.within_tolerance) {
        fprintf(stderr,
                "attention_cuda_core_input_mismatch coordinate=%llu max_abs=%.17g "
                "max_rel=%.17g\n",
                input_comparison.first_failing_coordinate,
                input_comparison.maximum_absolute_error,
                input_comparison.maximum_relative_error);
        goto mismatch;
    }
    direct_options = options;
    direct_options.operation_scope = YVEX_ATTENTION_OPERATION_CORE;
    direct_options.input = cpu_trace.input;
    direct_options.input_stride = layer->hidden_dimension;
    direct_options.publication = &cpu_direct_trace;
    rc = attention_execution_api()->cpu_chunk_execute(
        plan, ir, session, descriptor, &direct_options, &cpu_result, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    direct_options.input = cuda_trace.input;
    direct_options.publication = &cuda_direct_trace;
    rc = attention_execution_api()->cuda_token_execute(
        plan, ir, session, descriptor, backend, &direct_options,
        &cuda_result, failure, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!attention_reference_contract_init(
            &contract, summary, "envelope-core-routing", 0.0, 0.0) ||
        !yvex_test_attention_reference_compare_contract(
            &cpu_trace, &cpu_direct_trace, &contract, &cpu_direct) ||
        !yvex_test_attention_reference_compare_contract(
            &cuda_trace, &cuda_direct_trace, &contract, &cuda_direct)) {
        fprintf(stderr,
                "attention_cuda_core_routing_mismatch cpu_stage=%s cpu_coordinate=%llu "
                "cuda_stage=%s cuda_coordinate=%llu\n",
                cpu_direct.first_failed_stage ? cpu_direct.first_failed_stage : "",
                cpu_direct.first_failed_index,
                cuda_direct.first_failed_stage ? cuda_direct.first_failed_stage : "",
                cuda_direct.first_failed_index);
        goto mismatch;
    }
    rc = YVEX_OK;
    goto cleanup;

mismatch:
    if (!yvex_error_is_set(err))
        yvex_error_set(err, YVEX_ERR_FORMAT, "attention.cuda.core_input",
                       "CUDA rolling and index projections diverged from transformed core input");
    rc = YVEX_ERR_FORMAT;
cleanup:
    free(residual);
    live_history_release(&history);
    yvex_attention_execution_trace_release(&cpu_trace);
    yvex_attention_execution_trace_release(&cuda_trace);
    yvex_attention_execution_trace_release(&cpu_direct_trace);
    yvex_attention_execution_trace_release(&cuda_direct_trace);
    return rc;
}

static void runtime_oracle_graph_failure_report(
    const yvex_backend *backend, yvex_runtime_execution_mode mode,
    unsigned long long layer, unsigned int repeat,
    const yvex_attention_failure *failure)
{
    yvex_backend_cuda_attention_graph_entry entry;
    yvex_error inspect_error;
    unsigned long long count = 0ull, index;

    fprintf(stderr,
            "attention_runtime_oracle_dispatch_failed mode=%u layer=%llu "
            "repeat=%u failure=%u expected=%llu actual=%llu reason=%s\n",
            (unsigned int)mode, layer, repeat,
            failure ? (unsigned int)failure->code : 0u,
            failure ? failure->expected : 0ull,
            failure ? failure->actual : 0ull,
            failure && failure->reason ? failure->reason : "");
    yvex_error_clear(&inspect_error);
    if (yvex_backend_cuda_attention_graph_registry_count(
            backend, &count, &inspect_error) != YVEX_OK) {
        fprintf(stderr,
                "attention_runtime_oracle_registry_failed where=%s message=%s\n",
                yvex_error_where(&inspect_error),
                yvex_error_message(&inspect_error));
        return;
    }
    fprintf(stderr, "attention_runtime_oracle_registry_count=%llu\n", count);
    for (index = 0ull; index < count; ++index) {
        yvex_error_clear(&inspect_error);
        if (yvex_backend_cuda_attention_graph_registry_get(
                backend, index, &entry, &inspect_error) != YVEX_OK) {
            fprintf(stderr,
                    "attention_runtime_oracle_registry_entry_failed index=%llu "
                    "where=%s message=%s\n",
                    index, yvex_error_where(&inspect_error),
                    yvex_error_message(&inspect_error));
            continue;
        }
        fprintf(stderr,
                "attention_runtime_oracle_registry_entry index=%llu state=%u "
                "reason=%u captures=%llu instantiates=%llu launches=%llu "
                "replays=%llu kernels=%llu key=%s\n",
                index, (unsigned int)entry.graph.state,
                (unsigned int)entry.graph.reason, entry.graph.capture_count,
                entry.graph.instantiate_count, entry.graph.launch_count,
                entry.graph.replay_count, entry.graph.inventory.kernel_node_count,
                entry.compatibility_identity);
    }
}

/*
 * Execute the canonical SWA/CSA/HCA fixture twice through one resident CUDA mode.
 *
 * Lifecycle, capture, publication, oracle, or cleanup disagreement refuses the mode.
 */
static int run_runtime_oracle_mode(
    yvex_model_engine *model, yvex_runtime_execution_session *runtime_session,
    const yvex_graph_attention_capacity_plan *capacity,
    const yvex_deepseek_v4_ir *ir, yvex_runtime_execution_mode mode,
    unsigned long long swa_layer, unsigned long long csa_layer,
    unsigned long long hca_layer, runtime_oracle_mode_result *result,
    yvex_error *err)
{
    const yvex_model_engine_view *model_view = yvex_model_engine_view_get(model);
    const yvex_runtime_session_view *session_view =
        yvex_runtime_session_view_get(runtime_session);
    const yvex_attention_plan *plan = model_view ? model_view->attention : NULL;
    const yvex_attention_summary *summary = yvex_attention_plan_summary(plan);
    const yvex_graph_attention_capacity_summary *capacity_summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    const unsigned long long layers[] = {swa_layer, csa_layer, hca_layer};
    yvex_model_engine_failure model_failure;
    yvex_attention_probe_request request;
    yvex_attention_probe_result probe;
    yvex_runtime_session_summary session_summary;
    yvex_graph_attention_state_summary state_summary;
    yvex_backend_cuda_attention_mode backend_mode;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long layer_index;
    unsigned int repeat;
    int rc;

    if (!result || !model_view || !session_view || !plan || !summary ||
        !capacity_summary || !ir ||
        !model_view->materialization || !model_view->descriptor ||
        !model_view->binding || !session_view->backend ||
        !session_view->attention_workspace ||
        capacity_summary->selected_layer_count != 3ull ||
        strcmp(capacity_summary->attention_plan_identity,
               summary->attention_plan_identity) != 0 ||
        (mode != YVEX_RUNTIME_MODE_PIECEWISE && mode != YVEX_RUNTIME_MODE_FULL)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "attention.runtime_oracle",
                       "sealed resident CUDA graph owners are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(result, 0, sizeof(*result));
    result->oracle.plan = plan;
    result->oracle.ir = ir;
    result->oracle.materialization = model_view->materialization;
    result->oracle.descriptor = model_view->descriptor;
    yvex_sha256_init(&result->oracle.output_hash);
    yvex_sha256_init(&result->oracle.state_hash);
    if (!yvex_sha256_update_text(
            &result->oracle.output_hash,
            "yvex.test.runtime-graph.oracle-output-set.v1") ||
        !yvex_sha256_update_text(
            &result->oracle.state_hash,
            "yvex.test.runtime-graph.oracle-state-set.v1")) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                       "runtime oracle identity initialization failed");
        return YVEX_ERR_STATE;
    }
    rc = YVEX_OK;
    backend_mode = mode == YVEX_RUNTIME_MODE_FULL
        ? YVEX_BACKEND_CUDA_ATTENTION_FULL
        : YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE;
    if (rc == YVEX_OK)
        rc = yvex_backend_cuda_attention_configure(
            session_view->backend, YVEX_BACKEND_ATTENTION_PHASE_DECODE,
            backend_mode,
            model_view->binding->executable_graph_identity,
            "decode-1",
            capacity_summary->components[
                YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].maximum_capacity,
            capacity_summary->components[
                YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].maximum_capacity,
            capacity_summary->components[
                YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].maximum_capacity,
            err);
    for (layer_index = 0ull; rc == YVEX_OK && layer_index < 3ull;
         ++layer_index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(plan, layers[layer_index]);

        if (!layer || layer->layer_index != layers[layer_index]) {
            yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                           "representative runtime layer is unavailable");
            rc = YVEX_ERR_STATE;
            break;
        }
        for (repeat = 0u; rc == YVEX_OK && repeat < 2u; ++repeat) {
            unsigned long long position = 0ull;

            memset(&request, 0, sizeof(request));
            memset(&probe, 0, sizeof(probe));
            memset(&model_failure, 0, sizeof(model_failure));
            request.backend = YVEX_BACKEND_KIND_CUDA;
            request.execution_phase = YVEX_EXECUTION_PHASE_DECODE;
            request.probe = YVEX_ATTENTION_PROBE_CANONICAL_V2;
            request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
            request.operation_scope = YVEX_ATTENTION_OPERATION_CORE;
            request.evidence_level = YVEX_ATTENTION_EVIDENCE_FULL;
            request.token_count = 1ull;
            request.layer_ordinal = layer->layer_index;
            rc = yvex_attention_probe_position_resolve(
                layer, 0, (unsigned long long)repeat, &position, err);
            request.token_position = position;
            request.select_layer = 1;
            request.select_position = 1;
            request.evidence = runtime_oracle_compare_publication;
            request.evidence_context = &result->oracle;
            result->oracle.history =
                session_view->attention_state_provider->view
                    ? session_view->attention_state_provider->view(
                          session_view->attention_state_provider->context,
                          layer->layer_index,
                          YVEX_ATTENTION_STATE_VIEW_COMMITTED)
                    : NULL;
            if (!result->oracle.history ||
                result->oracle.history->token_count != position) {
                yvex_error_set(err, YVEX_ERR_STATE,
                               "attention.runtime_oracle.history",
                               "committed runtime history is not contiguous");
                rc = YVEX_ERR_STATE;
            }
            if (rc == YVEX_OK)
                rc = yvex_runtime_attention_probe_execute(
                    runtime_session, model, &request, &probe, &model_failure, err);
            result->oracle.history = NULL;
            if (rc != YVEX_OK)
                runtime_oracle_graph_failure_report(
                    session_view->backend, mode, layer->layer_index, repeat, NULL);
            if (rc == YVEX_OK &&
                (probe.layers_executed != 1ull ||
                 probe.bindings_executed != layer->required_binding_count ||
                 probe.swa_layers_executed !=
                     (layer->attention_class == YVEX_ATTENTION_CLASS_SWA) ||
                 probe.csa_layers_executed !=
                     (layer->attention_class == YVEX_ATTENTION_CLASS_CSA) ||
                 probe.hca_layers_executed !=
                     (layer->attention_class == YVEX_ATTENTION_CLASS_HCA))) {
                yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                               "runtime mode published incomplete layer evidence");
                rc = YVEX_ERR_STATE;
            }
        }
    }
    memset(&session_summary, 0, sizeof(session_summary));
    memset(&state_summary, 0, sizeof(state_summary));
    if (rc == YVEX_OK &&
        (yvex_runtime_session_summary_copy(
             runtime_session, &session_summary, err) != YVEX_OK ||
         !session_view->attention_state_provider ||
         session_view->attention_state_provider->summary(
             session_view->attention_state_provider->context,
             &state_summary, err) != YVEX_OK ||
         session_summary.execution_count != 6ull ||
         state_summary.commit_count != 6ull ||
         state_summary.transaction_active || state_summary.candidate_active ||
         state_summary.staged_layer_count)) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                       "session-scoped runtime execution did not commit six state batches");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_cuda_attention_graph_summary_get(
            session_view->backend, &result->graph, err);
    if (rc == YVEX_OK &&
        (!result->graph.configured ||
         result->graph.selected_mode != backend_mode ||
         !result->graph.graph_count || !result->graph.capture_count ||
         !result->graph.instantiate_count ||
         result->graph.capture_count != result->graph.graph_count ||
         result->graph.instantiate_count != result->graph.graph_count ||
         result->graph.replay_count <= result->graph.graph_count ||
         !result->graph.kernel_node_count ||
         (mode == YVEX_RUNTIME_MODE_PIECEWISE &&
          result->graph.piece_count < 2ull))) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                       "runtime CUDA graph lifecycle evidence is incomplete");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK &&
        (result->oracle.publications != 6ull ||
         !result->oracle.compared_values ||
         !yvex_sha256_final(&result->oracle.output_hash, digest))) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                       "runtime oracle comparison set is incomplete");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        yvex_sha256_hex(digest, result->output_identity);
        if (!yvex_sha256_final(&result->oracle.state_hash, digest)) {
            yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                           "runtime oracle state identity finalization failed");
            rc = YVEX_ERR_STATE;
        } else {
            yvex_sha256_hex(digest, result->state_identity);
        }
    }
    return rc;
}

static int run_runtime_residency_close_order(yvex_model_engine *model,
                                             yvex_error *err)
{
    const yvex_model_engine_view *view = yvex_model_engine_view_get(model);
    yvex_runtime_residency *borrowed;
    yvex_runtime_residency_summary before, after;
    yvex_error refusal;
    int rc, unset_rc;

    memset(&before, 0, sizeof(before));
    memset(&after, 0, sizeof(after));
    if (!view || !view->residency ||
        yvex_runtime_residency_snapshot(
            view->residency, &before, NULL, NULL, err) != YVEX_OK ||
        !before.cuda_ready || !before.binding_count ||
        before.cuda_upload_bytes || before.cuda_upload_count ||
        !((before.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED &&
           before.artifact_backed_bytes &&
           !before.host_resident_bytes &&
           !before.device_resident_bytes &&
           before.cuda_addressable_bytes == before.artifact_backed_bytes &&
           before.cuda_pageable_map_bytes == before.artifact_backed_bytes &&
           before.cuda_pageable_map_count == 1ull &&
           before.cuda_host_registration_count == 1ull &&
           !before.cuda_pageable_prefetch_bytes && !before.cuda_pageable_prefetch_count &&
           !before.cuda_managed_bytes && !before.cuda_managed_allocation_count &&
           !before.cuda_managed_prefetch_bytes && !before.cuda_managed_prefetch_count) ||
          (before.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED &&
           !before.host_resident_bytes &&
           before.device_resident_bytes == before.encoded_bytes &&
           before.cuda_addressable_bytes == before.encoded_bytes &&
           before.cuda_managed_bytes == before.encoded_bytes &&
           before.cuda_managed_allocation_count == 1ull &&
           before.cuda_managed_prefetch_bytes == before.encoded_bytes &&
           before.cuda_managed_prefetch_count == 1ull))) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_residency.close_order",
                       "one populated live CUDA residency is required");
        return YVEX_ERR_STATE;
    }
    borrowed = (yvex_runtime_residency *)view->residency;
    if (setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE", "tensor-alloc", 1) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_residency.close_order",
                       "CUDA tensor cleanup seam could not be installed");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(&refusal);
    rc = yvex_runtime_residency_close(&borrowed, &refusal);
    unset_rc = unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE");
    if (unset_rc != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_residency.close_order",
                       "CUDA tensor cleanup seam could not be cleared");
        return YVEX_ERR_STATE;
    }
    if (rc != YVEX_ERR_STATE ||
        strcmp(yvex_error_where(&refusal), "backend.close.admit") != 0 ||
        borrowed != view->residency ||
        yvex_runtime_residency_snapshot(
            borrowed, &after, NULL, NULL, err) != YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_residency.close_order",
                       "live child lease did not refuse residency teardown first");
        return YVEX_ERR_STATE;
    }
    if (after.schema_version != before.schema_version ||
        after.placement != before.placement ||
        after.sealed != before.sealed || after.attached != before.attached ||
        after.host_ready != before.host_ready ||
        after.host_locked != before.host_locked ||
        after.cuda_ready != before.cuda_ready ||
        after.invalidated != before.invalidated ||
        after.generation != before.generation ||
        after.model_complete != before.model_complete ||
        after.expected_model_binding_count != before.expected_model_binding_count ||
        after.model_binding_count != before.model_binding_count ||
        after.core_binding_count != before.core_binding_count ||
        after.envelope_binding_count != before.envelope_binding_count ||
        after.binding_count != before.binding_count ||
        after.encoded_bytes != before.encoded_bytes ||
        after.accelerator_encoded_bytes != before.accelerator_encoded_bytes ||
        after.host_resident_bytes != before.host_resident_bytes ||
        after.device_resident_bytes != before.device_resident_bytes ||
        after.artifact_backed_bytes != before.artifact_backed_bytes ||
        after.cuda_upload_bytes != before.cuda_upload_bytes ||
        after.cuda_upload_count != before.cuda_upload_count ||
        after.cuda_host_registration_count != before.cuda_host_registration_count ||
        after.cuda_pageable_map_bytes != before.cuda_pageable_map_bytes ||
        after.cuda_pageable_map_count != before.cuda_pageable_map_count ||
        after.cuda_managed_bytes != before.cuda_managed_bytes ||
        after.cuda_managed_allocation_count != before.cuda_managed_allocation_count ||
        after.cuda_managed_prefetch_bytes != before.cuda_managed_prefetch_bytes ||
        after.cuda_managed_prefetch_count != before.cuda_managed_prefetch_count ||
        after.cuda_pageable_prefetch_bytes != before.cuda_pageable_prefetch_bytes ||
        after.cuda_pageable_prefetch_count != before.cuda_pageable_prefetch_count ||
        memcmp(after.qtype_binding_counts, before.qtype_binding_counts,
               sizeof(before.qtype_binding_counts)) != 0 ||
        memcmp(after.qtype_bytes, before.qtype_bytes,
               sizeof(before.qtype_bytes)) != 0 ||
        strcmp(after.payload_digest, before.payload_digest) != 0 ||
        strcmp(after.residency_identity, before.residency_identity) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "attention.runtime_residency.close_order",
                       "refused residency teardown changed admitted resident bytes or identity");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Prove operator phase cleanup retains its complete owner and succeeds on retry.
 *
 * Missing retained ownership, wrong failure typing, or retry disagreement refuses.
 */
static int run_runtime_phase_cleanup_retry(
    const char *artifact_path, const char *runtime_binding_path,
    unsigned long long swa_layer, yvex_error *err)
{
    yvex_graph_attention_operator_request request;
    yvex_graph_attention_operator_result result;
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_error cleanup_error;
    int rc, close_rc, expected, unset_rc;

    memset(&request, 0, sizeof(request));
    memset(&result, 0, sizeof(result));
    request.target = "deepseek4-v4-flash-dspark";
    request.artifact_path = artifact_path;
    request.runtime_binding_path = runtime_binding_path;
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.probe = YVEX_ATTENTION_PROBE_CANONICAL_V2;
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    request.phase = YVEX_EXECUTION_PHASE_PREFILL;
    request.mode = YVEX_RUNTIME_MODE_EAGER;
    request.operation_scope = YVEX_RUNTIME_SCOPE_ATTENTION_CORE;
    request.operator_action = YVEX_RUNTIME_OPERATOR_STATE_EXERCISE;
    request.token_count = 2ull;
    request.repeat = 1ull;
    request.select_layer = 1;
    request.layer_start = swa_layer;
    request.layer_count = 1ull;
    if (setenv("YVEX_TEST_RUNTIME_PHASE_CLEANUP_FAILURE", "1", 1) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_phase_cleanup",
                       "phase cleanup fault could not be installed");
        return YVEX_ERR_STATE;
    }
    rc = yvex_graph_attention_operator_execute(
        &request, &result, &cleanup, err);
    unset_rc = unsetenv("YVEX_TEST_RUNTIME_PHASE_CLEANUP_FAILURE");
    expected = rc == YVEX_ERR_STATE && cleanup && !result.completed &&
               strcmp(result.status, "refused") == 0 &&
               strcmp(yvex_error_where(err),
                      "runtime.attention.phase.cleanup") == 0 &&
               strcmp(result.failure_where,
                      "runtime.attention.phase.cleanup") == 0;
    yvex_error_clear(&cleanup_error);
    close_rc = yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error);
    if (unset_rc != 0 || close_rc != YVEX_OK || cleanup ||
        yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error) != YVEX_OK) {
        *err = cleanup_error;
        if (!yvex_error_is_set(err))
            yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_phase_cleanup",
                           "retained phase cleanup could not be retried");
        return YVEX_ERR_STATE;
    }
    if (!expected) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_phase_cleanup",
                       "phase cleanup fault did not retain one typed retry owner");
        return YVEX_ERR_STATE;
    }
    printf("attention_runtime_phase_cleanup_retry=1\n");
    return YVEX_OK;
}

/*
 * Prove piecewise and full runtime-resident CUDA graphs against one oracle fixture set.
 *
 * Authenticates one runtime model, reuses one resident session, and reports parity facts.
 */
static int run_runtime_graph_oracle_suite(
    const char *artifact_path, const char *runtime_binding_path,
    const char *attention_plan_identity, const char *runtime_descriptor_identity,
    const yvex_deepseek_v4_ir *ir, unsigned long long swa_layer,
    unsigned long long csa_layer, unsigned long long hca_layer,
    yvex_error *err)
{
    yvex_model_engine_open_request model_request;
    yvex_runtime_session_open_request session_request;
    yvex_model_engine_failure failure;
    yvex_model_engine *model = NULL;
    yvex_runtime_execution_session *runtime_session = NULL;
    yvex_graph_attention_capacity_plan *capacity = NULL;
    yvex_graph_attention_capacity_request capacity_request;
    yvex_attention_state_provider_factory state_factory;
    runtime_oracle_state_factory_control state_factory_control;
    yvex_attention_failure attention_failure;
    runtime_oracle_mode_result piecewise, full;
    int rc;

    memset(&model_request, 0, sizeof(model_request));
    memset(&session_request, 0, sizeof(session_request));
    memset(&failure, 0, sizeof(failure));
    memset(&capacity_request, 0, sizeof(capacity_request));
    memset(&state_factory_control, 0, sizeof(state_factory_control));
    memset(&attention_failure, 0, sizeof(attention_failure));
    memset(&piecewise, 0, sizeof(piecewise));
    memset(&full, 0, sizeof(full));
    state_factory = (yvex_attention_state_provider_factory){
        .context = &state_factory_control,
        .open = runtime_oracle_state_factory_open,
        .discard = runtime_oracle_state_factory_discard,
    };
    model_request.artifact_path = artifact_path;
    model_request.runtime_binding_path = runtime_binding_path;
    model_request.target_id = "deepseek4-v4-flash-dspark";
    rc = yvex_model_engine_open(&model, &model_request, &failure, err);
    session_request.backend = YVEX_BACKEND_KIND_CUDA;
    session_request.attention_state_factory = &state_factory;
    if (rc == YVEX_OK) {
        const yvex_model_engine_view *view = yvex_model_engine_view_get(model);

        if (!view || !view->binding ||
            strcmp(view->binding->attention_plan_identity,
                   attention_plan_identity) != 0 ||
            strcmp(view->binding->runtime_descriptor_identity,
                   runtime_descriptor_identity) != 0) {
            yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle",
                           "external runtime binding does not match the live oracle fixture");
            rc = YVEX_ERR_STATE;
        }
    }
    capacity_request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    capacity_request.token_count = 1ull;
    capacity_request.execution_count = 2ull;
    if (rc == YVEX_OK) {
        const yvex_model_engine_view *view = yvex_model_engine_view_get(model);

        rc = yvex_graph_attention_capacity_plan_build(
            &capacity, view ? view->attention : NULL, &capacity_request, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_open(
            &runtime_session, model, &session_request, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_prepare_attention_workspace(
            runtime_session, YVEX_RUNTIME_MODE_PIECEWISE,
            YVEX_RUNTIME_SCOPE_ATTENTION_CORE, YVEX_ATTENTION_EVIDENCE_FULL,
            capacity, capacity_request.token_count, 0ull, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_prepare_attention_probe_state(
            runtime_session, model, capacity, &attention_failure, err);
    if (rc == YVEX_OK)
        rc = run_runtime_oracle_mode(
            model, runtime_session, capacity, ir, YVEX_RUNTIME_MODE_PIECEWISE,
            swa_layer, csa_layer, hca_layer, &piecewise, err);
    if (rc == YVEX_OK) {
        yvex_error cleanup;
        int close_rc;

        yvex_error_clear(&cleanup);
        close_rc = yvex_runtime_session_close(&runtime_session, &cleanup);
        if (close_rc != YVEX_OK) {
            rc = close_rc;
            *err = cleanup;
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_open(
            &runtime_session, model, &session_request, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_prepare_attention_workspace(
            runtime_session, YVEX_RUNTIME_MODE_FULL,
            YVEX_RUNTIME_SCOPE_ATTENTION_CORE, YVEX_ATTENTION_EVIDENCE_FULL,
            capacity, capacity_request.token_count, 0ull, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_prepare_attention_probe_state(
            runtime_session, model, capacity, &attention_failure, err);
    if (rc == YVEX_OK)
        rc = run_runtime_oracle_mode(
            model, runtime_session, capacity, ir, YVEX_RUNTIME_MODE_FULL,
            swa_layer, csa_layer, hca_layer, &full, err);
    if (rc == YVEX_OK &&
        (state_factory_control.opens != 2u ||
         state_factory_control.discards != 0u)) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_oracle.factory",
                       "injected state factory did not own both production sessions");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK &&
        (strcmp(piecewise.output_identity, full.output_identity) != 0 ||
         strcmp(piecewise.state_identity, full.state_identity) != 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "attention.runtime_oracle",
                       "piecewise and full modes did not consume the same oracle fixture set");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        rc = run_runtime_residency_close_order(model, err);
    if (rc == YVEX_OK) {
        printf("attention_runtime_oracle_modes=piecewise,full\n");
        printf("attention_runtime_oracle_classes=swa,csa,hca\n");
        printf("attention_runtime_oracle_publications=%llu\n",
               piecewise.oracle.publications + full.oracle.publications);
        printf("attention_runtime_piecewise_reference_values=%llu\n",
               piecewise.oracle.compared_values);
        printf("attention_runtime_piecewise_reference_max_abs=%.17g\n",
               piecewise.oracle.maximum_absolute_error);
        printf("attention_runtime_piecewise_reference_max_rel=%.17g\n",
               piecewise.oracle.maximum_relative_error);
        printf("attention_runtime_piecewise_reference_rmse=%.17g\n",
               reference_metrics_rmse(&(yvex_test_attention_reference_metrics){
                   .compared_values = piecewise.oracle.compared_values,
                   .squared_error_sum = piecewise.oracle.squared_error}));
        printf("attention_runtime_full_reference_values=%llu\n",
               full.oracle.compared_values);
        printf("attention_runtime_full_reference_max_abs=%.17g\n",
               full.oracle.maximum_absolute_error);
        printf("attention_runtime_full_reference_max_rel=%.17g\n",
               full.oracle.maximum_relative_error);
        printf("attention_runtime_full_reference_rmse=%.17g\n",
               reference_metrics_rmse(&(yvex_test_attention_reference_metrics){
                   .compared_values = full.oracle.compared_values,
                   .squared_error_sum = full.oracle.squared_error}));
        printf("attention_runtime_oracle_output_identity=%s\n",
               full.output_identity);
        printf("attention_runtime_oracle_state_identity=%s\n",
               full.state_identity);
        printf("attention_runtime_piecewise_graphs=%llu\n",
               piecewise.graph.graph_count);
        printf("attention_runtime_piecewise_pieces=%llu\n",
               piecewise.graph.piece_count);
        printf("attention_runtime_full_graphs=%llu\n", full.graph.graph_count);
        printf("attention_runtime_graph_replay_proven=1\n");
        printf("attention_runtime_injected_state_factory_opens=%u\n",
               state_factory_control.opens);
        printf("attention_runtime_session_scoped_api=1\n");
    }
    {
        yvex_error cleanup;
        int close_rc;

        yvex_error_clear(&cleanup);
        close_rc = yvex_runtime_session_close(&runtime_session, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) {
            rc = close_rc;
            *err = cleanup;
        }
    }
    yvex_graph_attention_capacity_plan_close(&capacity);
    yvex_model_engine_close(&model);
    if (rc == YVEX_OK && model) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.runtime_residency.close_order",
                       "model residency did not complete checked teardown");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        printf("attention_runtime_residency_close_order=1\n");
    if (rc == YVEX_OK)
        rc = run_runtime_phase_cleanup_retry(
            artifact_path, runtime_binding_path, swa_layer, err);
    return rc;
}

static int run_cuda_live_suite(
    const yvex_attention_plan *plan,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_summary *summary,
    unsigned long long swa_layer,
    unsigned long long csa_layer,
    unsigned long long hca_layer,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    yvex_backend_options backend_options;
    yvex_backend_capability_result capability;
    yvex_backend *backend = NULL;
    yvex_attention_cpu_options options;
    yvex_attention_cpu_result cpu_result;
    yvex_attention_cpu_result cuda_result;
    yvex_test_attention_reference_metrics cpu_reference;
    yvex_test_attention_reference_metrics cuda_reference;
    yvex_test_attention_reference_metrics cpu_cuda;
    yvex_test_attention_reference_evidence cpu_oracle;
    yvex_test_attention_reference_evidence cuda_oracle;
    yvex_attention_execution_trace failed_trace;
    attention_live_evidence evidence;
    live_attention_history history;
    const yvex_attention_layer_plan *layer;
    float *input = NULL, *native_input = NULL;
    unsigned long long input_extent = 0ull;
    unsigned long long layer_index;
    unsigned long long launches = 0ull;
    unsigned long long tensor_core_launches = 0ull;
    unsigned long long peak_host_bytes = 0ull;
    unsigned long long peak_device_bytes = 0ull;
    unsigned long long peak_encoded_weight_staging_bytes = 0ull;
    unsigned long long configured_host_scratch_bytes = 0ull;
    unsigned long long peak_total_host_owned_bytes = 0ull;
    unsigned long long mutation_detected = 0ull;
    attention_cuda_workspace_evidence workspace_evidence = {0};
    char repeat_identity[YVEX_ATTENTION_IDENTITY_CAP];
    char evidence_identity[YVEX_SHA256_HEX_CAP];
    static const struct {
        const char *seam;
        yvex_attention_failure_code expected_failure;
    } fault_cases[] = {
        {"allocation", YVEX_ATTENTION_FAILURE_ALLOCATION},
        {"host-staging", YVEX_ATTENTION_FAILURE_ALLOCATION},
        {"copy-input", YVEX_ATTENTION_FAILURE_BACKEND},
        {"cuda.attention.q_a",
         YVEX_ATTENTION_FAILURE_BACKEND},
        {"copy-output", YVEX_ATTENTION_FAILURE_BACKEND}
    };
    unsigned int fault_index;
    unsigned int qtype;
    int rc;

    memset(&backend_options, 0, sizeof(backend_options));
    memset(&capability, 0, sizeof(capability));
    memset(&failed_trace, 0, sizeof(failed_trace));
    memset(&evidence, 0, sizeof(evidence));
    memset(&history, 0, sizeof(history));
    memset(repeat_identity, 0, sizeof(repeat_identity));
    memset(evidence_identity, 0, sizeof(evidence_identity));
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &backend_options, err);
    if (rc != YVEX_OK) goto cleanup;
    rc = yvex_backend_query_capability(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &capability, err);
    if (rc != YVEX_OK ||
        capability.state != YVEX_BACKEND_CAPABILITY_SUPPORTED) {
        if (rc == YVEX_OK) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED,
                           "attention.cuda.capability",
                           "device-complete DeepSeek attention is not admitted");
            rc = YVEX_ERR_UNSUPPORTED;
        }
        goto cleanup;
    }
    if (!attention_cuda_workspace_prepare(backend, plan, err)) {
        rc = yvex_error_is_set(err) ? yvex_error_code(err) : YVEX_ERR_STATE;
        goto cleanup;
    }
    layer = yvex_attention_plan_layer_at(plan, swa_layer);
    if (!layer) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    input_extent = layer->hidden_dimension;
    input = (float *)malloc((size_t)input_extent * sizeof(*input));
    if (!input) {
        rc = YVEX_ERR_NOMEM;
        goto cleanup;
    }
    if (!attention_evidence_init(&evidence, summary)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    for (layer_index = 0ull;
         layer_index <
             yvex_attention_plan_layer_count(plan);
         ++layer_index) {
        layer = yvex_attention_plan_layer_at(
            plan, layer_index);
        if (!layer || layer->hidden_dimension != input_extent) {
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        fill_history_values(input, input_extent, 401ull + layer_index);
        attention_execution_api()->cpu_options_default(&options);
        configured_host_scratch_bytes = options.scratch_limit_bytes;
        options.layer_index = layer_index;
        options.token_position = 0ull;
        options.token_count = 1ull;
        options.input = input;
        options.input_stride = input_extent;
        if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA) {
            if (!live_history_init(&history, layer, summary, 0ull)) {
                rc = YVEX_ERR_NOMEM;
                goto cleanup;
            }
            options.history = &history.view;
        }
        rc = run_cpu_cuda_reference_compare(
            plan, ir, session, descriptor, backend, &options, &cpu_result,
            &cuda_result, &cpu_reference, &cuda_reference, &cpu_cuda,
            &cpu_oracle, &cuda_oracle, failure, err);
        if (rc != YVEX_OK) {
            fprintf(stderr,
                    "attention_cuda_layer_execution_failed layer=%llu rc=%d "
                    "failure=%u where=%s message=%s\n",
                    layer_index, rc, (unsigned int)failure->code,
                    yvex_error_where(err), yvex_error_message(err));
            live_history_release(&history);
            goto cleanup;
        }
        if (!attention_evidence_add(
                &evidence, &cpu_result, &cuda_result, &cpu_reference,
                &cuda_reference, &cpu_cuda, &cpu_oracle, &cuda_oracle)) {
            fprintf(stderr,
                    "attention_cuda_layer_evidence_failed layer=%llu\n",
                    layer_index);
            live_history_release(&history);
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        if (!attention_cuda_workspace_add(&workspace_evidence, &cuda_result)) {
            fprintf(stderr,
                    "attention_cuda_layer_workspace_failed layer=%llu capacity=%llu "
                    "used=%llu peak=%llu allocations=%llu reused=%d\n",
                    layer_index, cuda_result.cuda_host_workspace_capacity,
                    cuda_result.cuda_host_workspace_used,
                    cuda_result.cuda_host_workspace_peak,
                    cuda_result.cuda_host_workspace_allocations,
                    cuda_result.cuda_host_workspace_reused);
            live_history_release(&history);
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        live_history_release(&history);
        if (layer_index == swa_layer)
            memcpy(repeat_identity, cuda_result.output_identity,
                   sizeof(repeat_identity));
        if (!yvex_core_u64_add(
                launches, cuda_result.cuda_kernel_launches, &launches)) {
            rc = YVEX_ERR_FORMAT;
            goto cleanup;
        }
        if (!yvex_core_u64_add(
                tensor_core_launches, cuda_result.cuda_tensor_core_launches,
                &tensor_core_launches)) {
            rc = YVEX_ERR_FORMAT;
            goto cleanup;
        }
        if (cuda_result.cuda_peak_device_bytes > peak_device_bytes)
            peak_device_bytes = cuda_result.cuda_peak_device_bytes;
        if (cuda_result.cuda_peak_host_bytes > peak_host_bytes)
            peak_host_bytes = cuda_result.cuda_peak_host_bytes;
        if (cuda_result.payload_bytes_read >
            peak_encoded_weight_staging_bytes)
            peak_encoded_weight_staging_bytes =
                cuda_result.payload_bytes_read;
    }

#ifdef YVEX_HAVE_CUDA_KERNEL_CUBIN
    layer = yvex_attention_plan_layer_at(plan, swa_layer);
    fill_history_values(input, input_extent, 1401ull);
    attention_execution_api()->cpu_options_default(&options);
    options.layer_index = swa_layer;
    options.token_position = 0ull;
    options.token_count = 1ull;
    options.input = input;
    options.input_stride = input_extent;
    options.evidence_level = YVEX_ATTENTION_EVIDENCE_STAGES;
    options.execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
    rc = run_cuda_reference_compare(
        plan, ir, session, descriptor, backend, &options, &cuda_result,
        &cuda_reference, NULL, NULL,
        ATTENTION_NATIVE_Q8_ABSOLUTE_TOLERANCE,
        ATTENTION_NATIVE_Q8_RELATIVE_TOLERANCE, failure, err);
    if (rc != YVEX_OK || cuda_result.cuda_tensor_core_launches) {
        fprintf(stderr,
                "attention_native_sparse_projection_failed rc=%d launches=%llu where=%s message=%s\n",
                rc, cuda_result.cuda_tensor_core_launches,
                yvex_error_where(err), yvex_error_message(err));
        if (rc == YVEX_OK) rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    tensor_core_launches += cuda_result.cuda_tensor_core_launches;
    if (input_extent > SIZE_MAX / ATTENTION_NATIVE_TENSORCORE_ROWS /
                           sizeof(*native_input) ||
        !(native_input = (float *)malloc(
              (size_t)(input_extent * ATTENTION_NATIVE_TENSORCORE_ROWS) *
              sizeof(*native_input)))) {
        rc = YVEX_ERR_NOMEM;
        goto cleanup;
    }
    fill_history_values(native_input, input_extent, 1401ull);
    for (unsigned long long row = 1ull; row < ATTENTION_NATIVE_TENSORCORE_ROWS; ++row)
        memcpy(native_input + row * input_extent, native_input,
               (size_t)input_extent * sizeof(*native_input));
    options.token_count = ATTENTION_NATIVE_TENSORCORE_ROWS;
    options.input = native_input;
    rc = run_cuda_reference_compare(
        plan, ir, session, descriptor, backend, &options, &cuda_result,
        &cuda_reference, NULL, NULL, ATTENTION_NATIVE_Q8_ABSOLUTE_TOLERANCE,
        ATTENTION_NATIVE_Q8_RELATIVE_TOLERANCE, failure, err);
    if (rc != YVEX_OK || !cuda_result.cuda_tensor_core_launches) {
        fprintf(stderr,
                "attention_native_row_batch_failed rc=%d launches=%llu where=%s message=%s\n",
                rc, cuda_result.cuda_tensor_core_launches,
                yvex_error_where(err), yvex_error_message(err));
        if (rc == YVEX_OK) rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    tensor_core_launches += cuda_result.cuda_tensor_core_launches;
    free(native_input);
    native_input = NULL;
#endif

    rc = run_cuda_core_input_regression(
        plan, ir, session, descriptor, backend, summary, csa_layer, failure, err);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "attention_cuda_core_input_regression_failed rc=%d failure=%u "
                "where=%s message=%s\n",
                rc, (unsigned int)failure->code, yvex_error_where(err),
                yvex_error_message(err));
        goto cleanup;
    }
    printf("attention_cuda_envelope_core_input_regression=1\n");

    fill_history_values(input, input_extent, 401ull + swa_layer);
    attention_execution_api()->cpu_options_default(&options);
    options.layer_index = swa_layer;
    options.token_position = 0ull;
    options.token_count = 1ull;
    options.input = input;
    options.input_stride = input_extent;

    rc = run_real_mutation_proof(
        plan, ir, session, descriptor, &options, &mutation_detected,
        failure, err);
    if (rc != YVEX_OK) goto cleanup;

    options.input = NULL;
    rc = attention_execution_api()->cpu_chunk_execute(
        plan, ir, session, descriptor, &options, &cpu_result, failure, err);
    if (rc != YVEX_ERR_INVALID_ARG || cpu_result.executed ||
        failure->code != YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    options.input = input;
    options.token_position = 1ull;
    rc = attention_execution_api()->cpu_chunk_execute(
        plan, ir, session, descriptor, &options, &cpu_result, failure, err);
    if (rc != YVEX_ERR_STATE || cpu_result.executed ||
        failure->code != YVEX_ATTENTION_FAILURE_HISTORY) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    rc = attention_execution_api()->cuda_token_execute(
        plan, ir, session, descriptor, backend, &options, &cuda_result,
        failure, err);
    if (rc != YVEX_ERR_STATE || cuda_result.executed ||
        failure->code != YVEX_ATTENTION_FAILURE_HISTORY) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    options.token_position = 0ull;
    printf("attention_explicit_input_refusal=1\n");
    printf("attention_missing_history_refusal=1\n");

    {
        float finite_input = input[0];

        input[0] = NAN;
        memset(&failed_trace, 0, sizeof(failed_trace));
        options.trace = &failed_trace;
        rc = attention_execution_api()->cuda_token_execute(
            plan, ir, session, descriptor, backend, &options, &cuda_result,
            failure, err);
        options.trace = NULL;
        input[0] = finite_input;
        if (rc != YVEX_ERR_FORMAT || failed_trace.owned ||
            cuda_result.executed ||
            failure->code != YVEX_ATTENTION_FAILURE_NUMERIC) {
            fprintf(stderr,
                    "attention_cuda_nonfinite_ingress_not_refused rc=%d "
                    "failure=%u owned=%d executed=%d\n",
                    rc, (unsigned int)failure->code, failed_trace.owned,
                    cuda_result.executed);
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        yvex_error_clear(err);
        memset(failure, 0, sizeof(*failure));
        printf("attention_cuda_nonfinite_ingress_refused=1\n");
    }

    for (fault_index = 0u;
         fault_index < sizeof(fault_cases) / sizeof(fault_cases[0]);
         ++fault_index) {
        if (setenv("YVEX_TEST_CUDA_ATTENTION_FAILURE",
                   fault_cases[fault_index].seam, 1) != 0) {
            rc = YVEX_ERR_IO;
            goto cleanup;
        }
        options.trace = &failed_trace;
        rc = attention_execution_api()->cuda_token_execute(
            plan, ir, session, descriptor, backend, &options, &cuda_result,
            failure, err);
        (void)unsetenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
        options.trace = NULL;
        if (rc == YVEX_OK || failed_trace.owned || cuda_result.executed ||
            failure->code != fault_cases[fault_index].expected_failure) {
            fprintf(stderr,
                    "attention_cuda_injected_fault_failed seam=%s rc=%d expected=%u "
                    "actual=%u trace_owned=%d executed=%d where=%s message=%s\n",
                    fault_cases[fault_index].seam, rc,
                    (unsigned int)fault_cases[fault_index].expected_failure,
                    (unsigned int)failure->code, failed_trace.owned,
                    cuda_result.executed, yvex_error_where(err),
                    yvex_error_message(err));
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
    }
    rc = run_cuda_reference_compare(
        plan, ir, session, descriptor, backend, &options, &cuda_result,
        &cuda_reference, NULL, NULL,
        ATTENTION_CUDA_ABSOLUTE_TOLERANCE,
        ATTENTION_CUDA_RELATIVE_TOLERANCE, failure, err);
    if (rc != YVEX_OK ||
        strcmp(repeat_identity, cuda_result.output_identity) != 0 ||
        !attention_cuda_workspace_add(&workspace_evidence, &cuda_result))
        goto cleanup;
    if (!yvex_core_u64_add(
            launches, cuda_result.cuda_kernel_launches, &launches)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (!yvex_core_u64_add(
            tensor_core_launches, cuda_result.cuda_tensor_core_launches,
            &tensor_core_launches)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (cuda_result.cuda_peak_device_bytes > peak_device_bytes)
        peak_device_bytes = cuda_result.cuda_peak_device_bytes;
    if (cuda_result.cuda_peak_host_bytes > peak_host_bytes)
        peak_host_bytes = cuda_result.cuda_peak_host_bytes;
    if (cuda_result.payload_bytes_read > peak_encoded_weight_staging_bytes)
        peak_encoded_weight_staging_bytes = cuda_result.payload_bytes_read;
    if (!run_cuda_fault_case(
            plan, ir, session, descriptor, &options,
            "YVEX_TEST_CUDA_SYNC_FAILURE", "encoded-attention", 0ull,
            YVEX_ATTENTION_FAILURE_BACKEND, err) ||
        !run_cuda_fault_case(
            plan, ir, session, descriptor, &options,
            "YVEX_TEST_CUDA_CLEANUP_FAILURE", "encoded-attention", 0ull,
            YVEX_ATTENTION_FAILURE_CLEANUP, err) ||
        !run_cuda_fault_case(
            plan, ir, session, descriptor, &options, NULL, NULL, 1ull,
            YVEX_ATTENTION_FAILURE_CANCELLED, err) ||
        !run_cuda_fault_case(
            plan, ir, session, descriptor, &options, NULL, NULL, 6ull,
            YVEX_ATTENTION_FAILURE_CANCELLED, err) ||
        !run_cuda_bundle_refusal(err) ||
        !run_cuda_workspace_cleanup_fault(plan, err)) {
        fprintf(stderr, "attention_cuda_fault_matrix_failed=1\n");
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }

    layer = yvex_attention_plan_layer_at(plan, csa_layer);
    if (!layer || layer->hidden_dimension != input_extent ||
        !live_history_init(&history, layer, summary, 2052ull)) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    fill_history_values(input, input_extent, 503ull);
    attention_execution_api()->cpu_options_default(&options);
    options.layer_index = csa_layer;
    options.token_position = 2052ull;
    options.token_count = 1ull;
    options.input = input;
    options.input_stride = input_extent;
    options.history = &history.view;
    {
        yvex_attention_history_view stale = history.view;
        stale.main_rolling_state.current_fill++;
        options.history = &stale;
        rc = attention_execution_api()->cpu_chunk_execute(
            plan, ir, session, descriptor, &options, &cpu_result, failure,
            err);
        if (rc != YVEX_ERR_STATE || cpu_result.executed ||
            failure->code != YVEX_ATTENTION_FAILURE_HISTORY) {
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        rc = attention_execution_api()->cuda_token_execute(
            plan, ir, session, descriptor, backend, &options, &cuda_result,
            failure, err);
        if (rc != YVEX_ERR_STATE || cuda_result.executed ||
            failure->code != YVEX_ATTENTION_FAILURE_HISTORY) {
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        stale = history.view;
        stale.main_rolling_state.previous_fill--;
        options.history = &stale;
        rc = attention_execution_api()->cuda_token_execute(
            plan, ir, session, descriptor, backend, &options, &cuda_result,
            failure, err);
        if (rc != YVEX_ERR_STATE || cuda_result.executed ||
            failure->code != YVEX_ATTENTION_FAILURE_HISTORY) {
            rc = YVEX_ERR_STATE;
            goto cleanup;
        }
        options.history = &history.view;
        printf("attention_stale_rolling_refusal=1\n");
    }
    rc = run_cpu_cuda_reference_compare(
        plan, ir, session, descriptor, backend, &options, &cpu_result,
        &cuda_result, &cpu_reference, &cuda_reference, &cpu_cuda,
        &cpu_oracle, &cuda_oracle, failure, err);
    if (rc != YVEX_OK || cuda_result.topk_candidates != 513ull ||
        cuda_result.topk_selected != 512ull ||
        !attention_cuda_workspace_add(&workspace_evidence, &cuda_result))
        goto cleanup;
    if (!attention_evidence_add_case(
            &evidence, "csa-topk-513", &cpu_result, &cuda_result,
            &cpu_reference, &cuda_reference, &cpu_cuda, &cpu_oracle,
            &cuda_oracle)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    rc = run_real_mutation_proof(
        plan, ir, session, descriptor, &options, &mutation_detected,
        failure, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!yvex_core_u64_add(
            launches, cuda_result.cuda_kernel_launches, &launches)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (cuda_result.cuda_peak_device_bytes > peak_device_bytes)
        peak_device_bytes = cuda_result.cuda_peak_device_bytes;
    if (cuda_result.cuda_peak_host_bytes > peak_host_bytes)
        peak_host_bytes = cuda_result.cuda_peak_host_bytes;
    if (cuda_result.payload_bytes_read > peak_encoded_weight_staging_bytes)
        peak_encoded_weight_staging_bytes = cuda_result.payload_bytes_read;
    live_history_release(&history);

    layer = yvex_attention_plan_layer_at(plan, hca_layer);
    if (!layer || layer->hidden_dimension != input_extent ||
        !live_history_init(&history, layer, summary, 127ull)) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    fill_history_values(input, input_extent, 607ull);
    attention_execution_api()->cpu_options_default(&options);
    options.layer_index = hca_layer;
    options.token_position = 127ull;
    options.token_count = 1ull;
    options.input = input;
    options.input_stride = input_extent;
    options.history = &history.view;
    rc = run_cpu_cuda_reference_compare(
        plan, ir, session, descriptor, backend, &options, &cpu_result,
        &cuda_result, &cpu_reference, &cuda_reference, &cpu_cuda,
        &cpu_oracle, &cuda_oracle, failure, err);
    if (rc != YVEX_OK || cuda_result.compressed_entries != 1ull ||
        !attention_cuda_workspace_add(&workspace_evidence, &cuda_result))
        goto cleanup;
    if (!attention_evidence_add_case(
            &evidence, "hca-ratio-128", &cpu_result, &cuda_result,
            &cpu_reference, &cuda_reference, &cpu_cuda, &cpu_oracle,
            &cuda_oracle)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    rc = run_real_mutation_proof(
        plan, ir, session, descriptor, &options, &mutation_detected,
        failure, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!yvex_core_u64_add(
            launches, cuda_result.cuda_kernel_launches, &launches)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (cuda_result.cuda_peak_device_bytes > peak_device_bytes)
        peak_device_bytes = cuda_result.cuda_peak_device_bytes;
    if (cuda_result.cuda_peak_host_bytes > peak_host_bytes)
        peak_host_bytes = cuda_result.cuda_peak_host_bytes;
    if (cuda_result.payload_bytes_read > peak_encoded_weight_staging_bytes)
        peak_encoded_weight_staging_bytes = cuda_result.payload_bytes_read;
    live_history_release(&history);

    if (evidence.layer_count != summary->layer_count ||
        evidence.swa_count != summary->swa_layer_count ||
        evidence.csa_count != summary->csa_layer_count ||
        evidence.hca_count != summary->hca_layer_count ||
        !attention_evidence_final(&evidence, evidence_identity)) {
        fprintf(stderr,
                "attention_cuda_evidence_incomplete layers=%llu swa=%llu "
                "csa=%llu hca=%llu\n",
                evidence.layer_count, evidence.swa_count, evidence.csa_count,
                evidence.hca_count);
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    if (workspace_evidence.cold_executions != 1ull ||
        workspace_evidence.warm_executions < 2ull ||
        workspace_evidence.allocation_count != 1ull) {
        fprintf(stderr,
                "attention_cuda_workspace_evidence_invalid cold=%llu warm=%llu "
                "allocations=%llu capacity=%llu peak=%llu\n",
                workspace_evidence.cold_executions,
                workspace_evidence.warm_executions,
                workspace_evidence.allocation_count,
                workspace_evidence.capacity, workspace_evidence.peak);
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    if (!yvex_core_u64_add(
            peak_host_bytes, peak_encoded_weight_staging_bytes,
            &peak_total_host_owned_bytes)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    rc = yvex_backend_close_checked(&backend, err);
    if (rc != YVEX_OK) goto cleanup;
    printf("attention_cuda_layers_executed=%llu\n", evidence.layer_count);
    printf("attention_publication_abi_consumed=1\n");
    printf("attention_real_stage_mutations_detected=%llu\n",
           mutation_detected);
    printf("attention_cuda_swa_layers=%llu\n", evidence.swa_count);
    printf("attention_cuda_csa_layers=%llu\n", evidence.csa_count);
    printf("attention_cuda_hca_layers=%llu\n", evidence.hca_count);
    printf("attention_cuda_classes_executed=3\n");
    printf("attention_cuda_kernel_launches=%llu\n", launches);
    printf("attention_cuda_tensor_core_launches=%llu\n",
           tensor_core_launches);
    printf("attention_cuda_peak_host_bytes=%llu\n", peak_host_bytes);
    printf("attention_cuda_peak_device_bytes=%llu\n", peak_device_bytes);
    printf("attention_cuda_peak_encoded_weight_staging_bytes=%llu\n",
           peak_encoded_weight_staging_bytes);
    printf("attention_cuda_peak_total_host_owned_bytes=%llu\n",
           peak_total_host_owned_bytes);
    printf("attention_cuda_configured_host_scratch_bytes=%llu\n",
           configured_host_scratch_bytes);
    printf("attention_cuda_host_workspace_capacity=%llu\n",
           workspace_evidence.capacity);
    printf("attention_cuda_host_workspace_peak=%llu\n",
           workspace_evidence.peak);
    printf("attention_cuda_host_workspace_allocations=%llu\n",
           workspace_evidence.allocation_count);
    printf("attention_cuda_host_workspace_cold_executions=%llu\n",
           workspace_evidence.cold_executions);
    printf("attention_cuda_host_workspace_warm_executions=%llu\n",
           workspace_evidence.warm_executions);
    printf("attention_cuda_warm_host_allocation_delta=0\n");
    printf("attention_required_bindings_executed=%llu\n",
           summary->required_binding_count);
    printf("attention_qtype_compute_refusals=%llu\n",
           summary->qtype_compute_refusal_count);
    printf("attention_qtype_paths=%llu\n", evidence.qtype_paths);
    for (qtype = 0u; qtype < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++qtype) {
        if (!summary->qtype_binding_counts[qtype]) continue;
        printf("attention_qtype_bindings.%s=%llu\n",
               yvex_gguf_qtype_name(qtype),
               summary->qtype_binding_counts[qtype]);
    }
    printf("attention_cpu_reference_values=%llu\n",
           evidence.cpu_reference_values);
    printf("attention_cpu_reference_max_abs=%.17g\n",
           evidence.cpu_maximum_absolute_error);
    printf("attention_cpu_reference_max_rel=%.17g\n",
           evidence.cpu_maximum_relative_error);
    printf("attention_cpu_reference_rmse=%.17g\n",
           evidence.cpu_reference_values ?
               sqrt(evidence.cpu_squared_error /
                    (double)evidence.cpu_reference_values) : 0.0);
    printf("attention_cuda_reference_values=%llu\n",
           evidence.cuda_reference_values);
    printf("attention_cuda_reference_max_abs=%.17g\n",
           evidence.cuda_maximum_absolute_error);
    printf("attention_cuda_reference_max_rel=%.17g\n",
           evidence.cuda_maximum_relative_error);
    printf("attention_cuda_reference_rmse=%.17g\n",
           evidence.cuda_reference_values ?
               sqrt(evidence.cuda_squared_error /
                    (double)evidence.cuda_reference_values) : 0.0);
    printf("attention_cpu_cuda_values=%llu\n", evidence.cpu_cuda_values);
    printf("attention_cpu_cuda_max_abs=%.17g\n",
           evidence.cpu_cuda_maximum_absolute_error);
    printf("attention_cpu_cuda_max_rel=%.17g\n",
           evidence.cpu_cuda_maximum_relative_error);
    printf("attention_cpu_cuda_rmse=%.17g\n",
           evidence.cpu_cuda_values ?
               sqrt(evidence.cpu_cuda_squared_error /
                    (double)evidence.cpu_cuda_values) : 0.0);
    printf("attention_execution_evidence_identity=%s\n", evidence_identity);
    printf("attention_reference_oracle_trace_identity=%s\n",
           evidence.oracle_trace_identity);
    printf("attention_reference_oracle_output_identity=%s\n",
           evidence.oracle_output_identity);
    printf("attention_reference_fixture_identity=%s\n",
           evidence.fixture_identity);
    printf("attention_reference_history_identity=%s\n",
           evidence.history_identity);
    printf("attention_reference_comparison_contract_identity=%s\n",
           evidence.comparison_contract_identity);
    printf("attention_reference_evidence_schema=%u\n",
           YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2);
    printf("attention_reference_exact_topk_positions=%llu\n",
           evidence.exact_topk_positions);
    printf("attention_cuda_swa_repeat_deterministic=1\n");
    printf("attention_cuda_fault_cases=%zu\n",
           sizeof(fault_cases) / sizeof(fault_cases[0]) + 7u);
    printf("attention_cuda_fault_cleanup=1\n");
    printf("attention_cuda_cancel_before_dispatch=1\n");
    printf("attention_cuda_cancel_before_publish=1\n");
    printf("attention_cuda_missing_symbol_refused=1\n");
    printf("attention_cuda_csa_topk_candidates=513\n");
    printf("attention_cuda_csa_topk_selected=512\n");
    printf("attention_cuda_hca_ratio128_boundary=1\n");
    rc = YVEX_OK;

cleanup:
    (void)unsetenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
    yvex_attention_execution_trace_release(&failed_trace);
    live_history_release(&history);
    free(native_input);
    free(input);
    {
        int close_rc = yvex_backend_close_checked(&backend, err);
        if (rc == YVEX_OK && close_rc != YVEX_OK) rc = close_rc;
    }
    return rc;
}

int main(int argc, char **argv)
{
    const char *source_path;
    const char *models_root;
    const char *manifest_path;
    const char *runtime_binding_path = getenv("YVEX_ATTENTION_RUNTIME_BINDING");
    int plan_only = 0;
    char artifact_path[YVEX_ARTIFACT_PATH_CAP];
    yvex_deepseek_payload_handoff_options handoff_options;
    yvex_deepseek_payload_handoff *handoff = NULL;
    yvex_deepseek_payload_failure handoff_failure;
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_projection materialization_projection;
    yvex_materialization_plan *materialization_plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_materialization_failure materialization_failure;
    yvex_deepseek_v4_ir *architecture_ir = NULL;
    yvex_semantic_model_ir *semantic_model = NULL;
    yvex_deepseek_v4_ir_failure architecture_failure;
    yvex_runtime_descriptor *descriptor = NULL;
    yvex_runtime_descriptor_failure descriptor_failure;
    yvex_attention_plan *attention_plan = NULL;
    yvex_attention_failure attention_failure;
    const yvex_materialization_summary *materialization_summary;
    const yvex_runtime_descriptor_summary *descriptor_summary;
    const yvex_attention_summary *attention_summary;
    yvex_error err;
    int rc;

    if (argc == 5 && strcmp(argv[1], "--plan-only") == 0) {
        plan_only = 1;
        source_path = argv[2];
        models_root = argv[3];
        manifest_path = argv[4];
    } else if (argc == 4) {
        source_path = argv[1];
        models_root = argv[2];
        manifest_path = argv[3];
    } else {
        fprintf(stderr,
                "usage: attention_deepseek [--plan-only] SOURCE MODELS_ROOT SOURCE_MANIFEST\n");
        return 2;
    }

    if (!path_join_selected(artifact_path, sizeof(artifact_path),
                            models_root)) {
        fprintf(stderr, "artifact_path_build=fail\n");
        return 1;
    }

    memset(&handoff_options, 0, sizeof(handoff_options));
    handoff_options.source_path = source_path;
    handoff_options.models_root = models_root;
    handoff_options.manifest_path = manifest_path;
    yvex_source_payload_budget_default(&handoff_options.budget);
    handoff_options.budget.maximum_open_handles = 32u;
    handoff_options.budget.maximum_streams = 16u;
    handoff_options.budget.maximum_inflight_host_bytes =
        handoff_options.budget.chunk_bytes *
        handoff_options.budget.maximum_streams;
    handoff_options.chunk_bytes = handoff_options.budget.chunk_bytes;
    handoff_options.page_bytes = handoff_options.budget.page_bytes;

    rc = yvex_model_register_deepseek_v4()->payload.open(
        &handoff, &handoff_options, &handoff_failure, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "handoff_failure=%d where=%s message=%s\n",
                handoff_failure.code, yvex_error_where(&err),
                yvex_error_message(&err));
        return 1;
    }

    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = artifact_path;
    artifact_options.readonly = 1;
    artifact_options.map = 0;
    rc = yvex_artifact_open(&artifact, &artifact_options, &err);
    if (rc == YVEX_OK &&
        yvex_artifact_size(artifact) != YVEX_SELECTED_DEEPSEEK_FILE_BYTES) {
        fprintf(stderr, "artifact_size_mismatch expected=%llu actual=%llu\n",
                YVEX_SELECTED_DEEPSEEK_FILE_BYTES,
                yvex_artifact_size(artifact));
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "artifact_open_failure where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        goto cleanup_fail;
    }

    rc = yvex_artifact_admit_deepseek(
        artifact, &admission, &admission_failure, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "admission_failure=%s field=%s expected=%llu actual=%llu message=%s\n",
                yvex_artifact_admission_code_name(admission_failure.code),
                admission_failure.field, admission_failure.expected,
                admission_failure.actual, yvex_error_message(&err));
        goto cleanup_fail;
    }

    yvex_materialization_options_default(&materialization_options);
    materialization_options.require_terminal_projection = 1;
    materialization_options.max_chunk_bytes = 16ull * 1024ull * 1024ull;
    materialization_options.cache_budget_bytes = 256ull * 1024ull * 1024ull;
    materialization_options.backend_resident_budget_bytes = 0ull;
    materialization_options.future_graph_scratch_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    materialization_options.future_kv_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;

    rc = yvex_materialization_project_artifact_lowering(
        yvex_model_register_deepseek_v4()->payload.map(handoff),
        &materialization_projection, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &materialization_plan, &admission, artifact, gguf, tensors,
            &materialization_projection, &materialization_options,
            &materialization_failure, &err);
    if (rc != YVEX_OK) {
        print_materialization_failure("materialization-plan",
                                      &materialization_failure, &err);
        goto cleanup_fail;
    }
    rc = yvex_materialization_session_open(
        &session, materialization_plan, artifact, &materialization_options,
        &materialization_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(
            session, &materialization_failure, &err);
    if (rc != YVEX_OK) {
        print_materialization_failure("materialization-commit",
                                      &materialization_failure, &err);
        goto cleanup_fail;
    }

    rc = yvex_model_register_deepseek_v4()->ir.build(
        &architecture_ir, yvex_model_register_deepseek_v4()->payload.verification(handoff),
        &architecture_failure, &err);
    if (rc != YVEX_OK) {
        print_architecture_failure(&architecture_failure, &err);
        goto cleanup_fail;
    }

    rc = semantic_model_borrow(
        &semantic_model,
        yvex_model_register_deepseek_v4()->payload.verification(handoff), &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "semantic_model_failure where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        goto cleanup_fail;
    }

    rc = yvex_runtime_descriptor_build_deepseek(
        &descriptor, &admission, session,
        yvex_model_register_deepseek_v4()->payload.map(handoff), semantic_model,
        &descriptor_failure, &err);
    if (rc != YVEX_OK) {
        print_descriptor_failure("descriptor", &descriptor_failure, &err);
        goto cleanup_fail;
    }

    rc = yvex_compiler_family_deepseek_v4()->graph()->plan_build(
        &attention_plan, semantic_model, session, descriptor,
        &attention_failure, &err);
    if (rc != YVEX_OK) {
        print_attention_failure(&attention_failure, &err);
        goto cleanup_fail;
    }
    if (!run_topk_contract_cases()) {
        fprintf(stderr, "attention_topk_contract=fail\n");
        goto cleanup_fail;
    }

    {
        yvex_deepseek_v4_layer_spec *mutable_layer =
            (yvex_deepseek_v4_layer_spec *)
                yvex_model_register_deepseek_v4()->ir.layer_at(architecture_ir, 2ull);
        const yvex_runtime_descriptor_summary *canonical_runtime =
            yvex_runtime_descriptor_summary_get(descriptor);
        const yvex_attention_summary *canonical_attention =
            yvex_attention_plan_summary(attention_plan);
        const yvex_attention_layer_plan *canonical_layer =
            yvex_attention_plan_layer_at(attention_plan, 2ull);
        const yvex_materialization_summary *canonical_materialization =
            yvex_materialization_session_summary(session);
        unsigned long long original_topk;
        char canonical_logical[YVEX_TRANSFORM_IR_IDENTITY_CAP];
        char mutated_logical[YVEX_TRANSFORM_IR_IDENTITY_CAP];
        char repeated_logical[YVEX_TRANSFORM_IR_IDENTITY_CAP];
        char canonical_runtime_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
        char canonical_plan_identity[YVEX_ATTENTION_IDENTITY_CAP];

        if (!mutable_layer || !canonical_runtime || !canonical_attention ||
            !canonical_layer ||
            !canonical_materialization ||
            !yvex_model_register_deepseek_v4()->transform.architecture_identity(
                architecture_ir, canonical_logical)) {
            fprintf(stderr, "attention_identity_fixture=invalid\n");
            goto cleanup_fail;
        }
        yvex_core_text_copy(canonical_runtime_identity,
                            sizeof(canonical_runtime_identity),
                            canonical_runtime->runtime_descriptor_identity);
        yvex_core_text_copy(canonical_plan_identity, sizeof(canonical_plan_identity),
                            canonical_attention->attention_plan_identity);
        original_topk = mutable_layer->sparse_topk.k;
        mutable_layer->sparse_topk.k = original_topk - 1ull;
        if (!yvex_model_register_deepseek_v4()->transform.architecture_identity(
                architecture_ir, mutated_logical) ||
            !yvex_model_register_deepseek_v4()->transform.architecture_identity(
                architecture_ir, repeated_logical) ||
            strcmp(canonical_logical, mutated_logical) == 0 ||
            strcmp(mutated_logical, repeated_logical) != 0) {
            fprintf(stderr, "attention_identity_mutation=invalid\n");
            goto identity_mutation_fail;
        }
        if (strcmp(canonical_runtime_identity,
                   canonical_runtime->runtime_descriptor_identity) != 0 ||
            strcmp(canonical_plan_identity,
                   canonical_attention->attention_plan_identity) != 0 ||
            canonical_layer->sparse_topk.k != original_topk ||
            canonical_materialization->payload_bytes_accessed != 0ull) {
            fprintf(stderr, "attention_identity_compiled_isolation=invalid\n");
            goto identity_mutation_fail;
        }
        mutable_layer->sparse_topk.k = original_topk;
        if (!yvex_model_register_deepseek_v4()->transform.architecture_identity(
                architecture_ir, repeated_logical) ||
            strcmp(canonical_logical, repeated_logical) != 0 ||
            strcmp(admission.artifact_identity,
                   canonical_attention->artifact_identity) != 0 ||
            strcmp(canonical_materialization->plan_identity,
                   canonical_attention->materialization_plan_identity) != 0) {
            fprintf(stderr, "attention_identity_restore=invalid\n");
            goto identity_mutation_fail;
        }
        printf("attention_identity_canonical_logical=%s\n", canonical_logical);
        printf("attention_identity_mutated_logical=%s\n", mutated_logical);
        printf("attention_identity_mutation_deterministic=1\n");
        printf("attention_identity_semantic_ir_sealed=1\n");
        printf("attention_identity_compiled_plan_isolated=1\n");
        printf("attention_identity_artifact_unchanged=1\n");
        printf("attention_identity_materialization_unchanged=1\n");
        yvex_error_clear(&err);
        memset(&attention_failure, 0, sizeof(attention_failure));
        goto identity_mutation_done;

identity_mutation_fail:
        mutable_layer->sparse_topk.k = original_topk;
        goto cleanup_fail;
identity_mutation_done:
        ;
    }

    materialization_summary =
        yvex_materialization_session_summary(session);
    descriptor_summary = yvex_runtime_descriptor_summary_get(descriptor);
    attention_summary = yvex_attention_plan_summary(attention_plan);

    printf("mode=%s\n",
           plan_only ? "plan-only" :
           (getenv("YVEX_ATTENTION_CUDA_ONLY") ?
                "cuda-execution" : "cpu-reference-cuda"));
    printf("artifact_path=%s\n", artifact_path);
    printf("artifact_identity=%s\n", admission.artifact_identity);
    printf("materialization_plan_identity=%s\n",
           materialization_summary->plan_identity);
    printf("runtime_descriptor_identity=%s\n",
           descriptor_summary->runtime_descriptor_identity);
    printf("runtime_numeric_identity=%s\n",
           descriptor_summary->runtime_numeric_identity);
    printf("runtime_numeric_schema_version=%u\n",
           descriptor_summary->runtime_numeric_schema_version);
    printf("runtime_hadamard_revision=%s\n",
           descriptor_summary->runtime_hadamard_revision);
    printf("attention_plan_status=%s\n",
           yvex_test_attention_status_name(attention_summary->status));
    printf("attention_plan_identity=%s\n",
           attention_summary->attention_plan_identity);
    printf("attention_layers=%llu\n", attention_summary->layer_count);
    printf("attention_swa_layers=%llu\n", attention_summary->swa_layer_count);
    printf("attention_csa_layers=%llu\n", attention_summary->csa_layer_count);
    printf("attention_hca_layers=%llu\n", attention_summary->hca_layer_count);
    printf("attention_required_bindings=%llu\n",
           attention_summary->required_binding_count);
    printf("attention_payload_bytes_bound=%llu\n",
           attention_summary->payload_bytes_bound);
    printf("attention_history_contract_ready=%d\n",
           attention_summary->history_contract_ready);
    printf("attention_state_delta_contract_ready=%d\n",
           attention_summary->state_delta_contract_ready);
    printf("attention_cpu_reference_ready=%d\n",
           attention_summary->cpu_reference_ready);
    printf("attention_cuda_execution_ready=%d\n",
           attention_summary->cuda_execution_ready);
    printf("attention_full_execution_ready=%d\n",
           attention_summary->full_execution_ready);
    printf("attention_execution_supported=%d\n",
           attention_summary->full_execution_ready);
    printf("attention_execution_refusal=\n");
    printf("runtime_generation_ready=%d\n",
           descriptor_summary->generation_ready);
    printf("payload_bytes_accessed=%llu\n",
           materialization_summary->payload_bytes_accessed);

    if (!plan_only) {
        yvex_attention_cpu_options exec_options;
        yvex_attention_cpu_result exec_result;
        yvex_test_attention_reference_metrics reference_metrics;
        unsigned long long first_swa = ~0ull;
        unsigned long long first_csa = ~0ull;
        unsigned long long first_hca = ~0ull;
        unsigned long long first_token_payload_read = 0ull;
        unsigned long long first_token_executed = 0ull;
        unsigned long long chunk_payload_read = 0ull;
        unsigned long long chunk_executed = 0ull;
        unsigned long long chunk_repeat_payload_read = 0ull;
        unsigned long long chunk_repeat_executed = 0ull;
        unsigned long long history_chunk_payload_read = 0ull;
        unsigned long long history_chunk_executed = 0ull;
        unsigned long long i;
        double first_token_checksum = 0.0;
        double chunk_checksum = 0.0;
        double history_chunk_checksum = 0.0;
        char chunk_identity[3][YVEX_ATTENTION_IDENTITY_CAP];
        int cuda_only = getenv("YVEX_ATTENTION_CUDA_ONLY") != NULL;

        memset(chunk_identity, 0, sizeof(chunk_identity));

        for (i = 0ull; i < yvex_attention_plan_layer_count(attention_plan); ++i) {
            const yvex_attention_layer_plan *layer =
                yvex_attention_plan_layer_at(attention_plan, i);
            if (!layer) continue;
            if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA &&
                first_swa == ~0ull)
                first_swa = i;
            if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
                first_csa == ~0ull)
                first_csa = i;
            if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA &&
                first_hca == ~0ull)
                first_hca = i;
        }
        if (!cuda_only) {
#define RUN_FIRST_TOKEN(layer_id) do {                                          \
    if ((layer_id) != ~0ull) {                                                  \
        attention_execution_api()->cpu_options_default(&exec_options);             \
        exec_options.layer_index = (layer_id);                                  \
        exec_options.token_position = 0ull;                                     \
        exec_options.token_count = 1ull;                                         \
        rc = run_reference_compare(                                              \
            attention_plan, architecture_ir, session, descriptor,               \
            &exec_options, &exec_result, &reference_metrics, NULL,               \
            NULL, &attention_failure, &err);                                     \
        if (rc != YVEX_OK) {                                                    \
            print_attention_failure(&attention_failure, &err);                  \
            goto cleanup_fail;                                                  \
        }                                                                       \
        printf("attention_cpu_first_token.layer.%llu.class=%s\n",              \
               first_token_executed,                                           \
               attention_class_name(exec_result.attention_class));                                 \
        printf("attention_cpu_first_token.layer.%llu.index=%llu\n",            \
               first_token_executed, exec_result.layer_index);                 \
        printf("attention_cpu_first_token.layer.%llu.q_a_rows=%llu\n",         \
               first_token_executed, exec_result.q_a_rows);                    \
        printf("attention_cpu_first_token.layer.%llu.q_b_rows=%llu\n",         \
               first_token_executed, exec_result.q_b_rows);                    \
        printf("attention_cpu_first_token.layer.%llu.kv_rows=%llu\n",          \
               first_token_executed, exec_result.kv_rows);                     \
        printf("attention_cpu_first_token.layer.%llu.compressor_rows=%llu\n",  \
               first_token_executed, exec_result.compressor_rows);             \
        printf("attention_cpu_first_token.layer.%llu.indexer_rows=%llu\n",     \
               first_token_executed, exec_result.indexer_rows);                \
        printf("attention_cpu_first_token.layer.%llu.local_entries=%llu\n",    \
               first_token_executed, exec_result.local_entries);               \
        printf("attention_cpu_first_token.layer.%llu.payload_bytes_read=%llu\n", \
               first_token_executed, exec_result.payload_bytes_read);          \
        printf("attention_cpu_first_token.layer.%llu.q_checksum=%.17g\n",      \
               first_token_executed, exec_result.q_projection_checksum);       \
        printf("attention_cpu_first_token.layer.%llu.kv_checksum=%.17g\n",     \
               first_token_executed, exec_result.kv_projection_checksum);      \
        printf("attention_cpu_first_token.layer.%llu.attention_checksum=%.17g\n", \
               first_token_executed, exec_result.attention_checksum);          \
        printf("attention_cpu_first_token.layer.%llu.output_checksum=%.17g\n", \
               first_token_executed, exec_result.output_checksum);             \
        printf("attention_cpu_first_token.layer.%llu.output_identity=%s\n",   \
               first_token_executed, exec_result.output_identity);             \
        printf("attention_cpu_first_token.layer.%llu.full_attention=%d\n",     \
               first_token_executed, exec_result.full_attention);              \
        printf("attention_cpu_first_token.layer.%llu.reference_values=%llu\n", \
               first_token_executed, reference_metrics.compared_values);        \
        printf("attention_cpu_first_token.layer.%llu.reference_max_abs=%.17g\n", \
               first_token_executed, reference_metrics.maximum_absolute_error); \
        printf("attention_cpu_first_token.layer.%llu.reference_max_rel=%.17g\n", \
               first_token_executed, reference_metrics.maximum_relative_error); \
        first_token_payload_read += exec_result.payload_bytes_read;             \
        first_token_checksum += exec_result.output_checksum;                    \
        first_token_executed++;                                                 \
    }                                                                           \
} while (0)
        RUN_FIRST_TOKEN(first_swa);
        RUN_FIRST_TOKEN(first_csa);
        RUN_FIRST_TOKEN(first_hca);
#undef RUN_FIRST_TOKEN
#define RUN_CHUNK(layer_id, token_start, tokens) do {                          \
    if ((layer_id) != ~0ull) {                                                  \
        attention_execution_api()->cpu_options_default(&exec_options);             \
        exec_options.layer_index = (layer_id);                                  \
        exec_options.token_position = (token_start);                            \
        exec_options.token_count = (tokens);                                    \
        rc = run_reference_compare(                                             \
            attention_plan, architecture_ir, session, descriptor,               \
            &exec_options, &exec_result, &reference_metrics, NULL, NULL,         \
            &attention_failure, &err);                                           \
        if (rc != YVEX_OK) {                                                    \
            print_attention_failure(&attention_failure, &err);                  \
            goto cleanup_fail;                                                  \
        }                                                                       \
        printf("attention_cpu_chunk.layer.%llu.class=%s\n", chunk_executed,    \
               attention_class_name(exec_result.attention_class));                                 \
        printf("attention_cpu_chunk.layer.%llu.index=%llu\n", chunk_executed,  \
               exec_result.layer_index);                                        \
        printf("attention_cpu_chunk.layer.%llu.token_start=%llu\n",            \
               chunk_executed, exec_result.token_position);                     \
        printf("attention_cpu_chunk.layer.%llu.token_count=%llu\n",            \
               chunk_executed, exec_result.local_entries);                      \
        printf("attention_cpu_chunk.layer.%llu.q_a_rows=%llu\n",               \
               chunk_executed, exec_result.q_a_rows);                           \
        printf("attention_cpu_chunk.layer.%llu.q_b_rows=%llu\n",               \
               chunk_executed, exec_result.q_b_rows);                           \
        printf("attention_cpu_chunk.layer.%llu.kv_rows=%llu\n",                \
               chunk_executed, exec_result.kv_rows);                            \
        printf("attention_cpu_chunk.layer.%llu.compressed_entries=%llu\n",      \
               chunk_executed, exec_result.compressed_entries);                 \
        printf("attention_cpu_chunk.layer.%llu.indexer_entries=%llu\n",        \
               chunk_executed, exec_result.state_indexer_entries);              \
        printf("attention_cpu_chunk.layer.%llu.payload_bytes_read=%llu\n",      \
               chunk_executed, exec_result.payload_bytes_read);                 \
        printf("attention_cpu_chunk.layer.%llu.attention_checksum=%.17g\n",    \
               chunk_executed, exec_result.attention_checksum);                 \
        printf("attention_cpu_chunk.layer.%llu.output_checksum=%.17g\n",       \
               chunk_executed, exec_result.output_checksum);                    \
        printf("attention_cpu_chunk.layer.%llu.output_identity=%s\n",          \
               chunk_executed, exec_result.output_identity);                    \
        printf("attention_cpu_chunk.layer.%llu.full_attention=%d\n",           \
               chunk_executed, exec_result.full_attention);                     \
        if (chunk_executed < 3ull)                                              \
            snprintf(chunk_identity[chunk_executed],                            \
                     sizeof(chunk_identity[chunk_executed]), "%s",             \
                     exec_result.output_identity);                              \
        chunk_payload_read += exec_result.payload_bytes_read;                   \
        chunk_checksum += exec_result.output_checksum;                          \
        chunk_executed++;                                                       \
    }                                                                           \
} while (0)
        RUN_CHUNK(first_swa, 0ull, 4ull);
        RUN_CHUNK(first_csa, 0ull, 4ull);
        RUN_CHUNK(first_hca, 0ull, 4ull);
#undef RUN_CHUNK
#define RUN_CHUNK_REPEAT(layer_id, token_start, tokens) do {                   \
    if ((layer_id) != ~0ull) {                                                  \
        attention_execution_api()->cpu_options_default(&exec_options);             \
        exec_options.layer_index = (layer_id);                                  \
        exec_options.token_position = (token_start);                            \
        exec_options.token_count = (tokens);                                    \
        rc = run_reference_compare(                                             \
            attention_plan, architecture_ir, session, descriptor,               \
            &exec_options, &exec_result, &reference_metrics, NULL, NULL,         \
            &attention_failure, &err);                                           \
        if (rc != YVEX_OK) {                                                    \
            print_attention_failure(&attention_failure, &err);                  \
            goto cleanup_fail;                                                  \
        }                                                                       \
        if (chunk_repeat_executed >= 3ull ||                                    \
            strcmp(chunk_identity[chunk_repeat_executed],                       \
                   exec_result.output_identity) != 0) {                         \
            fprintf(stderr,                                                     \
                    "attention chunk repeat identity mismatch at %llu\n",       \
                    chunk_repeat_executed);                                     \
            goto cleanup_fail;                                                  \
        }                                                                       \
        printf("attention_cpu_chunk_repeat.layer.%llu.identity_match=1\n",     \
               chunk_repeat_executed);                                          \
        printf("attention_cpu_chunk_repeat.layer.%llu.output_identity=%s\n",   \
               chunk_repeat_executed, exec_result.output_identity);             \
        chunk_repeat_payload_read += exec_result.payload_bytes_read;            \
        chunk_repeat_executed++;                                                \
    }                                                                           \
} while (0)
        RUN_CHUNK_REPEAT(first_swa, 0ull, 4ull);
        RUN_CHUNK_REPEAT(first_csa, 0ull, 4ull);
        RUN_CHUNK_REPEAT(first_hca, 0ull, 4ull);
#undef RUN_CHUNK_REPEAT
        if (first_swa != ~0ull) {
            const yvex_attention_layer_plan *layer =
                yvex_attention_plan_layer_at(attention_plan,
                                                      first_swa);
            yvex_attention_history_view history;
            float *history_values = NULL;
            unsigned long long history_positions[2] = {0ull, 1ull};
            if (!layer) goto cleanup_fail;
            history_values = (float *)calloc(
                2ull * (size_t)layer->head_dimension, sizeof(float));
            if (!history_values) goto cleanup_fail;
            fill_history_values(history_values, 2ull * layer->head_dimension,
                                first_swa);
            memset(&history, 0, sizeof(history));
            history.immutable = 1;
            history.token_count = 2ull;
            history.local_tail_count = 2ull;
            history.local_kv = history_values;
            history.local_kv_stride = layer->head_dimension;
            history.local_positions = history_positions;
            attention_execution_api()->cpu_options_default(&exec_options);
            exec_options.layer_index = first_swa;
            exec_options.token_position = 2ull;
            exec_options.token_count = 2ull;
            exec_options.history = &history;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &exec_result, &reference_metrics, NULL, NULL,
                &attention_failure, &err);
            free(history_values);
            if (rc != YVEX_OK) {
                print_attention_failure(&attention_failure, &err);
                goto cleanup_fail;
            }
            printf("attention_cpu_history_chunk.layer.0.class=%s\n",
                   attention_class_name(exec_result.attention_class));
            printf("attention_cpu_history_chunk.layer.0.index=%llu\n",
                   exec_result.layer_index);
            printf("attention_cpu_history_chunk.layer.0.token_start=%llu\n",
                   exec_result.token_position);
            printf("attention_cpu_history_chunk.layer.0.local_entries=%llu\n",
                   exec_result.local_entries);
            printf("attention_cpu_history_chunk.layer.0.payload_bytes_read=%llu\n",
                   exec_result.payload_bytes_read);
            printf("attention_cpu_history_chunk.layer.0.output_checksum=%.17g\n",
                   exec_result.output_checksum);
            printf("attention_cpu_history_chunk.layer.0.output_identity=%s\n",
                   exec_result.output_identity);
            printf("attention_cpu_history_chunk.layer.0.full_attention=%d\n",
                   exec_result.full_attention);
            history_chunk_payload_read += exec_result.payload_bytes_read;
            history_chunk_checksum += exec_result.output_checksum;
            history_chunk_executed++;
        }
        if (first_csa != ~0ull) {
            const yvex_attention_layer_plan *layer =
                yvex_attention_plan_layer_at(attention_plan,
                                                      first_csa);
            live_attention_history history;
            yvex_attention_cpu_result changed_result;
            yvex_attention_execution_trace selection_trace;
            double baseline_checksum;
            char baseline_identity[YVEX_ATTENTION_IDENTITY_CAP];
            unsigned long long selected_index = ~0ull;
            unsigned long long unselected_index = ~0ull;
            unsigned long long csa_reference_values = 0ull;
            unsigned long long j;

            memset(&selection_trace, 0, sizeof(selection_trace));
            memset(baseline_identity, 0, sizeof(baseline_identity));
            if (!layer || !live_history_init(
                    &history, layer, attention_summary, 2052ull))
                goto cleanup_fail;
            attention_execution_api()->cpu_options_default(&exec_options);
            exec_options.layer_index = first_csa;
            exec_options.token_position = 2052ull;
            exec_options.token_count = 1ull;
            exec_options.history = &history.view;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &exec_result, &reference_metrics,
                &selection_trace, NULL,
                &attention_failure, &err);
            if (rc != YVEX_OK || exec_result.topk_candidates != 513ull ||
                exec_result.topk_selected != 512ull ||
                !selection_trace.topk_counts ||
                selection_trace.topk_counts[0] != 512ull) {
                if (rc != YVEX_OK)
                    print_attention_failure(&attention_failure, &err);
                yvex_attention_execution_trace_release(
                    &selection_trace);
                live_history_release(&history);
                goto cleanup_fail;
            }
            baseline_checksum = exec_result.output_checksum;
            csa_reference_values = reference_metrics.compared_values;
            snprintf(baseline_identity, sizeof(baseline_identity), "%s",
                     exec_result.output_identity);
            for (j = 0ull; j < 513ull; ++j) {
                unsigned long long selected;
                int chosen = 0;

                for (selected = 0ull; selected < 512ull; ++selected) {
                    if (selection_trace.topk_positions[selected] ==
                        history.compressed_positions[j]) {
                        chosen = 1;
                        break;
                    }
                }
                if (chosen && selected_index == ~0ull)
                    selected_index = j;
                if (!chosen) unselected_index = j;
            }
            yvex_attention_execution_trace_release(
                &selection_trace);
            if (selected_index == ~0ull || unselected_index == ~0ull) {
                live_history_release(&history);
                goto cleanup_fail;
            }
            for (j = 0ull; j < layer->head_dimension; ++j)
                history.compressed_kv[
                    selected_index * layer->head_dimension + j] *= -1.0f;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &changed_result, &reference_metrics, NULL,
                NULL, &attention_failure, &err);
            if (rc != YVEX_OK ||
                changed_result.output_checksum == baseline_checksum) {
                live_history_release(&history);
                goto cleanup_fail;
            }
            for (j = 0ull; j < layer->head_dimension; ++j)
                history.compressed_kv[
                    selected_index * layer->head_dimension + j] *= -1.0f;
            for (j = 0ull; j < layer->head_dimension; ++j)
                history.compressed_kv[
                    unselected_index * layer->head_dimension + j] *= -1.0f;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &changed_result, &reference_metrics, NULL,
                NULL, &attention_failure, &err);
            if (rc != YVEX_OK ||
                strcmp(changed_result.output_identity, baseline_identity) != 0) {
                live_history_release(&history);
                goto cleanup_fail;
            }
            for (j = 0ull; j < layer->head_dimension; ++j)
                history.compressed_kv[
                    unselected_index * layer->head_dimension + j] *= -1.0f;
            live_history_release(&history);
            if (!live_history_init(
                    &history, layer, attention_summary, 28ull))
                goto cleanup_fail;
            exec_options.token_position = 28ull;
            exec_options.history = &history.view;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &changed_result, &reference_metrics, NULL,
                NULL, &attention_failure, &err);
            if (rc != YVEX_OK || changed_result.topk_candidates != 7ull ||
                changed_result.topk_selected != 7ull) {
                live_history_release(&history);
                goto cleanup_fail;
            }
            live_history_release(&history);
            if (!live_history_init(
                    &history, layer, attention_summary, 2048ull))
                goto cleanup_fail;
            exec_options.token_position = 2048ull;
            exec_options.history = &history.view;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &changed_result, &reference_metrics, NULL,
                NULL, &attention_failure, &err);
            if (rc != YVEX_OK || changed_result.topk_candidates != 512ull ||
                changed_result.topk_selected != 512ull) {
                live_history_release(&history);
                goto cleanup_fail;
            }
            live_history_release(&history);
            {
                yvex_attention_execution_trace masked_trace;

                memset(&masked_trace, 0, sizeof(masked_trace));
                if (!live_history_init(
                        &history, layer, attention_summary, 2056ull))
                    goto cleanup_fail;
                history.compressed_positions[513] = 2055ull;
                history.indexer_positions[513] = 2055ull;
                for (j = 0ull; j < layer->indexer_head_dimension; ++j)
                    history.indexer_kv[
                        513ull * layer->indexer_head_dimension + j] = 32.0f;
                exec_options.token_position = 2056ull;
                exec_options.history = &history.view;
                rc = run_reference_compare(
                    attention_plan, architecture_ir, session, descriptor,
                    &exec_options, &exec_result, &reference_metrics,
                    &masked_trace, NULL, &attention_failure, &err);
                if (rc == YVEX_OK ||
                    attention_failure.code !=
                        YVEX_ATTENTION_FAILURE_HISTORY ||
                    masked_trace.owned || exec_result.executed) {
                    yvex_attention_execution_trace_release(
                        &masked_trace);
                    live_history_release(&history);
                    goto cleanup_fail;
                }
                yvex_attention_execution_trace_release(
                    &masked_trace);
            }
            printf("attention_csa_real_candidates=513\n");
            printf("attention_csa_fewer_than_topk=7\n");
            printf("attention_csa_exact_topk=512\n");
            printf("attention_csa_topk_selected=512\n");
            printf("attention_csa_selected_kv_affects_output=1\n");
            printf("attention_csa_unselected_kv_invariant=1\n");
            printf("attention_csa_incomplete_candidate_refused=1\n");
            printf("attention_csa_reference_values=%llu\n",
                   csa_reference_values);
            live_history_release(&history);
        }
        if (first_hca != ~0ull) {
            const yvex_attention_layer_plan *layer =
                yvex_attention_plan_layer_at(attention_plan,
                                                      first_hca);
            yvex_attention_layer_plan wrong_ratio;
            live_attention_history boundary_history;
            live_attention_history multi_history;
            yvex_attention_execution_trace boundary_trace;
            yvex_attention_cpu_result after_result;
            yvex_attention_cpu_result changed_result;

            memset(&boundary_trace, 0, sizeof(boundary_trace));
            if (!layer) goto cleanup_fail;
            wrong_ratio = *layer;
            wrong_ratio.compression_ratio = 0ull;
            rc = yvex_attention_class_geometry_validate(
                &wrong_ratio, &attention_failure, &err);
            if (rc == YVEX_OK ||
                attention_failure.code !=
                    YVEX_ATTENTION_FAILURE_DIMENSION)
                goto cleanup_fail;
            printf("attention_hca_wrong_ratio_refused=1\n");
            if (!layer || !live_history_init(
                    &boundary_history, layer, attention_summary, 127ull))
                goto cleanup_fail;
            attention_execution_api()->cpu_options_default(&exec_options);
            exec_options.layer_index = first_hca;
            exec_options.token_position = 127ull;
            exec_options.token_count = 1ull;
            exec_options.history = &boundary_history.view;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &exec_result, &reference_metrics,
                &boundary_trace, NULL, &attention_failure, &err);
            if (rc != YVEX_OK || boundary_trace.compressed_count != 1ull ||
                !boundary_trace.next_main_rolling_state.present) {
                if (rc != YVEX_OK)
                    print_attention_failure(&attention_failure, &err);
                yvex_attention_execution_trace_release(
                    &boundary_trace);
                live_history_release(&boundary_history);
                goto cleanup_fail;
            }
            boundary_history.compressed_kv = (float *)malloc(
                (size_t)layer->head_dimension * sizeof(float));
            boundary_history.compressed_positions =
                (unsigned long long *)malloc(sizeof(unsigned long long));
            if (!boundary_history.compressed_kv ||
                !boundary_history.compressed_positions) {
                yvex_attention_execution_trace_release(
                    &boundary_trace);
                live_history_release(&boundary_history);
                goto cleanup_fail;
            }
            memcpy(boundary_history.compressed_kv,
                   boundary_trace.compressed_kv,
                   (size_t)layer->head_dimension * sizeof(float));
            boundary_history.compressed_positions[0] =
                boundary_trace.compressed_positions[0];
            memmove(boundary_history.local_kv,
                    boundary_history.local_kv + layer->head_dimension,
                    (size_t)(126ull * layer->head_dimension) * sizeof(float));
            memmove(boundary_history.local_positions,
                    boundary_history.local_positions + 1ull,
                    126ull * sizeof(*boundary_history.local_positions));
            memcpy(boundary_history.local_kv +
                       126ull * layer->head_dimension,
                   boundary_trace.raw_kv,
                   (size_t)layer->head_dimension * sizeof(float));
            boundary_history.local_positions[126] = 127ull;
            boundary_history.view.local_tail_count = 127ull;
            boundary_history.view.local_kv = boundary_history.local_kv;
            boundary_history.view.local_positions =
                boundary_history.local_positions;
            boundary_history.view.compressed_entry_count = 1ull;
            boundary_history.view.compressed_kv =
                boundary_history.compressed_kv;
            boundary_history.view.compressed_positions =
                boundary_history.compressed_positions;
            boundary_history.view.compressed_kv_stride =
                layer->head_dimension;
            live_history_bind_next_state(&boundary_history, &boundary_trace);
            attention_execution_api()->cpu_options_default(&exec_options);
            exec_options.layer_index = first_hca;
            exec_options.token_position = 128ull;
            exec_options.token_count = 1ull;
            exec_options.history = &boundary_history.view;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &after_result, &reference_metrics, NULL,
                NULL, &attention_failure, &err);
            if (rc != YVEX_OK || after_result.compressed_entries != 0ull ||
                after_result.topk_candidates != 1ull) {
                if (rc != YVEX_OK)
                    print_attention_failure(&attention_failure, &err);
                yvex_attention_execution_trace_release(
                    &boundary_trace);
                live_history_release(&boundary_history);
                goto cleanup_fail;
            }
            printf("attention_hca_fill_before_boundary=127\n");
            printf("attention_hca_boundary_emissions=1\n");
            printf("attention_hca_first_after_boundary=1\n");
            printf("attention_hca_external_compressed_used=%d\n",
                   after_result.topk_candidates == 1ull);
            yvex_attention_execution_trace_release(&boundary_trace);
            live_history_release(&boundary_history);

            if (!live_history_init(&multi_history, layer, attention_summary,
                                   384ull)) {
                fprintf(stderr, "attention_hca_multi_history_init=fail\n");
                goto cleanup_fail;
            }
            attention_execution_api()->cpu_options_default(&exec_options);
            exec_options.layer_index = first_hca;
            exec_options.token_position = 384ull;
            exec_options.token_count = 1ull;
            exec_options.history = &multi_history.view;
            rc = run_reference_compare(
                attention_plan, architecture_ir, session, descriptor,
                &exec_options, &exec_result, &reference_metrics, NULL,
                NULL, &attention_failure, &err);
            if (rc != YVEX_OK || exec_result.topk_candidates != 3ull ||
                exec_result.local_entries != 128ull) {
                if (rc != YVEX_OK)
                    print_attention_failure(&attention_failure, &err);
                else
                    fprintf(stderr,
                            "attention_hca_multi_result candidates=%llu local=%llu\n",
                            exec_result.topk_candidates,
                            exec_result.local_entries);
                live_history_release(&multi_history);
                goto cleanup_fail;
            }
            {
                double baseline = exec_result.output_checksum;
                unsigned long long overlap = 2ull;
                unsigned long long group_start =
                    multi_history.compressed_positions[overlap];
                unsigned long long group_end =
                    group_start + layer->compression_ratio - 1ull;
                unsigned long long raw_inside =
                    multi_history.local_positions[1];
                unsigned long long j;

                if (raw_inside <= group_start || raw_inside > group_end) {
                    fprintf(stderr,
                            "attention_hca_overlap_interval_mismatch=%llu,%llu,%llu\n",
                            group_start, group_end, raw_inside);
                    live_history_release(&multi_history);
                    goto cleanup_fail;
                }
                for (j = 0ull; j < layer->head_dimension; ++j) {
                    unsigned long long offset =
                        overlap * layer->head_dimension + j;
                    multi_history.compressed_kv[offset] =
                        -multi_history.compressed_kv[offset];
                }
                rc = run_reference_compare(
                    attention_plan, architecture_ir, session, descriptor,
                    &exec_options, &changed_result, &reference_metrics, NULL,
                    NULL, &attention_failure, &err);
                if (rc != YVEX_OK ||
                    changed_result.output_checksum == baseline) {
                    if (rc != YVEX_OK)
                        print_attention_failure(&attention_failure, &err);
                    else
                        fprintf(stderr,
                                "attention_hca_compressed_perturbation_dead=1\n");
                    live_history_release(&multi_history);
                    goto cleanup_fail;
                }
                for (j = 0ull; j < layer->head_dimension; ++j) {
                    unsigned long long offset =
                        overlap * layer->head_dimension + j;
                    multi_history.compressed_kv[offset] =
                        -multi_history.compressed_kv[offset];
                }
                for (j = 0ull; j < layer->head_dimension; ++j)
                    multi_history.local_kv[layer->head_dimension + j] =
                        -multi_history.local_kv[
                            layer->head_dimension + j];
                rc = run_reference_compare(
                    attention_plan, architecture_ir, session, descriptor,
                    &exec_options, &changed_result, &reference_metrics, NULL,
                    NULL, &attention_failure, &err);
                if (rc != YVEX_OK ||
                    changed_result.output_checksum == baseline) {
                    if (rc != YVEX_OK)
                        print_attention_failure(&attention_failure, &err);
                    else
                        fprintf(stderr,
                                "attention_hca_local_perturbation_dead=1\n");
                    live_history_release(&multi_history);
                    goto cleanup_fail;
                }
            }
            printf("attention_hca_multiple_compressed_groups=3\n");
            printf("attention_hca_local_window_entries=128\n");
            printf("attention_hca_full_output_reference=1\n");
            printf("attention_hca_compressed_affects_output=1\n");
            printf("attention_hca_local_affects_output=1\n");
            printf("attention_hca_raw_compressed_overlap_distinct=1\n");
            printf("attention_hca_raw_inside_compressed_interval=1\n");
            live_history_release(&multi_history);
        }
        {
            unsigned long long full_layer_count = first_token_executed;
            unsigned long long full_layer_payload = first_token_payload_read;
            double full_layer_checksum = first_token_checksum;

            for (i = 0ull;
                 i < yvex_attention_plan_layer_count(attention_plan);
                 ++i) {
                if (i == first_swa || i == first_csa || i == first_hca)
                    continue;
                attention_execution_api()->cpu_options_default(&exec_options);
                exec_options.layer_index = i;
                exec_options.token_position = 0ull;
                exec_options.token_count = 1ull;
                rc = run_reference_compare(
                    attention_plan, architecture_ir, session, descriptor,
                    &exec_options, &exec_result, &reference_metrics, NULL,
                    NULL, &attention_failure, &err);
                if (rc != YVEX_OK || !exec_result.full_attention) {
                    if (rc != YVEX_OK)
                        print_attention_failure(&attention_failure, &err);
                    goto cleanup_fail;
                }
                full_layer_count++;
                full_layer_payload += exec_result.payload_bytes_read;
                full_layer_checksum += exec_result.output_checksum;
            }
            if (full_layer_count !=
                yvex_attention_plan_layer_count(attention_plan))
                goto cleanup_fail;
            printf("attention_full_release_layers_executed=%llu\n",
                   full_layer_count);
            printf("attention_full_release_layers_payload_bytes=%llu\n",
                   full_layer_payload);
            printf("attention_full_release_layers_checksum=%.17g\n",
                   full_layer_checksum);
        }
        printf("attention_cpu_first_token_executed=%llu\n",
               first_token_executed);
        printf("attention_cpu_first_token_payload_bytes_read=%llu\n",
               first_token_payload_read);
        printf("attention_cpu_first_token_execution_checksum=%.17g\n",
               first_token_checksum);
        printf("attention_cpu_chunk_executed=%llu\n", chunk_executed);
        printf("attention_cpu_chunk_payload_bytes_read=%llu\n",
               chunk_payload_read);
        printf("attention_cpu_chunk_execution_checksum=%.17g\n",
               chunk_checksum);
        printf("attention_cpu_chunk_repeat_executed=%llu\n",
               chunk_repeat_executed);
        printf("attention_cpu_chunk_repeat_payload_bytes_read=%llu\n",
               chunk_repeat_payload_read);
        printf("attention_cpu_history_chunk_executed=%llu\n",
               history_chunk_executed);
        printf("attention_cpu_history_chunk_payload_bytes_read=%llu\n",
               history_chunk_payload_read);
        printf("attention_cpu_history_chunk_execution_checksum=%.17g\n",
               history_chunk_checksum);
        }
        rc = run_cuda_live_suite(
            attention_plan, architecture_ir, session, descriptor,
            attention_summary, first_swa, first_csa, first_hca,
            &attention_failure, &err);
        if (rc != YVEX_OK) {
            print_attention_failure(&attention_failure, &err);
            goto cleanup_fail;
        }
        printf("attention_cuda_evidence_exercised=1\n");
        if (runtime_binding_path && runtime_binding_path[0]) {
            rc = run_runtime_graph_oracle_suite(
                artifact_path, runtime_binding_path,
                attention_summary->attention_plan_identity,
                descriptor_summary->runtime_descriptor_identity,
                architecture_ir,
                first_swa, first_csa, first_hca, &err);
            if (rc != YVEX_OK) {
                fprintf(stderr,
                        "attention_runtime_oracle_failure where=%s message=%s\n",
                        yvex_error_where(&err), yvex_error_message(&err));
                goto cleanup_fail;
            }
        }
    }

    yvex_attention_plan_close(attention_plan);
    yvex_semantic_model_ir_close(&semantic_model);
    yvex_model_register_deepseek_v4()->ir.close(architecture_ir);
    yvex_runtime_descriptor_close(descriptor);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(materialization_plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    yvex_model_register_deepseek_v4()->payload.close(handoff);
    return 0;

cleanup_fail:
    yvex_attention_plan_close(attention_plan);
    yvex_semantic_model_ir_close(&semantic_model);
    yvex_model_register_deepseek_v4()->ir.close(architecture_ir);
    yvex_runtime_descriptor_close(descriptor);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(materialization_plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    yvex_model_register_deepseek_v4()->payload.close(handoff);
    return 1;
}
