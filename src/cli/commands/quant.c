/* Owner: CLI physical-variant command adapter.
 * Owns: argv admission and compact rendering for policy-driven full-model plan and emission.
 * Does not own: policy semantics, IR lowering, codecs, artifact layout, materialization, or runtime.
 * Invariants: every emit rederives and exactly validates its external plan before reading model payloads.
 * Boundary: CLI selects production owners but never resolves tensor qtypes or writes artifact bytes itself.
 * Purpose: make physical-variant planning, explanation, and complete GGUF emission operator reachable.
 * Inputs: explicit source authority, one sealed policy or preset, optional imatrix, and output paths.
 * Effects: may publish an external plan, policy, or complete artifact through typed domain APIs.
 * Failure: typed lower-owner refusal publishes no partial final plan or artifact. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/quant.h>

typedef enum {
    QUANT_CLI_NONE = 0,
    QUANT_CLI_PLAN,
    QUANT_CLI_EMIT,
    QUANT_CLI_SUMMARIZE,
    QUANT_CLI_EXPLAIN
} quant_cli_action;

typedef struct {
    quant_cli_action action;
    const char *target;
    const char *source;
    const char *models_root;
    const char *manifest;
    const char *preset;
    const char *policy_path;
    const char *imatrix_path;
    const char *plan_path;
    const char *out_plan;
    const char *out_artifact;
    const char *tensor;
    const char *role;
    const char *backend;
} quant_cli_options;

typedef struct {
    yvex_deepseek_payload_handoff *handoff;
    yvex_quant_policy *policy;
    yvex_imatrix_data *imatrix;
    yvex_quant_plan *plan;
    yvex_gguf_writer_plan *writer;
    yvex_imatrix_data_summary imatrix_summary;
} quant_cli_context;

static int quant_cli_writer_build(yvex_gguf_writer_plan **out,
                                  const quant_cli_context *context,
                                  yvex_gguf_writer_failure *failure,
                                  yvex_error *err);

/* Purpose: map one lower-owner refusal to a stable operator exit and bounded diagnostic. */
static int quant_cli_fail(const char *phase, const yvex_error *err)
{
    yvex_cli_out_writef(stderr, "yvex: quant %s failed: %s (%s)\n", phase,
                        yvex_error_message(err), yvex_error_where(err));
    return 1;
}

/* Purpose: consume one required option value without accepting truncation or implicit booleans. */
static const char *quant_cli_value(int argc, char **argv, int *index)
{
    if (!index || *index + 1 >= argc) return NULL;
    (*index)++;
    return argv[*index];
}

/* Purpose: parse the common full-model physical-variant selector and path grammar.
 * Inputs: process argv beginning after `yvex quant ACTION`.
 * Effects: fills caller-owned borrowed option pointers.
 * Failure: unknown, duplicate-semantic, or value-less arguments refuse.
 * Boundary: parsing assigns no qtype and opens no source or artifact. */
static int quant_cli_parse(int argc, char **argv, quant_cli_options *options)
{
    int index;

    memset(options, 0, sizeof(*options));
    if (argc < 3) return 0;
    if (strcmp(argv[2], "plan") == 0) options->action = QUANT_CLI_PLAN;
    else if (strcmp(argv[2], "emit") == 0) options->action = QUANT_CLI_EMIT;
    else if (strcmp(argv[2], "summarize") == 0) options->action = QUANT_CLI_SUMMARIZE;
    else if (strcmp(argv[2], "explain") == 0) options->action = QUANT_CLI_EXPLAIN;
    else return 0;
    for (index = 3; index < argc; ++index) {
        const char *value;

        if (strcmp(argv[index], "--help") == 0 || strcmp(argv[index], "-h") == 0)
            return -1;
        value = quant_cli_value(argc, argv, &index);
        if (!value) {
            yvex_cli_out_writef(stderr, "yvex: quant option requires a value: %s\n", argv[index]);
            return 0;
        }
        if (strcmp(argv[index - 1], "--target") == 0) options->target = value;
        else if (strcmp(argv[index - 1], "--source") == 0) options->source = value;
        else if (strcmp(argv[index - 1], "--models-root") == 0) options->models_root = value;
        else if (strcmp(argv[index - 1], "--source-manifest") == 0) options->manifest = value;
        else if (strcmp(argv[index - 1], "--preset") == 0) options->preset = value;
        else if (strcmp(argv[index - 1], "--policy") == 0) options->policy_path = value;
        else if (strcmp(argv[index - 1], "--imatrix-manifest") == 0)
            options->imatrix_path = value;
        else if (strcmp(argv[index - 1], "--plan") == 0) options->plan_path = value;
        else if (strcmp(argv[index - 1], "--out-plan") == 0) options->out_plan = value;
        else if (strcmp(argv[index - 1], "--out") == 0) options->out_artifact = value;
        else if (strcmp(argv[index - 1], "--tensor") == 0) options->tensor = value;
        else if (strcmp(argv[index - 1], "--role") == 0) options->role = value;
        else if (strcmp(argv[index - 1], "--backend") == 0) options->backend = value;
        else {
            yvex_cli_out_writef(stderr, "yvex: unknown quant option: %s\n", argv[index - 1]);
            return 0;
        }
    }
    return 1;
}

/* Purpose: require the complete common planning authority before opening expensive resources. */
static int quant_cli_options_valid(const quant_cli_options *options)
{
    if (!options || !options->target || !options->source || !options->models_root ||
        !options->manifest || (!!options->preset == !!options->policy_path) ||
        strcmp(options->target, "deepseek4-v4-flash") != 0)
        return 0;
    if (options->backend && strcmp(options->backend, "cpu") != 0 &&
        strcmp(options->backend, "cuda") != 0)
        return 0;
    if (options->action == QUANT_CLI_PLAN) return options->out_plan != NULL;
    if (options->action == QUANT_CLI_EMIT)
        return options->plan_path && options->out_artifact;
    if (options->action == QUANT_CLI_SUMMARIZE) return options->plan_path != NULL;
    if (options->action == QUANT_CLI_EXPLAIN)
        return options->plan_path && (!!options->tensor != !!options->role);
    return 0;
}

/* Purpose: release compiler-plane objects in reverse dependency order.
 * Inputs: an optional partially initialized CLI context.
 * Effects: closes every owned lower-layer handle and clears borrowed state.
 * Failure: cleanup is best-effort under lower-owner close contracts and returns no status.
 * Boundary: does not publish, unlink, or reinterpret an external plan or artifact. */
static void quant_cli_context_close(quant_cli_context *context)
{
    if (!context) return;
    yvex_gguf_writer_plan_release(&context->writer);
    yvex_quant_plan_release(&context->plan);
    yvex_imatrix_data_close(context->imatrix);
    yvex_quant_policy_close(context->policy);
    if (context->handoff)
        yvex_model_register_deepseek_v4()->payload.close(context->handoff);
    memset(context, 0, sizeof(*context));
}

/* Purpose: admit exact calibration bytes and bind them to source IR identity.
 * Inputs: optional imatrix path and an already-sealed transform summary.
 * Effects: opens one immutable mapped calibration owner in the context.
 * Failure: unsupported format, identity, or bounds leave the context releasable.
 * Boundary: component names and coverage remain validated by the quant plan owner. */
static int quant_cli_imatrix_open(quant_cli_context *context, const char *path,
                                  const yvex_transform_ir_summary *transform, yvex_error *err)
{
    yvex_imatrix_data_options options;
    int rc;

    if (!path) return YVEX_OK;
    memset(&options, 0, sizeof(options));
    options.path = path;
    options.source_model_identity = transform->transform_identity;
    options.calibration_dataset_identity =
        "deepseek-v4-flash-chat-v2-rendered-prompts-v1";
    options.producer = "llama.cpp-imatrix";
    options.producer_version = 1u;
    options.maximum_mapped_bytes = 1024u * 1024u * 1024u;
    rc = yvex_imatrix_data_open(&context->imatrix, &options, err);
    if (rc == YVEX_OK)
        rc = yvex_imatrix_data_get_summary(context->imatrix, &context->imatrix_summary, err);
    return rc;
}

/* Purpose: reduce per-terminal compute facts into backend compatibility without changing the plan.
 * Inputs: one sealed complete physical plan and caller-owned result slots.
 * Effects: writes CPU/CUDA all-terminal compatibility facts only.
 * Failure: a missing plan or decision yields both facts false.
 * Boundary: compatibility reporting never selects or rewrites a qtype. */
static void quant_cli_plan_compatibility(const yvex_quant_plan *plan, int *cpu, int *cuda)
{
    const yvex_quant_plan_summary *summary = yvex_quant_plan_summary_get(plan);
    unsigned long long ordinal;

    *cpu = summary && summary->complete ? 1 : 0;
    *cuda = *cpu;
    if (!summary) return;
    for (ordinal = 0u; ordinal < summary->decision_count; ++ordinal) {
        const yvex_quant_decision *decision = yvex_quant_plan_decision_at(plan, ordinal);
        if (!decision) {
            *cpu = 0;
            *cuda = 0;
            return;
        }
        if (!decision->cpu_compute_available) *cpu = 0;
        if (!decision->cuda_compute_available) *cuda = 0;
    }
}

/* Purpose: make an explicit CLI backend request a real all-terminal admission gate.
 * Inputs: optional canonical backend spelling and one sealed physical plan.
 * Effects: performs validation only.
 * Failure: unavailable required compute refuses before payload reads or plan publication.
 * Boundary: backend selection does not enter physical-variant identity. */
static int quant_cli_backend_validate(const char *backend, const yvex_quant_plan *plan,
                                      yvex_error *err)
{
    int cpu;
    int cuda;

    if (!backend) return YVEX_OK;
    quant_cli_plan_compatibility(plan, &cpu, &cuda);
    if ((strcmp(backend, "cpu") == 0 && cpu) ||
        (strcmp(backend, "cuda") == 0 && cuda))
        return YVEX_OK;
    yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "quant_cli_backend",
                    "physical variant is not executable on requested backend: %s", backend);
    return YVEX_ERR_UNSUPPORTED;
}

/* Purpose: regenerate one complete policy-resolved DeepSeek physical plan from source authority.
 * Inputs: validated common CLI options and empty lifecycle context.
 * Effects: opens source handoff, policy, optional imatrix, and sealed plan.
 * Failure: no payload bytes are read by planning and partial owners remain releasable.
 * Boundary: CLI never selects a terminal qtype; family lowering and policy owners do. */
static int quant_cli_context_open(quant_cli_context *context, const quant_cli_options *options,
                                  yvex_quant_failure *failure, yvex_error *err)
{
    yvex_deepseek_payload_handoff_options handoff_options;
    yvex_deepseek_payload_failure handoff_failure;
    const yvex_transform_ir_summary *transform;
    int rc;

    memset(context, 0, sizeof(*context));
    memset(&handoff_options, 0, sizeof(handoff_options));
    handoff_options.source_path = options->source;
    handoff_options.models_root = options->models_root;
    handoff_options.manifest_path = options->manifest;
    yvex_source_payload_budget_default(&handoff_options.budget);
    handoff_options.budget.maximum_open_handles = 32u;
    handoff_options.budget.maximum_streams = 16u;
    handoff_options.budget.maximum_inflight_host_bytes =
        handoff_options.budget.chunk_bytes * handoff_options.budget.maximum_streams;
    handoff_options.chunk_bytes = handoff_options.budget.chunk_bytes;
    handoff_options.page_bytes = handoff_options.budget.page_bytes;
    rc = yvex_model_register_deepseek_v4()->payload.open(
        &context->handoff, &handoff_options, &handoff_failure, err);
    if (rc == YVEX_OK)
        rc = options->preset
                 ? yvex_quant_policy_preset_open(&context->policy, options->preset, err)
                 : yvex_quant_policy_open(&context->policy, options->policy_path, err);
    transform = context->handoff
                    ? yvex_transform_ir_summary_get(
                          yvex_model_register_deepseek_v4()->payload.transform_ir(context->handoff))
                    : NULL;
    if (rc == YVEX_OK && !transform) rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK)
        rc = quant_cli_imatrix_open(context, options->imatrix_path, transform, err);
    if (rc == YVEX_OK)
        rc = yvex_quant_plan_build_deepseek_policy(
            &context->plan,
            yvex_model_register_deepseek_v4()->payload.transform_ir(context->handoff),
            yvex_model_register_deepseek_v4()->payload.binding(context->handoff),
            yvex_model_register_deepseek_v4()->payload.map(context->handoff), context->policy,
            context->imatrix_summary.complete ? context->imatrix_summary.imatrix_identity : NULL,
            NULL, failure, err);
    if (rc == YVEX_OK)
        rc = quant_cli_backend_validate(options->backend, context->plan, err);
    if (rc == YVEX_OK) {
        yvex_gguf_writer_failure writer_failure;
        memset(&writer_failure, 0, sizeof(writer_failure));
        rc = quant_cli_writer_build(&context->writer, context, &writer_failure, err);
    }
    return rc;
}

/* Purpose: render exact size, qtype, and identity facts from one sealed plan.
 * Inputs: one validated immutable quant plan.
 * Effects: writes bounded human-readable evidence to the canonical CLI output owner.
 * Failure: accepts no nullable plan and performs no domain mutation.
 * Boundary: rendered facts do not establish capability beyond the sealed plan. */
static void quant_cli_summary_print(const quant_cli_context *context)
{
    const yvex_quant_plan *plan = context->plan;
    const yvex_quant_plan_summary *summary = yvex_quant_plan_summary_get(plan);
    const yvex_gguf_writer_plan_summary *writer =
        yvex_gguf_writer_plan_summary_get(context->writer);
    unsigned int qtype;
    unsigned int role;
    unsigned int collection;
    unsigned int scope;
    unsigned long long ordinal;
    unsigned long long maximum_layer = 0u;
    unsigned long long role_bytes[YVEX_TENSOR_ROLE_COUNT] = {0};
    unsigned long long collection_bytes[YVEX_TENSOR_COLLECTION_COUNT] = {0};
    unsigned long long scope_bytes[3] = {0};
    int cpu_compatible;
    int cuda_compatible;

    quant_cli_plan_compatibility(plan, &cpu_compatible, &cuda_compatible);

    for (ordinal = 0u; ordinal < summary->decision_count; ++ordinal) {
        const yvex_quant_decision *decision = yvex_quant_plan_decision_at(plan, ordinal);
        if (!decision) continue;
        if (decision->role < YVEX_TENSOR_ROLE_COUNT)
            role_bytes[decision->role] += decision->encoded_bytes;
        if (decision->collection < YVEX_TENSOR_COLLECTION_COUNT)
            collection_bytes[decision->collection] += decision->encoded_bytes;
        if ((unsigned int)decision->scope < 3u)
            scope_bytes[(unsigned int)decision->scope] += decision->encoded_bytes;
        if (decision->scope != YVEX_TRANSFORM_SCOPE_GLOBAL &&
            decision->logical_key.layer_index > maximum_layer)
            maximum_layer = decision->logical_key.layer_index;
    }

    yvex_cli_out_writef(stdout, "status: physical-variant-plan-complete\n");
    yvex_cli_out_writef(stdout, "profile: %s\n", summary->profile_name);
    yvex_cli_out_writef(stdout, "policy_identity: %s\n", summary->policy_identity);
    yvex_cli_out_writef(stdout, "imatrix_identity: %s\n", summary->imatrix_identity);
    yvex_cli_out_writef(stdout, "variant_identity: %s\n", summary->physical_variant_identity);
    yvex_cli_out_writef(stdout, "terminal_decisions: %llu\n", summary->decision_count);
    yvex_cli_out_writef(stdout, "predicted_payload_bytes: %llu\n", summary->encoded_bytes);
    yvex_cli_out_writef(stdout, "predicted_gguf_bytes: %llu\n", writer->final_file_bytes);
    yvex_cli_out_writef(stdout, "predicted_metadata_bytes: %llu\n",
                        writer->structural_bytes + writer->pre_data_padding_bytes);
    yvex_cli_out_writef(stdout, "writer_plan_identity: %s\n",
                        writer->writer_plan_identity);
    yvex_cli_out_writef(stdout, "source_payload_bytes_read: %llu\n", summary->payload_bytes_read);
    yvex_cli_out_writef(stdout, "artifact_emittable: %d\n", summary->complete);
    yvex_cli_out_writef(stdout, "cpu_materializable: %d\n", cpu_compatible);
    yvex_cli_out_writef(stdout, "cuda_materializable: %d\n", cuda_compatible);
    yvex_cli_out_writef(stdout, "cpu_runtime_executable: %d\n", cpu_compatible);
    yvex_cli_out_writef(stdout, "cuda_runtime_executable: %d\n", cuda_compatible);
    for (qtype = 0u; qtype <= YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID; ++qtype) {
        if (!summary->qtype_tensor_counts[qtype]) continue;
        yvex_cli_out_writef(stdout, "qtype_%u_tensors: %llu\n", qtype,
                            summary->qtype_tensor_counts[qtype]);
        yvex_cli_out_writef(stdout, "qtype_%u_bytes: %llu\n", qtype,
                            summary->qtype_encoded_bytes[qtype]);
    }
    for (role = 0u; role < YVEX_TENSOR_ROLE_COUNT; ++role)
        if (role_bytes[role])
            yvex_cli_out_writef(stdout, "role_%u_%s_bytes: %llu\n", role,
                                yvex_tensor_role_name((yvex_tensor_role)role),
                                role_bytes[role]);
    for (collection = 0u; collection < YVEX_TENSOR_COLLECTION_COUNT; ++collection)
        if (collection_bytes[collection])
            yvex_cli_out_writef(stdout, "collection_%u_bytes: %llu\n", collection,
                                collection_bytes[collection]);
    for (scope = 0u; scope < 3u; ++scope)
        yvex_cli_out_writef(stdout, "scope_%u_bytes: %llu\n", scope,
                            scope_bytes[scope]);
    for (ordinal = 0u; ordinal <= maximum_layer; ++ordinal) {
        unsigned long long layer_bytes = 0u;
        unsigned long long decision_ordinal;
        for (decision_ordinal = 0u; decision_ordinal < summary->decision_count;
             ++decision_ordinal) {
            const yvex_quant_decision *decision =
                yvex_quant_plan_decision_at(plan, decision_ordinal);
            if (decision && decision->scope != YVEX_TRANSFORM_SCOPE_GLOBAL &&
                decision->logical_key.layer_index == ordinal)
                layer_bytes += decision->encoded_bytes;
        }
        if (layer_bytes)
            yvex_cli_out_writef(stdout, "layer_%llu_bytes: %llu\n", ordinal,
                                layer_bytes);
    }
}

/* Purpose: render each selected decision without reinterpreting policy precedence.
 * Inputs: one exclusive tensor-or-role selector and a validated immutable plan.
 * Effects: writes matched decision evidence and its exact match count.
 * Failure: a selector matching no terminal returns a nonzero operator result.
 * Boundary: selection is report-only and cannot modify or reseal decisions. */
static int quant_cli_explain(const quant_cli_options *options, const yvex_quant_plan *plan)
{
    const yvex_quant_plan_summary *summary = yvex_quant_plan_summary_get(plan);
    unsigned long long ordinal;
    unsigned long long matches = 0u;

    for (ordinal = 0u; ordinal < summary->decision_count; ++ordinal) {
        const yvex_quant_decision *decision = yvex_quant_plan_decision_at(plan, ordinal);
        const char *role = yvex_tensor_role_name(decision->role);
        if (options->tensor && strcmp(options->tensor, decision->physical_tensor_name) != 0)
            continue;
        if (options->role && (!role || strcmp(options->role, role) != 0)) continue;
        yvex_cli_out_writef(
            stdout,
            "decision ordinal=%llu tensor=%s role=%s layer=%llu qtype=%u rule=%llu priority=%d "
            "label=%s imatrix=%d bytes=%llu cpu=%d cuda=%d identity=%s\n",
            ordinal, decision->physical_tensor_name, role ? role : "unknown",
            decision->logical_key.layer_index,
            decision->qtype, decision->policy_rule_ordinal, decision->policy_priority,
            decision->policy_label, decision->policy_requires_imatrix, decision->encoded_bytes,
            decision->cpu_compute_available, decision->cuda_compute_available,
            decision->decision_identity);
        matches++;
    }
    if (!matches) {
        yvex_cli_out_writef(stderr, "yvex: quant explain selector matched no terminal\n");
        return 1;
    }
    yvex_cli_out_writef(stdout, "matched_decisions: %llu\n", matches);
    return 0;
}

/* Purpose: construct the complete writer plan from the exact resolved physical plan.
 * Inputs: a context retaining family lowering, verification, and the sealed quant plan.
 * Effects: allocates one caller-owned writer plan through the writer subsystem.
 * Failure: typed writer refusal leaves no usable plan in the output handle.
 * Boundary: neither this adapter nor the writer re-resolves policy rules. */
static int quant_cli_writer_build(yvex_gguf_writer_plan **out, const quant_cli_context *context,
                                  yvex_gguf_writer_failure *failure, yvex_error *err)
{
    yvex_gguf_writer_plan_options options;
    yvex_gguf_writer_plan_request request;

    yvex_gguf_writer_plan_options_default(&options);
    memset(&request, 0, sizeof(request));
    request.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    request.quant_plan = context->plan;
    request.options = &options;
    request.input.complete.family_adapter = yvex_model_register_deepseek_v4();
    request.input.complete.lowering =
        yvex_model_register_deepseek_v4()->payload.map(context->handoff);
    request.input.complete.verification =
        yvex_model_register_deepseek_v4()->payload.verification(context->handoff);
    return yvex_gguf_writer_plan_build(out, &request, failure, err);
}

/* Purpose: execute, native-roundtrip, and atomically publish one complete physical artifact.
 * Inputs: regenerated context and exact previously published plan path.
 * Effects: streams bounded source reads through codecs and the transactional GGUF file sink.
 * Failure: failed terminal, roundtrip, or publication leaves no final artifact.
 * Boundary: official-reader admission, materialization, and runtime binding remain later gates. */
static int quant_cli_emit(const quant_cli_options *options, quant_cli_context *context,
                          yvex_error *err)
{
    yvex_gguf_file_sink *file_sink = NULL;
    yvex_quant_output_sink sink;
    yvex_quant_executor_options executor;
    yvex_quant_execution_summary execution;
    yvex_quant_failure quant_failure;
    yvex_gguf_file_failure file_failure;
    yvex_gguf_roundtrip_failure roundtrip_failure;
    yvex_gguf_file_sink_options file_options;
    yvex_gguf_file_sink_summary emission;
    yvex_gguf_roundtrip_options roundtrip_options;
    yvex_gguf_roundtrip_summary roundtrip;
    const yvex_gguf_writer_plan_summary *writer_summary;
    int rc;

    rc = yvex_quant_plan_file_validate(options->plan_path, context->plan, err);
    writer_summary = yvex_gguf_writer_plan_summary_get(context->writer);
    if (rc == YVEX_OK) {
        yvex_gguf_file_sink_options_default(&file_options);
        file_options.destination_path = options->out_artifact;
        rc = yvex_gguf_file_sink_create(&file_sink, context->writer, context->plan, &file_options,
                                        &file_failure, err);
    }
    if (rc == YVEX_OK) {
        yvex_gguf_file_sink_adapter(file_sink, &sink);
        yvex_quant_executor_options_default(&executor);
        executor.worker_count = 16u;
        executor.maximum_owned_bytes = 64u * 1024u * 1024u;
        executor.imatrix = context->imatrix;
        rc = yvex_quant_execute(context->plan, &sink, &executor, &execution, &quant_failure, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_gguf_file_sink_finalize(file_sink, &emission, &file_failure, err);
    if (rc == YVEX_OK) {
        yvex_gguf_roundtrip_options_default(&roundtrip_options);
        rc = yvex_gguf_roundtrip_validate(
            yvex_gguf_file_sink_temporary_path(file_sink), context->writer,
            yvex_gguf_file_sink_digest(file_sink), &roundtrip_options, &roundtrip,
            &roundtrip_failure, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_gguf_file_sink_publish(file_sink, &roundtrip, &emission, &file_failure, err);
    if (rc == YVEX_OK) {
        yvex_cli_out_writef(stdout, "status: complete-physical-artifact-emitted\n");
        yvex_cli_out_writef(stdout, "artifact: %s\n", options->out_artifact);
        yvex_cli_out_writef(stdout, "artifact_identity: %s\n", roundtrip.artifact_identity);
        yvex_cli_out_writef(stdout, "writer_plan_identity: %s\n",
                            writer_summary->writer_plan_identity);
        yvex_cli_out_writef(stdout, "quant_execution_identity: %s\n",
                            emission.execution_identity);
        yvex_cli_out_writef(stdout, "file_bytes: %llu\n", roundtrip.file_bytes);
        yvex_cli_out_writef(stdout, "payload_bytes: %llu\n", roundtrip.payload_bytes_verified);
        yvex_cli_out_writef(stdout, "terminal_count: %llu\n", roundtrip.terminals_verified);
        yvex_cli_out_writef(stdout, "native_roundtrip: accepted\n");
        yvex_cli_out_writef(stdout, "official_reader_admission: pending\n");
    }
    yvex_gguf_file_sink_release(&file_sink);
    return rc;
}

/* Purpose: list, inspect, or transactionally export normal sealed preset policies.
 * Inputs: the bounded `quant preset` argv grammar.
 * Effects: may render metadata or publish one policy JSON through the policy owner.
 * Failure: unknown presets, malformed grammar, or publication errors return nonzero.
 * Boundary: presets are ordinary policies and this adapter assigns no tensor qtype. */
static int quant_cli_preset(int argc, char **argv)
{
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy_summary summary;
    yvex_error err;
    unsigned long long index;
    int rc;

    if (argc < 4 || strcmp(argv[3], "list") == 0) {
        for (index = 0u; index < yvex_quant_policy_preset_count(); ++index)
            yvex_cli_out_writef(stdout, "%s\n", yvex_quant_policy_preset_name(index));
        return 0;
    }
    yvex_error_clear(&err);
    if (strcmp(argv[3], "show") == 0 && argc == 5) {
        rc = yvex_quant_policy_preset_open(&policy, argv[4], &err);
        if (rc == YVEX_OK) rc = yvex_quant_policy_get_summary(policy, &summary, &err);
        if (rc == YVEX_OK) {
            yvex_cli_out_writef(stdout, "preset: %s\n", summary.preset_name);
            yvex_cli_out_writef(stdout, "schema_version: %u\n", summary.schema_version);
            yvex_cli_out_writef(stdout, "policy_identity: %s\n", summary.policy_identity);
            yvex_cli_out_writef(stdout, "rule_count: %llu\n", summary.rule_count);
            yvex_cli_out_writef(stdout, "imatrix_rules: %llu\n", summary.requires_imatrix_count);
        }
        yvex_quant_policy_close(policy);
        return rc == YVEX_OK ? 0 : quant_cli_fail("preset show", &err);
    }
    if (strcmp(argv[3], "export") == 0 && argc == 7 && strcmp(argv[5], "--out") == 0) {
        rc = yvex_quant_policy_preset_open(&policy, argv[4], &err);
        if (rc == YVEX_OK) rc = yvex_quant_policy_write_json(argv[6], policy, &err);
        yvex_quant_policy_close(policy);
        return rc == YVEX_OK ? 0 : quant_cli_fail("preset export", &err);
    }
    yvex_cli_out_writef(stderr, "yvex: quant preset expects list, show NAME, or export NAME --out FILE\n");
    return 2;
}

/* Purpose: dispatch one canonical physical-variant command through production domain APIs.
 * Inputs: process arguments rooted at the `quant` command.
 * Effects: may publish a plan, artifact, policy export, or bounded evidence.
 * Failure: parsing and typed domain refusals return nonzero without partial final publication.
 * Boundary: contains no codec, rule resolution, artifact serialization, or materialization math. */
int yvex_quant_command(int arg_count, char **args)
{
    quant_cli_options options;
    quant_cli_context context;
    yvex_quant_failure failure;
    yvex_error err;
    int parsed;
    int rc;

    if (arg_count >= 3 && strcmp(args[2], "preset") == 0)
        return quant_cli_preset(arg_count, args);
    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_quant_help(stdout);
        return 0;
    }
    parsed = quant_cli_parse(arg_count, args, &options);
    if (parsed < 0) {
        yvex_quant_help(stdout);
        return 0;
    }
    if (!parsed || !quant_cli_options_valid(&options)) {
        yvex_quant_help(stderr);
        return 2;
    }
    yvex_error_clear(&err);
    memset(&failure, 0, sizeof(failure));
    rc = quant_cli_context_open(&context, &options, &failure, &err);
    if (rc == YVEX_OK && options.action == QUANT_CLI_PLAN)
        rc = yvex_quant_plan_file_write(options.out_plan, context.plan, &err);
    if (rc == YVEX_OK && (options.action == QUANT_CLI_SUMMARIZE ||
                          options.action == QUANT_CLI_EXPLAIN))
        rc = yvex_quant_plan_file_validate(options.plan_path, context.plan, &err);
    if (rc == YVEX_OK && options.action == QUANT_CLI_EMIT)
        rc = quant_cli_emit(&options, &context, &err);
    if (rc == YVEX_OK && (options.action == QUANT_CLI_PLAN ||
                          options.action == QUANT_CLI_SUMMARIZE))
        quant_cli_summary_print(&context);
    if (rc == YVEX_OK && options.action == QUANT_CLI_EXPLAIN)
        rc = quant_cli_explain(&options, context.plan) == 0 ? YVEX_OK : YVEX_ERR_FORMAT;
    quant_cli_context_close(&context);
    return rc == YVEX_OK ? 0 : quant_cli_fail("execution", &err);
}

/* Purpose: render the exact production physical-variant CLI grammar and boundary.
 * Inputs: one writable CLI stream selected by the caller.
 * Effects: writes usage text only.
 * Failure: stream failures follow the canonical CLI output owner's behavior.
 * Boundary: help text is not capability or execution evidence. */
void yvex_quant_help(FILE *fp)
{
    yvex_cli_out_writef(fp, "usage:\n");
    yvex_cli_out_writef(fp, "  yvex quant preset list|show NAME|export NAME --out FILE\n");
    yvex_cli_out_writef(fp,
                        "  yvex quant plan --target TARGET --source DIR --models-root DIR "
                        "--source-manifest FILE (--preset NAME|--policy FILE) [--imatrix-manifest FILE] "
                        "[--backend cpu|cuda] --out-plan FILE\n");
    yvex_cli_out_writef(fp,
                        "  yvex quant emit --target TARGET --source DIR --models-root DIR "
                        "--source-manifest FILE (--preset NAME|--policy FILE) [--imatrix-manifest FILE] "
                        "--plan FILE --out FILE\n");
    yvex_cli_out_writef(fp,
                        "  yvex quant summarize|explain [same source/policy options] --plan FILE "
                        "[--tensor NAME|--role ROLE]\n\n");
    yvex_cli_out_writef(fp,
                        "The plan is regenerated from source/policy authority before emit. "
                        "Materialization never chooses qtypes.\n");
}
