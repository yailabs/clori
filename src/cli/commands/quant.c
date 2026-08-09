/*
 * Make physical-variant planning, explanation, and complete GGUF emission operator reachable.
 *
 * Every emit rederives and exactly validates its external plan before reading model payloads. CLI
 * selects production owners but never resolves tensor qtypes or writes artifact bytes itself.
 */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/quant.h>

typedef enum {
    QUANT_CLI_NONE = 0,
    QUANT_CLI_PLAN,
    QUANT_CLI_EMIT,
    QUANT_CLI_PROBE,
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
    const char *component;
} quant_cli_options;

typedef struct {
    yvex_deepseek_payload_handoff *handoff;
    yvex_minimax_h3_handoff *minimax_handoff;
    yvex_minimax_h3_component_id minimax_component;
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

static int quant_cli_fail(const char *phase, const yvex_error *err)
{
    yvex_cli_out_writef(stderr, "yvex: quant %s failed: %s (%s)\n", phase,
                        yvex_error_message(err), yvex_error_where(err));
    return 1;
}

static const char *quant_cli_value(int argc, char **argv, int *index)
{
    if (!index || *index + 1 >= argc) return NULL;
    (*index)++;
    return argv[*index];
}

static int quant_cli_parse(int argc, char **argv, quant_cli_options *options)
{
    int index;

    memset(options, 0, sizeof(*options));
    if (argc < 3) return 0;
    if (strcmp(argv[2], "plan") == 0) options->action = QUANT_CLI_PLAN;
    else if (strcmp(argv[2], "emit") == 0) options->action = QUANT_CLI_EMIT;
    else if (strcmp(argv[2], "probe") == 0) options->action = QUANT_CLI_PROBE;
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
        else if (strcmp(argv[index - 1], "--component") == 0) options->component = value;
        else {
            yvex_cli_out_writef(stderr, "yvex: unknown quant option: %s\n", argv[index - 1]);
            return 0;
        }
    }
    return 1;
}

static int quant_cli_options_valid(const quant_cli_options *options)
{
    int minimax;

    if (!options || !options->target || !options->source) return 0;
    minimax = strcmp(options->target, YVEX_MINIMAX_H3_TARGET_ID) == 0;
    if (minimax) {
        if (!options->component || options->models_root || options->manifest ||
            options->preset || options->policy_path || options->imatrix_path ||
            options->backend || options->role)
            return 0;
    } else if (strcmp(options->target, "deepseek4-v4-flash-dspark") != 0 ||
               !options->models_root || !options->manifest || options->component ||
               (!!options->preset == !!options->policy_path)) {
        return 0;
    }
    if (options->backend && strcmp(options->backend, "cpu") != 0 &&
        strcmp(options->backend, "cuda") != 0)
        return 0;
    if (options->action == QUANT_CLI_PLAN) return options->out_plan != NULL;
    if (options->action == QUANT_CLI_EMIT)
        return options->plan_path && options->out_artifact;
    if (options->action == QUANT_CLI_PROBE)
        return options->plan_path && options->tensor && !options->role;
    if (options->action == QUANT_CLI_SUMMARIZE) return options->plan_path != NULL;
    if (options->action == QUANT_CLI_EXPLAIN)
        return options->plan_path &&
               (minimax ? options->tensor != NULL : (!!options->tensor != !!options->role));
    return 0;
}

static void quant_cli_context_close(quant_cli_context *context)
{
    if (!context) return;
    yvex_gguf_writer_plan_release(&context->writer);
    yvex_quant_plan_release(&context->plan);
    yvex_imatrix_data_close(context->imatrix);
    yvex_quant_policy_close(context->policy);
    if (context->minimax_handoff)
        yvex_model_minimax_h3_handoff_api()->close(&context->minimax_handoff);
    if (context->handoff)
        yvex_model_register_deepseek_v4()->payload.close(context->handoff);
    memset(context, 0, sizeof(*context));
}

static int quant_cli_imatrix_open(quant_cli_context *context, const char *path,
                                  yvex_error *err)
{
    yvex_imatrix_data_options options;
    int rc;

    if (!path) return YVEX_OK;
    memset(&options, 0, sizeof(options));
    options.path = path;
    /* The bootstrap profile deliberately carries forward the predecessor's
     * routed-expert importance prior. Bind that provenance honestly; calling
     * it calibration of the new DSpark transform would make two different
     * source snapshots share one semantic claim. */
    options.source_model_identity =
        YVEX_QUANT_DSPARK_IMATRIX_SOURCE_IDENTITY;
    options.calibration_dataset_identity =
        YVEX_QUANT_DSPARK_IMATRIX_DATASET_IDENTITY;
    options.producer = "llama.cpp-imatrix";
    options.producer_version = 1u;
    options.maximum_mapped_bytes = 1024u * 1024u * 1024u;
    rc = yvex_imatrix_data_open(&context->imatrix, &options, err);
    if (rc == YVEX_OK)
        rc = yvex_imatrix_data_get_summary(context->imatrix, &context->imatrix_summary, err);
    return rc;
}

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

static int quant_cli_context_open_deepseek(
    quant_cli_context *context, const quant_cli_options *options,
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
        rc = quant_cli_imatrix_open(context, options->imatrix_path, err);
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

static int quant_cli_minimax_component_find(
    const char *name, yvex_minimax_h3_component_id *out)
{
    static const yvex_minimax_h3_component_id weighted[] = {
        YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER,
        YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER,
        YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE,
        YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE
    };
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    size_t index;

    if (!name || !out) return 0;
    for (index = 0u; index < sizeof(weighted) / sizeof(weighted[0]); ++index) {
        if (strcmp(name, family->component_name(weighted[index])) == 0) {
            *out = weighted[index];
            return 1;
        }
    }
    return 0;
}

static int quant_cli_context_open_minimax(
    quant_cli_context *context, const quant_cli_options *options,
    yvex_quant_failure *failure, yvex_error *err)
{
    const yvex_minimax_h3_handoff_api *handoff_api =
        yvex_model_minimax_h3_handoff_api();
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    yvex_minimax_h3_handoff_options handoff_options;
    yvex_minimax_h3_handoff_failure handoff_failure;
    const yvex_minimax_h3_target *target;
    const yvex_minimax_h3_component *component;
    yvex_gguf_writer_failure writer_failure;
    int rc;

    memset(context, 0, sizeof(*context));
    if (!quant_cli_minimax_component_find(options->component,
                                          &context->minimax_component)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_cli_minimax",
                       "component must be text_encoder, transformer, video_vae, or audio_vae");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(&handoff_options, 0, sizeof(handoff_options));
    handoff_options.source_root = options->source;
    handoff_options.component = context->minimax_component;
    yvex_source_payload_budget_default(&handoff_options.budget);
    handoff_options.budget.maximum_open_handles = 4u;
    handoff_options.budget.maximum_streams = 1u;
    handoff_options.budget.maximum_inflight_host_bytes =
        handoff_options.budget.chunk_bytes;
    handoff_options.chunk_bytes = handoff_options.budget.chunk_bytes;
    handoff_options.page_bytes = handoff_options.budget.page_bytes;
    memset(&handoff_failure, 0, sizeof(handoff_failure));
    rc = handoff_api->open(&context->minimax_handoff, &handoff_options,
                           &handoff_failure, err);
    target = context->minimax_handoff
                 ? handoff_api->target(context->minimax_handoff) : NULL;
    component = target ? family->component_at(target, context->minimax_component) : NULL;
    if (rc == YVEX_OK && !component) {
        yvex_error_set(err, YVEX_ERR_STATE, "quant_cli_minimax",
                       "weighted component disappeared after source admission");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = yvex_quant_plan_build_source_faithful(
            &context->plan, handoff_api->transform_ir(context->minimax_handoff),
            handoff_api->binding(context->minimax_handoff),
            "minimax-h3-source-faithful-v1",
            yvex_transform_hash_string(component->identity), NULL, failure, err);
    if (rc == YVEX_OK) {
        memset(&writer_failure, 0, sizeof(writer_failure));
        rc = quant_cli_writer_build(&context->writer, context, &writer_failure, err);
    }
    return rc;
}

static int quant_cli_context_open(quant_cli_context *context,
                                  const quant_cli_options *options,
                                  yvex_quant_failure *failure, yvex_error *err)
{
    return strcmp(options->target, YVEX_MINIMAX_H3_TARGET_ID) == 0
               ? quant_cli_context_open_minimax(context, options, failure, err)
               : quant_cli_context_open_deepseek(context, options, failure, err);
}

/*
 * Render exact size, qtype, and identity facts from one sealed plan.
 *
 * Writes bounded human-readable evidence to the canonical CLI output owner.
 */
static void quant_cli_minimax_summary_print(const quant_cli_context *context)
{
    const yvex_minimax_h3_handoff_api *handoff =
        yvex_model_minimax_h3_handoff_api();
    const yvex_minimax_h3_handoff_summary *source =
        handoff->summary(context->minimax_handoff);
    const yvex_quant_plan_summary *plan = yvex_quant_plan_summary_get(context->plan);
    const yvex_gguf_writer_plan_summary *writer =
        yvex_gguf_writer_plan_summary_get(context->writer);
    const char *component = yvex_model_register_minimax_h3()->component_name(
        context->minimax_component);

    yvex_cli_out_writef(stdout, "status: component-physical-variant-plan-complete\n");
    yvex_cli_out_writef(stdout, "target: %s\n", YVEX_MINIMAX_H3_TARGET_ID);
    yvex_cli_out_writef(stdout, "family: minimax-h3\n");
    yvex_cli_out_writef(stdout, "component: %s\n", component);
    yvex_cli_out_writef(stdout, "source_revision: %s\n", YVEX_MINIMAX_H3_REVISION);
    yvex_cli_out_writef(stdout, "source_verified: %d\n", source && source->complete);
    yvex_cli_out_writef(stdout, "source_snapshot_identity: %s\n",
                        source ? source->source_snapshot_identity : "");
    yvex_cli_out_writef(stdout, "component_identity: %s\n",
                        source ? source->component_identity : "");
    yvex_cli_out_writef(stdout, "transform_identity: %s\n",
                        source ? source->transform_identity : "");
    yvex_cli_out_writef(stdout, "profile: %s\n", plan ? plan->profile_name : "");
    yvex_cli_out_writef(stdout, "variant_identity: %s\n",
                        plan ? plan->physical_variant_identity : "");
    yvex_cli_out_writef(stdout, "writer_plan_identity: %s\n",
                        writer ? writer->writer_plan_identity : "");
    yvex_cli_out_writef(stdout, "shards: %llu\n", source ? source->shards : 0u);
    yvex_cli_out_writef(stdout, "tensors: %llu\n", source ? source->tensors : 0u);
    yvex_cli_out_writef(stdout, "elements: %llu\n", source ? source->elements : 0u);
    yvex_cli_out_writef(stdout, "predicted_payload_bytes: %llu\n",
                        plan ? plan->encoded_bytes : 0u);
    yvex_cli_out_writef(stdout, "predicted_gguf_bytes: %llu\n",
                        writer ? writer->final_file_bytes : 0u);
    yvex_cli_out_writef(stdout, "bf16_tensors: %llu\n",
                        plan ? plan->qtype_tensor_counts[YVEX_GGUF_QTYPE_BF16] : 0u);
    yvex_cli_out_writef(stdout, "f32_tensors: %llu\n",
                        plan ? plan->qtype_tensor_counts[YVEX_GGUF_QTYPE_F32] : 0u);
    yvex_cli_out_writef(stdout, "transformation_payload_bytes_read: %llu\n",
                        source ? source->payload_execution_bytes_read : 0u);
    yvex_cli_out_writef(stdout, "artifact_emittable: %d\n", writer && writer->complete);
    yvex_cli_out_writef(stdout, "component_artifact_emitted: 0\n");
    yvex_cli_out_writef(stdout, "runtime_ready: 0\n");
    yvex_cli_out_writef(stdout, "backend_ready: 0\n");
    yvex_cli_out_writef(stdout, "media_generation_ready: 0\n");
    yvex_cli_out_writef(stdout,
                        "next_boundary: emit-and-admit-component-artifacts\n");
}

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

    if (context->minimax_handoff) {
        quant_cli_minimax_summary_print(context);
        return;
    }

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
        yvex_cli_out_writef(stderr, "yvex: inspect quant decision selector matched no terminal\n");
        return 1;
    }
    yvex_cli_out_writef(stdout, "matched_decisions: %llu\n", matches);
    return 0;
}

static int quant_cli_writer_build(yvex_gguf_writer_plan **out, const quant_cli_context *context,
                                  yvex_gguf_writer_failure *failure, yvex_error *err)
{
    yvex_gguf_writer_plan_options options;
    yvex_gguf_writer_plan_request request;

    yvex_gguf_writer_plan_options_default(&options);
    memset(&request, 0, sizeof(request));
    request.quant_plan = context->plan;
    request.options = &options;
    if (context->minimax_handoff) {
        const yvex_minimax_h3_handoff_api *handoff =
            yvex_model_minimax_h3_handoff_api();
        const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
        const yvex_minimax_h3_target *target = handoff->target(context->minimax_handoff);
        const yvex_minimax_h3_summary *summary = family->summary(target);
        const yvex_minimax_h3_component *component =
            family->component_at(target, context->minimax_component);

        if (!summary || !component) {
            yvex_error_set(err, YVEX_ERR_STATE, "quant_cli_writer",
                           "MiniMax component source facts are unavailable");
            return YVEX_ERR_STATE;
        }
        request.input_class = YVEX_GGUF_WRITER_INPUT_LOGICAL_COMPONENT;
        request.input.component.architecture = "minimax-h3";
        request.input.component.target_id = YVEX_MINIMAX_H3_TARGET_ID;
        request.input.component.component_id = component->canonical_id;
        request.input.component.source_snapshot_identity =
            summary->source_snapshot_identity;
        request.input.component.source_snapshot_key = summary->source_snapshot_key;
        request.input.component.component_identity = component->identity;
        request.input.component.component_manifest_identity =
            summary->component_manifest_identity;
        request.input.component.architecture_identity = summary->architecture_identity;
        request.input.component.role_map_identity = summary->role_map_identity;
        return yvex_gguf_writer_plan_build(out, &request, failure, err);
    }
    request.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    request.input.complete.lowering = yvex_model_deepseek_writer_lowering_api();
    request.input.complete.lowering_context =
        yvex_model_register_deepseek_v4()->payload.map(context->handoff);
    request.input.complete.verification =
        yvex_model_register_deepseek_v4()->payload.verification(context->handoff);
    return yvex_gguf_writer_plan_build(out, &request, failure, err);
}

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
    const yvex_quant_plan_summary *plan_summary;
    const yvex_gguf_writer_plan_summary *writer_summary;
    int rc;

    rc = yvex_quant_plan_file_validate(options->plan_path, context->plan, err);
    plan_summary = yvex_quant_plan_summary_get(context->plan);
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
        executor.worker_count = context->minimax_handoff ? 1u : 16u;
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
        if (rc != YVEX_OK)
            yvex_error_setf(err, (yvex_status)rc, "gguf.roundtrip",
                            "roundtrip code=%u parser/layout code=%llu record=%llu offset=%llu name=%s",
                            (unsigned int)roundtrip_failure.code, roundtrip_failure.actual,
                            roundtrip_failure.metadata_index, roundtrip_failure.file_offset,
                            roundtrip_failure.name[0] ? roundtrip_failure.name : "-");
    }
    if (rc == YVEX_OK)
        rc = yvex_gguf_file_sink_publish(file_sink, &roundtrip, &emission, &file_failure, err);
    if (rc == YVEX_OK) {
        yvex_cli_out_writef(
            stdout, "status: %s\n",
            context->minimax_handoff ? "component-physical-artifact-emitted"
                                     : "complete-physical-artifact-emitted");
        if (context->minimax_handoff)
            yvex_cli_out_writef(
                stdout, "component: %s\n",
                yvex_model_register_minimax_h3()->component_name(
                    context->minimax_component));
        yvex_cli_out_writef(stdout, "artifact: %s\n", options->out_artifact);
        yvex_cli_out_writef(stdout, "profile: %s\n", writer_summary->profile_name);
        yvex_cli_out_writef(stdout, "profile_identity: %s\n",
                            writer_summary->profile_identity);
        yvex_cli_out_writef(stdout, "physical_variant_identity: %s\n",
                            plan_summary->physical_variant_identity);
        yvex_cli_out_writef(stdout, "source_snapshot_identity: %016llx\n",
                            writer_summary->source_snapshot_identity);
        yvex_cli_out_writef(stdout, "mapping_identity: %016llx\n",
                            writer_summary->mapping_identity);
        yvex_cli_out_writef(stdout, "payload_identity: %s\n",
                            writer_summary->payload_identity);
        yvex_cli_out_writef(stdout, "transform_identity: %s\n",
                            writer_summary->transform_identity);
        yvex_cli_out_writef(stdout, "payload_plan_identity: %s\n",
                            writer_summary->payload_plan_identity);
        yvex_cli_out_writef(stdout, "artifact_identity: %s\n", roundtrip.artifact_identity);
        yvex_cli_out_writef(stdout, "writer_plan_identity: %s\n",
                            writer_summary->writer_plan_identity);
        yvex_cli_out_writef(stdout, "quant_execution_identity: %s\n",
                            emission.execution_identity);
        yvex_cli_out_writef(stdout, "payload_byte_identity: %s\n",
                            emission.payload_byte_identity);
        yvex_cli_out_writef(stdout, "file_bytes: %llu\n", roundtrip.file_bytes);
        yvex_cli_out_writef(stdout, "payload_bytes: %llu\n", roundtrip.payload_bytes_verified);
        yvex_cli_out_writef(stdout, "metadata_count: %llu\n", roundtrip.metadata_count);
        yvex_cli_out_writef(stdout, "tensor_count: %llu\n", roundtrip.tensor_count);
        yvex_cli_out_writef(stdout, "terminal_count: %llu\n", roundtrip.terminals_verified);
        yvex_cli_out_writef(stdout, "tokenizer_tokens: %llu\n",
                            writer_summary->tokenizer_token_count);
        yvex_cli_out_writef(stdout, "tokenizer_merges: %llu\n",
                            writer_summary->tokenizer_merge_count);
        yvex_cli_out_writef(stdout, "policy_identity: %s\n", plan_summary->policy_identity);
        yvex_cli_out_writef(stdout, "imatrix_identity: %s\n", plan_summary->imatrix_identity);
        yvex_cli_out_writef(stdout, "native_roundtrip: accepted\n");
        yvex_cli_out_writef(stdout, "official_reader_admission: pending\n");
        if (context->minimax_handoff) {
            yvex_cli_out_writef(stdout, "runtime_ready: 0\n");
            yvex_cli_out_writef(stdout, "media_generation_ready: 0\n");
        }
    }
    yvex_gguf_file_sink_release(&file_sink);
    return rc;
}

static int quant_cli_probe(const quant_cli_options *options, quant_cli_context *context,
                           yvex_error *err)
{
    const yvex_quant_plan_summary *plan = yvex_quant_plan_summary_get(context->plan);
    const yvex_quant_decision *selected = NULL;
    yvex_quant_digest_sink *digest = NULL;
    yvex_quant_output_sink sink;
    yvex_quant_executor_options executor;
    yvex_quant_execution_summary execution;
    yvex_quant_failure failure;
    const yvex_quant_metrics *metrics;
    struct timespec started;
    struct timespec stopped;
    unsigned long long ordinal;
    double seconds;
    int rc;

    rc = yvex_quant_plan_file_validate(options->plan_path, context->plan, err);
    for (ordinal = 0u; rc == YVEX_OK && ordinal < plan->decision_count; ++ordinal) {
        const yvex_quant_decision *decision =
            yvex_quant_plan_decision_at(context->plan, ordinal);
        if (!decision || strcmp(decision->physical_tensor_name, options->tensor) != 0)
            continue;
        if (selected) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "quant_cli_probe",
                           "probe tensor selector is not unique");
            rc = YVEX_ERR_FORMAT;
            break;
        }
        selected = decision;
    }
    if (rc == YVEX_OK && !selected) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "quant_cli_probe",
                       "probe tensor is absent from the sealed physical plan");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        rc = yvex_quant_digest_sink_create(&digest, context->plan,
                                           plan->required_payload_identity, &failure, err);
    if (rc == YVEX_OK) {
        yvex_quant_digest_sink_adapter(digest, &sink);
        yvex_quant_executor_options_default(&executor);
        executor.worker_count = 1u;
        executor.maximum_owned_bytes = 64u * 1024u * 1024u;
        executor.first_terminal = selected->terminal_ordinal;
        executor.terminal_count = 1u;
        executor.imatrix = context->imatrix;
        if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) {
            yvex_error_set(err, YVEX_ERR_IO, "quant_cli_probe",
                           "probe monotonic clock start failed");
            rc = YVEX_ERR_IO;
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_quant_execute(context->plan, &sink, &executor, &execution, &failure, err);
    if (rc == YVEX_OK && clock_gettime(CLOCK_MONOTONIC, &stopped) != 0) {
        yvex_error_set(err, YVEX_ERR_IO, "quant_cli_probe",
                       "probe monotonic clock stop failed");
        rc = YVEX_ERR_IO;
    }
    if (rc == YVEX_OK) {
        metrics = &execution.role_metrics[selected->role];
        seconds = (double)(stopped.tv_sec - started.tv_sec) +
                  (double)(stopped.tv_nsec - started.tv_nsec) / 1000000000.0;
        yvex_cli_out_writef(stdout, "status: quant-role-probe-complete\n");
        yvex_cli_out_writef(stdout, "profile_identity: %s\n", plan->profile_identity);
        yvex_cli_out_writef(stdout, "tensor: %s\n", selected->physical_tensor_name);
        yvex_cli_out_writef(stdout, "role: %s\n", yvex_tensor_role_name(selected->role));
        yvex_cli_out_writef(stdout, "terminal_ordinal: %llu\n", selected->terminal_ordinal);
        yvex_cli_out_writef(stdout, "qtype: %u\n", selected->qtype);
        yvex_cli_out_writef(stdout, "elements: %llu\n", metrics->element_count);
        yvex_cli_out_writef(stdout, "finite_elements: %llu\n", metrics->finite_count);
        yvex_cli_out_writef(stdout, "nonfinite_elements: %llu\n", metrics->nonfinite_count);
        yvex_cli_out_writef(stdout, "encoded_bytes: %llu\n", execution.encoded_output_bytes);
        yvex_cli_out_writef(stdout, "payload_bytes_read: %llu\n", execution.payload_bytes_read);
        yvex_cli_out_writef(stdout, "maximum_absolute_error: %.17g\n",
                            metrics->maximum_absolute_error);
        yvex_cli_out_writef(stdout, "rmse: %.17g\n", yvex_quant_metrics_rmse(metrics));
        yvex_cli_out_writef(stdout, "mean_absolute_error: %.17g\n",
                            metrics->finite_count
                                ? metrics->absolute_error_sum / (double)metrics->finite_count
                                : 0.0);
        yvex_cli_out_writef(stdout, "mean_relative_error: %.17g\n",
                            metrics->finite_count
                                ? metrics->relative_error_sum / (double)metrics->finite_count
                                : 0.0);
        yvex_cli_out_writef(stdout, "reference_squared_sum: %.17g\n",
                            metrics->reference_squared_sum);
        yvex_cli_out_writef(stdout, "dot_reference: %.17g\n", metrics->dot_reference);
        yvex_cli_out_writef(stdout, "dot_reconstructed: %.17g\n",
                            metrics->dot_reconstructed);
        yvex_cli_out_writef(stdout, "wall_seconds: %.9f\n", seconds);
    }
    yvex_quant_digest_sink_release(&digest);
    return rc;
}

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
    yvex_cli_out_writef(stderr, "yvex: compile quant preset expects list, show NAME, or export NAME --out FILE\n");
    return 2;
}

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
    if (rc == YVEX_OK && options.action == QUANT_CLI_PROBE)
        rc = quant_cli_probe(&options, &context, &err);
    if (rc == YVEX_OK && (options.action == QUANT_CLI_PLAN ||
                          options.action == QUANT_CLI_SUMMARIZE))
        quant_cli_summary_print(&context);
    if (rc == YVEX_OK && options.action == QUANT_CLI_EXPLAIN)
        rc = quant_cli_explain(&options, context.plan) == 0 ? YVEX_OK : YVEX_ERR_FORMAT;
    quant_cli_context_close(&context);
    return rc == YVEX_OK ? 0 : quant_cli_fail("execution", &err);
}

void yvex_quant_help(FILE *fp)
{
    yvex_cli_out_writef(fp, "usage:\n");
    yvex_cli_out_writef(fp, "  yvex compile quant preset list|show NAME|export NAME --out FILE\n");
    yvex_cli_out_writef(fp,
                        "  yvex compile quant plan --target TARGET --source DIR --models-root DIR "
                        "--source-manifest FILE (--preset NAME|--policy FILE) [--imatrix-manifest FILE] "
                        "[--backend cpu|cuda] --out-plan FILE\n");
    yvex_cli_out_writef(fp,
                        "  yvex compile quant emit --target TARGET --source DIR --models-root DIR "
                        "--source-manifest FILE (--preset NAME|--policy FILE) [--imatrix-manifest FILE] "
                        "--plan FILE --out FILE\n");
    yvex_cli_out_writef(
        fp,
        "  yvex compile quant plan --target minimax-h3-fl2va --source DIR "
        "--component text_encoder|transformer|video_vae|audio_vae --out-plan FILE\n");
    yvex_cli_out_writef(
        fp,
        "  yvex compile quant emit --target minimax-h3-fl2va --source DIR "
        "--component COMPONENT --plan FILE --out FILE\n");
    yvex_cli_out_writef(fp,
                        "  yvex compile quant probe --target TARGET --source DIR --models-root DIR "
                        "--source-manifest FILE (--preset NAME|--policy FILE) [--imatrix-manifest FILE] "
                        "--plan FILE --tensor NAME\n");
    yvex_cli_out_writef(fp,
                        "  yvex inspect quant summary|explain [same source/policy options] --plan FILE "
                        "[--tensor NAME|--role ROLE]\n\n");
    yvex_cli_out_writef(fp,
                        "Plan an identity-bound physical variant. The plan is regenerated from "
                        "source/policy authority before emit; materialization never chooses qtypes.\n");
}
