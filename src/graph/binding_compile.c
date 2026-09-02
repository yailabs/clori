/*
 * Own the generic execution-compilation lifecycle that lowers one family into binding products.
 *
 * Family callbacks provide semantic projections and numeric policy. This owner opens and closes
 * artifact, materialization, graph, quant, and writer resources in one deterministic order; it
 * never identifies a concrete family or reconstructs its topology.
 */
#include "src/graph/private.h"

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/tokenizer.h>

#include <stdlib.h>
#include <string.h>

typedef struct {
    const yvex_family_compiler_adapter *adapter;
    const yvex_family_binding_pipeline *pipeline;
    const yvex_graph_compiler_api *graph;
    yvex_family_compilation_source source;
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_complete_artifact_admission admission;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_semantic_model_ir *semantic_model;
    yvex_operator_graph_ir *operator_graph;
    yvex_physical_execution_ir *physical_execution;
    yvex_compiled_model_plan *compiled_plan;
    yvex_runtime_descriptor *descriptor;
    yvex_attention_plan *attention;
    yvex_attention_plan *draft_attention;
    yvex_quant_policy *quant_policy;
    yvex_imatrix_data *imatrix;
    yvex_quant_plan *quant;
    yvex_gguf_writer_plan *writer;
    yvex_artifact_physical_compatibility compatibility;
    yvex_artifact_compatibility_failure compatibility_failure;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_projection materialization_projection;
    yvex_materialization_failure materialization_failure;
    yvex_attention_failure attention_failure;
    yvex_gguf_writer_failure writer_failure;
    yvex_runtime_capabilities capabilities;
    yvex_transformer_family_policy transformer_policy;
    yvex_logits_family_policy logits_policy;
    yvex_speculation_family_policy speculation_policy;
    yvex_tokenizer_family_policy tokenizer_policy;
} binding_compiler;

static void family_runtime_binding_release(void *owner);

static int pipeline_valid(const yvex_family_compiler_adapter *adapter)
{
    const yvex_family_binding_pipeline *pipeline = adapter ? adapter->binding_pipeline : NULL;

    return adapter && adapter->schema_version == YVEX_FAMILY_COMPILER_SCHEMA_V2 &&
           adapter->adapter_id && adapter->adapter_version && adapter->target_id &&
           adapter->family && adapter->graph &&
           adapter->operator_graph_build &&
           adapter->execution_capabilities &&
           adapter->transformer_policy && adapter->logits_policy &&
           adapter->speculation_policy && adapter->tokenizer_policy &&
           adapter->physical_variant && pipeline &&
           pipeline->schema_version == YVEX_FAMILY_BINDING_PIPELINE_SCHEMA_V1 &&
           pipeline->source_open && pipeline->source_close && pipeline->artifact_admit &&
           pipeline->semantic_model_build &&
           pipeline->runtime_descriptor_build &&
           pipeline->quant_plan_default && pipeline->quant_plan_policy &&
           pipeline->tokenizer_architecture && pipeline->tokenizer_architecture[0] &&
           pipeline->tokenizer_pre && pipeline->tokenizer_pre[0];
}

static void binding_compiler_close(binding_compiler *compiler)
{
    if (!compiler) return;
    yvex_compiled_model_plan_close(&compiler->compiled_plan);
    yvex_physical_execution_ir_close(&compiler->physical_execution);
    yvex_gguf_writer_plan_release(&compiler->writer);
    yvex_quant_plan_release(&compiler->quant);
    yvex_imatrix_data_close(compiler->imatrix);
    yvex_quant_policy_close(compiler->quant_policy);
    yvex_attention_plan_close(compiler->attention);
    yvex_attention_plan_close(compiler->draft_attention);
    yvex_operator_graph_ir_close(&compiler->operator_graph);
    yvex_runtime_descriptor_close(compiler->descriptor);
    yvex_semantic_model_ir_close(&compiler->semantic_model);
    yvex_materialization_session_close(compiler->materialization);
    yvex_materialization_plan_close(compiler->materialization_plan);
    yvex_tensor_table_close(compiler->tensors);
    yvex_gguf_close(compiler->gguf);
    yvex_artifact_close(compiler->artifact);
    if (compiler->pipeline) compiler->pipeline->source_close(compiler->source.owner);
    memset(compiler, 0, sizeof(*compiler));
}

static int binding_compiler_open(
    binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    yvex_artifact_options options = {0};
    int rc;

    rc = compiler->pipeline->source_open(&compiler->source, request, err);
    options.path = request->artifact_path;
    options.readonly = 1;
    if (rc == YVEX_OK) rc = yvex_artifact_open(&compiler->artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&compiler->gguf, compiler->artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&compiler->tensors, compiler->gguf, err);
    if (rc == YVEX_OK)
        rc = compiler->pipeline->artifact_admit(
            compiler->artifact, &compiler->admission, &compiler->admission_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(
            compiler->artifact, &compiler->admission, NULL, NULL,
            &compiler->admission_failure, err);
    return rc;
}

static int binding_compiler_materialize(binding_compiler *compiler, yvex_error *err)
{
    int rc;

    yvex_materialization_options_default(&compiler->materialization_options);
    compiler->materialization_options.require_terminal_projection = 1;
    compiler->materialization_options.max_chunk_bytes = 16ull * 1024ull * 1024ull;
    compiler->materialization_options.cache_budget_bytes = 256ull * 1024ull * 1024ull;
    compiler->materialization_options.future_graph_scratch_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    compiler->materialization_options.future_kv_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    rc = yvex_materialization_project_artifact_lowering(
        compiler->source.lowering_context, &compiler->materialization_projection, err);
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
    return rc;
}

static int binding_compiler_graph(binding_compiler *compiler, yvex_error *err)
{
    int rc = compiler->pipeline->semantic_model_build(
        &compiler->semantic_model, compiler->source.verification, err);

    if (rc == YVEX_OK)
        rc = compiler->pipeline->runtime_descriptor_build(
            &compiler->descriptor, &compiler->admission, compiler->materialization,
            compiler->source.lowering_context, compiler->semantic_model, err);
    if (rc == YVEX_OK)
        rc = compiler->graph->plan_build(
            &compiler->attention, compiler->semantic_model, compiler->materialization,
            compiler->descriptor, &compiler->attention_failure, err);
    if (rc == YVEX_OK && compiler->graph->draft_plan_build)
        rc = compiler->graph->draft_plan_build(
            &compiler->draft_attention, compiler->semantic_model,
            compiler->materialization, compiler->descriptor,
            &compiler->attention_failure, err);
    if (rc == YVEX_OK)
        rc = compiler->adapter->operator_graph_build(
            &compiler->operator_graph, compiler->semantic_model,
            compiler->attention, compiler->draft_attention, err);
    return rc;
}

static int binding_compiler_imatrix(
    binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request,
    yvex_imatrix_data_summary *summary, yvex_error *err)
{
    yvex_imatrix_data_options options = {0};
    int rc;

    if (!request->imatrix_path) return YVEX_OK;
    if (!compiler->pipeline->imatrix_source_identity ||
        !compiler->pipeline->imatrix_dataset_identity ||
        !compiler->pipeline->imatrix_producer ||
        !compiler->pipeline->imatrix_producer_version) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.runtime-binding",
                       "family imatrix provenance is incomplete");
        return YVEX_ERR_STATE;
    }
    options.path = request->imatrix_path;
    options.source_model_identity = compiler->pipeline->imatrix_source_identity;
    options.calibration_dataset_identity = compiler->pipeline->imatrix_dataset_identity;
    options.producer = compiler->pipeline->imatrix_producer;
    options.producer_version = compiler->pipeline->imatrix_producer_version;
    options.maximum_mapped_bytes = 1024u * 1024u * 1024u;
    rc = yvex_imatrix_data_open(&compiler->imatrix, &options, err);
    return rc == YVEX_OK ? yvex_imatrix_data_get_summary(compiler->imatrix, summary, err) : rc;
}

static int binding_compiler_quant(
    binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    yvex_imatrix_data_summary imatrix = {0};
    int rc = YVEX_OK;

    if (request->physical_variant_plan_path) {
        if ((request->quant_policy_path != NULL) == (request->quant_preset_name != NULL)) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "compilation.runtime-binding",
                           "variant preparation requires exactly one quant policy or preset");
            return YVEX_ERR_INVALID_ARG;
        }
        rc = request->quant_policy_path
                 ? yvex_quant_policy_open(&compiler->quant_policy,
                                          request->quant_policy_path, err)
                 : yvex_quant_policy_preset_open(&compiler->quant_policy,
                                                 request->quant_preset_name, err);
        if (rc == YVEX_OK) rc = binding_compiler_imatrix(compiler, request, &imatrix, err);
        if (rc == YVEX_OK)
            rc = compiler->pipeline->quant_plan_policy(
                &compiler->quant, compiler->source.transform_ir,
                compiler->source.transform_binding, compiler->source.lowering_context,
                compiler->quant_policy,
                imatrix.complete ? imatrix.imatrix_identity : NULL, err);
        if (rc == YVEX_OK)
            rc = yvex_quant_plan_file_validate(
                request->physical_variant_plan_path, compiler->quant, err);
        return rc;
    }
    if (request->quant_policy_path || request->quant_preset_name || request->imatrix_path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "compilation.runtime-binding",
                       "policy, preset, and imatrix require a sealed physical variant plan");
        return YVEX_ERR_INVALID_ARG;
    }
    return compiler->pipeline->quant_plan_default(
        &compiler->quant, compiler->source.transform_ir,
        compiler->source.transform_binding, compiler->source.lowering_context, err);
}

static int binding_compiler_writer_build(
    binding_compiler *compiler, const char *required_execution_identity,
    yvex_error *err)
{
    const yvex_runtime_descriptor_summary *descriptor =
        yvex_runtime_descriptor_summary_get(compiler->descriptor);
    unsigned long long vocabulary_size =
        compiler->source.tokenizer_vocabulary_size
            ? compiler->source.tokenizer_vocabulary_size
            : descriptor ? descriptor->vocabulary_size : 0ull;
    yvex_gguf_writer_plan_options options;
    yvex_gguf_writer_plan_request writer = {0};

    if (!vocabulary_size) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.runtime-binding",
                       "sealed runtime vocabulary is required for artifact emission");
        return YVEX_ERR_STATE;
    }

    yvex_gguf_writer_plan_options_default(&options);
    options.required_execution_identity = required_execution_identity;
    writer.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    writer.quant_plan = compiler->quant;
    writer.options = &options;
    writer.input.complete.lowering = yvex_gguf_writer_artifact_lowering_api();
    writer.input.complete.lowering_context = compiler->source.lowering_context;
    writer.input.complete.verification = compiler->source.verification;
    writer.input.complete.tokenizer_architecture = compiler->pipeline->tokenizer_pre;
    writer.input.complete.tokenizer_vocabulary_size = vocabulary_size;
    return yvex_gguf_writer_plan_build(
        &compiler->writer, &writer, &compiler->writer_failure, err);
}

static int binding_compiler_writer(
    binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    const char *required = request->physical_variant_plan_path
                               ? NULL
                               : compiler->admission.quant_execution_identity;

    return binding_compiler_writer_build(compiler, required, err);
}

static int binding_compiler_prepare(
    binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request,
    yvex_family_compilation_products *products, yvex_error *err)
{
    const yvex_gguf_writer_plan_summary *writer =
        yvex_gguf_writer_plan_summary_get(compiler->writer);
    const yvex_transform_ir_summary *transform =
        yvex_transform_ir_summary_get(compiler->source.transform_ir);
    int rc = yvex_artifact_physical_compatibility_validate(
        compiler->writer, &compiler->admission, compiler->artifact, compiler->gguf,
        &compiler->compatibility, &compiler->compatibility_failure, err);

    if (rc == YVEX_OK &&
        (!writer || !transform || !yvex_sha256_hex_is_valid(transform->transform_identity) ||
         !compiler->compatibility.physical_payload_compatible)) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.runtime-binding",
                       "logical transform and physical compatibility proof are required");
        rc = YVEX_ERR_STATE;
    }
    if (rc != YVEX_OK) return rc;
    if (!compiler->adapter->execution_capabilities(&compiler->capabilities) ||
        !yvex_runtime_capabilities_contract_valid(&compiler->capabilities) ||
        !compiler->adapter->transformer_policy(
            yvex_runtime_descriptor_summary_get(compiler->descriptor),
            &compiler->transformer_policy) ||
        !compiler->adapter->logits_policy(&compiler->logits_policy) ||
        !compiler->adapter->speculation_policy(
            yvex_runtime_descriptor_summary_get(compiler->descriptor),
            &compiler->speculation_policy) ||
        !compiler->adapter->tokenizer_policy(&compiler->tokenizer_policy, err)) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.runtime-binding",
                       "family execution envelope compilation failed");
        return YVEX_ERR_STATE;
    }
    rc = yvex_physical_execution_ir_build(
        &compiler->physical_execution, compiler->materialization,
        compiler->descriptor, compiler->admission.profile_identity, err);
    if (rc == YVEX_OK) {
        yvex_compiled_model_plan_request plan = {
            .semantic_model = compiler->semantic_model,
            .operator_graph = compiler->operator_graph,
            .materialization = compiler->materialization,
            .descriptor = compiler->descriptor,
            .attention = compiler->attention,
            .draft_attention = compiler->draft_attention,
            .graph = compiler->graph,
            .family_adapter_id = request->family_adapter_id,
            .family_adapter_version = request->family_adapter_version,
            .capabilities = compiler->capabilities,
            .transformer_policy = compiler->transformer_policy,
            .logits_policy = compiler->logits_policy};
        rc = yvex_compiled_model_plan_build(
            &compiler->compiled_plan, &plan, err);
    }
    if (rc != YVEX_OK) return rc;
    products->directory = request->directory;
    products->admission = &compiler->admission;
    products->physical_compatibility = &compiler->compatibility;
    products->materialization = compiler->materialization;
    products->runtime_descriptor = compiler->descriptor;
    products->operator_graph = compiler->operator_graph;
    products->physical_execution = compiler->physical_execution;
    products->compiled_plan = compiler->compiled_plan;
    products->attention_plan = compiler->attention;
    products->draft_attention_plan = compiler->draft_attention;
    products->family_adapter_id = request->family_adapter_id;
    products->family_adapter_version = request->family_adapter_version;
    products->artifact_format = "gguf";
    products->artifact_format_version = writer->gguf_version;
    products->logical_transform_identity = transform->transform_identity;
    products->capabilities = &compiler->capabilities;
    products->transformer_policy = &compiler->transformer_policy;
    products->logits_policy = &compiler->logits_policy;
    products->speculation_policy = &compiler->speculation_policy;
    products->tokenizer_policy = &compiler->tokenizer_policy;
    return YVEX_OK;
}

int yvex_family_binding_compile(
    const yvex_family_compiler_adapter *adapter,
    const yvex_compilation_runtime_binding_request *request,
    yvex_family_compilation_products *products, void **owner, yvex_error *err)
{
    binding_compiler *compiler;
    int rc;

    if (products) {
        memset(products, 0, sizeof(*products));
        products->release = family_runtime_binding_release;
    }
    if (owner) *owner = NULL;
    if (!pipeline_valid(adapter) || !request || !products || !owner ||
        !request->source_path || !request->models_root ||
        !request->source_manifest_path || !request->artifact_path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "compilation.runtime-binding",
                       "complete family pipeline, source, artifact, and manifest are required");
        return YVEX_ERR_INVALID_ARG;
    }
    compiler = (binding_compiler *)calloc(1u, sizeof(*compiler));
    if (!compiler) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "compilation.runtime-binding",
                       "binding compiler allocation failed");
        return YVEX_ERR_NOMEM;
    }
    *owner = compiler;
    compiler->adapter = adapter;
    compiler->pipeline = adapter->binding_pipeline;
    compiler->graph = adapter->graph();
    if (!compiler->graph) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.runtime-binding",
                       "family graph compiler is unavailable");
        return YVEX_ERR_STATE;
    }
    rc = binding_compiler_open(compiler, request, err);
    if (rc == YVEX_OK) rc = binding_compiler_materialize(compiler, err);
    if (rc == YVEX_OK) rc = binding_compiler_graph(compiler, err);
    if (rc == YVEX_OK && !compiler->source.transform_ir) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.runtime-binding",
                       "family source projection did not provide sealed transform IR");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) rc = binding_compiler_quant(compiler, request, err);
    if (rc == YVEX_OK) rc = binding_compiler_writer(compiler, request, err);
    if (rc == YVEX_OK) rc = binding_compiler_prepare(compiler, request, products, err);
    return rc;
}

static void family_runtime_binding_release(void *owner)
{
    binding_compiler *compiler = (binding_compiler *)owner;

    if (!compiler) return;
    binding_compiler_close(compiler);
    free(compiler);
}

struct yvex_physical_variant_session {
    binding_compiler complete;
    yvex_component_variant_source component;
    yvex_quant_plan *component_quant;
    yvex_gguf_writer_plan *component_writer;
    yvex_physical_variant_summary summary;
    yvex_physical_variant_view view;
};

static int variant_refuse(yvex_status status, const char *reason, yvex_error *err)
{
    yvex_error_set(err, status, "compilation.physical-variant", reason);
    return status;
}

static void variant_release(yvex_physical_variant_session *session)
{
    if (!session) return;
    yvex_gguf_writer_plan_release(&session->component_writer);
    yvex_quant_plan_release(&session->component_quant);
    if (session->component.close) session->component.close(session->component.owner);
    binding_compiler_close(&session->complete);
    memset(session, 0, sizeof(*session));
}

static int variant_backend_validate(const char *backend, const yvex_quant_plan *plan,
                                    yvex_error *err)
{
    const yvex_quant_plan_summary *summary = yvex_quant_plan_summary_get(plan);
    unsigned long long ordinal;
    int compatible;

    if (!backend) return YVEX_OK;
    compatible = summary && summary->complete;
    for (ordinal = 0u; compatible && ordinal < summary->decision_count; ++ordinal) {
        const yvex_quant_decision *decision =
            yvex_quant_plan_decision_at(plan, ordinal);

        compatible = decision &&
                     ((strcmp(backend, "cpu") == 0 && decision->cpu_compute_available) ||
                      (strcmp(backend, "cuda") == 0 && decision->cuda_compute_available));
    }
    if (compatible) return YVEX_OK;
    yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "compilation.physical-variant.backend",
                    "physical variant is not executable on requested backend: %s", backend);
    return YVEX_ERR_UNSUPPORTED;
}

static int variant_complete_request_validate(
    const yvex_physical_variant_request *request, yvex_error *err)
{
    if (request->component_id)
        return variant_refuse(
            YVEX_ERR_INVALID_ARG,
            "complete-model preparation does not accept a component", err);
    if (!!request->quant_preset_name == !!request->quant_policy_path)
        return variant_refuse(
            YVEX_ERR_INVALID_ARG,
            "complete-model preparation requires exactly one quant preset or policy", err);
    if (request->backend && strcmp(request->backend, "cpu") != 0 &&
        strcmp(request->backend, "cuda") != 0)
        return variant_refuse(
            YVEX_ERR_INVALID_ARG,
            "complete-model backend must be cpu or cuda", err);
    return YVEX_OK;
}

static int variant_complete_quant(
    yvex_physical_variant_session *session,
    const yvex_physical_variant_request *request,
    const yvex_compilation_runtime_binding_request *source,
    yvex_error *err)
{
    binding_compiler *compiler = &session->complete;
    yvex_imatrix_data_summary imatrix = {0};
    int rc;

    rc = request->quant_preset_name
             ? yvex_quant_policy_preset_open(
                   &compiler->quant_policy, request->quant_preset_name, err)
             : yvex_quant_policy_open(
                   &compiler->quant_policy, request->quant_policy_path, err);
    if (rc == YVEX_OK) rc = binding_compiler_imatrix(compiler, source, &imatrix, err);
    if (rc == YVEX_OK)
        rc = compiler->pipeline->quant_plan_policy(
            &compiler->quant, compiler->source.transform_ir,
            compiler->source.transform_binding, compiler->source.lowering_context,
            compiler->quant_policy, imatrix.complete ? imatrix.imatrix_identity : NULL, err);
    if (rc == YVEX_OK)
        rc = variant_backend_validate(request->backend, compiler->quant, err);
    if (rc == YVEX_OK) rc = binding_compiler_writer_build(compiler, NULL, err);
    return rc;
}

static int variant_complete_open(
    yvex_physical_variant_session *session,
    const yvex_graph_execution_binding *execution,
    const yvex_physical_variant_request *request,
    yvex_error *err)
{
    binding_compiler *compiler = &session->complete;
    yvex_compilation_runtime_binding_request source = {0};
    const yvex_transform_ir_summary *transform;
    int rc;

    rc = variant_complete_request_validate(request, err);
    if (rc != YVEX_OK) return rc;
    if (!pipeline_valid(execution->compiler))
        return variant_refuse(YVEX_ERR_STATE,
                              "target published an invalid compiler adapter", err);
    compiler->adapter = execution->compiler;
    compiler->pipeline = execution->compiler->binding_pipeline;
    source.source_path = request->source_path;
    source.models_root = request->models_root;
    source.source_manifest_path = request->source_manifest_path;
    source.quant_policy_path = request->quant_policy_path;
    source.quant_preset_name = request->quant_preset_name;
    source.imatrix_path = request->imatrix_path;
    source.source_stream_count = request->worker_count;
    rc = compiler->pipeline->source_open(&compiler->source, &source, err);
    if (rc == YVEX_OK &&
        (!compiler->source.verification || !compiler->source.verification->verified))
        rc = variant_refuse(YVEX_ERR_STATE,
                            "family source did not publish verified source facts", err);
    if (rc == YVEX_OK) rc = variant_complete_quant(session, request, &source, err);
    transform = rc == YVEX_OK
                    ? yvex_transform_ir_summary_get(compiler->source.transform_ir)
                    : NULL;
    if (rc == YVEX_OK && !transform)
        rc = variant_refuse(YVEX_ERR_STATE,
                            "family source did not publish a sealed transformation", err);
    if (rc == YVEX_OK) {
        session->summary.schema_version = YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1;
        session->summary.kind = YVEX_PHYSICAL_VARIANT_COMPLETE_MODEL;
        session->summary.worker_count = request->worker_count;
        session->summary.source_verified =
            compiler->source.verification && compiler->source.verification->verified;
        yvex_core_text_copy(session->summary.target_id,
                            sizeof(session->summary.target_id), execution->target_id);
        yvex_core_text_copy(session->summary.family,
                            sizeof(session->summary.family), execution->compiler->family);
        yvex_core_text_copy(session->summary.transform_identity,
                            sizeof(session->summary.transform_identity),
                            transform->transform_identity);
    }
    return rc;
}

static int variant_component_adapter_validate(
    const yvex_component_variant_adapter *adapter,
    const yvex_physical_variant_request *request, yvex_error *err)
{
    if (!adapter ||
        adapter->schema_version != YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1 ||
        !adapter->target_id || strcmp(adapter->target_id, request->target_id) != 0 ||
        !adapter->family || !adapter->family[0] ||
        !adapter->source_revision || !adapter->source_revision[0] ||
        !adapter->profile_name || !adapter->profile_name[0] ||
        !adapter->source_open || !adapter->physical_variant ||
        (!!adapter->candidate_profile_name != !!adapter->candidate_component_id) ||
        (!!adapter->candidate_profile_name !=
         !!adapter->candidate_q8_semantic_role_mask))
        return variant_refuse(YVEX_ERR_STATE,
                              "target published an invalid component compiler adapter", err);
    return YVEX_OK;
}

static int variant_q8_select(const yvex_transform_value *terminal,
                             unsigned int *qtype, int *approximation, void *context)
{
    const yvex_component_variant_adapter *adapter = context;
    unsigned long long role = terminal ? terminal->logical_key.semantic_role : 64ull;

    if (!terminal || !qtype || !approximation || !adapter) return 0;
    if (terminal->dtype == YVEX_TRANSFORM_DTYPE_BF16 && terminal->shape.rank == 2u &&
        terminal->shape.dims[1] % 256ull == 0ull && role < 64ull &&
        (adapter->candidate_q8_semantic_role_mask & (1ull << role)) != 0ull) {
        *qtype = YVEX_GGUF_QTYPE_Q8_0;
        *approximation = 1;
    }
    return 1;
}

static int variant_component_source_validate(
    const yvex_component_variant_source *source,
    const yvex_component_variant_adapter *adapter,
    const yvex_physical_variant_request *request, yvex_error *err)
{
    const yvex_physical_variant_summary *summary = &source->summary;

    if (!source->owner || !source->close || !source->transform_ir ||
        !source->transform_binding || !source->architecture[0] ||
        strcmp(source->target_id, adapter->target_id) != 0 ||
        strcmp(source->component_id, request->component_id) != 0 ||
        !source->source_snapshot_identity[0] || !source->component_identity[0] ||
        !source->component_manifest_identity[0] || !source->architecture_identity[0] ||
        !source->role_map_identity[0] ||
        summary->schema_version != YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1 ||
        summary->kind != YVEX_PHYSICAL_VARIANT_COMPONENT ||
        strcmp(summary->target_id, adapter->target_id) != 0 ||
        strcmp(summary->component_id, request->component_id) != 0 ||
        !summary->source_verified)
        return variant_refuse(YVEX_ERR_STATE,
                              "component compiler adapter published incomplete source facts", err);
    return YVEX_OK;
}

static int variant_component_writer(yvex_physical_variant_session *session,
                                    yvex_error *err)
{
    yvex_gguf_writer_plan_options options;
    yvex_gguf_writer_plan_request writer = {0};
    yvex_gguf_writer_failure failure = {0};

    yvex_gguf_writer_plan_options_default(&options);
    writer.input_class = YVEX_GGUF_WRITER_INPUT_LOGICAL_COMPONENT;
    writer.quant_plan = session->component_quant;
    writer.options = &options;
    writer.input.component.architecture = session->component.architecture;
    writer.input.component.target_id = session->component.target_id;
    writer.input.component.component_id = session->component.component_id;
    writer.input.component.source_snapshot_identity =
        session->component.source_snapshot_identity;
    writer.input.component.source_snapshot_key = session->component.source_snapshot_key;
    writer.input.component.component_identity = session->component.component_identity;
    writer.input.component.component_manifest_identity =
        session->component.component_manifest_identity;
    writer.input.component.architecture_identity =
        session->component.architecture_identity;
    writer.input.component.role_map_identity = session->component.role_map_identity;
    return yvex_gguf_writer_plan_build(
        &session->component_writer, &writer, &failure, err);
}

static int variant_component_open(
    yvex_physical_variant_session *session,
    const yvex_component_variant_adapter *adapter,
    const yvex_physical_variant_request *request,
    yvex_error *err)
{
    const char *profile_name = request->quant_preset_name
                                   ? request->quant_preset_name
                                   : adapter->profile_name;
    yvex_quant_failure failure = {0};
    yvex_quant_plan_options quant_options = {0};
    int candidate_profile = adapter && adapter->candidate_profile_name &&
                            strcmp(profile_name, adapter->candidate_profile_name) == 0;
    yvex_component_variant_source_request source = {
        .source_path = request->source_path,
        .component_id = request->component_id,
        .candidate_q8 = candidate_profile};
    int rc;

    if (candidate_profile) {
        quant_options.identity_override = variant_q8_select;
        quant_options.identity_override_context = (void *)adapter;
    }

    if (!request->component_id || request->models_root ||
        request->source_manifest_path || request->quant_policy_path ||
        request->imatrix_path ||
        (request->backend && strcmp(request->backend, "cpu") != 0 &&
         strcmp(request->backend, "cuda") != 0))
        return variant_refuse(
            YVEX_ERR_INVALID_ARG,
            "component preparation accepts source, component, preset, and backend", err);
    rc = variant_component_adapter_validate(adapter, request, err);
    if (rc == YVEX_OK && strcmp(profile_name, adapter->profile_name) != 0 &&
        !candidate_profile)
        rc = variant_refuse(YVEX_ERR_UNSUPPORTED,
                            "component physical profile is not admitted", err);
    if (rc == YVEX_OK && candidate_profile &&
        strcmp(request->component_id, adapter->candidate_component_id) != 0)
        rc = variant_refuse(YVEX_ERR_UNSUPPORTED,
                            "component alternate profile targets another component", err);
    if (rc == YVEX_OK) rc = adapter->source_open(&session->component, &source, err);
    if (rc == YVEX_OK)
        rc = variant_component_source_validate(
            &session->component, adapter, request, err);
    if (rc == YVEX_OK)
        rc = yvex_quant_plan_build_identity(
            &session->component_quant, session->component.transform_ir,
            session->component.transform_binding, profile_name,
            yvex_transform_hash_string(candidate_profile
                                           ? profile_name
                                           : session->component.component_identity),
            candidate_profile ? &quant_options : NULL, &failure, err);
    if (rc == YVEX_OK)
        rc = variant_backend_validate(request->backend,
                                      session->component_quant, err);
    if (rc == YVEX_OK) rc = variant_component_writer(session, err);
    if (rc == YVEX_OK) {
        session->summary = session->component.summary;
        session->summary.worker_count = request->worker_count;
    }
    return rc;
}

static int physical_variant_session_open(
    yvex_physical_variant_session **out,
    const yvex_physical_variant_request *request,
    yvex_error *err)
{
    const yvex_graph_execution_binding *execution;
    const yvex_component_variant_adapter *component;
    yvex_physical_variant_session *session;
    int rc;

    if (out) *out = NULL;
    yvex_error_clear(err);
    if (!out || !request || !request->target_id || !request->target_id[0] ||
        !request->source_path || !request->source_path[0] ||
        request->worker_count == 0u || request->worker_count > 64u)
        return variant_refuse(YVEX_ERR_INVALID_ARG,
                              "target, source, output, and bounded workers are required", err);
    execution = yvex_graph_execution_find(0u, 0u, request->target_id);
    component = execution ? NULL : yvex_graph_component_variant_find(request->target_id);
    if (!execution && !component)
        return variant_refuse(
            YVEX_ERR_UNSUPPORTED,
            "target has no physical-variant compiler adapter", err);
    session = (yvex_physical_variant_session *)calloc(1u, sizeof(*session));
    if (!session)
        return variant_refuse(YVEX_ERR_NOMEM, "session allocation failed", err);
    rc = execution ? variant_complete_open(session, execution, request, err)
                   : variant_component_open(session, component, request, err);
    if (rc != YVEX_OK) {
        variant_release(session);
        free(session);
        return rc;
    }
    session->view.summary = &session->summary;
    session->view.plan = session->summary.kind == YVEX_PHYSICAL_VARIANT_COMPONENT
                             ? session->component_quant
                             : session->complete.quant;
    session->view.writer = session->summary.kind == YVEX_PHYSICAL_VARIANT_COMPONENT
                               ? session->component_writer
                               : session->complete.writer;
    session->view.imatrix = session->summary.kind == YVEX_PHYSICAL_VARIANT_COMPLETE_MODEL
                                ? session->complete.imatrix
                                : NULL;
    *out = session;
    return YVEX_OK;
}

static void physical_variant_session_close(yvex_physical_variant_session **address)
{
    yvex_physical_variant_session *session;

    if (!address || !*address) return;
    session = *address;
    *address = NULL;
    variant_release(session);
    free(session);
}

static const yvex_physical_variant_view *physical_variant_session_view(
    const yvex_physical_variant_session *session)
{
    return session ? &session->view : NULL;
}

const yvex_physical_variant_api *yvex_graph_physical_variant_api_get(void)
{
    static const yvex_physical_variant_api api = {
        YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1,
        physical_variant_session_open,
        physical_variant_session_close,
        physical_variant_session_view};

    return &api;
}
