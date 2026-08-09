/*
 * Own the generic execution-compilation lifecycle that lowers one family into binding products.
 *
 * Family callbacks provide semantic projections and numeric policy. This owner opens and closes
 * artifact, materialization, graph, quant, and writer resources in one deterministic order; it
 * never identifies a concrete family or reconstructs its topology.
 */
#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/compiler.h>
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
           adapter->adapter_id && adapter->adapter_version && adapter->graph &&
           adapter->operator_graph_build && adapter->physical_execution_policy &&
           adapter->execution_capabilities &&
           adapter->transformer_policy && adapter->logits_policy &&
           adapter->speculation_policy && adapter->tokenizer_policy && pipeline &&
           pipeline->schema_version == YVEX_FAMILY_BINDING_PIPELINE_SCHEMA_V1 &&
           pipeline->source_open && pipeline->source_close && pipeline->artifact_admit &&
           pipeline->materialization_project && pipeline->semantic_model_build &&
           pipeline->runtime_descriptor_build &&
           pipeline->quant_plan_default && pipeline->quant_plan_policy &&
           pipeline->writer_lowering;
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
    rc = compiler->pipeline->materialization_project(
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
    const yvex_runtime_descriptor_summary *descriptor_summary;
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
    descriptor_summary = rc == YVEX_OK
                             ? yvex_runtime_descriptor_summary_get(compiler->descriptor)
                             : NULL;
    if (rc == YVEX_OK && !descriptor_summary) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.operator-graph",
                       "family descriptor did not publish sealed model semantics");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = compiler->adapter->operator_graph_build(
            &compiler->operator_graph, compiler->semantic_model,
            &descriptor_summary->model_execution, compiler->attention,
            compiler->draft_attention, err);
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

static int binding_compiler_writer(
    binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    yvex_gguf_writer_plan_options options;
    yvex_gguf_writer_plan_request writer = {0};

    yvex_gguf_writer_plan_options_default(&options);
    options.required_execution_identity =
        request->physical_variant_plan_path ? NULL : compiler->admission.quant_execution_identity;
    writer.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    writer.quant_plan = compiler->quant;
    writer.options = &options;
    writer.input.complete.lowering = compiler->pipeline->writer_lowering();
    writer.input.complete.lowering_context = compiler->source.lowering_context;
    writer.input.complete.verification = compiler->source.verification;
    if (!writer.input.complete.lowering) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.runtime-binding",
                       "family writer lowering is unavailable");
        return YVEX_ERR_STATE;
    }
    return yvex_gguf_writer_plan_build(
        &compiler->writer, &writer, &compiler->writer_failure, err);
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
        compiler->descriptor, compiler->admission.profile_identity,
        compiler->adapter->physical_execution_policy, err);
    if (rc == YVEX_OK) {
        yvex_compiled_model_plan_request plan = {
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
