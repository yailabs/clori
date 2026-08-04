/* Exercise the family facts and component-aware IR without source payloads or execution. */
#include "tests/test.h"

#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/model_target.h>

#include <string.h>

#define TEST_ID_A "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#define TEST_ID_B "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

static int test_components(void)
{
    yvex_minimax_h3_component first[YVEX_MINIMAX_H3_COMPONENT_COUNT];
    yvex_minimax_h3_component second[YVEX_MINIMAX_H3_COMPONENT_COUNT];
    yvex_minimax_h3_component invalid[YVEX_MINIMAX_H3_COMPONENT_COUNT];
    yvex_minimax_h3_failure failure;
    yvex_error err;
    char first_identity[65];
    char second_identity[65];
    unsigned int index;
    unsigned int weighted = 0u;

    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->components_canonical(
                         first, first_identity, &failure, &err) == YVEX_OK,
                     "canonical composite target is admitted");
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->components_canonical(
                         second, second_identity, &failure, &err) == YVEX_OK,
                     "canonical composite target repeats");
    YVEX_TEST_ASSERT_STREQ(first_identity, second_identity,
                           "component identity is deterministic");
    for (index = 0u; index < YVEX_MINIMAX_H3_COMPONENT_COUNT; ++index) {
        YVEX_TEST_ASSERT(first[index].id == (yvex_minimax_h3_component_id)index,
                         "every required component is present exactly once");
        YVEX_TEST_ASSERT_STREQ(first[index].identity, second[index].identity,
                               "component identities repeat");
        weighted += first[index].weighted ? 1u : 0u;
    }
    YVEX_TEST_ASSERT(weighted == YVEX_MINIMAX_H3_WEIGHTED_COMPONENTS,
                     "exactly four components own weights");
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->component_graph_validate(
                         first, YVEX_MINIMAX_H3_COMPONENT_COUNT,
                         &failure, &err) == YVEX_OK,
                     "canonical component DAG validates");
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->component_graph_validate(
                         first, YVEX_MINIMAX_H3_COMPONENT_COUNT - 1u,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                     "missing component is refused");

    memcpy(invalid, first, sizeof(invalid));
    invalid[1].id = invalid[0].id;
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->component_graph_validate(
                         invalid, YVEX_MINIMAX_H3_COMPONENT_COUNT,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                     "duplicate component is refused");
    memcpy(invalid, first, sizeof(invalid));
    invalid[YVEX_MINIMAX_H3_COMPONENT_PROCESSOR].dependency_mask =
        1u << YVEX_MINIMAX_H3_COMPONENT_TOKENIZER;
    invalid[YVEX_MINIMAX_H3_COMPONENT_TOKENIZER].dependency_mask =
        1u << YVEX_MINIMAX_H3_COMPONENT_PROCESSOR;
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->component_graph_validate(
                         invalid, YVEX_MINIMAX_H3_COMPONENT_COUNT,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_COMPONENT_CYCLE,
                     "component cycle is refused");
    memcpy(invalid, first, sizeof(invalid));
    invalid[YVEX_MINIMAX_H3_COMPONENT_PROCESSOR].phase =
        YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE;
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->component_graph_validate(
                         invalid, YVEX_MINIMAX_H3_COMPONENT_COUNT,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_PHASE_ORDER,
                     "dependency-inconsistent phase order is refused");
    memcpy(invalid, first, sizeof(invalid));
    invalid[YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE].output_classes = 0u;
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->component_graph_validate(
                         invalid, YVEX_MINIMAX_H3_COMPONENT_COUNT,
                         &failure, &err) != YVEX_OK,
                     "output component without an output domain is refused");
    return 0;
}

static int test_architecture(void)
{
    yvex_minimax_h3_architecture first;
    yvex_minimax_h3_architecture second;
    yvex_minimax_h3_failure failure;
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->architecture_canonical(
                         &first, &failure, &err) == YVEX_OK,
                     "canonical architecture is available");
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->architecture_canonical(
                         &second, &failure, &err) == YVEX_OK,
                     "canonical architecture repeats");
    YVEX_TEST_ASSERT_STREQ(first.identity, second.identity,
                           "architecture identity is deterministic");
    YVEX_TEST_ASSERT(first.encoder.text_layers == 64u &&
                         first.encoder.text_width == 5120u &&
                         first.encoder.vision_layers == 27u,
                     "encoder signature retains exact source geometry");
    YVEX_TEST_ASSERT(first.omni.blocks == 50u && first.omni.width == 5376u &&
                         first.omni.video_channels == 24u &&
                         first.omni.audio_channels == 32u,
                     "Omni signature retains exact source geometry");
    YVEX_TEST_ASSERT(first.video_vae.spatial_ratio == 16u &&
                         first.video_vae.temporal_ratio == 4u &&
                         first.audio_vae.decoder_rate_product == 800u &&
                         first.audio_vae.latent_steps_per_second == 40u,
                     "VAE signatures retain exact source geometry");
    return 0;
}

static yvex_native_weight_info test_tensor(const char *name,
                                           const char *shard,
                                           yvex_native_dtype dtype,
                                           unsigned long long width)
{
    yvex_native_weight_info tensor;

    memset(&tensor, 0, sizeof(tensor));
    tensor.name = name;
    tensor.shard_path = shard;
    tensor.dtype = dtype;
    tensor.rank = 1u;
    tensor.dims[0] = width;
    tensor.data_start = 8u;
    tensor.data_end = 8u + width * (dtype == YVEX_NATIVE_DTYPE_BF16 ? 2u : 4u);
    tensor.data_bytes = tensor.data_end - tensor.data_start;
    return tensor;
}

static int test_roles(void)
{
    yvex_native_weight_info tensor = test_tensor(
        "blocks.0.norm1.weight", "FL2VA/transformer/model-00001-of-00013.safetensors",
        YVEX_NATIVE_DTYPE_BF16, 5376u);
    yvex_minimax_h3_tensor_role first;
    yvex_minimax_h3_tensor_role second;
    yvex_minimax_h3_failure failure;
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->tensor_classify(
                         &tensor, &first, &failure, &err) == YVEX_OK,
                     "valid tensor receives one canonical role");
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->tensor_classify(
                         &tensor, &second, &failure, &err) == YVEX_OK,
                     "role classification repeats");
    YVEX_TEST_ASSERT(first.role == YVEX_MINIMAX_H3_ROLE_OMNI_NORM &&
                         first.unresolved_requirement_identity == 0u,
                     "role and explicit known-state are retained");
    YVEX_TEST_ASSERT_STREQ(first.destination_identity, second.destination_identity,
                           "logical destination identity is deterministic");

    tensor.dtype = YVEX_NATIVE_DTYPE_F16;
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->tensor_classify(
                         &tensor, &first, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_DTYPE,
                     "component dtype mismatch is refused");
    tensor = test_tensor("unknown.weight",
                         "FL2VA/transformer/model-00001-of-00013.safetensors",
                         YVEX_NATIVE_DTYPE_BF16, 5376u);
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->tensor_classify(
                         &tensor, &first, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_TENSOR_ROLE,
                     "missing role is refused");
    tensor = test_tensor("blocks.0.attn.mlp.weight",
                         "FL2VA/transformer/model-00001-of-00013.safetensors",
                         YVEX_NATIVE_DTYPE_BF16, 5376u);
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->tensor_classify(
                         &tensor, &first, &failure, &err) != YVEX_OK,
                     "ambiguous role is refused");
    tensor = test_tensor("model.language_model.embed_tokens.weight",
                         "FL2VA/text_encoder/model-00001-of-00014.safetensors",
                         YVEX_NATIVE_DTYPE_BF16, 5120u);
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->tensor_classify(
                         &tensor, &first, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_SHAPE,
                     "canonical role shape mismatch is refused");
    tensor = test_tensor("blocks.0.norm1.weight",
                         "FL2VA/transformer/model-00001-of-00013.safetensors",
                         YVEX_NATIVE_DTYPE_BF16, 5376u);
    tensor.data_bytes--;
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->tensor_classify(
                         &tensor, &first, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_SOURCE_RANGE,
                     "source range mismatch is refused");
    return 0;
}

static int build_component_ir(char identity[65], unsigned long long *unknown)
{
    yvex_transform_header header;
    yvex_transform_builder_options options;
    yvex_transform_source_spec source;
    yvex_transform_value_spec value;
    yvex_transform_node_spec node;
    yvex_transform_builder *builder = NULL;
    yvex_transform_ir *ir = NULL;
    yvex_transform_failure failure;
    yvex_error err;
    const yvex_transform_ir_summary *summary;
    const yvex_transform_source_value *stored;
    unsigned long long source_id;
    unsigned long long terminal_id;
    unsigned long long node_id;
    int rc;

    memset(&header, 0, sizeof(header));
    header.schema_version = YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION;
    header.logical_model_identity = TEST_ID_A;
    header.source_snapshot_identity = 11u;
    header.coverage_identity = 12u;
    header.required_payload_identity = TEST_ID_B;
    header.payload_trust_class = "verified-source";
    header.component_manifest_identity = TEST_ID_A;
    header.architecture_identity = TEST_ID_B;
    header.role_map_identity = TEST_ID_A;
    header.unresolved_requirements_identity = TEST_ID_B;
    header.expected_source_count = 1u;
    header.expected_terminal_count = 1u;
    header.header_scan_count = 1u;
    memset(&options, 0, sizeof(options));
    yvex_transform_budget_default(&options.budget);
    rc = yvex_transform_builder_create(&builder, &header, &options, &failure, &err);
    if (rc != YVEX_OK) return 1;

    memset(&source, 0, sizeof(source));
    source.source_name = "time_embedder.proj_in.weight";
    source.shard_name = "FL2VA/transformer/model-00001-of-00013.safetensors";
    source.source_snapshot_identity = 11u;
    source.source_dtype = YVEX_NATIVE_DTYPE_F32;
    source.value_dtype = YVEX_TRANSFORM_DTYPE_F32;
    source.shape.rank = 2u;
    source.shape.dims[0] = 5376u;
    source.shape.dims[1] = 256u;
    source.relative_begin = 8u;
    source.relative_end = 16u;
    source.requirement_identity = 21u;
    source.scope = YVEX_TRANSFORM_SCOPE_GLOBAL;
    source.subsystem = YVEX_TRANSFORM_SUBSYSTEM_AUXILIARY;
    source.component_identity = 22u;
    source.semantic_role = YVEX_MINIMAX_H3_ROLE_TIMESTEP_PROJECTION;
    source.phase_identity = YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE;
    source.lifetime_identity = YVEX_MINIMAX_H3_LIFETIME_PHASE;
    source.unresolved_requirement_identity = 3u;
    source.layer_index = YVEX_TRANSFORM_IR_NO_ID;
    source.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    source.expert_index = YVEX_TRANSFORM_IR_NO_ID;
    source.required_uses = 1u;
    rc = yvex_transform_builder_add_source(builder, &source, &source_id, &failure, &err);
    if (rc != YVEX_OK) goto done;

    memset(&value, 0, sizeof(value));
    value.kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    value.semantic_id = 21u;
    value.canonical_ordinal = 0u;
    value.shape = source.shape;
    value.dtype = source.value_dtype;
    value.precision.flags = YVEX_TRANSFORM_PRECISION_EXACT;
    value.precision.allowed_physical_classes = YVEX_TRANSFORM_PHYSICAL_F32;
    value.logical_key.scope = source.scope;
    value.logical_key.subsystem = source.subsystem;
    value.logical_key.component_identity = source.component_identity;
    value.logical_key.semantic_role = source.semantic_role;
    value.logical_key.phase_identity = source.phase_identity;
    value.logical_key.lifetime_identity = source.lifetime_identity;
    value.logical_key.layer_index = YVEX_TRANSFORM_IR_NO_ID;
    value.logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    rc = yvex_transform_builder_declare_value(builder, &value, &terminal_id, &failure, &err);
    if (rc != YVEX_OK) goto done;
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.output_value_id = terminal_id;
    node.input_value_ids = &source_id;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    rc = yvex_transform_builder_add_node(builder, &node, &node_id, &failure, &err);
    if (rc != YVEX_OK) goto done;
    rc = yvex_transform_builder_seal(builder, &ir, &failure, &err);
    if (rc != YVEX_OK) goto done;
    summary = yvex_transform_ir_summary_get(ir);
    stored = yvex_transform_ir_source_at(ir, 0u);
    if (!summary || !stored || summary->schema_version != 2u ||
        summary->payload_bytes_read != 0u || stored->unresolved_requirement_identity != 3u) {
        rc = YVEX_ERR_STATE;
        goto done;
    }
    strcpy(identity, summary->transform_identity);
    *unknown = stored->unresolved_requirement_identity;
done:
    yvex_transform_ir_release(&ir);
    yvex_transform_builder_release(&builder);
    return rc == YVEX_OK ? 0 : 1;
}

static int test_component_ir(void)
{
    char first[65];
    char second[65];
    unsigned long long first_unknown = 0u;
    unsigned long long second_unknown = 0u;

    YVEX_TEST_ASSERT(build_component_ir(first, &first_unknown) == 0 &&
                         build_component_ir(second, &second_unknown) == 0,
                     "component-aware Transformation IR seals twice");
    YVEX_TEST_ASSERT_STREQ(first, second, "component-aware IR identity is deterministic");
    YVEX_TEST_ASSERT(first_unknown == 3u && second_unknown == 3u,
                     "unknown scheduler remains an explicit later-stage blocker");
    return 0;
}

static int test_operator_truth(void)
{
    yvex_model_target_request request;
    yvex_model_target_report report;
    const yvex_model_target_record *record;
    yvex_error err;

    memset(&request, 0, sizeof(request));
    request.kind = YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE;
    strcpy(request.target_id, YVEX_MINIMAX_H3_TARGET_ID);
    YVEX_TEST_ASSERT(yvex_model_target_report_build(&request, &report, &err) == YVEX_OK,
                     "existing model-target report route accepts MiniMax target identity");
    YVEX_TEST_ASSERT(report.exit_code == 5 && !report.family_architecture &&
                         strcmp(report.status, "source-acquisition-required") == 0,
                     "missing exact source refuses IR and support promotion");
    yvex_model_target_report_close(&report);
    record = yvex_model_target_find(YVEX_MINIMAX_H3_TARGET_ID);
    YVEX_TEST_ASSERT(record && strcmp(record->runtime_execution, "unsupported") == 0 &&
                         strcmp(record->generation, "unsupported") == 0,
                     "catalog target is explicitly non-runtime and non-generation");
    return 0;
}

int yvex_test_minimax_h3(void)
{
    if (test_components() != 0) return 1;
    if (test_architecture() != 0) return 1;
    if (test_roles() != 0) return 1;
    if (test_component_ir() != 0) return 1;
    if (test_operator_truth() != 0) return 1;
    return 0;
}
