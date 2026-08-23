/* Exercise the family facts and component-aware IR without source payloads or execution. */
#include "tests/test.h"

#include <yvex/internal/families/minimax_h3.h>

#include "src/graph/private.h"
#include <yvex/internal/artifact.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/joint_transformer.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/model_target.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/runtime.h>

#include <stdint.h>
#include <string.h>

#define TEST_ID_A "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#define TEST_ID_B "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int test_family_catalog(void)
{
    const yvex_component_variant_adapter *resolved =
        yvex_graph_component_variant_find(YVEX_MINIMAX_H3_TARGET_ID);
    yvex_compilation_runtime_binding_request request = {0};
    yvex_family_source_products products = {0};
    yvex_error err;

    YVEX_TEST_ASSERT(resolved &&
                         resolved->schema_version ==
                             YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1 &&
                         strcmp(resolved->target_id, YVEX_MINIMAX_H3_TARGET_ID) == 0 &&
                         resolved->source_open && resolved->candidate_profile_name &&
                         resolved->candidate_q8_semantic_role_mask ==
                             YVEX_MINIMAX_H3_TRANSFORMER_Q8_ROLE_MASK &&
                         !(resolved->candidate_q8_semantic_role_mask &
                           (1ull << YVEX_MINIMAX_H3_ROLE_OMNI_NORM)),
                     "generic family catalog resolves the exact MiniMax component adapter");
    YVEX_TEST_ASSERT(yvex_graph_component_variant_find(NULL) == NULL &&
                         yvex_graph_component_variant_find("unknown-family") == NULL,
                     "generic family catalog refuses absent and unknown component targets");
    request.source_path = "build/tests/missing-minimax-source";
    YVEX_TEST_ASSERT(
        yvex_family_source_compile("unknown-family", &request, &products, &err) ==
                YVEX_ERR_UNSUPPORTED &&
            !products.owner,
        "generic source catalog refuses an unknown family without partial products");
    YVEX_TEST_ASSERT(
        yvex_family_source_compile(
            YVEX_MINIMAX_H3_TARGET_ID, &request, &products, &err) != YVEX_OK &&
            !products.owner,
        "MiniMax source compilation is reached and refuses a missing exact source");
    return 0;
}

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
    yvex_minimax_h3_phase_edge edges[YVEX_MINIMAX_H3_PHASE_EDGES];

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
    for (index = 0u; index < YVEX_MINIMAX_H3_PHASE_EDGES; ++index) {
        const yvex_minimax_h3_phase_edge *edge =
            yvex_model_register_minimax_h3()->phase_edge_at(index);
        YVEX_TEST_ASSERT(edge != NULL, "every canonical phase edge is present");
        edges[index] = *edge;
    }
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->phase_graph_validate(
                         edges, YVEX_MINIMAX_H3_PHASE_EDGES,
                         &failure, &err) == YVEX_OK,
                     "canonical phase DAG reaches media publication");
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->phase_graph_validate(
                         edges, YVEX_MINIMAX_H3_PHASE_EDGES - 1u,
                         &failure, &err) != YVEX_OK,
                     "incomplete phase DAG is refused");
    edges[0].destination_phase = YVEX_MINIMAX_H3_PHASE_PREPARE;
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->phase_graph_validate(
                         edges, YVEX_MINIMAX_H3_PHASE_EDGES,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_MINIMAX_H3_FAILURE_PHASE_ORDER,
                     "cyclic or reversed phase edge is refused");
    edges[0] = *yvex_model_register_minimax_h3()->phase_edge_at(0u);
    edges[6] = edges[5];
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->phase_graph_validate(
                         edges, YVEX_MINIMAX_H3_PHASE_EDGES,
                         &failure, &err) != YVEX_OK,
                     "duplicate phase edge and missing audio publication are refused");
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
                         first.omni.audio_channels == 32u &&
                         first.omni.audio_patch_steps == 1u &&
                         first.omni.audio_patch_channels == 32u,
                     "Omni signature retains exact source geometry");
    YVEX_TEST_ASSERT(first.video_vae.spatial_ratio == 16u &&
                         first.video_vae.temporal_ratio == 4u &&
                         first.video_vae.base_channels == 128u &&
                         first.video_vae.stage_count == 6u &&
                         first.video_vae.channel_multipliers[5] == 8u &&
                         first.video_vae.conv3d && first.video_vae.encoder_tiling &&
                         first.audio_vae.decoder_rate_product == 800u &&
                         first.audio_vae.latent_steps_per_second == 40u &&
                         first.audio_vae.encoder_rates[4] == 5u &&
                         first.audio_vae.decoder_rates[6] == 2u,
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
    YVEX_TEST_ASSERT(report.exit_code == 5 &&
                         report.detail_kind == YVEX_MODEL_TARGET_DETAIL_NONE &&
                         strcmp(report.status, "source-acquisition-required") == 0,
                     "missing exact source refuses IR and support promotion");
    yvex_model_target_report_close(&report);
    record = yvex_model_target_find(YVEX_MINIMAX_H3_TARGET_ID);
    YVEX_TEST_ASSERT(record && strcmp(record->runtime_execution, "unsupported") == 0 &&
                         strcmp(record->generation, "unsupported") == 0,
                     "catalog target is explicitly non-runtime and non-generation");
    return 0;
}

static int test_semantic_composite(void)
{
    yvex_semantic_component components[2] = {0};
    yvex_semantic_phase_edge edges[1] = {{1u, 2u, 3u, 4u}};
    yvex_semantic_composite_request composite = {
        .repository = YVEX_MINIMAX_H3_REPOSITORY,
        .revision = YVEX_MINIMAX_H3_REVISION,
        .subtree = YVEX_MINIMAX_H3_SUBTREE,
        .source_snapshot_identity = TEST_ID_A,
        .component_manifest_identity = TEST_ID_B,
        .phase_dag_identity = TEST_ID_A,
        .architecture_identity = TEST_ID_B,
        .role_map_identity = TEST_ID_A,
        .unresolved_requirements_identity = TEST_ID_B,
        .weighted_components = 1ull,
        .shards = 2ull, .tensors = 3ull, .elements = 4ull, .payload_bytes = 5ull,
        .components = components, .component_count = 2ull,
        .phase_edges = edges, .phase_edge_count = 1ull};
    yvex_semantic_model_ir_request request = {
        .schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1,
        .family_adapter_id = 0x4d494e494d4158ull, .family_adapter_version = 1ull,
        .target_id = YVEX_MINIMAX_H3_TARGET_ID,
        .source_model_identity = TEST_ID_A,
        .logical_model_identity = TEST_ID_B,
        .semantic_payload_identity = TEST_ID_A,
        .composite = &composite};
    const yvex_semantic_component *stored_components = NULL;
    const yvex_semantic_phase_edge *stored_edges = NULL;
    unsigned long long component_count = 0ull, edge_count = 0ull;
    yvex_semantic_model_ir *semantic = NULL;
    yvex_error err;

    strcpy(components[0].canonical_id, "text");
    strcpy(components[0].identity, TEST_ID_A);
    components[0].weighted = 1;
    strcpy(components[1].canonical_id, "output");
    strcpy(components[1].identity, TEST_ID_B);
    YVEX_TEST_ASSERT(
        yvex_semantic_model_ir_seal(&semantic, &request, &err) == YVEX_OK &&
            yvex_semantic_model_ir_composite_view(
                semantic, &stored_components, &component_count,
                &stored_edges, &edge_count) &&
            component_count == 2ull && edge_count == 1ull &&
            strcmp(stored_components[1].canonical_id, "output") == 0 &&
            stored_edges[0].destination_phase == 2u,
        "generic Semantic Model IR owns a deterministic composite topology view");
    yvex_semantic_model_ir_close(&semantic);
    return 0;
}

static int test_canonical_operator_graph(void)
{
    yvex_semantic_model_ir_request semantic_request = {
        .schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1,
        .family_adapter_id = 0x4d494e494d4158ull,
        .family_adapter_version = 1ull,
        .target_id = YVEX_MINIMAX_H3_TARGET_ID,
        .source_model_identity = YVEX_MINIMAX_H3_SOURCE_TREE_IDENTITY,
        .logical_model_identity = YVEX_MINIMAX_H3_MODEL_INDEX_IDENTITY,
        .semantic_payload_identity = YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY};
    const yvex_operator_kind kinds[] = {
        YVEX_OPERATOR_TEXT_CONDITIONING, YVEX_OPERATOR_AUDIO_CODEC,
        YVEX_OPERATOR_MULTIMODAL_TRANSFORMER,
        YVEX_OPERATOR_RECTIFIED_FLOW};
    yvex_operator_node nodes[4] = {0};
    yvex_operator_edge edges[3] = {0};
    yvex_operator_graph_request graph_request = {0};
    yvex_semantic_model_ir *semantic = NULL;
    yvex_operator_graph_ir *first = NULL, *second = NULL;
    const yvex_operator_graph_summary *summary;
    yvex_error err;
    unsigned int index;

    YVEX_TEST_ASSERT(yvex_semantic_model_ir_seal(
                         &semantic, &semantic_request, &err) == YVEX_OK,
                     "MiniMax component semantics seal without transformer context");
    for (index = 0u; index < 4u; ++index) {
        nodes[index].schema_version = YVEX_OPERATOR_GRAPH_SCHEMA_V1;
        nodes[index].ordinal = index;
        nodes[index].kind = kinds[index];
        nodes[index].scope = YVEX_TENSOR_SCOPE_GLOBAL;
        nodes[index].layer_index = YVEX_OPERATOR_GRAPH_NO_NODE;
        nodes[index].input_width = nodes[index].output_width = 1ull;
        nodes[index].numeric_contract =
            YVEX_OPERATOR_NUMERIC_REFERENCE_TOLERANCE;
        strcpy(nodes[index].attribute_identity,
               YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY);
        if (index < 3u) {
            edges[index].schema_version = YVEX_OPERATOR_GRAPH_SCHEMA_V1;
            edges[index].ordinal = index;
            edges[index].source_node = index;
            edges[index].target_node = index + 1u;
            edges[index].kind = YVEX_OPERATOR_EDGE_ORDER;
            edges[index].state_class = YVEX_MODEL_STATE_CLASS_COUNT;
        }
    }
    graph_request.semantic_model = semantic;
    graph_request.nodes = nodes;
    graph_request.node_count = 4ull;
    graph_request.edges = edges;
    graph_request.edge_count = 3ull;
    YVEX_TEST_ASSERT(yvex_operator_graph_ir_seal(
                         &first, &graph_request, &err) == YVEX_OK,
                     "MiniMax component DAG lowers into canonical operator truth");
    summary = yvex_operator_graph_ir_summary(first);
    YVEX_TEST_ASSERT(summary && summary->node_count == 4ull &&
                         summary->edge_count == 3ull &&
                         summary->maximum_context == 0ull &&
                         yvex_sha256_hex_valid(summary->identity),
                     "non-transformer graph keeps context capability absent");
    YVEX_TEST_ASSERT(yvex_operator_graph_ir_seal(
                         &second, &graph_request, &err) == YVEX_OK &&
                         strcmp(summary->identity,
                                yvex_operator_graph_ir_summary(second)->identity) == 0,
                     "canonical component graph identity repeats");
    yvex_operator_graph_ir_close(&second);
    edges[1].source_node = 2ull;
    edges[1].target_node = 1ull;
    YVEX_TEST_ASSERT(yvex_operator_graph_ir_seal(
                         &second, &graph_request, &err) == YVEX_ERR_INVALID_ARG &&
                         !second,
                     "reversed component dependency is refused");
    yvex_operator_graph_ir_close(&first);
    yvex_semantic_model_ir_close(&semantic);
    return 0;
}

static int test_audio_numeric_primitives(void)
{
    const float input[] = {1.0f, 2.0f, 3.0f};
    const float weight[] = {1.0f, 2.0f, 1.0f};
    const float gain[] = {2.449489742783178f};
    const float transpose_input[] = {1.0f, 2.0f};
    float output[3];
    float alias_input[] = {1.0f, 3.0f};
    float alias_output[2];
    float scratch[4];
    float alpha[] = {0.0f};
    float beta[] = {0.0f};
    float up_filter[12] = {0};
    float down_filter[12] = {0};
    yvex_graph_conv1d_geometry geometry = {1ull, 1ull, 1ull, 3ull, 3ull,
                                           1ull, 1ull, 1ull, 0ull, 0};
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_graph_conv1d_f32(
                         &geometry, input, 3ull, weight, 3ull, NULL, 0ull,
                         gain, 1ull, output, 3ull, &err) == YVEX_OK,
                     "weight-normalized Conv1D executes");
    YVEX_TEST_ASSERT(fabsf(output[0] - 4.0f) < 1.0e-5f &&
                         fabsf(output[1] - 8.0f) < 1.0e-5f &&
                         fabsf(output[2] - 8.0f) < 1.0e-5f,
                     "Conv1D matches the independent hand calculation");

    geometry.input_length = 2ull;
    geometry.stride = 2ull;
    geometry.transposed = 1;
    YVEX_TEST_ASSERT(yvex_graph_conv1d_f32(
                         &geometry, transpose_input, 2ull, weight, 3ull,
                         NULL, 0ull, gain, 1ull, output, 3ull, &err) == YVEX_OK,
                     "weight-normalized transposed Conv1D executes");
    YVEX_TEST_ASSERT(fabsf(output[0] - 2.0f) < 1.0e-5f &&
                         fabsf(output[1] - 3.0f) < 1.0e-5f &&
                         fabsf(output[2] - 4.0f) < 1.0e-5f,
                     "transposed Conv1D matches the independent hand calculation");
    geometry.output_padding = geometry.stride;
    YVEX_TEST_ASSERT(yvex_graph_conv1d_f32(
                         &geometry, transpose_input, 2ull, weight, 3ull,
                         NULL, 0ull, gain, 1ull, output, 3ull, &err) == YVEX_ERR_INVALID_ARG,
                     "transposed Conv1D refuses output padding outside its stride");
    geometry.output_padding = 0ull;

    up_filter[5] = 0.5f;
    down_filter[5] = 1.0f;
    YVEX_TEST_ASSERT(yvex_graph_alias_snake_f32(
                         alias_input, 1ull, 1ull, 2ull, alpha, beta,
                         up_filter, down_filter, alias_output, scratch, 4ull,
                         &err) == YVEX_OK,
                     "alias-free SnakeBeta executes with bounded scratch");
    YVEX_TEST_ASSERT(fabsf(alias_output[0] - (1.0f + sinf(1.0f) * sinf(1.0f))) < 1.0e-5f &&
                         fabsf(alias_output[1] - (3.0f + sinf(3.0f) * sinf(3.0f))) < 1.0e-5f,
                     "alias-free activation matches the independent filter calculation");
    YVEX_TEST_ASSERT(yvex_graph_alias_snake_f32(
                         alias_input, 1ull, 1ull, 2ull, alpha, beta,
                         up_filter, down_filter, alias_output, scratch, 3ull,
                         &err) == YVEX_ERR_INVALID_ARG,
                     "alias-free activation refuses insufficient scratch");
    down_filter[5] = INFINITY;
    YVEX_TEST_ASSERT(yvex_graph_alias_snake_f32(
                         alias_input, 1ull, 1ull, 2ull, alpha, beta,
                         up_filter, down_filter, alias_output, scratch, 4ull,
                         &err) == YVEX_ERR_FORMAT,
                     "alias-free activation refuses a non-finite downsampling result");
    return 0;
}

static int test_video_numeric_primitives(void)
{
    const float linear_input[4] = {1.0f, 2.0f, -1.0f, 3.0f};
    const float linear_weight[4] = {1.0f, 0.0f, 0.0f, 2.0f};
    const float linear_bias[2] = {0.5f, -1.0f};
    float linear_output[4];
    float layer_values[2] = {1.0f, 3.0f};
    const float layer_weight[2] = {1.0f, 1.0f};
    const float layer_bias[2] = {0.0f, 0.0f};
    const float fused[2] = {0.0f, 2.0f};
    float gated[1];
    const float qkv[12] = {
        1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f,
        0.0f, 1.0f, 0.0f, 1.0f, 3.0f, 4.0f,
    };
    float attention[4];
    float scratch[2];
    float selected = expf(1.0f / sqrtf(2.0f));
    float expected_first = (selected * 1.0f + 3.0f) / (selected + 1.0f);
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_graph_linear_source_f32(
                         linear_input, 4ull, 2ull, 2ull,
                         linear_weight, 4ull, linear_bias, 2ull, 2ull,
                         linear_output, 4ull, &err) == YVEX_OK,
                     "source-layout linear projection is executable");
    YVEX_TEST_ASSERT(fabsf(linear_output[0] - 1.5f) < 1.0e-6f &&
                         fabsf(linear_output[1] - 3.0f) < 1.0e-6f &&
                         fabsf(linear_output[2] + 0.5f) < 1.0e-6f &&
                         fabsf(linear_output[3] - 5.0f) < 1.0e-6f,
                     "source-layout linear projection matches the independent result");
    YVEX_TEST_ASSERT(yvex_graph_linear_source_f32(
                         linear_input, 3ull, 2ull, 2ull,
                         linear_weight, 4ull, linear_bias, 2ull, 2ull,
                         linear_output, 4ull, &err) == YVEX_ERR_BOUNDS,
                     "source-layout linear projection refuses mismatched extents");
    YVEX_TEST_ASSERT(yvex_graph_layer_norm_f32(
                         layer_values, 1ull, 2ull, layer_weight, layer_bias,
                         1.0e-5, &err) == YVEX_OK &&
                         fabsf(layer_values[0] + 0.999995f) < 1.0e-5f &&
                         fabsf(layer_values[1] - 0.999995f) < 1.0e-5f,
                     "LayerNorm matches the independent two-value result");
    YVEX_TEST_ASSERT(yvex_graph_silu_gate_f32(
                         fused, 1ull, 1ull, gated, &err) == YVEX_OK &&
                         gated[0] == 0.0f,
                     "gated SiLU applies gate-first source semantics");
    YVEX_TEST_ASSERT(yvex_graph_full_attention_f32(
                         qkv, 2ull, 1ull, 2ull, attention, scratch, 2ull,
                         &err) == YVEX_OK &&
                         fabsf(attention[0] - expected_first) < 1.0e-6f &&
                         fabsf(attention[1] - (expected_first + 1.0f)) < 1.0e-6f,
                     "full attention matches an independent noncausal softmax result");
    YVEX_TEST_ASSERT(yvex_graph_full_attention_f32(
                         qkv, 2ull, 1ull, 2ull, attention, scratch, 1ull,
                         &err) == YVEX_ERR_INVALID_ARG,
                     "full attention refuses insufficient scratch");
    return 0;
}

static int test_t2va_plan(void)
{
    const yvex_transformer_joint_recipe *recipe =
        yvex_graph_register_minimax_h3()->omni_recipe();
    yvex_minimax_h3_t2va_plan first, repeated, source_scale;
    float sample[2] = {0.5f, -1.0f}, velocity[2] = {2.0f, 4.0f};
    float stepped[2] = {13.0f, 13.0f};
    yvex_error err;

    YVEX_TEST_ASSERT(recipe &&
                         recipe->qkv_layout ==
                             YVEX_TRANSFORMER_QKV_LAYOUT_PER_HEAD_THREE,
                     "Omni recipe preserves the released per-head Q/K/V row layout");
    YVEX_TEST_ASSERT(recipe &&
                         recipe->swiglu_layout ==
                             YVEX_TRANSFORMER_SWIGLU_LAYOUT_GATE_THEN_UP,
                     "Omni recipe preserves the released gate-before-up SwiGLU row layout");
    YVEX_TEST_ASSERT(
        recipe && recipe->schema_version == YVEX_TRANSFORMER_JOINT_SCHEMA_V2 &&
            recipe->video_output_numeric.tile_rows == 32u &&
            recipe->video_output_numeric.split_k == 10u &&
            recipe->video_output_numeric.reduction ==
                YVEX_BACKEND_LINEAR_REDUCTION_INPLACE &&
            recipe->audio_output_numeric.tile_rows == 128u &&
            recipe->audio_output_numeric.split_k == 3u &&
            recipe->audio_output_numeric.reduction ==
                YVEX_BACKEND_LINEAR_REDUCTION_COMPUTE_TYPE,
        "Omni recipe seals the independently qualified final projection reductions");

    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->t2va_plan_build(
                         &first, 16ull, 1344ull, 768ull, 124ull, 19u, &err) == YVEX_OK &&
                         first.complete && first.video_latent_frames == 37ull &&
                         first.video_latent_height == 48ull &&
                         first.video_latent_width == 84ull &&
                         first.audio_latent_steps == 207ull &&
                         first.audio_rows == 414ull && first.video_rows == 37296ull &&
                         first.packed_rows == 37726ull && first.sigma_grid_points == 20u &&
                         first.model_evaluations == 19u,
                     "t2va plan reconstructs the exact source geometry and packed layout");
    YVEX_TEST_ASSERT(fabsf(first.video_sigmas[0] - 1.0f) < 1.0e-7f &&
                         fabsf(first.audio_sigmas[0] - 1.0f) < 1.0e-7f &&
                         fabsf(first.video_sigmas[18] - 0.4f) < 1.0e-7f &&
                         fabsf(first.audio_sigmas[18] - 0.142857149f) < 1.0e-7f &&
                         first.video_sigmas[19] == 0.0f && first.audio_sigmas[19] == 0.0f,
                     "t2va plan includes terminal zero in the paired shifted sigma grids");
    YVEX_TEST_ASSERT(
        yvex_graph_register_minimax_h3()->t2va_plan_build(
            &source_scale, 28ull, 768ull, 768ull, 124ull, 49u, &err) == YVEX_OK &&
            float_bits(source_scale.video_sigmas[26]) == UINT32_C(0x3f69f5d3) &&
            float_bits(source_scale.video_sigmas[32]) == UINT32_C(0x3f5d49c4) &&
            float_bits(source_scale.video_sigmas[42]) == UINT32_C(0x3f2aaaaa) &&
            float_bits(source_scale.video_sigmas[44]) == UINT32_C(0x3f13b13a) &&
            float_bits(source_scale.audio_sigmas[16]) == UINT32_C(0x3f5c61f2) &&
            float_bits(source_scale.audio_sigmas[22]) == UINT32_C(0x3f495207) &&
            float_bits(source_scale.audio_sigmas[26]) == UINT32_C(0x3f39efd4) &&
            float_bits(source_scale.audio_sigmas[32]) == UINT32_C(0x3f1d4d1c) &&
            float_bits(source_scale.audio_sigmas[35]) == UINT32_C(0x3f0ba2e8) &&
            float_bits(source_scale.audio_sigmas[39]) == UINT32_C(0x3ede9bd2) &&
            float_bits(source_scale.audio_sigmas[42]) == UINT32_C(0x3eaaaaaa) &&
            float_bits(source_scale.audio_sigmas[44]) == UINT32_C(0x3e822b63),
        "source-scale sigma grids reproduce the released torch.linspace F32 bit patterns");
    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->scheduler_step(
                         stepped, sample, velocity, 2ull, 0.75f, 0.25f, 0.1f,
                         &err) == YVEX_OK &&
                         fabsf(stepped[0] - 0.8f) < 1.0e-7f &&
                         fabsf(stepped[1] + 0.4f) < 1.0e-7f,
                     "t2va scheduler applies the exact data-ward rectified-flow update");
    velocity[1] = NAN;
    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->scheduler_step(
                         stepped, sample, velocity, 2ull, 0.75f, 0.25f, 0.1f,
                         &err) == YVEX_ERR_FORMAT && stepped[0] == 0.8f &&
                         stepped[1] == -0.4f,
                     "t2va scheduler validates every value before publishing output");
    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->t2va_plan_build(
                         &repeated, 16ull, 1344ull, 768ull, 124ull, 19u, &err) == YVEX_OK &&
                         strcmp(first.identity, repeated.identity) == 0,
                     "t2va plan identity is deterministic");
    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->t2va_plan_build(
                         &repeated, 16ull, 1344ull, 768ull, 123ull, 19u, &err) ==
                         YVEX_ERR_INVALID_ARG &&
                         yvex_graph_register_minimax_h3()->t2va_plan_build(
                             &repeated, 16ull, 1343ull, 768ull, 124ull, 19u, &err) ==
                         YVEX_ERR_INVALID_ARG,
                     "t2va plan refuses invalid temporal and spatial grids");
    return 0;
}

static int test_component_admission_routing(void)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_minimax_h3_architecture architecture;
    yvex_backend_text_encoder_geometry geometry, invalid_geometry;
    yvex_minimax_h3_failure family_failure;
    yvex_minimax_h3_conditioning_result conditioning;
    yvex_backend_text_execution_result backend_result;
    unsigned int token = 1u;
    float output[5120];
    int rc;
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->component_admit(
                         "unknown", NULL, NULL, NULL, &admission, &failure, &err) ==
                         YVEX_ERR_INVALID_ARG &&
                         failure.code == YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT &&
                         strcmp(failure.field, "component") == 0,
                     "component admission refuses an unknown family-owned component");
    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->component_admit(
                         "audio_vae", NULL, NULL, NULL, &admission, &failure, &err) ==
                         YVEX_ERR_INVALID_ARG &&
                         failure.code == YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT,
                     "component admission refuses absent generic structural views");
    YVEX_TEST_ASSERT(yvex_model_register_minimax_h3()->architecture_canonical(
                         &architecture, &family_failure, &err) == YVEX_OK,
                     "component execution receives canonical family geometry");
    geometry = (yvex_backend_text_encoder_geometry){
        .schema_version = YVEX_BACKEND_TEXT_ENCODER_SCHEMA_V1,
        .semantic_identity = YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY,
        .embedding_identity_domain = "yvex.minimax-h3.text-conditioning.cuda.v1",
        .encoder_identity_domain = "yvex.minimax-h3.qwen-text-stack.cuda.v1",
        .layer_capacity = architecture.encoder.text_layers,
        .hidden_width = architecture.encoder.text_width,
        .ffn_width = architecture.encoder.text_ffn_width,
        .query_heads = architecture.encoder.text_query_heads,
        .kv_heads = architecture.encoder.text_kv_heads,
        .head_dimension = architecture.encoder.text_head_dimension,
        .vocabulary_size = architecture.encoder.vocabulary_size,
        .rope_theta = architecture.encoder.rope_theta,
        .normalization_epsilon = 1.0e-6f};
    memset(output, 0x5a, sizeof(output));
    rc = yvex_backend_text_embedding_execute(
        NULL, &geometry, NULL, 0ull, 0u, 0ull, 0ull, 0ull, NULL, 0ull,
        &token, 1ull, output, 5120ull, &backend_result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "CUDA conditioning reports invalid absent materialization");
    YVEX_TEST_ASSERT(!backend_result.complete && ((unsigned char *)output)[0] == 0x5a,
                     "CUDA conditioning does not publish a refused execution");
    YVEX_TEST_ASSERT(strcmp(yvex_error_where(&err),
                            "cuda.text-embedding.validate") == 0,
                     "CUDA conditioning refuses absent materialization without publication");
    invalid_geometry = geometry;
    ++invalid_geometry.query_heads;
    YVEX_TEST_ASSERT(yvex_backend_text_embedding_execute(
                         NULL, &invalid_geometry, NULL, 0ull, 0u, 0ull, 0ull, 0ull,
                         NULL, 0ull, &token, 1ull, output, 5120ull, &backend_result,
                         &err) == YVEX_ERR_INVALID_ARG &&
                         strcmp(yvex_error_where(&err),
                                "cuda.text-geometry") == 0,
                     "CUDA conditioning refuses inconsistent family geometry");
    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->text_encoder_artifact_cuda(
                         NULL, NULL, NULL, &token, 1ull, 0ull,
                         output, 5120ull, 1ull, 1ull,
                         &conditioning, &err) == YVEX_ERR_INVALID_ARG &&
                         !conditioning.complete && ((unsigned char *)output)[0] == 0x5a,
                     "artifact conditioning refuses absent exact component views");
    YVEX_TEST_ASSERT(yvex_graph_register_minimax_h3()->text_encoder_artifact_cuda(
                         NULL, NULL, NULL, &token, 1ull, 50ull,
                         output, 5120ull, 1ull, 1ull,
                         &conditioning, &err) == YVEX_ERR_INVALID_ARG &&
                         !conditioning.complete && ((unsigned char *)output)[0] == 0x5a,
                     "artifact text layer refuses absent exact component views");
    return 0;
}

static int test_component_execution_plans(void)
{
    yvex_component_plan_request request = {
        .target_id = YVEX_MINIMAX_H3_TARGET_ID,
        .component_id = "audio-vae",
        .backend = YVEX_BACKEND_KIND_CPU,
        .batch = 2ull,
        .geometry_rank = 1u,
        .geometry = {3ull},
        .maximum_host_bytes = 16ull * 1024ull * 1024ull,
    };
    yvex_component_plan plan;
    yvex_component_failure failure;
    yvex_error err;
    char oversized_target[129];

    YVEX_TEST_ASSERT(
        yvex_runtime_component_api_get()->plan_build(
            &request, &plan, &failure, &err) == YVEX_OK &&
            plan.complete && plan.input_values == 192ull &&
            plan.output_values == 4800ull && plan.output_rank == 3u &&
            plan.output_dims[0] == 2ull && plan.output_dims[1] == 1ull &&
            plan.output_dims[2] == 2400ull &&
            yvex_sha256_hex_valid(plan.identity),
        "component compiler derives Audio VAE extents from family semantics");
    YVEX_TEST_ASSERT(
        yvex_runtime_component_api_get()->plan_validate(
            &plan, &failure, &err) == YVEX_OK,
        "component compiler validates the sealed canonical plan");
    ++plan.output_values;
    YVEX_TEST_ASSERT(
        yvex_runtime_component_api_get()->plan_validate(
            &plan, &failure, &err) == YVEX_ERR_FORMAT &&
            failure.code == YVEX_COMPONENT_FAILURE_LIFECYCLE,
        "component compiler refuses a mutated sealed plan");

    request.component_id = "video-vae";
    request.batch = 1ull;
    request.geometry_rank = 3u;
    request.geometry[0] = 2ull;
    request.geometry[1] = 3ull;
    request.geometry[2] = 4ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_component_api_get()->plan_build(
            &request, &plan, &failure, &err) == YVEX_OK &&
            plan.input_values == 576ull && plan.output_values == 73728ull &&
            plan.output_rank == 5u && plan.output_dims[0] == 1ull &&
            plan.output_dims[1] == 3ull && plan.output_dims[2] == 8ull &&
            plan.output_dims[3] == 48ull && plan.output_dims[4] == 64ull,
        "component compiler derives Visual VAE extents from family semantics");
    request.backend = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(
        yvex_runtime_component_api_get()->plan_build(
            &request, &plan, &failure, &err) ==
                YVEX_ERR_UNSUPPORTED &&
            failure.code == YVEX_COMPONENT_FAILURE_UNSUPPORTED,
        "component compiler refuses an unadmitted backend explicitly");
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.target_id = "unknown";
    YVEX_TEST_ASSERT(
        yvex_runtime_component_api_get()->plan_build(
            &request, &plan, &failure, &err) ==
                YVEX_ERR_UNSUPPORTED &&
            failure.code == YVEX_COMPONENT_FAILURE_UNSUPPORTED,
        "component compiler refuses an unknown target without CLI family policy");
    memset(oversized_target, 'x', sizeof(oversized_target) - 1u);
    oversized_target[sizeof(oversized_target) - 1u] = '\0';
    request.target_id = oversized_target;
    YVEX_TEST_ASSERT(
        yvex_runtime_component_api_get()->plan_build(
            &request, &plan, &failure, &err) == YVEX_ERR_INVALID_ARG &&
            failure.code == YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT,
        "component compiler refuses identifiers that cannot be sealed losslessly");
    return 0;
}

int yvex_test_minimax_h3(void)
{
    if (test_family_catalog() != 0) return 1;
    if (test_components() != 0) return 1;
    if (test_architecture() != 0) return 1;
    if (test_roles() != 0) return 1;
    if (test_component_ir() != 0) return 1;
    if (test_operator_truth() != 0) return 1;
    if (test_semantic_composite() != 0) return 1;
    if (test_canonical_operator_graph() != 0) return 1;
    if (test_audio_numeric_primitives() != 0) return 1;
    if (test_video_numeric_primitives() != 0) return 1;
    if (test_t2va_plan() != 0) return 1;
    if (test_component_admission_routing() != 0) return 1;
    if (test_component_execution_plans() != 0) return 1;
    return 0;
}
