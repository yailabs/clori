/* Admit the pinned Qwen3.8 source as one source-faithful Qwen3.5 text specialization. */
#include "src/graph/private.h"

#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/compiler_source.h>
#include <yvex/internal/core.h>
#include <yvex/internal/deployment.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/families/qwen3_5.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/source_catalog.h>
#include <yvex/internal/tokenizer.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#define QWEN_TEXT_TENSORS 851ull
#define QWEN_PINNED_NAMES 131ull
#define QWEN_EXTENSION_NAMES 432ull
#define QWEN_MAPPING_IDENTITY 9266396127046126464ull
#define QWEN_SOURCE_FAITHFUL_PRESET "qwen3.8-source-faithful"

typedef struct {
    yvex_compilation_source_session *source;
    yvex_semantic_model_ir *semantic;
} qwen_source_owner;

static void qwen_source_release(void *pointer);
static int qwen_tokenizer_policy(yvex_tokenizer_family_policy *out,
                                 yvex_error *err);

static int qwen_role_project(yvex_qwen3_5_tensor_role source,
                             yvex_tensor_role *role,
                             yvex_tensor_collection *collection)
{
    static const yvex_tensor_role roles[YVEX_QWEN3_5_ROLE_COUNT] = {
        [YVEX_QWEN3_5_ROLE_TOKEN_EMBEDDING] = YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
        [YVEX_QWEN3_5_ROLE_OUTPUT_NORM] = YVEX_TENSOR_ROLE_OUTPUT_NORM,
        [YVEX_QWEN3_5_ROLE_OUTPUT_HEAD] = YVEX_TENSOR_ROLE_OUTPUT_HEAD,
        [YVEX_QWEN3_5_ROLE_INPUT_NORM] = YVEX_TENSOR_ROLE_ATTENTION_NORM,
        [YVEX_QWEN3_5_ROLE_FFN_GATE] = YVEX_TENSOR_ROLE_FFN_GATE,
        [YVEX_QWEN3_5_ROLE_FFN_UP] = YVEX_TENSOR_ROLE_FFN_UP,
        [YVEX_QWEN3_5_ROLE_FFN_DOWN] = YVEX_TENSOR_ROLE_FFN_DOWN,
        [YVEX_QWEN3_5_ROLE_POST_ATTENTION_NORM] = YVEX_TENSOR_ROLE_FFN_NORM,
        [YVEX_QWEN3_5_ROLE_ATTENTION_Q] = YVEX_TENSOR_ROLE_ATTENTION_Q,
        [YVEX_QWEN3_5_ROLE_ATTENTION_K] = YVEX_TENSOR_ROLE_ATTENTION_K,
        [YVEX_QWEN3_5_ROLE_ATTENTION_V] = YVEX_TENSOR_ROLE_ATTENTION_V,
        [YVEX_QWEN3_5_ROLE_ATTENTION_OUT] = YVEX_TENSOR_ROLE_ATTENTION_OUT,
        [YVEX_QWEN3_5_ROLE_ATTENTION_Q_NORM] = YVEX_TENSOR_ROLE_ATTENTION_Q_NORM,
        [YVEX_QWEN3_5_ROLE_ATTENTION_K_NORM] = YVEX_TENSOR_ROLE_ATTENTION_K_NORM,
        [YVEX_QWEN3_5_ROLE_DELTA_DECAY_LOG] = YVEX_TENSOR_ROLE_SEQUENCE_MIXER_DECAY_LOG,
        [YVEX_QWEN3_5_ROLE_DELTA_CONVOLUTION] = YVEX_TENSOR_ROLE_SEQUENCE_MIXER_CONVOLUTION,
        [YVEX_QWEN3_5_ROLE_DELTA_TIME_BIAS] = YVEX_TENSOR_ROLE_SEQUENCE_MIXER_TIME_BIAS,
        [YVEX_QWEN3_5_ROLE_DELTA_DECAY_PROJECTION] =
            YVEX_TENSOR_ROLE_SEQUENCE_MIXER_DECAY_PROJECTION,
        [YVEX_QWEN3_5_ROLE_DELTA_BETA_PROJECTION] =
            YVEX_TENSOR_ROLE_SEQUENCE_MIXER_BETA_PROJECTION,
        [YVEX_QWEN3_5_ROLE_DELTA_QKV_PROJECTION] =
            YVEX_TENSOR_ROLE_SEQUENCE_MIXER_QKV_PROJECTION,
        [YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_GATE] =
            YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT_GATE,
        [YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_NORM] =
            YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT_NORM,
        [YVEX_QWEN3_5_ROLE_DELTA_OUTPUT] = YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT};

    if (!role || !collection || source <= YVEX_QWEN3_5_ROLE_UNKNOWN ||
        source >= YVEX_QWEN3_5_ROLE_COUNT || roles[source] == YVEX_TENSOR_ROLE_UNKNOWN)
        return 0;
    *role = roles[source];
    switch (source) {
    case YVEX_QWEN3_5_ROLE_TOKEN_EMBEDDING:
    case YVEX_QWEN3_5_ROLE_OUTPUT_HEAD:
        *collection = YVEX_TENSOR_COLLECTION_GLOBAL;
        break;
    case YVEX_QWEN3_5_ROLE_OUTPUT_NORM:
    case YVEX_QWEN3_5_ROLE_INPUT_NORM:
    case YVEX_QWEN3_5_ROLE_POST_ATTENTION_NORM:
        *collection = YVEX_TENSOR_COLLECTION_NORM;
        break;
    case YVEX_QWEN3_5_ROLE_FFN_GATE:
    case YVEX_QWEN3_5_ROLE_FFN_UP:
    case YVEX_QWEN3_5_ROLE_FFN_DOWN:
        *collection = YVEX_TENSOR_COLLECTION_DENSE_FFN;
        break;
    case YVEX_QWEN3_5_ROLE_ATTENTION_Q:
    case YVEX_QWEN3_5_ROLE_ATTENTION_K:
    case YVEX_QWEN3_5_ROLE_ATTENTION_V:
    case YVEX_QWEN3_5_ROLE_ATTENTION_OUT:
    case YVEX_QWEN3_5_ROLE_ATTENTION_Q_NORM:
    case YVEX_QWEN3_5_ROLE_ATTENTION_K_NORM:
        *collection = YVEX_TENSOR_COLLECTION_ATTENTION;
        break;
    default:
        *collection = YVEX_TENSOR_COLLECTION_SEQUENCE_MIXER;
        break;
    }
    return 1;
}

static int qwen_transform_add(yvex_transform_recipe_sink *sink,
                              const yvex_native_weight_info *tensor,
                              const yvex_qwen3_5_tensor_binding *binding,
                              unsigned long long ordinal,
                              yvex_transform_failure *failure, yvex_error *err)
{
    yvex_transform_direct_recipe recipe = {0};
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    unsigned int dimension;

    if (!tensor || !binding || tensor->dtype != YVEX_NATIVE_DTYPE_BF16 ||
        tensor->rank == 0u || tensor->rank > YVEX_TRANSFORM_IR_MAX_RANK ||
        !qwen_role_project(binding->role, &role, &collection)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "qwen3_5.transform",
                       "text tensor cannot be projected into the source-faithful transform");
        return YVEX_ERR_FORMAT;
    }
    recipe.source_name = tensor->name;
    recipe.role = role;
    recipe.collection = collection;
    recipe.scope = binding->layer_index == YVEX_QWEN3_5_GLOBAL_TENSOR_LAYER
                       ? YVEX_TENSOR_SCOPE_GLOBAL
                       : YVEX_TENSOR_SCOPE_MAIN_LAYER;
    recipe.layer = binding->layer_index;
    recipe.auxiliary = YVEX_TRANSFORM_IR_NO_ID;
    recipe.expert = YVEX_TRANSFORM_IR_NO_ID;
    recipe.requirement_index = ordinal;
    recipe.source_dtype = tensor->dtype;
    recipe.shape.rank = tensor->rank;
    for (dimension = 0u; dimension < tensor->rank; ++dimension)
        recipe.shape.dims[dimension] = tensor->dims[dimension];
    return yvex_transform_recipe_add_direct(sink, &recipe, failure, err);
}

typedef struct {
    const yvex_qwen3_5_architecture *architecture;
    yvex_source_tensor_snapshot *snapshot;
    unsigned long long tensor_count;
} qwen_transform_projection;

static int qwen_transform_project(void *context,
                                  yvex_transform_recipe_sink *sink,
                                  yvex_transform_failure *failure,
                                  yvex_error *err)
{
    qwen_transform_projection *projection = context;
    unsigned long long index, ordinal = 0ull;
    int rc = YVEX_OK;

    if (!projection || !sink) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "qwen3_5.transform",
                       "transform projection and compiler sink are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; rc == YVEX_OK && index < projection->tensor_count; ++index) {
        const yvex_native_weight_info *tensor =
            yvex_source_tensor_snapshot_at(projection->snapshot, index);
        yvex_qwen3_5_tensor_binding binding = {0};
        yvex_qwen3_5_failure family_failure = {0};

        rc = yvex_model_register_qwen3_5()->tensor_classify(
            projection->architecture, tensor, &binding, &family_failure, err);
        if (rc == YVEX_OK &&
            binding.classification == YVEX_QWEN3_5_TENSOR_TEXT_EXECUTION_REQUIRED)
            rc = qwen_transform_add(
                sink, tensor, &binding, ordinal++, failure, err);
    }
    if (rc == YVEX_OK && ordinal != QWEN_TEXT_TENSORS) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "qwen3_5.transform",
                       "text specialization tensor population changed after audit");
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
}

static int qwen_transform_build(yvex_transform_ir **out,
                                const yvex_source_verification *verification,
                                const yvex_qwen3_5_architecture *architecture,
                                yvex_source_tensor_snapshot *snapshot,
                                const yvex_qwen3_5_tensor_inventory *inventory,
                                yvex_transform_failure *failure, yvex_error *err)
{
    yvex_source_tensor_snapshot_facts facts = {0};
    yvex_transform_builder_options options = {0};
    yvex_transform_header header = {0};
    qwen_transform_projection projection = {0};

    if (out) *out = NULL;
    if (!out || !verification || !architecture || !snapshot || !inventory ||
        !inventory->complete || !verification->verified || verification->blocker_count ||
        !verification->manifest_payload_trusted ||
        !yvex_sha256_hex_valid(verification->manifest_payload_identity) ||
        yvex_source_tensor_snapshot_facts_get(snapshot, &facts, err) != YVEX_OK ||
        facts.identity != verification->source_snapshot_identity ||
        inventory->class_counts[YVEX_QWEN3_5_TENSOR_TEXT_EXECUTION_REQUIRED] !=
            QWEN_TEXT_TENSORS) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "qwen3_5.transform",
                       "verified pinned source and complete tensor accounting are required");
        return YVEX_ERR_FORMAT;
    }
    header.schema_version = YVEX_TRANSFORM_IR_SPECIALIZATION_SCHEMA_VERSION;
    header.logical_model_identity = architecture->architecture_identity;
    header.source_snapshot_identity = facts.identity;
    header.required_payload_identity = verification->manifest_payload_identity;
    header.payload_trust_class = verification->manifest_payload_trust_class;
    header.architecture_identity = architecture->architecture_identity;
    header.role_map_identity = inventory->role_map_identity;
    header.source_population_count = facts.tensor_count;
    header.expected_source_count = QWEN_TEXT_TENSORS;
    header.expected_terminal_count = QWEN_TEXT_TENSORS;
    header.header_scan_count = facts.header_scan_count;
    yvex_transform_budget_default(&options.budget);
    options.source_snapshot = snapshot;
    projection.architecture = architecture;
    projection.snapshot = snapshot;
    projection.tensor_count = facts.tensor_count;
    return yvex_transform_recipe_compile(
        out, &header, qwen_transform_project, &projection, &options, failure, err);
}

static void qwen_metadata_string(yvex_artifact_lowering_metadata *entry,
                                 const char *key, const char *value)
{
    memset(entry, 0, sizeof(*entry));
    yvex_core_text_copy(entry->key, sizeof(entry->key), key);
    entry->type = YVEX_ARTIFACT_LOWERING_METADATA_STRING;
    yvex_core_text_copy(entry->string_value, sizeof(entry->string_value), value);
}

static void qwen_metadata_u64(yvex_artifact_lowering_metadata *entry,
                              const char *key, unsigned long long value)
{
    memset(entry, 0, sizeof(*entry));
    yvex_core_text_copy(entry->key, sizeof(entry->key), key);
    entry->type = YVEX_ARTIFACT_LOWERING_METADATA_U64;
    entry->u64_value = value;
}

static int qwen_lowering_build(yvex_artifact_lowering_map **out,
                               const yvex_transform_ir *transform,
                               const yvex_qwen3_5_architecture *architecture,
                               yvex_artifact_lowering_failure *failure,
                               yvex_error *err)
{
    yvex_artifact_lowering_metadata metadata[18];
    yvex_artifact_lowering_policy policy = {0};
    unsigned long long count = 0ull;

    qwen_metadata_string(&metadata[count++], "general.architecture", "qwen3_5");
    qwen_metadata_string(&metadata[count++], "general.name", "Qwen3.8-27B Text");
    qwen_metadata_string(&metadata[count++], "general.source.repository",
                         YVEX_SOURCE_QWEN3_8_27B_REPOSITORY);
    qwen_metadata_string(&metadata[count++], "general.source.revision",
                         YVEX_SOURCE_QWEN3_8_27B_REVISION);
    qwen_metadata_string(&metadata[count++], "yvex.source.capability", "multimodal");
    qwen_metadata_string(&metadata[count++], "yvex.specialization", "text");
    qwen_metadata_string(&metadata[count++], "yvex.vision.execution", "deferred");
    qwen_metadata_string(&metadata[count++], "yvex.mtp.acceleration", "deferred");
    qwen_metadata_u64(&metadata[count++], "qwen3_5.block_count",
                      architecture->text.layer_count);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.embedding_length",
                      architecture->text.hidden_size);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.context_length",
                      architecture->text.maximum_positions);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.vocabulary_size",
                      architecture->text.vocabulary_size);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.feed_forward_length",
                      architecture->text.intermediate_size);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.attention.head_count",
                      architecture->text.attention_heads);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.attention.head_count_kv",
                      architecture->text.kv_heads);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.attention.key_length",
                      architecture->text.attention_head_dimension);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.full_attention_interval",
                      architecture->text.full_attention_interval);
    qwen_metadata_u64(&metadata[count++], "qwen3_5.linear_attention.conv_kernel",
                      architecture->text.linear_convolution_kernel);
    policy.schema_version = YVEX_ARTIFACT_LOWERING_POLICY_SCHEMA_V1;
    policy.source_contribution_count = QWEN_TEXT_TENSORS;
    policy.descriptor_count = QWEN_TEXT_TENSORS;
    policy.trunk_descriptor_count = QWEN_TEXT_TENSORS;
    policy.pinned_standard_count = QWEN_PINNED_NAMES;
    policy.extension_count = QWEN_EXTENSION_NAMES;
    policy.trunk_collection_counts[YVEX_TENSOR_COLLECTION_GLOBAL] = 2ull;
    policy.trunk_collection_counts[YVEX_TENSOR_COLLECTION_ATTENTION] = 96ull;
    policy.trunk_collection_counts[YVEX_TENSOR_COLLECTION_NORM] = 129ull;
    policy.trunk_collection_counts[YVEX_TENSOR_COLLECTION_SEQUENCE_MIXER] = 432ull;
    policy.trunk_collection_counts[YVEX_TENSOR_COLLECTION_DENSE_FFN] = 192ull;
    policy.metadata = metadata;
    policy.metadata_count = count;
    return yvex_artifact_lowering_operations.build(
        out, transform, &policy, failure, err);
}

static int qwen_source_lower(yvex_transform_ir **transform,
                             yvex_artifact_lowering_map **lowering,
                             const yvex_source_verification *verification,
                             yvex_source_tensor_snapshot *snapshot,
                             yvex_compilation_source_failure *failure,
                             yvex_error *err)
{
    const yvex_qwen3_5_api *family = yvex_model_register_qwen3_5();
    yvex_qwen3_5_failure family_failure = {0};
    yvex_qwen3_5_tensor_inventory inventory = {0};
    yvex_transform_failure transform_failure = {0};
    yvex_artifact_lowering_failure lowering_failure = {0};
    yvex_qwen3_5_model *model = NULL;
    const yvex_qwen3_5_architecture *architecture;
    int rc;

    if (transform) *transform = NULL;
    if (lowering) *lowering = NULL;
    rc = family->open(&model, verification, &family_failure, err);
    architecture = rc == YVEX_OK ? family->architecture(model) : NULL;
    if (rc == YVEX_OK)
        rc = family->tensor_snapshot_audit(
            architecture, snapshot, &inventory, &family_failure, err);
    if (rc != YVEX_OK && failure)
        failure->code = YVEX_COMPILATION_SOURCE_FAILURE_SEMANTIC_MODEL;
    if (rc == YVEX_OK)
        rc = qwen_transform_build(
            transform, verification, architecture, snapshot, &inventory,
            &transform_failure, err);
    if (rc != YVEX_OK && failure &&
        failure->code == YVEX_COMPILATION_SOURCE_FAILURE_NONE)
        failure->code = YVEX_COMPILATION_SOURCE_FAILURE_TRANSFORM_IR;
    if (rc == YVEX_OK)
        rc = qwen_lowering_build(
            lowering, *transform, architecture, &lowering_failure, err);
    if (rc != YVEX_OK && failure && *transform && !*lowering)
        failure->code = YVEX_COMPILATION_SOURCE_FAILURE_LOWERING;
    family->close(&model);
    return rc;
}

static const void *qwen_source_identity(void)
{
    return yvex_source_target_identity_find(YVEX_QWEN3_8_27B_TARGET_ID);
}

static const yvex_compilation_source_projection qwen_source_projection = {
    .schema_version = YVEX_COMPILATION_SOURCE_PROJECTION_SCHEMA_V1,
    .expected_mapping_identity = QWEN_MAPPING_IDENTITY,
    .required_contribution_mask =
        YVEX_COMPILATION_SOURCE_REQUIRE_DIRECT |
        YVEX_COMPILATION_SOURCE_REQUIRE_GLOBAL |
        YVEX_COMPILATION_SOURCE_REQUIRE_NORM |
        YVEX_COMPILATION_SOURCE_REQUIRE_OUTPUT_HEAD,
    .source_identity = qwen_source_identity,
    .lower = qwen_source_lower,
    .lowering = &yvex_artifact_lowering_operations};

static int qwen_semantic_identity(
    const char *domain, const yvex_qwen3_5_architecture *architecture,
    char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long layer;

    if (!domain || !architecture || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_text(&hash, architecture->architecture_identity) ||
        !yvex_sha256_update_u64(&hash, architecture->text.layer_count) ||
        !yvex_sha256_update_u64(&hash, architecture->text.maximum_positions) ||
        !yvex_sha256_update_u64(&hash, architecture->text.hidden_size) ||
        !yvex_sha256_update_u64(&hash, architecture->text.intermediate_size))
        return 0;
    for (layer = 0ull; layer < architecture->text.layer_count; ++layer)
        if (!yvex_sha256_update_u64(
                &hash, architecture->text.layers[layer])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int qwen_numeric_contract_build(
    const yvex_qwen3_5_architecture *architecture,
    yvex_semantic_numeric_contract *contract, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    if (!architecture || !contract) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "qwen3_5.numeric-contract",
                       "pinned architecture and numeric contract output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.qwen3_5.text.numeric.v1") ||
        !yvex_sha256_update_text(&hash, architecture->architecture_identity) ||
        !yvex_sha256_update_u64(&hash, architecture->text.layer_count) ||
        !yvex_sha256_update_u64(&hash, architecture->text.hidden_size) ||
        !yvex_sha256_update_u64(&hash, architecture->text.intermediate_size) ||
        !yvex_sha256_update_u64(&hash, architecture->text.attention_head_dimension) ||
        !yvex_sha256_update_u64(&hash, architecture->text.rotary_dimension) ||
        !yvex_sha256_update_u64(&hash, architecture->text.linear_key_head_dimension) ||
        !yvex_sha256_update_u64(&hash, architecture->text.linear_value_head_dimension) ||
        !yvex_sha256_update_u64(&hash, architecture->text.linear_convolution_kernel) ||
        !yvex_sha256_update_u64(&hash, YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1) ||
        !yvex_sha256_update_u64(&hash, YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE) ||
        !yvex_sha256_final(&hash, digest)) {
        yvex_error_set(err, YVEX_ERR_STATE, "qwen3_5.numeric-contract",
                       "Qwen numeric contract identity could not be derived");
        return YVEX_ERR_STATE;
    }
    memset(contract, 0, sizeof(*contract));
    contract->schema_version = YVEX_SEMANTIC_NUMERIC_CONTRACT_SCHEMA_V1;
    contract->numeric_schema_version = 1u;
    contract->compute_policy_count = 2ull;
    contract->activation_policy_count = 2ull;
    yvex_core_text_copy(contract->algorithm_revision,
                        sizeof(contract->algorithm_revision),
                        "qwen3_5-text-bf16-f32-recurrence-v1");
    yvex_sha256_hex(digest, contract->identity);
    return YVEX_OK;
}

static yvex_gated_delta_requirement qwen_delta_requirement(
    const yvex_qwen3_5_architecture *architecture)
{
    yvex_gated_delta_requirement requirement = {
        .schema_version = YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V1,
        .query_heads = architecture->text.linear_key_heads,
        .key_heads = architecture->text.linear_key_heads,
        .value_heads = architecture->text.linear_value_heads,
        .key_head_dimension = architecture->text.linear_key_head_dimension,
        .value_head_dimension = architecture->text.linear_value_head_dimension,
        .convolution_kernel = architecture->text.linear_convolution_kernel,
        .projected_dtype = YVEX_DTYPE_F32,
        .convolution_state_dtype = YVEX_DTYPE_F32,
        .recurrent_state_dtype = YVEX_DTYPE_F32,
        .accumulation_dtype = YVEX_DTYPE_F32,
        .output_dtype = YVEX_DTYPE_F32,
        .numeric_contract = YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE,
        .qk_normalization_epsilon = 1.0e-6,
        .output_normalization_epsilon = architecture->text.rms_norm_epsilon,
        .query_scale = 1.0 / sqrt(
            (double)architecture->text.linear_key_head_dimension),
        .deterministic = 1};

    return requirement;
}

static int qwen_decoder_layer(
    const void *context, unsigned long long index,
    yvex_semantic_decoder_layer *out)
{
    const yvex_qwen3_5_architecture *architecture = context;
    yvex_qwen3_5_layer_kind kind;

    if (!architecture || !out || index >= architecture->text.layer_count)
        return 0;
    kind = architecture->text.layers[index];
    memset(out, 0, sizeof(*out));
    out->ordinal = index;
    out->layer_index = index;
    out->tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER;
    out->mixer = kind == YVEX_QWEN3_5_LAYER_FULL_ATTENTION
                     ? YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION
                     : YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA;
    out->feed_forward = YVEX_SEMANTIC_DECODER_FFN_DENSE_SILU_GATED;
    out->hidden_width = architecture->text.hidden_size;
    out->intermediate_width = architecture->text.intermediate_size;
    out->normalization_epsilon = architecture->text.rms_norm_epsilon;
    out->mixer_output_gate = 1;
    if (kind == YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION)
        out->gated_delta = qwen_delta_requirement(architecture);
    return kind == YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION ||
           kind == YVEX_QWEN3_5_LAYER_FULL_ATTENTION;
}

static int qwen_attention_layer(
    const void *context, unsigned long long ordinal,
    yvex_semantic_attention_layer *out)
{
    const yvex_qwen3_5_architecture *architecture = context;
    unsigned long long index, found = 0ull;

    if (!architecture || !out ||
        ordinal >= architecture->text.full_attention_layers) return 0;
    for (index = 0ull; index < architecture->text.layer_count; ++index) {
        if (architecture->text.layers[index] !=
            YVEX_QWEN3_5_LAYER_FULL_ATTENTION) continue;
        if (found++ != ordinal) continue;
        memset(out, 0, sizeof(*out));
        out->ordinal = ordinal;
        out->layer_index = index;
        out->predictor_index = YVEX_ATTENTION_NO_TENSOR_INDEX;
        out->tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER;
        /* A maximum-context window selects the exact bounded full-causal H28 path. */
        out->attention_class = YVEX_ATTENTION_CLASS_SWA;
        out->compute_contract = YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1;
        out->sliding_window = architecture->text.maximum_positions;
        out->query_heads = architecture->text.attention_heads;
        out->kv_heads = architecture->text.kv_heads;
        out->head_dimension = architecture->text.attention_head_dimension;
        out->rope_head_dimension = architecture->text.rotary_dimension;
        out->hidden_dimension = architecture->text.hidden_size;
        out->rms_norm_epsilon = architecture->text.rms_norm_epsilon;
        out->position.rope_dimension = architecture->text.rotary_dimension;
        out->position.theta = architecture->text.rope_theta;
        out->position.scaling_factor = 1ull;
        out->position.original_context = architecture->text.maximum_positions;
        out->position.maximum_context = architecture->text.maximum_positions;
        out->position.partial_rope = 1;
        return 1;
    }
    return 0;
}

static int qwen_execution_descriptor(
    yvex_model_execution_descriptor *out,
    const yvex_source_verification *verification,
    const yvex_qwen3_5_architecture *architecture, yvex_error *err)
{
    char schedule[YVEX_SHA256_HEX_BYTES], state[YVEX_SHA256_HEX_BYTES];
    yvex_model_execution_descriptor_request request = {0};

    if (!out || !verification || !architecture ||
        !qwen_semantic_identity(
            "yvex.qwen3_5.attention-schedule.v1", architecture, schedule) ||
        !qwen_semantic_identity(
            "yvex.qwen3_5.persistent-state.v1", architecture, state)) {
        yvex_error_set(err, YVEX_ERR_STATE, "qwen3_5.execution-descriptor",
                       "Qwen execution identities could not be derived");
        return YVEX_ERR_STATE;
    }
    request = (yvex_model_execution_descriptor_request){
        .schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2,
        .logical_model_identity = architecture->architecture_identity,
        .source_model_identity = verification->manifest_payload_identity,
        .attention_schedule_identity = schedule,
        .persistent_state_identity = state,
        .maximum_context = architecture->text.maximum_positions,
        .original_context = architecture->text.maximum_positions,
        .rope_scaling = YVEX_MODEL_ROPE_SCALING_NONE,
        .rope_theta = architecture->text.rope_theta,
        .rope_scaling_factor = 1ull,
        .layer_count = architecture->text.layer_count,
        .hidden_width = architecture->text.hidden_size,
        .vocabulary_size = architecture->text.vocabulary_size,
        .attention_heads = architecture->text.attention_heads,
        .kv_heads = architecture->text.kv_heads,
        .head_width = architecture->text.attention_head_dimension,
        .swa_layers = architecture->text.full_attention_layers,
        .sequence_mixer_layers = architecture->text.linear_attention_layers,
        .sliding_window = architecture->text.maximum_positions,
        .normalization_epsilon = architecture->text.rms_norm_epsilon,
        .dense_ffn_width = architecture->text.intermediate_size,
        .output_input_width = architecture->text.hidden_size,
        .output_vocabulary_size = architecture->text.vocabulary_size,
        .persistent_state_class_mask =
            YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_SWA_RING) |
            YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_RECURRENT_SEQUENCE),
        .bos_token_id = architecture->text.bos_token_id,
        .eos_token_id = architecture->text.eos_token_id};
    return yvex_model_execution_descriptor_seal(&request, out, err);
}

static int qwen_semantic_model_build(yvex_semantic_model_ir **out,
                                     const yvex_source_verification *verification,
                                     yvex_error *err)
{
    const yvex_qwen3_5_api *family = yvex_model_register_qwen3_5();
    yvex_qwen3_5_failure failure = {0};
    yvex_qwen3_5_model *model = NULL;
    const yvex_qwen3_5_architecture *architecture;
    yvex_semantic_reference_request references[4];
    yvex_semantic_model_ir_request request = {0};
    yvex_model_execution_descriptor execution = {0};
    yvex_semantic_numeric_contract numeric = {0};
    int rc;

    if (out) *out = NULL;
    rc = family->open(&model, verification, &failure, err);
    architecture = rc == YVEX_OK ? family->architecture(model) : NULL;
    if (rc == YVEX_OK && (!architecture ||
        !yvex_sha256_hex_valid(verification->manifest_payload_identity))) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "qwen3_5.semantic-model",
                       "pinned architecture and source payload identity are required");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        rc = qwen_execution_descriptor(
            &execution, verification, architecture, err);
    if (rc == YVEX_OK)
        rc = qwen_numeric_contract_build(architecture, &numeric, err);
    if (rc == YVEX_OK) {
        references[0] = (yvex_semantic_reference_request){
            "source-revision", architecture->source_revision};
        references[1] = (yvex_semantic_reference_request){
            "source-architecture", YVEX_SOURCE_QWEN3_8_27B_CONFIG_ARCHITECTURE};
        references[2] = (yvex_semantic_reference_request){
            "transformers-requirement", architecture->transformers_version};
        references[3] = (yvex_semantic_reference_request){
            "execution-specialization", "text-first"};
        request.schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2;
        request.family_adapter_id = YVEX_QWEN3_5_ADAPTER_ID;
        request.family_adapter_version = YVEX_QWEN3_5_ADAPTER_VERSION;
        request.target_id = YVEX_QWEN3_8_27B_TARGET_ID;
        request.source_model_identity = verification->manifest_payload_identity;
        request.logical_model_identity = architecture->architecture_identity;
        request.semantic_payload_identity = execution.identity;
        request.execution_descriptor = &execution;
        request.numeric_contract = &numeric;
        request.attention_context = architecture;
        request.attention_layer = qwen_attention_layer;
        request.attention_layer_count = architecture->text.full_attention_layers;
        request.decoder_context = architecture;
        request.decoder_layer = qwen_decoder_layer;
        request.decoder_layer_count = architecture->text.layer_count;
        request.references = references;
        request.reference_count = sizeof(references) / sizeof(references[0]);
        rc = yvex_semantic_model_ir_seal(out, &request, err);
    }
    family->close(&model);
    return rc;
}

static int qwen_compilation_source_open(
    yvex_family_compilation_source *out,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    yvex_compilation_source_options options = {0};
    yvex_compilation_source_failure failure = {0};
    qwen_source_owner *owner;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (!out || !request || !request->source_path || !request->models_root) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "qwen3_5.compilation-source",
                       "exact source path and models root are required");
        return YVEX_ERR_INVALID_ARG;
    }
    owner = calloc(1u, sizeof(*owner));
    if (!owner) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "qwen3_5.compilation-source",
                       "source compiler ownership allocation failed");
        return YVEX_ERR_NOMEM;
    }
    options.source_path = request->source_path;
    options.models_root = request->models_root;
    options.manifest_path = request->source_manifest_path;
    yvex_source_payload_budget_default(&options.budget);
    if (request->source_stream_count > 64u) {
        qwen_source_release(owner);
        yvex_error_set(err, YVEX_ERR_BOUNDS, "qwen3_5.compilation-source",
                       "source stream count exceeds the bounded compiler worker limit");
        return YVEX_ERR_BOUNDS;
    }
    if (request->source_stream_count) {
        options.budget.maximum_streams = request->source_stream_count;
        if (options.budget.maximum_open_handles < request->source_stream_count)
            options.budget.maximum_open_handles = request->source_stream_count;
        options.budget.maximum_inflight_host_bytes =
            options.budget.chunk_bytes * request->source_stream_count;
    }
    options.chunk_bytes = options.budget.chunk_bytes;
    options.page_bytes = options.budget.page_bytes;
    rc = yvex_compilation_source_operations.open(
        &owner->source, &options, &qwen_source_projection, &failure, err);
    if (rc == YVEX_OK)
        rc = qwen_semantic_model_build(
            &owner->semantic,
            yvex_compilation_source_operations.verification(owner->source), err);
    if (rc != YVEX_OK) {
        qwen_source_release(owner);
        return rc;
    }
    out->owner = owner;
    out->verification = yvex_compilation_source_operations.verification(owner->source);
    out->transform_ir = yvex_compilation_source_operations.transform(owner->source);
    out->transform_binding = yvex_compilation_source_operations.binding(owner->source);
    out->artifact_lowering = yvex_compilation_source_operations.lowering(owner->source);
    out->source_summary = yvex_compilation_source_operations.summary(owner->source);
    out->lowering_context = out->artifact_lowering;
    out->tokenizer_vocabulary_size =
        out->verification ? out->verification->tokenizer_effective_vocab_size : 0ull;
    if (!out->verification || !out->transform_ir || !out->transform_binding ||
        !out->artifact_lowering || !out->source_summary ||
        !out->tokenizer_vocabulary_size) {
        qwen_source_release(owner);
        memset(out, 0, sizeof(*out));
        yvex_error_set(err, YVEX_ERR_STATE, "qwen3_5.compilation-source",
                       "Qwen source projection omitted a compiler input");
        return YVEX_ERR_STATE;
    }
    return YVEX_OK;
}

static void qwen_compilation_source_close(void *owner)
{
    qwen_source_release(owner);
}

static void qwen_source_release(void *pointer)
{
    qwen_source_owner *owner = pointer;

    if (!owner) return;
    yvex_semantic_model_ir_close(&owner->semantic);
    yvex_compilation_source_operations.close(owner->source);
    free(owner);
}

static int qwen_source_compile(yvex_family_source_products *out,
                               const yvex_compilation_runtime_binding_request *request,
                               yvex_error *err)
{
    yvex_family_compilation_source source = {0};
    qwen_source_owner *owner = NULL;
    const yvex_semantic_model_ir_summary *semantic;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (!out || !request) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "qwen3_5.source-compiler",
                       "exact source path and models root are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = qwen_compilation_source_open(&source, request, err);
    owner = source.owner;
    semantic = rc == YVEX_OK
                   ? yvex_semantic_model_ir_summary_get(owner->semantic) : NULL;
    if (rc != YVEX_OK || !semantic) {
        qwen_source_release(owner);
        return rc != YVEX_OK ? rc : YVEX_ERR_STATE;
    }
    out->owner = owner;
    out->release = qwen_source_release;
    out->verification = source.verification;
    out->source_summary = source.source_summary;
    out->semantic_model = owner->semantic;
    out->transform_ir = source.transform_ir;
    out->lowering = source.artifact_lowering;
    yvex_core_text_copy(out->derivation_identity,
                        sizeof(out->derivation_identity), semantic->identity);
    return YVEX_OK;
}

static int qwen_graph_plan_build(
    yvex_attention_plan **out, const yvex_semantic_model_ir *semantic_model,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    return yvex_attention_plan_build_semantic(
        out, semantic_model, YVEX_TENSOR_SCOPE_MAIN_LAYER,
        session, descriptor, failure, err);
}

static const yvex_graph_compiler_api *qwen_graph_compile(void)
{
    static const yvex_graph_compiler_api compiler = {
        .plan_build = qwen_graph_plan_build};

    return &compiler;
}

static int qwen_execution_capabilities(yvex_runtime_capabilities *out)
{
    if (!out) return 0;
    *out = (yvex_runtime_capabilities){
        .attention_semantics_ready = 1,
        .attention_core_ready = 1,
        .attention_envelope_ready = 1,
        .cpu_prefill_eager_ready = 1,
        .cpu_decode_eager_ready = 1,
        .cuda_eager_implemented = 1,
        .attention_state_delta_ready = 1,
        .attention_operator_ready = 1,
        .attention_trace_ready = 1,
        .attention_profile_ready = 1,
        .attention_benchmark_ready = 1,
        .output_head_binding_ready = 1,
        .output_head_projection_ready = 1};
    return yvex_runtime_capabilities_contract_valid(out);
}

static int qwen_transformer_policy(
    const yvex_runtime_descriptor_summary *runtime,
    yvex_transformer_family_policy *out)
{
    if (!runtime || !out ||
        runtime->model_execution.schema_version !=
            YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2)
        return 0;
    memset(out, 0, sizeof(*out));
    return 1;
}

static int qwen_logits_policy(yvex_logits_family_policy *out)
{
    if (!out) return 0;
    *out = (yvex_logits_family_policy){
        .schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1,
        .separate_output_head = 1,
        .tied_output_head = 0,
        .output_head_bias = 0};
    return 1;
}

static int qwen_speculation_policy(
    const yvex_runtime_descriptor_summary *runtime,
    yvex_speculation_family_policy *out)
{
    if (!runtime || !out ||
        runtime->model_execution.schema_version !=
            YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2)
        return 0;
    memset(out, 0, sizeof(*out));
    return 1;
}

static int qwen_artifact_admit(
    const yvex_artifact *artifact, yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    if (out) memset(out, 0, sizeof(*out));
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH;
        yvex_core_text_copy(failure->field, sizeof(failure->field),
                            "qwen-qualified-physical-catalog");
        failure->actual = artifact ? yvex_artifact_size(artifact) : 0ull;
    }
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "qwen3_5.artifact-catalog",
                   "Qwen source-faithful artifact has not entered the qualified physical catalog");
    return YVEX_ERR_UNSUPPORTED;
}

static int qwen_runtime_descriptor(
    yvex_runtime_descriptor **out,
    const yvex_complete_artifact_admission *admission,
    yvex_materialization_session *materialization,
    const void *lowering_context,
    const yvex_semantic_model_ir *semantic_model, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic =
        yvex_semantic_model_ir_summary_get(semantic_model);
    const yvex_model_execution_descriptor *execution =
        semantic ? &semantic->execution_descriptor : NULL;
    const yvex_semantic_numeric_contract *numeric =
        semantic ? &semantic->numeric_contract : NULL;
    yvex_runtime_descriptor_family_facts facts = {0};
    yvex_runtime_descriptor_failure failure = {0};
    yvex_materialization_projection projection;
    int rc;

    if (!semantic || !execution || !numeric ||
        semantic->schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2 ||
        execution->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2 ||
        numeric->schema_version != YVEX_SEMANTIC_NUMERIC_CONTRACT_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(numeric->identity)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "qwen3_5.runtime-descriptor",
                       "sealed hybrid execution and numeric contracts are required");
        return YVEX_ERR_FORMAT;
    }
    facts.logical_model_identity = execution->logical_model_identity;
    facts.runtime_numeric_identity = numeric->identity;
    facts.runtime_hadamard_revision = numeric->algorithm_revision;
    facts.runtime_numeric_schema_version = numeric->numeric_schema_version;
    facts.runtime_compute_policy_count = numeric->compute_policy_count;
    facts.runtime_activation_policy_count = numeric->activation_policy_count;
    facts.runtime_sparse_topk_policy_count = numeric->sparse_topk_policy_count;
    facts.layer_count = execution->layer_count;
    facts.vocabulary_size = execution->vocabulary_size;
    facts.model_execution = execution;
    rc = yvex_materialization_project_artifact_lowering(
        (const yvex_artifact_lowering_map *)lowering_context, &projection, err);
    return rc == YVEX_OK
               ? yvex_runtime_descriptor_build_projected(
                     out, admission, materialization, &facts, &projection,
                     &failure, err)
               : rc;
}

static const yvex_quant_artifact_lowering_rule qwen_quant_lowering_rules[] = {
    {YVEX_ARTIFACT_LOWERING_TRANSFORM_DIRECT,
     YVEX_TRANSFORM_OP_IDENTITY,
     YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, 0}};

static const yvex_quant_artifact_lowering_policy qwen_quant_lowering_policy = {
    QWEN_SOURCE_FAITHFUL_PRESET, QWEN_SOURCE_FAITHFUL_PRESET,
    qwen_quant_lowering_rules,
    sizeof(qwen_quant_lowering_rules) / sizeof(qwen_quant_lowering_rules[0])};

static int qwen_quant_default(
    yvex_quant_plan **out, const yvex_transform_ir *transform,
    const yvex_transform_binding *binding, const void *lowering_context,
    yvex_error *err)
{
    yvex_quant_failure failure = {0};

    return yvex_quant_plan_build_artifact_lowering_profile(
        out, transform, binding,
        (const yvex_artifact_lowering_map *)lowering_context,
        &qwen_quant_lowering_policy, YVEX_QUANT_PROFILE_SOURCE_FAITHFUL,
        NULL, &failure, err);
}

static int qwen_quant_policy(
    yvex_quant_plan **out, const yvex_transform_ir *transform,
    const yvex_transform_binding *binding, const void *lowering_context,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    yvex_error *err)
{
    yvex_quant_failure failure = {0};

    return yvex_quant_plan_build_artifact_lowering_policy(
        out, transform, binding,
        (const yvex_artifact_lowering_map *)lowering_context,
        &qwen_quant_lowering_policy, policy, imatrix_identity,
        NULL, &failure, err);
}

static void qwen_preset_rule(yvex_quant_policy_rule *rule)
{
    memset(rule, 0, sizeof(*rule));
    rule->schema_version = YVEX_QUANT_POLICY_SCHEMA_VERSION;
    rule->match_mask = YVEX_QUANT_MATCH_PHYSICAL_CLASS;
    rule->operation = YVEX_QUANT_POLICY_OPERATION_ANY;
    rule->scope = YVEX_TENSOR_SCOPE_GLOBAL;
    rule->physical_class = YVEX_QUANT_POLICY_PHYSICAL_QUANTIZABLE;
    rule->qtype = YVEX_QUANT_QTYPE_SOURCE;
    rule->requires_cpu_compute = 1;
    rule->requires_cuda_compute = 1;
    rule->priority = 10u;
    rule->label = "preserve pinned Qwen BF16 source representation";
}

static unsigned long long qwen_preset_count(void)
{
    return 1ull;
}

static const char *qwen_preset_name(unsigned long long index)
{
    return index == 0ull ? QWEN_SOURCE_FAITHFUL_PRESET : NULL;
}

static int qwen_preset_open(
    yvex_quant_policy **out, const char *name, yvex_error *err)
{
    yvex_quant_policy_rule rule;
    yvex_quant_policy_definition definition;

    if (!out || !name || strcmp(name, QWEN_SOURCE_FAITHFUL_PRESET) != 0) {
        if (out) *out = NULL;
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "quant_policy_preset",
                        "unknown Qwen quantization preset: %s",
                        name ? name : "-");
        return YVEX_ERR_UNSUPPORTED;
    }
    qwen_preset_rule(&rule);
    definition = (yvex_quant_policy_definition){
        QWEN_SOURCE_FAITHFUL_PRESET, YVEX_QWEN3_8_27B_TARGET_ID,
        "built-in-preset", &rule, 1ull};
    return yvex_quant_policy_create_definition(out, &definition, err);
}

static const yvex_quant_preset_catalog *qwen_quant_presets(void)
{
    static const yvex_quant_preset_catalog catalog = {
        YVEX_QUANT_PRESET_CATALOG_SCHEMA_V1,
        YVEX_QWEN3_8_27B_TARGET_ID,
        qwen_preset_count, qwen_preset_name, qwen_preset_open};

    return &catalog;
}

static const yvex_family_binding_pipeline qwen_binding_pipeline = {
    .schema_version = YVEX_FAMILY_BINDING_PIPELINE_SCHEMA_V1,
    .source_open = qwen_compilation_source_open,
    .source_close = qwen_compilation_source_close,
    .artifact_admit = qwen_artifact_admit,
    .semantic_model_build = qwen_semantic_model_build,
    .runtime_descriptor_build = qwen_runtime_descriptor,
    .quant_plan_default = qwen_quant_default,
    .quant_plan_policy = qwen_quant_policy,
    .tokenizer_architecture = YVEX_QWEN3_5_FAMILY_KEY,
    .tokenizer_pre = "qwen2"};

static const yvex_family_compiler_adapter qwen_compiler = {
    .schema_version = YVEX_FAMILY_COMPILER_SCHEMA_V2,
    .adapter_id = YVEX_QWEN3_5_ADAPTER_ID,
    .adapter_version = YVEX_QWEN3_5_ADAPTER_VERSION,
    .target_id = YVEX_QWEN3_8_27B_TARGET_ID,
    .family = YVEX_QWEN3_5_FAMILY_KEY,
    .graph = qwen_graph_compile,
    .operator_graph_build = yvex_operator_graph_ir_build_decoder,
    .execution_capabilities = qwen_execution_capabilities,
    .transformer_policy = qwen_transformer_policy,
    .logits_policy = qwen_logits_policy,
    .speculation_policy = qwen_speculation_policy,
    .tokenizer_policy = qwen_tokenizer_policy,
    .physical_variant = yvex_graph_physical_variant_api_get,
    .binding_pipeline = &qwen_binding_pipeline,
    .binding_compile = yvex_family_binding_compile};

static const yvex_model_deployment_defaults qwen_deployment_defaults = {
    .schema_version = YVEX_MODEL_DEPLOYMENT_DEFAULTS_SCHEMA_V1,
    .logical_family = YVEX_QWEN3_5_FAMILY_KEY,
    .logical_model = YVEX_QWEN3_8_27B_TARGET_ID,
    .quant_preset = QWEN_SOURCE_FAITHFUL_PRESET,
    .backend = "cuda",
    .engine_kind = "text",
    .execution_strategy = "target-only"};

static const yvex_graph_execution_binding *qwen_execution_binding(void)
{
    static const yvex_graph_execution_binding execution = {
        .schema_version = YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1,
        .adapter_id = YVEX_QWEN3_5_ADAPTER_ID,
        .adapter_version = YVEX_QWEN3_5_ADAPTER_VERSION,
        .target_id = YVEX_QWEN3_8_27B_TARGET_ID,
        .family_name = YVEX_QWEN3_5_FAMILY_KEY,
        .operator_family_key = "qwen",
        .operator_artifact_filename = "qwen3.8-27b-source-faithful.gguf",
        .source_manifest_filename = "qwen3.8-27b.source-manifest.json",
        .deployment_defaults = &qwen_deployment_defaults,
        .compiler = &qwen_compiler,
        .api = &yvex_attention_execution_api};

    return &execution;
}

static int qwen_tokenizer_policy(yvex_tokenizer_family_policy *out,
                                 yvex_error *err)
{
    return yvex_tokenizer_family_policy_compile(
               out, yvex_model_qwen3_5_conversation(),
               YVEX_TOKENIZER_KIND_GGML_GPT2,
               YVEX_TOKENIZER_MODEL_BPE_BYTELEVEL,
               YVEX_TOKENIZER_PROMPT_CONVERSATION, err) == YVEX_OK;
}

static const yvex_family_source_adapter *qwen_source_adapter(void)
{
    static const yvex_family_source_adapter adapter = {
        .schema_version = YVEX_FAMILY_SOURCE_ADAPTER_SCHEMA_V1,
        .target_id = YVEX_QWEN3_8_27B_TARGET_ID,
        .family = YVEX_QWEN3_5_FAMILY_KEY,
        .tokenizer_architecture = YVEX_QWEN3_5_FAMILY_KEY,
        .tokenizer_pre = "qwen2",
        .tokenizer_policy = qwen_tokenizer_policy,
        .compile = qwen_source_compile};

    return &adapter;
}

const yvex_family_descriptor yvex_graph_family_descriptor_qwen3_5 = {
    .schema_version = YVEX_FAMILY_DESCRIPTOR_SCHEMA_V1,
    .target_id = YVEX_QWEN3_8_27B_TARGET_ID,
    .family = YVEX_QWEN3_5_FAMILY_KEY,
    .tokenizer_architecture = YVEX_QWEN3_5_FAMILY_KEY,
    .tokenizer_pre = "qwen2",
    .execution = qwen_execution_binding,
    .quant_presets = qwen_quant_presets,
    .source = qwen_source_adapter};
