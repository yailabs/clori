/* Account every tensor class in the pinned Qwen3.8-27B source inventory. */
#include "tests/test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/families/qwen3_5.h>
#include <yvex/internal/source.h>

static void qwen_tensor_architecture(yvex_qwen3_5_architecture *architecture)
{
    unsigned long long layer;

    memset(architecture, 0, sizeof(*architecture));
    architecture->text.hidden_size = 5120ull;
    architecture->text.layer_count = 64ull;
    architecture->text.vocabulary_size = 248320ull;
    architecture->text.intermediate_size = 17408ull;
    architecture->text.full_attention_interval = 4ull;
    architecture->text.full_attention_layers = 16ull;
    architecture->text.linear_attention_layers = 48ull;
    architecture->text.attention_heads = 24ull;
    architecture->text.kv_heads = 4ull;
    architecture->text.attention_head_dimension = 256ull;
    architecture->text.linear_key_heads = 16ull;
    architecture->text.linear_value_heads = 48ull;
    architecture->text.linear_key_head_dimension = 128ull;
    architecture->text.linear_value_head_dimension = 128ull;
    architecture->text.linear_convolution_kernel = 4ull;
    architecture->text.mtp_hidden_layers = 1ull;
    architecture->vision.depth = 27ull;
    memset(architecture->architecture_identity, 'a', 64u);
    architecture->architecture_identity[64] = '\0';
    for (layer = 0ull; layer < architecture->text.layer_count; ++layer)
        architecture->text.layers[layer] = (layer + 1ull) % 4ull == 0ull
                                               ? YVEX_QWEN3_5_LAYER_FULL_ATTENTION
                                               : YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION;
}

static int qwen_tensor_add(yvex_native_weight_table *table, const char *name,
                           unsigned int rank, const unsigned long long *dims)
{
    unsigned long long elements = 1ull;
    unsigned int index;
    yvex_error err;

    for (index = 0u; index < rank; ++index) elements *= dims[index];
    yvex_error_clear(&err);
    return yvex_native_weight_table_add(table, name, "fixture.safetensors", "BF16",
                                        rank, dims, 0ull, elements * 2ull, &err) == YVEX_OK;
}

static int qwen_tensor_add_layer(yvex_native_weight_table *table,
                                 const yvex_qwen3_5_architecture *architecture,
                                 unsigned long long layer)
{
    static const char *const common[] = {
        "input_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight", "post_attention_layernorm.weight"};
    static const char *const full[] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight"};
    static const char *const delta[] = {
        "linear_attn.A_log", "linear_attn.conv1d.weight", "linear_attn.dt_bias",
        "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.norm.weight", "linear_attn.out_proj.weight"};
    const char *const *rows;
    size_t count, index;
    char name[192];
    unsigned long long dims[3];

    for (index = 0u; index < sizeof(common) / sizeof(common[0]); ++index) {
        if (strstr(common[index], "gate_proj") || strstr(common[index], "up_proj")) {
            dims[0] = 17408ull; dims[1] = 5120ull;
        } else if (strstr(common[index], "down_proj")) {
            dims[0] = 5120ull; dims[1] = 17408ull;
        } else {
            dims[0] = 5120ull;
        }
        snprintf(name, sizeof(name), "model.language_model.layers.%llu.%s",
                 layer, common[index]);
        if (!qwen_tensor_add(table, name,
                             strstr(common[index], "proj") ? 2u : 1u, dims)) return 0;
    }
    rows = architecture->text.layers[layer] == YVEX_QWEN3_5_LAYER_FULL_ATTENTION
               ? full : delta;
    count = architecture->text.layers[layer] == YVEX_QWEN3_5_LAYER_FULL_ATTENTION
                ? sizeof(full) / sizeof(full[0]) : sizeof(delta) / sizeof(delta[0]);
    for (index = 0u; index < count; ++index) {
        unsigned int rank = 1u;

        if (rows == full) {
            if (strstr(rows[index], "q_proj")) {
                dims[0] = 12288ull; dims[1] = 5120ull; rank = 2u;
            } else if (strstr(rows[index], "k_proj") || strstr(rows[index], "v_proj")) {
                dims[0] = 1024ull; dims[1] = 5120ull; rank = 2u;
            } else if (strstr(rows[index], "o_proj")) {
                dims[0] = 5120ull; dims[1] = 6144ull; rank = 2u;
            } else {
                dims[0] = 256ull;
            }
        } else if (strstr(rows[index], "conv1d")) {
            dims[0] = 10240ull; dims[1] = 1ull; dims[2] = 4ull; rank = 3u;
        } else if (strstr(rows[index], "in_proj_qkv")) {
            dims[0] = 10240ull; dims[1] = 5120ull; rank = 2u;
        } else if (strstr(rows[index], "in_proj_z")) {
            dims[0] = 6144ull; dims[1] = 5120ull; rank = 2u;
        } else if (strstr(rows[index], "in_proj_a") ||
                   strstr(rows[index], "in_proj_b")) {
            dims[0] = 48ull; dims[1] = 5120ull; rank = 2u;
        } else if (strstr(rows[index], "out_proj")) {
            dims[0] = 5120ull; dims[1] = 6144ull; rank = 2u;
        } else if (strstr(rows[index], "norm")) {
            dims[0] = 128ull;
        } else {
            dims[0] = 48ull;
        }
        snprintf(name, sizeof(name), "model.language_model.layers.%llu.%s",
                 layer, rows[index]);
        if (!qwen_tensor_add(table, name, rank, dims)) return 0;
    }
    return 1;
}

static int qwen_tensor_add_deferred(yvex_native_weight_table *table)
{
    static const char *const vision_block[] = {
        "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
        "mlp.linear_fc1.bias", "mlp.linear_fc1.weight", "mlp.linear_fc2.bias",
        "mlp.linear_fc2.weight", "norm1.bias", "norm1.weight", "norm2.bias",
        "norm2.weight"};
    static const char *const vision_global[] = {
        "model.visual.merger.linear_fc1.bias", "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc2.bias", "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.norm.bias", "model.visual.merger.norm.weight",
        "model.visual.patch_embed.proj.bias", "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight"};
    static const char *const mtp[] = {
        "mtp.fc.weight", "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight", "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight", "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight", "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight", "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight"};
    unsigned long long dims[] = {1ull};
    unsigned long long layer;
    size_t index;
    char name[192];

    for (layer = 0ull; layer < 27ull; ++layer)
        for (index = 0u; index < sizeof(vision_block) / sizeof(vision_block[0]); ++index) {
            snprintf(name, sizeof(name), "model.visual.blocks.%llu.%s", layer,
                     vision_block[index]);
            if (!qwen_tensor_add(table, name, 1u, dims)) return 0;
        }
    for (index = 0u; index < sizeof(vision_global) / sizeof(vision_global[0]); ++index)
        if (!qwen_tensor_add(table, vision_global[index], 1u, dims)) return 0;
    for (index = 0u; index < sizeof(mtp) / sizeof(mtp[0]); ++index)
        if (!qwen_tensor_add(table, mtp[index], 1u, dims)) return 0;
    return 1;
}

static yvex_native_weight_table *qwen_tensor_inventory_fixture(
    const yvex_qwen3_5_architecture *architecture)
{
    yvex_native_weight_table *table = calloc(1u, sizeof(*table));
    unsigned long long dims[2], layer;
    yvex_error err;

    if (!table) return NULL;
    dims[0] = 248320ull; dims[1] = 5120ull;
    if (!qwen_tensor_add(table, "model.language_model.embed_tokens.weight", 2u, dims) ||
        !qwen_tensor_add(table, "lm_head.weight", 2u, dims)) goto fail;
    dims[0] = 5120ull;
    if (!qwen_tensor_add(table, "model.language_model.norm.weight", 1u, dims)) goto fail;
    for (layer = 0ull; layer < architecture->text.layer_count; ++layer)
        if (!qwen_tensor_add_layer(table, architecture, layer)) goto fail;
    if (!qwen_tensor_add_deferred(table)) goto fail;
    yvex_error_clear(&err);
    if (yvex_native_weight_table_finalize(table, &err) != YVEX_OK) goto fail;
    return table;
fail:
    yvex_native_weight_table_close(table);
    return NULL;
}

int yvex_test_qwen3_5_tensors(void)
{
    const yvex_qwen3_5_api *api = yvex_model_register_qwen3_5();
    yvex_qwen3_5_architecture architecture;
    yvex_qwen3_5_tensor_inventory inventory;
    yvex_qwen3_5_tensor_inventory snapshot_inventory;
    yvex_qwen3_5_tensor_binding binding;
    yvex_qwen3_5_failure failure;
    yvex_native_weight_table *table;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_native_weight_info invalid = {0};
    unsigned long long invalid_dims[] = {1ull};
    yvex_error err;

    qwen_tensor_architecture(&architecture);
    table = qwen_tensor_inventory_fixture(&architecture);
    YVEX_TEST_ASSERT(api && table, "construct the complete pinned tensor population");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        api->tensor_inventory_audit(&architecture, table, &inventory,
                                    &failure, &err) == YVEX_OK &&
            inventory.complete && inventory.tensor_count == 1199ull &&
            inventory.class_counts[YVEX_QWEN3_5_TENSOR_TEXT_EXECUTION_REQUIRED] == 851ull &&
            inventory.class_counts[YVEX_QWEN3_5_TENSOR_VISION_DEFERRED] == 333ull &&
            inventory.class_counts[YVEX_QWEN3_5_TENSOR_MTP_DEFERRED] == 15ull &&
            inventory.class_counts[YVEX_QWEN3_5_TENSOR_UNKNOWN] == 0ull &&
            yvex_sha256_hex_valid(inventory.role_map_identity),
        "all text, vision-deferred, and MTP-deferred tensors are accounted");
    YVEX_TEST_ASSERT(
        inventory.role_counts[YVEX_QWEN3_5_ROLE_DELTA_QKV_PROJECTION] == 48ull &&
            inventory.role_counts[YVEX_QWEN3_5_ROLE_ATTENTION_Q] == 16ull &&
            inventory.role_counts[YVEX_QWEN3_5_ROLE_FFN_GATE] == 64ull,
        "hybrid topology owns exact role populations");
    memset(&invalid, 0, sizeof(invalid));
    invalid.name = "model.language_model.layers.0.unclassified.weight";
    invalid.shard_path = "fixture.safetensors";
    invalid.dtype = YVEX_NATIVE_DTYPE_BF16;
    invalid.dtype_name = "BF16";
    invalid.rank = 1u;
    invalid.dims[0] = invalid_dims[0];
    invalid.data_end = 2ull;
    invalid.data_bytes = 2ull;
    YVEX_TEST_ASSERT(
        api->tensor_classify(&architecture, &invalid, &binding,
                             &failure, &err) != YVEX_OK &&
            failure.code == YVEX_QWEN3_5_FAILURE_TENSOR_ROLE,
        "unknown pinned-source tensor blocks semantic closure");
    invalid.name = "model.language_model.layers.0.linear_attn.norm.weight";
    YVEX_TEST_ASSERT(
        api->tensor_classify(&architecture, &invalid, &binding,
                             &failure, &err) != YVEX_OK &&
            failure.code == YVEX_QWEN3_5_FAILURE_TENSOR_ROLE,
        "known tensor name with false geometry fails closed");
    YVEX_TEST_ASSERT(
        strcmp(api->tensor_class_name(YVEX_QWEN3_5_TENSOR_VISION_DEFERRED),
               "vision-deferred") == 0 &&
            strcmp(api->tensor_role_name(YVEX_QWEN3_5_ROLE_DELTA_OUTPUT),
                   "delta-output") == 0,
        "tensor classes and semantic roles have stable names");
    YVEX_TEST_ASSERT(
        yvex_source_tensor_snapshot_take_table(
            &snapshot, &table, 18ull, 1ull, &err) == YVEX_OK && snapshot && !table,
        "retain the canonical tensor table as an authenticated source snapshot");
    YVEX_TEST_ASSERT(
        api->tensor_snapshot_audit(&architecture, snapshot, &snapshot_inventory,
                                   &failure, &err) == YVEX_OK &&
            snapshot_inventory.complete &&
            snapshot_inventory.tensor_count == inventory.tensor_count &&
            snapshot_inventory.tensor_bytes == inventory.tensor_bytes &&
            strcmp(snapshot_inventory.role_map_identity,
                   inventory.role_map_identity) == 0,
        "snapshot and table audits project one exact role-map identity");
    yvex_source_tensor_snapshot_release(snapshot);
    return 0;
}
