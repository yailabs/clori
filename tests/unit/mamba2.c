/* Source-contract fixtures are not model acquisition or hosted-generation evidence. */
#include "tests/test.h"
#include <yvex/internal/families/mamba2.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/source_catalog.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static int mamba_write(const char *root, const char *name, const char *text)
{
    char path[768], metadata[768], oid[41];
    yvex_error err;
    FILE *fp;
    snprintf(path, sizeof(path), "%s/%s", root, name);
    fp = fopen(path, "wb");
    if (!fp) return 0;
    if (fputs(text, fp) < 0 || fclose(fp)) return 0;
    if (yvex_source_git_blob_oid_file(path, oid, &err) != YVEX_OK) return 0;
    snprintf(metadata, sizeof(metadata), "%s/.cache/huggingface/download/%s.metadata", root, name);
    fp = fopen(metadata, "wb");
    if (!fp) return 0;
    if (fprintf(fp, "%s\n%s\n0\n", YVEX_MAMBA2_REVISION, oid) < 0 || fclose(fp)) return 0;
    return 1;
}

static int mamba_fixture_config(const char *root, const char *model_type, unsigned int groups)
{
    char config[2048];
    snprintf(config, sizeof(config),
        "{\"architectures\":[\"Mamba2ForCausalLM\"],\"model_type\":\"%s\","
        "\"hidden_size\":4,\"num_hidden_layers\":2,\"vocab_size\":4,\"expand\":2,"
        "\"num_heads\":4,\"head_dim\":2,\"state_size\":2,\"n_groups\":%u,"
        "\"conv_kernel\":3,\"chunk_size\":2,\"bos_token_id\":0,\"eos_token_id\":0,"
        "\"pad_token_id\":0,\"layer_norm_epsilon\":0.00001,\"rms_norm\":true,"
        "\"residual_in_fp32\":true,\"tie_word_embeddings\":false,\"use_bias\":false,"
        "\"use_conv_bias\":true,\"norm_before_gate\":true,\"hidden_act\":\"silu\","
        "\"torch_dtype\":\"bfloat16\",\"time_step_limit\":[0.0,Infinity]}", model_type, groups);
    return mamba_write(root, "config.json", config);
}

static int mamba_inventory(const yvex_mamba2_architecture *a, yvex_native_weight_table **out)
{
    const char *const global[] = {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"};
    const char *const suffix[] = {"norm.weight", "mixer.in_proj.weight", "mixer.conv1d.weight",
        "mixer.conv1d.bias", "mixer.A_log", "mixer.D", "mixer.dt_bias", "mixer.norm.weight",
        "mixer.out_proj.weight"};
    const unsigned long long shapes[12][3] = {
        {4,4,0}, {4,0,0}, {4,4,0}, {4,0,0}, {28,4,0}, {16,1,3},
        {16,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {8,0,0}, {4,8,0}};
    unsigned long long i, layer, offset = 0;
    yvex_error err;
    *out = calloc(1, sizeof(**out));
    if (!*out) return 0;
    for (layer = 0; layer <= a->layer_count; ++layer)
        for (i = layer == a->layer_count ? 0u : 3u;
             i < (layer == a->layer_count ? 3u : 12u); ++i) {
            char name[128];
            unsigned int rank = shapes[i][2] ? 3u : shapes[i][1] ? 2u : 1u;
            unsigned long long bytes = 2u, j;
            if (i < 3u) snprintf(name, sizeof(name), "%s", global[i]);
            else snprintf(name, sizeof(name), "backbone.layers.%llu.%s", layer, suffix[i - 3u]);
            for (j = 0; j < rank; ++j) bytes *= shapes[i][j];
            if (yvex_native_weight_table_add(*out, name, "fixture.safetensors", "BF16",
                    rank, shapes[i], offset, offset + bytes, &err) != YVEX_OK) return 0;
            offset += bytes;
        }
    return yvex_native_weight_table_finalize(*out, &err) == YVEX_OK;
}

static int mamba_contract(const char *root)
{
    const yvex_mamba2_api *api = yvex_model_register_mamba2();
    yvex_source_verification verification = {0};
    yvex_mamba2_architecture a;
    yvex_mamba2_inventory inventory;
    yvex_mamba2_tensor_binding binding;
    yvex_native_weight_table *table = NULL;
    yvex_native_weight_info bad;
    yvex_error err;
    verification.verified = verification.config_valid = 1;
    snprintf(verification.resolved_source_path, sizeof(verification.resolved_source_path), "%s", root);
    snprintf(verification.repository_id, sizeof(verification.repository_id), "%s", YVEX_MAMBA2_REPOSITORY);
    snprintf(verification.revision, sizeof(verification.revision), "%s", YVEX_MAMBA2_REVISION);
    YVEX_TEST_ASSERT(api->open(&verification, &a, &err) == YVEX_OK && a.architecture_complete &&
        a.layer_count == 2u && a.mixer.width == 8u && a.mixer.convolution_state_values == 48u &&
        a.mixer.recurrent_state_values == 16u && a.token_policy_conflict &&
        a.normalization_policy_conflict && a.tokenizer_bos == 1u && a.tokenizer_eos == 2u,
        "source geometry and upstream conflicts are represented, not silently reconciled");
    YVEX_TEST_ASSERT(mamba_inventory(&a, &table) &&
        api->tensor_audit(&a, table, &inventory, &err) == YVEX_OK &&
        inventory.complete && inventory.tensors == 21u,
        "bidirectional role coverage is exact");
    bad = *yvex_native_weight_table_at(table, 0);
    bad.dims[0]++;
    YVEX_TEST_ASSERT(api->tensor_classify(&a, &bad, &binding, &err) != YVEX_OK,
        "wrong tensor shape rejected");
    bad = *yvex_native_weight_table_at(table, 0);
    bad.dtype = YVEX_NATIVE_DTYPE_F32;
    YVEX_TEST_ASSERT(api->tensor_classify(&a, &bad, &binding, &err) != YVEX_OK,
        "wrong source dtype rejected");
    bad = *yvex_native_weight_table_at(table, 0);
    bad.name = "backbone.layers.0.self_attn.q_proj.weight";
    YVEX_TEST_ASSERT(api->tensor_classify(&a, &bad, &binding, &err) != YVEX_OK,
        "attention tensors cannot be relabeled as SSM");
    table->count--;
    YVEX_TEST_ASSERT(api->tensor_audit(&a, table, &inventory, &err) != YVEX_OK && !inventory.complete,
        "missing required tensor prevents complete coverage");
    table->count++;
    yvex_native_weight_table_close(table);
    YVEX_TEST_ASSERT(!yvex_graph_execution_find(0, 0, YVEX_MAMBA2_TARGET),
        "family inspection does not advertise executable capability");
    verification.revision[0] = '0';
    YVEX_TEST_ASSERT(api->open(&verification, &a, &err) != YVEX_OK, "revision drift rejected");
    snprintf(verification.revision, sizeof(verification.revision), "%s", YVEX_MAMBA2_REVISION);
    YVEX_TEST_ASSERT(mamba_fixture_config(root, "mamba", 2u) &&
        api->open(&verification, &a, &err) != YVEX_OK, "Mamba1 declaration rejected");
    YVEX_TEST_ASSERT(mamba_fixture_config(root, "mamba2", 3u) &&
        api->open(&verification, &a, &err) != YVEX_OK, "head/group and params disagreement rejected");
    return 0;
}

static int mamba_acquired_contract(void)
{
    const char *source = getenv("YVEX_MAMBA2_SOURCE");
    const yvex_mamba2_api *api = yvex_model_register_mamba2();
    yvex_source_verify_options options = {0};
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_mamba2_architecture architecture;
    yvex_mamba2_inventory inventory;
    yvex_error err;
    int rc;
    if (!source || !source[0]) return 0;
    options.identity = yvex_source_target_identity_find(YVEX_MAMBA2_TARGET);
    options.source_path = source;
    options.models_root = getenv("YVEX_MAMBA2_MODELS_ROOT");
    options.manifest_path = getenv("YVEX_MAMBA2_MANIFEST");
    YVEX_TEST_ASSERT(options.models_root && options.manifest_path,
                     "real source proof requires its existing root and verified manifest");
    rc = yvex_source_verify_with_snapshot(&options, &verification, &snapshot, &err);
    if (rc == YVEX_OK) rc = api->open(&verification, &architecture, &err);
    if (rc == YVEX_OK) rc = api->snapshot_audit(&architecture, snapshot, &inventory, &err);
    yvex_source_tensor_snapshot_release(snapshot);
    YVEX_TEST_ASSERT(rc == YVEX_OK && inventory.complete, "real acquired source contract");
    printf("acquired-mamba2: revision=%s architecture=%s roles=%s tensors=%llu bytes=%llu "
           "conv_values_per_layer=%llu recurrent_values_per_layer=%llu layers=%llu "
           "token_conflict=%d normalization_conflict=%d executable=false\n",
           architecture.source_revision, architecture.architecture_identity, inventory.role_identity,
           inventory.tensors, inventory.tensor_bytes, architecture.mixer.convolution_state_values,
           architecture.mixer.recurrent_state_values, architecture.layer_count,
           architecture.token_policy_conflict, architecture.normalization_policy_conflict);
    return 0;
}

int yvex_test_mamba2_source(void)
{
    char root[] = "build/tests/mamba2-source.XXXXXX", path[768];
    const char *const dirs[] = {".cache", ".cache/huggingface", ".cache/huggingface/download"};
    const char *const files[] = {"config.json", "params.json", "generation_config.json", "tokenizer.json"};
    size_t i;
    int rc;
    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "create bounded family fixture root");
    for (i = 0; i < 3u; ++i) {
        snprintf(path, sizeof(path), "%s/%s", root, dirs[i]);
        YVEX_TEST_ASSERT(mkdir(path, 0700) == 0, "create fixture provider metadata directory");
    }
    YVEX_TEST_ASSERT(mamba_fixture_config(root, "mamba2", 2u) &&
        mamba_write(root, "params.json", "{\"dim\":4,\"n_layers\":2,\"vocab_size\":4,\"n_groups\":2,"
            "\"rms_norm\":true,\"residual_in_fp32\":true,\"tie_embeddings\":false,\"model_type\":\"mamba\"}") &&
        mamba_write(root, "generation_config.json", "{\"bos_token_id\":0,\"eos_token_id\":2,\"pad_token_id\":1}") &&
        mamba_write(root, "tokenizer.json", "{\"model\":{\"type\":\"BPE\","
            "\"vocab\":{\"<unk>\":0,\"<s>\":1,\"</s>\":2,\"a\":3}}}"), "write identity-bound fixtures");
    rc = mamba_contract(root);
    for (i = 0; i < 4u; ++i) {
        snprintf(path, sizeof(path), "%s/%s", root, files[i]);
        (void)unlink(path);
        snprintf(path, sizeof(path), "%s/.cache/huggingface/download/%s.metadata", root, files[i]);
        (void)unlink(path);
    }
    for (i = 3u; i > 0; --i) {
        snprintf(path, sizeof(path), "%s/%s", root, dirs[i - 1u]);
        (void)rmdir(path);
    }
    (void)rmdir(root);
    return rc ? rc : mamba_acquired_contract();
}
