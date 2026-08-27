/*
 * Exercises structured DeepSeek source verification with tiny metadata and safetensors headers.
 * No model payload fixture is used.
 */
#include "tests/test.h"

#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char source_verify_revision[] =
    "62af8fffb2f7030cac4de2f0169f5b8d1101b646";

static int source_verify_make_dir(const char *path)
{
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

static int source_verify_write_text(const char *path, const char *text)
{
    FILE *fp = fopen(path, "wb");

    if (!fp) return 0;
    if (fputs(text, fp) < 0) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static int source_verify_write_safetensors(const char *path)
{
    static const char header[] =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[0,8]},"
        "\"model.scale\":{\"dtype\":\"F8_E8M0\","
        "\"shape\":[1],\"data_offsets\":[8,9]},"
        "\"model.values\":{\"dtype\":\"I8\","
        "\"shape\":[2],\"data_offsets\":[9,11]}}";
    unsigned long long length = (unsigned long long)strlen(header);
    unsigned char bytes[8];
    FILE *fp;
    unsigned int i;

    fp = fopen(path, "wb");
    if (!fp) return 0;
    for (i = 0; i < 8u; ++i) {
        bytes[i] = (unsigned char)((length >> (8u * i)) & 0xffu);
    }
    if (fwrite(bytes, 1u, sizeof(bytes), fp) != sizeof(bytes) ||
        fwrite(header, 1u, (size_t)length, fp) != (size_t)length ||
        fwrite("12345678901", 1u, 11u, fp) != 11u) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static int source_verify_write_config(const char *root,
                                      const char *model_type,
                                      const char *architecture)
{
    char path[512];
    char json[4096];
    int n;

    n = snprintf(path, sizeof(path), "%s/config.json", root);
    if (n < 0 || (size_t)n >= sizeof(path)) return 0;
    n = snprintf(
        json, sizeof(json),
        "{\"architectures\":[\"%s\"],\"model_type\":\"%s\","
        "\"hidden_size\":4096,\"num_hidden_layers\":43,"
        "\"num_attention_heads\":64,\"num_key_value_heads\":1,"
        "\"head_dim\":512,\"qk_rope_head_dim\":64,"
        "\"max_position_embeddings\":1048576,"
        "\"moe_intermediate_size\":2048,\"n_routed_experts\":256,"
        "\"n_shared_experts\":1,\"num_experts_per_tok\":6,"
        "\"num_hash_layers\":3,\"q_lora_rank\":1024,"
        "\"o_lora_rank\":1024,\"vocab_size\":129280,"
        "\"sliding_window\":128,\"tie_word_embeddings\":false,"
        "\"torch_dtype\":\"bfloat16\",\"expert_dtype\":\"fp4\","
        "\"hidden_act\":\"silu\",\"attention_bias\":false,"
        "\"attention_dropout\":0.0,\"bos_token_id\":0,\"eos_token_id\":1,"
        "\"compress_ratios\":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0],"
        "\"compress_rope_theta\":160000,\"hc_eps\":0.000001,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":20,"
        "\"index_head_dim\":128,\"index_n_heads\":64,\"index_topk\":512,"
        "\"num_nextn_predict_layers\":1,\"dspark_block_size\":5,"
        "\"dspark_noise_token_id\":128799,"
        "\"dspark_target_layer_ids\":[40,41,42],"
        "\"dspark_markov_rank\":256,\"o_groups\":8,"
        "\"rms_norm_eps\":0.000001,\"rope_theta\":10000,"
        "\"routed_scaling_factor\":1.5,\"scoring_func\":\"sqrtsoftplus\","
        "\"topk_method\":\"noaux_tc\",\"norm_topk_prob\":true,"
        "\"swiglu_limit\":10.0,\"use_cache\":true,\"rope_scaling\":{"
        "\"type\":\"yarn\",\"factor\":16,"
        "\"original_max_position_embeddings\":65536,"
        "\"beta_fast\":32,\"beta_slow\":1},"
        "\"quantization_config\":{\"quant_method\":\"fp8\","
        "\"fmt\":\"e4m3\",\"activation_scheme\":\"dynamic\","
        "\"scale_fmt\":\"ue8m0\","
        "\"weight_block_size\":[128,128]}}",
        architecture, model_type);
    return n >= 0 && (size_t)n < sizeof(json) &&
           source_verify_write_text(path, json);
}

static int source_verify_write_manifest(const char *root, const char *kind,
                                        const char *repo,
                                        const char *status,
                                        const char *revision)
{
    char path[512];
    char json[2048];
    char revision_json[256];
    int n;

    n = snprintf(path, sizeof(path), "%s/source-manifest.json", root);
    if (n < 0 || (size_t)n >= sizeof(path)) return 0;
    n = revision
            ? snprintf(revision_json, sizeof(revision_json),
                       ",\"revision\":\"%s\"", revision)
            : snprintf(revision_json, sizeof(revision_json), "%s", "");
    if (n < 0 || (size_t)n >= sizeof(revision_json)) return 0;
    n = snprintf(
        json, sizeof(json),
        "{\"schema\":\"yvex.source_manifest.v1\",\"status\":\"%s\","
        "\"source\":{\"kind\":\"%s\",\"repo\":\"%s\"%s},"
        "\"local\":{\"path\":\"%s\"}}",
        status,
        kind,
        repo,
        revision_json,
        root);
    return n >= 0 && (size_t)n < sizeof(json) &&
           source_verify_write_text(path, json);
}

static int source_verify_write_metadata_revision(const char *root,
                                                 const char *name,
                                                 const char *revision)
{
    char path[768];
    char source_path[768];
    char text[256];
    char oid[41];
    yvex_error err;
    int n;

    n = snprintf(source_path, sizeof(source_path), "%s/%s", root, name);
    if (n < 0 || (size_t)n >= sizeof(source_path)) return 0;
    yvex_error_clear(&err);
    if (yvex_source_git_blob_oid_file(source_path, oid, &err) != YVEX_OK) {
        return 0;
    }
    n = snprintf(path, sizeof(path),
                 "%s/.cache/huggingface/download/%s.metadata", root, name);
    if (n < 0 || (size_t)n >= sizeof(path)) return 0;
    n = snprintf(text, sizeof(text), "%s\n%s\n0\n", revision, oid);
    return n >= 0 && (size_t)n < sizeof(text) &&
           source_verify_write_text(path, text);
}

static int source_verify_write_metadata(const char *root, const char *name)
{
    return source_verify_write_metadata_revision(root, name,
                                                 source_verify_revision);
}

static int source_verify_make_valid(const char *root)
{
    char path[512];

    if (!source_verify_make_dir("build") ||
        !source_verify_make_dir("build/tests") ||
        !source_verify_make_dir(root)) return 0;
    snprintf(path, sizeof(path), "%s/.cache", root);
    if (!source_verify_make_dir(path)) return 0;
    snprintf(path, sizeof(path), "%s/.cache/huggingface", root);
    if (!source_verify_make_dir(path)) return 0;
    snprintf(path, sizeof(path), "%s/.cache/huggingface/download", root);
    if (!source_verify_make_dir(path)) return 0;
    snprintf(path, sizeof(path), "%s/inference", root);
    if (!source_verify_make_dir(path)) return 0;
    snprintf(path, sizeof(path), "%s/.cache/huggingface/download/inference", root);
    if (!source_verify_make_dir(path)) return 0;
    if (!source_verify_write_manifest(root, "huggingface",
                                      yvex_source_release_identity()->upstream_repo_id,
                                      "in-progress", source_verify_revision) ||
        !source_verify_write_config(root,
                                    yvex_source_release_identity()->config_model_type,
                                    yvex_source_release_identity()->config_architecture)) return 0;
    snprintf(path, sizeof(path), "%s/tokenizer.json", root);
    if (!source_verify_write_text(path,
                                  "{\"version\":\"1.0\",\"added_tokens\":[{\"id\":129279,\"content\":\"<extra>\"}],"
                                  "\"normalizer\":null,\"pre_tokenizer\":{},"
                                  "\"post_processor\":{},\"decoder\":{},"
                                  "\"model\":{\"type\":\"BPE\",\"vocab\":{\"base\":127999}}}")) return 0;
    snprintf(path, sizeof(path), "%s/tokenizer_config.json", root);
    if (!source_verify_write_text(
            path,
            "{\"tokenizer_class\":\"PreTrainedTokenizerFast\","
            "\"model_max_length\":1048576,\"bos_token\":{},\"eos_token\":{}}")) return 0;
    snprintf(path, sizeof(path), "%s/generation_config.json", root);
    if (!source_verify_write_text(
            path,
            "{\"_from_model_config\":true,\"bos_token_id\":0,"
            "\"eos_token_id\":1,\"do_sample\":true,"
            "\"temperature\":1.0,\"top_p\":1.0,"
            "\"transformers_version\":\"4.46.3\"}")) return 0;
    snprintf(path, sizeof(path), "%s/inference/config.json", root);
    if (!source_verify_write_text(
            path,
            "{\"n_mtp_layers\":3,\"dspark_block_size\":5,"
            "\"dspark_noise_token_id\":128799,"
            "\"dspark_target_layer_ids\":[40,41,42],"
            "\"dspark_markov_rank\":256}")) return 0;
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", root);
    if (!source_verify_write_text(
            path,
            "{\"metadata\":{\"total_size\":11},\"weight_map\":{"
            "\"model.embed_tokens.weight\":\"model-00001-of-00001.safetensors\","
            "\"model.scale\":\"model-00001-of-00001.safetensors\","
            "\"model.values\":\"model-00001-of-00001.safetensors\"}}")) return 0;
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", root);
    if (!source_verify_write_safetensors(path)) return 0;
    return source_verify_write_metadata(root, "config.json") &&
           source_verify_write_metadata(root, "tokenizer.json") &&
           source_verify_write_metadata(root, "tokenizer_config.json") &&
           source_verify_write_metadata(root, "generation_config.json") &&
           source_verify_write_metadata(root, "inference/config.json") &&
           source_verify_write_metadata(root, "model.safetensors.index.json") &&
           source_verify_write_metadata(root,
                                        "model-00001-of-00001.safetensors");
}

static int source_verify_has_blocker(const yvex_source_verification *result,
                                     const char *blocker)
{
    unsigned int i;

    for (i = 0; result && i < result->blocker_count; ++i) {
        if (strcmp(result->blockers[i], blocker) == 0) return 1;
    }
    return 0;
}

static int source_verify_run_mode_snapshot(
    const char *root,
    int promote_manifest,
    yvex_source_verification *result,
    yvex_source_tensor_snapshot **snapshot,
    yvex_error *err)
{
    yvex_source_verify_options options;
    yvex_source_target_identity identity;
    char manifest_path[512];
    char index_path[512];
    char index_oid[41];
    struct stat st;

    memset(&options, 0, sizeof(options));
    identity = *yvex_source_release_identity();
    snprintf(index_path, sizeof(index_path),
             "%s/model.safetensors.index.json", root);
    yvex_error_clear(err);
    if (stat(index_path, &st) != 0 ||
        yvex_source_git_blob_oid_file(index_path, index_oid, err) != YVEX_OK) {
        return yvex_error_code(err) == YVEX_OK ? YVEX_ERR_IO
                                                : yvex_error_code(err);
    }
    identity.upstream_index_oid = index_oid;
    identity.upstream_index_size = (unsigned long long)st.st_size;
    snprintf(manifest_path, sizeof(manifest_path), "%s/source-manifest.json",
             root);
    options.identity = &identity;
    options.source_path = root;
    options.models_root = "build/tests";
    options.manifest_path = manifest_path;
    options.promote_manifest = promote_manifest;
    return yvex_source_verify_with_snapshot(&options, result, snapshot, err);
}

static int source_verify_run_mode(const char *root,
                                  int promote_manifest,
                                  yvex_source_verification *result,
                                  yvex_error *err)
{
    return source_verify_run_mode_snapshot(root, promote_manifest, result,
                                           NULL, err);
}

static int source_verify_run(const char *root,
                             yvex_source_verification *result,
                             yvex_error *err)
{
    return source_verify_run_mode(root, 0, result, err);
}

static int source_verify_payload_publication(void)
{
    const char *root = "build/tests/source-payload-publication";
    const char *manifest_path =
        "build/tests/source-payload-publication.json";
    yvex_source_payload_verification_result first;
    yvex_source_payload_verification_result second;
    yvex_source_payload_failure failure;
    yvex_source_payload_budget budget;
    yvex_source_verify_options options;
    yvex_source_target_identity identity;
    yvex_error err;
    char initial_manifest_path[512];
    char index_path[512];
    char index_oid[41];
    char metadata_path[768];
    char metadata_text[256];
    struct stat status;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/source-payload-publication") == 0,
                     "remove stale payload publication fixture");
    unlink(manifest_path);
    YVEX_TEST_ASSERT(source_verify_make_valid(root),
                     "create payload publication fixture");
    snprintf(metadata_path, sizeof(metadata_path),
             "%s/.cache/huggingface/download/"
             "model-00001-of-00001.safetensors.metadata",
             root);
    snprintf(metadata_text, sizeof(metadata_text), "%s\n%s\n0\n",
             source_verify_revision,
             "7c3a10eeb47de03729b98276827c00e954457cbb8c664ff265c7350a3ffbd14f");
    YVEX_TEST_ASSERT(source_verify_write_text(metadata_path, metadata_text),
                     "bind fixture shard to its provider SHA-256");
    identity = *yvex_source_release_identity();
    snprintf(index_path, sizeof(index_path),
             "%s/model.safetensors.index.json", root);
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(stat(index_path, &status) == 0 &&
                         yvex_source_git_blob_oid_file(
                             index_path, index_oid, &err) == YVEX_OK,
                     "bind fixture index identity");
    identity.upstream_index_oid = index_oid;
    identity.upstream_index_size = (unsigned long long)status.st_size;
    snprintf(initial_manifest_path, sizeof(initial_manifest_path),
             "%s/source-manifest.json", root);
    YVEX_TEST_ASSERT(rename(initial_manifest_path, manifest_path) == 0,
                     "place mutable provenance outside the source snapshot");
    memset(&options, 0, sizeof(options));
    options.identity = &identity;
    options.source_path = root;
    options.models_root = "build/tests";
    options.manifest_path = manifest_path;
    options.promote_manifest = 1;
    yvex_source_payload_budget_default(&budget);
    budget.allow_local_snapshot_seal = 0;
    memset(&first, 0, sizeof(first));
    memset(&failure, 0, sizeof(failure));
    rc = yvex_source_payload_verify_snapshot(
        &options, &budget, &first, &failure, &err);
    if (rc != YVEX_OK)
        fprintf(stderr,
                "payload publication refusal: %s; verified=%d trusted=%d "
                "schema=%s manifest_identity=%s payload_identity=%s "
                "manifest_source=%llu payload_source=%llu\n",
                yvex_error_message(&err), first.verification.verified,
                first.verification.manifest_payload_trusted,
                first.verification.manifest_schema,
                first.verification.manifest_payload_identity,
                first.payload.payload_identity,
                first.verification.manifest_payload_source_snapshot_identity,
                first.payload.source_snapshot_identity);
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "payload publication succeeds");
    YVEX_TEST_ASSERT(first.verification.verified &&
                         strcmp(first.verification.manifest_schema,
                                "yvex.source_manifest.v3") == 0,
                     "payload publication reopens its v3 manifest");
    YVEX_TEST_ASSERT(first.payload.state == YVEX_SOURCE_PAYLOAD_STATE_READY &&
                         first.payload.trust_class ==
                             YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED &&
                         strlen(first.payload.payload_identity) == 64u,
                     "payload publication retains its trusted session facts");
    YVEX_TEST_ASSERT(first.stream.complete &&
                         first.stream.physical_bytes_read > 0u &&
                         !first.reused_published_identity,
                     "first payload publication reads the admitted shard");

    memset(&second, 0, sizeof(second));
    memset(&failure, 0, sizeof(failure));
    yvex_error_clear(&err);
    rc = yvex_source_payload_verify_snapshot(
        &options, &budget, &second, &failure, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "payload identity reuse refusal: %s\n",
                yvex_error_message(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         second.reused_published_identity &&
                         second.stream.complete &&
                         second.stream.physical_bytes_read == 0u,
                     "a current v3 identity reopens without rereading payload bytes");
    YVEX_TEST_ASSERT(strcmp(second.payload.payload_identity,
                            first.payload.payload_identity) == 0,
                     "payload identity remains stable across reopen");
    return 0;
}

static int source_verify_run_identity(
    const char *root,
    const yvex_source_target_identity *identity,
    const char *upstream_inventory_path,
    const char *derived_inventory_path,
    int promote_manifest,
    yvex_source_verification *result,
    yvex_error *err)
{
    yvex_source_verify_options options;
    char manifest_path[512];

    memset(&options, 0, sizeof(options));
    snprintf(manifest_path, sizeof(manifest_path), "%s/source-manifest.json",
             root);
    options.identity = identity;
    options.source_path = root;
    options.models_root = "build/tests";
    options.manifest_path = manifest_path;
    options.upstream_inventory_path = upstream_inventory_path;
    options.derived_inventory_path = derived_inventory_path;
    options.promote_manifest = promote_manifest;
    return yvex_source_verify(&options, result, err);
}

static int source_verify_write_upstream_inventory(const char *path,
                                                  const char *repo,
                                                  const char *revision,
                                                  const char *shard,
                                                  unsigned long long size)
{
    char json[2048];
    int n = snprintf(
        json, sizeof(json),
        "{\"schema\":\"yvex.source_upstream_inventory.v1\","
        "\"repository\":\"%s\",\"revision\":\"%s\","
        "\"files\":[{\"path\":\"%s\",\"size_bytes\":%llu}]}",
        repo, revision, shard, size);

    return n >= 0 && (size_t)n < sizeof(json) &&
           source_verify_write_text(path, json);
}

static int source_verify_write_upstream_inventory_two(
    const char *path,
    const char *repo,
    const char *revision,
    const char *first,
    unsigned long long first_size,
    const char *second,
    unsigned long long second_size)
{
    char json[4096];
    int n = snprintf(
        json, sizeof(json),
        "{\"schema\":\"yvex.source_upstream_inventory.v1\","
        "\"repository\":\"%s\",\"revision\":\"%s\","
        "\"files\":[{\"path\":\"%s\",\"size_bytes\":%llu},"
        "{\"path\":\"%s\",\"size_bytes\":%llu}]}",
        repo, revision, first, first_size, second, second_size);

    return n >= 0 && (size_t)n < sizeof(json) &&
           source_verify_write_text(path, json);
}

static int source_verify_json_iteration(void)
{
    static const char document[] = "{\"values\":[1,2,],\"empty\":{},}";
    static const char malformed[] = "{\"first\":1 \"second\":2}";
    yvex_json json;
    yvex_json_iter object;
    yvex_json_iter array;
    yvex_json_item item;
    unsigned long long value;
    char key[32];

    yvex_json_init(&json, document, sizeof(document) - 1u);
    YVEX_TEST_ASSERT(yvex_json_iter_begin(&json, &object, YVEX_JSON_COLLECTION_OBJECT), "JSON object iterator begins");
    YVEX_TEST_ASSERT(yvex_json_object_member(&object, key, sizeof(key)) ==
                             YVEX_JSON_ITEM_READY &&
                         strcmp(key, "values") == 0 && yvex_json_iter_begin(&json, &array, YVEX_JSON_COLLECTION_ARRAY),
                     "JSON object member exposes its unconsumed array");
    YVEX_TEST_ASSERT(yvex_json_array_value(&array) == YVEX_JSON_ITEM_READY &&
                         yvex_json_u64(&json, &value) && value == 1u &&
                         yvex_json_array_value(&array) == YVEX_JSON_ITEM_READY &&
                         yvex_json_u64(&json, &value) && value == 2u &&
                         yvex_json_array_value(&array) == YVEX_JSON_ITEM_END &&
                         array.trailing_separator,
                     "JSON array iterator preserves trailing-separator evidence");
    YVEX_TEST_ASSERT(yvex_json_object_member(&object, key, sizeof(key)) ==
                             YVEX_JSON_ITEM_READY &&
                         strcmp(key, "empty") == 0 && yvex_json_iter_begin(&json, &array, YVEX_JSON_COLLECTION_OBJECT) &&
                         yvex_json_object_member(&array, key, sizeof(key)) ==
                             YVEX_JSON_ITEM_END &&
                         yvex_json_object_member(&object, key, sizeof(key)) ==
                             YVEX_JSON_ITEM_END &&
                         object.trailing_separator && yvex_json_complete(&json),
                     "JSON iterators close nested and trailing-comma objects exactly");

    yvex_json_init(&json, malformed, sizeof(malformed) - 1u);
    YVEX_TEST_ASSERT(yvex_json_iter_begin(&json, &object, YVEX_JSON_COLLECTION_OBJECT) &&
                         yvex_json_object_member(&object, key, sizeof(key)) ==
                             YVEX_JSON_ITEM_READY &&
                         yvex_json_u64(&json, &value),
                     "JSON malformed-separator fixture consumes its first member");
    item = yvex_json_object_member(&object, key, sizeof(key));
    YVEX_TEST_ASSERT(item == YVEX_JSON_ITEM_ERROR,
                     "JSON object iterator rejects a missing comma");

    yvex_json_init(&json, "[1,]", 4u);
    YVEX_TEST_ASSERT(!yvex_json_skip_value(&json),
                     "canonical recursive JSON arrays still reject trailing commas");
    yvex_json_init(&json, "{\"key\":1,}", 10u);
    YVEX_TEST_ASSERT(!yvex_json_skip_value(&json),
                     "canonical recursive JSON objects still reject trailing commas");
    return 0;
}

static int source_acquisition_digest(const char *text, char output[65])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, text, strlen(text)) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int source_acquisition_fixture_write(const char *root,
                                            const char *relative_path,
                                            const char *payload,
                                            const char *git_oid_override)
{
    static const char repository[] = "Example/Tiny";
    static const char revision[] = "0123456789abcdef0123456789abcdef01234567";
    char payload_path[512];
    char manifest_path[512];
    char actual_sha[65];
    char git_oid[41];
    char identity[65];
    char json[4096];
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    size_t size = strlen(payload);
    int n;

    if (!source_acquisition_digest(payload, actual_sha)) return 0;
    memset(git_oid, '0', sizeof(git_oid) - 1u);
    git_oid[sizeof(git_oid) - 1u] = '\0';
    if (strcmp(relative_path, "FL2VA/config.json") == 0) {
        n = snprintf(payload_path, sizeof(payload_path), "%s/FL2VA/config.json", root);
        if (n < 0 || (size_t)n >= sizeof(payload_path) ||
            !source_verify_write_text(payload_path, payload) ||
            yvex_source_git_blob_oid_file(payload_path, git_oid, NULL) != YVEX_OK) return 0;
    }
    if (git_oid_override) {
        if (strlen(git_oid_override) != 40u) return 0;
        memcpy(git_oid, git_oid_override, sizeof(git_oid));
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, YVEX_SOURCE_ACQUISITION_SCHEMA) ||
        !yvex_sha256_update_text(&hash, repository) ||
        !yvex_sha256_update_text(&hash, revision) ||
        !yvex_sha256_update_text(&hash, "FL2VA") ||
        !yvex_sha256_update_u64(&hash, 1u) ||
        !yvex_sha256_update_text(&hash, relative_path) ||
        !yvex_sha256_update_u64(&hash, size) ||
        !yvex_sha256_update_text(&hash, "") ||
        !yvex_sha256_update_text(&hash, actual_sha) ||
        !yvex_sha256_update_text(&hash, git_oid) ||
        !yvex_sha256_update_text(&hash, "") ||
        !yvex_sha256_update_text(&hash, "") ||
        !yvex_sha256_update_text(&hash, "metadata") ||
        !yvex_sha256_update_text(&hash, "pipeline") ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    n = snprintf(
        json, sizeof(json),
        "{\"schema\":\"%s\",\"repository\":\"%s\","
        "\"revision\":\"%s\",\"admitted_subtree\":\"FL2VA\","
        "\"acquisition_complete\":true,\"acquisition_identity\":\"%s\","
        "\"shards\":0,\"metadata_bytes\":%zu,\"shard_bytes\":0,"
        "\"source_bytes\":%zu,\"files\":[{\"path\":\"%s\","
        "\"component\":\"pipeline\",\"classification\":\"metadata\","
        "\"expected_size\":%zu,\"actual_size\":%zu,"
        "\"expected_sha256\":\"\",\"actual_sha256\":\"%s\","
        "\"git_oid\":\"%s\",\"lfs_oid\":\"\",\"xet_hash\":\"\","
        "\"verified\":true}]}",
        YVEX_SOURCE_ACQUISITION_SCHEMA, repository, revision, identity,
        size, size, relative_path, size, size, actual_sha, git_oid);
    if (n < 0 || (size_t)n >= sizeof(json)) return 0;
    n = snprintf(manifest_path, sizeof(manifest_path), "%s/%s", root,
                 YVEX_SOURCE_ACQUISITION_MANIFEST);
    if (n < 0 || (size_t)n >= sizeof(manifest_path) ||
        !source_verify_write_text(manifest_path, json)) return 0;
    return 1;
}

static int source_verify_acquisition(void)
{
    static const char repository[] = "Example/Tiny";
    static const char revision[] = "0123456789abcdef0123456789abcdef01234567";
    const char *root = "build/tests/source-acquisition";
    const char *payload = "{\"model_type\":\"tiny\"}\n";
    yvex_source_acquisition_options options;
    yvex_source_acquisition_failure failure;
    yvex_source_acquisition *acquisition = NULL;
    const yvex_source_acquisition_facts *facts;
    const yvex_source_acquisition_file *file;
    yvex_source_metadata_blob blob;
    yvex_error err;
    char path[512];
    int rc;

    (void)system("rm -rf build/tests/source-acquisition");
    YVEX_TEST_ASSERT(source_verify_make_dir(root), "create acquisition root");
    snprintf(path, sizeof(path), "%s/FL2VA", root);
    YVEX_TEST_ASSERT(source_verify_make_dir(path), "create acquisition subtree");
    YVEX_TEST_ASSERT(source_acquisition_fixture_write(
                         root, "FL2VA/config.json", payload, NULL),
                     "write canonical acquisition fixture");
    yvex_source_acquisition_options_default(&options);
    options.source_root = root;
    options.expected_repository = repository;
    options.expected_revision = revision;
    options.expected_subtree = "FL2VA";
    options.maximum_files = 1u;
    options.maximum_source_bytes = strlen(payload);
    yvex_error_clear(&err);
    rc = yvex_source_acquisition_open(&acquisition, &options, &failure, &err);
    facts = yvex_source_acquisition_facts_get(acquisition);
    file = yvex_source_acquisition_file_at(acquisition, 0u);
    YVEX_TEST_ASSERT(rc == YVEX_OK && facts && facts->complete &&
                         facts->file_count == 1u && facts->shard_count == 0u &&
                         facts->source_bytes == strlen(payload) &&
                         facts->payload_bytes_read == strlen(payload) && file &&
                         file->local_identity_verified && file->verified_inode,
                     "source admission binds complete bytes and their local file identity");
    memset(&blob, 0, sizeof(blob));
    YVEX_TEST_ASSERT(
        yvex_source_acquisition_metadata_read(acquisition, root, "FL2VA/config.json",
                                              strlen(payload), &blob, &err) == YVEX_OK &&
            blob.byte_count == strlen(payload) &&
            memcmp(blob.bytes, payload, strlen(payload)) == 0 &&
            blob.identity.identity_verified && blob.identity.revision_matches &&
            strcmp(blob.identity.expected_git_blob_oid,
                   blob.identity.observed_git_blob_oid) == 0,
        "verified acquisition metadata is read exactly through its retained local identity");
    yvex_source_metadata_blob_release(&blob);
    YVEX_TEST_ASSERT(
        yvex_source_acquisition_metadata_read(acquisition, root, "FL2VA/config.json",
                                              strlen(payload) - 1u, &blob, &err) ==
            YVEX_ERR_BOUNDS && !blob.bytes,
        "verified acquisition metadata refuses a caller budget below its exact size");
    yvex_source_acquisition_release(&acquisition);

    options.expected_repository = "Example/Wrong";
    YVEX_TEST_ASSERT(yvex_source_acquisition_open(
                         &acquisition, &options, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_ACQUISITION_FAILURE_SOURCE_IDENTITY,
                     "source admission rejects the wrong repository");
    options.expected_repository = repository;
    options.maximum_source_bytes = strlen(payload) - 1u;
    YVEX_TEST_ASSERT(yvex_source_acquisition_open(
                         &acquisition, &options, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_ACQUISITION_FAILURE_RESOURCE_BUDGET,
                     "source admission enforces its byte budget");
    options.maximum_source_bytes = strlen(payload);
    snprintf(path, sizeof(path), "%s/FL2VA/config.json", root);
    YVEX_TEST_ASSERT(source_verify_write_text(path, "{\"model_type\":\"evil\"}\n") &&
                         yvex_source_acquisition_open(
                             &acquisition, &options, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_ACQUISITION_FAILURE_DIGEST,
                     "source admission rejects same-sized digest corruption");
    YVEX_TEST_ASSERT(source_verify_write_text(path, payload),
                     "restore acquisition payload");
    YVEX_TEST_ASSERT(source_acquisition_fixture_write(
                         root, "FL2VA/config.json", payload,
                         "0000000000000000000000000000000000000000") &&
                         yvex_source_acquisition_open(
                             &acquisition, &options, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_ACQUISITION_FAILURE_DIGEST,
                     "source admission rejects bytes outside their Git identity");
    YVEX_TEST_ASSERT(source_acquisition_fixture_write(
                         root, "FL2VA/config.json", payload, NULL),
                     "restore authoritative acquisition identity");
    YVEX_TEST_ASSERT(unlink(path) == 0 && symlink("/dev/null", path) == 0 &&
                         yvex_source_acquisition_open(
                             &acquisition, &options, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_ACQUISITION_FAILURE_SYMLINK,
                     "source admission rejects a symlinked file");
    YVEX_TEST_ASSERT(unlink(path) == 0 && source_verify_write_text(path, payload) &&
                         source_acquisition_fixture_write(
                             root, "../outside.json", payload, NULL) &&
                         yvex_source_acquisition_open(
                             &acquisition, &options, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_ACQUISITION_FAILURE_MANIFEST_FORMAT,
                     "source admission rejects subtree traversal");
    (void)system("rm -rf build/tests/source-acquisition");
    return 0;
}

int yvex_test_source_verify(void)
{
    const char *root = "build/tests/source-verify";
    yvex_source_verification result;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_source_tensor_snapshot_facts snapshot_facts;
    yvex_error err;
    unsigned long long total;
    char path[512];
    int rc;

    if (source_verify_json_iteration() != 0)
        return 1;
    if (source_verify_payload_publication() != 0)
        return 1;
    if (source_verify_acquisition() != 0)
        return 1;

    YVEX_TEST_ASSERT(
        yvex_source_is_release_target(YVEX_SOURCE_RELEASE_TARGET_ID) &&
            !yvex_source_is_release_target("deepseek4-v4-flash-dspark-other") &&
            strcmp(yvex_source_release_identity()->upstream_revision,
                   YVEX_SOURCE_RELEASE_REVISION) == 0 &&
            strcmp(yvex_source_release_identity()->upstream_index_oid,
                   YVEX_SOURCE_RELEASE_INDEX_OID) == 0 &&
            yvex_source_target_path(path, sizeof(path), "/models",
                                    yvex_source_release_identity()) &&
            strcmp(path, "/models/hf/deepseek/DeepSeek-V4-Flash-DSpark") == 0,
        "source owner exposes the exact release identity and canonical path");
    YVEX_TEST_ASSERT(
        yvex_source_target_identity_find_repository(
                YVEX_SOURCE_RELEASE_REPOSITORY) ==
                yvex_source_release_identity() &&
            strcmp(yvex_source_target_identity_find(
                       YVEX_SOURCE_MINIMAX_H3_TARGET_ID)
                       ->upstream_repo_id,
                   YVEX_SOURCE_MINIMAX_H3_REPOSITORY) == 0 &&
            strcmp(yvex_source_target_identity_find_repository(
                       YVEX_SOURCE_MINIMAX_H3_REPOSITORY)
                       ->upstream_revision,
                   YVEX_SOURCE_MINIMAX_H3_REVISION) == 0 &&
            !yvex_source_target_identity_find_repository("unknown/model"),
        "one source catalog owns qualified target repository and revision truth");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root), "create valid source fixture");
    {
        yvex_source_manifest_file_list files;
        size_t file_index;

        yvex_source_manifest_file_list_init(&files);
        yvex_error_clear(&err);
        YVEX_TEST_ASSERT(
            yvex_source_manifest_scan_files(root, 1, &files, &err) == YVEX_OK &&
                files.summary.file_count == 8u && files.summary.safetensors_count == 1u,
            "source footprint excludes acquisition cache metadata");
        for (file_index = 0u; file_index < files.count; ++file_index) {
            YVEX_TEST_ASSERT(strncmp(files.items[file_index].path, ".cache/", 7u) != 0,
                             "source footprint exposes no acquisition-cache row");
        }
        yvex_source_manifest_file_list_free(&files);
    }
    rc = source_verify_run_mode_snapshot(root, 1, &result, &snapshot, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && result.verified,
                     "valid exact source verifies");
    YVEX_TEST_ASSERT(result.manifest_verified && result.manifest_published &&
                     result.manifest_reopened && result.header_scan_count == 1,
                     "verifier promotes, reopens, and scans headers once");
    YVEX_TEST_ASSERT_STREQ(result.repository_id,
                           yvex_source_release_identity()->upstream_repo_id,
                           "repository identity matches");
    YVEX_TEST_ASSERT_STREQ(result.revision, source_verify_revision,
                           "exact revision matches");
    YVEX_TEST_ASSERT(result.shard_count == 1 &&
                     result.header_tensor_count == 3 &&
                     result.shard_index_headers_match,
                     "index and header inventory agree");
    YVEX_TEST_ASSERT(snapshot &&
                         yvex_source_tensor_snapshot_facts_get(
                             snapshot, &snapshot_facts, &err) == YVEX_OK &&
                         snapshot_facts.tensor_count == 3u &&
                         snapshot_facts.shard_count == 1u &&
                         snapshot_facts.header_scan_count == 1u &&
                         snapshot_facts.payload_bytes_read == 0u,
                     "strict verification publishes its immutable one-pass snapshot");
    YVEX_TEST_ASSERT(yvex_source_tensor_snapshot_find(
                         snapshot, "model.embed_tokens.weight") != NULL,
                     "published snapshot retains header tensor facts");
    yvex_source_tensor_snapshot_release(snapshot);
    snapshot = NULL;
    YVEX_TEST_ASSERT(result.dtype_bf16_count == 1 &&
                     result.dtype_f8_e8m0_count == 1 &&
                     result.dtype_i8_count == 1 &&
                     result.dtype_other_count == 0,
                     "raw source dtype facts remain distinct");
    YVEX_TEST_ASSERT(result.generation_config_valid &&
                     result.generation_bos_token_id == 0 &&
                     result.generation_eos_token_id == 1,
                     "generation sidecar facts are verified");
    YVEX_TEST_ASSERT(result.compress_ratio_count == 46 &&
                     result.index_topk == 512 &&
                     strcmp(result.scoring_func, "sqrtsoftplus") == 0 &&
                     strcmp(result.tokenizer_model_type, "BPE") == 0 &&
                     result.tokenizer_effective_vocab_size == 129280 &&
                     result.rope_beta_fast == 32 &&
                     result.rope_beta_slow == 1 &&
                     strcmp(result.quant_scale_format, "ue8m0") == 0 &&
                     result.inference_config_valid &&
                     result.dspark_block_size == 5u &&
                     result.dspark_noise_token_id == 128799u &&
                     result.dspark_target_layer_count == 3u &&
                     result.dspark_target_layer_ids[0] == 40u &&
                     result.dspark_target_layer_ids[1] == 41u &&
                     result.dspark_target_layer_ids[2] == 42u &&
                     result.dspark_markov_rank == 256u &&
                     result.dspark_inference_layer_count == 3u,
                     "execution-affecting config and tokenizer facts are preserved");
    {
        yvex_deepseek_v4_ir *ir = NULL;
        yvex_deepseek_v4_ir_failure failure;

        yvex_error_clear(&err);
        YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->ir.build(
                             &ir, &result, &failure, &err) != YVEX_OK && !ir &&
                         failure.code ==
                             YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_NOT_VERIFIED,
                         "bounded fixture cannot impersonate the full pinned release topology");
        YVEX_TEST_ASSERT(result.header_scan_count == 1,
                         "rejected architecture construction performs no rescan");
        yvex_model_register_deepseek_v4()->ir.close(ir);
    }

    {
        yvex_source_target_identity indexless =
            *yvex_source_release_identity();
        char upstream_path[512];
        char derived_path[512];
        char shard_path[512];
        char index_path[512];
        char index_metadata_path[768];
        struct stat shard_stat;
        FILE *fp;
        char first[4096];
        char second[4096];
        size_t first_size;
        size_t second_size;

        snprintf(index_path, sizeof(index_path),
                 "%s/model.safetensors.index.json", root);
        snprintf(index_metadata_path, sizeof(index_metadata_path),
                 "%s/.cache/huggingface/download/model.safetensors.index.json.metadata",
                 root);
        YVEX_TEST_ASSERT(unlink(index_path) == 0 &&
                         unlink(index_metadata_path) == 0,
                         "remove index for official indexless fixture");
        snprintf(shard_path, sizeof(shard_path),
                 "%s/model-00001-of-00001.safetensors", root);
        YVEX_TEST_ASSERT(stat(shard_path, &shard_stat) == 0,
                         "stat indexless shard");
        snprintf(upstream_path, sizeof(upstream_path),
                 "%s-upstream.json", root);
        snprintf(derived_path, sizeof(derived_path),
                 "%s-derived.json", root);
        unlink(upstream_path);
        unlink(derived_path);
        YVEX_TEST_ASSERT(source_verify_write_manifest(
                             root, "huggingface",
                             yvex_source_release_identity()->upstream_repo_id,
                             "in-progress", source_verify_revision) &&
                         source_verify_write_upstream_inventory(
                             upstream_path,
                             yvex_source_release_identity()->upstream_repo_id,
                             source_verify_revision,
                             "model-00001-of-00001.safetensors",
                             (unsigned long long)shard_stat.st_size),
                         "write official indexless snapshot fixture");
        indexless.upstream_index_path = NULL;
        indexless.upstream_index_oid = "not-applicable";
        indexless.upstream_index_size = 0u;
        indexless.upstream_inventory_authority = "header-derived";
        YVEX_TEST_ASSERT(
            source_verify_run_identity(root, &indexless, upstream_path,
                                       derived_path, 1, &result, &err) == YVEX_OK &&
                result.verified &&
                strcmp(result.inventory_authority, "header-derived") == 0 &&
                access(derived_path, F_OK) == 0,
            "official indexless source derives deterministic inventory");
        fp = fopen(derived_path, "rb");
        YVEX_TEST_ASSERT(fp != NULL, "open first derived inventory");
        first_size = fread(first, 1u, sizeof(first), fp);
        YVEX_TEST_ASSERT(!ferror(fp) && fclose(fp) == 0,
                         "read first derived inventory");
        YVEX_TEST_ASSERT(
            source_verify_run_identity(root, &indexless, upstream_path,
                                       derived_path, 1, &result, &err) == YVEX_OK &&
                result.verified,
            "repeat indexless verification");
        fp = fopen(derived_path, "rb");
        YVEX_TEST_ASSERT(fp != NULL, "open repeated derived inventory");
        second_size = fread(second, 1u, sizeof(second), fp);
        YVEX_TEST_ASSERT(!ferror(fp) && fclose(fp) == 0 &&
                         first_size == second_size &&
                         memcmp(first, second, first_size) == 0,
                         "derived inventory is deterministic");
    }

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root), "recreate wrong repo fixture");
    YVEX_TEST_ASSERT(source_verify_write_manifest(root, "huggingface",
                                                  "wrong/repository",
                                                  "complete",
                                                  source_verify_revision),
                     "write wrong repo manifest");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     !result.verified &&
                     source_verify_has_blocker(&result, "wrong-source-repository"),
                     "wrong repository is refused");

    YVEX_TEST_ASSERT(source_verify_write_manifest(
                         root, "local", yvex_source_release_identity()->upstream_repo_id,
                         "complete", source_verify_revision),
                     "write unsupported source kind manifest");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     !result.verified &&
                     source_verify_has_blocker(&result,
                                               "unsupported-source-kind"),
                     "unsupported source kind is refused");

    YVEX_TEST_ASSERT(source_verify_write_manifest(
                         root, "huggingface",
                         yvex_source_release_identity()->upstream_repo_id,
                         "complete", NULL),
                     "write absent revision manifest");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     !result.verified &&
                     source_verify_has_blocker(&result, "missing-source-revision"),
                     "absent revision is refused");
    YVEX_TEST_ASSERT(source_verify_write_manifest(
                         root, "huggingface",
                         yvex_source_release_identity()->upstream_repo_id,
                         "complete", "unknown"),
                     "write unverifiable revision manifest");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     !result.verified &&
                     source_verify_has_blocker(
                         &result, "unverifiable-source-revision"),
                     "unknown revision is not promoted to verified provenance");
    YVEX_TEST_ASSERT(source_verify_write_manifest(
                         root, "huggingface",
                         yvex_source_release_identity()->upstream_repo_id,
                         "in-progress", source_verify_revision),
                     "write incomplete manifest status");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     !result.verified &&
                     source_verify_has_blocker(
                         &result, "source-manifest-incomplete"),
                     "in-progress manifest is refused");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root), "recreate wrong config fixture");
    YVEX_TEST_ASSERT(source_verify_write_config(root, "not_deepseek_v4",
                                                yvex_source_release_identity()->config_architecture),
                     "write wrong config identity");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result, "wrong-source-model-type"),
                     "wrong structured config identity is refused");
    snprintf(path, sizeof(path), "%s/config.json", root);
    YVEX_TEST_ASSERT(source_verify_write_text(path, "{"),
                     "write malformed config");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result, "malformed-source-config"),
                     "malformed config is refused");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root), "recreate tokenizer fixture");
    snprintf(path, sizeof(path), "%s/tokenizer.json", root);
    YVEX_TEST_ASSERT(unlink(path) == 0, "remove tokenizer");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result, "missing-tokenizer-json"),
                     "missing tokenizer is refused");
    YVEX_TEST_ASSERT(source_verify_write_text(path, "{}"),
                     "write structurally incomplete tokenizer");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(
                         &result, "malformed-tokenizer-json"),
                     "tokenizer structure is validated, not only JSON syntax");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root),
                     "recreate generation config fixture");
    snprintf(path, sizeof(path), "%s/generation_config.json", root);
    YVEX_TEST_ASSERT(unlink(path) == 0, "remove generation config");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result,
                                               "missing-generation-config"),
                     "missing generation config is refused");
    YVEX_TEST_ASSERT(source_verify_write_text(
                         path,
                         "{\"_from_model_config\":true,\"bos_token_id\":9,"
                         "\"eos_token_id\":1,\"do_sample\":true,"
                         "\"temperature\":1.0,\"top_p\":1.0,"
                         "\"transformers_version\":\"4.46.3\"}"),
                     "write inconsistent generation config");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(
                         &result, "generation-config-token-mismatch"),
                     "generation token identity must match model config");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root),
                     "recreate stale revision fixture");
    YVEX_TEST_ASSERT(source_verify_write_metadata_revision(
                         root, "config.json",
                         "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                     "write stale config revision metadata");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result,
                                               "stale-source-revision") &&
                     source_verify_has_blocker(
                         &result, "inconsistent-source-revision"),
                     "stale provider metadata fails provenance");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root),
                     "recreate manifest promotion refusal fixture");
    YVEX_TEST_ASSERT(source_verify_write_config(
                         root, "not_deepseek_v4",
                         yvex_source_release_identity()->config_architecture),
                     "invalidate config before manifest promotion");
    YVEX_TEST_ASSERT(source_verify_run_mode(root, 1, &result, &err) == YVEX_OK &&
                     !result.verified && !result.manifest_published &&
                     source_verify_has_blocker(&result,
                                               "wrong-source-model-type"),
                     "invalid verifier facts cannot promote manifest");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root),
                     "recreate atomic publication fixture");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_FAIL_SOURCE_PUBLISH_AFTER_WRITE",
                           "1", 1) == 0,
                     "enable manifest publication failure injection");
    YVEX_TEST_ASSERT(source_verify_run_mode(root, 1, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(
                         &result, "source-manifest-publish-failed") &&
                     !result.manifest_published,
                     "atomic manifest publication fails closed");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_FAIL_SOURCE_PUBLISH_AFTER_WRITE") == 0,
                     "disable manifest publication failure injection");
    snprintf(path, sizeof(path), "%s/source-manifest.json.tmp.%ld", root,
             (long)getpid());
    YVEX_TEST_ASSERT(access(path, F_OK) != 0,
                     "failed publication removes temporary output");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root),
                     "recreate missing upstream index fixture");
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", root);
    YVEX_TEST_ASSERT(unlink(path) == 0, "remove required upstream index");
    YVEX_TEST_ASSERT(source_verify_run_identity(
                         root, yvex_source_release_identity(), NULL,
                         NULL, 0, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result,
                                               "missing-shard-index"),
                     "upstream index claim requires the exact local file");

    {
        yvex_source_target_identity indexless =
            *yvex_source_release_identity();
        char upstream_path[512];
        char derived_path[512];
        char first_path[512];
        char second_path[512];
        char old_path[512];
        char index_metadata_path[768];
        struct stat first_stat;
        struct stat second_stat;

        indexless.upstream_index_path = NULL;
        indexless.upstream_index_oid = "not-applicable";
        indexless.upstream_index_size = 0u;
        indexless.upstream_inventory_authority = "header-derived";
        snprintf(upstream_path, sizeof(upstream_path), "%s-upstream.json", root);
        snprintf(derived_path, sizeof(derived_path), "%s-derived.json", root);
        snprintf(old_path, sizeof(old_path),
                 "%s/model-00001-of-00001.safetensors", root);
        YVEX_TEST_ASSERT(stat(old_path, &first_stat) == 0 &&
                         source_verify_write_upstream_inventory(
                             upstream_path,
                             yvex_source_release_identity()->upstream_repo_id,
                             source_verify_revision,
                             "model-00001-of-00001.safetensors",
                             (unsigned long long)first_stat.st_size + 1u),
                         "write drifting upstream snapshot");
        YVEX_TEST_ASSERT(source_verify_run_identity(
                             root, &indexless, upstream_path, derived_path, 0,
                             &result, &err) == YVEX_OK &&
                         source_verify_has_blocker(
                             &result, "upstream-local-inventory-drift"),
                         "upstream and local file metadata drift fails closed");

        snprintf(first_path, sizeof(first_path),
                 "%s/model-00001-of-00002.safetensors", root);
        snprintf(second_path, sizeof(second_path),
                 "%s/model-00002-of-00002.safetensors", root);
        YVEX_TEST_ASSERT(unlink(old_path) == 0 &&
                         source_verify_write_safetensors(first_path) &&
                         source_verify_write_safetensors(second_path) &&
                         source_verify_write_metadata(
                             root, "model-00001-of-00002.safetensors") &&
                         source_verify_write_metadata(
                             root, "model-00002-of-00002.safetensors") &&
                         stat(first_path, &first_stat) == 0 &&
                         stat(second_path, &second_stat) == 0,
                         "create duplicate tensor headers across two shards");
        snprintf(index_metadata_path, sizeof(index_metadata_path),
                 "%s/.cache/huggingface/download/model-00001-of-00001.safetensors.metadata",
                 root);
        unlink(index_metadata_path);
        YVEX_TEST_ASSERT(source_verify_write_upstream_inventory_two(
                             upstream_path,
                             yvex_source_release_identity()->upstream_repo_id,
                             source_verify_revision,
                             "model-00001-of-00002.safetensors",
                             (unsigned long long)first_stat.st_size,
                             "model-00002-of-00002.safetensors",
                             (unsigned long long)second_stat.st_size),
                         "write two-shard upstream snapshot");
        YVEX_TEST_ASSERT(source_verify_run_identity(
                             root, &indexless, upstream_path, derived_path, 0,
                             &result, &err) == YVEX_OK &&
                         source_verify_has_blocker(
                             &result, "duplicate-header-tensor"),
                         "duplicate tensor names across headers fail closed");
    }

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root), "recreate index fixture");
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", root);
    YVEX_TEST_ASSERT(source_verify_write_text(path, "{"),
                     "write malformed index");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result, "malformed-shard-index"),
                     "malformed shard index is refused");
    YVEX_TEST_ASSERT(source_verify_write_text(
                         path,
                         "{\"weight_map\":{\"model.embed_tokens.weight\":"
                         "\"model-00001-of-00001.safetensors\","
                         "\"model.embed_tokens.weight\":"
                         "\"model-00001-of-00001.safetensors\"}}"),
                     "write duplicate tensor index");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result,
                                               "duplicate-index-tensor"),
                     "duplicate index tensor is refused explicitly");
    YVEX_TEST_ASSERT(source_verify_write_text(
                         path,
                         "{\"metadata\":{\"total_size\":9},\"weight_map\":{"
                         "\"model.embed_tokens.weight\":"
                         "\"model-00001-of-00001.safetensors\"}}"),
                     "write inconsistent index size");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result,
                                               "shard-index-size-mismatch"),
                     "index total size must match header tensor spans");
    YVEX_TEST_ASSERT(source_verify_write_text(
                         path,
                         "{\"weight_map\":{\"model.embed_tokens.weight\":"
                         "\"model-00002-of-00002.safetensors\"}}"),
                     "write missing referenced shard index");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result, "missing-referenced-shard"),
                     "missing referenced shard is refused");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root), "recreate unexpected shard fixture");
    snprintf(path, sizeof(path), "%s/weights.safetensors", root);
    YVEX_TEST_ASSERT(source_verify_write_safetensors(path),
                     "write unexpected shard");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result, "unexpected-shard"),
                     "unexpected shard is refused");
    snprintf(path, sizeof(path), "%s/model-00001-of-00002.safetensors", root);
    YVEX_TEST_ASSERT(source_verify_write_safetensors(path),
                     "write inconsistent shard series");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result,
                                               "inconsistent-shard-set") &&
                     source_verify_has_blocker(&result,
                                               "duplicate-source-shard"),
                     "inconsistent and duplicate shard numbering is refused");

    system("rm -rf build/tests/source-verify");
    YVEX_TEST_ASSERT(source_verify_make_valid(root), "recreate invalid header fixture");
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", root);
    YVEX_TEST_ASSERT(source_verify_write_text(path, "bad"),
                     "write invalid safetensors header");
    YVEX_TEST_ASSERT(source_verify_run(root, &result, &err) == YVEX_OK &&
                     source_verify_has_blocker(&result, "invalid-safetensors-header"),
                     "invalid safetensors header is refused");

    total = ULLONG_MAX - 1u;
    YVEX_TEST_ASSERT(!yvex_core_u64_add(total, 2u, &total) &&
                     total == ULLONG_MAX - 1u,
                     "footprint overflow fails without mutation");
    YVEX_TEST_ASSERT(yvex_core_u64_add(total, 1u, &total) &&
                     total == ULLONG_MAX,
                     "checked footprint addition accepts exact limit");
    return 0;
}
