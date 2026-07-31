/*
 * YVEX - Local model registry tests
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int write_file(const char *path, const char *text)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fputs(text, fp);
    return fclose(fp) == 0;
}

static int file_contains(const char *path, const char *needle)
{
    FILE *fp = fopen(path, "rb");
    char *bytes;
    long extent;
    int found;
    if (!fp || fseek(fp, 0, SEEK_END) != 0 || (extent = ftell(fp)) < 0 ||
        fseek(fp, 0, SEEK_SET) != 0) {
        if (fp) (void)fclose(fp);
        return 0;
    }
    bytes = malloc((size_t)extent + 1u);
    if (!bytes) {
        (void)fclose(fp);
        return 0;
    }
    if (fread(bytes, 1u, (size_t)extent, fp) != (size_t)extent) {
        free(bytes);
        (void)fclose(fp);
        return 0;
    }
    bytes[extent] = '\0';
    found = strstr(bytes, needle) != NULL;
    free(bytes);
    return fclose(fp) == 0 && found;
}

static int test_alias_validation(void)
{
    yvex_error err;
    yvex_error_clear(&err);

    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-selected-embed", &err) == YVEX_OK,
                     "valid DeepSeek alias");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("qwen3-8b-selected-embed", &err) == YVEX_OK,
                     "valid Qwen alias");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("llama-7b-full-model", &err) == YVEX_OK,
                     "valid full alias");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("DeepSeek4-v4-flash-selected-embed", &err) != YVEX_OK,
                     "uppercase rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4 selected embed", &err) != YVEX_OK,
                     "spaces rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4/v4-flash", &err) != YVEX_OK,
                     "path slash rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("../model", &err) != YVEX_OK,
                     "path traversal rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("latest", &err) != YVEX_OK,
                     "latest rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-final-embed", &err) != YVEX_OK,
                     "final segment rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-new-embed", &err) != YVEX_OK,
                     "new segment rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-test-embed", &err) != YVEX_OK,
                     "test segment rejected");
    return 0;
}

static int test_derive_metadata(void)
{
    yvex_model_registry_entry entry;
    yvex_error err;
    const char *path = "build/tests/model-registry/deepseek4-v4-flash-selected-embed-F16-noimatrix-yvex-v1.gguf";

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_registry_entry_derive_from_path(&entry, path, &err) == YVEX_OK,
                     "derive canonical filename");
    YVEX_TEST_ASSERT_STREQ(entry.alias, "deepseek4-v4-flash-selected-embed", "derived alias");
    YVEX_TEST_ASSERT_STREQ(entry.family, "deepseek4", "derived family");
    YVEX_TEST_ASSERT_STREQ(entry.model, "v4-flash", "derived model");
    YVEX_TEST_ASSERT_STREQ(entry.scope, "selected", "derived scope");
    YVEX_TEST_ASSERT_STREQ(entry.artifact_class, "embed", "derived class");
    YVEX_TEST_ASSERT_STREQ(entry.qprofile, "F16", "derived qprofile");
    YVEX_TEST_ASSERT_STREQ(entry.calibration, "noimatrix", "derived calibration");
    YVEX_TEST_ASSERT_STREQ(entry.producer, "yvex", "derived producer");
    YVEX_TEST_ASSERT_STREQ(entry.schema_version, "v1", "derived schema");
    return 0;
}

static void fill_entry(yvex_model_registry_entry *entry, const char *path)
{
    memset(entry, 0, sizeof(*entry));
    entry->alias = "deepseek4-v4-flash-selected-embed";
    entry->family = "deepseek4";
    entry->model = "v4-flash";
    entry->scope = "selected";
    entry->artifact_class = "embed";
    entry->qprofile = "F16";
    entry->calibration = "noimatrix";
    entry->producer = "yvex";
    entry->schema_version = "v1";
    entry->path = path;
    entry->sha256 = "abc123";
    entry->file_size = 42ull;
    entry->format = "gguf";
    entry->architecture = "deepseek";
    entry->tensor_count = 1ull;
    entry->known_tensor_bytes = 64ull;
    entry->primary_tensor_name = "token_embd.weight";
    entry->primary_tensor_role = "token_embedding";
    entry->primary_tensor_dtype = "F16";
    entry->primary_tensor_rank = 2u;
    entry->primary_tensor_dims = "[4,8]";
    entry->primary_tensor_bytes = 64ull;
    entry->support_level = "selected-tensor-materialized";
    entry->selected_embedding_ready = 1;
    entry->selected_embedding_hidden_size = 4ull;
    entry->selected_embedding_vocab_size = 8ull;
    entry->selected_embedding_output_count = 4ull;
    entry->selected_embedding_slice_bytes = 8ull;
    entry->execution_ready = 0;
}

static int test_registry_lifecycle(void)
{
    const char *dir = "build/tests/model-registry";
    const char *registry_path = "build/tests/model-registry/models.local.json";
    const char *model_path = "build/tests/model-registry/deepseek4-v4-flash-selected-embed-F16-noimatrix-yvex-v1.gguf";
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    yvex_model_registry_entry entry;
    const yvex_model_registry_entry *found;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/model-registry && mkdir -p build/tests/model-registry") == 0,
                     "prepare model registry dir");
    YVEX_TEST_ASSERT(write_file(model_path, "not a real gguf for registry unit test\n"),
                     "write model path");

    memset(&options, 0, sizeof(options));
    options.registry_path = registry_path;
    options.create_if_missing = 1;
    yvex_error_clear(&err);
    rc = yvex_model_registry_open(&registry, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open missing registry with create");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 0, "initial count");

    fill_entry(&entry, model_path);
    rc = yvex_model_registry_add(registry, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "add entry");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 1, "count after add");
    found = yvex_model_registry_find(registry, "deepseek4-v4-flash-selected-embed");
    YVEX_TEST_ASSERT(found != NULL, "find entry");
    YVEX_TEST_ASSERT_STREQ(found->path, model_path, "found path");

    rc = yvex_model_registry_save(registry, registry_path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "save registry");
    YVEX_TEST_ASSERT(file_contains(registry_path, "\"schema\": \"yvex.models.local.v2\""),
                     "registry writer publishes schema v2");
    YVEX_TEST_ASSERT(!file_contains(registry_path, "\"selected\":"),
                     "registry writer has no selected startup state");
    yvex_model_registry_close(registry);
    registry = NULL;

    options.create_if_missing = 0;
    rc = yvex_model_registry_open(&registry, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "reload registry");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 1, "count after reload");
    found = yvex_model_registry_find(registry, "deepseek4-v4-flash-selected-embed");
    YVEX_TEST_ASSERT(found != NULL, "entry after reload");
    YVEX_TEST_ASSERT_STREQ(found->support_level, "selected-tensor-materialized", "support after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_name, "token_embd.weight", "primary tensor after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_role, "token_embedding", "primary role after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_dtype, "F16", "primary dtype after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_dims, "[4,8]", "primary dims after reload");
    YVEX_TEST_ASSERT(found->selected_embedding_ready == 1, "selected embedding readiness after reload");

    rc = yvex_model_registry_remove(registry, "deepseek4-v4-flash-selected-embed", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "remove entry");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 0, "count after remove");
    yvex_model_registry_close(registry);
    (void)dir;
    return 0;
}

static int test_invalid_args(void)
{
    yvex_model_registry_entry entry;
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    yvex_error err;

    memset(&options, 0, sizeof(options));
    options.registry_path = "build/tests/model-registry/missing.json";
    options.create_if_missing = 0;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_registry_open(&registry, &options, &err) != YVEX_OK,
                     "open missing without create fails");
    YVEX_TEST_ASSERT(yvex_model_registry_add(NULL, NULL, &err) != YVEX_OK,
                     "add invalid args fails");
    memset(&entry, 0, sizeof(entry));
    YVEX_TEST_ASSERT(yvex_model_registry_entry_derive_from_path(&entry, "some-model.gguf", &err) != YVEX_OK,
                     "unknown filename derive fails");
    return 0;
}

int yvex_test_model_registry(void)
{
    if (test_alias_validation() != 0) return 1;
    if (test_derive_metadata() != 0) return 1;
    if (test_registry_lifecycle() != 0) return 1;
    if (test_invalid_args() != 0) return 1;
    return 0;
}
