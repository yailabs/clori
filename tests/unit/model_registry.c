
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static int write_legacy_registry_copy(const char *source, const char *destination,
                                      const char *legacy_schema,
                                      const char *legacy_mode)
{
    FILE *fp;
    char *bytes, *output, *schema, *kind, *strategy_end;
    long extent;
    size_t size, prefix, suffix, replacement_size;
    int replacement_length;
    char replacement[96];

    fp = fopen(source, "rb");
    if (!fp || fseek(fp, 0, SEEK_END) != 0 || (extent = ftell(fp)) < 0 ||
        fseek(fp, 0, SEEK_SET) != 0) {
        if (fp) (void)fclose(fp);
        return 0;
    }
    size = (size_t)extent;
    bytes = malloc(size + 1u);
    if (!bytes) {
        (void)fclose(fp);
        return 0;
    }
    if (fread(bytes, 1u, size, fp) != size) {
        (void)fclose(fp);
        free(bytes);
        return 0;
    }
    if (fclose(fp) != 0) {
        free(bytes);
        return 0;
    }
    bytes[size] = '\0';
    schema = strstr(bytes, YVEX_MODEL_REGISTRY_SCHEMA_CURRENT);
    kind = strstr(bytes, "      \"runtime_engine_kind\":");
    strategy_end = kind
        ? strstr(kind, "      \"runtime_execution_strategy\":") : NULL;
    strategy_end = strategy_end ? strchr(strategy_end, '\n') : NULL;
    if (!schema || !kind || !strategy_end ||
        strlen(legacy_schema) != strlen(YVEX_MODEL_REGISTRY_SCHEMA_CURRENT)) {
        free(bytes);
        return 0;
    }
    memcpy(schema, legacy_schema, strlen(legacy_schema));
    strategy_end++;
    replacement_length = legacy_mode
        ? snprintf(replacement, sizeof(replacement),
                   "      \"runtime_mode\": \"%s\",\n", legacy_mode)
        : 0;
    if (replacement_length < 0 || (size_t)replacement_length >= sizeof(replacement)) {
        free(bytes);
        return 0;
    }
    replacement_size = (size_t)replacement_length;
    prefix = (size_t)(kind - bytes);
    suffix = size - (size_t)(strategy_end - bytes);
    output = malloc(prefix + replacement_size + suffix + 1u);
    if (!output) {
        free(bytes);
        return 0;
    }
    memcpy(output, bytes, prefix);
    if (replacement_size)
        memcpy(output + prefix, replacement, replacement_size);
    memcpy(output + prefix + replacement_size, strategy_end, suffix);
    size = prefix + replacement_size + suffix;
    output[size] = '\0';
    fp = fopen(destination, "wb");
    if (!fp) {
        free(output);
        free(bytes);
        return 0;
    }
    if (fwrite(output, 1u, size, fp) != size) {
        (void)fclose(fp);
        free(output);
        free(bytes);
        return 0;
    }
    if (fclose(fp) != 0) {
        free(output);
        free(bytes);
        return 0;
    }
    free(output);
    free(bytes);
    return 1;
}

static int test_alias_validation(void)
{
    yvex_error err;
    yvex_error_clear(&err);

    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-dspark-selected-embed", &err) == YVEX_OK,
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
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-dspark-final-embed", &err) != YVEX_OK,
                     "final segment rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-dspark-new-embed", &err) != YVEX_OK,
                     "new segment rejected");
    YVEX_TEST_ASSERT(yvex_model_alias_validate("deepseek4-v4-flash-dspark-test-embed", &err) != YVEX_OK,
                     "test segment rejected");
    return 0;
}

static int test_derive_metadata(void)
{
    yvex_model_registry_entry entry;
    yvex_error err;
    const char *path = "build/tests/model-registry/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf";

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_registry_entry_derive_from_path(&entry, path, &err) == YVEX_OK,
                     "derive canonical filename");
    YVEX_TEST_ASSERT_STREQ(entry.alias, "deepseek4-v4-flash-dspark-selected-embed", "derived alias");
    YVEX_TEST_ASSERT_STREQ(entry.family, "deepseek4", "derived family");
    YVEX_TEST_ASSERT_STREQ(entry.model, "v4-flash-dspark", "derived model");
    YVEX_TEST_ASSERT_STREQ(entry.scope, "selected", "derived scope");
    YVEX_TEST_ASSERT_STREQ(entry.artifact_class, "embed", "derived class");
    YVEX_TEST_ASSERT_STREQ(entry.qprofile, "F16", "derived qprofile");
    YVEX_TEST_ASSERT_STREQ(entry.calibration, "noimatrix", "derived calibration");
    YVEX_TEST_ASSERT_STREQ(entry.producer, "yvex", "derived producer");
    YVEX_TEST_ASSERT(entry.schema_version == YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT,
                     "derived entry schema");
    YVEX_TEST_ASSERT_STREQ(entry.artifact_schema, "v1", "derived artifact schema");
    return 0;
}

static void fill_entry(yvex_model_registry_entry *entry, const char *path,
                       const char *binding)
{
    memset(entry, 0, sizeof(*entry));
    entry->schema_version = YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT;
    entry->alias = "deepseek4-v4-flash-dspark-selected-embed";
    entry->family = "deepseek4";
    entry->model = "v4-flash-dspark";
    entry->scope = "selected";
    entry->artifact_class = "embed";
    entry->qprofile = "F16";
    entry->calibration = "noimatrix";
    entry->producer = "yvex";
    entry->artifact_schema = "v1";
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
    entry->runtime_profile = "single-artifact";
    entry->runtime_installation = "";
    entry->runtime_binding = binding;
    entry->runtime_target = "deepseek4-v4-flash-dspark";
    entry->runtime_backend = "cuda";
    entry->runtime_engine_kind = "text";
    entry->runtime_execution_strategy = "speculative";
    entry->runtime_context = 4096ull;
}

static int test_registry_lifecycle(void)
{
    const char *dir = "build/tests/model-registry";
    const char *registry_path = "build/tests/model-registry/models.local.json";
    const char *model_path = "build/tests/model-registry/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf";
    const char *binding_path = "build/tests/model-registry/runtime.binding";
    char absolute_model[YVEX_PATH_CAP];
    char absolute_binding[YVEX_PATH_CAP];
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
    YVEX_TEST_ASSERT(write_file(binding_path, "runtime binding fixture\n"),
                     "write binding path");
    YVEX_TEST_ASSERT(realpath(model_path, absolute_model) != NULL,
                     "resolve absolute model path");
    YVEX_TEST_ASSERT(realpath(binding_path, absolute_binding) != NULL,
                     "resolve absolute binding path");

    memset(&options, 0, sizeof(options));
    options.registry_path = registry_path;
    options.create_if_missing = 1;
    yvex_error_clear(&err);
    rc = yvex_model_registry_open(&registry, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open missing registry with create");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 0, "initial count");

    fill_entry(&entry, absolute_model, absolute_binding);
    entry.schema_version = 0u;
    YVEX_TEST_ASSERT(yvex_model_registry_startup_validate(&entry, &err) ==
                         YVEX_ERR_INVALID_ARG &&
                         strstr(yvex_error_message(&err), "schema") != NULL,
                     "startup validation rejects a stale entry schema first");
    YVEX_TEST_ASSERT(yvex_model_registry_add(registry, &entry, &err) ==
                         YVEX_ERR_INVALID_ARG &&
                         strstr(yvex_error_message(&err), "schema") != NULL,
                     "registry add rejects a stale entry schema first");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 0,
                     "stale entry schema leaves registry unchanged");
    fill_entry(&entry, absolute_model, absolute_binding);
    YVEX_TEST_ASSERT(yvex_model_registry_startup_validate(&entry, &err) == YVEX_OK,
                     "complete startup profile validates");
    entry.support_level = "runtime-profile-configured";
    YVEX_TEST_ASSERT(yvex_model_registry_add(registry, &entry, &err) == YVEX_ERR_INVALID_ARG,
                     "startup profile cannot masquerade as artifact support");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 0,
                     "invalid support level leaves registry unchanged");
    fill_entry(&entry, absolute_model, absolute_binding);
    rc = yvex_model_registry_add(registry, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "add entry");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 1, "count after add");
    found = yvex_model_registry_find(registry, "deepseek4-v4-flash-dspark-selected-embed");
    YVEX_TEST_ASSERT(found != NULL, "find entry");
    YVEX_TEST_ASSERT_STREQ(found->path, absolute_model, "found path");

    rc = yvex_model_registry_save(registry, registry_path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "save registry");
    YVEX_TEST_ASSERT(file_contains(registry_path, "\"schema\": \"yvex.models.local.v6\""),
                     "registry writer publishes schema v6");
    YVEX_TEST_ASSERT(file_contains(registry_path, "\"runtime_backend\": \"cuda\""),
                     "registry writer persists runtime profile");
    YVEX_TEST_ASSERT(file_contains(registry_path,
                                   "\"runtime_engine_kind\": \"text\""),
                     "registry writer persists engine kind");
    YVEX_TEST_ASSERT(file_contains(
                         registry_path,
                         "\"runtime_execution_strategy\": \"speculative\""),
                     "registry writer persists semantic execution strategy");
    YVEX_TEST_ASSERT(!file_contains(registry_path, "\"runtime_mode\":"),
                     "registry writer removes the mixed legacy mode axis");
    YVEX_TEST_ASSERT(!file_contains(registry_path, "\"selected\":"),
                     "registry writer has no selected startup state");
    yvex_model_registry_close(registry);
    registry = NULL;

    options.create_if_missing = 0;
    rc = yvex_model_registry_open(&registry, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "reload registry");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 1, "count after reload");
    found = yvex_model_registry_find(registry, "deepseek4-v4-flash-dspark-selected-embed");
    YVEX_TEST_ASSERT(found != NULL, "entry after reload");
    YVEX_TEST_ASSERT_STREQ(found->support_level, "selected-tensor-materialized", "support after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_name, "token_embd.weight", "primary tensor after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_role, "token_embedding", "primary role after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_dtype, "F16", "primary dtype after reload");
    YVEX_TEST_ASSERT_STREQ(found->primary_tensor_dims, "[4,8]", "primary dims after reload");
    YVEX_TEST_ASSERT(found->selected_embedding_ready == 1, "selected embedding readiness after reload");
    YVEX_TEST_ASSERT_STREQ(found->runtime_binding, absolute_binding,
                           "runtime binding after reload");
    YVEX_TEST_ASSERT_STREQ(found->runtime_profile, "single-artifact",
                           "single-artifact profile after reload");
    YVEX_TEST_ASSERT_STREQ(found->runtime_target, "deepseek4-v4-flash-dspark",
                           "runtime target after reload");
    YVEX_TEST_ASSERT_STREQ(found->runtime_backend, "cuda",
                           "runtime backend after reload");
    YVEX_TEST_ASSERT_STREQ(found->runtime_engine_kind, "text",
                           "engine kind after reload");
    YVEX_TEST_ASSERT_STREQ(found->runtime_execution_strategy, "speculative",
                           "execution strategy after reload");
    YVEX_TEST_ASSERT(found->runtime_context == 4096ull,
                     "runtime context after reload");
    YVEX_TEST_ASSERT(yvex_model_registry_startup_validate(found, &err) == YVEX_OK,
                     "reloaded startup profile validates");

    rc = yvex_model_registry_remove(registry, "deepseek4-v4-flash-dspark-selected-embed", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "remove entry");
    YVEX_TEST_ASSERT(yvex_model_registry_count(registry) == 0, "count after remove");
    yvex_model_registry_close(registry);
    (void)dir;
    return 0;
}

static int test_composite_profile(void)
{
    const char *registry_path = "build/tests/model-registry/composite.local.json";
    const char *model_path =
        "build/tests/model-registry/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf";
    yvex_model_registry_options options = {0};
    yvex_model_registry *registry = NULL;
    yvex_model_registry_entry entry;
    const yvex_model_registry_entry *found;
    char absolute_model[YVEX_PATH_CAP], absolute_root[YVEX_PATH_CAP];
    yvex_error err;

    YVEX_TEST_ASSERT(realpath(model_path, absolute_model) != NULL,
                     "resolve composite reference artifact");
    YVEX_TEST_ASSERT(realpath("build/tests/model-registry", absolute_root) != NULL,
                     "resolve composite installation");
    fill_entry(&entry, absolute_model, "");
    entry.alias = "minimax-h3-fl2va-runtime-media";
    entry.family = "minimax-h3";
    entry.runtime_profile = "composite";
    entry.runtime_installation = absolute_root;
    entry.runtime_target = "minimax-h3-fl2va";
    entry.runtime_backend = "cuda";
    entry.runtime_engine_kind = "media";
    entry.runtime_execution_strategy = "not-applicable";
    entry.runtime_context = 0ull;
    YVEX_TEST_ASSERT(yvex_model_registry_startup_validate(&entry, &err) == YVEX_OK,
                     "complete composite startup profile validates");
    entry.runtime_installation = "";
    YVEX_TEST_ASSERT(yvex_model_registry_startup_validate(&entry, &err) == YVEX_ERR_STATE,
                     "incomplete composite startup profile refuses");
    entry.runtime_installation = absolute_root;
    entry.runtime_binding = absolute_model;
    YVEX_TEST_ASSERT(yvex_model_registry_startup_validate(&entry, &err) == YVEX_ERR_STATE,
                     "composite startup refuses a fake runtime binding");
    entry.runtime_binding = "";

    options.registry_path = registry_path;
    options.create_if_missing = 1;
    YVEX_TEST_ASSERT(yvex_model_registry_open(&registry, &options, &err) == YVEX_OK &&
                         yvex_model_registry_add(registry, &entry, &err) == YVEX_OK &&
                         yvex_model_registry_save(registry, registry_path, &err) == YVEX_OK,
                     "persist composite startup profile");
    yvex_model_registry_close(registry);
    registry = NULL;
    options.create_if_missing = 0;
    YVEX_TEST_ASSERT(yvex_model_registry_open(&registry, &options, &err) == YVEX_OK,
                     "reload composite startup profile");
    found = yvex_model_registry_find(registry, entry.alias);
    YVEX_TEST_ASSERT(found && !strcmp(found->runtime_profile, "composite") &&
                         !strcmp(found->runtime_installation, absolute_root) &&
                         !found->runtime_binding[0] && found->runtime_context == 0ull &&
                         yvex_model_registry_startup_validate(found, &err) == YVEX_OK,
                     "composite profile roundtrip preserves its deployment contract");
    yvex_model_registry_close(registry);
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
    YVEX_TEST_ASSERT(yvex_model_registry_startup_validate(&entry, &err) != YVEX_OK,
                     "incomplete startup profile fails");
    return 0;
}

static int test_legacy_startup_axes(void)
{
    const char *current = "build/tests/model-registry/models.local.json";
    const char *legacy_v5 = "build/tests/model-registry/models.local.v5.json";
    const char *legacy = "build/tests/model-registry/models.local.v3.json";
    yvex_model_registry_options options = {0};
    yvex_model_registry *registry = NULL;
    const yvex_model_registry_entry *entry;
    yvex_error err;

    YVEX_TEST_ASSERT(write_legacy_registry_copy(
                         current, legacy_v5, YVEX_MODEL_REGISTRY_SCHEMA_V5,
                         "dspark"),
                     "construct a prior registry schema v5 fixture");
    options.registry_path = legacy_v5;
    YVEX_TEST_ASSERT(yvex_model_registry_open(&registry, &options, &err) == YVEX_OK,
                     "registry schema v5 remains readable");
    entry = yvex_model_registry_find(
        registry, "deepseek4-v4-flash-dspark-selected-embed");
    YVEX_TEST_ASSERT(entry != NULL, "schema v5 retains its model entry");
    YVEX_TEST_ASSERT_STREQ(entry->runtime_engine_kind, "text",
                           "schema v5 maps DSpark to a text engine");
    YVEX_TEST_ASSERT_STREQ(entry->runtime_execution_strategy, "speculative",
                           "schema v5 maps DSpark to semantic speculation");
    yvex_model_registry_close(registry);
    registry = NULL;

    YVEX_TEST_ASSERT(write_legacy_registry_copy(
                         current, legacy, "yvex.models.local.v3", NULL),
                     "construct a prior registry schema fixture");
    options.registry_path = legacy;
    YVEX_TEST_ASSERT(yvex_model_registry_open(&registry, &options, &err) == YVEX_OK,
                     "registry schema v3 remains readable");
    entry = yvex_model_registry_find(
        registry, "deepseek4-v4-flash-dspark-selected-embed");
    YVEX_TEST_ASSERT(entry != NULL, "schema v3 retains its model entry");
    YVEX_TEST_ASSERT_STREQ(entry->runtime_engine_kind, "text",
                           "schema v3 acquires the safe text engine kind");
    YVEX_TEST_ASSERT_STREQ(entry->runtime_execution_strategy, "target-only",
                           "schema v3 acquires the safe target-only strategy");
    yvex_model_registry_close(registry);
    return 0;
}

static int test_logical_model_library(void)
{
    const char *registry_path = "build/tests/model-registry/library.local.json";
    const char *model_path =
        "build/tests/model-registry/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf";
    const char *binding_path = "build/tests/model-registry/runtime.binding";
    static const char *const identities[] = {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    };
    yvex_model_registry_options registry_options = {0};
    yvex_local_catalog_options library_options = {0};
    yvex_model_registry *registry = NULL;
    yvex_model_library *library = NULL;
    yvex_model_registry_entry entry;
    const yvex_model_library_entry *logical;
    const yvex_model_runtime_profile_fact *profile;
    char aliases[8][64];
    char absolute_model[YVEX_PATH_CAP], absolute_binding[YVEX_PATH_CAP];
    yvex_error err;
    size_t index;

    YVEX_TEST_ASSERT(realpath(model_path, absolute_model) != NULL &&
                         realpath(binding_path, absolute_binding) != NULL,
                     "resolve logical-library fixture paths");
    (void)unlink(registry_path);
    YVEX_TEST_ASSERT(system("mkdir -p build/tests/model-library-root") == 0,
                     "prepare isolated logical-library source root");
    registry_options.registry_path = registry_path;
    registry_options.create_if_missing = 1;
    YVEX_TEST_ASSERT(yvex_model_registry_open(&registry, &registry_options, &err) == YVEX_OK,
                     "open logical-library registry");
    for (index = 0u; index < 8u; ++index) {
        fill_entry(&entry, absolute_model, absolute_binding);
        (void)snprintf(aliases[index], sizeof(aliases[index]),
                       "deepseek4-v4-flash-profile-%zu", index);
        entry.alias = aliases[index];
        entry.sha256 = identities[index / 4u];
        YVEX_TEST_ASSERT(yvex_model_registry_add(registry, &entry, &err) == YVEX_OK,
                         "add subordinate runtime profile");
    }
    YVEX_TEST_ASSERT(yvex_model_registry_save(registry, registry_path, &err) == YVEX_OK,
                     "persist logical-library registry");
    yvex_model_registry_close(registry);
    library_options.models_root = "build/tests/model-library-root";
    library_options.registry_path = registry_path;
    YVEX_TEST_ASSERT(yvex_model_library_open(&library, &library_options, &err) == YVEX_OK,
                     "open canonical logical model library");
    logical = yvex_model_library_at(library, 0u);
    YVEX_TEST_ASSERT(yvex_model_library_count(library) == 1u && logical &&
                         logical->profile_count == 8u && logical->artifact_count == 2u &&
                         logical->launchable_profile_count == 8u &&
                         logical->profile_launchable,
                     "eight profiles and two artifacts aggregate under one logical model");
    profile = yvex_model_library_profile_at(library, 0u, 0u);
    YVEX_TEST_ASSERT(profile && profile->launchable,
                     "startup validation admits v5 profiles without the legacy readiness bit");
    YVEX_TEST_ASSERT(!strcmp(logical->family, "deepseek4") &&
                         !strcmp(logical->model, "v4-flash-dspark") &&
                         logical->identity_kind == YVEX_MODEL_IDENTITY_FAMILY_MODEL_TARGET,
                     "logical identity uses exact family model and runtime target facts");
    YVEX_TEST_ASSERT(yvex_model_library_profile_at(library, 0u, 7u) &&
                         !strcmp(yvex_model_library_profile_at(library, 0u, 7u)->alias,
                                 "deepseek4-v4-flash-profile-7"),
                     "subordinate profiles retain their canonical aliases");
    yvex_model_library_close(library);
    return 0;
}

int yvex_test_model_registry(void)
{
    if (test_alias_validation() != 0) return 1;
    if (test_derive_metadata() != 0) return 1;
    if (test_registry_lifecycle() != 0) return 1;
    if (test_composite_profile() != 0) return 1;
    if (test_legacy_startup_axes() != 0) return 1;
    if (test_logical_model_library() != 0) return 1;
    if (test_invalid_args() != 0) return 1;
    return 0;
}
