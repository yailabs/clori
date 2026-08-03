
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

static int write_registry(const char *path, const char *model_path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fprintf(fp,
            "{\n"
            "  \"schema\": \"yvex.models.local.v1\",\n"
            "  \"selected\": \"deepseek4-v4-flash-dspark-selected-embed\",\n"
            "  \"models\": [\n"
            "    {\n"
            "      \"alias\": \"deepseek4-v4-flash-dspark-selected-embed\",\n"
            "      \"family\": \"deepseek4\",\n"
            "      \"model\": \"v4-flash-dspark\",\n"
            "      \"scope\": \"selected\",\n"
            "      \"artifact_class\": \"embed\",\n"
            "      \"qprofile\": \"F16\",\n"
            "      \"calibration\": \"noimatrix\",\n"
            "      \"producer\": \"yvex\",\n"
            "      \"schema_version\": \"v1\",\n"
            "      \"path\": \"%s\",\n"
            "      \"sha256\": \"abc123\",\n"
            "      \"support_level\": \"selected-tensor-materialized\",\n"
            "      \"execution_ready\": false\n"
            "    }\n"
            "  ]\n"
            "}\n",
            model_path);
    return fclose(fp) == 0;
}

static int test_path_resolution(void)
{
    yvex_model_ref ref;
    yvex_model_ref_options options;
    yvex_error err;
    char cwd[4096];
    char absolute[4096];

    YVEX_TEST_ASSERT(system("rm -rf build/tests/model-ref && mkdir -p build/tests/model-ref") == 0,
                     "prepare model-ref dir");
    YVEX_TEST_ASSERT(write_file("build/tests/model-ref/model.gguf", "fake\n"), "write relative model");

    memset(&options, 0, sizeof(options));
    options.allow_registry = 0;
    yvex_error_clear(&err);
    memset(&ref, 0, sizeof(ref));
    YVEX_TEST_ASSERT(yvex_model_ref_resolve(&ref, "build/tests/model-ref/model.gguf", &options, &err) == YVEX_OK,
                     "resolve relative path");
    YVEX_TEST_ASSERT(ref.kind == YVEX_MODEL_REF_PATH, "relative path kind");
    YVEX_TEST_ASSERT_STREQ(ref.path, "build/tests/model-ref/model.gguf", "relative path value");
    yvex_model_ref_clear(&ref);

    YVEX_TEST_ASSERT(getcwd(cwd, sizeof(cwd)) != NULL, "getcwd");
    YVEX_TEST_ASSERT(strlen(cwd) + 1u + strlen("build/tests/model-ref/model.gguf") < sizeof(absolute),
                     "absolute path fits");
    strcpy(absolute, cwd);
    strcat(absolute, "/");
    strcat(absolute, "build/tests/model-ref/model.gguf");
    YVEX_TEST_ASSERT(yvex_model_ref_resolve(&ref, absolute, &options, &err) == YVEX_OK,
                     "resolve absolute path");
    YVEX_TEST_ASSERT(ref.kind == YVEX_MODEL_REF_PATH, "absolute path kind");
    YVEX_TEST_ASSERT_STREQ(ref.path, absolute, "absolute path value");
    yvex_model_ref_clear(&ref);

    YVEX_TEST_ASSERT(yvex_model_ref_resolve(&ref, "build/tests/model-ref/missing.gguf", &options, &err) == YVEX_OK,
                     "resolve missing path-like reference");
    YVEX_TEST_ASSERT(ref.kind == YVEX_MODEL_REF_PATH, "missing path-like kind");
    YVEX_TEST_ASSERT_STREQ(ref.path, "build/tests/model-ref/missing.gguf", "missing path-like value");
    yvex_model_ref_clear(&ref);
    return 0;
}

static int test_alias_resolution(void)
{
    yvex_model_ref ref;
    yvex_model_ref_options options;
    yvex_error err;
    const char *model_path = "build/tests/model-ref/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf";
    const char *registry_path = "build/tests/model-ref/models.local.json";

    YVEX_TEST_ASSERT(write_file(model_path, "fake\n"), "write alias model");
    YVEX_TEST_ASSERT(write_registry(registry_path, model_path), "write registry");

    memset(&options, 0, sizeof(options));
    options.registry_path = registry_path;
    options.allow_registry = 1;
    yvex_error_clear(&err);
    memset(&ref, 0, sizeof(ref));
    YVEX_TEST_ASSERT(yvex_model_ref_resolve(&ref, "deepseek4-v4-flash-dspark-selected-embed", &options, &err) == YVEX_OK,
                     "resolve alias");
    YVEX_TEST_ASSERT(ref.kind == YVEX_MODEL_REF_ALIAS, "alias kind");
    YVEX_TEST_ASSERT_STREQ(ref.alias, "deepseek4-v4-flash-dspark-selected-embed", "alias value");
    YVEX_TEST_ASSERT_STREQ(ref.path, model_path, "alias path");
    YVEX_TEST_ASSERT_STREQ(ref.family, "deepseek4", "alias family");
    YVEX_TEST_ASSERT_STREQ(ref.support_level, "selected-tensor-materialized", "alias support");
    YVEX_TEST_ASSERT(ref.execution_ready == 0, "alias execution false");
    yvex_model_ref_clear(&ref);
    return 0;
}

static int test_failures(void)
{
    yvex_model_ref ref;
    yvex_model_ref_options options;
    yvex_error err;

    memset(&options, 0, sizeof(options));
    options.registry_path = "build/tests/model-ref/models.local.json";
    options.allow_registry = 1;

    memset(&ref, 0, sizeof(ref));
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_ref_resolve(&ref, "missing-alias", &options, &err) == YVEX_ERR_INVALID_ARG,
                     "missing alias fails");
    YVEX_TEST_ASSERT(ref.status == YVEX_MODEL_REF_STATUS_NOT_FOUND, "missing alias status");
    yvex_model_ref_clear(&ref);

    YVEX_TEST_ASSERT(write_registry("build/tests/model-ref/missing-path.json",
                                    "build/tests/model-ref/missing.gguf"),
                     "write missing path registry");
    options.registry_path = "build/tests/model-ref/missing-path.json";
    memset(&ref, 0, sizeof(ref));
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_ref_resolve(&ref, "deepseek4-v4-flash-dspark-selected-embed", &options, &err) == YVEX_ERR_IO,
                     "alias path missing fails");
    YVEX_TEST_ASSERT(ref.status == YVEX_MODEL_REF_STATUS_ALIAS_PATH_MISSING, "alias path missing status");
    yvex_model_ref_clear(&ref);

    options.registry_path = "build/tests/model-ref/no-registry.json";
    memset(&ref, 0, sizeof(ref));
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_ref_resolve(&ref, "deepseek4-v4-flash-dspark-selected-embed", &options, &err) == YVEX_ERR_IO,
                     "registry unavailable fails");
    YVEX_TEST_ASSERT(ref.status == YVEX_MODEL_REF_STATUS_REGISTRY_UNAVAILABLE, "registry unavailable status");
    yvex_model_ref_clear(&ref);
    return 0;
}

int yvex_test_model_ref(void)
{
    if (test_path_resolution() != 0) return 1;
    if (test_alias_resolution() != 0) return 1;
    if (test_failures() != 0) return 1;
    return 0;
}
