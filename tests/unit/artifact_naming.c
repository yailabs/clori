
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int suggest(char *out, size_t n,
                   const char *family,
                   const char *model,
                   const char *qprofile,
                   yvex_error *err)
{
    return yvex_artifact_name_suggest(out,
                                      n,
                                      family,
                                      model,
                                      "selected",
                                      "embed",
                                      qprofile,
                                      "noimatrix",
                                      "yvex",
                                      "v1",
                                      err);
}

static int test_qwen_name(void)
{
    char out[128];
    yvex_error err;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(suggest(out, sizeof(out), "qwen3", "8b", "F16", &err) == YVEX_OK,
                     "qwen suggest");
    YVEX_TEST_ASSERT_STREQ(out,
                           "qwen3-8b-selected-embed-F16-noimatrix-yvex-v1.gguf",
                           "qwen name");
    return 0;
}

static int test_deepseek_name(void)
{
    char out[128];
    yvex_error err;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_conversion_suggest_artifact_name(out,
                                                           sizeof(out),
                                                           "deepseek4",
                                                           "v4-flash",
                                                           "selected",
                                                           "embed",
                                                           "F16",
                                                           "noimatrix",
                                                           "yvex",
                                                           "v1",
                                                           &err) == YVEX_OK,
                     "deepseek conversion suggest");
    YVEX_TEST_ASSERT_STREQ(out,
                           "deepseek4-v4-flash-selected-embed-F16-noimatrix-yvex-v1.gguf",
                           "deepseek name");
    return 0;
}

static int test_rejects_missing(void)
{
    char out[128];
    yvex_error err;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(suggest(out, sizeof(out), NULL, "8b", "F16", &err) == YVEX_ERR_INVALID_ARG,
                     "missing family rejected");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(suggest(out, sizeof(out), "qwen3", NULL, "F16", &err) == YVEX_ERR_INVALID_ARG,
                     "missing model rejected");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(suggest(out, sizeof(out), "qwen3", "8b", NULL, &err) == YVEX_ERR_INVALID_ARG,
                     "missing qprofile rejected");
    return 0;
}

static int test_rejects_policy_violations(void)
{
    char out[128];
    yvex_error err;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_artifact_name_suggest(out, sizeof(out), "qwen3", "8b",
                                                "selected", "embed", "F16",
                                                "noimatrix", "external", "v1",
                                                &err) == YVEX_ERR_INVALID_ARG,
                     "non-yvex producer rejected");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_artifact_name_suggest(out, sizeof(out), "qwen3", "8b",
                                                "selected", "embed", "F16",
                                                "noimatrix", "yvex", "v3",
                                                &err) == YVEX_ERR_INVALID_ARG,
                     "bad schema rejected");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_artifact_name_suggest(out, sizeof(out), "qwen3", "latest",
                                                "selected", "embed", "F16",
                                                "noimatrix", "yvex", "v1",
                                                &err) == YVEX_ERR_INVALID_ARG,
                     "ambiguous word rejected");
    return 0;
}

int yvex_test_artifact_naming(void)
{
    if (test_qwen_name() != 0) return 1;
    if (test_deepseek_name() != 0) return 1;
    if (test_rejects_missing() != 0) return 1;
    if (test_rejects_policy_violations() != 0) return 1;
    return 0;
}
