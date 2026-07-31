
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int test_valid_template_fixture(void)
{
    yvex_gguf_template *tmpl = NULL;
    yvex_gguf_template_summary summary;
    yvex_error err;
    yvex_gguf_template_options options;
    int rc;

    memset(&options, 0, sizeof(options));
    options.template_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    yvex_error_clear(&err);
    rc = yvex_gguf_template_open(&tmpl, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open valid tokenizer template");
    YVEX_TEST_ASSERT(yvex_gguf_template_get_summary(tmpl, &summary, &err) == YVEX_OK, "summary");
    YVEX_TEST_ASSERT(summary.has_architecture, "summary has architecture");
    YVEX_TEST_ASSERT(summary.model_name && summary.model_name[0] != '\0', "summary has model name");
    YVEX_TEST_ASSERT(summary.tensor_count == 1, "summary tensor_count=1");
    YVEX_TEST_ASSERT(summary.has_tokenizer, "summary has tokenizer");
    YVEX_TEST_ASSERT(summary.known_role_count > 0, "known_roles > 0");
    YVEX_TEST_ASSERT(summary.status == YVEX_GGUF_TEMPLATE_STATUS_VALID ||
                     summary.status == YVEX_GGUF_TEMPLATE_STATUS_PARTIAL,
                     "status valid or partial");
    yvex_gguf_template_close(tmpl);
    return 0;
}

static int test_missing_template_fails(void)
{
    yvex_gguf_template *tmpl = NULL;
    yvex_error err;
    yvex_gguf_template_options options;
    int rc;

    memset(&options, 0, sizeof(options));
    options.template_path = "tests/fixtures/gguf/missing-template.gguf";
    yvex_error_clear(&err);
    rc = yvex_gguf_template_open(&tmpl, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing template file fails");
    yvex_gguf_template_close(tmpl);
    return 0;
}

static int test_malformed_template_fails(void)
{
    yvex_gguf_template *tmpl = NULL;
    yvex_error err;
    yvex_gguf_template_options options;
    int rc;

    memset(&options, 0, sizeof(options));
    options.template_path = "tests/fixtures/gguf/bad-magic.gguf";
    yvex_error_clear(&err);
    rc = yvex_gguf_template_open(&tmpl, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "malformed GGUF fails");
    yvex_gguf_template_close(tmpl);
    return 0;
}

int yvex_test_gguf_template(void)
{
    if (test_valid_template_fixture() != 0) return 1;
    if (test_missing_template_fails() != 0) return 1;
    if (test_malformed_template_fails() != 0) return 1;
    return 0;
}
