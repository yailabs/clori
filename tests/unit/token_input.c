/*
 * Exercises explicit token input parsing, bounded token sequence storage, vocabulary bounds
 * validation, and selected-token lookup.
 */
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int expect_parse_error(const char *text, const char *needle)
{
    yvex_token_input input;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    memset(&input, 0, sizeof(input));
    rc = yvex_token_input_parse_explicit(text, &input, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "parse should fail");
    YVEX_TEST_ASSERT(strstr(yvex_error_message(&err), needle) != NULL,
                     "expected parse error code");
    return 0;
}

static int test_explicit_parse(void)
{
    yvex_token_input input;
    yvex_error err;
    unsigned int selected = 0u;

    yvex_error_clear(&err);
    memset(&input, 0, sizeof(input));
    YVEX_TEST_ASSERT(yvex_token_input_parse_explicit("0, 1,2", &input, &err) == YVEX_OK,
                     "parse explicit list");
    YVEX_TEST_ASSERT(input.kind == YVEX_TOKEN_INPUT_EXPLICIT, "explicit kind");
    YVEX_TEST_ASSERT(input.token_count == 3ull, "token count");
    YVEX_TEST_ASSERT(input.tokens[0] == 0u && input.tokens[1] == 1u && input.tokens[2] == 2u,
                     "token values");
    YVEX_TEST_ASSERT(yvex_token_input_validate_bounds(&input, 8ull, &err) == YVEX_OK,
                     "bounds pass");
    YVEX_TEST_ASSERT(input.token_bounds_checked && input.token_bounds_valid, "bounds flags");
    YVEX_TEST_ASSERT(yvex_token_input_select(&input, 1ull, &selected, &err) == YVEX_OK,
                     "select token");
    YVEX_TEST_ASSERT(selected == 1u, "selected token");
    return 0;
}

static int test_parse_failures(void)
{
    YVEX_TEST_ASSERT(expect_parse_error("", "token-list-empty") == 0, "empty list");
    YVEX_TEST_ASSERT(expect_parse_error("1,,2", "token-parse-invalid") == 0, "double comma");
    YVEX_TEST_ASSERT(expect_parse_error("1, ", "token-parse-invalid") == 0, "trailing comma");
    YVEX_TEST_ASSERT(expect_parse_error("abc", "token-parse-invalid") == 0, "non-number");
    YVEX_TEST_ASSERT(expect_parse_error("-1", "token-parse-invalid") == 0, "negative");
    YVEX_TEST_ASSERT(expect_parse_error("184467440737095516160", "token-id-overflow") == 0,
                     "overflow");
    return 0;
}

static int test_bounds_and_select_failures(void)
{
    yvex_token_input input;
    yvex_error err;
    unsigned int selected = 0u;

    yvex_error_clear(&err);
    memset(&input, 0, sizeof(input));
    YVEX_TEST_ASSERT(yvex_token_input_parse_explicit("0,8", &input, &err) == YVEX_OK,
                     "parse out-of-vocab list");
    YVEX_TEST_ASSERT(yvex_token_input_validate_bounds(&input, 8ull, &err) == YVEX_ERR_BOUNDS,
                     "bounds fail");
    YVEX_TEST_ASSERT(strstr(yvex_error_message(&err), "token-out-of-vocab") != NULL,
                     "out of vocab code");

    YVEX_TEST_ASSERT(yvex_token_input_parse_explicit("0,1", &input, &err) == YVEX_OK,
                     "parse select list");
    YVEX_TEST_ASSERT(yvex_token_input_select(&input, 2ull, &selected, &err) == YVEX_ERR_BOUNDS,
                     "select out of range");
    YVEX_TEST_ASSERT(strstr(yvex_error_message(&err), "token-index-out-of-range") != NULL,
                     "index out of range code");
    return 0;
}

int yvex_test_token_input(void)
{
    if (test_explicit_parse() != 0) return 1;
    if (test_parse_failures() != 0) return 1;
    if (test_bounds_and_select_failures() != 0) return 1;
    return 0;
}
