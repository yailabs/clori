/*
 * Exercises tokenizer layer tokenizer metadata extraction, vocabulary records, special-token
 * IDs, fixture encode/decode, malformed tokenizer metadata handling, and unsupported tokenizer
 * behavior.
 */
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *table;
    yvex_model_descriptor *model;
    yvex_tokenizer *tokenizer;
} tokenizer_stack;

static void stack_close(tokenizer_stack *stack)
{
    if (!stack) {
        return;
    }
    yvex_tokenizer_close(stack->tokenizer);
    yvex_model_descriptor_close(stack->model);
    yvex_tensor_table_close(stack->table);
    yvex_gguf_close(stack->gguf);
    yvex_artifact_close(stack->artifact);
    memset(stack, 0, sizeof(*stack));
}

static int stack_open_until_model(const char *path, tokenizer_stack *stack)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(stack, 0, sizeof(*stack));
    memset(&options, 0, sizeof(options));
    options.path = path;
    options.readonly = 1;

    rc = yvex_artifact_open(&stack->artifact, &options, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "artifact open failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }
    rc = yvex_gguf_open(&stack->gguf, stack->artifact, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "gguf open failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        stack_close(stack);
        return 1;
    }
    rc = yvex_tensor_table_from_gguf(&stack->table, stack->gguf, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "tensor table failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        stack_close(stack);
        return 1;
    }
    rc = yvex_model_descriptor_from_gguf(&stack->model, stack->gguf, stack->table, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "descriptor failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        stack_close(stack);
        return 1;
    }
    return 0;
}

static int stack_open_tokenizer(const char *path, tokenizer_stack *stack)
{
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(stack_open_until_model(path, stack) == 0, "open stack until model");
    rc = yvex_tokenizer_from_gguf(&stack->tokenizer, stack->gguf, stack->model, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "tokenizer failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        stack_close(stack);
        return 1;
    }
    return 0;
}

static int expect_tokenizer_status(const char *path, int expected)
{
    tokenizer_stack stack;
    yvex_tokenizer *tokenizer = NULL;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(stack_open_until_model(path, &stack) == 0, "open malformed tokenizer stack");
    rc = yvex_tokenizer_from_gguf(&tokenizer, stack.gguf, stack.model, &err);
    if (rc != expected) {
        fprintf(stderr, "expected %d for %s, got %d: %s: %s\n",
                expected, path, rc, yvex_error_where(&err), yvex_error_message(&err));
        yvex_tokenizer_close(tokenizer);
        stack_close(&stack);
        return 1;
    }
    yvex_tokenizer_close(tokenizer);
    stack_close(&stack);
    return 0;
}

static int test_valid_tokenizer(void)
{
    tokenizer_stack stack;
    const yvex_token_info *token;
    unsigned int id;

    YVEX_TEST_ASSERT(stack_open_tokenizer("tests/fixtures/gguf/valid-tokenizer-simple.gguf", &stack) == 0,
                     "open valid tokenizer");

    YVEX_TEST_ASSERT(yvex_tokenizer_kind_of(stack.tokenizer) == YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE,
                     "fixture kind");
    YVEX_TEST_ASSERT(yvex_tokenizer_support_of(stack.tokenizer) == YVEX_TOKENIZER_SUPPORT_FIXTURE_ENCODE_DECODE,
                     "fixture support");
    YVEX_TEST_ASSERT_STREQ(yvex_tokenizer_kind_name(yvex_tokenizer_kind_of(stack.tokenizer)),
                           "yvex-fixture-simple", "kind name");
    YVEX_TEST_ASSERT_STREQ(yvex_tokenizer_support_name(yvex_tokenizer_support_of(stack.tokenizer)),
                           "fixture-encode-decode", "support name");
    YVEX_TEST_ASSERT(yvex_tokenizer_vocab_size(stack.tokenizer) == 8, "vocab size");

    token = yvex_tokenizer_token_at(stack.tokenizer, 3);
    YVEX_TEST_ASSERT(token != NULL, "token 3 exists");
    YVEX_TEST_ASSERT(token->id == 3, "token id");
    YVEX_TEST_ASSERT_STREQ(token->text, "hello", "token text");
    YVEX_TEST_ASSERT(token->text_len == 5, "token len");
    YVEX_TEST_ASSERT(token->type == YVEX_TOKEN_TYPE_NORMAL, "token type");

    YVEX_TEST_ASSERT(yvex_tokenizer_bos_id(stack.tokenizer, &id) == YVEX_OK && id == 1, "bos id");
    YVEX_TEST_ASSERT(yvex_tokenizer_eos_id(stack.tokenizer, &id) == YVEX_OK && id == 2, "eos id");
    YVEX_TEST_ASSERT(yvex_tokenizer_unk_id(stack.tokenizer, &id) == YVEX_OK && id == 0, "unk id");
    YVEX_TEST_ASSERT(yvex_tokenizer_pad_id(stack.tokenizer, &id) == YVEX_ERR_UNSUPPORTED, "missing pad clean");
    YVEX_TEST_ASSERT(yvex_tokenizer_token_at(stack.tokenizer, 99) == NULL, "token out of range null");

    stack_close(&stack);
    return 0;
}

static int test_encode_decode(void)
{
    tokenizer_stack stack;
    yvex_tokens tokens;
    yvex_error err;
    char text[128];
    unsigned int ids[] = {3, 4, 5};
    int rc;

    YVEX_TEST_ASSERT(stack_open_tokenizer("tests/fixtures/gguf/valid-tokenizer-simple.gguf", &stack) == 0,
                     "open tokenizer for encode");

    rc = yvex_tokenize_text(stack.tokenizer, "hello world", &tokens, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "tokenize hello world");
    YVEX_TEST_ASSERT(tokens.len == 3, "token count");
    YVEX_TEST_ASSERT(tokens.ids[0] == 3 && tokens.ids[1] == 4 && tokens.ids[2] == 5, "token ids");
    yvex_tokens_free(&tokens);

    rc = yvex_tokenize_text(stack.tokenizer, "hello!", &tokens, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "tokenize unknown char uses unk");
    YVEX_TEST_ASSERT(tokens.len == 2, "unknown token count");
    YVEX_TEST_ASSERT(tokens.ids[0] == 3 && tokens.ids[1] == 0, "unknown token ids");
    yvex_tokens_free(&tokens);

    rc = yvex_detokenize_ids(stack.tokenizer, ids, 3, text, sizeof(text), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "detokenize ids");
    YVEX_TEST_ASSERT_STREQ(text, "hello world", "detokenized text");

    ids[0] = 99;
    rc = yvex_detokenize_ids(stack.tokenizer, ids, 1, text, sizeof(text), &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "out-of-range id fails");

    stack_close(&stack);
    return 0;
}

static int test_malformed_and_unsupported(void)
{
    tokenizer_stack stack;
    yvex_tokens tokens;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(expect_tokenizer_status("tests/fixtures/gguf/tokenizer-missing-tokens.gguf",
                                             YVEX_ERR_UNSUPPORTED) == 0,
                     "missing tokens unsupported");
    YVEX_TEST_ASSERT(expect_tokenizer_status("tests/fixtures/gguf/tokenizer-type-len.gguf",
                                             YVEX_ERR_FORMAT) == 0,
                     "bad token type len fails");
    YVEX_TEST_ASSERT(expect_tokenizer_status("tests/fixtures/gguf/tokenizer-bad-score-len.gguf",
                                             YVEX_ERR_FORMAT) == 0,
                     "bad score len fails");
    YVEX_TEST_ASSERT(expect_tokenizer_status("tests/fixtures/gguf/tokenizer-bad-special-id.gguf",
                                             YVEX_ERR_BOUNDS) == 0,
                     "bad special id fails");

    YVEX_TEST_ASSERT(stack_open_tokenizer("tests/fixtures/gguf/tokenizer-unsupported-arch.gguf", &stack) == 0,
                     "open unsupported tokenizer model");
    YVEX_TEST_ASSERT(yvex_tokenizer_kind_of(stack.tokenizer) == YVEX_TOKENIZER_KIND_UNKNOWN,
                     "unsupported kind unknown");
    YVEX_TEST_ASSERT(yvex_tokenizer_support_of(stack.tokenizer) == YVEX_TOKENIZER_SUPPORT_UNSUPPORTED,
                     "unsupported support");
    rc = yvex_tokenize_text(stack.tokenizer, "hello", &tokens, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED, "unsupported tokenizer does not encode");
    stack_close(&stack);
    return 0;
}

int yvex_test_tokenizer(void)
{
    if (test_valid_tokenizer() != 0) {
        return 1;
    }
    if (test_encode_decode() != 0) {
        return 1;
    }
    if (test_malformed_and_unsupported() != 0) {
        return 1;
    }
    return 0;
}
