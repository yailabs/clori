/*
 * Exercises the tokenizer layer default prompt renderer for explicit role messages and its
 * interaction with the fixture tokenizer.
 */
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int test_render_system_user(void)
{
    yvex_prompt_message messages[2];
    yvex_prompt_options options;
    yvex_rendered_prompt prompt;
    yvex_error err;
    int rc;
    const char *expected =
        "<system>\n"
        "You are helpful\n"
        "</system>\n"
        "<user>\n"
        "hello world\n"
        "</user>\n"
        "<assistant>\n";

    messages[0].role = YVEX_PROMPT_ROLE_SYSTEM;
    messages[0].content = "You are helpful";
    messages[1].role = YVEX_PROMPT_ROLE_USER;
    messages[1].content = "hello world";
    options.add_bos = 0;
    options.add_eos = 0;
    options.add_generation_prompt = 1;

    rc = yvex_prompt_render(&prompt, NULL, messages, 2, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "render system user");
    YVEX_TEST_ASSERT_STREQ(prompt.text, expected, "rendered prompt");
    YVEX_TEST_ASSERT(prompt.len == strlen(expected), "rendered len");
    yvex_rendered_prompt_free(&prompt);
    return 0;
}

static int test_render_variants_and_errors(void)
{
    yvex_prompt_message message;
    yvex_prompt_options options;
    yvex_rendered_prompt prompt;
    yvex_error err;
    int rc;

    message.role = YVEX_PROMPT_ROLE_USER;
    message.content = "hello world";
    options.add_bos = 0;
    options.add_eos = 0;
    options.add_generation_prompt = 0;
    rc = yvex_prompt_render(&prompt, NULL, &message, 1, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "render user only");
    YVEX_TEST_ASSERT_STREQ(prompt.text, "<user>\nhello world\n</user>\n", "user-only prompt");
    yvex_rendered_prompt_free(&prompt);

    message.role = YVEX_PROMPT_ROLE_ASSISTANT;
    message.content = "done";
    rc = yvex_prompt_render(&prompt, NULL, &message, 1, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "render assistant");
    YVEX_TEST_ASSERT_STREQ(prompt.text, "<assistant>\ndone\n</assistant>\n", "assistant prompt");
    yvex_rendered_prompt_free(&prompt);

    rc = yvex_prompt_render(&prompt, NULL, &message, 0, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "empty message list fails");

    message.content = NULL;
    rc = yvex_prompt_render(&prompt, NULL, &message, 1, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "null content fails");

    return 0;
}

static int test_rendered_prompt_tokenizes(void)
{
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *table = NULL;
    yvex_model_descriptor *model = NULL;
    yvex_tokenizer *tokenizer = NULL;
    yvex_prompt_message message;
    yvex_prompt_options options;
    yvex_rendered_prompt prompt;
    yvex_tokens tokens;
    yvex_error err;
    int rc;

    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    artifact_options.readonly = 1;

    rc = yvex_artifact_open(&artifact, &artifact_options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open artifact");
    rc = yvex_gguf_open(&gguf, artifact, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open gguf");
    rc = yvex_tensor_table_from_gguf(&table, gguf, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "build table");
    rc = yvex_model_descriptor_from_gguf(&model, gguf, table, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "build model");
    rc = yvex_tokenizer_from_gguf(&tokenizer, gguf, model, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "build tokenizer");

    message.role = YVEX_PROMPT_ROLE_USER;
    message.content = "hello world";
    options.add_bos = 0;
    options.add_eos = 0;
    options.add_generation_prompt = 1;
    rc = yvex_prompt_render(&prompt, tokenizer, &message, 1, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "render prompt with tokenizer");
    rc = yvex_tokenize_text(tokenizer, prompt.text, &tokens, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "rendered prompt tokenizes");
    YVEX_TEST_ASSERT(tokens.len > 0, "rendered prompt token count");

    yvex_tokens_free(&tokens);
    yvex_rendered_prompt_free(&prompt);
    yvex_tokenizer_close(tokenizer);
    yvex_model_descriptor_close(model);
    yvex_tensor_table_close(table);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

int yvex_test_prompt(void)
{
    if (test_render_system_user() != 0) {
        return 1;
    }
    if (test_render_variants_and_errors() != 0) {
        return 1;
    }
    if (test_rendered_prompt_tokenizes() != 0) {
        return 1;
    }
    return 0;
}
