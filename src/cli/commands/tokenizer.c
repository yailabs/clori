#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include <yvex/core.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/model.h>
#include <yvex/tokenizer.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const literal_pair_0[] = { "yvex: input requires tokens or prompt",
    "usage: yvex execute input tokens --model FILE_OR_ALIAS --tokens IDS | "
    "yvex execute input prompt --model FILE_OR_ALIAS --text TEXT"
};

static const char *const literal_pair_1[] = { "token_input_status: fail",
    "token_input_kind: prompt-text"};

static const char *const literal_pair_2[] = { "prefill_ready: false",
    "status: token-input-fail"};

static const char *const literal_pair_3[] = { "",
    "status: detokenized"};

static const char *const literal_lines_0[] = { "prefill_ready: false",
    "logits_ready: false",
    "generation: unsupported"};

static const char *const literal_lines_1[] = { "token_bounds_status: not-checked",
    "prefill_ready: false",
    "logits_ready: false",
    "generation: unsupported",
    "status: token-input-fail"};

static const char *const literal_lines_2[] = { "prefill_ready: false",
    "logits_ready: false",
    "generation: unsupported"};

static const char *const literal_lines_3[] = { "usage: yvex execute input tokens --model FILE_OR_ALIAS --tokens IDS",
    "       yvex execute input prompt --model FILE_OR_ALIAS --text TEXT",
    "\nInput parses explicit tokens or tokenizer-backed prompt text into validated token input."};

static int context_tokenizer_open(yvex_model_context *context, yvex_error *err)
{
    int rc = yvex_family_tokenizer_open(&context->tokenizer, context->gguf, err);

    if (rc != YVEX_ERR_UNSUPPORTED) return rc;
    yvex_error_clear(err);
    return yvex_tokenizer_from_gguf(
        &context->tokenizer, context->gguf, context->model, err);
}

static int model_context_open_tokenizer(
    const char *path, yvex_model_context *context, yvex_error *err)
{
    int rc = yvex_model_context_open(path, context, err);

    if (rc == YVEX_OK) rc = context_tokenizer_open(context, err);
    if (rc != YVEX_OK) yvex_model_context_close(context);
    return rc;
}

static int print_special_id_line(const char *name, int (*fn)(const yvex_tokenizer *, unsigned int *),
    const yvex_tokenizer *tokenizer)
{
    unsigned int id;
    int rc = fn(tokenizer, &id);

    if (rc == YVEX_OK) {
        yvex_cli_out_writef(stdout, "%s: %u\n", name, id);
    } else {
        yvex_cli_out_writef(stdout, "%s: absent\n", name);
    }
    return YVEX_OK;
}

static int command_tokenizer(int arg_count, char **args)
{
    yvex_model_context ctx;
    yvex_error err;
    const yvex_tokenizer_plan_summary *plan;
    int rc;

    yvex_error_clear(&err);

    if (arg_count != 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        if (arg_count == 3) {
            yvex_tokenizer_help(stdout);
            return 0;
        }
        yvex_cli_out_writef(stderr, "yvex: tokenizer requires exactly one path\n");
        yvex_cli_out_writef(stderr, "usage: yvex inspect tokenizer <path>\n");
        return 2;
    }

    rc = model_context_open_tokenizer(args[2], &ctx, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }

    yvex_cli_out_writef(stdout, "format: gguf\n");
    yvex_cli_out_writef(stdout, "architecture: %s\n", yvex_arch_name(yvex_model_arch(ctx.model)));
    yvex_cli_out_writef(stdout, "model_name: %s\n", yvex_model_name(ctx.model));
    yvex_cli_out_writef(stdout, "tokenizer_model: %s\n",
        yvex_tokenizer_kind_name(yvex_tokenizer_kind_of(ctx.tokenizer)));
    yvex_cli_out_writef(stdout, "support: %s\n", yvex_tokenizer_support_name(yvex_tokenizer_support_of(ctx.tokenizer)));
    yvex_cli_out_writef(stdout, "vocab_size: %llu\n", yvex_tokenizer_vocab_size(ctx.tokenizer));
    plan = yvex_tokenizer_plan_summary_get(ctx.tokenizer);
    if (plan) {
        yvex_cli_out_writef(stdout, "runtime_support: exact-artifact-bpe\n");
        yvex_cli_out_writef(stdout, "base_vocab_size: %llu\n", plan->base_vocabulary_size);
        yvex_cli_out_writef(stdout, "merge_count: %llu\n", plan->merge_count);
        yvex_cli_out_writef(stdout, "added_token_count: %llu\n", plan->added_token_count);
        yvex_cli_out_writef(stdout, "special_token_count: %llu\n", plan->special_token_count);
        yvex_cli_out_writef(stdout, "tokenizer_json_identity: %s\n", plan->tokenizer_json_identity);
        yvex_cli_out_writef(stdout, "tokenizer_config_identity: %s\n", plan->tokenizer_config_identity);
        yvex_cli_out_writef(stdout, "tokenizer_plan_identity: %s\n", plan->tokenizer_plan_identity);
        yvex_cli_out_writef(
            stdout, "chat_template: %s\n",
            plan->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION
                ? "conversation-family-policy" : "verbatim-no-special");
    } else {
        yvex_cli_out_writef(stdout, "runtime_support: unavailable\n");
        yvex_cli_out_writef(stdout, "chat_template: absent\n");
    }
    (void)print_special_id_line("bos_token_id", yvex_tokenizer_bos_id, ctx.tokenizer);
    (void)print_special_id_line("eos_token_id", yvex_tokenizer_eos_id, ctx.tokenizer);
    (void)print_special_id_line("pad_token_id", yvex_tokenizer_pad_id, ctx.tokenizer);
    (void)print_special_id_line("unk_token_id", yvex_tokenizer_unk_id, ctx.tokenizer);
    yvex_cli_out_writef(stdout, "tokenizer_runtime_ready: %s\n", plan ? "true" : "false");
    yvex_cli_out_writef(stdout, "generation_ready: false\n");
    yvex_cli_out_writef(stdout, "status: %s\n", plan ? "tokenizer-ready" : "tokenizer-descriptor");

    yvex_model_context_close(&ctx);
    return 0;
}

static int command_tokenize(int arg_count, char **args)
{
    yvex_model_context ctx;
    yvex_tokenizer_encode_result encoded;
    yvex_tokens fixture_tokens;
    const yvex_tokens *tokens;
    yvex_tokenizer_encode_options options = {0, 0, 1, ULLONG_MAX};
    yvex_error err;
    const char *text = NULL;
    int want_pieces = 0;
    int i;
    int rc;

    yvex_error_clear(&err);

    if (arg_count < 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        yvex_tokenize_help(stdout);
        return arg_count >= 3 ? 0 : 2;
    }

    for (i = 3; i < arg_count; ++i) {
        if (strcmp(args[i], "--text") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --text requires a value\n");
                return 2;
            }
            text = args[++i];
        } else if (strcmp(args[i], "--pieces") == 0) {
            want_pieces = 1;
        } else if (strcmp(args[i], "--bos") == 0) {
            options.add_bos = 1;
        } else if (strcmp(args[i], "--no-bos") == 0) {
            options.add_bos = 0;
        } else if (strcmp(args[i], "--eos") == 0) {
            options.add_eos = 1;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown tokenize option: %s\n", args[i]);
            yvex_cli_out_writef(stderr, "Try 'yvex help tokenize' for usage.\n");
            return 2;
        }
    }
    if (!text) {
        yvex_cli_out_writef(stderr, "yvex: tokenize requires --text\n");
        return 2;
    }

    rc = model_context_open_tokenizer(args[2], &ctx, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }

    memset(&encoded, 0, sizeof(encoded));
    memset(&fixture_tokens, 0, sizeof(fixture_tokens));
    if (yvex_tokenizer_plan_summary_get(ctx.tokenizer)) {
        rc = yvex_tokenizer_encode(ctx.tokenizer, (const unsigned char *)text,
                                   (unsigned long long)strlen(text), &options, &encoded, &err);
        tokens = &encoded.tokens;
    } else {
        if (options.add_bos || options.add_eos) {
            yvex_error_set(&err, YVEX_ERR_UNSUPPORTED, "cli.tokenize.special-policy",
                           "fixture tokenizers do not admit explicit BOS/EOS insertion");
            rc = YVEX_ERR_UNSUPPORTED;
        } else {
            rc = yvex_tokenize_text(ctx.tokenizer, text, &fixture_tokens, &err);
        }
        tokens = &fixture_tokens;
    }
    if (rc != YVEX_OK) {
        yvex_model_context_close(&ctx);
        return print_yvex_error(&err, exit_for_status(rc));
    }

    yvex_cli_out_writef(stdout, "tokens: %llu\n", tokens->len);
    print_token_ids(tokens);
    if (want_pieces) {
        yvex_cli_out_writef(stdout, "pieces:\n");
        for (i = 0; (unsigned long long)i < tokens->len; ++i) {
            const yvex_token_info *token = yvex_tokenizer_token_at(ctx.tokenizer, tokens->ids[i]);
            yvex_cli_out_writef(stdout, "  %u ", tokens->ids[i]);
            print_quoted_bytes(token ? token->text : "", token ? token->text_len : 0);
            yvex_cli_out_writef(stdout, "\n");
        }
    }
    yvex_cli_out_writef(stdout, "encoding_identity: %s\n",
                        encoded.completed ? encoded.encoding_identity : "unavailable");
    yvex_cli_out_writef(stdout, "tokenizer_runtime_ready: %s\n",
                        encoded.completed ? "true" : "false");
    yvex_cli_out_writef(stdout, "generation_ready: false\n");
    yvex_cli_out_writef(stdout, "status: tokenized\n");

    yvex_tokenizer_encode_result_clear(&encoded);
    yvex_tokens_clear(&fixture_tokens);
    yvex_model_context_close(&ctx);
    return 0;
}

static int command_detokenize(int arg_count, char **args)
{
    yvex_model_context ctx;
    yvex_error err;
    const char *ids_text = NULL;
    unsigned int *ids = NULL;
    unsigned long long ids_len = 0;
    yvex_tokenizer_decode_result decoded;
    yvex_tokenizer_decode_options options = {0};
    const yvex_tokenizer_plan_summary *plan;
    int i;
    int rc;

    yvex_error_clear(&err);

    if (arg_count < 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        yvex_detokenize_help(stdout);
        return arg_count >= 3 ? 0 : 2;
    }

    for (i = 3; i < arg_count; ++i) {
        if (strcmp(args[i], "--ids") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --ids requires a value\n");
                return 2;
            }
            ids_text = args[++i];
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown detokenize option: %s\n", args[i]);
            return 2;
        }
    }
    if (!ids_text || !parse_id_list(ids_text, &ids, &ids_len)) {
        yvex_cli_out_writef(stderr, "yvex: detokenize requires comma-separated --ids\n");
        return 2;
    }

    rc = model_context_open_tokenizer(args[2], &ctx, &err);
    if (rc != YVEX_OK) {
        free(ids);
        return print_yvex_error(&err, exit_for_status(rc));
    }

    memset(&decoded, 0, sizeof(decoded));
    plan = yvex_tokenizer_plan_summary_get(ctx.tokenizer);
    if (plan) {
        rc = yvex_tokenizer_decode(ctx.tokenizer, ids, ids_len, &options, &decoded, &err);
    } else {
        unsigned long long byte_capacity = 1u, index;
        for (index = 0u; index < ids_len; ++index) {
            const yvex_token_info *token = yvex_tokenizer_token_at(ctx.tokenizer, ids[index]);
            if (!token || token->text_len > ULLONG_MAX - byte_capacity) {
                yvex_error_set(&err, token ? YVEX_ERR_BOUNDS : YVEX_ERR_INVALID_ARG,
                               "tokenizer.cli.detokenize", "fixture decode extent is invalid");
                rc = token ? YVEX_ERR_BOUNDS : YVEX_ERR_INVALID_ARG;
                break;
            }
            byte_capacity += token->text_len;
        }
        if (rc == YVEX_OK && byte_capacity <= SIZE_MAX) {
            decoded.bytes = malloc((size_t)byte_capacity);
            if (!decoded.bytes) {
                yvex_error_set(&err, YVEX_ERR_NOMEM, "tokenizer.cli.detokenize",
                               "fixture decode allocation failed");
                rc = YVEX_ERR_NOMEM;
            } else {
                rc = yvex_detokenize_ids(ctx.tokenizer, ids, ids_len,
                                         (char *)decoded.bytes, byte_capacity, &err);
                if (rc == YVEX_OK)
                    decoded.byte_count = (unsigned long long)strlen((char *)decoded.bytes);
            }
        } else if (rc == YVEX_OK) {
            yvex_error_set(&err, YVEX_ERR_BOUNDS, "tokenizer.cli.detokenize",
                           "fixture decode exceeds host address space");
            rc = YVEX_ERR_BOUNDS;
        }
    }
    free(ids);
    if (rc != YVEX_OK) {
        yvex_model_context_close(&ctx);
        return print_yvex_error(&err, exit_for_status(rc));
    }

    yvex_cli_out_writef(stdout, "text: ");
    print_quoted_bytes((const char *)decoded.bytes, decoded.byte_count);
    yvex_cli_out_lines(stdout, literal_pair_3, sizeof(literal_pair_3) / sizeof(literal_pair_3[0]));
    yvex_cli_out_writef(stdout, "decoder_identity: %s\n",
                        plan ? decoded.decoder_identity : "unavailable");
    yvex_cli_out_writef(stdout, "detokenization_ready: %s\n", plan ? "true" : "false");
    yvex_cli_out_writef(stdout, "generation_ready: false\n");

    yvex_tokenizer_decode_result_clear(&decoded);
    yvex_model_context_close(&ctx);
    return 0;
}

static int command_prompt(int arg_count, char **args)
{
    yvex_model_context ctx;
    yvex_prompt_message messages[16];
    unsigned long long message_count = 0;
    yvex_prompt_options options;
    yvex_rendered_prompt rendered;
    yvex_tokenizer_encode_result encoded;
    yvex_error err;
    int want_tokens = 0;
    int i;
    int rc;

    yvex_error_clear(&err);
    memset(&rendered, 0, sizeof(rendered));
    options.add_bos = 1;
    options.add_eos = 0;
    options.add_generation_prompt = 1;
    options.drop_thinking = 1;
    options.mode = YVEX_PROMPT_MODE_CHAT;

    if (arg_count < 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        yvex_prompt_help(stdout);
        return arg_count >= 3 ? 0 : 2;
    }

    for (i = 3; i < arg_count; ++i) {
        yvex_prompt_role role;
        if (strcmp(args[i], "--system") == 0) {
            role = YVEX_PROMPT_ROLE_SYSTEM;
        } else if (strcmp(args[i], "--user") == 0) {
            role = YVEX_PROMPT_ROLE_USER;
        } else if (strcmp(args[i], "--assistant") == 0) {
            role = YVEX_PROMPT_ROLE_ASSISTANT;
        } else if (strcmp(args[i], "--tool") == 0) {
            role = YVEX_PROMPT_ROLE_TOOL;
        } else if (strcmp(args[i], "--thinking") == 0) {
            options.mode = YVEX_PROMPT_MODE_THINKING;
            continue;
        } else if (strcmp(args[i], "--no-generation-prompt") == 0) {
            options.add_generation_prompt = 0;
            continue;
        } else if (strcmp(args[i], "--tokens") == 0) {
            want_tokens = 1;
            continue;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown prompt option: %s\n", args[i]);
            return 2;
        }

        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: prompt option %s requires text\n", args[i]);
            return 2;
        }
        if (message_count >= sizeof(messages) / sizeof(messages[0])) {
            yvex_cli_out_writef(stderr, "yvex: too many prompt messages\n");
            return 2;
        }
        messages[message_count].role = role;
        messages[message_count].content = args[++i];
        messages[message_count].content_len =
            (unsigned long long)strlen(messages[message_count].content);
        message_count += 1;
    }

    if (message_count == 0) {
        yvex_cli_out_writef(stderr, "yvex: prompt requires at least one message\n");
        return 2;
    }

    rc = model_context_open_tokenizer(args[2], &ctx, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }

    rc = yvex_prompt_render(&rendered, ctx.tokenizer, messages, message_count, &options, &err);
    if (rc != YVEX_OK) {
        yvex_model_context_close(&ctx);
        return print_yvex_error(&err, exit_for_status(rc));
    }

    yvex_cli_out_writef(stdout, "template: conversation-family-policy\n");
    yvex_cli_out_writef(stdout, "rendered_bytes: %llu\n", rendered.len);
    yvex_cli_out_writef(stdout, "rendered:\n%s\n", rendered.text);

    if (want_tokens) {
        yvex_tokenizer_encode_options encode_options = {0, 0, 1, ULLONG_MAX};
        memset(&encoded, 0, sizeof(encoded));
        rc = yvex_tokenizer_encode(ctx.tokenizer, (const unsigned char *)rendered.text,
                                   rendered.len, &encode_options, &encoded, &err);
        if (rc != YVEX_OK) {
            yvex_rendered_prompt_free(&rendered);
            yvex_model_context_close(&ctx);
            return print_yvex_error(&err, exit_for_status(rc));
        }
        yvex_cli_out_writef(stdout, "tokens: %llu\n", encoded.tokens.len);
        print_token_ids(&encoded.tokens);
        yvex_cli_out_writef(stdout, "encoding_identity: %s\n", encoded.encoding_identity);
        yvex_tokenizer_encode_result_clear(&encoded);
    }
    yvex_cli_out_writef(stdout, "prompt_identity: %s\n", rendered.prompt_identity);
    yvex_cli_out_writef(stdout, "tokenizer_runtime_ready: true\n");
    yvex_cli_out_writef(stdout, "generation_ready: false\n");
    yvex_cli_out_writef(stdout, "status: rendered\n");

    yvex_rendered_prompt_free(&rendered);
    yvex_model_context_close(&ctx);
    return 0;
}

static void print_token_input_tokens(const yvex_token_input *input)
{
    unsigned long long i;

    for (i = 0; input && i < input->token_count; ++i) {
        yvex_cli_out_writef(stdout, "token_%llu: %u\n", i, input->tokens[i]);
    }
}

void print_token_input_summary(const yvex_token_input *input,
                               const char *status,
                               const char *bounds_status,
                               unsigned long long selected_index,
                               unsigned int selected_token,
                               int has_selected)
{
    yvex_cli_out_writef(stdout, "token_input_status: %s\n", status ? status : "fail");
    yvex_cli_out_writef(stdout, "token_input_kind: %s\n",
                        input ? yvex_token_input_kind_name(input->kind) : "unknown");
    yvex_cli_out_writef(stdout, "token_count: %llu\n", input ? input->token_count : 0ull);
    if (input) {
        yvex_cli_out_writef(stdout, "selected_token_index: %llu\n", selected_index);
    }
    if (has_selected) {
        yvex_cli_out_writef(stdout, "selected_token_id: %u\n", selected_token);
    } else if (input) {
        yvex_cli_out_writef(stdout, "selected_token_id: unavailable\n");
    }
    yvex_cli_out_writef(stdout, "token_bounds_status: %s\n",
                        bounds_status ? bounds_status : "not-checked");
}

static int command_input_tokens(yvex_model_ref *ref,
                                const char *model_arg,
                                const char *tokens_text)
{
    yvex_token_input input;
    yvex_error err;
    unsigned long long vocab_size = 0ull;
    int rc;

    memset(&input, 0, sizeof(input));
    yvex_error_clear(&err);
    rc = yvex_token_input_parse_explicit(tokens_text, &input, &err);
    if (rc == YVEX_OK) rc = yvex_model_context_vocab_size(ref->path, &vocab_size, &err);
    if (rc == YVEX_OK) rc = yvex_token_input_validate_bounds(&input, vocab_size, &err);
    yvex_cli_out_writef(stdout, "token_input: tokens\n");
    yvex_cli_out_writef(stdout, "model: %s\n", model_arg);
    yvex_cli_out_writef(stdout, "resolved_path: %s\n", ref->path ? ref->path : "");
    yvex_cli_out_writef(stdout, "model_input_kind: %s\n",
                        ref->kind == YVEX_MODEL_REF_ALIAS ? "alias" : "path");
    yvex_cli_out_writef(stdout, "identity_status: %s\n",
                        ref->kind == YVEX_MODEL_REF_ALIAS ? "pass" : "unregistered");
    print_token_input_summary(&input, rc == YVEX_OK ? "pass" : "fail",
                              rc == YVEX_OK ? "pass" :
                                  input.token_bounds_checked ? "fail" : "not-checked",
                              0ull, input.token_count ? input.tokens[0] : 0u,
                              input.token_count > 0ull);
    yvex_cli_out_writef(stdout, "vocab_size: %llu\n", vocab_size);
    print_token_input_tokens(&input);
    yvex_cli_out_lines(stdout, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
    yvex_cli_out_writef(stdout, "status: %s\n",
                        rc == YVEX_OK ? "token-input-pass" : "token-input-fail");
    yvex_model_ref_clear(ref);
    return rc == YVEX_OK ? 0 : print_yvex_error(&err, exit_for_status(rc));
}

/* Render an admitted-model identity failure without attempting tokenization. */

static int command_input_identity_failure(yvex_model_ref *ref,
                                          const char *subcommand,
                                          const char *model_arg,
                                          int rc)
{
    yvex_cli_out_writef(stdout, "token_input: %s\n", subcommand);
    yvex_cli_out_writef(stdout, "model: %s\n", model_arg);
    yvex_cli_out_writef(stdout, "resolved_path: %s\n", ref->path ? ref->path : "");
    yvex_cli_out_writef(stdout, "identity_status: fail\n");
    print_token_input_summary(NULL, "fail", "not-checked", 0ull, 0u, 0);
    yvex_cli_out_lines(stdout, literal_pair_2, sizeof(literal_pair_2) / sizeof(literal_pair_2[0]));
    yvex_model_ref_clear(ref);
    return exit_for_status(rc);
}

static int command_input_prompt_failure(yvex_model_ref *ref,
                                        yvex_model_context *ctx,
                                        const char *model_arg,
                                        const char *tokenizer_status,
                                        const char *reason,
                                        yvex_error *err,
                                        int rc)
{
    yvex_cli_out_writef(stdout, "token_input: prompt\n");
    yvex_cli_out_writef(stdout, "model: %s\n", model_arg);
    yvex_cli_out_writef(stdout, "resolved_path: %s\n", ref->path ? ref->path : "");
    yvex_cli_out_writef(stdout, "model_input_kind: %s\n",
                        ref->kind == YVEX_MODEL_REF_ALIAS ? "alias" : "path");
    yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
    yvex_cli_out_writef(stdout, "tokenizer_status: %s\n", tokenizer_status);
    if (ctx && ctx->tokenizer && strcmp(tokenizer_status, "unsupported") == 0) {
        yvex_cli_out_writef(stdout, "tokenizer_support: %s\n",
                            yvex_tokenizer_support_name(yvex_tokenizer_support_of(ctx->tokenizer)));
    }
    yvex_cli_out_writef(stdout, "reason: %s\n", reason);
    yvex_cli_out_lines(stdout, literal_lines_1, sizeof(literal_lines_1) / sizeof(literal_lines_1[0]));
    if (ctx && ctx->model) yvex_model_context_close(ctx);
    yvex_model_ref_clear(ref);
    return print_yvex_error(err, exit_for_status(rc));
}

static int command_input(int arg_count, char **args)
{
    yvex_model_ref ref;
    yvex_model_context ctx;
    yvex_token_input input;
    yvex_tokens tokens;
    yvex_error err;
    const char *subcommand;
    const char *model_arg = NULL;
    const char *tokens_text = NULL;
    const char *prompt_text = NULL;
    int i;
    int rc;

    yvex_error_clear(&err);
    memset(&ref, 0, sizeof(ref));
    memset(&ctx, 0, sizeof(ctx));
    memset(&input, 0, sizeof(input));
    memset(&tokens, 0, sizeof(tokens));
    if (arg_count < 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        yvex_input_help(stdout);
        return arg_count >= 3 ? 0 : 2;
    }
    subcommand = args[2];
    if (strcmp(subcommand, "tokens") != 0 && strcmp(subcommand, "prompt") != 0) {
        yvex_cli_out_lines(stderr, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
        return 2;
    }
    for (i = 3; i < arg_count; ++i) {
        if (strcmp(args[i], "--model") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --model requires FILE_OR_ALIAS\n");
                return 2;
            }
            model_arg = args[++i];
        } else if (strcmp(args[i], "--tokens") == 0 && strcmp(subcommand, "tokens") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --tokens requires IDS\n");
                return 2;
            }
            tokens_text = args[++i];
        } else if (strcmp(args[i], "--text") == 0 && strcmp(subcommand, "prompt") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --text requires TEXT\n");
                return 2;
            }
            prompt_text = args[++i];
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown input option: %s\n", args[i]);
            yvex_cli_out_writef(stderr, "Try 'yvex help input' for usage.\n");
            return 2;
        }
    }
    if (!model_arg) {
        yvex_cli_out_writef(stderr, "yvex: input requires --model FILE_OR_ALIAS\n");
        return 2;
    }
    if (strcmp(subcommand, "tokens") == 0 && !tokens_text) {
        yvex_cli_out_writef(stderr, "yvex: input tokens requires --tokens IDS\n");
        return 2;
    }
    if (strcmp(subcommand, "prompt") == 0 && !prompt_text) {
        yvex_cli_out_writef(stderr, "yvex: input prompt requires --text TEXT\n");
        return 2;
    }
    rc = yvex_model_ref_resolve(&ref, model_arg, NULL, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    rc = enforce_registered_identity_cli(&ref, "input");
    if (rc != YVEX_OK) return command_input_identity_failure(&ref, subcommand, model_arg, rc);
    if (strcmp(subcommand, "tokens") == 0) return command_input_tokens(&ref, model_arg, tokens_text);

    rc = yvex_model_context_open(ref.path, &ctx, &err);
    if (rc != YVEX_OK) {
        return command_input_prompt_failure(&ref, NULL, model_arg, "not-checked",
                                            yvex_error_message(&err), &err, rc);
    }
    rc = context_tokenizer_open(&ctx, &err);
    if (rc != YVEX_OK ||
        yvex_tokenizer_support_of(ctx.tokenizer) != YVEX_TOKENIZER_SUPPORT_FIXTURE_ENCODE_DECODE) {
        const char *status = rc == YVEX_OK ? "unsupported" : "missing";
        yvex_error_set(&err, YVEX_ERR_UNSUPPORTED, "yvex_input_prompt",
                       "tokenizer-metadata-missing");
        return command_input_prompt_failure(&ref, &ctx, model_arg, status,
                                            "tokenizer-metadata-missing", &err,
                                            YVEX_ERR_UNSUPPORTED);
    }
    rc = yvex_tokenize_text(ctx.tokenizer, prompt_text, &tokens, &err);
    if (rc == YVEX_OK) {
        rc = yvex_token_input_from_ids(YVEX_TOKEN_INPUT_PROMPT_TEXT,
                                       tokens.ids, tokens.len, &input, &err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_token_input_validate_bounds(&input,
                                              yvex_tokenizer_vocab_size(ctx.tokenizer), &err);
    }
    yvex_cli_out_writef(stdout, "token_input: prompt\n");
    yvex_cli_out_writef(stdout, "model: %s\n", model_arg);
    yvex_cli_out_writef(stdout, "resolved_path: %s\n", ref.path ? ref.path : "");
    yvex_cli_out_writef(stdout, "model_input_kind: %s\n",
                        ref.kind == YVEX_MODEL_REF_ALIAS ? "alias" : "path");
    yvex_cli_out_writef(stdout, "tokenizer_status: present\n");
    yvex_cli_out_writef(stdout, "tokenizer_support: %s\n",
                        yvex_tokenizer_support_name(yvex_tokenizer_support_of(ctx.tokenizer)));
    print_token_input_summary(&input, rc == YVEX_OK ? "pass" : "fail",
                              rc == YVEX_OK ? "pass" :
                                  input.token_bounds_checked ? "fail" : "not-checked",
                              0ull, input.token_count ? input.tokens[0] : 0u,
                              input.token_count > 0ull);
    yvex_cli_out_writef(stdout, "vocab_size: %llu\n", yvex_tokenizer_vocab_size(ctx.tokenizer));
    print_token_input_tokens(&input);
    yvex_cli_out_lines(stdout, literal_lines_2, sizeof(literal_lines_2) / sizeof(literal_lines_2[0]));
    yvex_cli_out_writef(stdout, "status: %s\n",
                        rc == YVEX_OK ? "token-input-pass" : "token-input-fail");
    yvex_tokens_free(&tokens);
    yvex_model_context_close(&ctx);
    yvex_model_ref_clear(&ref);
    return rc == YVEX_OK ? 0 : print_yvex_error(&err, exit_for_status(rc));
}

int yvex_detokenize_command(int arg_count, char **args)
{
    return command_detokenize(arg_count, args);
}

int yvex_input_command(int arg_count, char **args)
{
    return command_input(arg_count, args);
}

int yvex_prompt_command(int arg_count, char **args)
{
    return command_prompt(arg_count, args);
}

int yvex_tokenize_command(int arg_count, char **args)
{
    return command_tokenize(arg_count, args);
}

int yvex_tokenizer_command(int arg_count, char **args)
{
    return command_tokenizer(arg_count, args);
}

void yvex_detokenize_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex execute tokenizer decode <path> --ids IDS\n\nDecodes IDs through the exact artifact tokenizer.\n");
}

void yvex_input_help(FILE *fp)
{
    yvex_cli_out_lines(fp, literal_lines_3, sizeof(literal_lines_3) / sizeof(literal_lines_3[0]));
}

void yvex_prompt_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex execute tokenizer prompt <path> [--system TEXT] --user TEXT [--assistant TEXT] "
        "[--tool TEXT] [--thinking] [--tokens]\n\nRenders the exact bounded DeepSeek prompt policy.\n");
}

void yvex_tokenize_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex execute tokenizer encode <path> --text TEXT [--bos] [--eos] [--pieces]\n\n"
        "Encodes an explicit byte span through the exact artifact tokenizer.\n");
}

void yvex_tokenizer_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex inspect tokenizer <path>\n\nPrints tokenizer kind, support level, vocabulary facts, special "
            "token IDs, and chat template presence.\n");
}
