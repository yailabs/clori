/* Render role-enveloped source conversation policies without evaluating provider templates. */
#include "src/tokenizer/private.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

typedef struct {
    unsigned char *data;
    unsigned long long count, capacity;
} prompt_builder;

typedef struct {
    const unsigned char *bytes;
    unsigned long long count;
} prompt_span;

static int prompt_reserve(prompt_builder *builder, unsigned long long add,
                          yvex_error *err)
{
    unsigned long long need, capacity;
    unsigned char *grown;

    if (!builder || builder->count > ULLONG_MAX - add - 1u) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.prompt",
                       "rendered prompt extent overflowed");
        return YVEX_ERR_BOUNDS;
    }
    need = builder->count + add + 1u;
    if (need <= builder->capacity) return YVEX_OK;
    capacity = builder->capacity ? builder->capacity : 256u;
    while (capacity < need) {
        if (capacity > ULLONG_MAX / 2u || capacity * 2u > SIZE_MAX) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.prompt",
                           "rendered prompt exceeds address space");
            return YVEX_ERR_NOMEM;
        }
        capacity *= 2u;
    }
    grown = realloc(builder->data, (size_t)capacity);
    if (!grown) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.prompt",
                       "rendered prompt allocation failed");
        return YVEX_ERR_NOMEM;
    }
    builder->data = grown;
    builder->capacity = capacity;
    return YVEX_OK;
}

static int prompt_append(prompt_builder *builder, const void *bytes,
                         unsigned long long count, yvex_error *err)
{
    int rc = prompt_reserve(builder, count, err);
    if (rc != YVEX_OK) return rc;
    if (count) memcpy(builder->data + builder->count, bytes, (size_t)count);
    builder->count += count;
    builder->data[builder->count] = '\0';
    return YVEX_OK;
}

static int prompt_literal(prompt_builder *builder, const char *text,
                          yvex_error *err)
{
    return prompt_append(builder, text, strlen(text), err);
}

static int prompt_utf8_span(const char *text, unsigned long long declared,
                            prompt_span *span)
{
    unsigned long long count, offset = 0u, first = ULLONG_MAX, last = 0u;
    uint32_t point;

    if (!text || !span) return 0;
    count = declared ? declared : (unsigned long long)strlen(text);
    if (count > SIZE_MAX) return 0;
    while (offset < count) {
        unsigned long long begin = offset;
        if (!yvex_tokenizer_utf8_next((const unsigned char *)text, count,
                                      &offset, &point))
            return 0;
        if ((yvex_tokenizer_unicode_class(point) & TOKENIZER_UNICODE_SPACE) == 0u) {
            if (first == ULLONG_MAX) first = begin;
            last = offset;
        }
    }
    if (first == ULLONG_MAX) {
        span->bytes = (const unsigned char *)text;
        span->count = 0u;
    } else {
        span->bytes = (const unsigned char *)text + first;
        span->count = last - first;
    }
    return 1;
}

static int prompt_messages_valid(const yvex_prompt_message *messages,
                                 unsigned long long count,
                                 int generation_prompt)
{
    unsigned long long index;
    yvex_prompt_role prior = YVEX_PROMPT_ROLE_SYSTEM;

    if (!messages || !count) return 0;
    for (index = 0u; index < count; ++index) {
        yvex_prompt_role role = messages[index].role;
        prompt_span content, reasoning;
        if (messages[index].schema_version != YVEX_PROMPT_MESSAGE_SCHEMA_V1 ||
            role > YVEX_PROMPT_ROLE_TOOL ||
            (role == YVEX_PROMPT_ROLE_SYSTEM && index != 0u) ||
            (role == YVEX_PROMPT_ROLE_ASSISTANT &&
             prior != YVEX_PROMPT_ROLE_USER && prior != YVEX_PROMPT_ROLE_TOOL) ||
            (role == YVEX_PROMPT_ROLE_TOOL &&
             prior != YVEX_PROMPT_ROLE_ASSISTANT && prior != YVEX_PROMPT_ROLE_TOOL) ||
            !prompt_utf8_span(messages[index].content,
                              messages[index].content_len, &content) ||
            (messages[index].reasoning_content &&
             !prompt_utf8_span(messages[index].reasoning_content,
                               messages[index].reasoning_content_len,
                               &reasoning)))
            return 0;
        prior = role;
    }
    return !generation_prompt || prior == YVEX_PROMPT_ROLE_USER ||
           prior == YVEX_PROMPT_ROLE_TOOL;
}

static const char *reasoning_instruction(
    const yvex_conversation_protocol *conversation,
    yvex_reasoning_policy policy)
{
    if (policy == YVEX_REASONING_MAXIMUM)
        return conversation->reasoning_effort_max;
    if (policy == YVEX_REASONING_LOW)
        return conversation->reasoning_effort_low;
    return "";
}

static int prompt_system_append(prompt_builder *builder,
                                const yvex_conversation_protocol *conversation,
                                const yvex_prompt_message *message,
                                yvex_reasoning_policy policy,
                                yvex_error *err)
{
    const char *instruction = reasoning_instruction(conversation, policy);
    prompt_span content = {(const unsigned char *)"", 0u};
    int rc;

    if (message && !prompt_utf8_span(message->content, message->content_len,
                                     &content))
        return YVEX_ERR_FORMAT;
    if (!instruction[0] && !content.count) return YVEX_OK;
    rc = prompt_literal(builder, conversation->system, err);
    if (rc == YVEX_OK && instruction[0])
        rc = prompt_literal(builder, instruction, err);
    if (rc == YVEX_OK && instruction[0] && content.count)
        rc = prompt_literal(builder, "\n\n", err);
    if (rc == YVEX_OK && content.count)
        rc = prompt_append(builder, content.bytes, content.count, err);
    return rc == YVEX_OK
               ? prompt_literal(builder, conversation->message_end, err) : rc;
}

static int prompt_reasoning_append(
    prompt_builder *builder, const yvex_conversation_protocol *conversation,
    const yvex_prompt_message *message, yvex_error *err)
{
    prompt_span reasoning = {(const unsigned char *)"", 0u};
    int rc;

    if (message->reasoning_content &&
        !prompt_utf8_span(message->reasoning_content,
                          message->reasoning_content_len, &reasoning))
        return YVEX_ERR_FORMAT;
    rc = prompt_literal(builder, conversation->thinking_start, err);
    if (rc == YVEX_OK)
        rc = prompt_literal(builder, conversation->thinking_start_suffix, err);
    if (rc == YVEX_OK && reasoning.count)
        rc = prompt_append(builder, reasoning.bytes, reasoning.count, err);
    if (rc == YVEX_OK)
        rc = prompt_literal(builder, conversation->thinking_end_prefix, err);
    if (rc == YVEX_OK)
        rc = prompt_literal(builder, conversation->thinking_end, err);
    return rc == YVEX_OK
               ? prompt_literal(builder, conversation->thinking_end_suffix, err) : rc;
}

static int prompt_turns_append(prompt_builder *builder,
                               const yvex_conversation_protocol *conversation,
                               const yvex_prompt_message *messages,
                               unsigned long long count, int drop_reasoning,
                               yvex_error *err)
{
    unsigned long long index = messages[0].role == YVEX_PROMPT_ROLE_SYSTEM;
    unsigned long long last_user = ULLONG_MAX, scan;
    int tool_group = 0, rc = YVEX_OK;

    for (scan = count; scan > index; --scan)
        if (messages[scan - 1u].role == YVEX_PROMPT_ROLE_USER) {
            last_user = scan - 1u;
            break;
        }
    for (; index < count && rc == YVEX_OK; ++index) {
        const yvex_prompt_message *message = &messages[index];
        prompt_span content;
        if (!prompt_utf8_span(message->content, message->content_len, &content))
            return YVEX_ERR_FORMAT;
        if (message->role == YVEX_PROMPT_ROLE_USER) {
            tool_group = 0;
            rc = prompt_literal(builder, conversation->user, err);
            if (rc == YVEX_OK)
                rc = prompt_append(builder, content.bytes, content.count, err);
            if (rc == YVEX_OK)
                rc = prompt_literal(builder, conversation->message_end, err);
        } else if (message->role == YVEX_PROMPT_ROLE_ASSISTANT) {
            tool_group = 0;
            rc = prompt_literal(builder, conversation->assistant, err);
            if (rc == YVEX_OK &&
                (!drop_reasoning || index > last_user))
                rc = prompt_reasoning_append(builder, conversation, message, err);
            if (rc == YVEX_OK)
                rc = prompt_append(builder, content.bytes, content.count, err);
            if (rc == YVEX_OK)
                rc = prompt_literal(builder, conversation->message_end, err);
        } else if (message->role == YVEX_PROMPT_ROLE_TOOL) {
            if (!tool_group)
                rc = prompt_literal(builder,
                                    conversation->tool_result_group_start, err);
            if (rc == YVEX_OK)
                rc = prompt_literal(builder, conversation->tool_result_start, err);
            if (rc == YVEX_OK)
                rc = prompt_append(builder, content.bytes, content.count, err);
            if (rc == YVEX_OK)
                rc = prompt_literal(builder, conversation->tool_result_end, err);
            tool_group = 1;
            if (rc == YVEX_OK &&
                (index + 1u == count ||
                 messages[index + 1u].role != YVEX_PROMPT_ROLE_TOOL)) {
                rc = prompt_literal(builder, conversation->message_end, err);
                tool_group = 0;
            }
        }
    }
    return rc;
}

static int generation_prompt_append(
    prompt_builder *builder, const yvex_conversation_protocol *conversation,
    yvex_reasoning_policy policy, yvex_error *err)
{
    int rc = prompt_literal(builder, conversation->assistant, err);
    if (rc != YVEX_OK) return rc;
    rc = prompt_literal(builder, conversation->thinking_start, err);
    if (rc == YVEX_OK)
        rc = prompt_literal(builder, conversation->thinking_start_suffix, err);
    if (rc == YVEX_OK && policy == YVEX_REASONING_DISABLED)
        rc = prompt_literal(builder, conversation->thinking_end_prefix, err);
    if (rc == YVEX_OK && policy == YVEX_REASONING_DISABLED)
        rc = prompt_literal(builder, conversation->thinking_end, err);
    if (rc == YVEX_OK && policy == YVEX_REASONING_DISABLED)
        rc = prompt_literal(builder, conversation->thinking_end_suffix, err);
    return rc;
}

static int prompt_identity_build(const yvex_tokenizer *tokenizer,
                                 const yvex_prompt_message *messages,
                                 unsigned long long message_count,
                                 const yvex_prompt_options *options,
                                 yvex_rendered_prompt *prompt)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.messages.v2") ||
        !yvex_sha256_update_u64_be(&hash, message_count)) return 0;
    for (index = 0u; index < message_count; ++index) {
        prompt_span content, reasoning = {(const unsigned char *)"", 0u};
        if (!prompt_utf8_span(messages[index].content,
                              messages[index].content_len, &content) ||
            (messages[index].reasoning_content &&
             !prompt_utf8_span(messages[index].reasoning_content,
                               messages[index].reasoning_content_len,
                               &reasoning)) ||
            !yvex_sha256_update_u64_be(&hash, messages[index].role) ||
            !yvex_sha256_update_u64_be(&hash, content.count) ||
            !yvex_sha256_update(&hash, content.bytes, (size_t)content.count) ||
            !yvex_sha256_update_u64_be(&hash, reasoning.count) ||
            !yvex_sha256_update(&hash, reasoning.bytes,
                                (size_t)reasoning.count))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, prompt->message_sequence_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.rendered.v2") ||
        !yvex_sha256_update_u64_be(&hash, prompt->len) ||
        !yvex_sha256_update(&hash, prompt->text, (size_t)prompt->len) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, prompt->rendered_bytes_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.prompt.v2") ||
        !yvex_sha256_update_text(&hash, tokenizer->plan.tokenizer_plan_identity) ||
        !yvex_sha256_update_text(&hash, tokenizer->plan.prompt_policy_identity) ||
        !yvex_sha256_update_text(&hash, prompt->message_sequence_identity) ||
        !yvex_sha256_update_text(&hash, prompt->rendered_bytes_identity) ||
        !yvex_sha256_update_u64_be(&hash, options->add_bos) ||
        !yvex_sha256_update_u64_be(&hash, options->add_eos) ||
        !yvex_sha256_update_u64_be(&hash, options->add_generation_prompt) ||
        !yvex_sha256_update_u64_be(&hash, options->drop_thinking) ||
        !yvex_sha256_update_u64_be(&hash, options->mode) ||
        !yvex_sha256_update_u64_be(&hash, options->reasoning_policy) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, prompt->prompt_identity);
    return 1;
}

int yvex_tokenizer_prompt_render_v2(
    yvex_rendered_prompt *out, const yvex_tokenizer *tokenizer,
    const yvex_prompt_message *messages, unsigned long long message_count,
    const yvex_prompt_options *options, yvex_error *err)
{
    yvex_prompt_options defaults;
    const yvex_conversation_protocol *conversation;
    const yvex_prompt_message *system = NULL;
    prompt_builder builder = {0};
    yvex_rendered_prompt candidate = {0};
    int rc;

    if (!out || !tokenizer || !tokenizer->plan.sealed ||
        !tokenizer->conversation || tokenizer->conversation->schema_version !=
            YVEX_CONVERSATION_PROTOCOL_SCHEMA_V2 ||
        tokenizer->conversation->grammar !=
            YVEX_CONVERSATION_GRAMMAR_ROLE_ENVELOPED) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.prompt",
                       "an admitted role-enveloped tokenizer is required");
        return YVEX_ERR_INVALID_ARG;
    }
    conversation = tokenizer->conversation;
    memset(&defaults, 0, sizeof(defaults));
    defaults.add_bos = 1;
    defaults.add_generation_prompt = 1;
    defaults.drop_thinking = conversation->drop_prior_reasoning_by_default;
    defaults.reasoning_policy = conversation->default_reasoning_policy;
    defaults.mode = defaults.reasoning_policy == YVEX_REASONING_DISABLED
                        ? YVEX_PROMPT_MODE_CHAT : YVEX_PROMPT_MODE_THINKING;
    if (!options) options = &defaults;
    if (!yvex_reasoning_policy_valid(options->reasoning_policy) ||
        ((options->reasoning_policy == YVEX_REASONING_DISABLED) !=
         (options->mode == YVEX_PROMPT_MODE_CHAT)) ||
        (options->reasoning_policy == YVEX_REASONING_LOW &&
         !tokenizer->plan.low_reasoning_supported) ||
        !prompt_messages_valid(messages, message_count,
                               options->add_generation_prompt)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.prompt",
                       "messages and reasoning policy are not admitted");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    if (messages[0].role == YVEX_PROMPT_ROLE_SYSTEM) system = &messages[0];
    rc = options->add_bos
             ? prompt_literal(&builder, conversation->bos, err) : YVEX_OK;
    if (rc == YVEX_OK)
        rc = prompt_system_append(&builder, conversation, system,
                                  options->reasoning_policy, err);
    if (rc == YVEX_OK)
        rc = prompt_turns_append(&builder, conversation, messages,
                                 message_count, options->drop_thinking, err);
    if (rc == YVEX_OK && options->add_generation_prompt)
        rc = generation_prompt_append(&builder, conversation,
                                      options->reasoning_policy, err);
    if (rc == YVEX_OK && options->add_eos)
        rc = prompt_literal(&builder, conversation->eos, err);
    if (rc != YVEX_OK) {
        free(builder.data);
        return rc;
    }
    candidate.text = (char *)builder.data;
    candidate.len = builder.count;
    candidate.generation_prompt = options->add_generation_prompt;
    if (!prompt_identity_build(tokenizer, messages, message_count, options,
                               &candidate)) {
        free(candidate.text);
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.prompt.identity",
                       "role-enveloped prompt identity derivation failed");
        return YVEX_ERR_STATE;
    }
    *out = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}
