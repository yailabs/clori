/* Compile source-authored tokenizer and conversation facts into one pointer-free policy. */
#include <yvex/internal/tokenizer.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const unsigned char *data;
    size_t count, offset;
} policy_cursor;

static const char *const policy_domain = "yvex.tokenizer.family-policy.v1";

static int bytes_u64(yvex_core_bytes *bytes, unsigned long long value)
{
    unsigned char encoded[8];
    unsigned int index;
    for (index = 0u; index < sizeof(encoded); ++index)
        encoded[index] = (unsigned char)(value >> (56u - index * 8u));
    return yvex_core_bytes_append(bytes, encoded, sizeof(encoded));
}

static int bytes_text(yvex_core_bytes *bytes, const char *text)
{
    size_t count = text ? strlen(text) : 0u;
    return bytes_u64(bytes, count) && yvex_core_bytes_append(bytes, text, count);
}

static int cursor_u64(policy_cursor *cursor, unsigned long long *value)
{
    unsigned int index;
    if (!cursor || !value || cursor->offset > cursor->count ||
        cursor->count - cursor->offset < 8u)
        return 0;
    *value = 0ull;
    for (index = 0u; index < 8u; ++index)
        *value = (*value << 8u) | cursor->data[cursor->offset + index];
    cursor->offset += 8u;
    return 1;
}

static int cursor_text(policy_cursor *cursor, char *text, size_t capacity)
{
    unsigned long long count;
    if (!cursor_u64(cursor, &count) || !capacity || count >= capacity ||
        count > cursor->count - cursor->offset)
        return 0;
    if (count) memcpy(text, cursor->data + cursor->offset, (size_t)count);
    text[count] = '\0';
    cursor->offset += (size_t)count;
    return 1;
}

static const char *policy_text(const yvex_tokenizer_family_policy *policy,
                               yvex_tokenizer_policy_text field)
{
    unsigned int offset, length;
    if (!policy || (unsigned int)field >= YVEX_TOKENIZER_POLICY_TEXT_COUNT)
        return NULL;
    offset = policy->text_offsets[field];
    length = policy->text_lengths[field];
    if (offset > policy->text_bytes || length > policy->text_bytes - offset ||
        offset + length >= YVEX_TOKENIZER_POLICY_TEXT_CAP ||
        policy->text[offset + length] != '\0')
        return NULL;
    return policy->text + offset;
}

static int policy_identity_build(const yvex_tokenizer_family_policy *policy,
                                 char identity[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const char *fixed[] = {
        policy->architecture, policy->tokenizer_model, policy->tokenizer_pre,
        policy->tokenizer_json_identity, policy->tokenizer_config_identity};
    unsigned long long values[] = {
        policy->schema_version, policy->family_adapter_id, policy->family_adapter_version,
        policy->tokenizer_kind, policy->model_policy, policy->prompt_policy,
        policy->vocabulary_size, policy->base_vocabulary_size, policy->merge_count,
        policy->added_token_count, policy->special_token_count,
        policy->bos_token_id, policy->eos_token_id, policy->pad_token_id,
        policy->unk_token_id, policy->bos_present, policy->eos_present,
        policy->pad_present, policy->unk_present, policy->add_bos_token,
        policy->add_eos_token, policy->byte_fallback,
        policy->drop_prior_reasoning_by_default, policy->tools_preserve_reasoning,
        policy->tool_results_merge_into_user};
    size_t index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, policy_domain)) return 0;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        if (!yvex_sha256_update_u64_be(&hash, values[index])) return 0;
    for (index = 0u; index < sizeof(fixed) / sizeof(fixed[0]); ++index)
        if (!yvex_sha256_update_text(&hash, fixed[index])) return 0;
    if (policy->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION) {
        for (index = 0u; index < YVEX_TOKENIZER_POLICY_TEXT_COUNT; ++index) {
            const char *text = policy_text(policy, (yvex_tokenizer_policy_text)index);
            if (!text || !yvex_sha256_update_u64_be(&hash, strlen(text)) ||
                !yvex_sha256_update(&hash, text, strlen(text)))
                return 0;
        }
    } else if (!yvex_sha256_update_text(&hash, policy->direct_prompt_name)) {
        return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}

static int policy_boolean(int value)
{
    return value == 0 || value == 1;
}

int yvex_tokenizer_family_policy_validate(
    const yvex_tokenizer_family_policy *policy, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned int index;
    if (!policy || policy->schema_version != YVEX_TOKENIZER_FAMILY_POLICY_SCHEMA_V1 ||
        !policy->family_adapter_id || !policy->family_adapter_version ||
        policy->tokenizer_kind <= YVEX_TOKENIZER_KIND_UNKNOWN ||
        policy->tokenizer_kind > YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE ||
        policy->model_policy > YVEX_TOKENIZER_MODEL_BPE_BYTELEVEL ||
        (policy->prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION &&
         policy->prompt_policy != YVEX_TOKENIZER_PROMPT_MINIMAX_H3_FL2VA) ||
        !policy->vocabulary_size || !policy->base_vocabulary_size ||
        policy->base_vocabulary_size > policy->vocabulary_size ||
        !policy->architecture[0] || !policy->tokenizer_model[0] ||
        !policy->tokenizer_pre[0] ||
        !yvex_sha256_hex_is_valid(policy->tokenizer_json_identity) ||
        !yvex_sha256_hex_is_valid(policy->tokenizer_config_identity) ||
        !yvex_sha256_hex_is_valid(policy->policy_identity) ||
        policy->text_bytes > YVEX_TOKENIZER_POLICY_TEXT_CAP ||
        !policy_boolean(policy->bos_present) || !policy_boolean(policy->eos_present) ||
        !policy_boolean(policy->pad_present) || !policy_boolean(policy->unk_present) ||
        !policy_boolean(policy->add_bos_token) || !policy_boolean(policy->add_eos_token) ||
        !policy_boolean(policy->byte_fallback) ||
        !policy_boolean(policy->drop_prior_reasoning_by_default) ||
        !policy_boolean(policy->tools_preserve_reasoning) ||
        !policy_boolean(policy->tool_results_merge_into_user) ||
        (policy->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION &&
         policy->direct_prompt_name[0]) ||
        (policy->prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION &&
         (!policy->direct_prompt_name[0] || policy->text_bytes ||
          policy->drop_prior_reasoning_by_default || policy->tools_preserve_reasoning ||
          policy->tool_results_merge_into_user)))
        goto invalid;
    if (policy->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION)
        for (index = 0u; index < YVEX_TOKENIZER_POLICY_TEXT_COUNT; ++index)
            if (!policy_text(policy, (yvex_tokenizer_policy_text)index)) goto invalid;
    if (!policy_identity_build(policy, identity) ||
        strcmp(identity, policy->policy_identity) != 0)
        goto invalid;
    yvex_error_clear(err);
    return YVEX_OK;
invalid:
    yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.family-policy",
                   "compiled tokenizer policy is malformed or identity-mismatched");
    return YVEX_ERR_FORMAT;
}

static int policy_text_copy(yvex_tokenizer_family_policy *policy,
                            yvex_tokenizer_policy_text field, const char *text)
{
    size_t count = text ? strlen(text) : 0u;
    unsigned int offset = policy->text_bytes;
    if (!text || count > UINT32_MAX || offset > YVEX_TOKENIZER_POLICY_TEXT_CAP ||
        count + 1u > YVEX_TOKENIZER_POLICY_TEXT_CAP - offset)
        return 0;
    memcpy(policy->text + offset, text, count + 1u);
    policy->text_offsets[field] = offset;
    policy->text_lengths[field] = (unsigned int)count;
    policy->text_bytes += (unsigned int)count + 1u;
    return 1;
}

int yvex_tokenizer_family_policy_compile(
    yvex_tokenizer_family_policy *out, const yvex_conversation_protocol *source,
    yvex_tokenizer_kind tokenizer_kind, yvex_tokenizer_model_policy model_policy,
    yvex_tokenizer_prompt_policy prompt_policy, yvex_error *err)
{
    const char *texts[YVEX_TOKENIZER_POLICY_TEXT_COUNT];
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned int index;
    if (!out || !source || source->schema_version != YVEX_CONVERSATION_PROTOCOL_SCHEMA_V1 ||
        !source->family_adapter_id || !source->family_adapter_version ||
        !source->architecture || !source->tokenizer_model || !source->tokenizer_pre ||
        !source->tokenizer_json_identity || !source->tokenizer_config_identity) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.family-policy.compile",
                       "complete source-authored tokenizer facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    texts[YVEX_TOKENIZER_POLICY_SOURCE_REVISION] = source->source_revision;
    texts[YVEX_TOKENIZER_POLICY_SOURCE_ENCODING_PATH] = source->source_encoding_path;
    texts[YVEX_TOKENIZER_POLICY_SOURCE_ENCODING_IDENTITY] = source->source_encoding_identity;
    texts[YVEX_TOKENIZER_POLICY_BOS] = source->bos;
    texts[YVEX_TOKENIZER_POLICY_EOS] = source->eos;
    texts[YVEX_TOKENIZER_POLICY_USER] = source->user;
    texts[YVEX_TOKENIZER_POLICY_ASSISTANT] = source->assistant;
    texts[YVEX_TOKENIZER_POLICY_LATEST_REMINDER] = source->latest_reminder;
    texts[YVEX_TOKENIZER_POLICY_THINKING_START] = source->thinking_start;
    texts[YVEX_TOKENIZER_POLICY_THINKING_END] = source->thinking_end;
    texts[YVEX_TOKENIZER_POLICY_TOOL_RESULT_START] = source->tool_result_start;
    texts[YVEX_TOKENIZER_POLICY_TOOL_RESULT_END] = source->tool_result_end;
    texts[YVEX_TOKENIZER_POLICY_DSML] = source->dsml;
    texts[YVEX_TOKENIZER_POLICY_TOOL_CALLS_START] = source->tool_calls_start;
    texts[YVEX_TOKENIZER_POLICY_TOOL_CALLS_END] = source->tool_calls_end;
    texts[YVEX_TOKENIZER_POLICY_TOOL_INVOKE_START] = source->tool_invoke_start;
    texts[YVEX_TOKENIZER_POLICY_TOOL_INVOKE_NAME_END] = source->tool_invoke_name_end;
    texts[YVEX_TOKENIZER_POLICY_TOOL_INVOKE_END] = source->tool_invoke_end;
    texts[YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_START] = source->tool_parameter_start;
    texts[YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_NAME_END] = source->tool_parameter_name_end;
    texts[YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_KIND_END] = source->tool_parameter_kind_end;
    texts[YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_END] = source->tool_parameter_end;
    texts[YVEX_TOKENIZER_POLICY_REASONING_EFFORT_MAX] = source->reasoning_effort_max;
    texts[YVEX_TOKENIZER_POLICY_TOOLS_PREFIX] = source->tools_prefix;
    texts[YVEX_TOKENIZER_POLICY_TOOLS_SUFFIX] = source->tools_suffix;
    texts[YVEX_TOKENIZER_POLICY_RESPONSE_FORMAT_PREFIX] = source->response_format_prefix;
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_TOKENIZER_FAMILY_POLICY_SCHEMA_V1;
    out->family_adapter_id = source->family_adapter_id;
    out->family_adapter_version = source->family_adapter_version;
    out->tokenizer_kind = tokenizer_kind;
    out->model_policy = model_policy;
    out->prompt_policy = prompt_policy;
    out->vocabulary_size = source->vocabulary_size;
    out->base_vocabulary_size = source->base_vocabulary_size;
    out->merge_count = source->merge_count;
    out->added_token_count = source->added_token_count;
    out->special_token_count = source->special_token_count;
    out->bos_token_id = source->bos_token_id;
    out->eos_token_id = source->eos_token_id;
    out->pad_token_id = source->pad_token_id;
    out->unk_token_id = source->unk_token_id;
    out->bos_present = source->bos_present;
    out->eos_present = source->eos_present;
    out->pad_present = source->pad_present;
    out->unk_present = source->unk_present;
    out->add_bos_token = source->add_bos_token;
    out->add_eos_token = source->add_eos_token;
    out->byte_fallback = source->byte_fallback;
    out->drop_prior_reasoning_by_default = source->drop_prior_reasoning_by_default;
    out->tools_preserve_reasoning = source->tools_preserve_reasoning;
    out->tool_results_merge_into_user = source->tool_results_merge_into_user;
    yvex_core_text_copy(out->architecture, sizeof(out->architecture), source->architecture);
    yvex_core_text_copy(out->tokenizer_model, sizeof(out->tokenizer_model),
                        source->tokenizer_model);
    yvex_core_text_copy(out->tokenizer_pre, sizeof(out->tokenizer_pre), source->tokenizer_pre);
    yvex_core_text_copy(out->tokenizer_json_identity,
                        sizeof(out->tokenizer_json_identity), source->tokenizer_json_identity);
    yvex_core_text_copy(out->tokenizer_config_identity,
                        sizeof(out->tokenizer_config_identity), source->tokenizer_config_identity);
    for (index = 0u; index < YVEX_TOKENIZER_POLICY_TEXT_COUNT; ++index)
        if (!policy_text_copy(out, (yvex_tokenizer_policy_text)index, texts[index]))
            goto invalid;
    if (!policy_identity_build(out, identity)) goto invalid;
    yvex_core_text_copy(out->policy_identity, sizeof(out->policy_identity), identity);
    return yvex_tokenizer_family_policy_validate(out, err);
invalid:
    memset(out, 0, sizeof(*out));
    yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.family-policy.compile",
                   "source-authored tokenizer policy exceeds its compiled envelope");
    return YVEX_ERR_BOUNDS;
}

int yvex_tokenizer_family_policy_compile_direct(
    yvex_tokenizer_family_policy *out,
    const yvex_tokenizer_direct_policy *source, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    const char *const names[] = {
        source ? source->architecture : NULL,
        source ? source->tokenizer_model : NULL,
        source ? source->tokenizer_pre : NULL,
        source ? source->prompt_name : NULL};
    size_t index;

    if (!out || !source || !source->family_adapter_id ||
        !source->family_adapter_version ||
        source->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION ||
        !source->tokenizer_json_identity || !source->tokenizer_config_identity) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.family-policy.compile",
                       "complete direct tokenizer facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0u; index < sizeof(names) / sizeof(names[0]); ++index)
        if (!names[index] || !names[index][0] ||
            strlen(names[index]) >= YVEX_TOKENIZER_POLICY_NAME_CAP)
            goto invalid;
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_TOKENIZER_FAMILY_POLICY_SCHEMA_V1;
    out->family_adapter_id = source->family_adapter_id;
    out->family_adapter_version = source->family_adapter_version;
    out->tokenizer_kind = source->tokenizer_kind;
    out->model_policy = source->model_policy;
    out->prompt_policy = source->prompt_policy;
    out->vocabulary_size = source->vocabulary_size;
    out->base_vocabulary_size = source->base_vocabulary_size;
    out->merge_count = source->merge_count;
    out->added_token_count = source->added_token_count;
    out->special_token_count = source->special_token_count;
    out->bos_token_id = source->bos_token_id;
    out->eos_token_id = source->eos_token_id;
    out->pad_token_id = source->pad_token_id;
    out->unk_token_id = source->unk_token_id;
    out->bos_present = source->bos_present;
    out->eos_present = source->eos_present;
    out->pad_present = source->pad_present;
    out->unk_present = source->unk_present;
    out->add_bos_token = source->add_bos_token;
    out->add_eos_token = source->add_eos_token;
    out->byte_fallback = source->byte_fallback;
    yvex_core_text_copy(out->architecture, sizeof(out->architecture), source->architecture);
    yvex_core_text_copy(out->tokenizer_model, sizeof(out->tokenizer_model),
                        source->tokenizer_model);
    yvex_core_text_copy(out->tokenizer_pre, sizeof(out->tokenizer_pre), source->tokenizer_pre);
    yvex_core_text_copy(out->tokenizer_json_identity,
                        sizeof(out->tokenizer_json_identity), source->tokenizer_json_identity);
    yvex_core_text_copy(out->tokenizer_config_identity,
                        sizeof(out->tokenizer_config_identity),
                        source->tokenizer_config_identity);
    yvex_core_text_copy(out->direct_prompt_name, sizeof(out->direct_prompt_name),
                        source->prompt_name);
    if (!policy_identity_build(out, identity)) goto invalid;
    yvex_core_text_copy(out->policy_identity, sizeof(out->policy_identity), identity);
    return yvex_tokenizer_family_policy_validate(out, err);
invalid:
    memset(out, 0, sizeof(*out));
    yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.family-policy.compile",
                   "direct tokenizer policy exceeds its compiled envelope");
    return YVEX_ERR_BOUNDS;
}

int yvex_tokenizer_family_policy_encode(
    const yvex_tokenizer_family_policy *policy, yvex_core_bytes *bytes,
    yvex_error *err)
{
    const unsigned long long values[] = {
        policy ? policy->schema_version : 0u,
        policy ? policy->family_adapter_id : 0u,
        policy ? policy->family_adapter_version : 0u,
        policy ? (unsigned int)policy->tokenizer_kind : 0u,
        policy ? (unsigned int)policy->model_policy : 0u,
        policy ? (unsigned int)policy->prompt_policy : 0u,
        policy ? policy->vocabulary_size : 0u,
        policy ? policy->base_vocabulary_size : 0u,
        policy ? policy->merge_count : 0u,
        policy ? policy->added_token_count : 0u,
        policy ? policy->special_token_count : 0u,
        policy ? policy->bos_token_id : 0u, policy ? policy->eos_token_id : 0u,
        policy ? policy->pad_token_id : 0u, policy ? policy->unk_token_id : 0u,
        policy ? (unsigned int)policy->bos_present : 0u,
        policy ? (unsigned int)policy->eos_present : 0u,
        policy ? (unsigned int)policy->pad_present : 0u,
        policy ? (unsigned int)policy->unk_present : 0u,
        policy ? (unsigned int)policy->add_bos_token : 0u,
        policy ? (unsigned int)policy->add_eos_token : 0u,
        policy ? (unsigned int)policy->byte_fallback : 0u,
        policy ? (unsigned int)policy->drop_prior_reasoning_by_default : 0u,
        policy ? (unsigned int)policy->tools_preserve_reasoning : 0u,
        policy ? (unsigned int)policy->tool_results_merge_into_user : 0u};
    size_t index;
    if (!bytes || yvex_tokenizer_family_policy_validate(policy, err) != YVEX_OK)
        return YVEX_ERR_INVALID_ARG;
    if (!bytes_text(bytes, policy_domain)) goto allocation;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        if (!bytes_u64(bytes, values[index])) goto allocation;
    if (!bytes_text(bytes, policy->architecture) ||
        !bytes_text(bytes, policy->tokenizer_model) ||
        !bytes_text(bytes, policy->tokenizer_pre) ||
        !bytes_text(bytes, policy->tokenizer_json_identity) ||
        !bytes_text(bytes, policy->tokenizer_config_identity) ||
        !bytes_text(bytes, policy->policy_identity)) goto allocation;
    if (policy->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION) {
        if (!bytes_u64(bytes, YVEX_TOKENIZER_POLICY_TEXT_COUNT)) goto allocation;
        for (index = 0u; index < YVEX_TOKENIZER_POLICY_TEXT_COUNT; ++index)
            if (!bytes_text(bytes, policy_text(policy, (yvex_tokenizer_policy_text)index)))
                goto allocation;
    } else if (!bytes_text(bytes, policy->direct_prompt_name) || !bytes_u64(bytes, 0u)) {
        goto allocation;
    }
    yvex_error_clear(err);
    return YVEX_OK;
allocation:
    yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.family-policy.encode",
                   "compiled tokenizer policy exceeds its serialization budget");
    return YVEX_ERR_NOMEM;
}

int yvex_tokenizer_family_policy_decode(
    yvex_tokenizer_family_policy *policy, const unsigned char *data,
    size_t count, yvex_error *err)
{
    policy_cursor cursor = {data, count, 0u};
    char domain[64], identity[YVEX_SHA256_HEX_CAP];
    unsigned long long values[25], text_count;
    char text[YVEX_TOKENIZER_POLICY_TEXT_CAP];
    size_t index;
    if (!policy || (!data && count)) return YVEX_ERR_INVALID_ARG;
    memset(policy, 0, sizeof(*policy));
    if (!cursor_text(&cursor, domain, sizeof(domain)) || strcmp(domain, policy_domain) != 0)
        goto invalid;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        if (!cursor_u64(&cursor, &values[index])) goto invalid;
    policy->schema_version = (unsigned int)values[0];
    policy->family_adapter_id = values[1]; policy->family_adapter_version = values[2];
    policy->tokenizer_kind = (yvex_tokenizer_kind)values[3];
    policy->model_policy = (yvex_tokenizer_model_policy)values[4];
    policy->prompt_policy = (yvex_tokenizer_prompt_policy)values[5];
    policy->vocabulary_size = values[6]; policy->base_vocabulary_size = values[7];
    policy->merge_count = values[8]; policy->added_token_count = values[9];
    policy->special_token_count = values[10];
    policy->bos_token_id = (unsigned int)values[11];
    policy->eos_token_id = (unsigned int)values[12];
    policy->pad_token_id = (unsigned int)values[13];
    policy->unk_token_id = (unsigned int)values[14];
    policy->bos_present = (int)values[15]; policy->eos_present = (int)values[16];
    policy->pad_present = (int)values[17]; policy->unk_present = (int)values[18];
    policy->add_bos_token = (int)values[19]; policy->add_eos_token = (int)values[20];
    policy->byte_fallback = (int)values[21];
    policy->drop_prior_reasoning_by_default = (int)values[22];
    policy->tools_preserve_reasoning = (int)values[23];
    policy->tool_results_merge_into_user = (int)values[24];
    if (!cursor_text(&cursor, policy->architecture, sizeof(policy->architecture)) ||
        !cursor_text(&cursor, policy->tokenizer_model, sizeof(policy->tokenizer_model)) ||
        !cursor_text(&cursor, policy->tokenizer_pre, sizeof(policy->tokenizer_pre)) ||
        !cursor_text(&cursor, policy->tokenizer_json_identity,
                     sizeof(policy->tokenizer_json_identity)) ||
        !cursor_text(&cursor, policy->tokenizer_config_identity,
                     sizeof(policy->tokenizer_config_identity)) ||
        !cursor_text(&cursor, identity, sizeof(identity)))
        goto invalid;
    if (policy->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION) {
        if (!cursor_u64(&cursor, &text_count) ||
            text_count != YVEX_TOKENIZER_POLICY_TEXT_COUNT)
            goto invalid;
        for (index = 0u; index < YVEX_TOKENIZER_POLICY_TEXT_COUNT; ++index) {
            if (!cursor_text(&cursor, text, sizeof(text)) ||
                !policy_text_copy(policy, (yvex_tokenizer_policy_text)index, text))
                goto invalid;
        }
    } else if (!cursor_text(&cursor, policy->direct_prompt_name,
                            sizeof(policy->direct_prompt_name)) ||
               !cursor_u64(&cursor, &text_count) || text_count != 0u) {
        goto invalid;
    }
    if (cursor.offset != cursor.count) goto invalid;
    yvex_core_text_copy(policy->policy_identity, sizeof(policy->policy_identity), identity);
    return yvex_tokenizer_family_policy_validate(policy, err);
invalid:
    memset(policy, 0, sizeof(*policy));
    yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.family-policy.decode",
                   "compiled tokenizer policy encoding is malformed");
    return YVEX_ERR_FORMAT;
}

int yvex_tokenizer_family_policy_conversation(
    const yvex_tokenizer_family_policy *policy,
    yvex_conversation_protocol *conversation)
{
    if (!conversation || yvex_tokenizer_family_policy_validate(policy, NULL) != YVEX_OK ||
        policy->prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION)
        return 0;
    memset(conversation, 0, sizeof(*conversation));
    conversation->schema_version = YVEX_CONVERSATION_PROTOCOL_SCHEMA_V1;
    conversation->family_adapter_id = policy->family_adapter_id;
    conversation->family_adapter_version = policy->family_adapter_version;
    conversation->architecture = policy->architecture;
#define VIEW(member, field) conversation->member = policy_text(policy, field)
    VIEW(source_revision, YVEX_TOKENIZER_POLICY_SOURCE_REVISION);
    VIEW(source_encoding_path, YVEX_TOKENIZER_POLICY_SOURCE_ENCODING_PATH);
    VIEW(source_encoding_identity, YVEX_TOKENIZER_POLICY_SOURCE_ENCODING_IDENTITY);
    VIEW(bos, YVEX_TOKENIZER_POLICY_BOS); VIEW(eos, YVEX_TOKENIZER_POLICY_EOS);
    VIEW(user, YVEX_TOKENIZER_POLICY_USER); VIEW(assistant, YVEX_TOKENIZER_POLICY_ASSISTANT);
    VIEW(latest_reminder, YVEX_TOKENIZER_POLICY_LATEST_REMINDER);
    VIEW(thinking_start, YVEX_TOKENIZER_POLICY_THINKING_START);
    VIEW(thinking_end, YVEX_TOKENIZER_POLICY_THINKING_END);
    VIEW(tool_result_start, YVEX_TOKENIZER_POLICY_TOOL_RESULT_START);
    VIEW(tool_result_end, YVEX_TOKENIZER_POLICY_TOOL_RESULT_END);
    VIEW(dsml, YVEX_TOKENIZER_POLICY_DSML);
    VIEW(tool_calls_start, YVEX_TOKENIZER_POLICY_TOOL_CALLS_START);
    VIEW(tool_calls_end, YVEX_TOKENIZER_POLICY_TOOL_CALLS_END);
    VIEW(tool_invoke_start, YVEX_TOKENIZER_POLICY_TOOL_INVOKE_START);
    VIEW(tool_invoke_name_end, YVEX_TOKENIZER_POLICY_TOOL_INVOKE_NAME_END);
    VIEW(tool_invoke_end, YVEX_TOKENIZER_POLICY_TOOL_INVOKE_END);
    VIEW(tool_parameter_start, YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_START);
    VIEW(tool_parameter_name_end, YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_NAME_END);
    VIEW(tool_parameter_kind_end, YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_KIND_END);
    VIEW(tool_parameter_end, YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_END);
    VIEW(reasoning_effort_max, YVEX_TOKENIZER_POLICY_REASONING_EFFORT_MAX);
    VIEW(tools_prefix, YVEX_TOKENIZER_POLICY_TOOLS_PREFIX);
    VIEW(tools_suffix, YVEX_TOKENIZER_POLICY_TOOLS_SUFFIX);
    VIEW(response_format_prefix, YVEX_TOKENIZER_POLICY_RESPONSE_FORMAT_PREFIX);
#undef VIEW
    conversation->drop_prior_reasoning_by_default = policy->drop_prior_reasoning_by_default;
    conversation->tools_preserve_reasoning = policy->tools_preserve_reasoning;
    conversation->tool_results_merge_into_user = policy->tool_results_merge_into_user;
    conversation->tokenizer_model = policy->tokenizer_model;
    conversation->tokenizer_pre = policy->tokenizer_pre;
    conversation->tokenizer_json_identity = policy->tokenizer_json_identity;
    conversation->tokenizer_config_identity = policy->tokenizer_config_identity;
    conversation->vocabulary_size = policy->vocabulary_size;
    conversation->base_vocabulary_size = policy->base_vocabulary_size;
    conversation->merge_count = policy->merge_count;
    conversation->added_token_count = policy->added_token_count;
    conversation->special_token_count = policy->special_token_count;
    conversation->bos_token_id = policy->bos_token_id;
    conversation->eos_token_id = policy->eos_token_id;
    conversation->pad_token_id = policy->pad_token_id;
    conversation->unk_token_id = policy->unk_token_id;
    conversation->bos_present = policy->bos_present;
    conversation->eos_present = policy->eos_present;
    conversation->pad_present = policy->pad_present;
    conversation->unk_present = policy->unk_present;
    conversation->add_bos_token = policy->add_bos_token;
    conversation->add_eos_token = policy->add_eos_token;
    conversation->byte_fallback = policy->byte_fallback;
    return 1;
}
