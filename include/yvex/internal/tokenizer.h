/* Compiler-owned tokenizer policy consumed by artifact admission and runtime instantiation. */
#ifndef INCLUDE_YVEX_INTERNAL_TOKENIZER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_TOKENIZER_H_INCLUDED

#include <stddef.h>
#include <yvex/internal/conversation.h>
#include <yvex/internal/core.h>
#include <yvex/tokenizer.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_TOKENIZER_FAMILY_POLICY_SCHEMA_V1 1u
#define YVEX_TOKENIZER_POLICY_TEXT_CAP 4096u
#define YVEX_TOKENIZER_POLICY_NAME_CAP 64u

typedef enum {
    YVEX_TOKENIZER_POLICY_SOURCE_REVISION = 0,
    YVEX_TOKENIZER_POLICY_SOURCE_ENCODING_PATH,
    YVEX_TOKENIZER_POLICY_SOURCE_ENCODING_IDENTITY,
    YVEX_TOKENIZER_POLICY_BOS,
    YVEX_TOKENIZER_POLICY_EOS,
    YVEX_TOKENIZER_POLICY_USER,
    YVEX_TOKENIZER_POLICY_ASSISTANT,
    YVEX_TOKENIZER_POLICY_LATEST_REMINDER,
    YVEX_TOKENIZER_POLICY_THINKING_START,
    YVEX_TOKENIZER_POLICY_THINKING_END,
    YVEX_TOKENIZER_POLICY_TOOL_RESULT_START,
    YVEX_TOKENIZER_POLICY_TOOL_RESULT_END,
    YVEX_TOKENIZER_POLICY_DSML,
    YVEX_TOKENIZER_POLICY_TOOL_CALLS_START,
    YVEX_TOKENIZER_POLICY_TOOL_CALLS_END,
    YVEX_TOKENIZER_POLICY_TOOL_INVOKE_START,
    YVEX_TOKENIZER_POLICY_TOOL_INVOKE_NAME_END,
    YVEX_TOKENIZER_POLICY_TOOL_INVOKE_END,
    YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_START,
    YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_NAME_END,
    YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_KIND_END,
    YVEX_TOKENIZER_POLICY_TOOL_PARAMETER_END,
    YVEX_TOKENIZER_POLICY_REASONING_EFFORT_MAX,
    YVEX_TOKENIZER_POLICY_TOOLS_PREFIX,
    YVEX_TOKENIZER_POLICY_TOOLS_SUFFIX,
    YVEX_TOKENIZER_POLICY_RESPONSE_FORMAT_PREFIX,
    YVEX_TOKENIZER_POLICY_TEXT_COUNT
} yvex_tokenizer_policy_text;

typedef struct yvex_tokenizer_family_policy {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    yvex_tokenizer_kind tokenizer_kind;
    yvex_tokenizer_model_policy model_policy;
    yvex_tokenizer_prompt_policy prompt_policy;
    unsigned long long vocabulary_size, base_vocabulary_size, merge_count;
    unsigned long long added_token_count, special_token_count;
    unsigned int bos_token_id, eos_token_id, pad_token_id, unk_token_id;
    int bos_present, eos_present, pad_present, unk_present;
    int add_bos_token, add_eos_token, byte_fallback;
    int drop_prior_reasoning_by_default, tools_preserve_reasoning;
    int tool_results_merge_into_user;
    char architecture[YVEX_TOKENIZER_POLICY_NAME_CAP];
    char tokenizer_model[YVEX_TOKENIZER_POLICY_NAME_CAP];
    char tokenizer_pre[YVEX_TOKENIZER_POLICY_NAME_CAP];
    char tokenizer_json_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_config_identity[YVEX_SHA256_HEX_CAP];
    char direct_prompt_name[YVEX_TOKENIZER_POLICY_NAME_CAP];
    char policy_identity[YVEX_SHA256_HEX_CAP];
    unsigned int text_offsets[YVEX_TOKENIZER_POLICY_TEXT_COUNT];
    unsigned int text_lengths[YVEX_TOKENIZER_POLICY_TEXT_COUNT];
    unsigned int text_bytes;
    char text[YVEX_TOKENIZER_POLICY_TEXT_CAP];
} yvex_tokenizer_family_policy;

typedef struct {
    unsigned long long family_adapter_id, family_adapter_version;
    yvex_tokenizer_kind tokenizer_kind;
    yvex_tokenizer_model_policy model_policy;
    yvex_tokenizer_prompt_policy prompt_policy;
    unsigned long long vocabulary_size, base_vocabulary_size, merge_count;
    unsigned long long added_token_count, special_token_count;
    unsigned int bos_token_id, eos_token_id, pad_token_id, unk_token_id;
    int bos_present, eos_present, pad_present, unk_present;
    int add_bos_token, add_eos_token, byte_fallback;
    const char *architecture, *tokenizer_model, *tokenizer_pre;
    const char *tokenizer_json_identity, *tokenizer_config_identity;
    const char *prompt_name;
} yvex_tokenizer_direct_policy;

int yvex_tokenizer_family_policy_compile(
    yvex_tokenizer_family_policy *out,
    const yvex_conversation_protocol *source,
    yvex_tokenizer_kind tokenizer_kind,
    yvex_tokenizer_model_policy model_policy,
    yvex_tokenizer_prompt_policy prompt_policy, yvex_error *err);
int yvex_tokenizer_family_policy_compile_direct(
    yvex_tokenizer_family_policy *out,
    const yvex_tokenizer_direct_policy *source, yvex_error *err);
int yvex_tokenizer_family_policy_validate(
    const yvex_tokenizer_family_policy *policy, yvex_error *err);
int yvex_tokenizer_family_policy_encode(
    const yvex_tokenizer_family_policy *policy, yvex_core_bytes *bytes,
    yvex_error *err);
int yvex_tokenizer_family_policy_decode(
    yvex_tokenizer_family_policy *policy, const unsigned char *data,
    size_t count, yvex_error *err);
int yvex_tokenizer_family_policy_conversation(
    const yvex_tokenizer_family_policy *policy,
    yvex_conversation_protocol *conversation);

struct yvex_gguf;
int yvex_tokenizer_from_compiled_gguf(
    yvex_tokenizer **out, const struct yvex_gguf *gguf,
    const yvex_tokenizer_family_policy *policy, yvex_error *err);
const yvex_conversation_protocol *yvex_tokenizer_conversation_protocol_get(
    const yvex_tokenizer *tokenizer);

#ifdef __cplusplus
}
#endif
#endif
