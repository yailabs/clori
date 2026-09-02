/*
 * Family projections publish source-authored conversation syntax through this immutable view.
 * Tokenizer and provider owners interpret the view; they never select a family or reconstruct its
 * special tokens. Returned strings have process lifetime and contain exact UTF-8 bytes.
 */
#ifndef INCLUDE_YVEX_INTERNAL_CONVERSATION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_CONVERSATION_H_INCLUDED

#include <yvex/provider.h>

#define YVEX_CONVERSATION_PROTOCOL_SCHEMA_V1 1u
#define YVEX_CONVERSATION_PROTOCOL_SCHEMA_V2 2u

typedef enum {
    YVEX_CONVERSATION_GRAMMAR_SEGMENTED = 0,
    YVEX_CONVERSATION_GRAMMAR_ROLE_ENVELOPED
} yvex_conversation_grammar;

typedef enum {
    YVEX_CONVERSATION_TOOL_GRAMMAR_TYPED_ATTRIBUTES = 0,
    YVEX_CONVERSATION_TOOL_GRAMMAR_XML_ELEMENTS
} yvex_conversation_tool_grammar;

typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    const char *architecture;
    const char *source_revision;
    const char *source_encoding_path;
    const char *source_encoding_identity;
    const char *bos;
    const char *eos;
    const char *system;
    const char *user;
    const char *assistant;
    const char *message_end;
    const char *latest_reminder;
    const char *thinking_start;
    const char *thinking_start_suffix;
    const char *thinking_end_prefix;
    const char *thinking_end;
    const char *thinking_end_suffix;
    const char *tool_result_start;
    const char *tool_result_end;
    const char *tool_result_group_start;
    const char *dsml;
    const char *tool_calls_start;
    const char *tool_calls_end;
    const char *tool_invoke_start;
    const char *tool_invoke_name_end;
    const char *tool_invoke_end;
    const char *tool_parameter_start;
    const char *tool_parameter_name_end;
    const char *tool_parameter_kind_end;
    const char *tool_parameter_end;
    const char *reasoning_effort_low;
    const char *reasoning_effort_max;
    const char *tools_prefix;
    const char *tools_suffix;
    const char *response_format_prefix;
    yvex_conversation_grammar grammar;
    yvex_conversation_tool_grammar tool_grammar;
    yvex_reasoning_policy default_reasoning_policy;
    int drop_prior_reasoning_by_default;
    int tools_preserve_reasoning;
    int tool_results_merge_into_user;
    const char *tokenizer_model;
    const char *tokenizer_pre;
    const char *tokenizer_json_identity;
    const char *tokenizer_config_identity;
    unsigned long long vocabulary_size, base_vocabulary_size, merge_count;
    unsigned long long added_token_count, special_token_count;
    unsigned int bos_token_id, eos_token_id, pad_token_id, unk_token_id;
    int bos_present, eos_present, pad_present, unk_present;
    int add_bos_token, add_eos_token, byte_fallback;
} yvex_conversation_protocol;

#endif /* INCLUDE_YVEX_INTERNAL_CONVERSATION_H_INCLUDED */
