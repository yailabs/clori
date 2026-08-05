/*
 * Family projections publish source-authored conversation syntax through this immutable view.
 * Tokenizer and provider owners interpret the view; they never select a family or reconstruct its
 * special tokens. Returned strings have process lifetime and contain exact UTF-8 bytes.
 */
#ifndef INCLUDE_YVEX_INTERNAL_CONVERSATION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_CONVERSATION_H_INCLUDED

#define YVEX_CONVERSATION_PROTOCOL_SCHEMA_V1 1u

typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    const char *architecture;
    const char *source_revision;
    const char *source_encoding_path;
    const char *source_encoding_identity;
    const char *bos;
    const char *eos;
    const char *user;
    const char *assistant;
    const char *latest_reminder;
    const char *thinking_start;
    const char *thinking_end;
    const char *tool_result_start;
    const char *tool_result_end;
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
    const char *reasoning_effort_max;
    const char *tools_prefix;
    const char *tools_suffix;
    const char *response_format_prefix;
    int drop_prior_reasoning_by_default;
    int tools_preserve_reasoning;
    int tool_results_merge_into_user;
} yvex_conversation_protocol;

/* Generic consumers enumerate this model-owned registry and match architecture facts. */
const yvex_conversation_protocol *
yvex_model_conversation_protocol_at(unsigned long long index);

#endif /* INCLUDE_YVEX_INTERNAL_CONVERSATION_H_INCLUDED */
