/* Tokenization and prompt rendering are derived from admitted artifact metadata and share one
 * immutable plan. Producing token IDs or text fragments neither mutates model state nor selects a
 * generated token. */
#ifndef YVEX_TOKENIZER_H
#define YVEX_TOKENIZER_H

#include <yvex/artifact.h>
#include <yvex/core.h>
#include <yvex/model.h>
#include <yvex/provider.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_gguf yvex_gguf;

/* Tokenizer. */
typedef struct yvex_tokenizer yvex_tokenizer;

typedef enum {
    YVEX_TOKENIZER_KIND_UNKNOWN = 0,
    YVEX_TOKENIZER_KIND_GGML_LLAMA,
    YVEX_TOKENIZER_KIND_GGML_GPT2,
    YVEX_TOKENIZER_KIND_GGML_REPLIT,
    YVEX_TOKENIZER_KIND_GGML_RWKV,
    YVEX_TOKENIZER_KIND_HUGGINGFACE_JSON,
    YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE
} yvex_tokenizer_kind;

typedef enum {
    YVEX_TOKENIZER_SUPPORT_METADATA_ONLY = 0,
    YVEX_TOKENIZER_SUPPORT_VOCAB_ONLY,
    YVEX_TOKENIZER_SUPPORT_FIXTURE_ENCODE_DECODE,
    YVEX_TOKENIZER_SUPPORT_ARTIFACT_BPE,
    YVEX_TOKENIZER_SUPPORT_UNSUPPORTED
} yvex_tokenizer_support;

typedef enum {
    YVEX_TOKEN_TYPE_UNKNOWN = 0,
    YVEX_TOKEN_TYPE_NORMAL = 1,
    YVEX_TOKEN_TYPE_UNK = 2,
    YVEX_TOKEN_TYPE_CONTROL = 3,
    YVEX_TOKEN_TYPE_USER_DEFINED = 4,
    YVEX_TOKEN_TYPE_UNUSED = 5,
    YVEX_TOKEN_TYPE_BYTE = 6
} yvex_token_type;

typedef struct {
    unsigned int id;
    const char *text;
    unsigned long long text_len;
    float score;
    yvex_token_type type;
} yvex_token_info;

typedef struct {
    unsigned int *ids;
    unsigned long long len;
    unsigned long long cap;
} yvex_tokens;

#define YVEX_TOKENIZER_PLAN_SCHEMA_V1 1u
#define YVEX_TOKENIZER_PLAN_SCHEMA_V2 2u
#define YVEX_TOKENIZER_PLAN_SCHEMA_V3 3u
#define YVEX_TOKENIZER_EXECUTION_SCHEMA_V1 1u
#define YVEX_TOKENIZER_DECODER_SCHEMA_V1 1u
#define YVEX_TOKENIZER_APPEND_SCHEMA_V1 1u
#define YVEX_TOKENIZER_PROVIDER_RESULT_SCHEMA_V1 1u
#define YVEX_TOKENIZER_PROVIDER_RESULT_SCHEMA_V2 2u

typedef enum {
    YVEX_TOKENIZER_MODEL_FIXTURE = 0,
    YVEX_TOKENIZER_MODEL_BPE_BYTELEVEL
} yvex_tokenizer_model_policy;

typedef enum {
    YVEX_TOKENIZER_PROMPT_NONE = 0,
    YVEX_TOKENIZER_PROMPT_DEEPSEEK_V4
} yvex_tokenizer_prompt_policy;

typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long vocabulary_size, base_vocabulary_size, merge_count;
    unsigned long long added_token_count, special_token_count, owned_bytes;
    unsigned int bos_token_id, eos_token_id, pad_token_id, unk_token_id;
    unsigned int reasoning_start_token_id, reasoning_end_token_id;
    int bos_present, eos_present, pad_present, unk_present;
    int add_bos_token, add_eos_token, byte_fallback, sealed, runtime_bound;
    int explicit_reasoning_supported, maximum_reasoning_supported;
    yvex_tokenizer_model_policy model_policy;
    yvex_tokenizer_prompt_policy prompt_policy;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_json_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_config_identity[YVEX_SHA256_HEX_CAP];
    char vocabulary_identity[YVEX_SHA256_HEX_CAP];
    char merge_table_identity[YVEX_SHA256_HEX_CAP];
    char added_token_identity[YVEX_SHA256_HEX_CAP];
    char special_policy_identity[YVEX_SHA256_HEX_CAP];
    char prompt_policy_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_plan_identity[YVEX_SHA256_HEX_CAP];
} yvex_tokenizer_plan_summary;

typedef struct {
    int add_bos, add_eos, allow_special_tokens;
    unsigned long long maximum_tokens;
} yvex_tokenizer_encode_options;

typedef struct {
    unsigned int schema_version;
    yvex_tokens tokens;
    unsigned long long input_bytes, added_token_matches, byte_fallback_count;
    int bos_inserted, eos_inserted, completed;
    char input_identity[YVEX_SHA256_HEX_CAP];
    char token_ids_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_plan_identity[YVEX_SHA256_HEX_CAP];
    char encoding_identity[YVEX_SHA256_HEX_CAP];
} yvex_tokenizer_encode_result;

typedef int (*yvex_tokenizer_cancel_fn)(void *context);

typedef struct {
    int skip_special_tokens, require_complete_utf8;
    yvex_tokenizer_cancel_fn cancelled;
    void *cancel_context;
} yvex_tokenizer_decode_options;

typedef struct {
    unsigned int schema_version;
    unsigned char *bytes;
    unsigned long long byte_count, token_count, suppressed_special_count;
    int completed;
    char token_ids_identity[YVEX_SHA256_HEX_CAP];
    char decoded_bytes_identity[YVEX_SHA256_HEX_CAP];
    char decoder_identity[YVEX_SHA256_HEX_CAP];
} yvex_tokenizer_decode_result;

typedef struct yvex_tokenizer_decoder yvex_tokenizer_decoder;
typedef struct yvex_tokenizer_decoder_transaction yvex_tokenizer_decoder_transaction;

typedef struct {
    unsigned int schema_version, token_id;
    unsigned char *bytes;
    unsigned long long byte_count, pending_byte_count, processed_token_count;
    int special, eos, suppressed, completed;
    char state_before_identity[YVEX_SHA256_HEX_CAP];
    char state_after_identity[YVEX_SHA256_HEX_CAP];
    char fragment_identity[YVEX_SHA256_HEX_CAP];
} yvex_tokenizer_fragment;

typedef struct {
    unsigned int token_id;
    int special, eos, pad, unknown, stop, suppressed_by_default;
} yvex_tokenizer_token_classification;

typedef enum {
    YVEX_TOKEN_APPEND_PROPOSED = 0,
    YVEX_TOKEN_APPEND_APPENDED,
    YVEX_TOKEN_APPEND_SUBMITTED,
    YVEX_TOKEN_APPEND_MODEL_COMMITTED,
    YVEX_TOKEN_APPEND_DETOKENIZED,
    YVEX_TOKEN_APPEND_TEXT_PUBLISHED
} yvex_token_append_state;

typedef struct yvex_token_sequence yvex_token_sequence;
typedef struct yvex_token_sequence_transaction yvex_token_sequence_transaction;

typedef struct {
    unsigned int schema_version;
    unsigned long long count, capacity, generation;
    char token_ids_identity[YVEX_SHA256_HEX_CAP];
    char state_identity[YVEX_SHA256_HEX_CAP];
} yvex_token_sequence_summary;

/* Bounded explicit token input used by tokenizer and graph inspection owners. */
#define YVEX_TOKEN_INPUT_MAX_TOKENS 1024ull

typedef enum {
    YVEX_TOKEN_INPUT_EXPLICIT = 1,
    YVEX_TOKEN_INPUT_PROMPT_TEXT = 2
} yvex_token_input_kind;

typedef struct {
    yvex_token_input_kind kind;
    unsigned long long token_count;
    unsigned int tokens[YVEX_TOKEN_INPUT_MAX_TOKENS];
    unsigned long long max_tokens;
    int token_bounds_checked;
    int token_bounds_valid;
    unsigned long long vocab_size;
} yvex_token_input;

void yvex_token_input_init(yvex_token_input *input, yvex_token_input_kind kind);
const char *yvex_token_input_kind_name(yvex_token_input_kind kind);
int yvex_token_input_parse_explicit(const char *text, yvex_token_input *out, yvex_error *err);
int yvex_token_input_from_ids(yvex_token_input_kind kind, const unsigned int *ids,
                              unsigned long long count, yvex_token_input *out, yvex_error *err);
int yvex_token_input_validate_bounds(yvex_token_input *input, unsigned long long vocab_size,
                                     yvex_error *err);
int yvex_token_input_select(const yvex_token_input *input, unsigned long long token_index,
                            unsigned int *out_token, yvex_error *err);

int yvex_tokenizer_from_gguf(yvex_tokenizer **out,
                             const yvex_gguf *gguf,
                             const yvex_model_descriptor *model,
                             yvex_error *err);

void yvex_tokenizer_close(yvex_tokenizer *tokenizer);

const yvex_tokenizer_plan_summary *yvex_tokenizer_plan_summary_get(
    const yvex_tokenizer *tokenizer);
int yvex_tokenizer_bind_runtime(yvex_tokenizer *tokenizer,
                                const char *artifact_identity,
                                const char *logical_model_identity,
                                const char *runtime_descriptor_identity,
                                yvex_error *err);

yvex_tokenizer_kind yvex_tokenizer_kind_of(const yvex_tokenizer *tokenizer);
yvex_tokenizer_support yvex_tokenizer_support_of(const yvex_tokenizer *tokenizer);
const char *yvex_tokenizer_kind_name(yvex_tokenizer_kind kind);
const char *yvex_tokenizer_support_name(yvex_tokenizer_support support);

unsigned long long yvex_tokenizer_vocab_size(const yvex_tokenizer *tokenizer);
const yvex_token_info *yvex_tokenizer_token_at(const yvex_tokenizer *tokenizer,
                                               unsigned long long id);

int yvex_tokenizer_bos_id(const yvex_tokenizer *tokenizer, unsigned int *out);
int yvex_tokenizer_eos_id(const yvex_tokenizer *tokenizer, unsigned int *out);
int yvex_tokenizer_unk_id(const yvex_tokenizer *tokenizer, unsigned int *out);
int yvex_tokenizer_pad_id(const yvex_tokenizer *tokenizer, unsigned int *out);

int yvex_tokenizer_chat_template(const yvex_tokenizer *tokenizer,
                                 const char **data,
                                 unsigned long long *len);

int yvex_tokenize_text(const yvex_tokenizer *tokenizer,
                       const char *text,
                       yvex_tokens *out,
                       yvex_error *err);

int yvex_tokenizer_encode(const yvex_tokenizer *tokenizer,
                          const unsigned char *bytes,
                          unsigned long long byte_count,
                          const yvex_tokenizer_encode_options *options,
                          yvex_tokenizer_encode_result *result,
                          yvex_error *err);
void yvex_tokenizer_encode_result_clear(yvex_tokenizer_encode_result *result);

int yvex_detokenize_ids(const yvex_tokenizer *tokenizer,
                        const unsigned int *ids,
                        unsigned long long len,
                        char *out,
                        unsigned long long cap,
                        yvex_error *err);

int yvex_tokenizer_decode(const yvex_tokenizer *tokenizer,
                          const unsigned int *ids,
                          unsigned long long count,
                          const yvex_tokenizer_decode_options *options,
                          yvex_tokenizer_decode_result *result,
                          yvex_error *err);
void yvex_tokenizer_decode_result_clear(yvex_tokenizer_decode_result *result);
int yvex_tokenizer_decoder_open(yvex_tokenizer_decoder **out,
                                const yvex_tokenizer *tokenizer,
                                const yvex_tokenizer_decode_options *options,
                                yvex_error *err);
int yvex_tokenizer_decoder_push(yvex_tokenizer_decoder *decoder,
                                unsigned int token_id,
                                yvex_tokenizer_fragment *fragment,
                                yvex_error *err);
/*
 * A decoder transaction makes a complete candidate prefix readable before any decoder state is
 * published. Preparing retains exclusive decoder admission until publish or abort, allowing a
 * caller to coordinate text state with a separate model-state transaction.
 */
int yvex_tokenizer_decoder_transaction_begin(
    yvex_tokenizer_decoder *decoder,
    yvex_tokenizer_decoder_transaction **out, yvex_error *err);
int yvex_tokenizer_decoder_transaction_push(
    yvex_tokenizer_decoder_transaction *transaction, unsigned int token_id,
    yvex_tokenizer_fragment *fragment, yvex_error *err);
int yvex_tokenizer_decoder_transaction_prepare(
    yvex_tokenizer_decoder_transaction *transaction, yvex_error *err);
void yvex_tokenizer_decoder_transaction_publish(
    yvex_tokenizer_decoder_transaction **transaction);
void yvex_tokenizer_decoder_transaction_abort(
    yvex_tokenizer_decoder_transaction **transaction);
int yvex_tokenizer_decoder_finish(yvex_tokenizer_decoder *decoder,
                                  yvex_tokenizer_fragment *fragment,
                                  yvex_error *err);
int yvex_tokenizer_decoder_reset(yvex_tokenizer_decoder *decoder,
                                 yvex_error *err);
void yvex_tokenizer_fragment_clear(yvex_tokenizer_fragment *fragment);
void yvex_tokenizer_decoder_close(yvex_tokenizer_decoder **decoder);
int yvex_tokenizer_token_classify(
    const yvex_tokenizer *tokenizer, unsigned int token_id,
    yvex_tokenizer_token_classification *classification, yvex_error *err);

int yvex_token_sequence_open(yvex_token_sequence **out,
                             unsigned long long capacity,
                             yvex_error *err);
int yvex_token_sequence_append(yvex_token_sequence *sequence,
                               unsigned int token_id,
                               unsigned long long vocabulary_size,
                               unsigned long long *ordinal,
                               yvex_error *err);
/* Stage a bounded append prefix so all token-ledger rows become visible together. */
int yvex_token_sequence_transaction_begin(
    yvex_token_sequence *sequence, unsigned long long maximum_rows,
    yvex_token_sequence_transaction **out, yvex_error *err);
int yvex_token_sequence_transaction_append(
    yvex_token_sequence_transaction *transaction, unsigned int token_id,
    unsigned long long vocabulary_size, unsigned long long *ordinal,
    yvex_error *err);
int yvex_token_sequence_transaction_transition(
    yvex_token_sequence_transaction *transaction, unsigned long long ordinal,
    yvex_token_append_state expected, yvex_token_append_state next,
    yvex_error *err);
int yvex_token_sequence_transaction_prepare(
    yvex_token_sequence_transaction *transaction, yvex_error *err);
void yvex_token_sequence_transaction_publish(
    yvex_token_sequence_transaction **transaction);
void yvex_token_sequence_transaction_abort(
    yvex_token_sequence_transaction **transaction);
int yvex_token_sequence_transition(yvex_token_sequence *sequence,
                                   unsigned long long ordinal,
                                   yvex_token_append_state expected,
                                   yvex_token_append_state next,
                                   yvex_error *err);
int yvex_token_sequence_summary_get(const yvex_token_sequence *sequence,
                                    yvex_token_sequence_summary *summary,
                                    yvex_error *err);
int yvex_token_sequence_reset(yvex_token_sequence *sequence,
                              yvex_error *err);
void yvex_token_sequence_close(yvex_token_sequence **sequence);

void yvex_tokens_clear(yvex_tokens *tokens);
void yvex_tokens_free(yvex_tokens *tokens);

/* Prompt rendering. */
typedef enum {
    YVEX_PROMPT_ROLE_SYSTEM = 0,
    YVEX_PROMPT_ROLE_USER,
    YVEX_PROMPT_ROLE_ASSISTANT,
    YVEX_PROMPT_ROLE_TOOL
} yvex_prompt_role;

typedef struct {
    yvex_prompt_role role;
    const char *content;
    unsigned long long content_len;
} yvex_prompt_message;

typedef enum {
    YVEX_PROMPT_MODE_CHAT = 0,
    YVEX_PROMPT_MODE_THINKING
} yvex_prompt_mode;

typedef struct {
    int add_bos;
    int add_eos;
    int add_generation_prompt;
    int drop_thinking;
    yvex_prompt_mode mode;
    yvex_reasoning_policy reasoning_policy;
} yvex_prompt_options;

typedef struct {
    char *text;
    unsigned long long len;
    int generation_prompt;
    char message_sequence_identity[YVEX_SHA256_HEX_CAP];
    char rendered_bytes_identity[YVEX_SHA256_HEX_CAP];
    char prompt_identity[YVEX_SHA256_HEX_CAP];
} yvex_rendered_prompt;

typedef enum {
    YVEX_REASONING_SEGMENT_FINAL_TEXT = 0,
    YVEX_REASONING_SEGMENT_EXPLICIT
} yvex_reasoning_segment;

typedef struct yvex_tokenizer_reasoning_stream yvex_tokenizer_reasoning_stream;
typedef int (*yvex_tokenizer_reasoning_sink)(
    void *context, yvex_reasoning_segment segment, const unsigned char *bytes,
    unsigned long long byte_count, yvex_error *err);

/* Classification is enabled only by an admitted source-authored prompt policy. The tokenizer
 * consumes its own delimiter and never reclassifies arbitrary final prose by style. */
int yvex_tokenizer_reasoning_stream_open(
    yvex_tokenizer_reasoning_stream **out, const yvex_tokenizer *tokenizer,
    yvex_reasoning_policy policy, yvex_tokenizer_reasoning_sink sink,
    void *sink_context, yvex_error *err);
int yvex_tokenizer_reasoning_stream_push(
    yvex_tokenizer_reasoning_stream *stream, const unsigned char *bytes,
    unsigned long long byte_count, yvex_error *err);
int yvex_tokenizer_reasoning_stream_finish(
    yvex_tokenizer_reasoning_stream *stream, yvex_error *err);
void yvex_tokenizer_reasoning_stream_close(
    yvex_tokenizer_reasoning_stream **stream);

const char *yvex_prompt_role_name(yvex_prompt_role role);

int yvex_prompt_render(yvex_rendered_prompt *out,
                       const yvex_tokenizer *tokenizer,
                       const yvex_prompt_message *messages,
                       unsigned long long message_count,
                       const yvex_prompt_options *options,
                       yvex_error *err);

int yvex_tokenizer_encode_prompt(const yvex_tokenizer *tokenizer,
                                 const yvex_prompt_message *messages,
                                 unsigned long long message_count,
                                 const yvex_prompt_options *prompt_options,
                                 const yvex_tokenizer_encode_options *encode_options,
                                 yvex_rendered_prompt *rendered,
                                 yvex_tokenizer_encode_result *encoded,
                                 yvex_error *err);

void yvex_rendered_prompt_free(yvex_rendered_prompt *prompt);

typedef struct {
    unsigned int schema_version;
    yvex_provider_output_kind kind;
    unsigned char *reasoning_content;
    unsigned long long reasoning_content_count;
    unsigned char *content;
    unsigned long long content_count;
    yvex_provider_tool_call *tool_calls;
    unsigned long long tool_call_count;
    char output_identity[YVEX_SHA256_HEX_CAP];
    int completed;
} yvex_tokenizer_provider_result;

int yvex_tokenizer_provider_prompt(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    yvex_rendered_prompt *rendered, yvex_error *err);
int yvex_tokenizer_encode_provider_prompt(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    const yvex_tokenizer_encode_options *encode_options,
    yvex_rendered_prompt *rendered, yvex_tokenizer_encode_result *encoded,
    yvex_error *err);
int yvex_tokenizer_parse_provider_completion(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    const unsigned char *bytes, unsigned long long byte_count,
    yvex_tokenizer_provider_result *result, yvex_error *err);
void yvex_tokenizer_provider_result_clear(
    yvex_tokenizer_provider_result *result);

#ifdef __cplusplus
}
#endif

#endif /* YVEX_TOKENIZER_H */
