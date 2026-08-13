/* Compatibility adapters exchange bounded application intent through this transport-neutral ABI.
 * Requests and results are field-wise identity-sealed and contain neither engine ownership nor
 * transport pointers. */
#ifndef YVEX_PROVIDER_H
#define YVEX_PROVIDER_H

#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_PROVIDER_SCHEMA_V1 1u
#define YVEX_PROVIDER_SCHEMA_V2 2u
#define YVEX_PROVIDER_SCHEMA_V3 3u
#define YVEX_PROVIDER_WIRE_SCHEMA_V1 1u
#define YVEX_PROVIDER_WIRE_SCHEMA_V2 2u
#define YVEX_PROVIDER_WIRE_SCHEMA_V3 3u
#define YVEX_PROVIDER_MODEL_CAP 128u
#define YVEX_PROVIDER_ID_CAP 65u
#define YVEX_PROVIDER_ADAPTER_CAP 32u
#define YVEX_PROVIDER_TOOL_NAME_CAP 65u
#define YVEX_PROVIDER_MAX_MESSAGES 128u
#define YVEX_PROVIDER_MAX_TOOLS 32u
#define YVEX_PROVIDER_MAX_STOPS 4u
#define YVEX_PROVIDER_MAX_MESSAGE_BYTES 262144u
#define YVEX_PROVIDER_MAX_CONTENT_BYTES 1048576u
#define YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES 65536u
#define YVEX_PROVIDER_MAX_STOP_BYTES 256u
#define YVEX_PROVIDER_WIRE_MAX_BYTES 1048576u

typedef struct {
    const unsigned char *bytes;
    unsigned long long count;
} yvex_provider_span;

typedef enum {
    YVEX_PROVIDER_ROLE_DEVELOPER = 0,
    YVEX_PROVIDER_ROLE_SYSTEM,
    YVEX_PROVIDER_ROLE_USER,
    YVEX_PROVIDER_ROLE_ASSISTANT,
    YVEX_PROVIDER_ROLE_TOOL
} yvex_provider_role;

typedef struct {
    char call_id[YVEX_PROVIDER_ID_CAP];
    char name[YVEX_PROVIDER_TOOL_NAME_CAP];
    yvex_provider_span arguments_json;
} yvex_provider_tool_call;

typedef struct {
    yvex_provider_role role;
    yvex_provider_span content;
    yvex_provider_span reasoning_content;
    char tool_call_id[YVEX_PROVIDER_ID_CAP];
    const yvex_provider_tool_call *tool_calls;
    unsigned long long tool_call_count;
} yvex_provider_message;

typedef struct {
    char name[YVEX_PROVIDER_TOOL_NAME_CAP];
    yvex_provider_span description;
    yvex_provider_span parameters_json;
    int description_present, strict, strict_present;
} yvex_provider_function_tool;

typedef enum {
    YVEX_PROVIDER_TOOL_CHOICE_NONE = 0,
    YVEX_PROVIDER_TOOL_CHOICE_AUTO,
    YVEX_PROVIDER_TOOL_CHOICE_REQUIRED,
    YVEX_PROVIDER_TOOL_CHOICE_FUNCTION
} yvex_provider_tool_choice_kind;

typedef struct {
    yvex_provider_tool_choice_kind kind;
    char function_name[YVEX_PROVIDER_TOOL_NAME_CAP];
    int parallel_calls;
} yvex_provider_tool_choice;

typedef enum {
    YVEX_PROVIDER_RESPONSE_TEXT = 0,
    YVEX_PROVIDER_RESPONSE_JSON_OBJECT
} yvex_provider_response_format;

/* Source-authored prompt policy. It controls only reasoning text the active model explicitly
 * emits; it never authorizes reconstruction or exposure of hidden chain of thought. */
typedef enum {
    YVEX_REASONING_DISABLED = 0,
    YVEX_REASONING_ENABLED,
    YVEX_REASONING_MAXIMUM
} yvex_reasoning_policy;

typedef struct {
    int stochastic, seed_present;
    unsigned long long seed, top_k;
    double temperature, top_p, min_p, typical_p;
} yvex_provider_sampling;

typedef struct {
    unsigned int schema_version;
    char model[YVEX_PROVIDER_MODEL_CAP];
    const yvex_provider_message *messages;
    unsigned long long message_count;
    const yvex_provider_function_tool *tools;
    unsigned long long tool_count;
    const yvex_provider_span *stop_strings;
    unsigned long long stop_count;
    yvex_provider_tool_choice tool_choice;
    yvex_provider_response_format response_format;
    yvex_reasoning_policy reasoning_policy;
    yvex_provider_sampling sampling;
    unsigned long long maximum_output_tokens;
    int stream, include_usage, drop_thinking;
    char adapter[YVEX_PROVIDER_ADAPTER_CAP];
    char previous_response_id[YVEX_PROVIDER_ID_CAP];
    char external_correlation_id[YVEX_PROVIDER_ID_CAP];
    char request_identity[YVEX_PROVIDER_ID_CAP];
    int sealed;
} yvex_provider_request;

typedef enum {
    YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT = 0,
    YVEX_PROVIDER_OUTPUT_FUNCTION_CALL,
    YVEX_PROVIDER_OUTPUT_USAGE,
    YVEX_PROVIDER_OUTPUT_TERMINAL,
    YVEX_PROVIDER_OUTPUT_ERROR,
    YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING
} yvex_provider_output_kind;

typedef enum {
    YVEX_PROVIDER_FINISH_STOP = 0,
    YVEX_PROVIDER_FINISH_LENGTH,
    YVEX_PROVIDER_FINISH_TOOL_CALLS,
    YVEX_PROVIDER_FINISH_CANCELLED,
    YVEX_PROVIDER_FINISH_FAILED
} yvex_provider_finish_class;

typedef struct {
    unsigned int schema_version;
    yvex_provider_output_kind kind;
    yvex_provider_finish_class finish;
    yvex_provider_span bytes;
    yvex_provider_tool_call tool_call;
    unsigned long long prompt_tokens, completion_tokens, total_tokens;
    char request_identity[YVEX_PROVIDER_ID_CAP];
    char output_identity[YVEX_PROVIDER_ID_CAP];
    int completed;
} yvex_provider_output;

void yvex_provider_sampling_default(yvex_provider_sampling *sampling);
void yvex_provider_request_default(yvex_provider_request *request);

const char *yvex_provider_role_name(yvex_provider_role role);
const char *yvex_provider_finish_name(yvex_provider_finish_class finish);

int yvex_provider_request_seal(yvex_provider_request *request, yvex_error *err);
int yvex_provider_request_validate(const yvex_provider_request *request,
                                   yvex_error *err);
int yvex_provider_request_clone(const yvex_provider_request *source,
                                yvex_provider_request **out, yvex_error *err);
void yvex_provider_request_close(yvex_provider_request **request);

int yvex_provider_request_wire_encode(const yvex_provider_request *request,
                                      unsigned char *output,
                                      unsigned long long capacity,
                                      unsigned long long *byte_count,
                                      yvex_error *err);
int yvex_provider_request_wire_decode(const unsigned char *input,
                                      unsigned long long byte_count,
                                      yvex_provider_request **out,
                                      yvex_error *err);

int yvex_provider_json_value_validate(const unsigned char *bytes,
                                      unsigned long long byte_count,
                                      int require_object, yvex_error *err);
int yvex_provider_output_seal(yvex_provider_output *output, yvex_error *err);
int yvex_provider_output_validate(const yvex_provider_output *output,
                                  yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* YVEX_PROVIDER_H */
