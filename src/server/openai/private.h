/* Owner: server.openai source-local interface.
 * Owns: bounded HTTP request/result storage and collaboration between OpenAI adapter owners.
 * Does not own: provider semantics, YVEX protocol framing, model execution, or installed ABI.
 * Invariants: only OpenAI adapter translation units consume these daemon-local structures.
 * Boundary: OpenAI syntax remains below this interface and never enters runtime owners.
 * Purpose: share the smallest in-process adapter contracts across independently compiled owners.
 * Inputs: loopback descriptors, explicit bytes, provider requests, and YVEX protocol messages.
 * Effects: declarations only.
 * Failure: implementations own and clear all allocations described here. */
#ifndef SRC_SERVER_OPENAI_PRIVATE_H_INCLUDED
#define SRC_SERVER_OPENAI_PRIVATE_H_INCLUDED
#include <stdatomic.h>
#include <stddef.h>
#include "src/server/private.h"
#include <yvex/provider.h>
#include <yvex/server.h>
#define OPENAI_COMPAT_PROFILE "yvex.openai.compat.v1"
#define OPENAI_HTTP_BODY_MAX YVEX_PROVIDER_WIRE_MAX_BYTES
#define OPENAI_HTTP_HEADER_MAX 32768u
#define OPENAI_HTTP_HEADER_COUNT_MAX 64u
#define OPENAI_RESPONSE_RECORD_MAX 64u
#define OPENAI_RESPONSE_TTL_SECONDS 3600u
typedef enum {
    OPENAI_ENDPOINT_HEALTH = 0,
    OPENAI_ENDPOINT_MODELS,
    OPENAI_ENDPOINT_MODEL,
    OPENAI_ENDPOINT_CHAT,
    OPENAI_ENDPOINT_RESPONSES
} openai_endpoint;
typedef enum {
    OPENAI_RESPONSE_EVENT_CREATED = 0,
    OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_ADDED,
    OPENAI_RESPONSE_EVENT_CONTENT_PART_ADDED,
    OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DELTA,
    OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DONE,
    OPENAI_RESPONSE_EVENT_CONTENT_PART_DONE,
    OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA,
    OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DONE,
    OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_DONE,
    OPENAI_RESPONSE_EVENT_COMPLETED,
    OPENAI_RESPONSE_EVENT_INCOMPLETE,
    OPENAI_RESPONSE_EVENT_FAILED
} openai_response_event_kind;
typedef struct {
    char method[8];
    char path[256];
    unsigned char *body;
    unsigned long long body_count;
} openai_http_request;
typedef struct {
    int fd;
    int headers_sent;
    int stream;
    openai_endpoint endpoint;
    unsigned long long response_sequence;
    int response_item_started;
} openai_http_sink;
typedef struct {
    yvex_provider_request *provider;
    openai_endpoint endpoint;
} openai_admitted_request;
typedef struct {
    unsigned char *text, *arguments;
    unsigned long long text_count, text_capacity;
    unsigned long long arguments_count, arguments_capacity;
    unsigned long long prompt_tokens, completion_tokens, total_tokens;
    yvex_provider_finish_class finish;
    char tool_call_id[YVEX_PROVIDER_ID_CAP];
    char tool_name[YVEX_PROVIDER_TOOL_NAME_CAP];
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    char turn_identity[YVEX_SHA256_HEX_CAP];
    yvex_client_failure_class failure_class;
    int has_tool_call, complete;
} openai_generation_result;
typedef struct {
    int occupied;
    char response_id[YVEX_PROVIDER_ID_CAP];
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    char model[YVEX_PROVIDER_MODEL_CAP];
    yvex_provider_request *context;
    unsigned long long created_seconds, last_used_sequence;
} openai_response_record;
typedef struct {
    char host[64];
    unsigned short port;
    unsigned long long yvex_timeout_ms;
    atomic_int *stop;
    char yvex_socket[YVEX_SERVER_SOCKET_PATH_CAP];
    unsigned long long next_id, request_count;
    server_telemetry *telemetry;
    openai_response_record records[OPENAI_RESPONSE_RECORD_MAX];
} openai_gateway;
int openai_http_read(int fd, openai_http_request *request, yvex_error *err);
void openai_http_request_clear(openai_http_request *request);
int openai_http_json(int fd, int status, const unsigned char *body,
                     unsigned long long count, yvex_error *err);
int openai_http_sse_begin(int fd, yvex_error *err);
int openai_http_sse_event(int fd, const char *event,
                          const unsigned char *json,
                          unsigned long long count, yvex_error *err);
int openai_http_sse_done(int fd, yvex_error *err);
int openai_http_peer_wait(int fd, unsigned int milliseconds, int *closed,
                          yvex_error *err);
int openai_json_admit(const openai_http_request *http,
                      openai_endpoint endpoint, const char *model,
                      openai_admitted_request *request, yvex_error *err);
void openai_admitted_request_clear(openai_admitted_request *request);
int openai_json_error(int status, const char *type, const char *param,
                      const char *code, const char *message,
                      unsigned char **output, unsigned long long *count,
                      yvex_error *err);
int openai_json_models(const yvex_server_summary *summary,
                       const char *selected_model, int list,
                       unsigned char **output, unsigned long long *count,
                       yvex_error *err);
int openai_json_result(openai_endpoint endpoint, const char *id,
                       const char *model, unsigned long long created,
                       const openai_generation_result *result,
                       unsigned char **output, unsigned long long *count,
                       yvex_error *err);
int openai_json_stream_chunk(openai_endpoint endpoint, const char *id,
                             const char *model, unsigned long long created,
                             const yvex_client_message *message, int initial,
                             unsigned char **output, unsigned long long *count,
                             yvex_error *err);
int openai_json_chat_usage_chunk(const char *id, const char *model,
                                 unsigned long long created,
                                 const openai_generation_result *result,
                                 unsigned char **output,
                                 unsigned long long *count, yvex_error *err);
int openai_json_response_event(openai_response_event_kind kind,
                               const char *id, const char *model,
                               unsigned long long created,
                               const yvex_client_message *message,
                               const openai_generation_result *result,
                               unsigned long long sequence,
                               unsigned char **output,
                               unsigned long long *count, yvex_error *err);
openai_response_record *openai_state_find(openai_gateway *gateway,
                                          const char *response_id,
                                          unsigned long long now);
int openai_state_store(openai_gateway *gateway, const char *response_id,
                       const char *session_name,
                       const yvex_provider_request *context,
                       unsigned long long now, yvex_error *err);
int openai_state_replace(openai_gateway *gateway,
                         openai_response_record *record,
                         const char *response_id,
                         const yvex_provider_request *context,
                         unsigned long long now, yvex_error *err);
void openai_state_remove(openai_response_record *record);
void openai_state_clear(openai_gateway *gateway);
#endif
