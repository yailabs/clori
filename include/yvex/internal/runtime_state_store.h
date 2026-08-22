/* Versioned committed model-state checkpoints bound to one admitted runtime model. */
#ifndef INCLUDE_YVEX_INTERNAL_RUNTIME_STATE_STORE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_RUNTIME_STATE_STORE_H_INCLUDED

#include <yvex/internal/generation.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_STATE_STORE_SCHEMA_V1 1u
#define YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 2u

#define YVEX_RUNTIME_GENERATION_CHECKPOINT_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    yvex_runtime_sampling_checkpoint sampling;
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
    char checkpoint_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_checkpoint;

typedef struct {
    unsigned int schema_version;
    unsigned long long file_bytes, scope_count, committed_sequence_length;
    unsigned long long payload_bytes;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char payload_identity[YVEX_SHA256_HEX_CAP];
    char file_digest[YVEX_SHA256_HEX_CAP];
} yvex_runtime_state_store_summary;

typedef struct {
    unsigned char *bytes;
    unsigned long long byte_count;
    char payload_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_state_store_payload;

int yvex_runtime_generation_context_checkpoint(
    yvex_runtime_generation_context *context,
    yvex_runtime_generation_checkpoint *checkpoint, yvex_error *err);
int yvex_runtime_generation_context_restore(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_checkpoint *checkpoint, yvex_error *err);

/* Save and restore require an idle session. Publication is atomic and restore
 * is admitted only after the complete digest and bound identities validate. */
int yvex_runtime_session_state_save(
    yvex_runtime_execution_session *session, const char *path,
    const void *payload, unsigned long long payload_bytes,
    yvex_runtime_state_store_summary *summary, yvex_error *err);
int yvex_runtime_session_state_inspect(
    yvex_runtime_execution_session *session, const char *path,
    unsigned long long maximum_file_bytes,
    yvex_runtime_state_store_payload *payload,
    yvex_runtime_state_store_summary *summary, yvex_error *err);
int yvex_runtime_session_state_restore(
    yvex_runtime_execution_session *session, const char *path,
    unsigned long long maximum_file_bytes,
    unsigned long long expected_committed_sequence_length,
    const char *expected_file_digest, const char *expected_payload_identity,
    yvex_runtime_state_store_summary *summary, yvex_error *err);
void yvex_runtime_state_store_payload_close(
    yvex_runtime_state_store_payload *payload);

#ifdef __cplusplus
}
#endif
#endif
