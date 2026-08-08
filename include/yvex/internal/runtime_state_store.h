/* Versioned committed model-state checkpoints bound to one admitted runtime model. */
#ifndef INCLUDE_YVEX_INTERNAL_RUNTIME_STATE_STORE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_RUNTIME_STATE_STORE_H_INCLUDED

#include <yvex/internal/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_STATE_STORE_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    unsigned long long file_bytes, scope_count, committed_sequence_length;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char file_digest[YVEX_SHA256_HEX_CAP];
} yvex_runtime_state_store_summary;

/* Save and restore require an idle session. Publication is atomic and restore
 * is admitted only after the complete digest and bound identities validate. */
int yvex_runtime_session_state_save(
    yvex_runtime_execution_session *session, const char *path,
    yvex_runtime_state_store_summary *summary, yvex_error *err);
int yvex_runtime_session_state_restore(
    yvex_runtime_execution_session *session, const char *path,
    unsigned long long maximum_file_bytes,
    unsigned long long expected_committed_sequence_length,
    yvex_runtime_state_store_summary *summary, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
