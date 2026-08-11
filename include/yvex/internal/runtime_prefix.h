/* Share one immutable committed state prefix across isolated runtime sessions. */
#ifndef INCLUDE_YVEX_INTERNAL_RUNTIME_PREFIX_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_RUNTIME_PREFIX_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_SESSION_PREFIX_SCHEMA_V1 1u

typedef struct yvex_runtime_execution_session yvex_runtime_execution_session;
typedef struct yvex_runtime_model_failure yvex_runtime_model_failure;
typedef struct yvex_runtime_session_prefix yvex_runtime_session_prefix;

typedef struct {
    unsigned int schema_version;
    unsigned long long scope_count, committed_sequence_length;
    unsigned long long shared_bytes, mapped_bytes, reference_count;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char target_prefix_identity[YVEX_SHA256_HEX_CAP];
    char draft_prefix_identity[YVEX_SHA256_HEX_CAP];
    char prefix_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_session_prefix_summary;

/* Capture and attach require idle sessions. Attach consumes an empty session
 * before any backend state residency has been prepared. */
int yvex_runtime_session_prefix_capture(
    yvex_runtime_execution_session *source,
    unsigned long long maximum_shared_bytes,
    yvex_runtime_session_prefix **out,
    yvex_runtime_session_prefix_summary *summary,
    yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_prefix_attach(
    yvex_runtime_execution_session *destination,
    const yvex_runtime_session_prefix *prefix,
    yvex_runtime_session_prefix_summary *summary,
    yvex_runtime_model_failure *failure, yvex_error *err);
void yvex_runtime_session_prefix_close(yvex_runtime_session_prefix **prefix);

#ifdef __cplusplus
}
#endif
#endif
