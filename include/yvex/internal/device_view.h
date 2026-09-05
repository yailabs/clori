/* Checked device-value views used by runtime operators within one authenticated engine. */
#ifndef INCLUDE_YVEX_INTERNAL_DEVICE_VIEW_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_DEVICE_VIEW_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1 1u

typedef enum {
    YVEX_EXECUTION_DEVICE_HIDDEN = 0,
    YVEX_EXECUTION_DEVICE_EXPANDED_RESIDUAL,
    YVEX_EXECUTION_DEVICE_FEATURE_TAP,
    YVEX_EXECUTION_DEVICE_LOGITS,
    YVEX_EXECUTION_DEVICE_PROBABILITIES,
    YVEX_EXECUTION_DEVICE_CANDIDATE_STATE,
    YVEX_EXECUTION_DEVICE_ACCEPTED_PREFIX,
    YVEX_EXECUTION_DEVICE_WORKSPACE
} yvex_execution_device_value_kind;

typedef enum {
    YVEX_EXECUTION_MATERIALIZE_NONE = 0,
    YVEX_EXECUTION_MATERIALIZE_SCALAR,
    YVEX_EXECUTION_MATERIALIZE_ROW,
    YVEX_EXECUTION_MATERIALIZE_AUDIT_DIGEST,
    YVEX_EXECUTION_MATERIALIZE_FORENSIC_FULL
} yvex_execution_materialization_policy;

typedef struct {
    unsigned int schema_version;
    yvex_execution_device_value_kind kind;
    yvex_backend *backend;
    const yvex_device_tensor *tensor;
    unsigned long long element_offset;
    unsigned long long resource_generation, session_generation, state_generation;
    unsigned long long rows, columns, element_bytes;
    yvex_dtype dtype;
    int mutable_view, synchronization_required;
    yvex_execution_materialization_policy materialization;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char execution_profile_identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_device_view;

int yvex_execution_device_view_validate(
    const yvex_execution_device_view *view, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_DEVICE_VIEW_H_INCLUDED */
