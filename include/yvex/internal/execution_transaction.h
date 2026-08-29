/* Govern retained iterative execution without owning family semantics or backend policy. */
#ifndef INCLUDE_YVEX_INTERNAL_EXECUTION_TRANSACTION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_EXECUTION_TRANSACTION_H_INCLUDED

#include <yvex/internal/core.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_runtime_execution_transaction yvex_runtime_execution_transaction;

typedef enum {
    YVEX_EXECUTION_TRANSACTION_PREPARED = 1,
    YVEX_EXECUTION_TRANSACTION_EXECUTING,
    YVEX_EXECUTION_TRANSACTION_SAFE,
    YVEX_EXECUTION_TRANSACTION_YIELDED,
    YVEX_EXECUTION_TRANSACTION_READY_TO_COMMIT,
    YVEX_EXECUTION_TRANSACTION_COMMITTED,
    YVEX_EXECUTION_TRANSACTION_ABORTED
} yvex_execution_transaction_state;

typedef enum {
    YVEX_EXECUTION_SAFE_POINT_CONTINUE = 1,
    YVEX_EXECUTION_SAFE_POINT_YIELD,
    YVEX_EXECUTION_SAFE_POINT_CANCEL,
    YVEX_EXECUTION_SAFE_POINT_COMMIT
} yvex_execution_safe_point_action;

typedef int (*yvex_execution_resource_retain_fn)(void *, yvex_error *);
typedef int (*yvex_execution_resource_release_fn)(void *, yvex_error *);

/* The resource owner retains its process-local state for the transaction lifetime. */
typedef struct {
    const char *identity;
    void *context;
    yvex_execution_resource_retain_fn retain;
    yvex_execution_resource_release_fn release;
} yvex_execution_resource_lease;

typedef int (*yvex_execution_control_requested_fn)(void *);
typedef int (*yvex_execution_quantum_execute_fn)(
    void *, unsigned long long, yvex_error *);
typedef int (*yvex_execution_transaction_publish_fn)(void *, yvex_error *);
typedef void (*yvex_execution_transaction_discard_fn)(void *);

typedef struct {
    const char *request_identity;
    unsigned long long quantum_count;
    const yvex_execution_resource_lease *resource;
    yvex_execution_quantum_execute_fn execute;
    void *execution_context;
    yvex_execution_control_requested_fn cancel_requested;
    void *cancel_context;
    yvex_execution_control_requested_fn yield_requested;
    void *yield_context;
    yvex_execution_transaction_publish_fn publish;
    yvex_execution_transaction_discard_fn discard;
    void *publication_context;
} yvex_execution_transaction_options;

typedef struct {
    yvex_execution_transaction_state state;
    unsigned long long admitted_quanta, started_quanta, completed_quanta;
    unsigned long long safe_points, yields, cancellations;
    unsigned long long publications, discards, retained_resources;
    unsigned long long setup_nanoseconds, quantum_wall_nanoseconds;
    unsigned long long safe_point_nanoseconds;
    char request_identity[YVEX_SHA256_HEX_BYTES];
    char resource_identity[YVEX_SHA256_HEX_BYTES];
} yvex_execution_transaction_summary;

/* One caller thread drives a transaction. Callback contexts remain borrowed until close, while
 * the resource lease keeps its owner alive across every admitted quantum. */
int yvex_runtime_execution_transaction_open(
    yvex_runtime_execution_transaction **out,
    const yvex_execution_transaction_options *options, yvex_error *err);
int yvex_runtime_execution_transaction_execute_quantum(
    yvex_runtime_execution_transaction *transaction,
    yvex_execution_safe_point_action *action, yvex_error *err);
int yvex_runtime_execution_transaction_resume(
    yvex_runtime_execution_transaction *transaction, yvex_error *err);
int yvex_runtime_execution_transaction_commit(
    yvex_runtime_execution_transaction *transaction, yvex_error *err);
int yvex_runtime_execution_transaction_abort(
    yvex_runtime_execution_transaction *transaction, yvex_error *err);
int yvex_runtime_execution_transaction_summary_copy(
    const yvex_runtime_execution_transaction *transaction,
    yvex_execution_transaction_summary *summary, yvex_error *err);
int yvex_runtime_execution_transaction_close(
    yvex_runtime_execution_transaction **transaction, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_EXECUTION_TRANSACTION_H_INCLUDED */
