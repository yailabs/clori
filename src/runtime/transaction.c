/* Retain admitted execution state across cooperative, transactionally published quanta. */
#include <yvex/internal/execution_transaction.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <yvex/internal/core.h>

struct yvex_runtime_execution_transaction {
    yvex_execution_transaction_options options;
    yvex_execution_resource_lease resource;
    yvex_execution_transaction_summary summary;
    struct timespec quantum_started;
    int resource_retained, discarded;
};

static int transaction_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.execution-transaction", reason);
    return status;
}

static int transaction_now(struct timespec *now)
{
    return now && clock_gettime(CLOCK_MONOTONIC, now) == 0;
}

static unsigned long long transaction_elapsed(
    const struct timespec *start, const struct timespec *finish)
{
    unsigned long long seconds, nanoseconds;
    if (!start || !finish || finish->tv_sec < start->tv_sec ||
        (finish->tv_sec == start->tv_sec && finish->tv_nsec < start->tv_nsec))
        return 0ull;
    seconds = (unsigned long long)(finish->tv_sec - start->tv_sec);
    if (finish->tv_nsec >= start->tv_nsec) {
        nanoseconds = (unsigned long long)(finish->tv_nsec - start->tv_nsec);
    } else {
        if (!seconds) return 0ull;
        seconds--;
        nanoseconds = 1000000000ull + (unsigned long long)finish->tv_nsec -
                      (unsigned long long)start->tv_nsec;
    }
    return seconds > (ULLONG_MAX - nanoseconds) / 1000000000ull
               ? ULLONG_MAX
               : seconds * 1000000000ull + nanoseconds;
}

static void transaction_add(unsigned long long *total, unsigned long long value)
{
    *total = ULLONG_MAX - *total < value ? ULLONG_MAX : *total + value;
}

static void transaction_discard(yvex_runtime_execution_transaction *transaction)
{
    if (!transaction || transaction->discarded ||
        transaction->summary.publications)
        return;
    if (transaction->options.discard)
        transaction->options.discard(transaction->options.publication_context);
    transaction->discarded = 1;
    transaction->summary.discards++;
}

static void transaction_stop_quantum(yvex_runtime_execution_transaction *transaction)
{
    struct timespec finish;
    if (!transaction ||
        transaction->summary.state != YVEX_EXECUTION_TRANSACTION_EXECUTING)
        return;
    if (transaction_now(&finish))
        transaction_add(&transaction->summary.quantum_wall_nanoseconds,
                        transaction_elapsed(&transaction->quantum_started, &finish));
}

static void transaction_cancel(yvex_runtime_execution_transaction *transaction)
{
    if (!transaction) return;
    transaction_stop_quantum(transaction);
    transaction->summary.cancellations++;
    transaction_discard(transaction);
    transaction->summary.state = YVEX_EXECUTION_TRANSACTION_ABORTED;
}

static int transaction_cancel_requested(
    const yvex_runtime_execution_transaction *transaction)
{
    return transaction && transaction->options.cancel_requested &&
           transaction->options.cancel_requested(transaction->options.cancel_context);
}

int yvex_runtime_execution_transaction_open(
    yvex_runtime_execution_transaction **out,
    const yvex_execution_transaction_options *options, yvex_error *err)
{
    yvex_runtime_execution_transaction *transaction;
    struct timespec start, finish;
    int timed, rc;
    if (out) *out = NULL;
    if (!out || !options || !options->request_identity ||
        !yvex_sha256_hex_valid(options->request_identity) || !options->quantum_count ||
        !options->execute ||
        (options->resource &&
         (!options->resource->identity ||
          !yvex_sha256_hex_valid(options->resource->identity) ||
          !options->resource->context || !options->resource->retain ||
          !options->resource->release)))
        return transaction_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "one bounded identity-bound execution request is required");
    timed = transaction_now(&start);
    transaction = calloc(1u, sizeof(*transaction));
    if (!transaction)
        return transaction_refuse(err, YVEX_ERR_NOMEM,
                                  "execution transaction allocation failed");
    transaction->options = *options;
    transaction->summary.state = YVEX_EXECUTION_TRANSACTION_PREPARED;
    transaction->summary.admitted_quanta = options->quantum_count;
    yvex_core_text_copy(transaction->summary.request_identity,
                        sizeof(transaction->summary.request_identity),
                        options->request_identity);
    if (options->resource) {
        transaction->resource = *options->resource;
        rc = transaction->resource.retain(transaction->resource.context, err);
        if (rc != YVEX_OK) {
            free(transaction);
            return rc;
        }
        transaction->resource_retained = 1;
        transaction->summary.retained_resources = 1ull;
        yvex_core_text_copy(transaction->summary.resource_identity,
                            sizeof(transaction->summary.resource_identity),
                            options->resource->identity);
    }
    if (timed && transaction_now(&finish))
        transaction->summary.setup_nanoseconds = transaction_elapsed(&start, &finish);
    *out = transaction;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int transaction_quantum_begin(
    yvex_runtime_execution_transaction *transaction, unsigned long long ordinal,
    yvex_error *err)
{
    yvex_execution_transaction_state state;
    if (!transaction)
        return transaction_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "an execution transaction is required");
    state = transaction->summary.state;
    if ((state != YVEX_EXECUTION_TRANSACTION_PREPARED &&
         state != YVEX_EXECUTION_TRANSACTION_SAFE) ||
        ordinal != transaction->summary.completed_quanta ||
        ordinal >= transaction->summary.admitted_quanta)
        return transaction_refuse(err, YVEX_ERR_STATE,
                                  "the next admitted execution quantum is required");
    if (transaction_cancel_requested(transaction)) {
        transaction_cancel(transaction);
        return transaction_refuse(err, YVEX_ERR_CANCELLED,
                                  "execution was cancelled before the next quantum");
    }
    transaction->summary.state = YVEX_EXECUTION_TRANSACTION_EXECUTING;
    transaction->summary.started_quanta++;
    memset(&transaction->quantum_started, 0, sizeof(transaction->quantum_started));
    (void)transaction_now(&transaction->quantum_started);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int transaction_safe_point(
    yvex_runtime_execution_transaction *transaction,
    yvex_execution_safe_point_action *action, yvex_error *err)
{
    struct timespec start, finish;
    int timed;
    if (action) *action = 0;
    if (!transaction || !action ||
        transaction->summary.state != YVEX_EXECUTION_TRANSACTION_EXECUTING)
        return transaction_refuse(err, YVEX_ERR_STATE,
                                  "one completed execution quantum is required");
    transaction_stop_quantum(transaction);
    transaction->summary.completed_quanta++;
    transaction->summary.safe_points++;
    timed = transaction_now(&start);
    if (transaction_cancel_requested(transaction)) {
        transaction->summary.cancellations++;
        transaction_discard(transaction);
        transaction->summary.state = YVEX_EXECUTION_TRANSACTION_ABORTED;
        *action = YVEX_EXECUTION_SAFE_POINT_CANCEL;
    } else if (transaction->summary.completed_quanta ==
               transaction->summary.admitted_quanta) {
        transaction->summary.state = YVEX_EXECUTION_TRANSACTION_READY_TO_COMMIT;
        *action = YVEX_EXECUTION_SAFE_POINT_COMMIT;
    } else if (transaction->options.yield_requested &&
               transaction->options.yield_requested(transaction->options.yield_context)) {
        transaction->summary.yields++;
        transaction->summary.state = YVEX_EXECUTION_TRANSACTION_YIELDED;
        *action = YVEX_EXECUTION_SAFE_POINT_YIELD;
    } else {
        transaction->summary.state = YVEX_EXECUTION_TRANSACTION_SAFE;
        *action = YVEX_EXECUTION_SAFE_POINT_CONTINUE;
    }
    if (timed && transaction_now(&finish))
        transaction_add(&transaction->summary.safe_point_nanoseconds,
                        transaction_elapsed(&start, &finish));
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_execution_transaction_execute_quantum(
    yvex_runtime_execution_transaction *transaction,
    yvex_execution_safe_point_action *action, yvex_error *err)
{
    unsigned long long ordinal;
    yvex_error primary, cleanup;
    int rc;
    if (action) *action = 0;
    if (!transaction || !action)
        return transaction_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "an execution transaction and action output are required");
    ordinal = transaction->summary.completed_quanta;
    rc = transaction_quantum_begin(transaction, ordinal, err);
    if (rc != YVEX_OK) return rc;
    rc = transaction->options.execute(
        transaction->options.execution_context, ordinal, err);
    if (rc != YVEX_OK) {
        if (!yvex_error_is_set(err))
            transaction_refuse(err, (yvex_status)rc,
                               "execution quantum failed without an error");
        primary = err ? *err : (yvex_error){0};
        yvex_error_clear(&cleanup);
        (void)yvex_runtime_execution_transaction_abort(transaction, &cleanup);
        if (err) *err = primary;
        return rc;
    }
    return transaction_safe_point(transaction, action, err);
}

int yvex_runtime_execution_transaction_resume(
    yvex_runtime_execution_transaction *transaction, yvex_error *err)
{
    if (!transaction ||
        transaction->summary.state != YVEX_EXECUTION_TRANSACTION_YIELDED)
        return transaction_refuse(err, YVEX_ERR_STATE,
                                  "a yielded execution transaction is required");
    if (transaction_cancel_requested(transaction)) {
        transaction_cancel(transaction);
        return transaction_refuse(err, YVEX_ERR_CANCELLED,
                                  "yielded execution was cancelled before resumption");
    }
    transaction->summary.state = YVEX_EXECUTION_TRANSACTION_SAFE;
    transaction->summary.resumes++;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_execution_transaction_commit(
    yvex_runtime_execution_transaction *transaction, yvex_error *err)
{
    int rc;
    if (!transaction ||
        transaction->summary.state != YVEX_EXECUTION_TRANSACTION_READY_TO_COMMIT)
        return transaction_refuse(err, YVEX_ERR_STATE,
                                  "a fully executed transaction is required for publication");
    if (transaction_cancel_requested(transaction)) {
        transaction_cancel(transaction);
        return transaction_refuse(err, YVEX_ERR_CANCELLED,
                                  "execution was cancelled before publication");
    }
    rc = transaction->options.publish
             ? transaction->options.publish(transaction->options.publication_context, err)
             : YVEX_OK;
    if (rc != YVEX_OK) {
        if (!yvex_error_is_set(err))
            transaction_refuse(err, (yvex_status)rc,
                               "transaction publication failed without an error");
        transaction_discard(transaction);
        transaction->summary.state = YVEX_EXECUTION_TRANSACTION_ABORTED;
        return rc;
    }
    transaction->summary.publications++;
    transaction->summary.state = YVEX_EXECUTION_TRANSACTION_COMMITTED;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_execution_transaction_abort(
    yvex_runtime_execution_transaction *transaction, yvex_error *err)
{
    if (!transaction)
        return transaction_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "an execution transaction is required");
    if (transaction->summary.state == YVEX_EXECUTION_TRANSACTION_COMMITTED)
        return transaction_refuse(err, YVEX_ERR_STATE,
                                  "a committed execution transaction cannot be aborted");
    if (transaction->summary.state != YVEX_EXECUTION_TRANSACTION_ABORTED) {
        transaction_stop_quantum(transaction);
        transaction_discard(transaction);
        transaction->summary.state = YVEX_EXECUTION_TRANSACTION_ABORTED;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_execution_transaction_summary_copy(
    const yvex_runtime_execution_transaction *transaction,
    yvex_execution_transaction_summary *summary, yvex_error *err)
{
    if (!transaction || !summary)
        return transaction_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "execution transaction summary storage is required");
    *summary = transaction->summary;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_execution_transaction_close(
    yvex_runtime_execution_transaction **transaction, yvex_error *err)
{
    yvex_runtime_execution_transaction *owned;
    int rc = YVEX_OK;
    if (!transaction || !*transaction) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owned = *transaction;
    if (owned->summary.state != YVEX_EXECUTION_TRANSACTION_COMMITTED &&
        owned->summary.state != YVEX_EXECUTION_TRANSACTION_ABORTED) {
        transaction_stop_quantum(owned);
        transaction_discard(owned);
        owned->summary.state = YVEX_EXECUTION_TRANSACTION_ABORTED;
    }
    if (owned->resource_retained) {
        rc = owned->resource.release(owned->resource.context, err);
        if (rc != YVEX_OK) return rc;
        owned->resource_retained = 0;
    }
    memset(owned, 0, sizeof(*owned));
    free(owned);
    *transaction = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
