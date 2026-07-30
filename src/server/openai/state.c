/* Owner: server.openai.state.
 * Owns: bounded in-memory Responses ID to YVEX session/provider-context mapping and deterministic eviction.
 * Does not own: session KV, hidden state, protocol framing, HTTP parsing, or persistence.
 * Invariants: every record is model/session bound, uniquely owned, TTL-limited, and absent after restart.
 * Boundary: previous_response_id can name only state created by this daemon adapter lifetime.
 * Purpose: retain exact application-visible context references without reconstructing model state from text.
 * Inputs: sealed provider requests, response/session IDs, monotonic process sequence, and wall seconds.
 * Effects: clones bounded request state and evicts/releases expired or least-recent records.
 * Failure: allocation failure leaves prior records intact and publishes no new response mapping. */

#include "src/server/openai/private.h"

#include <string.h>

#include <yvex/internal/core.h>

/* Purpose: release one complete response record and its provider context.
 * Inputs: one process-local response record.
 * Effects: closes its provider graph and clears lookup/session facts.
 * Failure: none; empty records are accepted.
 * Boundary: does not close the server-owned YVEX session. */
void openai_state_remove(openai_response_record *record)
{
    if (!record) return;
    yvex_provider_request_close(&record->context);
    memset(record, 0, sizeof(*record));
}

/* Purpose: find one live instance-local response mapping.
 * Inputs: gateway table, exact response ID, and current wall seconds.
 * Effects: advances the selected live row's LRU sequence.
 * Failure: returns null for missing, expired, or invalid identifiers.
 * Boundary: maps IDs only; compatibility with daemon session state is checked by the caller. */
openai_response_record *openai_state_find(openai_gateway *gateway,
                                          const char *response_id,
                                          unsigned long long now)
{
    unsigned long long index;
    if (!gateway || !response_id || !response_id[0]) return NULL;
    for (index = 0u; index < OPENAI_RESPONSE_RECORD_MAX; ++index) {
        openai_response_record *record = &gateway->records[index];
        if (record->occupied && strcmp(record->response_id, response_id) == 0 &&
            (now < record->created_seconds ||
             now - record->created_seconds <= OPENAI_RESPONSE_TTL_SECONDS)) {
            record->last_used_sequence = ++gateway->request_count;
            return record;
        }
    }
    return NULL;
}

/* Purpose: atomically store one bounded retained Responses context in a prepared free slot.
 * Inputs: gateway table, response/session IDs, sealed context, timestamp, and error output.
 * Effects: clones context and publishes it through one free record.
 * Failure: leaves the table unchanged when validation or cloning fails.
 * Boundary: retains application history, never model/KV memory. */
int openai_state_store(openai_gateway *gateway, const char *response_id,
                       const char *session_name,
                       const yvex_provider_request *context,
                       unsigned long long now, yvex_error *err)
{
    yvex_provider_request *clone = NULL;
    openai_response_record *slot = NULL;
    unsigned long long index;
    if (!gateway || !response_id || !response_id[0] || !session_name ||
        !session_name[0] || yvex_provider_request_validate(context, err) != YVEX_OK)
        return YVEX_ERR_INVALID_ARG;
    if (yvex_provider_request_clone(context, &clone, err) != YVEX_OK)
        return yvex_error_code(err);
    for (index = 0u; index < OPENAI_RESPONSE_RECORD_MAX; ++index) {
        openai_response_record *candidate = &gateway->records[index];
        if (!candidate->occupied) {
            slot = candidate;
            break;
        }
    }
    if (!slot) {
        yvex_provider_request_close(&clone);
        yvex_error_set(err, YVEX_ERR_STATE, "server.openai.state",
                       "response-state directory has no admissible slot");
        return YVEX_ERR_STATE;
    }
    slot->occupied = 1;
    yvex_core_text_copy(slot->response_id, sizeof(slot->response_id), response_id);
    yvex_core_text_copy(slot->session_name, sizeof(slot->session_name), session_name);
    yvex_core_text_copy(slot->model, sizeof(slot->model), context->model);
    slot->context = clone;
    slot->created_seconds = now;
    slot->last_used_sequence = ++gateway->request_count;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: atomically advance one retained Responses mapping after exact session continuation.
 * Inputs: gateway, occupied record, successor response ID, sealed context, timestamp, and error output.
 * Effects: clones the successor context before replacing the prior application mapping in place.
 * Failure: leaves the prior mapping intact when validation or allocation fails.
 * Boundary: preserves the same server session; consumed response IDs do not create branches. */
int openai_state_replace(openai_gateway *gateway,
                         openai_response_record *record,
                         const char *response_id,
                         const yvex_provider_request *context,
                         unsigned long long now, yvex_error *err)
{
    yvex_provider_request *clone = NULL;
    if (!gateway || !record || !record->occupied || !response_id ||
        !response_id[0] ||
        yvex_provider_request_validate(context, err) != YVEX_OK)
        return YVEX_ERR_INVALID_ARG;
    if (yvex_provider_request_clone(context, &clone, err) != YVEX_OK)
        return yvex_error_code(err);
    yvex_provider_request_close(&record->context);
    record->context = clone;
    yvex_core_text_copy(record->response_id, sizeof(record->response_id),
                        response_id);
    yvex_core_text_copy(record->model, sizeof(record->model), context->model);
    record->created_seconds = now;
    record->last_used_sequence = ++gateway->request_count;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: close every process-local response mapping during graceful gateway shutdown.
 * Inputs: gateway state containing the bounded record table.
 * Effects: releases every context and resets record sequencing.
 * Failure: none; null or already-cleared state is accepted.
 * Boundary: gateway restart intentionally loses all previous_response_id mappings. */
void openai_state_clear(openai_gateway *gateway)
{
    unsigned long long index;
    if (!gateway) return;
    for (index = 0u; index < OPENAI_RESPONSE_RECORD_MAX; ++index)
        openai_state_remove(&gateway->records[index]);
}
