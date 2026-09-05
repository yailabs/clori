/*
 * Retain exact application-visible context references without reconstructing model state from
 * text.
 *
 * Every record is model/session bound, uniquely owned, TTL-limited, and absent after restart.
 * Previous_response_id can name only state created by this daemon adapter lifetime.
 */

#include "src/server/openai/private.h"

#include <string.h>

#include <yvex/internal/core.h>

void openai_state_remove(openai_response_record *record)
{
    if (!record) return;
    yvex_provider_request_close(&record->context);
    memset(record, 0, sizeof(*record));
}

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

int openai_state_store(openai_gateway *gateway, const char *response_id,
                       const char *session_name,
                       unsigned long long engine_generation,
                       const yvex_provider_request *context,
                       unsigned long long now, yvex_error *err)
{
    yvex_provider_request *clone = NULL;
    openai_response_record *slot = NULL;
    unsigned long long index;
    if (!gateway || !response_id || !response_id[0] || !session_name ||
        !session_name[0] || !engine_generation ||
        yvex_provider_request_validate(context, err) != YVEX_OK)
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
    slot->engine_generation = engine_generation;
    slot->context = clone;
    slot->created_seconds = now;
    slot->last_used_sequence = ++gateway->request_count;
    yvex_error_clear(err);
    return YVEX_OK;
}

int openai_state_replace(openai_gateway *gateway,
                         openai_response_record *record,
                         const char *response_id,
                         unsigned long long engine_generation,
                         const yvex_provider_request *context,
                         unsigned long long now, yvex_error *err)
{
    yvex_provider_request *clone = NULL;
    if (!gateway || !record || !record->occupied || !response_id ||
        !response_id[0] || !engine_generation ||
        yvex_provider_request_validate(context, err) != YVEX_OK)
        return YVEX_ERR_INVALID_ARG;
    if (yvex_provider_request_clone(context, &clone, err) != YVEX_OK)
        return yvex_error_code(err);
    yvex_provider_request_close(&record->context);
    record->context = clone;
    yvex_core_text_copy(record->response_id, sizeof(record->response_id),
                        response_id);
    yvex_core_text_copy(record->model, sizeof(record->model), context->model);
    record->engine_generation = engine_generation;
    record->created_seconds = now;
    record->last_used_sequence = ++gateway->request_count;
    yvex_error_clear(err);
    return YVEX_OK;
}

void openai_state_clear(openai_gateway *gateway)
{
    unsigned long long index;
    if (!gateway) return;
    for (index = 0u; index < OPENAI_RESPONSE_RECORD_MAX; ++index)
        openai_state_remove(&gateway->records[index]);
}
