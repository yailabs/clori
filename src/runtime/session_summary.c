/* Project one live session's typed resource ownership without folding in model bytes. */
#include "src/runtime/private.h"

#include <string.h>

#include <yvex/internal/core.h>

static int summary_refuse(yvex_error *err, yvex_status status,
                          const char *reason)
{
    yvex_error_set(err, status, "runtime.session.summary", reason);
    return status;
}

static int attention_summary_add(
    const yvex_attention_state_provider *provider,
    yvex_runtime_session_summary *summary, yvex_error *err)
{
    yvex_graph_attention_state_summary state = {0};
    unsigned long long allocated, resident, virtual_bytes, page_table;
    int rc;
    if (!provider || !provider->context) return YVEX_OK;
    rc = provider->summary(provider->context, &state, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_add(summary->attention_state_allocated_bytes,
                           state.allocated_bytes, &allocated) ||
        !yvex_core_u64_add(summary->attention_state_resident_bytes,
                           state.resident_bytes, &resident) ||
        !yvex_core_u64_add(summary->attention_state_virtual_bytes,
                           state.virtual_bytes, &virtual_bytes) ||
        !yvex_core_u64_add(summary->attention_state_page_table_bytes,
                           state.page_table_bytes, &page_table))
        return summary_refuse(err, YVEX_ERR_BOUNDS,
                              "attention state byte totals overflowed");
    summary->attention_state_allocated_bytes = allocated;
    summary->attention_state_resident_bytes = resident;
    summary->attention_state_virtual_bytes = virtual_bytes;
    summary->attention_state_page_table_bytes = page_table;
    return YVEX_OK;
}

static int sequence_summary_bind(
    const yvex_sequence_state *state, yvex_runtime_session_summary *summary,
    yvex_error *err)
{
    yvex_sequence_state_summary sequence = {0};
    if (!state) return YVEX_OK;
    if (yvex_sequence_state_summary_copy(state, &sequence, err) != YVEX_OK)
        return yvex_error_code(err);
    summary->sequence_state_binding_count = sequence.binding_count;
    summary->sequence_state_generation = sequence.generation;
    summary->sequence_committed_state_bytes = sequence.committed_state_bytes;
    summary->sequence_candidate_state_bytes = sequence.candidate_state_bytes;
    summary->sequence_host_state_bytes = sequence.host_state_bytes;
    summary->sequence_device_state_bytes = sequence.device_state_bytes;
    summary->sequence_recurrent_state_bytes = sequence.recurrent_state_bytes;
    summary->sequence_convolution_state_bytes = sequence.convolution_state_bytes;
    return YVEX_OK;
}

int yvex_runtime_session_summary_copy(
    const yvex_runtime_execution_session *session,
    yvex_runtime_session_summary *out, yvex_error *err)
{
    yvex_runtime_execution_session *mutable_session =
        (yvex_runtime_execution_session *)session;
    int rc;
    if (!session || !out)
        return summary_refuse(err, YVEX_ERR_INVALID_ARG,
                              "runtime session and summary output are required");
    if (!session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&mutable_session->lifecycle_mutex) != 0)
        return summary_refuse(err, YVEX_ERR_STATE,
                              "runtime session synchronization is unavailable");
    *out = session->summary;
    out->attention_state_allocated_bytes = 0ull;
    out->attention_state_resident_bytes = 0ull;
    out->attention_state_virtual_bytes = 0ull;
    out->attention_state_page_table_bytes = 0ull;
    rc = attention_summary_add(
        session->attention_state_provider_ready > 0
            ? &session->attention_state_provider : NULL,
        out, err);
    if (rc == YVEX_OK)
        rc = attention_summary_add(
            session->draft_attention_state_provider_ready > 0
                ? &session->draft_attention_state_provider : NULL,
            out, err);
    if (rc == YVEX_OK)
        rc = sequence_summary_bind(session->sequence_state, out, err);
    (void)pthread_mutex_unlock(&mutable_session->lifecycle_mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
