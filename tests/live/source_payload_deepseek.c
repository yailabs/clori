/*
 * The source tree is opened read-only and no payload byte is printed or retained. Exhaustive
 * source delivery is not conversion, artifact emission, or runtime support.
 */
#define _POSIX_C_SOURCE 200809L
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>

#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct {
    unsigned long long bytes;
    unsigned long long chunks;
    unsigned long long aborts;
    unsigned long long commits;
    unsigned long long current_tensor;
    unsigned long long current_tensor_end;
    int began;
    int invalid;
} live_sink;

/* Starts one accounting-only transaction without allocating consumer storage. */
static int live_begin(void *opaque,
                      const yvex_source_payload_plan_summary *summary)
{
    live_sink *sink = (live_sink *)opaque;

    if (!sink || !summary || summary->range_count == 0u ||
        summary->chunk_count == 0u || summary->logical_bytes == 0u) return 1;
    sink->began = 1;
    sink->current_tensor = ~0ull;
    return 0;
}

/* Accepts borrowed bytes only long enough to validate chunk identity and order. */
static int live_chunk(void *opaque,
                      const yvex_source_payload_chunk *chunk,
                      const unsigned char *bytes)
{
    live_sink *sink = (live_sink *)opaque;

    if (!sink || !sink->began || !chunk || !bytes ||
        !chunk->source_tensor_name || !chunk->source_tensor_name[0] ||
        chunk->byte_length == 0u) return 1;
    if (sink->current_tensor == chunk->source_tensor_index) {
        if (chunk->logical_offset != sink->current_tensor_end) sink->invalid = 1;
    } else {
        if (chunk->logical_offset != 0u) sink->invalid = 1;
        sink->current_tensor = chunk->source_tensor_index;
    }
    sink->current_tensor_end = chunk->logical_offset + chunk->byte_length;
    sink->bytes += chunk->byte_length;
    sink->chunks++;
    return sink->invalid ? 1 : 0;
}

static int live_commit(void *opaque,
                       const yvex_source_payload_stream_result *result)
{
    live_sink *sink = (live_sink *)opaque;

    if (!sink || !result || !result->complete || sink->invalid ||
        sink->bytes != result->delivered_logical_bytes ||
        sink->chunks != result->chunks_completed) return 1;
    sink->commits++;
    return 0;
}

static void live_abort(void *opaque,
                       const yvex_source_payload_failure *failure,
                       const yvex_source_payload_stream_result *result)
{
    live_sink *sink = (live_sink *)opaque;

    (void)failure;
    (void)result;
    if (sink) sink->aborts++;
}

static unsigned long long live_elapsed(const struct timespec *begin,
                                       const struct timespec *end)
{
    if (end->tv_nsec >= begin->tv_nsec)
        return (unsigned long long)(end->tv_sec - begin->tv_sec) *
                   1000000000ull +
               (unsigned long long)(end->tv_nsec - begin->tv_nsec);
    return (unsigned long long)(end->tv_sec - begin->tv_sec - 1) *
               1000000000ull +
           1000000000ull -
               (unsigned long long)(begin->tv_nsec - end->tv_nsec);
}

int main(int argc, char **argv)
{
    yvex_deepseek_payload_handoff_options options;
    yvex_deepseek_payload_handoff *handoff = NULL;
    yvex_deepseek_payload_failure handoff_failure;
    const yvex_deepseek_payload_handoff_summary *summary;
    const yvex_source_payload_plan_summary *plan_summary;
    const yvex_transform_ir_summary *transform_summary;
    const yvex_transform_binding_summary *binding_summary;
    const yvex_deepseek_gguf_map_summary *map_summary;
    const yvex_source_verification *verification;
    yvex_source_payload_session *session;
    yvex_source_payload_session_facts facts;
    yvex_source_payload_failure failure;
    yvex_transform_failure transform_failure;
    yvex_source_payload_stream_result result;
    yvex_source_payload_sink sink;
    live_sink sink_state;
    yvex_error error;
    struct timespec begin;
    struct timespec end;
    unsigned long long elapsed = 0u;
    int binding_readable = 0;
    int plan_only = 0;
    int argument = 1;
    int rc;

    if (argc > 1 && strcmp(argv[1], "--plan-only") == 0) {
        plan_only = 1;
        argument++;
    }
    if (argc - argument != 3) {
        fprintf(stderr, "usage: %s [--plan-only] SOURCE MODELS_ROOT MANIFEST\n",
                argv[0]);
        return 2;
    }
    memset(&options, 0, sizeof(options));
    options.source_path = argv[argument];
    options.models_root = argv[argument + 1];
    options.manifest_path = argv[argument + 2];
    yvex_source_payload_budget_default(&options.budget);
    options.chunk_bytes = options.budget.chunk_bytes;
    options.page_bytes = options.budget.page_bytes;
    yvex_error_clear(&error);
    rc = yvex_model_register_deepseek_v4()->payload.open(
        &handoff, &options, &handoff_failure, &error);
    if (rc != YVEX_OK) {
        fprintf(stderr, "handoff_failure=%s payload_failure=%u status=%s where=%s\n",
                yvex_model_register_deepseek_v4()->payload.failure_name(handoff_failure.code),
                (unsigned int)handoff_failure.payload_failure.code,
                yvex_status_name(yvex_error_code(&error)),
                yvex_error_where(&error));
        return 1;
    }
    summary = yvex_model_register_deepseek_v4()->payload.summary(handoff);
    plan_summary = yvex_source_payload_plan_summary_get(
        yvex_model_register_deepseek_v4()->payload.plan(handoff));
    transform_summary = yvex_transform_ir_summary_get(
        yvex_model_register_deepseek_v4()->payload.transform_ir(handoff));
    binding_summary = yvex_transform_binding_summary_get(
        yvex_model_register_deepseek_v4()->payload.binding(handoff));
    map_summary = yvex_model_register_deepseek_v4()->lowering.summary(
        yvex_model_register_deepseek_v4()->payload.map(handoff));
    verification = yvex_model_register_deepseek_v4()->payload.verification(handoff);
    session = yvex_model_register_deepseek_v4()->payload.session(handoff);
    if (!summary || !summary->complete || !plan_summary || !transform_summary ||
        !transform_summary->complete || !binding_summary ||
        !binding_summary->complete || !map_summary || !map_summary->complete ||
        !verification ||
        summary->mapping_identity != YVEX_DEEPSEEK_PAYLOAD_MAPPING_IDENTITY ||
        map_summary->mapping_identity != YVEX_DEEPSEEK_GGUF_MAPPING_IDENTITY ||
        strcmp(summary->transform_identity,
               transform_summary->transform_identity) != 0 ||
        transform_summary->schema_version !=
            YVEX_TRANSFORM_IR_SCHEMA_VERSION ||
        transform_summary->source_value_count != 69187u ||
        transform_summary->intermediate_value_count != 0u ||
        transform_summary->value_count != 70547u ||
        transform_summary->node_count != 1360u ||
        transform_summary->edge_count != 69187u ||
        transform_summary->terminal_count != 1360u ||
        transform_summary->maximum_fan_in != 512u ||
        transform_summary->maximum_depth != 1u ||
        transform_summary->operation_counts[YVEX_TRANSFORM_OP_IDENTITY] !=
            850u ||
        transform_summary->operation_counts[
            YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR] != 375u ||
        transform_summary->operation_counts[
            YVEX_TRANSFORM_OP_CHECKED_CAST] != 3u ||
        transform_summary->operation_counts[
            YVEX_TRANSFORM_OP_EXPERT_AGGREGATE] != 132u ||
        transform_summary->payload_bytes_read != 0u ||
        binding_summary->resolved_range_count != 69187u ||
        binding_summary->source_count != 69187u ||
        binding_summary->payload_bytes_read != 0u ||
        summary->descriptors_covered != 1360u ||
        summary->contributions_resolved != 69187u ||
        plan_summary->range_count != 69187u ||
        verification->shard_count != 46u ||
        verification->header_tensor_count != 69187u ||
        verification->header_scan_count != 1u) {
        fprintf(stderr, "live_plan_invariant=failed\n");
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return 1;
    }
    memset(&result, 0, sizeof(result));
    memset(&sink_state, 0, sizeof(sink_state));
    if (!plan_only) {
        memset(&sink, 0, sizeof(sink));
        sink.begin = live_begin;
        sink.chunk = live_chunk;
        sink.commit = live_commit;
        sink.abort = live_abort;
        sink.context = &sink_state;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        rc = yvex_source_payload_session_verify(
            session, yvex_model_register_deepseek_v4()->payload.plan(handoff), &sink,
            &result, &failure, &error);
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = live_elapsed(&begin, &end);
        if (rc != YVEX_OK) {
            fprintf(stderr,
                    "stream_failure=%u status=%s where=%s requested=%llu delivered=%llu\n",
                    (unsigned int)failure.code,
                    yvex_status_name(yvex_error_code(&error)),
                    yvex_error_where(&error), failure.requested_bytes,
                    failure.delivered_bytes);
            yvex_model_register_deepseek_v4()->payload.close(handoff);
            return 1;
        }
    }
    if (yvex_source_payload_session_facts_get(session, &facts, &error) !=
        YVEX_OK) {
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return 1;
    }
    rc = yvex_transform_binding_readable_validate(
        yvex_model_register_deepseek_v4()->payload.binding(handoff), &transform_failure,
        &error);
    binding_readable = rc == YVEX_OK;
    if (!binding_readable) {
        fprintf(stderr, "transform_binding_readability=failed\n");
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return 1;
    }
    if (!plan_only &&
        (!result.complete || !result.committed || result.aborted ||
         sink_state.commits != 1u || sink_state.aborts != 0u ||
         result.physical_bytes_read != verification->shard_bytes ||
         result.trust_bytes_read != verification->shard_bytes ||
         result.delivered_logical_bytes != plan_summary->logical_bytes ||
         facts.trust_class != YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED ||
         facts.trusted_shard_count != 46u || facts.short_reads != 0u ||
         facts.digest_mismatches != 0u || facts.identity_drifts != 0u)) {
        fprintf(stderr, "live_stream_invariant=failed\n");
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return 1;
    }
    printf("mode=%s\n", plan_only ? "plan-only" : "trust-and-deliver");
    printf("source_revision=%s\n", verification->revision);
    printf("source_snapshot_identity=%016llx\n",
           verification->source_snapshot_identity);
    printf("payload_identity=%s\n",
           facts.payload_identity[0] ? facts.payload_identity : "not-sealed");
    printf("admitted_payload_identity=%s\n",
           facts.admitted_payload_identity[0]
               ? facts.admitted_payload_identity
               : "not-admitted");
    printf("architecture_identity=%s\n",
           transform_summary->logical_model_identity);
    printf("transform_ir_schema_version=%u\n",
           transform_summary->schema_version);
    printf("transform_ir_identity=%s\n",
           transform_summary->transform_identity);
    printf("required_payload_identity=%s\n",
           transform_summary->required_payload_identity);
    printf("mapping_identity=%016llx\n", summary->mapping_identity);
    printf("trust_class=%s\n",
           yvex_source_payload_trust_class_name(facts.trust_class));
    printf("shards_admitted=%llu\n", facts.shard_count);
    printf("shards_trusted=%llu\n", facts.trusted_shard_count);
    printf("source_tensors_indexed=%llu\n", facts.tensor_count);
    printf("transform_source_inputs=%llu\n",
           transform_summary->source_value_count);
    printf("transform_intermediate_values=%llu\n",
           transform_summary->intermediate_value_count);
    printf("transform_value_count=%llu\n", transform_summary->value_count);
    printf("transform_node_count=%llu\n", transform_summary->node_count);
    printf("transform_edge_count=%llu\n", transform_summary->edge_count);
    printf("transform_terminal_outputs=%llu\n",
           transform_summary->terminal_count);
    printf("transform_identity_operations=%llu\n",
           transform_summary->operation_counts[YVEX_TRANSFORM_OP_IDENTITY]);
    printf("transform_scale_pair_operations=%llu\n",
           transform_summary->operation_counts[
               YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR]);
    printf("transform_checked_cast_operations=%llu\n",
           transform_summary->operation_counts[
               YVEX_TRANSFORM_OP_CHECKED_CAST]);
    printf("transform_expert_aggregate_operations=%llu\n",
           transform_summary->operation_counts[
               YVEX_TRANSFORM_OP_EXPERT_AGGREGATE]);
    printf("transform_maximum_fan_in=%llu\n",
           transform_summary->maximum_fan_in);
    printf("transform_maximum_depth=%llu\n",
           transform_summary->maximum_depth);
    printf("transform_builder_peak_bytes=%zu\n",
           transform_summary->builder_peak_bytes);
    printf("transform_sealed_ir_bytes=%zu\n",
           transform_summary->sealed_ir_bytes);
    printf("transform_temporary_validation_bytes=%zu\n",
           transform_summary->temporary_validation_bytes);
    printf("transform_index_capacity=%llu\n",
           transform_summary->index_capacity);
    printf("transform_validation_steps=%llu\n",
           transform_summary->validation_steps);
    printf("transform_binding_ranges=%llu\n",
           binding_summary->resolved_range_count);
    printf("transform_binding_lookups=%llu\n",
           binding_summary->range_lookup_count);
    printf("transform_binding_readable=%d\n", binding_readable);
    printf("transform_unconsumed_inputs=0\n");
    printf("transform_duplicate_outputs=0\n");
    printf("transform_cycles=0\n");
    printf("transform_unresolved_nodes=0\n");
    printf("transform_unsupported_operations=0\n");
    printf("transform_payload_bytes_read=%llu\n",
           transform_summary->payload_bytes_read +
               binding_summary->payload_bytes_read);
    printf("gguf_lowering_equivalence=complete\n");
    printf("mapping_contributions_resolved=%llu\n",
           summary->contributions_resolved);
    printf("logical_descriptors_covered=%llu\n",
           summary->descriptors_covered);
    printf("total_source_footprint=%llu\n", verification->source_total_bytes);
    printf("shard_file_bytes=%llu\n", verification->shard_bytes);
    printf("logical_range_bytes=%llu\n", plan_summary->logical_bytes);
    printf("trust_bytes_read=%llu\n", result.trust_bytes_read);
    printf("consumer_bytes_delivered=%llu\n",
           result.delivered_logical_bytes);
    printf("physical_bytes_read=%llu\n", result.physical_bytes_read);
    printf("chunks=%llu\n", result.chunks_completed);
    printf("planned_chunks=%llu\n", plan_summary->chunk_count);
    printf("peak_inflight_host_bytes=%llu\n",
           facts.peak_inflight_host_bytes);
    printf("buffer_allocations=%llu\n", facts.buffer_allocations);
    printf("buffer_reuses=%llu\n", facts.buffer_reuses);
    printf("buffer_releases=%llu\n", facts.buffer_releases);
    printf("maximum_open_handles=%u\n", facts.peak_open_handles);
    printf("handle_opens=%llu\n", facts.handle_opens);
    printf("handle_reopens=%llu\n", facts.handle_reopens);
    printf("handle_evictions=%llu\n", facts.handle_evictions);
    printf("short_reads=%llu\n", facts.short_reads);
    printf("digest_mismatches=%llu\n", facts.digest_mismatches);
    printf("identity_drift=%llu\n", facts.identity_drifts);
    printf("cancellations=%llu\n", facts.cancellations);
    printf("header_scans=%llu\n", facts.header_scan_count);
    printf("range_lookups=%llu\n", summary->range_lookup_count);
    printf("elapsed_nanoseconds=%llu\n", elapsed);
    printf("direct_contributions=%llu\n", summary->direct_contributions);
    printf("fp8_weight_contributions=%llu\n",
           summary->fp8_weight_contributions);
    printf("e8m0_scale_contributions=%llu\n",
           summary->e8m0_scale_contributions);
    printf("expert_contributions=%llu\n", summary->expert_contributions);
    printf("i64_router_contributions=%llu\n",
           summary->i64_router_contributions);
    printf("global_contributions=%llu\n", summary->global_contributions);
    printf("norm_contributions=%llu\n", summary->norm_contributions);
    printf("shared_expert_contributions=%llu\n",
           summary->shared_expert_contributions);
    printf("output_head_contributions=%llu\n",
           summary->output_head_contributions);
    printf("mtp_contributions=%llu\n", summary->mtp_contributions);
    yvex_model_register_deepseek_v4()->payload.close(handoff);
    return 0;
}
