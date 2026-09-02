/*
 * Compose verified source metadata, family lowering, payload ranges, transform binding, and the
 * bounded delivery plan into one compiler-owned session. Family callbacks never own these
 * resources and runtime consumers never reopen them.
 */
#include <yvex/internal/compiler_source.h>

#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/source.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const failure_names[] = {
    "none",          "invalid-argument", "source-verification", "semantic-model",
    "transform-ir",  "artifact-lowering", "mapping-identity",   "mapping-contribution",
    "payload-range", "transform-binding", "payload-plan",       "allocation-failure"};

struct yvex_compilation_source_session {
    char *source_path, *models_root, *manifest_path;
    yvex_source_verify_options source_options;
    yvex_source_verification verification;
    yvex_transform_ir *transform;
    yvex_artifact_lowering_map *lowering;
    yvex_source_payload_session *payload;
    yvex_transform_binding *binding;
    yvex_source_payload_plan *plan;
    const yvex_compilation_source_projection *projection;
    yvex_compilation_source_summary summary;
};

static void source_session_close(yvex_compilation_source_session *session);

static int source_refuse(
    yvex_compilation_source_failure *failure,
    yvex_compilation_source_failure_code code,
    unsigned long long descriptor, unsigned long long contribution,
    yvex_status status, const char *reason, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->descriptor_index = descriptor;
        failure->contribution_index = contribution;
    }
    yvex_error_set(err, status, "compilation.source", reason);
    return status;
}

static char *source_text_copy(const char *text)
{
    size_t length;
    char *copy;

    if (!text) return NULL;
    length = strlen(text);
    copy = malloc(length + 1u);
    if (copy) memcpy(copy, text, length + 1u);
    return copy;
}

static int projection_valid(const yvex_compilation_source_projection *projection)
{
    return projection &&
           projection->schema_version == YVEX_COMPILATION_SOURCE_PROJECTION_SCHEMA_V1 &&
           projection->source_identity && projection->source_identity() &&
           projection->lower && projection->lowering &&
           projection->lowering->summary && projection->lowering->descriptor_at &&
           projection->lowering->contribution_at && projection->lowering->close;
}

static int source_account_bytes(
    unsigned long long *total, unsigned long long amount,
    unsigned long long descriptor, unsigned long long contribution,
    yvex_compilation_source_failure *failure, yvex_error *err)
{
    if (ULLONG_MAX - *total < amount)
        return source_refuse(
            failure, YVEX_COMPILATION_SOURCE_FAILURE_RANGE, descriptor,
            contribution, YVEX_ERR_BOUNDS, "payload byte accounting overflow", err);
    *total += amount;
    return YVEX_OK;
}

static int source_contribution_account(
    yvex_compilation_source_session *session,
    const yvex_artifact_lowering_descriptor *descriptor,
    const yvex_artifact_lowering_contribution *contribution,
    const yvex_source_payload_range *range,
    unsigned long long contribution_index,
    yvex_compilation_source_failure *failure, yvex_error *err)
{
    yvex_compilation_source_summary *summary = &session->summary;

    if (descriptor->transform == YVEX_ARTIFACT_LOWERING_TRANSFORM_DIRECT)
        summary->direct_contributions++;
    if (contribution->kind == YVEX_ARTIFACT_LOWERING_CONTRIBUTION_PRIMARY &&
        contribution->source_dtype == YVEX_NATIVE_DTYPE_F8_E4M3)
        summary->fp8_weight_contributions++;
    if (contribution->kind == YVEX_ARTIFACT_LOWERING_CONTRIBUTION_SCALE &&
        contribution->source_dtype == YVEX_NATIVE_DTYPE_F8_E8M0)
        summary->e8m0_scale_contributions++;
    if (contribution->kind == YVEX_ARTIFACT_LOWERING_CONTRIBUTION_EXPERT_WEIGHT ||
        contribution->kind == YVEX_ARTIFACT_LOWERING_CONTRIBUTION_EXPERT_SCALE) {
        int rc = source_account_bytes(
            &summary->routed_expert_logical_bytes, range->byte_length,
            contribution->descriptor_index, contribution_index, failure, err);

        if (rc != YVEX_OK) return rc;
        summary->expert_contributions++;
    }
    if (descriptor->transform == YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32 &&
        contribution->source_dtype == YVEX_NATIVE_DTYPE_I64)
        summary->i64_router_contributions++;
    if (descriptor->collection == YVEX_TENSOR_COLLECTION_GLOBAL)
        summary->global_contributions++;
    if (descriptor->collection == YVEX_TENSOR_COLLECTION_NORM)
        summary->norm_contributions++;
    if (descriptor->collection == YVEX_TENSOR_COLLECTION_SHARED_EXPERT)
        summary->shared_expert_contributions++;
    if (descriptor->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD) {
        int rc = source_account_bytes(
            &summary->output_head_logical_bytes, range->byte_length,
            contribution->descriptor_index, contribution_index, failure, err);

        if (rc != YVEX_OK) return rc;
        summary->output_head_contributions++;
    }
    if (descriptor->scope == YVEX_TENSOR_SCOPE_DRAFT)
        summary->draft_contributions++;
    return YVEX_OK;
}

static int source_contributions_resolve(
    yvex_compilation_source_session *session,
    const yvex_artifact_lowering_summary *lowering,
    const yvex_transform_binding_summary *binding,
    yvex_compilation_source_failure *failure, yvex_error *err)
{
    unsigned long long index;

    for (index = 0u; index < lowering->source_contribution_count; ++index) {
        const yvex_artifact_lowering_contribution *contribution =
            session->projection->lowering->contribution_at(session->lowering, index);
        const yvex_artifact_lowering_descriptor *descriptor;
        const yvex_transform_source_value *source;
        const yvex_source_payload_range *range;
        int rc;

        if (!contribution || contribution->descriptor_index >= lowering->descriptor_count)
            return source_refuse(
                failure, YVEX_COMPILATION_SOURCE_FAILURE_CONTRIBUTION, ULLONG_MAX,
                index, YVEX_ERR_FORMAT, "lowering contribution is incomplete", err);
        descriptor = session->projection->lowering->descriptor_at(
            session->lowering, contribution->descriptor_index);
        source = yvex_transform_ir_source_find(
            session->transform, contribution->source_name);
        range = yvex_source_payload_range_find(
            session->payload, contribution->source_name);
        session->summary.range_lookup_count++;
        if (!descriptor || !source || !range ||
            source->requirement_index != contribution->source_row_index ||
            source->source_dtype != contribution->source_dtype ||
            source->shape.rank != contribution->source_rank ||
            range->source_snapshot_identity != lowering->source_identity ||
            range->dtype != contribution->source_dtype ||
            range->rank != contribution->source_rank)
            return source_refuse(
                failure, YVEX_COMPILATION_SOURCE_FAILURE_RANGE,
                contribution->descriptor_index, index, YVEX_ERR_FORMAT,
                "lowering contribution does not resolve to its exact source range", err);
        rc = source_contribution_account(
            session, descriptor, contribution, range, index, failure, err);
        if (rc != YVEX_OK) return rc;
        session->summary.contributions_resolved++;
    }
    return binding->source_count == session->summary.contributions_resolved
               ? YVEX_OK
               : source_refuse(
                     failure, YVEX_COMPILATION_SOURCE_FAILURE_BINDING, ULLONG_MAX,
                     ULLONG_MAX, YVEX_ERR_FORMAT,
                     "transform binding and lowering contribution counts differ", err);
}

static int source_descriptors_resolve(
    yvex_compilation_source_session *session,
    const yvex_artifact_lowering_summary *lowering,
    yvex_compilation_source_failure *failure, yvex_error *err)
{
    unsigned long long index;

    for (index = 0u; index < lowering->descriptor_count; ++index) {
        const yvex_artifact_lowering_descriptor *descriptor =
            session->projection->lowering->descriptor_at(session->lowering, index);
        unsigned long long end;

        if (!descriptor || !descriptor->contribution_count ||
            ULLONG_MAX - descriptor->contribution_offset < descriptor->contribution_count)
            return source_refuse(
                failure, YVEX_COMPILATION_SOURCE_FAILURE_CONTRIBUTION, index,
                ULLONG_MAX, YVEX_ERR_FORMAT,
                "logical descriptor has no bounded source contribution set", err);
        end = descriptor->contribution_offset + descriptor->contribution_count;
        if (end > session->summary.contributions_resolved)
            return source_refuse(
                failure, YVEX_COMPILATION_SOURCE_FAILURE_CONTRIBUTION, index,
                end, YVEX_ERR_FORMAT,
                "logical descriptor contribution span exceeds resolved lowering", err);
        session->summary.descriptors_covered++;
    }
    return YVEX_OK;
}

static unsigned long long source_present_requirements(
    const yvex_compilation_source_summary *summary)
{
    unsigned long long present = 0u;

#define REQUIRE_IF(field, flag) \
    do { if (summary->field) present |= (flag); } while (0)
    REQUIRE_IF(direct_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_DIRECT);
    REQUIRE_IF(fp8_weight_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_FP8_WEIGHT);
    REQUIRE_IF(e8m0_scale_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_E8M0_SCALE);
    REQUIRE_IF(expert_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_EXPERT);
    REQUIRE_IF(i64_router_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_I64_ROUTER);
    REQUIRE_IF(global_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_GLOBAL);
    REQUIRE_IF(norm_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_NORM);
    REQUIRE_IF(shared_expert_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_SHARED_EXPERT);
    REQUIRE_IF(output_head_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_OUTPUT_HEAD);
    REQUIRE_IF(draft_contributions, YVEX_COMPILATION_SOURCE_REQUIRE_DRAFT);
#undef REQUIRE_IF
    return present;
}

static int source_resolve(
    yvex_compilation_source_session *session,
    const yvex_compilation_source_options *options,
    yvex_compilation_source_failure *failure, yvex_error *err)
{
    const yvex_artifact_lowering_summary *lowering =
        session->projection->lowering->summary(session->lowering);
    const yvex_transform_binding_summary *binding =
        yvex_transform_binding_summary_get(session->binding);
    const yvex_transform_ir_summary *transform =
        yvex_transform_ir_summary_get(session->transform);
    unsigned long long present;
    int rc;

    if (!lowering || !lowering->complete || !binding || !binding->complete || !transform ||
        !transform->complete ||
        lowering->source_identity != session->verification.source_snapshot_identity ||
        binding->source_count != lowering->source_contribution_count)
        return source_refuse(
            failure, YVEX_COMPILATION_SOURCE_FAILURE_MAPPING_IDENTITY,
            ULLONG_MAX, ULLONG_MAX, YVEX_ERR_FORMAT,
            "lowering, transform, binding, and verified source identities differ", err);
    if (lowering->mapping_identity != session->projection->expected_mapping_identity) {
        if (failure) {
            memset(failure, 0, sizeof(*failure));
            failure->code = YVEX_COMPILATION_SOURCE_FAILURE_MAPPING_IDENTITY;
        }
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "compilation.source",
            "lowering mapping identity differs: expected=%llu actual=%llu",
            session->projection->expected_mapping_identity,
            lowering->mapping_identity);
        return YVEX_ERR_FORMAT;
    }
    session->summary.mapping_identity = lowering->mapping_identity;
    yvex_core_text_copy(session->summary.transform_identity,
                        sizeof(session->summary.transform_identity),
                        transform->transform_identity);
    session->summary.source_snapshot_identity = lowering->source_identity;
    session->summary.descriptor_count = lowering->descriptor_count;
    session->summary.contribution_count = lowering->source_contribution_count;
    rc = source_contributions_resolve(session, lowering, binding, failure, err);
    if (rc == YVEX_OK)
        rc = source_descriptors_resolve(session, lowering, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_transform_binding_payload_plan_build(
            &session->plan, session->binding, options->chunk_bytes,
            options->page_bytes, failure ? &failure->payload_failure : NULL, err);
    if (rc != YVEX_OK) {
        if (failure && failure->code == YVEX_COMPILATION_SOURCE_FAILURE_NONE)
            failure->code = YVEX_COMPILATION_SOURCE_FAILURE_PLAN;
        return rc;
    }
    present = source_present_requirements(&session->summary);
    session->summary.complete =
        session->summary.descriptors_covered == lowering->descriptor_count &&
        session->summary.contributions_resolved == lowering->source_contribution_count &&
        (present & session->projection->required_contribution_mask) ==
            session->projection->required_contribution_mask;
    return session->summary.complete
               ? YVEX_OK
               : source_refuse(
                     failure, YVEX_COMPILATION_SOURCE_FAILURE_CONTRIBUTION,
                     ULLONG_MAX, ULLONG_MAX, YVEX_ERR_FORMAT,
                     "compiled source lacks a family-required contribution class", err);
}

static int source_session_open(
    yvex_compilation_source_session **out,
    const yvex_compilation_source_options *options,
    const yvex_compilation_source_projection *projection,
    yvex_compilation_source_failure *failure, yvex_error *err)
{
    yvex_compilation_source_session *session;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_source_tensor_snapshot_facts snapshot_facts = {0};
    yvex_source_payload_open_options payload_options = {0};
    yvex_transform_failure transform_failure = {0};
    int rc;

    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!out || !options || !options->source_path || !options->source_path[0] ||
        !options->models_root || !options->models_root[0] ||
        !projection_valid(projection))
        return source_refuse(
            failure, YVEX_COMPILATION_SOURCE_FAILURE_INVALID_ARGUMENT,
            ULLONG_MAX, ULLONG_MAX, YVEX_ERR_INVALID_ARG,
            "source paths, projection, and output are required", err);
    session = calloc(1u, sizeof(*session));
    if (!session)
        return source_refuse(
            failure, YVEX_COMPILATION_SOURCE_FAILURE_ALLOCATION,
            ULLONG_MAX, ULLONG_MAX, YVEX_ERR_NOMEM,
            "compilation source allocation failed", err);
    session->projection = projection;
    session->source_path = source_text_copy(options->source_path);
    session->models_root = source_text_copy(options->models_root);
    session->manifest_path = options->manifest_path
                                 ? source_text_copy(options->manifest_path) : NULL;
    if (!session->source_path || !session->models_root ||
        (options->manifest_path && !session->manifest_path)) {
        source_session_close(session);
        return source_refuse(
            failure, YVEX_COMPILATION_SOURCE_FAILURE_ALLOCATION,
            ULLONG_MAX, ULLONG_MAX, YVEX_ERR_NOMEM,
            "compilation source path allocation failed", err);
    }
    session->source_options.identity =
        (const yvex_source_target_identity *)projection->source_identity();
    session->source_options.source_path = session->source_path;
    session->source_options.models_root = session->models_root;
    session->source_options.manifest_path = session->manifest_path;
    session->source_options.promote_manifest = 1;
    rc = yvex_source_verify_with_snapshot(
        &session->source_options, &session->verification, &snapshot, err);
    if (rc == YVEX_OK && (!session->verification.verified || !snapshot)) {
        yvex_error_set(err, YVEX_ERR_STATE, "compilation.source",
                       "source verification did not retain its snapshot");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = yvex_source_tensor_snapshot_facts_get(snapshot, &snapshot_facts, err);
    if (rc == YVEX_OK) {
        session->summary.source_tensor_count = snapshot_facts.tensor_count;
        session->summary.source_shard_count = snapshot_facts.shard_count;
        session->summary.source_header_scan_count = snapshot_facts.header_scan_count;
        session->summary.source_payload_bytes_read = snapshot_facts.payload_bytes_read;
        session->summary.source_lookup_count = snapshot_facts.lookup_count;
        session->summary.source_collision_count = snapshot_facts.collision_count;
        session->summary.source_maximum_probe = snapshot_facts.maximum_probe;
    }
    if (rc != YVEX_OK && failure)
        failure->code = YVEX_COMPILATION_SOURCE_FAILURE_SOURCE;
    if (rc == YVEX_OK)
        rc = projection->lower(
            &session->transform, &session->lowering, &session->verification,
            snapshot, failure, err);
    payload_options.verification_options = &session->source_options;
    payload_options.verification = &session->verification;
    payload_options.snapshot = snapshot;
    payload_options.budget = options->budget;
    payload_options.manifest_path = session->verification.manifest_path;
    if (rc == YVEX_OK)
        rc = yvex_source_payload_session_open(
            &session->payload, &payload_options,
            failure ? &failure->payload_failure : NULL, err);
    if (rc != YVEX_OK && session->lowering && !session->payload && failure &&
        failure->code == YVEX_COMPILATION_SOURCE_FAILURE_NONE)
        failure->code = YVEX_COMPILATION_SOURCE_FAILURE_SOURCE;
    yvex_source_tensor_snapshot_release(snapshot);
    if (rc == YVEX_OK)
        rc = yvex_transform_binding_create(
            &session->binding, session->transform, session->payload, NULL,
            &transform_failure, err);
    if (rc != YVEX_OK && session->payload && !session->binding && failure &&
        failure->code == YVEX_COMPILATION_SOURCE_FAILURE_NONE)
        failure->code = YVEX_COMPILATION_SOURCE_FAILURE_BINDING;
    if (rc == YVEX_OK)
        rc = source_resolve(session, options, failure, err);
    if (rc != YVEX_OK) {
        source_session_close(session);
        return rc;
    }
    *out = session;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static void source_session_close(yvex_compilation_source_session *session)
{
    if (!session) return;
    yvex_source_payload_plan_close(session->plan);
    yvex_transform_binding_release(&session->binding);
    (void)yvex_source_payload_session_release(&session->payload, NULL, NULL);
    if (session->projection && session->projection->lowering)
        session->projection->lowering->close(session->lowering);
    yvex_transform_ir_release(&session->transform);
    free(session->manifest_path);
    free(session->models_root);
    free(session->source_path);
    memset(session, 0, sizeof(*session));
    free(session);
}

static const yvex_compilation_source_summary *source_session_summary(
    const yvex_compilation_source_session *session)
{
    return session ? &session->summary : NULL;
}

static const yvex_source_verification *source_session_verification(
    const yvex_compilation_source_session *session)
{
    return session ? &session->verification : NULL;
}

static const yvex_artifact_lowering_map *source_session_lowering(
    const yvex_compilation_source_session *session)
{
    return session ? session->lowering : NULL;
}

static const yvex_transform_ir *source_session_transform(
    const yvex_compilation_source_session *session)
{
    return session ? session->transform : NULL;
}

static const yvex_transform_binding *source_session_binding(
    const yvex_compilation_source_session *session)
{
    return session ? session->binding : NULL;
}

static yvex_source_payload_session *source_session_payload(
    yvex_compilation_source_session *session)
{
    return session ? session->payload : NULL;
}

static const yvex_source_payload_plan *source_session_plan(
    const yvex_compilation_source_session *session)
{
    return session ? session->plan : NULL;
}

static const char *source_failure_name(
    yvex_compilation_source_failure_code code)
{
    size_t count = sizeof(failure_names) / sizeof(failure_names[0]);

    return (unsigned int)code < count ? failure_names[code]
                                      : "unknown-compilation-source-failure";
}

const yvex_compilation_source_api yvex_compilation_source_operations = {
    source_session_open,
    source_session_close,
    source_session_summary,
    source_session_verification,
    source_session_lowering,
    source_session_transform,
    source_session_binding,
    source_session_payload,
    source_session_plan,
    source_failure_name};
