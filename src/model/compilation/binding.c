/*
 * Bind sealed transformation inputs to one admitted source snapshot and compose the family
 * pipeline without reopening transformation truth.
 *
 * Publication follows exact source/payload/IR/map identity checks; construction resolves every
 * range and reads zero payload bytes. Consumers borrow immutable binding and session views only
 * for the owning binding or family-handoff lifetime.
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>

struct yvex_transform_binding {
    yvex_transform_allocator allocator;
    const yvex_transform_ir *ir;
    yvex_source_payload_session *session;
    unsigned long long *tensor_indices;
    yvex_transform_binding_summary summary;
};

static const char *const payload_failure_names[] = {"none",
                                                    "invalid-argument",
                                                    "source-verification",
                                                    "architecture-ir",
                                                    "tensor-coverage",
                                                    "transform-ir",
                                                    "gguf-mapping",
                                                    "mapping-identity-mismatch",
                                                    "mapping-contribution",
                                                    "payload-range",
                                                    "transform-binding",
                                                    "payload-plan",
                                                    "allocation-failure"};

static void *binding_default_allocate(size_t size, void *context) {
    (void)context;
    return malloc(size);
}

static void binding_default_release(void *allocation, void *context) {
    (void)context;
    free(allocation);
}

static int binding_fail(yvex_transform_failure *failure, yvex_transform_failure_code code,
                        unsigned long long source, unsigned long long expected,
                        unsigned long long actual, yvex_error *err, const char *where) {
    return yvex_transform_fail(failure, code, YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                               source, YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID, expected,
                               actual, 0u, err, where);
}

static int binding_range_equal(const yvex_transform_source_value *source,
                               const yvex_source_payload_range *range,
                               const yvex_source_payload_shard *shard) {
    unsigned int dimension;

    if (!source || !range || !shard ||
        strcmp(source->source_name, range->source_tensor_name) != 0 ||
        strcmp(source->shard_name, shard->canonical_name) != 0 ||
        source->source_tensor_index != range->source_tensor_index ||
        source->source_snapshot_identity != range->source_snapshot_identity ||
        source->source_dtype != range->dtype || source->shape.rank != range->rank ||
        source->relative_begin != range->relative_begin ||
        source->relative_end != range->relative_end)
        return 0;
    for (dimension = 0u; dimension < source->shape.rank; ++dimension)
        if (source->shape.dims[dimension] != range->dims[dimension])
            return 0;
    return 1;
}

int yvex_transform_binding_create(yvex_transform_binding **out, const yvex_transform_ir *ir,
                                  yvex_source_payload_session *session,
                                  const yvex_transform_allocator *requested_allocator,
                                  yvex_transform_failure *failure, yvex_error *err) {
    const yvex_transform_ir_summary *ir_summary = yvex_transform_ir_summary_get(ir);
    yvex_source_payload_session_facts session_facts;
    yvex_transform_allocator allocator;
    yvex_transform_binding *binding;
    unsigned long long source_index;
    size_t index_bytes;

    if (out)
        *out = NULL;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!out || !ir || !session || !ir_summary || !ir_summary->complete ||
        ir_summary->state != YVEX_TRANSFORM_IR_STATE_SEALED)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
                            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, err, "transform_binding_create");
    allocator.allocate = binding_default_allocate;
    allocator.release = binding_default_release;
    allocator.context = NULL;
    if (requested_allocator) {
        if (!requested_allocator->allocate || !requested_allocator->release)
            return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
                                YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, err, "transform_binding_create");
        allocator = *requested_allocator;
    }
    memset(&session_facts, 0, sizeof(session_facts));
    if (yvex_source_payload_session_facts_get(session, &session_facts, err) != YVEX_OK)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_STATE, YVEX_TRANSFORM_IR_NO_ID,
                            YVEX_SOURCE_PAYLOAD_STATE_READY, session_facts.state, err,
                            "transform_binding_create");
    if (session_facts.state != YVEX_SOURCE_PAYLOAD_STATE_READY &&
        session_facts.state != YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_STATE, YVEX_TRANSFORM_IR_NO_ID,
                            YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED, session_facts.state, err,
                            "transform_binding_create");
    if (session_facts.source_snapshot_identity != ir_summary->source_snapshot_identity)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
                            YVEX_TRANSFORM_IR_NO_ID, ir_summary->source_snapshot_identity,
                            session_facts.source_snapshot_identity, err,
                            "transform_binding_create");
    if (strcmp(session_facts.state == YVEX_SOURCE_PAYLOAD_STATE_READY
                   ? session_facts.payload_identity
                   : session_facts.admitted_payload_identity,
               ir_summary->required_payload_identity) != 0)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
                            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, err, "transform_binding_create");
    if (ir_summary->source_value_count >
        (unsigned long long)(SIZE_MAX / sizeof(binding->tensor_indices[0])))
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
                            YVEX_TRANSFORM_IR_NO_ID, SIZE_MAX, ir_summary->source_value_count, err,
                            "transform_binding_create");
    binding = (yvex_transform_binding *)allocator.allocate(sizeof(*binding), allocator.context);
    if (!binding)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_ALLOCATION, YVEX_TRANSFORM_IR_NO_ID,
                            sizeof(*binding), 0u, err, "transform_binding_create");
    memset(binding, 0, sizeof(*binding));
    binding->allocator = allocator;
    binding->ir = ir;
    binding->session = session;
    index_bytes = (size_t)ir_summary->source_value_count * sizeof(binding->tensor_indices[0]);
    binding->tensor_indices =
        (unsigned long long *)allocator.allocate(index_bytes, allocator.context);
    if (!binding->tensor_indices) {
        allocator.release(binding, allocator.context);
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_ALLOCATION, YVEX_TRANSFORM_IR_NO_ID,
                            index_bytes, 0u, err, "transform_binding_create");
    }
    for (source_index = 0u; source_index < ir_summary->source_value_count; ++source_index) {
        const yvex_transform_source_value *source = yvex_transform_ir_source_at(ir, source_index);
        const yvex_source_payload_range *range =
            source ? yvex_source_payload_range_find(session, source->source_name) : NULL;
        const yvex_source_payload_shard *shard =
            range ? yvex_source_payload_shard_at(session, range->shard_index) : NULL;
        binding->summary.range_lookup_count++;
        if (!binding_range_equal(source, range, shard) ||
            (session_facts.state == YVEX_SOURCE_PAYLOAD_STATE_READY &&
             (!range->payload_identity ||
              strcmp(range->payload_identity, ir_summary->required_payload_identity) != 0))) {
            allocator.release(binding->tensor_indices, allocator.context);
            allocator.release(binding, allocator.context);
            return binding_fail(failure,
                                range ? YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH
                                      : YVEX_TRANSFORM_FAILURE_MISSING_SOURCE,
                                source_index, 1u, 0u, err, "transform_binding_create");
        }
        binding->tensor_indices[source_index] = range->source_tensor_index;
        binding->summary.resolved_range_count++;
    }
    binding->summary.source_count = ir_summary->source_value_count;
    binding->summary.terminal_count = ir_summary->terminal_count;
    binding->summary.node_count = ir_summary->node_count;
    binding->summary.source_snapshot_identity = ir_summary->source_snapshot_identity;
    yvex_core_text_copy(binding->summary.required_payload_identity,
                        sizeof(binding->summary.required_payload_identity),
                        ir_summary->required_payload_identity);
    yvex_core_text_copy(binding->summary.transform_identity,
                        sizeof(binding->summary.transform_identity),
                        ir_summary->transform_identity);
    binding->summary.owned_bytes = sizeof(*binding) + index_bytes;
    binding->summary.payload_bytes_read = 0u;
    binding->summary.payload_readable_at_bind =
        session_facts.state == YVEX_SOURCE_PAYLOAD_STATE_READY;
    binding->summary.complete =
        binding->summary.resolved_range_count == binding->summary.source_count;
    *out = binding;
    return YVEX_OK;
}

int yvex_transform_binding_readable_validate(const yvex_transform_binding *binding,
                                             yvex_transform_failure *failure, yvex_error *err) {
    const yvex_transform_ir_summary *ir_summary;
    yvex_source_payload_session_facts facts;

    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!binding || !binding->ir || !binding->session)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
                            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, err, "transform_binding_readable");
    ir_summary = yvex_transform_ir_summary_get(binding->ir);
    memset(&facts, 0, sizeof(facts));
    if (!ir_summary ||
        yvex_source_payload_session_facts_get(binding->session, &facts, err) != YVEX_OK ||
        facts.state != YVEX_SOURCE_PAYLOAD_STATE_READY)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_STATE, YVEX_TRANSFORM_IR_NO_ID,
                            YVEX_SOURCE_PAYLOAD_STATE_READY, facts.state, err,
                            "transform_binding_readable");
    if (facts.source_snapshot_identity != ir_summary->source_snapshot_identity)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
                            YVEX_TRANSFORM_IR_NO_ID, ir_summary->source_snapshot_identity,
                            facts.source_snapshot_identity, err, "transform_binding_readable");
    if (strcmp(facts.payload_identity, ir_summary->required_payload_identity) != 0)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
                            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, err, "transform_binding_readable");
    return YVEX_OK;
}

void yvex_transform_binding_release(yvex_transform_binding **binding_ptr) {
    yvex_transform_binding *binding;
    yvex_transform_allocator allocator;

    if (!binding_ptr || !*binding_ptr)
        return;
    binding = *binding_ptr;
    allocator = binding->allocator;
    if (binding->tensor_indices)
        allocator.release(binding->tensor_indices, allocator.context);
    allocator.release(binding, allocator.context);
    *binding_ptr = NULL;
}

const yvex_transform_binding_summary *
yvex_transform_binding_summary_get(const yvex_transform_binding *binding) {
    return binding ? &binding->summary : NULL;
}

const yvex_transform_ir *yvex_transform_binding_ir(const yvex_transform_binding *binding) {
    return binding ? binding->ir : NULL;
}

yvex_source_payload_session *
yvex_transform_binding_payload_session(const yvex_transform_binding *binding) {
    return binding ? binding->session : NULL;
}

const yvex_transform_value *
yvex_transform_binding_terminal_at(const yvex_transform_binding *binding,
                                   unsigned long long ordinal) {
    return binding ? yvex_transform_ir_terminal_at(binding->ir, ordinal) : NULL;
}

const yvex_transform_node *
yvex_transform_binding_terminal_operation(const yvex_transform_binding *binding,
                                          unsigned long long ordinal) {
    const yvex_transform_value *terminal = yvex_transform_binding_terminal_at(binding, ordinal);
    return terminal ? yvex_transform_ir_node_at(binding->ir, terminal->producer_node_id) : NULL;
}

const yvex_transform_source_value *
yvex_transform_binding_source_at(const yvex_transform_binding *binding,
                                 unsigned long long source_index) {
    return binding ? yvex_transform_ir_source_at(binding->ir, source_index) : NULL;
}

const yvex_source_payload_range *
yvex_transform_binding_range_at(const yvex_transform_binding *binding,
                                unsigned long long source_index) {
    if (!binding || source_index >= binding->summary.source_count)
        return NULL;
    return yvex_source_payload_range_at(binding->session, binding->tensor_indices[source_index]);
}

int yvex_transform_binding_decision_validate(const yvex_transform_binding *binding,
                                             unsigned long long terminal_ordinal,
                                             const yvex_transform_physical_decision *decision,
                                             yvex_transform_failure *failure, yvex_error *err) {
    const yvex_transform_value *terminal =
        yvex_transform_binding_terminal_at(binding, terminal_ordinal);

    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);

    if (!binding || !decision || !terminal || decision->physical_class == 0u ||
        (decision->physical_class & (decision->physical_class - 1u)) != 0u)
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
                            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, err, "transform_binding_decision");
    if ((terminal->precision.allowed_physical_classes & decision->physical_class) == 0u ||
        (decision->approximation_selected && !terminal->precision.approximation_allowed))
        return binding_fail(failure, YVEX_TRANSFORM_FAILURE_INVALID_DTYPE, YVEX_TRANSFORM_IR_NO_ID,
                            terminal->precision.allowed_physical_classes, decision->physical_class,
                            err, "transform_binding_decision");
    return YVEX_OK;
}
/* Payload handoff resolves typed family inputs through the common source ABI. */

/* Local composed lifecycle operation used by construction-failure unwinds. */
static void payload_close(yvex_deepseek_payload_handoff *handoff);

struct yvex_deepseek_payload_handoff {
    char *source_path;
    char *models_root;
    char *manifest_path;
    yvex_source_verify_options source_options;
    yvex_source_verification verification;
    yvex_deepseek_tensor_coverage *coverage;
    yvex_transform_ir *transform_ir;
    yvex_deepseek_gguf_map *map;
    yvex_source_payload_session *session;
    yvex_transform_binding *binding;
    yvex_source_payload_plan *plan;
    yvex_deepseek_payload_handoff_summary summary;
};

static char *handoff_strdup(const char *text) {
    size_t length;
    char *copy;

    if (!text)
        return NULL;
    length = strlen(text);
    copy = (char *)malloc(length + 1u);
    if (copy)
        memcpy(copy, text, length + 1u);
    return copy;
}

static int handoff_reject(yvex_deepseek_payload_failure *failure,
                          yvex_deepseek_payload_failure_code code, unsigned long long descriptor,
                          unsigned long long contribution, int status, yvex_error *err,
                          const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->descriptor_index = descriptor;
        failure->contribution_index = contribution;
    }
    yvex_error_set(err, (yvex_status)status, "deepseek_payload_handoff", message);
    return status;
}

static int handoff_resolve(yvex_deepseek_payload_handoff *handoff,
                           const yvex_deepseek_payload_handoff_options *options,
                           yvex_deepseek_payload_failure *failure, yvex_error *err) {
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_model_family_lowering_api *lowering = yvex_model_deepseek_lowering_api();
    const yvex_deepseek_gguf_map_summary *map_summary = lowering->summary(handoff->map);
    unsigned long long *tensor_indices;
    unsigned long long contribution_index;
    unsigned long long descriptor_index;
    char identity_message[192];
    int rc;

    if (!map_summary || !map_summary->complete ||
        map_summary->mapping_identity != YVEX_DEEPSEEK_PAYLOAD_MAPPING_IDENTITY ||
        map_summary->source_identity != handoff->verification.source_snapshot_identity) {
        (void)snprintf(
            identity_message, sizeof(identity_message),
            "canonical DeepSeek mapping identity mismatch: expected=%016llx actual=%016llx "
            "source=%016llx verified_source=%016llx",
            (unsigned long long)YVEX_DEEPSEEK_PAYLOAD_MAPPING_IDENTITY,
            map_summary ? map_summary->mapping_identity : 0ull,
            map_summary ? map_summary->source_identity : 0ull,
            handoff->verification.source_snapshot_identity);
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_MAPPING_IDENTITY, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_FORMAT, err,
                              identity_message);
    }
    if (map_summary->source_contribution_count >
        (unsigned long long)(SIZE_MAX / sizeof(tensor_indices[0]))) {
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_BOUNDS, err,
                              "mapping contribution index allocation overflow");
    }
    tensor_indices = (unsigned long long *)calloc((size_t)map_summary->source_contribution_count,
                                                  sizeof(tensor_indices[0]));
    if (!tensor_indices) {
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_NOMEM, err,
                              "mapping contribution index allocation failed");
    }
    handoff->summary.mapping_identity = map_summary->mapping_identity;
    yvex_core_text_copy(
        handoff->summary.transform_identity,
        sizeof(handoff->summary.transform_identity),
        yvex_transform_ir_summary_get(handoff->transform_ir)->transform_identity);
    handoff->summary.source_snapshot_identity = map_summary->source_identity;
    handoff->summary.descriptor_count = map_summary->descriptor_count;
    handoff->summary.contribution_count = map_summary->source_contribution_count;
    for (contribution_index = 0u; contribution_index < map_summary->source_contribution_count;
         ++contribution_index) {
        const yvex_deepseek_gguf_contribution *contribution =
            lowering->contribution_at(handoff->map, contribution_index);
        const yvex_source_payload_range *range;
        const yvex_deepseek_tensor_coverage_row *coverage_row;
        const yvex_deepseek_gguf_descriptor *descriptor;

        if (!contribution || contribution->descriptor_index >= map_summary->descriptor_count) {
            free(tensor_indices);
            return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION, ULLONG_MAX,
                                  contribution_index, YVEX_ERR_FORMAT, err,
                                  "mapping contribution is incomplete");
        }
        descriptor = lowering->at(handoff->map, contribution->descriptor_index);
        coverage_row = family->coverage.at(handoff->coverage, contribution->source_row_index);
        range = yvex_source_payload_range_find(handoff->session, contribution->source_name);
        handoff->summary.range_lookup_count++;
        if (!descriptor || !coverage_row || !coverage_row->source || !range ||
            strcmp(coverage_row->source->name, contribution->source_name) != 0 ||
            range->source_snapshot_identity != map_summary->source_identity ||
            range->dtype != contribution->source_dtype ||
            range->rank != contribution->source_rank) {
            free(tensor_indices);
            return handoff_reject(
                failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_RANGE, contribution->descriptor_index,
                contribution_index, YVEX_ERR_FORMAT, err,
                "mapping contribution does not resolve to its exact source range");
        }
        tensor_indices[contribution_index] = range->source_tensor_index;
        handoff->summary.contributions_resolved++;
        if (descriptor->transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT)
            handoff->summary.direct_contributions++;
        if (contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY &&
            contribution->source_dtype == YVEX_NATIVE_DTYPE_F8_E4M3)
            handoff->summary.fp8_weight_contributions++;
        if (contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_SCALE &&
            contribution->source_dtype == YVEX_NATIVE_DTYPE_F8_E8M0)
            handoff->summary.e8m0_scale_contributions++;
        if (contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_WEIGHT ||
            contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_SCALE) {
            if (ULLONG_MAX - handoff->summary.routed_expert_logical_bytes < range->byte_length) {
                free(tensor_indices);
                return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_RANGE,
                                      contribution->descriptor_index, contribution_index,
                                      YVEX_ERR_BOUNDS, err,
                                      "routed expert payload accounting overflow");
            }
            handoff->summary.expert_contributions++;
            handoff->summary.routed_expert_logical_bytes += range->byte_length;
        }
        if (descriptor->transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32 &&
            contribution->source_dtype == YVEX_NATIVE_DTYPE_I64)
            handoff->summary.i64_router_contributions++;
        if (descriptor->collection == YVEX_TENSOR_COLLECTION_GLOBAL)
            handoff->summary.global_contributions++;
        if (descriptor->collection == YVEX_TENSOR_COLLECTION_NORM)
            handoff->summary.norm_contributions++;
        if (descriptor->collection == YVEX_TENSOR_COLLECTION_SHARED_EXPERT)
            handoff->summary.shared_expert_contributions++;
        if (descriptor->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD) {
            if (ULLONG_MAX - handoff->summary.output_head_logical_bytes < range->byte_length) {
                free(tensor_indices);
                return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_RANGE,
                                      contribution->descriptor_index, contribution_index,
                                      YVEX_ERR_BOUNDS, err,
                                      "output head payload accounting overflow");
            }
            handoff->summary.output_head_contributions++;
            handoff->summary.output_head_logical_bytes += range->byte_length;
        }
        if (descriptor->scope == YVEX_TENSOR_SCOPE_DRAFT)
            handoff->summary.draft_contributions++;
    }
    for (descriptor_index = 0u; descriptor_index < map_summary->descriptor_count;
         ++descriptor_index) {
        const yvex_deepseek_gguf_descriptor *descriptor =
            lowering->at(handoff->map, descriptor_index);
        unsigned long long end;

        if (!descriptor || descriptor->contribution_count == 0u ||
            ULLONG_MAX - descriptor->contribution_offset < descriptor->contribution_count) {
            free(tensor_indices);
            return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION,
                                  descriptor_index, ULLONG_MAX, YVEX_ERR_FORMAT, err,
                                  "logical descriptor has no bounded source contribution set");
        }
        end = descriptor->contribution_offset + descriptor->contribution_count;
        if (end > handoff->summary.contributions_resolved) {
            free(tensor_indices);
            return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION,
                                  descriptor_index, end, YVEX_ERR_FORMAT, err,
                                  "logical descriptor contribution span exceeds resolved mapping");
        }
        handoff->summary.descriptors_covered++;
    }
    rc = yvex_source_payload_plan_build(
        &handoff->plan, handoff->session, tensor_indices, map_summary->source_contribution_count,
        options->chunk_bytes, options->page_bytes, failure ? &failure->payload_failure : NULL, err);
    free(tensor_indices);
    if (rc != YVEX_OK) {
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_PLAN;
        return rc;
    }
    handoff->summary.complete =
        handoff->summary.descriptors_covered == YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT &&
        handoff->summary.contributions_resolved == YVEX_DEEPSEEK_GGUF_SOURCE_COUNT &&
        handoff->summary.fp8_weight_contributions != 0u &&
        handoff->summary.e8m0_scale_contributions != 0u &&
        handoff->summary.expert_contributions != 0u &&
        handoff->summary.i64_router_contributions != 0u &&
        handoff->summary.global_contributions != 0u && handoff->summary.norm_contributions != 0u &&
        handoff->summary.shared_expert_contributions != 0u &&
        handoff->summary.output_head_contributions != 0u &&
        handoff->summary.draft_contributions != 0u;
    if (!handoff->summary.complete)
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_FORMAT, err,
                              "mapping payload handoff lacks one required contribution class");
    return YVEX_OK;
}

static int payload_open(yvex_deepseek_payload_handoff **out,
                        const yvex_deepseek_payload_handoff_options *options,
                        yvex_deepseek_payload_failure *failure, yvex_error *err) {
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_model_family_lowering_api *lowering = yvex_model_deepseek_lowering_api();
    yvex_deepseek_payload_handoff *handoff;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure ir_failure;
    yvex_deepseek_tensor_coverage_failure coverage_failure;
    yvex_deepseek_gguf_map_failure map_failure;
    yvex_transform_failure transform_failure;
    yvex_source_payload_open_options payload_options;
    int rc;

    if (out)
        *out = NULL;
    if (!out || !options || !options->source_path || !options->source_path[0] ||
        !options->models_root || !options->models_root[0]) {
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_INVALID_ARGUMENT, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_INVALID_ARG, err,
                              "source path, models root, and output are required");
    }
    handoff = (yvex_deepseek_payload_handoff *)calloc(1u, sizeof(*handoff));
    if (!handoff)
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_NOMEM, err, "payload handoff allocation failed");
    handoff->source_path = handoff_strdup(options->source_path);
    handoff->models_root = handoff_strdup(options->models_root);
    handoff->manifest_path = options->manifest_path ? handoff_strdup(options->manifest_path) : NULL;
    if (!handoff->source_path || !handoff->models_root ||
        (options->manifest_path && !handoff->manifest_path)) {
        payload_close(handoff);
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_NOMEM, err,
                              "payload handoff path allocation failed");
    }
    handoff->source_options.identity = yvex_source_release_identity();
    handoff->source_options.source_path = handoff->source_path;
    handoff->source_options.models_root = handoff->models_root;
    handoff->source_options.manifest_path = handoff->manifest_path;
    handoff->source_options.promote_manifest = 1;
    rc = yvex_source_verify_with_snapshot(&handoff->source_options, &handoff->verification,
                                          &snapshot, err);
    if (rc != YVEX_OK || !handoff->verification.verified || !snapshot) {
        yvex_source_tensor_snapshot_release(snapshot);
        payload_close(handoff);
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_SOURCE, ULLONG_MAX, ULLONG_MAX,
                              rc == YVEX_OK ? YVEX_ERR_STATE : rc, err,
                              "exact source verification did not produce a retained snapshot");
    }
    rc = family->ir.build(&ir, &handoff->verification, &ir_failure, err);
    if (rc != YVEX_OK) {
        yvex_source_tensor_snapshot_release(snapshot);
        payload_close(handoff);
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_ARCHITECTURE;
        return rc;
    }
    rc = family->coverage.build(&handoff->coverage, &handoff->verification, ir, snapshot, NULL,
                                &coverage_failure, err);
    if (rc == YVEX_OK)
        rc = family->transform.build(&handoff->transform_ir, &handoff->verification, ir,
                                     handoff->coverage, NULL, &transform_failure, err);
    if (rc == YVEX_OK)
        rc = lowering->build(&handoff->map, ir, handoff->transform_ir, &map_failure, err);
    family->ir.close(ir);
    if (rc != YVEX_OK) {
        yvex_deepseek_payload_failure_code code =
            !handoff->coverage
                ? YVEX_DEEPSEEK_PAYLOAD_FAILURE_COVERAGE
                : (!handoff->transform_ir ? YVEX_DEEPSEEK_PAYLOAD_FAILURE_TRANSFORM_IR
                                          : YVEX_DEEPSEEK_PAYLOAD_FAILURE_MAPPING);
        yvex_source_tensor_snapshot_release(snapshot);
        payload_close(handoff);
        if (failure)
            failure->code = code;
        return rc;
    }
    memset(&payload_options, 0, sizeof(payload_options));
    payload_options.verification_options = &handoff->source_options;
    payload_options.verification = &handoff->verification;
    payload_options.snapshot = snapshot;
    payload_options.budget = options->budget;
    payload_options.manifest_path = handoff->verification.manifest_path;
    rc = yvex_source_payload_session_open(&handoff->session, &payload_options,
                                          failure ? &failure->payload_failure : NULL, err);
    yvex_source_tensor_snapshot_release(snapshot);
    if (rc != YVEX_OK) {
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_SOURCE;
        payload_close(handoff);
        return rc;
    }
    rc = yvex_transform_binding_create(&handoff->binding, handoff->transform_ir, handoff->session,
                                       NULL, &transform_failure, err);
    if (rc != YVEX_OK) {
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_BINDING;
        payload_close(handoff);
        return rc;
    }
    rc = handoff_resolve(handoff, options, failure, err);
    if (rc != YVEX_OK) {
        payload_close(handoff);
        return rc;
    }
    *out = handoff;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static void payload_close(yvex_deepseek_payload_handoff *handoff) {
    const yvex_model_family_api *family;
    const yvex_model_family_lowering_api *lowering;

    if (!handoff)
        return;
    family = yvex_model_register_deepseek_v4();
    lowering = yvex_model_deepseek_lowering_api();
    yvex_source_payload_plan_close(handoff->plan);
    yvex_transform_binding_release(&handoff->binding);
    (void)yvex_source_payload_session_release(&handoff->session, NULL, NULL);
    lowering->close(handoff->map);
    yvex_transform_ir_release(&handoff->transform_ir);
    family->coverage.close(handoff->coverage);
    free(handoff->manifest_path);
    free(handoff->models_root);
    free(handoff->source_path);
    free(handoff);
}

static const yvex_deepseek_payload_handoff_summary *
payload_summary(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? &handoff->summary : NULL;
}

static const yvex_source_verification *
payload_verification(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? &handoff->verification : NULL;
}

static const yvex_deepseek_gguf_map *payload_map(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->map : NULL;
}

static const yvex_transform_ir *payload_transform_ir(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->transform_ir : NULL;
}

static const yvex_transform_binding *payload_binding(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->binding : NULL;
}

static yvex_source_payload_session *payload_session(yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->session : NULL;
}

static const yvex_source_payload_plan *payload_plan(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->plan : NULL;
}

static const char *payload_failure_name(yvex_deepseek_payload_failure_code code) {
    size_t count = sizeof(payload_failure_names) / sizeof(payload_failure_names[0]);

    return code >= 0 && (size_t)code < count ? payload_failure_names[code]
                                             : "unknown-handoff-failure";
}

/*
 * Publish the immutable trusted-payload handoff operation table used by the family registration.
 *
 * Returns process-lifetime immutable storage; no allocation or I/O.
 */
const yvex_model_family_payload_api *yvex_model_deepseek_payload_api(void) {
    static const yvex_model_family_payload_api api = {
        payload_open, payload_close,        payload_summary, payload_verification,
        payload_map,  payload_transform_ir, payload_binding, payload_session,
        payload_plan, payload_failure_name};

    return &api;
}
