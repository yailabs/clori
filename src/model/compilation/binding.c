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

struct yvex_transform_binding {
    yvex_transform_allocator allocator;
    const yvex_transform_ir *ir;
    yvex_source_payload_session *session;
    unsigned long long *tensor_indices;
    yvex_transform_binding_summary summary;
};


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

int yvex_transform_binding_payload_plan_build(
    yvex_source_payload_plan **out, const yvex_transform_binding *binding, size_t chunk_bytes,
    size_t page_bytes, yvex_source_payload_failure *failure, yvex_error *err) {
    if (out)
        *out = NULL;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!out || !binding || !binding->session || !binding->summary.complete ||
        binding->summary.source_count == 0u) {
        if (failure)
            failure->code = YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT;
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "transform_binding_payload_plan",
                       "one complete transform binding is required");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_source_payload_plan_build(out, binding->session, binding->tensor_indices,
                                          binding->summary.source_count, chunk_bytes, page_bytes,
                                          failure, err);
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
