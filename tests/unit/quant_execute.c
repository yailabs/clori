#define _XOPEN_SOURCE 700

#include "tests/test.h"

#include <yvex/internal/artifact.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define QUANT_EXEC_EXPERTS 256u
#define QUANT_EXEC_EXPERT_PACKED_COLUMNS 4224u
#define QUANT_EXEC_EXPERT_LOGICAL_COLUMNS 8448u
#define QUANT_EXEC_EXPERT_SCALE_COLUMNS 264u
#define QUANT_EXEC_SOURCE_COUNT (4u + QUANT_EXEC_EXPERTS * 2u)
#define QUANT_EXEC_TERMINAL_COUNT 4u

typedef struct {
    char root[512];
    char shard_path[640];
    char manifest_path[640];
    yvex_source_tensor_snapshot *snapshot;
    yvex_source_payload_session *session;
    yvex_transform_ir *ir;
    yvex_transform_binding *binding;
    yvex_quant_plan *plan;
    unsigned long long payload_bytes;
} quant_execute_fixture;

typedef struct {
    unsigned long long bytes;
} quant_verify_sink;

typedef struct {
    unsigned long long calls;
    unsigned long long fail_at;
    unsigned long long live;
} quant_execute_allocator;

/* Injects one deterministic executor allocation failure without retaining bytes. */
static void *quant_execute_allocate(size_t size, void *context)
{
    quant_execute_allocator *state = (quant_execute_allocator *)context;
    void *allocation;

    if (!state || state->calls++ == state->fail_at) return NULL;
    allocation = malloc(size);
    if (allocation) state->live++;
    return allocation;
}

/* Releases one injected executor allocation and maintains exact ownership count. */
static void quant_execute_release(void *allocation, void *context)
{
    quant_execute_allocator *state = (quant_execute_allocator *)context;

    if (!allocation) return;
    if (state && state->live) state->live--;
    free(allocation);
}

static int quant_fixture_file_copy(const char *source, const char *destination)
{
    unsigned char bytes[8192];
    FILE *input = NULL;
    FILE *output = NULL;
    size_t count;
    int ok = 0;

    if (!source || !destination) return 0;
    input = fopen(source, "rb");
    output = fopen(destination, "wb");
    if (!input || !output) goto cleanup;
    while ((count = fread(bytes, 1u, sizeof(bytes), input)) != 0u) {
        if (fwrite(bytes, 1u, count, output) != count) goto cleanup;
    }
    if (ferror(input) || fflush(output) != 0) goto cleanup;
    if (fclose(output) != 0) {
        output = NULL;
        goto cleanup;
    }
    output = NULL;
    if (fclose(input) != 0) {
        input = NULL;
        goto cleanup;
    }
    input = NULL;
    ok = 1;
cleanup:
    if (input) (void)fclose(input);
    if (output) (void)fclose(output);
    if (!ok) (void)unlink(destination);
    return ok;
}

/* Accepts trust-scan delivery while retaining only an exact byte count. */
static int quant_verify_begin(
    void *context, const yvex_source_payload_plan_summary *summary)
{
    quant_verify_sink *sink = (quant_verify_sink *)context;
    if (!sink || !summary) return 1;
    sink->bytes = 0u;
    return 0;
}

static int quant_verify_chunk(
    void *context,
    const yvex_source_payload_chunk *chunk,
    const unsigned char *bytes)
{
    quant_verify_sink *sink = (quant_verify_sink *)context;
    if (!sink || !chunk || !bytes ||
        ULLONG_MAX - sink->bytes < chunk->byte_length) return 1;
    sink->bytes += chunk->byte_length;
    return 0;
}

/* Admits the trust transaction only after exact delivery. */
static int quant_verify_commit(
    void *context, const yvex_source_payload_stream_result *result)
{
    quant_verify_sink *sink = (quant_verify_sink *)context;
    return !sink || !result || !result->complete || result->aborted ||
           sink->bytes != result->delivered_logical_bytes;
}

/* Leaves the fixture count defined after an aborted trust transaction. */
static void quant_verify_abort(
    void *context,
    const yvex_source_payload_failure *failure,
    const yvex_source_payload_stream_result *result)
{
    (void)failure;
    (void)result;
    if (context) ((quant_verify_sink *)context)->bytes = 0u;
}

static int quant_mkdir(const char *path)
{
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

static int quant_write_u16(FILE *file, unsigned short value)
{
    unsigned char bytes[2];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8u) & 0xffu);
    return fwrite(bytes, 1u, sizeof(bytes), file) == sizeof(bytes);
}

static int quant_write_i64(FILE *file, int64_t value)
{
    unsigned char bytes[8];
    uint64_t bits = (uint64_t)value;
    unsigned int index;
    for (index = 0u; index < sizeof(bytes); ++index)
        bytes[index] = (unsigned char)((bits >> (index * 8u)) & 0xffu);
    return fwrite(bytes, 1u, sizeof(bytes), file) == sizeof(bytes);
}

/* Writes one deterministic multi-operation source shard under bounded size. */
static int quant_fixture_write_payload(
    quant_execute_fixture *fixture,
    int bad_cast,
    int bad_scale,
    yvex_error *err)
{
    static const char shard_name[] = "quant-execute-00001.safetensors";
    yvex_native_weight_table *table = NULL;
    yvex_source_shard_snapshot shard;
    yvex_source_tensor_snapshot_facts snapshot_facts;
    unsigned long long offset = 0u;
    unsigned long long dims[2];
    unsigned int expert;
    unsigned int index;
    FILE *file;

    file = fopen(fixture->shard_path, "wb");
    if (!file) return 0;
    for (index = 0u; index < 16u; ++index) {
        unsigned char prefix = (unsigned char)(0xa0u + index);
        if (fwrite(&prefix, 1u, 1u, file) != 1u) goto fail;
    }
    table = (yvex_native_weight_table *)calloc(1u, sizeof(*table));
    if (!table) goto fail;

    dims[0] = 4096u;
    if (yvex_native_weight_table_add(
            table, "identity.weight", shard_name, "BF16", 1u, dims,
            offset, offset + 8192u, err) != YVEX_OK) goto fail;
    for (index = 0u; index < 4096u; ++index) {
        float value = ((int)(index % 257u) - 128) / 64.0f;
        if (!quant_write_u16(file, yvex_quant_bf16_encode(value))) goto fail;
    }
    offset += 8192u;

    dims[0] = 64u;
    dims[1] = 128u;
    if (yvex_native_weight_table_add(
            table, "pair.weight", shard_name, "F8_E4M3", 2u, dims,
            offset, offset + 8192u, err) != YVEX_OK) goto fail;
    for (index = 0u; index < 8192u; ++index) {
        unsigned char value = (unsigned char)(0x20u + index % 0x4fu);
        if ((value & 0x7fu) == 0x7fu) value = 0x3fu;
        if (fwrite(&value, 1u, 1u, file) != 1u) goto fail;
    }
    offset += 8192u;
    dims[0] = 2u;
    dims[1] = 4u;
    if (yvex_native_weight_table_add(
            table, "pair.scale", shard_name, "F8_E8M0", 2u, dims,
            offset, offset + 8u, err) != YVEX_OK) goto fail;
    for (index = 0u; index < 8u; ++index) {
        unsigned char scale = bad_scale && index == 0u ? 0xffu : 127u;
        if (fwrite(&scale, 1u, 1u, file) != 1u) goto fail;
    }
    offset += 8u;

    dims[0] = 4u;
    if (yvex_native_weight_table_add(
            table, "cast.table", shard_name, "I64", 1u, dims,
            offset, offset + 32u, err) != YVEX_OK) goto fail;
    if (!quant_write_i64(file, INT32_MIN) || !quant_write_i64(file, -1) ||
        !quant_write_i64(file, 0) ||
        !quant_write_i64(file, bad_cast == 1
            ? (int64_t)INT32_MAX + 1
            : bad_cast == 2
                ? (int64_t)INT32_MIN - 1 : INT32_MAX)) goto fail;
    offset += 32u;

    for (expert = 0u; expert < QUANT_EXEC_EXPERTS; ++expert) {
        char name[80];
        dims[0] = 1u;
        dims[1] = QUANT_EXEC_EXPERT_PACKED_COLUMNS;
        (void)snprintf(name, sizeof(name), "expert.%03u.weight", expert);
        if (yvex_native_weight_table_add(
                table, name, shard_name, "I8", 2u, dims, offset,
                offset + QUANT_EXEC_EXPERT_PACKED_COLUMNS, err) != YVEX_OK)
            goto fail;
        for (index = 0u; index < QUANT_EXEC_EXPERT_PACKED_COLUMNS; ++index) {
            unsigned char packed = (unsigned char)(
                ((expert + index) & 0x0fu) |
                (((expert * 3u + index * 5u) & 0x0fu) << 4u));
            if (fwrite(&packed, 1u, 1u, file) != 1u) goto fail;
        }
        offset += QUANT_EXEC_EXPERT_PACKED_COLUMNS;
        dims[1] = QUANT_EXEC_EXPERT_SCALE_COLUMNS;
        (void)snprintf(name, sizeof(name), "expert.%03u.scale", expert);
        if (yvex_native_weight_table_add(
                table, name, shard_name, "F8_E8M0", 2u, dims, offset,
                offset + QUANT_EXEC_EXPERT_SCALE_COLUMNS, err) != YVEX_OK)
            goto fail;
        for (index = 0u; index < QUANT_EXEC_EXPERT_SCALE_COLUMNS; ++index) {
            unsigned char scale = (unsigned char)(125u + (index % 5u));
            if (fwrite(&scale, 1u, 1u, file) != 1u) goto fail;
        }
        offset += QUANT_EXEC_EXPERT_SCALE_COLUMNS;
    }
    if (fclose(file) != 0) {
        file = NULL;
        goto fail_after_close;
    }
    file = NULL;
    memset(&shard, 0, sizeof(shard));
    shard.canonical_id = 0u;
    shard.canonical_name = shard_name;
    shard.file_bytes = offset + 16u;
    shard.data_region_offset = 16u;
    shard.payload_bytes = offset;
    if (yvex_source_tensor_snapshot_take_table_with_shards(
            &fixture->snapshot, &table, &shard, 1u, 1u, err) != YVEX_OK)
        goto fail_after_close;
    if (yvex_source_tensor_snapshot_facts_get(
            fixture->snapshot, &snapshot_facts, err) != YVEX_OK)
        goto fail_after_close;
    fixture->payload_bytes = offset;
    return snapshot_facts.tensor_count == QUANT_EXEC_SOURCE_COUNT;

fail:
    (void)fclose(file);
fail_after_close:
    free(table);
    return 0;
}

static int quant_fixture_open_session(
    quant_execute_fixture *fixture,
    yvex_error *err)
{
    yvex_source_tensor_snapshot_facts snapshot_facts;
    yvex_source_verification verification;
    yvex_source_verify_options verify_options;
    yvex_source_payload_open_options open_options;
    yvex_source_payload_budget budget;
    yvex_source_payload_failure failure;
    yvex_source_payload_plan *trust_plan = NULL;
    yvex_source_payload_sink sink;
    yvex_source_payload_stream_result result;
    quant_verify_sink sink_state;
    int rc;

    if (yvex_source_tensor_snapshot_facts_get(
            fixture->snapshot, &snapshot_facts, err) != YVEX_OK) return 0;
    memset(&verification, 0, sizeof(verification));
    verification.verified = 1;
    verification.manifest_verified = 1;
    verification.path_verified = 1;
    verification.repository_verified = 1;
    verification.revision_verified = 1;
    verification.shard_index_headers_match = 1;
    verification.header_scan_count = 1u;
    verification.header_shard_count = 1u;
    verification.header_tensor_count = QUANT_EXEC_SOURCE_COUNT;
    verification.shard_count = 1u;
    verification.source_file_count = 1u;
    verification.shard_bytes = fixture->payload_bytes + 16u;
    verification.source_total_bytes = fixture->payload_bytes + 16u;
    verification.declared_tensor_bytes = fixture->payload_bytes;
    verification.source_snapshot_identity = snapshot_facts.identity;
    (void)snprintf(verification.revision, sizeof(verification.revision), "%s",
                   "62af8fffb2f7030cac4de2f0169f5b8d1101b646");
    (void)snprintf(verification.repository_id,
                   sizeof(verification.repository_id), "%s",
                   "deepseek-ai/DeepSeek-V4-Flash-DSpark");
    (void)snprintf(verification.inventory_authority,
                   sizeof(verification.inventory_authority), "%s",
                   "fixture-retained-snapshot");
    if (!realpath(fixture->root, verification.resolved_source_path)) return 0;
    (void)snprintf(verification.manifest_path,
                   sizeof(verification.manifest_path), "%s",
                   fixture->manifest_path);
    memset(&verify_options, 0, sizeof(verify_options));
    verify_options.identity = yvex_source_release_identity();
    verify_options.source_path = fixture->root;
    verify_options.models_root = "build/tests";
    verify_options.manifest_path = fixture->manifest_path;
    yvex_source_payload_budget_default(&budget);
    budget.maximum_open_handles = 1u;
    budget.maximum_streams = 4u;
    budget.chunk_bytes = 8192u;
    budget.maximum_inflight_host_bytes = 64u * 1024u;
    budget.allow_local_snapshot_seal = 1;
    memset(&open_options, 0, sizeof(open_options));
    open_options.verification_options = &verify_options;
    open_options.verification = &verification;
    open_options.snapshot = fixture->snapshot;
    open_options.budget = budget;
    open_options.manifest_path = fixture->manifest_path;
    rc = yvex_source_payload_session_open(
        &fixture->session, &open_options, &failure, err);
    if (rc != YVEX_OK) return 0;
    rc = yvex_source_payload_plan_build_all(
        &trust_plan, fixture->session, 4096u, 4096u, &failure, err);
    if (rc != YVEX_OK) return 0;
    memset(&sink_state, 0, sizeof(sink_state));
    memset(&sink, 0, sizeof(sink));
    sink.begin = quant_verify_begin;
    sink.chunk = quant_verify_chunk;
    sink.commit = quant_verify_commit;
    sink.abort = quant_verify_abort;
    sink.context = &sink_state;
    rc = yvex_source_payload_session_verify(
        fixture->session, trust_plan, &sink, &result, &failure, err);
    yvex_source_payload_plan_close(trust_plan);
    return rc == YVEX_OK && result.complete && result.committed &&
           sink_state.bytes == fixture->payload_bytes;
}

static int quant_fixture_add_source(
    yvex_transform_builder *builder,
    yvex_source_payload_session *session,
    unsigned long long snapshot_identity,
    const char *name,
    yvex_transform_dtype value_dtype,
    yvex_transform_subsystem subsystem,
    yvex_tensor_role role,
    unsigned long long expert_index,
    unsigned long long requirement_index,
    unsigned long long *value_id,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    const yvex_source_payload_range *range =
        yvex_source_payload_range_find(session, name);
    const yvex_source_payload_shard *shard = range
        ? yvex_source_payload_shard_at(session, range->shard_index) : NULL;
    yvex_transform_source_spec source;
    unsigned int dimension;

    if (!range || !shard) return YVEX_ERR_FORMAT;
    memset(&source, 0, sizeof(source));
    source.source_name = range->source_tensor_name;
    source.shard_name = shard->canonical_name;
    source.source_tensor_index = range->source_tensor_index;
    source.requirement_index = requirement_index;
    source.source_snapshot_identity = snapshot_identity;
    source.source_dtype = range->dtype;
    source.value_dtype = value_dtype;
    source.shape.rank = range->rank;
    for (dimension = 0u; dimension < range->rank; ++dimension)
        source.shape.dims[dimension] = range->dims[dimension];
    source.relative_begin = range->relative_begin;
    source.relative_end = range->relative_end;
    source.requirement_identity = requirement_index + 0x1000u;
    source.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    source.subsystem = subsystem;
    source.role_hint = role;
    source.layer_index = 0u;
    source.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    source.expert_index = expert_index;
    source.required_uses = 1u;
    return yvex_transform_builder_add_source(
        builder, &source, value_id, failure, err);
}

static int quant_fixture_add_terminal(
    yvex_transform_builder *builder,
    unsigned long long ordinal,
    yvex_transform_subsystem subsystem,
    yvex_tensor_role role,
    const yvex_transform_shape *shape,
    yvex_transform_dtype dtype,
    const yvex_transform_precision_constraint *precision,
    yvex_transform_node_spec *node,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_value_spec terminal;
    unsigned long long output_value;
    unsigned long long node_id;
    int rc;

    memset(&terminal, 0, sizeof(terminal));
    terminal.kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    terminal.semantic_id = 0x2000u + ordinal;
    terminal.canonical_ordinal = ordinal;
    terminal.shape = *shape;
    terminal.dtype = dtype;
    terminal.precision = *precision;
    terminal.logical_key.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    terminal.logical_key.subsystem = subsystem;
    terminal.logical_key.role = role;
    terminal.logical_key.layer_index = 0u;
    terminal.logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    terminal.logical_key.group_index = ordinal;
    rc = yvex_transform_builder_declare_value(
        builder, &terminal, &output_value, failure, err);
    if (rc != YVEX_OK) return rc;
    node->output_value_id = output_value;
    return yvex_transform_builder_add_node(
        builder, node, &node_id, failure, err);
}

static int quant_fixture_build_ir(
    quant_execute_fixture *fixture,
    yvex_error *err)
{
    static const char logical_identity[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    yvex_source_payload_session_facts session_facts;
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_failure failure;
    yvex_transform_precision_constraint precision;
    yvex_transform_node_spec node;
    yvex_transform_shape shape;
    unsigned long long identity_input;
    unsigned long long pair_inputs[2];
    unsigned long long cast_input;
    unsigned long long *expert_inputs = NULL;
    unsigned long long requirement = 0u;
    unsigned int expert;
    int rc;

    if (yvex_source_payload_session_facts_get(
            fixture->session, &session_facts, err) != YVEX_OK ||
        session_facts.state != YVEX_SOURCE_PAYLOAD_STATE_READY) return 0;
    memset(&header, 0, sizeof(header));
    header.schema_version = YVEX_TRANSFORM_IR_SCHEMA_VERSION;
    header.logical_model_identity = logical_identity;
    header.source_snapshot_identity = session_facts.source_snapshot_identity;
    header.coverage_identity = 0x8899aabbccddeeffull;
    header.required_payload_identity = session_facts.payload_identity;
    header.payload_trust_class =
        yvex_source_payload_trust_class_name(session_facts.trust_class);
    header.expected_source_count = QUANT_EXEC_SOURCE_COUNT;
    header.expected_terminal_count = QUANT_EXEC_TERMINAL_COUNT;
    header.header_scan_count = 1u;
    rc = yvex_transform_builder_create(
        &builder, &header, NULL, &failure, err);
    if (rc != YVEX_OK) goto cleanup;

    rc = quant_fixture_add_source(
        builder, fixture->session, session_facts.source_snapshot_identity,
        "identity.weight", YVEX_TRANSFORM_DTYPE_BF16,
        YVEX_TRANSFORM_SUBSYSTEM_GLOBAL, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
        YVEX_TRANSFORM_IR_NO_ID, requirement++, &identity_input,
        &failure, err);
    memset(&shape, 0, sizeof(shape));
    shape.rank = 1u;
    shape.dims[0] = 4096u;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_EXACT;
    precision.allowed_physical_classes = YVEX_TRANSFORM_PHYSICAL_BF16;
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.input_value_ids = &identity_input;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    if (rc == YVEX_OK) rc = quant_fixture_add_terminal(
        builder, 0u, YVEX_TRANSFORM_SUBSYSTEM_GLOBAL,
        YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, &shape,
        YVEX_TRANSFORM_DTYPE_BF16, &precision, &node, &failure, err);

    if (rc == YVEX_OK) rc = quant_fixture_add_source(
        builder, fixture->session, session_facts.source_snapshot_identity,
        "pair.weight", YVEX_TRANSFORM_DTYPE_FP8_E4M3,
        YVEX_TRANSFORM_SUBSYSTEM_ATTENTION, YVEX_TENSOR_ROLE_ATTENTION_Q,
        YVEX_TRANSFORM_IR_NO_ID, requirement++, &pair_inputs[0],
        &failure, err);
    if (rc == YVEX_OK) rc = quant_fixture_add_source(
        builder, fixture->session, session_facts.source_snapshot_identity,
        "pair.scale", YVEX_TRANSFORM_DTYPE_E8M0_SCALE,
        YVEX_TRANSFORM_SUBSYSTEM_ATTENTION, YVEX_TENSOR_ROLE_ATTENTION_Q,
        YVEX_TRANSFORM_IR_NO_ID, requirement++, &pair_inputs[1],
        &failure, err);
    memset(&shape, 0, sizeof(shape));
    shape.rank = 2u;
    shape.dims[0] = 64u;
    shape.dims[1] = 128u;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_SCALE_PAIRED |
                      YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                      YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
    precision.allowed_physical_classes = YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    precision.approximation_allowed = 1;
    precision.reference_compute_required = 1;
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR;
    node.input_value_ids = pair_inputs;
    node.input_count = 2u;
    node.scale_block_rows = 32u;
    node.scale_block_columns = 32u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    if (rc == YVEX_OK) rc = quant_fixture_add_terminal(
        builder, 1u, YVEX_TRANSFORM_SUBSYSTEM_ATTENTION,
        YVEX_TENSOR_ROLE_ATTENTION_Q, &shape, YVEX_TRANSFORM_DTYPE_REAL,
        &precision, &node, &failure, err);

    if (rc == YVEX_OK) rc = quant_fixture_add_source(
        builder, fixture->session, session_facts.source_snapshot_identity,
        "cast.table", YVEX_TRANSFORM_DTYPE_I64,
        YVEX_TRANSFORM_SUBSYSTEM_ROUTER, YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE,
        YVEX_TRANSFORM_IR_NO_ID, requirement++, &cast_input, &failure, err);
    memset(&shape, 0, sizeof(shape));
    shape.rank = 1u;
    shape.dims[0] = 4u;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_LOSSLESS |
                      YVEX_TRANSFORM_PRECISION_RANGE_PROOF |
                      YVEX_TRANSFORM_PRECISION_INTEGER_ONLY;
    precision.allowed_physical_classes = YVEX_TRANSFORM_PHYSICAL_I32;
    precision.range_proof_required = 1;
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_CHECKED_CAST;
    node.input_value_ids = &cast_input;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_RANGE_PROOF;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    if (rc == YVEX_OK) rc = quant_fixture_add_terminal(
        builder, 2u, YVEX_TRANSFORM_SUBSYSTEM_ROUTER,
        YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE, &shape,
        YVEX_TRANSFORM_DTYPE_I32, &precision, &node, &failure, err);

    expert_inputs = (unsigned long long *)calloc(
        QUANT_EXEC_EXPERTS * 2u, sizeof(*expert_inputs));
    if (rc == YVEX_OK && !expert_inputs) rc = YVEX_ERR_NOMEM;
    for (expert = 0u; rc == YVEX_OK && expert < QUANT_EXEC_EXPERTS; ++expert) {
        char weight[80];
        char scale[80];
        (void)snprintf(weight, sizeof(weight),
                       "expert.%03u.weight", expert);
        (void)snprintf(scale, sizeof(scale),
                       "expert.%03u.scale", expert);
        rc = quant_fixture_add_source(
            builder, fixture->session, session_facts.source_snapshot_identity,
            weight, YVEX_TRANSFORM_DTYPE_PACKED_FP4,
            YVEX_TRANSFORM_SUBSYSTEM_ROUTED_EXPERT,
            YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, expert, requirement++,
            &expert_inputs[expert * 2u], &failure, err);
        if (rc == YVEX_OK) rc = quant_fixture_add_source(
            builder, fixture->session, session_facts.source_snapshot_identity,
            scale, YVEX_TRANSFORM_DTYPE_E8M0_SCALE,
            YVEX_TRANSFORM_SUBSYSTEM_ROUTED_EXPERT,
            YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, expert, requirement++,
            &expert_inputs[expert * 2u + 1u], &failure, err);
    }
    memset(&shape, 0, sizeof(shape));
    shape.rank = 3u;
    shape.dims[0] = QUANT_EXEC_EXPERTS;
    shape.dims[1] = 1u;
    shape.dims[2] = QUANT_EXEC_EXPERT_LOGICAL_COLUMNS;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_SCALE_PAIRED |
                      YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                      YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
    precision.allowed_physical_classes = YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    precision.approximation_allowed = 1;
    precision.reference_compute_required = 1;
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_EXPERT_AGGREGATE;
    node.input_value_ids = expert_inputs;
    node.input_count = QUANT_EXEC_EXPERTS * 2u;
    node.axis = 0u;
    node.expert_count = QUANT_EXEC_EXPERTS;
    node.packing_factor = 2u;
    node.scale_group_width = 32u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_EXPERT_INDEX;
    node.payload_execution_required = 1;
    if (rc == YVEX_OK) rc = quant_fixture_add_terminal(
        builder, 3u, YVEX_TRANSFORM_SUBSYSTEM_ROUTED_EXPERT,
        YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, &shape,
        YVEX_TRANSFORM_DTYPE_REAL, &precision, &node, &failure, err);
    if (rc == YVEX_OK && requirement != QUANT_EXEC_SOURCE_COUNT)
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK) rc = yvex_transform_builder_seal(
        builder, &fixture->ir, &failure, err);

cleanup:
    free(expert_inputs);
    yvex_transform_builder_release(&builder);
    return rc == YVEX_OK && fixture->ir;
}

static void quant_fixture_decisions(yvex_quant_explicit_decision decisions[QUANT_EXEC_TERMINAL_COUNT],
                                    unsigned int expert_qtype)
{
    memset(decisions, 0, sizeof(*decisions) * QUANT_EXEC_TERMINAL_COUNT);
    decisions[0].qtype = YVEX_GGUF_QTYPE_BF16;
    decisions[0].rank = 1u;
    decisions[0].dims[0] = 4096u;
    decisions[1].qtype = YVEX_GGUF_QTYPE_Q8_0;
    decisions[1].approximation = 1;
    decisions[1].rank = 2u;
    decisions[1].dims[0] = 128u;
    decisions[1].dims[1] = 64u;
    decisions[2].qtype = YVEX_GGUF_QTYPE_I32;
    decisions[2].rank = 1u;
    decisions[2].dims[0] = 4u;
    decisions[3].qtype = expert_qtype;
    decisions[3].approximation = 1;
    decisions[3].rank = 3u;
    decisions[3].dims[0] = QUANT_EXEC_EXPERT_LOGICAL_COLUMNS;
    decisions[3].dims[1] = 1u;
    decisions[3].dims[2] = QUANT_EXEC_EXPERTS;
}

static int quant_fixture_build_selected_plan(quant_execute_fixture *fixture,
                                             unsigned int expert_qtype,
                                             unsigned long long lowering_identity,
                                             yvex_quant_plan **out, yvex_error *err)
{
    yvex_quant_failure quant_failure;
    yvex_quant_explicit_decision decisions[QUANT_EXEC_TERMINAL_COUNT];
    int rc;

    quant_fixture_decisions(decisions, expert_qtype);
    rc = yvex_quant_plan_build_explicit(
        out, fixture->ir, fixture->binding, "quant-executor-four-ops-v1", lowering_identity,
        decisions,
        QUANT_EXEC_TERMINAL_COUNT, NULL, &quant_failure, err);
    return rc == YVEX_OK && *out && yvex_quant_plan_summary_get(*out) &&
           yvex_quant_plan_summary_get(*out)->complete;
}

static int quant_fixture_build_plan(quant_execute_fixture *fixture, yvex_error *err)
{
    yvex_transform_failure transform_failure;
    const yvex_quant_plan_summary *summary;
    int rc;

    rc = yvex_transform_binding_create(&fixture->binding, fixture->ir, fixture->session, NULL,
                                       &transform_failure, err);
    if (rc != YVEX_OK) return 0;
    if (!quant_fixture_build_selected_plan(fixture, YVEX_GGUF_QTYPE_Q2_K, 0xf00dcafeu,
                                           &fixture->plan, err))
        return 0;
    summary = yvex_quant_plan_summary_get(fixture->plan);
    return summary->source_value_count == QUANT_EXEC_SOURCE_COUNT &&
           summary->decision_count == QUANT_EXEC_TERMINAL_COUNT &&
           summary->encoded_bytes == 726544u;
}

/* Constructs a completely trusted four-operation fixture. */
static void quant_fixture_release(quant_execute_fixture *fixture);

static int quant_fixture_create(
    quant_execute_fixture *fixture,
    const char *suffix,
    int bad_cast,
    int bad_scale,
    yvex_error *err)
{
    memset(fixture, 0, sizeof(*fixture));
    (void)snprintf(fixture->root, sizeof(fixture->root),
                   "build/tests/quant-execute-%s-XXXXXX", suffix);
    if (!quant_mkdir("build") || !quant_mkdir("build/tests") ||
        !mkdtemp(fixture->root))
        return 0;
    (void)snprintf(fixture->shard_path, sizeof(fixture->shard_path),
                   "%s/quant-execute-00001.safetensors", fixture->root);
    (void)snprintf(fixture->manifest_path, sizeof(fixture->manifest_path),
                   "%s/manifest.json", fixture->root);
    if (!quant_fixture_write_payload(fixture, bad_cast, bad_scale, err) ||
        !quant_fixture_open_session(fixture, err) ||
        !quant_fixture_build_ir(fixture, err) ||
        !quant_fixture_build_plan(fixture, err)) {
        quant_fixture_release(fixture);
        return 0;
    }
    return 1;
}

static void quant_fixture_release(quant_execute_fixture *fixture)
{
    if (!fixture) return;
    yvex_quant_plan_release(&fixture->plan);
    yvex_transform_binding_release(&fixture->binding);
    yvex_transform_ir_release(&fixture->ir);
    (void)yvex_source_payload_session_release(&fixture->session, NULL, NULL);
    yvex_source_tensor_snapshot_release(fixture->snapshot);
    fixture->snapshot = NULL;
    if (fixture->shard_path[0]) (void)unlink(fixture->shard_path);
    if (fixture->manifest_path[0]) (void)unlink(fixture->manifest_path);
    if (fixture->root[0]) (void)rmdir(fixture->root);
}

static int quant_fixture_execute(
    quant_execute_fixture *fixture,
    size_t source_chunk,
    size_t output_chunk,
    unsigned int workers,
    yvex_quant_digest_summary *digest,
    yvex_quant_execution_summary *execution,
    yvex_quant_failure *failure,
    yvex_error *err)
{
    yvex_source_payload_session_facts session_facts;
    yvex_quant_digest_sink *digest_sink = NULL;
    yvex_quant_output_sink sink;
    yvex_quant_executor_options options;
    int rc;

    if (yvex_source_payload_session_facts_get(
            fixture->session, &session_facts, err) != YVEX_OK) return 0;
    rc = yvex_quant_digest_sink_create(
        &digest_sink, fixture->plan, session_facts.payload_identity,
        failure, err);
    if (rc != YVEX_OK) return 0;
    yvex_quant_digest_sink_adapter(digest_sink, &sink);
    yvex_quant_executor_options_default(&options);
    options.worker_count = workers;
    options.source_chunk_bytes = source_chunk;
    options.output_chunk_bytes = output_chunk;
    options.maximum_owned_bytes = 8u * 1024u * 1024u;
    rc = yvex_quant_execute(
        fixture->plan, &sink, &options, execution, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_quant_digest_sink_finalize(
            digest_sink, digest, failure, err);
    yvex_quant_digest_sink_release(&digest_sink);
    return rc == YVEX_OK;
}

static int quant_test_executor_success(void)
{
    quant_execute_fixture fixture;
    yvex_quant_digest_summary first_digest;
    yvex_quant_digest_summary second_digest;
    yvex_quant_execution_summary first;
    yvex_quant_execution_summary second;
    yvex_quant_failure failure;
    yvex_quant_digest_sink *selection_digest = NULL;
    yvex_quant_output_sink selection_sink;
    yvex_quant_executor_options selection_options;
    yvex_quant_execution_summary selection;
    yvex_source_payload_session_facts session_facts;
    const yvex_quant_decision *selected;
    yvex_error err;
    int second_ok;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(
                         &fixture, "success", 0, 0, &err),
                     "four-operation trusted fixture must construct");
    YVEX_TEST_ASSERT(quant_fixture_execute(
                         &fixture, 4096u, 4096u, 1u, &first_digest,
                         &first, &failure, &err),
                     "single-worker four-operation execution must commit");
    YVEX_TEST_ASSERT(first.complete &&
                         first.terminals_executed == 4u &&
                         first.committed_terminals == 4u &&
                         first.aborted_terminals == 0u &&
                         first.source_values_consumed ==
                             QUANT_EXEC_SOURCE_COUNT &&
                         first.source_ranges_read ==
                             QUANT_EXEC_SOURCE_COUNT &&
                         first.payload_bytes_read == 1165352u &&
                         first.encoded_output_bytes == 726544u &&
                         first.qtype_tensor_counts[YVEX_GGUF_QTYPE_BF16] == 1u &&
                         first.qtype_tensor_counts[YVEX_GGUF_QTYPE_Q8_0] == 1u &&
                         first.qtype_tensor_counts[YVEX_GGUF_QTYPE_I32] == 1u &&
                         first.qtype_tensor_counts[YVEX_GGUF_QTYPE_Q2_K] == 1u &&
                         first.numeric_bound_violations == 0u &&
                         first.peak_owned_bytes <= first.configured_memory_budget &&
                         first_digest.complete &&
                         first_digest.committed_terminals == 4u,
                     "execution accounting must cover every operation exactly");
    second_ok = quant_fixture_execute(
        &fixture, 8192u, 8192u, 4u, &second_digest,
        &second, &failure, &err);
    if (!second_ok || !second.complete ||
        strcmp(first_digest.execution_identity,
               second_digest.execution_identity) != 0)
        fprintf(stderr,
                "quant executor repeat ok=%d complete=%d failure=%u "
                "terminal=%llu source=%llu expected=%llu actual=%llu "
                "committed=%llu aborted=%llu workers=%u error=%s first=%s second=%s\n",
                second_ok, second.complete, failure.code,
                failure.terminal_ordinal, failure.source_index,
                failure.expected, failure.actual,
                second.committed_terminals, second.aborted_terminals,
                second.workers_started, err.message,
                first_digest.execution_identity,
                second_ok ? second_digest.execution_identity : "unavailable");
    YVEX_TEST_ASSERT(second_ok && second.complete &&
                         second.workers_started == 4u &&
                         strcmp(first_digest.execution_identity,
                                second_digest.execution_identity) == 0 &&
                         strcmp(first_digest.payload_byte_identity,
                                second_digest.payload_byte_identity) == 0,
                     "chunk boundaries and worker scheduling preserve identity");
    YVEX_TEST_ASSERT(first.source_chunks != second.source_chunks,
                     "different source chunks must exercise distinct plans");
    selected = yvex_quant_plan_decision_at(fixture.plan, 3u);
    YVEX_TEST_ASSERT(selected &&
                         yvex_source_payload_session_facts_get(
                             fixture.session, &session_facts, &err) == YVEX_OK &&
                         yvex_quant_digest_sink_create(
                             &selection_digest, fixture.plan, session_facts.payload_identity,
                             &failure, &err) == YVEX_OK,
                     "one-terminal probe sink must bind the complete physical plan");
    yvex_quant_digest_sink_adapter(selection_digest, &selection_sink);
    yvex_quant_executor_options_default(&selection_options);
    selection_options.first_terminal = 3u;
    selection_options.terminal_count = 1u;
    selection_options.source_chunk_bytes = 4096u;
    selection_options.output_chunk_bytes = 4096u;
    selection_options.maximum_owned_bytes = 8u * 1024u * 1024u;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &selection_sink, &selection_options, &selection,
                         &failure, &err) == YVEX_OK && selection.complete &&
                         selection.partial_plan_execution &&
                         selection.plan_terminal_decisions == QUANT_EXEC_TERMINAL_COUNT &&
                         selection.first_terminal == 3u && selection.terminal_decisions == 1u &&
                         selection.terminals_executed == 1u &&
                         selection.committed_terminals == 1u &&
                         selection.source_values_consumed == QUANT_EXEC_EXPERTS * 2u &&
                         selection.encoded_output_bytes == selected->encoded_bytes &&
                         selection.qtype_tensor_counts[YVEX_GGUF_QTYPE_Q2_K] == 1u &&
                         selection.role_metrics[YVEX_TENSOR_ROLE_MOE_EXPERT_GATE].element_count ==
                             selected->element_count,
                     "one-terminal execution must preserve exact bounded accounting and metrics");
    yvex_quant_digest_sink_release(&selection_digest);
    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &selection_digest, fixture.plan, session_facts.payload_identity,
                         &failure, &err) == YVEX_OK,
                     "invalid-selection refusal sink must construct");
    yvex_quant_digest_sink_adapter(selection_digest, &selection_sink);
    selection_options.first_terminal = QUANT_EXEC_TERMINAL_COUNT;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &selection_sink, &selection_options, &selection,
                         &failure, &err) != YVEX_OK && !selection.complete &&
                         selection.terminals_attempted == 0u,
                     "terminal selection beyond the sealed plan must refuse before payload work");
    yvex_quant_digest_sink_release(&selection_digest);
    quant_fixture_release(&fixture);
    return 0;
}

static int quant_test_variant_file(void)
{
    quant_execute_fixture fixture;
    char path[640];
    char link_path[640];
    FILE *file;
    int byte;
    yvex_error err;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(&fixture, "variant-file", 0, 0, &err),
                     "physical-variant fixture must construct");
    (void)snprintf(path, sizeof(path), "%s/physical.plan", fixture.root);
    (void)snprintf(link_path, sizeof(link_path), "%s/physical-link.plan", fixture.root);
    YVEX_TEST_ASSERT(yvex_quant_plan_file_write(path, fixture.plan, &err) == YVEX_OK &&
                         yvex_quant_plan_file_validate(path, fixture.plan, &err) == YVEX_OK &&
                         yvex_quant_plan_file_write(path, fixture.plan, &err) != YVEX_OK,
                     "canonical physical plan must publish once and revalidate exactly");
    YVEX_TEST_ASSERT(symlink("physical.plan", link_path) == 0 &&
                         yvex_quant_plan_file_validate(link_path, fixture.plan, &err) != YVEX_OK,
                     "physical plan symlink admission must refuse");
    file = fopen(path, "r+b");
    YVEX_TEST_ASSERT(file && (byte = fgetc(file)) != EOF && fseek(file, 0L, SEEK_SET) == 0 &&
                         fputc(byte ^ 1, file) != EOF && fclose(file) == 0 &&
                         yvex_quant_plan_file_validate(path, fixture.plan, &err) != YVEX_OK,
                     "one external physical-plan mutation must refuse admission");
    (void)unlink(link_path);
    (void)unlink(path);
    quant_fixture_release(&fixture);
    return 0;
}

/* Proves payload recipe identity tracks byte semantics, not opaque lowering provenance. */
static int quant_test_payload_recipe_identity(void)
{
    quant_execute_fixture fixture;
    yvex_quant_plan *q8_plan = NULL;
    yvex_quant_plan *provenance_plan = NULL;
    yvex_quant_plan *original_plan;
    yvex_quant_digest_summary q2_digest;
    yvex_quant_digest_summary q8_digest;
    yvex_quant_execution_summary execution;
    yvex_quant_failure failure;
    yvex_error err;
    const yvex_quant_plan_summary *q2_summary;
    const yvex_quant_plan_summary *q8_summary;
    const yvex_quant_plan_summary *provenance_summary;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(&fixture, "payload-recipe", 0, 0, &err),
                     "payload-recipe fixture must construct");
    original_plan = fixture.plan;
    YVEX_TEST_ASSERT(quant_fixture_build_selected_plan(
                         &fixture, YVEX_GGUF_QTYPE_Q8_0, 0xf00dcafeu, &q8_plan, &err) &&
                         quant_fixture_build_selected_plan(
                             &fixture, YVEX_GGUF_QTYPE_Q2_K, 0x5aa55aa5u,
                             &provenance_plan, &err),
                     "alternate physical and provenance plans must seal");
    q2_summary = yvex_quant_plan_summary_get(original_plan);
    q8_summary = yvex_quant_plan_summary_get(q8_plan);
    provenance_summary = yvex_quant_plan_summary_get(provenance_plan);
    YVEX_TEST_ASSERT(q2_summary && q8_summary && provenance_summary &&
                         strcmp(q2_summary->payload_plan_identity,
                                q8_summary->payload_plan_identity) != 0 &&
                         strcmp(q2_summary->payload_plan_identity,
                                provenance_summary->payload_plan_identity) == 0 &&
                         q2_summary->encoded_bytes != q8_summary->encoded_bytes,
                     "byte-changing codec selection changes recipe while provenance does not");
    YVEX_TEST_ASSERT(quant_fixture_execute(&fixture, 4096u, 4096u, 1u, &q2_digest,
                                           &execution, &failure, &err),
                     "baseline payload recipe must execute");
    fixture.plan = q8_plan;
    YVEX_TEST_ASSERT(quant_fixture_execute(&fixture, 4096u, 4096u, 1u, &q8_digest,
                                           &execution, &failure, &err),
                     "changed payload recipe must execute");
    fixture.plan = original_plan;
    YVEX_TEST_ASSERT(strcmp(q2_digest.payload_byte_identity,
                            q8_digest.payload_byte_identity) != 0 &&
                         q2_digest.encoded_bytes != q8_digest.encoded_bytes,
                     "byte-changing recipe produces a distinct aggregate payload identity");
    yvex_quant_plan_release(&q8_plan);
    yvex_quant_plan_release(&provenance_plan);
    quant_fixture_release(&fixture);
    return 0;
}

static int quant_test_cast_refusal(void)
{
    quant_execute_fixture fixture;
    yvex_quant_digest_summary digest;
    yvex_quant_execution_summary execution;
    yvex_quant_failure failure;
    yvex_error err;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(
                         &fixture, "cast-refusal", 1, 0, &err),
                     "out-of-range cast fixture must construct");
    YVEX_TEST_ASSERT(!quant_fixture_execute(
                         &fixture, 4096u, 4096u, 1u, &digest,
                         &execution, &failure, &err) &&
                         failure.code == YVEX_QUANT_FAILURE_CAST_RANGE &&
                         failure.terminal_ordinal == 2u &&
                         execution.aborted_terminals == 1u &&
                         execution.committed_terminals == 2u &&
                         !execution.complete,
                     "I64 overflow must abort only the uncommitted cast tensor");
    quant_fixture_release(&fixture);
    return 0;
}

static int quant_test_negative_cast_refusal(void)
{
    quant_execute_fixture fixture;
    yvex_quant_digest_summary digest;
    yvex_quant_execution_summary execution;
    yvex_quant_failure failure;
    yvex_error err;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(
                         &fixture, "negative-cast-refusal", 2, 0, &err),
                     "negative out-of-range cast fixture must construct");
    YVEX_TEST_ASSERT(!quant_fixture_execute(
                         &fixture, 4096u, 4096u, 1u, &digest,
                         &execution, &failure, &err) &&
                         failure.code == YVEX_QUANT_FAILURE_CAST_RANGE &&
                         failure.terminal_ordinal == 2u &&
                         execution.aborted_terminals == 1u &&
                         execution.committed_terminals == 2u &&
                         !execution.complete,
                     "I64 underflow must abort the uncommitted cast tensor");
    quant_fixture_release(&fixture);
    return 0;
}

static int quant_test_scale_refusal(void)
{
    quant_execute_fixture fixture;
    yvex_quant_digest_summary digest;
    yvex_quant_execution_summary execution;
    yvex_quant_failure failure;
    yvex_error err;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(
                         &fixture, "scale-refusal", 0, 1, &err),
                     "malformed scale fixture must construct");
    YVEX_TEST_ASSERT(!quant_fixture_execute(
                         &fixture, 8192u, 4096u, 1u, &digest,
                         &execution, &failure, &err) &&
                         failure.code == YVEX_QUANT_FAILURE_NONFINITE &&
                         failure.terminal_ordinal == 1u &&
                         failure.source_index != ULLONG_MAX &&
                         execution.aborted_terminals == 1u &&
                         execution.committed_terminals == 1u &&
                         !execution.complete,
                     "non-finite scale must abort before paired output commit");
    quant_fixture_release(&fixture);
    return 0;
}

/* Sweeps each distinct executor ownership transition across all four operations. */
static int quant_test_executor_allocation_sweep(void)
{
    quant_execute_fixture fixture;
    yvex_quant_digest_sink *digest_sink = NULL;
    yvex_quant_output_sink sink;
    yvex_quant_execution_summary execution;
    yvex_quant_executor_options options;
    yvex_quant_failure failure;
    yvex_error err;
    unsigned long long fail_at;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(
                         &fixture, "allocation-sweep", 0, 0, &err),
                     "allocation-sweep fixture must construct");
    for (fail_at = 0u; fail_at < 8u; ++fail_at) {
        quant_execute_allocator allocator;
        const yvex_quant_plan_summary *summary =
            yvex_quant_plan_summary_get(fixture.plan);

        memset(&allocator, 0, sizeof(allocator));
        allocator.fail_at = fail_at;
        YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                             &digest_sink, fixture.plan,
                             summary->required_payload_identity,
                             &failure, &err) == YVEX_OK,
                         "allocation-sweep digest sink must open");
        yvex_quant_digest_sink_adapter(digest_sink, &sink);
        yvex_quant_executor_options_default(&options);
        options.source_chunk_bytes = 4096u;
        options.output_chunk_bytes = 4096u;
        options.maximum_owned_bytes = 8u * 1024u * 1024u;
        options.allocate = quant_execute_allocate;
        options.release = quant_execute_release;
        options.context = &allocator;
        YVEX_TEST_ASSERT(yvex_quant_execute(
                             fixture.plan, &sink, &options, &execution,
                             &failure, &err) != YVEX_OK &&
                             failure.code == YVEX_QUANT_FAILURE_ALLOCATION &&
                             allocator.live == 0u && !execution.complete,
                         "executor allocation refusal must unwind all ownership");
        yvex_quant_digest_sink_release(&digest_sink);
    }
    quant_fixture_release(&fixture);
    return 0;
}

typedef struct {
    unsigned long long calls;
    unsigned long long last_bytes;
    unsigned long long planned_bytes;
} quant_roundtrip_progress_fixture;

/* Captures synchronous roundtrip progress without retaining borrowed views. */
static void quant_roundtrip_progress_capture(
    void *opaque,
    const yvex_gguf_roundtrip_summary *summary,
    unsigned long long planned_file_bytes)
{
    quant_roundtrip_progress_fixture *fixture =
        (quant_roundtrip_progress_fixture *)opaque;
    if (!fixture || !summary) return;
    fixture->calls++;
    fixture->last_bytes = summary->bytes_hashed;
    fixture->planned_bytes = planned_file_bytes;
}

/*
 * Exercises structural planning, transactional file delivery, native
 * roundtrip, physical corruption refusal, support refusal without tokenizer,
 * and deterministic cleanup on preallocation/write/incomplete faults.
 */
int yvex_test_gguf_writer_artifact(void)
{
    static const yvex_gguf_writer_proof_tensor names[] = {
        {"fixture.direct"}, {"fixture.scale-pair"},
        {"fixture.cast"}, {"fixture.experts"}
    };
    static const yvex_gguf_writer_proof_tensor duplicate_names[] = {
        {"fixture.duplicate"}, {"fixture.duplicate"},
        {"fixture.cast"}, {"fixture.experts"}
    };
    quant_execute_fixture fixture;
    yvex_gguf_writer_plan *writer = NULL;
    yvex_gguf_writer_plan *duplicate_writer = NULL;
    yvex_quant_plan *source_faithful = NULL;
    yvex_gguf_file_sink *file_sink = NULL;
    yvex_quant_output_sink output_sink;
    yvex_quant_executor_options executor_options;
    yvex_quant_execution_summary execution;
    yvex_gguf_file_sink_options file_options;
    yvex_gguf_file_sink_summary emission;
    yvex_gguf_file_sink_summary tampered_emission;
    yvex_gguf_writer_plan_request writer_request;
    yvex_gguf_roundtrip_options roundtrip_options;
    yvex_gguf_roundtrip_summary roundtrip;
    yvex_gguf_writer_failure writer_failure;
    yvex_gguf_file_failure file_failure;
    yvex_gguf_roundtrip_failure roundtrip_failure;
    yvex_artifact_official_reader_fact official;
    yvex_artifact_admission_request admission_request;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_artifact_file_identity independent_identity;
    yvex_quant_failure quant_failure;
    yvex_quant_failure protocol_failure;
    yvex_error err;
    quant_roundtrip_progress_fixture progress_fixture;
    const yvex_gguf_writer_plan_summary *summary;
    const yvex_gguf_writer_tensor *first_tensor;
    const yvex_quant_decision *first_decision;
    char artifact_path[640];
    char corrupt_path[640];
    char truncated_path[640];
    char tail_path[640];
    char drift_path[640];
    char partial_path[640];
    char symlink_directory[640];
    char symlink_path[700];
    char traversal_path[700];
    char fault_path[640];
    char temp_path[YVEX_ARTIFACT_PATH_CAP];
    FILE *fp;
    int byte;
    unsigned char protocol_byte = 0u;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(quant_fixture_create(
                         &fixture, "gguf-writer", 0, 0, &err),
                     "writer fixture must construct trusted quant inputs");
    YVEX_TEST_ASSERT(yvex_quant_plan_build_source_faithful(
                         &source_faithful, fixture.ir, fixture.binding,
                         "invalid-source-faithful-v1", 0x112233u, NULL,
                         &quant_failure, &err) != YVEX_OK && !source_faithful &&
                         quant_failure.code == YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION,
                     "source-faithful plan refuses non-identity transformation composition");
    memset(&writer_request, 0, sizeof(writer_request));
    writer_request.input_class = YVEX_GGUF_WRITER_INPUT_TENSOR_PROOF;
    writer_request.quant_plan = fixture.plan;
    writer_request.input.tensor_proof.tensors = names;
    writer_request.input.tensor_proof.tensor_count = QUANT_EXEC_TERMINAL_COUNT;
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &writer, &writer_request, &writer_failure, &err) == YVEX_OK,
                     "explicit fixture writer plan must seal");
    summary = yvex_gguf_writer_plan_summary_get(writer);
    YVEX_TEST_ASSERT(summary && summary->complete &&
                         summary->tensor_count == QUANT_EXEC_TERMINAL_COUNT &&
                         summary->tensor_payload_bytes == 726544u &&
                         summary->final_file_bytes >
                             summary->tensor_payload_bytes &&
                         summary->tokenizer_token_count == 0u,
                     "fixture writer plan must account exact physical bytes");
    writer_request.input.tensor_proof.tensors = duplicate_names;
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &duplicate_writer, &writer_request, &writer_failure, &err) != YVEX_OK &&
                         writer_failure.code ==
                             YVEX_GGUF_WRITER_DUPLICATE_TENSOR &&
                         !duplicate_writer,
                     "duplicate tensor names must refuse during planning");
    writer_request.input_class = YVEX_GGUF_WRITER_INPUT_INVALID;
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &duplicate_writer, &writer_request, &writer_failure, &err) != YVEX_OK &&
                         writer_failure.code == YVEX_GGUF_WRITER_INVALID_ARGUMENT &&
                         !duplicate_writer,
                     "unknown writer input classes must refuse without a plan");
    writer_request.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    memset(&writer_request.input, 0, sizeof(writer_request.input));
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &duplicate_writer, &writer_request, &writer_failure, &err) != YVEX_OK &&
                         writer_failure.code == YVEX_GGUF_WRITER_INVALID_ARGUMENT &&
                         !duplicate_writer,
                     "complete input without a lowering projection must refuse");

    (void)snprintf(symlink_directory, sizeof(symlink_directory),
                   "%s/linkdir", fixture.root);
    (void)snprintf(symlink_path, sizeof(symlink_path), "%s/output.gguf",
                   symlink_directory);
    YVEX_TEST_ASSERT(symlink(".", symlink_directory) == 0,
                     "path-admission fixture symlink must construct");
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = symlink_path;
    file_options.safety_margin_bytes = 0u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) != YVEX_OK && !file_sink &&
                         file_failure.code ==
                             YVEX_GGUF_FILE_DIRECTORY_OPEN,
                     "destination directory symlink must refuse");
    YVEX_TEST_ASSERT(unlink(symlink_directory) == 0,
                     "path-admission symlink must clean up");

    (void)snprintf(traversal_path, sizeof(traversal_path),
                   "%s/../escaped.gguf", fixture.root);
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = traversal_path;
    file_options.safety_margin_bytes = 0u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) != YVEX_OK && !file_sink &&
                         file_failure.code ==
                             YVEX_GGUF_FILE_UNSAFE_DESTINATION,
                     "destination traversal must refuse before file creation");

    (void)snprintf(artifact_path, sizeof(artifact_path), "%s/output.gguf",
                   fixture.root);
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = artifact_path;
    file_options.safety_margin_bytes = 0u;
    file_options.inject_temp_create_failure = 1;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) != YVEX_OK && !file_sink &&
                         file_failure.code == YVEX_GGUF_FILE_TEMP_CREATE,
                     "temporary-file creation failure must publish no sink");

    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = artifact_path;
    file_options.safety_margin_bytes = 0u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "transactional fixture file sink must preallocate");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    yvex_quant_executor_options_default(&executor_options);
    executor_options.worker_count = 4u;
    executor_options.source_chunk_bytes = 4096u;
    executor_options.output_chunk_bytes = 4096u;
    executor_options.maximum_owned_bytes = 8u * 1024u * 1024u;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         execution.complete,
                     "quant executor must deliver every fixture terminal");
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_finalize(
                         file_sink, &emission, &file_failure, &err) ==
                         YVEX_OK && emission.finalized &&
                         emission.committed_terminals == 4u &&
                         emission.encoded_bytes_written == 726544u &&
                         emission.source_ranges_committed ==
                             execution.source_ranges_read &&
                         emission.source_bytes_committed ==
                             execution.payload_bytes_read,
                     "exact fixture file must finalize transactionally");
    yvex_gguf_roundtrip_options_default(&roundtrip_options);
    roundtrip_options.verification_chunk_bytes = 4096u;
    memset(&progress_fixture, 0, sizeof(progress_fixture));
    roundtrip_options.progress = quant_roundtrip_progress_capture;
    roundtrip_options.progress_context = &progress_fixture;
    YVEX_TEST_ASSERT(yvex_gguf_roundtrip_validate(
                         temp_path, writer,
                         yvex_gguf_file_sink_digest(file_sink),
                         &roundtrip_options, &roundtrip,
                         &roundtrip_failure, &err) == YVEX_OK &&
                         roundtrip.complete && roundtrip.payload_accepted &&
                         strcmp(roundtrip.payload_byte_identity,
                                emission.payload_byte_identity) == 0 &&
                         !roundtrip.tokenizer_complete &&
                         roundtrip.bytes_hashed == summary->final_file_bytes &&
                         progress_fixture.calls > 0u &&
                         progress_fixture.last_bytes ==
                             summary->final_file_bytes &&
                         progress_fixture.planned_bytes ==
                             summary->final_file_bytes,
                     "native roundtrip must verify every fixture byte");
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_publish(
                         file_sink, &roundtrip, &emission,
                         &file_failure, &err) ==
                         YVEX_OK && emission.published,
                     "validated fixture must publish without replacement");
    YVEX_TEST_ASSERT(yvex_artifact_identity_read(
                         artifact_path, &independent_identity, &err) ==
                         YVEX_OK &&
                         independent_identity.file_size ==
                             roundtrip.file_bytes &&
                         strcmp(independent_identity.sha256,
                                roundtrip.artifact_identity) == 0,
                     "streaming artifact identity must match independent exact-file identity");

    memset(&official, 0, sizeof(official));
    (void)snprintf(official.revision, sizeof(official.revision), "%s",
                   YVEX_GGUF_OFFICIAL_READER_REVISION);
    official.metadata_count = summary->metadata_count;
    official.tensor_count = summary->tensor_count;
    official.file_bytes = summary->final_file_bytes;
    official.file_device = emission.file_device;
    official.file_inode = emission.file_inode;
    official.file_mtime_seconds = emission.validated_mtime_seconds;
    official.file_mtime_nanoseconds = emission.validated_mtime_nanoseconds;
    official.file_ctime_seconds = emission.validated_ctime_seconds;
    official.file_ctime_nanoseconds = emission.validated_ctime_nanoseconds;
    official.accepted = 1;
    memset(&admission_request, 0, sizeof(admission_request));
    admission_request.artifact_path = artifact_path;
    admission_request.writer_plan = writer;
    admission_request.emission = &emission;
    admission_request.native_roundtrip = &roundtrip;
    admission_request.official_reader = &official;
    tampered_emission = emission;
    tampered_emission.payload_byte_identity[0] =
        tampered_emission.payload_byte_identity[0] == '0' ? '1' : '0';
    admission_request.emission = &tampered_emission;
    YVEX_TEST_ASSERT(yvex_complete_artifact_admit(
                         &admission_request, &admission,
                         &admission_failure, &err) != YVEX_OK &&
                         admission_failure.code ==
                             YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH &&
                         !admission.complete,
                     "tampered emission payload identity must refuse admission");
    admission_request.emission = &emission;
    YVEX_TEST_ASSERT(yvex_complete_artifact_admit(
                         &admission_request, &admission,
                         &admission_failure, &err) != YVEX_OK &&
                         admission_failure.code ==
                             YVEX_ARTIFACT_ADMISSION_TOKENIZER_INCOMPLETE &&
                         !admission.complete,
                     "tensor proof without tokenizer must not enter complete path");
    (void)snprintf(official.revision, sizeof(official.revision), "%s",
                   "0000000000000000000000000000000000000000");
    YVEX_TEST_ASSERT(yvex_complete_artifact_admit(
                         &admission_request, &admission,
                         &admission_failure, &err) != YVEX_OK &&
                         admission_failure.code ==
                             YVEX_ARTIFACT_ADMISSION_OFFICIAL_READER &&
                         !admission.complete,
                     "mismatched official-reader revision must refuse admission");

    (void)snprintf(corrupt_path, sizeof(corrupt_path), "%s/corrupt.gguf",
                   fixture.root);
    YVEX_TEST_ASSERT(quant_fixture_file_copy(artifact_path, corrupt_path),
                     "corruption proof must use an independent fixture copy");
    first_tensor = yvex_gguf_writer_plan_tensor_at(writer, 0u);
    fp = fopen(corrupt_path, "r+b");
    YVEX_TEST_ASSERT(fp && first_tensor &&
                         fseeko(fp, (off_t)first_tensor->absolute_offset,
                                SEEK_SET) == 0,
                     "published fixture payload must be addressable for corruption");
    byte = fgetc(fp);
    YVEX_TEST_ASSERT(byte != EOF &&
                         fseeko(fp, (off_t)first_tensor->absolute_offset,
                                SEEK_SET) == 0 &&
                         fputc(byte ^ 0x5a, fp) != EOF && fclose(fp) == 0,
                     "bounded corruption fixture must mutate one payload byte");
    fp = NULL;
    YVEX_TEST_ASSERT(yvex_gguf_roundtrip_validate(
                         corrupt_path, writer,
                         yvex_gguf_file_sink_digest(file_sink),
                         &roundtrip_options, &roundtrip,
                         &roundtrip_failure, &err) != YVEX_OK &&
                         roundtrip_failure.code ==
                             YVEX_GGUF_ROUNDTRIP_PAYLOAD_DIGEST,
                     "one corrupted payload byte must fail terminal digest");
    YVEX_TEST_ASSERT(unlink(corrupt_path) == 0,
                     "corruption fixture copy must clean up deterministically");

    (void)snprintf(truncated_path, sizeof(truncated_path),
                   "%s/truncated.gguf", fixture.root);
    YVEX_TEST_ASSERT(quant_fixture_file_copy(artifact_path, truncated_path),
                     "truncation proof must use an independent fixture copy");
    fp = fopen(truncated_path, "r+b");
    YVEX_TEST_ASSERT(fp && summary->final_file_bytes > 0u &&
                         ftruncate(fileno(fp),
                                   (off_t)(summary->final_file_bytes - 1u)) == 0 &&
                         fclose(fp) == 0,
                     "truncation fixture must remove one physical byte");
    fp = NULL;
    YVEX_TEST_ASSERT(yvex_gguf_roundtrip_validate(
                         truncated_path, writer,
                         yvex_gguf_file_sink_digest(file_sink),
                         &roundtrip_options, &roundtrip,
                         &roundtrip_failure, &err) != YVEX_OK &&
                         roundtrip_failure.code ==
                             YVEX_GGUF_ROUNDTRIP_HEADER_DIVERGENCE,
                     "one-byte truncation must refuse before payload acceptance");
    YVEX_TEST_ASSERT(unlink(truncated_path) == 0,
                     "truncation fixture must clean up deterministically");

    (void)snprintf(tail_path, sizeof(tail_path), "%s/extra-tail.gguf",
                   fixture.root);
    YVEX_TEST_ASSERT(quant_fixture_file_copy(artifact_path, tail_path),
                     "extra-tail proof must use an independent fixture copy");
    fp = fopen(tail_path, "ab");
    YVEX_TEST_ASSERT(fp && fputc(0, fp) != EOF && fclose(fp) == 0,
                     "extra-tail fixture must append one physical byte");
    fp = NULL;
    YVEX_TEST_ASSERT(yvex_gguf_roundtrip_validate(
                         tail_path, writer,
                         yvex_gguf_file_sink_digest(file_sink),
                         &roundtrip_options, &roundtrip,
                         &roundtrip_failure, &err) != YVEX_OK &&
                         roundtrip_failure.code ==
                             YVEX_GGUF_ROUNDTRIP_HEADER_DIVERGENCE,
                     "one-byte trailing data must refuse before payload acceptance");
    YVEX_TEST_ASSERT(unlink(tail_path) == 0,
                     "extra-tail fixture must clean up deterministically");
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_withdraw(
                         file_sink, &file_failure, &err) == YVEX_OK &&
                         access(artifact_path, F_OK) != 0,
                     "owned refused artifact must withdraw and fsync cleanup");
    yvex_gguf_file_sink_release(&file_sink);

    (void)snprintf(drift_path, sizeof(drift_path), "%s/drift.gguf",
                   fixture.root);
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = drift_path;
    file_options.safety_margin_bytes = 0u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "snapshot drift fixture must preallocate");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_finalize(
                             file_sink, &emission, &file_failure, &err) ==
                             YVEX_OK,
                     "snapshot drift fixture must reach finalized state");
    fp = fopen(temp_path, "r+b");
    YVEX_TEST_ASSERT(fp && first_tensor &&
                         fseeko(fp, (off_t)first_tensor->absolute_offset,
                                SEEK_SET) == 0,
                     "finalized temporary artifact must be mutable by drift test");
    byte = fgetc(fp);
    YVEX_TEST_ASSERT(byte != EOF &&
                         fseeko(fp, (off_t)first_tensor->absolute_offset,
                                SEEK_SET) == 0 &&
                         fputc(byte ^ 0x33, fp) != EOF && fclose(fp) == 0,
                     "drift fixture must change the finalized inode");
    fp = NULL;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_publish(
                         file_sink, &roundtrip, &emission,
                         &file_failure, &err) !=
                         YVEX_OK &&
                         file_failure.code == YVEX_GGUF_FILE_SNAPSHOT_DRIFT &&
                         access(drift_path, F_OK) != 0,
                     "changed finalized inode must refuse before publication");
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0,
                     "snapshot drift refusal must remove its owned temp");

    (void)snprintf(partial_path, sizeof(partial_path), "%s/partial.gguf",
                   fixture.root);
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = partial_path;
    file_options.safety_margin_bytes = 0u;
    file_options.injected_write_eintr_call = 1u;
    file_options.injected_write_max_bytes = 1024u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "EINTR retry and bounded partial writes must admit");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_finalize(
                             file_sink, &emission, &file_failure, &err) ==
                             YVEX_OK && emission.write_calls >
                                 emission.output_chunks &&
                         yvex_gguf_roundtrip_validate(
                             temp_path, writer,
                             yvex_gguf_file_sink_digest(file_sink),
                             &roundtrip_options, &roundtrip,
                             &roundtrip_failure, &err) == YVEX_OK,
                     "partial/EINTR path must produce exact roundtrip bytes");
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(partial_path, F_OK) != 0,
                     "unpublished retry fixture must clean up exactly");

    (void)snprintf(fault_path, sizeof(fault_path), "%s/fault.gguf",
                   fixture.root);
    fp = fopen(fault_path, "wb");
    YVEX_TEST_ASSERT(fp && fputc(0x5a, fp) != EOF && fclose(fp) == 0,
                     "destination-conflict fixture must create one owned sentinel");
    fp = NULL;
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = fault_path;
    file_options.safety_margin_bytes = 0u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) != YVEX_OK && !file_sink &&
                         file_failure.code ==
                             YVEX_GGUF_FILE_DESTINATION_EXISTS,
                     "pre-existing destination must refuse without replacement");
    fp = fopen(fault_path, "rb");
    YVEX_TEST_ASSERT(fp && fgetc(fp) == 0x5a && fclose(fp) == 0 &&
                         unlink(fault_path) == 0,
                     "destination refusal must preserve the pre-existing file");
    fp = NULL;

    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = fault_path;
    file_options.safety_margin_bytes = ULLONG_MAX;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) != YVEX_OK && !file_sink &&
                         file_failure.code ==
                             YVEX_GGUF_FILE_INSUFFICIENT_SPACE &&
                         access(fault_path, F_OK) != 0,
                     "capacity arithmetic overflow must refuse before temp creation");

    first_decision = yvex_quant_plan_decision_at(fixture.plan, 0u);
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = fault_path;
    file_options.safety_margin_bytes = 0u;
    YVEX_TEST_ASSERT(first_decision && yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "terminal-protocol fixture must construct");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    memset(&protocol_failure, 0, sizeof(protocol_failure));
    YVEX_TEST_ASSERT(output_sink.begin_terminal(
                         output_sink.context, first_decision) == 0 &&
                         output_sink.begin_terminal(
                             output_sink.context, first_decision) != 0 &&
                         output_sink.deliver_chunk(
                             output_sink.context, first_decision, 0u,
                             &protocol_byte, 1u) != 0,
                     "duplicate begin must poison subsequent terminal delivery");
    output_sink.abort_terminal(output_sink.context, first_decision,
                               &protocol_failure, 0u);
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(fault_path, F_OK) != 0,
                     "terminal protocol abort must remove its unpublished temp");

    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "overlapping-chunk fixture must construct independently");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(output_sink.begin_terminal(
                         output_sink.context, first_decision) == 0 &&
                         output_sink.deliver_chunk(
                             output_sink.context, first_decision, 0u,
                             &protocol_byte, 1u) == 0 &&
                         output_sink.deliver_chunk(
                             output_sink.context, first_decision, 0u,
                             &protocol_byte, 1u) != 0,
                     "overlapping terminal chunks must poison the session");
    output_sink.abort_terminal(output_sink.context, first_decision,
                               &protocol_failure, 1u);
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(fault_path, F_OK) != 0,
                     "overlapping chunk refusal must remove its owned temp");

    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "out-of-range terminal fixture must construct");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(output_sink.begin_terminal(
                         output_sink.context, first_decision) == 0 &&
                         output_sink.deliver_chunk(
                             output_sink.context, first_decision,
                             first_decision->encoded_bytes,
                             &protocol_byte, 1u) != 0,
                     "terminal delivery beyond its planned range must refuse");
    output_sink.abort_terminal(output_sink.context, first_decision,
                               &protocol_failure, 0u);
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(fault_path, F_OK) != 0,
                     "out-of-range refusal must clean its owned temporary file");

    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "duplicate-commit fixture must construct");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         output_sink.commit_terminal(
                             output_sink.context, first_decision,
                             first_decision->encoded_bytes) != 0 &&
                         yvex_gguf_file_sink_finalize(
                             file_sink, &emission, &file_failure, &err) !=
                             YVEX_OK,
                     "duplicate terminal commit must poison publication");
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(fault_path, F_OK) != 0,
                     "duplicate commit refusal must clean its owned temp");

    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = fault_path;
    file_options.safety_margin_bytes = 0u;
    file_options.inject_preallocate_failure = 1;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) != YVEX_OK &&
                         file_failure.code == YVEX_GGUF_FILE_PREALLOCATE &&
                         !file_sink && access(fault_path, F_OK) != 0,
                     "preallocation failure must leave no output or sink");
    file_options.inject_preallocate_failure = 0;
    file_options.injected_write_failure_call = 1u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) != YVEX_OK &&
                         file_failure.code == YVEX_GGUF_FILE_WRITE &&
                         !file_sink && access(fault_path, F_OK) != 0,
                     "prefix write failure must remove its owned temp");
    file_options.injected_write_failure_call = 2u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "payload-write-failure sink must pass structural prefix");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) != YVEX_OK &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_SINK_SHORT_WRITE &&
                         !execution.complete,
                     "hard positioned payload write failure must abort execution");
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(fault_path, F_OK) != 0,
                     "payload write failure must remove its owned temporary file");
    file_options.injected_write_failure_call = 0u;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_finalize(
                             file_sink, &emission, &file_failure, &err) !=
                             YVEX_OK &&
                         file_failure.code == YVEX_GGUF_FILE_INCOMPLETE,
                     "missing terminal commits must refuse finalization");
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(fault_path, F_OK) != 0,
                     "incomplete sink release must remove its owned temp");

    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = fault_path;
    file_options.safety_margin_bytes = 0u;
    file_options.inject_fsync_failure = 1;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "flush-failure sink must construct");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_finalize(
                             file_sink, &emission, &file_failure, &err) !=
                             YVEX_OK &&
                         file_failure.code == YVEX_GGUF_FILE_FLUSH,
                     "injected file flush must refuse finalization");
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(fault_path, F_OK) != 0,
                     "flush failure must remove only its owned temporary file");

    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = fault_path;
    file_options.safety_margin_bytes = 0u;
    file_options.inject_publish_failure = 1;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "publication-failure sink must construct");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_finalize(
                             file_sink, &emission, &file_failure, &err) ==
                             YVEX_OK &&
                         yvex_gguf_file_sink_publish(
                             file_sink, NULL, &emission,
                             &file_failure, &err) != YVEX_OK &&
                         file_failure.code ==
                             YVEX_GGUF_FILE_VALIDATION_REQUIRED &&
                         yvex_gguf_roundtrip_validate(
                             temp_path, writer,
                             yvex_gguf_file_sink_digest(file_sink),
                             &roundtrip_options, &roundtrip,
                             &roundtrip_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_publish(
                             file_sink, &roundtrip, &emission,
                             &file_failure, &err) !=
                             YVEX_OK &&
                         file_failure.code == YVEX_GGUF_FILE_PUBLICATION,
                     "injected no-replace publication must refuse");
    yvex_gguf_file_sink_release(&file_sink);
    YVEX_TEST_ASSERT(access(temp_path, F_OK) != 0 &&
                         access(fault_path, F_OK) != 0,
                     "publication failure must clean its unpublished temp");

    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = fault_path;
    file_options.safety_margin_bytes = 0u;
    file_options.inject_directory_fsync_failure = 1;
    YVEX_TEST_ASSERT(yvex_gguf_file_sink_create(
                         &file_sink, writer, fixture.plan, &file_options,
                         &file_failure, &err) == YVEX_OK,
                     "directory-flush-failure sink must construct");
    (void)snprintf(temp_path, sizeof(temp_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         fixture.plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_finalize(
                             file_sink, &emission, &file_failure, &err) ==
                             YVEX_OK &&
                         yvex_gguf_roundtrip_validate(
                             temp_path, writer,
                             yvex_gguf_file_sink_digest(file_sink),
                             &roundtrip_options, &roundtrip,
                             &roundtrip_failure, &err) == YVEX_OK &&
                         yvex_gguf_file_sink_publish(
                             file_sink, &roundtrip, &emission,
                             &file_failure, &err) !=
                             YVEX_OK &&
                         file_failure.code ==
                             YVEX_GGUF_FILE_DIRECTORY_FLUSH &&
                         access(fault_path, F_OK) != 0,
                     "directory flush failure must roll publication back");
    yvex_gguf_file_sink_release(&file_sink);
    yvex_gguf_writer_plan_release(&writer);
    quant_fixture_release(&fixture);
    return 0;
}

int yvex_test_quant_execute(void)
{
    if (quant_test_executor_success() != 0) return 1;
    if (quant_test_variant_file() != 0) return 1;
    if (quant_test_payload_recipe_identity() != 0) return 1;
    if (quant_test_cast_refusal() != 0) return 1;
    if (quant_test_negative_cast_refusal() != 0) return 1;
    if (quant_test_scale_refusal() != 0) return 1;
    if (quant_test_executor_allocation_sweep() != 0) return 1;
    return 0;
}
