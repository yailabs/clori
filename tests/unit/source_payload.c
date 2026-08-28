#define _XOPEN_SOURCE 700

#include "tests/test.h"

#include <yvex/internal/core.h>
#include <yvex/internal/source_payload.h>
#include <yvex/internal/source.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/quant_numeric.h>

#include <yvex/artifact.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PAYLOAD_COMPONENT_IDENTITY \
    "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
#define PAYLOAD_SOURCE_SNAPSHOT_IDENTITY \
    "67890abcdef1234567890abcdef1234567890abcdef1234567890abcdef12345"
#define PAYLOAD_COMPONENT_MANIFEST_IDENTITY \
    "234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1"
#define PAYLOAD_ARCHITECTURE_IDENTITY \
    "34567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef12"
#define PAYLOAD_ROLE_MAP_IDENTITY \
    "4567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123"
#define PAYLOAD_UNRESOLVED_IDENTITY \
    "567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234"

typedef struct {
    char root[512];
    char manifest[640];
    yvex_source_verify_options verify_options;
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot;
    yvex_source_payload_session *session;
} payload_fixture;

typedef struct {
    yvex_sha256 hash;
    unsigned long long bytes;
    unsigned long long chunks;
    unsigned long long aborts;
    unsigned long long commits;
    unsigned long long fail_after_chunks;
    yvex_source_payload_session *cancel_session;
    unsigned long long cancel_after_chunks;
} payload_sink_state;

typedef struct {
    unsigned long long pread_calls;
    unsigned long long eintr_at;
    unsigned long long eof_at;
    unsigned long long io_at;
    size_t partial_limit;
    unsigned long long open_calls;
    unsigned long long open_fail_at;
    unsigned long long fstat_calls;
    unsigned long long fstat_fail_at;
    unsigned long long close_calls;
    unsigned long long close_fail_at;
    unsigned long long allocation_calls;
    unsigned long long allocation_fail_at;
    long long live_allocations;
} payload_fault_state;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned int arrived;
    int released;
} payload_stream_gate;

typedef struct {
    payload_sink_state base;
    payload_stream_gate *gate;
    int waited;
} payload_blocking_sink;

static void payload_sink_make(yvex_source_payload_sink *sink,
                              payload_sink_state *state);

typedef struct {
    yvex_source_payload_session *session;
    const yvex_source_payload_plan *plan;
    yvex_source_payload_sink sink;
    yvex_source_payload_stream_result result;
    yvex_source_payload_failure failure;
    yvex_error error;
    int rc;
    int verify;
} payload_stream_thread;

typedef struct {
    unsigned int calls;
    unsigned int fail_at;
    unsigned int live;
} payload_binding_allocator;

typedef struct {
    unsigned int begins;
    unsigned int chunks;
    unsigned int commits;
    unsigned int aborts;
    unsigned int fail_chunk;
    unsigned int cancel_chunk;
    unsigned long long abort_delivered_bytes;
    yvex_quant_cancellation *cancellation;
} payload_quant_sink_state;

static payload_fault_state payload_fault;

/* Injects deterministic quantizer-binding allocation failures. */
static void *payload_binding_allocate(size_t size, void *context)
{
    payload_binding_allocator *state =
        (payload_binding_allocator *)context;
    void *allocation;

    if (state->calls++ == state->fail_at) return NULL;
    allocation = malloc(size);
    if (allocation) state->live++;
    return allocation;
}

static void payload_binding_release(void *allocation, void *context)
{
    payload_binding_allocator *state =
        (payload_binding_allocator *)context;

    if (!allocation) return;
    free(allocation);
    state->live--;
}

/* Records fixture transactions and injects one deterministic chunk refusal. */
static int payload_quant_sink_begin(
    void *context, const yvex_quant_decision *decision)
{
    payload_quant_sink_state *state = (payload_quant_sink_state *)context;
    if (!state || !decision) return 1;
    state->begins++;
    return 0;
}

static int payload_quant_sink_chunk(
    void *context,
    const yvex_quant_decision *decision,
    unsigned long long output_offset,
    const unsigned char *bytes,
    size_t byte_count)
{
    payload_quant_sink_state *state = (payload_quant_sink_state *)context;
    (void)output_offset;
    if (!state || !decision || !bytes || !byte_count) return 1;
    state->chunks++;
    if (state->cancellation && state->cancel_chunk == state->chunks)
        atomic_store_explicit(&state->cancellation->requested, 1, memory_order_release);
    return state->fail_chunk && state->chunks == state->fail_chunk;
}

static int payload_quant_sink_commit(
    void *context,
    const yvex_quant_decision *decision,
    unsigned long long delivered_bytes)
{
    payload_quant_sink_state *state = (payload_quant_sink_state *)context;
    if (!state || !decision || delivered_bytes != decision->encoded_bytes)
        return 1;
    state->commits++;
    return 0;
}

static void payload_quant_sink_abort(
    void *context,
    const yvex_quant_decision *decision,
    const yvex_quant_failure *failure,
    unsigned long long delivered_bytes)
{
    payload_quant_sink_state *state = (payload_quant_sink_state *)context;
    (void)decision;
    (void)failure;
    if (state) {
        state->aborts++;
        state->abort_delivered_bytes = delivered_bytes;
    }
}

static int payload_quant_thread_fail(
    pthread_t *thread,
    void *(*entry)(void *),
    void *argument,
    void *context)
{
    (void)thread;
    (void)entry;
    (void)argument;
    (void)context;
    return 1;
}

static int test_storage_shard_index_foundation(void)
{
    yvex_shard_index_entry entries[] = {
        {0u, "artifact-00001"},
        {1u, "artifact-00002"},
        {2u, "artifact-00003"}
    };
    yvex_shard_index index;
    unsigned long long comparisons;

    YVEX_TEST_ASSERT(yvex_shard_index_init(
                         &index, entries, 3u, 3u) == YVEX_SHARD_INDEX_OK,
                     "common shard index admits canonical entries");
    YVEX_TEST_ASSERT(yvex_shard_index_at(&index, 1u) == &entries[1] &&
                         yvex_shard_index_find(
                             &index, "artifact-00003", &comparisons) ==
                             &entries[2] && comparisons <= 2u,
                     "common shard index provides bounded deterministic lookup");
    YVEX_TEST_ASSERT(yvex_shard_index_init(
                         &index, entries, 3u, 2u) == YVEX_SHARD_INDEX_BUDGET,
                     "common shard index enforces entry budget");
    entries[2].canonical_key = entries[1].canonical_key;
    YVEX_TEST_ASSERT(yvex_shard_index_init(
                         &index, entries, 3u, 3u) ==
                         YVEX_SHARD_INDEX_DUPLICATE_KEY,
                     "common shard index refuses duplicate ownership");
    entries[2].canonical_key = "artifact-00003";
    entries[2].canonical_id = 1u;
    YVEX_TEST_ASSERT(yvex_shard_index_init(
                         &index, entries, 3u, 3u) ==
                         YVEX_SHARD_INDEX_DUPLICATE_ID,
                     "common shard index refuses duplicate canonical IDs");
    yvex_shard_index_reset(&index);
    return 0;
}

static int payload_test_mkdir(const char *path)
{
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

static int payload_test_write_bytes(const char *path,
                                    size_t prefix,
                                    size_t payload,
                                    unsigned int seed)
{
    FILE *file = fopen(path, "wb");
    size_t index;

    if (!file) return 0;
    for (index = 0u; index < prefix + payload; ++index) {
        unsigned char byte = index < prefix
            ? (unsigned char)(0xa0u + (index & 15u))
            : (unsigned char)((index + seed * 17u) & 0x3fu);
        if (fwrite(&byte, 1u, 1u, file) != 1u) {
            fclose(file);
            return 0;
        }
    }
    return fclose(file) == 0;
}

static int payload_test_write_text(const char *path, const char *text)
{
    FILE *file = fopen(path, "wb");

    if (!file) return 0;
    if (fputs(text, file) < 0) {
        fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

static int payload_test_write_sparse_gguf(const char *path,
                                          const unsigned char *prefix,
                                          size_t prefix_bytes,
                                          unsigned long long file_bytes)
{
    FILE *file = fopen(path, "wb");
    int descriptor;

    if (!file || !prefix || file_bytes < prefix_bytes) return 0;
    descriptor = fileno(file);
    if (fwrite(prefix, 1u, prefix_bytes, file) != prefix_bytes ||
        ftruncate(descriptor, (off_t)file_bytes) != 0) {
        fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

static int payload_test_read_text(const char *path, char *text, size_t cap)
{
    FILE *file = fopen(path, "rb");
    size_t got;

    if (!file || cap == 0u) {
        if (file) fclose(file);
        return 0;
    }
    got = fread(text, 1u, cap - 1u, file);
    text[got] = '\0';
    return fclose(file) == 0;
}

static int payload_acquisition_fixture_open(
    const payload_fixture *fixture, yvex_source_acquisition **out,
    yvex_source_acquisition_failure *failure, yvex_error *err)
{
    static const char repository[] = "Example/Payload";
    static const char revision[] = "0123456789abcdef0123456789abcdef01234567";
    static const char subtree[] = "fixture";
    static const char *const names[] = {
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    };
    char digests[2][65];
    char paths[2][768];
    char identity[65];
    char manifest[4096];
    char manifest_path[768];
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_source_acquisition_options options;
    unsigned int index;
    int count;

    for (index = 0u; index < 2u; ++index) {
        count = snprintf(paths[index], sizeof(paths[index]), "%s/%s",
                         fixture->root, names[index]);
        if (count < 0 || (size_t)count >= sizeof(paths[index]) ||
            yvex_artifact_compute_sha256(paths[index], digests[index], err) != YVEX_OK)
            return 0;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, YVEX_SOURCE_ACQUISITION_SCHEMA) ||
        !yvex_sha256_update_text(&hash, repository) ||
        !yvex_sha256_update_text(&hash, revision) ||
        !yvex_sha256_update_text(&hash, subtree) ||
        !yvex_sha256_update_u64(&hash, 2u)) return 0;
    for (index = 0u; index < 2u; ++index) {
        if (!yvex_sha256_update_text(&hash, names[index]) ||
            !yvex_sha256_update_u64(&hash, 16400u) ||
            !yvex_sha256_update_text(&hash, digests[index]) ||
            !yvex_sha256_update_text(&hash, digests[index]) ||
            !yvex_sha256_update_text(&hash, "") ||
            !yvex_sha256_update_text(&hash, digests[index]) ||
            !yvex_sha256_update_text(&hash, "") ||
            !yvex_sha256_update_text(&hash, "shard") ||
            !yvex_sha256_update_text(&hash, "fixture")) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    count = snprintf(
        manifest, sizeof(manifest),
        "{\"schema\":\"%s\",\"repository\":\"%s\",\"revision\":\"%s\","
        "\"admitted_subtree\":\"%s\",\"acquisition_complete\":true,"
        "\"acquisition_identity\":\"%s\",\"shards\":2,\"metadata_bytes\":0,"
        "\"shard_bytes\":32800,\"source_bytes\":32800,\"files\":["
        "{\"path\":\"%s\",\"component\":\"fixture\",\"classification\":\"shard\","
        "\"expected_size\":16400,\"actual_size\":16400,\"expected_sha256\":\"%s\","
        "\"actual_sha256\":\"%s\",\"git_oid\":\"\",\"lfs_oid\":\"%s\","
        "\"xet_hash\":\"\",\"verified\":true},"
        "{\"path\":\"%s\",\"component\":\"fixture\",\"classification\":\"shard\","
        "\"expected_size\":16400,\"actual_size\":16400,\"expected_sha256\":\"%s\","
        "\"actual_sha256\":\"%s\",\"git_oid\":\"\",\"lfs_oid\":\"%s\","
        "\"xet_hash\":\"\",\"verified\":true}]}",
        YVEX_SOURCE_ACQUISITION_SCHEMA, repository, revision, subtree, identity,
        names[0], digests[0], digests[0], digests[0], names[1], digests[1],
        digests[1], digests[1]);
    if (count < 0 || (size_t)count >= sizeof(manifest)) return 0;
    count = snprintf(manifest_path, sizeof(manifest_path), "%s/%s", fixture->root,
                     YVEX_SOURCE_ACQUISITION_MANIFEST);
    if (count < 0 || (size_t)count >= sizeof(manifest_path) ||
        !payload_test_write_text(manifest_path, manifest)) return 0;
    yvex_source_acquisition_options_default(&options);
    options.source_root = fixture->root;
    options.expected_repository = repository;
    options.expected_revision = revision;
    options.expected_subtree = subtree;
    options.maximum_files = 2u;
    options.maximum_source_bytes = 32800u;
    return yvex_source_acquisition_open(out, &options, failure, err) == YVEX_OK;
}

static int payload_test_metadata(const char *root,
                                 const char *name,
                                 const char *digest)
{
    char path[768];
    char text[512];
    int count;

    count = snprintf(path, sizeof(path),
                     "%s/.cache/huggingface/download/%s.metadata",
                     root, name);
    if (count < 0 || (size_t)count >= sizeof(path)) return 0;
    count = snprintf(text, sizeof(text),
        "62af8fffb2f7030cac4de2f0169f5b8d1101b646\n%s\n0\n", digest);
    return count >= 0 && (size_t)count < sizeof(text) &&
           payload_test_write_text(path, text);
}

static int payload_fixture_create_with_alpha_shape(
    payload_fixture *fixture,
    const char *suffix,
    int second_digest,
    int bad_digest,
    unsigned int handles,
    const yvex_source_payload_ops *ops,
    unsigned int alpha_rank,
    const unsigned long long *alpha_dims,
    yvex_source_payload_failure *failure,
    yvex_error *err)
{
    static const char *const shard_names[] = {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"
    };
    yvex_native_weight_table *table = NULL;
    yvex_source_shard_snapshot shards[2];
    yvex_source_tensor_snapshot_facts snapshot_facts;
    yvex_source_payload_open_options open_options;
    yvex_source_payload_budget budget;
    unsigned long long dims[1];
    char path[768];
    char digest[65];
    unsigned int index;
    int rc;

    memset(fixture, 0, sizeof(*fixture));
    snprintf(fixture->root, sizeof(fixture->root),
             "build/tests/source-payload-%s", suffix);
    snprintf(fixture->manifest, sizeof(fixture->manifest),
             "%s/manifest.json", fixture->root);
    snprintf(path, sizeof(path), "rm -rf %s", fixture->root);
    YVEX_TEST_ASSERT(system(path) == 0, "clear source payload fixture");
    YVEX_TEST_ASSERT(payload_test_mkdir("build"), "payload fixture build dir");
    YVEX_TEST_ASSERT(payload_test_mkdir("build/tests"), "payload fixture tests dir");
    YVEX_TEST_ASSERT(payload_test_mkdir(fixture->root), "payload fixture root");
    snprintf(path, sizeof(path), "%s/.cache", fixture->root);
    YVEX_TEST_ASSERT(payload_test_mkdir(path), "payload fixture cache");
    snprintf(path, sizeof(path), "%s/.cache/huggingface", fixture->root);
    YVEX_TEST_ASSERT(payload_test_mkdir(path), "payload fixture provider cache");
    snprintf(path, sizeof(path), "%s/.cache/huggingface/download", fixture->root);
    YVEX_TEST_ASSERT(payload_test_mkdir(path), "payload fixture metadata cache");

    table = (yvex_native_weight_table *)calloc(1u, sizeof(*table));
    YVEX_TEST_ASSERT(table != NULL, "payload fixture tensor table");
    YVEX_TEST_ASSERT(alpha_rank && alpha_dims,
                     "payload alpha shape is explicit");
    YVEX_TEST_ASSERT(yvex_native_weight_table_add(
        table, "alpha.weight", shard_names[0], "BF16", alpha_rank,
        alpha_dims,
        0u, 8192u, err) == YVEX_OK, "payload alpha tensor");
    dims[0] = 6000u;
    YVEX_TEST_ASSERT(yvex_native_weight_table_add(
        table, "beta.weight", shard_names[0], "I8", 1u, dims,
        8192u, 14192u, err) == YVEX_OK, "payload beta tensor");
    dims[0] = 2048u;
    YVEX_TEST_ASSERT(yvex_native_weight_table_add(
        table, "gamma.weight", shard_names[1], "F32", 1u, dims,
        4096u, 12288u, err) == YVEX_OK, "payload gamma tensor");
    memset(shards, 0, sizeof(shards));
    for (index = 0u; index < 2u; ++index) {
        shards[index].canonical_id = index;
        shards[index].canonical_name = shard_names[index];
        shards[index].file_bytes = 16400u;
        shards[index].data_region_offset = 16u;
        shards[index].payload_bytes = 16384u;
        snprintf(path, sizeof(path), "%s/%s", fixture->root,
                 shard_names[index]);
        YVEX_TEST_ASSERT(payload_test_write_bytes(path, 16u, 16384u,
                                                  index + 1u),
                         "payload fixture shard bytes");
        YVEX_TEST_ASSERT(yvex_artifact_compute_sha256(path, digest, err) ==
                             YVEX_OK,
                         "payload fixture reference digest");
        if (bad_digest && index == 0u)
            memset(digest, '0', 64u), digest[64] = '\0';
        if (index == 0u || second_digest)
            YVEX_TEST_ASSERT(payload_test_metadata(
                                 fixture->root, shard_names[index], digest),
                             "payload fixture LFS metadata");
    }
    YVEX_TEST_ASSERT(yvex_source_tensor_snapshot_take_table_with_shards(
                         &fixture->snapshot, &table, shards, 2u, 1u, err) ==
                         YVEX_OK,
                     "payload retained snapshot");
    YVEX_TEST_ASSERT(yvex_source_tensor_snapshot_facts_get(
                         fixture->snapshot, &snapshot_facts, err) == YVEX_OK,
                     "payload retained snapshot facts");
    memset(&fixture->verification, 0, sizeof(fixture->verification));
    fixture->verification.verified = 1;
    fixture->verification.manifest_verified = 1;
    fixture->verification.header_scan_count = 1u;
    fixture->verification.header_tensor_count = 3u;
    fixture->verification.shard_count = 2u;
    fixture->verification.shard_bytes = 32800u;
    fixture->verification.source_file_count = 2u;
    fixture->verification.source_total_bytes = 32800u;
    fixture->verification.declared_tensor_bytes = 22384u;
    fixture->verification.source_snapshot_identity = snapshot_facts.identity;
    fixture->verification.path_verified = 1;
    fixture->verification.repository_verified = 1;
    fixture->verification.revision_verified = 1;
    fixture->verification.shard_index_headers_match = 1;
    snprintf(fixture->verification.revision,
             sizeof(fixture->verification.revision), "%s",
             "62af8fffb2f7030cac4de2f0169f5b8d1101b646");
    snprintf(fixture->verification.repository_id,
             sizeof(fixture->verification.repository_id), "%s",
             "deepseek-ai/DeepSeek-V4-Flash-DSpark");
    snprintf(fixture->verification.inventory_authority,
             sizeof(fixture->verification.inventory_authority), "%s",
             "upstream-index");
    snprintf(fixture->verification.upstream_index_oid,
             sizeof(fixture->verification.upstream_index_oid), "%s",
             "84692cbe7af556a01e2e5353341100079c387aee");
    YVEX_TEST_ASSERT(realpath(fixture->root,
                             fixture->verification.resolved_source_path) != NULL,
                     "payload fixture canonical root");
    snprintf(fixture->verification.manifest_path,
             sizeof(fixture->verification.manifest_path), "%s",
             fixture->manifest);
    memset(&fixture->verify_options, 0, sizeof(fixture->verify_options));
    fixture->verify_options.identity = yvex_source_release_identity();
    fixture->verify_options.source_path = fixture->root;
    fixture->verify_options.models_root = "build/tests";
    fixture->verify_options.manifest_path = fixture->manifest;
    yvex_source_payload_budget_default(&budget);
    budget.maximum_open_handles = handles;
    budget.chunk_bytes = 8193u;
    budget.maximum_inflight_host_bytes = 8193u * 4u;
    budget.allow_local_snapshot_seal = 1;
    memset(&open_options, 0, sizeof(open_options));
    open_options.verification_options = &fixture->verify_options;
    open_options.verification = &fixture->verification;
    open_options.snapshot = fixture->snapshot;
    open_options.budget = budget;
    open_options.manifest_path = fixture->manifest;
    rc = yvex_source_payload_session_open_with_ops(
        &fixture->session, &open_options, ops, failure, err);
    if (rc != YVEX_OK) return rc;
    return YVEX_OK;
}

static int payload_fixture_create(payload_fixture *fixture,
                                  const char *suffix,
                                  int second_digest,
                                  int bad_digest,
                                  unsigned int handles,
                                  const yvex_source_payload_ops *ops,
                                  yvex_source_payload_failure *failure,
                                  yvex_error *err)
{
    static const unsigned long long alpha_dims[] = {4096u};

    return payload_fixture_create_with_alpha_shape(
        fixture, suffix, second_digest, bad_digest, handles, ops, 1u,
        alpha_dims, failure, err);
}

static void payload_fixture_close(payload_fixture *fixture)
{
    char command[768];

    if (!fixture) return;
    (void)yvex_source_payload_session_release(&fixture->session, NULL, NULL);
    yvex_source_tensor_snapshot_release(fixture->snapshot);
    fixture->snapshot = NULL;
    snprintf(command, sizeof(command), "rm -rf %s", fixture->root);
    if (system(command) != 0)
        fprintf(stderr, "source payload fixture cleanup failed: %s\n",
                fixture->root);
}

static int payload_fixture_transform_ir(
    yvex_transform_ir **out,
    yvex_source_payload_session *session,
    unsigned long long source_snapshot_identity,
    const char *required_payload_identity,
    const char *shard_name_override,
    int component_schema,
    int quantizable,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    const yvex_source_payload_range *range =
        yvex_source_payload_range_find(session, "alpha.weight");
    const yvex_source_payload_shard *shard = range
        ? yvex_source_payload_shard_at(session, range->shard_index) : NULL;
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_source_spec source;
    yvex_transform_value_spec terminal;
    yvex_transform_node_spec node;
    unsigned long long input_value;
    unsigned long long output_value;
    unsigned long long node_id;
    unsigned int dimension;
    int rc;

    if (out) *out = NULL;
    if (!out || !range || !shard || !required_payload_identity) return 1;
    memset(&header, 0, sizeof(header));
    header.schema_version = component_schema ? YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION
                                             : YVEX_TRANSFORM_IR_SCHEMA_VERSION;
    header.logical_model_identity = PAYLOAD_COMPONENT_IDENTITY;
    header.source_snapshot_identity = source_snapshot_identity;
    header.coverage_identity = 0x1122334455667788ull;
    header.required_payload_identity = required_payload_identity;
    header.payload_trust_class = "upstream_payload_verified";
    if (component_schema) {
        header.component_manifest_identity = PAYLOAD_COMPONENT_MANIFEST_IDENTITY;
        header.architecture_identity = PAYLOAD_ARCHITECTURE_IDENTITY;
        header.role_map_identity = PAYLOAD_ROLE_MAP_IDENTITY;
        header.unresolved_requirements_identity = PAYLOAD_UNRESOLVED_IDENTITY;
    }
    header.expected_source_count = 1u;
    header.expected_terminal_count = 1u;
    rc = yvex_transform_builder_create(
        &builder, &header, NULL, failure, err);
    memset(&source, 0, sizeof(source));
    source.source_name = range->source_tensor_name;
    source.shard_name = shard_name_override
        ? shard_name_override : shard->canonical_name;
    source.source_tensor_index = range->source_tensor_index;
    source.requirement_index = 0u;
    source.source_snapshot_identity = source_snapshot_identity;
    source.source_dtype = range->dtype;
    source.value_dtype = YVEX_TRANSFORM_DTYPE_BF16;
    source.shape.rank = range->rank;
    for (dimension = 0u; dimension < range->rank; ++dimension)
        source.shape.dims[dimension] = range->dims[dimension];
    source.relative_begin = range->relative_begin;
    source.relative_end = range->relative_end;
    source.requirement_identity = 0x99u;
    source.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    source.subsystem = YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    source.role_hint = YVEX_TENSOR_ROLE_ATTENTION_Q_A;
    if (component_schema) {
        source.component_identity = 0x778899u;
        source.semantic_role = 1u;
        source.phase_identity = 2u;
        source.lifetime_identity = 3u;
    }
    source.layer_index = 0u;
    source.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    source.expert_index = YVEX_TRANSFORM_IR_NO_ID;
    source.required_uses = 1u;
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_add_source(
            builder, &source, &input_value, failure, err);
    memset(&terminal, 0, sizeof(terminal));
    terminal.kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    terminal.semantic_id = 0x101u;
    terminal.canonical_ordinal = 0u;
    terminal.shape = source.shape;
    terminal.dtype = YVEX_TRANSFORM_DTYPE_BF16;
    terminal.precision.flags = YVEX_TRANSFORM_PRECISION_EXACT;
    if (quantizable)
        terminal.precision.flags |= YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                                    YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
    terminal.precision.allowed_physical_classes =
        YVEX_TRANSFORM_PHYSICAL_BF16 |
        (quantizable ? YVEX_TRANSFORM_PHYSICAL_QUANTIZED : 0u);
    terminal.precision.approximation_allowed = quantizable;
    terminal.precision.reference_compute_required = quantizable;
    terminal.logical_key.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    terminal.logical_key.subsystem = YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    terminal.logical_key.role = YVEX_TENSOR_ROLE_ATTENTION_Q_A;
    if (component_schema) {
        terminal.logical_key.component_identity = source.component_identity;
        terminal.logical_key.semantic_role = source.semantic_role;
        terminal.logical_key.phase_identity = source.phase_identity;
        terminal.logical_key.lifetime_identity = source.lifetime_identity;
    }
    terminal.logical_key.layer_index = 0u;
    terminal.logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_declare_value(
            builder, &terminal, &output_value, failure, err);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.output_value_id = output_value;
    node.input_value_ids = &input_value;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_add_node(
            builder, &node, &node_id, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_seal(builder, out, failure, err);
    yvex_transform_builder_release(&builder);
    return rc;
}

static int test_payload_transform_binding(
    payload_fixture *fixture,
    const yvex_source_payload_session_facts *facts)
{
    static const char wrong_payload[] =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    static const char wrong_execution[] =
        "0000000000000000000000000000000000000000000000000000000000000000";
    yvex_transform_ir *ir = NULL;
    yvex_transform_ir *mismatch = NULL;
    yvex_transform_binding *binding = NULL;
    yvex_transform_binding_summary const *summary;
    yvex_source_payload_plan *binding_plan = NULL;
    const yvex_source_payload_plan_summary *binding_plan_summary;
    yvex_source_payload_failure payload_failure;
    yvex_transform_physical_decision decision;
    yvex_transform_failure failure;
    yvex_transform_allocator allocator;
    payload_binding_allocator allocation_state;
    yvex_quant_plan *quant_plan = NULL;
    yvex_quant_plan *second_plan = NULL;
    yvex_gguf_writer_plan *writer_plan = NULL;
    yvex_gguf_writer_plan *second_writer_plan = NULL;
    yvex_gguf_writer_plan_request writer_request;
    yvex_gguf_writer_failure writer_failure;
    yvex_quant_explicit_decision quant_spec;
    yvex_quant_plan_options quant_plan_options;
    yvex_quant_sink_allocator sink_allocator;
    const yvex_quant_plan_summary *quant_summary;
    const yvex_quant_plan_summary *second_summary;
    const yvex_gguf_writer_plan_summary *writer_summary;
    const yvex_gguf_writer_plan_summary *second_writer_summary;
    const unsigned char *writer_prefix;
    const unsigned char *second_writer_prefix;
    size_t writer_prefix_bytes;
    size_t second_writer_prefix_bytes;
    char component_path[640];
    yvex_artifact *component_artifact = NULL;
    yvex_artifact_options artifact_options;
    yvex_gguf *component_gguf = NULL;
    yvex_gguf_parse_result parse;
    yvex_quant_digest_sink *digest_sink = NULL;
    yvex_quant_output_sink output_sink;
    yvex_quant_digest_summary digest_first;
    yvex_quant_digest_summary digest_second;
    yvex_quant_execution_summary execution;
    yvex_quant_executor_options executor_options;
    yvex_quant_cancellation cancellation;
    yvex_quant_failure quant_failure;
    payload_quant_sink_state refusal_state;
    const yvex_transform_value *terminal;
    unsigned char protocol_byte = 0u;
    yvex_error err;
    unsigned int fail_at;
    int quant_rc;

    YVEX_TEST_ASSERT(payload_fixture_transform_ir(
                         &ir, fixture->session,
                         facts->source_snapshot_identity,
                         facts->payload_identity, NULL, 0, 0, &failure, &err) ==
                         YVEX_OK &&
                         ir,
                     "trusted fixture Transformation IR seals");
    YVEX_TEST_ASSERT(yvex_transform_binding_create(
                         &binding, ir, fixture->session, NULL,
                         &failure, &err) == YVEX_OK && binding,
                     "sealed IR binds every trusted source range");
    summary = yvex_transform_binding_summary_get(binding);
    YVEX_TEST_ASSERT(summary && summary->complete &&
                         summary->source_count == 1u &&
                         summary->resolved_range_count == 1u &&
                         summary->range_lookup_count == 1u &&
                         summary->payload_bytes_read == 0u &&
                         summary->payload_readable_at_bind &&
                         yvex_transform_binding_terminal_at(binding, 0u) &&
                         yvex_transform_binding_terminal_operation(binding, 0u) &&
                         yvex_transform_binding_range_at(binding, 0u),
                     "quantizer boundary exposes immutable terminal dependencies");
    YVEX_TEST_ASSERT(yvex_transform_binding_readable_validate(
                         binding, &failure, &err) == YVEX_OK,
                     "ready session passes execution-time identity revalidation");
    YVEX_TEST_ASSERT(yvex_transform_binding_payload_plan_build(
                         &binding_plan, binding, 4096u, 4096u, &payload_failure,
                         &err) == YVEX_OK && binding_plan,
                     "transform binding projects its exact payload range plan");
    binding_plan_summary = yvex_source_payload_plan_summary_get(binding_plan);
    YVEX_TEST_ASSERT(binding_plan_summary && binding_plan_summary->range_count == 1u &&
                         binding_plan_summary->logical_bytes == 8192u,
                     "binding payload plan preserves the admitted source selection");
    yvex_source_payload_plan_close(binding_plan);
    memset(&decision, 0, sizeof(decision));
    decision.physical_class = YVEX_TRANSFORM_PHYSICAL_BF16;
    decision.encoding_id = 7u;
    YVEX_TEST_ASSERT(yvex_transform_binding_decision_validate(
                         binding, 0u, &decision, &failure, &err) == YVEX_OK,
                     "allowed exact physical decision attaches externally");
    decision.physical_class = YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    YVEX_TEST_ASSERT(yvex_transform_binding_decision_validate(
                         binding, 0u, &decision, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_TRANSFORM_FAILURE_INVALID_DTYPE,
                     "disallowed physical class refuses without mutating IR");

    terminal = yvex_transform_binding_terminal_at(binding, 0u);
    memset(&quant_spec, 0, sizeof(quant_spec));
    quant_spec.qtype = YVEX_GGUF_QTYPE_BF16;
    quant_spec.rank = terminal->shape.rank;
    memcpy(quant_spec.dims, terminal->shape.dims,
           sizeof(quant_spec.dims));
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &quant_plan, ir, binding, "fixture-bf16-exact-v1",
                         0xfaceb00cull, &quant_spec, 1u, NULL,
                         &quant_failure, &err) == YVEX_OK && quant_plan,
                     "fixture physical plan seals from the immutable IR");
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-bf16-exact-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) ==
                         YVEX_OK && second_plan,
                     "repeated fixture physical plan seals");
    quant_summary = yvex_quant_plan_summary_get(quant_plan);
    second_summary = yvex_quant_plan_summary_get(second_plan);
    YVEX_TEST_ASSERT(quant_summary && second_summary &&
                         quant_summary->complete &&
                         quant_summary->decision_count == 1u &&
                         quant_summary->source_value_count == 1u &&
                         quant_summary->encoded_bytes == 8192u &&
                         quant_summary->payload_bytes_read == 0u &&
                         strcmp(quant_summary->profile_identity,
                                second_summary->profile_identity) == 0 &&
                         yvex_quant_plan_find(
                             quant_plan, &terminal->logical_key) ==
                             yvex_quant_plan_decision_at(quant_plan, 0u),
                     "sealed plan identity, accounting, and index are deterministic");
    yvex_quant_plan_release(&second_plan);

    memset(&sink_allocator, 0, sizeof(sink_allocator));
    sink_allocator.allocate = payload_binding_allocate;
    sink_allocator.release = payload_binding_release;
    for (fail_at = 0u; fail_at < 3u; ++fail_at) {
        memset(&allocation_state, 0, sizeof(allocation_state));
        allocation_state.fail_at = fail_at;
        sink_allocator.context = &allocation_state;
        quant_rc = yvex_quant_digest_sink_create_with_allocator(
            &digest_sink, quant_plan, facts->payload_identity,
            &sink_allocator, &quant_failure, &err);
        if (fail_at < 2u) {
            YVEX_TEST_ASSERT(quant_rc != YVEX_OK && !digest_sink &&
                                 quant_failure.code ==
                                     YVEX_QUANT_FAILURE_ALLOCATION &&
                                 allocation_state.live == 0u,
                             "digest sink allocation failure unwinds all ownership");
        } else {
            YVEX_TEST_ASSERT(quant_rc == YVEX_OK && digest_sink &&
                                 allocation_state.live == 2u,
                             "digest sink allocator seam reaches complete construction");
            yvex_quant_digest_sink_release(&digest_sink);
            YVEX_TEST_ASSERT(allocation_state.live == 0u,
                             "digest sink allocator seam releases all ownership");
        }
    }

    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &digest_sink, quant_plan, facts->payload_identity,
                         &quant_failure, &err) == YVEX_OK,
                     "digest/discard sink binds the trusted payload identity");
    yvex_quant_digest_sink_adapter(digest_sink, &output_sink);
    yvex_quant_executor_options_default(&executor_options);
    executor_options.source_chunk_bytes = 4096u;
    executor_options.output_chunk_bytes = 4096u;
    executor_options.maximum_owned_bytes = 1024u * 1024u;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         quant_plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) == YVEX_OK &&
                         execution.complete &&
                         execution.terminals_executed == 1u &&
                         execution.source_values_consumed == 1u &&
                         execution.payload_bytes_read == 8192u &&
                         execution.encoded_output_bytes == 8192u &&
                         execution.peak_owned_bytes <=
                             executor_options.maximum_owned_bytes,
                     "trusted exact execution streams and commits bounded bytes");
    YVEX_TEST_ASSERT(yvex_quant_digest_sink_finalize(
                         digest_sink, &digest_first, &quant_failure, &err) ==
                         YVEX_OK && digest_first.complete &&
                         digest_first.committed_terminals == 1u &&
                         digest_first.encoded_bytes == 8192u,
                     "digest sink seals exact terminal output identity");
    YVEX_TEST_ASSERT(yvex_quant_digest_summary_validate(
                         &digest_first, digest_first.execution_identity,
                         &quant_failure, &err) == YVEX_OK,
                     "matching execution digest validates");
    YVEX_TEST_ASSERT(yvex_quant_digest_summary_validate(
                         &digest_first, wrong_execution,
                         &quant_failure, &err) != YVEX_OK &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_DIGEST_MISMATCH,
                     "mismatched execution digest has a typed refusal");
    yvex_quant_digest_sink_release(&digest_sink);

    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &digest_sink, quant_plan, wrong_payload,
                         &quant_failure, &err) != YVEX_OK && !digest_sink &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_PAYLOAD_IDENTITY,
                     "digest sink refuses a mismatched payload identity");

    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &digest_sink, quant_plan, facts->payload_identity,
                         &quant_failure, &err) == YVEX_OK,
                     "second digest sink opens");
    yvex_quant_digest_sink_adapter(digest_sink, &output_sink);
    executor_options.source_chunk_bytes = 8193u;
    executor_options.output_chunk_bytes = 4099u;
    quant_rc = yvex_quant_execute(
        quant_plan, &output_sink, &executor_options,
        &execution, &quant_failure, &err);
    if (quant_rc != YVEX_OK)
        fprintf(stderr,
                "odd-chunk execution failure=%u expected=%llu actual=%llu status=%s\n",
                (unsigned int)quant_failure.code,
                quant_failure.expected, quant_failure.actual,
                yvex_status_name(yvex_error_code(&err)));
    YVEX_TEST_ASSERT(quant_rc == YVEX_OK &&
                         yvex_quant_digest_sink_finalize(
                             digest_sink, &digest_second,
                             &quant_failure, &err) == YVEX_OK &&
                         strcmp(digest_first.execution_identity,
                                digest_second.execution_identity) == 0,
                     "output chunk size does not change encoded identity");
    yvex_quant_digest_sink_release(&digest_sink);

    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &digest_sink, quant_plan, facts->payload_identity,
                         &quant_failure, &err) == YVEX_OK,
                     "protocol-refusal digest sink opens");
    yvex_quant_digest_sink_adapter(digest_sink, &output_sink);
    YVEX_TEST_ASSERT(output_sink.begin_terminal(
                         output_sink.context,
                         yvex_quant_plan_decision_at(quant_plan, 0u)) == 0 &&
                         output_sink.deliver_chunk(
                             output_sink.context,
                             yvex_quant_plan_decision_at(quant_plan, 0u),
                             1u, &protocol_byte, 1u) != 0,
                     "digest sink refuses non-monotonic output offsets");
    output_sink.abort_terminal(
        output_sink.context, yvex_quant_plan_decision_at(quant_plan, 0u),
        &quant_failure, 0u);
    YVEX_TEST_ASSERT(output_sink.commit_terminal(
                         output_sink.context,
                         yvex_quant_plan_decision_at(quant_plan, 0u), 0u) != 0,
                     "digest sink forbids commit after abort");
    yvex_quant_digest_sink_release(&digest_sink);

    memset(&refusal_state, 0, sizeof(refusal_state));
    refusal_state.fail_chunk = 1u;
    memset(&output_sink, 0, sizeof(output_sink));
    output_sink.begin_terminal = payload_quant_sink_begin;
    output_sink.deliver_chunk = payload_quant_sink_chunk;
    output_sink.commit_terminal = payload_quant_sink_commit;
    output_sink.abort_terminal = payload_quant_sink_abort;
    output_sink.context = &refusal_state;
    yvex_quant_executor_options_default(&executor_options);
    executor_options.source_chunk_bytes = 4096u;
    executor_options.output_chunk_bytes = 4096u;
    executor_options.maximum_owned_bytes = 1024u * 1024u;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         quant_plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) != YVEX_OK &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_SINK_SHORT_WRITE &&
                         refusal_state.begins == 1u &&
                         refusal_state.chunks == 1u &&
                         refusal_state.commits == 0u &&
                         refusal_state.aborts == 1u &&
                         refusal_state.abort_delivered_bytes == 0u &&
                         !execution.complete,
                     "sink refusal aborts without publishing partial output");

    memset(&refusal_state, 0, sizeof(refusal_state));
    atomic_init(&cancellation.requested, 0);
    refusal_state.cancel_chunk = 1u;
    refusal_state.cancellation = &cancellation;
    memset(&output_sink, 0, sizeof(output_sink));
    output_sink.begin_terminal = payload_quant_sink_begin;
    output_sink.deliver_chunk = payload_quant_sink_chunk;
    output_sink.commit_terminal = payload_quant_sink_commit;
    output_sink.abort_terminal = payload_quant_sink_abort;
    output_sink.context = &refusal_state;
    executor_options.cancellation = &cancellation;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         quant_plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) != YVEX_OK &&
                         quant_failure.code == YVEX_QUANT_FAILURE_CANCELLED &&
                         refusal_state.chunks == 1u &&
                         refusal_state.commits == 0u &&
                         refusal_state.aborts == 1u &&
                         refusal_state.abort_delivered_bytes == 4096u &&
                         execution.committed_terminals == 0u &&
                         !execution.complete,
                     "mid-stream cancellation aborts between source chunks");

    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &digest_sink, quant_plan, facts->payload_identity,
                         &quant_failure, &err) == YVEX_OK,
                     "cancellation digest sink opens");
    yvex_quant_digest_sink_adapter(digest_sink, &output_sink);
    atomic_init(&cancellation.requested, 0);
    atomic_store_explicit(&cancellation.requested, 1, memory_order_release);
    yvex_quant_executor_options_default(&executor_options);
    executor_options.source_chunk_bytes = 4096u;
    executor_options.output_chunk_bytes = 4096u;
    executor_options.maximum_owned_bytes = 1024u * 1024u;
    executor_options.cancellation = &cancellation;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         quant_plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) != YVEX_OK &&
                         quant_failure.code == YVEX_QUANT_FAILURE_CANCELLED &&
                         execution.committed_terminals == 0u &&
                         !execution.complete,
                     "pre-execution cancellation publishes no terminal");
    yvex_quant_digest_sink_release(&digest_sink);

    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &digest_sink, quant_plan, facts->payload_identity,
                         &quant_failure, &err) == YVEX_OK,
                     "worker-failure digest sink opens");
    yvex_quant_digest_sink_adapter(digest_sink, &output_sink);
    yvex_quant_executor_options_default(&executor_options);
    executor_options.worker_count = 2u;
    executor_options.source_chunk_bytes = 4096u;
    executor_options.output_chunk_bytes = 4096u;
    executor_options.maximum_owned_bytes = 1024u * 1024u;
    executor_options.thread_create = payload_quant_thread_fail;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         quant_plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) != YVEX_OK &&
                         quant_failure.code == YVEX_QUANT_FAILURE_WORKER &&
                         execution.terminals_attempted == 0u &&
                         execution.worker_failures == 1u,
                     "worker startup failure occurs before any transaction");
    yvex_quant_digest_sink_release(&digest_sink);

    for (fail_at = 0u; fail_at < 2u; ++fail_at) {
        memset(&allocation_state, 0, sizeof(allocation_state));
        allocation_state.fail_at = fail_at;
        YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                             &digest_sink, quant_plan,
                             facts->payload_identity, &quant_failure,
                             &err) == YVEX_OK,
                         "allocation-fault digest sink opens");
        yvex_quant_digest_sink_adapter(digest_sink, &output_sink);
        yvex_quant_executor_options_default(&executor_options);
        executor_options.source_chunk_bytes = 4096u;
        executor_options.output_chunk_bytes = 4096u;
        executor_options.maximum_owned_bytes = 1024u * 1024u;
        executor_options.allocate = payload_binding_allocate;
        executor_options.release = payload_binding_release;
        executor_options.context = &allocation_state;
        YVEX_TEST_ASSERT(yvex_quant_execute(
                             quant_plan, &output_sink, &executor_options,
                             &execution, &quant_failure, &err) != YVEX_OK &&
                             quant_failure.code ==
                                 YVEX_QUANT_FAILURE_ALLOCATION &&
                             allocation_state.live == 0u,
                         "executor allocation failure unwinds every buffer");
        yvex_quant_digest_sink_release(&digest_sink);
    }

    YVEX_TEST_ASSERT(yvex_quant_digest_sink_create(
                         &digest_sink, quant_plan, facts->payload_identity,
                         &quant_failure, &err) == YVEX_OK,
                     "resource-budget digest sink opens");
    yvex_quant_digest_sink_adapter(digest_sink, &output_sink);
    yvex_quant_executor_options_default(&executor_options);
    executor_options.source_chunk_bytes = 4096u;
    executor_options.output_chunk_bytes = 4096u;
    executor_options.maximum_owned_bytes = 4096u;
    YVEX_TEST_ASSERT(yvex_quant_execute(
                         quant_plan, &output_sink, &executor_options,
                         &execution, &quant_failure, &err) != YVEX_OK &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_RESOURCE_BUDGET &&
                         !execution.complete,
                     "executor distinguishes exhausted budget from allocator refusal");
    yvex_quant_digest_sink_release(&digest_sink);

    memset(&quant_plan_options, 0, sizeof(quant_plan_options));
    quant_plan_options.allocate = payload_binding_allocate;
    quant_plan_options.release = payload_binding_release;
    quant_plan_options.maximum_owned_bytes = 1024u * 1024u;
    for (fail_at = 0u; fail_at < 3u; ++fail_at) {
        memset(&allocation_state, 0, sizeof(allocation_state));
        allocation_state.fail_at = fail_at;
        quant_plan_options.context = &allocation_state;
        YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                             &second_plan, ir, binding,
                             "fixture-bf16-exact-v1", 0xfaceb00cull,
                             &quant_spec, 1u, &quant_plan_options,
                             &quant_failure, &err) != YVEX_OK &&
                             !second_plan &&
                             quant_failure.code ==
                                 YVEX_QUANT_FAILURE_ALLOCATION &&
                             allocation_state.live == 0u,
                         "plan allocation failure unwinds every owner");
    }
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-bf16-exact-v1", 0xfaceb00cull,
                         &quant_spec, 0u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_MISSING_DECISION,
                     "missing explicit terminal decision refuses sealing");
    quant_spec.row_axis = 1u;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-bf16-exact-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_INVALID_ROW_AXIS,
                     "invalid physical row axis refuses sealing");
    quant_spec.row_axis = 0u;
    quant_spec.dims[0] /= 2u;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-bf16-exact-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_INVALID_DIMENSION,
                     "physical element-count mismatch refuses sealing");
    quant_spec.dims[0] *= 2u;
    quant_spec.qtype = YVEX_GGUF_QTYPE_Q4_2_REMOVED;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-refusal-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_REMOVED_QTYPE,
                     "removed qtype identity has a stable plan refusal");
    quant_spec.qtype = YVEX_GGUF_QTYPE_NVFP4_OUTSIDE_BASELINE;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-refusal-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_QTYPE_OUTSIDE_BASELINE,
                     "post-baseline qtype identity has a stable plan refusal");
    quant_spec.qtype = YVEX_GGUF_QTYPE_Q4_0;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-refusal-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_ENCODER_UNAVAILABLE,
                     "admitted storage without codec refuses encoding");
    quant_spec.qtype = YVEX_GGUF_QTYPE_BF16;
    quant_spec.rank = 0u;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-refusal-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_INVALID_RANK,
                     "invalid physical rank retains its typed geometry refusal");
    quant_spec.rank = terminal->shape.rank;
    quant_spec.dims[0] = 0u;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &second_plan, ir, binding,
                         "fixture-refusal-v1", 0xfaceb00cull,
                         &quant_spec, 1u, NULL, &quant_failure, &err) !=
                         YVEX_OK && !second_plan &&
                         quant_failure.code ==
                             YVEX_QUANT_FAILURE_INVALID_DIMENSION,
                     "zero physical dimension retains its typed geometry refusal");
    memcpy(quant_spec.dims, terminal->shape.dims,
           sizeof(quant_spec.dims));
    yvex_quant_plan_release(&quant_plan);
    yvex_transform_binding_release(&binding);

    YVEX_TEST_ASSERT(payload_fixture_transform_ir(
                         &mismatch, fixture->session,
                         facts->source_snapshot_identity, facts->payload_identity,
                         NULL, 1, 0, &failure, &err) == YVEX_OK && mismatch &&
                         yvex_transform_binding_create(
                             &binding, mismatch, fixture->session, NULL,
                             &failure, &err) == YVEX_OK && binding,
                     "component-schema source binds without payload reads");
    YVEX_TEST_ASSERT(yvex_quant_plan_build_identity(
                         &quant_plan, mismatch, binding,
                         "minimax-h3-source-faithful-v1", 0x778899u, NULL,
                         &quant_failure, &err) == YVEX_OK && quant_plan &&
                         yvex_quant_plan_build_identity(
                             &second_plan, mismatch, binding,
                             "minimax-h3-source-faithful-v1", 0x778899u,
                             NULL, &quant_failure, &err) == YVEX_OK && second_plan,
                     "source-faithful component physical plan seals twice");
    quant_summary = yvex_quant_plan_summary_get(quant_plan);
    second_summary = yvex_quant_plan_summary_get(second_plan);
    YVEX_TEST_ASSERT(quant_summary && second_summary && quant_summary->complete &&
                         quant_summary->qtype_tensor_counts[YVEX_GGUF_QTYPE_BF16] == 1u &&
                         quant_summary->encoded_bytes == 8192u &&
                         quant_summary->payload_bytes_read == 0u &&
                         strcmp(quant_summary->physical_variant_identity,
                                second_summary->physical_variant_identity) == 0,
                     "exact component physical identity and BF16 accounting repeat");
    memset(&writer_request, 0, sizeof(writer_request));
    writer_request.input_class = YVEX_GGUF_WRITER_INPUT_LOGICAL_COMPONENT;
    writer_request.quant_plan = quant_plan;
    writer_request.input.component.architecture = "minimax-h3";
    writer_request.input.component.target_id = "minimax-h3-fl2va";
    writer_request.input.component.component_id = "transformer";
    writer_request.input.component.source_snapshot_identity =
        PAYLOAD_SOURCE_SNAPSHOT_IDENTITY;
    writer_request.input.component.source_snapshot_key = facts->source_snapshot_identity;
    writer_request.input.component.component_identity = PAYLOAD_COMPONENT_IDENTITY;
    writer_request.input.component.component_manifest_identity =
        PAYLOAD_COMPONENT_MANIFEST_IDENTITY;
    writer_request.input.component.architecture_identity = PAYLOAD_ARCHITECTURE_IDENTITY;
    writer_request.input.component.role_map_identity = PAYLOAD_ROLE_MAP_IDENTITY;
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &writer_plan, &writer_request, &writer_failure, &err) == YVEX_OK &&
                         writer_plan &&
                         yvex_gguf_writer_plan_build(
                             &second_writer_plan, &writer_request,
                             &writer_failure, &err) == YVEX_OK && second_writer_plan,
                     "logical component writer plan seals twice");
    writer_summary = yvex_gguf_writer_plan_summary_get(writer_plan);
    second_writer_summary = yvex_gguf_writer_plan_summary_get(second_writer_plan);
    writer_prefix = yvex_gguf_writer_plan_prefix(writer_plan, &writer_prefix_bytes);
    second_writer_prefix = yvex_gguf_writer_plan_prefix(
        second_writer_plan, &second_writer_prefix_bytes);
    YVEX_TEST_ASSERT(writer_summary && second_writer_summary && writer_summary->complete &&
                         writer_summary->tensor_count == 1u &&
                         writer_summary->tensor_payload_bytes == 8192u &&
                         writer_summary->payload_bytes_read == 0u && writer_prefix &&
                         second_writer_prefix && writer_prefix_bytes == second_writer_prefix_bytes &&
                         memcmp(writer_prefix, second_writer_prefix, writer_prefix_bytes) == 0 &&
                         strcmp(writer_summary->writer_plan_identity,
                                second_writer_summary->writer_plan_identity) == 0,
                     "component GGUF structure is byte-identical and payload-neutral");
    YVEX_TEST_ASSERT(snprintf(component_path, sizeof(component_path), "%s/component.gguf",
                              fixture->root) > 0 &&
                         payload_test_write_sparse_gguf(
                             component_path, writer_prefix, writer_prefix_bytes,
                             writer_summary->final_file_bytes),
                     "component fixture writes one bounded structural GGUF");
    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = component_path;
    artifact_options.readonly = 1;
    memset(&parse, 0, sizeof(parse));
    YVEX_TEST_ASSERT(yvex_artifact_open(&component_artifact, &artifact_options, &err) == YVEX_OK &&
                         yvex_gguf_open_ex(&component_gguf, component_artifact, NULL, &parse,
                                           &err) == YVEX_OK &&
                         yvex_gguf_metadata_find(
                             component_gguf, "yvex.logical.component_manifest.identity") &&
                         yvex_gguf_metadata_find(
                             component_gguf, "yvex.logical.role_map.identity") &&
                         yvex_gguf_metadata_find(
                             component_gguf, "yvex.logical.unresolved_requirements.identity") &&
                         yvex_gguf_metadata_find(
                             component_gguf, "yvex.physical.payload_plan.identity"),
                     "native reader accepts every component metadata key");
    yvex_gguf_close(component_gguf);
    yvex_artifact_close(component_artifact);
    YVEX_TEST_ASSERT(unlink(component_path) == 0,
                     "component structural fixture is removed");
    yvex_gguf_writer_plan_release(&second_writer_plan);
    writer_request.input.component.component_identity = PAYLOAD_ROLE_MAP_IDENTITY;
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &second_writer_plan, &writer_request,
                         &writer_failure, &err) != YVEX_OK && !second_writer_plan &&
                         writer_failure.code == YVEX_GGUF_WRITER_IDENTITY_MISMATCH,
                     "mismatched component identity refuses writer publication");
    yvex_gguf_writer_plan_release(&writer_plan);
    yvex_quant_plan_release(&second_plan);
    yvex_quant_plan_release(&quant_plan);
    yvex_transform_binding_release(&binding);
    yvex_transform_ir_release(&mismatch);

    YVEX_TEST_ASSERT(payload_fixture_transform_ir(
                         &mismatch, fixture->session,
                         facts->source_snapshot_identity, wrong_payload,
                         NULL, 0, 0, &failure, &err) == YVEX_OK && mismatch &&
                         yvex_transform_binding_create(
                             &binding, mismatch, fixture->session, NULL,
                             &failure, &err) != YVEX_OK && !binding &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
                     "required payload identity mismatch refuses before execution");
    yvex_transform_ir_release(&mismatch);

    YVEX_TEST_ASSERT(payload_fixture_transform_ir(
                         &mismatch, fixture->session,
                         facts->source_snapshot_identity,
                         facts->payload_identity, "wrong.safetensors",
                         0, 0, &failure, &err) == YVEX_OK && mismatch &&
                         yvex_transform_binding_create(
                             &binding, mismatch, fixture->session, NULL,
                             &failure, &err) != YVEX_OK && !binding &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
                     "source shard identity mismatch refuses binding");
    yvex_transform_ir_release(&mismatch);

    memset(&allocator, 0, sizeof(allocator));
    allocator.allocate = payload_binding_allocate;
    allocator.release = payload_binding_release;
    for (fail_at = 0u; fail_at < 3u; ++fail_at) {
        memset(&allocation_state, 0, sizeof(allocation_state));
        allocation_state.fail_at = fail_at;
        allocator.context = &allocation_state;
        if (yvex_transform_binding_create(
                &binding, ir, fixture->session, &allocator,
                &failure, &err) == YVEX_OK) {
            yvex_transform_binding_release(&binding);
        } else {
            YVEX_TEST_ASSERT(!binding &&
                                 failure.code ==
                                     YVEX_TRANSFORM_FAILURE_ALLOCATION,
                             "binding allocation seam returns typed refusal");
        }
        YVEX_TEST_ASSERT(allocation_state.live == 0u,
                         "binding allocation seam unwinds completely");
    }
    yvex_transform_binding_release(&binding);
    yvex_transform_binding_release(&binding);
    yvex_transform_ir_release(&ir);
    return 0;
}

/* Proves an untrusted source session cannot become an executable binding. */
static int test_payload_transform_binding_untrusted(payload_fixture *fixture)
{
    static const char pending_payload[] =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    yvex_transform_ir *ir = NULL;
    yvex_transform_binding *binding = NULL;
    yvex_transform_failure failure;
    yvex_error err;

    YVEX_TEST_ASSERT(payload_fixture_transform_ir(
                         &ir, fixture->session,
                         fixture->verification.source_snapshot_identity,
                         pending_payload, NULL, 0, 0, &failure, &err) == YVEX_OK &&
                         ir,
                     "untrusted-session plan remains valid planning state");
    YVEX_TEST_ASSERT(yvex_transform_binding_create(
                         &binding, ir, fixture->session, NULL,
                         &failure, &err) != YVEX_OK && !binding &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
                     "untrusted session without admitted identity refuses binding");
    yvex_transform_ir_release(&ir);
    return 0;
}

static int payload_fixture_q8_override(const yvex_transform_value *terminal,
                                       unsigned int *qtype, int *approximation,
                                       void *context)
{
    unsigned int *calls = context;

    if (!terminal || !qtype || !approximation || !calls) return 0;
    *qtype = YVEX_GGUF_QTYPE_Q8_0;
    *approximation = 1;
    (*calls)++;
    return 1;
}

static int test_payload_component_physical_shape_fold(void)
{
    static const unsigned long long logical_dims[] = {1u, 2u, 4u, 2u, 256u};
    payload_fixture fixture;
    yvex_source_payload_failure payload_failure;
    yvex_source_payload_plan *payload_plan = NULL;
    yvex_source_payload_sink trust_sink;
    payload_sink_state trust_sink_state;
    yvex_source_payload_stream_result payload_result;
    yvex_source_payload_session_facts facts;
    yvex_transform_ir *ir = NULL;
    yvex_transform_binding *binding = NULL;
    yvex_transform_failure transform_failure;
    yvex_quant_plan *plan = NULL;
    yvex_quant_plan *candidate = NULL;
    yvex_quant_plan *arbitrary = NULL;
    yvex_quant_failure quant_failure;
    const yvex_quant_plan_summary *summary;
    const yvex_quant_plan_summary *candidate_summary;
    const yvex_quant_decision *decision;
    const yvex_quant_decision *candidate_decision;
    yvex_quant_explicit_decision arbitrary_spec;
    yvex_quant_plan_options candidate_options;
    yvex_gguf_writer_plan_request request;
    yvex_gguf_writer_plan *writer = NULL;
    yvex_gguf_writer_failure writer_failure;
    yvex_error err;
    unsigned int override_calls = 0u;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create_with_alpha_shape(
                         &fixture, "rank-five", 1, 0, 1u, NULL, 5u,
                         logical_dims, &payload_failure, &err) == YVEX_OK,
                     "rank-five source payload fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &payload_plan, fixture.session, 4096u, 4096u,
                         &payload_failure, &err) == YVEX_OK,
                     "rank-five source payload trust plan seals");
    memset(&trust_sink_state, 0, sizeof(trust_sink_state));
    payload_sink_make(&trust_sink, &trust_sink_state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, payload_plan, &trust_sink,
                         &payload_result, &payload_failure, &err) == YVEX_OK &&
                         payload_result.complete && payload_result.committed,
                     "rank-five source payload trust publishes transactionally");
    yvex_source_payload_plan_close(payload_plan);
    YVEX_TEST_ASSERT(yvex_source_payload_session_facts_get(
                         fixture.session, &facts, &err) == YVEX_OK &&
                         facts.state == YVEX_SOURCE_PAYLOAD_STATE_READY,
                     "rank-five source payload identity is ready");
    YVEX_TEST_ASSERT(payload_fixture_transform_ir(
                         &ir, fixture.session, facts.source_snapshot_identity,
                         facts.payload_identity, NULL, 1, 1, &transform_failure,
                         &err) == YVEX_OK && ir &&
                         yvex_transform_binding_create(
                             &binding, ir, fixture.session, NULL,
                             &transform_failure, &err) == YVEX_OK && binding,
                     "rank-five logical component binds exact source bytes");
    YVEX_TEST_ASSERT(yvex_quant_plan_build_identity(
                         &plan, ir, binding, "source-faithful-rank-fold-v1",
                         0x887766u, NULL, &quant_failure, &err) == YVEX_OK &&
                         plan,
                     "rank-five source-faithful physical plan seals");
    summary = yvex_quant_plan_summary_get(plan);
    decision = yvex_quant_plan_decision_at(plan, 0u);
    YVEX_TEST_ASSERT(summary && decision && summary->complete &&
                         decision->rank == 4u &&
                         decision->dims[0] == 256u && decision->dims[1] == 2u &&
                         decision->dims[2] == 4u && decision->dims[3] == 2u &&
                         decision->element_count == 4096u &&
                         decision->encoded_bytes == 8192u,
                     "canonical GGUF projection preserves elements and exact bytes");

    memset(&candidate_options, 0, sizeof(candidate_options));
    candidate_options.identity_override = payload_fixture_q8_override;
    candidate_options.identity_override_context = &override_calls;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_identity(
                         &candidate, ir, binding, "fixture-q8-candidate-v1",
                         0x1122aabbu, &candidate_options, &quant_failure, &err) == YVEX_OK &&
                         candidate,
                     "bounded physical override seals a quantized identity candidate");
    candidate_summary = yvex_quant_plan_summary_get(candidate);
    candidate_decision = yvex_quant_plan_decision_at(candidate, 0u);
    YVEX_TEST_ASSERT(override_calls == 1u && candidate_summary && candidate_summary->complete &&
                         candidate_summary->qtype_tensor_counts[YVEX_GGUF_QTYPE_Q8_0] == 1u &&
                         candidate_decision && candidate_decision->qtype == YVEX_GGUF_QTYPE_Q8_0 &&
                         candidate_decision->approximation,
                     "candidate override changes only the admitted physical decision");

    memset(&request, 0, sizeof(request));
    request.input_class = YVEX_GGUF_WRITER_INPUT_LOGICAL_COMPONENT;
    request.quant_plan = plan;
    request.input.component.architecture = "minimax-h3";
    request.input.component.target_id = "minimax-h3-fl2va";
    request.input.component.component_id = "video_vae";
    request.input.component.source_snapshot_identity =
        PAYLOAD_SOURCE_SNAPSHOT_IDENTITY;
    request.input.component.source_snapshot_key = facts.source_snapshot_identity;
    request.input.component.component_identity = PAYLOAD_COMPONENT_IDENTITY;
    request.input.component.component_manifest_identity =
        PAYLOAD_COMPONENT_MANIFEST_IDENTITY;
    request.input.component.architecture_identity = PAYLOAD_ARCHITECTURE_IDENTITY;
    request.input.component.role_map_identity = PAYLOAD_ROLE_MAP_IDENTITY;
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &writer, &request, &writer_failure, &err) == YVEX_OK &&
                         writer,
                     "logical component writer admits only the canonical fold");
    yvex_gguf_writer_plan_release(&writer);

    memset(&arbitrary_spec, 0, sizeof(arbitrary_spec));
    arbitrary_spec.qtype = YVEX_GGUF_QTYPE_BF16;
    arbitrary_spec.rank = 4u;
    arbitrary_spec.dims[0] = 4u;
    arbitrary_spec.dims[1] = 2u;
    arbitrary_spec.dims[2] = 2u;
    arbitrary_spec.dims[3] = 256u;
    YVEX_TEST_ASSERT(yvex_quant_plan_build_explicit(
                         &arbitrary, ir, binding, "arbitrary-rank-fold-v1",
                         0x887766u, &arbitrary_spec, 1u, NULL, &quant_failure,
                         &err) == YVEX_OK && arbitrary,
                     "equal-element arbitrary physical geometry reaches writer guard");
    request.quant_plan = arbitrary;
    YVEX_TEST_ASSERT(yvex_gguf_writer_plan_build(
                         &writer, &request, &writer_failure, &err) != YVEX_OK &&
                         !writer &&
                         writer_failure.code ==
                             YVEX_GGUF_WRITER_TENSOR_DIVERGENCE,
                     "logical component writer refuses an arbitrary shape fold");

    yvex_quant_plan_release(&arbitrary);
    yvex_quant_plan_release(&candidate);
    yvex_quant_plan_release(&plan);
    yvex_transform_binding_release(&binding);
    yvex_transform_ir_release(&ir);
    payload_fixture_close(&fixture);
    return 0;
}

static int payload_fixture_reopen_policy(payload_fixture *fixture,
                                         unsigned int handles,
                                         int allow_local_seal,
                                         const yvex_source_payload_ops *ops,
                                         yvex_source_payload_failure *failure,
                                         yvex_error *err)
{
    yvex_source_payload_open_options open_options;
    yvex_source_payload_budget budget;

    memset(&open_options, 0, sizeof(open_options));
    yvex_source_payload_budget_default(&budget);
    budget.maximum_open_handles = handles;
    budget.chunk_bytes = 4096u;
    budget.maximum_inflight_host_bytes = 4096u * 4u;
    budget.allow_local_snapshot_seal = allow_local_seal;
    open_options.verification_options = &fixture->verify_options;
    open_options.verification = &fixture->verification;
    open_options.snapshot = fixture->snapshot;
    open_options.budget = budget;
    open_options.manifest_path = fixture->manifest;
    return yvex_source_payload_session_open_with_ops(
        &fixture->session, &open_options, ops, failure, err);
}

static int payload_fixture_reopen(payload_fixture *fixture,
                                  unsigned int handles,
                                  const yvex_source_payload_ops *ops,
                                  yvex_source_payload_failure *failure,
                                  yvex_error *err)
{
    return payload_fixture_reopen_policy(
        fixture, handles, 1, ops, failure, err);
}

static int payload_fixture_replace_snapshot_facts(
    payload_fixture *fixture,
    const char *tensor_shard_name,
    const char *catalog_shard_name,
    const char *dtype,
    unsigned long long dimension,
    unsigned long long data_start,
    unsigned long long data_end,
    unsigned long long data_region_offset,
    unsigned long long payload_bytes,
    yvex_error *err)
{
    yvex_native_weight_table *table;
    yvex_source_shard_snapshot shard;
    yvex_source_tensor_snapshot_facts facts;
    unsigned long long dims[1];
    int rc;

    table = (yvex_native_weight_table *)calloc(1u, sizeof(*table));
    if (!table) return 0;
    dims[0] = dimension;
    rc = yvex_native_weight_table_add(
        table, "probe.weight", tensor_shard_name, dtype, 1u, dims,
        data_start, data_end, err);
    if (rc != YVEX_OK) {
        yvex_native_weight_table_close(table);
        return 0;
    }
    memset(&shard, 0, sizeof(shard));
    shard.canonical_name = catalog_shard_name;
    shard.file_bytes = 16400u;
    shard.data_region_offset = data_region_offset;
    shard.payload_bytes = payload_bytes;
    yvex_source_tensor_snapshot_release(fixture->snapshot);
    fixture->snapshot = NULL;
    rc = yvex_source_tensor_snapshot_take_table_with_shards(
        &fixture->snapshot, &table, &shard, 1u, 1u, err);
    if (rc != YVEX_OK) {
        yvex_native_weight_table_close(table);
        return 0;
    }
    if (yvex_source_tensor_snapshot_facts_get(
            fixture->snapshot, &facts, err) != YVEX_OK) return 0;
    fixture->verification.source_snapshot_identity = facts.identity;
    fixture->verification.shard_count = 1u;
    fixture->verification.header_shard_count = 1u;
    fixture->verification.header_tensor_count = 1u;
    fixture->verification.declared_tensor_bytes = data_end - data_start;
    return 1;
}

static int payload_fixture_replace_snapshot(payload_fixture *fixture,
                                            const char *shard_name,
                                            unsigned long long data_end,
                                            yvex_error *err)
{
    return payload_fixture_replace_snapshot_facts(
        fixture, shard_name, shard_name, "I8", data_end, 0u, data_end,
        16u, 16384u, err);
}

static int payload_sink_begin(void *context,
                              const yvex_source_payload_plan_summary *summary)
{
    payload_sink_state *state = (payload_sink_state *)context;

    if (!state || !summary || summary->logical_bytes == 0u) return 1;
    yvex_sha256_init(&state->hash);
    return 0;
}

static int payload_sink_chunk(void *context,
                              const yvex_source_payload_chunk *chunk,
                              const unsigned char *bytes)
{
    payload_sink_state *state = (payload_sink_state *)context;

    if (!state || !chunk || !bytes) return 1;
    if (state->fail_after_chunks != 0u &&
        state->chunks == state->fail_after_chunks) return 1;
    if (!yvex_sha256_update(&state->hash, bytes, chunk->byte_length)) return 1;
    state->bytes += chunk->byte_length;
    state->chunks++;
    if (state->cancel_session && state->cancel_after_chunks != 0u &&
        state->chunks == state->cancel_after_chunks)
        (void)yvex_source_payload_session_cancel(
            state->cancel_session, NULL, NULL);
    return 0;
}

static int payload_sink_commit(void *context,
                               const yvex_source_payload_stream_result *result)
{
    payload_sink_state *state = (payload_sink_state *)context;
    unsigned char digest[32];

    if (!state || !result || !result->complete ||
        !yvex_sha256_final(&state->hash, digest)) return 1;
    state->commits++;
    return 0;
}

static void payload_sink_abort(void *context,
                               const yvex_source_payload_failure *failure,
                               const yvex_source_payload_stream_result *result)
{
    payload_sink_state *state = (payload_sink_state *)context;

    (void)failure;
    (void)result;
    if (state) state->aborts++;
}

static void payload_sink_make(yvex_source_payload_sink *sink,
                              payload_sink_state *state)
{
    memset(sink, 0, sizeof(*sink));
    sink->begin = payload_sink_begin;
    sink->chunk = payload_sink_chunk;
    sink->commit = payload_sink_commit;
    sink->abort = payload_sink_abort;
    sink->context = state;
}

/* Blocks one accepted chunk at a deterministic synchronization point. */
static int payload_blocking_chunk(
    void *context,
    const yvex_source_payload_chunk *chunk,
    const unsigned char *bytes)
{
    payload_blocking_sink *state = (payload_blocking_sink *)context;

    if (!state || payload_sink_chunk(&state->base, chunk, bytes) != 0) return 1;
    if (!state->waited) {
        state->waited = 1;
        pthread_mutex_lock(&state->gate->mutex);
        state->gate->arrived++;
        pthread_cond_broadcast(&state->gate->condition);
        while (!state->gate->released)
            pthread_cond_wait(&state->gate->condition, &state->gate->mutex);
        pthread_mutex_unlock(&state->gate->mutex);
    }
    return 0;
}

/* Adapts the ordinary sink transaction to a blocking chunk callback. */
static void payload_blocking_sink_make(yvex_source_payload_sink *sink,
                                       payload_blocking_sink *state,
                                       payload_stream_gate *gate)
{
    memset(state, 0, sizeof(*state));
    state->gate = gate;
    payload_sink_make(sink, &state->base);
    sink->chunk = payload_blocking_chunk;
    sink->context = state;
}

static void *payload_stream_thread_run(void *opaque)
{
    payload_stream_thread *thread = (payload_stream_thread *)opaque;

    yvex_error_clear(&thread->error);
    thread->rc = thread->verify
        ? yvex_source_payload_session_verify(
              thread->session, thread->plan, &thread->sink, &thread->result,
              &thread->failure, &thread->error)
        : yvex_source_payload_session_stream(
              thread->session, thread->plan, &thread->sink, &thread->result,
              &thread->failure, &thread->error);
    return NULL;
}

/* Waits until a required number of callbacks are concurrently inside the gate. */
static void payload_gate_wait(payload_stream_gate *gate, unsigned int count)
{
    pthread_mutex_lock(&gate->mutex);
    while (gate->arrived < count)
        pthread_cond_wait(&gate->condition, &gate->mutex);
    pthread_mutex_unlock(&gate->mutex);
}

static void payload_gate_release(payload_stream_gate *gate)
{
    pthread_mutex_lock(&gate->mutex);
    gate->released = 1;
    pthread_cond_broadcast(&gate->condition);
    pthread_mutex_unlock(&gate->mutex);
}

static int payload_has_blocker(const yvex_source_verification *verification,
                               const char *blocker)
{
    unsigned int index;

    for (index = 0u; verification && index < verification->blocker_count;
         ++index) {
        if (strcmp(verification->blockers[index], blocker) == 0) return 1;
    }
    return 0;
}

static int payload_fault_openat(int directory, const char *name, int flags)
{
    unsigned long long call = payload_fault.open_calls++;
    if (call == payload_fault.open_fail_at) {
        errno = EMFILE;
        return -1;
    }
    return openat(directory, name, flags);
}

static int payload_fault_fstat(int fd, struct stat *status)
{
    unsigned long long call = payload_fault.fstat_calls++;
    if (call == payload_fault.fstat_fail_at) {
        errno = EIO;
        return -1;
    }
    return fstat(fd, status);
}

static int payload_fault_fstatat(int directory, const char *name,
                                 struct stat *status, int flags)
{
    return fstatat(directory, name, status, flags);
}

static int payload_fault_close(int fd)
{
    unsigned long long call = payload_fault.close_calls++;
    int rc = close(fd);

    if (call == payload_fault.close_fail_at && rc == 0) {
        errno = EIO;
        return -1;
    }
    return rc;
}

static ssize_t payload_fault_pread(int fd, void *buffer,
                                   size_t length, off_t offset)
{
    unsigned long long call = payload_fault.pread_calls++;

    if (call == payload_fault.eintr_at) {
        errno = EINTR;
        return -1;
    }
    if (call == payload_fault.eof_at) return 0;
    if (call == payload_fault.io_at) {
        errno = EIO;
        return -1;
    }
    if (payload_fault.partial_limit != 0u &&
        length > payload_fault.partial_limit)
        length = payload_fault.partial_limit;
    return pread(fd, buffer, length, offset);
}

static void *payload_fault_malloc(size_t size)
{
    unsigned long long call = payload_fault.allocation_calls++;
    void *allocation;

    if (call == payload_fault.allocation_fail_at) return NULL;
    allocation = malloc(size);
    if (allocation) payload_fault.live_allocations++;
    return allocation;
}

static void *payload_fault_calloc(size_t count, size_t size)
{
    unsigned long long call = payload_fault.allocation_calls++;
    void *allocation;

    if (call == payload_fault.allocation_fail_at) return NULL;
    allocation = calloc(count, size);
    if (allocation) payload_fault.live_allocations++;
    return allocation;
}

static void payload_fault_free(void *allocation)
{
    if (!allocation) return;
    payload_fault.live_allocations--;
    free(allocation);
}

static void payload_fault_ops(yvex_source_payload_ops *ops)
{
    yvex_source_payload_default_ops(ops);
    ops->openat_fn = payload_fault_openat;
    ops->fstat_fn = payload_fault_fstat;
    ops->fstatat_fn = payload_fault_fstatat;
    ops->pread_fn = payload_fault_pread;
    ops->close_fn = payload_fault_close;
    ops->malloc_fn = payload_fault_malloc;
    ops->calloc_fn = payload_fault_calloc;
    ops->free_fn = payload_fault_free;
}

static void payload_fault_reset(void)
{
    memset(&payload_fault, 0, sizeof(payload_fault));
    payload_fault.eintr_at = ULLONG_MAX;
    payload_fault.eof_at = ULLONG_MAX;
    payload_fault.io_at = ULLONG_MAX;
    payload_fault.open_fail_at = ULLONG_MAX;
    payload_fault.fstat_fail_at = ULLONG_MAX;
    payload_fault.close_fail_at = ULLONG_MAX;
    payload_fault.allocation_fail_at = ULLONG_MAX;
}

static int test_sha256_primitive(void)
{
    static const char expected_empty[] =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    static const char expected_abc[] =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    static const char expected_long[] =
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
    static const char long_text[] =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    static const char expected_million_a[] =
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    char block[1000];
    unsigned int block_index;
    yvex_sha256 hash;
    unsigned char digest[32];
    char hex[65];

    yvex_sha256_init(&hash);
    YVEX_TEST_ASSERT(yvex_sha256_update(&hash, NULL, 0u),
                     "SHA-256 accepts zero-length update");
    YVEX_TEST_ASSERT(yvex_sha256_final(&hash, digest), "SHA-256 empty final");
    yvex_sha256_hex(digest, hex);
    YVEX_TEST_ASSERT_STREQ(hex, expected_empty, "SHA-256 empty known answer");
    yvex_sha256_init(&hash);
    YVEX_TEST_ASSERT(yvex_sha256_update(&hash, "a", 1u) &&
                     yvex_sha256_update(&hash, "bc", 2u) &&
                     yvex_sha256_final(&hash, digest),
                     "SHA-256 incremental update");
    yvex_sha256_hex(digest, hex);
    YVEX_TEST_ASSERT_STREQ(hex, expected_abc, "SHA-256 abc known answer");
    yvex_sha256_init(&hash);
    YVEX_TEST_ASSERT(yvex_sha256_update(&hash, long_text, strlen(long_text)) &&
                     yvex_sha256_final(&hash, digest),
                     "SHA-256 multi-block update");
    yvex_sha256_hex(digest, hex);
    YVEX_TEST_ASSERT_STREQ(hex, expected_long, "SHA-256 multi-block known answer");
    YVEX_TEST_ASSERT(!yvex_sha256_update(&hash, "x", 1u),
                     "SHA-256 refuses update after final");
    memset(block, 'a', sizeof(block));
    yvex_sha256_init(&hash);
    for (block_index = 0u; block_index < 1000u; ++block_index)
        YVEX_TEST_ASSERT(yvex_sha256_update(&hash, block, sizeof(block)),
                         "SHA-256 large incremental stream");
    YVEX_TEST_ASSERT(yvex_sha256_final(&hash, digest),
                     "SHA-256 million-a final");
    yvex_sha256_hex(digest, hex);
    YVEX_TEST_ASSERT_STREQ(hex, expected_million_a,
                           "SHA-256 million-a known answer");
    return 0;
}

static int test_payload_happy_path(void)
{
    payload_fixture fixture;
    yvex_source_payload_failure failure;
    yvex_source_payload_plan *plan = NULL;
    yvex_source_payload_plan *single = NULL;
    yvex_source_payload_sink sink;
    payload_sink_state sink_state;
    yvex_source_payload_stream_result result;
    yvex_source_payload_session_facts facts;
    yvex_source_verification parsed;
    yvex_source_payload_probe_result probe;
    yvex_error err;
    unsigned long long alpha;
    unsigned long long duplicate_selection[2];
    const yvex_source_payload_plan_summary *summary;
    char manifest_text[8192];
    char malformed_text[8192];
    char *schema_version;
    char *field;
    int rc;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "happy", 1, 0, 1u, NULL,
                         &failure, &err) == YVEX_OK,
                     "payload fixture session opens");
    YVEX_TEST_ASSERT(yvex_source_tensor_snapshot_find_index(
                         fixture.snapshot, "alpha.weight", &alpha),
                     "payload alpha index");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "payload exhaustive chunk plan");
    summary = yvex_source_payload_plan_summary_get(plan);
    YVEX_TEST_ASSERT(summary && summary->range_count == 3u &&
                         summary->chunk_count == 6u &&
                         summary->logical_bytes == 22384u,
                     "payload plan exact accounting");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build(
                         &single, fixture.session, &alpha, 1u, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "payload single tensor plan");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_summary_get(single)->chunk_count ==
                         2u &&
                     yvex_source_payload_plan_chunk_at(single, 0u)->first_page ==
                         0u &&
                     yvex_source_payload_plan_chunk_at(single, 1u)->last_page ==
                         2u,
                     "payload page and chunk boundaries");
    yvex_source_payload_plan_close(single);
    single = NULL;
    duplicate_selection[0] = alpha;
    duplicate_selection[1] = alpha;
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build(
                         &single, fixture.session, duplicate_selection, 2u,
                         4096u, 4096u, &failure, &err) == YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_TENSOR,
                     "payload plan rejects duplicate tensor ownership");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build(
                         &single, fixture.session, &alpha, 1u,
                         128u, 4096u, &failure, &err) == YVEX_ERR_BOUNDS &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_CHUNK,
                     "payload plan rejects undersized chunk configuration");
    {
        unsigned long long absent = ULLONG_MAX;

        YVEX_TEST_ASSERT(yvex_source_payload_plan_build(
                             &single, fixture.session, &absent, 1u,
                             4096u, 4096u, &failure, &err) ==
                             YVEX_ERR_BOUNDS &&
                             failure.code ==
                                 YVEX_SOURCE_PAYLOAD_FAILURE_TENSOR_NOT_INDEXED,
                         "payload plan rejects an absent tensor index");
    }
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build(
                         &single, fixture.session, &alpha, 1u,
                         4096u, 4096u, &failure, &err) == YVEX_OK,
                     "payload single tensor plan rebuilds after refusals");
    YVEX_TEST_ASSERT(yvex_source_payload_session_close(
                         fixture.session, &failure, &err) == YVEX_ERR_STATE &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_CLOSE_BUSY,
                     "payload close refuses registered plans before IO");
    memset(&sink_state, 0, sizeof(sink_state));
    payload_sink_make(&sink, &sink_state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_stream(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) == YVEX_ERR_STATE &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_NOT_TRUSTED &&
                         !result.committed && !result.aborted,
                     "ordinary stream refuses an untrusted payload session");
    if (test_payload_transform_binding_untrusted(&fixture) != 0) return 1;
    memset(&sink_state, 0, sizeof(sink_state));
    rc = yvex_source_payload_session_verify(
        fixture.session, plan, &sink, &result, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && result.complete && result.committed &&
                         !result.aborted,
                     "payload trust stream commits exactly");
    YVEX_TEST_ASSERT(result.trust_bytes_read == 32800u &&
                         result.physical_bytes_read == 32800u &&
                         result.delivered_logical_bytes == 22384u &&
                         result.chunks_completed == 6u &&
                         sink_state.commits == 1u && sink_state.aborts == 0u,
                     "payload trust and consumer accounting separate");
    YVEX_TEST_ASSERT(yvex_source_payload_session_facts_get(
                         fixture.session, &facts, &err) == YVEX_OK &&
                         facts.state == YVEX_SOURCE_PAYLOAD_STATE_READY &&
                         facts.trust_class ==
                             YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED &&
                         facts.trusted_shard_count == 2u &&
                         strlen(facts.payload_identity) == 64u &&
                         facts.peak_inflight_host_bytes == 4096u &&
                         facts.peak_open_handles == 1u &&
                         facts.handle_evictions != 0u,
                     "payload session publishes trust and enforced resource facts");
    if (test_payload_transform_binding(&fixture, &facts) != 0) return 1;
    YVEX_TEST_ASSERT(yvex_source_payload_plan_range_at(plan, 0u) &&
                         yvex_source_payload_plan_range_at(plan, 0u)->trust_class ==
                             YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED &&
                         strcmp(yvex_source_payload_plan_range_at(
                                    plan, 0u)->payload_identity,
                                facts.payload_identity) == 0,
                     "pre-trust plan projects the committed range trust binding");
    memset(&parsed, 0, sizeof(parsed));
    snprintf(parsed.resolved_source_path,
             sizeof(parsed.resolved_source_path), "%s",
             fixture.verification.resolved_source_path);
    YVEX_TEST_ASSERT(yvex_source_provenance_manifest_read(
                         &fixture.verify_options, &parsed, &err) == YVEX_OK &&
                         strcmp(parsed.manifest_schema,
                                "yvex.source_manifest.v3") == 0 &&
                         strcmp(parsed.manifest_payload_identity,
                                facts.payload_identity) == 0 &&
                         parsed.manifest_payload_shard_count == 2u &&
                         !payload_has_blocker(
                             &parsed, "malformed-source-manifest"),
                     "payload manifest parser roundtrip");
    YVEX_TEST_ASSERT(payload_test_read_text(
                         fixture.manifest, manifest_text,
                         sizeof(manifest_text)),
                     "read payload manifest for version refusal");
    snprintf(malformed_text, sizeof(malformed_text), "%s", manifest_text);
    field = strstr(malformed_text, "\"observed_digest\": \"");
    YVEX_TEST_ASSERT(field != NULL, "payload observed digest is present");
    field += strlen("\"observed_digest\": \"");
    *field = *field == '0' ? '1' : '0';
    YVEX_TEST_ASSERT(payload_test_write_text(
                         fixture.manifest, malformed_text),
                     "write inconsistent payload digest manifest");
    memset(&parsed, 0, sizeof(parsed));
    snprintf(parsed.resolved_source_path,
             sizeof(parsed.resolved_source_path), "%s",
             fixture.verification.resolved_source_path);
    YVEX_TEST_ASSERT(yvex_source_provenance_manifest_read(
                         &fixture.verify_options, &parsed, &err) == YVEX_OK &&
                         payload_has_blocker(
                             &parsed, "malformed-source-manifest"),
                     "payload manifest parser refuses unequal trusted digests");
    YVEX_TEST_ASSERT(payload_test_write_text(
                         fixture.manifest, manifest_text),
                     "restore canonical payload manifest");
    schema_version = strstr(manifest_text, "yvex.source_manifest.v3");
    YVEX_TEST_ASSERT(schema_version != NULL,
                     "payload manifest schema is present");
    schema_version[strlen("yvex.source_manifest.v") ] = '9';
    YVEX_TEST_ASSERT(payload_test_write_text(
                         fixture.manifest, manifest_text),
                     "write unsupported payload manifest version");
    memset(&parsed, 0, sizeof(parsed));
    snprintf(parsed.resolved_source_path,
             sizeof(parsed.resolved_source_path), "%s",
             fixture.verification.resolved_source_path);
    YVEX_TEST_ASSERT(yvex_source_provenance_manifest_read(
                         &fixture.verify_options, &parsed, &err) == YVEX_OK &&
                         payload_has_blocker(
                             &parsed, "unsupported-source-manifest-version"),
                     "payload manifest parser refuses unsupported version");
    memset(&sink_state, 0, sizeof(sink_state));
    payload_sink_make(&sink, &sink_state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_stream(
                         fixture.session, single, &sink, &result,
                         &failure, &err) == YVEX_OK && result.committed &&
                         result.physical_bytes_read == 8192u,
                     "trusted exact repeated range stream");
    YVEX_TEST_ASSERT(yvex_source_payload_probe(
                         fixture.session, single,
                         YVEX_SOURCE_PAYLOAD_PROBE_COLD_ADVISORY, 1u, &probe,
                         &failure, &err) == YVEX_OK &&
                         probe.page_cache_advice_requested &&
                         probe.logical_bytes == 8192u &&
                         probe.physical_bytes == 8192u,
                     "cold-advisory probe reports real advice and IO facts");
    YVEX_TEST_ASSERT(yvex_source_payload_probe(
                         fixture.session, single,
                         YVEX_SOURCE_PAYLOAD_PROBE_WARM, 1u, &probe,
                         &failure, &err) == YVEX_OK &&
                         probe.logical_bytes == 8192u &&
                         probe.physical_bytes == 8192u &&
                         probe.buffer_reuses == 1u,
                     "warm diagnostic probe accounts actual buffer reuse");
    YVEX_TEST_ASSERT(yvex_source_payload_probe(
                         fixture.session, single,
                         YVEX_SOURCE_PAYLOAD_PROBE_REPEATED, 2u, &probe,
                         &failure, &err) == YVEX_OK &&
                         probe.logical_bytes == 16384u &&
                         probe.physical_bytes == 16384u &&
                         probe.buffer_reuses == 2u,
                     "repeated diagnostic probe accounts both exact reads");
    YVEX_TEST_ASSERT(yvex_source_payload_probe(
                         fixture.session, plan,
                         YVEX_SOURCE_PAYLOAD_PROBE_STAGED, 1u, &probe,
                         &failure, &err) == YVEX_OK &&
                         probe.logical_bytes == 22384u &&
                         probe.physical_bytes == 22384u &&
                         probe.chunks == 6u && probe.buffer_reuses == 3u,
                     "staged probe performs one bounded transaction per range");
    yvex_source_payload_plan_close(single);
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);
    return 0;
}

static int test_payload_local_seal_and_digest_failure(void)
{
    payload_fixture fixture;
    yvex_source_payload_failure failure;
    yvex_source_payload_plan *plan = NULL;
    yvex_source_payload_sink sink;
    payload_sink_state state;
    yvex_source_payload_stream_result result;
    yvex_source_payload_session_facts facts;
    yvex_source_verification parsed;
    yvex_error err;
    char invalid_digest[65];

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "seal", 0, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "local seal fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "local seal plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) == YVEX_OK,
                     "local payload seal completes");
    YVEX_TEST_ASSERT(yvex_source_payload_session_facts_get(
                         fixture.session, &facts, &err) == YVEX_OK &&
                         facts.trust_class ==
                             YVEX_SOURCE_PAYLOAD_TRUST_LOCAL_SNAPSHOT_SEALED,
                     "local seal is not promoted to upstream verification");
    memset(&parsed, 0, sizeof(parsed));
    snprintf(parsed.resolved_source_path,
             sizeof(parsed.resolved_source_path), "%s",
             fixture.verification.resolved_source_path);
    YVEX_TEST_ASSERT(yvex_source_provenance_manifest_read(
                         &fixture.verify_options, &parsed, &err) == YVEX_OK &&
                         strcmp(parsed.manifest_payload_trust_class,
                                "local_payload_snapshot_sealed") == 0 &&
                         !payload_has_blocker(
                             &parsed, "malformed-source-manifest"),
                     "mixed-authority local seal manifest reopens exactly");
    yvex_source_payload_plan_close(plan);
    plan = NULL;
    YVEX_TEST_ASSERT(yvex_source_payload_session_release(
                         &fixture.session, &failure, &err) == YVEX_OK &&
                         payload_fixture_reopen_policy(
                             &fixture, 2u, 0, NULL, &failure, &err) ==
                             YVEX_ERR_UNSUPPORTED &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_EXPECTED_DIGEST_UNAVAILABLE,
                     "missing upstream digest refuses when local sealing is disabled");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "digest-syntax", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "digest syntax fixture closes admitted session");
    memset(invalid_digest, 'g', 64u);
    invalid_digest[64] = '\0';
    YVEX_TEST_ASSERT(payload_test_metadata(
                         fixture.root,
                         "model-00001-of-00002.safetensors",
                         invalid_digest) &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_DIGEST_ALGORITHM_UNSUPPORTED,
                     "non-SHA provider digest syntax is refused");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "identity-mismatch", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "prior payload identity fixture opens");
    fixture.session->verification.manifest_payload_trusted = 1;
    memset(fixture.session->verification.manifest_payload_identity,
           '0', 64u);
    fixture.session->verification.manifest_payload_identity[64] = '\0';
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "prior payload identity plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) != YVEX_OK &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_IDENTITY_MISMATCH &&
                         result.aborted && !result.committed,
                     "published payload identity mismatch poisons without reseal");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "digest-mismatch", 1, 1, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "digest mismatch fixture opens before trust");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "digest mismatch plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) != YVEX_OK &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_DIGEST_MISMATCH &&
                         result.aborted && !result.committed && state.aborts == 1u,
                     "digest mismatch aborts provisional delivery");
    YVEX_TEST_ASSERT(yvex_source_payload_session_facts_get(
                         fixture.session, &facts, &err) == YVEX_OK &&
                         facts.state == YVEX_SOURCE_PAYLOAD_STATE_POISONED &&
                         facts.digest_mismatches == 1u,
                     "digest mismatch poisons session without trust publication");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);
    return 0;
}

static int test_payload_exact_read_faults(void)
{
    payload_fixture fixture;
    yvex_source_payload_ops ops;
    yvex_source_payload_failure failure;
    yvex_source_payload_plan *plan = NULL;
    yvex_source_payload_sink sink;
    payload_sink_state state;
    yvex_source_payload_stream_result result;
    yvex_error err;

    payload_fault_reset();
    payload_fault.eintr_at = 0u;
    payload_fault.partial_limit = 37u;
    payload_fault_ops(&ops);
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "partial", 1, 0, 2u, &ops,
                         &failure, &err) == YVEX_OK,
                     "partial/EINTR fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "partial/EINTR plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) == YVEX_OK && result.committed &&
                         payload_fault.pread_calls > 800u,
                     "EINTR and repeated partial reads complete exactly");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);
    YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                     "partial-read fixture releases injected allocations");

    payload_fault_reset();
    payload_fault.eof_at = 2u;
    payload_fault_ops(&ops);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "short", 1, 0, 2u, &ops,
                         &failure, &err) == YVEX_OK,
                     "short-read fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "short-read plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_SHORT_READ &&
                         result.aborted && !result.committed,
                     "premature EOF is typed short-read abort");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);
    YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                     "short-read fixture releases injected allocations");

    payload_fault_reset();
    payload_fault.io_at = 1u;
    payload_fault_ops(&ops);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "hard-io", 1, 0, 2u, &ops,
                         &failure, &err) == YVEX_OK,
                     "hard IO fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "hard IO plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) == YVEX_ERR_IO &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_IO &&
                         result.aborted && !result.committed,
                     "hard positioned-read failure aborts transaction");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);
    YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                     "hard IO fixture releases injected allocations");
    return 0;
}

static int test_payload_consumer_cancel_and_publication_failure(void)
{
    payload_fixture fixture;
    yvex_source_payload_failure failure;
    yvex_source_payload_plan *plan = NULL;
    yvex_source_payload_sink sink;
    payload_sink_state state;
    yvex_source_payload_stream_result result;
    yvex_source_payload_session_facts facts;
    yvex_error err;
    char manifest[256];

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "consumer", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "consumer refusal fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "consumer refusal plan");
    memset(&state, 0, sizeof(state));
    state.fail_after_chunks = 2u;
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_CONSUMER &&
                         result.delivered_logical_bytes == 8192u &&
                         result.aborted && state.aborts == 1u,
                     "consumer refusal aborts with exact delivered count");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "cancel", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "cancellation fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "cancellation plan");
    memset(&state, 0, sizeof(state));
    state.cancel_session = fixture.session;
    state.cancel_after_chunks = 1u;
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) == YVEX_ERR_CANCELLED &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_CANCELLED &&
                         result.delivered_logical_bytes == 4096u &&
                         result.aborted,
                     "cancellation aborts between bounded chunks");
    YVEX_TEST_ASSERT(yvex_source_payload_session_reset_cancel(
                         fixture.session, &failure, &err) == YVEX_OK,
                     "idle untrusted session cancellation resets explicitly");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "publish", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "publication failure fixture opens");
    YVEX_TEST_ASSERT(payload_test_write_text(
                         fixture.manifest, "previous-valid-manifest\n"),
                     "publication prior manifest");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "publication failure plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_FAIL_SOURCE_PUBLISH_AFTER_WRITE",
                            "1", 1) == 0,
                     "publication failure seam enabled");
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) != YVEX_OK && result.aborted &&
                         !result.committed,
                     "atomic publication failure aborts trust stream");
    unsetenv("YVEX_TEST_FAIL_SOURCE_PUBLISH_AFTER_WRITE");
    YVEX_TEST_ASSERT(payload_test_read_text(
                         fixture.manifest, manifest, sizeof(manifest)) &&
                         strcmp(manifest, "previous-valid-manifest\n") == 0,
                     "atomic failure preserves prior manifest");
    YVEX_TEST_ASSERT(yvex_source_payload_session_facts_get(
                         fixture.session, &facts, &err) == YVEX_OK &&
                         facts.state == YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED,
                     "publication failure publishes no partial trust state");
    YVEX_TEST_ASSERT(facts.trust_class == YVEX_SOURCE_PAYLOAD_TRUST_NONE &&
                         facts.trusted_shard_count == 0u &&
                         facts.payload_identity[0] == '\0',
                     "publication failure rolls back every aggregate trust fact");
    yvex_source_payload_plan_close(plan);
    payload_fixture_close(&fixture);
    return 0;
}

static int test_payload_construction_faults(void)
{
    payload_fixture fixture;
    yvex_source_payload_ops ops;
    yvex_source_payload_failure failure;
    yvex_error err;
    unsigned long long allocation;
    unsigned int plan_allocation;
    int reached_success = 0;

    yvex_error_clear(&err);
    payload_fault_reset();
    payload_fault.open_fail_at = 0u;
    payload_fault_ops(&ops);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "open-failure", 1, 0, 2u, &ops,
                         &failure, &err) == YVEX_ERR_IO &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_OPEN,
                     "source root open failure is typed");
    payload_fixture_close(&fixture);
    YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                     "root open failure unwinds all allocations");

    payload_fault_reset();
    payload_fault.fstat_fail_at = 0u;
    payload_fault_ops(&ops);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "fstat-failure", 1, 0, 2u, &ops,
                         &failure, &err) == YVEX_ERR_IO &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_STAT,
                     "source shard fstat failure is typed");
    payload_fixture_close(&fixture);
    YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                     "fstat failure unwinds all allocations");

    for (allocation = 0u; allocation < 32u; ++allocation) {
        int rc;

        payload_fault_reset();
        payload_fault.allocation_fail_at = allocation;
        payload_fault_ops(&ops);
        rc = payload_fixture_create(
            &fixture, "allocation-failure", 1, 0, 2u, &ops,
            &failure, &err);
        if (rc == YVEX_OK) reached_success = 1;
        else
            YVEX_TEST_ASSERT(
                rc == YVEX_ERR_NOMEM &&
                    failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION,
                "session construction allocation failure is typed");
        payload_fixture_close(&fixture);
        YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                         "session construction allocation failure has no leak");
        if (reached_success) break;
    }
    YVEX_TEST_ASSERT(reached_success,
                     "allocation sweep crosses every session ownership transition");

    for (plan_allocation = 0u; plan_allocation < 3u; ++plan_allocation) {
        yvex_source_payload_plan *plan = NULL;
        unsigned long long next;

        payload_fault_reset();
        payload_fault_ops(&ops);
        YVEX_TEST_ASSERT(payload_fixture_create(
                             &fixture, "plan-allocation", 1, 0, 2u, &ops,
                             &failure, &err) == YVEX_OK,
                         "plan allocation fixture opens");
        next = payload_fault.allocation_calls + plan_allocation;
        payload_fault.allocation_fail_at = next;
        YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                             &plan, fixture.session, 4096u, 4096u,
                             &failure, &err) == YVEX_ERR_NOMEM &&
                             failure.code ==
                                 YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION,
                         "plan ownership allocation failure is typed");
        payload_fixture_close(&fixture);
        YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                         "plan allocation failure has no leak");
    }
    {
        yvex_source_payload_plan *plan = NULL;
        yvex_source_payload_sink sink;
        payload_sink_state state;
        yvex_source_payload_stream_result result;

        payload_fault_reset();
        payload_fault_ops(&ops);
        YVEX_TEST_ASSERT(payload_fixture_create(
                             &fixture, "stream-allocation", 1, 0, 2u, &ops,
                             &failure, &err) == YVEX_OK &&
                             yvex_source_payload_plan_build_all(
                                 &plan, fixture.session, 4096u, 4096u,
                                 &failure, &err) == YVEX_OK,
                         "stream allocation fixture and plan");
        payload_fault.allocation_fail_at = payload_fault.allocation_calls;
        memset(&state, 0, sizeof(state));
        payload_sink_make(&sink, &state);
        YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                             fixture.session, plan, &sink, &result,
                             &failure, &err) == YVEX_ERR_NOMEM &&
                             failure.code ==
                                 YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION &&
                             !result.committed && !result.aborted,
                         "pre-begin stream buffer allocation failure is typed");
        yvex_source_payload_plan_close(plan);
        payload_fixture_close(&fixture);
        YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                         "stream buffer allocation failure has no leak");
    }
    payload_fault_reset();
    payload_fault_ops(&ops);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "cleanup-failure", 1, 0, 2u, &ops,
                         &failure, &err) == YVEX_OK,
                     "cleanup failure fixture opens");
    payload_fault.close_fail_at = payload_fault.close_calls;
    YVEX_TEST_ASSERT(yvex_source_payload_session_close(
                         fixture.session, &failure, &err) == YVEX_ERR_IO &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_CLEANUP,
                     "descriptor cleanup failure remains typed after close");
    YVEX_TEST_ASSERT(yvex_source_payload_session_release(
                         &fixture.session, &failure, &err) == YVEX_OK,
                     "closed session releases memory after cleanup refusal");
    payload_fixture_close(&fixture);
    YVEX_TEST_ASSERT(payload_fault.live_allocations == 0,
                     "cleanup failure releases every owned allocation");
    return 0;
}

static int test_payload_path_admission(void)
{
    payload_fixture fixture;
    yvex_source_payload_failure failure;
    yvex_error err;
    char shard[768];
    char replacement[800];
    FILE *file;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        yvex_source_payload_name_is_canonical(
            "FL2VA/text_encoder/model-00001-of-00014.safetensors"),
        "normalized nested source shard path is admitted");
    YVEX_TEST_ASSERT(
        !yvex_source_payload_name_is_canonical(
            "FL2VA//text_encoder/model-00001-of-00014.safetensors") &&
            !yvex_source_payload_name_is_canonical(
                "FL2VA/../text_encoder/model-00001-of-00014.safetensors") &&
            !yvex_source_payload_name_is_canonical(
                "/FL2VA/text_encoder/model-00001-of-00014.safetensors") &&
            !yvex_source_payload_name_is_canonical(
                "FL2VA\\text_encoder\\model-00001-of-00014.safetensors"),
        "empty, traversal, absolute, and backslash shard paths are refused");
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "missing-shard", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "missing shard fixture closes admitted session");
    snprintf(shard, sizeof(shard), "%s/model-00002-of-00002.safetensors",
             fixture.root);
    YVEX_TEST_ASSERT(unlink(shard) == 0 &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) == YVEX_ERR_IO &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_OPEN,
                     "missing indexed shard is refused on admission");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "symlink-shard", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "symlink fixture closes admitted session");
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors",
             fixture.root);
    snprintf(replacement, sizeof(replacement), "%s.real", shard);
    YVEX_TEST_ASSERT(rename(shard, replacement) == 0 &&
                         symlink("model-00001-of-00002.safetensors.real",
                                 shard) == 0 &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) == YVEX_ERR_IO &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_SYMLINK_REFUSED,
                     "unexpected source shard symlink is refused");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "size-drift", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "size drift fixture closes admitted session");
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors",
             fixture.root);
    file = fopen(shard, "ab");
    YVEX_TEST_ASSERT(file != NULL && fputc(0, file) != EOF && fclose(file) == 0 &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_SIZE_MISMATCH,
                     "source shard size drift is refused");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "non-regular", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "non-regular fixture closes admitted session");
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors",
             fixture.root);
    YVEX_TEST_ASSERT(unlink(shard) == 0 && mkdir(shard, 0777) == 0 &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_NON_REGULAR_SHARD,
                     "non-regular source shard is refused");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "path-escape", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK &&
                         payload_fixture_replace_snapshot(
                             &fixture, "../outside.safetensors", 8u, &err) &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_PATH_ESCAPE,
                     "root-relative shard traversal is refused before open");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "range-outside", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK &&
                         payload_fixture_replace_snapshot(
                             &fixture,
                             "model-00001-of-00002.safetensors",
                             20000u, &err) &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_BOUNDS &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OUTSIDE_DATA_REGION,
                     "tensor range outside retained data region is refused");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "missing-owner", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK &&
                         payload_fixture_replace_snapshot_facts(
                             &fixture, "absent.safetensors",
                             "model-00001-of-00002.safetensors", "I8",
                             8u, 0u, 8u, 16u, 16384u, &err) &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_NOT_INDEXED,
                     "tensor ownership by an absent shard is refused");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "range-file", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK &&
                         payload_fixture_replace_snapshot_facts(
                             &fixture,
                             "model-00001-of-00002.safetensors",
                             "model-00001-of-00002.safetensors", "I8",
                             8u, 0u, 8u, 16399u, 16384u, &err) &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_BOUNDS &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OUTSIDE_FILE,
                     "checked absolute tensor range cannot exceed EOF");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "range-overflow", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK &&
                         payload_fixture_replace_snapshot_facts(
                             &fixture,
                             "model-00001-of-00002.safetensors",
                             "model-00001-of-00002.safetensors", "I8",
                             8u, 0u, 8u, ULLONG_MAX - 4u, 16384u, &err) &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_BOUNDS &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                     "checked tensor absolute offset addition refuses overflow");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "length-mismatch", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK &&
                         payload_fixture_replace_snapshot_facts(
                             &fixture,
                             "model-00001-of-00002.safetensors",
                             "model-00001-of-00002.safetensors", "I8",
                             7u, 0u, 8u, 16u, 16384u, &err) &&
                         payload_fixture_reopen(
                             &fixture, 2u, NULL, &failure, &err) ==
                             YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_TENSOR_LENGTH_MISMATCH,
                     "dtype and shape byte-length inconsistency is refused");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "metadata-gate", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "metadata gate fixture closes admitted session");
    fixture.verification.verified = 0;
    YVEX_TEST_ASSERT(payload_fixture_reopen(
                         &fixture, 2u, NULL, &failure, &err) ==
                         YVEX_ERR_STATE &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_METADATA_NOT_VERIFIED,
                     "directory plus snapshot cannot bypass metadata verification");
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "identity-gate", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "source identity fixture closes admitted session");
    fixture.verification.repository_id[0] = 'x';
    YVEX_TEST_ASSERT(payload_fixture_reopen(
                         &fixture, 2u, NULL, &failure, &err) ==
                         YVEX_ERR_FORMAT &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_SOURCE_IDENTITY_MISMATCH,
                     "session refuses verification bound to another source identity");
    payload_fixture_close(&fixture);
    return 0;
}

static int test_payload_concurrency_and_contention(void)
{
    payload_fixture fixture;
    yvex_source_payload_failure failure;
    yvex_source_payload_plan *all = NULL;
    yvex_source_payload_plan *alpha_plan = NULL;
    yvex_source_payload_plan *gamma_plan = NULL;
    yvex_source_payload_sink sink;
    payload_sink_state state;
    yvex_source_payload_stream_result result;
    yvex_source_payload_session_facts facts;
    payload_stream_gate gate;
    payload_blocking_sink blocking;
    payload_stream_thread first;
    payload_stream_thread second;
    pthread_t first_thread;
    pthread_t second_thread;
    unsigned long long alpha;
    unsigned long long gamma;
    int pinned_fd;
    yvex_error err;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "contention", 1, 0, 1u, NULL,
                         &failure, &err) == YVEX_OK,
                     "contention fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &all, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "contention trust plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, all, &sink, &result,
                         &failure, &err) == YVEX_OK,
                     "contention fixture becomes trusted");
    yvex_source_payload_plan_close(all);
    all = NULL;
    YVEX_TEST_ASSERT(yvex_source_tensor_snapshot_find_index(
                         fixture.snapshot, "alpha.weight", &alpha) &&
                         yvex_source_tensor_snapshot_find_index(
                             fixture.snapshot, "gamma.weight", &gamma),
                     "contention tensor indexes");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build(
                         &alpha_plan, fixture.session, &alpha, 1u,
                         4096u, 4096u, &failure, &err) == YVEX_OK &&
                         yvex_source_payload_plan_build(
                             &gamma_plan, fixture.session, &gamma, 1u,
                             4096u, 4096u, &failure, &err) == YVEX_OK,
                     "contention single-range plans");
    fixture.session->budget.maximum_streams = 1u;
    memset(&gate, 0, sizeof(gate));
    pthread_mutex_init(&gate.mutex, NULL);
    pthread_cond_init(&gate.condition, NULL);
    memset(&first, 0, sizeof(first));
    first.session = fixture.session;
    first.plan = alpha_plan;
    payload_blocking_sink_make(&first.sink, &blocking, &gate);
    YVEX_TEST_ASSERT(pthread_create(
                         &first_thread, NULL, payload_stream_thread_run,
                         &first) == 0,
                     "start blocked payload stream");
    payload_gate_wait(&gate, 1u);
    YVEX_TEST_ASSERT(yvex_source_payload_session_close(
                         fixture.session, &failure, &err) == YVEX_ERR_STATE &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_CLOSE_BUSY,
                     "session close refuses an active stream");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_stream(
                         fixture.session, gamma_plan, &sink, &result,
                         &failure, &err) == YVEX_ERR_BOUNDS &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET,
                     "maximum stream budget refuses contention");
    YVEX_TEST_ASSERT(yvex_source_payload_handle_acquire(
                         fixture.session, gamma_plan->ranges[0].shard_index,
                         &pinned_fd, &failure, &err) == YVEX_ERR_STATE &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_HANDLE_CACHE_EXHAUSTED,
                     "pinned handle cannot be evicted under low FD budget");
    payload_gate_release(&gate);
    pthread_join(first_thread, NULL);
    YVEX_TEST_ASSERT(first.rc == YVEX_OK && first.result.committed,
                     "blocked stream resumes and commits exactly");
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);
    yvex_source_payload_plan_close(gamma_plan);
    yvex_source_payload_plan_close(alpha_plan);
    payload_fixture_close(&fixture);

    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "concurrent", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "concurrent fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &all, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "concurrent trust plan");
    memset(&gate, 0, sizeof(gate));
    pthread_mutex_init(&gate.mutex, NULL);
    pthread_cond_init(&gate.condition, NULL);
    memset(&first, 0, sizeof(first));
    first.session = fixture.session;
    first.plan = all;
    first.verify = 1;
    payload_blocking_sink_make(&first.sink, &blocking, &gate);
    YVEX_TEST_ASSERT(pthread_create(
                         &first_thread, NULL, payload_stream_thread_run,
                         &first) == 0,
                     "start atomic payload verification transition");
    payload_gate_wait(&gate, 1u);
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, all, &sink, &result,
                         &failure, &err) == YVEX_ERR_STATE &&
                         failure.code ==
                             YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_STATE,
                     "second verifier cannot enter a VERIFYING session");
    payload_gate_release(&gate);
    pthread_join(first_thread, NULL);
    YVEX_TEST_ASSERT(first.rc == YVEX_OK && first.result.committed,
                     "single admitted verifier commits after contention");
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);
    yvex_source_payload_plan_close(all);
    YVEX_TEST_ASSERT(yvex_source_tensor_snapshot_find_index(
                         fixture.snapshot, "alpha.weight", &alpha) &&
                         yvex_source_tensor_snapshot_find_index(
                             fixture.snapshot, "gamma.weight", &gamma) &&
                         yvex_source_payload_plan_build(
                             &alpha_plan, fixture.session, &alpha, 1u,
                             4096u, 4096u, &failure, &err) == YVEX_OK &&
                         yvex_source_payload_plan_build(
                             &gamma_plan, fixture.session, &gamma, 1u,
                             4096u, 4096u, &failure, &err) == YVEX_OK,
                     "concurrent range plans");
    memset(&gate, 0, sizeof(gate));
    pthread_mutex_init(&gate.mutex, NULL);
    pthread_cond_init(&gate.condition, NULL);
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    first.session = second.session = fixture.session;
    first.plan = alpha_plan;
    second.plan = gamma_plan;
    {
        static payload_blocking_sink first_blocking;
        static payload_blocking_sink second_blocking;
        payload_blocking_sink_make(&first.sink, &first_blocking, &gate);
        payload_blocking_sink_make(&second.sink, &second_blocking, &gate);
    }
    YVEX_TEST_ASSERT(pthread_create(
                         &first_thread, NULL, payload_stream_thread_run,
                         &first) == 0 &&
                         pthread_create(
                             &second_thread, NULL, payload_stream_thread_run,
                             &second) == 0,
                     "start two concurrent range streams");
    payload_gate_wait(&gate, 2u);
    payload_gate_release(&gate);
    pthread_join(first_thread, NULL);
    pthread_join(second_thread, NULL);
    YVEX_TEST_ASSERT(first.rc == YVEX_OK && second.rc == YVEX_OK &&
                         first.result.committed && second.result.committed,
                     "different-shard positioned reads commit concurrently");
    YVEX_TEST_ASSERT(yvex_source_payload_session_facts_get(
                         fixture.session, &facts, &err) == YVEX_OK &&
                         facts.peak_active_streams >= 2u,
                     "concurrent stream accounting observes overlap");
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);

    memset(&gate, 0, sizeof(gate));
    pthread_mutex_init(&gate.mutex, NULL);
    pthread_cond_init(&gate.condition, NULL);
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    first.session = second.session = fixture.session;
    first.plan = second.plan = alpha_plan;
    {
        static payload_blocking_sink first_repeat;
        static payload_blocking_sink second_repeat;
        payload_blocking_sink_make(&first.sink, &first_repeat, &gate);
        payload_blocking_sink_make(&second.sink, &second_repeat, &gate);
    }
    YVEX_TEST_ASSERT(pthread_create(
                         &first_thread, NULL, payload_stream_thread_run,
                         &first) == 0 &&
                         pthread_create(
                             &second_thread, NULL, payload_stream_thread_run,
                             &second) == 0,
                     "start concurrent repeated-range streams");
    payload_gate_wait(&gate, 2u);
    payload_gate_release(&gate);
    pthread_join(first_thread, NULL);
    pthread_join(second_thread, NULL);
    YVEX_TEST_ASSERT(first.rc == YVEX_OK && second.rc == YVEX_OK &&
                         first.result.committed && second.result.committed,
                     "same-range positioned reads commit concurrently");
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);
    yvex_source_payload_plan_close(gamma_plan);
    yvex_source_payload_plan_close(alpha_plan);
    payload_fixture_close(&fixture);
    return 0;
}

static int test_payload_drift_and_lifecycle(void)
{
    payload_fixture fixture;
    yvex_source_payload_failure failure;
    yvex_source_payload_plan *plan = NULL;
    yvex_source_payload_sink sink;
    payload_sink_state state;
    yvex_source_payload_stream_result result;
    yvex_source_payload_session_facts facts;
    yvex_error err;
    char shard[768];
    char prior[800];

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "drift", 1, 0, 1u, NULL,
                         &failure, &err) == YVEX_OK,
                     "drift fixture opens");
    YVEX_TEST_ASSERT(yvex_source_payload_plan_build_all(
                         &plan, fixture.session, 4096u, 4096u,
                         &failure, &err) == YVEX_OK,
                     "drift fixture plan");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_verify(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) == YVEX_OK,
                     "drift fixture trusts initial snapshot");
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors",
             fixture.root);
    snprintf(prior, sizeof(prior), "%s.replaced", shard);
    YVEX_TEST_ASSERT(rename(shard, prior) == 0 &&
                         payload_test_write_bytes(shard, 16u, 16384u, 1u),
                     "replace shard path with same-size file");
    memset(&state, 0, sizeof(state));
    payload_sink_make(&sink, &state);
    YVEX_TEST_ASSERT(yvex_source_payload_session_stream(
                         fixture.session, plan, &sink, &result,
                         &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_DRIFT &&
                         !result.committed,
                     "same-size shard replacement poisons cached reader");
    yvex_source_payload_plan_close(plan);
    plan = NULL;
    YVEX_TEST_ASSERT(yvex_source_payload_session_close(
                         fixture.session, &failure, &err) == YVEX_OK &&
                         yvex_source_payload_session_facts_get(
                             fixture.session, &facts, &err) == YVEX_OK &&
                         facts.buffer_allocations != 0u &&
                         facts.buffer_releases == facts.buffer_allocations,
                     "payload close releases every retained bounded buffer");
    YVEX_TEST_ASSERT(yvex_source_payload_session_release(
                         &fixture.session, &failure, &err) == YVEX_OK &&
                         fixture.session == NULL &&
                         yvex_source_payload_session_release(
                             &fixture.session, &failure, &err) == YVEX_OK,
                     "payload session release is pointer-idempotent");
    payload_fixture_close(&fixture);
    return 0;
}

static int test_payload_acquired_identity(void)
{
    payload_fixture fixture;
    yvex_source_acquisition *acquisition = NULL;
    yvex_source_acquisition_failure acquisition_failure;
    const yvex_source_acquisition_facts *acquisition_facts;
    yvex_source_payload_open_options options;
    yvex_source_payload_session *session = NULL;
    yvex_source_payload_failure failure;
    yvex_source_tensor_snapshot_facts snapshot;
    yvex_error err;
    char shard[768];

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(payload_fixture_create(
                         &fixture, "acquired", 1, 0, 2u, NULL,
                         &failure, &err) == YVEX_OK,
                     "acquired payload fixture opens");
    YVEX_TEST_ASSERT(payload_acquisition_fixture_open(
                         &fixture, &acquisition, &acquisition_failure, &err) && acquisition,
                     "acquired source binds exact shard digests and local identities");
    acquisition_facts = yvex_source_acquisition_facts_get(acquisition);
    YVEX_TEST_ASSERT(acquisition_facts && acquisition_facts->complete &&
                         yvex_source_tensor_snapshot_facts_get(
                             fixture.snapshot, &snapshot, &err) == YVEX_OK,
                     "acquired payload fixture identities are available");
    memset(&options, 0, sizeof(options));
    options.acquisition = acquisition;
    options.acquired_source_root = fixture.root;
    options.acquired_target_id = "fixture-target";
    options.acquired_family_key = "fixture-family";
    options.acquired_payload_identity = acquisition_facts->acquisition_identity;
    options.acquired_source_snapshot_identity = snapshot.identity;
    options.snapshot = fixture.snapshot;
    yvex_source_payload_budget_default(&options.budget);
    YVEX_TEST_ASSERT(yvex_source_payload_session_open(
                         &session, &options, &failure, &err) == YVEX_OK && session,
                     "unchanged acquired shards open through verified local identities");
    YVEX_TEST_ASSERT(yvex_source_payload_session_release(
                         &session, &failure, &err) == YVEX_OK && !session,
                     "acquired source session releases cleanly");
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors", fixture.root);
    YVEX_TEST_ASSERT(payload_test_write_bytes(shard, 16u, 16384u, 9u),
                     "same-size acquired shard corruption is prepared");
    YVEX_TEST_ASSERT(yvex_source_payload_session_open(
                         &session, &options, &failure, &err) != YVEX_OK && !session &&
                         failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_DRIFT,
                     "post-verification acquired shard replacement is refused");
    yvex_source_acquisition_release(&acquisition);
    payload_fixture_close(&fixture);
    return 0;
}

int yvex_test_source_payload(void)
{
    if (test_storage_shard_index_foundation() != 0) return 1;
    if (test_sha256_primitive() != 0) return 1;
    if (test_payload_happy_path() != 0) return 1;
    if (test_payload_component_physical_shape_fold() != 0) return 1;
    if (test_payload_local_seal_and_digest_failure() != 0) return 1;
    if (test_payload_exact_read_faults() != 0) return 1;
    if (test_payload_consumer_cancel_and_publication_failure() != 0) return 1;
    if (test_payload_drift_and_lifecycle() != 0) return 1;
    if (test_payload_acquired_identity() != 0) return 1;
    if (test_payload_construction_faults() != 0) return 1;
    if (test_payload_path_admission() != 0) return 1;
    if (test_payload_concurrency_and_contention() != 0) return 1;
    return 0;
}
