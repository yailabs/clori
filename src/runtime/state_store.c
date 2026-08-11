/*
 * Persist committed target and draft sequence state as one immutable checkpoint.
 *
 * The file is streamed into an unpublished sibling, hashed over explicit fields and payload
 * bytes, synchronized, and linked into place without replacement. Restore maps a stable file,
 * validates its complete digest and model identities, then lets each graph provider copy the
 * admitted state into its inactive bank. The native scalar marker makes this first schema
 * deliberately host-ABI bound instead of pretending raw F32 state is cross-endian.
 */
#define _POSIX_C_SOURCE 200809L

#include "src/runtime/private.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <yvex/internal/core.h>
#include <yvex/internal/runtime_state_store.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif

#define STATE_FILE_MAGIC "YVSTATE1"
#define STATE_FILE_MAGIC_BYTES 8u
#define STATE_FILE_NAME_CAP 256u
#define STATE_FILE_ENDIAN_MARKER 0x0102030405060708ull
#define STATE_FILE_F32_MARKER 0x000000003f800000ull
#define STATE_FILE_DIGEST_BYTES YVEX_SHA256_DIGEST_BYTES
#define STATE_FILE_HEADER_V1_BYTES 240ull
#define STATE_FILE_HEADER_V2_BYTES 312ull
#define STATE_FILE_SCOPE_HEADER_BYTES 216ull
#define STATE_FILE_CAPACITY_FIELDS 27ull
#define STATE_FILE_CAPACITY_CLASS_FIELDS 16ull
#define STATE_FILE_CAPACITY_IDENTITIES 4ull
#define STATE_FILE_RECIPE_HEADER_BYTES 176ull
#define STATE_FILE_RECIPE_COMPONENT_BYTES 264ull
#define STATE_FILE_ROLLING_BYTES 216ull
#define STATE_FILE_LAYER_HEADER_BYTES 552ull

typedef struct {
    int directory_fd, file_fd;
    char destination[STATE_FILE_NAME_CAP], temporary[STATE_FILE_NAME_CAP];
    unsigned long long offset;
    yvex_sha256 hash;
    int temporary_exists, destination_exists;
} state_file_writer;

typedef struct {
    const unsigned char *data;
    size_t count, offset, digest_offset;
} state_file_parser;

typedef struct {
    const yvex_attention_state_provider *provider;
    const yvex_execution_capacity_plan *capacity;
    yvex_graph_attention_state_summary summary;
    unsigned long long scope, file_bytes;
} state_save_scope;

typedef struct {
    yvex_attention_state_checkpoint checkpoint;
    yvex_execution_capacity_plan capacity;
    yvex_graph_attention_capacity_plan *graph_capacity;
    yvex_attention_state_recipe *recipes;
    yvex_attention_history_view *layers;
    char (*identities)[YVEX_SHA256_HEX_CAP];
    unsigned long long scope;
} state_restore_scope;

typedef struct {
    const unsigned char *mapping, *payload;
    size_t mapping_count;
    state_restore_scope scopes[2];
    yvex_runtime_model_summary model;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long schema, scope_count, payload_bytes;
    char payload_identity[YVEX_SHA256_HEX_CAP];
} state_restore_file;

static int state_store_fail(yvex_status status, const char *reason,
                            yvex_error *err)
{
    yvex_error_set(err, status, "runtime.state-store", reason);
    return status;
}

static int state_store_add(unsigned long long left, unsigned long long right,
                           unsigned long long *out, yvex_error *err)
{
    if (!yvex_core_u64_add(left, right, out))
        return state_store_fail(YVEX_ERR_BOUNDS,
                                "state checkpoint size overflowed", err);
    return YVEX_OK;
}

static int state_store_multiply(unsigned long long left,
                                unsigned long long right,
                                unsigned long long *out, yvex_error *err)
{
    if (!yvex_core_u64_mul(left, right, out))
        return state_store_fail(YVEX_ERR_BOUNDS,
                                "state checkpoint extent overflowed", err);
    return YVEX_OK;
}

static int state_store_align(unsigned long long value,
                             unsigned long long *out, yvex_error *err)
{
    unsigned long long adjusted;
    if (!yvex_core_u64_add(value, 7ull, &adjusted))
        return state_store_fail(YVEX_ERR_BOUNDS,
                                "state checkpoint alignment overflowed", err);
    *out = adjusted & ~7ull;
    return YVEX_OK;
}

static int state_file_parent_open(const char *path, int *directory_fd,
                                  char name[STATE_FILE_NAME_CAP],
                                  yvex_error *err)
{
    char copy[YVEX_PATH_CAP];
    char *slash, *cursor, *next;
    int fd = -1;
    if (!path || !path[0] || strnlen(path, sizeof(copy)) >= sizeof(copy))
        return state_store_fail(YVEX_ERR_INVALID_ARG,
                                "state checkpoint path is invalid", err);
    yvex_core_text_copy(copy, sizeof(copy), path);
    slash = strrchr(copy, '/');
    if (!slash) {
        if (snprintf(name, STATE_FILE_NAME_CAP, "%s", copy) >=
            (int)STATE_FILE_NAME_CAP)
            goto unsafe;
        cursor = NULL;
        fd = open(".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    } else {
        if (!slash[1] ||
            snprintf(name, STATE_FILE_NAME_CAP, "%s", slash + 1) >=
                (int)STATE_FILE_NAME_CAP)
            goto unsafe;
        *slash = '\0';
        fd = open(path[0] == '/' ? "/" : ".",
                  O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        cursor = slash == copy ? NULL : copy + (path[0] == '/');
    }
    if (!strcmp(name, ".") || !strcmp(name, "..") || !name[0]) goto unsafe;
    while (fd >= 0 && cursor && *cursor) {
        int child;
        next = strchr(cursor, '/');
        if (next) *next = '\0';
        if (!cursor[0] || !strcmp(cursor, ".") || !strcmp(cursor, ".."))
            goto unsafe;
        child = openat(fd, cursor,
                       O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (child < 0) goto unsafe;
        (void)close(fd);
        fd = child;
        cursor = next ? next + 1 : NULL;
    }
    if (fd < 0) goto unsafe;
    *directory_fd = fd;
    return YVEX_OK;
unsafe:
    if (fd >= 0) (void)close(fd);
    return state_store_fail(YVEX_ERR_IO,
                            "state checkpoint path is unsafe", err);
}

static int state_file_write_exact(int fd, const void *data, size_t count)
{
    const unsigned char *bytes = data;
    size_t offset = 0u;
    while (offset < count) {
        ssize_t wrote = write(fd, bytes + offset, count - offset);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return 0;
        offset += (size_t)wrote;
    }
    return 1;
}

static int state_file_writer_open(state_file_writer *writer, const char *path,
                                  yvex_error *err)
{
    unsigned int attempt;
    memset(writer, 0, sizeof(*writer));
    writer->directory_fd = writer->file_fd = -1;
    if (state_file_parent_open(path, &writer->directory_fd,
                               writer->destination, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_IO;
    for (attempt = 0u; attempt < 64u; ++attempt) {
        if (snprintf(writer->temporary, sizeof(writer->temporary),
                     ".%s.%llu.%u.tmp", writer->destination,
                     (unsigned long long)getpid(), attempt) >=
            (int)sizeof(writer->temporary))
            break;
        writer->file_fd = openat(
            writer->directory_fd, writer->temporary,
            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (writer->file_fd >= 0) break;
        if (errno != EEXIST) break;
    }
    if (writer->file_fd < 0)
        return state_store_fail(YVEX_ERR_IO,
                                "state checkpoint temporary file creation failed",
                                err);
    writer->temporary_exists = 1;
    yvex_sha256_init(&writer->hash);
    return YVEX_OK;
}

static int state_file_writer_write(state_file_writer *writer,
                                   const void *data, size_t count, int digest,
                                   yvex_error *err)
{
    unsigned long long next;
    if (!writer || writer->file_fd < 0 || (!data && count) ||
        !yvex_core_u64_add(writer->offset, count, &next) ||
        (digest && !yvex_sha256_update(&writer->hash, data, count)) ||
        !state_file_write_exact(writer->file_fd, data, count))
        return state_store_fail(YVEX_ERR_IO,
                                "state checkpoint write failed", err);
    writer->offset = next;
    return YVEX_OK;
}

static int state_file_writer_publish(state_file_writer *writer,
                                     yvex_error *err)
{
    int saved;
    if (fsync(writer->file_fd) != 0)
        return state_store_fail(YVEX_ERR_IO,
                                "state checkpoint file synchronization failed",
                                err);
    if (close(writer->file_fd) != 0) {
        writer->file_fd = -1;
        return state_store_fail(YVEX_ERR_IO,
                                "state checkpoint file close failed", err);
    }
    writer->file_fd = -1;
    if (linkat(writer->directory_fd, writer->temporary,
               writer->directory_fd, writer->destination, 0) != 0) {
        saved = errno;
        return state_store_fail(saved == EEXIST ? YVEX_ERR_STATE : YVEX_ERR_IO,
                                saved == EEXIST
                                    ? "state checkpoint destination already exists"
                                    : "state checkpoint publication failed",
                                err);
    }
    writer->destination_exists = 1;
    if (unlinkat(writer->directory_fd, writer->temporary, 0) != 0)
        return state_store_fail(YVEX_ERR_IO,
                                "state checkpoint temporary name cleanup failed",
                                err);
    writer->temporary_exists = 0;
    if (fsync(writer->directory_fd) != 0)
        return state_store_fail(YVEX_ERR_IO,
                                "state checkpoint directory synchronization failed",
                                err);
    return YVEX_OK;
}

static void state_file_writer_close(state_file_writer *writer, int success)
{
    if (!writer) return;
    if (writer->file_fd >= 0) (void)close(writer->file_fd);
    if (writer->directory_fd >= 0) {
        if (!success && writer->destination_exists)
            (void)unlinkat(writer->directory_fd, writer->destination, 0);
        if (writer->temporary_exists)
            (void)unlinkat(writer->directory_fd, writer->temporary, 0);
        if (!success) (void)fsync(writer->directory_fd);
        (void)close(writer->directory_fd);
    }
    memset(writer, 0, sizeof(*writer));
    writer->directory_fd = writer->file_fd = -1;
}

static int state_file_put_u64(state_file_writer *writer,
                              unsigned long long value, yvex_error *err)
{
    return state_file_writer_write(writer, &value, sizeof(value), 1, err);
}

static int state_file_put_identity(state_file_writer *writer, const char *value,
                                   yvex_error *err)
{
    char encoded[64];
    if (!yvex_sha256_hex_valid(value))
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state checkpoint identity is invalid", err);
    memcpy(encoded, value, sizeof(encoded));
    return state_file_writer_write(writer, encoded, sizeof(encoded), 1, err);
}

static int state_file_put_optional_identity(state_file_writer *writer,
                                            const char *value,
                                            yvex_error *err)
{
    static const unsigned char absent[64] = {0};
    return value && value[0]
               ? state_file_put_identity(writer, value, err)
               : state_file_writer_write(writer, absent, sizeof(absent), 1, err);
}

static int state_file_put_blob(state_file_writer *writer, const void *data,
                               unsigned long long bytes, yvex_error *err)
{
    static const unsigned char padding[8] = {0};
    unsigned long long aligned;
    if (bytes > SIZE_MAX || (!data && bytes))
        return state_store_fail(YVEX_ERR_BOUNDS,
                                "state checkpoint payload is unavailable", err);
    if (state_file_writer_write(writer, data, (size_t)bytes, 1, err) != YVEX_OK ||
        state_store_align(bytes, &aligned, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_IO;
    return aligned == bytes
               ? YVEX_OK
               : state_file_writer_write(writer, padding,
                                          (size_t)(aligned - bytes), 1, err);
}

static int state_array_bytes(unsigned long long count,
                             unsigned long long width,
                             unsigned long long element_bytes,
                             unsigned long long *bytes, yvex_error *err)
{
    unsigned long long elements;
    if (state_store_multiply(count, width, &elements, err) != YVEX_OK ||
        state_store_multiply(elements, element_bytes, bytes, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_BOUNDS;
    return YVEX_OK;
}

static int state_view_blob_bytes(const yvex_attention_history_view *view,
                                 unsigned long long blobs[10],
                                 yvex_error *err)
{
    if (!view ||
        state_array_bytes(view->local_tail_count, view->local_kv_stride,
                          sizeof(float), &blobs[0], err) != YVEX_OK ||
        state_array_bytes(view->local_tail_count, 1ull,
                          sizeof(unsigned long long), &blobs[1], err) != YVEX_OK ||
        state_array_bytes(view->compressed_entry_count,
                          view->compressed_kv_stride, sizeof(float),
                          &blobs[2], err) != YVEX_OK ||
        state_array_bytes(view->compressed_entry_count, 1ull,
                          sizeof(unsigned long long), &blobs[3], err) != YVEX_OK ||
        state_array_bytes(view->indexer_entry_count, view->indexer_kv_stride,
                          sizeof(float), &blobs[4], err) != YVEX_OK ||
        state_array_bytes(view->indexer_entry_count, 1ull,
                          sizeof(unsigned long long), &blobs[5], err) != YVEX_OK ||
        state_array_bytes(view->main_rolling_state.kv_state_extent, 1ull,
                          sizeof(float), &blobs[6], err) != YVEX_OK ||
        state_array_bytes(view->main_rolling_state.score_state_extent, 1ull,
                          sizeof(float), &blobs[7], err) != YVEX_OK ||
        state_array_bytes(view->indexer_rolling_state.kv_state_extent, 1ull,
                          sizeof(float), &blobs[8], err) != YVEX_OK ||
        state_array_bytes(view->indexer_rolling_state.score_state_extent, 1ull,
                          sizeof(float), &blobs[9], err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_BOUNDS;
    return YVEX_OK;
}

static int state_view_payload_valid(const yvex_attention_history_view *view,
                                    const unsigned long long blobs[10],
                                    yvex_error *err)
{
    if ((blobs[0] && !view->local_kv) || (blobs[1] && !view->local_positions) ||
        (blobs[2] && !view->compressed_kv) ||
        (blobs[3] && !view->compressed_positions) ||
        (blobs[4] && !view->indexer_kv) ||
        (blobs[5] && !view->indexer_positions) ||
        (blobs[6] && !view->main_rolling_state.kv_state) ||
        (blobs[7] && !view->main_rolling_state.score_state) ||
        (blobs[8] && !view->indexer_rolling_state.kv_state) ||
        (blobs[9] && !view->indexer_rolling_state.score_state))
        return state_store_fail(YVEX_ERR_STATE,
                                "state checkpoint payload pointer is missing", err);
    return YVEX_OK;
}

static int state_view_measure(const yvex_attention_history_view *view,
                              unsigned long long *bytes, yvex_error *err)
{
    unsigned long long blobs[10], total = STATE_FILE_LAYER_HEADER_BYTES;
    unsigned int index;
    if (state_view_blob_bytes(view, blobs, err) != YVEX_OK ||
        state_view_payload_valid(view, blobs, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_BOUNDS;
    for (index = 0u; index < 10u; ++index) {
        unsigned long long aligned;
        if (state_store_align(blobs[index], &aligned, err) != YVEX_OK ||
            state_store_add(total, aligned, &total, err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_BOUNDS;
    }
    *bytes = total;
    return YVEX_OK;
}

static int state_capacity_measure(
    const yvex_execution_capacity_plan *capacity,
    unsigned long long *bytes, yvex_error *err)
{
    unsigned long long classes, total;
    if (!capacity || !bytes)
        return state_store_fail(YVEX_ERR_INVALID_ARG,
                                "state capacity output is invalid", err);
    if (yvex_execution_capacity_plan_validate(capacity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    if (!yvex_core_u64_mul(capacity->state_class_count,
                           STATE_FILE_CAPACITY_CLASS_FIELDS * 8ull, &classes) ||
        !yvex_core_u64_add(STATE_FILE_CAPACITY_FIELDS * 8ull +
                               STATE_FILE_CAPACITY_IDENTITIES * 64ull,
                           classes, &total))
        return state_store_fail(YVEX_ERR_BOUNDS,
                                "state capacity size overflowed", err);
    *bytes = total;
    return YVEX_OK;
}

static int state_recipe_measure(
    const yvex_attention_state_recipe *recipe,
    unsigned long long *bytes, yvex_error *err)
{
    yvex_attention_state_recipe candidate;
    unsigned long long components;
    if (!recipe || !bytes) return YVEX_ERR_INVALID_ARG;
    candidate = *recipe;
    if (yvex_attention_state_recipe_seal(&candidate, err) != YVEX_OK ||
        !yvex_core_u64_mul(candidate.component_count,
                           STATE_FILE_RECIPE_COMPONENT_BYTES, &components) ||
        !yvex_core_u64_add(STATE_FILE_RECIPE_HEADER_BYTES, components, bytes))
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state recipe is not persistable", err);
    return YVEX_OK;
}

static int state_save_scope_prepare(
    state_save_scope *scope, const yvex_attention_state_provider *provider,
    unsigned long long ordinal, yvex_error *err)
{
    unsigned long long layer, capacity_bytes = 0ull;
    unsigned long long bytes = STATE_FILE_SCOPE_HEADER_BYTES;
    char identity[YVEX_SHA256_HEX_CAP];
    if (!scope || !provider ||
        provider->schema_version != YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V8 ||
        !provider->summary || !provider->capacity || !provider->recipe ||
        !provider->view || !provider->identity ||
        !provider->restore ||
        provider->summary(provider->context, &scope->summary, err) != YVEX_OK ||
        !scope->summary.sealed || scope->summary.invalidated ||
        scope->summary.cancelled || scope->summary.transaction_active ||
        scope->summary.prepared_layer_count != scope->summary.layer_count ||
        !scope->summary.position_consistent)
        return state_store_fail(YVEX_ERR_STATE,
                                "only idle complete persistent state may be saved",
                                err);
    scope->provider = provider;
    scope->capacity = provider->capacity(provider->context);
    scope->scope = ordinal;
    if (state_capacity_measure(scope->capacity, &capacity_bytes, err) != YVEX_OK ||
        state_store_add(bytes, capacity_bytes, &bytes, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (layer = 0ull; layer < scope->summary.layer_count; ++layer) {
        const yvex_attention_state_recipe *recipe =
            provider->recipe(provider->context, layer);
        const yvex_attention_history_view *view = provider->view(
            provider->context, layer, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        unsigned long long recipe_bytes = 0ull, layer_bytes = 0ull;
        if (!view || view->token_count !=
                         scope->summary.committed_sequence_length ||
            state_recipe_measure(recipe, &recipe_bytes, err) != YVEX_OK ||
            provider->identity(provider->context, layer, identity, err) !=
                YVEX_OK ||
            !yvex_sha256_hex_valid(identity) ||
            state_view_measure(view, &layer_bytes, err) != YVEX_OK ||
            state_store_add(bytes, recipe_bytes, &bytes, err) != YVEX_OK ||
            state_store_add(bytes, layer_bytes, &bytes, err) != YVEX_OK)
            return state_store_fail(YVEX_ERR_STATE,
                                    "committed state view is not checkpointable",
                                    err);
    }
    scope->file_bytes = bytes;
    return YVEX_OK;
}

static int state_file_put_rolling(
    state_file_writer *writer, const yvex_attention_rolling_state_view *rolling,
    yvex_error *err)
{
    const unsigned long long fields[] = {
        (unsigned long long)rolling->present,
        (unsigned long long)rolling->schema_version,
        (unsigned long long)rolling->kind,
        (unsigned long long)rolling->attention_class,
        rolling->layer_index,
        rolling->next_token_position,
        rolling->ratio,
        rolling->head_dimension,
        rolling->state_width,
        rolling->state_slots,
        rolling->previous_fill,
        rolling->current_fill,
        rolling->cursor,
        rolling->kv_state_stride,
        rolling->score_state_stride,
        rolling->kv_state_extent,
        rolling->score_state_extent,
        (unsigned long long)rolling->overlap,
        (unsigned long long)rolling->rotated};
    unsigned int index;
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (state_file_put_u64(writer, fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_IO;
    if (rolling->present)
        return state_file_put_identity(
            writer, rolling->attention_plan_identity, err);
    {
        static const unsigned char absent_identity[64] = {0};
        return state_file_writer_write(
            writer, absent_identity, sizeof(absent_identity), 1, err);
    }
}

static int state_file_put_view(state_file_writer *writer,
                               const yvex_attention_history_view *view,
                               const char *identity, yvex_error *err)
{
    const unsigned long long fields[] = {
        view->token_count, view->local_tail_count,
        view->compressed_entry_count, view->indexer_entry_count,
        view->local_kv_stride, view->compressed_kv_stride,
        view->indexer_kv_stride};
    const void *payloads[10];
    unsigned long long blobs[10];
    unsigned int index;
    if (state_view_blob_bytes(view, blobs, err) != YVEX_OK ||
        state_view_payload_valid(view, blobs, err) != YVEX_OK ||
        state_file_put_identity(writer, identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_STATE;
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (state_file_put_u64(writer, fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_IO;
    if (state_file_put_rolling(writer, &view->main_rolling_state, err) != YVEX_OK ||
        state_file_put_rolling(writer, &view->indexer_rolling_state, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_IO;
    payloads[0] = view->local_kv;
    payloads[1] = view->local_positions;
    payloads[2] = view->compressed_kv;
    payloads[3] = view->compressed_positions;
    payloads[4] = view->indexer_kv;
    payloads[5] = view->indexer_positions;
    payloads[6] = view->main_rolling_state.kv_state;
    payloads[7] = view->main_rolling_state.score_state;
    payloads[8] = view->indexer_rolling_state.kv_state;
    payloads[9] = view->indexer_rolling_state.score_state;
    for (index = 0u; index < 10u; ++index)
        if (state_file_put_blob(writer, payloads[index], blobs[index], err) !=
            YVEX_OK)
            return err ? err->code : YVEX_ERR_IO;
    return YVEX_OK;
}

static int state_file_put_capacity(
    state_file_writer *writer, const yvex_execution_capacity_plan *capacity,
    yvex_error *err)
{
    const unsigned long long fields[] = {
        capacity->schema_version, capacity->model_maximum_context,
        capacity->admitted_execution_maximum, capacity->per_session_maximum,
        capacity->per_request_maximum, capacity->total_logical_context_tokens,
        capacity->physical_state_pool_tokens, capacity->candidate_reserve_tokens,
        capacity->concurrent_sequences, capacity->logical_batch_tokens,
        capacity->attention_microbatch_rows, capacity->moe_row_tile,
        capacity->output_head_rows, capacity->model_bytes,
        capacity->derived_layout_bytes, capacity->state_pool_bytes,
        capacity->candidate_reserve_bytes, capacity->workspace_bytes,
        capacity->scheduler_bytes, capacity->graph_bytes,
        capacity->prefix_cache_bytes, capacity->persistent_state_bytes,
        capacity->system_reserve_bytes, capacity->required_bytes,
        capacity->usable_memory_bytes, capacity->unreserved_bytes,
        capacity->state_class_count};
    unsigned long long index;
    if (sizeof(fields) / sizeof(fields[0]) != STATE_FILE_CAPACITY_FIELDS)
        return state_store_fail(YVEX_ERR_STATE,
                                "capacity field inventory diverged", err);
    for (index = 0ull; index < STATE_FILE_CAPACITY_FIELDS; ++index)
        if (state_file_put_u64(writer, fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_IO;
    if (state_file_put_identity(writer, capacity->model_execution_identity, err) !=
            YVEX_OK ||
        state_file_put_identity(writer, capacity->hardware_profile_identity, err) !=
            YVEX_OK ||
        state_file_put_identity(writer, capacity->workload_profile_identity, err) !=
            YVEX_OK ||
        state_file_put_identity(writer, capacity->identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_IO;
    for (index = 0ull; index < capacity->state_class_count; ++index) {
        const yvex_execution_state_class_plan *state =
            &capacity->state_classes[index];
        const unsigned long long state_fields[] = {
            state->state_class, state->extent, state->logical_block_tokens,
            state->bytes_per_block, state->page_tokens, state->page_bytes,
            state->tokens_per_sequence, state->pool_tokens, state->pool_bytes,
            state->page_count, state->page_table_bytes,
            state->tail_fragmentation_bytes, state->copy_on_write_tail_bytes,
            state->promotion_fragmentation_bytes,
            (unsigned long long)state->shared,
            (unsigned long long)state->copy_on_write};
        unsigned int field;
        for (field = 0u; field < sizeof(state_fields) / sizeof(state_fields[0]);
             ++field)
            if (state_file_put_u64(writer, state_fields[field], err) != YVEX_OK)
                return err ? err->code : YVEX_ERR_IO;
    }
    return YVEX_OK;
}

static int state_file_put_recipe(
    state_file_writer *writer, const yvex_attention_state_recipe *recipe,
    yvex_error *err)
{
    const unsigned long long fields[] = {
        recipe->schema_version, recipe->layer_index, recipe->selection_key,
        recipe->initial_position, recipe->final_position,
        recipe->component_count};
    unsigned long long index;
    for (index = 0ull; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (state_file_put_u64(writer, fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_IO;
    if (state_file_put_identity(writer, recipe->attention_plan_identity, err) !=
            YVEX_OK ||
        state_file_put_identity(writer, recipe->identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_IO;
    for (index = 0ull; index < recipe->component_count; ++index) {
        const yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        const yvex_attention_rolling_state_view *rolling = &component->rolling;
        const unsigned long long component_fields[] = {
            component->schema_version, component->ordinal, component->kind,
            component->binding, component->capacity, component->value_width,
            (unsigned long long)rolling->present, rolling->schema_version,
            rolling->kind, rolling->attention_class, rolling->layer_index,
            rolling->next_token_position, rolling->ratio,
            rolling->head_dimension, rolling->state_width, rolling->state_slots,
            rolling->previous_fill, rolling->current_fill, rolling->cursor,
            rolling->kv_state_stride, rolling->score_state_stride,
            rolling->kv_state_extent, rolling->score_state_extent,
            (unsigned long long)rolling->overlap,
            (unsigned long long)rolling->rotated};
        unsigned int field;
        for (field = 0u;
             field < sizeof(component_fields) / sizeof(component_fields[0]);
             ++field)
            if (state_file_put_u64(writer, component_fields[field], err) !=
                YVEX_OK)
                return err ? err->code : YVEX_ERR_IO;
        if (state_file_put_optional_identity(
                writer, rolling->attention_plan_identity, err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_IO;
    }
    return YVEX_OK;
}

static int state_file_put_scope(state_file_writer *writer,
                                const state_save_scope *scope,
                                yvex_error *err)
{
    unsigned long long layer;
    if (state_file_put_u64(writer, scope->scope, err) != YVEX_OK ||
        state_file_put_u64(writer, scope->summary.layer_count, err) != YVEX_OK ||
        state_file_put_u64(writer,
                           scope->summary.committed_sequence_length,
                           err) != YVEX_OK ||
        state_file_put_identity(writer,
                                scope->summary.state_layout_identity,
                                err) != YVEX_OK ||
        state_file_put_identity(writer,
                                scope->summary.state_content_identity,
                                err) != YVEX_OK ||
        state_file_put_optional_identity(
            writer, scope->summary.capacity_plan_identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_IO;
    if (state_file_put_capacity(writer, scope->capacity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_IO;
    for (layer = 0ull; layer < scope->summary.layer_count; ++layer) {
        const yvex_attention_state_recipe *recipe =
            scope->provider->recipe(scope->provider->context, layer);
        const yvex_attention_history_view *view = scope->provider->view(
            scope->provider->context, layer,
            YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        char identity[YVEX_SHA256_HEX_CAP];
        if (!recipe || !view ||
            scope->provider->identity(scope->provider->context, layer,
                                      identity, err) != YVEX_OK ||
            state_file_put_recipe(writer, recipe, err) != YVEX_OK ||
            state_file_put_view(writer, view, identity, err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_STATE;
    }
    return YVEX_OK;
}

static int state_store_payload_identity(
    const void *payload, unsigned long long payload_bytes,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!output || payload_bytes > SIZE_MAX || (!payload && payload_bytes))
        return 0;
    output[0] = '\0';
    if (!payload_bytes) return 1;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash,
                                 "yvex.runtime.state-store.payload.v1") ||
        !yvex_sha256_update_u64(&hash, payload_bytes) ||
        !yvex_sha256_update(&hash, payload, (size_t)payload_bytes) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int state_file_header_write(
    state_file_writer *writer, unsigned long long file_bytes,
    unsigned long long scope_count, const yvex_runtime_model_summary *model,
    unsigned long long payload_bytes, const char *payload_identity,
    yvex_error *err)
{
    return state_file_writer_write(writer, STATE_FILE_MAGIC,
                                   STATE_FILE_MAGIC_BYTES, 1, err) != YVEX_OK ||
                   state_file_put_u64(writer,
                                      YVEX_RUNTIME_STATE_STORE_SCHEMA_V2,
                                      err) != YVEX_OK ||
                   state_file_put_u64(writer, STATE_FILE_ENDIAN_MARKER,
                                      err) != YVEX_OK ||
                   state_file_put_u64(writer, STATE_FILE_F32_MARKER,
                                      err) != YVEX_OK ||
                   state_file_put_u64(writer, file_bytes, err) != YVEX_OK ||
                   state_file_put_u64(writer, scope_count, err) != YVEX_OK ||
                   state_file_put_identity(
                       writer, model->runtime_model_identity, err) != YVEX_OK ||
                   state_file_put_identity(
                       writer, model->runtime_binding_identity, err) != YVEX_OK ||
                   state_file_put_identity(writer, model->artifact_identity,
                                           err) != YVEX_OK ||
                   state_file_put_u64(writer, payload_bytes, err) != YVEX_OK ||
                   state_file_put_optional_identity(
                       writer, payload_identity, err) != YVEX_OK
               ? (err ? err->code : YVEX_ERR_IO)
               : YVEX_OK;
}

static void state_store_summary_set(
    yvex_runtime_state_store_summary *summary,
    unsigned int schema, const yvex_runtime_model_summary *model,
    unsigned long long file_bytes,
    unsigned long long scope_count, unsigned long long position,
    unsigned long long payload_bytes, const char *payload_identity,
    const unsigned char digest[YVEX_SHA256_DIGEST_BYTES])
{
    if (!summary) return;
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = schema;
    summary->file_bytes = file_bytes;
    summary->scope_count = scope_count;
    summary->committed_sequence_length = position;
    summary->payload_bytes = payload_bytes;
    yvex_runtime_identity_copy(summary->runtime_model_identity,
                               model->runtime_model_identity);
    yvex_runtime_identity_copy(summary->runtime_binding_identity,
                               model->runtime_binding_identity);
    yvex_runtime_identity_copy(summary->artifact_identity,
                               model->artifact_identity);
    if (payload_identity && payload_identity[0])
        yvex_runtime_identity_copy(summary->payload_identity,
                                   payload_identity);
    yvex_sha256_hex(digest, summary->file_digest);
}

int yvex_runtime_session_state_save(
    yvex_runtime_execution_session *session, const char *path,
    const void *payload, unsigned long long payload_bytes,
    yvex_runtime_state_store_summary *summary, yvex_error *err)
{
    state_save_scope scopes[2];
    state_file_writer writer;
    yvex_runtime_model_summary model;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long file_bytes = STATE_FILE_HEADER_V2_BYTES;
    unsigned long long scope_count = 0ull, index, payload_aligned = 0ull;
    char payload_identity[YVEX_SHA256_HEX_CAP] = {0};
    int rc = YVEX_ERR_STATE, published = 0;
    memset(scopes, 0, sizeof(scopes));
    memset(&writer, 0, sizeof(writer));
    writer.directory_fd = writer.file_fd = -1;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!session || !path || !summary || payload_bytes > SIZE_MAX ||
        (!payload && payload_bytes) || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return state_store_fail(YVEX_ERR_INVALID_ARG,
                                "state save requires a synchronized session",
                                err);
    if (!session->summary.open || session->summary.busy || session->closing ||
        session->summary.cancelled || session->summary.invalidated ||
        !session->attention_state_provider_ready) {
        rc = state_store_fail(YVEX_ERR_STATE,
                              "state save requires an idle valid session", err);
        goto done;
    }
    model = session->model->summary;
    rc = state_save_scope_prepare(&scopes[scope_count],
                                  &session->attention_state_provider, 0ull,
                                  err);
    if (rc != YVEX_OK) goto done;
    scope_count++;
    if (session->draft_attention_state_provider_ready) {
        rc = state_save_scope_prepare(
            &scopes[scope_count], &session->draft_attention_state_provider,
            1ull, err);
        if (rc != YVEX_OK) goto done;
        scope_count++;
    }
    for (index = 0ull; index < scope_count; ++index)
        if (state_store_add(file_bytes, scopes[index].file_bytes,
                            &file_bytes, err) != YVEX_OK)
            goto done;
    if (!state_store_payload_identity(payload, payload_bytes,
                                      payload_identity) ||
        state_store_align(payload_bytes, &payload_aligned, err) != YVEX_OK ||
        state_store_add(file_bytes, payload_aligned, &file_bytes, err) != YVEX_OK) {
        rc = state_store_fail(YVEX_ERR_BOUNDS,
                              "state checkpoint extension is invalid", err);
        goto done;
    }
    if (state_store_add(file_bytes, STATE_FILE_DIGEST_BYTES, &file_bytes,
                        err) != YVEX_OK ||
        file_bytes > SIZE_MAX) {
        rc = state_store_fail(YVEX_ERR_BOUNDS,
                              "state checkpoint exceeds host file limits", err);
        goto done;
    }
    rc = state_file_writer_open(&writer, path, err);
    if (rc != YVEX_OK) goto done;
    rc = state_file_header_write(&writer, file_bytes, scope_count, &model,
                                 payload_bytes, payload_identity, err);
    for (index = 0ull; rc == YVEX_OK && index < scope_count; ++index)
        rc = state_file_put_scope(&writer, &scopes[index], err);
    if (rc == YVEX_OK)
        rc = state_file_put_blob(&writer, payload, payload_bytes, err);
    if (rc == YVEX_OK &&
        (writer.offset != file_bytes - STATE_FILE_DIGEST_BYTES ||
         !yvex_sha256_final(&writer.hash, digest)))
        rc = state_store_fail(YVEX_ERR_STATE,
                              "state checkpoint size or digest diverged", err);
    if (rc == YVEX_OK)
        rc = state_file_writer_write(&writer, digest, sizeof(digest), 0, err);
    if (rc == YVEX_OK && writer.offset != file_bytes)
        rc = state_store_fail(YVEX_ERR_STATE,
                              "state checkpoint final size diverged", err);
    if (rc == YVEX_OK) rc = state_file_writer_publish(&writer, err);
    if (rc == YVEX_OK) {
        published = 1;
        state_store_summary_set(
            summary, YVEX_RUNTIME_STATE_STORE_SCHEMA_V2, &model,
            file_bytes, scope_count,
            scopes[0].summary.committed_sequence_length, payload_bytes,
            payload_identity, digest);
        yvex_error_clear(err);
    }
done:
    state_file_writer_close(&writer, published);
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}

static int state_file_map(const char *path, unsigned long long maximum_bytes,
                          const unsigned char **data, size_t *count,
                          yvex_error *err)
{
    char name[STATE_FILE_NAME_CAP];
    struct stat before, after;
    void *mapping = MAP_FAILED;
    int directory_fd = -1, fd = -1, rc = YVEX_ERR_IO;
    *data = NULL;
    *count = 0u;
    if (!maximum_bytes || maximum_bytes > SIZE_MAX)
        return state_store_fail(YVEX_ERR_INVALID_ARG,
                                "state restore file bound is invalid", err);
    rc = state_file_parent_open(path, &directory_fd, name, err);
    if (rc != YVEX_OK) goto done;
    fd = openat(directory_fd, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0 || fstat(fd, &before) != 0 || !S_ISREG(before.st_mode)) {
        rc = state_store_fail(YVEX_ERR_IO,
                              "state checkpoint could not be opened safely",
                              err);
        goto done;
    }
    if (before.st_size <= 0 ||
        (unsigned long long)before.st_size > maximum_bytes ||
        (unsigned long long)before.st_size <
            STATE_FILE_HEADER_V1_BYTES + STATE_FILE_DIGEST_BYTES) {
        rc = state_store_fail(YVEX_ERR_BOUNDS,
                              "state checkpoint size is outside its bound", err);
        goto done;
    }
    mapping = mmap(NULL, (size_t)before.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        rc = state_store_fail(YVEX_ERR_IO,
                              "state checkpoint mapping failed", err);
        goto done;
    }
    if (fstat(fd, &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
        before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
        rc = state_store_fail(YVEX_ERR_IO,
                              "state checkpoint drifted while mapping", err);
        goto done;
    }
    *data = mapping;
    *count = (size_t)before.st_size;
    mapping = MAP_FAILED;
    rc = YVEX_OK;
done:
    if (mapping != MAP_FAILED) (void)munmap(mapping, (size_t)before.st_size);
    if (fd >= 0) (void)close(fd);
    if (directory_fd >= 0) (void)close(directory_fd);
    return rc;
}

static int state_file_parser_take(state_file_parser *parser, size_t count,
                                  const unsigned char **out, yvex_error *err)
{
    if (!parser || !out || parser->offset > parser->digest_offset ||
        count > parser->digest_offset - parser->offset) {
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state checkpoint is truncated", err);
    }
    *out = parser->data + parser->offset;
    parser->offset += count;
    return YVEX_OK;
}

static int state_file_parser_u64(state_file_parser *parser,
                                 unsigned long long *out, yvex_error *err)
{
    const unsigned char *bytes;
    if (state_file_parser_take(parser, sizeof(*out), &bytes, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    memcpy(out, bytes, sizeof(*out));
    return YVEX_OK;
}

static int state_file_parser_identity(
    state_file_parser *parser, char out[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    const unsigned char *bytes;
    if (state_file_parser_take(parser, 64u, &bytes, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    memcpy(out, bytes, 64u);
    out[64] = '\0';
    return yvex_sha256_hex_valid(out)
               ? YVEX_OK
               : state_store_fail(YVEX_ERR_FORMAT,
                                  "state checkpoint identity is malformed", err);
}


static int state_file_parser_optional_identity(
    state_file_parser *parser, char out[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    const unsigned char *bytes;
    size_t index;
    if (state_file_parser_take(parser, 64u, &bytes, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (index = 0u; index < 64u; ++index)
        if (bytes[index]) break;
    if (index == 64u) {
        out[0] = '\0';
        return YVEX_OK;
    }
    memcpy(out, bytes, 64u);
    out[64] = '\0';
    return yvex_sha256_hex_valid(out)
               ? YVEX_OK
               : state_store_fail(YVEX_ERR_FORMAT,
                                  "optional checkpoint identity is malformed", err);
}

static int state_file_parser_blob(state_file_parser *parser,
                                  unsigned long long bytes,
                                  const void **out, yvex_error *err)
{
    const unsigned char *payload, *padding;
    unsigned long long aligned;
    size_t padding_bytes, index;
    if (bytes > SIZE_MAX ||
        state_store_align(bytes, &aligned, err) != YVEX_OK ||
        aligned > SIZE_MAX ||
        state_file_parser_take(parser, (size_t)bytes, &payload, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    padding_bytes = (size_t)(aligned - bytes);
    if (state_file_parser_take(parser, padding_bytes, &padding, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (index = 0u; index < padding_bytes; ++index)
        if (padding[index] != 0u)
            return state_store_fail(YVEX_ERR_FORMAT,
                                    "state checkpoint padding is noncanonical",
                                    err);
    *out = bytes ? payload : NULL;
    return YVEX_OK;
}

static int state_file_parser_rolling(
    state_file_parser *parser, yvex_attention_rolling_state_view *rolling,
    yvex_error *err)
{
    unsigned long long fields[19];
    unsigned int index;
    memset(rolling, 0, sizeof(*rolling));
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (state_file_parser_u64(parser, &fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
    if (fields[0] > 1ull || fields[1] > UINT_MAX || fields[2] > UINT_MAX ||
        fields[3] > UINT_MAX || fields[17] > 1ull || fields[18] > 1ull)
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state checkpoint rolling metadata is invalid",
                                err);
    rolling->present = (int)fields[0];
    rolling->schema_version = (unsigned int)fields[1];
    rolling->kind = (yvex_attention_rolling_kind)fields[2];
    rolling->attention_class = (yvex_attention_class)fields[3];
    rolling->layer_index = fields[4];
    rolling->next_token_position = fields[5];
    rolling->ratio = fields[6];
    rolling->head_dimension = fields[7];
    rolling->state_width = fields[8];
    rolling->state_slots = fields[9];
    rolling->previous_fill = fields[10];
    rolling->current_fill = fields[11];
    rolling->cursor = fields[12];
    rolling->kv_state_stride = fields[13];
    rolling->score_state_stride = fields[14];
    rolling->kv_state_extent = fields[15];
    rolling->score_state_extent = fields[16];
    rolling->overlap = (int)fields[17];
    rolling->rotated = (int)fields[18];
    if (rolling->present)
        return state_file_parser_identity(
            parser, rolling->attention_plan_identity, err);
    {
        const unsigned char *identity;
        if (state_file_parser_take(parser, 64u, &identity, err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
        for (index = 0u; index < 64u; ++index)
            if (identity[index])
                return state_store_fail(
                    YVEX_ERR_FORMAT,
                    "absent rolling state has a nonempty identity", err);
        rolling->attention_plan_identity[0] = '\0';
    }
    return YVEX_OK;
}

static int state_file_parser_view(
    state_file_parser *parser, yvex_attention_history_view *view,
    char identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    unsigned long long fields[7], blobs[10];
    const void *payloads[10];
    unsigned int index;
    memset(view, 0, sizeof(*view));
    if (state_file_parser_identity(parser, identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (state_file_parser_u64(parser, &fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
    view->token_count = fields[0];
    view->local_tail_count = fields[1];
    view->compressed_entry_count = fields[2];
    view->indexer_entry_count = fields[3];
    view->local_kv_stride = fields[4];
    view->compressed_kv_stride = fields[5];
    view->indexer_kv_stride = fields[6];
    if (state_file_parser_rolling(parser, &view->main_rolling_state, err) !=
            YVEX_OK ||
        state_file_parser_rolling(parser, &view->indexer_rolling_state, err) !=
            YVEX_OK ||
        state_view_blob_bytes(view, blobs, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (index = 0u; index < 10u; ++index)
        if (state_file_parser_blob(parser, blobs[index], &payloads[index], err) !=
            YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
    view->local_kv = payloads[0];
    view->local_positions = payloads[1];
    view->compressed_kv = payloads[2];
    view->compressed_positions = payloads[3];
    view->indexer_kv = payloads[4];
    view->indexer_positions = payloads[5];
    view->main_rolling_state.kv_state = payloads[6];
    view->main_rolling_state.score_state = payloads[7];
    view->indexer_rolling_state.kv_state = payloads[8];
    view->indexer_rolling_state.score_state = payloads[9];
    view->immutable = 1;
    return YVEX_OK;
}

static int state_file_parser_capacity(
    state_file_parser *parser, yvex_execution_capacity_plan *capacity,
    yvex_error *err)
{
    unsigned long long fields[STATE_FILE_CAPACITY_FIELDS];
    unsigned long long index;
    memset(capacity, 0, sizeof(*capacity));
    for (index = 0ull; index < STATE_FILE_CAPACITY_FIELDS; ++index)
        if (state_file_parser_u64(parser, &fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
    if (fields[0] > UINT_MAX || fields[26] > YVEX_MODEL_STATE_CLASS_COUNT)
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state capacity field is invalid", err);
    capacity->schema_version = (unsigned int)fields[0];
    capacity->model_maximum_context = fields[1];
    capacity->admitted_execution_maximum = fields[2];
    capacity->per_session_maximum = fields[3];
    capacity->per_request_maximum = fields[4];
    capacity->total_logical_context_tokens = fields[5];
    capacity->physical_state_pool_tokens = fields[6];
    capacity->candidate_reserve_tokens = fields[7];
    capacity->concurrent_sequences = fields[8];
    capacity->logical_batch_tokens = fields[9];
    capacity->attention_microbatch_rows = fields[10];
    capacity->moe_row_tile = fields[11];
    capacity->output_head_rows = fields[12];
    capacity->model_bytes = fields[13];
    capacity->derived_layout_bytes = fields[14];
    capacity->state_pool_bytes = fields[15];
    capacity->candidate_reserve_bytes = fields[16];
    capacity->workspace_bytes = fields[17];
    capacity->scheduler_bytes = fields[18];
    capacity->graph_bytes = fields[19];
    capacity->prefix_cache_bytes = fields[20];
    capacity->persistent_state_bytes = fields[21];
    capacity->system_reserve_bytes = fields[22];
    capacity->required_bytes = fields[23];
    capacity->usable_memory_bytes = fields[24];
    capacity->unreserved_bytes = fields[25];
    capacity->state_class_count = fields[26];
    if (state_file_parser_identity(
            parser, capacity->model_execution_identity, err) != YVEX_OK ||
        state_file_parser_identity(
            parser, capacity->hardware_profile_identity, err) != YVEX_OK ||
        state_file_parser_identity(
            parser, capacity->workload_profile_identity, err) != YVEX_OK ||
        state_file_parser_identity(parser, capacity->identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (index = 0ull; index < capacity->state_class_count; ++index) {
        yvex_execution_state_class_plan *state = &capacity->state_classes[index];
        unsigned long long values[STATE_FILE_CAPACITY_CLASS_FIELDS];
        unsigned int field;
        for (field = 0u; field < STATE_FILE_CAPACITY_CLASS_FIELDS; ++field)
            if (state_file_parser_u64(parser, &values[field], err) != YVEX_OK)
                return err ? err->code : YVEX_ERR_FORMAT;
        if (values[0] > UINT_MAX || values[1] > UINT_MAX ||
            values[14] > 1ull || values[15] > 1ull)
            return state_store_fail(YVEX_ERR_FORMAT,
                                    "state capacity class is invalid", err);
        state->state_class = (yvex_model_state_class)values[0];
        state->extent = (yvex_execution_state_extent)values[1];
        state->logical_block_tokens = values[2];
        state->bytes_per_block = values[3];
        state->page_tokens = values[4];
        state->page_bytes = values[5];
        state->tokens_per_sequence = values[6];
        state->pool_tokens = values[7];
        state->pool_bytes = values[8];
        state->page_count = values[9];
        state->page_table_bytes = values[10];
        state->tail_fragmentation_bytes = values[11];
        state->copy_on_write_tail_bytes = values[12];
        state->promotion_fragmentation_bytes = values[13];
        state->shared = (int)values[14];
        state->copy_on_write = (int)values[15];
    }
    if (yvex_execution_capacity_plan_validate(capacity, err) != YVEX_OK)
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state capacity plan failed validation", err);
    return YVEX_OK;
}

static int state_file_parser_recipe_component(
    state_file_parser *parser, yvex_attention_state_component_recipe *component,
    yvex_error *err)
{
    unsigned long long fields[25];
    unsigned int index;
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (state_file_parser_u64(parser, &fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
    if (fields[0] > UINT_MAX || fields[1] > UINT_MAX || fields[2] > UINT_MAX ||
        fields[3] > UINT_MAX || fields[6] > 1ull || fields[7] > UINT_MAX ||
        fields[8] > UINT_MAX || fields[9] > UINT_MAX || fields[23] > 1ull ||
        fields[24] > 1ull)
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state recipe component is invalid", err);
    component->schema_version = (unsigned int)fields[0];
    component->ordinal = (unsigned int)fields[1];
    component->kind = (yvex_attention_state_component_kind)fields[2];
    component->binding = (yvex_attention_state_binding)fields[3];
    component->capacity = fields[4];
    component->value_width = fields[5];
    component->rolling.present = (int)fields[6];
    component->rolling.schema_version = (unsigned int)fields[7];
    component->rolling.kind = (yvex_attention_rolling_kind)fields[8];
    component->rolling.attention_class = (yvex_attention_class)fields[9];
    component->rolling.layer_index = fields[10];
    component->rolling.next_token_position = fields[11];
    component->rolling.ratio = fields[12];
    component->rolling.head_dimension = fields[13];
    component->rolling.state_width = fields[14];
    component->rolling.state_slots = fields[15];
    component->rolling.previous_fill = fields[16];
    component->rolling.current_fill = fields[17];
    component->rolling.cursor = fields[18];
    component->rolling.kv_state_stride = fields[19];
    component->rolling.score_state_stride = fields[20];
    component->rolling.kv_state_extent = fields[21];
    component->rolling.score_state_extent = fields[22];
    component->rolling.overlap = (int)fields[23];
    component->rolling.rotated = (int)fields[24];
    return state_file_parser_optional_identity(
        parser, component->rolling.attention_plan_identity, err);
}

static int state_file_parser_recipe(
    state_file_parser *parser, yvex_attention_state_recipe *recipe,
    yvex_error *err)
{
    unsigned long long fields[6], index;
    memset(recipe, 0, sizeof(*recipe));
    for (index = 0ull; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (state_file_parser_u64(parser, &fields[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
    if (fields[0] > UINT_MAX ||
        fields[5] > YVEX_ATTENTION_STATE_COMPONENT_CAP)
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state recipe header is invalid", err);
    recipe->schema_version = (unsigned int)fields[0];
    recipe->layer_index = fields[1];
    recipe->selection_key = fields[2];
    recipe->initial_position = fields[3];
    recipe->final_position = fields[4];
    recipe->component_count = (unsigned int)fields[5];
    if (state_file_parser_identity(
            parser, recipe->attention_plan_identity, err) != YVEX_OK ||
        state_file_parser_identity(parser, recipe->identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (index = 0ull; index < recipe->component_count; ++index)
        if (state_file_parser_recipe_component(
                parser, &recipe->components[index], err) != YVEX_OK)
            return err ? err->code : YVEX_ERR_FORMAT;
    return yvex_attention_state_recipe_seal(recipe, err);
}

static void state_restore_scope_close(state_restore_scope *scope)
{
    if (!scope) return;
    free(scope->recipes);
    free(scope->layers);
    free(scope->identities);
    yvex_graph_attention_capacity_plan_close(&scope->graph_capacity);
    memset(scope, 0, sizeof(*scope));
}

static int state_file_parser_scope(
    state_file_parser *parser, state_restore_scope *scope,
    const yvex_attention_state_provider *provider,
    unsigned long long schema, unsigned long long expected_scope,
    yvex_error *err)
{
    yvex_graph_attention_state_summary current;
    unsigned long long layer, layer_count = 0ull, position = 0ull;
    if (!provider || !provider->summary || !provider->restore ||
        provider->summary(provider->context, &current, err) != YVEX_OK ||
        state_file_parser_u64(parser, &scope->scope, err) != YVEX_OK ||
        state_file_parser_u64(parser, &layer_count, err) != YVEX_OK ||
        state_file_parser_u64(parser, &position, err) != YVEX_OK ||
        scope->scope != expected_scope || layer_count != current.layer_count ||
        (schema != YVEX_RUNTIME_STATE_STORE_SCHEMA_V1 &&
         schema != YVEX_RUNTIME_STATE_STORE_SCHEMA_V2) ||
        provider->schema_version != YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V8 ||
        !provider->configure_pages || !provider->prepare ||
        layer_count > SIZE_MAX / sizeof(*scope->layers) ||
        layer_count > SIZE_MAX / sizeof(*scope->recipes) ||
        layer_count > SIZE_MAX / sizeof(*scope->identities))
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state checkpoint scope is incompatible", err);
    scope->layers = calloc((size_t)layer_count, sizeof(*scope->layers));
    scope->recipes = calloc((size_t)layer_count, sizeof(*scope->recipes));
    scope->identities = calloc((size_t)layer_count, sizeof(*scope->identities));
    if ((layer_count && !scope->layers) || (layer_count && !scope->recipes) ||
        (layer_count && !scope->identities))
        return state_store_fail(YVEX_ERR_NOMEM,
                                "state checkpoint scope allocation failed", err);
    scope->checkpoint.schema_version =
        YVEX_ATTENTION_STATE_CHECKPOINT_SCHEMA_V1;
    scope->checkpoint.layer_count = layer_count;
    scope->checkpoint.committed_sequence_length = position;
    scope->checkpoint.capacity =
        schema == YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 ? &scope->capacity : NULL;
    scope->checkpoint.recipes =
        schema == YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 ? scope->recipes : NULL;
    scope->checkpoint.layers = scope->layers;
    scope->checkpoint.layer_identities =
        (const char (*)[YVEX_SHA256_HEX_CAP])scope->identities;
    if (state_file_parser_identity(
            parser, scope->checkpoint.state_layout_identity, err) != YVEX_OK ||
        state_file_parser_identity(
            parser, scope->checkpoint.state_content_identity, err) != YVEX_OK ||
        state_file_parser_optional_identity(
            parser, scope->checkpoint.capacity_plan_identity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    if (schema == YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 &&
        state_file_parser_capacity(parser, &scope->capacity, err) != YVEX_OK)
        return err ? err->code : YVEX_ERR_FORMAT;
    for (layer = 0ull; layer < layer_count; ++layer)
        if ((schema == YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 &&
             state_file_parser_recipe(
                 parser, &scope->recipes[layer], err) != YVEX_OK) ||
            state_file_parser_view(parser, &scope->layers[layer],
                                   scope->identities[layer], err) != YVEX_OK ||
            scope->layers[layer].token_count != position)
            return state_store_fail(YVEX_ERR_FORMAT,
                                    "state checkpoint layer position diverged",
                                    err);
    return yvex_attention_state_checkpoint_validate(
        &scope->checkpoint, current.prepared_layer_count == current.layer_count ? &current : NULL, err);
}

static int state_file_parser_header(
    state_file_parser *parser, yvex_runtime_model_summary *model,
    unsigned long long *schema, unsigned long long *scope_count,
    unsigned long long *payload_bytes,
    char payload_identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    const unsigned char *magic;
    unsigned long long endian = 0ull, f32 = 0ull;
    unsigned long long file_bytes = 0ull;
    if (schema) *schema = 0ull;
    if (payload_bytes) *payload_bytes = 0ull;
    if (payload_identity) payload_identity[0] = '\0';
    if (!parser || !model || !schema || !scope_count || !payload_bytes ||
        !payload_identity)
        return state_store_fail(YVEX_ERR_INVALID_ARG,
                                "state checkpoint header output is invalid", err);
    if (state_file_parser_take(parser, STATE_FILE_MAGIC_BYTES, &magic, err) !=
            YVEX_OK ||
        memcmp(magic, STATE_FILE_MAGIC, STATE_FILE_MAGIC_BYTES) != 0 ||
        state_file_parser_u64(parser, schema, err) != YVEX_OK ||
        state_file_parser_u64(parser, &endian, err) != YVEX_OK ||
        state_file_parser_u64(parser, &f32, err) != YVEX_OK ||
        state_file_parser_u64(parser, &file_bytes, err) != YVEX_OK ||
        state_file_parser_u64(parser, scope_count, err) != YVEX_OK ||
        (*schema != YVEX_RUNTIME_STATE_STORE_SCHEMA_V1 &&
         *schema != YVEX_RUNTIME_STATE_STORE_SCHEMA_V2) ||
        endian != STATE_FILE_ENDIAN_MARKER || f32 != STATE_FILE_F32_MARKER ||
        file_bytes != parser->count || (*scope_count != 1ull && *scope_count != 2ull) ||
        state_file_parser_identity(parser, model->runtime_model_identity, err) !=
            YVEX_OK ||
        state_file_parser_identity(parser, model->runtime_binding_identity,
                                   err) != YVEX_OK ||
        state_file_parser_identity(parser, model->artifact_identity, err) !=
            YVEX_OK ||
        (*schema == YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 &&
         (state_file_parser_u64(parser, payload_bytes, err) != YVEX_OK ||
          state_file_parser_optional_identity(
              parser, payload_identity, err) != YVEX_OK ||
          ((*payload_bytes != 0ull) != (payload_identity[0] != '\0')))))
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state checkpoint header is invalid", err);
    return YVEX_OK;
}

static int state_file_digest_validate(
    const unsigned char *data, size_t count,
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES], yvex_error *err)
{
    yvex_sha256 hash;
    if (count < STATE_FILE_DIGEST_BYTES)
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state checkpoint digest is truncated", err);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, data, count - STATE_FILE_DIGEST_BYTES) ||
        !yvex_sha256_final(&hash, digest) ||
        memcmp(digest, data + count - STATE_FILE_DIGEST_BYTES,
               STATE_FILE_DIGEST_BYTES) != 0)
        return state_store_fail(YVEX_ERR_FORMAT,
                                "state checkpoint digest is invalid", err);
    return YVEX_OK;
}

static void state_restore_file_close(state_restore_file *file)
{
    unsigned long long index;
    if (!file) return;
    for (index = 0ull; index < 2ull; ++index)
        state_restore_scope_close(&file->scopes[index]);
    if (file->mapping)
        (void)munmap((void *)file->mapping, file->mapping_count);
    memset(file, 0, sizeof(*file));
}

static int state_restore_file_parse(
    yvex_runtime_execution_session *session, const char *path,
    unsigned long long maximum_file_bytes, state_restore_file *file,
    yvex_error *err)
{
    state_file_parser parser = {0};
    yvex_runtime_model_summary current_model;
    const void *payload = NULL;
    char identity[YVEX_SHA256_HEX_CAP];
    int rc;
    memset(file, 0, sizeof(*file));
    current_model = session->model->summary;
    rc = state_file_map(path, maximum_file_bytes, &file->mapping,
                        &file->mapping_count, err);
    if (rc != YVEX_OK) goto failure;
    rc = state_file_digest_validate(file->mapping, file->mapping_count,
                                    file->digest, err);
    if (rc != YVEX_OK) goto failure;
    parser.data = file->mapping;
    parser.count = file->mapping_count;
    parser.digest_offset = file->mapping_count - STATE_FILE_DIGEST_BYTES;
    rc = state_file_parser_header(
        &parser, &file->model, &file->schema, &file->scope_count,
        &file->payload_bytes, file->payload_identity, err);
    if (rc != YVEX_OK ||
        strcmp(file->model.runtime_model_identity,
               current_model.runtime_model_identity) != 0 ||
        strcmp(file->model.runtime_binding_identity,
               current_model.runtime_binding_identity) != 0 ||
        strcmp(file->model.artifact_identity,
               current_model.artifact_identity) != 0 ||
        file->scope_count !=
            (session->draft_attention_state_provider_ready ? 2ull : 1ull)) {
        rc = state_store_fail(YVEX_ERR_FORMAT,
                              "state checkpoint model or scope identity mismatched",
                              err);
        goto failure;
    }
    rc = state_file_parser_scope(
        &parser, &file->scopes[0], &session->attention_state_provider,
        file->schema, 0ull, err);
    if (rc == YVEX_OK && file->scope_count == 2ull)
        rc = state_file_parser_scope(
            &parser, &file->scopes[1],
            &session->draft_attention_state_provider,
            file->schema, 1ull, err);
    if (rc == YVEX_OK && file->schema == YVEX_RUNTIME_STATE_STORE_SCHEMA_V2)
        rc = state_file_parser_blob(&parser, file->payload_bytes, &payload, err);
    if (rc == YVEX_OK && file->payload_bytes &&
        (!state_store_payload_identity(payload, file->payload_bytes, identity) ||
         strcmp(identity, file->payload_identity) != 0))
        rc = state_store_fail(YVEX_ERR_FORMAT,
                              "state checkpoint extension identity is invalid", err);
    if (rc == YVEX_OK && parser.offset != parser.digest_offset)
        rc = state_store_fail(YVEX_ERR_FORMAT,
                              "state checkpoint has trailing records", err);
    if (rc != YVEX_OK) goto failure;
    file->payload = payload;
    return YVEX_OK;
failure:
    state_restore_file_close(file);
    return rc;
}

static int state_restore_scope_prepare(
    yvex_runtime_execution_session *session, state_restore_scope *scope,
    yvex_tensor_scope tensor_scope, yvex_error *err)
{
    const yvex_attention_plan *attention =
        tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
            ? session->model->draft_attention
            : session->model->attention;
    yvex_attention_state_provider *provider =
        tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
            ? &session->draft_attention_state_provider
            : &session->attention_state_provider;
    yvex_graph_attention_capacity_request request = {0};
    yvex_graph_attention_state_summary current;
    yvex_attention_failure failure = {0};
    unsigned long long layer;
    int rc;
    if (!scope->checkpoint.capacity || !scope->checkpoint.recipes)
        return provider->summary(provider->context, &current, err) == YVEX_OK &&
                       current.prepared_layer_count == current.layer_count
                   ? YVEX_OK
                   : state_store_fail(
                         YVEX_ERR_STATE,
                         "legacy state restore requires an already prepared provider",
                         err);
    if (!attention ||
        strcmp(scope->capacity.model_execution_identity,
               session->model->binding_summary.model_execution_identity) != 0)
        return state_store_fail(
            YVEX_ERR_FORMAT,
            "state capacity does not belong to the runtime model", err);
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    request.history_tokens = request.start_position =
        scope->recipes[0].initial_position;
    request.token_count = scope->recipes[0].final_position -
                          scope->recipes[0].initial_position;
    request.execution_count = 1ull;
    request.use_requested_position = 1;
    rc = yvex_graph_attention_capacity_plan_build(
        &scope->graph_capacity, attention, &request, err);
    for (layer = 0ull; rc == YVEX_OK &&
                        layer < scope->checkpoint.layer_count; ++layer) {
        const yvex_graph_attention_capacity_layer *compiled =
            yvex_graph_attention_capacity_plan_layer(
                scope->graph_capacity, layer);
        if (!compiled || !compiled->selected ||
            strcmp(compiled->recipe.identity,
                   scope->recipes[layer].identity) != 0)
            rc = state_store_fail(
                YVEX_ERR_FORMAT,
                "persisted state recipe differs from compiled model geometry",
                err);
    }
    if (rc == YVEX_OK)
        rc = provider->summary(provider->context, &current, err);
    if (rc == YVEX_OK && !current.paging_configured)
        rc = provider->configure_pages(
            provider->context, &scope->capacity, &failure, err);
    else if (rc == YVEX_OK &&
             strcmp(current.capacity_plan_identity,
                    scope->capacity.identity) != 0)
        rc = state_store_fail(
            YVEX_ERR_FORMAT,
            "persisted state capacity differs from the live provider", err);
    if (rc == YVEX_OK)
        rc = provider->summary(provider->context, &current, err);
    if (rc == YVEX_OK && !current.prepared_layer_count)
        for (layer = 0ull; rc == YVEX_OK &&
                            layer < scope->checkpoint.layer_count; ++layer)
            rc = provider->prepare(
                provider->context, layer, &scope->recipes[layer], NULL,
                &failure, err);
    if (rc == YVEX_OK)
        rc = provider->summary(provider->context, &current, err);
    if (rc == YVEX_OK &&
        (current.prepared_layer_count != current.layer_count ||
         strcmp(current.state_layout_identity,
                scope->checkpoint.state_layout_identity) != 0 ||
         strcmp(current.capacity_plan_identity,
                scope->checkpoint.capacity_plan_identity) != 0))
        rc = state_store_fail(
            YVEX_ERR_FORMAT,
            "reconstructed provider layout differs from its checkpoint", err);
    return rc;
}

static int state_restore_residencies_close(
    yvex_runtime_execution_session *session, yvex_error *err)
{
    int rc = yvex_runtime_state_residency_close(
        &session->state_residency, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_state_residency_close(
            &session->draft_state_residency, err);
    session->view.state_residency = session->state_residency;
    session->view.draft_state_residency = session->draft_state_residency;
    if (rc == YVEX_OK && session->state_resolver_attached) {
        yvex_backend_state_residency_detach(session->backend);
        session->state_resolver_attached = 0;
    }
    return rc;
}

static int state_restore_publish(
    yvex_runtime_execution_session *session, state_restore_scope scopes[2],
    unsigned long long scope_count, yvex_error *err)
{
    yvex_attention_failure failure = {0};
    yvex_runtime_model_failure model_failure = {0};
    yvex_error cleanup;
    int rc;
    rc = state_restore_scope_prepare(
        session, &scopes[0], YVEX_TENSOR_SCOPE_GLOBAL, err);
    if (rc == YVEX_OK && scope_count == 2ull)
        rc = state_restore_scope_prepare(
            session, &scopes[1], YVEX_TENSOR_SCOPE_DRAFT, err);
    if (rc == YVEX_OK && scopes[0].checkpoint.capacity)
        rc = state_restore_residencies_close(session, err);
    if (rc == YVEX_OK)
        rc = session->attention_state_provider.restore(
        session->attention_state_provider.context, &scopes[0].checkpoint,
        &failure, err);
    if (rc == YVEX_OK && scope_count == 2ull)
        rc = session->draft_attention_state_provider.restore(
            session->draft_attention_state_provider.context,
            &scopes[1].checkpoint, &failure, err);
    if (rc == YVEX_OK && scopes[0].checkpoint.capacity)
        rc = yvex_runtime_private_session_prepare_persistent_scope_state_locked(
            session, YVEX_TENSOR_SCOPE_GLOBAL, scopes[0].graph_capacity,
            &model_failure, err);
    if (rc == YVEX_OK && scope_count == 2ull &&
        scopes[0].checkpoint.capacity)
        rc = yvex_runtime_private_session_prepare_persistent_scope_state_locked(
            session, YVEX_TENSOR_SCOPE_DRAFT, scopes[1].graph_capacity,
            &model_failure, err);
    if (rc == YVEX_OK && !scopes[0].checkpoint.capacity &&
        session->state_residency)
        rc = yvex_runtime_state_residency_reset(session->state_residency, err);
    if (rc == YVEX_OK && !scopes[0].checkpoint.capacity &&
        session->draft_state_residency)
        rc = yvex_runtime_state_residency_reset(
            session->draft_state_residency, err);
    if (rc != YVEX_OK) {
        session->summary.invalidated = 1;
        yvex_error_clear(&cleanup);
        (void)yvex_runtime_private_session_invalidate(session, 1, &cleanup);
    }
    return rc;
}

void yvex_runtime_state_store_payload_close(
    yvex_runtime_state_store_payload *payload)
{
    if (!payload) return;
    free(payload->bytes);
    memset(payload, 0, sizeof(*payload));
}

int yvex_runtime_session_state_inspect(
    yvex_runtime_execution_session *session, const char *path,
    unsigned long long maximum_file_bytes,
    yvex_runtime_state_store_payload *payload,
    yvex_runtime_state_store_summary *summary, yvex_error *err)
{
    state_restore_file file;
    int rc;
    if (payload) memset(payload, 0, sizeof(*payload));
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!session || !path || !payload || !summary ||
        !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return state_store_fail(YVEX_ERR_INVALID_ARG,
                                "state inspect requires a synchronized session",
                                err);
    if (!session->summary.open || session->summary.busy || session->closing ||
        session->summary.cancelled || session->summary.invalidated ||
        !session->attention_state_provider_ready) {
        rc = state_store_fail(YVEX_ERR_STATE,
                              "state inspect requires an idle valid session", err);
        goto done;
    }
    rc = state_restore_file_parse(session, path, maximum_file_bytes, &file, err);
    if (rc != YVEX_OK) goto done;
    if (file.payload_bytes) {
        payload->bytes = malloc((size_t)file.payload_bytes);
        if (!payload->bytes) {
            rc = state_store_fail(YVEX_ERR_NOMEM,
                                  "state checkpoint extension allocation failed",
                                  err);
            goto parsed;
        }
        memcpy(payload->bytes, file.payload, (size_t)file.payload_bytes);
        payload->byte_count = file.payload_bytes;
        yvex_runtime_identity_copy(payload->payload_identity,
                                   file.payload_identity);
    }
    state_store_summary_set(
        summary, (unsigned int)file.schema, &file.model, file.mapping_count,
        file.scope_count,
        file.scopes[0].checkpoint.committed_sequence_length,
        file.payload_bytes, file.payload_identity, file.digest);
    yvex_error_clear(err);
parsed:
    state_restore_file_close(&file);
done:
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (rc != YVEX_OK) yvex_runtime_state_store_payload_close(payload);
    return rc;
}

int yvex_runtime_session_state_restore(
    yvex_runtime_execution_session *session, const char *path,
    unsigned long long maximum_file_bytes,
    unsigned long long expected_committed_sequence_length,
    const char *expected_file_digest, const char *expected_payload_identity,
    yvex_runtime_state_store_summary *summary, yvex_error *err)
{
    state_restore_file file;
    int rc;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!session || !path || !summary ||
        (expected_file_digest && expected_file_digest[0] &&
         !yvex_sha256_hex_valid(expected_file_digest)) ||
        (expected_payload_identity && expected_payload_identity[0] &&
         !yvex_sha256_hex_valid(expected_payload_identity)) ||
        !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return state_store_fail(YVEX_ERR_INVALID_ARG,
                                "state restore requires a synchronized session",
                                err);
    if (!session->summary.open || session->summary.busy || session->closing ||
        session->summary.cancelled || session->summary.invalidated ||
        !session->attention_state_provider_ready) {
        rc = state_store_fail(YVEX_ERR_STATE,
                              "state restore requires an idle valid session", err);
        goto done;
    }
    rc = state_restore_file_parse(session, path, maximum_file_bytes, &file, err);
    if (rc != YVEX_OK) goto done;
    {
        char file_digest[YVEX_SHA256_HEX_CAP];
        yvex_sha256_hex(file.digest, file_digest);
        if (file.scopes[0].checkpoint.committed_sequence_length !=
                expected_committed_sequence_length ||
            (expected_file_digest && expected_file_digest[0] &&
             strcmp(expected_file_digest, file_digest) != 0) ||
            (expected_payload_identity && expected_payload_identity[0] &&
             strcmp(expected_payload_identity, file.payload_identity) != 0))
            rc = state_store_fail(
                YVEX_ERR_STATE,
                "state checkpoint changed or mismatches its semantic session", err);
    }
    if (rc == YVEX_OK)
        rc = state_restore_publish(session, file.scopes, file.scope_count, err);
    if (rc == YVEX_OK) {
        state_store_summary_set(
            summary, (unsigned int)file.schema, &file.model,
            file.mapping_count, file.scope_count,
            file.scopes[0].checkpoint.committed_sequence_length,
            file.payload_bytes, file.payload_identity, file.digest);
        yvex_error_clear(err);
    }
    state_restore_file_close(&file);
done:
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}
