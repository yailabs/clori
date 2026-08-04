/*
 * Build the retained shard and tensor inventory from verified metadata.
 *
 * Indexed and indexless authority modes remain explicit and deterministic. Header inventory is not
 * payload trust or transform execution.
 */
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

static int inventory_refuse(yvex_error *err,
                            yvex_status status,
                            const char *where,
                            const char *message) {
    yvex_error_set(err, status, where, message);
    return status;
}

#define SOURCE_INDEX_CAP (128u * 1024u * 1024u)
#define SOURCE_UPSTREAM_INVENTORY_CAP (8u * 1024u * 1024u)

typedef struct {
    char *tensor;
    char *shard;
    int seen_in_header;
} source_index_entry;

typedef struct {
    source_index_entry *items;
    size_t count;
    size_t cap;
    unsigned long long declared_total_size;
    int has_declared_total_size;
    int duplicate_tensor;
} source_index;

typedef struct {
    char *name;
    unsigned long long size;
} source_named_size;

typedef struct {
    source_named_size *items;
    size_t count;
    size_t cap;
    unsigned int declared_total;
} source_shards;

typedef struct {
    source_named_size *items;
    size_t count;
    size_t cap;
    char repo[256];
    char revision[128];
} source_upstream_inventory;

typedef struct {
    char name[YVEX_PATH_CAP];
    unsigned long long size;
    unsigned int seen;
} source_upstream_parse_state;

struct yvex_source_tensor_snapshot {
    yvex_native_weight_table *table;
    yvex_source_shard_snapshot *shards;
    unsigned long long shard_count;
    unsigned long long header_scan_count;
    unsigned long long payload_bytes_read;
    unsigned long long identity;
    unsigned int references;
};

typedef struct {
    size_t initial_capacity;
    const char *where;
    const char *storage_failure;
    const char *name_failure;
} source_row_policy;

static const source_row_policy source_shard_row_policy = {
    64u, "source_shards", "shard name allocation failed", "shard name copy failed"};
static const source_row_policy source_upstream_row_policy = {
    64u,
    "source_upstream_inventory",
    "upstream inventory allocation failed",
    "upstream inventory name allocation failed"};

/*
 * Append one owned name/size row for source catalogs that share vector semantics.
 *
 * Overflow or allocation failure preserves the logical row count and prior rows. Vector ownership
 * does not assign shard or upstream authority semantics.
 */
static int source_named_size_append(source_named_size **items,
                                    size_t *count,
                                    size_t *capacity,
                                    const char *name,
                                    unsigned long long size,
                                    const source_row_policy *policy,
                                    yvex_error *err) {
    source_named_size *grown;
    size_t next_capacity;
    char *copy;

    if (!items || !count || !capacity || !name || !policy || policy->initial_capacity == 0u)
        return inventory_refuse(
            err, YVEX_ERR_BOUNDS, policy ? policy->where : "source_rows",
            policy ? policy->storage_failure : "source row policy is invalid");
    if (*count == *capacity) {
        next_capacity = *capacity ? *capacity * 2u : policy->initial_capacity;
        if (next_capacity < *capacity || next_capacity > SIZE_MAX / sizeof(**items))
            return inventory_refuse(
                err, YVEX_ERR_BOUNDS, policy->where, policy->storage_failure);
        grown = (source_named_size *)realloc(*items, next_capacity * sizeof(**items));
        if (!grown)
            return inventory_refuse(
                err, YVEX_ERR_NOMEM, policy->where, policy->storage_failure);
        *items = grown;
        *capacity = next_capacity;
    }
    copy = yvex_core_strdup(name);
    if (!copy)
        return inventory_refuse(err, YVEX_ERR_NOMEM, policy->where, policy->name_failure);
    (*items)[*count] = (source_named_size){.name = copy, .size = size};
    (*count)++;
    return YVEX_OK;
}

static void source_named_size_free(source_named_size *items, size_t count) {
    size_t index;

    for (index = 0u; index < count; ++index)
        free(items[index].name);
    free(items);
}

static int source_named_size_compare(const void *left, const void *right) {
    const source_named_size *a = (const source_named_size *)left;
    const source_named_size *b = (const source_named_size *)right;

    return strcmp(a->name, b->name);
}

static int source_named_size_key_compare(const void *key, const void *row) {
    return strcmp((const char *)key, ((const source_named_size *)row)->name);
}

static const size_t source_dtype_counter_offsets[YVEX_NATIVE_DTYPE_OTHER + 1u] = {
    [YVEX_NATIVE_DTYPE_F16] = offsetof(yvex_source_verification, dtype_f16_count),
    [YVEX_NATIVE_DTYPE_BF16] = offsetof(yvex_source_verification, dtype_bf16_count),
    [YVEX_NATIVE_DTYPE_F32] = offsetof(yvex_source_verification, dtype_f32_count),
    [YVEX_NATIVE_DTYPE_I64] = offsetof(yvex_source_verification, dtype_i64_count),
    [YVEX_NATIVE_DTYPE_I8] = offsetof(yvex_source_verification, dtype_i8_count),
    [YVEX_NATIVE_DTYPE_FP4] = offsetof(yvex_source_verification, dtype_fp4_count),
    [YVEX_NATIVE_DTYPE_F8_E4M3] = offsetof(yvex_source_verification, dtype_f8_count),
    [YVEX_NATIVE_DTYPE_F8_E5M2] = offsetof(yvex_source_verification, dtype_f8_count),
    [YVEX_NATIVE_DTYPE_F8_E8M0] = offsetof(yvex_source_verification, dtype_f8_e8m0_count),
};

static void source_count_dtype(yvex_source_verification *out, yvex_native_dtype dtype) {
    size_t offset = (unsigned int)dtype <= YVEX_NATIVE_DTYPE_OTHER
                        ? source_dtype_counter_offsets[(unsigned int)dtype]
                        : 0u;
    unsigned long long *counter;

    if (offset == 0u)
        offset = offsetof(yvex_source_verification, dtype_other_count);
    counter = (unsigned long long *)(void *)((unsigned char *)out + offset);
    (*counter)++;
}

static int source_snapshot_copy_shards(yvex_source_tensor_snapshot *snapshot,
                                       const yvex_source_shard_snapshot *shards,
                                       unsigned long long shard_count,
                                       yvex_error *err) {
    unsigned long long index;

    if (!shards || shard_count == 0u)
        return YVEX_OK;
    if (shard_count > (unsigned long long)(SIZE_MAX / sizeof(shards[0]))) {
        return inventory_refuse(err, YVEX_ERR_BOUNDS, "source_tensor_snapshot",
            "source shard catalog allocation overflow");
    }
    snapshot->shards =
        (yvex_source_shard_snapshot *)calloc((size_t)shard_count, sizeof(snapshot->shards[0]));
    if (!snapshot->shards) {
        return inventory_refuse(err, YVEX_ERR_NOMEM, "source_tensor_snapshot",
            "source shard catalog allocation failed");
    }
    for (index = 0u; index < shard_count; ++index) {
        char *name = yvex_core_strdup(shards[index].canonical_name);
        if (!name) {
            return inventory_refuse(err, YVEX_ERR_NOMEM, "source_tensor_snapshot",
                "source shard name allocation failed");
        }
        snapshot->shards[index] = shards[index];
        snapshot->shards[index].canonical_name = name;
    }
    return YVEX_OK;
}

static unsigned long long
source_snapshot_hash_bytes(unsigned long long hash, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    size_t i;

    for (i = 0u; i < length; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static unsigned long long source_snapshot_hash_u64(unsigned long long hash,
                                                   unsigned long long value) {
    unsigned char bytes[8];
    unsigned int i;

    for (i = 0u; i < sizeof(bytes); ++i)
        bytes[i] = (unsigned char)((value >> (i * 8u)) & 0xffu);
    return source_snapshot_hash_bytes(hash, bytes, sizeof(bytes));
}

static unsigned long long source_snapshot_identity(const yvex_native_weight_table *table) {
    unsigned long long hash = 1469598103934665603ull;
    unsigned long long i;

    for (i = 0u; table && i < table->count; ++i) {
        const yvex_native_weight_info *item = &table->items[i];
        hash = source_snapshot_hash_bytes(hash, item->name, strlen(item->name) + 1u);
        hash = source_snapshot_hash_bytes(hash, item->shard_path, strlen(item->shard_path) + 1u);
        hash = source_snapshot_hash_bytes(hash,
                                          yvex_native_dtype_name(item->dtype),
                                          strlen(yvex_native_dtype_name(item->dtype)) + 1u);
        hash = source_snapshot_hash_u64(hash, item->rank);
        {
            unsigned int dimension;
            for (dimension = 0u; dimension < item->rank; ++dimension)
                hash = source_snapshot_hash_u64(hash, item->dims[dimension]);
        }
        hash = source_snapshot_hash_u64(hash, item->data_start);
        hash = source_snapshot_hash_u64(hash, item->data_end);
    }
    return hash;
}

int yvex_source_tensor_snapshot_take_table(yvex_source_tensor_snapshot **out,
                                           yvex_native_weight_table **table,
                                           unsigned long long shard_count,
                                           unsigned long long header_scan_count,
                                           yvex_error *err) {
    yvex_source_tensor_snapshot *snapshot;
    int rc;

    if (!out || !table || !*table) {
        return inventory_refuse(err, YVEX_ERR_INVALID_ARG, "source_tensor_snapshot",
            "output and native tensor table are required");
    }
    *out = NULL;
    rc = yvex_native_weight_table_finalize(*table, err);
    if (rc != YVEX_OK)
        return rc;
    snapshot = (yvex_source_tensor_snapshot *)calloc(1u, sizeof(*snapshot));
    if (!snapshot) {
        return inventory_refuse(err, YVEX_ERR_NOMEM, "source_tensor_snapshot",
            "source tensor snapshot allocation failed");
    }
    snapshot->table = *table;
    snapshot->shard_count = shard_count;
    snapshot->header_scan_count = header_scan_count;
    snapshot->identity = source_snapshot_identity(*table);
    snapshot->references = 1u;
    *table = NULL;
    *out = snapshot;
    return YVEX_OK;
}

int yvex_source_tensor_snapshot_take_table_with_shards(yvex_source_tensor_snapshot **out,
                                                       yvex_native_weight_table **table,
                                                       const yvex_source_shard_snapshot *shards,
                                                       unsigned long long shard_count,
                                                       unsigned long long header_scan_count,
                                                       yvex_error *err) {
    yvex_source_tensor_snapshot *snapshot = NULL;
    unsigned long long index;
    int rc;

    if (!shards || shard_count == 0u) {
        return inventory_refuse(err, YVEX_ERR_INVALID_ARG, "source_tensor_snapshot",
            "a nonempty canonical shard catalog is required");
    }
    for (index = 0u; index < shard_count; ++index) {
        if (!shards[index].canonical_name || shards[index].canonical_id != index ||
            (index &&
             strcmp(shards[index - 1u].canonical_name, shards[index].canonical_name) >= 0)) {
            return inventory_refuse(err, YVEX_ERR_FORMAT, "source_tensor_snapshot",
                "source shard catalog is not canonical and unique");
        }
    }
    rc = yvex_source_tensor_snapshot_take_table(
        &snapshot, table, shard_count, header_scan_count, err);
    if (rc != YVEX_OK)
        return rc;
    rc = source_snapshot_copy_shards(snapshot, shards, shard_count, err);
    if (rc != YVEX_OK) {
        yvex_source_tensor_snapshot_release(snapshot);
        return rc;
    }
    *out = snapshot;
    return YVEX_OK;
}

void yvex_source_tensor_snapshot_retain(yvex_source_tensor_snapshot *snapshot) {
    if (snapshot && snapshot->references < UINT_MAX)
        snapshot->references++;
}

/*
 * Release one retained immutable tensor snapshot reference.
 *
 * Releases only resources owned by source inventory; cleanup remains deterministic.
 */
void yvex_source_tensor_snapshot_release(yvex_source_tensor_snapshot *snapshot) {
    unsigned long long index;

    if (!snapshot || snapshot->references == 0u)
        return;
    snapshot->references--;
    if (snapshot->references != 0u)
        return;
    for (index = 0u; snapshot->shards && index < snapshot->shard_count; ++index)
        free((char *)snapshot->shards[index].canonical_name);
    free(snapshot->shards);
    yvex_native_weight_table_close(snapshot->table);
    free(snapshot);
}

const yvex_source_shard_snapshot *
yvex_source_tensor_snapshot_shard_at(const yvex_source_tensor_snapshot *snapshot,
                                     unsigned long long index) {
    return snapshot && snapshot->shards && index < snapshot->shard_count ? &snapshot->shards[index]
                                                                         : NULL;
}

static int source_snapshot_shard_key_compare(const void *key, const void *row) {
    return strcmp((const char *)key,
                  ((const yvex_source_shard_snapshot *)row)->canonical_name);
}

const yvex_source_shard_snapshot *
yvex_source_tensor_snapshot_shard_find(const yvex_source_tensor_snapshot *snapshot,
                                       const char *canonical_name) {
    if (!snapshot || !snapshot->shards || !canonical_name)
        return NULL;
    return (const yvex_source_shard_snapshot *)bsearch(canonical_name,
                                                       snapshot->shards,
                                                       (size_t)snapshot->shard_count,
                                                       sizeof(snapshot->shards[0]),
                                                       source_snapshot_shard_key_compare);
}

int yvex_source_tensor_snapshot_has_shard_catalog(const yvex_source_tensor_snapshot *snapshot) {
    return snapshot && snapshot->shards && snapshot->shard_count != 0u;
}

const yvex_native_weight_info *
yvex_source_tensor_snapshot_at(const yvex_source_tensor_snapshot *snapshot,
                               unsigned long long index) {
    return snapshot ? yvex_native_weight_table_at(snapshot->table, index) : NULL;
}

const yvex_native_weight_info *
yvex_source_tensor_snapshot_find(const yvex_source_tensor_snapshot *snapshot, const char *name) {
    return snapshot ? yvex_native_weight_table_find(snapshot->table, name) : NULL;
}

int yvex_source_tensor_snapshot_find_index(const yvex_source_tensor_snapshot *snapshot,
                                           const char *name,
                                           unsigned long long *index) {
    const yvex_native_weight_info *item;

    if (!snapshot || !name || !index)
        return 0;
    item = yvex_native_weight_table_find(snapshot->table, name);
    if (!item)
        return 0;
    *index = (unsigned long long)(item - snapshot->table->items);
    return 1;
}

int yvex_source_tensor_snapshot_facts_get(const yvex_source_tensor_snapshot *snapshot,
                                          yvex_source_tensor_snapshot_facts *out,
                                          yvex_error *err) {
    if (!snapshot || !out) {
        return inventory_refuse(err, YVEX_ERR_INVALID_ARG, "source_tensor_snapshot",
            "snapshot and facts output are required");
    }
    memset(out, 0, sizeof(*out));
    out->tensor_count = snapshot->table->count;
    out->shard_count = snapshot->shard_count;
    out->header_scan_count = snapshot->header_scan_count;
    out->payload_bytes_read = snapshot->payload_bytes_read;
    out->lookup_count = snapshot->table->lookup_count;
    out->collision_count = snapshot->table->collision_count;
    out->maximum_probe = snapshot->table->maximum_probe;
    out->identity = snapshot->identity;
    return YVEX_OK;
}

static int
source_index_append(source_index *index, const char *tensor, const char *shard, yvex_error *err) {
    source_index_entry *next;
    size_t cap;

    if (index->count == index->cap) {
        cap = index->cap ? index->cap * 2u : 256u;
        if (cap < index->cap || cap > SIZE_MAX / sizeof(index->items[0])) {
            return inventory_refuse(err, YVEX_ERR_BOUNDS, "source_index", "shard index allocation overflow");
        }
        next = (source_index_entry *)realloc(index->items, cap * sizeof(index->items[0]));
        if (!next) {
            return inventory_refuse(err, YVEX_ERR_NOMEM, "source_index", "shard index allocation failed");
        }
        index->items = next;
        index->cap = cap;
    }
    memset(&index->items[index->count], 0, sizeof(index->items[0]));
    index->items[index->count].tensor = yvex_core_strdup(tensor);
    index->items[index->count].shard = yvex_core_strdup(shard);
    if (!index->items[index->count].tensor || !index->items[index->count].shard) {
        free(index->items[index->count].tensor);
        free(index->items[index->count].shard);
        return inventory_refuse(err, YVEX_ERR_NOMEM, "source_index", "shard index row allocation failed");
    }
    index->count++;
    return YVEX_OK;
}

static void source_index_free(source_index *index) {
    size_t i;

    if (!index)
        return;
    for (i = 0; i < index->count; ++i) {
        free(index->items[i].tensor);
        free(index->items[i].shard);
    }
    free(index->items);
    memset(index, 0, sizeof(*index));
}

static int source_index_compare(const void *left, const void *right) {
    const source_index_entry *a = (const source_index_entry *)left;
    const source_index_entry *b = (const source_index_entry *)right;
    return strcmp(a->tensor, b->tensor);
}

static int source_index_key_compare(const void *key, const void *row) {
    return strcmp((const char *)key, ((const source_index_entry *)row)->tensor);
}

static int source_parse_index_metadata(yvex_json *json, source_index *index) {
    char key[YVEX_JSON_KEY_CAP];
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "total_size") == 0) {
            if (index->has_declared_total_size ||
                !yvex_json_u64(json, &index->declared_total_size))
                return 0;
            index->has_declared_total_size = 1;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END;
}

static int source_parse_weight_map(yvex_json *json, source_index *index, yvex_error *err) {
    char tensor[YVEX_JSON_KEY_CAP];
    char shard[YVEX_PATH_CAP];
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, tensor, sizeof(tensor))) ==
           YVEX_JSON_ITEM_READY) {
        int rc;
        if (!yvex_json_string(json, shard, sizeof(shard)))
            return 0;
        rc = source_index_append(index, tensor, shard, err);
        if (rc != YVEX_OK)
            return -1;
    }
    return item == YVEX_JSON_ITEM_END && index->count > 0u;
}

static int
source_parse_index_json(const char *data, size_t length, source_index *index, yvex_error *err) {
    yvex_json json;
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];
    unsigned int seen = 0u;
    size_t i;

    yvex_json_init(&json, data, length);
    if (!yvex_json_iter_begin(&json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        int parsed = 1;
        if (strcmp(key, "metadata") == 0) {
            if ((seen & 1u) || !source_parse_index_metadata(&json, index))
                return 0;
            seen |= 1u;
        } else if (strcmp(key, "weight_map") == 0) {
            if (seen & 2u)
                return 0;
            parsed = source_parse_weight_map(&json, index, err);
            if (parsed < 0)
                return -1;
            if (!parsed)
                return 0;
            seen |= 2u;
        } else if (!yvex_json_skip_value(&json)) {
            return 0;
        }
    }
    if (item != YVEX_JSON_ITEM_END || !yvex_json_complete(&json) || !(seen & 2u))
        return 0;
    qsort(index->items, index->count, sizeof(index->items[0]), source_index_compare);
    for (i = 1u; i < index->count; ++i) {
        if (strcmp(index->items[i - 1u].tensor, index->items[i].tensor) == 0) {
            index->duplicate_tensor = 1;
            return 0;
        }
    }
    return 1;
}

static source_index_entry *source_index_find(source_index *index, const char *tensor) {
    return index && tensor
               ? (source_index_entry *)bsearch(tensor,
                                               index->items,
                                               index->count,
                                               sizeof(index->items[0]),
                                               source_index_key_compare)
               : NULL;
}

static int source_shards_append(source_shards *shards,
                                const char *name,
                                unsigned long long size,
                                yvex_error *err) {
    return source_named_size_append(
        &shards->items, &shards->count, &shards->cap, name, size, &source_shard_row_policy, err);
}

static void source_shards_free(source_shards *shards) {
    if (!shards)
        return;
    source_named_size_free(shards->items, shards->count);
    memset(shards, 0, sizeof(*shards));
}

static int
source_shard_name_parse(const char *name, unsigned int *index_out, unsigned int *total_out) {
    unsigned int index = 0u;
    unsigned int total = 0u;
    size_t i;

    if (!name || strlen(name) != strlen("model-00000-of-00000.safetensors") ||
        strncmp(name, "model-", 6u) != 0 || strncmp(name + 11u, "-of-", 4u) != 0 ||
        strcmp(name + 20u, ".safetensors") != 0)
        return 0;
    for (i = 6u; i < 11u; ++i) {
        if (!isdigit((unsigned char)name[i]))
            return 0;
        index = index * 10u + (unsigned int)(name[i] - '0');
    }
    for (i = 15u; i < 20u; ++i) {
        if (!isdigit((unsigned char)name[i]))
            return 0;
        total = total * 10u + (unsigned int)(name[i] - '0');
    }
    if (index == 0u || total == 0u || index > total)
        return 0;
    if (index_out)
        *index_out = index;
    if (total_out)
        *total_out = total;
    return 1;
}

/* Order canonical shard rows by numeric shard identity and relative name. */

static void source_shards_sort(source_shards *shards) {
    qsort(shards->items, shards->count, sizeof(shards->items[0]), source_named_size_compare);
}

static long source_shards_find(const source_shards *shards, const char *name) {
    const source_named_size *row;

    if (!shards || !name)
        return -1;
    row = (const source_named_size *)bsearch(name,
                                             shards->items,
                                             shards->count,
                                             sizeof(shards->items[0]),
                                             source_named_size_key_compare);
    return row ? (long)(row - shards->items) : -1;
}

static int source_scan_root(const char *source_path,
                            source_shards *shards,
                            yvex_source_verification *out,
                            yvex_error *err) {
    yvex_source_manifest_file_list files;
    size_t row;
    int rc;

    yvex_source_manifest_file_list_init(&files);
    rc = yvex_source_manifest_scan_files(source_path, 1, &files, err);
    if (rc == YVEX_ERR_BOUNDS) {
        out->footprint_overflow = 1;
        yvex_source_verification_add_blocker(out, "source-footprint-overflow");
        yvex_error_clear(err);
        rc = YVEX_OK;
    } else if (rc == YVEX_OK) {
        out->source_file_count = files.summary.file_count;
        out->source_total_bytes = files.summary.total_size_bytes;
    }
    if (rc != YVEX_OK)
        goto cleanup;
    for (row = 0u; row < files.count; ++row) {
        const yvex_source_manifest_file *file = &files.items[row];
        unsigned int shard_index = 0u;
        unsigned int shard_total = 0u;

        if (strchr(file->path, '/') || strcmp(file->kind, "safetensors") != 0)
            continue;
        if (!source_shard_name_parse(file->path, &shard_index, &shard_total)) {
            yvex_source_verification_add_blocker(out, "unexpected-shard");
            continue;
        }
        if (!shards->declared_total) {
            shards->declared_total = shard_total;
        } else if (shards->declared_total != shard_total) {
            yvex_source_verification_add_blocker(out, "inconsistent-shard-set");
        }
        rc = source_shards_append(shards, file->path, file->size_bytes, err);
        if (rc != YVEX_OK)
            goto cleanup;
        if (!yvex_core_u64_add(out->shard_bytes, file->size_bytes, &out->shard_bytes)) {
            out->footprint_overflow = 1;
            yvex_source_verification_add_blocker(out, "source-footprint-overflow");
        }
    }
    source_shards_sort(shards);
    out->shard_count = (unsigned long long)shards->count;
    if (!shards->count) {
        yvex_source_verification_add_blocker(out, "missing-source-shards");
    }
    if (shards->declared_total && shards->count != (size_t)shards->declared_total) {
        yvex_source_verification_add_blocker(out, "incomplete-shard-set");
    }
    for (size_t i = 0u; i < shards->count; ++i) {
        unsigned int index = 0u;
        unsigned int prior_index = 0u;
        if (i > 0u &&
            source_shard_name_parse(shards->items[i - 1u].name, &prior_index, NULL) &&
            source_shard_name_parse(shards->items[i].name, &index, NULL) &&
            prior_index == index) {
            yvex_source_verification_add_blocker(out, "duplicate-source-shard");
        }
        if (!source_shard_name_parse(shards->items[i].name, &index, NULL) || index != i + 1u) {
            yvex_source_verification_add_blocker(out, "discontinuous-shard-set");
            break;
        }
    }
    rc = YVEX_OK;
cleanup:
    yvex_source_manifest_file_list_free(&files);
    return rc;
}

static int source_upstream_append(source_upstream_inventory *inventory,
                                  const char *name,
                                  unsigned long long size,
                                  yvex_error *err) {
    return source_named_size_append(&inventory->items,
                                    &inventory->count,
                                    &inventory->cap,
                                    name,
                                    size,
                                    &source_upstream_row_policy,
                                    err);
}

static void source_upstream_free(source_upstream_inventory *inventory) {
    if (!inventory)
        return;
    source_named_size_free(inventory->items, inventory->count);
    memset(inventory, 0, sizeof(*inventory));
}

static int
source_upstream_parse_file(yvex_json *json, source_upstream_inventory *inventory, yvex_error *err) {
    char key[YVEX_JSON_KEY_CAP];
    source_upstream_parse_state state = {{0}, 0u, 0u};
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "path") == 0) {
            if ((state.seen & 1u) ||
                !yvex_json_string(json, state.name, sizeof(state.name)))
                return 0;
            state.seen |= 1u;
        } else if (strcmp(key, "size_bytes") == 0) {
            if ((state.seen & 2u) || !yvex_json_u64(json, &state.size))
                return 0;
            state.seen |= 2u;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END && state.seen == 3u &&
           source_upstream_append(inventory, state.name, state.size, err) == YVEX_OK;
}

static int source_upstream_parse_files(yvex_json *json,
                                       source_upstream_inventory *inventory,
                                       yvex_error *err) {
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
        return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (!source_upstream_parse_file(json, inventory, err))
            return 0;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator && inventory->count > 0u;
}

/* Parses and verifies repository/revision identity for an indexless snapshot. */
static int source_upstream_parse(const char *data,
                                 size_t length,
                                 source_upstream_inventory *inventory,
                                 yvex_error *err) {
    yvex_json json;
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];
    char schema[64];
    unsigned int seen = 0u;

    yvex_json_init(&json, data, length);
    if (!yvex_json_iter_begin(&json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "schema") == 0) {
            if ((seen & 1u) || !yvex_json_string(&json, schema, sizeof(schema)) ||
                strcmp(schema, "yvex.source_upstream_inventory.v1") != 0)
                return 0;
            seen |= 1u;
        } else if (strcmp(key, "repository") == 0) {
            if ((seen & 2u) ||
                !yvex_json_string(&json, inventory->repo, sizeof(inventory->repo)))
                return 0;
            seen |= 2u;
        } else if (strcmp(key, "revision") == 0) {
            if ((seen & 4u) ||
                !yvex_json_string(&json, inventory->revision, sizeof(inventory->revision)))
                return 0;
            seen |= 4u;
        } else if (strcmp(key, "files") == 0) {
            if ((seen & 8u) || !source_upstream_parse_files(&json, inventory, err))
                return 0;
            seen |= 8u;
        } else if (!yvex_json_skip_value(&json)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END && yvex_json_complete(&json) && seen == 15u;
}

/* Verifies the official index file, provider identity, and declared shard map. */
static int source_verify_index(const yvex_source_verify_options *options,
                               source_index *index,
                               yvex_source_verification *out,
                               yvex_error *err) {
    char path[YVEX_PATH_CAP];
    char *data;
    size_t length;
    int parsed;
    int rc;

    if (!yvex_source_path_join(
            path, sizeof(path), options->source_path, options->identity->upstream_index_path) ||
        !yvex_source_regular_file(path, NULL)) {
        yvex_source_verification_add_blocker(out, "missing-shard-index");
        return YVEX_OK;
    }
    out->shard_index_present = 1;
    data = yvex_read_bounded_file(path, SOURCE_INDEX_CAP, &length, err);
    if (!data) {
        yvex_source_verification_add_blocker(out, "malformed-shard-index");
        return yvex_error_code(err) == YVEX_ERR_NOMEM ? YVEX_ERR_NOMEM : YVEX_OK;
    }
    parsed = source_parse_index_json(data, length, index, err);
    free(data);
    if (parsed < 0)
        return yvex_error_code(err);
    if (!parsed) {
        yvex_source_verification_add_blocker(
            out, index->duplicate_tensor ? "duplicate-index-tensor" : "malformed-shard-index");
        return YVEX_OK;
    }
    out->shard_index_valid = 1;
    out->indexed_tensor_count = (unsigned long long)index->count;
    rc = yvex_source_provenance_verify_file(
        options, options->identity->upstream_index_path, 1, out, err);
    if (rc != YVEX_OK)
        return rc;
    snprintf(out->inventory_authority, sizeof(out->inventory_authority), "%s", "upstream-index");
    return YVEX_OK;
}

static void source_verify_index_shards(source_index *index,
                                       const source_shards *shards,
                                       yvex_source_verification *out) {
    unsigned char *referenced;
    size_t i;
    unsigned long long unique = 0u;

    if (!out->shard_index_valid)
        return;
    referenced = (unsigned char *)calloc(shards->count ? shards->count : 1u, sizeof(*referenced));
    if (!referenced) {
        yvex_source_verification_add_blocker(out, "shard-reference-allocation-failed");
        return;
    }
    for (i = 0; i < index->count; ++i) {
        long shard_index = source_shards_find(shards, index->items[i].shard);
        if (shard_index < 0) {
            yvex_source_verification_add_blocker(out, "missing-referenced-shard");
        } else if (!referenced[(size_t)shard_index]) {
            referenced[(size_t)shard_index] = 1u;
            unique++;
        }
    }
    out->referenced_shard_count = unique;
    for (i = 0; i < shards->count; ++i) {
        if (!referenced[i]) {
            yvex_source_verification_add_blocker(out, "unexpected-shard");
        }
    }
    free(referenced);
}

static int source_verify_upstream_indexless(const yvex_source_verify_options *options,
                                            const source_shards *shards,
                                            yvex_source_verification *out,
                                            yvex_error *err) {
    source_upstream_inventory upstream;
    char *data;
    size_t length;
    size_t i;

    memset(&upstream, 0, sizeof(upstream));
    if (!options->upstream_inventory_path ||
        !yvex_source_regular_file(options->upstream_inventory_path, NULL)) {
        yvex_source_verification_add_blocker(out, "missing-upstream-inventory");
        return YVEX_OK;
    }
    data = yvex_read_bounded_file(
        options->upstream_inventory_path, SOURCE_UPSTREAM_INVENTORY_CAP, &length, err);
    if (!data) {
        yvex_source_verification_add_blocker(out, "malformed-upstream-inventory");
        return yvex_error_code(err) == YVEX_ERR_NOMEM ? YVEX_ERR_NOMEM : YVEX_OK;
    }
    if (!source_upstream_parse(data, length, &upstream, err)) {
        free(data);
        source_upstream_free(&upstream);
        yvex_source_verification_add_blocker(out, "malformed-upstream-inventory");
        return YVEX_OK;
    }
    free(data);
    if (strcmp(upstream.repo, options->identity->upstream_repo_id) != 0 ||
        strcmp(upstream.revision, options->identity->upstream_revision) != 0) {
        yvex_source_verification_add_blocker(out, "stale-upstream-inventory");
    }
    for (i = 0u; i < upstream.count; ++i) {
        long local = source_shards_find(shards, upstream.items[i].name);
        if (local < 0 || shards->items[local].size != upstream.items[i].size) {
            yvex_source_verification_add_blocker(out, "upstream-local-inventory-drift");
        }
    }
    if (upstream.count != shards->count) {
        yvex_source_verification_add_blocker(out, "upstream-local-inventory-drift");
    }
    if (!yvex_source_verification_has_blocker(out, "upstream-local-inventory-drift") &&
        !yvex_source_verification_has_blocker(out, "stale-upstream-inventory")) {
        snprintf(
            out->inventory_authority, sizeof(out->inventory_authority), "%s", "header-derived");
    }
    source_upstream_free(&upstream);
    return YVEX_OK;
}

static int source_native_info_compare(const void *left, const void *right) {
    const yvex_native_weight_info *const *a = (const yvex_native_weight_info *const *)left;
    const yvex_native_weight_info *const *b = (const yvex_native_weight_info *const *)right;
    return strcmp((*a)->name, (*b)->name);
}

/* Builds deterministic owned tensor-to-shard rows from the canonical header table. */
static int source_derived_build(const yvex_native_weight_table *table,
                                yvex_source_derived_inventory *derived,
                                yvex_error *err) {
    const yvex_native_weight_info **sorted;
    size_t i;

    if (!derived)
        return YVEX_OK;
    sorted = (const yvex_native_weight_info **)calloc((size_t)table->count, sizeof(sorted[0]));
    if (!sorted) {
        return inventory_refuse(err, YVEX_ERR_NOMEM, "source_derived_inventory",
            "derived inventory sort allocation failed");
    }
    derived->rows =
        (yvex_source_inventory_row *)calloc((size_t)table->count, sizeof(derived->rows[0]));
    if (!derived->rows) {
        free(sorted);
        return inventory_refuse(err, YVEX_ERR_NOMEM, "source_derived_inventory",
            "derived inventory row allocation failed");
    }
    for (i = 0u; i < (size_t)table->count; ++i)
        sorted[i] = &table->items[i];
    qsort(sorted, (size_t)table->count, sizeof(sorted[0]), source_native_info_compare);
    for (i = 0u; i < (size_t)table->count; ++i) {
        derived->rows[i].tensor = yvex_core_strdup(sorted[i]->name);
        derived->rows[i].shard = yvex_core_strdup(sorted[i]->shard_path);
        if (!derived->rows[i].tensor || !derived->rows[i].shard) {
            derived->count = i + 1u;
            free(sorted);
            yvex_source_derived_inventory_free(derived);
            return inventory_refuse(err, YVEX_ERR_NOMEM, "source_derived_inventory",
                "derived inventory string allocation failed");
        }
    }
    derived->count = (size_t)table->count;
    free(sorted);
    return YVEX_OK;
}

static int source_verify_headers(const yvex_source_verify_options *options,
                                 const source_shards *shards,
                                 source_index *index,
                                 yvex_source_verification *out,
                                 yvex_source_derived_inventory *derived,
                                 yvex_source_tensor_snapshot **snapshot,
                                 yvex_error *err) {
    yvex_native_weight_table *table;
    yvex_source_shard_snapshot *shard_facts = NULL;
    size_t i;
    int mismatch = 0;
    int shard_catalog_complete = 1;
    int indexed = strcmp(out->inventory_authority, "upstream-index") == 0;
    int rc = YVEX_OK;

    table = (yvex_native_weight_table *)calloc(1u, sizeof(*table));
    if (!table) {
        yvex_error_set(
            err, YVEX_ERR_NOMEM, "source_verify_headers", "native header table allocation failed");
        return YVEX_ERR_NOMEM;
    }
    if (snapshot) {
        shard_facts = (yvex_source_shard_snapshot *)calloc(shards->count, sizeof(shard_facts[0]));
        if (!shard_facts) {
            yvex_native_weight_table_close(table);
            return inventory_refuse(err, YVEX_ERR_NOMEM, "source_verify_headers",
                "retained shard geometry allocation failed");
        }
    }
    out->header_scan_count++;
    for (i = 0; i < shards->count; ++i) {
        char path[YVEX_PATH_CAP];
        yvex_error header_error;
        yvex_safetensors_file_facts file_facts;
        unsigned long long before = table->count;
        unsigned long long row;

        rc = yvex_source_provenance_verify_file(options, shards->items[i].name, 0, out, err);
        if (rc != YVEX_OK)
            goto cleanup;
        if (!yvex_source_path_join(
                path, sizeof(path), options->source_path, shards->items[i].name)) {
            yvex_source_verification_add_blocker(out, "invalid-safetensors-header");
            shard_catalog_complete = 0;
            continue;
        }
        yvex_error_clear(&header_error);
        memset(&file_facts, 0, sizeof(file_facts));
        rc = yvex_safetensors_read_header_file_with_facts(
            path, shards->items[i].name, table, &file_facts, &header_error);
        if (rc == YVEX_ERR_NOMEM) {
            yvex_error_set(err,
                           YVEX_ERR_NOMEM,
                           "source_verify_headers",
                           "native header inventory allocation failed");
            goto cleanup;
        }
        if (rc != YVEX_OK) {
            if (strcmp(yvex_error_where(&header_error), "native_weight_add") == 0 &&
                strncmp(yvex_error_message(&header_error),
                        "duplicate tensor name:",
                        strlen("duplicate tensor name:")) == 0) {
                yvex_source_verification_add_blocker(out, "duplicate-header-tensor");
                mismatch = 1;
            } else {
                yvex_source_verification_add_blocker(out, "invalid-safetensors-header");
            }
            shard_catalog_complete = 0;
            rc = YVEX_OK;
            continue;
        }
        if (shard_facts) {
            shard_facts[i].canonical_id = (unsigned long long)i;
            shard_facts[i].canonical_name = shards->items[i].name;
            shard_facts[i].file_bytes = file_facts.file_bytes;
            shard_facts[i].data_region_offset = file_facts.data_region_offset;
            shard_facts[i].payload_bytes = file_facts.payload_bytes;
        }
        for (row = before; row < table->count && indexed; ++row) {
            source_index_entry *entry = source_index_find(index, table->items[row].name);
            if (!entry || strcmp(entry->shard, shards->items[i].name) != 0) {
                mismatch = 1;
            } else {
                entry->seen_in_header = 1;
            }
        }
    }
    rc = yvex_native_weight_table_finalize(table, err);
    if (rc != YVEX_OK)
        goto cleanup;
    if (indexed) {
        for (i = 0; i < index->count; ++i) {
            if (!index->items[i].seen_in_header)
                mismatch = 1;
        }
        if (mismatch || table->count != index->count) {
            yvex_source_verification_add_blocker(out, "shard-index-header-mismatch");
        } else {
            out->shard_index_headers_match = 1;
        }
    } else if (!mismatch) {
        out->shard_index_headers_match = 1;
        out->indexed_tensor_count = table->count;
        out->referenced_shard_count = shards->count;
        rc = source_derived_build(table, derived, err);
        if (rc != YVEX_OK)
            goto cleanup;
    }
    for (i = 0; i < (size_t)table->count; ++i) {
        const yvex_native_weight_info *info = &table->items[i];
        if (info->rank > out->max_tensor_rank)
            out->max_tensor_rank = info->rank;
        source_count_dtype(out, info->dtype);
    }
    out->header_shard_count = table->header_read_count;
    out->header_tensor_count = table->count;
    out->header_bytes = table->header_bytes;
    out->declared_tensor_bytes = table->summary.total_tensor_bytes;
    if (index->has_declared_total_size &&
        index->declared_total_size != out->declared_tensor_bytes) {
        yvex_source_verification_add_blocker(out, "shard-index-size-mismatch");
    }
    if (snapshot && shard_catalog_complete) {
        rc = yvex_source_tensor_snapshot_take_table_with_shards(snapshot,
                                                                &table,
                                                                shard_facts,
                                                                (unsigned long long)shards->count,
                                                                out->header_scan_count,
                                                                err);
        if (rc != YVEX_OK)
            goto cleanup;
    }
    rc = YVEX_OK;
cleanup:
    free(shard_facts);
    yvex_native_weight_table_close(table);
    return rc;
}

/*
 * Release the complete derived inventory, retained snapshot, and all owned rows.
 *
 * Releases only resources owned by source inventory; cleanup remains deterministic.
 */
void yvex_source_derived_inventory_free(yvex_source_derived_inventory *inventory) {
    size_t i;

    if (!inventory)
        return;
    for (i = 0u; i < inventory->count; ++i) {
        free(inventory->rows[i].tensor);
        free(inventory->rows[i].shard);
    }
    free(inventory->rows);
    memset(inventory, 0, sizeof(*inventory));
}

int yvex_source_inventory_verify(const yvex_source_verify_options *options,
                                 yvex_source_verification *out,
                                 yvex_source_derived_inventory *derived,
                                 yvex_source_tensor_snapshot **snapshot,
                                 yvex_error *err) {
    source_shards shards;
    source_index index;
    int rc;

    if (!options || !out || !options->identity) {
        return inventory_refuse(err, YVEX_ERR_INVALID_ARG, "source_inventory_verify",
            "source verification options and output are required");
    }
    memset(&shards, 0, sizeof(shards));
    memset(&index, 0, sizeof(index));
    if (derived)
        memset(derived, 0, sizeof(*derived));
    if (snapshot)
        *snapshot = NULL;
    rc = source_scan_root(options->source_path, &shards, out, err);
    if (rc != YVEX_OK)
        goto cleanup;
    if (strcmp(options->identity->upstream_inventory_authority, "upstream-index") == 0) {
        rc = source_verify_index(options, &index, out, err);
        if (rc != YVEX_OK)
            goto cleanup;
        source_verify_index_shards(&index, &shards, out);
    } else if (strcmp(options->identity->upstream_inventory_authority, "header-derived") == 0) {
        rc = source_verify_upstream_indexless(options, &shards, out, err);
        if (rc != YVEX_OK)
            goto cleanup;
    } else {
        yvex_source_verification_add_blocker(out, "unsupported-inventory-authority");
    }
    rc = source_verify_headers(options, &shards, &index, out, derived, snapshot, err);
cleanup:
    source_index_free(&index);
    source_shards_free(&shards);
    return rc;
}

static int component_inventory_prefix(const char *path, const char *prefix)
{
    size_t length = prefix ? strlen(prefix) : 0u;

    return path && length && strncmp(path, prefix, length) == 0;
}

static int component_inventory_shard_add(const char **shards,
                                         size_t *count,
                                         size_t capacity,
                                         const char *path)
{
    size_t index;

    for (index = 0u; index < *count; ++index)
        if (strcmp(shards[index], path) == 0) return 1;
    if (*count >= capacity) return 0;
    shards[(*count)++] = path;
    return 1;
}

static int component_inventory_index_open(const char *source_root,
                                          const char *relative_path,
                                          source_index *index,
                                          yvex_error *err)
{
    char path[YVEX_PATH_CAP];
    char *data;
    size_t length = 0u;
    int parsed;

    if (!yvex_source_path_join(path, sizeof(path), source_root, relative_path)) {
        return inventory_refuse(err, YVEX_ERR_BOUNDS,
                                "source_component_inventory",
                                "component index path exceeds bounds");
    }
    data = yvex_read_bounded_file(path, 16u * 1024u * 1024u, &length, err);
    if (!data) return err ? err->code : YVEX_ERR_IO;
    parsed = source_parse_index_json(data, length, index, err);
    free(data);
    if (parsed == 1) return YVEX_OK;
    return inventory_refuse(err, YVEX_ERR_FORMAT,
                            "source_component_inventory",
                            "component shard index is malformed or duplicated");
}

static int component_inventory_expected_shard(char *out,
                                              size_t cap,
                                              const char *index_path,
                                              const char *shard)
{
    const char *slash;
    size_t prefix;
    size_t shard_length;

    if (!out || !index_path || !shard || strchr(shard, '/') ||
        !yvex_source_payload_name_is_canonical(shard)) return 0;
    slash = strrchr(index_path, '/');
    prefix = slash ? (size_t)(slash - index_path + 1) : 0u;
    shard_length = strlen(shard);
    if (prefix + shard_length >= cap) return 0;
    if (prefix) memcpy(out, index_path, prefix);
    memcpy(out + prefix, shard, shard_length + 1u);
    return 1;
}

static int component_inventory_verify_one(
    const char *source_root,
    const yvex_native_weight_table *table,
    const yvex_source_component_inventory_spec *spec,
    unsigned long long *tensor_count,
    unsigned long long *shard_count,
    yvex_error *err)
{
    const char *shards[YVEX_SOURCE_ACQUISITION_FILE_CAP];
    source_index index;
    unsigned long long payload_bytes = 0u;
    unsigned long long row;
    size_t unique_shards = 0u;
    int rc = YVEX_OK;

    memset(&index, 0, sizeof(index));
    *tensor_count = 0u;
    *shard_count = 0u;
    if (!spec->component || !spec->component[0] || !spec->shard_prefix ||
        !spec->shard_prefix[0] || !spec->expected_shards ||
        !spec->expected_tensors) {
        return inventory_refuse(err, YVEX_ERR_INVALID_ARG,
                                "source_component_inventory",
                                "complete component inventory spec is required");
    }
    if (spec->index_path) {
        rc = component_inventory_index_open(source_root, spec->index_path,
                                            &index, err);
        if (rc != YVEX_OK) goto cleanup;
    }
    for (row = 0u; row < table->count; ++row) {
        const yvex_native_weight_info *tensor = &table->items[row];
        source_index_entry *entry;

        if (!component_inventory_prefix(tensor->shard_path,
                                        spec->shard_prefix)) continue;
        if (!component_inventory_shard_add(
                shards, &unique_shards,
                sizeof(shards) / sizeof(shards[0]), tensor->shard_path) ||
            !yvex_core_u64_add(payload_bytes, tensor->data_bytes,
                               &payload_bytes)) {
            rc = inventory_refuse(err, YVEX_ERR_BOUNDS,
                                  "source_component_inventory",
                                  "component inventory exceeds resource bounds");
            goto cleanup;
        }
        (*tensor_count)++;
        if (!spec->index_path) continue;
        entry = source_index_find(&index, tensor->name);
        if (!entry) {
            rc = inventory_refuse(err, YVEX_ERR_FORMAT,
                                  "source_component_inventory",
                                  "component tensor is absent from its shard index");
            goto cleanup;
        }
        {
            char expected[YVEX_PATH_CAP];
            if (!component_inventory_expected_shard(
                    expected, sizeof(expected), spec->index_path,
                    entry->shard) ||
                strcmp(expected, tensor->shard_path) != 0) {
                rc = inventory_refuse(err, YVEX_ERR_FORMAT,
                                      "source_component_inventory",
                                      "component tensor shard disagrees with its index");
                goto cleanup;
            }
        }
    }
    *shard_count = (unsigned long long)unique_shards;
    if (*tensor_count != spec->expected_tensors ||
        *shard_count != spec->expected_shards ||
        (spec->index_path && index.count != *tensor_count) ||
        (spec->index_path && index.has_declared_total_size &&
         index.declared_total_size != payload_bytes)) {
        rc = inventory_refuse(err, YVEX_ERR_FORMAT,
                              "source_component_inventory",
                              "component shard, tensor, or payload totals disagree");
    }
cleanup:
    source_index_free(&index);
    return rc;
}

int yvex_source_component_inventory_verify(
    const char *source_root,
    const yvex_native_weight_table *table,
    const yvex_source_component_inventory_spec *specs,
    size_t spec_count,
    yvex_source_component_inventory_facts *facts,
    yvex_error *err)
{
    unsigned long long matched = 0u;
    size_t index;

    if (!source_root || !table || !specs || !spec_count || spec_count > 32u ||
        !facts) {
        return inventory_refuse(err, YVEX_ERR_INVALID_ARG,
                                "source_component_inventory",
                                "source, table, specs, and facts are required");
    }
    memset(facts, 0, sizeof(*facts));
    for (index = 0u; index < spec_count; ++index) {
        unsigned long long tensors;
        unsigned long long shards;
        int rc = component_inventory_verify_one(
            source_root, table, &specs[index], &tensors, &shards, err);
        if (rc != YVEX_OK) return rc;
        if (!yvex_core_u64_add(matched, tensors, &matched) ||
            !yvex_core_u64_add(facts->shard_count, shards,
                               &facts->shard_count)) {
            return inventory_refuse(err, YVEX_ERR_BOUNDS,
                                    "source_component_inventory",
                                    "component totals overflow");
        }
        if (specs[index].index_path) facts->indexed_component_count++;
    }
    if (matched != table->count) {
        return inventory_refuse(err, YVEX_ERR_FORMAT,
                                "source_component_inventory",
                                "one or more source tensors lack a component owner");
    }
    facts->component_count = (unsigned long long)spec_count;
    facts->tensor_count = matched;
    facts->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
