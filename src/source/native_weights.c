/*
 * Own deterministic native tensor tables and lookup indexes.
 *
 * Finalized tables contain unique names and checked byte geometry. Native tables describe headers
 * and never materialize payloads.
 */
#define _XOPEN_SOURCE 700
#include <dirent.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yvex/internal/source_payload.h>
#include <yvex/source.h>

static int native_refuse(yvex_error *err,
                         yvex_status status,
                         const char *where,
                         const char *message) {
    yvex_error_set(err, status, where, message);
    return status;
}

typedef struct {
    yvex_native_dtype dtype;
    const char *names[3];
} native_dtype_row;

static const native_dtype_row native_dtype_rows[] = {
    {YVEX_NATIVE_DTYPE_UNKNOWN, {"UNKNOWN", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_F64, {"F64", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_F32, {"F32", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_F16, {"F16", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_BF16, {"BF16", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_I64, {"I64", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_I32, {"I32", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_I16, {"I16", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_I8, {"I8", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_U8, {"U8", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_BOOL, {"BOOL", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_F8_E4M3, {"F8_E4M3", "F8_E4M3FN", NULL}},
    {YVEX_NATIVE_DTYPE_F8_E5M2, {"F8_E5M2", NULL, NULL}},
    {YVEX_NATIVE_DTYPE_F8_E8M0, {"F8_E8M0", "F8_E8M0FNU", "F8_E8M0FNUZ"}},
    {YVEX_NATIVE_DTYPE_FP4, {"FP4", "F4", NULL}},
    {YVEX_NATIVE_DTYPE_OTHER, {"OTHER", NULL, NULL}},
};

const char *yvex_native_dtype_name(yvex_native_dtype dtype) {
    size_t index;

    for (index = 0u; index < sizeof(native_dtype_rows) / sizeof(native_dtype_rows[0]); ++index)
        if (native_dtype_rows[index].dtype == dtype)
            return native_dtype_rows[index].names[0];
    return "UNKNOWN";
}

static yvex_native_dtype nw_dtype_from_name(const char *name) {
    size_t row, alias;

    if (!name)
        return YVEX_NATIVE_DTYPE_UNKNOWN;
    for (row = 0u; row < sizeof(native_dtype_rows) / sizeof(native_dtype_rows[0]); ++row)
        for (alias = 0u; alias < 3u && native_dtype_rows[row].names[alias]; ++alias)
            if (strcmp(name, native_dtype_rows[row].names[alias]) == 0)
                return native_dtype_rows[row].dtype;
    return YVEX_NATIVE_DTYPE_OTHER;
}

static uint64_t nw_name_hash(const char *name) {
    uint64_t hash = UINT64_C(1469598103934665603);

    while (name && *name) {
        hash ^= (unsigned char)*name++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int nw_index_rebuild(yvex_native_weight_table *table, size_t requested, yvex_error *err) {
    unsigned long long *slots;
    size_t cap = 128u;
    unsigned long long i;

    if (requested > SIZE_MAX / 2u) {
        return native_refuse(err, YVEX_ERR_BOUNDS, "native_weight_index", "native tensor name index capacity overflow");
    }
    while (cap < requested * 2u) {
        if (cap > SIZE_MAX / 2u) {
            return native_refuse(err, YVEX_ERR_BOUNDS, "native_weight_index",
                "native tensor name index capacity overflow");
        }
        cap *= 2u;
    }
    slots = (unsigned long long *)calloc(cap, sizeof(slots[0]));
    if (!slots) {
        return native_refuse(err, YVEX_ERR_NOMEM, "native_weight_index", "native tensor name index allocation failed");
    }
    for (i = 0u; i < table->count; ++i) {
        size_t slot = (size_t)(nw_name_hash(table->items[i].name) & (cap - 1u));
        unsigned long long probe = 0u;
        while (slots[slot] != 0u) {
            slot = (slot + 1u) & (cap - 1u);
            probe++;
        }
        slots[slot] = i + 1u;
        if (probe > table->maximum_probe)
            table->maximum_probe = probe;
    }
    free(table->name_slots);
    table->name_slots = slots;
    table->name_slot_count = cap;
    return YVEX_OK;
}

static long long nw_index_find(const yvex_native_weight_table *table, const char *name) {
    size_t slot;
    size_t visited = 0u;

    if (!table || !name || !table->name_slots || !table->name_slot_count)
        return -1;
    ((yvex_native_weight_table *)table)->lookup_count++;
    slot = (size_t)(nw_name_hash(name) & (table->name_slot_count - 1u));
    while (visited++ < table->name_slot_count && table->name_slots[slot] != 0u) {
        unsigned long long index = table->name_slots[slot] - 1u;
        if (strcmp(table->items[index].name, name) == 0)
            return (long long)index;
        ((yvex_native_weight_table *)table)->collision_count++;
        slot = (slot + 1u) & (table->name_slot_count - 1u);
    }
    return -1;
}

static int nw_row_compare(const void *left, const void *right) {
    const yvex_native_weight_info *a = (const yvex_native_weight_info *)left;
    const yvex_native_weight_info *b = (const yvex_native_weight_info *)right;
    int result = strcmp(a->name, b->name);

    return result ? result : strcmp(a->shard_path, b->shard_path);
}

/* Construct canonical native tensor table indexes and seal their deterministic facts. */
int yvex_native_weight_table_finalize(yvex_native_weight_table *table, yvex_error *err) {
    if (!table) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "native_weight_finalize", "native tensor table is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (table->finalized)
        return YVEX_OK;
    if (table->count > (unsigned long long)SIZE_MAX) {
        return native_refuse(err, YVEX_ERR_BOUNDS, "native_weight_finalize",
            "native tensor count exceeds addressable memory");
    }
    if (table->count > 1u)
        qsort(table->items, (size_t)table->count, sizeof(table->items[0]), nw_row_compare);
    table->maximum_probe = 0u;
    if (nw_index_rebuild(table, (size_t)table->count + 1u, err) != YVEX_OK)
        return yvex_error_code(err);
    table->finalized = 1;
    return YVEX_OK;
}

int yvex_native_weight_table_add(yvex_native_weight_table *table,
                                 const char *name,
                                 const char *shard_path,
                                 const char *dtype_name,
                                 unsigned int rank,
                                 const unsigned long long *dims,
                                 unsigned long long data_start,
                                 unsigned long long data_end,
                                 yvex_error *err) {
    yvex_native_weight_info *next;
    yvex_native_weight_info *row;
    unsigned int i;

    if (!table || !name || !shard_path || !dtype_name || rank > YVEX_NATIVE_WEIGHT_MAX_DIMS ||
        data_end < data_start) {
        return native_refuse(err, YVEX_ERR_INVALID_ARG, "native_weight_add", "invalid native tensor row");
    }
    if (table->finalized) {
        return native_refuse(err, YVEX_ERR_STATE, "native_weight_add", "cannot mutate a finalized native tensor table");
    }
    if (table->count >= (unsigned long long)SIZE_MAX || table->count == ULLONG_MAX) {
        return native_refuse(err, YVEX_ERR_BOUNDS, "native_weight_add",
            "native tensor count exceeds addressable memory");
    }
    if (!table->name_slots || (size_t)table->count + 1u >= table->name_slot_count / 2u) {
        int index_rc = nw_index_rebuild(table, (size_t)table->count + 1u, err);
        if (index_rc != YVEX_OK)
            return index_rc;
    }
    if (nw_index_find(table, name) >= 0) {
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "native_weight_add", "duplicate tensor name: %s", name);
        return YVEX_ERR_FORMAT;
    }
    if (table->count == table->cap) {
        unsigned long long cap;
        if (table->cap > (unsigned long long)(SIZE_MAX / (2u * sizeof(table->items[0]))) ||
            table->cap > ULLONG_MAX / 2u) {
            yvex_error_set(
                err, YVEX_ERR_BOUNDS, "native_weight_add", "native tensor table capacity overflow");
            return YVEX_ERR_BOUNDS;
        }
        cap = table->cap == 0 ? 64u : table->cap * 2u;
        next =
            (yvex_native_weight_info *)realloc(table->items, (size_t)cap * sizeof(table->items[0]));
        if (!next) {
            yvex_error_set(
                err, YVEX_ERR_NOMEM, "native_weight_add", "native tensor table allocation failed");
            return YVEX_ERR_NOMEM;
        }
        table->items = next;
        table->cap = cap;
    }
    row = &table->items[table->count];
    memset(row, 0, sizeof(*row));
    row->name = yvex_core_strdup(name);
    row->shard_path = yvex_core_strdup(shard_path);
    row->dtype_name = yvex_core_strdup(dtype_name);
    if (!row->name || !row->shard_path || !row->dtype_name) {
        yvex_error_set(
            err, YVEX_ERR_NOMEM, "native_weight_add", "native tensor row allocation failed");
        free((char *)row->name);
        free((char *)row->shard_path);
        free((char *)row->dtype_name);
        memset(row, 0, sizeof(*row));
        return YVEX_ERR_NOMEM;
    }
    row->dtype = nw_dtype_from_name(dtype_name);
    row->rank = rank;
    for (i = 0; i < rank; ++i) {
        row->dims[i] = dims[i];
    }
    row->data_start = data_start;
    row->data_end = data_end;
    row->data_bytes = data_end - data_start;
    if (table->count == ULLONG_MAX || table->summary.tensor_count == ULLONG_MAX ||
        ULLONG_MAX - table->summary.total_tensor_bytes < row->data_bytes) {
        free((char *)row->name);
        free((char *)row->shard_path);
        free((char *)row->dtype_name);
        memset(row, 0, sizeof(*row));
        yvex_error_set(
            err, YVEX_ERR_BOUNDS, "native_weight_add", "native tensor inventory overflow");
        return YVEX_ERR_BOUNDS;
    }
    table->count++;
    table->summary.tensor_count++;
    table->summary.total_tensor_bytes += row->data_bytes;
    if (row->dtype == YVEX_NATIVE_DTYPE_UNKNOWN || row->dtype == YVEX_NATIVE_DTYPE_OTHER) {
        table->summary.unknown_dtype_count++;
    }
    {
        size_t slot = (size_t)(nw_name_hash(row->name) & (table->name_slot_count - 1u));
        unsigned long long probe = 0u;
        while (table->name_slots[slot] != 0u) {
            slot = (slot + 1u) & (table->name_slot_count - 1u);
            probe++;
        }
        table->name_slots[slot] = table->count;
        if (probe > table->maximum_probe)
            table->maximum_probe = probe;
    }
    return YVEX_OK;
}

static int nw_scan_dir(const char *root,
                       const char *rel_dir,
                       int recursive,
                       yvex_native_weight_table *table,
                       yvex_error *err) {
    char *abs_dir = rel_dir && rel_dir[0] ? yvex_source_path_alloc(root, rel_dir) : yvex_core_strdup(root);
    DIR *dir;
    struct dirent *ent;
    int rc = YVEX_OK;

    if (!abs_dir) {
        return native_refuse(err, YVEX_ERR_NOMEM, "native_weights", "directory path allocation failed");
    }
    dir = opendir(abs_dir);
    if (!dir) {
        yvex_error_setf(
            err, YVEX_ERR_IO, "native_weights", "cannot open source directory: %s", abs_dir);
        free(abs_dir);
        return YVEX_ERR_IO;
    }
    while ((ent = readdir(dir)) != NULL) {
        char *rel_path;
        char *abs_path;
        struct stat st;

        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        rel_path = rel_dir && rel_dir[0]
                       ? yvex_source_path_alloc(rel_dir, ent->d_name)
                       : yvex_core_strdup(ent->d_name);
        abs_path = rel_path ? yvex_source_path_alloc(root, rel_path) : NULL;
        if (!rel_path || !abs_path) {
            free(rel_path);
            free(abs_path);
            yvex_error_set(err, YVEX_ERR_NOMEM, "native_weights", "scan path allocation failed");
            rc = YVEX_ERR_NOMEM;
            break;
        }
        if (lstat(abs_path, &st) != 0) {
            yvex_error_setf(err, YVEX_ERR_IO, "native_weights", "cannot stat path: %s", abs_path);
            rc = YVEX_ERR_IO;
        } else if (S_ISDIR(st.st_mode) && recursive) {
            rc = nw_scan_dir(root, rel_path, recursive, table, err);
        } else if (S_ISREG(st.st_mode) && yvex_source_ends_with(rel_path, ".safetensors")) {
            table->summary.shard_count++;
            rc = yvex_safetensors_read_header_file(abs_path, rel_path, table, err);
        }
        free(abs_path);
        free(rel_path);
        if (rc != YVEX_OK) {
            break;
        }
    }
    closedir(dir);
    free(abs_dir);
    return rc;
}

int yvex_native_weight_table_open(yvex_native_weight_table **out,
                                  const yvex_native_weight_options *options,
                                  yvex_error *err) {
    yvex_native_weight_table *table;
    struct stat st;
    int rc;

    if (!out || !options || !options->source_dir) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "native_weights_open", "out and source_dir are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    if (lstat(options->source_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        yvex_error_setf(err,
                        YVEX_ERR_IO,
                        "native_weights_open",
                        "source directory not found: %s",
                        options->source_dir);
        return YVEX_ERR_IO;
    }
    table = (yvex_native_weight_table *)calloc(1, sizeof(*table));
    if (!table) {
        yvex_error_set(
            err, YVEX_ERR_NOMEM, "native_weights_open", "native weight table allocation failed");
        return YVEX_ERR_NOMEM;
    }
    rc = nw_scan_dir(options->source_dir, "", options->recursive, table, err);
    if (rc != YVEX_OK) {
        yvex_native_weight_table_close(table);
        return rc;
    }
    rc = yvex_native_weight_table_finalize(table, err);
    if (rc != YVEX_OK) {
        yvex_native_weight_table_close(table);
        return rc;
    }
    *out = table;
    return YVEX_OK;
}

/*
 * Release resources owned by one native tensor table object and clear its observable state.
 *
 * Releases only resources owned by native source inventory; cleanup remains deterministic.
 */
void yvex_native_weight_table_close(yvex_native_weight_table *table) {
    unsigned long long i;

    if (!table) {
        return;
    }
    for (i = 0; i < table->count; ++i) {
        free((char *)table->items[i].name);
        free((char *)table->items[i].shard_path);
        free((char *)table->items[i].dtype_name);
    }
    free(table->items);
    free(table->name_slots);
    free(table);
}

unsigned long long yvex_native_weight_table_count(const yvex_native_weight_table *table) {
    return table ? table->count : 0;
}

const yvex_native_weight_info *yvex_native_weight_table_at(const yvex_native_weight_table *table,
                                                           unsigned long long index) {
    if (!table || index >= table->count) {
        return NULL;
    }
    return &table->items[index];
}

const yvex_native_weight_info *yvex_native_weight_table_find(const yvex_native_weight_table *table,
                                                             const char *name) {
    if (!table || !name) {
        return NULL;
    }
    {
        long long index = nw_index_find(table, name);
        return index >= 0 ? &table->items[index] : NULL;
    }
}

int yvex_native_weight_table_summary(const yvex_native_weight_table *table,
                                     yvex_native_weight_summary *out,
                                     yvex_error *err) {
    if (!table || !out) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "native_weights_summary", "table and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = table->summary;
    return YVEX_OK;
}
