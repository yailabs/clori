/*
 * Build deterministic source footprint rows beneath one root.
 *
 * Scans metadata only and never reads model payload contents. Footprint discovery does not create
 * trust.
 */
#define _XOPEN_SOURCE 700
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yvex/internal/source_payload.h>

static int scan_refuse(yvex_error *err,
                       yvex_status status,
                       const char *where,
                       const char *message) {
    yvex_error_set(err, status, where, message);
    return status;
}

typedef enum {
    SCAN_NAME_EXACT,
    SCAN_NAME_PREFIX,
    SCAN_NAME_SUFFIX
} scan_name_match;

typedef struct {
    const char *pattern;
    const char *kind;
    scan_name_match match;
} scan_kind_rule;

static const scan_kind_rule scan_kind_rules[] = {
    {"generation_config.json", "config", SCAN_NAME_EXACT},
    {"config", "config", SCAN_NAME_PREFIX},
    {"tokenizer", "tokenizer", SCAN_NAME_PREFIX},
    {".model", "tokenizer", SCAN_NAME_SUFFIX},
    {"readme", "readme", SCAN_NAME_PREFIX},
    {"license", "license", SCAN_NAME_PREFIX},
    {"copying", "license", SCAN_NAME_PREFIX},
    {".json", "metadata", SCAN_NAME_SUFFIX},
    {".txt", "metadata", SCAN_NAME_SUFFIX},
    {".md", "metadata", SCAN_NAME_SUFFIX},
};

static int scan_name_matches(const char *name, const scan_kind_rule *rule) {
    size_t index;

    if (rule->match == SCAN_NAME_EXACT)
        return strcmp(name, rule->pattern) == 0;
    if (rule->match == SCAN_NAME_SUFFIX)
        return yvex_source_ends_with(name, rule->pattern);
    for (index = 0u; rule->pattern[index] != '\0'; ++index) {
        if (name[index] == '\0' ||
            tolower((unsigned char)name[index]) !=
                tolower((unsigned char)rule->pattern[index]))
            return 0;
    }
    return 1;
}

static const char *scan_kind_for_path(const char *rel_path) {
    const char *base = rel_path && rel_path[0] ? yvex_source_path_basename(rel_path) : "";
    size_t index;

    if (yvex_source_ends_with(base, ".safetensors"))
        return "safetensors";
    for (index = 0u; index < sizeof(scan_kind_rules) / sizeof(scan_kind_rules[0]); ++index)
        if (scan_name_matches(base, &scan_kind_rules[index]))
            return scan_kind_rules[index].kind;
    return "other";
}

static int scan_append_file(yvex_source_manifest_file_list *list,
                            const char *rel_path,
                            unsigned long long size_bytes,
                            yvex_error *err) {
    yvex_source_manifest_file *next;
    const char *kind;
    size_t new_cap;

    if (list->count == list->cap) {
        new_cap = list->cap == 0 ? 16u : list->cap * 2u;
        next = (yvex_source_manifest_file *)realloc(list->items, new_cap * sizeof(list->items[0]));
        if (!next) {
            yvex_error_set(
                err, YVEX_ERR_NOMEM, "source_manifest_scan", "file list allocation failed");
            return YVEX_ERR_NOMEM;
        }
        list->items = next;
        list->cap = new_cap;
    }

    kind = scan_kind_for_path(rel_path);
    list->items[list->count].path = yvex_core_strdup(rel_path);
    if (!list->items[list->count].path) {
        return scan_refuse(err, YVEX_ERR_NOMEM, "source_manifest_scan", "file path allocation failed");
    }
    list->items[list->count].size_bytes = size_bytes;
    list->items[list->count].kind = kind;
    list->count++;

    if (list->summary.file_count == ULLONG_MAX ||
        ULLONG_MAX - list->summary.total_size_bytes < size_bytes) {
        free(list->items[list->count - 1u].path);
        memset(&list->items[list->count - 1u], 0, sizeof(list->items[0]));
        list->count--;
        return scan_refuse(err, YVEX_ERR_BOUNDS, "source_manifest_scan", "source footprint overflow");
    }
    list->summary.file_count++;
    list->summary.total_size_bytes += size_bytes;
    if (strcmp(kind, "safetensors") == 0) {
        list->summary.safetensors_count++;
        list->summary.has_safetensors = 1;
    } else if (strcmp(kind, "config") == 0) {
        list->summary.has_config = 1;
    } else if (strcmp(kind, "tokenizer") == 0) {
        list->summary.has_tokenizer = 1;
    }
    return YVEX_OK;
}

static int scan_file_compare(const void *a, const void *b) {
    const yvex_source_manifest_file *fa = (const yvex_source_manifest_file *)a;
    const yvex_source_manifest_file *fb = (const yvex_source_manifest_file *)b;

    return strcmp(fa->path, fb->path);
}

static int scan_dir(const char *root,
                    const char *rel_dir,
                    int include_files,
                    yvex_source_manifest_file_list *out,
                    yvex_error *err) {
    char *abs_dir;
    DIR *dir;
    struct dirent *ent;
    int rc = YVEX_OK;

    abs_dir = rel_dir && rel_dir[0] != '\0' ? yvex_source_path_alloc(root, rel_dir) : yvex_core_strdup(root);
    if (!abs_dir) {
        yvex_error_set(
            err, YVEX_ERR_NOMEM, "source_manifest_scan", "directory path allocation failed");
        return YVEX_ERR_NOMEM;
    }

    dir = opendir(abs_dir);
    if (!dir) {
        yvex_error_setf(
            err, YVEX_ERR_IO, "source_manifest_scan", "cannot open directory: %s", abs_dir);
        free(abs_dir);
        return YVEX_ERR_IO;
    }

    for (;;) {
        char *rel_path;
        char *abs_path;
        struct stat st;

        errno = 0;
        ent = readdir(dir);
        if (!ent) {
            if (errno != 0 && rc == YVEX_OK) {
                yvex_error_setf(
                    err, YVEX_ERR_IO, "source_manifest_scan", "cannot read directory: %s", abs_dir);
                rc = YVEX_ERR_IO;
            }
            break;
        }

        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        rel_path = rel_dir && rel_dir[0] != '\0' ? yvex_source_path_alloc(rel_dir, ent->d_name)
                                                 : yvex_core_strdup(ent->d_name);
        if (!rel_path) {
            yvex_error_set(
                err, YVEX_ERR_NOMEM, "source_manifest_scan", "relative path allocation failed");
            rc = YVEX_ERR_NOMEM;
            break;
        }
        abs_path = yvex_source_path_alloc(root, rel_path);
        if (!abs_path) {
            free(rel_path);
            yvex_error_set(
                err, YVEX_ERR_NOMEM, "source_manifest_scan", "absolute path allocation failed");
            rc = YVEX_ERR_NOMEM;
            break;
        }

        if (lstat(abs_path, &st) != 0) {
            yvex_error_setf(
                err, YVEX_ERR_IO, "source_manifest_scan", "cannot stat path: %s", abs_path);
            free(abs_path);
            free(rel_path);
            rc = YVEX_ERR_IO;
            break;
        }

        if (S_ISDIR(st.st_mode)) {
            rc = scan_dir(root, rel_path, include_files, out, err);
        } else if (S_ISREG(st.st_mode)) {
            rc = scan_append_file(out, rel_path, (unsigned long long)st.st_size, err);
            if (!include_files && rc == YVEX_OK) {
                free(out->items[out->count - 1].path);
                out->items[out->count - 1].path = NULL;
                out->count--;
            }
        }

        free(abs_path);
        free(rel_path);
        if (rc != YVEX_OK) {
            break;
        }
    }

    if (closedir(dir) != 0 && rc == YVEX_OK) {
        yvex_error_setf(
            err, YVEX_ERR_IO, "source_manifest_scan", "cannot close directory: %s", abs_dir);
        rc = YVEX_ERR_IO;
    }
    free(abs_dir);
    return rc;
}

void yvex_source_manifest_file_list_init(yvex_source_manifest_file_list *list) {
    if (!list) {
        return;
    }
    memset(list, 0, sizeof(*list));
}

/*
 * Release resources owned by one source footprint object and clear its observable state.
 *
 * Releases only resources owned by source footprint scanning; cleanup remains deterministic.
 */
void yvex_source_manifest_file_list_free(yvex_source_manifest_file_list *list) {
    size_t i;

    if (!list) {
        return;
    }
    for (i = 0; i < list->count; ++i) {
        free(list->items[i].path);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

/* Enumerate deterministic source footprint rows beneath one admitted root. */
int yvex_source_manifest_scan_files(const char *local_path,
                                    int include_files,
                                    yvex_source_manifest_file_list *out,
                                    yvex_error *err) {
    struct stat st;
    int rc;

    if (!local_path || !out) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "source_manifest_scan", "local_path and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (lstat(local_path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        yvex_error_setf(err,
                        YVEX_ERR_IO,
                        "source_manifest_scan",
                        "local path is not a directory: %s",
                        local_path);
        return YVEX_ERR_IO;
    }

    rc = scan_dir(local_path, "", include_files, out, err);
    if (rc == YVEX_OK && include_files && out->count > 1) {
        qsort(out->items, out->count, sizeof(out->items[0]), scan_file_compare);
    }
    return rc;
}
