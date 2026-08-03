/* Registry entries describe locally available model references. Registry membership, selected
 * startup configuration, and the model currently open in yvexd remain independent facts. */
#ifndef YVEX_REGISTRY_H
#define YVEX_REGISTRY_H

#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model registry. */
typedef struct yvex_model_registry yvex_model_registry;

typedef struct {
    const char *alias;
    const char *family;
    const char *model;
    const char *scope;
    const char *artifact_class;
    const char *qprofile;
    const char *calibration;
    const char *producer;
    const char *schema_version;
    const char *path;
    const char *sha256;
    unsigned long long file_size;
    const char *format;
    const char *architecture;
    unsigned long long tensor_count;
    unsigned long long known_tensor_bytes;
    const char *primary_tensor_name;
    const char *primary_tensor_role;
    const char *primary_tensor_dtype;
    unsigned int primary_tensor_rank;
    const char *primary_tensor_dims;
    unsigned long long primary_tensor_bytes;
    const char *support_level;
    int selected_embedding_ready;
    unsigned long long selected_embedding_hidden_size;
    unsigned long long selected_embedding_vocab_size;
    unsigned long long selected_embedding_output_count;
    unsigned long long selected_embedding_slice_bytes;
    int execution_ready;
    const char *runtime_binding;
    const char *runtime_target;
    const char *runtime_backend;
    const char *runtime_mode;
    unsigned long long runtime_context;
} yvex_model_registry_entry;

#define YVEX_MODEL_METADATA_STATUS_CAP 24u
#define YVEX_MODEL_METADATA_CODE_CAP 64u
#define YVEX_MODEL_METADATA_VALUE_CAP 128u
#define YVEX_MODEL_METADATA_MAX_ISSUES 16u

typedef struct {
    char code[YVEX_MODEL_METADATA_CODE_CAP];
    char registered_value[YVEX_MODEL_METADATA_VALUE_CAP];
    char current_value[YVEX_MODEL_METADATA_VALUE_CAP];
} yvex_model_metadata_issue;

typedef struct {
    char metadata_status[YVEX_MODEL_METADATA_STATUS_CAP];
    char readiness_status[YVEX_MODEL_METADATA_STATUS_CAP];
    unsigned int issue_count;
    yvex_model_metadata_issue issues[YVEX_MODEL_METADATA_MAX_ISSUES];
} yvex_model_metadata_drift_report;

typedef struct {
    const char *registry_path;
    int create_if_missing;
} yvex_model_registry_options;

int yvex_model_registry_open(yvex_model_registry **out,
                             const yvex_model_registry_options *options,
                             yvex_error *err);

void yvex_model_registry_close(yvex_model_registry *registry);

unsigned long long yvex_model_registry_count(const yvex_model_registry *registry);

const yvex_model_registry_entry *yvex_model_registry_at(const yvex_model_registry *registry,
                                                        unsigned long long index);

const yvex_model_registry_entry *yvex_model_registry_find(const yvex_model_registry *registry,
                                                          const char *alias);

int yvex_model_registry_add(yvex_model_registry *registry,
                            const yvex_model_registry_entry *entry,
                            yvex_error *err);

int yvex_model_registry_remove(yvex_model_registry *registry,
                               const char *alias,
                               yvex_error *err);

int yvex_model_registry_save(const yvex_model_registry *registry,
                             const char *path,
                             yvex_error *err);

int yvex_model_alias_validate(const char *alias,
                              yvex_error *err);

int yvex_model_registry_entry_derive_from_path(yvex_model_registry_entry *entry,
                                               const char *path,
                                               yvex_error *err);

int yvex_model_registry_scan_root(const char *root,
                                  yvex_model_registry_entry **entries_out,
                                  unsigned long long *count_out,
                                  yvex_error *err);

void yvex_model_registry_scan_free(yvex_model_registry_entry *entries,
                                   unsigned long long count);

int yvex_model_registry_compare_metadata(
    const yvex_model_registry_entry *registered,
    const yvex_model_registry_entry *current,
    yvex_model_metadata_drift_report *out,
    yvex_error *err);

int yvex_model_registry_default_path(char *out,
                                     unsigned long long out_size,
                                     yvex_error *err);

/* A startup profile is selectable only when its artifact, binding, target, backend, and context
 * form one complete local configuration. Runtime admission still occurs inside yvexd. */
int yvex_model_registry_startup_validate(const yvex_model_registry_entry *entry,
                                         yvex_error *err);

/* Model references. */
typedef enum {
    YVEX_MODEL_REF_UNKNOWN = 0,
    YVEX_MODEL_REF_PATH,
    YVEX_MODEL_REF_ALIAS
} yvex_model_ref_kind;

typedef enum {
    YVEX_MODEL_REF_STATUS_UNKNOWN = 0,
    YVEX_MODEL_REF_STATUS_RESOLVED,
    YVEX_MODEL_REF_STATUS_NOT_FOUND,
    YVEX_MODEL_REF_STATUS_ALIAS_PATH_MISSING,
    YVEX_MODEL_REF_STATUS_REGISTRY_UNAVAILABLE,
    YVEX_MODEL_REF_STATUS_INVALID
} yvex_model_ref_status;

typedef struct {
    const char *registry_path;
    int allow_registry;
} yvex_model_ref_options;

typedef struct {
    yvex_model_ref_status status;
    yvex_model_ref_kind kind;
    const char *input;
    const char *path;
    const char *alias;
    const char *family;
    const char *sha256;
    unsigned long long registered_file_size;
    const char *support_level;
    const char *format;
    const char *architecture;
    unsigned long long tensor_count;
    unsigned long long known_tensor_bytes;
    const char *primary_tensor_name;
    const char *primary_tensor_role;
    const char *primary_tensor_dtype;
    unsigned int primary_tensor_rank;
    const char *primary_tensor_dims;
    unsigned long long primary_tensor_bytes;
    int selected_embedding_ready;
    unsigned long long selected_embedding_hidden_size;
    unsigned long long selected_embedding_vocab_size;
    unsigned long long selected_embedding_output_count;
    unsigned long long selected_embedding_slice_bytes;
    int execution_ready;
} yvex_model_ref;

typedef struct {
    yvex_model_registry_entry entry;
    char format[16];
    char architecture[64];
    char primary_tensor_name[128];
    char primary_tensor_role[64];
    char primary_tensor_dtype[32];
    char primary_tensor_dims[128];
    char support_level[64];
} yvex_model_metadata_snapshot;

int yvex_model_ref_resolve(yvex_model_ref *out,
                           const char *input,
                           const yvex_model_ref_options *options,
                           yvex_error *err);

void yvex_model_ref_clear(yvex_model_ref *ref);

int yvex_model_metadata_snapshot_read(yvex_model_metadata_snapshot *out,
                                      const char *path_or_alias,
                                      yvex_error *err);
void yvex_model_ref_registry_entry_view(const yvex_model_ref *ref,
                                        yvex_model_registry_entry *out);
#ifdef __cplusplus
}
#endif

#endif /* YVEX_REGISTRY_H */
