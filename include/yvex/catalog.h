/* Provider and local catalogs expose separate records that projections may join without
 * transferring source, package, or live-engine authority into one another. */
#ifndef YVEX_CATALOG_H
#define YVEX_CATALOG_H

#include <yvex/core.h>
#include <yvex/source.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_REMOTE_REPOSITORY_CAP 256u
#define YVEX_REMOTE_NAME_CAP 128u
#define YVEX_REMOTE_REVISION_CAP 128u
#define YVEX_REMOTE_FAMILY_CAP 64u
#define YVEX_REMOTE_FORMAT_CAP 32u
#define YVEX_REMOTE_PRECISION_CAP 96u
#define YVEX_REMOTE_REASON_CAP 192u
#define YVEX_REMOTE_MAX_REPRESENTATIONS 96u

typedef enum {
    YVEX_MODEL_REPRESENTATION_UNKNOWN = 0,
    YVEX_MODEL_REPRESENTATION_SAFETENSORS,
    YVEX_MODEL_REPRESENTATION_GGUF
} yvex_model_representation_kind;

typedef enum {
    YVEX_REMOTE_MODEL_UNKNOWN = 0,
    YVEX_REMOTE_MODEL_FULL,
    YVEX_REMOTE_MODEL_CONVERSION,
    YVEX_REMOTE_MODEL_ADAPTER,
    YVEX_REMOTE_MODEL_COMPONENT,
    YVEX_REMOTE_MODEL_DELTA,
    YVEX_REMOTE_MODEL_DERIVATIVE
} yvex_remote_model_kind;

typedef enum {
    YVEX_REMOTE_FILE_UNKNOWN = 0,
    YVEX_REMOTE_FILE_SAFETENSORS,
    YVEX_REMOTE_FILE_GGUF,
    YVEX_REMOTE_FILE_CONFIGURATION,
    YVEX_REMOTE_FILE_TOKENIZER,
    YVEX_REMOTE_FILE_SIDECAR
} yvex_remote_file_kind;

typedef enum {
    YVEX_MODEL_SUPPORT_REMOTE_ONLY = 0,
    YVEX_MODEL_SUPPORT_ARCHITECTURE_RECOGNIZED,
    YVEX_MODEL_SUPPORT_SOURCE_INGEST,
    YVEX_MODEL_SUPPORT_SEMANTIC_FAMILY,
    YVEX_MODEL_SUPPORT_PHYSICAL_INSPECTION,
    YVEX_MODEL_SUPPORT_PACKAGE_PREPARATION
} yvex_model_support_stage;

typedef struct {
    char identity[YVEX_REMOTE_NAME_CAP];
    char format[YVEX_REMOTE_FORMAT_CAP];
    char precision[YVEX_REMOTE_PRECISION_CAP];
    char precision_evidence[32];
    char file_pattern[YVEX_REMOTE_REPOSITORY_CAP];
    char compatibility[YVEX_REMOTE_REASON_CAP];
    char recommendation[YVEX_REMOTE_REASON_CAP];
    yvex_model_representation_kind kind;
    unsigned long long file_count;
    unsigned long long size_bytes;
    int size_known;
    int provisional;
    int source_ingest_supported;
    int package_preparation_supported;
    int direct_admission_requires_inspection;
} yvex_model_representation;

typedef struct {
    char path[YVEX_REMOTE_REPOSITORY_CAP];
    char representation[YVEX_REMOTE_NAME_CAP];
    unsigned long long size_bytes;
    yvex_remote_file_kind kind;
    int size_known;
} yvex_remote_file;

typedef struct {
    char provider[YVEX_ACCOUNT_PROVIDER_CAP];
    char repository[YVEX_REMOTE_REPOSITORY_CAP];
    char author[YVEX_REMOTE_NAME_CAP];
    char revision_reference[YVEX_REMOTE_REVISION_CAP];
    char resolved_revision[YVEX_REMOTE_REVISION_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    char family_evidence[32];
    char model_identity[YVEX_REMOTE_REPOSITORY_CAP];
    char kind_evidence[32];
    char architecture[YVEX_REMOTE_NAME_CAP];
    char pipeline[YVEX_REMOTE_NAME_CAP];
    char base_model[YVEX_REMOTE_REPOSITORY_CAP];
    char lineage_relation[32];
    char support_reason[YVEX_REMOTE_REASON_CAP];
    unsigned long long parameter_count;
    yvex_remote_model_kind kind;
    yvex_model_support_stage support_stage;
    unsigned int ranking_score;
    unsigned int provider_rank;
    unsigned int representation_count;
    unsigned int available_file_count;
    int parameter_count_known;
    int gated;
    int gated_known;
    int canonical;
    int kind_provisional;
} yvex_remote_model;

typedef struct yvex_remote_catalog yvex_remote_catalog;

typedef struct {
    yvex_account_provider provider;
    const char *query;
    const char *author;
    const char *filter;
    unsigned int page;
    unsigned int page_size;
} yvex_remote_search_options;

typedef struct {
    yvex_account_provider provider;
    const char *repository;
    const char *revision;
} yvex_remote_inspect_options;

int yvex_remote_model_search(yvex_remote_catalog **out,
                             const yvex_remote_search_options *options,
                             yvex_error *err);
int yvex_remote_model_inspect(yvex_remote_catalog **out,
                              const yvex_remote_inspect_options *options,
                              yvex_error *err);
void yvex_remote_catalog_close(yvex_remote_catalog *catalog);
unsigned long long yvex_remote_catalog_count(const yvex_remote_catalog *catalog);
const yvex_remote_model *yvex_remote_catalog_at(const yvex_remote_catalog *catalog,
                                                unsigned long long index);
const yvex_model_representation *yvex_remote_catalog_representation_at(
    const yvex_remote_catalog *catalog, unsigned long long model_index,
    unsigned int representation_index);
const yvex_remote_file *yvex_remote_catalog_file_at(
    const yvex_remote_catalog *catalog, unsigned long long model_index,
    unsigned int file_index);
const char *yvex_model_representation_kind_name(yvex_model_representation_kind kind);
const char *yvex_remote_file_kind_name(yvex_remote_file_kind kind);
const char *yvex_model_support_stage_name(yvex_model_support_stage stage);

typedef struct {
    char name[YVEX_REMOTE_NAME_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    char provider[YVEX_ACCOUNT_PROVIDER_CAP];
    char repository[YVEX_REMOTE_REPOSITORY_CAP];
    char revision[YVEX_REMOTE_REVISION_CAP];
    char representation[YVEX_REMOTE_PRECISION_CAP];
    char acquisition_state[32];
    char verification_state[32];
    char blocker[YVEX_REMOTE_REASON_CAP];
    char path[YVEX_PATH_CAP];
    unsigned long long size_bytes;
    int size_known;
} yvex_local_source_record;

typedef struct {
    char name[YVEX_REMOTE_NAME_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    char repository[YVEX_REMOTE_REPOSITORY_CAP];
    char revision[YVEX_REMOTE_REVISION_CAP];
    char representation[YVEX_REMOTE_PRECISION_CAP];
    char package_state[32];
    char verification_state[32];
    char backend[32];
    char blocker[YVEX_REMOTE_REASON_CAP];
    char path[YVEX_PATH_CAP];
    unsigned long long size_bytes;
    int size_known;
    int ready;
} yvex_local_package_record;

typedef struct yvex_local_catalog yvex_local_catalog;
typedef struct {
    const char *models_root;
    const char *registry_path;
} yvex_local_catalog_options;

int yvex_local_catalog_open(yvex_local_catalog **out,
                            const yvex_local_catalog_options *options,
                            yvex_error *err);
void yvex_local_catalog_close(yvex_local_catalog *catalog);
unsigned long long yvex_local_catalog_source_count(const yvex_local_catalog *catalog);
const yvex_local_source_record *yvex_local_catalog_source_at(
    const yvex_local_catalog *catalog, unsigned long long index);
unsigned long long yvex_local_catalog_package_count(const yvex_local_catalog *catalog);
const yvex_local_package_record *yvex_local_catalog_package_at(
    const yvex_local_catalog *catalog, unsigned long long index);

#ifdef __cplusplus
}
#endif
#endif /* YVEX_CATALOG_H */
