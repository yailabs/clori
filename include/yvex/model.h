/* Model and tensor facts in this ABI are artifact-neutral. They describe admitted logical
 * structure but do not select a physical encoding, materialize payload bytes, or establish runtime
 * support. */
#ifndef YVEX_MODEL_H
#define YVEX_MODEL_H

#include <yvex/core.h>
#include <yvex/source.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_gguf yvex_gguf;
typedef struct yvex_backend yvex_backend;
typedef struct yvex_device_tensor yvex_device_tensor;

/* Dtype geometry. */
typedef enum {
    YVEX_DTYPE_UNKNOWN = 0,

    YVEX_DTYPE_F32,
    YVEX_DTYPE_F16,
    YVEX_DTYPE_BF16,
    YVEX_DTYPE_F64,

    YVEX_DTYPE_I8,
    YVEX_DTYPE_I16,
    YVEX_DTYPE_I32,
    YVEX_DTYPE_I64,

    YVEX_DTYPE_Q4_0,
    YVEX_DTYPE_Q4_1,
    YVEX_DTYPE_Q5_0,
    YVEX_DTYPE_Q5_1,
    YVEX_DTYPE_Q8_0,
    YVEX_DTYPE_Q8_1,

    YVEX_DTYPE_Q2_K,
    YVEX_DTYPE_Q3_K,
    YVEX_DTYPE_Q4_K,
    YVEX_DTYPE_Q5_K,
    YVEX_DTYPE_Q6_K,
    YVEX_DTYPE_Q8_K,

    YVEX_DTYPE_IQ2_XXS,
    YVEX_DTYPE_IQ2_XS,
    YVEX_DTYPE_IQ3_XXS,
    YVEX_DTYPE_IQ1_S,
    YVEX_DTYPE_IQ4_NL,
    YVEX_DTYPE_IQ3_S,
    YVEX_DTYPE_IQ2_S,
    YVEX_DTYPE_IQ4_XS,
    YVEX_DTYPE_IQ1_M,

    YVEX_DTYPE_TQ1_0,
    YVEX_DTYPE_TQ2_0,
    YVEX_DTYPE_MXFP4
} yvex_dtype;

typedef struct {
    yvex_dtype dtype;
    unsigned int ggml_type;
} yvex_dtype_info;

const yvex_dtype_info *yvex_dtype_get_info(yvex_dtype dtype);
const yvex_dtype_info *yvex_dtype_from_ggml_type(unsigned int ggml_type);
const char *yvex_dtype_name(yvex_dtype dtype);
int yvex_dtype_is_quantized(yvex_dtype dtype);
int yvex_dtype_storage_supported(yvex_dtype dtype);

int yvex_dtype_tensor_storage_bytes(yvex_dtype dtype,
                                    const unsigned long long *dims,
                                    unsigned int rank,
                                    unsigned long long *out,
                                    yvex_error *err);

/* The compatibility form interprets element_count as one logical row only. */
int yvex_dtype_storage_bytes(yvex_dtype dtype,
                             unsigned long long row_element_count,
                             unsigned long long *out,
                             yvex_error *err);

/* Tensor inventory. */
#define YVEX_TENSOR_MAX_DIMS 4u

typedef enum {
    YVEX_TENSOR_ROLE_UNKNOWN = 0,
    YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
    YVEX_TENSOR_ROLE_OUTPUT_NORM,
    YVEX_TENSOR_ROLE_OUTPUT_HEAD,
    YVEX_TENSOR_ROLE_ATTENTION_NORM,
    YVEX_TENSOR_ROLE_ATTENTION_Q,
    YVEX_TENSOR_ROLE_ATTENTION_K,
    YVEX_TENSOR_ROLE_ATTENTION_V,
    YVEX_TENSOR_ROLE_ATTENTION_OUT,
    YVEX_TENSOR_ROLE_FFN_NORM,
    YVEX_TENSOR_ROLE_FFN_GATE,
    YVEX_TENSOR_ROLE_FFN_UP,
    YVEX_TENSOR_ROLE_FFN_DOWN,
    YVEX_TENSOR_ROLE_MOE_ROUTER,
    YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
    YVEX_TENSOR_ROLE_MOE_EXPERT_UP,
    YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
    YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION,
    YVEX_TENSOR_ROLE_HC_HEAD_BASE,
    YVEX_TENSOR_ROLE_HC_HEAD_SCALE,
    YVEX_TENSOR_ROLE_ATTENTION_SINKS,
    YVEX_TENSOR_ROLE_ATTENTION_Q_A,
    YVEX_TENSOR_ROLE_ATTENTION_Q_B,
    YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM,
    YVEX_TENSOR_ROLE_ATTENTION_KV,
    YVEX_TENSOR_ROLE_ATTENTION_KV_NORM,
    YVEX_TENSOR_ROLE_ATTENTION_OUT_A,
    YVEX_TENSOR_ROLE_ATTENTION_OUT_B,
    YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION,
    YVEX_TENSOR_ROLE_HC_ATTENTION_BASE,
    YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE,
    YVEX_TENSOR_ROLE_HC_FFN_FUNCTION,
    YVEX_TENSOR_ROLE_HC_FFN_BASE,
    YVEX_TENSOR_ROLE_HC_FFN_SCALE,
    YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV,
    YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE,
    YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE,
    YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM,
    YVEX_TENSOR_ROLE_INDEXER_PROJECTION,
    YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B,
    YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV,
    YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE,
    YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE,
    YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM,
    YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS,
    YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE,
    YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE,
    YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP,
    YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN,
    YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION,
    YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM,
    YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM,
    YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING,
    YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
    YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE,
    YVEX_TENSOR_ROLE_COUNT
} yvex_tensor_role;

/* Artifact-neutral tensor ownership projected by family recipes. */
typedef enum {
    YVEX_TENSOR_COLLECTION_GLOBAL = 0,
    YVEX_TENSOR_COLLECTION_ATTENTION,
    YVEX_TENSOR_COLLECTION_COMPRESSOR,
    YVEX_TENSOR_COLLECTION_INDEXER,
    YVEX_TENSOR_COLLECTION_NORM,
    YVEX_TENSOR_COLLECTION_MHC,
    YVEX_TENSOR_COLLECTION_ROUTER,
    YVEX_TENSOR_COLLECTION_ROUTED_EXPERT,
    YVEX_TENSOR_COLLECTION_SHARED_EXPERT,
    YVEX_TENSOR_COLLECTION_AUXILIARY,
    YVEX_TENSOR_COLLECTION_COUNT
} yvex_tensor_collection;

/* Artifact-neutral model scope for a logical tensor. */
typedef enum {
    YVEX_TENSOR_SCOPE_GLOBAL = 0,
    YVEX_TENSOR_SCOPE_MAIN_LAYER,
    YVEX_TENSOR_SCOPE_DRAFT
} yvex_tensor_scope;

typedef struct {
    const char *name;
    unsigned int rank;
    unsigned long long dims[YVEX_TENSOR_MAX_DIMS];
    yvex_dtype dtype;
    unsigned int ggml_type;
    yvex_tensor_role role;
    unsigned long long element_count;
    unsigned long long storage_bytes;
    unsigned long long relative_offset;
    unsigned long long absolute_offset;
} yvex_tensor_info;

typedef struct yvex_tensor_table yvex_tensor_table;

int yvex_tensor_table_from_gguf(yvex_tensor_table **out,
                                const yvex_gguf *gguf,
                                yvex_error *err);

void yvex_tensor_table_close(yvex_tensor_table *table);

unsigned long long yvex_tensor_table_count(const yvex_tensor_table *table);
const yvex_tensor_info *yvex_tensor_table_at(const yvex_tensor_table *table,
                                             unsigned long long index);
const yvex_tensor_info *yvex_tensor_table_find(const yvex_tensor_table *table,
                                               const char *name);

const char *yvex_tensor_role_name(yvex_tensor_role role);
yvex_tensor_role yvex_tensor_role_classify(const char *architecture,
                                           const char *tensor_name,
                                           unsigned int rank,
                                           const unsigned long long *dims,
                                           yvex_dtype dtype);

/* Model descriptor. */
typedef enum {
    YVEX_ARCH_UNKNOWN = 0,
    YVEX_ARCH_LLAMA,
    YVEX_ARCH_QWEN,
    YVEX_ARCH_DEEPSEEK,
    YVEX_ARCH_GEMMA,
    YVEX_ARCH_PHI,
    YVEX_ARCH_KIMI,
    YVEX_ARCH_GLM
} yvex_arch;

typedef struct yvex_model_descriptor yvex_model_descriptor;
typedef struct yvex_artifact yvex_artifact;
typedef struct yvex_tokenizer yvex_tokenizer;

/*
 * Owns one admitted, read-only artifact/model view.  The pointers are exposed
 * as borrowed views so graph, tokenizer, and diagnostic consumers share one
 * lifecycle instead of rebuilding this stack in CLI translation units. */
typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *table;
    yvex_model_descriptor *model;
    yvex_tokenizer *tokenizer;
} yvex_model_context;

int yvex_model_descriptor_from_gguf(yvex_model_descriptor **out,
                                    const yvex_gguf *gguf,
                                    const yvex_tensor_table *tensors,
                                    yvex_error *err);

void yvex_model_descriptor_close(yvex_model_descriptor *model);

yvex_arch yvex_model_arch(const yvex_model_descriptor *model);
const char *yvex_arch_name(yvex_arch arch);

const char *yvex_model_name(const yvex_model_descriptor *model);
unsigned long long yvex_model_context_length(const yvex_model_descriptor *model);
unsigned long long yvex_model_tensor_count(const yvex_model_descriptor *model);
unsigned long long yvex_model_total_storage_bytes(const yvex_model_descriptor *model);
unsigned long long yvex_model_unsupported_tensor_accounting_count(const yvex_model_descriptor *model);
unsigned long long yvex_model_role_count(const yvex_model_descriptor *model, yvex_tensor_role role);

int yvex_model_context_open(const char *path_or_alias,
                            yvex_model_context *out,
                            yvex_error *err);
int yvex_model_context_open_tokenizer(const char *path_or_alias,
                                      yvex_model_context *out,
                                      yvex_error *err);
void yvex_model_context_close(yvex_model_context *context);
int yvex_model_context_vocab_size(const char *path_or_alias,
                                  unsigned long long *out_vocab_size,
                                  yvex_error *err);

/* Provider discovery describes remote availability without promoting local or runtime state. */
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
    YVEX_MODEL_SUPPORT_PACKAGE_PREPARATION,
    YVEX_MODEL_SUPPORT_PACKAGE_READY
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
    int local;
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
    char engine_state[32];
    char local_source_revision[YVEX_REMOTE_REVISION_CAP];
    char local_package_revision[YVEX_REMOTE_REVISION_CAP];
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
    int local;
    int local_source;
    int local_package;
    int local_related_revision;
    int canonical;
    int kind_provisional;
} yvex_remote_model;

typedef struct yvex_remote_catalog yvex_remote_catalog;

typedef struct {
    yvex_account_provider provider;
    const char *query;
    const char *author;
    const char *filter;
    const char *cli_override;
    const char *models_root;
    unsigned int page;
    unsigned int page_size;
} yvex_remote_search_options;

typedef struct {
    yvex_account_provider provider;
    const char *repository;
    const char *revision;
    const char *cli_override;
    const char *models_root;
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
    const yvex_remote_catalog *catalog,
    unsigned long long model_index,
    unsigned int representation_index);
const yvex_remote_file *yvex_remote_catalog_file_at(const yvex_remote_catalog *catalog,
                                                    unsigned long long model_index,
                                                    unsigned int file_index);
const char *yvex_model_representation_kind_name(yvex_model_representation_kind kind);
const char *yvex_remote_file_kind_name(yvex_remote_file_kind kind);
const char *yvex_model_support_stage_name(yvex_model_support_stage stage);

/* The local catalog projects acquired source and package state without opening an engine. */
typedef enum {
    YVEX_LOCAL_MODEL_ACQUIRED_SOURCE = 0,
    YVEX_LOCAL_MODEL_PACKAGE
} yvex_local_model_kind;

typedef struct {
    char name[YVEX_REMOTE_NAME_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    char provider[YVEX_ACCOUNT_PROVIDER_CAP];
    char repository[YVEX_REMOTE_REPOSITORY_CAP];
    char revision[YVEX_REMOTE_REVISION_CAP];
    char representation[YVEX_REMOTE_PRECISION_CAP];
    char package_state[32];
    char verification_state[32];
    char engine_state[32];
    char backend[32];
    char blocker[YVEX_REMOTE_REASON_CAP];
    char path[YVEX_PATH_CAP];
    unsigned long long size_bytes;
    yvex_local_model_kind kind;
    int size_known;
    int package_ready;
} yvex_local_model;

typedef struct yvex_local_model_catalog yvex_local_model_catalog;
typedef struct {
    const char *models_root;
    const char *registry_path;
} yvex_local_catalog_options;
int yvex_local_model_catalog_open(yvex_local_model_catalog **out,
                                  const yvex_local_catalog_options *options,
                                  yvex_error *err);
void yvex_local_model_catalog_close(yvex_local_model_catalog *catalog);
unsigned long long yvex_local_model_catalog_count(const yvex_local_model_catalog *catalog);
const yvex_local_model *yvex_local_model_catalog_at(const yvex_local_model_catalog *catalog,
                                                    unsigned long long index);

/* Materialized weights. */
typedef struct yvex_weight_table yvex_weight_table;
typedef struct yvex_materialized_weight yvex_materialized_weight;

typedef enum {
    YVEX_WEIGHT_STATUS_EMPTY = 0,
    YVEX_WEIGHT_STATUS_MATERIALIZED,
    YVEX_WEIGHT_STATUS_PARTIAL,
    YVEX_WEIGHT_STATUS_FAILED
} yvex_weight_status;

typedef enum {
    YVEX_WEIGHT_RESIDENCY_HOST = 0,
    YVEX_WEIGHT_RESIDENCY_CPU_BACKEND,
    YVEX_WEIGHT_RESIDENCY_CUDA_BACKEND
} yvex_weight_residency;

typedef struct {
    const char *backend_name;
    int require_all_tensors;
    int allow_unsupported_dtype;
} yvex_materialize_options;

typedef struct {
    yvex_weight_status status;
    const char *backend_name;
    const char *materialization_gate;
    const char *materialization_phase;
    const char *shape_status;
    const char *range_status;
    const char *backend_status;
    const char *cleanup_status;
    int allocation_attempted;
    int transfer_attempted;
    int cleanup_attempted;
    unsigned long long tensors_total;
    unsigned long long tensors_materialized;
    unsigned long long tensors_failed;
    unsigned long long bytes_total;
    unsigned long long bytes_materialized;
    unsigned long long backend_allocated_bytes;
    unsigned long long bytes_planned;
    unsigned long long bytes_allocated;
    unsigned long long bytes_transferred;
    int execution_ready;
} yvex_materialize_summary;

int yvex_weight_table_materialize(yvex_weight_table **out,
                                  const yvex_artifact *artifact,
                                  const yvex_gguf *gguf,
                                  const yvex_tensor_table *tensors,
                                  yvex_backend *backend,
                                  const yvex_materialize_options *options,
                                  yvex_error *err);

void yvex_weight_table_close(yvex_weight_table *weights);

unsigned long long yvex_weight_table_count(const yvex_weight_table *weights);
const yvex_materialized_weight *yvex_weight_table_at(const yvex_weight_table *weights,
                                                     unsigned long long index);
const yvex_materialized_weight *yvex_weight_table_find(const yvex_weight_table *weights,
                                                       const char *name);

int yvex_weight_table_get_summary(const yvex_weight_table *weights,
                                  yvex_materialize_summary *out,
                                  yvex_error *err);

const char *yvex_weight_status_name(yvex_weight_status status);
const char *yvex_weight_residency_name(yvex_weight_residency residency);

const char *yvex_weight_name(const yvex_materialized_weight *weight);
yvex_dtype yvex_weight_dtype(const yvex_materialized_weight *weight);
yvex_tensor_role yvex_weight_role(const yvex_materialized_weight *weight);
unsigned long long yvex_weight_bytes(const yvex_materialized_weight *weight);
yvex_weight_residency yvex_weight_residency_of(const yvex_materialized_weight *weight);
const yvex_device_tensor *yvex_weight_device_tensor(const yvex_materialized_weight *weight);

/* Graph operation vocabulary. */
#define YVEX_GRAPH_MAX_DIMS 4u

typedef enum {
    YVEX_VALUE_TOKEN_IDS = 0,
    YVEX_VALUE_ACTIVATION,
    YVEX_VALUE_WEIGHT,
    YVEX_VALUE_KV_CACHE,
    YVEX_VALUE_LOGITS,
    YVEX_VALUE_TEMPORARY,
    YVEX_VALUE_UNKNOWN
} yvex_value_kind;

typedef enum {
    YVEX_RESIDENCY_HOST = 0,
    YVEX_RESIDENCY_DEVICE,
    YVEX_RESIDENCY_BACKEND_DECIDES
} yvex_residency;

typedef struct {
    unsigned int id;
    yvex_value_kind kind;
    const char *name;
    unsigned int rank;
    unsigned long long dims[YVEX_GRAPH_MAX_DIMS];
    yvex_dtype dtype;
    yvex_residency residency;
    const char *source_tensor_name;
} yvex_graph_value_info;

typedef enum {
    YVEX_OP_EMBED = 0,
    YVEX_OP_RMS_NORM,
    YVEX_OP_MATMUL,
    YVEX_OP_ROPE,
    YVEX_OP_ATTENTION_PREFILL,
    YVEX_OP_ATTENTION_DECODE,
    YVEX_OP_KV_WRITE,
    YVEX_OP_KV_READ,
    YVEX_OP_SWIGLU,
    YVEX_OP_RESIDUAL_ADD,
    YVEX_OP_LOGITS,
    YVEX_OP_SAMPLER,
    YVEX_OP_UNSUPPORTED
} yvex_op_kind;

typedef enum {
    YVEX_OP_STATUS_PLANNED = 0,
    YVEX_OP_STATUS_MISSING_INPUT,
    YVEX_OP_STATUS_UNSUPPORTED,
    YVEX_OP_STATUS_INVALID_SHAPE
} yvex_op_status;

typedef struct {
    unsigned int id;
    yvex_op_kind kind;
    yvex_op_status status;
    const char *name;
    unsigned int input_count;
    unsigned int output_count;
    const char *reason;
} yvex_graph_op_info;

#ifdef __cplusplus
}
#endif

#endif /* YVEX_MODEL_H */
