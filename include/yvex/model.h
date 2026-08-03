/* Model and tensor facts in this ABI are artifact-neutral. They describe admitted logical
 * structure but do not select a physical encoding, materialize payload bytes, or establish runtime
 * support. */
#ifndef YVEX_MODEL_H
#define YVEX_MODEL_H

#include <yvex/core.h>

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
