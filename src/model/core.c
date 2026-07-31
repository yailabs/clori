/*
 * Own model descriptors, tensor tables, and materialized weight lifecycles.
 *
 * Model facts are built from parsed GGUF and tensor metadata; tensor payload bytes are read only
 * through explicit materialization paths; domain code returns facts and errors, not operator
 * prose. Model metadata/materialization facts are not model support, runtime support, generation
 * support, eval evidence, benchmark evidence, or release readiness.
 */

#include <yvex/model.h>
#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/qtype.h>
#include <yvex/registry.h>
#include <yvex/source.h>
#include <yvex/tokenizer.h>
#include <yvex/internal/core.h>

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Dtype registry */

static const yvex_dtype_info dtype_table[] = {
    {YVEX_DTYPE_UNKNOWN, UINT_MAX},
    {YVEX_DTYPE_F32, YVEX_GGUF_QTYPE_F32},
    {YVEX_DTYPE_F16, YVEX_GGUF_QTYPE_F16},
    {YVEX_DTYPE_BF16, YVEX_GGUF_QTYPE_BF16},
    {YVEX_DTYPE_F64, YVEX_GGUF_QTYPE_F64},
    {YVEX_DTYPE_I8, YVEX_GGUF_QTYPE_I8},
    {YVEX_DTYPE_I16, YVEX_GGUF_QTYPE_I16},
    {YVEX_DTYPE_I32, YVEX_GGUF_QTYPE_I32},
    {YVEX_DTYPE_I64, YVEX_GGUF_QTYPE_I64},
    {YVEX_DTYPE_Q4_0, YVEX_GGUF_QTYPE_Q4_0},
    {YVEX_DTYPE_Q4_1, YVEX_GGUF_QTYPE_Q4_1},
    {YVEX_DTYPE_Q5_0, YVEX_GGUF_QTYPE_Q5_0},
    {YVEX_DTYPE_Q5_1, YVEX_GGUF_QTYPE_Q5_1},
    {YVEX_DTYPE_Q8_0, YVEX_GGUF_QTYPE_Q8_0},
    {YVEX_DTYPE_Q8_1, YVEX_GGUF_QTYPE_Q8_1},
    {YVEX_DTYPE_Q2_K, YVEX_GGUF_QTYPE_Q2_K},
    {YVEX_DTYPE_Q3_K, YVEX_GGUF_QTYPE_Q3_K},
    {YVEX_DTYPE_Q4_K, YVEX_GGUF_QTYPE_Q4_K},
    {YVEX_DTYPE_Q5_K, YVEX_GGUF_QTYPE_Q5_K},
    {YVEX_DTYPE_Q6_K, YVEX_GGUF_QTYPE_Q6_K},
    {YVEX_DTYPE_Q8_K, YVEX_GGUF_QTYPE_Q8_K},
    {YVEX_DTYPE_IQ2_XXS, YVEX_GGUF_QTYPE_IQ2_XXS},
    {YVEX_DTYPE_IQ2_XS, YVEX_GGUF_QTYPE_IQ2_XS},
    {YVEX_DTYPE_IQ3_XXS, YVEX_GGUF_QTYPE_IQ3_XXS},
    {YVEX_DTYPE_IQ1_S, YVEX_GGUF_QTYPE_IQ1_S},
    {YVEX_DTYPE_IQ4_NL, YVEX_GGUF_QTYPE_IQ4_NL},
    {YVEX_DTYPE_IQ3_S, YVEX_GGUF_QTYPE_IQ3_S},
    {YVEX_DTYPE_IQ2_S, YVEX_GGUF_QTYPE_IQ2_S},
    {YVEX_DTYPE_IQ4_XS, YVEX_GGUF_QTYPE_IQ4_XS},
    {YVEX_DTYPE_IQ1_M, YVEX_GGUF_QTYPE_IQ1_M},
    {YVEX_DTYPE_TQ1_0, YVEX_GGUF_QTYPE_TQ1_0},
    {YVEX_DTYPE_TQ2_0, YVEX_GGUF_QTYPE_TQ2_0},
    {YVEX_DTYPE_MXFP4, YVEX_GGUF_QTYPE_MXFP4},
};

static const unsigned long dtype_table_count = sizeof(dtype_table) / sizeof(dtype_table[0]);

typedef struct {
    const char *name;
    yvex_arch value;
} architecture_name;

static const architecture_name architecture_names[] = {
    {"llama", YVEX_ARCH_LLAMA}, {"qwen", YVEX_ARCH_QWEN},
    {"deepseek", YVEX_ARCH_DEEPSEEK}, {"gemma", YVEX_ARCH_GEMMA},
    {"phi", YVEX_ARCH_PHI}, {"kimi", YVEX_ARCH_KIMI}, {"glm", YVEX_ARCH_GLM}
};

const yvex_dtype_info *yvex_dtype_get_info(yvex_dtype dtype)
{
    unsigned long i;

    for (i = 0; i < dtype_table_count; ++i) {
        if (dtype_table[i].dtype == dtype) {
            return &dtype_table[i];
        }
    }

    return &dtype_table[0];
}

const yvex_dtype_info *yvex_dtype_from_ggml_type(unsigned int ggml_type)
{
    unsigned long i;

    for (i = 0; i < dtype_table_count; ++i) {
        if (dtype_table[i].ggml_type == ggml_type) {
            return &dtype_table[i];
        }
    }

    return &dtype_table[0];
}

const char *yvex_dtype_name(yvex_dtype dtype)
{
    const yvex_dtype_info *info = yvex_dtype_get_info(dtype);

    return info->dtype == YVEX_DTYPE_UNKNOWN
        ? "UNKNOWN"
        : yvex_gguf_qtype_name(info->ggml_type);
}

int yvex_dtype_is_quantized(yvex_dtype dtype)
{
    const yvex_dtype_info *info = yvex_dtype_get_info(dtype);
    const yvex_gguf_qtype_geometry *geometry =
        yvex_gguf_qtype_geometry_find(info->ggml_type);

    return geometry &&
        geometry->storage_class == YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED;
}

int yvex_dtype_storage_supported(yvex_dtype dtype)
{
    const yvex_dtype_info *info = yvex_dtype_get_info(dtype);

    return info->dtype != YVEX_DTYPE_UNKNOWN &&
        yvex_gguf_qtype_supported_for_storage(info->ggml_type, NULL);
}

static int dtype_storage_error_code(yvex_gguf_qtype_storage_status status)
{
    switch (status) {
    case YVEX_GGUF_QTYPE_STORAGE_OK:
        return YVEX_OK;
    case YVEX_GGUF_QTYPE_STORAGE_INVALID_ARGUMENT:
        return YVEX_ERR_INVALID_ARG;
    case YVEX_GGUF_QTYPE_STORAGE_INVALID_RANK:
    case YVEX_GGUF_QTYPE_STORAGE_INVALID_DIMENSION:
    case YVEX_GGUF_QTYPE_STORAGE_ROW_BLOCK_MISMATCH:
    case YVEX_GGUF_QTYPE_STORAGE_EXPECTED_ACTUAL_MISMATCH:
        return YVEX_ERR_FORMAT;
    case YVEX_GGUF_QTYPE_STORAGE_ELEMENT_COUNT_OVERFLOW:
    case YVEX_GGUF_QTYPE_STORAGE_ROW_BYTE_OVERFLOW:
    case YVEX_GGUF_QTYPE_STORAGE_ROW_COUNT_OVERFLOW:
    case YVEX_GGUF_QTYPE_STORAGE_TOTAL_BYTE_OVERFLOW:
        return YVEX_ERR_BOUNDS;
    case YVEX_GGUF_QTYPE_STORAGE_UNKNOWN_ID:
    case YVEX_GGUF_QTYPE_STORAGE_REMOVED_ID:
    case YVEX_GGUF_QTYPE_STORAGE_RESERVED_ID:
    case YVEX_GGUF_QTYPE_STORAGE_OUTSIDE_BASELINE:
    case YVEX_GGUF_QTYPE_STORAGE_GEOMETRY_UNAVAILABLE:
        return YVEX_ERR_UNSUPPORTED;
    }
    return YVEX_ERR_UNSUPPORTED;
}

int yvex_dtype_tensor_storage_bytes(yvex_dtype dtype,
                                    const unsigned long long *dims,
                                    unsigned int rank,
                                    unsigned long long *out,
                                    yvex_error *err)
{
    const yvex_dtype_info *info;
    yvex_gguf_qtype_storage_result result;
    yvex_gguf_qtype_storage_status status;
    int rc;

    if (!out || !dims) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "yvex_dtype_tensor_storage_bytes",
                       "dims and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = 0ull;
    info = yvex_dtype_get_info(dtype);
    if (info->dtype == YVEX_DTYPE_UNKNOWN) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED,
                       "yvex_dtype_tensor_storage_bytes",
                       "unknown dtype has no GGUF storage identity");
        return YVEX_ERR_UNSUPPORTED;
    }
    status = yvex_gguf_qtype_tensor_storage(info->ggml_type, dims, rank, &result);
    rc = dtype_storage_error_code(status);
    if (rc != YVEX_OK) {
        yvex_error_setf(err, rc, "yvex_dtype_tensor_storage_bytes",
                        "%s: %s",
                        yvex_gguf_qtype_storage_status_name(status),
                        result.reason ? result.reason : "qtype storage refused");
        return rc;
    }
    *out = result.total_bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_dtype_storage_bytes(yvex_dtype dtype,
                             unsigned long long row_element_count,
                             unsigned long long *out,
                             yvex_error *err)
{
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "yvex_dtype_storage_bytes", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_dtype_tensor_storage_bytes(dtype, &row_element_count, 1u, out, err);
}

/* Model descriptors */

struct yvex_model_descriptor {
    yvex_arch arch;
    char *name;
    unsigned long long context_length;
    unsigned long long tensor_count;
    unsigned long long total_storage_bytes;
    unsigned long long unsupported_tensor_accounting_count;
    unsigned long long role_counts[YVEX_TENSOR_ROLE_COUNT];
};

static char *copy_bytes_string(const char *data, unsigned long long len)
{
    char *copy;

    if (!data) {
        return NULL;
    }
    if (len > (unsigned long long)(SIZE_MAX - 1)) {
        return NULL;
    }
    copy = (char *)malloc((size_t)len + 1u);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, data, (size_t)len);
    copy[len] = '\0';
    return copy;
}

static yvex_arch arch_from_name(const char *name)
{
    size_t index;

    if (!name) return YVEX_ARCH_UNKNOWN;
    for (index = 0u; index < sizeof(architecture_names) / sizeof(architecture_names[0]);
         ++index) {
        if (strcmp(name, architecture_names[index].name) == 0)
            return architecture_names[index].value;
    }
    return YVEX_ARCH_UNKNOWN;
}

/*
 * Build a model descriptor from parsed GGUF metadata and tensor-table facts.
 *
 * Returns invalid-arg or allocation failures and releases partial descriptor ownership before
 * returning.
 */
int yvex_model_descriptor_from_gguf(yvex_model_descriptor **out,
                                    const yvex_gguf *gguf,
                                    const yvex_tensor_table *tensors,
                                    yvex_error *err)
{
    yvex_model_descriptor *model;
    const yvex_gguf_value *value;
    const char *text;
    unsigned long long len;
    unsigned long long i;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_model_descriptor_from_gguf", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!gguf || !tensors) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_model_descriptor_from_gguf", "gguf and tensors are required");
        return YVEX_ERR_INVALID_ARG;
    }

    model = (yvex_model_descriptor *)calloc(1, sizeof(*model));
    if (!model) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_model_descriptor_from_gguf", "failed to allocate model descriptor");
        return YVEX_ERR_NOMEM;
    }

    value = yvex_gguf_metadata_find(gguf, "general.architecture");
    if (value && yvex_gguf_value_as_string(value, &text, &len) == YVEX_OK) {
        char *arch_text = copy_bytes_string(text, len);
        model->arch = arch_from_name(arch_text);
        free(arch_text);
    }

    value = yvex_gguf_metadata_find(gguf, "general.name");
    if (value && yvex_gguf_value_as_string(value, &text, &len) == YVEX_OK) {
        model->name = copy_bytes_string(text, len);
        if (!model->name) {
            yvex_model_descriptor_close(model);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_model_descriptor_from_gguf", "failed to copy model name");
            return YVEX_ERR_NOMEM;
        }
    }

    value = yvex_gguf_metadata_find(gguf, "llama.context_length");
    if (value) {
        (void)yvex_gguf_value_as_u64(value, &model->context_length);
    }

    model->tensor_count = yvex_tensor_table_count(tensors);
    for (i = 0; i < model->tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, i);
        if (!tensor) {
            continue;
        }
        if (tensor->storage_bytes == 0) {
            model->unsupported_tensor_accounting_count += 1;
        } else {
            model->total_storage_bytes += tensor->storage_bytes;
        }
        if (tensor->role < YVEX_TENSOR_ROLE_COUNT) {
            model->role_counts[tensor->role] += 1;
        }
    }

    *out = model;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_model_descriptor_close(yvex_model_descriptor *model)
{
    if (!model) {
        return;
    }
    free(model->name);
    free(model);
}

yvex_arch yvex_model_arch(const yvex_model_descriptor *model)
{
    return model ? model->arch : YVEX_ARCH_UNKNOWN;
}

const char *yvex_arch_name(yvex_arch arch)
{
    size_t index;

    for (index = 0u; index < sizeof(architecture_names) / sizeof(architecture_names[0]);
         ++index)
        if (architecture_names[index].value == arch) return architecture_names[index].name;
    return "unknown";
}

/* Return the immutable model descriptor name without transferring ownership. */
const char *yvex_model_name(const yvex_model_descriptor *model)
{
    if (!model || !model->name) {
        return "";
    }
    return model->name;
}

int yvex_model_context_open(const char *path_or_alias,
                            yvex_model_context *out,
                            yvex_error *err)
{
    yvex_model_ref ref;
    yvex_artifact_options artifact_options;
    yvex_artifact_integrity_options integrity_options;
    yvex_artifact_integrity_report integrity_report;
    int rc;

    if (!path_or_alias || !path_or_alias[0] || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_context",
                       "path or alias and output context are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    memset(&ref, 0, sizeof(ref));
    memset(&artifact_options, 0, sizeof(artifact_options));
    memset(&integrity_options, 0, sizeof(integrity_options));
    memset(&integrity_report, 0, sizeof(integrity_report));

    rc = yvex_model_ref_resolve(&ref, path_or_alias, NULL, err);
    if (rc == YVEX_OK) {
        artifact_options.path = ref.path;
        artifact_options.readonly = 1;
        rc = yvex_artifact_open(&out->artifact, &artifact_options, err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_gguf_open(&out->gguf, out->artifact, err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_tensor_table_from_gguf(&out->table, out->gguf, err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_model_descriptor_from_gguf(&out->model,
                                             out->gguf,
                                             out->table,
                                             err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_artifact_integrity_validate(out->artifact,
                                              out->gguf,
                                              out->table,
                                              &integrity_options,
                                              &integrity_report,
                                              err);
    }
    yvex_model_ref_clear(&ref);
    if (rc != YVEX_OK) {
        yvex_model_context_close(out);
    }
    return rc;
}

int yvex_model_context_open_tokenizer(const char *path_or_alias,
                                      yvex_model_context *out,
                                      yvex_error *err)
{
    int rc = yvex_model_context_open(path_or_alias, out, err);

    if (rc == YVEX_OK) {
        rc = yvex_tokenizer_from_gguf(&out->tokenizer,
                                      out->gguf,
                                      out->model,
                                      err);
    }
    if (rc != YVEX_OK) {
        yvex_model_context_close(out);
    }
    return rc;
}

void yvex_model_context_close(yvex_model_context *context)
{
    if (!context) return;
    yvex_tokenizer_close(context->tokenizer);
    yvex_model_descriptor_close(context->model);
    yvex_tensor_table_close(context->table);
    yvex_gguf_close(context->gguf);
    yvex_artifact_close(context->artifact);
    memset(context, 0, sizeof(*context));
}

int yvex_model_context_vocab_size(const char *path_or_alias,
                                  unsigned long long *out_vocab_size,
                                  yvex_error *err)
{
    yvex_model_context context;
    const yvex_tensor_info *tensor;
    yvex_tokenizer *tokenizer = NULL;
    int rc;

    if (!path_or_alias || !out_vocab_size) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_vocab",
                       "path or alias and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out_vocab_size = 0ull;
    memset(&context, 0, sizeof(context));
    rc = yvex_model_context_open(path_or_alias, &context, err);
    if (rc != YVEX_OK) return rc;

    tensor = yvex_tensor_table_find(context.table, "token_embd.weight");
    if (tensor && tensor->rank == 2u && tensor->dims[1] > 0ull) {
        *out_vocab_size = tensor->dims[1];
        yvex_model_context_close(&context);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_tokenizer_from_gguf(&tokenizer,
                                  context.gguf,
                                  context.model,
                                  err);
    if (rc == YVEX_OK && yvex_tokenizer_vocab_size(tokenizer) > 0ull) {
        *out_vocab_size = yvex_tokenizer_vocab_size(tokenizer);
        yvex_tokenizer_close(tokenizer);
        yvex_model_context_close(&context);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    yvex_tokenizer_close(tokenizer);
    yvex_model_context_close(&context);
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "model_vocab",
                   "tokenizer-metadata-missing");
    return YVEX_ERR_UNSUPPORTED;
}

unsigned long long yvex_model_context_length(const yvex_model_descriptor *model)
{
    return model ? model->context_length : 0;
}

unsigned long long yvex_model_tensor_count(const yvex_model_descriptor *model)
{
    return model ? model->tensor_count : 0;
}

unsigned long long yvex_model_total_storage_bytes(const yvex_model_descriptor *model)
{
    return model ? model->total_storage_bytes : 0;
}

unsigned long long yvex_model_unsupported_tensor_accounting_count(const yvex_model_descriptor *model)
{
    return model ? model->unsupported_tensor_accounting_count : 0;
}

unsigned long long yvex_model_role_count(const yvex_model_descriptor *model, yvex_tensor_role role)
{
    if (!model || role >= YVEX_TENSOR_ROLE_COUNT) {
        return 0;
    }
    return model->role_counts[role];
}

/* Tensor roles */

static const char *const tensor_role_names[YVEX_TENSOR_ROLE_COUNT] = {
    [YVEX_TENSOR_ROLE_UNKNOWN] = "unknown",
    [YVEX_TENSOR_ROLE_TOKEN_EMBEDDING] = "token_embedding",
    [YVEX_TENSOR_ROLE_OUTPUT_NORM] = "output_norm",
    [YVEX_TENSOR_ROLE_OUTPUT_HEAD] = "output_head",
    [YVEX_TENSOR_ROLE_ATTENTION_NORM] = "attention_norm",
    [YVEX_TENSOR_ROLE_ATTENTION_Q] = "attention_q",
    [YVEX_TENSOR_ROLE_ATTENTION_K] = "attention_k",
    [YVEX_TENSOR_ROLE_ATTENTION_V] = "attention_v",
    [YVEX_TENSOR_ROLE_ATTENTION_OUT] = "attention_out",
    [YVEX_TENSOR_ROLE_FFN_NORM] = "ffn_norm",
    [YVEX_TENSOR_ROLE_FFN_GATE] = "ffn_gate",
    [YVEX_TENSOR_ROLE_FFN_UP] = "ffn_up",
    [YVEX_TENSOR_ROLE_FFN_DOWN] = "ffn_down",
    [YVEX_TENSOR_ROLE_MOE_ROUTER] = "moe_router",
    [YVEX_TENSOR_ROLE_MOE_EXPERT_GATE] = "moe_expert_gate",
    [YVEX_TENSOR_ROLE_MOE_EXPERT_UP] = "moe_expert_up",
    [YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN] = "moe_expert_down",
    [YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION] = "hc_head_function",
    [YVEX_TENSOR_ROLE_HC_HEAD_BASE] = "hc_head_base",
    [YVEX_TENSOR_ROLE_HC_HEAD_SCALE] = "hc_head_scale",
    [YVEX_TENSOR_ROLE_ATTENTION_SINKS] = "attention_sinks",
    [YVEX_TENSOR_ROLE_ATTENTION_Q_A] = "attention_q_a",
    [YVEX_TENSOR_ROLE_ATTENTION_Q_B] = "attention_q_b",
    [YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM] = "attention_q_a_norm",
    [YVEX_TENSOR_ROLE_ATTENTION_KV] = "attention_kv",
    [YVEX_TENSOR_ROLE_ATTENTION_KV_NORM] = "attention_kv_norm",
    [YVEX_TENSOR_ROLE_ATTENTION_OUT_A] = "attention_out_a",
    [YVEX_TENSOR_ROLE_ATTENTION_OUT_B] = "attention_out_b",
    [YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION] = "hc_attention_function",
    [YVEX_TENSOR_ROLE_HC_ATTENTION_BASE] = "hc_attention_base",
    [YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE] = "hc_attention_scale",
    [YVEX_TENSOR_ROLE_HC_FFN_FUNCTION] = "hc_ffn_function",
    [YVEX_TENSOR_ROLE_HC_FFN_BASE] = "hc_ffn_base",
    [YVEX_TENSOR_ROLE_HC_FFN_SCALE] = "hc_ffn_scale",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV] = "attention_compressor_kv",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE] = "attention_compressor_gate",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE] = "attention_compressor_ape",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM] = "attention_compressor_norm",
    [YVEX_TENSOR_ROLE_INDEXER_PROJECTION] = "indexer_projection",
    [YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B] = "indexer_attention_q_b",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV] = "indexer_compressor_kv",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE] = "indexer_compressor_gate",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE] = "indexer_compressor_ape",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM] = "indexer_compressor_norm",
    [YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS] = "moe_router_bias",
    [YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE] = "moe_router_table",
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE] = "moe_shared_expert_gate",
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP] = "moe_shared_expert_up",
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN] = "moe_shared_expert_down",
    [YVEX_TENSOR_ROLE_MTP_EMBEDDING_PROJECTION] = "mtp_embedding_projection",
    [YVEX_TENSOR_ROLE_MTP_HIDDEN_PROJECTION] = "mtp_hidden_projection",
    [YVEX_TENSOR_ROLE_MTP_EMBEDDING_NORM] = "mtp_embedding_norm",
    [YVEX_TENSOR_ROLE_MTP_HIDDEN_NORM] = "mtp_hidden_norm",
    [YVEX_TENSOR_ROLE_MTP_OUTPUT_NORM] = "mtp_output_norm"
};

const char *yvex_tensor_role_name(yvex_tensor_role role)
{
    return role >= 0 && role < YVEX_TENSOR_ROLE_COUNT ? tensor_role_names[role] : "unknown";
}

static int ends_with(const char *text, const char *suffix)
{
    size_t text_len;
    size_t suffix_len;

    if (!text || !suffix) {
        return 0;
    }

    text_len = strlen(text);
    suffix_len = strlen(suffix);
    if (suffix_len > text_len) {
        return 0;
    }

    return strcmp(text + text_len - suffix_len, suffix) == 0;
}

static int contains(const char *text, const char *needle)
{
    return text && needle && strstr(text, needle) != NULL;
}

typedef enum {
    TENSOR_ROLE_EXACT = 0,
    TENSOR_ROLE_SUFFIX
} tensor_role_match;

typedef struct {
    const char *pattern;
    const char *required_fragment;
    yvex_tensor_role role;
    tensor_role_match match;
} tensor_role_rule;

static const tensor_role_rule tensor_role_rules[] = {
    {"token_embd.weight", NULL, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, TENSOR_ROLE_EXACT},
    {"model.embed_tokens.weight", NULL, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
     TENSOR_ROLE_EXACT},
    {"tok_embeddings.weight", NULL, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, TENSOR_ROLE_EXACT},
    {"output_norm.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_NORM, TENSOR_ROLE_EXACT},
    {"model.norm.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_NORM, TENSOR_ROLE_EXACT},
    {"norm.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_NORM, TENSOR_ROLE_EXACT},
    {"output.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_HEAD, TENSOR_ROLE_EXACT},
    {"lm_head.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_HEAD, TENSOR_ROLE_EXACT},
    {".attn_norm.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_NORM, TENSOR_ROLE_SUFFIX},
    {".input_layernorm.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_NORM, TENSOR_ROLE_SUFFIX},
    {".attn_q.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_Q, TENSOR_ROLE_SUFFIX},
    {".self_attn.q_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_Q, TENSOR_ROLE_SUFFIX},
    {".attn_k.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_K, TENSOR_ROLE_SUFFIX},
    {".self_attn.k_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_K, TENSOR_ROLE_SUFFIX},
    {".attn_v.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_V, TENSOR_ROLE_SUFFIX},
    {".self_attn.v_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_V, TENSOR_ROLE_SUFFIX},
    {".attn_output.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_OUT, TENSOR_ROLE_SUFFIX},
    {".self_attn.o_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_OUT, TENSOR_ROLE_SUFFIX},
    {".ffn_norm.weight", NULL, YVEX_TENSOR_ROLE_FFN_NORM, TENSOR_ROLE_SUFFIX},
    {".post_attention_layernorm.weight", NULL, YVEX_TENSOR_ROLE_FFN_NORM,
     TENSOR_ROLE_SUFFIX},
    {".ffn_gate.weight", NULL, YVEX_TENSOR_ROLE_FFN_GATE, TENSOR_ROLE_SUFFIX},
    {".mlp.gate_proj.weight", NULL, YVEX_TENSOR_ROLE_FFN_GATE, TENSOR_ROLE_SUFFIX},
    {".ffn_up.weight", NULL, YVEX_TENSOR_ROLE_FFN_UP, TENSOR_ROLE_SUFFIX},
    {".mlp.up_proj.weight", NULL, YVEX_TENSOR_ROLE_FFN_UP, TENSOR_ROLE_SUFFIX},
    {".ffn_down.weight", NULL, YVEX_TENSOR_ROLE_FFN_DOWN, TENSOR_ROLE_SUFFIX},
    {".mlp.down_proj.weight", NULL, YVEX_TENSOR_ROLE_FFN_DOWN, TENSOR_ROLE_SUFFIX},
    {".ffn_gate_inp.weight", NULL, YVEX_TENSOR_ROLE_MOE_ROUTER, TENSOR_ROLE_SUFFIX},
    {".mlp.gate.weight", NULL, YVEX_TENSOR_ROLE_MOE_ROUTER, TENSOR_ROLE_SUFFIX},
    {".gate.weight", ".ffn.experts.", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
     TENSOR_ROLE_SUFFIX},
    {".up.weight", ".ffn.experts.", YVEX_TENSOR_ROLE_MOE_EXPERT_UP, TENSOR_ROLE_SUFFIX},
    {".down.weight", ".ffn.experts.", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
     TENSOR_ROLE_SUFFIX}
};

yvex_tensor_role yvex_tensor_role_classify(const char *architecture,
                                           const char *tensor_name,
                                           unsigned int rank,
                                           const unsigned long long *dims,
                                           yvex_dtype dtype)
{
    size_t index;

    (void)architecture;
    (void)rank;
    (void)dims;
    (void)dtype;

    if (!tensor_name) return YVEX_TENSOR_ROLE_UNKNOWN;
    for (index = 0u; index < sizeof(tensor_role_rules) / sizeof(tensor_role_rules[0]);
         ++index) {
        const tensor_role_rule *rule = &tensor_role_rules[index];
        int matched = rule->match == TENSOR_ROLE_EXACT
                          ? strcmp(tensor_name, rule->pattern) == 0
                          : ends_with(tensor_name, rule->pattern);

        if (matched && (!rule->required_fragment ||
                        contains(tensor_name, rule->required_fragment)))
            return rule->role;
    }

    return YVEX_TENSOR_ROLE_UNKNOWN;
}

/* Tensor table */

struct yvex_tensor_table {
    yvex_tensor_info *items;
    unsigned long long count;
};

static int product_dims(const yvex_gguf_tensor_info *src, unsigned long long *out, yvex_error *err)
{
    unsigned int i;
    unsigned long long product = 1;

    for (i = 0; i < src->rank; ++i) {
        if (src->dims[i] == 0) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "yvex_tensor_table_from_gguf",
                            "tensor %s has zero dimension", src->name);
            return YVEX_ERR_FORMAT;
        }
        if (product > ULLONG_MAX / src->dims[i]) {
            yvex_error_setf(err, YVEX_ERR_BOUNDS, "yvex_tensor_table_from_gguf",
                            "element count overflow for tensor %s", src->name);
            return YVEX_ERR_BOUNDS;
        }
        product *= src->dims[i];
    }

    *out = product;
    return YVEX_OK;
}

int yvex_tensor_table_from_gguf(yvex_tensor_table **out,
                                const yvex_gguf *gguf,
                                yvex_error *err)
{
    yvex_tensor_table *table;
    unsigned long long count;
    unsigned long long i;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tensor_table_from_gguf", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!gguf) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tensor_table_from_gguf", "gguf is required");
        return YVEX_ERR_INVALID_ARG;
    }

    count = yvex_gguf_tensor_count(gguf);
    table = (yvex_tensor_table *)calloc(1, sizeof(*table));
    if (!table) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "failed to allocate tensor table");
        return YVEX_ERR_NOMEM;
    }
    table->count = count;

    if (count > 0) {
        if (count > (unsigned long long)(SIZE_MAX / sizeof(*table->items))) {
            yvex_tensor_table_close(table);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "tensor count too large");
            return YVEX_ERR_NOMEM;
        }
        table->items = (yvex_tensor_info *)calloc((size_t)count, sizeof(*table->items));
        if (!table->items) {
            yvex_tensor_table_close(table);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "failed to allocate tensor rows");
            return YVEX_ERR_NOMEM;
        }
    }

    for (i = 0; i < count; ++i) {
        const yvex_gguf_tensor_info *src = yvex_gguf_tensor_at(gguf, i);
        yvex_tensor_info *dst = &table->items[i];
        const yvex_dtype_info *dtype_info;
        int rc;

        dst->name = yvex_core_strdup(src->name);
        if (!dst->name) {
            yvex_tensor_table_close(table);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "failed to copy tensor name");
            return YVEX_ERR_NOMEM;
        }
        dst->rank = src->rank;
        memcpy(dst->dims, src->dims, sizeof(dst->dims));
        dst->ggml_type = src->ggml_type;
        dst->relative_offset = src->relative_offset;
        dst->absolute_offset = src->absolute_offset;

        dtype_info = yvex_dtype_from_ggml_type(src->ggml_type);
        dst->dtype = dtype_info->dtype;

        rc = product_dims(src, &dst->element_count, err);
        if (rc != YVEX_OK) {
            yvex_tensor_table_close(table);
            return rc;
        }

        rc = yvex_dtype_tensor_storage_bytes(dst->dtype,
                                             dst->dims,
                                             dst->rank,
                                             &dst->storage_bytes,
                                             err);
        if (rc == YVEX_ERR_UNSUPPORTED) {
            dst->storage_bytes = 0;
            yvex_error_clear(err);
        } else if (rc != YVEX_OK) {
            yvex_tensor_table_close(table);
            return rc;
        }

        dst->role = yvex_tensor_role_classify(NULL, dst->name, dst->rank, dst->dims, dst->dtype);
    }

    *out = table;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tensor_table_close(yvex_tensor_table *table)
{
    unsigned long long i;

    if (!table) {
        return;
    }
    if (table->items) {
        for (i = 0; i < table->count; ++i) {
            free((char *)table->items[i].name);
            table->items[i].name = NULL;
        }
        free(table->items);
    }
    free(table);
}

unsigned long long yvex_tensor_table_count(const yvex_tensor_table *table)
{
    return table ? table->count : 0;
}

const yvex_tensor_info *yvex_tensor_table_at(const yvex_tensor_table *table,
                                             unsigned long long index)
{
    if (!table || index >= table->count) {
        return NULL;
    }
    return &table->items[index];
}

const yvex_tensor_info *yvex_tensor_table_find(const yvex_tensor_table *table,
                                               const char *name)
{
    unsigned long long i;

    if (!table || !name) {
        return NULL;
    }

    for (i = 0; i < table->count; ++i) {
        if (table->items[i].name && strcmp(table->items[i].name, name) == 0) {
            return &table->items[i];
        }
    }

    return NULL;
}

/* Materialized weights */

struct yvex_materialized_weight {
    char *name;
    yvex_dtype dtype;
    yvex_tensor_role role;
    unsigned long long bytes;
    yvex_weight_residency residency;
    yvex_device_tensor *device_tensor;
};

struct yvex_weight_table {
    yvex_backend *backend;
    char *backend_name;
    yvex_materialized_weight *items;
    unsigned long long count;
    yvex_materialize_summary summary;
};

static int test_env_enabled(const char *name)
{
    const char *value = getenv(name);

    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void materialized_weight_clear(yvex_weight_table *table,
                                           yvex_materialized_weight *weight)
{
    if (!weight) {
        return;
    }
    if (table && table->backend && weight->device_tensor) {
        yvex_backend_tensor_free(table->backend, weight->device_tensor);
    }
    weight->device_tensor = NULL;
    free(weight->name);
    weight->name = NULL;
}

/* Release backend materialized weights and table ownership in dependency order. */
void yvex_weight_table_close(yvex_weight_table *weights)
{
    unsigned long long i;

    if (!weights) {
        return;
    }
    for (i = 0; i < weights->count; ++i) {
        materialized_weight_clear(weights, &weights->items[i]);
    }
    free(weights->items);
    free(weights->backend_name);
    free(weights);
}

unsigned long long yvex_weight_table_count(const yvex_weight_table *weights)
{
    return weights ? weights->count : 0;
}

const yvex_materialized_weight *yvex_weight_table_at(const yvex_weight_table *weights,
                                                     unsigned long long index)
{
    if (!weights || index >= weights->count) {
        return NULL;
    }
    return &weights->items[index];
}

const yvex_materialized_weight *yvex_weight_table_find(const yvex_weight_table *weights,
                                                       const char *name)
{
    unsigned long long i;

    if (!weights || !name) {
        return NULL;
    }
    for (i = 0; i < weights->count; ++i) {
        if (weights->items[i].name && strcmp(weights->items[i].name, name) == 0) {
            return &weights->items[i];
        }
    }
    return NULL;
}

const char *yvex_weight_status_name(yvex_weight_status status)
{
    switch (status) {
    case YVEX_WEIGHT_STATUS_EMPTY: return "empty";
    case YVEX_WEIGHT_STATUS_MATERIALIZED: return "materialized";
    case YVEX_WEIGHT_STATUS_PARTIAL: return "partial";
    case YVEX_WEIGHT_STATUS_FAILED: return "failed";
    }
    return "unknown";
}

const char *yvex_weight_residency_name(yvex_weight_residency residency)
{
    switch (residency) {
    case YVEX_WEIGHT_RESIDENCY_HOST: return "host";
    case YVEX_WEIGHT_RESIDENCY_CPU_BACKEND: return "cpu_backend";
    case YVEX_WEIGHT_RESIDENCY_CUDA_BACKEND: return "cuda_backend";
    }
    return "unknown";
}

/* Return one borrowed materialized-weight name without ownership transfer. */
const char *yvex_weight_name(const yvex_materialized_weight *weight)
{
    return weight && weight->name ? weight->name : "";
}

yvex_dtype yvex_weight_dtype(const yvex_materialized_weight *weight)
{
    return weight ? weight->dtype : YVEX_DTYPE_UNKNOWN;
}

yvex_tensor_role yvex_weight_role(const yvex_materialized_weight *weight)
{
    return weight ? weight->role : YVEX_TENSOR_ROLE_UNKNOWN;
}

unsigned long long yvex_weight_bytes(const yvex_materialized_weight *weight)
{
    return weight ? weight->bytes : 0;
}

yvex_weight_residency yvex_weight_residency_of(const yvex_materialized_weight *weight)
{
    return weight ? weight->residency : YVEX_WEIGHT_RESIDENCY_HOST;
}

const yvex_device_tensor *yvex_weight_device_tensor(const yvex_materialized_weight *weight)
{
    return weight ? weight->device_tensor : NULL;
}

static yvex_weight_residency residency_from_backend(const yvex_backend *backend)
{
    if (yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA) {
        return YVEX_WEIGHT_RESIDENCY_CUDA_BACKEND;
    }
    if (yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CPU) {
        return YVEX_WEIGHT_RESIDENCY_CPU_BACKEND;
    }
    return YVEX_WEIGHT_RESIDENCY_HOST;
}

static int copy_tensor_dims(yvex_backend_tensor_desc *desc, const yvex_tensor_info *tensor)
{
    unsigned int i;

    if (!desc || !tensor || tensor->rank > YVEX_TENSOR_MAX_DIMS) {
        return 0;
    }
    for (i = 0; i < tensor->rank; ++i) {
        desc->dims[i] = tensor->dims[i];
    }
    return 1;
}

static int materialize_one(yvex_weight_table *table,
                           const yvex_artifact *artifact,
                           const yvex_gguf *gguf,
                           const yvex_tensor_info *tensor,
                           yvex_error *err)
{
    yvex_backend_tensor_desc desc;
    yvex_device_tensor *device_tensor = NULL;
    yvex_materialized_weight *weight;
    yvex_tensor_range range;
    const unsigned char *data;
    unsigned char *owned_payload = NULL;
    int rc;

    if (!table || !artifact || !gguf || !tensor) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_weight_table_materialize",
                       "table, artifact, gguf, and tensor are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(&range, 0, sizeof(range));
    rc = yvex_tensor_range_validate(artifact, gguf, tensor, &range, err);
    if (rc != YVEX_OK) {
        return rc;
    }

    memset(&desc, 0, sizeof(desc));
    desc.name = tensor->name;
    desc.dtype = tensor->dtype;
    desc.rank = tensor->rank;
    desc.bytes = range.tensor_bytes;
    if (!copy_tensor_dims(&desc, tensor)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_weight_table_materialize",
                       "invalid tensor rank");
        return YVEX_ERR_INVALID_ARG;
    }

    table->summary.materialization_phase = "allocation";
    rc = yvex_backend_tensor_alloc(table->backend, &desc, &device_tensor, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    table->summary.allocation_attempted = 1;
    table->summary.bytes_allocated += range.tensor_bytes;
    if (test_env_enabled("YVEX_TEST_FAIL_MATERIALIZE_AFTER_ALLOC")) {
        yvex_backend_tensor_free(table->backend, device_tensor);
        table->summary.materialization_gate = "fail";
        table->summary.materialization_phase = "allocation";
        table->summary.cleanup_attempted = 1;
        table->summary.cleanup_status = "pass";
        table->summary.status = YVEX_WEIGHT_STATUS_FAILED;
        yvex_error_set(err, YVEX_ERR_BACKEND, "yvex_weight_table_materialize",
                       "test materialization failure after allocation");
        return YVEX_ERR_BACKEND;
    }

    data = yvex_artifact_data(artifact);
    if (!data) {
        if (range.tensor_bytes > (unsigned long long)SIZE_MAX) {
            yvex_backend_tensor_free(table->backend, device_tensor);
            table->summary.cleanup_attempted = 1;
            table->summary.cleanup_status = "pass";
            yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_weight_table_materialize",
                           "tensor payload exceeds host staging address space");
            return YVEX_ERR_BOUNDS;
        }
        owned_payload = (unsigned char *)malloc((size_t)range.tensor_bytes);
        if (!owned_payload) {
            yvex_backend_tensor_free(table->backend, device_tensor);
            table->summary.cleanup_attempted = 1;
            table->summary.cleanup_status = "pass";
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_weight_table_materialize",
                           "failed to allocate tensor payload staging buffer");
            return YVEX_ERR_NOMEM;
        }
        rc = yvex_artifact_read_at(artifact, range.tensor_absolute_offset,
                                   owned_payload, (size_t)range.tensor_bytes, err);
        if (rc != YVEX_OK) {
            free(owned_payload);
            yvex_backend_tensor_free(table->backend, device_tensor);
            table->summary.cleanup_attempted = 1;
            table->summary.cleanup_status = "pass";
            return rc;
        }
        data = owned_payload;
    }

    table->summary.materialization_phase = "transfer";
    table->summary.transfer_attempted = 1;
    rc = yvex_backend_tensor_write(table->backend,
                                   device_tensor,
                                   owned_payload ? data : data + range.tensor_absolute_offset,
                                   range.tensor_bytes,
                                   err);
    free(owned_payload);
    owned_payload = NULL;
    if (rc != YVEX_OK) {
        yvex_backend_tensor_free(table->backend, device_tensor);
        table->summary.cleanup_attempted = 1;
        table->summary.cleanup_status = "pass";
        return rc;
    }
    table->summary.bytes_transferred += range.tensor_bytes;
    if (test_env_enabled("YVEX_TEST_FAIL_MATERIALIZE_AFTER_TRANSFER")) {
        yvex_backend_tensor_free(table->backend, device_tensor);
        table->summary.materialization_gate = "fail";
        table->summary.materialization_phase = "transfer";
        table->summary.cleanup_attempted = 1;
        table->summary.cleanup_status = "pass";
        table->summary.status = YVEX_WEIGHT_STATUS_FAILED;
        yvex_error_set(err, YVEX_ERR_BACKEND, "yvex_weight_table_materialize",
                       "test materialization write failure after transfer");
        return YVEX_ERR_BACKEND;
    }

    weight = &table->items[table->count];
    weight->name = yvex_core_strdup(tensor->name);
    if (!weight->name) {
        yvex_backend_tensor_free(table->backend, device_tensor);
        table->summary.cleanup_attempted = 1;
        table->summary.cleanup_status = "pass";
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_weight_table_materialize",
                       "failed to copy weight name");
        return YVEX_ERR_NOMEM;
    }
    weight->dtype = tensor->dtype;
    weight->role = tensor->role;
    weight->bytes = range.tensor_bytes;
    weight->residency = residency_from_backend(table->backend);
    weight->device_tensor = device_tensor;
    table->count += 1;
    return YVEX_OK;
}

int yvex_weight_table_materialize(yvex_weight_table **out,
                                  const yvex_artifact *artifact,
                                  const yvex_gguf *gguf,
                                  const yvex_tensor_table *tensors,
                                  yvex_backend *backend,
                                  const yvex_materialize_options *options,
                                  yvex_error *err)
{
    yvex_weight_table *table;
    yvex_backend_memory_stats stats;
    yvex_gguf_layout_result layout;
    unsigned long long tensor_count;
    unsigned long long i;
    int require_all = 0;
    int allow_unsupported = 0;
    int rc = YVEX_OK;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_weight_table_materialize", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!artifact || !gguf || !tensors || !backend) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_weight_table_materialize",
                       "artifact, gguf, tensors and backend are required");
        return YVEX_ERR_INVALID_ARG;
    }

    rc = yvex_gguf_layout_validate(artifact, gguf, &layout, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (options) {
        require_all = options->require_all_tensors;
        allow_unsupported = options->allow_unsupported_dtype;
    }

    tensor_count = yvex_tensor_table_count(tensors);
    table = (yvex_weight_table *)calloc(1, sizeof(*table));
    if (!table) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_weight_table_materialize",
                       "failed to allocate weight table");
        return YVEX_ERR_NOMEM;
    }
    table->backend = backend;
    table->backend_name = yvex_core_strdup(options && options->backend_name
                                             ? options->backend_name
                                             : yvex_backend_kind_name(yvex_backend_kind_of(backend)));
    table->items = (yvex_materialized_weight *)calloc((size_t)(tensor_count ? tensor_count : 1),
                                                      sizeof(*table->items));
    if (!table->backend_name || !table->items) {
        yvex_weight_table_close(table);
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_weight_table_materialize",
                       "failed to allocate materialized weight rows");
        return YVEX_ERR_NOMEM;
    }

    table->summary.backend_name = table->backend_name;
    table->summary.materialization_gate = "fail";
    table->summary.materialization_phase = "preflight";
    table->summary.shape_status = "unchecked";
    table->summary.range_status = "unchecked";
    table->summary.backend_status = "ready";
    table->summary.cleanup_status = "not-needed";
    table->summary.tensors_total = tensor_count;
    table->summary.execution_ready = 0;

    for (i = 0; i < tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, i);
        yvex_tensor_range range;

        if (!tensor) {
            if (require_all) {
                yvex_weight_table_close(table);
                yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_weight_table_materialize",
                               "missing tensor table row");
                return YVEX_ERR_INVALID_ARG;
            }
            continue;
        }
        if (tensor->storage_bytes == 0) {
            if (require_all && !allow_unsupported) {
                yvex_weight_table_close(table);
                yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "yvex_weight_table_materialize",
                                "tensor %s has unsupported storage accounting", tensor->name);
                return YVEX_ERR_UNSUPPORTED;
            }
            continue;
        }
        memset(&range, 0, sizeof(range));
        rc = yvex_tensor_range_validate(artifact, gguf, tensor, &range, err);
        if (rc != YVEX_OK) {
            yvex_weight_table_close(table);
            return rc;
        }
        table->summary.bytes_planned += range.tensor_bytes;
    }
    table->summary.shape_status = "pass";
    table->summary.range_status = "pass";

    for (i = 0; i < tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, i);
        if (!tensor) {
            table->summary.tensors_failed += 1;
            if (require_all) {
                rc = YVEX_ERR_INVALID_ARG;
                yvex_error_set(err, rc, "yvex_weight_table_materialize",
                               "missing tensor table row");
                break;
            }
            continue;
        }

        if (tensor->storage_bytes > 0) {
            table->summary.bytes_total += tensor->storage_bytes;
        } else {
            table->summary.tensors_failed += 1;
            if (require_all && !allow_unsupported) {
                rc = YVEX_ERR_UNSUPPORTED;
                yvex_error_setf(err, rc, "yvex_weight_table_materialize",
                                "tensor %s has unsupported storage accounting", tensor->name);
                break;
            }
            continue;
        }

        rc = materialize_one(table, artifact, gguf, tensor, err);
        if (rc != YVEX_OK) {
            table->summary.tensors_failed += 1;
            if (require_all || rc == YVEX_ERR_BOUNDS || rc == YVEX_ERR_FORMAT ||
                rc == YVEX_ERR_BACKEND || rc == YVEX_ERR_NOMEM) {
                break;
            }
            rc = YVEX_OK;
            continue;
        }
        table->summary.tensors_materialized += 1;
        table->summary.bytes_materialized += tensor->storage_bytes;
    }

    if (rc != YVEX_OK) {
        yvex_weight_table_close(table);
        return rc;
    }

    if (table->summary.tensors_materialized == tensor_count &&
        table->summary.tensors_failed == 0) {
        table->summary.status = tensor_count == 0
            ? YVEX_WEIGHT_STATUS_EMPTY
            : YVEX_WEIGHT_STATUS_MATERIALIZED;
    } else {
        table->summary.status = YVEX_WEIGHT_STATUS_PARTIAL;
    }
    table->summary.materialization_gate =
        table->summary.status == YVEX_WEIGHT_STATUS_MATERIALIZED ? "pass" : "fail";
    table->summary.materialization_phase = "complete";
    table->summary.cleanup_status = "not-needed";

    if (yvex_backend_get_memory_stats(backend, &stats, err) == YVEX_OK) {
        table->summary.backend_allocated_bytes = stats.allocated_bytes;
    }

    *out = table;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_weight_table_get_summary(const yvex_weight_table *weights,
                                  yvex_materialize_summary *out,
                                  yvex_error *err)
{
    yvex_backend_memory_stats stats;

    if (!weights || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_weight_table_get_summary",
                       "weights and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memcpy(out, &weights->summary, sizeof(*out));
    if (weights->backend &&
        yvex_backend_get_memory_stats(weights->backend, &stats, err) == YVEX_OK) {
        out->backend_allocated_bytes = stats.allocated_bytes;
    }
    return YVEX_OK;
}
