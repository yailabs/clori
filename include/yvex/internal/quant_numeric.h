/* Owner: gguf.quant_numeric (gguf).
 * Owns: numeric codecs, immutable quant plans, and transactional quant sinks.
 * Does not own: Transformation IR semantics, GGUF writing, or backend topology.
 * Invariants: declarations have one owner, stable ordering, and no hidden capability promotion.
 * Boundary: physical numeric execution contracts.
 * Purpose: provide the canonical physical numeric execution contracts contract.
 * Inputs: typed immutable facts and explicitly owned mutable lifecycle objects.
 * Effects: only declared lifecycle, allocation, I/O, and publication operations mutate state.
 * Failure: typed refusals leave outputs defined and preserve caller-owned state. */
#ifndef INCLUDE_YVEX_INTERNAL_QUANT_NUMERIC_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_QUANT_NUMERIC_H_INCLUDED

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <yvex/core.h>
#include <yvex/qtype.h>
#include <yvex/quant.h>
#include <yvex/source.h>
#include <yvex/internal/compilation.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quant Numeric contract. */
#define YVEX_QUANT_NUMERIC_CONTRACT_VERSION 1u
#define YVEX_QUANT_Q8_0_ELEMENTS 32u
#define YVEX_QUANT_Q8_0_BYTES 34u
#define YVEX_QUANT_MXFP4_ELEMENTS 32u
#define YVEX_QUANT_MXFP4_BYTES 17u
#define YVEX_QUANT_Q2_K_ELEMENTS 256u
#define YVEX_QUANT_Q2_K_BYTES 84u
#define YVEX_QUANT_IQ2_XXS_ELEMENTS 256u
#define YVEX_QUANT_IQ2_XXS_BYTES 66u
typedef enum {
    YVEX_QUANT_FAILURE_NONE = 0,
    YVEX_QUANT_FAILURE_INVALID_ARGUMENT,
    YVEX_QUANT_FAILURE_INVALID_STATE,
    YVEX_QUANT_FAILURE_PROFILE_SCHEMA,
    YVEX_QUANT_FAILURE_TRANSFORM_IDENTITY,
    YVEX_QUANT_FAILURE_MAPPING_IDENTITY,
    YVEX_QUANT_FAILURE_SOURCE_IDENTITY,
    YVEX_QUANT_FAILURE_PAYLOAD_IDENTITY,
    YVEX_QUANT_FAILURE_PAYLOAD_NOT_READABLE,
    YVEX_QUANT_FAILURE_MISSING_DECISION,
    YVEX_QUANT_FAILURE_DUPLICATE_DECISION,
    YVEX_QUANT_FAILURE_UNMATCHED_LOWERING,
    YVEX_QUANT_FAILURE_DUPLICATE_LOWERING,
    YVEX_QUANT_FAILURE_PRECISION_CONSTRAINT,
    YVEX_QUANT_FAILURE_APPROXIMATION_FORBIDDEN,
    YVEX_QUANT_FAILURE_UNKNOWN_QTYPE,
    YVEX_QUANT_FAILURE_REMOVED_QTYPE,
    YVEX_QUANT_FAILURE_QTYPE_OUTSIDE_BASELINE,
    YVEX_QUANT_FAILURE_ENCODER_UNAVAILABLE,
    YVEX_QUANT_FAILURE_DECODER_UNAVAILABLE,
    YVEX_QUANT_FAILURE_CPU_COMPUTE_UNAVAILABLE,
    YVEX_QUANT_FAILURE_CUDA_COMPUTE_UNAVAILABLE,
    YVEX_QUANT_FAILURE_CALIBRATION_REQUIRED,
    YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY,
    YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION,
    YVEX_QUANT_FAILURE_INVALID_RANK,
    YVEX_QUANT_FAILURE_INVALID_DIMENSION,
    YVEX_QUANT_FAILURE_INVALID_ROW_AXIS,
    YVEX_QUANT_FAILURE_ROW_DIVISIBILITY,
    YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW,
    YVEX_QUANT_FAILURE_BYTE_OVERFLOW,
    YVEX_QUANT_FAILURE_FP8_SCALE_PAIR,
    YVEX_QUANT_FAILURE_MXFP4_BLOCK,
    YVEX_QUANT_FAILURE_Q8_0_BLOCK,
    YVEX_QUANT_FAILURE_Q2_K_BLOCK,
    YVEX_QUANT_FAILURE_IQ2_XXS_BLOCK,
    YVEX_QUANT_FAILURE_CAST_RANGE,
    YVEX_QUANT_FAILURE_NONFINITE,
    YVEX_QUANT_FAILURE_SOURCE_SHORT_READ,
    YVEX_QUANT_FAILURE_SOURCE_DRIFT,
    YVEX_QUANT_FAILURE_SINK_SHORT_WRITE,
    YVEX_QUANT_FAILURE_SINK_PROTOCOL,
    YVEX_QUANT_FAILURE_CANCELLED,
    YVEX_QUANT_FAILURE_RESOURCE_BUDGET,
    YVEX_QUANT_FAILURE_ALLOCATION,
    YVEX_QUANT_FAILURE_WORKER,
    YVEX_QUANT_FAILURE_CLEANUP,
    YVEX_QUANT_FAILURE_NUMERIC_BOUND,
    YVEX_QUANT_FAILURE_DIGEST_MISMATCH,
    YVEX_QUANT_FAILURE_INCOMPLETE
} yvex_quant_failure_code;
typedef struct yvex_quant_failure {
    yvex_quant_failure_code code;
    unsigned long long terminal_ordinal;
    unsigned long long source_index;
    unsigned long long row_index;
    unsigned long long block_index;
    unsigned long long expected;
    unsigned long long actual;
    unsigned int qtype;
    yvex_transform_operation_kind operation;
} yvex_quant_failure;
typedef enum {
    YVEX_QUANT_CALIBRATION_NONE = 0,
    YVEX_QUANT_CALIBRATION_OPTIONAL,
    YVEX_QUANT_CALIBRATION_REQUIRED,
    YVEX_QUANT_CALIBRATION_UNSUPPORTED
} yvex_quant_calibration_requirement;
typedef enum {
    YVEX_QUANT_REFUSAL_NONE = 0,
    YVEX_QUANT_REFUSAL_UNKNOWN_IDENTITY,
    YVEX_QUANT_REFUSAL_REMOVED_IDENTITY,
    YVEX_QUANT_REFUSAL_OUTSIDE_PINNED_BASELINE,
    YVEX_QUANT_REFUSAL_STORAGE_UNAVAILABLE,
    YVEX_QUANT_REFUSAL_ENCODER_UNAVAILABLE,
    YVEX_QUANT_REFUSAL_DECODER_UNAVAILABLE,
    YVEX_QUANT_REFUSAL_CPU_COMPUTE_UNAVAILABLE,
    YVEX_QUANT_REFUSAL_CUDA_COMPUTE_UNAVAILABLE,
    YVEX_QUANT_REFUSAL_CALIBRATION_REQUIRED
} yvex_quant_refusal_code;
enum {
    YVEX_QUANT_TRANSFORM_IDENTITY = 1u << YVEX_TRANSFORM_OP_IDENTITY,
    YVEX_QUANT_TRANSFORM_SCALE_PAIR =
        1u << YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR,
    YVEX_QUANT_TRANSFORM_CHECKED_CAST =
        1u << YVEX_TRANSFORM_OP_CHECKED_CAST,
    YVEX_QUANT_TRANSFORM_EXPERT =
        1u << YVEX_TRANSFORM_OP_EXPERT_AGGREGATE
};
typedef struct {
    unsigned int qtype;
    int identity_known;
    int storage_admitted;
    int encoder_available;
    int reference_decoder_available;
    int dedicated_cpu_compute_available;
    int dedicated_cuda_compute_available;
    yvex_quant_calibration_requirement calibration;
    unsigned int physical_class_mask;
    unsigned int transform_kind_mask;
    unsigned int numeric_contract_version;
    int deterministic_encoding;
    yvex_quant_refusal_code refusal;
} yvex_quant_numeric_capability;
typedef struct {
    unsigned long long element_count;
    unsigned long long finite_count;
    unsigned long long nonfinite_count;
    double maximum_absolute_error;
    double absolute_error_sum;
    double squared_error_sum;
    double relative_error_sum;
    double reference_squared_sum;
    double dot_reference;
    double dot_reconstructed;
} yvex_quant_metrics;
const yvex_quant_numeric_capability *yvex_quant_numeric_capability_at(
    unsigned int qtype);
const yvex_quant_numeric_capability *yvex_quant_numeric_capability_by_name(
    const char *name);
const char *yvex_quant_calibration_name(
    yvex_quant_calibration_requirement requirement);
float yvex_quant_f16_decode(unsigned short bits);
unsigned short yvex_quant_f16_encode(float value);
float yvex_quant_bf16_decode(unsigned short bits);
unsigned short yvex_quant_bf16_encode(float value);
float yvex_quant_fp8_e4m3fn_decode(unsigned char bits);
float yvex_quant_e8m0_decode(unsigned char bits);
int yvex_quant_source_scalar_decode(yvex_native_dtype dtype,
                                    const unsigned char *source,
                                    float *out,
                                    yvex_quant_failure *failure,
                                    yvex_error *err);
int yvex_quant_source_i64_decode(const unsigned char source[8],
                                 int64_t *out,
                                 yvex_quant_failure *failure,
                                 yvex_error *err);
int yvex_quant_source_mxfp4_decode(const unsigned char packed[16],
                                  unsigned char scale,
                                  float out[32],
                                  yvex_quant_failure *failure,
                                  yvex_error *err);
int yvex_quant_encode_block(unsigned int qtype,
                            const float *source,
                            unsigned long long elements,
                            unsigned char *encoded,
                            size_t encoded_capacity,
                            size_t *encoded_bytes,
                            yvex_quant_failure *failure,
                            yvex_error *err);
int yvex_quant_encode_block_weighted(unsigned int qtype,
                                     const float *source,
                                     const float *calibration_weights,
                                     unsigned long long elements,
                                     unsigned char *encoded,
                                     size_t encoded_capacity,
                                     size_t *encoded_bytes,
                                     yvex_quant_failure *failure,
                                     yvex_error *err);
int yvex_quant_decode_block(unsigned int qtype,
                            const unsigned char *encoded,
                            size_t encoded_bytes,
                            float *out,
                            unsigned long long out_elements,
                            yvex_quant_failure *failure,
                            yvex_error *err);
void yvex_quant_metrics_init(yvex_quant_metrics *metrics);
int yvex_quant_metrics_update(yvex_quant_metrics *metrics,
                              const float *reference,
                              const float *reconstructed,
                              const float *dot_vector,
                              unsigned long long count);
double yvex_quant_metrics_rmse(const yvex_quant_metrics *metrics);
int yvex_quant_cpu_dot(unsigned int qtype,
                       const unsigned char *encoded,
                       size_t encoded_bytes,
                       const float *vector,
                       unsigned long long elements,
                       float *out,
                       yvex_quant_failure *failure,
                       yvex_error *err);

/* Quant Plan contract. */
#define YVEX_QUANT_PROFILE_SCHEMA_VERSION 1u
#define YVEX_QUANT_PLAN_IDENTITY_CAP 65u
#define YVEX_QUANT_RELEASE_PROFILE_NAME \
    "deepseek-v4-flash-q8_0-q2_k-v1"
#define YVEX_QUANT_REFERENCE_PROFILE_NAME \
    "deepseek-v4-flash-source-faithful-v1"
#define YVEX_QUANT_DS4_PROFILE_NAME \
    "deepseek-v4-flash-ds4-like-q2-v1"
typedef enum {
    YVEX_QUANT_PLAN_BUILDING = 0,
    YVEX_QUANT_PLAN_SEALED,
    YVEX_QUANT_PLAN_RELEASED
} yvex_quant_plan_state;
typedef enum {
    YVEX_QUANT_PHYSICAL_EXACT_SCALAR = 0,
    YVEX_QUANT_PHYSICAL_BLOCK_QUANTIZED
} yvex_quant_physical_class;
typedef struct {
    yvex_transform_logical_key logical_key;
    unsigned long long terminal_ordinal;
    unsigned long long terminal_value_id;
    unsigned long long node_id;
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_transform_scope scope;
    yvex_transform_operation_kind operation;
    yvex_quant_physical_class physical_class;
    unsigned int qtype;
    unsigned int rank;
    unsigned long long dims[YVEX_GGUF_QTYPE_MAX_DIMS];
    unsigned int row_axis;
    unsigned long long row_width;
    unsigned long long row_count;
    unsigned long long element_count;
    unsigned long long encoded_bytes;
    int approximation;
    yvex_quant_calibration_requirement calibration;
    int reference_decoder_required;
    int cpu_compute_available;
    int cuda_compute_available;
    unsigned int numeric_contract_version;
    int policy_bound;
    unsigned long long policy_rule_ordinal;
    unsigned int policy_priority;
    int policy_requires_imatrix;
    char policy_label[64];
    char physical_tensor_name[192];
    unsigned long long physical_expert_count;
    char policy_rule_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char decision_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
} yvex_quant_decision;
typedef struct {
    yvex_quant_profile_kind kind;
    const char *name;
    unsigned long long terminal_count;
    unsigned long long encoded_bytes;
    unsigned long long exact_scalar_bytes;
    unsigned long long q8_0_bytes;
    unsigned long long q2_k_bytes;
    unsigned long long iq2_xxs_bytes;
    unsigned long long mxfp4_bytes;
    unsigned long long qtype_tensor_counts[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    int calibration_required;
    int numerically_admissible;
    int compute_admissible;
} yvex_quant_candidate_summary;
typedef struct {
    unsigned int schema_version;
    yvex_quant_plan_state state;
    char profile_name[64];
    char profile_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char payload_plan_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char transform_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    unsigned long long source_snapshot_identity;
    char required_payload_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    unsigned long long mapping_identity;
    char backend_compute_contract[64];
    char calibration_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char policy_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char imatrix_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char physical_variant_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    unsigned long long terminal_count;
    unsigned long long decision_count;
    unsigned long long source_value_count;
    unsigned long long encoded_bytes;
    unsigned long long exact_scalar_bytes;
    unsigned long long q8_0_bytes;
    unsigned long long q2_k_bytes;
    unsigned long long iq2_xxs_bytes;
    unsigned long long mxfp4_bytes;
    unsigned long long qtype_tensor_counts[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    unsigned long long qtype_encoded_bytes[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    unsigned long long role_tensor_counts[YVEX_TENSOR_ROLE_COUNT];
    unsigned long long index_capacity;
    unsigned long long lookup_count;
    size_t owned_bytes;
    size_t peak_builder_bytes;
    unsigned long long payload_bytes_read;
    int calibration_required;
    int complete;
    yvex_quant_candidate_summary candidates[2];
} yvex_quant_plan_summary;
typedef void *(*yvex_quant_allocate_fn)(size_t size, void *context);
typedef void (*yvex_quant_release_fn)(void *allocation, void *context);
typedef struct yvex_quant_plan_options {
    yvex_quant_allocate_fn allocate;
    yvex_quant_release_fn release;
    void *context;
    size_t maximum_owned_bytes;
} yvex_quant_plan_options;
typedef struct {
    unsigned int qtype;
    int approximation;
    unsigned int rank;
    unsigned long long dims[YVEX_GGUF_QTYPE_MAX_DIMS];
    unsigned int row_axis;
} yvex_quant_explicit_decision;
typedef struct yvex_quant_plan yvex_quant_plan;
int yvex_quant_plan_build_explicit(
    yvex_quant_plan **out,
    const yvex_transform_ir *ir,
    const yvex_transform_binding *binding,
    const char *profile_name,
    unsigned long long lowering_identity,
    const yvex_quant_explicit_decision *decisions,
    unsigned long long decision_count,
    const yvex_quant_plan_options *options,
    yvex_quant_failure *failure,
    yvex_error *err);
void yvex_quant_plan_release(yvex_quant_plan **plan);
const yvex_quant_plan_summary *yvex_quant_plan_summary_get(
    const yvex_quant_plan *plan);
const yvex_quant_decision *yvex_quant_plan_decision_at(
    const yvex_quant_plan *plan, unsigned long long ordinal);
const yvex_quant_decision *yvex_quant_plan_find(
    const yvex_quant_plan *plan, const yvex_transform_logical_key *key);
const yvex_transform_ir *yvex_quant_plan_transform_ir(
    const yvex_quant_plan *plan);
const yvex_transform_binding *yvex_quant_plan_binding(
    const yvex_quant_plan *plan);

/* Physical-variant plan file contract. */
#define YVEX_PHYSICAL_VARIANT_FILE_SCHEMA_VERSION 1u
int yvex_quant_plan_file_write(const char *path, const yvex_quant_plan *plan,
                               yvex_error *err);
int yvex_quant_plan_file_validate(const char *path, const yvex_quant_plan *plan,
                                  yvex_error *err);

/* Quant Sink contract. */
typedef struct {
    int (*begin_terminal)(void *context,
                          const yvex_quant_decision *decision);
    int (*deliver_chunk)(void *context,
                         const yvex_quant_decision *decision,
                         unsigned long long output_offset,
                         const unsigned char *bytes,
                         size_t byte_count);
    int (*commit_terminal)(void *context,
                           const yvex_quant_decision *decision,
                           unsigned long long delivered_bytes);
    void (*abort_terminal)(void *context,
                           const yvex_quant_decision *decision,
                           const yvex_quant_failure *failure,
                           unsigned long long delivered_bytes);
    void *context;
} yvex_quant_output_sink;
typedef struct {
    unsigned long long terminal_count;
    unsigned long long committed_terminals;
    unsigned long long aborted_terminals;
    unsigned long long output_chunks;
    unsigned long long encoded_bytes;
    unsigned long long qtype_bytes[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    char execution_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char payload_byte_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    int complete;
} yvex_quant_digest_summary;
typedef struct {
    unsigned long long terminal_ordinal;
    unsigned int qtype;
    unsigned long long delivered_bytes;
    unsigned long long chunks;
    char sha256[YVEX_QUANT_PLAN_IDENTITY_CAP];
    int committed;
} yvex_quant_terminal_digest;
typedef struct yvex_quant_digest_sink yvex_quant_digest_sink;
typedef struct {
    yvex_quant_allocate_fn allocate;
    yvex_quant_release_fn release;
    void *context;
} yvex_quant_sink_allocator;
int yvex_quant_digest_sink_create(
    yvex_quant_digest_sink **out,
    const yvex_quant_plan *plan,
    const char *payload_identity,
    yvex_quant_failure *failure,
    yvex_error *err);
int yvex_quant_digest_sink_create_with_allocator(
    yvex_quant_digest_sink **out,
    const yvex_quant_plan *plan,
    const char *payload_identity,
    const yvex_quant_sink_allocator *allocator,
    yvex_quant_failure *failure,
    yvex_error *err);
void yvex_quant_digest_sink_release(yvex_quant_digest_sink **sink);
void yvex_quant_digest_sink_adapter(yvex_quant_digest_sink *sink,
                                    yvex_quant_output_sink *out);
int yvex_quant_digest_sink_finalize(
    yvex_quant_digest_sink *sink,
    yvex_quant_digest_summary *out,
    yvex_quant_failure *failure,
    yvex_error *err);
int yvex_quant_digest_summary_validate(
    const yvex_quant_digest_summary *summary,
    const char *expected_execution_identity,
    yvex_quant_failure *failure,
    yvex_error *err);
int yvex_quant_digest_sink_terminal_at(
    yvex_quant_digest_sink *sink,
    unsigned long long ordinal,
    yvex_quant_terminal_digest *out,
    yvex_quant_failure *failure,
    yvex_error *err);
size_t yvex_quant_digest_sink_owned_bytes(
    const yvex_quant_digest_sink *sink);
typedef struct {
    atomic_int requested;
} yvex_quant_cancellation;
typedef void *(*yvex_quant_executor_allocate_fn)(size_t size, void *context);
typedef void (*yvex_quant_executor_release_fn)(void *allocation, void *context);
typedef struct {
    unsigned int worker_count;
    size_t source_chunk_bytes;
    size_t output_chunk_bytes;
    size_t maximum_owned_bytes;
    yvex_quant_cancellation *cancellation;
    const yvex_imatrix_data *imatrix;
    yvex_quant_executor_allocate_fn allocate;
    yvex_quant_executor_release_fn release;
    int (*thread_create)(pthread_t *thread,
                         void *(*entry)(void *),
                         void *argument,
                         void *context);
    void *context;
} yvex_quant_executor_options;
typedef struct {
    unsigned long long terminal_decisions;
    unsigned long long terminals_attempted;
    unsigned long long terminals_executed;
    unsigned long long committed_terminals;
    unsigned long long aborted_terminals;
    unsigned long long source_values_consumed;
    unsigned long long source_ranges_read;
    unsigned long long payload_bytes_read;
    unsigned long long source_chunks;
    unsigned long long output_chunks;
    unsigned long long encoded_output_bytes;
    unsigned long long reference_decode_elements;
    unsigned long long qtype_tensor_counts[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    unsigned long long qtype_output_bytes[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    unsigned long long role_tensor_counts[YVEX_TENSOR_ROLE_COUNT];
    yvex_quant_metrics qtype_metrics[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    yvex_quant_metrics role_metrics[YVEX_TENSOR_ROLE_COUNT];
    unsigned long long numeric_bound_violations;
    unsigned long long short_reads;
    unsigned long long payload_drifts;
    unsigned long long sink_failures;
    unsigned long long cancellations;
    unsigned long long worker_failures;
    size_t configured_memory_budget;
    size_t peak_owned_bytes;
    unsigned int configured_workers;
    unsigned int workers_started;
    int complete;
} yvex_quant_execution_summary;
void yvex_quant_executor_options_default(yvex_quant_executor_options *options);
int yvex_quant_execute(
    const yvex_quant_plan *plan,
    const yvex_quant_output_sink *sink,
    const yvex_quant_executor_options *options,
    yvex_quant_execution_summary *summary,
    yvex_quant_failure *failure,
    yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_QUANT_NUMERIC_H_INCLUDED */
