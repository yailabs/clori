/*
 * The reference implementation never calls production attention numeric primitives. Independent
 * test evidence is not a runtime implementation.
 */
#ifndef YVEX_TEST_DEEPSEEK_ATTENTION_REFERENCE_H
#define YVEX_TEST_DEEPSEEK_ATTENTION_REFERENCE_H

#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/graph.h>

#define YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP 192u
#define YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2 2u
#define YVEX_TEST_ATTENTION_EXTERNAL_SCHEMA_V1 1u
#define YVEX_TEST_ATTENTION_EXTERNAL_HADAMARD_WIDTH 3u
#define YVEX_TEST_ATTENTION_EXTERNAL_POSITION_WIDTH 6u
#define YVEX_TEST_ATTENTION_EXTERNAL_BF16_COUNT 3u
#define YVEX_TEST_ATTENTION_EXTERNAL_FP4_COUNT 6u
#define YVEX_TEST_ATTENTION_EXTERNAL_FP8_COUNT 5u
#define YVEX_TEST_ATTENTION_EXTERNAL_COMPRESS_COUNT 4u
#define YVEX_TEST_ATTENTION_EXTERNAL_TOPK_COUNT 6u
#define YVEX_TEST_ATTENTION_EXTERNAL_TOPK_SELECTED 4u
#define YVEX_TEST_ATTENTION_EXTERNAL_HCA_CASES 4u
#define YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH 2u

typedef enum {
    YVEX_TEST_ATTENTION_STAGE_INPUT = 0,
    YVEX_TEST_ATTENTION_STAGE_Q_LOW,
    YVEX_TEST_ATTENTION_STAGE_QUERY,
    YVEX_TEST_ATTENTION_STAGE_RAW_KV,
    YVEX_TEST_ATTENTION_STAGE_COMPRESSED_KV,
    YVEX_TEST_ATTENTION_STAGE_INDEXER_KV,
    YVEX_TEST_ATTENTION_STAGE_INDEX_QUERY,
    YVEX_TEST_ATTENTION_STAGE_INDEX_WEIGHTS,
    YVEX_TEST_ATTENTION_STAGE_ATTENTION,
    YVEX_TEST_ATTENTION_STAGE_OUTPUT,
    YVEX_TEST_ATTENTION_STAGE_MAIN_STATE,
    YVEX_TEST_ATTENTION_STAGE_INDEXER_STATE,
    YVEX_TEST_ATTENTION_STAGE_COUNT
} yvex_test_attention_reference_stage;

typedef struct {
    unsigned int schema_version;
    const char *path_name;
    double absolute[YVEX_TEST_ATTENTION_STAGE_COUNT];
    double relative[YVEX_TEST_ATTENTION_STAGE_COUNT];
    unsigned long long qtype_binding_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    int compressed_positions_exact;
    int indexer_positions_exact;
    int topk_positions_exact;
} yvex_test_attention_reference_contract;

typedef struct {
    unsigned int schema_version;
    char oracle_trace_identity[YVEX_SHA256_HEX_CAP];
    char oracle_output_identity[YVEX_SHA256_HEX_CAP];
    char fixture_identity[YVEX_SHA256_HEX_CAP];
    char history_identity[YVEX_SHA256_HEX_CAP];
    char comparison_contract_identity[YVEX_SHA256_HEX_CAP];
    char evidence_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long exact_topk_positions;
} yvex_test_attention_reference_evidence;

typedef struct {
    unsigned long long compared_values;
    unsigned long long discrete_mismatches;
    unsigned long long first_failed_index;
    double maximum_absolute_error;
    double maximum_relative_error;
    double squared_error_sum;
    const char *first_failed_stage;
} yvex_test_attention_reference_metrics;

typedef struct {
    const yvex_attention_layer_plan *layer;
    const float *residual, *linear_mixes, *scale, *base, *norm_weights, *core_output;
    unsigned long long token_count, residual_stride, mix_stride, core_stride;
    float *core_input, *post, *combination, *envelope_output;
    unsigned long long core_input_stride, post_stride, combination_stride, envelope_stride;
} yvex_test_attention_envelope_case;

/* Bounded literals independently derived from pinned primary semantic authorities.
 * They are immutable test evidence and never become runtime policy. */
typedef struct {
    unsigned int schema_version;
    const char *paper_revision;
    const char *paper_locator;
    const char *sglang_revision;
    const char *sglang_locator;
    const char *vllm_revision;
    const char *vllm_locator;
    const char *hadamard_revision;
    const char *hadamard_locator;
    float hadamard_input[YVEX_TEST_ATTENTION_EXTERNAL_HADAMARD_WIDTH];
    float hadamard_expected[YVEX_TEST_ATTENTION_EXTERNAL_HADAMARD_WIDTH];
    float hadamard_scale;
    float position_input[YVEX_TEST_ATTENTION_EXTERNAL_POSITION_WIDTH];
    float position_forward_expected[YVEX_TEST_ATTENTION_EXTERNAL_POSITION_WIDTH];
    float position_inverse_expected[YVEX_TEST_ATTENTION_EXTERNAL_POSITION_WIDTH];
    unsigned long long position, compressed_source_position, compressed_position;
    unsigned short bf16_expected_codes[YVEX_TEST_ATTENTION_EXTERNAL_BF16_COUNT];
    float bf16_input[YVEX_TEST_ATTENTION_EXTERNAL_BF16_COUNT];
    float bf16_expected[YVEX_TEST_ATTENTION_EXTERNAL_BF16_COUNT];
    float fp4_input[YVEX_TEST_ATTENTION_EXTERNAL_FP4_COUNT];
    unsigned char fp4_expected_codes[YVEX_TEST_ATTENTION_EXTERNAL_FP4_COUNT];
    unsigned char fp4_expected_scale;
    float fp4_expected[YVEX_TEST_ATTENTION_EXTERNAL_FP4_COUNT];
    float fp8_input[YVEX_TEST_ATTENTION_EXTERNAL_FP8_COUNT];
    unsigned char fp8_expected_codes[YVEX_TEST_ATTENTION_EXTERNAL_FP8_COUNT];
    unsigned char fp8_expected_scale;
    float fp8_expected[YVEX_TEST_ATTENTION_EXTERNAL_FP8_COUNT];
    float csa_compress_logits[YVEX_TEST_ATTENTION_EXTERNAL_COMPRESS_COUNT];
    float csa_compress_values[YVEX_TEST_ATTENTION_EXTERNAL_COMPRESS_COUNT];
    float csa_compress_expected;
    float hca_compress_logits[YVEX_TEST_ATTENTION_EXTERNAL_COMPRESS_COUNT];
    float hca_compress_values[YVEX_TEST_ATTENTION_EXTERNAL_COMPRESS_COUNT];
    float hca_compress_expected;
    float topk_scores[YVEX_TEST_ATTENTION_EXTERNAL_TOPK_COUNT];
    unsigned long long topk_positions[YVEX_TEST_ATTENTION_EXTERNAL_TOPK_COUNT];
    unsigned long long topk_expected[YVEX_TEST_ATTENTION_EXTERNAL_TOPK_SELECTED];
    unsigned long long hca_ratio;
    unsigned long long hca_tokens[YVEX_TEST_ATTENTION_EXTERNAL_HCA_CASES];
    unsigned long long hca_emissions[YVEX_TEST_ATTENTION_EXTERNAL_HCA_CASES];
    unsigned long long hca_tails[YVEX_TEST_ATTENTION_EXTERNAL_HCA_CASES];
    float reduction_logits[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH];
    float reduction_values[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH * 2u];
    float reduction_sink_logit;
    float reduction_expected[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH];
    float mhc_core[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH];
    float mhc_residual[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH * 2u];
    float mhc_post[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH];
    float mhc_combination[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH * 2u];
    float mhc_expected[YVEX_TEST_ATTENTION_EXTERNAL_REDUCTION_WIDTH * 2u];
} yvex_test_attention_external_vectors;

typedef struct {
    unsigned int schema_version;
    char vector_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long vector_count;
    int provenance_complete;
    int position_policy_exact;
    int bf16_publication_exact;
    int fp8_fake_quant_exact;
    int fp4_fake_quant_exact;
    int compressor_transition_exact;
    int csa_topk_order_exact;
    int hca_ratio_exact;
    int local_compressed_reduction_exact;
    int envelope_mhc_exact;
    int hadamard_exact;
} yvex_test_attention_external_summary;

const char *yvex_test_attention_status_name(yvex_attention_status status);
const char *yvex_test_attention_failure_name(yvex_attention_failure_code code);

int yvex_test_attention_reference_hadamard(const float *input,
                                           unsigned long long length,
                                           float scale,
                                           int reject_nonfinite,
                                           float *output);
int yvex_test_attention_reference_execute(
    const yvex_attention_layer_plan *layer,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options,
    yvex_attention_execution_trace *trace,
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP]);
int yvex_test_attention_reference_compare(
    const yvex_attention_execution_trace *production,
    const yvex_attention_execution_trace *reference,
    double absolute_tolerance,
    double relative_tolerance,
    yvex_test_attention_reference_metrics *metrics);
int yvex_test_attention_reference_compare_contract(
    const yvex_attention_execution_trace *production,
    const yvex_attention_execution_trace *reference,
    const yvex_test_attention_reference_contract *contract,
    yvex_test_attention_reference_metrics *metrics);
int yvex_test_attention_reference_evidence_build(
    const yvex_attention_execution_trace *reference,
    const yvex_attention_history_view *history,
    const char *attention_plan_identity,
    const yvex_test_attention_reference_contract *contract,
    yvex_test_attention_reference_evidence *evidence);
int yvex_test_attention_reference_topk(
    const float *scores, const unsigned long long *positions,
    unsigned long long count, unsigned long long k,
    unsigned long long *selected, unsigned long long *selected_count);
int yvex_test_attention_reference_envelope(
    const yvex_test_attention_envelope_case *test_case);
const yvex_test_attention_external_vectors *
yvex_test_attention_external_vectors_get(void);
int yvex_test_attention_external_conformance_validate(
    yvex_test_attention_external_summary *summary);

#endif
