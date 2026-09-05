/*
 * Bind one immutable vocabulary projection to its exact hidden-state producer.
 *
 * The output head is shared by distinct decoder architectures.  Its identity names the
 * producing execution plan explicitly; runtime must never infer that lineage from dimensions.
 */
#ifndef INCLUDE_YVEX_INTERNAL_OUTPUT_HEAD_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_OUTPUT_HEAD_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/model.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_LOGITS_SCHEMA_V3 3u
#define YVEX_RUNTIME_LOGITS_SCHEMA_V2 YVEX_RUNTIME_LOGITS_SCHEMA_V3
#define YVEX_RUNTIME_LOGITS_SCHEMA_V1 YVEX_RUNTIME_LOGITS_SCHEMA_V3
#define YVEX_OUTPUT_HEAD_PLAN_SCHEMA_V1 YVEX_RUNTIME_LOGITS_SCHEMA_V3
#define YVEX_OUTPUT_HEAD_PLAN_SCHEMA_V2 4u
#define YVEX_OUTPUT_HEAD_PLAN_SCHEMA_CURRENT YVEX_OUTPUT_HEAD_PLAN_SCHEMA_V2

typedef struct yvex_logits_family_policy {
    unsigned int schema_version;
    int separate_output_head, tied_output_head, output_head_bias;
} yvex_logits_family_policy;

typedef enum {
    YVEX_EXECUTION_PLAN_UNKNOWN = 0,
    YVEX_EXECUTION_PLAN_TRANSFORMER,
    YVEX_EXECUTION_PLAN_DECODER
} yvex_execution_plan_kind;

typedef struct yvex_runtime_logits_plan_summary {
    unsigned int schema_version;
    yvex_execution_plan_kind producer_kind;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long output_head_tensor_id, row_width, row_count, row_bytes;
    unsigned long long encoded_bytes, vocabulary_size, hidden_width;
    yvex_tensor_role role;
    unsigned int qtype;
    int separate_output_head, output_head_bias;
    char artifact_identity[YVEX_SHA256_HEX_BYTES];
    char materialization_identity[YVEX_SHA256_HEX_BYTES];
    char logical_model_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_numeric_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_BYTES];
    char transformer_plan_identity[YVEX_SHA256_HEX_BYTES];
    char decoder_plan_identity[YVEX_SHA256_HEX_BYTES];
    char output_head_plan_identity[YVEX_SHA256_HEX_BYTES];
} yvex_runtime_logits_plan_summary;

struct yvex_materialization_session;
struct yvex_runtime_descriptor;
struct yvex_transformer_plan;
struct yvex_decoder_plan;

int yvex_output_head_plan_build_transformer(
    yvex_runtime_logits_plan_summary *out,
    unsigned long long family_adapter_id,
    unsigned long long family_adapter_version,
    const struct yvex_materialization_session *materialization,
    const struct yvex_runtime_descriptor *descriptor,
    const struct yvex_transformer_plan *transformer,
    const yvex_logits_family_policy *policy, yvex_error *err);
int yvex_output_head_plan_build_decoder(
    yvex_runtime_logits_plan_summary *out,
    unsigned long long family_adapter_id,
    unsigned long long family_adapter_version,
    const struct yvex_materialization_session *materialization,
    const struct yvex_runtime_descriptor *descriptor,
    const struct yvex_decoder_plan *decoder,
    const yvex_logits_family_policy *policy, yvex_error *err);
int yvex_output_head_plan_seal(
    yvex_runtime_logits_plan_summary *summary, yvex_error *err);
int yvex_output_head_plan_validate(
    const yvex_runtime_logits_plan_summary *summary, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
