/* Compile immutable physical package facts without backend topology reconstruction. */
#include <yvex/internal/execution.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
struct yvex_physical_execution_ir {
    yvex_physical_execution_decision *decisions;
    yvex_physical_execution_summary summary;
};
static int physical_ir_identity(yvex_physical_execution_ir *ir, yvex_error *err);
static int execution_refuse(yvex_error *err, yvex_status status,
                            const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}
static int execution_hash_finish(yvex_sha256 *hash,
                                 char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
static int execution_model_schema_supported(unsigned int schema_version)
{
    return schema_version == YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
           schema_version == YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2;
}
static yvex_execution_consumer_class execution_consumer(yvex_tensor_role role)
{
    switch (role) {
    case YVEX_TENSOR_ROLE_TOKEN_EMBEDDING:
        return YVEX_EXECUTION_CONSUMER_EMBEDDING;
    case YVEX_TENSOR_ROLE_OUTPUT_NORM:
    case YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM:
    case YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION:
    case YVEX_TENSOR_ROLE_HC_HEAD_BASE:
    case YVEX_TENSOR_ROLE_HC_HEAD_SCALE:
        return YVEX_EXECUTION_CONSUMER_FINAL_NORMALIZATION;
    case YVEX_TENSOR_ROLE_OUTPUT_HEAD:
        return YVEX_EXECUTION_CONSUMER_OUTPUT_HEAD;
    case YVEX_TENSOR_ROLE_MOE_ROUTER:
    case YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS:
    case YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE:
        return YVEX_EXECUTION_CONSUMER_MOE_ROUTER;
    case YVEX_TENSOR_ROLE_MOE_EXPERT_GATE:
    case YVEX_TENSOR_ROLE_MOE_EXPERT_UP:
        return YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP;
    case YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN:
        return YVEX_EXECUTION_CONSUMER_ROUTED_DOWN;
    case YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE:
    case YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP:
    case YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN:
        return YVEX_EXECUTION_CONSUMER_SHARED_EXPERT;
    case YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION:
    case YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM:
        return YVEX_EXECUTION_CONSUMER_DRAFT_FEATURE_PROJECTION;
    case YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING:
    case YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT:
        return YVEX_EXECUTION_CONSUMER_MARKOV;
    case YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE:
        return YVEX_EXECUTION_CONSUMER_CONFIDENCE;
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV:
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE:
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE:
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM:
    case YVEX_TENSOR_ROLE_INDEXER_PROJECTION:
    case YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM:
        return YVEX_EXECUTION_CONSUMER_ATTENTION_STATE;
    default:
        return role >= YVEX_TENSOR_ROLE_ATTENTION_NORM &&
                       role <= YVEX_TENSOR_ROLE_HC_FFN_SCALE
                   ? YVEX_EXECUTION_CONSUMER_ATTENTION_PROJECTION
                   : YVEX_EXECUTION_CONSUMER_DRAFT_BACKBONE;
    }
}
static int physical_execution_decision_seal(
    yvex_physical_execution_decision *decision, yvex_error *err)
{
    yvex_sha256 hash;
    if (!decision ||
        decision->schema_version != YVEX_PHYSICAL_EXECUTION_SCHEMA_V5 ||
        decision->role <= YVEX_TENSOR_ROLE_UNKNOWN ||
        decision->role >= YVEX_TENSOR_ROLE_COUNT ||
        decision->scope > YVEX_TENSOR_SCOPE_DRAFT ||
        decision->consumer >= YVEX_EXECUTION_CONSUMER_COUNT ||
        decision->layout > YVEX_EXECUTION_LAYOUT_EXPERT_MAJOR ||
        decision->sharing > YVEX_EXECUTION_SHARING_ALIAS ||
        !decision->canonical_row_width || !decision->canonical_row_count ||
        !decision->encoded_bytes || !decision->alignment ||
        !yvex_sha256_hex_valid(decision->terminal_identity))
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "complete physical execution decision facts are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.decision.v5") ||
        !yvex_sha256_update_u64(&hash, decision->schema_version) ||
        !yvex_sha256_update_u64(&hash, decision->terminal_tensor_id) ||
        !yvex_sha256_update_u64(&hash, decision->role) ||
        !yvex_sha256_update_u64(&hash, decision->scope) ||
        !yvex_sha256_update_u64(&hash, decision->layer_index) ||
        !yvex_sha256_update_u64(&hash, decision->predictor_index) ||
        !yvex_sha256_update_u64(&hash, decision->expert_count) ||
        !yvex_sha256_update_u64(&hash, decision->canonical_qtype) ||
        !yvex_sha256_update_u64(&hash, decision->canonical_row_width) ||
        !yvex_sha256_update_u64(&hash, decision->canonical_row_count) ||
        !yvex_sha256_update_u64(&hash, decision->encoded_offset) ||
        !yvex_sha256_update_u64(&hash, decision->encoded_bytes) ||
        !yvex_sha256_update_u64(&hash, decision->alignment) ||
        !yvex_sha256_update_u64(&hash, decision->consumer) ||
        !yvex_sha256_update_u64(&hash, decision->layout) ||
        !yvex_sha256_update_u64(&hash, decision->sharing) ||
        !yvex_sha256_update_text(&hash, decision->terminal_identity) ||
        !execution_hash_finish(&hash, decision->decision_identity))
        return execution_refuse(
            err, YVEX_ERR_STATE, "runtime.execution.physical",
            "physical execution decision identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
static void execution_decision_from_binding(
    yvex_physical_execution_decision *decision,
    const yvex_runtime_tensor_binding *binding,
    const yvex_materialized_tensor_binding *physical)
{
    memset(decision, 0, sizeof(*decision));
    decision->schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V5;
    decision->terminal_tensor_id = binding->tensor_id;
    decision->role = binding->role;
    decision->scope = binding->scope;
    decision->layer_index = binding->layer_index;
    decision->predictor_index = binding->predictor_index;
    decision->expert_count = physical->expert_count;
    decision->canonical_qtype = binding->qtype;
    decision->canonical_row_width = physical->row_width;
    decision->canonical_row_count = physical->row_count;
    decision->encoded_offset = physical->absolute_offset;
    decision->encoded_bytes = physical->encoded_bytes;
    decision->alignment = physical->alignment;
    decision->consumer = execution_consumer(binding->role);
    decision->layout = physical->expert_count > 1ull
                           ? YVEX_EXECUTION_LAYOUT_EXPERT_MAJOR
                           : YVEX_EXECUTION_LAYOUT_CANONICAL_ROW;
    decision->sharing = binding->scope == YVEX_TENSOR_SCOPE_GLOBAL
                            ? YVEX_EXECUTION_SHARING_MODEL_READ_ONLY
                            : YVEX_EXECUTION_SHARING_EXCLUSIVE;
    {
        yvex_sha256 hash;
        yvex_sha256_init(&hash);
        if (yvex_sha256_update_text(&hash, "yvex.terminal-tensor.v1") &&
            yvex_sha256_update_text(&hash, physical->name) &&
            yvex_sha256_update_u64(&hash, binding->tensor_id) &&
            yvex_sha256_update_u64(&hash, binding->role) &&
            yvex_sha256_update_u64(&hash, binding->scope))
            (void)execution_hash_finish(&hash, decision->terminal_identity);
    }
}
typedef int (*physical_binding_at_fn)(
    const void *, unsigned long long, const yvex_runtime_tensor_binding **,
    const yvex_materialized_tensor_binding **);
static int physical_ir_compile(
    yvex_physical_execution_ir **out, unsigned long long count,
    const char *physical_variant_identity, physical_binding_at_fn binding_at,
    const void *context, yvex_error *err)
{
    yvex_physical_execution_ir *ir;
    unsigned long long index;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !count || !yvex_sha256_hex_valid(physical_variant_identity) ||
        count > SIZE_MAX / sizeof(*ir->decisions) || !binding_at)
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "physical execution compiler inputs are incomplete");
    ir = calloc(1u, sizeof(*ir));
    if (ir) ir->decisions = calloc((size_t)count, sizeof(*ir->decisions));
    if (!ir || !ir->decisions) {
        yvex_physical_execution_ir_close(&ir);
        return execution_refuse(err, YVEX_ERR_NOMEM,
                                "runtime.execution.physical",
                                "physical execution IR allocation failed");
    }
    ir->summary.schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V5;
    ir->summary.decision_count = count;
    yvex_core_text_copy(ir->summary.physical_variant_identity,
                        sizeof(ir->summary.physical_variant_identity),
                        physical_variant_identity);
    for (index = 0ull; index < count; ++index) {
        const yvex_runtime_tensor_binding *binding = NULL;
        const yvex_materialized_tensor_binding *physical = NULL;
        yvex_physical_execution_decision *decision = &ir->decisions[index];
        if (!binding_at(context, index, &binding, &physical) || !binding || !physical) {
            rc = execution_refuse(err, YVEX_ERR_FORMAT,
                                  "runtime.execution.physical",
                                  "runtime descriptor contains an orphan tensor");
            break;
        }
        execution_decision_from_binding(decision, binding, physical);
        rc = physical_execution_decision_seal(decision, err);
        if (rc != YVEX_OK) break;
        ir->summary.consumer_counts[decision->consumer]++;
        ir->summary.layout_counts[decision->layout]++;
        if (!yvex_core_u64_add(ir->summary.encoded_bytes,
                               decision->encoded_bytes,
                               &ir->summary.encoded_bytes)) {
            rc = execution_refuse(err, YVEX_ERR_BOUNDS,
                                  "runtime.execution.physical",
                                  "physical execution byte accounting overflowed");
            break;
        }
    }
    if (rc == YVEX_OK) rc = physical_ir_identity(ir, err);
    if (rc != YVEX_OK) {
        yvex_physical_execution_ir_close(&ir);
        return rc;
    }
    *out = ir;
    yvex_error_clear(err);
    return YVEX_OK;
}
typedef struct {
    const yvex_runtime_descriptor *descriptor;
} physical_descriptor_source;
static int physical_descriptor_binding_at(
    const void *opaque, unsigned long long index,
    const yvex_runtime_tensor_binding **binding,
    const yvex_materialized_tensor_binding **physical)
{
    const physical_descriptor_source *source = opaque;
    *binding = source ? yvex_runtime_descriptor_tensor_at(source->descriptor, index) : NULL;
    *physical = *binding ? (*binding)->binding : NULL;
    return *binding && *physical;
}
static int physical_ir_identity(yvex_physical_execution_ir *ir,
                                yvex_error *err)
{
    yvex_sha256 hash;
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.ir.v5") ||
        !yvex_sha256_update_u64(&hash, ir->summary.schema_version) ||
        !yvex_sha256_update_text(&hash,
                                 ir->summary.physical_variant_identity) ||
        !yvex_sha256_update_u64(&hash, ir->summary.decision_count) ||
        !yvex_sha256_update_u64(&hash, ir->summary.encoded_bytes))
        goto failed;
    for (index = 0ull; index < ir->summary.decision_count; ++index)
        if (!yvex_sha256_update_text(
                &hash, ir->decisions[index].decision_identity))
            goto failed;
    if (!execution_hash_finish(&hash, ir->summary.identity)) goto failed;
    return YVEX_OK;
failed:
    return execution_refuse(err, YVEX_ERR_STATE,
                            "runtime.execution.physical",
                            "physical execution IR identity derivation failed");
}
int yvex_physical_execution_ir_build(
    yvex_physical_execution_ir **out,
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor,
    const char *physical_variant_identity, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *descriptor_summary;
    const yvex_materialization_summary *materialization_summary;
    physical_descriptor_source source;
    if (out) *out = NULL;
    descriptor_summary = yvex_runtime_descriptor_summary_get(descriptor);
    materialization_summary =
        yvex_materialization_session_summary(materialization);
    if (!out || !descriptor_summary || !materialization_summary ||
        descriptor_summary->status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        materialization_summary->status != YVEX_MATERIALIZATION_STATUS_COMMITTED ||
        descriptor_summary->tensor_count != materialization_summary->tensor_count ||
        !descriptor_summary->tensor_count ||
        !execution_model_schema_supported(
            descriptor_summary->model_execution.schema_version) ||
        !yvex_sha256_hex_valid(physical_variant_identity))
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "admitted descriptor, materialization, and physical variant are required");
    source.descriptor = descriptor;
    return physical_ir_compile(
        out, descriptor_summary->tensor_count, physical_variant_identity,
        physical_descriptor_binding_at, &source, err);
}

int yvex_physical_execution_ir_import(
    yvex_physical_execution_ir **out, const yvex_physical_execution_summary *summary,
    const yvex_physical_execution_decision *decisions, unsigned long long count,
    yvex_error *err)
{
    yvex_physical_execution_ir *ir;
    unsigned long long index;
    char expected[YVEX_SHA256_HEX_CAP];
    int authenticated;
    if (out) *out = NULL;
    if (!out || !summary || !decisions || !count || summary->decision_count != count ||
        summary->schema_version != YVEX_PHYSICAL_EXECUTION_SCHEMA_V5 ||
        !yvex_sha256_hex_valid(summary->physical_variant_identity) ||
        count > SIZE_MAX / sizeof(*ir->decisions))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.physical",
                                "sealed physical execution records are required");
    authenticated = summary->identity[0] != '\0';
    ir = calloc(1u, sizeof(*ir));
    if (ir) ir->decisions = calloc((size_t)count, sizeof(*ir->decisions));
    if (!ir || !ir->decisions) {
        yvex_physical_execution_ir_close(&ir);
        return execution_refuse(err, YVEX_ERR_NOMEM,
                                "runtime.execution.physical",
                                "physical execution import allocation failed");
    }
    ir->summary.schema_version = summary->schema_version;
    ir->summary.decision_count = count;
    yvex_core_text_copy(ir->summary.physical_variant_identity,
                        sizeof(ir->summary.physical_variant_identity),
                        summary->physical_variant_identity);
    for (index = 0ull; index < count; ++index) {
        ir->decisions[index] = decisions[index];
        yvex_core_text_copy(expected, sizeof(expected), decisions[index].decision_identity);
        if (physical_execution_decision_seal(&ir->decisions[index], err) != YVEX_OK ||
            (authenticated && strcmp(expected, ir->decisions[index].decision_identity) != 0))
            goto refused;
        ir->summary.consumer_counts[ir->decisions[index].consumer]++;
        ir->summary.layout_counts[ir->decisions[index].layout]++;
        if (!yvex_core_u64_add(ir->summary.encoded_bytes,
                               ir->decisions[index].encoded_bytes,
                               &ir->summary.encoded_bytes))
            goto refused;
    }
    if (authenticated &&
        (ir->summary.encoded_bytes != summary->encoded_bytes ||
         memcmp(ir->summary.consumer_counts, summary->consumer_counts,
                sizeof(ir->summary.consumer_counts)) != 0 ||
         memcmp(ir->summary.layout_counts, summary->layout_counts,
                sizeof(ir->summary.layout_counts)) != 0))
        goto refused;
    yvex_core_text_copy(expected, sizeof(expected), summary->identity);
    if (physical_ir_identity(ir, err) != YVEX_OK ||
        (authenticated && strcmp(expected, ir->summary.identity) != 0))
        goto refused;
    *out = ir;
    return YVEX_OK;
refused:
    yvex_physical_execution_ir_close(&ir);
    return execution_refuse(err, YVEX_ERR_FORMAT,
                            "runtime.execution.physical",
                            "physical execution records failed identity validation");
}
const yvex_physical_execution_summary *yvex_physical_execution_ir_summary(
    const yvex_physical_execution_ir *ir)
{
    return ir ? &ir->summary : NULL;
}
const yvex_physical_execution_decision *yvex_physical_execution_ir_decision_at(
    const yvex_physical_execution_ir *ir, unsigned long long index)
{
    return ir && index < ir->summary.decision_count ? &ir->decisions[index] : NULL;
}
void yvex_physical_execution_ir_close(yvex_physical_execution_ir **ir)
{
    if (!ir || !*ir) return;
    free((*ir)->decisions);
    memset(*ir, 0, sizeof(**ir));
    free(*ir);
    *ir = NULL;
}
