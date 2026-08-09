/*
 * Seal the family-neutral semantic identity and context capability before graph lowering.
 *
 * The optional family payload is process-local compiler state. It never participates by address
 * in identity and never crosses the compiled binding boundary; only its sealed identity does.
 */
#include <yvex/internal/compiler.h>

#include <stdlib.h>
#include <string.h>

struct yvex_semantic_model_ir {
    yvex_semantic_model_ir_summary summary;
    void *family_payload;
    yvex_semantic_model_payload_close_fn family_payload_close;
};

static int semantic_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "compilation.semantic-model", reason);
    return status;
}

static int semantic_identity(yvex_semantic_model_ir_summary *summary)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.semantic-model-ir.v1") ||
        !yvex_sha256_update_u64(&hash, summary->schema_version) ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_version) ||
        !yvex_sha256_update_text(&hash, summary->target_id) ||
        !yvex_sha256_update_text(&hash, summary->source_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->semantic_payload_identity) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned int)summary->context_capability_present) ||
        !yvex_sha256_update_u64(&hash, summary->maximum_context) ||
        !yvex_sha256_update_u64(&hash, summary->original_context) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, summary->identity);
    return 1;
}

int yvex_semantic_model_ir_seal(
    yvex_semantic_model_ir **out,
    const yvex_semantic_model_ir_request *request, yvex_error *err)
{
    yvex_semantic_model_ir *model;
    if (out) *out = NULL;
    if (!out || !request ||
        request->schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1 ||
        !request->family_adapter_id || !request->family_adapter_version ||
        !request->target_id || !request->target_id[0] ||
        strlen(request->target_id) >=
            sizeof(((yvex_semantic_model_ir_summary *)0)->target_id) ||
        !yvex_sha256_hex_valid(request->source_model_identity) ||
        !yvex_sha256_hex_valid(request->logical_model_identity) ||
        !yvex_sha256_hex_valid(request->semantic_payload_identity) ||
        (request->context_capability_present &&
         (!request->maximum_context || !request->original_context ||
          request->original_context > request->maximum_context)) ||
        (!request->context_capability_present &&
         (request->maximum_context || request->original_context)) ||
        (!request->family_payload &&
         (request->family_payload_owned || request->family_payload_close)) ||
        (request->family_payload_owned && !request->family_payload_close) ||
        (!request->family_payload_owned && request->family_payload_close))
        return semantic_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "complete immutable semantic facts and balanced payload ownership are required");
    model = calloc(1u, sizeof(*model));
    if (!model)
        return semantic_refuse(err, YVEX_ERR_NOMEM,
                               "semantic model allocation failed");
    model->summary.schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1;
    model->summary.family_adapter_id = request->family_adapter_id;
    model->summary.family_adapter_version = request->family_adapter_version;
    model->summary.maximum_context = request->maximum_context;
    model->summary.original_context = request->original_context;
    model->summary.context_capability_present = request->context_capability_present;
    yvex_core_text_copy(model->summary.target_id,
                        sizeof(model->summary.target_id), request->target_id);
    yvex_core_text_copy(model->summary.source_model_identity,
                        sizeof(model->summary.source_model_identity),
                        request->source_model_identity);
    yvex_core_text_copy(model->summary.logical_model_identity,
                        sizeof(model->summary.logical_model_identity),
                        request->logical_model_identity);
    yvex_core_text_copy(model->summary.semantic_payload_identity,
                        sizeof(model->summary.semantic_payload_identity),
                        request->semantic_payload_identity);
    if (!semantic_identity(&model->summary)) {
        free(model);
        return semantic_refuse(err, YVEX_ERR_STATE,
                               "semantic model identity derivation failed");
    }
    model->family_payload = request->family_payload;
    model->family_payload_close = request->family_payload_owned
                                      ? request->family_payload_close : NULL;
    *out = model;
    yvex_error_clear(err);
    return YVEX_OK;
}

const yvex_semantic_model_ir_summary *yvex_semantic_model_ir_summary_get(
    const yvex_semantic_model_ir *model)
{
    return model ? &model->summary : NULL;
}

const void *yvex_semantic_model_ir_family_payload(
    const yvex_semantic_model_ir *model,
    unsigned long long family_adapter_id,
    unsigned long long family_adapter_version)
{
    return model && model->summary.family_adapter_id == family_adapter_id &&
                   model->summary.family_adapter_version == family_adapter_version
               ? model->family_payload : NULL;
}

void yvex_semantic_model_ir_close(yvex_semantic_model_ir **model)
{
    yvex_semantic_model_ir *owner;
    if (!model || !*model) return;
    owner = *model;
    if (owner->family_payload_close)
        owner->family_payload_close(owner->family_payload);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *model = NULL;
}
