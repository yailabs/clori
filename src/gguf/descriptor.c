/*
 * Centralize the small pure projections that define one GGUF tensor descriptor.
 *
 * Every projection is checked, deterministic, and independent of mutable runtime state. Structural
 * descriptor acceptance does not imply artifact or execution support.
 */
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <yvex/internal/gguf.h>

/*
 * Map one admitted role to its pinned GGUF name or layer-local suffix.
 *
 * This is format naming, not logical tensor identity.
 */
typedef struct {
    const char *name;
    int layer_scoped;
} gguf_role_name;

static const gguf_role_name gguf_role_names[YVEX_TENSOR_ROLE_COUNT] = {
    [YVEX_TENSOR_ROLE_TOKEN_EMBEDDING] = {"token_embd.weight", 0},
    [YVEX_TENSOR_ROLE_OUTPUT_NORM] = {"output_norm.weight", 0},
    [YVEX_TENSOR_ROLE_OUTPUT_HEAD] = {"output.weight", 0},
    [YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION] = {"output_hc_fn.weight", 0},
    [YVEX_TENSOR_ROLE_HC_HEAD_BASE] = {"output_hc_base.weight", 0},
    [YVEX_TENSOR_ROLE_HC_HEAD_SCALE] = {"output_hc_scale.weight", 0},
    [YVEX_TENSOR_ROLE_ATTENTION_NORM] = {"attn_norm.weight", 1},
    [YVEX_TENSOR_ROLE_FFN_NORM] = {"ffn_norm.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_SINKS] = {"attn_sinks.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_Q_A] = {"attn_q_a.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_Q_B] = {"attn_q_b.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM] = {"attn_q_a_norm.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_KV] = {"attn_kv.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_KV_NORM] = {"attn_kv_a_norm.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_OUT_A] = {"attn_output_a.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_OUT_B] = {"attn_output_b.weight", 1},
    [YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION] = {"hc_attn_fn.weight", 1},
    [YVEX_TENSOR_ROLE_HC_ATTENTION_BASE] = {"hc_attn_base.weight", 1},
    [YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE] = {"hc_attn_scale.weight", 1},
    [YVEX_TENSOR_ROLE_HC_FFN_FUNCTION] = {"hc_ffn_fn.weight", 1},
    [YVEX_TENSOR_ROLE_HC_FFN_BASE] = {"hc_ffn_base.weight", 1},
    [YVEX_TENSOR_ROLE_HC_FFN_SCALE] = {"hc_ffn_scale.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV] = {"attn_compressor_kv.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE] = {"attn_compressor_gate.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE] = {"attn_compressor_ape.weight", 1},
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM] = {"attn_compressor_norm.weight", 1},
    [YVEX_TENSOR_ROLE_INDEXER_PROJECTION] = {"indexer.proj.weight", 1},
    [YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B] = {"indexer.attn_q_b.weight", 1},
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV] = {"indexer_compressor_kv.weight", 1},
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE] = {"indexer_compressor_gate.weight", 1},
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE] = {"indexer_compressor_ape.weight", 1},
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM] = {"indexer_compressor_norm.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_ROUTER] = {"ffn_gate_inp.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS] = {"exp_probs_b.bias", 1},
    [YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE] = {"ffn_gate_tid2eid.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_EXPERT_GATE] = {"ffn_gate_exps.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN] = {"ffn_down_exps.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_EXPERT_UP] = {"ffn_up_exps.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE] = {"ffn_gate_shexp.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN] = {"ffn_down_shexp.weight", 1},
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP] = {"ffn_up_shexp.weight", 1},
};

static const char *gguf_standard_role_name(yvex_tensor_role role, int *layer_scoped) {
    if (role <= YVEX_TENSOR_ROLE_UNKNOWN || role >= YVEX_TENSOR_ROLE_COUNT)
        return NULL;
    *layer_scoped = gguf_role_names[role].layer_scoped;
    return gguf_role_names[role].name;
}

/*
 * Resolve one typed role and scope into a deterministic GGUF tensor name.
 *
 * Writes one bounded name and provenance value. Emitted names remain format facts and do not
 * define logical model identity.
 */
int yvex_gguf_name_map_resolve(yvex_tensor_role role, int mtp_extension,
                               unsigned long long layer_index, unsigned long long predictor_index,
                               char *out, size_t out_cap, yvex_gguf_name_provenance *provenance,
                               const char **reason) {
    const char *name;
    int layer_scoped;
    int written;

    if (!out || out_cap == 0u || !provenance || role <= YVEX_TENSOR_ROLE_UNKNOWN ||
        role >= YVEX_TENSOR_ROLE_COUNT) {
        if (reason)
            *reason = "invalid typed role or output buffer";
        return 0;
    }
    if (mtp_extension) {
        written = snprintf(out, out_cap, "yvex.mtp.v1.%llu.%s.weight", predictor_index,
                           yvex_tensor_role_name(role));
        *provenance = YVEX_GGUF_NAME_YVEX_EXTENSION;
    } else {
        name = gguf_standard_role_name(role, &layer_scoped);
        if (!name) {
            if (reason)
                *reason = "role has no pinned DeepSeek-V4 GGUF name";
            return 0;
        }
        if (layer_scoped) {
            written = snprintf(out, out_cap, "blk.%llu.%s", layer_index, name);
        } else {
            written = snprintf(out, out_cap, "%s", name);
        }
        *provenance = YVEX_GGUF_NAME_PINNED_STANDARD;
    }
    if (written < 0 || (size_t)written >= out_cap) {
        if (reason)
            *reason = "emitted GGUF name exceeds mapping bounds";
        return 0;
    }
    if (reason)
        *reason = "admitted canonical logical name";
    return 1;
}

int yvex_gguf_layout_map_shape_supported(yvex_tensor_role role, unsigned int qtype,
                                         unsigned int rank, const unsigned long long *dims,
                                         const char **reason) {
    unsigned int i;

    if (role <= YVEX_TENSOR_ROLE_UNKNOWN || role >= YVEX_TENSOR_ROLE_COUNT || !dims || rank == 0u ||
        rank > YVEX_TENSOR_MAX_DIMS) {
        if (reason)
            *reason = "invalid role, rank, or dimensions";
        return 0;
    }
    for (i = 0u; i < rank; ++i) {
        if (!dims[i]) {
            if (reason)
                *reason = "logical GGML dimension is zero";
            return 0;
        }
    }
    if (qtype != YVEX_GGUF_NO_FORCED_QTYPE) {
        yvex_gguf_qtype_storage_result storage;
        if (yvex_gguf_qtype_tensor_storage(qtype, dims, rank, &storage) !=
            YVEX_GGUF_QTYPE_STORAGE_OK) {
            if (reason)
                *reason = storage.reason;
            return 0;
        }
    }
    if (reason)
        *reason = "admitted logical GGML shape";
    return 1;
}

static int is_empty(const char *s) {
    return !s || s[0] == '\0';
}

static int contains_space(const char *s) {
    while (s && *s) {
        if (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')
            return 1;
        ++s;
    }
    return 0;
}

static int has_ambiguous_word(const char *s) {
    return s && (strstr(s, "test") || strstr(s, "final") || strstr(s, "new") ||
                 strstr(s, "fixed") || strstr(s, "latest"));
}

static int validate_part(const char *part, const char *name, yvex_error *err) {
    if (is_empty(part)) {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_name_suggest", "%s is required",
                        name);
        return YVEX_ERR_INVALID_ARG;
    }
    if (contains_space(part)) {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_name_suggest",
                        "%s must not contain spaces", name);
        return YVEX_ERR_INVALID_ARG;
    }
    if (has_ambiguous_word(part)) {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_name_suggest",
                        "%s contains ambiguous naming vocabulary", name);
        return YVEX_ERR_INVALID_ARG;
    }
    return YVEX_OK;
}

/*
 * Construct one deterministic artifact filename from admitted semantic components.
 *
 * Bounded output plus repository, revision, model, precision, hardware, and version text. Naming
 * does not create, admit, or publish an artifact.
 */
int yvex_artifact_name_suggest(char *out, size_t out_size, const char *family, const char *model,
                               const char *scope, const char *artifact_class, const char *qprofile,
                               const char *calibration, const char *producer, const char *schema,
                               yvex_error *err) {
    int n;
    int rc;

    if (!out || out_size == 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_name_suggest",
                       "output buffer is required");
        return YVEX_ERR_INVALID_ARG;
    }
    out[0] = '\0';
    rc = validate_part(family, "family", err);
    if (rc != YVEX_OK)
        return rc;
    rc = validate_part(model, "model", err);
    if (rc != YVEX_OK)
        return rc;
    rc = validate_part(scope, "scope", err);
    if (rc != YVEX_OK)
        return rc;
    rc = validate_part(artifact_class, "artifact_class", err);
    if (rc != YVEX_OK)
        return rc;
    rc = validate_part(qprofile, "qprofile", err);
    if (rc != YVEX_OK)
        return rc;
    rc = validate_part(calibration, "calibration", err);
    if (rc != YVEX_OK)
        return rc;
    rc = validate_part(producer, "producer", err);
    if (rc != YVEX_OK)
        return rc;
    rc = validate_part(schema, "schema", err);
    if (rc != YVEX_OK)
        return rc;
    if (strcmp(producer, "yvex") != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_name_suggest",
                       "producer must be yvex");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(schema, "v1") != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_name_suggest",
                       "schema must be v1");
        return YVEX_ERR_INVALID_ARG;
    }

    n = snprintf(out, out_size, "%s-%s-%s-%s-%s-%s-%s-%s.gguf", family, model, scope,
                 artifact_class, qprofile, calibration, producer, schema);
    if (n < 0 || (size_t)n >= out_size) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_artifact_name_suggest",
                       "artifact filename buffer too small");
        out[0] = '\0';
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}
