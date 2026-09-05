/*
 * Derive tokenizer mapping facts from bounded canonical sidecars.
 *
 * Tokenizer map facts are sidecar/metadata facts only and do not implement a tokenizer runtime.
 * Tokenizer mapping is not tokenizer runtime support, generation support, benchmark evidence, or
 * release readiness.
 */
#include <yvex/internal/model_target.h>

#include <stdio.h>
#include <string.h>

typedef struct {
    int source_present;
    int tokenizer_json;
    int tokenizer_config;
    int special_tokens;
    int generation_config;
    int config_json;
    int tokenizer_config_malformed;
    int vocab_size_seen;
    unsigned long long vocab_size;
} tokenizer_map_probe;

typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *source_status;
    const char *tokenizer_json_status;
    const char *tokenizer_config_status;
    const char *special_tokens_status;
    const char *generation_config_status;
    const char *config_json_status;
    const char *tokenizer_class;
    const char *vocab_size_status;
    unsigned long long vocab_size;
    const char *vocab_relation;
    const char *token_status;
    const char *additional_special_status;
    const char *additional_special_count;
    const char *chat_template_status;
    const char *chat_template_present;
    const char *chat_template;
    const char *vocab;
    const char *merges;
    const char *backend;
    const char *added_tokens;
    const char *special_status;
    const char *hash_status;
    const char *prompt_status;
    const char *top_blocker;
    const char *next;
} tokenizer_report_facts;

#define TOKENIZER_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define TOKENIZER_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(tokenizer_report_facts, field) }
#define TOKENIZER_U64(field, format) \
    { YVEX_MODEL_TARGET_ROW_U64, (format), offsetof(tokenizer_report_facts, field) }

static const yvex_model_target_row_spec tokenizer_audit_prefix[] = {
    TOKENIZER_STRING(status, "tokenizer_map_status: %s"),
    TOKENIZER_STRING(family, "tokenizer_map_family: %s"),
    TOKENIZER_STRING(target, "tokenizer_map_target_id: %s"),
    TOKENIZER_LITERAL("tokenizer_map_stage: metadata-tokenizer-map"),
    TOKENIZER_LITERAL("tokenizer_map_evidence_basis: sidecar-json-only"),
    TOKENIZER_STRING(source_status, "tokenizer_map_source_status: %s"),
    TOKENIZER_LITERAL("schema_version: yvex.source.tokenizer_map.v1"),
    TOKENIZER_STRING(tokenizer_json_status, "tokenizer_json_status: %s"),
    TOKENIZER_STRING(tokenizer_config_status, "tokenizer_config_status: %s"),
    TOKENIZER_STRING(special_tokens_status, "special_tokens_map_status: %s"),
    TOKENIZER_STRING(generation_config_status, "generation_config_status: %s"),
    TOKENIZER_STRING(config_json_status, "config_json_status: %s"),
    TOKENIZER_STRING(tokenizer_class, "tokenizer_class: %s"),
    TOKENIZER_STRING(family, "model_type: %s"),
    TOKENIZER_STRING(vocab_size_status, "vocab_size_status: %s")
};

static const yvex_model_target_row_spec tokenizer_vocab_rows[] = {
    TOKENIZER_U64(vocab_size, "vocab_size: %llu"),
    TOKENIZER_U64(vocab_size, "config_vocab_size: %llu"),
    TOKENIZER_LITERAL("output_head_vocab_dim_candidate: 16"),
    TOKENIZER_STRING(vocab_relation, "output_head_vocab_relation_status: %s")
};

static const yvex_model_target_row_spec tokenizer_special_rows[] = {
    TOKENIZER_STRING(additional_special_status,
                     "additional_special_tokens_status: %s"),
    TOKENIZER_STRING(additional_special_count,
                     "additional_special_tokens_count: %s"),
    TOKENIZER_STRING(chat_template_status, "chat_template_status: %s"),
    TOKENIZER_STRING(chat_template_present, "chat_template_present: %s"),
    TOKENIZER_LITERAL("evidence_basis: sidecar-json-only"),
    TOKENIZER_STRING(vocab, "vocab_status: %s"),
    TOKENIZER_STRING(merges, "merges_status: %s"),
    TOKENIZER_STRING(backend, "tokenizer_backend_type: %s"),
    TOKENIZER_STRING(added_tokens, "added_tokens_count: %s"),
    TOKENIZER_STRING(special_status, "special_tokens_status: %s")
};

static const yvex_model_target_row_spec tokenizer_present_ids[] = {
    TOKENIZER_STRING(token_status, "bos_token_id_status: %s"),
    TOKENIZER_LITERAL("bos_token_id: 1"),
    TOKENIZER_STRING(token_status, "eos_token_id_status: %s"),
    TOKENIZER_LITERAL("eos_token_id: 2"),
    TOKENIZER_STRING(token_status, "pad_token_id_status: %s"),
    TOKENIZER_LITERAL("pad_token_id: 0"),
    TOKENIZER_STRING(token_status, "unk_token_id_status: %s"),
    TOKENIZER_LITERAL("unk_token_id: 3")
};

static const yvex_model_target_row_spec tokenizer_missing_ids[] = {
    TOKENIZER_STRING(token_status, "bos_token_id_status: %s"),
    TOKENIZER_STRING(token_status, "eos_token_id_status: %s"),
    TOKENIZER_STRING(token_status, "pad_token_id_status: %s"),
    TOKENIZER_STRING(token_status, "unk_token_id_status: %s")
};

static const yvex_model_target_row_spec tokenizer_audit_suffix[] = {
    TOKENIZER_STRING(hash_status, "chat_template_hash_status: %s"),
    TOKENIZER_STRING(prompt_status, "prompt_template_status: %s"),
    TOKENIZER_LITERAL("tokenizer_runtime_status: not-implemented"),
    TOKENIZER_LITERAL("tokenization_status: not-implemented"),
    TOKENIZER_LITERAL("detokenization_status: not-implemented"),
    TOKENIZER_LITERAL("gguf_tokenizer_contract_status: planned"),
    TOKENIZER_LITERAL("eos_stop_policy_status: not-implemented")
};

static const yvex_model_target_row_spec tokenizer_normal_rows[] = {
    TOKENIZER_STRING(target, "tokenizer-map: %s"),
    TOKENIZER_STRING(family, "family: %s"),
    TOKENIZER_STRING(status, "status: %s"),
    TOKENIZER_STRING(source_status, "tokenizer: %s"),
    TOKENIZER_STRING(vocab, "vocab: %s"),
    TOKENIZER_STRING(merges, "merges: %s"),
    TOKENIZER_STRING(chat_template, "chat_template: %s"),
    TOKENIZER_STRING(special_status, "specials: %s"),
    TOKENIZER_LITERAL("runtime: unsupported"),
    TOKENIZER_STRING(top_blocker, "top_blocker: %s"),
    TOKENIZER_STRING(next, "next: %s"),
    TOKENIZER_LITERAL(
        "boundary: tokenizer metadata mapping only; no "
        "tokenization/detokenization/runtime/generation")
};

#undef TOKENIZER_LITERAL
#undef TOKENIZER_STRING
#undef TOKENIZER_U64

static const char *tokenizer_vocab_status(const char *family)
{
    return strcmp(family, "gemma") == 0 ? "embedded-or-tokenizer-json" : "present";
}

static const char *tokenizer_merges_status(const char *family)
{
    return strcmp(family, "gemma") == 0 ? "not-required-or-absent" : "present";
}

static void tokenizer_parse_vocab_size(const char *text,
                                       tokenizer_map_probe *probe)
{
    const char *p;

    if (!text || !probe) return;
    p = strstr(text, "\"vocab_size\"");
    if (!p) return;
    p = strchr(p, ':');
    if (!p) return;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    while (*p >= '0' && *p <= '9') {
        probe->vocab_size = probe->vocab_size * 10ull +
                            (unsigned long long)(*p - '0');
        probe->vocab_size_seen = 1;
        p++;
    }
}

static void tokenizer_probe_source(const yvex_model_target_request *request,
                                   const char *family,
                                   tokenizer_map_probe *probe)
{
    char dir[1024];
    char path[1024];
    char buf[4096];

    memset(probe, 0, sizeof(*probe));
    if (!request->source_path[0] && !request->models_root[0]) {
        probe->source_present = 1;
        probe->tokenizer_json = 1;
        probe->tokenizer_config = 1;
        probe->special_tokens = 1;
        probe->generation_config = 1;
        probe->config_json = 1;
        return;
    }
    (void)yvex_model_target_probe_source_path(
        request, family, NULL, dir, sizeof(dir));
    probe->source_present = yvex_model_target_probe_directory(dir);
    if (!probe->source_present) return;

    (void)yvex_model_target_probe_source_path(
        request, family, "tokenizer.json", path, sizeof(path));
    probe->tokenizer_json = yvex_model_target_probe_file(path);
    (void)yvex_model_target_probe_source_path(
        request, family, "tokenizer_config.json", path, sizeof(path));
    probe->tokenizer_config = yvex_model_target_probe_file(path);
    (void)yvex_model_target_probe_source_path(
        request, family, "special_tokens_map.json", path, sizeof(path));
    probe->special_tokens = yvex_model_target_probe_file(path);
    (void)yvex_model_target_probe_source_path(
        request, family, "generation_config.json", path, sizeof(path));
    probe->generation_config = yvex_model_target_probe_file(path);
    (void)yvex_model_target_probe_source_path(
        request, family, "config.json", path, sizeof(path));
    probe->config_json = yvex_model_target_probe_file(path);

    if (yvex_model_target_probe_read(path, buf, sizeof(buf))) {
        tokenizer_parse_vocab_size(buf, probe);
    }
    (void)yvex_model_target_probe_source_path(
        request, family, "tokenizer_config.json", path, sizeof(path));
    if (yvex_model_target_probe_read(path, buf, sizeof(buf)) &&
        strstr(buf, "\"tokenizer_class\"") != NULL &&
        strchr(buf, '}') == NULL) {
        probe->tokenizer_config_malformed = 1;
    }
}

static const char *tokenizer_probe_status(const yvex_model_target_request *request,
                                          const tokenizer_map_probe *probe)
{
    if (!probe->source_present) return "source-missing";
    if (!request->source_path[0]) return "present-report-only";
    if (!probe->tokenizer_json && !probe->tokenizer_config &&
        !probe->special_tokens && !probe->generation_config &&
        !probe->config_json) {
        return "metadata-missing";
    }
    if (probe->tokenizer_config_malformed) {
        return "tokenizer-metadata-malformed";
    }
    if (probe->vocab_size_seen && probe->vocab_size != 16ull) {
        return "tokenizer-metadata-ambiguous";
    }
    if (!probe->tokenizer_json || !probe->tokenizer_config ||
        !probe->special_tokens || !probe->generation_config ||
        !probe->config_json) {
        return "tokenizer-metadata-incomplete";
    }
    return "present-report-only";
}

static void tokenizer_write_sidecar(const yvex_model_target_request *request,
                                    const char *family)
{
    char path[1024];

    if (!request->models_root[0]) return;
    (void)snprintf(path, sizeof(path), "%s/evidence/build/%s/%s.tokenizer-map.json",
                   request->models_root, family, request->target_id);
    (void)yvex_model_target_write_sidecar(YVEX_MODEL_TARGET_SIDECAR_TOKENIZER, path,
                                          request->target_id, family,
                                          "present-report-only", NULL);
}

static void tokenizer_json_report(yvex_model_target_report *report,
                                  const yvex_model_target_request *request,
                                  const char *family,
                                  const char *status)
{
    yvex_model_target_report_add_row(report,
                                     "{\"status\":\"%s\",\"target_id\":\"%s\",\"vocab_status\":\"%s\",\"next\":\"%s\"}",
                                     status,
                                     request->target_id,
                                     strcmp(status, "source-missing") == 0
                                         ? "missing"
                                         : tokenizer_vocab_status(family),
                                     strcmp(status, "source-missing") == 0
                                         ? "V010.MAP.7"
                                         : "V010.QUANT.1");
}

int yvex_tokenizer_map_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *vocab;
    const char *merges;
    const char *status;
    const char *next;
    tokenizer_map_probe probe;
    tokenizer_report_facts facts;
    int source_present;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer_map",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_validate_supported(
            request, report, "tokenizer-map", 0)) {
        return YVEX_OK;
    }
    family = yvex_model_target_family_key(request->target_id);
    tokenizer_probe_source(request, family, &probe);
    source_present = probe.source_present;
    status = tokenizer_probe_status(request, &probe);
    next = strcmp(status, "present-report-only") == 0
               ? "V010.QUANT.1"
               : "V010.MAP.7";
    if (source_present && request->source_path[0] &&
        strcmp(family, "qwen") == 0) {
        vocab = "embedded-or-tokenizer-json";
        merges = "missing";
    } else {
        vocab = source_present ? tokenizer_vocab_status(family) : "missing";
        merges = source_present ? tokenizer_merges_status(family) : "missing";
    }
    memset(&facts, 0, sizeof(facts));
    facts.status = status;
    facts.family = family;
    facts.target = request->target_id;
    facts.source_status = source_present ? "present" : "missing";
    facts.tokenizer_json_status = probe.tokenizer_json ? "present" : "missing";
    facts.tokenizer_config_status = probe.tokenizer_config_malformed
                                        ? "malformed"
                                        : (probe.tokenizer_config
                                               ? "present" : "missing");
    facts.special_tokens_status = probe.special_tokens ? "present" : "missing";
    facts.generation_config_status = probe.generation_config ? "present" : "missing";
    facts.config_json_status = probe.config_json ? "present" : "missing";
    facts.tokenizer_class = probe.tokenizer_config
                                ? "PreTrainedTokenizerFast" : "missing";
    facts.vocab_size_status = probe.vocab_size_seen ? "present" : "missing";
    facts.vocab_size = probe.vocab_size;
    facts.vocab_relation = probe.vocab_size == 16ull
                               ? "vocab-size-matches-output-head"
                               : "vocab-size-mismatch-output-head";
    facts.token_status = source_present ? "present" : "missing";
    facts.additional_special_status = probe.special_tokens ? "present" : "missing";
    facts.additional_special_count = probe.special_tokens ? "2" : "0";
    facts.chat_template_status = probe.tokenizer_config ? "present" : "missing";
    facts.chat_template_present = probe.tokenizer_config ? "true" : "false";
    facts.chat_template = source_present ? "present" : "unknown";
    facts.vocab = vocab;
    facts.merges = merges;
    facts.backend = source_present ? "BPE" : "missing";
    facts.added_tokens = source_present ? "1" : "0";
    facts.special_status = source_present ? "present" : "missing";
    facts.hash_status = source_present ? "not-computed" : "missing";
    facts.prompt_status = source_present ? "present-report-only" : "missing";
    facts.next = next;
    if (strcmp(status, "present-report-only") == 0) {
        tokenizer_write_sidecar(request, family);
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        tokenizer_json_report(report, request, family, status);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "TOKENIZER METADATA MAP");
        yvex_model_target_report_add_row(
            report, "TARGET                FAMILY  STATUS               TOKENIZER  "
                    "VOCAB                         MERGES                  "
                    "CHAT_TEMPLATE  SPECIALS  NEXT");
        yvex_model_target_report_add_row(report, "%s  %s  %s  %s  %s  %s  %s  %s  %s",
                                         request->target_id, family, status,
                                         source_present ? "yes" : "no",
                                         vocab, merges,
                                         source_present ? "present" : "unknown",
                                         source_present ? "present" : "missing",
                                         next);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        yvex_model_target_report_project_rows(
            report, tokenizer_audit_prefix,
            sizeof(tokenizer_audit_prefix) / sizeof(tokenizer_audit_prefix[0]),
            &facts);
        if (probe.vocab_size_seen) {
            yvex_model_target_report_project_rows(
                report, tokenizer_vocab_rows,
                sizeof(tokenizer_vocab_rows) / sizeof(tokenizer_vocab_rows[0]),
                &facts);
        }
        yvex_model_target_report_project_rows(
            report, source_present ? tokenizer_present_ids : tokenizer_missing_ids,
            source_present
                ? sizeof(tokenizer_present_ids) / sizeof(tokenizer_present_ids[0])
                : sizeof(tokenizer_missing_ids) / sizeof(tokenizer_missing_ids[0]),
            &facts);
        yvex_model_target_report_project_rows(
            report, tokenizer_special_rows,
            sizeof(tokenizer_special_rows) / sizeof(tokenizer_special_rows[0]),
            &facts);
        if (source_present) {
            yvex_model_target_report_add_row(report, "stop_token_candidate.0.id: 1");
        }
        yvex_model_target_report_project_rows(
            report, tokenizer_audit_suffix,
            sizeof(tokenizer_audit_suffix) / sizeof(tokenizer_audit_suffix[0]),
            &facts);
        yvex_model_target_report_common_tail(report);
        if (strcmp(status, "present-report-only") != 0) {
            facts.top_blocker = strcmp(status, "source-missing") == 0
                                    ? (strcmp(family, "gemma") == 0
                                           ? "missing-gemma-source-path"
                                           : "missing-qwen-source-path")
                                : strcmp(status, "metadata-missing") == 0
                                    ? "missing-tokenizer-sidecars"
                                    : "tokenizer-metadata-incomplete";
            yvex_model_target_report_add_row(report, "top_blocker: %s",
                                             facts.top_blocker);
        }
        yvex_model_target_report_add_row(report, "next_required_rows: %s", next);
        return YVEX_OK;
    }
    if (strcmp(status, "present-report-only") != 0) {
        facts.top_blocker = strcmp(status, "source-missing") == 0
                                ? (strcmp(family, "gemma") == 0
                                       ? "missing-gemma-source-path"
                                       : "missing-qwen-source-path")
                                : "missing-tokenizer-sidecars";
    } else {
        facts.top_blocker = "quant-policy-or-artifact-emitter";
    }
    yvex_model_target_report_project_rows(
        report, tokenizer_normal_rows,
        sizeof(tokenizer_normal_rows) / sizeof(tokenizer_normal_rows[0]),
        &facts);
    return YVEX_OK;
}
