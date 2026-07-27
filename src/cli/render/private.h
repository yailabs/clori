/* Owner: cli.render.private (cli.render).
 * Owns: domain render entrypoints and shared output formatting.
 * Does not own: domain policy, input parsing, or capability truth.
 * Invariants: declarations have one owner, stable ordering, and no hidden capability promotion.
 * Boundary: CLI presentation over typed reports.
 * Purpose: provide the canonical CLI presentation over typed reports contract.
 * Inputs: typed immutable facts and explicitly owned mutable lifecycle objects.
 * Effects: only declared lifecycle, allocation, I/O, and publication operations mutate state.
 * Failure: typed refusals leave outputs defined and preserve caller-owned state. */
#ifndef SRC_CLI_RENDER_PRIVATE_H_INCLUDED
#define SRC_CLI_RENDER_PRIVATE_H_INCLUDED

#include "src/cli/io/private.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/model_artifact.h>
#include <yvex/internal/model_target.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>
#include <yvex/internal/source_payload.h>


#ifdef __cplusplus
extern "C" {
#endif

/* Private contract. */
typedef enum {
    YVEX_RENDER_MODE_PORCELAIN = 0,
    YVEX_RENDER_MODE_TABLE,
    YVEX_RENDER_MODE_AUDIT
} yvex_render_mode;
typedef struct {
    FILE *fp;
    yvex_render_mode mode;
} yvex_render_out;
/* Purpose: read one immutable collection counter selected by a renderer table. */
static inline unsigned long long cli_collection_value(
    const yvex_fullmodel_collections *collections, size_t offset) {
    if (!collections || offset == (size_t)-1)
        return 0ull;
    return *(const unsigned long long *)((const unsigned char *)collections + offset);
}

typedef yvex_cli_field_kind yvex_render_field_kind;
typedef yvex_cli_field_spec yvex_render_field_spec;
#define YVEX_RENDER_FIELD_TEXT YVEX_CLI_FIELD_TEXT
#define YVEX_RENDER_FIELD_TEXT_ARRAY YVEX_CLI_FIELD_TEXT_ARRAY
#define YVEX_RENDER_FIELD_U64 YVEX_CLI_FIELD_U64
#define YVEX_RENDER_FIELD_U32 YVEX_CLI_FIELD_U32
#define YVEX_RENDER_FIELD_I32 YVEX_CLI_FIELD_I32
#define YVEX_RENDER_FIELD_BOOL YVEX_CLI_FIELD_BOOL
#define YVEX_RENDER_FIELD_DOUBLE YVEX_CLI_FIELD_DOUBLE
#define render_object_fields yvex_cli_out_fields
static inline void render_out_init(yvex_render_out *out, FILE *fp, yvex_render_mode mode) {
    if (!out) {
        return;
    }
    out->fp = fp ? fp : stdout;
    out->mode = mode;
}
static inline FILE *render_fp(const yvex_render_out *out) {
    return out && out->fp ? out->fp : stdout;
}
static inline const char *render_text(const char *text) {
    return text && text[0] ? text : "unknown";
}
static inline void render_report_title(const yvex_render_out *out, const char *report,
                                       const char *subject, const char *state) {
    FILE *fp = render_fp(out);
    yvex_cli_out_writef(fp, "%s: %s", render_text(report), render_text(subject));
    if (state && state[0]) {
        yvex_cli_out_writef(fp, " [%s]", state);
    }
    yvex_cli_out_writef(fp, "\n");
}
static inline void render_kv(const yvex_render_out *out, const char *key, const char *value) {
    yvex_cli_out_writef(render_fp(out), "%s: %s\n", render_text(key), render_text(value));
}
static inline void render_kv_u(const yvex_render_out *out, const char *key, unsigned int value) {
    yvex_cli_out_writef(render_fp(out), "%s: %u\n", render_text(key), value);
}
static inline void render_status(const yvex_render_out *out, const char *value) {
    render_kv(out, "status", value);
}
static inline void render_top_blocker(const yvex_render_out *out, const char *value) {
    render_kv(out, "top_blocker", value);
}
static inline void render_next(const yvex_render_out *out, const char *value) {
    render_kv(out, "next", value);
}
static inline void render_boundary(const yvex_render_out *out, const char *value) {
    render_kv(out, "boundary", value);
}
static inline void render_section(const yvex_render_out *out, const char *title) {
    yvex_cli_out_writef(render_fp(out), "%s\n", render_text(title));
}
static inline void render_fields2(const yvex_render_out *out, const char *key0, const char *value0,
                                  const char *key1, const char *value1) {
    yvex_cli_out_writef(render_fp(out), "%s: %s  %s: %s\n", render_text(key0), render_text(value0),
                        render_text(key1), render_text(value1));
}
static inline void render_fields3(const yvex_render_out *out, const char *key0, const char *value0,
                                  const char *key1, const char *value1, const char *key2,
                                  const char *value2) {
    yvex_cli_out_writef(render_fp(out), "%s: %s  %s: %s  %s: %s\n", render_text(key0),
                        render_text(value0), render_text(key1), render_text(value1),
                        render_text(key2), render_text(value2));
}
static inline void render_table_header(const yvex_render_out *out, const char *header) {
    yvex_cli_out_writef(render_fp(out), "%s\n", render_text(header));
}
static inline void render_table_row(const yvex_render_out *out, const char *row) {
    yvex_cli_out_writef(render_fp(out), "%s\n", render_text(row));
}

/* Emit one immutable declarative line set without reconstructing domain facts. */
static inline void render_lines(FILE *fp, const char *const *lines, size_t line_count) {
    yvex_cli_out_lines(fp, lines, line_count);
}

/* Backend contract. */
int yvex_backend_render(FILE *fp, const yvex_backend_report *report);
int yvex_backend_render_help(FILE *fp);
int yvex_cuda_info_render_help(FILE *fp);

/* Graph contract. */
int yvex_graph_attention_render(FILE *fp, yvex_graph_report_mode mode,
                                const yvex_graph_attention_operator_result *result);
int yvex_graph_moe_render(FILE *fp, yvex_graph_report_mode mode,
                          const yvex_moe_operator_result *result);
int yvex_graph_transformer_render(FILE *fp, yvex_graph_report_mode mode,
                                  const yvex_transformer_operator_result *result);
int yvex_graph_decode_render(FILE *fp, yvex_graph_report_mode mode,
                             const yvex_decode_operator_result *result);
int yvex_graph_render_help(FILE *fp);

/* Model Target contract. */
int yvex_model_target_render(FILE *fp, yvex_model_target_render_mode mode,
                             const yvex_model_target_report *report);
int yvex_model_target_render_errors(FILE *fp, const yvex_model_target_report *report);
int yvex_model_target_render_help(FILE *fp);

/* Source contract. */
int yvex_source_render(FILE *fp, yvex_source_render_mode mode, const yvex_source_report *report);
int yvex_source_render_normal(FILE *fp, const yvex_source_report *report);
int yvex_source_render_table(FILE *fp, const yvex_source_report *report);
int yvex_source_render_audit(FILE *fp, const yvex_source_report *report);
int yvex_source_render_json(FILE *fp, const yvex_source_report *report);
void yvex_source_render_help(FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* SRC_CLI_RENDER_PRIVATE_H_INCLUDED */
