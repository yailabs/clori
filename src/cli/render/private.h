/* Renderers project typed reports without reclassifying their capability or failure state. */
#ifndef SRC_CLI_RENDER_PRIVATE_H_INCLUDED
#define SRC_CLI_RENDER_PRIVATE_H_INCLUDED

#include "src/cli/io/private.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/io.h>
#include <yvex/internal/media.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/model_artifact.h>
#include <yvex/internal/model_target.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/runtime_operator.h>
#include <yvex/internal/sampling.h>
#include <yvex/internal/transformer.h>
#include <yvex/internal/source_payload.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    YVEX_RENDER_MODE_PORCELAIN = 0,
    YVEX_RENDER_MODE_TABLE,
    YVEX_RENDER_MODE_AUDIT
} yvex_render_mode;
typedef struct {
    FILE *fp;
    yvex_render_mode mode;
} yvex_render_out;

typedef struct {
    const char *output_path;
    unsigned long long width, height, fps_numerator, fps_denominator;
    unsigned long long audio_channels, audio_sample_rate;
    yvex_media_avi_result publication;
} yvex_cli_media_report;

typedef struct {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    char identity[YVEX_SERVER_ID_CAP];
    char state[24];
    unsigned long long position;
    unsigned long long turns;
    int ready;
} yvex_cli_session_table_fact;
void yvex_cli_host_status_render(FILE *fp, const yvex_server_summary *status,
                                 int json);
void yvex_cli_host_memory_render(FILE *fp, const yvex_server_summary *status,
                                 int json);
void yvex_cli_engine_render(FILE *fp,
                            const yvex_server_engine_summary *engine,
                            int json);
int yvex_cli_session_table_render(FILE *fp,
                                  const yvex_cli_session_table_fact *facts,
                                  size_t count);
int yvex_cli_session_json_render(FILE *fp,
                                 const yvex_cli_session_table_fact *facts,
                                 size_t count, int list);
void yvex_cli_session_table_fact_set(yvex_cli_session_table_fact *fact,
                                     const yvex_client_message *message);

int yvex_media_publish_render(FILE *fp, yvex_graph_report_mode mode,
                              const yvex_cli_media_report *report);
int yvex_media_generate_render(FILE *, yvex_graph_report_mode, const char *,
                               const yvex_runtime_av_generation_result *);
/* Read one immutable collection counter selected by a renderer table. */
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

/* Backend rendering. */
int yvex_backend_render(FILE *fp, const yvex_backend_report *report);
int yvex_backend_render_help(FILE *fp);
int yvex_cuda_info_render_help(FILE *fp);

/* Graph rendering. */
int yvex_graph_attention_render(FILE *fp, yvex_graph_report_mode mode,
                                const yvex_graph_attention_operator_result *result);
int yvex_graph_moe_render(FILE *fp, yvex_graph_report_mode mode,
                          const yvex_moe_operator_result *result);
int yvex_graph_transformer_render(FILE *fp, yvex_graph_report_mode mode,
                                  const yvex_transformer_operator_result *result);
int yvex_graph_decode_render(FILE *fp, yvex_graph_report_mode mode,
                             const yvex_decode_operator_result *result);
int yvex_graph_logits_render(FILE *fp, yvex_graph_report_mode mode,
                             const yvex_logits_operator_result *result);
int yvex_graph_sampling_render(FILE *fp, yvex_graph_report_mode mode,
                               const yvex_sampling_operator_result *result);
int yvex_graph_generation_render(FILE *fp, yvex_graph_report_mode mode,
                                 const yvex_generation_operator_result *result);
int yvex_graph_render_help(FILE *fp);

/* Model-target rendering. */
int yvex_model_target_render(FILE *fp, yvex_model_target_render_mode mode,
                             const yvex_model_target_report *report);
int yvex_model_target_render_errors(FILE *fp, const yvex_model_target_report *report);
int yvex_model_target_render_help(FILE *fp);

/* Source rendering. */
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
