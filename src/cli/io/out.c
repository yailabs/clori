/*
 * Provide approved direct text output calls for operator normal/table/audit text.
 *
 * Direct stdio output stays in this file; callers provide target streams; wrappers preserve stdio
 * return behavior where legacy code checks it. Writer calls serialize existing facts only and do
 * not create capability.
 */
#include <build_commit.h>
#include "src/cli/io/private.h"
#include "src/cli/io/terminal/private.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <operator/registry.h>
#include <yvex/internal/core.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
enum { COMPLETION_CANDIDATE_CAP = 256, COMPLETION_TEXT_CAP = 128 };
typedef struct {
    char text[COMPLETION_CANDIDATE_CAP][COMPLETION_TEXT_CAP];
    size_t count;
} completion_candidates;
static int completion_visible(const yvex_operator_descriptor *descriptor)
{
    return descriptor->cli_projection &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_REMOVED &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_API_ONLY &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_TEST_ONLY;
}
static int completion_prefix_matches(const yvex_operator_descriptor *descriptor,
                                     size_t count, const char *const *words)
{
    size_t index;
    if (count > descriptor->command_word_count) return 0;
    for (index = 0u; index < count; ++index)
        if (strcmp(descriptor->command_words[index], words[index])) return 0;
    return 1;
}
static void completion_add(completion_candidates *candidates, const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    size_t index;
    if (!value[0] || strlen(value) >= COMPLETION_TEXT_CAP) return;
    while (*cursor) {
        if (!(isalnum(*cursor) || strchr("._:/@+-", *cursor))) return;
        cursor++;
    }
    for (index = 0u; index < candidates->count; ++index)
        if (!strcmp(candidates->text[index], value)) return;
    if (candidates->count >= COMPLETION_CANDIDATE_CAP) return;
    (void)snprintf(candidates->text[candidates->count], COMPLETION_TEXT_CAP, "%s", value);
    candidates->count++;
}
static void completion_add_metadata(completion_candidates *candidates, const char *values)
{
    const char *cursor = values;
    if (!strcmp(values, "none")) return;
    while (*cursor) {
        const char *end = strchr(cursor, '|');
        size_t extent = end ? (size_t)(end - cursor) : strlen(cursor);
        char item[COMPLETION_TEXT_CAP];
        if (extent < sizeof(item)) {
            memcpy(item, cursor, extent);
            item[extent] = '\0';
            completion_add(candidates, item);
        }
        if (!end) break;
        cursor = end + 1;
    }
}

static completion_candidates completion_collect(size_t prefix_count,
                                                const char *const *prefix)
{
    completion_candidates candidates = {{{0}}, 0u};
    size_t descriptor_index;
    for (descriptor_index = 0u; descriptor_index < yvex_operator_descriptor_count;
         ++descriptor_index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[descriptor_index];
        size_t index;
        if (!completion_visible(descriptor) ||
            (!prefix_count && descriptor->visibility !=
                                  YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT) ||
            !completion_prefix_matches(descriptor, prefix_count, prefix))
            continue;
        if (descriptor->command_word_count > prefix_count) {
            completion_add(&candidates, descriptor->command_words[prefix_count]);
            continue;
        }
        for (index = 0u; index < descriptor->flag_count; ++index) {
            completion_add(&candidates, descriptor->flags[index].name);
            completion_add_metadata(&candidates, descriptor->flags[index].aliases);
        }
        for (index = 0u; index < descriptor->argument_count; ++index)
            completion_add_metadata(&candidates, descriptor->arguments[index].enum_values);
    }
    return candidates;
}

static void completion_emit_case(FILE *output, const char *shell,
                                 size_t prefix_count, const char *const *prefix)
{
    completion_candidates candidates = completion_collect(prefix_count, prefix);
    size_t index;
    if (!candidates.count) return;
    if (!strcmp(shell, "fish")) fputs("    case '", output);
    else fputs("    '", output);
    for (index = 0u; index < prefix_count; ++index)
        fprintf(output, "%s%s", index ? " " : "", prefix[index]);
    if (!strcmp(shell, "fish")) fputs("'\n      set candidates", output);
    else fputs("') candidates='", output);
    for (index = 0u; index < candidates.count; ++index)
        fprintf(output, " %s", candidates.text[index]);
    if (!strcmp(shell, "fish")) fputc('\n', output);
    else fputs(" ' ;;\n", output);
}

static void completion_emit_cases(FILE *output, const char *shell)
{
    size_t descriptor_index, prefix_count, prior;
    for (descriptor_index = 0u; descriptor_index < yvex_operator_descriptor_count;
         ++descriptor_index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[descriptor_index];
        if (!completion_visible(descriptor)) continue;
        for (prefix_count = 0u; prefix_count <= descriptor->command_word_count;
             ++prefix_count) {
            int seen = 0;
            for (prior = 0u; prior < descriptor_index && !seen; ++prior) {
                const yvex_operator_descriptor *candidate =
                    &yvex_operator_descriptors[prior];
                if (completion_visible(candidate) &&
                    candidate->command_word_count >= prefix_count &&
                    completion_prefix_matches(candidate, prefix_count,
                                              descriptor->command_words))
                    seen = 1;
            }
            if (!seen)
                completion_emit_case(output, shell, prefix_count,
                                     descriptor->command_words);
        }
    }
}

int yvex_cli_completion_command(int argc, char **argv, size_t consumed)
{
    const char *shell = consumed + 1u < (size_t)argc ? argv[consumed + 1u] : NULL;
    if (!shell) return 2;
    if (!strcmp(shell, "bash")) {
        fputs("_yvex_complete() {\n"
              "  local cur=${COMP_WORDS[COMP_CWORD]} path='' candidates=''\n"
              "  if (( COMP_CWORD > 1 )); then "
              "path=${COMP_WORDS[*]:1:$((COMP_CWORD-1))}; fi\n"
              "  case \"$path\" in\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  esac\n  COMPREPLY=( $(compgen -W \"$candidates\" -- \"$cur\") )\n"
              "}\ncomplete -F _yvex_complete yvex\n", stdout);
        return 0;
    }
    if (!strcmp(shell, "zsh")) {
        fputs("#compdef yvex\n_yvex_complete() {\n"
              "  local path='' candidates=''\n"
              "  if (( CURRENT > 2 )); then path=${(j: :)words[2,$((CURRENT-1))]}; fi\n"
              "  case \"$path\" in\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  esac\n  compadd -- ${(z)candidates}\n}\ncompdef _yvex_complete yvex\n", stdout);
        return 0;
    }
    if (!strcmp(shell, "fish")) {
        fputs("function __yvex_candidates\n"
              "  set -l tokens (commandline -opc)\n"
              "  set -e tokens[1]\n"
              "  set -l path (string join ' ' $tokens)\n"
              "  set -l candidates\n"
              "  switch $path\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  end\n  printf '%s\\n' $candidates\nend\n"
              "complete -c yvex -f -a '(__yvex_candidates)'\n", stdout);
        return 0;
    }
    fprintf(stderr, "yvex: completion shell must be bash, zsh, or fish\n");
    return 2;
}

int yvex_cli_out_vwritef(FILE *fp, const char *fmt, va_list ap)
{
    return vfprintf(fp ? fp : stdout, fmt ? fmt : "", ap);
}

int yvex_cli_out_writef(FILE *fp, const char *fmt, ...)
{
    va_list ap;
    int rc;

    va_start(ap, fmt);
    rc = yvex_cli_out_vwritef(fp, fmt, ap);
    va_end(ap);
    return rc;
}

int yvex_cli_out_puts(FILE *fp, const char *text)
{
    return fputs(text ? text : "", fp ? fp : stdout);
}

int yvex_cli_out_fputs(const char *text, FILE *fp)
{
    return yvex_cli_out_puts(fp, text);
}

int yvex_cli_out_char(FILE *fp, int ch)
{
    return fputc(ch, fp ? fp : stdout);
}

int yvex_cli_out_flush(FILE *fp)
{
    FILE *stream = fp ? fp : stdout;

    return fflush(stream) == 0 && !ferror(stream) ? YVEX_OK : YVEX_ERR_IO;
}

FILE *yvex_cli_out_stdout(void)
{
    return stdout;
}

FILE *yvex_cli_out_stderr(void)
{
    return stderr;
}

void yvex_cli_terminal_style_get(FILE *fp, yvex_cli_terminal_style *style)
{
    FILE *stream = fp ? fp : stdout;
    const char *terminal;

    if (!style) return;
    memset(style, 0, sizeof(*style));
    style->reset = "";
    style->strong = "";
    style->accent = "";
    style->dim = "";
    style->success = "";
    style->warning = "";
    style->error = "";
    terminal = getenv("TERM");
    if (!yvex_cli_terminal_interactive(stream) || getenv("NO_COLOR") ||
        (terminal && !strcmp(terminal, "dumb")))
        return;
    style->reset = "\033[0m";
    style->strong = "\033[1;38;5;250m";
    style->accent = "\033[38;5;81m";
    style->dim = "\033[38;5;245m";
    style->success = "\033[38;5;114m";
    style->warning = "\033[38;5;179m";
    style->error = "\033[38;5;203m";
}



static int server_event_watch_visible(const yvex_server_event *event)
{
    return event && event->kind != YVEX_SERVER_EVENT_CLIENT_DISCONNECTED &&
           event->kind != YVEX_SERVER_EVENT_REQUEST_RECEIVED &&
           !(event->kind == YVEX_SERVER_EVENT_REQUEST_QUEUED && event->value_a <= 1u) &&
           !(event->kind >= YVEX_SERVER_EVENT_DRAFT_STARTED &&
             event->kind <= YVEX_SERVER_EVENT_CANDIDATE_REJECTED) &&
           event->kind != YVEX_SERVER_EVENT_GENERATION_FRAGMENT &&
           event->kind != YVEX_SERVER_EVENT_GENERATION_PROFILE;
}

static const char *server_event_category(yvex_server_event_kind kind)
{
    if (kind == YVEX_SERVER_EVENT_RUNTIME_READY) return "READY";
    if (kind >= YVEX_SERVER_EVENT_ENGINE_LOAD_REQUESTED) return "ENGINE";
    if (kind <= YVEX_SERVER_EVENT_LISTENER_READY) return "STARTUP";
    if (kind <= YVEX_SERVER_EVENT_SESSION_CLOSED) return "SESSION";
    if (kind <= YVEX_SERVER_EVENT_REQUEST_STARTED) return "REQUEST";
    if (kind <= YVEX_SERVER_EVENT_PREFILL_COMPLETED) return "PREFILL";
    if (kind <= YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED) return "SPECULATION";
    if (kind <= YVEX_SERVER_EVENT_GENERATION_FAILED) return "GENERATE";
    if (kind == YVEX_SERVER_EVENT_TELEMETRY_DROPPED) return "WARNING";
    return "RUNTIME";
}

static const char *server_event_color(const yvex_server_event *event,
                                      const yvex_cli_terminal_style *style)
{
    if (event->severity >= YVEX_SERVER_SEVERITY_ERROR) return style->error;
    if (event->severity == YVEX_SERVER_SEVERITY_WARNING) return style->warning;
    switch (event->kind) {
    case YVEX_SERVER_EVENT_RUNTIME_READY:
    case YVEX_SERVER_EVENT_LISTENER_READY:
    case YVEX_SERVER_EVENT_PREFILL_COMPLETED:
    case YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED:
    case YVEX_SERVER_EVENT_GENERATION_COMPLETED:
    case YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE:
    case YVEX_SERVER_EVENT_ENGINE_READY:
    case YVEX_SERVER_EVENT_ENGINE_UNLOADED:
        return style->success;
    case YVEX_SERVER_EVENT_GENERATION_CANCELLED:
    case YVEX_SERVER_EVENT_TELEMETRY_DROPPED:
        return style->warning;
    case YVEX_SERVER_EVENT_ENGINE_LOAD_FAILED:
    case YVEX_SERVER_EVENT_ENGINE_UNLOAD_FAILED:
        return style->error;
    case YVEX_SERVER_EVENT_PREFILL_STARTED:
    case YVEX_SERVER_EVENT_PREFILL_PROGRESS:
    case YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN:
        return style->accent;
    default:
        return style->strong;
    }
}

const char *yvex_cli_out_stop_reason(unsigned long long reason)
{
    static const char *const names[] = {
        "none", "EOS", "tokenizer stop", "maximum tokens", "context capacity",
        "cancelled", "model failure", "tokenizer failure", "output failure"};
    return reason < sizeof(names) / sizeof(names[0]) ? names[reason] : "unknown";
}

static void execution_rates(FILE *output,
                            const yvex_execution_measurement *measurement,
                            double fallback_rate, int compact)
{
    int typed = measurement &&
                measurement->schema_version ==
                    YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1;
    const char *scope = typed &&
                                measurement->scope ==
                                    YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE
                            ? "subsequent decode cumulative"
                        : typed && measurement->scope == YVEX_EXECUTION_SCOPE_PREFILL
                            ? "prefill"
                        : typed &&
                                  measurement->scope ==
                                      YVEX_EXECUTION_SCOPE_TOTAL_OPERATION
                            ? "total cumulative"
                            : "cumulative";
    if (compact) {
        double cumulative = 0.0;
        if (typed && (measurement->available &
                      YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE))
            cumulative = measurement->cumulative_rate;
        else if (!typed && fallback_rate > 0.0)
            cumulative = fallback_rate;
        const char *label = !typed ? "rate" :
            measurement->scope == YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE
                ? "decode-avg" :
            measurement->scope == YVEX_EXECUTION_SCOPE_TOTAL_OPERATION
                ? "total-avg" : "avg";
        if (cumulative > 0.0)
            fprintf(output, " | %s=%.2f tok/s", label, cumulative);
        if (typed && (measurement->available &
                      YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE)) {
            fprintf(output, "%s rolling[", cumulative > 0.0 ? "" : " |");
            if (measurement->rolling_units < measurement->rolling_window_units)
                fprintf(output, "%llu/", measurement->rolling_units);
            fprintf(output, "%llu]=%.2f tok/s", measurement->rolling_window_units,
                    measurement->rolling_rate);
        }
        return;
    }
    if (typed &&
        (measurement->available &
         YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE))
        fprintf(output, " · %s %.2f tok/s", scope,
                measurement->cumulative_rate);
    else if (fallback_rate > 0.0)
        fprintf(output, " · cumulative %.2f tok/s", fallback_rate);
    if (typed &&
        (measurement->available &
         YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE))
        fprintf(output, " · rolling%llu %.2f tok/s",
                measurement->rolling_window_units,
                measurement->rolling_rate);
}

void yvex_cli_out_turn_metrics(FILE *output, const yvex_client_message *message,
                               unsigned long long context_capacity,
                               const yvex_cli_terminal_style *style)
{
    if (message->engine_kind == YVEX_SERVER_ENGINE_MEDIA) {
        fprintf(output, "%smedia%s · %.2f s", style->success, style->reset,
                message->decode_seconds);
        return;
    }
    fprintf(output,
            "\n%s  generation %llu tokens · decode wall %.2f s",
            style->dim, message->generated_tokens, message->decode_seconds);
    execution_rates(output, &message->measurement, message->decode_rate, 0);
    fprintf(output,
            " · TTFT %.2f s%s\n"
            "%s  prefill %llu new/%llu prompt/%llu reused · %.2f s · %.2f tok/s",
            message->first_token_seconds, style->reset, style->dim,
            message->prefill_tokens, message->prompt_tokens,
            message->reused_tokens, message->prefill_seconds,
            message->prefill_rate);
    if (message->execution_strategy == YVEX_SERVER_EXECUTION_SPECULATIVE)
        fprintf(output, " · speculative %llu proposed/%llu accepted/%llu rejected/%llu verified",
                message->proposed_tokens,
                message->accepted_draft_tokens, message->rejected_draft_tokens,
                message->target_verification_count);
    if (context_capacity)
        fprintf(output, " · context %llu->%llu/%llu", message->initial_position,
                message->context_used, context_capacity);
    else
        fprintf(output, " · context %llu", message->context_used);
    if (message->output_limit_explicit)
        fprintf(output, " · output explicit %llu · envelope %llu",
                message->requested_maximum_new_tokens,
                message->resolved_maximum_new_tokens);
    else
        fprintf(output, " · output adaptive · envelope %llu",
                message->resolved_maximum_new_tokens);
    fputs(style->reset, output);
}

void yvex_cli_out_turn_complete(FILE *output,
                                const yvex_client_message *message,
                                unsigned long long context_capacity,
                                const yvex_cli_terminal_style *style)
{
    if (message->media_result.available) {
        fprintf(output,
                "%smedia complete%s · %s\n"
                "%llux%llu · %llu frames · %.3f s · %llu/%llu fps · "
                "%llu audio samples · %llu bytes · seed %llu · %llu evals\n"
                "%spreset %s · trajectory %s · execution %s%s\n",
                style->success, style->reset, message->media_result.output_path,
                message->media_result.width, message->media_result.height,
                message->media_result.frames,
                (double)message->media_result.duration_milliseconds / 1000.0,
                message->media_result.fps_numerator,
                message->media_result.fps_denominator,
                message->media_result.audio_samples,
                message->media_result.file_bytes, message->media_result.seed,
                message->media_result.model_evaluations, style->dim,
                message->media_result.preset_identity,
                message->media_result.trajectory_identity,
                message->media_result.execution_identity, style->reset);
    } else {
        yvex_cli_out_turn_metrics(output, message, context_capacity, style);
        fprintf(output, "%s · stop %s · session %s%s\n", style->dim,
                yvex_cli_out_stop_reason(message->stop_reason),
                message->session_name, style->reset);
    }
    if (message->reasoning_tokens || message->first_reasoning_seconds > 0.0)
        fprintf(output,
                "%s  reasoning %llu tokens · %.2f s · %.2f tok/s · "
                "TTFR %.2f s · final %llu tokens · %.2f s · %.2f tok/s · "
                "TTFF %.2f s · total %.2f tok/s%s\n",
                style->dim, message->reasoning_tokens,
                message->reasoning_seconds, message->reasoning_rate,
                message->first_reasoning_seconds, message->final_tokens,
                message->final_seconds, message->final_rate,
                message->first_final_seconds,
                message->total_completion_rate, style->reset);
}

static const char *server_backend_name(unsigned long long backend)
{
    return backend == YVEX_BACKEND_KIND_CUDA ? "CUDA" : "CPU";
}

static void server_event_name(const yvex_server_event *event)
{
    const char *name = yvex_server_event_kind_name(event->kind);
    while (*name) {
        int byte = *name++;
        putchar(byte == '.' || byte == '_' ? ' ' : byte);
    }
}

static void server_event_bytes(const char *name, unsigned long long bytes, int compact)
{
    double scale = bytes >= 1073741824u ? 1073741824.0 : 1048576.0;
    const char *unit = bytes >= 1073741824u ? "GiB" : "MiB";
    if (compact)
        printf(" %s=%.1f%s", name, (double)bytes / scale, unit);
    else
        printf(" · %s %.2f %s", name, (double)bytes / scale, unit);
}

static const char *execution_work_name(yvex_execution_work_unit unit)
{
    switch (unit) {
    case YVEX_EXECUTION_WORK_TOKENS: return "tokens";
    case YVEX_EXECUTION_WORK_BYTES: return "bytes";
    case YVEX_EXECUTION_WORK_TENSORS: return "tensors";
    case YVEX_EXECUTION_WORK_PLANS: return "plans";
    case YVEX_EXECUTION_WORK_COMPONENTS: return "components";
    case YVEX_EXECUTION_WORK_EVALUATIONS: return "evaluations";
    case YVEX_EXECUTION_WORK_FRAMES: return "frames";
    case YVEX_EXECUTION_WORK_SAMPLES: return "samples";
    case YVEX_EXECUTION_WORK_OPERATIONS: return "operations";
    case YVEX_EXECUTION_WORK_NONE: break;
    }
    return "work";
}

static void server_event_values(const yvex_server_event *event, int detailed)
{
    if (event->kind >= YVEX_SERVER_EVENT_DRAFT_STARTED &&
        event->kind <= YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED) {
        printf(" · cycle %llu", event->speculative_cycle);
        if (event->proposed_tokens)
            printf(" · accepted %llu/%llu", event->accepted_tokens,
                   event->proposed_tokens);
        if (event->selected_verification_tokens)
            printf(" · verified %llu", event->selected_verification_tokens);
        if (event->rejected_tokens) printf(" · rejected %llu", event->rejected_tokens);
        if (event->discarded_tokens) printf(" · discarded %llu", event->discarded_tokens);
        if (detailed && event->confidence_logit_count)
            printf(" · confidence %.4g..%.4g mean %.4g",
                   event->confidence_logit_minimum, event->confidence_logit_maximum,
                   event->confidence_logit_mean);
        return;
    }
    switch (event->kind) {
    case YVEX_SERVER_EVENT_MATERIALIZATION_COMPLETE:
        server_event_bytes("host", event->value_a, 0);
        server_event_bytes("device", event->value_b, 0);
        printf(" · %llu tensor binding%s", event->value_c,
               event->value_c == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_RESIDENCY_READY:
        server_event_bytes("host", event->value_a, 0);
        server_event_bytes("device", event->value_b, 0);
        printf(" · %llu upload%s", event->value_c, event->value_c == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_ARTIFACT_OPEN_COMPLETE:
        if (event->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
            strcmp(event->phase, "media-model")) {
            static const char *const modes[] = {
                "full-hash", "verified-reopen", "fallback-full-hash",
            };
            static const char *const states[] = {
                "disabled", "miss", "hit", "invalid",
            };
            unsigned long long mode = event->value_c & 0xffull;
            unsigned long long state = (event->value_c >> 8u) & 0xffull;
            server_event_bytes("hashed", event->value_a, 0);
            server_event_bytes("file", event->value_b, 0);
            printf(" · %s · lease %s",
                   mode < sizeof(modes) / sizeof(modes[0]) ? modes[mode] : "unknown",
                   state < sizeof(states) / sizeof(states[0]) ? states[state] : "unknown");
            if ((event->value_c >> 16u) & 1ull) printf(" · published");
            if ((event->value_c >> 17u) & 1ull) printf(" · repaired");
            if ((event->value_c >> 18u) & 1ull) printf(" · cache warning");
        } else if (event->engine_kind == YVEX_SERVER_ENGINE_MEDIA) {
            server_event_bytes("hashed", event->value_a, 0);
            server_event_bytes("files", event->value_b, 0);
            printf(" · %llu components", event->value_c);
        } else {
            server_event_bytes("hashed", event->value_a, 0);
            server_event_bytes("host", event->value_b, 0);
            server_event_bytes("device", event->value_c, 0);
        }
        break;
    case YVEX_SERVER_EVENT_LISTENER_READY:
        printf(" · socket %04llo · queue %llu · sessions %llu", event->value_a,
               event->value_b, event->value_c);
        break;
    case YVEX_SERVER_EVENT_RUNTIME_READY:
        printf(" · %s · context %llu", server_backend_name(event->value_c), event->value_b);
        break;
    case YVEX_SERVER_EVENT_SESSION_CREATED:
    case YVEX_SERVER_EVENT_SESSION_CLOSED:
        printf(" · %llu active", event->value_b);
        break;
    case YVEX_SERVER_EVENT_SESSION_ATTACHED:
    case YVEX_SERVER_EVENT_SESSION_DETACHED:
        printf(" · %llu client%s", event->value_a, event->value_a == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_REQUEST_RECEIVED:
        if (detailed) printf(" · prompt %llu bytes", event->value_a);
        break;
    case YVEX_SERVER_EVENT_REQUEST_QUEUED:
        printf(" · depth %llu/%llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_REQUEST_STARTED:
        printf(" · input %llu · prefix %llu · limit %llu", event->value_a,
               event->value_b, event->value_c);
        break;
    case YVEX_SERVER_EVENT_TOKENIZER_COMPLETED:
        printf(" · prompt %llu · reused %llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_PREFILL_STARTED:
        printf(" · %llu new · chunk %llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_PREFILL_PROGRESS:
        printf(" · %llu/%llu tokens", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_PREFILL_COMPLETED:
        printf(" · %llu tokens · %llu chunk%s", event->value_a, event->value_b,
               event->value_b == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN:
        if (detailed) printf(" · ordinal %llu · token %llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_GENERATION_COMPLETED:
    case YVEX_SERVER_EVENT_GENERATION_CANCELLED:
    case YVEX_SERVER_EVENT_GENERATION_FAILED:
        printf(" · %llu token%s · position %llu · stop %s", event->value_a,
               event->value_a == 1u ? "" : "s", event->value_b,
               yvex_cli_out_stop_reason(event->value_c));
        break;
    case YVEX_SERVER_EVENT_GENERATION_PROFILE:
        if (!strcmp(event->phase, "movement"))
            printf(" · H2D %llu · D2H %llu · D2D %llu bytes", event->value_a,
                   event->value_b, event->value_c);
        else if (!strcmp(event->phase, "transfers"))
            printf(" · uploads %llu · downloads %llu · expert subviews %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "launches"))
            printf(" · kernel launches %llu · stream syncs %llu · device syncs %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "target"))
            printf(" · target forwards %llu · rows %llu · replayed %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "speculation"))
            printf(" · draft forwards %llu · verified rows %llu · promoted rows %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "candidate"))
            printf(" · accepted %llu · discarded %llu · extensions %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "moe"))
            printf(" · row/expert pairs %llu · subviews %llu · bytes %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "output"))
            printf(" · rows %llu · D2H %llu bytes · committed %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "prefill"))
            printf(" · prompt %llu · reused %llu · new %llu", event->value_a,
                   event->value_b, event->value_c);
        else if (!strcmp(event->phase, "decode"))
            printf(" · first decode %llu · later decode %llu · tokens %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batches"))
            printf(" · physical %llu · multi-source %llu · max width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-rows"))
            printf(" · submitted %llu · executed %llu · admitted width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-multi-source"))
            printf(" · physical %llu · rows %llu · max real width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-sources"))
            printf(" · max sources %llu · active producers %llu",
                   event->value_a, event->value_b);
        else if (!strcmp(event->phase, "execution-batch-experts"))
            printf(" · worklists %llu · pairs %llu · max population %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-expert-rows"))
            printf(" · TC eligible %llu · TC executed %llu · narrow %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "batch-expert-population-1-3"))
            printf(" · population 1 %llu · 2 %llu · 3 %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "batch-expert-population-4-6"))
            printf(" · population 4 %llu · 5 %llu · 6 %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-coalescing"))
            printf(" · waits %llu · timeouts %llu · producers %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-policy"))
            printf(" · coalescing limit %llu ns · width %llu · producers %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-step-rendezvous"))
            printf(" · submissions %llu · multi-source %llu · max width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-step-policy"))
            printf(" · limit %llu ns · steps %llu · producers %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-mismatch"))
            printf(" · phase %llu · layer %llu · operation %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-mismatch-other"))
            printf(" · geometry %llu · profile %llu · identity %llu",
                   event->value_a, event->value_b, event->value_c);
        break;
    case YVEX_SERVER_EVENT_ENGINE_LOAD_PROGRESS:
        if (event->measurement.available &
            YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE)
            printf(" · %llu/%llu %s", event->measurement.completed_units,
                   event->measurement.total_units,
                   execution_work_name(event->measurement.work_unit));
        else
            printf(" · %llu %s completed · total unavailable",
                   event->measurement.completed_units,
                   execution_work_name(event->measurement.work_unit));
        break;
    case YVEX_SERVER_EVENT_TELEMETRY_DROPPED:
        printf(" · %llu dropped · capacity %llu", event->value_a, event->value_b);
        break;
    default:
        break;
    }
}

int yvex_cli_out_server_event(const yvex_server_event *event, int detailed)
{
    static const char *const severity[] = {"debug", "info", "warning", "error", "fatal"};
    yvex_cli_terminal_style style;
    const char *color;
    unsigned int level;
    if (!event || (!detailed && !server_event_watch_visible(event))) return 0;
    yvex_cli_terminal_style_get(stdout, &style);
    color = server_event_color(event, &style);
    level = event->severity <= YVEX_SERVER_SEVERITY_FATAL
                ? (unsigned int)event->severity : YVEX_SERVER_SEVERITY_FATAL;
    if (detailed) {
        const char *severity_color = event->severity >= YVEX_SERVER_SEVERITY_ERROR
                                         ? style.error
                                         : event->severity == YVEX_SERVER_SEVERITY_WARNING
                                               ? style.warning
                                               : style.dim;
        printf("%s#%llu%s %s%-7s%s ", style.dim, event->sequence, style.reset,
               severity_color, severity[level], style.reset);
    } else {
        time_t seconds = (time_t)(event->wall_time_ns / 1000000000u);
        struct tm clock;
        char stamp[16] = "--:--:--";
        if (event->wall_time_ns && localtime_r(&seconds, &clock))
            (void)strftime(stamp, sizeof(stamp), "%H:%M:%S", &clock);
        printf("%s%s%s  %s%-8s%s ", style.dim, stamp, style.reset, color,
               server_event_category(event->kind), style.reset);
    }
    fputs(color, stdout);
    server_event_name(event);
    fputs(style.reset, stdout);
    if (!detailed && event->session_id[0] && event->request_id[0])
        printf(" · %s/%s", event->session_id, event->request_id);
    else {
        if (event->session_id[0]) printf(" · session %s", event->session_id);
        if (event->request_id[0]) printf(" · request %s", event->request_id);
    }
    if (detailed && event->turn_id[0]) printf(" · turn %s", event->turn_id);
    if (detailed && event->phase[0]) printf(" · phase %s", event->phase);
    server_event_values(event, detailed);
    if (event->seconds > 0.0) printf(" · %.3f s", event->seconds);
    if (event->kind == YVEX_SERVER_EVENT_GENERATION_PROGRESS ||
        (event->kind >= YVEX_SERVER_EVENT_GENERATION_COMPLETED &&
         event->kind <= YVEX_SERVER_EVENT_GENERATION_FAILED))
        execution_rates(stdout, &event->measurement, event->rate, 0);
    else if (event->rate > 0.0)
        printf(" · %.2f %s/s", event->rate,
               execution_work_name(event->measurement.work_unit));
    putchar('\n');
    fflush(stdout);
    return 1;
}

static void watch_stamp(const yvex_server_event *event, char stamp[16])
{
    time_t seconds = (time_t)(event->wall_time_ns / 1000000000u);
    struct tm clock;
    memcpy(stamp, "--:--:--", 9u);
    if (event->wall_time_ns && localtime_r(&seconds, &clock))
        (void)strftime(stamp, 16u, "%H:%M:%S", &clock);
}

static void watch_line_begin(const yvex_cli_watch_renderer *renderer,
                             const yvex_server_event *event,
                             const char *color, const char *label)
{
    char stamp[16];
    watch_stamp(event, stamp);
    printf("%s%s%s  %s%-9s%s ", renderer->style.dim, stamp,
           renderer->style.reset, color, label, renderer->style.reset);
}

static void watch_text(const char *value, size_t maximum)
{
    size_t length = strlen(value);
    if (length <= maximum) fputs(value, stdout);
    else printf("%.*s..%s", (int)(maximum - 6u), value, value + length - 4u);
}
static void watch_request_id(const yvex_server_event *event)
{
    if (event->session_id[0] && event->request_id[0]) {
        watch_text(event->session_id, 12u);
        printf("/%s", event->request_id);
    }
    else if (event->session_id[0])
        watch_text(event->session_id, 12u);
    else if (event->request_id[0])
        fputs(event->request_id, stdout);
    else
        fputs("runtime", stdout);
}

static void watch_request_begin(yvex_cli_watch_renderer *renderer,
                                const yvex_server_event *event)
{
    renderer->request_open = 1;
    watch_line_begin(renderer, event, renderer->style.accent, "REQUEST");
    watch_request_id(event);
    printf(" %s%s=%llu prefix_tokens=%llu max_tokens=%llu%s",
           renderer->style.dim,
           event->provider_request_identity[0] ? "messages" : "input_bytes",
           event->value_a, event->value_b,
           event->value_c, renderer->style.reset);
    putchar('\n');
}
static void watch_session(const yvex_cli_watch_renderer *renderer,
                          const yvex_server_event *event)
{
    const char *verb = event->kind == YVEX_SERVER_EVENT_SESSION_CREATED ? "created" :
        event->kind == YVEX_SERVER_EVENT_SESSION_ATTACHED ? "attached" :
        event->kind == YVEX_SERVER_EVENT_SESSION_DETACHED ? "detached" :
        event->kind == YVEX_SERVER_EVENT_SESSION_RESET ? "reset" : "closed";
    watch_line_begin(renderer, event, server_event_color(event, &renderer->style),
                     "SESSION");
    watch_text(event->session_id, 16u);
    printf(" %s", verb);
    if (event->kind == YVEX_SERVER_EVENT_SESSION_ATTACHED ||
        event->kind == YVEX_SERVER_EVENT_SESSION_DETACHED)
        printf(" clients=%llu", event->value_a);
    else if (event->kind == YVEX_SERVER_EVENT_SESSION_CREATED ||
             event->kind == YVEX_SERVER_EVENT_SESSION_CLOSED)
        printf(" active_sessions=%llu", event->value_b);
    putchar('\n');
}
static void watch_cycle(yvex_cli_watch_renderer *renderer,
                        const yvex_server_event *event)
{
    if (!renderer->detailed) return;
    watch_line_begin(renderer, event, renderer->style.success, "SPEC");
    watch_request_id(event);
    printf(" cycle=%llu accepted=%llu/%llu", event->speculative_cycle,
           event->accepted_tokens, event->proposed_tokens);
    if (event->selected_verification_tokens)
        printf(" verified=%llu", event->selected_verification_tokens);
    if (event->rejected_tokens) printf(" rejected=%llu", event->rejected_tokens);
    if (event->discarded_tokens) printf(" discarded=%llu", event->discarded_tokens);
    if (event->seconds > 0.0) printf(" elapsed=%.3fs", event->seconds);
    putchar('\n');
}

static void watch_request_end(yvex_cli_watch_renderer *renderer,
                              const yvex_server_event *event)
{
    const char *label = event->kind == YVEX_SERVER_EVENT_GENERATION_COMPLETED
                            ? "DONE"
                        : event->kind == YVEX_SERVER_EVENT_GENERATION_CANCELLED
                            ? "CANCELLED"
                            : "FAIL";
    const char *color = server_event_color(event, &renderer->style);
    watch_line_begin(renderer, event, color, label);
    watch_request_id(event);
    printf(" generated=%llu position=%llu stop=\"%s\"", event->value_a,
           event->value_b, yvex_cli_out_stop_reason(event->value_c));
    if (event->seconds > 0.0) printf(" elapsed=%.1fs", event->seconds);
    execution_rates(stdout, &event->measurement, event->rate, 1);
    if (event->proposed_tokens)
        printf(" spec-accept=%.1f%%", 100.0 * (double)event->accepted_tokens /
                              (double)event->proposed_tokens);
    putchar('\n');
    renderer->request_open = 0;
}

static int watch_progress_due(yvex_cli_watch_renderer *renderer,
                              const yvex_server_event *event)
{
    int reasoning = !strcmp(event->phase, "reasoning");
    int same = !strcmp(renderer->session_id, event->session_id) &&
               !strcmp(renderer->request_id, event->request_id);
    if (same && reasoning == renderer->progress_reasoning &&
        event->value_a < renderer->progress_tokens + 16ull &&
        event->seconds < renderer->progress_seconds + 1.0)
        return 0;
    yvex_core_text_copy(renderer->session_id, sizeof(renderer->session_id),
                        event->session_id);
    yvex_core_text_copy(renderer->request_id, sizeof(renderer->request_id),
                        event->request_id);
    renderer->progress_tokens = event->value_a;
    renderer->progress_seconds = event->seconds;
    renderer->progress_reasoning = reasoning;
    return 1;
}

static void watch_live_resources(yvex_cli_watch_renderer *renderer,
                                 const yvex_server_event *event,
                                 const yvex_server_summary *live, int force)
{
    const yvex_execution_resource_summary *resource;
    if (!live) return;
    if (!force && renderer->resource_stamp_ns &&
        event->wall_time_ns >= renderer->resource_stamp_ns &&
        event->wall_time_ns - renderer->resource_stamp_ns < 10000000000ull) return;
    renderer->resource_stamp_ns = event->wall_time_ns;
    resource = &live->metrics.resources;
    watch_line_begin(renderer, event, renderer->style.dim, "RESOURCES");
    fputs("host snapshot", stdout);
    if (resource->available & YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE)
        server_event_bytes("process-rss", resource->process_rss_current_bytes, 1);
    if (resource->model_explicit_device_bytes)
        server_event_bytes("model-device-alloc", resource->model_explicit_device_bytes, 1);
    if (resource->available & YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE)
        server_event_bytes("workspace", resource->workspace_current_bytes, 1);
    if (resource->available & YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE)
        server_event_bytes("session-state", resource->session_physical_state_bytes, 1);
    printf(" active_requests=%llu queued=%llu\n", live->metrics.active_requests,
           live->metrics.queue_depth);
}

static int watch_generation_progress(yvex_cli_watch_renderer *renderer,
                                     const yvex_server_event *event,
                                     const yvex_server_summary *live)
{
    if (!watch_progress_due(renderer, event)) return 0;
    watch_line_begin(renderer, event, renderer->style.accent, "DECODE");
    watch_request_id(event);
    printf(" generated=%llu position=%llu phase=%s", event->value_a, event->value_b,
           event->phase[0] ? event->phase : "decode");
    if (event->value_c && strcmp(event->phase, "reasoning"))
        printf(" reasoning=%llu", event->value_c);
    if (event->seconds > 0.0) printf(" elapsed=%.1fs", event->seconds);
    execution_rates(stdout, &event->measurement, event->rate, 1);
    if (event->proposed_tokens)
        printf(" spec-accept=%.1f%%", 100.0 * (double)event->accepted_tokens /
                              (double)event->proposed_tokens);
    putchar('\n');
    watch_live_resources(renderer, event, live, 0);
    return 1;
}

void yvex_cli_watch_renderer_open(yvex_cli_watch_renderer *renderer, int detailed)
{
    if (!renderer) return;
    memset(renderer, 0, sizeof(*renderer));
    renderer->detailed = detailed != 0;
    yvex_cli_terminal_style_get(stdout, &renderer->style);
}

static int watch_operation_progress(yvex_cli_watch_renderer *renderer,
                                     const yvex_server_event *event, const char *label)
{
    const yvex_execution_measurement *measurement = &event->measurement;
    const char *phase = event->phase[0] ? event->phase : "work";
    const char *unit = execution_work_name(measurement->work_unit);
    watch_line_begin(renderer, event, renderer->style.accent, label);
    if (event->request_id[0]) {
        watch_request_id(event);
        putchar(' ');
    }
    printf("phase=%s", phase);
    if ((measurement->available & YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE) &&
        measurement->total_units) {
        double percent = 100.0 * (double)measurement->completed_units /
                         (double)measurement->total_units;
        if (measurement->work_unit == YVEX_EXECUTION_WORK_BYTES) {
            double scale = measurement->total_units >= 1073741824ull
                               ? 1073741824.0
                           : measurement->total_units >= 1048576ull
                               ? 1048576.0
                           : measurement->total_units >= 1024ull ? 1024.0 : 1.0;
            const char *suffix = scale == 1073741824.0 ? "GiB" :
                                 scale == 1048576.0 ? "MiB" :
                                 scale == 1024.0 ? "KiB" : "B";
            printf(" completed=%.1f/%.1f%s", (double)measurement->completed_units / scale,
                   (double)measurement->total_units / scale, suffix);
        } else {
            printf(" completed=%llu/%llu %s", measurement->completed_units,
                   measurement->total_units, unit);
        }
        printf(" (%.0f%%)", percent);
    } else {
        printf(" completed=%llu %s total=unknown", measurement->completed_units, unit);
    }
    if (measurement->available & YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE)
        printf(" elapsed=%.2fs", (double)measurement->duration_ns / 1000000000.0);
    putchar('\n');
    return 1;
}

int yvex_cli_watch_renderer_event(yvex_cli_watch_renderer *renderer,
                                  const yvex_server_event *event,
                                  const yvex_server_summary *live)
{
    if (!renderer || !event) return 0;
    if (event->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
        event->kind == YVEX_SERVER_EVENT_ARTIFACT_OPEN_COMPLETE &&
        strcmp(event->phase, "media-model")) {
        static const char *const modes[] = {"hash", "reopen", "rehash"};
        unsigned long long mode = event->value_c & 0xffull;
        watch_line_begin(renderer, event, renderer->style.accent, "COMPONENT");
        printf("%-12s %s", event->phase,
               mode < sizeof(modes) / sizeof(modes[0]) ? modes[mode] : "unknown");
        server_event_bytes("hashed", event->value_a, 1);
        server_event_bytes("file", event->value_b, 1);
        if (event->seconds > 0.0) printf(" elapsed=%.2fs", event->seconds);
        putchar('\n');
        return 1;
    }
    if (event->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
        event->kind == YVEX_SERVER_EVENT_REQUEST_STARTED) {
        watch_line_begin(renderer, event, renderer->style.accent, "MEDIA");
        watch_request_id(event);
        puts(" started");
        renderer->request_open = 1;
        return 1;
    }
    if (event->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
        (event->kind == YVEX_SERVER_EVENT_PREFILL_STARTED ||
         event->kind == YVEX_SERVER_EVENT_PREFILL_COMPLETED ||
         event->kind == YVEX_SERVER_EVENT_GENERATION_PROGRESS ||
         event->kind == YVEX_SERVER_EVENT_GENERATION_PROFILE)) {
        watch_line_begin(renderer, event, renderer->style.accent, "MEDIA");
        watch_request_id(event);
        printf(" phase=%s", event->phase[0] ? event->phase : "executing");
        if (event->value_b)
            printf(" completed=%llu/%llu %s", event->value_a, event->value_b,
                   execution_work_name(event->measurement.work_unit));
        if (event->value_c) printf(" value=%llu", event->value_c);
        putchar('\n');
        watch_live_resources(renderer, event, live, 0);
        return 1;
    }
    if (event->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
        event->kind >= YVEX_SERVER_EVENT_GENERATION_COMPLETED &&
        event->kind <= YVEX_SERVER_EVENT_GENERATION_FAILED) {
        const char *label = event->kind == YVEX_SERVER_EVENT_GENERATION_COMPLETED
                                ? "COMPLETE"
                            : event->kind == YVEX_SERVER_EVENT_GENERATION_CANCELLED
                                ? "CANCELLED" : "FAILED";
        watch_line_begin(renderer, event, server_event_color(event, &renderer->style),
                         !strcmp(label, "COMPLETE") ? "DONE" :
                         !strcmp(label, "CANCELLED") ? "CANCELLED" : "FAIL");
        watch_request_id(event);
        if (event->kind == YVEX_SERVER_EVENT_GENERATION_COMPLETED)
            printf(" frames=%llu bytes=%llu audio_samples=%llu",
                   event->value_a, event->value_b, event->value_c);
        else
            printf(" media phase=%s", event->phase[0] ? event->phase : "end");
        if (event->seconds > 0.0) printf(" elapsed=%.2fs", event->seconds);
        putchar('\n');
        watch_live_resources(renderer, event, live, 1);
        renderer->request_open = 0;
        return 1;
    }
    if (event->kind == YVEX_SERVER_EVENT_PROCESS_START ||
        event->kind == YVEX_SERVER_EVENT_TELEMETRY_READY ||
        event->kind == YVEX_SERVER_EVENT_LISTENER_READY ||
        (event->kind == YVEX_SERVER_EVENT_RUNTIME_READY &&
         event->engine_kind == YVEX_SERVER_ENGINE_NONE) ||
        event->kind == YVEX_SERVER_EVENT_CLIENT_DISCONNECTED ||
        event->kind == YVEX_SERVER_EVENT_REQUEST_RECEIVED ||
        event->kind == YVEX_SERVER_EVENT_GENERATION_FRAGMENT ||
        event->kind == YVEX_SERVER_EVENT_GENERATION_PROFILE ||
        (event->kind >= YVEX_SERVER_EVENT_DRAFT_STARTED &&
         event->kind <= YVEX_SERVER_EVENT_CANDIDATE_REJECTED))
        return 0;
    if (event->kind == YVEX_SERVER_EVENT_ENGINE_LOAD_PROGRESS)
        return watch_operation_progress(renderer, event, "LOAD");
    if (event->kind == YVEX_SERVER_EVENT_PREFILL_STARTED ||
        event->kind == YVEX_SERVER_EVENT_PREFILL_PROGRESS)
        return watch_operation_progress(renderer, event, "PREFILL");
    if (event->kind == YVEX_SERVER_EVENT_RUNTIME_READY ||
        event->kind >= YVEX_SERVER_EVENT_ENGINE_LOAD_REQUESTED) {
        const char *label = event->kind == YVEX_SERVER_EVENT_RUNTIME_READY ||
                            event->kind == YVEX_SERVER_EVENT_ENGINE_READY
                                ? "MODEL"
                            : event->kind == YVEX_SERVER_EVENT_ENGINE_LOAD_REQUESTED
                                ? "LOAD"
                            : event->kind == YVEX_SERVER_EVENT_ENGINE_LOAD_FAILED
                                ? "FAIL"
                            : event->kind == YVEX_SERVER_EVENT_ENGINE_UNLOAD_STARTED
                                ? "UNLOAD"
                            : event->kind == YVEX_SERVER_EVENT_ENGINE_UNLOADED
                                ? "UNLOADED" : "FAIL";
        watch_line_begin(renderer, event,
                         server_event_color(event, &renderer->style), label);
        watch_text(event->phase, sizeof(event->phase));
        printf(" generation=%llu backend=%s", event->value_a, server_backend_name(event->value_c));
        if (event->execution_strategy == YVEX_SERVER_EXECUTION_SPECULATIVE)
            printf(" strategy=speculative");
        else if (event->execution_strategy == YVEX_SERVER_EXECUTION_TARGET_ONLY)
            printf(" strategy=target-only");
        putchar('\n');
        return 1;
    }
    if (event->kind >= YVEX_SERVER_EVENT_ARTIFACT_OPEN_START &&
        event->kind <= YVEX_SERVER_EVENT_RESIDENCY_READY) return 0;
    if (event->kind >= YVEX_SERVER_EVENT_SESSION_CREATED &&
        event->kind <= YVEX_SERVER_EVENT_SESSION_CLOSED) {
        watch_session(renderer, event);
        return 1;
    }
    if (event->kind == YVEX_SERVER_EVENT_REQUEST_QUEUED) {
        if (event->value_a <= 1ull) return 0;
        watch_line_begin(renderer, event, renderer->style.warning, "QUEUE");
        watch_request_id(event);
        printf(" depth=%llu/%llu\n", event->value_a, event->value_b);
        return 1;
    }
    if (event->kind == YVEX_SERVER_EVENT_REQUEST_STARTED) {
        watch_request_begin(renderer, event);
        return 1;
    }
    if (event->kind == YVEX_SERVER_EVENT_TOKENIZER_COMPLETED) {
        watch_line_begin(renderer, event, renderer->style.strong, "PROMPT");
        watch_request_id(event);
        printf(" tokens=%llu reused=%llu\n", event->value_a, event->value_b);
    } else if (event->kind == YVEX_SERVER_EVENT_PREFILL_COMPLETED) {
        watch_line_begin(renderer, event, renderer->style.success, "PREFILL");
        watch_request_id(event);
        printf(" tokens=%llu chunks=%llu", event->value_a, event->value_b);
        if (event->seconds > 0.0) printf(" elapsed=%.2fs", event->seconds);
        execution_rates(stdout, &event->measurement, event->rate, 1);
        putchar('\n');
    } else if (event->kind == YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN) {
        watch_line_begin(renderer, event, renderer->style.accent, "FIRST");
        watch_request_id(event);
        if (event->seconds > 0.0) printf(" first-token=%.3fs", event->seconds);
        putchar('\n');
    } else if (event->kind == YVEX_SERVER_EVENT_GENERATION_PROGRESS) {
        if (!watch_generation_progress(renderer, event, live)) return 0;
    } else if (event->kind == YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED) {
        watch_cycle(renderer, event);
    } else if (event->kind >= YVEX_SERVER_EVENT_GENERATION_COMPLETED &&
               event->kind <= YVEX_SERVER_EVENT_GENERATION_FAILED) {
        watch_request_end(renderer, event);
        watch_live_resources(renderer, event, live, 1);
    } else if (event->kind == YVEX_SERVER_EVENT_TELEMETRY_DROPPED) {
        watch_line_begin(renderer, event, renderer->style.warning, "WARN");
        printf("telemetry coalesced=%llu dropped=%llu capacity=%llu\n",
               event->value_c, event->value_a, event->value_b);
    } else if (event->kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_START ||
               event->kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE) {
        watch_line_begin(renderer, event, server_event_color(event, &renderer->style),
                         "HOST");
        fputs(event->kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_START
                  ? "stopping" : "stopped", stdout);
        putchar('\n');
    } else {
        return 0;
    }
    fflush(stdout);
    return 1;
}

void yvex_cli_watch_renderer_finish(yvex_cli_watch_renderer *renderer)
{
    if (renderer) renderer->request_open = 0;
}

void yvex_cli_out_repl_catalog(void)
{
    yvex_cli_terminal_style style;
    size_t index, pass;
    yvex_cli_terminal_style_get(stdout, &style);
    printf("%scommands%s", style.strong, style.reset);
    for (pass = 0u; pass < 3u; ++pass) {
        for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
            const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
            int priority = descriptor->runtime_adapter == YVEX_OPERATOR_RUNTIME_HELP ? 0
                         : descriptor->repl_adapter == YVEX_OPERATOR_REPL_QUIT ? 2 : 1;
            if (!strcmp(descriptor->slash_projection, "none") || priority != (int)pass) continue;
            printf("\n  %s%-12s%s %s%s%s", style.accent,
                   descriptor->slash_projection, style.reset, style.dim,
                   descriptor->summary, style.reset);
        }
    }
    printf("\n\n  %s%-12s%s %scancel an active turn or clear input; press again to exit%s",
           style.warning, "Ctrl-C", style.reset, style.dim, style.reset);
    printf("\n  %s%-12s%s %sexit on empty input; otherwise delete at cursor%s", style.accent,
           "Ctrl-D", style.reset, style.dim, style.reset);
    printf("\n  %s%-12s%s %sclear and redraw input%s", style.accent, "Ctrl-L",
           style.reset, style.dim, style.reset);
    puts("\n");
}
void yvex_cli_out_line(FILE *fp, const char *text)
{
    (void)yvex_cli_out_puts(fp, text);
    (void)yvex_cli_out_char(fp, '\n');
}
void yvex_cli_out_lines(FILE *fp,
                        const char *const *lines,
                        size_t line_count)
{
    size_t i;
    if (!lines) {
        return;
    }
    for (i = 0; i < line_count; ++i) {
        yvex_cli_out_line(fp, lines[i]);
    }
}
void yvex_cli_out_kv_str(FILE *fp, const char *key, const char *value)
{
    (void)yvex_cli_out_writef(fp, "%s: %s\n", key ? key : "", value ? value : "");
}
void yvex_cli_out_kv_bool(FILE *fp, const char *key, int value)
{
    yvex_cli_out_kv_str(fp, key, value ? "true" : "false");
}
int yvex_cli_out_fields(FILE *fp,
                        const void *object,
                        const yvex_cli_field_spec *fields,
                        size_t field_count)
{
    const unsigned char *base = object;
    size_t i;
    if (!object || (!fields && field_count != 0u)) {
        return -1;
    }
    for (i = 0; i < field_count; ++i) {
        const yvex_cli_field_spec *field = &fields[i];
        const void *value = base + field->offset;
        const char *text;
        int rc;
        switch (field->kind) {
        case YVEX_CLI_FIELD_TEXT:
            text = *(const char *const *)value;
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     text && text[0]
                                         ? text
                                         : (field->fallback ? field->fallback : "unknown"));
            break;
        case YVEX_CLI_FIELD_TEXT_ARRAY:
            text = value;
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     text[0]
                                         ? text
                                         : (field->fallback ? field->fallback : "unknown"));
            break;
        case YVEX_CLI_FIELD_U64:
            rc = yvex_cli_out_writef(fp, "%s: %llu\n", field->key,
                                     *(const unsigned long long *)value);
            break;
        case YVEX_CLI_FIELD_U32:
            rc = yvex_cli_out_writef(fp, "%s: %u\n", field->key,
                                     *(const unsigned int *)value);
            break;
        case YVEX_CLI_FIELD_I32:
            rc = yvex_cli_out_writef(fp, "%s: %d\n", field->key, *(const int *)value);
            break;
        case YVEX_CLI_FIELD_BOOL:
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     *(const int *)value ? "true" : "false");
            break;
        case YVEX_CLI_FIELD_DOUBLE:
            rc = yvex_cli_out_writef(fp, "%s: %.17g\n", field->key,
                                     *(const double *)value);
            break;
        case YVEX_CLI_FIELD_FLOAT9:
            rc = yvex_cli_out_writef(fp, "%s: %.9g\n", field->key,
                                     *(const double *)value);
            break;
        case YVEX_CLI_FIELD_HEX64:
            rc = yvex_cli_out_writef(fp, "%s: %016llx\n", field->key,
                                     *(const unsigned long long *)value);
            break;
        default:
            return -1;
        }
        if (rc < 0) {
            return -1;
        }
    }
    return 0;
}
int print_yvex_error(const yvex_error *err, int exit_code)
{
    yvex_cli_out_writef(stderr, "yvex: %s: %s\n", yvex_error_where(err),
                        yvex_error_message(err));
    return exit_code;
}
int exit_for_status(int status)
{
    switch (status) {
    case YVEX_ERR_INVALID_ARG:
        return 2;
    case YVEX_ERR_IO:
        return 3;
    case YVEX_ERR_FORMAT:
    case YVEX_ERR_BOUNDS:
        return 4;
    case YVEX_ERR_UNSUPPORTED:
        return 5;
    default:
        return 1;
    }
}
/*
 * Parse one complete unsigned integer without accepting signs or suffixes.
 *
 * Returns false and leaves result ownership with the caller.
 */
int parse_ull_allow_zero(const char *text, unsigned long long *out)
{
    char *end = NULL;
    unsigned long long value;
    if (!text || !out || text[0] == '\0' || text[0] == '-') return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *out = value;
    return 1;
}
int parse_positive_ull(const char *text, unsigned long long *out)
{
    return parse_ull_allow_zero(text, out) && *out != 0;
}
int parse_uint_allow_zero(const char *text, unsigned int *out)
{
    unsigned long long value;
    if (!out || !parse_ull_allow_zero(text, &value) || value > UINT32_MAX) return 0;
    *out = (unsigned int)value;
    return 1;
}
void print_quoted_bytes(const char *data, unsigned long long len)
{
    unsigned long long i;
    yvex_cli_out_writef(stdout, "\"");
    for (i = 0; i < len; ++i) {
        unsigned char ch = (unsigned char)data[i];
        if (ch == '"' || ch == '\\') {
            yvex_cli_out_writef(stdout, "\\%c", (int)ch);
        } else if (ch == '\n') {
            yvex_cli_out_writef(stdout, "\\n");
        } else if (ch == '\r') {
            yvex_cli_out_writef(stdout, "\\r");
        } else if (ch == '\t') {
            yvex_cli_out_writef(stdout, "\\t");
        } else if (ch < 32 || ch > 126) {
            yvex_cli_out_writef(stdout, "\\x%02x", (unsigned int)ch);
        } else {
            yvex_cli_out_writef(stdout, "%c", (int)ch);
        }
    }
    yvex_cli_out_writef(stdout, "\"");
}
int open_artifact_for_gguf(const char *path, yvex_artifact **artifact, yvex_error *err)
{
    yvex_artifact_options options;
    yvex_model_ref ref;
    int rc;
    memset(&options, 0, sizeof(options));
    memset(&ref, 0, sizeof(ref));
    rc = yvex_model_ref_resolve(&ref, path, NULL, err);
    if (rc != YVEX_OK) return rc;
    options.path = ref.path;
    options.readonly = 1;
    rc = yvex_artifact_open(artifact, &options, err);
    yvex_model_ref_clear(&ref);
    return rc;
}
void print_tensor_dims(const unsigned long long *dims, unsigned int rank)
{
    unsigned int i;
    yvex_cli_out_writef(stdout, "[");
    for (i = 0; i < rank; ++i) {
        if (i > 0) yvex_cli_out_writef(stdout, ",");
        yvex_cli_out_writef(stdout, "%llu", dims[i]);
    }
    yvex_cli_out_writef(stdout, "]");
}
void print_native_dims(const unsigned long long *dims, unsigned int rank)
{
    print_tensor_dims(dims, rank);
}
void print_token_ids(const yvex_tokens *tokens)
{
    unsigned long long i;
    yvex_cli_out_writef(stdout, "ids:");
    for (i = 0; i < tokens->len; ++i) yvex_cli_out_writef(stdout, " %u", tokens->ids[i]);
    yvex_cli_out_writef(stdout, "\n");
}
int parse_id_list(const char *text, unsigned int **out_ids, unsigned long long *out_len)
{
    unsigned int *ids = NULL;
    unsigned long long len = 0;
    unsigned long long capacity = 0;
    const char *cursor = text;
    if (!text || !out_ids || !out_len) return 0;
    *out_ids = NULL;
    *out_len = 0;
    while (*cursor) {
        char *end = NULL;
        unsigned long value = strtoul(cursor, &end, 10);
        unsigned int *next;
        if (end == cursor || value > UINT32_MAX) goto fail;
        if (len == capacity) {
            unsigned long long next_capacity = capacity == 0 ? 8 : capacity * 2u;
            if (next_capacity > (unsigned long long)(SIZE_MAX / sizeof(*ids))) goto fail;
            next = realloc(ids, (size_t)next_capacity * sizeof(*ids));
            if (!next) goto fail;
            ids = next;
            capacity = next_capacity;
        }
        ids[len++] = (unsigned int)value;
        if (*end == ',') {
            cursor = end + 1;
        } else if (*end == '\0') {
            cursor = end;
        } else {
            goto fail;
        }
    }
    if (len == 0) goto fail;
    *out_ids = ids;
    *out_len = len;
    return 1;

fail:
    free(ids);
    return 0;
}

/*
 * Parse an exact-rank comma-separated tensor shape.
 *
 * Rejects invalid rank, malformed fields, zero dimensions, and overflow.
 */
int parse_dims_csv(const char *text, unsigned int rank, unsigned long long dims[4])
{
    const char *cursor = text;
    char *end = NULL;
    unsigned int i;

    if (!text || !dims || rank == 0 || rank > 4u) return 0;
    memset(dims, 0, 4u * sizeof(*dims));
    for (i = 0; i < rank; ++i) {
        errno = 0;
        dims[i] = strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor || dims[i] == 0) return 0;
        if (i + 1u < rank) {
            if (*end != ',') return 0;
            cursor = end + 1;
        } else if (*end != '\0') {
            return 0;
        }
    }
    return 1;
}

/*
 * Provide approved direct JSON text output for CLI plumbing surfaces.
 *
 * Helpers serialize only caller-provided fields and do not claim uniform JSON. JSON writer
 * primitives are not command-level JSON support by themselves.
 */
void yvex_cli_out_json_string(FILE *fp, const char *text) {
    const unsigned char *p = (const unsigned char *)(text ? text : "");

    (void)yvex_cli_out_char(fp, '"');
    while (*p) {
        if (*p == '"' || *p == '\\') {
            (void)yvex_cli_out_char(fp, '\\');
            (void)yvex_cli_out_char(fp, *p);
        } else if (*p == '\n') {
            (void)yvex_cli_out_puts(fp, "\\n");
        } else if (*p == '\r') {
            (void)yvex_cli_out_puts(fp, "\\r");
        } else if (*p == '\t') {
            (void)yvex_cli_out_puts(fp, "\\t");
        } else if (*p < 0x20u) {
            (void)yvex_cli_out_writef(fp, "\\u%04x", (unsigned int)*p);
        } else {
            (void)yvex_cli_out_char(fp, *p);
        }
        ++p;
    }
    (void)yvex_cli_out_char(fp, '"');
}

static void discovery_json_list(FILE *output, const char *value, int delimiter)
{
    const char *cursor = value;
    int first = 1;
    fputc('[', output);
    if (strcmp(value, "none")) {
        while (*cursor) {
            const char *end = strchr(cursor, delimiter);
            size_t extent = end ? (size_t)(end - cursor) : strlen(cursor);
            char item[512];
            if (extent >= sizeof(item)) extent = sizeof(item) - 1u;
            memcpy(item, cursor, extent);
            item[extent] = '\0';
            if (!first) fputc(',', output);
            yvex_cli_out_json_string(output, item);
            first = 0;
            if (!end) break;
            cursor = end + 1;
        }
    }
    fputc(']', output);
}
static const char *visibility_name(yvex_operator_visibility visibility)
{
    switch (visibility) {
    case YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT: return "product-default";
    case YVEX_OPERATOR_VISIBILITY_PRODUCT_ADVANCED: return "product-advanced";
    case YVEX_OPERATOR_VISIBILITY_ENGINEERING: return "engineering";
    case YVEX_OPERATOR_VISIBILITY_AUTOMATION: return "automation";
    case YVEX_OPERATOR_VISIBILITY_API_ONLY: return "API-only";
    case YVEX_OPERATOR_VISIBILITY_TEST_ONLY: return "test-only";
    case YVEX_OPERATOR_VISIBILITY_REMOVED: return "removed";
    }
    return "unknown";
}
static const char *plane_name(yvex_operator_plane plane)
{
    static const char *const names[] = {
        "Compile", "Execute", "Inspect", "Integrate", "Profile", "Run", "System"};
    return (unsigned int)plane < sizeof(names) / sizeof(names[0]) ? names[plane]
                                                                  : "Unknown";
}
static const char *lane_name(yvex_operator_lane lane)
{
    switch (lane) {
    case YVEX_OPERATOR_LANE_RUNTIME_CLIENT: return "runtime-client";
    case YVEX_OPERATOR_LANE_OFFLINE_ENGINE: return "offline-engine";
    case YVEX_OPERATOR_LANE_DAEMON_ENTRYPOINT: return "host-entrypoint";
    case YVEX_OPERATOR_LANE_REPL_LOCAL: return "REPL-local";
    case YVEX_OPERATOR_LANE_API_ONLY: return "API-only";
    case YVEX_OPERATOR_LANE_TEST_ONLY: return "test-only";
    }
    return "unknown";
}
static int descriptor_has_prefix(const yvex_operator_descriptor *descriptor,
                                 size_t count, const char *const *path)
{
    size_t index;
    if (count > descriptor->command_word_count) return 0;
    for (index = 0u; index < count; ++index)
        if (strcmp(path[index], descriptor->command_words[index])) return 0;
    return 1;
}
static void render_leaf_usage(FILE *output,
                              const yvex_operator_descriptor *descriptor)
{
    size_t index;
    fputs("usage: yvex", output);
    if (descriptor->command_path[0])
        fprintf(output, " %s", descriptor->command_path);
    for (index = 0u; index < descriptor->argument_count; ++index) {
        const yvex_operator_argument_descriptor *argument = &descriptor->arguments[index];
        if (!strcmp(argument->multiplicity, "many"))
            fprintf(output, " [%s ...]", argument->name);
        else if (argument->required)
            fprintf(output, " %s", argument->name);
        else
            fprintf(output, " [%s]", argument->name);
    }
    if (descriptor->flag_count) fputs(" [options]", output);
    fputc('\n', output);
}
static const char *flag_value_name(const yvex_operator_flag_descriptor *flag)
{
    if (strcmp(flag->enum_values, "none")) return flag->enum_values;
    if (!strcmp(flag->value_type, "u64")) return "N";
    if (!strcmp(flag->value_type, "number")) return "NUMBER";
    if (!strcmp(flag->value_type, "path")) return "PATH";
    if (!strcmp(flag->value_type, "name")) return "NAME";
    if (!strcmp(flag->value_type, "text")) return "TEXT";
    return "VALUE";
}
static void render_metadata_text(const char *value)
{
    for (; *value; ++value)
        if (*value == '|') fputs(", ", stdout); else fputc(*value, stdout);
}
static void render_leaf_help(const yvex_operator_descriptor *descriptor)
{
    size_t index;
    render_leaf_usage(stdout, descriptor);
    printf("\n%s\n\noperation: %s\nplane: %s\nvisibility: %s\nlane: %s\n",
           descriptor->summary, descriptor->operation_id, plane_name(descriptor->plane),
           visibility_name(descriptor->visibility), lane_name(descriptor->lane));
    if (descriptor->flag_count) {
        puts("\noptions:");
        for (index = 0u; index < descriptor->flag_count; ++index) {
            const yvex_operator_flag_descriptor *flag = &descriptor->flags[index];
            char syntax[384];
            const char *separator = "";
            snprintf(syntax, sizeof(syntax), "%s%s%s", flag->name,
                     flag->takes_value ? " " : "",
                     flag->takes_value ? flag_value_name(flag) : "");
            printf("  %-42s", syntax);
            if (!strcmp(flag->multiplicity, "repeatable")) {
                fputs("repeatable", stdout); separator = "; ";
            }
            if (strcmp(flag->range, "delegated")) {
                printf("%srange %s", separator, flag->range); separator = "; ";
            }
            if (strcmp(flag->dependencies, "none")) {
                printf("%srequires ", separator); render_metadata_text(flag->dependencies);
                separator = "; ";
            }
            if (strcmp(flag->conflicts, "none")) {
                printf("%sconflicts ", separator); render_metadata_text(flag->conflicts);
                separator = "; ";
            }
            if (strcmp(flag->aliases, "none")) {
                printf("%salias ", separator); render_metadata_text(flag->aliases);
            }
            fputc('\n', stdout);
        }
    }
}

static void render_command_index_line(const yvex_operator_descriptor *descriptor)
{
    size_t index, width = strlen(descriptor->command_path);
    fputs("  yvex", stdout);
    if (descriptor->command_path[0])
        printf(" %s", descriptor->command_path);
    for (index = 0u; index < descriptor->argument_count; ++index) {
        const yvex_operator_argument_descriptor *argument = &descriptor->arguments[index];
        if (!strcmp(argument->multiplicity, "many")) {
            printf(" [%s ...]", argument->name);
            width += strlen(argument->name) + 7u;
        } else if (argument->required) {
            printf(" %s", argument->name);
            width += strlen(argument->name) + 1u;
        } else {
            printf(" [%s]", argument->name);
            width += strlen(argument->name) + 3u;
        }
    }
    if (width < 42u) printf("%*s", (int)(42u - width), "");
    else fputc(' ', stdout);
    printf(" %s%s\n", descriptor->summary,
           descriptor->visibility == YVEX_OPERATOR_VISIBILITY_ENGINEERING
               ? " [engineering]" : "");
}
void yvex_client_render_usage_error(const yvex_operator_descriptor *operation)
{
    render_leaf_usage(stderr, operation);
}
static void render_discovery_aliases(size_t operation_index)
{
    size_t index;
    int first = 1;
    fputc('[', stdout);
    for (index = 0u; index < yvex_operator_alias_count; ++index) {
        if (yvex_operator_aliases[index].operation_index != operation_index) continue;
        if (!first) fputc(',', stdout);
        yvex_cli_out_json_string(stdout, yvex_operator_aliases[index].path);
        first = 0;
    }
    fputc(']', stdout);
}
static void render_discovery_arguments(
    const yvex_operator_argument_descriptor *arguments, size_t count)
{
    size_t index;
#define JSON_ARGUMENT_FIELD(name, value) \
    do { fputs("\"" name "\":", stdout); yvex_cli_out_json_string(stdout, (value)); } while (0)
    fputc('[', stdout);
    for (index = 0u; index < count; ++index) {
        const yvex_operator_argument_descriptor *argument = &arguments[index];
        if (index) fputc(',', stdout);
        fputc('{', stdout); JSON_ARGUMENT_FIELD("name", argument->name);
        fputc(',', stdout); JSON_ARGUMENT_FIELD("type", argument->value_type);
        printf(",\"required\":%s,", argument->required ? "true" : "false");
        JSON_ARGUMENT_FIELD("multiplicity", argument->multiplicity);
        fputc(',', stdout); JSON_ARGUMENT_FIELD("range", argument->range);
        fputs(",\"enum_values\":", stdout);
        discovery_json_list(stdout, argument->enum_values, '|');
        fputc(',', stdout);
        JSON_ARGUMENT_FIELD("completion_provider", argument->completion_provider);
        fputc(',', stdout);
        JSON_ARGUMENT_FIELD("sensitive_display", argument->sensitive_display);
        fputc(',', stdout); JSON_ARGUMENT_FIELD("validator", argument->validator);
        fputc('}', stdout);
    }
    fputc(']', stdout);
#undef JSON_ARGUMENT_FIELD
}
static void render_discovery_operation(size_t operation_index,
                                       const yvex_operator_descriptor *descriptor)
{
    size_t index;
#define JSON_FIELD(name, value) \
    do { fputs("\"" name "\":", stdout); yvex_cli_out_json_string(stdout, (value)); } while (0)
    fputc('{', stdout);
    printf("\"schema_version\":%u,", descriptor->schema_version);
    JSON_FIELD("operation_id", descriptor->operation_id);
    fputc(',', stdout); JSON_FIELD("command_path", descriptor->command_path);
    fputs(",\"aliases\":", stdout); render_discovery_aliases(operation_index);
    fputc(',', stdout); JSON_FIELD("visibility", visibility_name(descriptor->visibility));
    fputc(',', stdout); JSON_FIELD("plane", plane_name(descriptor->plane));
    fputc(',', stdout); JSON_FIELD("lane", lane_name(descriptor->lane));
    fputc(',', stdout); JSON_FIELD("summary", descriptor->summary);
    fputs(",\"arguments\":", stdout);
    render_discovery_arguments(descriptor->arguments, descriptor->argument_count);
    fputs(",\"slash_arguments\":", stdout);
    render_discovery_arguments(descriptor->slash_arguments,
                               descriptor->slash_argument_count);
    fputs(",\"flags\":[", stdout);
    for (index = 0u; index < descriptor->flag_count; ++index) {
        const yvex_operator_flag_descriptor *flag = &descriptor->flags[index];
        if (index) fputc(',', stdout);
        fputc('{', stdout); JSON_FIELD("name", flag->name);
        fputs(",\"aliases\":", stdout); discovery_json_list(stdout, flag->aliases, '|');
        fputc(',', stdout); JSON_FIELD("type", flag->value_type);
        printf(",\"takes_value\":%s,", flag->takes_value ? "true" : "false");
        printf("\"required\":%s,", flag->required ? "true" : "false");
        JSON_FIELD("multiplicity", flag->multiplicity);
        fputc(',', stdout); JSON_FIELD("default_provider", flag->default_provider);
        fputc(',', stdout); JSON_FIELD("range", flag->range);
        fputs(",\"enum_values\":", stdout); discovery_json_list(stdout, flag->enum_values, '|');
        fputs(",\"conflicts\":", stdout); discovery_json_list(stdout, flag->conflicts, '|');
        fputs(",\"dependencies\":", stdout);
        discovery_json_list(stdout, flag->dependencies, '|');
        fputc(',', stdout); JSON_FIELD("environment", flag->environment);
        fputc(',', stdout); JSON_FIELD("config", flag->config);
        fputc(',', stdout); JSON_FIELD("protocol_field", flag->protocol_field);
        fputc(',', stdout); JSON_FIELD("output_interaction", flag->output_interaction);
        fputc(',', stdout); JSON_FIELD("deprecation", flag->deprecation);
        fputc(',', stdout); JSON_FIELD("validator", flag->validator);
        fputc('}', stdout);
    }
    fputs("],\"default_providers\":", stdout);
    discovery_json_list(stdout, descriptor->default_providers, '|');
    fputs(",\"validators\":", stdout);
    discovery_json_list(stdout, descriptor->validator_ids, '|');
    fputs(",\"input_schema\":", stdout); yvex_cli_out_json_string(stdout, descriptor->input_schema);
    fputs(",\"result_schema\":", stdout); yvex_cli_out_json_string(stdout, descriptor->result_schema);
    fputs(",\"side_effects\":", stdout); yvex_cli_out_json_string(stdout, descriptor->side_effects);
    fputs(",\"tty_policy\":", stdout); yvex_cli_out_json_string(stdout, descriptor->tty_policy);
    fputs(",\"requirements\":{", stdout); JSON_FIELD("daemon", descriptor->daemon_requirement);
    fputc(',', stdout); JSON_FIELD("model", descriptor->model_requirement);
    fputc(',', stdout); JSON_FIELD("artifact", descriptor->artifact_requirement);
    fputc(',', stdout); JSON_FIELD("backend", descriptor->backend_requirement);
    fputs("},\"output_schemas\":[", stdout); yvex_cli_out_json_string(stdout, descriptor->result_schema);
    fputs("],\"adapter_id\":", stdout); yvex_cli_out_json_string(stdout, descriptor->adapter_id);
    fputs(",\"renderer_id\":", stdout); yvex_cli_out_json_string(stdout, descriptor->renderer_id);
    fputs(",\"completion_provider\":", stdout);
    yvex_cli_out_json_string(stdout, descriptor->completion_provider);
    fputs(",\"projections\":{\"cli\":", stdout);
    fputs(descriptor->cli_projection ? "true" : "false", stdout);
    fputs(",\"slash\":", stdout); yvex_cli_out_json_string(stdout, descriptor->slash_projection);
    fputs(",\"slash_aliases\":", stdout);
    discovery_json_list(stdout, descriptor->slash_aliases, ',');
    fputs(",\"protocol\":", stdout); yvex_cli_out_json_string(stdout, descriptor->protocol_operation);
    fputs("},\"deprecation\":", stdout); yvex_cli_out_json_string(stdout, descriptor->deprecation_state);
    fputs(",\"superseded_by\":", stdout);
    discovery_json_list(stdout, descriptor->superseded_by, '|');
    fputs(",\"test_owner\":", stdout); yvex_cli_out_json_string(stdout, descriptor->test_owner);
    fputs(",\"documentation_owner\":", stdout);
    yvex_cli_out_json_string(stdout, descriptor->documentation_owner);
    fputc('}', stdout);
#undef JSON_FIELD
}
static void render_discovery_json(void)
{
    size_t index;
    fputs("{\"schema\":", stdout); yvex_cli_out_json_string(stdout, YVEX_COMMAND_DISCOVERY_SCHEMA);
    fputs(",\"registry_identity\":", stdout); yvex_cli_out_json_string(stdout, yvex_operator_registry_identity);
    fputs(",\"build_commit\":", stdout); yvex_cli_out_json_string(stdout, YVEX_BUILD_COMMIT);
    fputs(",\"operations\":[", stdout);
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        if (index) fputc(',', stdout);
        render_discovery_operation(index, &yvex_operator_descriptors[index]);
    }
    fputs("]}\n", stdout);
}

static int root_group(const char *root)
{
    if (!strcmp(root, "chat")) return 0;
    if (!strcmp(root, "serve") || !strcmp(root, "model") ||
        !strcmp(root, "host")) return 1;
    if (!strcmp(root, "inspect")) return 2;
    if (!strcmp(root, "help") || !strcmp(root, "version")) return 3;
    return 4;
}

static int root_first_visible(size_t candidate)
{
    const yvex_operator_descriptor *row = &yvex_operator_descriptors[candidate];
    size_t index;
    if (!row->cli_projection || !row->command_word_count ||
        row->visibility != YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT ||
        root_group(row->command_words[0]) == 4)
        return 0;
    for (index = 0u; index < candidate; ++index) {
        const yvex_operator_descriptor *prior = &yvex_operator_descriptors[index];
        if (prior->cli_projection && prior->command_word_count &&
            prior->visibility == YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT &&
            !strcmp(prior->command_words[0], row->command_words[0]))
            return 0;
    }
    return 1;
}

static const char *root_summary(const char *root)
{
    static const struct { const char *root; const char *summary; } domains[] = {
        {"host", "Inspect and control the foreground host."},
        {"engine", "Load, inspect, and unload engine generations."},
        {"session", "Manage generation-bound conversation state."},
        {"model", "Find, pull, prepare, load, and manage models."},
        {"source", "Acquire, verify, and inspect exact source revisions."},
        {"artifact", "Inspect and verify immutable compiled packages."},
        {"profile", "Inspect durable deployment configurations."},
        {"inspect", "Read bounded system and package evidence."},
        {"bench", "Run bounded component execution and measurement."},
    };
    size_t index;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *row = &yvex_operator_descriptors[index];
        if (row->cli_projection && row->command_word_count == 1u &&
            row->visibility != YVEX_OPERATOR_VISIBILITY_REMOVED &&
            !strcmp(row->command_words[0], root))
            return row->summary;
    }
    for (index = 0u; index < sizeof(domains) / sizeof(domains[0]); ++index)
        if (!strcmp(domains[index].root, root)) return domains[index].summary;
    return "Domain operations.";
}

static void render_root_map(void)
{
    static const char *const labels[] = {"USE", "RUNTIME", "TOOLS", "META"};
    size_t group, index;
    puts("YVEX inference runtime");
    for (group = 0u; group < sizeof(labels) / sizeof(labels[0]); ++group) {
        printf("\n%s\n", labels[group]);
        for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
            const yvex_operator_descriptor *row = &yvex_operator_descriptors[index];
            if (root_first_visible(index) &&
                root_group(row->command_words[0]) == (int)group)
                printf("  %-10s %s\n", row->command_words[0],
                       root_summary(row->command_words[0]));
        }
    }
    puts("\nUse `yvex help COMMAND` for details.");
}

static void render_default_namespace(const char *root, const char *title)
{
    size_t index;
    printf("\n%s\n", title);
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[index];
        if (descriptor->cli_projection && descriptor->command_word_count > 1u &&
            descriptor->visibility == YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT &&
            !strcmp(descriptor->command_words[0], root))
            render_command_index_line(descriptor);
    }
}

static void render_product_grammar(void)
{
    puts("\nLIFECYCLE\n"
         "  model search -> model pull -> model prepare -> serve -> model load -> chat\n"
         "  model push distributes; model unload changes runtime residency.");
    render_default_namespace("model", "MODEL COMMANDS");
    render_default_namespace("host", "HOST CONTROL");
}

int yvex_client_render_help_path(size_t path_count, const char *const *path,
                                 int advanced, int json)
{
    const yvex_operator_descriptor *exact = NULL;
    size_t index, matches = 0u;
    if (json) {
        render_discovery_json();
        return 0;
    }
    if (!path_count) {
        render_root_map();
        render_product_grammar();
        if (advanced) {
            puts("\nADVANCED AND ENGINEERING\n");
            for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
                const yvex_operator_descriptor *descriptor =
                    &yvex_operator_descriptors[index];
                if (descriptor->cli_projection &&
                    (descriptor->visibility == YVEX_OPERATOR_VISIBILITY_PRODUCT_ADVANCED ||
                     descriptor->visibility == YVEX_OPERATOR_VISIBILITY_ENGINEERING))
                    render_command_index_line(descriptor);
            }
        } else {
            puts("Use `yvex help --advanced` for advanced and engineering commands.");
        }
        return 0;
    }
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        if (!descriptor->cli_projection || !descriptor_has_prefix(descriptor, path_count, path))
            continue;
        if (descriptor->command_word_count == path_count) exact = descriptor;
        matches++;
    }
    if (!matches) {
        fprintf(stderr, "yvex: unknown help path");
        for (index = 0u; index < path_count; ++index) fprintf(stderr, " %s", path[index]);
        fputc('\n', stderr);
        return 2;
    }
    if (exact) {
        render_leaf_help(exact);
        if (matches == 1u) return 0;
        puts("\nsubcommands:");
    } else {
        puts(path_count ? "YVEX command namespace\n" : "YVEX local inference\n");
    }
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        int visible = path_count != 0u ||
                      descriptor->visibility == YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT ||
                      (advanced && (descriptor->visibility == YVEX_OPERATOR_VISIBILITY_PRODUCT_ADVANCED ||
                                    descriptor->visibility == YVEX_OPERATOR_VISIBILITY_ENGINEERING));
        if (descriptor != exact && descriptor->cli_projection && visible &&
            descriptor_has_prefix(descriptor, path_count, path))
            render_command_index_line(descriptor);
    }
    if (!advanced) puts("\nUse `yvex help --advanced` for advanced and engineering commands.");
    return 0;
}
void yvex_cli_json_begin(FILE *fp) {
    yvex_cli_out_line(fp, "{");
}
void yvex_cli_json_end(FILE *fp) {
    yvex_cli_out_line(fp, "}");
}

static int json_field(FILE *fp, const char *key, yvex_cli_field_kind kind, const void *value,
                      int comma) {
    (void)yvex_cli_out_puts(fp, "  ");
    yvex_cli_out_json_string(fp, key);
    (void)yvex_cli_out_puts(fp, ": ");
    switch (kind) {
    case YVEX_CLI_FIELD_TEXT:
    case YVEX_CLI_FIELD_TEXT_ARRAY:
        yvex_cli_out_json_string(fp, value);
        break;
    case YVEX_CLI_FIELD_U64:
        (void)yvex_cli_out_writef(fp, "%llu", *(const unsigned long long *)value);
        break;
    case YVEX_CLI_FIELD_U32:
        (void)yvex_cli_out_writef(fp, "%u", *(const unsigned int *)value);
        break;
    case YVEX_CLI_FIELD_I32:
        (void)yvex_cli_out_writef(fp, "%d", *(const int *)value);
        break;
    case YVEX_CLI_FIELD_BOOL:
        (void)yvex_cli_out_puts(fp, *(const int *)value ? "true" : "false");
        break;
    case YVEX_CLI_FIELD_DOUBLE:
        if (isfinite(*(const double *)value))
            (void)yvex_cli_out_writef(fp, "%.17g", *(const double *)value);
        else
            (void)yvex_cli_out_puts(fp, "null");
        break;
    default:
        return YVEX_ERR_UNSUPPORTED;
    }
    (void)yvex_cli_out_writef(fp, "%s\n", comma ? "," : "");
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}

void yvex_cli_json_field_str(FILE *fp, const char *key, const char *value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_TEXT_ARRAY, value ? value : "", comma);
}

void yvex_cli_json_field_u64(FILE *fp, const char *key, unsigned long long value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_U64, &value, comma);
}

void yvex_cli_json_field_bool(FILE *fp, const char *key, int value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_BOOL, &value, comma);
}

int yvex_cli_json_fields(FILE *fp, const void *object, const yvex_cli_field_spec *fields,
                         size_t field_count, int comma) {
    const unsigned char *base = object;
    size_t index;

    if (!fp || !object || (!fields && field_count))
        return YVEX_ERR_INVALID_ARG;
    for (index = 0; index < field_count; ++index) {
        const yvex_cli_field_spec *field = &fields[index];
        const void *value = base + field->offset;
        int separator = comma || index + 1u < field_count;
        if (field->kind == YVEX_CLI_FIELD_TEXT)
            value = *(const char *const *)value;
        if (json_field(fp, field->key, field->kind, value, separator) != YVEX_OK)
            return ferror(fp) ? YVEX_ERR_IO : YVEX_ERR_UNSUPPORTED;
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
