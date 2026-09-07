/* Standalone CLI integration fixture: exercise the real renderer without a model. */
#include "src/cli/io/private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char output[16384];
static int failures;

static void expect(int condition, const char *reason)
{
    if (condition) return;
    fprintf(stderr, "log renderer: %s\n%s\n", reason, output);
    failures++;
}

static int capture(yvex_cli_watch_renderer *renderer, yvex_server_event *event,
                   const yvex_server_summary *live)
{
    yvex_server_event before = *event;
    FILE *file = tmpfile();
    int saved, shown;
    size_t size;
    if (!file || (saved = dup(STDOUT_FILENO)) < 0) exit(2);
    fflush(stdout);
    if (dup2(fileno(file), STDOUT_FILENO) < 0) exit(2);
    shown = yvex_cli_watch_renderer_event(renderer, event, live);
    fflush(stdout);
    if (dup2(saved, STDOUT_FILENO) < 0) exit(2);
    close(saved);
    rewind(file);
    size = fread(output, 1, sizeof(output) - 1, file);
    output[size] = '\0';
    expect(feof(file), "bounded output fits the capture");
    fclose(file);
    expect(!memcmp(&before, event, sizeof(before)), "typed event is unchanged");
    return shown;
}

static yvex_server_event decode_event(void)
{
    yvex_server_event event = {0};
    event.kind = YVEX_SERVER_EVENT_GENERATION_PROGRESS;
    event.engine_kind = YVEX_SERVER_ENGINE_TEXT;
    event.wall_time_ns = 1788708011000000000ull;
    strcpy(event.session_id, "main");
    strcpy(event.request_id, "r5");
    strcpy(event.phase, "answer");
    event.value_a = 280;
    event.value_b = 307;
    event.value_c = 97;
    event.seconds = 40;
    event.rate = 999; /* The typed measurement, not this legacy fallback, wins. */
    event.measurement.schema_version = YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1;
    event.measurement.scope = YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE;
    event.measurement.work_unit = YVEX_EXECUTION_WORK_TOKENS;
    event.measurement.available = YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE |
                                  YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE;
    event.measurement.cumulative_rate = 6.98;
    event.measurement.rolling_rate = 5.38;
    event.measurement.rolling_units = 32;
    event.measurement.rolling_window_units = 32;
    return event;
}

static void test_decode(void)
{
    yvex_cli_watch_renderer renderer;
    yvex_server_event event = decode_event();
    yvex_cli_watch_renderer_open(&renderer, 0);
    expect(capture(&renderer, &event, NULL), "first progress is visible");
    expect(strstr(output, "generated=280 position=307 phase=answer reasoning=97") != NULL,
           "counts and phase are named");
    expect(strstr(output, "decode-avg=6.98 tok/s rolling[32]=5.38 tok/s") != NULL,
           "slow tail and cumulative scope remain distinct");
    expect(!strstr(output, "999") && !strstr(output, "RESOURCES"),
           "no fallback rate or fabricated resources");
    expect(!capture(&renderer, &event, NULL), "duplicate progress remains coalesced");
    strcpy(event.phase, "reasoning");
    expect(capture(&renderer, &event, NULL), "phase transition is never hidden");
    expect(!strstr(output, "reasoning=97"), "reasoning count does not duplicate phase");
    strcpy(event.session_id, "auxiliary");
    event.measurement.rolling_units = 10;
    expect(capture(&renderer, &event, NULL), "another session's progress survives");
    expect(strstr(output, "rolling[10/32]=5.38 tok/s") != NULL,
           "warming rolling window exposes actual and maximum token counts");
    event.seconds++;
    event.measurement.available = 0;
    capture(&renderer, &event, NULL);
    expect(!strstr(output, "tok/s"), "typed unavailable rate rejects the legacy fallback");
    event.seconds++;
    event.rate = 0;
    capture(&renderer, &event, NULL);
    expect(!strstr(output, "tok/s"), "unavailable rate is not a false zero");
    event.seconds++;
    event.measurement.schema_version = 0;
    event.rate = 4;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "rate=4.00 tok/s") != NULL && !strstr(output, "decode-avg"),
           "legacy rate does not borrow a typed scope");
    event.kind = YVEX_SERVER_EVENT_GENERATION_CANCELLED;
    event.value_c = 5;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "CANCELLED") && strstr(output, "stop=\"cancelled\""),
           "cancellation is explicit even after coalesced progress");
    event.kind = YVEX_SERVER_EVENT_GENERATION_FAILED;
    event.value_c = 6;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "FAIL") && strstr(output, "stop=\"model failure\""),
           "model failure is not a cryptic stop code");
    event.kind = YVEX_SERVER_EVENT_GENERATION_COMPLETED;
    event.measurement.schema_version = YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1;
    event.measurement.scope = YVEX_EXECUTION_SCOPE_TOTAL_OPERATION;
    event.measurement.available = YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE;
    event.proposed_tokens = 100;
    event.accepted_tokens = 39;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "total-avg=6.98 tok/s") && strstr(output, "spec-accept=39.0%") &&
           !strstr(output, "decode-avg"), "total wall and speculative acceptance keep their scopes");
    event.kind = YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED;
    event.speculative_cycle = 8;
    event.proposed_tokens = 10;
    event.accepted_tokens = 4;
    event.rejected_tokens = 6;
    event.discarded_tokens = 2;
    yvex_cli_watch_renderer_open(&renderer, 1);
    capture(&renderer, &event, NULL);
    expect(strstr(output, "auxiliary/r5 cycle=8 accepted=4/10 rejected=6 discarded=2") != NULL,
           "detailed speculation retains request identity and exact counts");
}

static void test_lifecycle(void)
{
    yvex_cli_watch_renderer renderer;
    yvex_server_event event = decode_event();
    yvex_cli_watch_renderer_open(&renderer, 0);
    event.kind = YVEX_SERVER_EVENT_REQUEST_STARTED;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "input_bytes=280 prefix_tokens=307 max_tokens=97") != NULL,
           "native request input is bytes, not tokens");
    strcpy(event.provider_request_identity, "provider-request");
    capture(&renderer, &event, NULL);
    expect(strstr(output, "messages=280") && !strstr(output, "input_bytes"),
           "provider request input is messages, not bytes");
    event.kind = YVEX_SERVER_EVENT_TOKENIZER_COMPLETED;
    event.value_a = 126;
    event.value_b = 100;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "PROMPT") && strstr(output, "tokens=126 reused=100"),
           "prompt reuse is visible without detailed logging");
    event.kind = YVEX_SERVER_EVENT_SESSION_RESET;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "main reset") && !strstr(output, "closed"),
           "reset is not falsely rendered as session close");
    event.kind = YVEX_SERVER_EVENT_ENGINE_LOAD_PROGRESS;
    strcpy(event.phase, "artifact-verification");
    event.measurement.available = YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE;
    event.measurement.work_unit = YVEX_EXECUTION_WORK_BYTES;
    event.measurement.completed_units = 1024;
    event.measurement.total_units = 2048;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "phase=artifact-verification completed=1.0/2.0KiB (50%)") != NULL,
           "load has real phase, denominator and binary units");
    event.measurement.available = 0;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "completed=1024 bytes total=unknown") && !strchr(output, '%'),
           "unknown denominator stays unknown");
    event.measurement.available = YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE;
    event.measurement.total_units = 0;
    capture(&renderer, &event, NULL);
    expect(!strchr(output, '%'), "zero denominator cannot manufacture percentage");
    event.kind = YVEX_SERVER_EVENT_PREFILL_PROGRESS;
    strcpy(event.phase, "prefill");
    event.measurement.work_unit = YVEX_EXECUTION_WORK_TOKENS;
    event.measurement.total_units = 2048;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "PREFILL") && strstr(output, "main/r5 phase=prefill") &&
           strstr(output, "completed=1024/2048 tokens (50%)"),
           "long prefill progress keeps request identity and a real denominator");
    event.kind = YVEX_SERVER_EVENT_TELEMETRY_DROPPED;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "telemetry coalesced=97 dropped=126 capacity=100") != NULL,
           "telemetry pressure remains visible");
}

static void test_resources(void)
{
    yvex_cli_watch_renderer renderer;
    yvex_server_event event = decode_event();
    yvex_server_summary live = {0}, before;
    unsigned long long start = event.wall_time_ns;
    int index, snapshots = 0;
    live.metrics.resources.available = YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE | YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE;
    live.metrics.resources.process_rss_current_bytes = 98784247808ull;
    live.metrics.resources.workspace_current_bytes = 9985798963ull;
    live.metrics.resources.session_physical_state_bytes = 78957773ull;
    live.metrics.resources.model_device_addressable_bytes = 53687091200ull;
    live.metrics.active_requests = 1;
    before = live;
    yvex_cli_watch_renderer_open(&renderer, 0);
    for (index = 0; index < 100; ++index) {
        event.wall_time_ns = start + (unsigned long long)index * 1000000000ull;
        event.seconds = 40 + index;
        capture(&renderer, &event, &live);
        if (strstr(output, "RESOURCES")) snapshots++;
        if (index == 0) {
            expect(strstr(output, "host snapshot process-rss=92.0GiB workspace=9.3GiB") != NULL,
                   "resource snapshot is host-wide and has units");
            expect(!strstr(output, "device-alloc=0") && !strstr(output, "resident=0"),
                   "UMA mapping is not represented as zero GPU usage");
        }
    }
    expect(snapshots == 10, "100 progress events produce only 10 resource snapshots");
    event.kind = YVEX_SERVER_EVENT_GENERATION_COMPLETED;
    event.value_c = 3;
    capture(&renderer, &event, &live);
    expect(strstr(output, "DONE") && strstr(output, "RESOURCES"),
           "completion forces the final available snapshot");
    expect(!memcmp(&live, &before, sizeof(live)), "resource authority is unchanged");
    live.metrics.resources.available = 0;
    capture(&renderer, &event, &live);
    expect(!strstr(output, "process-rss=") && !strstr(output, "workspace="),
           "unavailable resources are not printed as measured");
    event.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    capture(&renderer, &event, NULL);
    expect(strstr(output, "frames=280 bytes=307 audio_samples=3") && !strstr(output, "tok/s"),
           "media completion does not borrow text-token units");
}

int main(void)
{
    yvex_cli_watch_renderer renderer;
    yvex_server_event event = decode_event();
    test_decode();
    test_lifecycle();
    test_resources();
    if (failures) return 1;
    yvex_cli_watch_renderer_open(&renderer, 0);
    yvex_cli_watch_renderer_event(&renderer, &event, NULL);
    puts("log renderer: PASS");
    return 0;
}
