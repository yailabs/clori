#!/bin/sh
# Purpose: prove one yvexd PID serves Unix and OpenAI Chat/Responses/SSE over one resident model.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEXD_BIN=${YVEXD_BIN:-./yvexd}
ARTIFACT=${YVEX_MODEL_ARTIFACT:?YVEX_MODEL_ARTIFACT is required}
BINDING=${YVEX_RUNTIME_BINDING:?YVEX_RUNTIME_BINDING is required}
. tests/support/cleanup.sh

test -f "$ARTIFACT"
test -f "$BINDING"
root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-openai-live.XXXXXX")
runtime="$root/runtime"
mkdir -m 700 "$runtime"
socket="$runtime/yvex/yvexd.sock"
daemon_pid=
cleanup()
{
    status=$?
    trap - EXIT HUP INT TERM
    if test -n "$daemon_pid" && kill -0 "$daemon_pid" 2>/dev/null; then
        XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime stop >/dev/null 2>&1 || true
        kill "$daemon_pid" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    if test "$status" -ne 0; then
        printf 'OpenAI live failure; diagnostics: %s\n' "$root" >&2
        for file in daemon.err status.json status.after.json chat.json chat.sse; do
            test ! -f "$root/$file" || {
                printf '[%s]\n' "$file" >&2
                tail -40 "$root/$file" >&2
            }
        done
    fi
    if test "${YVEX_KEEP_TEST_OUTPUT:-0}" = 1; then
        printf 'OpenAI live output retained: %s\n' "$root" >&2
    else
        yvex_test_cleanup "$root"
    fi
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
XDG_RUNTIME_DIR="$runtime" "$YVEXD_BIN" --model "$ARTIFACT" \
    --runtime-binding "$BINDING" --backend cuda --context 512 \
    --console raw --trace-level stages --openai on --openai-port "$port" \
    --openai-timeout-ms 3600000 >"$root/raw.jsonl" 2>"$root/daemon.err" &
daemon_pid=$!

ready=0
attempt=0
while test "$attempt" -lt 3600; do
    if XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime status --json \
        >"$root/status.json" 2>/dev/null; then
        ready=1
        break
    fi
    kill -0 "$daemon_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 1
done
test "$ready" -eq 1
grep -F '"model_open_count":1' "$root/status.json" >/dev/null
grep -F '"artifact_open_count":1' "$root/status.json" >/dev/null
grep -F '"materialization_count":1' "$root/status.json" >/dev/null
grep -F '"residency_build_count":1' "$root/status.json" >/dev/null
grep -F '"openai_enabled":true' "$root/status.json" >/dev/null
grep -F '"openai_ready":true' "$root/status.json" >/dev/null
grep -F "\"openai_port\":$port" "$root/status.json" >/dev/null
python3 - "$root/status.json" "$ARTIFACT" "$daemon_pid" <<'PY'
import json, os, sys
status = json.load(open(sys.argv[1]))
artifact_bytes = os.path.getsize(sys.argv[2])
assert status['resident_host_bytes'] * 100 >= artifact_bytes * 95
rollup = {}
for line in open(f'/proc/{sys.argv[3]}/smaps_rollup'):
    fields = line.split()
    if len(fields) >= 2 and fields[0].rstrip(':') == 'Anonymous':
        rollup[fields[0].rstrip(':')] = int(fields[1]) * 1024
locked = 0
for line in open(f'/proc/{sys.argv[3]}/status'):
    fields = line.split()
    if fields and fields[0] == 'VmLck:':
        locked = int(fields[1]) * 1024
assert rollup['Anonymous'] * 100 >= status['resident_host_bytes'] * 95
assert locked * 100 >= status['resident_host_bytes'] * 95
PY

base="http://127.0.0.1:$port"
attempt=0
while test "$attempt" -lt 100; do
    curl --fail-with-body -sS "$base/health" >"$root/health.json" 2>/dev/null && break
    kill -0 "$daemon_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.05
done
test "$attempt" -lt 100
curl --fail-with-body -sS "$base/v1/models" >"$root/models.json"
model=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$root/models.json")

chat_body="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply briefly.\"}],\"temperature\":0,\"max_completion_tokens\":2}"
curl --fail-with-body -sS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d "$chat_body" >"$root/chat.json"
chat_stream="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply briefly.\"}],\"temperature\":0,\"max_completion_tokens\":2,\"stream\":true,\"stream_options\":{\"include_usage\":true}}"
curl --fail-with-body -sS -N -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d "$chat_stream" >"$root/chat.sse"

python3 - "$root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
health = json.load(open(root / 'health.json'))
assert health['gateway'] == 'ready' and health['yvexd'] == 'ready'
chat = json.load(open(root / 'chat.json'))
assert chat['object'] == 'chat.completion' and chat['usage']['total_tokens'] >= 1
assert chat['choices'][0]['finish_reason'] in ('stop', 'length')
def events(path):
    return [json.loads(line[6:]) for line in path.read_text().splitlines()
            if line.startswith('data: ') and line != 'data: [DONE]']
chat_events = events(root / 'chat.sse')
assert chat_events and chat_events[0]['choices'][0]['delta']['role'] == 'assistant'
assert (root / 'chat.sse').read_text().rstrip().endswith('data: [DONE]')
chat_text = ''.join(
    event['choices'][0]['delta'].get('content', '')
    for event in chat_events if event.get('choices'))
chat_terminal = next(
    event for event in chat_events
    if event.get('choices') and event['choices'][0].get('finish_reason'))
chat_usage = next(event['usage'] for event in chat_events if not event.get('choices'))
assert chat_text == chat['choices'][0]['message']['content']
assert chat_terminal['choices'][0]['finish_reason'] == chat['choices'][0]['finish_reason']
assert chat_usage == chat['usage']
PY

curl --max-time 0.2 -sS -N -H 'Content-Type: application/json' \
    "$base/v1/chat/completions" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long answer.\"}],\"temperature\":0,\"max_completion_tokens\":64,\"stream\":true}" \
    >"$root/cancel.sse" 2>"$root/cancel.err" || true
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime status --json >"$root/status.after.json"
grep -F '"model_open_count":1' "$root/status.after.json" >/dev/null
grep -F '"artifact_open_count":1' "$root/status.after.json" >/dev/null
grep -F '"materialization_count":1' "$root/status.after.json" >/dev/null
grep -F '"residency_build_count":1' "$root/status.after.json" >/dev/null
grep -F '"output_head_upload_count":1' "$root/status.after.json" >/dev/null
python3 - "$root/status.after.json" "$daemon_pid" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))
rollup = {}
for line in open(f'/proc/{sys.argv[2]}/smaps_rollup'):
    fields = line.split()
    if len(fields) >= 2 and fields[0].rstrip(':') == 'Anonymous':
        rollup[fields[0].rstrip(':')] = int(fields[1]) * 1024
locked = 0
for line in open(f'/proc/{sys.argv[2]}/status'):
    fields = line.split()
    if fields and fields[0] == 'VmLck:':
        locked = int(fields[1]) * 1024
assert rollup['Anonymous'] * 100 >= status['resident_host_bytes'] * 95
assert locked * 100 >= status['resident_host_bytes'] * 95
PY

curl --fail-with-body -sS "$base/health" >"$root/health.after.json"
kill -0 "$daemon_pid"
grep -F '"provider":"openai"' "$root/raw.jsonl" >/dev/null
! grep -F 'Reply briefly.' "$root/raw.jsonl" >/dev/null

served_pid=$daemon_pid
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime stop >/dev/null
wait "$daemon_pid"
daemon_pid=
grep -F '"kind":"runtime.shutdown.complete"' "$root/raw.jsonl" >/dev/null
printf 'test: openai_live one_daemon_pid=%s model=%s profile=yvex.openai.compat.v1\n' \
    "$served_pid" "$model"
