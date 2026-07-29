#!/bin/sh
# Purpose: prove the real loopback gateway translates Chat/Responses/SSE over one process-resident model.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEXD_BIN=${YVEXD_BIN:-./yvexd}
YVEX_OPENAI_BIN=${YVEX_OPENAI_BIN:-./yvex-openai}
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
gateway_pid=
cleanup()
{
    status=$?
    trap - EXIT HUP INT TERM
    if test -n "$gateway_pid" && kill -0 "$gateway_pid" 2>/dev/null; then
        kill "$gateway_pid" 2>/dev/null || true
        wait "$gateway_pid" 2>/dev/null || true
    fi
    if test -n "$daemon_pid" && kill -0 "$daemon_pid" 2>/dev/null; then
        XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime stop >/dev/null 2>&1 || true
        kill "$daemon_pid" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    if test "$status" -ne 0; then
        printf 'OpenAI live failure; diagnostics: %s\n' "$root" >&2
        for file in daemon.err gateway.err status.json status.after.json chat.json chat.sse \
            response.json response.sse tool.json tool-final.json; do
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

XDG_RUNTIME_DIR="$runtime" "$YVEXD_BIN" --model "$ARTIFACT" \
    --runtime-binding "$BINDING" --backend cuda --context 512 \
    --console raw --trace-level stages >"$root/raw.jsonl" 2>"$root/daemon.err" &
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

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
"$YVEX_OPENAI_BIN" --host 127.0.0.1 --port "$port" \
    --yvex-socket "$socket" --timeout-ms 3600000 \
    >"$root/gateway.out" 2>"$root/gateway.err" &
gateway_pid=$!
base="http://127.0.0.1:$port"
attempt=0
while test "$attempt" -lt 100; do
    curl --fail-with-body -sS "$base/health" >"$root/health.json" 2>/dev/null && break
    kill -0 "$gateway_pid" 2>/dev/null || break
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

response_body="{\"model\":\"$model\",\"input\":\"Reply briefly.\",\"temperature\":0,\"max_output_tokens\":2,\"store\":false}"
curl --fail-with-body -sS -H 'Content-Type: application/json' "$base/v1/responses" \
    -d "$response_body" >"$root/response.json"
response_stream="{\"model\":\"$model\",\"input\":\"Reply briefly.\",\"temperature\":0,\"max_output_tokens\":2,\"stream\":true,\"store\":false}"
curl --fail-with-body -sS -N -H 'Content-Type: application/json' "$base/v1/responses" \
    -d "$response_stream" >"$root/response.sse"

tools='[{"type":"function","function":{"name":"get_match_context","description":"Return one match fixture","parameters":{"type":"object","properties":{"match_id":{"type":"string"}},"required":["match_id"],"additionalProperties":false},"strict":false}}]'
tool_choice='{"type":"function","function":{"name":"get_match_context"}}'
curl --fail-with-body -sS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Use get_match_context for match m1.\"}],\"temperature\":0,\"max_completion_tokens\":64,\"tools\":$tools,\"tool_choice\":$tool_choice,\"parallel_tool_calls\":false}" \
    >"$root/tool.json"
python3 - "$root/tool.json" "$root/tool-final-request.json" "$model" <<'PY'
import json, sys
tool = json.load(open(sys.argv[1]))
call = tool['choices'][0]['message']['tool_calls'][0]
request = {
    'model': sys.argv[3],
    'messages': [
        {'role': 'user', 'content': 'Use get_match_context for match m1.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [call]},
        {'role': 'tool', 'tool_call_id': call['id'],
         'content': '{"match_id":"m1","surface":"hard"}'},
    ],
    'temperature': 0,
    'max_completion_tokens': 4,
}
json.dump(request, open(sys.argv[2], 'w'), separators=(',', ':'))
PY
curl --fail-with-body -sS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    --data-binary "@$root/tool-final-request.json" >"$root/tool-final.json"

python3 - "$root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
health = json.load(open(root / 'health.json'))
assert health['gateway'] == 'ready' and health['yvexd'] == 'ready'
chat = json.load(open(root / 'chat.json'))
assert chat['object'] == 'chat.completion' and chat['usage']['total_tokens'] >= 1
assert chat['choices'][0]['finish_reason'] in ('stop', 'length')
response = json.load(open(root / 'response.json'))
assert response['object'] == 'response' and response['usage']['total_tokens'] >= 1
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
response_events = events(root / 'response.sse')
assert response_events[0]['type'] == 'response.created'
assert response_events[-1]['type'] in ('response.completed', 'response.incomplete')
response_text = ''.join(
    event['delta'] for event in response_events
    if event['type'] == 'response.output_text.delta')
stream_response = response_events[-1]['response']
assert response_text == response['output'][0]['content'][0]['text']
assert stream_response['output'][0]['content'][0]['text'] == response_text
assert stream_response['usage'] == response['usage']
tool = json.load(open(root / 'tool.json'))
call = tool['choices'][0]['message']['tool_calls'][0]
assert call['function']['name'] == 'get_match_context'
json.loads(call['function']['arguments'])
tool_final = json.load(open(root / 'tool-final.json'))
assert tool_final['choices'][0]['message']['content']
assert not tool_final['choices'][0]['message'].get('tool_calls')
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
grep -E '"resident_device_bytes":[1-9][0-9]*' "$root/status.after.json" >/dev/null
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

kill "$gateway_pid"
wait "$gateway_pid"
gateway_pid=
"$YVEX_OPENAI_BIN" --host 127.0.0.1 --port "$port" \
    --yvex-socket "$socket" --timeout-ms 3600000 \
    >"$root/gateway.restart.out" \
    2>"$root/gateway.restart.err" &
gateway_pid=$!
attempt=0
while test "$attempt" -lt 100; do
    curl --fail-with-body -sS "$base/health" >"$root/health.restart.json" 2>/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.05
done
test "$attempt" -lt 100
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime status --json >"$root/status.restart.json"
grep -F '"model_open_count":1' "$root/status.restart.json" >/dev/null
grep -F '"provider":"openai"' "$root/raw.jsonl" >/dev/null
! grep -F 'Reply briefly.' "$root/raw.jsonl" >/dev/null

kill "$gateway_pid"
wait "$gateway_pid"
gateway_pid=
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime stop >/dev/null
wait "$daemon_pid"
daemon_pid=
grep -F '"kind":"runtime.shutdown.complete"' "$root/raw.jsonl" >/dev/null
printf 'test: openai_live model=%s profile=yvex.openai.compat.v1\n' "$model"
