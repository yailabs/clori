#!/bin/sh
# Exercises production HTTP/SSE translation over the real local protocol v11 codec.
set -eu

YVEX_OPENAI_ADAPTER=${YVEX_OPENAI_ADAPTER:-build/tests/openai_adapter}
YVEX_OPENAI_HOST=${YVEX_OPENAI_HOST:-build/tests/openai_host}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-openai-integration.XXXXXX")
socket=$root/yvexd.sock
host_pid=
gateway_pid=
cleanup()
{
    status=$?
    trap - EXIT HUP INT TERM
    test -z "$gateway_pid" || kill "$gateway_pid" 2>/dev/null || true
    test -z "$host_pid" || kill "$host_pid" 2>/dev/null || true
    test -z "$gateway_pid" || wait "$gateway_pid" 2>/dev/null || true
    test -z "$host_pid" || wait "$host_pid" 2>/dev/null || true
    if test "$status" -ne 0; then
        test ! -f "$root/gateway.err" || cat "$root/gateway.err" >&2
        test ! -f "$root/gateway.out" || cat "$root/gateway.out" >&2
        test ! -f "$root/response-tool.json" || cat "$root/response-tool.json" >&2
    fi
    yvex_test_cleanup "$root" || test "$status" -ne 0
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
"$YVEX_OPENAI_HOST" "$socket" >"$root/host.out" 2>"$root/host.err" &
host_pid=$!
attempt=0
while test "$attempt" -lt 100; do
    test -S "$socket" && break
    kill -0 "$host_pid" 2>/dev/null || exit 1
    attempt=$((attempt + 1))
    sleep 0.02
done
test -S "$socket"

"$YVEX_OPENAI_ADAPTER" --host 127.0.0.1 --port "$port" --timeout-ms 500 \
    --yvex-socket "$socket" \
    >"$root/gateway.out" 2>"$root/gateway.err" &
gateway_pid=$!
base=http://127.0.0.1:$port
attempt=0
while test "$attempt" -lt 100; do
    if curl -fsS "$base/health" >"$root/health.json" 2>/dev/null; then break; fi
    kill -0 "$gateway_pid" 2>/dev/null || exit 1
    attempt=$((attempt + 1))
    sleep 0.02
done
test "$attempt" -lt 100

curl -fsS "$base/v1/models" >"$root/models.json"
curl -fsS "$base/v1/models/deepseek4-v4-flash-dspark" >"$root/model.json"
curl -fsS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"Hello"}],"temperature":0}' \
    >"$root/chat.json"
curl -fsS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"JSON"}],"temperature":0,"response_format":{"type":"json_object"}}' \
    >"$root/chat-json.json"
curl -fsS -N -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"Hello"}],"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
    >"$root/chat.sse"
curl -fsS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"Think"}],"temperature":0,"reasoning_effort":"high"}' \
    >"$root/chat-reasoning.json"
curl -fsS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"Think max"}],"temperature":0,"reasoning_effort":"max"}' \
    >"$root/chat-reasoning-max.json"
curl -fsS -N -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"Think stream"}],"temperature":0,"reasoning_effort":"high","stream":true,"stream_options":{"include_usage":true}}' \
    >"$root/chat-reasoning.sse"

tools='[{"type":"function","function":{"name":"get_match_context","description":"Get match context","parameters":{"type":"object","properties":{"match_id":{"type":"string"}},"required":["match_id"]},"strict":false}}]'
response_tools='[{"type":"function","name":"get_match_context","description":"Get match context","parameters":{"type":"object","properties":{"match_id":{"type":"string"}},"required":["match_id"]},"strict":false}]'
response_tool_choice='{"type":"function","name":"get_match_context"}'
curl -fsS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d "{\"model\":\"deepseek4-v4-flash-dspark\",\"messages\":[{\"role\":\"user\",\"content\":\"Load m1\"}],\"temperature\":0,\"tools\":$tools,\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}" \
    >"$root/tool.json"
curl -fsS -N -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d "{\"model\":\"deepseek4-v4-flash-dspark\",\"messages\":[{\"role\":\"user\",\"content\":\"Load m1\"}],\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true},\"tools\":$tools,\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}" \
    >"$root/tool.sse"
curl -fsS -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d "{\"model\":\"deepseek4-v4-flash-dspark\",\"messages\":[{\"role\":\"user\",\"content\":\"MULTI_TOOL\"}],\"temperature\":0,\"tools\":$tools,\"tool_choice\":\"auto\",\"parallel_tool_calls\":true}" \
    >"$root/multi-tool.json"

curl --fail-with-body -sS -H 'Content-Type: application/json' "$base/v1/responses" \
    -d "{\"model\":\"deepseek4-v4-flash-dspark\",\"input\":\"Load m1\",\"temperature\":0,\"reasoning_effort\":\"high\",\"tools\":$response_tools,\"tool_choice\":$response_tool_choice,\"parallel_tool_calls\":false,\"store\":false}" \
    >"$root/response-tool.json"
response_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$root/response-tool.json")
call_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"][0]["call_id"])' "$root/response-tool.json")
curl -fsS -H 'Content-Type: application/json' "$base/v1/responses" \
    -d "{\"model\":\"deepseek4-v4-flash-dspark\",\"previous_response_id\":\"$response_id\",\"input\":[{\"type\":\"function_call_output\",\"call_id\":\"$call_id\",\"output\":\"{\\\"score\\\":2}\"}],\"temperature\":0,\"reasoning_effort\":\"high\",\"store\":false}" \
    >"$root/response-final.json"
consumed_status=$(curl -sS -o "$root/consumed-response.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' "$base/v1/responses" \
    -d "{\"model\":\"deepseek4-v4-flash-dspark\",\"previous_response_id\":\"$response_id\",\"input\":\"branch\",\"temperature\":0,\"store\":false}")
test "$consumed_status" = 409
curl -fsS -N -H 'Content-Type: application/json' "$base/v1/responses" \
    -d '{"model":"deepseek4-v4-flash-dspark","input":"Hello","temperature":0,"stream":true,"store":false}' \
    >"$root/responses.sse"
curl -fsS -H 'Content-Type: application/json' "$base/v1/responses" \
    -d '{"model":"deepseek4-v4-flash-dspark","input":"Think","temperature":0,"reasoning_effort":"high","store":false}' \
    >"$root/response-reasoning.json"
curl -fsS -N -H 'Content-Type: application/json' "$base/v1/responses" \
    -d '{"model":"deepseek4-v4-flash-dspark","input":"Think stream","temperature":0,"reasoning_effort":"high","stream":true,"store":false}' \
    >"$root/response-reasoning.sse"
curl -fsS -N -H 'Content-Type: application/json' "$base/v1/responses" \
    -d "{\"model\":\"deepseek4-v4-flash-dspark\",\"input\":\"Load m1\",\"temperature\":0,\"stream\":true,\"tools\":$response_tools,\"tool_choice\":$response_tool_choice,\"parallel_tool_calls\":false,\"store\":false}" \
    >"$root/responses-tool.sse"

status=$(curl -sS -o "$root/refusal.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"x"}],"n":2}')
test "$status" = 422
queue_status=$(curl -sS -o "$root/queue-full.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"QUEUE_FULL"}],"temperature":0}')
test "$queue_status" = 429
timeout_status=$(curl -sS -o "$root/timeout.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"TIMEOUT"}],"temperature":0}')
test "$timeout_status" = 504

# A vanished HTTP consumer must trigger the typed daemon cancellation path
# before the gateway accepts another request.
if curl --max-time 0.2 -fsS -N -H 'Content-Type: application/json' \
    "$base/v1/chat/completions" \
    -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"REASONING_DISCONNECT"}],"temperature":0,"reasoning_effort":"high","stream":true}' \
    >"$root/disconnect.sse" 2>"$root/disconnect.err"; then
    echo 'disconnect fixture unexpectedly completed' >&2
    exit 1
fi
attempt=0
while test "$attempt" -lt 100; do
    grep -q '^generation.cancel ' "$root/host.err" && break
    attempt=$((attempt + 1))
    sleep 0.02
done
test "$attempt" -lt 100
grep -F '"reasoning_content":"reasoning before disconnect"' \
    "$root/disconnect.sse" >/dev/null
! grep -F '"content":"hello from yvex"' "$root/disconnect.sse" >/dev/null
curl -fsS "$base/health" >"$root/health-after-disconnect.json"

python3 - "$root" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
health=json.load(open(root/'health.json'))
assert health == {'status':'ok','adapter':'ready','server':'ready','profile':'yvex.openai.compat.v2'}
assert json.load(open(root/'health-after-disconnect.json')) == health
models=json.load(open(root/'models.json'))
assert models['object']=='list' and models['data'][0]['id']=='deepseek4-v4-flash-dspark'
assert json.load(open(root/'model.json'))['id']=='deepseek4-v4-flash-dspark'
chat=json.load(open(root/'chat.json'))
assert chat['choices'][0]['message']['content']=='hello from yvex'
assert chat['choices'][0]['finish_reason']=='stop'
assert chat['usage']['prompt_tokens']==5 and chat['usage']['completion_tokens']==3
assert chat['usage']['total_tokens']==8
assert chat['usage']['completion_tokens_details']['reasoning_tokens']==0
assert chat['yvex_completion_metrics']['final_tokens']==3
assert chat['choices'][0]['message']['reasoning_content']==''
assert json.load(open(root/'chat-json.json'))['choices'][0]['message']['content']=='{"ok":true}'
reasoning=json.load(open(root/'chat-reasoning.json'))
assert reasoning['choices'][0]['message']['reasoning_content']=='explicit model reasoning'
assert reasoning['choices'][0]['message']['content']=='hello from yvex'
assert reasoning['usage']['completion_tokens_details']['reasoning_tokens']==2
assert reasoning['yvex_completion_metrics']=={
    'reasoning_tokens':2,'final_tokens':1,'reasoning_tokens_per_second':8,
    'final_tokens_per_second':4,'total_tokens_per_second':6,
    'time_to_first_reasoning_token':2.5,'time_to_first_final_token':2.75,
    'reasoning_seconds':0.25,'final_seconds':0.25,
    'total_completion_seconds':0.5}
assert json.load(open(root/'chat-reasoning-max.json'))['choices'][0]['message']['reasoning_content']=='explicit model reasoning'
tool=json.load(open(root/'tool.json'))
call=tool['choices'][0]['message']['tool_calls'][0]
assert tool['choices'][0]['finish_reason']=='tool_calls'
assert call['id']=='call_fixture_1' and call['function']['name']=='get_match_context'
assert json.loads(call['function']['arguments'])=={'match_id':'m1'}
multi=json.load(open(root/'multi-tool.json'))
assert [c['id'] for c in multi['choices'][0]['message']['tool_calls']]==[
    'call_fixture_1','call_fixture_2']
assert [json.loads(c['function']['arguments'])['match_id']
        for c in multi['choices'][0]['message']['tool_calls']]==['m1','m2']
first=json.load(open(root/'response-tool.json'))
assert first['output'][0]['type']=='function_call'
assert first['reasoning_content']=='explicit model reasoning'
final=json.load(open(root/'response-final.json'))
assert final['output'][0]['content'][0]['text']=='Reasoning continuity accepted.'
assert final['reasoning_content']=='explicit model reasoning'
response_reasoning=json.load(open(root/'response-reasoning.json'))
assert response_reasoning['reasoning_content']=='explicit model reasoning'
assert response_reasoning['yvex_completion_metrics']['reasoning_tokens']==2
def records(path):
    out=[]
    event=None
    for line in path.read_text().splitlines():
        if line.startswith('event: '): event=line[7:]
        if line.startswith('data: ') and line != 'data: [DONE]':
            item=json.loads(line[6:])
            if event is not None: assert event == item.get('type')
            out.append(item)
            event=None
    return out
chat_events=records(root/'chat.sse')
assert chat_events[0]['choices'][0]['delta']['role']=='assistant'
assert ''.join(e['choices'][0]['delta'].get('content','')
               for e in chat_events if e.get('choices'))=='hello from yvex'
assert chat_events[-1]['choices']==[] and chat_events[-1]['usage']['total_tokens']==8
assert (root/'chat.sse').read_text().rstrip().endswith('data: [DONE]')
chat_reasoning_events=records(root/'chat-reasoning.sse')
assert ''.join(e['choices'][0]['delta'].get('reasoning_content','')
               for e in chat_reasoning_events if e.get('choices'))=='explicit model reasoning'
assert ''.join(e['choices'][0]['delta'].get('content','')
               for e in chat_reasoning_events if e.get('choices'))=='hello from yvex'
assert chat_reasoning_events[-1]['yvex_completion_metrics']['reasoning_tokens']==2
chat_tool_events=records(root/'tool.sse')
tool_delta=next(e for e in chat_tool_events
                if e.get('choices') and e['choices'][0]['delta'].get('tool_calls'))
stream_call=tool_delta['choices'][0]['delta']['tool_calls'][0]
assert stream_call['id']=='call_fixture_1'
assert stream_call['function']['name']=='get_match_context'
assert json.loads(stream_call['function']['arguments'])=={'match_id':'m1'}
assert next(e for e in chat_tool_events
            if e.get('choices') and e['choices'][0]['finish_reason']=='tool_calls')
response_events=records(root/'responses.sse')
assert [e['type'] for e in response_events] == [
    'response.created', 'response.output_item.added',
    'response.content_part.added', 'response.output_text.delta',
    'response.output_text.done', 'response.content_part.done',
    'response.output_item.done', 'response.completed']
assert [e['sequence_number'] for e in response_events] == list(range(8))
assert response_events[3]['delta']=='hello from yvex'
response_reasoning_events=records(root/'response-reasoning.sse')
assert [e['type'] for e in response_reasoning_events] == [
    'response.created', 'response.reasoning_content.delta',
    'response.output_item.added', 'response.content_part.added',
    'response.output_text.delta', 'response.reasoning_content.done',
    'response.output_text.done', 'response.content_part.done',
    'response.output_item.done', 'response.completed']
assert response_reasoning_events[1]['delta']=='explicit model reasoning'
assert response_reasoning_events[5]['reasoning_content']=='explicit model reasoning'
response_tool_events=records(root/'responses-tool.sse')
assert [e['type'] for e in response_tool_events] == [
    'response.created', 'response.output_item.added',
    'response.function_call_arguments.delta',
    'response.function_call_arguments.done',
    'response.output_item.done', 'response.completed']
assert [e['sequence_number'] for e in response_tool_events] == list(range(6))
assert json.loads(response_tool_events[3]['arguments'])=={'match_id':'m1'}
error=json.load(open(root/'refusal.json'))
assert error['error']['type']=='unsupported_parameter'
consumed=json.load(open(root/'consumed-response.json'))
assert consumed['error']['type']=='incompatible_state'
queue=json.load(open(root/'queue-full.json'))
assert queue['error']['type']=='rate_limit_error'
assert queue['error']['code']=='queue_full'
timeout=json.load(open(root/'timeout.json'))
assert timeout['error']['type']=='server_error'
assert timeout['error']['code']=='gateway_timeout'
PY

kill "$gateway_pid"
wait "$gateway_pid"
gateway_pid=
created=$(grep -c '^session.new ' "$root/host.err" || true)
closed=$(grep -c '^session.close ' "$root/host.err" || true)
test "$created" -gt 0
test "$created" = "$closed"

echo 'OpenAI adapter integration: protocol-v11 Chat/Responses/SSE/tool/state/cleanup/refusal passed'
