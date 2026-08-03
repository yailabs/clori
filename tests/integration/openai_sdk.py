"""Official OpenAI Python SDK acceptance over the bounded YVEX gateway."""

from __future__ import annotations

import json
import os

from openai import OpenAI


base_url = os.environ["YVEX_OPENAI_BASE_URL"] + "/v1"
client = OpenAI(base_url=base_url, api_key="yvex-local", timeout=10.0)

models = client.models.list()
assert [model.id for model in models.data] == ["deepseek4-v4-flash-dspark"]

chat = client.chat.completions.create(
    model="deepseek4-v4-flash-dspark",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
)
assert chat.choices[0].message.content == "hello from yvex"
assert chat.usage and chat.usage.total_tokens == 8

chat_stream = client.chat.completions.create(
    model="deepseek4-v4-flash-dspark",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},
)
chat_text = "".join(
    choice.delta.content or ""
    for chunk in chat_stream
    for choice in chunk.choices
)
assert chat_text == "hello from yvex"

response = client.responses.create(
    model="deepseek4-v4-flash-dspark",
    input="Hello",
    temperature=0,
    store=False,
)
assert response.output_text == "hello from yvex"

response_events = list(
    client.responses.create(
        model="deepseek4-v4-flash-dspark",
        input="Hello",
        temperature=0,
        store=False,
        stream=True,
    )
)
assert response_events[0].type == "response.created"
assert response_events[-1].type == "response.completed"
assert "".join(
    event.delta
    for event in response_events
    if event.type == "response.output_text.delta"
) == "hello from yvex"

tools = [{
    "type": "function",
    "function": {
        "name": "get_match_context",
        "description": "Get deterministic match context",
        "parameters": {
            "type": "object",
            "properties": {"match_id": {"type": "string"}},
            "required": ["match_id"],
        },
        "strict": False,
    },
}]
tool_chat = client.chat.completions.create(
    model="deepseek4-v4-flash-dspark",
    messages=[{"role": "user", "content": "Load m1"}],
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=False,
    temperature=0,
)
tool_call = tool_chat.choices[0].message.tool_calls[0]
assert tool_call.function.name == "get_match_context"
assert json.loads(tool_call.function.arguments) == {"match_id": "m1"}
tool_chat_final = client.chat.completions.create(
    model="deepseek4-v4-flash-dspark",
    messages=[
        {"role": "user", "content": "Load m1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": '{"score":2}',
        },
    ],
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=False,
    temperature=0,
)
assert tool_chat_final.choices[0].message.content == "Match context accepted."

tool_response = client.responses.create(
    model="deepseek4-v4-flash-dspark",
    input="Load m1",
    tools=[tool["function"] | {"type": "function"} for tool in tools],
    tool_choice={"type": "function", "name": "get_match_context"},
    parallel_tool_calls=False,
    temperature=0,
    store=False,
)
function_call = next(item for item in tool_response.output
                     if item.type == "function_call")
continued = client.responses.create(
    model="deepseek4-v4-flash-dspark",
    previous_response_id=tool_response.id,
    input=[{
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": '{"score":2}',
    }],
    temperature=0,
    store=False,
)
assert continued.output_text == "Match context accepted."

print("OpenAI Python SDK 2.50.0: models/chat/Responses/SSE/two function loops passed")
