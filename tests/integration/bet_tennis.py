"""Unchanged bet-tennis OpenAICompatibleProvider acceptance against YVEX."""

from __future__ import annotations

import asyncio
import json
import os

from tqa.foundation_provider.contracts import FoundationProviderConfig
from tqa.foundation_provider.openai_compatible import OpenAICompatibleFoundationProvider


async def main() -> None:
    provider = OpenAICompatibleFoundationProvider(
        FoundationProviderConfig(
            base_url=os.environ["YVEX_OPENAI_BASE_URL"],
            selected_model_id="deepseek-v4-flash",
            request_timeout_seconds=10,
            stream_idle_timeout_seconds=10,
        ),
        None,  # Authentication mode NONE never consults the secret store.
    )
    models, _ = await provider.models()
    assert models[0]["id"] == "deepseek-v4-flash"

    grounded = await provider.chat(
        messages=[
            {"role": "system", "content": "Use grounded facts and include DATA_UNAVAILABLE."},
            {"role": "user", "content": "The data source is unavailable."},
        ], model="deepseek-v4-flash", temperature=0, max_tokens=32,
    )
    assert "DATA_UNAVAILABLE" in grounded.content

    structured = await provider.chat(
        messages=[
            {"role": "system", "content": "Return one JSON object only with exactly these keys: status, operation_mode, real_data."},
            {"role": "user", "content": "Return the observe state."},
        ], model="deepseek-v4-flash", temperature=0, max_tokens=32,
        response_format={"type": "json_object"},
    )
    assert json.loads(structured.content) == {
        "status": "ok", "operation_mode": "observe", "real_data": False,
    }

    tools = [{"type": "function", "function": {
        "name": "query_match_context", "description": "Read one match fixture",
        "parameters": {"type": "object", "properties": {
            "match_id": {"type": "string"}}, "required": ["match_id"]},
    }}]
    selection = await provider.chat(
        messages=[{"role": "user", "content": "Read match m1."}],
        model="deepseek-v4-flash", temperature=0, max_tokens=32,
        tools=tools, tool_choice={"type": "function", "function": {
            "name": "query_match_context"}},
    )
    call = selection.tool_calls[0]
    assert call["function"]["name"] == "query_match_context"
    assert json.loads(call["function"]["arguments"]) == {"match_id": "m1"}

    streamed = []
    async for event in provider.stream_chat(
        messages=[
            {"role": "system", "content": "Reply with STREAM_OK."},
            {"role": "user", "content": "Confirm real SSE streaming."},
        ], model="deepseek-v4-flash", max_tokens=32,
    ):
        if event["type"] == "delta" and event["content"]:
            streamed.append(event["content"])
    assert streamed == ["STREAM_", "OK"]

    cancelled = provider.stream_chat(
        messages=[{"role": "user", "content": "Reply with STREAM_OK."}],
        model="deepseek-v4-flash", max_tokens=32,
    )
    assert (await anext(cancelled))["type"] == "delta"
    await cancelled.aclose()
    models_after, _ = await provider.models()
    assert models_after[0]["id"] == "deepseek-v4-flash"

    print(json.dumps({
        "provider": "tqa.foundation_provider.openai_compatible.OpenAICompatibleFoundationProvider",
        "grounded_chat": True,
        "structured_json": True,
        "tool_selection": True,
        "streaming_sse": True,
        "client_cancellation": True,
        "source_modified": False,
    }, sort_keys=True))


asyncio.run(main())
