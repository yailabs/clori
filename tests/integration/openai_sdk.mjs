// Official OpenAI JavaScript SDK acceptance over the bounded YVEX gateway.
const modulePath = `${process.env.YVEX_OPENAI_NODE_MODULE}/index.mjs`;
const { default: OpenAI } = await import(modulePath);
const client = new OpenAI({
  baseURL: `${process.env.YVEX_OPENAI_BASE_URL}/v1`,
  apiKey: "yvex-local",
  timeout: 10000,
});

const models = await client.models.list();
if (models.data[0].id !== "deepseek-v4-flash") throw new Error("model discovery");

const chat = await client.chat.completions.create({
  model: "deepseek-v4-flash",
  messages: [{ role: "user", content: "Hello" }],
  temperature: 0,
});
if (chat.choices[0].message.content !== "hello from yvex") throw new Error("chat");

const chatStream = await client.chat.completions.create({
  model: "deepseek-v4-flash",
  messages: [{ role: "user", content: "Hello" }],
  temperature: 0,
  stream: true,
  stream_options: { include_usage: true },
});
let chatText = "";
for await (const chunk of chatStream) chatText += chunk.choices[0]?.delta?.content ?? "";
if (chatText !== "hello from yvex") throw new Error("chat stream");

const response = await client.responses.create({
  model: "deepseek-v4-flash",
  input: "Hello",
  temperature: 0,
  store: false,
});
if (response.output_text !== "hello from yvex") throw new Error("response");

const responseStream = await client.responses.create({
  model: "deepseek-v4-flash",
  input: "Hello",
  temperature: 0,
  store: false,
  stream: true,
});
let responseText = "";
let completed = false;
for await (const event of responseStream) {
  if (event.type === "response.output_text.delta") responseText += event.delta;
  if (event.type === "response.completed") completed = true;
}
if (responseText !== "hello from yvex" || !completed) throw new Error("response stream");

const tools = [{
  type: "function",
  name: "get_match_context",
  description: "Get deterministic match context",
  parameters: {
    type: "object",
    properties: { match_id: { type: "string" } },
    required: ["match_id"],
  },
  strict: false,
}];
const chatTool = await client.chat.completions.create({
  model: "deepseek-v4-flash",
  messages: [{ role: "user", content: "Load m1" }],
  tools: tools.map(({ name, description, parameters, strict }) => ({
    type: "function", function: { name, description, parameters, strict },
  })),
  tool_choice: "auto",
  parallel_tool_calls: false,
  temperature: 0,
});
const chatCall = chatTool.choices[0].message.tool_calls[0];
const chatFinal = await client.chat.completions.create({
  model: "deepseek-v4-flash",
  messages: [
    { role: "user", content: "Load m1" },
    { role: "assistant", content: null, tool_calls: [chatCall] },
    { role: "tool", tool_call_id: chatCall.id, content: '{"score":2}' },
  ],
  tools: tools.map(({ name, description, parameters, strict }) => ({
    type: "function", function: { name, description, parameters, strict },
  })),
  tool_choice: "auto",
  parallel_tool_calls: false,
  temperature: 0,
});
if (chatFinal.choices[0].message.content !== "Match context accepted.") {
  throw new Error("chat function continuation");
}
const first = await client.responses.create({
  model: "deepseek-v4-flash",
  input: "Load m1",
  tools,
  tool_choice: { type: "function", name: "get_match_context" },
  parallel_tool_calls: false,
  temperature: 0,
  store: false,
});
const call = first.output.find((item) => item.type === "function_call");
if (!call || JSON.parse(call.arguments).match_id !== "m1") throw new Error("function call");
const final = await client.responses.create({
  model: "deepseek-v4-flash",
  previous_response_id: first.id,
  input: [{ type: "function_call_output", call_id: call.call_id, output: '{"score":2}' }],
  temperature: 0,
  store: false,
});
if (final.output_text !== "Match context accepted.") throw new Error("function continuation");

console.log("OpenAI JavaScript SDK 7.1.0: models/chat/Responses/SSE/two function loops passed");
