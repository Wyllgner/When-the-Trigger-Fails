import ollama
tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object",
      "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }
}]
r = ollama.chat(model="llama3.2:3b",
    messages=[{"role":"user","content":"What's the weather in Recife?"}],
    tools=tools)
print(r["message"].get("tool_calls"))