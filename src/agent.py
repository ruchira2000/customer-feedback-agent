import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
SERVER = Path(__file__).with_name("server.py")
MAX_STEPS = 8

SYSTEM_PROMPT = """You are a customer feedback analyst for a clothing retailer.
Answer questions using ONLY the tools provided. Never guess numbers.
Always quote the exact figures the tools return, including how many reviews they are based on.
If a group has fewer than 150 reviews, warn that the result is based on a small sample.
If the tools cannot answer the question, say so plainly.
When searching for a word or phrase, use exactly the wording the user gave. If you search for something different or broader, say so and explain how it differs. Never present results for a broader keyword as if they answered a question about a more specific phrase."""

gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def to_payload(result):
    """Turn an MCP tool result into a plain dict that Gemini accepts."""
    if result.structured_content is not None:
        return result.structured_content
    texts = [c.text for c in result.content if hasattr(c, "text")]
    return {"text": " ".join(texts)}


async def ask(question):
    trace = []
    async with Client(SERVER) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters_json_schema=t.input_schema,
            )
            for t in mcp_tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            tools=[types.Tool(function_declarations=declarations)],
        )
        contents = [types.Content(role="user", parts=[types.Part(text=question)])]

        for _ in range(MAX_STEPS):
            response = await gemini.aio.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
            if not response.candidates:
                return "No answer returned (the response was empty or blocked).", trace

            model_content = response.candidates[0].content
            calls = [p.function_call for p in (model_content.parts or []) if p.function_call]

            if not calls:
                return response.text, trace

            contents.append(model_content)
            reply_parts = []
            for call in calls:
                args = dict(call.args or {})
                result = await mcp_client.call_tool(call.name, args)
                payload = to_payload(result)
                trace.append({"tool": call.name, "args": args, "result": payload})
                print(f"  [tool call] {call.name} {args}")
                reply_parts.append(
                    types.Part.from_function_response(name=call.name, response=payload)
                )
            contents.append(types.Content(role="user", parts=reply_parts))

    return "Stopped: the agent used too many tool calls without answering.", trace


if __name__ == "__main__":
    question = input("Ask about the reviews: ")
    answer, trace = asyncio.run(ask(question))
    print("\n" + answer)
    print(f"\n(tools used: {len(trace)})")