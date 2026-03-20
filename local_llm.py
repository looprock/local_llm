#!/usr/bin/env python3
"""MCP server that delegates tasks to a local MLX model via mlx_lm.server."""

import asyncio
import httpx
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

MLX_BASE_URL = "http://localhost:8080/v1"
MODEL_ID = "AlejandroOlmedo/zeta-8bit-mlx"

server = Server("local-llm")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="local_llm_complete",
            description=(
                "Delegate a self-contained task to a fast local LLM running on-device via MLX. "
                "Best for: summarization, simple transforms, boilerplate generation, "
                "draft writing, format conversions, regex/template tasks. "
                "NOT for: complex reasoning, multi-step planning, or anything requiring "
                "full context of the current conversation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The full, self-contained prompt to send to the local model.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 1024,
                        "description": "Maximum number of tokens to generate.",
                    },
                    "temperature": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Sampling temperature (0.0 = deterministic, 1.0 = creative).",
                    },
                },
                "required": ["prompt"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "local_llm_complete":
        raise ValueError(f"Unknown tool: {name}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{MLX_BASE_URL}/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": arguments["prompt"]}],
                    "max_tokens": arguments.get("max_tokens", 1024),
                    "temperature": arguments.get("temperature", 0.2),
                },
            )
            response.raise_for_status()
        except httpx.ConnectError:
            return [types.TextContent(
                type="text",
                text="Error: local MLX model server is not running. Start it with:\n"
                     "mlx_lm.server --model AlejandroOlmedo/zeta-8bit-mlx --port 8080",
            )]

        result = response.json()
        text = result["choices"][0]["message"]["content"]

    return [types.TextContent(type="text", text=text)]


async def main() -> None:
    from mcp.server.stdio import stdio_server
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="local-llm",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
