# Local LLM MCP Server

Runs [AlejandroOlmedo/zeta-8bit-mlx](https://huggingface.co/AlejandroOlmedo/zeta-8bit-mlx) locally via MLX on Apple Silicon and exposes it to Claude Code as an MCP tool (`local_llm_complete`). Claude will automatically delegate self-contained tasks (summarization, boilerplate, format conversions, etc.) to the on-device model.

## Files

| File | Description |
|------|-------------|
| `local_llm.py` | MCP server that wraps the mlx_lm HTTP API |
| `com.dsl.mlx-lm-server.plist` | launchd agent — starts the model server at login, restarts on crash |

## Requirements

- Apple Silicon Mac (MLX requires it)
- Python 3.11+ with `uv`
- A global venv at `~/.venv` (adjust paths if yours differs)

## Setup

### 1. Install dependencies

```bash
uv pip install mlx-lm mcp httpx
```

### 2. Install the MCP server

```bash
mkdir -p ~/.claude/mcp
cp local_llm.py ~/.claude/mcp/local_llm.py
```

### 3. Register with Claude Code (user scope = all projects)

```bash
claude mcp add -s user local-llm -- /Users/$(whoami)/.venv/bin/python /Users/$(whoami)/.claude/mcp/local_llm.py
```

Verify:

```bash
claude mcp list
```

### 4. Install the launchd agent

The plist hardcodes `dsl` as the username. Update it first if your username differs:

```bash
sed "s|/Users/dsl|/Users/$(whoami)|g" com.dsl.mlx-lm-server.plist \
  > ~/Library/LaunchAgents/com.dsl.mlx-lm-server.plist
```

Then load it:

```bash
launchctl load ~/Library/LaunchAgents/com.dsl.mlx-lm-server.plist
```

The model will download from HuggingFace on first run (~a few minutes). Subsequent starts load from cache in seconds.

### 5. Add delegation instructions to CLAUDE.md

Append the following to your project's `CLAUDE.md`:

```markdown
## Local LLM Delegation

A `local_llm_complete` MCP tool is available. Use it proactively to offload self-contained tasks to a fast on-device model (AlejandroOlmedo/zeta-8bit-mlx via MLX).

Use it for:
- Summarizing long files, logs, or text before processing
- Generating repetitive boilerplate (YAML, config stubs, test skeletons)
- Format conversions (JSON↔YAML, naming convention transforms at scale)
- First-draft docstrings, commit messages, or changelog entries
- Simple isolated tasks where output will be reviewed or edited anyway

Do NOT use it for:
- Multi-step reasoning or planning
- Security-sensitive analysis
- Tasks requiring full conversation context
- Anything where correctness is critical without review

Before calling: ensure `mlx_lm.server --model AlejandroOlmedo/zeta-8bit-mlx --port 8080` is running. If the tool returns a "not running" error, surface that message to the user immediately.
```

## Operations

```bash
# Check service status
launchctl list | grep mlx

# Stop the model server
launchctl unload ~/Library/LaunchAgents/com.dsl.mlx-lm-server.plist

# Start the model server
launchctl load ~/Library/LaunchAgents/com.dsl.mlx-lm-server.plist

# Watch logs (stdout and stderr both go here)
tail -f ~/.claude/mcp/mlx-lm-server.log

# Test the HTTP endpoint directly
curl http://localhost:8080/v1/models
```

## Testing the API

The model server exposes an OpenAI-compatible REST API. Use these commands to verify it's working.

**Check the server is up and the model is loaded:**
```bash
curl -s http://localhost:8080/v1/models | jq
```

Expected output:
```json
{
  "object": "list",
  "data": [
    {
      "id": "AlejandroOlmedo/zeta-8bit-mlx",
      "object": "model",
      "created": 1774045181
    }
  ]
}
```

**Send a chat completion:**
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AlejandroOlmedo/zeta-8bit-mlx",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }' | jq '.choices[0].message.content'
```

**Stream a completion:**
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AlejandroOlmedo/zeta-8bit-mlx",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "stream": true
  }'
```

**Test the MCP tool directly from Claude Code:**

In any Claude Code session, ask:
> "Use the local model to summarize this in one sentence: The quick brown fox jumps over the lazy dog."

Claude will call `local_llm_complete` via MCP and return the result.

## How it works

```
Claude Code ──► local_llm_complete (MCP tool)
                      │
                      ▼
            ~/.claude/mcp/local_llm.py   (stdio MCP server)
                      │
                      ▼
            http://localhost:8080/v1     (mlx_lm.server — OpenAI-compatible)
                      │
                      ▼
            AlejandroOlmedo/zeta-8bit-mlx  (MLX model, runs on Apple GPU)
```

## Configuration

| Setting | Default | Location |
|---------|---------|----------|
| Model server port | `8080` | plist + `local_llm.py` `MLX_BASE_URL` |
| Model ID | `AlejandroOlmedo/zeta-8bit-mlx` | plist + `local_llm.py` `MODEL_ID` |
| Max tokens default | `1024` | `local_llm.py` |
| Temperature default | `0.2` | `local_llm.py` |
| Request timeout | `120s` | `local_llm.py` |
| Logs | `~/.claude/mcp/mlx-lm-server.{log,error.log}` | plist |
