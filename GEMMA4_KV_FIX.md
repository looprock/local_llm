# Gemma 4 MLX: KV-shared weight workaround

If you try to serve a Gemma 4 model from any current `mlx-community` or `lmstudio-community` MLX upload (verified 2026-05-01 against `mlx-community/gemma-4-e4b-8bit` and `lmstudio-community/gemma-4-E4B-it-MLX-8bit`), `mlx_lm.server` will fail to load the weights and the first `/v1/chat/completions` request will hang forever. This doc explains why, and how to work around it with `strip_gemma4_kv.py`.

## Symptoms

1. `launchctl list com.dsl.mlx-lm-server` shows the agent running and `lsof -iTCP:8080` shows it listening — looks healthy.
2. `curl http://localhost:8080/v1/models` returns 200 with the model listed — also looks healthy.
3. `curl http://localhost:8080/v1/chat/completions ...` **hangs forever** — no response, no timeout from the server side.
4. `~/.claude/mcp/mlx-lm-server.log` contains:
   ```
   ValueError: Received 126 parameters not in model:
     language_model.model.layers.24.self_attn.k_norm.weight,
     language_model.model.layers.24.self_attn.k_proj.{biases,scales,weight},
     language_model.model.layers.24.self_attn.v_proj.{biases,scales,weight},
     ... (through layer 41)
   ```

The exception fires inside `Thread-1 (_generate)`, so the HTTP handler that triggered the lazy weight load never gets notified — the request just dangles. That's why `/v1/models` looks fine (server up) but chat completions hang (worker thread crashes silently).

## Root cause

Gemma 4 uses **cross-layer KV sharing**: the late layers in the stack don't have their own `k_proj` / `v_proj` / `k_norm` weights — they reuse KV from earlier layers to save memory and compute. The model's `config.json` declares this:

```json
"text_config": {
  "num_hidden_layers": 42,
  "num_kv_shared_layers": 18,
  ...
}
```

So layers `42 - 18 = 24` through `41` (= 18 layers) are KV-shared, and `mlx_lm.models.gemma4_text` (in mlx-lm 0.31.3+) does not allocate K/V weights for them.

But the safetensors uploads on HuggingFace **include** those weights anyway. That's 18 layers × 7 weights each (`k_norm.weight`, `k_proj.{biases,scales,weight}`, `v_proj.{biases,scales,weight}`) = **126 extra parameters** that `mlx_lm.models.gemma4_text.Model.load_weights(strict=True)` rejects.

This is an `mlx_lm.convert` bug, not a per-uploader bug — every fresh Gemma 4 quant on HF is currently affected. Until upstream ships a fix to either the converter or the model class (whichever they choose), the weights have to be stripped client-side.

## Fix: `strip_gemma4_kv.py`

The script reads the snapshot from your local HuggingFace cache, drops the offending keys, and writes a clean copy to `~/.cache/mlx-models/<repo-name>-fixed/` that you can point `mlx_lm.server --model` at directly.

It auto-derives `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers` from the model's own `config.json`, so the same script works for any Gemma 4 size variant (E2B / E4B / 26B-A4B / 31B).

### Usage

```bash
# 1. Download the model first (mlx_lm.server will do this automatically on first launch,
#    or you can prefetch with: huggingface-cli download <repo>)
launchctl load ~/Library/LaunchAgents/com.dsl.mlx-lm-server.plist
# wait for download to complete (watch ~/.claude/mcp/mlx-lm-server.log)
launchctl unload ~/Library/LaunchAgents/com.dsl.mlx-lm-server.plist

# 2. Strip the KV-shared weights into a local fixed copy
python strip_gemma4_kv.py lmstudio-community/gemma-4-E4B-it-MLX-8bit
# -> writes ~/.cache/mlx-models/gemma-4-E4B-it-MLX-8bit-fixed/

# 3. Point launchd + the MCP server at the fixed local path
#    Edit com.dsl.mlx-lm-server.plist: change --model arg to:
#      /Users/<you>/.cache/mlx-models/gemma-4-E4B-it-MLX-8bit-fixed
#    Edit local_llm.py: change MODEL_ID to the same path

# 4. Reload
launchctl load ~/Library/LaunchAgents/com.dsl.mlx-lm-server.plist
```

### Verify

Always test with a real chat completion — `/v1/models` returns 200 even when the model can't load, so it is **not** a valid health check:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/Users/<you>/.cache/mlx-models/gemma-4-E4B-it-MLX-8bit-fixed","messages":[{"role":"user","content":"Say hi"}],"max_tokens":20,"stream":false}'
```

A working response will return in a second or two with `"finish_reason": "stop"` and real content. Hangs or 400s mean something is still off.

### Recommended models

For chat use cases, prefer the **instruct-tuned** variants (`-it` in the repo name) — the base/pretrained models won't follow instructions and will generate runaway fake transcripts even though they "work" technically.

| Repo | Quant | Notes |
|---|---|---|
| `lmstudio-community/gemma-4-E4B-it-MLX-8bit` | 8-bit | Recommended; ships `chat_template.jinja` |
| `mlx-community/gemma-4-e4b-it-8bit` | 8-bit | Also fine; same uploader as the broken base model |
| `mlx-community/gemma-4-e4b-it-4bit` | 4-bit | Smaller / faster, slightly lower quality |

All three need the strip step.

## When this workaround can be retired

Check after each `mlx-lm` release after `0.31.3`:

```bash
uv pip install --upgrade mlx-lm
```

If a new release fixes either:
- `mlx_lm.convert` to strip the unused KV weights at quantization time, OR
- `mlx_lm.models.gemma4_text.Model.load_weights` to accept and ignore extras (non-strict load),

then the strip step becomes unnecessary. At that point you can:

1. Delete `~/.cache/mlx-models/gemma-4-E4B-it-MLX-8bit-fixed/`
2. Change the `--model` arg back to the HF repo ID
3. Delete this doc and `strip_gemma4_kv.py`
