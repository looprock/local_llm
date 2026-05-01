"""Strip the spurious KV weights from a Gemma-4 MLX safetensors upload.

Current MLX Gemma-4 quants on HuggingFace serialize k_proj/v_proj/k_norm
weights for KV-shared layers, even though the gemma4_text.py model class
does not expect them. This script reads the downloaded safetensors shards,
drops the offending keys, and writes a fixed copy to a local directory that
mlx_lm.server can load via --model <path>.

Usage:
    python strip_gemma4_kv.py <hf_model_id> [output_dir]

Examples:
    python strip_gemma4_kv.py lmstudio-community/gemma-4-E4B-it-MLX-8bit
    python strip_gemma4_kv.py mlx-community/gemma-4-e4b-8bit /tmp/fixed-base
"""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
DEFAULT_OUT_BASE = Path.home() / ".cache" / "mlx-models"


def find_snapshot(model_id: str) -> Path:
    repo_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        raise SystemExit(f"HF cache not found for {model_id}: {repo_dir}")
    snapshots = [p for p in (repo_dir / "snapshots").iterdir() if p.is_dir()]
    if len(snapshots) != 1:
        raise SystemExit(f"expected exactly one snapshot, found {len(snapshots)}")
    return snapshots[0]


def load_kv_share_split(snapshot: Path) -> int:
    """Return first_kv_shared_layer_idx (layers >= this are KV-shared)."""
    cfg = json.loads((snapshot / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    n_layers = text["num_hidden_layers"]
    n_shared = text.get("num_kv_shared_layers", 0)
    return n_layers - n_shared


def make_dropper(first_shared: int):
    def is_dropped(key: str) -> bool:
        if not key.startswith("language_model.model.layers."):
            return False
        parts = key.split(".")
        if len(parts) < 6:
            return False
        try:
            layer_idx = int(parts[3])
        except ValueError:
            return False
        return (
            layer_idx >= first_shared
            and parts[4] == "self_attn"
            and parts[5] in ("k_norm", "k_proj", "v_proj")
        )

    return is_dropped


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)

    model_id = argv[1]
    snapshot = find_snapshot(model_id)
    out_dir = (
        Path(argv[2])
        if len(argv) >= 3
        else DEFAULT_OUT_BASE / f"{model_id.split('/')[-1]}-fixed"
    )

    first_shared = load_kv_share_split(snapshot)
    print(f"snapshot:       {snapshot}")
    print(f"output:         {out_dir}")
    print(f"first KV-shared layer: {first_shared}")

    is_dropped = make_dropper(first_shared)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_dropped = 0
    shards = sorted(snapshot.glob("model-*-of-*.safetensors"))
    print(f"processing {len(shards)} shards")
    for shard in shards:
        full = mx.load(str(shard), return_metadata=True)
        arrays, metadata = (full[0], full[1]) if isinstance(full, tuple) else (full, {})
        kept = {k: v for k, v in arrays.items() if not is_dropped(k)}
        dropped = len(arrays) - len(kept)
        total_dropped += dropped
        mx.save_safetensors(str(out_dir / shard.name), kept, metadata=metadata)
        print(f"  {shard.name}: kept {len(kept)}, dropped {dropped}")
    print(f"total dropped: {total_dropped}")

    idx_path = snapshot / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        orig = len(idx["weight_map"])
        idx["weight_map"] = {
            k: v for k, v in idx["weight_map"].items() if not is_dropped(k)
        }
        print(f"index weight_map: {orig} -> {len(idx['weight_map'])}")
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))

    skip_names = {s.name for s in shards} | {"model.safetensors.index.json"}
    for f in snapshot.iterdir():
        if f.name in skip_names:
            continue
        shutil.copy(f.resolve(), out_dir / f.name)
        print(f"  copied: {f.name}")

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv)
