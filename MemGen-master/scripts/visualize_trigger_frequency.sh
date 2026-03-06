#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/visualize_trigger_frequency.sh [GROUP_DIR]

Computes trigger call/invocation frequency for trigger-active evaluation runs under GROUP_DIR
and writes:
  <GROUP_DIR>/viz/trigger_frequency.png

Notes:
  - Requires a working MemGen conda env (torch/transformers/peft) for the compute step.
  - Uses system python3 (needs matplotlib) for plotting.

Env vars:
  MEMGEN_PYTHON_BIN   Path to python in the memgen env (default: $HOME/.miniconda/envs/memgen/bin/python)
  PYTHON_BIN          Plot python executable (default: python3)
  BATCH_SIZE          Batch size for trigger forward (default: 2)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

GROUP_DIR="${1:-results/evaluate/gsm8k/ssd}"
PLOT_PYTHON_BIN="${PYTHON_BIN:-python3}"
MEMGEN_PYTHON_BIN="${MEMGEN_PYTHON_BIN:-$HOME/.miniconda/envs/memgen/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-2}"

if ! command -v "${PLOT_PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PLOT_PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${MEMGEN_PYTHON_BIN}" ]]; then
  echo "MemGen python not found or not executable: ${MEMGEN_PYTHON_BIN}" >&2
  echo "Set MEMGEN_PYTHON_BIN to your memgen env python path." >&2
  exit 1
fi

TMP_JSON="$(mktemp)"
trap 'rm -f "${TMP_JSON}"' EXIT

"${MEMGEN_PYTHON_BIN}" - "${GROUP_DIR}" "${TMP_JSON}" "${BATCH_SIZE}" <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


_LOAD_MODEL_PATH_RE = re.compile(r"\"load_model_path\"\s*:\s*\"([^\"]+)\"")
_TRIGGER_ACTIVE_RE = re.compile(r"\"active\"\s*:\s*(true|false)", re.IGNORECASE)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    launcher_label: str
    load_model_path: str
    trigger_active: bool


def _find_repo_root(path: Path) -> Path:
    for p in [path] + list(path.parents):
        if (p / "main.py").is_file() and (p / "memgen").is_dir():
            return p
    raise RuntimeError(f"Could not locate repo root from: {path}")


def _map_remote_path(repo_root: Path, p: str) -> Path:
    raw = p.strip()
    if not raw:
        raise ValueError("Empty path")

    # Most logs were produced under ".../MemGen-master/...".
    m = re.search(r"/MemGen-master(/.*)$", raw)
    if m:
        return (repo_root / m.group(1).lstrip("/")).resolve()

    # Fallback: locate a "/results/" segment.
    i = raw.find("/results/")
    if i != -1:
        return (repo_root / raw[i + 1 :]).resolve()

    # Already local/relative.
    return Path(raw).expanduser().resolve()


def _infer_launcher_label(run_name: str, load_model_path: str, trigger_active: bool) -> str:
    if trigger_active is True:
        if "20251213-153948" in load_model_path:
            return "eval_gsm8k_then_rocstories_weaver_trigger.sh"
        return "eval_gsm8k_weaver_trigger.sh"
    if trigger_active is False:
        if "/results/train/rocstories/" in load_model_path and load_model_path.endswith("/weaver"):
            return "eval_gsm8k_then_rocstories_weaver_only.sh"
        if "/results/train/gsm8k/" in load_model_path and load_model_path.endswith("/weaver"):
            return "eval_gsm8k_weaver_only.sh"
        if load_model_path.endswith("model.safetensors"):
            return "eval.sh"
    return run_name


def _read_run_info(run_dir: Path) -> RunInfo | None:
    log_txt = run_dir / "logs" / "log.txt"
    if not log_txt.is_file():
        return None

    txt = log_txt.read_text(encoding="utf-8", errors="replace")
    m = _LOAD_MODEL_PATH_RE.search(txt)
    if not m:
        return None
    load_model_path = m.group(1)

    act = _TRIGGER_ACTIVE_RE.findall(txt)
    if not act:
        return None
    trigger_active = act[-1].lower() == "true"

    launcher_label = run_dir.name
    launcher_json = run_dir / "launcher.json"
    if launcher_json.is_file():
        try:
            payload = json.loads(launcher_json.read_text(encoding="utf-8"))
            script = (payload.get("launcher") or {}).get("script")
            if isinstance(script, str) and script.strip():
                launcher_label = Path(script).name
        except Exception:
            pass
    if launcher_label == run_dir.name:
        launcher_label = _infer_launcher_label(run_dir.name, load_model_path, trigger_active)

    return RunInfo(
        run_dir=run_dir,
        launcher_label=launcher_label,
        load_model_path=load_model_path,
        trigger_active=trigger_active,
    )


def _discover_runs(group_dir: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for log_path in group_dir.rglob("logs/log.txt"):
        run_dir = log_path.parent.parent
        info = _read_run_info(run_dir)
        if info is None:
            continue
        if info.trigger_active:
            runs.append(info)
    # Stable ordering
    runs.sort(key=lambda r: (r.launcher_label, r.run_dir.name))
    return runs


def _iter_answer_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _forward_trigger_logits(model: MemGenModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    trigger = model.trigger
    if not trigger.active:
        B, L = input_ids.shape
        logits = torch.zeros(B, L, 2, device=input_ids.device)
        logits[..., 1] = 1.0
        return logits

    position_ids = model._generate_position_ids(attention_mask)
    trigger.model.set_adapter(trigger.adapter_name)

    # Prefer the base *transformer* (e.g. Qwen2Model) to avoid returning all hidden states.
    hidden_states = None
    base = getattr(trigger.model, "base_model", None)  # e.g. LoraModel
    causal_lm = getattr(base, "model", None) if base is not None else None  # e.g. Qwen2ForCausalLM
    transformer = getattr(causal_lm, "model", None) if causal_lm is not None else None  # e.g. Qwen2Model

    if transformer is not None:
        out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = out.last_hidden_state
    else:
        out = trigger.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states[-1]

    logits = trigger.output_layer(hidden_states)
    trigger.model.disable_adapter()
    return logits


def compute_trigger_rates(
    repo_root: Path,
    run: RunInfo,
    *,
    base_model_dir: Path,
    batch_size: int,
) -> dict:
    # Map paths from older absolute logs to local repo paths.
    ckpt_dir = _map_remote_path(repo_root, run.load_model_path)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir} (from {run.load_model_path})")

    base_model_dir = base_model_dir.resolve()
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memgen_config = MemGenConfig.from_pretrained(str(ckpt_dir))

    base_config = AutoConfig.from_pretrained(str(base_model_dir))
    try:
        base_model = AutoModelForCausalLM.from_config(
            base_config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    except ImportError:
        base_model = AutoModelForCausalLM.from_config(
            base_config, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    base_tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))

    model = MemGenModel.from_pretrained(
        str(ckpt_dir),
        config=memgen_config,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        weaver_model=base_model,
        trigger_model=base_model,
    )
    model.eval()
    model.to(device)
    model.to(torch.bfloat16)

    pad_id = model.tokenizer.pad_token_id
    max_aug = int(model.config.max_inference_aug_num)

    # Cache token -> delimiter decision (decode is expensive).
    delimiter_cache: dict[int, bool] = {}
    delimiters = [",", ".", "\n"]

    def ends_with_delimiter(token_id: int) -> bool:
        hit = delimiter_cache.get(token_id)
        if hit is not None:
            return hit
        s = model.tokenizer.decode([token_id], skip_special_tokens=False)
        hit = any(s.endswith(d) for d in delimiters)
        delimiter_cache[token_id] = hit
        return hit

    answer_path = run.run_dir / "evaluate" / "answer.json"
    if not answer_path.is_file() or answer_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing answer.json: {answer_path}")

    total_tokens = 0
    total_calls = 0
    total_invokes = 0
    n_samples = 0

    batch_seqs: list[list[int]] = []
    batch_prompt_lens: list[int] = []
    batch_completion_lens: list[int] = []

    def flush_batch() -> None:
        nonlocal total_tokens, total_calls, total_invokes, n_samples
        if not batch_seqs:
            return

        max_len = max(len(s) for s in batch_seqs)
        input_ids = torch.full((len(batch_seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(batch_seqs):
            input_ids[i, -len(seq) :] = torch.tensor(seq, dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            logits = _forward_trigger_logits(model, input_ids, attention_mask)
            pred = logits.argmax(dim=-1).to("cpu")
            input_ids_cpu = input_ids.to("cpu")

        for b in range(len(batch_seqs)):
            prompt_len = batch_prompt_lens[b]
            completion_len = batch_completion_lens[b]
            seq_len = len(batch_seqs[b])
            offset = max_len - seq_len

            n_samples += 1
            total_tokens += completion_len

            inv_count = 0
            call_count = 0
            aug_count = 0

            # Step i=0 uses last prompt token.
            if completion_len > 0:
                call_count += 1
                pos0 = offset + prompt_len - 1
                if pred[b, pos0].item() == 1:
                    inv_count += 1

            # Steps i>=1 use last token of prefix (completion token i-1).
            for i in range(1, completion_len):
                if aug_count >= max_aug:
                    break
                pos = offset + prompt_len + i - 1
                last_token_id = int(input_ids_cpu[b, pos].item())
                if not ends_with_delimiter(last_token_id):
                    continue
                call_count += 1
                if pred[b, pos].item() == 1:
                    inv_count += 1
                    aug_count += 1

            total_calls += call_count
            total_invokes += inv_count

        batch_seqs.clear()
        batch_prompt_lens.clear()
        batch_completion_lens.clear()

    for obj in _iter_answer_json(answer_path):
        prompt = obj.get("prompt", "")
        completion = obj.get("completion", "")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue

        prompt_ids = model.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = model.tokenizer(completion, add_special_tokens=False)["input_ids"]
        seq = prompt_ids + completion_ids

        if len(prompt_ids) == 0 or len(completion_ids) == 0:
            continue

        batch_seqs.append(seq)
        batch_prompt_lens.append(len(prompt_ids))
        batch_completion_lens.append(len(completion_ids))

        if len(batch_seqs) >= batch_size:
            flush_batch()

    flush_batch()

    call_rate = (total_calls / total_tokens) if total_tokens else 0.0
    invoke_rate = (total_invokes / total_tokens) if total_tokens else 0.0

    # Best-effort cleanup to avoid VRAM spikes when multiple runs are processed.
    del model, base_model, base_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_dir": str(run.run_dir),
        "launcher_label": run.launcher_label,
        "n_samples": n_samples,
        "total_completion_tokens": total_tokens,
        "total_trigger_calls": total_calls,
        "total_trigger_invokes": total_invokes,
        "call_rate": call_rate,
        "invoke_rate": invoke_rate,
    }


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: <group_dir> <out_json> <batch_size>", file=sys.stderr)
        return 2

    group_dir = Path(sys.argv[1]).expanduser().resolve()
    out_json = Path(sys.argv[2]).expanduser().resolve()
    batch_size = int(sys.argv[3])

    if batch_size <= 0:
        print("batch_size must be > 0", file=sys.stderr)
        return 2

    repo_root = _find_repo_root(group_dir)
    runs = _discover_runs(group_dir)
    if not runs:
        print(f"No trigger-active runs found under: {group_dir}", file=sys.stderr)
        return 1

    base_model_dir = repo_root / "data_chache" / "models" / "Qwen2.5-1.5B-Instruct"
    results = []
    for run in runs:
        results.append(
            compute_trigger_rates(
                repo_root,
                run,
                base_model_dir=base_model_dir,
                batch_size=batch_size,
            )
        )

    payload = {"group_dir": str(group_dir), "runs": results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

"${PLOT_PYTHON_BIN}" - "${TMP_JSON}" "${GROUP_DIR}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: matplotlib\n"
            "Install with: python3 -m pip install matplotlib"
        ) from exc


_require_matplotlib()

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: <json_path> <group_dir>", file=sys.stderr)
        return 2

    json_path = Path(sys.argv[1]).expanduser().resolve()
    group_dir = Path(sys.argv[2]).expanduser().resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    if not runs:
        print("No runs in json.", file=sys.stderr)
        return 1

    labels = [r["launcher_label"] for r in runs]
    calls_per_100 = [r["call_rate"] * 100 for r in runs]
    invokes_per_100 = [r["invoke_rate"] * 100 for r in runs]

    out_dir = group_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trigger_frequency.png"

    fig = plt.figure(figsize=(max(8, 0.8 * len(runs)), 5))
    ax = fig.add_subplot(1, 1, 1)

    x = list(range(len(runs)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], calls_per_100, width=width, label="trigger calls / 100 tokens")
    ax.bar([i + width / 2 for i in x], invokes_per_100, width=width, label="trigger invokes / 100 tokens")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("rate (per 100 tokens)")
    ax.set_title(str(group_dir))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
