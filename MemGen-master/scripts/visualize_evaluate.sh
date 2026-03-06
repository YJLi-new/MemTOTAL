#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/visualize_evaluate.sh [PATH]

PATH can be:
  - results/evaluate (default): recursively finds all evaluation runs via */logs/log.txt
  - a group dir (e.g. results/evaluate/gsm8k/ssd): visualizes all runs under it
  - a single run dir (contains logs/log.txt): visualizes just that run

Outputs:
  - Per-run:   <run_dir>/viz/summary.png
  - Per-group: <group_dir>/viz/compare_accuracy.png (when applicable)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INPUT_PATH="${1:-results/evaluate}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${INPUT_PATH}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


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


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")  # matches literal "\boxed{...}"
_BOXED_RE2 = re.compile(r"\\bboxed\{([^}]*)\}")  # tolerate typo
_BOXED_RE3 = re.compile(r"\\\\boxed\{([^}]*)\}")  # tolerate double-escaped text
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*")
_RUN_TIME_RE = re.compile(r"_(\d{8}-\d{6})$")
_LOAD_MODEL_PATH_RE = re.compile(r"\"load_model_path\"\s*:\s*\"([^\"]+)\"")
_TRIGGER_ACTIVE_RE = re.compile(r"\"active\"\s*:\s*(true|false)", re.IGNORECASE)


def _extract_braced(text: str, open_brace_index: int) -> str | None:
    if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None
    depth = 0
    out: list[str] = []
    i = open_brace_index
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
            out.append(ch)
        else:
            if depth >= 1:
                out.append(ch)
        i += 1
    return None


def _extract_boxed(text: str) -> str | None:
    # Prefer brace-balanced parsing for the last occurrence.
    markers = ["\\boxed{", "\\bboxed{", "boxed{"]
    best: tuple[int, str] | None = None
    for marker in markers:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            open_brace_index = idx + len(marker) - 1
            content = _extract_braced(text, open_brace_index)
            if content is not None:
                best = (idx, content)
            start = idx + 1

    if best is not None:
        return best[1]

    # Fallback: regex (handles most cases when braces are simple).
    matches: list[str] = []
    matches.extend(_BOXED_RE.findall(text))
    matches.extend(_BOXED_RE2.findall(text))
    matches.extend(_BOXED_RE3.findall(text))
    if matches:
        return matches[-1]

    # Common failure mode: model outputs "boxed{<num> ..." without a closing brace.
    m = re.search(r"boxed\{\s*([-+]?\d[\d,]*)", text)
    if m:
        return m.group(1)
    return None


def _extract_int(text: str | None) -> int | None:
    if not text:
        return None
    boxed = _extract_boxed(text)
    candidate = boxed if boxed is not None else text
    nums = _NUMBER_RE.findall(candidate)
    if not nums and boxed is not None:
        nums = _NUMBER_RE.findall(text)
    if not nums:
        # Heuristics: look for "answer is/:" patterns, else last number in text.
        m = re.findall(r"(?i)(?:final answer|answer)\s*(?:is|:)\s*([-+]?\d[\d,]*)", text)
        if m:
            nums = [m[-1]]
        else:
            nums = _NUMBER_RE.findall(text)
            if not nums:
                return None
    raw = nums[-1].replace(",", "")
    try:
        return int(raw)
    except Exception:
        return None


def _safe_mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


@dataclass(frozen=True)
class RunMetrics:
    run_dir: Path
    dataset: str | None
    variant: str | None
    load_model_path: str | None
    trigger_active: bool | None
    launcher_label: str
    run_name: str
    run_time: datetime | None

    n_records: int
    n_json_errors: int

    n_gt_parsed: int
    n_pred_parsed: int
    n_both_parsed: int
    n_correct: int

    n_reward_present: int
    n_reward_ones: int
    n_reward_zeros: int
    n_reward_other: int

    mean_reward: float | None
    mean_completion_chars: float | None

    @property
    def parsed_accuracy(self) -> float | None:
        if self.n_both_parsed == 0:
            return None
        return self.n_correct / self.n_both_parsed

    @property
    def reward_accuracy(self) -> float | None:
        if self.n_reward_present == 0:
            return None
        if self.n_reward_other == 0 and (self.n_reward_ones + self.n_reward_zeros) == self.n_reward_present:
            return self.n_reward_ones / self.n_reward_present
        return None

    @property
    def score(self) -> float | None:
        # Prefer the environment-computed reward when available.
        if self.mean_reward is not None:
            return self.mean_reward
        return self.parsed_accuracy

    @property
    def pred_parse_rate(self) -> float | None:
        if self.n_records == 0:
            return None
        return self.n_pred_parsed / self.n_records


def _parse_run_time(run_name: str) -> datetime | None:
    m = _RUN_TIME_RE.search(run_name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_metrics(run_dir: Path) -> RunMetrics:
    run_name = run_dir.name
    run_time = _parse_run_time(run_name)

    dataset = None
    variant = None
    parts = run_dir.parts
    if "results" in parts:
        try:
            i = parts.index("evaluate")
            dataset = parts[i + 1] if i + 1 < len(parts) else None
            variant = parts[i + 2] if i + 2 < len(parts) else None
        except ValueError:
            pass

    # Try to infer how this run was launched.
    launcher_label = run_name
    launcher_json = run_dir / "launcher.json"
    if launcher_json.is_file():
        try:
            payload = json.loads(launcher_json.read_text(encoding="utf-8"))
            script = (payload.get("launcher") or {}).get("script")
            if isinstance(script, str) and script.strip():
                launcher_label = Path(script).name
        except Exception:
            pass

    log_txt = run_dir / "logs" / "log.txt"
    load_model_path = None
    trigger_active = None
    if log_txt.is_file():
        try:
            txt = log_txt.read_text(encoding="utf-8", errors="replace")
            m = _LOAD_MODEL_PATH_RE.search(txt)
            if m:
                load_model_path = m.group(1)
            # Use the *last* "active" occurrence (model.trigger.active is printed later in the log).
            act = _TRIGGER_ACTIVE_RE.findall(txt)
            if act:
                trigger_active = act[-1].lower() == "true"
        except Exception:
            pass

    if launcher_label == run_name:
        # Fallback heuristic for older runs without launcher.json: infer from config.
        lmp = load_model_path or ""
        if trigger_active is True:
            if "20251213-153948" in lmp:
                launcher_label = "eval_gsm8k_then_rocstories_weaver_trigger.sh"
            else:
                launcher_label = "eval_gsm8k_weaver_trigger.sh"
        elif trigger_active is False:
            if "/results/train/rocstories/" in lmp and lmp.endswith("/weaver"):
                launcher_label = "eval_gsm8k_then_rocstories_weaver_only.sh"
            elif "/results/train/gsm8k/" in lmp and lmp.endswith("/weaver"):
                launcher_label = "eval_gsm8k_weaver_only.sh"
            elif lmp.endswith("model.safetensors"):
                launcher_label = "eval.sh"

    answer_path = run_dir / "evaluate" / "answer.json"

    n_records = 0
    n_json_errors = 0
    n_gt_parsed = 0
    n_pred_parsed = 0
    n_both_parsed = 0
    n_correct = 0
    n_reward_present = 0
    n_reward_ones = 0
    n_reward_zeros = 0
    n_reward_other = 0
    rewards: list[float] = []
    completion_chars: list[int] = []

    if answer_path.is_file() and answer_path.stat().st_size > 0:
        for raw_line in answer_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_json_errors += 1
                continue

            n_records += 1
            gt = _extract_int(obj.get("solution"))
            pred = _extract_int(obj.get("completion"))

            if gt is not None:
                n_gt_parsed += 1
            if pred is not None:
                n_pred_parsed += 1
            if gt is not None and pred is not None:
                n_both_parsed += 1
                if gt == pred:
                    n_correct += 1

            m = obj.get("metrics") or {}
            if isinstance(m, dict):
                r = m.get("compute_reward")
                if isinstance(r, (int, float)):
                    rf = float(r)
                    rewards.append(rf)
                    n_reward_present += 1
                    # Most envs use 0/1 rewards; keep counts when they are exact.
                    if rf == 1.0:
                        n_reward_ones += 1
                    elif rf == 0.0:
                        n_reward_zeros += 1
                    else:
                        n_reward_other += 1

            completion = obj.get("completion")
            if isinstance(completion, str):
                completion_chars.append(len(completion))

    mean_reward = _safe_mean(rewards)
    mean_completion_chars = _safe_mean(completion_chars)

    return RunMetrics(
        run_dir=run_dir,
        dataset=dataset,
        variant=variant,
        load_model_path=load_model_path,
        trigger_active=trigger_active,
        launcher_label=launcher_label,
        run_name=run_name,
        run_time=run_time,
        n_records=n_records,
        n_json_errors=n_json_errors,
        n_gt_parsed=n_gt_parsed,
        n_pred_parsed=n_pred_parsed,
        n_both_parsed=n_both_parsed,
        n_correct=n_correct,
        n_reward_present=n_reward_present,
        n_reward_ones=n_reward_ones,
        n_reward_zeros=n_reward_zeros,
        n_reward_other=n_reward_other,
        mean_reward=mean_reward,
        mean_completion_chars=mean_completion_chars,
    )


def _format_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def render_run_summary(metrics: RunMetrics) -> Path:
    out_dir = metrics.run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.png"

    # Figure layout: a text panel on top + two plots below.
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    ax_text = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    title_parts = [p for p in [metrics.dataset, metrics.variant, metrics.run_name] if p]
    fig.suptitle(" / ".join(title_parts), fontsize=12, y=0.98)

    ax_text.axis("off")
    lines = [
        f"run_dir: {metrics.run_dir}",
        f"records: {metrics.n_records} (json_errors: {metrics.n_json_errors})",
        f"gt_parsed: {metrics.n_gt_parsed}/{metrics.n_records}  pred_parsed: {metrics.n_pred_parsed}/{metrics.n_records}",
        f"parsed_accuracy: {_format_pct(metrics.parsed_accuracy)}  pred_parse_rate: {_format_pct(metrics.pred_parse_rate)}",
        f"mean_compute_reward: {metrics.mean_reward if metrics.mean_reward is not None else 'n/a'}"
        + (f"  reward_accuracy: {_format_pct(metrics.reward_accuracy)}" if metrics.reward_accuracy is not None else ""),
        f"mean_completion_chars: {metrics.mean_completion_chars:.1f}"
        if metrics.mean_completion_chars is not None
        else "mean_completion_chars: n/a",
    ]

    if metrics.run_time is not None:
        lines.insert(1, f"run_time: {metrics.run_time.isoformat(sep=' ', timespec='seconds')}")

    if metrics.n_records == 0:
        lines.append("")
        lines.append("No evaluate/answer.json found or it is empty; only metadata was available.")

    ax_text.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    # Bar: prefer reward distribution when available; otherwise fallback to parsed counts.
    if metrics.n_reward_present > 0:
        labels = ["reward=1", "reward=0"]
        values = [metrics.n_reward_ones, metrics.n_reward_zeros]
        if metrics.n_reward_other > 0:
            labels.append("reward=other")
            values.append(metrics.n_reward_other)
        ax_bar.bar(labels, values)
        ax_bar.set_title("compute_reward distribution")
        ax_bar.set_ylabel("count")
    else:
        correct = metrics.n_correct
        incorrect = max(metrics.n_both_parsed - metrics.n_correct, 0)
        unparsed_pred = max(metrics.n_records - metrics.n_pred_parsed, 0)
        ax_bar.bar(["correct", "incorrect", "pred_unparsed"], [correct, incorrect, unparsed_pred])
        ax_bar.set_title("Parsed Answer Match (heuristic)")
        ax_bar.set_ylabel("count")

    # Hist: completion length distribution (chars)
    if metrics.n_records > 0 and metrics.mean_completion_chars is not None:
        # Re-read lengths cheaply only when we actually have data.
        answer_path = metrics.run_dir / "evaluate" / "answer.json"
        lengths: list[int] = []
        try:
            for raw_line in answer_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                c = obj.get("completion")
                if isinstance(c, str):
                    lengths.append(len(c))
        except Exception:
            lengths = []

        if lengths:
            ax_hist.hist(lengths, bins=30)
            ax_hist.set_title("Completion Length (chars)")
            ax_hist.set_xlabel("chars")
            ax_hist.set_ylabel("count")
        else:
            ax_hist.axis("off")
            ax_hist.text(0.5, 0.5, "No completion lengths available.", ha="center", va="center")
    else:
        ax_hist.axis("off")
        ax_hist.text(0.5, 0.5, "No answer.json data.", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_group_compare(group_dir: Path, runs: list[RunMetrics]) -> Path | None:
    # Only compare runs that have a score value.
    rows = [r for r in runs if r.score is not None]
    if len(rows) < 2:
        return None

    rows.sort(key=lambda r: (r.run_time or datetime.min, r.run_name))
    out_dir = group_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_accuracy.png"

    fig = plt.figure(figsize=(max(10, 0.55 * len(rows)), 5))
    ax = fig.add_subplot(1, 1, 1)

    labels = []
    scores = []
    for r in rows:
        label = r.launcher_label
        if sum(rr.launcher_label == label for rr in rows) > 1:
            # Disambiguate when multiple runs share the same launcher.
            if r.run_time is not None:
                label = f"{label}\\n{r.run_time.strftime('%m-%d %H:%M:%S')}"
            else:
                label = f"{label}\\n{r.run_name}"
        labels.append(label)
        scores.append(r.score * 100)

    ax.bar(range(len(scores)), scores)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("score (%)")
    ax.set_title(str(group_dir))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def discover_run_dirs(root: Path) -> list[Path]:
    if (root / "logs" / "log.txt").is_file():
        return [root]

    run_dirs: set[Path] = set()
    for log_path in root.rglob("logs/log.txt"):
        run_dir = log_path.parent.parent
        run_dirs.add(run_dir)
    return sorted(run_dirs)


def group_key(run_dir: Path) -> Path | None:
    # A "group" is dataset/variant folder (e.g. results/evaluate/gsm8k/ssd).
    parts = run_dir.parts
    try:
        i = parts.index("evaluate")
    except ValueError:
        return None
    if i + 2 >= len(parts):
        return None
    return Path(*parts[: i + 3])


def main() -> int:
    if len(sys.argv) < 2:
        print("Missing input path.", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        return 2

    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        print(f"No evaluation runs found under: {root}", file=sys.stderr)
        return 1

    all_metrics: list[RunMetrics] = []
    for run_dir in run_dirs:
        m = compute_metrics(run_dir)
        all_metrics.append(m)
        out_path = render_run_summary(m)
        print(out_path)

    # Group-level comparisons.
    grouped: dict[Path, list[RunMetrics]] = {}
    for m in all_metrics:
        g = group_key(m.run_dir)
        if g is None:
            continue
        grouped.setdefault(g, []).append(m)

    for g, rows in sorted(grouped.items(), key=lambda x: str(x[0])):
        compare = render_group_compare(g, rows)
        if compare is not None:
            print(compare)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
