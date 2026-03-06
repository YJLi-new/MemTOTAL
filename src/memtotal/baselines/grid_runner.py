from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from memtotal.analysis.reporting import collect_metrics, write_sanity_plot, write_summary_csv
from memtotal.eval.run_eval import main as eval_main
from memtotal.training.run_train import main as train_main
from memtotal.utils.config import load_config
from memtotal.utils.io import ensure_dir, write_json
from memtotal.utils.repro import set_seed


@dataclass(frozen=True)
class GridCell:
    family: str
    mode: str
    backbone: str
    shot: int
    step: int
    template_config: str

    @property
    def slug(self) -> str:
        return (
            f"{self.family}-{self.mode}-"
            f"{self.backbone.replace('.', '').replace('-', '').lower()}-"
            f"{self.shot}shot-{self.step}step"
        )


def _resolve_template(config_path: Path, template_config: str) -> Path:
    return (config_path.parent / template_config).resolve()


def build_grid_plan(config: dict[str, Any]) -> list[GridCell]:
    grid_cfg = config["grid"]
    shots = [int(value) for value in grid_cfg.get("shots", [0])]
    steps = [int(value) for value in grid_cfg.get("steps", [0])]
    cells: list[GridCell] = []
    for variant in grid_cfg.get("variants", []):
        family = str(variant["family"])
        mode = str(variant["mode"])
        backbone = str(variant["backbone"])
        template_config = str(variant["template_config"])
        if family in {"prompting", "meta_prompting"}:
            for shot in shots:
                cells.append(
                    GridCell(
                        family=family,
                        mode=mode,
                        backbone=backbone,
                        shot=shot,
                        step=0,
                        template_config=template_config,
                    )
                )
            continue
        if family == "adapter":
            for shot in shots:
                for step in steps:
                    if shot == 0 and step > 0:
                        continue
                    cells.append(
                        GridCell(
                            family=family,
                            mode=mode,
                            backbone=backbone,
                            shot=shot,
                            step=step,
                            template_config=template_config,
                        )
                    )
            continue
        raise ValueError(f"Unsupported baseline family in grid: {family}")
    return cells


def _write_variant_config(
    *,
    template_path: Path,
    shot: int,
    step: int,
    config_output_path: Path,
) -> dict[str, Any]:
    template = load_config(template_path)
    template.pop("_meta", None)
    template.setdefault("baseline", {})
    template.setdefault("runtime", {})
    template["baseline"]["support_examples"] = shot
    if template["baseline"].get("family") == "adapter":
        template["runtime"]["train_steps"] = step
    config_output_path.write_text(yaml.safe_dump(template, sort_keys=False))
    return template


def _write_adapt_curve(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "baseline_family",
        "baseline_mode",
        "backbone",
        "task_name",
        "smoke_subset",
        "support_examples",
        "train_steps",
        "primary_metric",
        "primary_score",
        "run_dir",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_grid(config: dict[str, Any], *, seed: int, output_dir: Path, dry_run: bool) -> None:
    set_seed(seed)
    config_path = Path(config["_meta"]["config_path"]).resolve()
    generated_config_root = ensure_dir(output_dir / "generated-configs")
    run_root = ensure_dir(output_dir / "runs")
    grid_cells = build_grid_plan(config)
    if dry_run:
        grid_cells = grid_cells[: max(1, min(4, len(grid_cells)))]

    executed_cells: list[dict[str, object]] = []
    train_run_count = 0
    eval_run_count = 0
    for cell in grid_cells:
        template_path = _resolve_template(config_path, cell.template_config)
        generated_config_path = generated_config_root / f"{cell.slug}.yaml"
        _write_variant_config(
            template_path=template_path,
            shot=cell.shot,
            step=cell.step,
            config_output_path=generated_config_path,
        )
        run_dir = run_root / cell.slug
        if cell.family == "adapter":
            train_dir = run_dir / "train"
            eval_dir = run_dir / "eval"
            train_main(
                [
                    "--config",
                    str(generated_config_path),
                    "--seed",
                    str(seed),
                    "--output_dir",
                    str(train_dir),
                ]
                + (["--dry-run"] if dry_run else [])
            )
            train_run_count += 1
            eval_main(
                [
                    "--config",
                    str(generated_config_path),
                    "--seed",
                    str(seed),
                    "--output_dir",
                    str(eval_dir),
                    "--checkpoint",
                    str(train_dir / "checkpoint.pt"),
                ]
                + (["--dry-run"] if dry_run else [])
            )
            eval_run_count += 1
            executed_cells.append(
                {
                    "family": cell.family,
                    "mode": cell.mode,
                    "backbone": cell.backbone,
                    "shot": cell.shot,
                    "step": cell.step,
                    "train_dir": str(train_dir),
                    "eval_dir": str(eval_dir),
                }
            )
        else:
            eval_dir = run_dir / "eval"
            eval_main(
                [
                    "--config",
                    str(generated_config_path),
                    "--seed",
                    str(seed),
                    "--output_dir",
                    str(eval_dir),
                ]
                + (["--dry-run"] if dry_run else [])
            )
            eval_run_count += 1
            executed_cells.append(
                {
                    "family": cell.family,
                    "mode": cell.mode,
                    "backbone": cell.backbone,
                    "shot": cell.shot,
                    "step": cell.step,
                    "eval_dir": str(eval_dir),
                }
            )

    rows = collect_metrics(run_root)
    eval_rows = [row for row in rows if str(row.get("mode")) == "eval_baseline"]
    summary_path = output_dir / "summary.csv"
    plot_path = output_dir / "summary.svg"
    adapt_curve_path = output_dir / "adapt_curve.csv"
    adapt_cost_path = output_dir / "adapt_cost.json"
    grid_plan_path = output_dir / "grid_plan.json"
    write_summary_csv(summary_path, rows)
    write_sanity_plot(plot_path, rows)
    _write_adapt_curve(adapt_curve_path, eval_rows)
    write_json(
        adapt_cost_path,
        {
            "shots": sorted({cell.shot for cell in grid_cells}),
            "steps": sorted({cell.step for cell in grid_cells}),
            "variant_count": len({(cell.family, cell.mode, cell.backbone) for cell in grid_cells}),
            "cell_count": len(grid_cells),
            "train_run_count": train_run_count,
            "eval_run_count": eval_run_count,
            "dry_run": dry_run,
        },
    )
    write_json(
        grid_plan_path,
        {
            "seed": seed,
            "output_dir": str(output_dir.resolve()),
            "cells": executed_cells,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemTOTAL baseline grid runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    run_grid(
        config=config,
        seed=args.seed,
        output_dir=Path(args.output_dir).resolve(),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
