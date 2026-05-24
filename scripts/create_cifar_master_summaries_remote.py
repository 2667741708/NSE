#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create master summaries for split CIFAR Bayes-Unified seed runs on c201."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev


PROJECT = Path("/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409")
ROOT = PROJECT / "out_ultimate" / "bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model"
SUMMARY_RE = re.compile(r"Epoch\s+(\d+)\s+Summary:\s+Acc=([0-9.]+)%\s+\|\s+Best=([0-9.]+)%")
SEED_RE = re.compile(r"seed_(\d+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    summaries = []
    for match in SUMMARY_RE.finditer(text):
        summaries.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    seed_match = SEED_RE.search(str(path))
    seed = int(seed_match.group(1)) if seed_match else -1
    has_epoch_500 = "Epoch 500/500" in text
    final_500 = [x for x in summaries if x[0] == 500]
    last_summary = summaries[-1] if summaries else None
    if final_500:
        _, final_acc, best_acc = final_500[-1]
        complete = has_epoch_500
    else:
        final_acc = None
        best_acc = None
        complete = False
    return {
        "path": path,
        "rel": path.relative_to(path.parents[2]),
        "seed": seed,
        "complete": complete,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "last_epoch": last_summary[0] if last_summary else None,
        "mtime": path.stat().st_mtime,
    }


def fmt_pm(values: list[float]) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f}±{pstdev(values):.2f}"


def condition_sort_key(path: Path) -> tuple:
    name = path.name
    m = re.match(r"c(10|100)_pr(\d+)_nr(\d+)_maxw05", name)
    if not m:
        return (name,)
    ds, pr, nr = m.groups()
    return (int(ds), int(pr), int(nr), name)


def write_master(condition: Path) -> dict | None:
    logs = sorted(condition.glob("*/seed_*/run.log"))
    if not logs:
        return None

    parsed = [parse_log(p) for p in logs]
    complete = [p for p in parsed if p["complete"] and p["final_acc"] is not None]

    selected: dict[int, dict] = {}
    excluded: list[tuple[dict, str]] = []
    for item in parsed:
        if not item["complete"]:
            reason = "Incomplete; no Epoch 500/500 final summary"
            if item["last_epoch"] is not None:
                reason = f"Incomplete; last parsed epoch is {item['last_epoch']}"
            excluded.append((item, reason))
            continue
        old = selected.get(item["seed"])
        if old is None or item["mtime"] > old["mtime"]:
            if old is not None:
                excluded.append((old, "Duplicate complete seed; newer complete run selected"))
            selected[item["seed"]] = item
        else:
            excluded.append((item, "Duplicate complete seed; newer complete run selected"))

    rows = [selected[k] for k in sorted(selected)]
    final_values = [r["final_acc"] for r in rows]
    best_values = [r["best_acc"] for r in rows]
    final_mean = mean(final_values) if final_values else math.nan
    final_std = pstdev(final_values) if len(final_values) > 1 else 0.0
    best_mean = mean(best_values) if best_values else math.nan
    best_std = pstdev(best_values) if len(best_values) > 1 else 0.0

    lines = []
    lines.append(f"# {condition.name} Master Summary")
    lines.append("")
    lines.append("## Integration Rule")
    lines.append("- Only logs that reached `Epoch 500/500` and contain `Epoch 500 Summary` are included.")
    lines.append("- If the same seed appears in multiple folders, the newest complete 500-epoch run is used.")
    lines.append("- Incomplete or duplicate runs are listed under `Excluded Runs` and are not used for mean/std.")
    lines.append("- Original experiment folders and logs are not modified.")
    lines.append("")
    lines.append("## Included Runs")
    lines.append("| Seed | Source log | Final Acc (%) | Best Acc (%) |")
    lines.append("|---:|---|---:|---:|")
    for row in rows:
        rel = row["path"].relative_to(condition)
        lines.append(f"| {row['seed']} | {rel} | {row['final_acc']:.2f} | {row['best_acc']:.2f} |")
    lines.append("")
    lines.append("## Integrated Result")
    lines.append(f"- Final Acc mean: {final_mean:.2f}" if final_values else "- Final Acc mean: --")
    lines.append(f"- Final Acc std: {final_std:.2f}" if final_values else "- Final Acc std: --")
    lines.append(f"- Best Acc mean: {best_mean:.2f}" if best_values else "- Best Acc mean: --")
    lines.append(f"- Best Acc std: {best_std:.2f}" if best_values else "- Best Acc std: --")
    lines.append(f"- Seed count: {len(rows)}")
    lines.append(f"- Paper-ready final result: {fmt_pm(final_values)}")
    lines.append(f"- Paper-ready best result: {fmt_pm(best_values)}")
    lines.append("")
    lines.append("## Excluded Runs")
    if excluded:
        lines.append("| Seed | Source log | Reason |")
        lines.append("|---:|---|---|")
        for item, reason in excluded:
            rel = item["path"].relative_to(condition)
            lines.append(f"| {item['seed']} | {rel} | {reason} |")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Raw Final Summary Lines")
    lines.append("```text")
    for row in rows:
        rel = row["path"].relative_to(condition)
        lines.append(f"{rel}: Epoch 500 Summary: Acc={row['final_acc']:.2f}% | Best={row['best_acc']:.2f}%")
    lines.append("```")
    lines.append("")
    lines.append("## Generated At")
    lines.append(f"- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai")
    lines.append("")
    (condition / "master.txt").write_text("\n".join(lines), encoding="utf-8")

    return {
        "condition": condition.name,
        "seed_count": len(rows),
        "final": fmt_pm(final_values),
        "best": fmt_pm(best_values),
        "final_mean": final_mean,
        "final_std": final_std,
        "best_mean": best_mean,
        "best_std": best_std,
        "included": rows,
        "excluded": excluded,
    }


def main() -> None:
    conditions = sorted(
        [p for p in ROOT.iterdir() if p.is_dir() and (p.name.startswith("c10_") or p.name.startswith("c100_"))],
        key=condition_sort_key,
    )
    results = []
    for condition in conditions:
        result = write_master(condition)
        if result is not None:
            results.append(result)

    summary_lines = []
    summary_lines.append("# CIFAR Main Result Master Summary")
    summary_lines.append("")
    summary_lines.append("| Condition | Seeds | Final Acc | Best Acc |")
    summary_lines.append("|---|---:|---:|---:|")
    for result in results:
        summary_lines.append(
            f"| {result['condition']} | {result['seed_count']} | {result['final']} | {result['best']} |"
        )
    summary_lines.append("")
    summary_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai")
    (ROOT / "master_c10_c100_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(ROOT / "master_c10_c100_summary.txt")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
