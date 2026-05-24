#!/usr/bin/env python3
import ast
import csv
import os
import re
import statistics
from pathlib import Path

BASE = Path("/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model")
OUT = BASE / "collected_results_with_paths.csv"

FINAL_RE = re.compile(r"Final Reported \(Final Epoch Acc\):\s*([0-9.]+)%.*?([0-9.]+)%")
BEST_RE = re.compile(r"Final Reported \(Best Acc\):\s*([0-9.]+)%.*?([0-9.]+)%")
IND_FINAL_RE = re.compile(r"Individual Final Epoch Accuracies:\s*(\[[^\]]*\])")
IND_BEST_RE = re.compile(r"Individual Best Accuracies:\s*(\[[^\]]*\])")
RUN_FINAL_RE = re.compile(r"Epoch\s+\d+\s+Summary:\s*Acc=([0-9.]+)%")
RUN_FINISHED_RE = re.compile(r"Run\s+(\d+)\s+Finished\..*?Best Acc:\s*([0-9.]+)%\s*\|\s*Final Acc:\s*([0-9.]+)%")


def parse_percent_list(text, regex):
    match = regex.search(text)
    if not match:
        return []
    try:
        return [float(x.strip("%")) for x in ast.literal_eval(match.group(1))]
    except Exception:
        return []


def std_for(values):
    if len(values) <= 1:
        return ""
    return f"{statistics.stdev(values):.2f}"


rows = []
for path in sorted(BASE.rglob("*")):
    if path.name not in {"master_log.txt", "master.txt"}:
        continue
    rel = path.relative_to(BASE)
    parts = rel.parts
    group = parts[0] if len(parts) > 0 else ""
    run = parts[1] if len(parts) > 1 else ""
    lower_group = group.lower()
    lower_run = run.lower()

    # User request: CIFAR100/CIFAR100H series only record e500 results.
    if (lower_group.startswith("c100") or lower_group.startswith("c100h")) and "e500" not in lower_run:
        continue

    text = path.read_text(encoding="utf-8", errors="ignore")
    final = FINAL_RE.search(text)
    best = BEST_RE.search(text)
    individual_final = parse_percent_list(text, IND_FINAL_RE)
    individual_best = parse_percent_list(text, IND_BEST_RE)
    finished = [(float(best), float(final)) for _, best, final in RUN_FINISHED_RE.findall(text)]
    if not individual_final and finished:
        individual_best = [best for best, _ in finished]
        individual_final = [final for _, final in finished]

    final_mean = final.group(1) if final else (f"{statistics.mean(individual_final):.2f}" if individual_final else "")
    final_std = final.group(2) if final else std_for(individual_final)
    best_mean = best.group(1) if best else (f"{statistics.mean(individual_best):.2f}" if individual_best else "")
    best_std = best.group(2) if best else std_for(individual_best)

    rows.append({
        "group": group,
        "run": run,
        "final_mean": final_mean,
        "final_std": final_std,
        "best_mean": best_mean,
        "best_std": best_std,
        "seed_count": len(individual_final) if individual_final else "",
        "individual_final": ";".join(f"{x:.2f}" for x in individual_final),
        "path": str(path),
        "note": "",
    })

# User-specified Plankton lpi10 convention: seed_1, seed_2, and seed_3_old.
special = BASE / "plankton_lpi10_maxw05_slice2_hl15_lsr00" / "refactored_e100_seed123"
values = []
paths = []
for seed_dir in ["seed_1", "seed_2", "seed_3_old"]:
    run_log = special / seed_dir / "run.log"
    if not run_log.exists():
        continue
    matches = RUN_FINAL_RE.findall(run_log.read_text(encoding="utf-8", errors="ignore"))
    if matches:
        values.append(float(matches[-1]))
        paths.append(str(run_log))

if values:
    rows.append({
        "group": "plankton_lpi10_maxw05_slice2_hl15_lsr00",
        "run": "refactored_e100_seed12_seed3old_manual",
        "final_mean": f"{statistics.mean(values):.2f}",
        "final_std": std_for(values),
        "best_mean": "",
        "best_std": "",
        "seed_count": len(values),
        "individual_final": ";".join(f"{x:.2f}" for x in values),
        "path": " | ".join(paths),
        "note": "manual from seed_1, seed_2, seed_3_old run.log final epoch acc",
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "group",
            "run",
            "final_mean",
            "final_std",
            "best_mean",
            "best_std",
            "seed_count",
            "individual_final",
            "path",
            "note",
        ],
    )
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row["group"].lower(), row["run"].lower(), row["note"])))

print(OUT)
