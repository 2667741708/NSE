#!/usr/bin/env python3
import csv
import re
from pathlib import Path

ROOT = Path("/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast")
OUT_CSV = ROOT / "contrast_summary.csv"

RUN_RE = re.compile(r"Run\s+\d+\s+Finished.*?Best Acc:\s*([0-9.]+)%\s*\|\s*Final Acc:\s*([0-9.]+)%")
BEST_RE = re.compile(r"Final Reported \(Best Acc\):\s*([0-9.]+)%\s*±\s*([0-9.]+)%")
FINAL_RE = re.compile(r"Final Reported \(Final Epoch Acc\):\s*([0-9.]+)%\s*±\s*([0-9.]+)%")

rows = []
for log_path in sorted(ROOT.glob("*/*/master_log.txt")):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    runs = RUN_RE.findall(text)
    best = BEST_RE.search(text)
    final = FINAL_RE.search(text)
    setting = log_path.parts[-3]
    variant = log_path.parts[-2]
    rows.append({
        "setting": setting,
        "variant": variant,
        "seeds_completed": len(runs),
        "best_mean": best.group(1) if best else "",
        "best_std": best.group(2) if best else "",
        "final_mean": final.group(1) if final else "",
        "final_std": final.group(2) if final else "",
        "log_path": str(log_path),
        "command_file": str(log_path.parent / "command.txt"),
        "alignment_file": str(log_path.parent / "baseline_alignment.txt"),
    })

ROOT.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
        "setting", "variant", "seeds_completed", "best_mean", "best_std",
        "final_mean", "final_std", "log_path", "command_file", "alignment_file",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
