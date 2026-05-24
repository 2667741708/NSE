#!/usr/bin/env python3
import csv
import re
from pathlib import Path

ROOT = Path("out_ultimate") / "bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast"
OUT_CSV = ROOT / "summary.csv"

best_re = re.compile(r"Final Reported \(Best Acc\):\s*([0-9.]+)%.*?([0-9.]+)%")
final_re = re.compile(r"Final Reported \(Final Epoch Acc\):\s*([0-9.]+)%.*?([0-9.]+)%")

rows = []
for log_path in sorted(ROOT.glob("*/*/master_log.txt")):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    best = best_re.search(text)
    final = final_re.search(text)
    dataset = log_path.parts[-3]
    ablation = log_path.parts[-2]
    rows.append({
        "dataset": dataset,
        "ablation": ablation,
        "best_mean": best.group(1) if best else "",
        "best_std": best.group(2) if best else "",
        "final_mean": final.group(1) if final else "",
        "final_std": final.group(2) if final else "",
        "log_path": str(log_path),
    })

ROOT.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["dataset", "ablation", "best_mean", "best_std", "final_mean", "final_std", "log_path"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
