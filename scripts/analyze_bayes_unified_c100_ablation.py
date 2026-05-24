#!/usr/bin/env python3
import csv
import re
from pathlib import Path


ROOT = Path("/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_ablast/c100_pr005_nr03")
BASELINE_FINAL = 79.17
BASELINE_STD = 0.15
BASELINE_SEEDS = 2
BASELINE_PATH = "/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/refactored_e500_seed23"

RUN_RE = re.compile(r"Run\s+\d+\s+Finished.*?Best Acc:\s*([0-9.]+)%\s*\|\s*Final Acc:\s*([0-9.]+)%")
SUMMARY_BEST_RE = re.compile(r"Final Reported \(Best Acc\):\s*([0-9.]+)%\s*±\s*([0-9.]+)%")
SUMMARY_FINAL_RE = re.compile(r"Final Reported \(Final Epoch Acc\):\s*([0-9.]+)%\s*±\s*([0-9.]+)%")

LABELS = {
    "A01_ablate_topology_daes_use_exp_both": ("K1/K2 topology-DAES -> exp", "Remove topology-aware voting in both propagation passes."),
    "A02_ablate_stage2_topology_daes_use_exp_s2": ("K2 topology-DAES -> exp", "Keep topology-DAES in pass 1, replace pass 2 with exponential kernel."),
    "A03_ablate_stage1_topology_daes_use_exp_s1": ("K1 topology-DAES -> exp", "Replace pass 1 with exponential kernel, keep topology-DAES in pass 2."),
    "A04_ablate_adaptive_ri_use_uniform_ri05": ("adaptive r_i -> 0.5", "Use a constant 0.5 gate instead of sample-wise confidence ratio."),
    "A05_ablate_candidate_prior_unmask_model_evidence": ("w/o candidate prior", "Remove omega mask from model evidence."),
    "A06_ablate_model_view_maxw0_knn_only": ("model cap 0.0", "KNN-only second-stage evidence, no model-view contribution."),
    "A07_stress_model_view_maxw1_full_model_influence": ("model cap 1.0", "Allow full model-view influence after warm-up."),
    "A08_ablate_salvage_training_log_only": ("detect salvage only", "Detect queue-salvaged samples but do not promote them into training."),
    "A09_ablate_queue_stability_short_history5": ("history_len 5", "Use a short temporal queue."),
    "A10_ablate_queue_stability_long_history30": ("history_len 30", "Use a long temporal queue; seed 3 is incomplete."),
}


def fmt(mean, std, seeds):
    if mean == "":
        return "--"
    if seeds <= 1 or std == "":
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def pop_std(values):
    if not values:
        return ""
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def parse_one(exp_dir):
    master = exp_dir / "master_log.txt"
    text = master.read_text(encoding="utf-8", errors="ignore") if master.exists() else ""
    runs = [(float(b), float(f)) for b, f in RUN_RE.findall(text)]
    summary_best = SUMMARY_BEST_RE.search(text)
    summary_final = SUMMARY_FINAL_RE.search(text)

    if summary_final:
        final_mean = float(summary_final.group(1))
        final_std = float(summary_final.group(2))
    elif runs:
        finals = [f for _, f in runs]
        final_mean = sum(finals) / len(finals)
        final_std = pop_std(finals)
    else:
        final_mean = ""
        final_std = ""

    if summary_best:
        best_mean = float(summary_best.group(1))
        best_std = float(summary_best.group(2))
    elif runs:
        bests = [b for b, _ in runs]
        best_mean = sum(bests) / len(bests)
        best_std = pop_std(bests)
    else:
        best_mean = ""
        best_std = ""

    command = (exp_dir / "command.txt").read_text(encoding="utf-8", errors="ignore").strip() if (exp_dir / "command.txt").exists() else ""
    alignment = (exp_dir / "baseline_alignment.txt").read_text(encoding="utf-8", errors="ignore").strip() if (exp_dir / "baseline_alignment.txt").exists() else ""

    name = exp_dir.name
    label, meaning = LABELS.get(name, (name, ""))
    delta = "" if final_mean == "" else final_mean - BASELINE_FINAL
    return {
        "id": name.split("_", 1)[0],
        "name": name,
        "label": label,
        "meaning": meaning,
        "seeds_completed": len(runs),
        "best_mean": best_mean,
        "best_std": best_std,
        "final_mean": final_mean,
        "final_std": final_std,
        "delta_vs_baseline": delta,
        "master_log": str(master),
        "command_file": str(exp_dir / "command.txt"),
        "alignment_file": str(exp_dir / "baseline_alignment.txt"),
        "command": command,
        "alignment": alignment,
    }


def main():
    rows = [parse_one(p) for p in sorted(ROOT.glob("A*")) if p.is_dir()]

    csv_path = ROOT / "c100_pr005_nr03_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id", "name", "label", "meaning", "seeds_completed", "best_mean", "best_std",
            "final_mean", "final_std", "delta_vs_baseline", "master_log", "command_file", "alignment_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    md_path = ROOT / "c100_pr005_nr03_ablation_analysis.md"
    lines = []
    lines.append("# Bayes-Unified CIFAR-100 q=0.05 eta=0.3 Ablation Analysis")
    lines.append("")
    lines.append(f"Baseline final accuracy: **{BASELINE_FINAL:.2f}±{BASELINE_STD:.2f}** over {BASELINE_SEEDS} seeds.")
    lines.append(f"Baseline source: `{BASELINE_PATH}`")
    lines.append("")
    lines.append("All ablation values below use final-epoch accuracy. A10 is summarized from two completed seeds because seed 3 did not finish.")
    lines.append("")
    lines.append("| ID | Ablation | Final Acc. | Delta | Seeds | Log |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in rows:
        final_text = fmt(row["final_mean"], row["final_std"], row["seeds_completed"])
        delta_text = "--" if row["delta_vs_baseline"] == "" else f"{row['delta_vs_baseline']:+.2f}"
        lines.append(f"| {row['id']} | {row['label']} | {final_text} | {delta_text} | {row['seeds_completed']} | `{row['master_log']}` |")

    lines.append("")
    lines.append("## Main Takeaways")
    lines.append("")
    lines.append("1. **Topology-DAES is mainly useful in the first pass.** Replacing both passes with the exponential kernel drops final accuracy to 76.50, and replacing only the first pass gives a similar 76.51. Keeping topology-DAES in pass 1 but using exp in pass 2 is much better at 78.56. This suggests the geometry-aware vote is most critical before Bayesian fusion, where it decides the quality of the internal belief that later stages inherit.")
    lines.append("2. **The current candidate-prior mask is beneficial at eta=0.3.** Removing the omega constraint gives 78.17, about 1.00 point below the baseline. At this noise level, the prior still suppresses enough false candidates to be useful, even though it may become risky when the true class is absent more often.")
    lines.append("3. **Model-view fusion is robust but not the sole source of gain.** KNN-only evidence reaches 78.99 and full model influence reaches 79.13, both close to the baseline. The two-view pathway appears to stabilize the method, but the exact cap is less sensitive than the topology kernel or candidate prior.")
    lines.append("4. **The adaptive r_i formula needs reinterpretation.** Forcing r_i=0.5 reaches 79.14, slightly above the current baseline. This does not mean the idea of reliability gating is wrong; it means the present confidence-ratio implementation may be overreacting to confidence calibration or early model/KNN imbalance. This is a strong clue for a revised r_i design.")
    lines.append("5. **Queue salvage helps, but the effect is moderate.** Disabling promotion of salvaged samples gives 78.67, about 0.50 below baseline. Short history 5 gives 78.97 and long history 30 gives 79.08 over two seeds, so the queue is useful and not extremely sensitive to the horizon around the tested range.")
    lines.append("")
    lines.append("## Paper-facing interpretation")
    lines.append("")
    lines.append("The strongest supported story is that Bayes-Unified's robustness comes from geometry-aware first-pass topology voting plus a constrained Bayesian evidence path. The candidate prior remains useful at eta=0.3, while the model-view and queue modules provide stabilizing secondary gains. The r_i ablation suggests that the current adaptive gate should be described carefully: sample-wise reliability is an important design axis, but this implementation's confidence-ratio gate is not yet strictly better than a constant balanced gate under this condition.")
    lines.append("")
    lines.append("## Command provenance")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['id']} {row['label']}")
        lines.append("")
        lines.append(f"- Command file: `{row['command_file']}`")
        lines.append(f"- Alignment file: `{row['alignment_file']}`")
        lines.append(f"- Master log: `{row['master_log']}`")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
