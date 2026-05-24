# Project Agent Notes

## Source Thread

- Summarized from Codex thread: `codex://threads/019e1a19-99e1-7bb1-bc8e-34953aacef34`
- Local session file inspected: `C:\Users\BILIBILI\.codex\sessions\2026\05\12\rollout-2026-05-12T10-52-25-019e1a19-99e1-7bb1-bc8e-34953aacef34.jsonl`
- Summary date: 2026-05-13

## Communication And Documentation Rules

- The project has an Antigravity-style engineering instruction file:
  `ANTIGRAVITY_SYSTEM_ENGINEERING_RESEARCH_INSTRUCTIONS.md`.
- Preferred technical style: objective, direct, structured Markdown.
- For experiment/code answers, include problem definition, evidence, code path, risk/constraint, and verification method.
- Avoid unsupported claims. When judging experiments, align condition name, command, code branch, and logs.

## Reference Library / 引用核验规则

Canonical local reference root:

```text
D:\文件\论文项目\参考文献列表
```

Use this reference library before changing or judging paper claims in the
Abstract, Introduction, Related Work, experiment protocol, dataset description,
backbone/pretraining description, or baseline-result provenance.

Required lookup order:

1. Read `D:\文件\论文项目\参考文献列表\README.md`.
2. Resolve the citation key through `00_index\bibkey_to_folder.csv`.
3. Inspect the target paper folder, especially `notes.md`, `metadata.md`,
   `bibtex.bib`, and the local PDF.
4. If a claim cannot be verified from the reference library, mark it as
   unverified instead of strengthening it from memory.

Current comparison datasets are limited to:

- CIFAR-10
- CIFAR-100
- CIFAR-100H
- Plankton
- Treeversity
- Benthic

Baseline-result provenance rule:

- Baseline numbers in the manuscript are taken from the PALS paper and its
  public comparison tables.
- Original baseline papers are cited to identify the compared methods and
  support mechanism/background discussion, not to indicate that baseline
  numbers were copied from those papers.
- If future work adds baseline numbers from any non-PALS source, explicitly
  document the source in the manuscript caption or experiment setup and record
  it in the provenance docs.

## Paper Positioning

Current strongest paper framing:

- Core idea: preserve the native weak-supervision source instead of rewriting the candidate/crowd prior.
- Recommended positioning:
  `Topology-Adaptive KNN Propagation with Prior-Constrained Evidence Fusion for Noisy Partial Label Learning`.
- Strong claims:
  - topology-aware KNN propagation is effective for NPLL.
  - candidate-prior-constrained model evidence reduces harmful model feedback.
  - active set plus salvage improves data usage.
- Weaker claims:
  - dual-view model belief is not consistently supported as a primary contribution.
  - `r_i` alone is not a strong standalone performance source because uniform `r_i=0.5` drops little in existing ablations.

Related-work/motivation changes already made in the thread:

- `sec/1_intro.tex`: strengthened VALEN/IDGP/POP motivation and the "protect original weak supervision source" story.
- `sec/2_relatedwork.tex`: added a method-difference table comparing POP, FREDIS, ALIM, PALS, and Bayes-Unified.
- `main.bib`: added `qiao2023idgp`; corrected `xu2022pop` to ICML 2023/PMLR.
- Verified with `main_verify_positioning.pdf`; no undefined citations/references at that time.

Useful sources used for paper positioning:

- VALEN / IDPLL, NeurIPS 2021.
- IDGP, ICLR 2023.
- POP, ICML 2023.

## C201 Remote Project

Primary remote project root:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409
```

Script/result organization root used for curated scripts:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果
```

Use this directory as the main experiment hub for future work. New experiment
scripts, launcher files, execution plans, mapping documents, and result indexes
should be placed here instead of adding more entry points under the older draft
project root.

Important result roots:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_unified_ablation
```

Use `ssh c201` for remote inspection and experiment control.

## Verified Ablation Findings

### A05 Candidate Prior / Model Evidence

Condition checked:

```text
A05_ablate_candidate_prior_unmask_model_evidence
```

Conclusion:

- This condition was a real ablation.
- The active flag was `--ablate_no_candidate_prior`.
- Logs showed `ablate_no_candidate_prior=True`.
- Each seed logged `[Ablation] candidate prior omega removed from model evidence.` for all epochs.

Exact meaning:

- It removes `omega/static_cand_mask` as a multiplicative constraint from the model-evidence branch.
- It does not remove final `_filter_logic()` candidate-set admission constraints.
- The phrase `unmask_model_evidence` means model evidence is no longer masked by the candidate prior before entering the evidence fusion path.

### Invalid Old Unified Ablations

Old unified script issue:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/bayes_unified_unified_ablation.py
```

Invalid conditions in the old script:

- `E5_no_salvage_training`
- `E8_no_reliable_mixup`

Reason:

- `--disable_salvage_training` was parsed but did not prevent salvage samples from being promoted into the reliable/active training pool.
- `--no_reliable_mixup` only changed experiment metadata and did not affect the actual reliable MixUp loss path.

Do not use old E5/E8 results in the paper.

## Audit Ablation Script

Implemented trustworthy audit script:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/bayes_unified_unified_ablation_audit.py
```

Curated copy:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py
```

Key fixed flags:

- `--ablate_no_salvage_training`
- `--ablate_no_reliable_mixup`

Old invalid flag names removed from the audit script:

- `--disable_salvage_training`
- `--no_reliable_mixup`
- `args.no_reliable_mixup`
- `disable_salvage_training`

Audit verification logs use `[AblationCheck]`.

Expected audit validation:

- No salvage training:
  - `detected_salvage > 0`
  - `promoted_salvage=0`
  - `Sampler Salvaged: 0`
- No reliable MixUp:
  - reliable weak/strong branches bypass MixUp and use direct supervised labels.

Smoke status from the thread:

- A5 smoke passed after `detected_salvage=1` and `promoted_salvage=0`.
- A8 smoke passed.
- Formal e500 audit queues were launched afterward.

Curated audit directory:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/
```

Contains:

- `bayes_unified_unified_ablation_audit.py`
- `launch_audit_smoke_then_e500_20260513.sh`
- `launch_audit_e500_gpu0_wait_20260513.sh`
- `launch_audit_e500_gpu1_20260513.sh`
- `AUDIT_ABLATION_EXECUTION_PLAN_20260513.md`
- `ABLATION_CODE_MAPPING_AUDIT_20260513.md`
- `README.md`

Deprecated old entries moved to:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/deprecated_invalid_20260513/
```

Script inventory:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/SCRIPT_INVENTORY_20260513.md
```

## R_i Formula Experiments

Contrast script:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast.py
```

Supported `ri_mode` values in the contrast script:

- `confidence`: simplified max-confidence ratio.
- `entropy_prior`: entropy/agreement-based formula plus adaptive candidate-prior gating.
- `fused_entropy`: simple replacement using entropy of fused KNN/model score.
- `uniform`: `r_i=0.5`.

Important formula-only note:

- `fused_entropy` was introduced as a simple `r_i` replacement baseline.
- It uses KNN/model fused-score entropy to compute `r_i`.
- It does not include the `entropy_prior` adaptive candidate-prior gate.

Smoke status:

- `entropy_prior` and `fused_entropy` both passed 2-epoch smoke under `model_contrast/smoke_e2/`.
- A previous v2 failure was due to `NameError: adap_gamma is not defined`; that was fixed and `py_compile` passed.

Baseline e100 root:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast/baseline_e100
```

Baseline e100 results recorded in the thread:

| Condition | Best Acc | Final Acc |
|---|---:|---:|
| C10 `pr=0.5 nr=0.3` | 91.76 +/- 0.02 | 91.70 +/- 0.06 |
| C100 `pr=0.1 nr=0.0` | 77.47 +/- 0.12 | 77.41 +/- 0.06 |
| C100 `pr=0.05 nr=0.5` | 72.53 +/- 0.41 | 72.53 +/- 0.41 |
| C100H `pr=0.5 nr=0.2` | 74.05 +/- 0.11 | 73.95 +/- 0.01 |

C10 confidence e100 comparison recorded in the thread:

| Method | max_w_model | Best Acc | Final Acc |
|---|---:|---:|---:|
| baseline/original | 0.5 | 91.76 +/- 0.02 | 91.70 +/- 0.06 |
| confidence `r_i` | 0.0 | 90.50 +/- 0.05 | 90.48 +/- 0.04 |
| confidence `r_i` | 0.5 | 91.76 +/- 0.02 | 91.70 +/- 0.06 |
| confidence `r_i` | 1.0 | 91.13 +/- 0.67 | 91.05 +/- 0.66 |

Interpretation at that time:

- `max_w_model=0.5` was most stable.
- `max_w_model=0.0` dropped clearly, so fully ignoring model prediction was bad.
- `max_w_model=1.0` was less stable.

## Unified Ablation Progress Before Audit Fix

Before the audit-script repair, partial old unified ablation progress included:

- Completed `pr005_nr03/E0_full`: Final about `79.22`.
- Completed `pr005_nr03/E1_no_candidate_prior`: Final about `78.17`, around `-1.05` vs full.
- Completed `pr005_nr03/E2_uniform_ri`: Final about `79.14`, around `-0.08`.
- Completed `pr005_nr03/E3_weak_only_belief`: Final about `79.14`, around `-0.08`.
- Old `E5_no_salvage_training` was later found invalid and must be discarded.
- Old `E8_no_reliable_mixup` was later found invalid and must be discarded.

Pre-reboot progress record written on C201:

```text
/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/out_ultimate/bayes_unified_unified_ablation/EXPERIMENT_PROGRESS_BEFORE_REBOOT_20260513_1957.md
```

## Operational Notes

- When launching long C201 experiments, avoid mixing old invalid launcher queues with audit queues.
- If old tmux clients are suspended and `tmux list-sessions` hangs, terminate only the stuck tmux client/query process; do not kill active Python training unless explicitly requested.
- In future experiment reports, always include:
  - script path,
  - command or `command.txt`,
  - flag values,
  - exact code branch,
  - log evidence,
  - seed/epoch completion status.
- Treat `scripts/active/audit/` as the trusted current entry point for unified ablation.
- Treat `scripts/deprecated_invalid_20260513/` as trace-only historical material.
- For paper-facing experiment provenance, use the 5080-only record in
  `docs/NSE_5080_experiment_result_storage_reproduction_20260518.md`.
  Official manuscript wording should state the NSE ablations are based on the
  two RTX 5080 GPUs on `c201` only after every reported row is verified or
  rerun there. Non-5080 historical runs are internal debugging evidence unless
  the user explicitly asks to report them.

## Extractive vs Destructive Experiment Hub

New source write-back and extraction-quality experiments live under:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/extractive_vs_destructive
```

Trusted write-back audit script:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/extractive_vs_destructive/bayes_unified_writeback_source_audit.py
```

This script is copied from the verified audit ablation script and adds only:

- `--source_update_mode none`
- `--source_update_mode add_pseudo`
- `--source_update_mode replace_pseudo`
- `--source_update_mode remove_low_conf`
- `--source_update_mode alpha_blend`

Diagnostics emitted by this script:

- `[SourceAudit]`: hard/soft source contamination and true-label source mass.
- `[ExtractionAudit]`: active-set precision, active-set coverage, and active count.
- `[WriteBackAudit]`: destructive update mode, changed samples, and source L1 delta.

Experiment plan:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/extractive_vs_destructive/EXTRACTIVE_VS_DESTRUCTIVE_PLAN_20260513.md
```

Launcher files:

```text
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/extractive_vs_destructive/run_writeback_smoke_e5_20260513.sh
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/extractive_vs_destructive/run_writeback_e500_seed1_20260513.sh
/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/extractive_vs_destructive/wait_audit_then_writeback_smoke_20260513.sh
```

Current watcher started in the thread:

```text
tmux: writeback_smoke_after_audit_20260513
```

It waits for `audit_ablation_gpu0_20260513` and
`audit_ablation_gpu1_20260513` to finish, then runs the 5-epoch write-back
smoke on GPU0.
