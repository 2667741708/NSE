# NSE Paper Reproducibility Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a paper-facing reproducibility package that separates trusted NSE scripts, result provenance, citation checks, and legacy/archive boundaries.

**Architecture:** Keep the active manuscript untouched except for bibliography correction. Store reproducibility material under `reproducibility/`, with copied script snapshots under `reproducibility/code/`, command wrappers under `reproducibility/commands/`, and manifest documents under `reproducibility/manifest/`.

**Tech Stack:** LaTeX/BibTeX, Python training scripts, shell launchers, c201 SSH paths, Markdown/CSV provenance files.

---

### Task 1: Correct IRNet Bibliography

**Files:**
- Modify: `main.bib`
- Modify: `D:/文件/论文项目/参考文献列表/02_deep_PLL_and_NPLL/lian2023irnet_2026_IRNet/bibtex.bib`

- [ ] Replace the `lian2023irnet` entry with the formal TPAMI 2026 citation:
  `IEEE Transactions on Pattern Analysis and Machine Intelligence`, volume `48`, number `2`, pages `1932--1948`, month `feb`, DOI `10.1109/TPAMI.2025.3620388`, and IEEE stamp URL.

- [ ] Verify `rg -n "lian2023irnet|3620388|11202439" main.bib`.

### Task 2: Copy Trusted Script Snapshots

**Files:**
- Create directory: `reproducibility/code/main/`
- Create directory: `reproducibility/code/component_ablation/`
- Create directory: `reproducibility/code/persistent_state/`

- [ ] Copy the main-result script from c201 into `reproducibility/code/main/bayes_unified_main_20260523.py`.
- [ ] Copy the trusted audit script from c201 into `reproducibility/code/component_ablation/bayes_unified_unified_ablation_audit_20260523.py`.
- [ ] Copy the local persistent-state training script and v2 launchers into `reproducibility/code/persistent_state/`.

- [ ] Generate `reproducibility/manifest/sha256sums.txt` for copied code files.

### Task 3: Write Reproduction Commands

**Files:**
- Create: `reproducibility/commands/reproduce_main_results_c201.sh`
- Create: `reproducibility/commands/reproduce_component_ablation_c201.sh`
- Create: `reproducibility/commands/reproduce_persistent_state_v2_c201.sh`

- [ ] Add commands that document the exact c201 roots, scripts, flags, result roots, and seed policy.
- [ ] Mark commands as reproduction templates, not commands already executed in this package.

### Task 4: Write Result and Reference Manifests

**Files:**
- Create: `reproducibility/README.md`
- Create: `reproducibility/manifest/result_index.csv`
- Create: `reproducibility/manifest/script_taxonomy.md`
- Create: `reproducibility/manifest/reference_check.md`
- Create: `reproducibility/archive_policy.md`

- [ ] Record which result roots are paper-facing, internal, historical, or deprecated.
- [ ] Record that final five v2 proxy rows come from `train_nse_source_mvp_family_20260521_v2.py`.
- [ ] Record that IRNet is cited as the formal TPAMI 2026 article while the local PDF is an arXiv public copy because IEEE PDF access requires sign-in.
- [ ] Record the legacy scripts that should not be used for final paper tables.

### Task 5: Verify

**Files:**
- Read: `main.bib`
- Read: `reproducibility/`

- [ ] Run `python -m py_compile` on copied Python scripts where local Python can parse them.
- [ ] Run `latexmk -pdf -interaction=nonstopmode main.tex`.
- [ ] Check `main.log` for undefined citations/references.
- [ ] Report any verification that cannot run due environment limitations.
