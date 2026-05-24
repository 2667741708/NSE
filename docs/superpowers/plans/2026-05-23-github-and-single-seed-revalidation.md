# GitHub Reproducibility Upload And Single-Seed Revalidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the curated NSE reproduction package to GitHub and start single-seed validation runs for paper-facing results.

**Architecture:** Keep the repository upload scoped to `reproducibility/`, the paper experiment section, and provenance docs. Put stable c201 reproduction entry points under a single remote hub and keep timestamped or exploratory scripts out of the active path. Use one-to-one manifests to map every paper result to a script, command family, result source, and single-seed validation slot.

**Tech Stack:** Git/GitHub CLI, LaTeX, Python training scripts, c201 SSH/tmux launchers, Markdown/CSV manifests.

---

### Task 1: Preserve Upload Scope

**Files:**
- Read: `git status -sb`
- Read: `git remote -v`

- [x] Confirm that the worktree is mixed and avoid `git add -A`.
- [x] Confirm that no `origin` remote is configured locally.
- [ ] After the user provides the GitHub repository URL, add it as `origin` and push a scoped branch.

### Task 2: Add One-To-One Reproduction Checklist

**Files:**
- Create: `reproducibility/manifest/paper_result_reproduction_checklist_20260523.md`
- Create: `reproducibility/manifest/paper_result_reproduction_matrix_20260523.csv`

- [ ] List every paper-facing NSE result family from the manuscript.
- [ ] Mark third-party baseline rows as literature-provenance rows, not rows reproduced by NSE scripts.
- [ ] Map each NSE row to the stable script family and single-seed validation command.

### Task 3: Add Single-Seed Validation Launchers

**Files:**
- Create: `reproducibility/commands/run_single_seed_validation_c201_gpu0.sh`
- Create: `reproducibility/commands/run_single_seed_validation_c201_gpu1.sh`

- [ ] Use stable script names only.
- [ ] Run separate `exp_name` values under `results/nse_single_seed_validation_20260523`.
- [ ] Split representative validation across c201 GPU0 and GPU1.

### Task 4: Sync Stable Scripts To c201

**Files:**
- Remote create/update: `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_reproducibility/`

- [ ] Copy main-result, component-ablation, and persistent-state scripts to stable remote names.
- [ ] Copy single-seed launchers to the remote hub.
- [ ] Verify Python syntax on c201.

### Task 5: Start Representative Single-Seed Runs

**Files:**
- Remote tmux sessions:
  - `nse_single_seed_gpu0_20260523`
  - `nse_single_seed_gpu1_20260523`

- [ ] Start GPU0 launcher for FREDIS, IRNet, and PALS/SARI PSS proxy validation.
- [ ] Start GPU1 launcher for UPLLRS, PiCO+, component A0, and main CIFAR100 `q=0.05, eta=0.3` validation.
- [ ] Record the result root and how to monitor it.

### Task 6: Verify And Prepare GitHub Push

**Files:**
- Modify: `reproducibility/README.md`
- Modify: `reproducibility/manifest/script_taxonomy.md`

- [ ] Run local Python syntax checks.
- [ ] Build `main.pdf`.
- [ ] Check `main.log` for undefined citations/references.
- [ ] Stage only scoped files after the user supplies the GitHub target.
