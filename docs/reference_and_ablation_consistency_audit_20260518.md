# Reference and Ablation Consistency Audit 2026-05-18

## Scope

This audit checks the active manuscript build, the local and canonical reference
libraries, and the C201 experiment/code provenance relevant to the current NSE
paper. The active LaTeX build path is `main.tex` plus
`sec/0_abstract.tex`, `sec/1_intro.tex`, `sec/2_relatedwork.tex`,
`sec/3_method.tex`, `sec/4_experiment.tex`, and `sec/5_conclusion.tex`.
Backup files such as `sec/4_experiment_backup.tex` are not part of the active
PDF build.

## Reference Metadata And Publication-Version Audit

The active manuscript cites 21 keys. Both reference roots now contain the same
active cited-reference index:

- Project-local root: `D:\文件\论文项目\CE泄露V3_Branch_Experiments - 副本\参考文献列表`
- Canonical root: `D:\文件\论文项目\参考文献列表`

The main issue found in the reference library was that `qiao2023idgp` was cited
in the manuscript but missing from both reference-library indexes. It is now
added to the project-local and canonical indexes with a local OpenReview PDF:
[project cited index:L22](../参考文献列表/00_index/cited_references.md#L22),
[project bibkey map:L17](../参考文献列表/00_index/bibkey_to_folder.csv#L17),
and
[project metadata:L3-L13](../参考文献列表/03_instance_dependent_PLL/qiao2023idgp_2023_decompositional_generation_process_IDPLL/metadata.md#L3-L13).

Direct Google Scholar pages are not treated as a canonical machine-verifiable
source because they are dynamic and not stable for scripted inspection. The
version check therefore uses official venue or publisher pages where possible:
[IDGP/OpenReview](https://openreview.net/forum?id=lKOfilXucGB),
[POP/PMLR](https://proceedings.mlr.press/v202/xu23l.html),
[FREDIS/PMLR](https://proceedings.mlr.press/v202/qiao23b.html),
[ALIM/NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7988e9b3876ad689e921ce05d711442f-Abstract-Conference.html),
[UPLLRS/IJCAI](https://www.ijcai.org/proceedings/2023/468),
[PALS/CVF](https://openaccess.thecvf.com/content/CVPR2025W/FGVC/html/Saravanan_Pseudo-labelling_meets_Label_Smoothing_for_Noisy_Partial_Label_Learning_CVPRW_2025_paper.html),
[PiCO+/IEEE DOI](https://doi.org/10.1109/TPAMI.2023.3342650), and
[IRNet/IEEE DOI](https://doi.org/10.1109/TPAMI.2025.3620388).

2026-05-20 flat-library completion check:

- The active `main.pdf` bibliography contains 21 cited keys, as confirmed by
  `main.aux` `\bibcite` entries.
- The canonical flat PDF folder
  `D:\文件\论文项目\参考文献列表\paper_pdf列表` and the project-local flat PDF
  folder
  `D:\文件\论文项目\CE泄露V3_Branch_Experiments - 副本\参考文献列表\paper_pdf列表`
  had only 19 PDFs. The two missing flat-list PDFs were `wu2022revisiting` and
  `qiao2023idgp`.
- `wu2022revisiting` is the formal ICML/PMLR 2022 paper "Revisiting
  Consistency Regularization for Deep Partial Label Learning"; the official
  PMLR record and PDF are `https://proceedings.mlr.press/v162/wu22l.html` and
  `https://proceedings.mlr.press/v162/wu22l/wu22l.pdf`.
- `qiao2023idgp` is the formal ICLR 2023 paper "Decompositional Generation
  Process for Instance-Dependent Partial Label Learning"; the official
  OpenReview record and PDF are `https://openreview.net/forum?id=lKOfilXucGB`
  and `https://openreview.net/pdf?id=lKOfilXucGB`.
- Both PDFs were copied from their verified category folders into both flat
  `paper_pdf列表` folders, bringing each flat list to 21 PDFs. The
  `wang2022pico+` PDF was also copied from the flat list into its category
  folder in both roots because that category folder previously had metadata but
  no PDF.
- Both `download_status.csv` ledgers were updated so they contain all 21 active
  cited keys and record official non-arXiv sources for `wu2022revisiting` and
  `qiao2023idgp`.

2026-05-20 CroSel addition:

- `tian2024crosel` was added as an active reference after the manuscript cited
  CroSel as inspiration for the salvage history queue.
- `lv2020progressive` was also added to both reference-library indexes because
  PRODEN is now an active citation in the Related Work taxonomy.
- After these additions, the active manuscript bibliography contains 23 cited
  keys.
- PRODEN formal source: "Progressive Identification of True Labels for
  Partial-Label Learning", ICML 2020, PMLR 119:6500--6510. Official PMLR page:
  `https://proceedings.mlr.press/v119/lv20a.html`; official PMLR PDF:
  `https://proceedings.mlr.press/v119/lv20a/lv20a.pdf`.
- Formal source: "CroSel: Cross Selection of Confident Pseudo Labels for
  Partial-Label Learning", CVPR 2024, pp. 19479--19488. Official CVF page:
  `https://openaccess.thecvf.com/content/CVPR2024/html/Tian_CroSel_Cross_Selection_of_Confident_Pseudo_Labels_for_Partial-Label_Learning_CVPR_2024_paper.html`.
  Official CVF PDF:
  `https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_CroSel_Cross_Selection_of_Confident_Pseudo_Labels_for_Partial-Label_Learning_CVPR_2024_paper.pdf`.
- The reference was saved in both reference roots under
  `02_deep_PLL_and_NPLL/tian2024crosel_2024_crosel_cross_selection_pll`, and
  both flat `paper_pdf列表` folders were updated.
- The scoped citation role is limited to the historical-prediction FIFO
  memory-bank / queue idea. CroSel is standard PLL and assumes the true label is
  contained in the candidate set; NSE adapts the temporal stability idea to NPLL
  through model--KNN--prototype consensus and epoch-local active supervision.

2026-05-20 persistent-supervision-state proxy update:

- The paper-facing ablation scope was narrowed to five conservative
  persistent-supervision-state proxies: FREDIS-style add/remove candidate state,
  IRNet-style unreliable-sample correction, PALS/SARI-style partial-label
  augmentation, UPLLRS-style reliable-promotion state, and PiCO+-style soft
  pseudo-target carry-over.
- `peng2025noise` remains an important related-work citation for candidate-label
  reconstruction, but it is excluded from the five-proxy experiment table. The
  reason is comparability: its reconstruction method is coupled to a fixed
  feature-space / large-feature setting and candidate-label reconstruction
  design, while the current diagnostic table is intended to transplant only the
  inherited supervision-state operator into the same NSE extractor.
- The local implementation adds the UPLLRS-style promotion proxy without
  modifying the source prior: [train_nse_source_mvp.py:L2026-L2094](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L2026-L2094).
- Promoted samples are stored in `persistent_promoted_mask` /
  `persistent_promoted_label` and forced into later active training at
  [train_nse_source_mvp.py:L2545-L2612](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L2545-L2612).
- The compact c201 launcher fixes the previous `No module named data.dataset`
  failure by changing into the old project root and setting `PYTHONPATH` before
  invoking the experiment script:
  [launch_c201_persistent_state_five_20260520.sh](../experiments/nse_mvp_source_writeback_20260515_windows/launch_c201_persistent_state_five_20260520.sh).
- Remote checks on 2026-05-20 passed on both hosts. c201 runs the four
  candidate/promotion proxies in tmux session `nse_pss_c201_20260520`; d437
  runs `PiCOPlus_PSS` through `nohup` because `tmux` is unavailable on that host.
- The 2-epoch c201 `UPLLRS_PSS` smoke produced `[PersistentState]` promotion
  diagnostics and kept `[MVPSource] mode=none ... SourceDrift=0.000000`, so the
  promotion proxy is sample-state carry-over rather than source mutation.

2026-05-21 strict proxy instrumentation update:

- The candidate-membership insertion proxy now inserts only when the evidence
  top-1 label is outside the current candidate support. This prevents hard
  insert/add-remove rows from counting or amplifying an already-candidate label
  as if it were a candidate-membership repair.
- `--source_insert_margin` was added for optional stricter non-candidate
  correction tests. The main five-proxy setting uses margin `0.0`.
- `[MVPSource]` now logs `MassRecN`, `DamageMass`, and `AUCDrift`, so
  PiCO+-style soft evidence carry-over can be judged by soft true-label mass and
  long-run drift, not only by hard support metrics.
- The first launched 2026-05-20 run should be treated as a partial
  instrumentation run if the strict metrics are required for the paper table.
- The partial 2026-05-20 c201 tmux session and d437 `nohup` process were stopped
  before completion. Strict runs were relaunched under separate result roots:
  c201 tmux session `nse_pss_c201_strict_20260521` and d437 `nohup` root
  `nse_persistent_state_five_20260521_strict`.
- A 2-epoch strict c201 hard-insert smoke verified that `[MVPSource]` now emits
  `MassRecN`, `DamageMass`, and `AUCDrift`.

2026-05-21 prior-aligned proxy correction:

- The literature check confirmed that FREDIS should not be approximated only by
  confidence-threshold add/remove. Its candidate movement is based on the score
  difference between the predicted label and instance-label candidates:
  refinement adds `j notin S` when `f_yhat(x)-f_j(x) <= zeta`, while
  disambiguation removes `j in S` when `f_yhat(x)-f_j(x) >= zeta_bar`.
- IRNet's correction step uses
  `tau=max_{j in S(x)} f_j(x)-max_{j notin S(x)} f_j(x)` for noisy-sample
  detection and then inserts the highest-probability non-candidate label for
  detected noisy samples. Optional swapping is not used in the corrected proxy.
- PALS/SARI's persistent candidate update is partial-label augmentation with a
  decaying confidence threshold: if the current top-1 classifier prediction
  exceeds `lambda_t`, that top-1 label is unioned into the next iteration's
  partial label.
- UPLLRS is not a hard candidate-membership rewrite. It carries sample state:
  high-confidence pseudo-labelled unreliable samples are added to the reliable
  set and removed from the unreliable set.
- PiCO+ is best treated as soft pseudo-target / sample-selection carry-over,
  not a hard NPL source rewrite. The corrected proxy isolates the inherited
  moving-average soft target state.
- The corrected local implementation is in
  [train_nse_source_mvp.py:L1978-L2050](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L1978-L2050)
  and the c201 GPU1 launcher is
  [launch_c201_persistent_state_prior_aligned_20260521.sh:L72-L102](../experiments/nse_mvp_source_writeback_20260515_windows/launch_c201_persistent_state_prior_aligned_20260521.sh#L72-L102).
- Verification completed: local helper tests passed, local `py_compile` passed,
  remote c201 `py_compile` passed, remote helper tests passed, and a c201 GPU1
  2-epoch smoke completed all five corrected proxy branches. The full corrected
  suite was launched on c201 GPU1 in tmux session
  `nse_pss_prior_gpu1_20260521`.

2026-05-21 v2 conservative proxy correction:

- The v1 corrected suite and the older strict suite were stopped before
  completion because the final proxy design is more conservative.
- FREDIS now uses top-non-candidate refinement with both gap and confidence
  conditions, plus candidate disambiguation with both gap and low-confidence
  conditions.
- IRNet now uses its own candidate/non-candidate score-gap detector and a
  high-confidence non-candidate threshold, rather than NSE-estimated noisy
  samples.
- PALS/SARI now uses all-sample top-1 augmentation with a conservative
  `0.95 -> 0.85` threshold schedule.
- UPLLRS now promotes only from the NSE-estimated noisy set and selects labels
  from the native non-candidate space.
- PiCO+ remains a soft pseudo-target persistence proxy using `P^(2)` evidence
  with `alpha=0.1`.
- Local helper tests, local `py_compile`, remote c201 `py_compile`, and remote
  helper tests all passed for the v2 script. Waiting launchers were created in
  tmux sessions `nse_pss_v2_gpu0_20260521` and `nse_pss_v2_gpu1_20260521`; they
  wait for unrelated GPU jobs to release memory before running.

No active manuscript citation currently needs to be downgraded from a formal
venue to an arXiv-only record. The remaining arXiv-style entries in `main.bib`
are not active citations in the current PDF build.

## Introduction And Related Work Claim Audit

The Introduction and Related Work descriptions are broadly consistent with the
referenced papers after the current edits. The Related Work table was softened
where the previous wording could overstate persistence or dominance:

- The design-tension paragraph now says model-induced decisions can make
  working supervision more model-dependent, instead of claiming a uniform
  destructive rewrite mechanism:
  [sec/2_relatedwork.tex:L47-L58](../sec/2_relatedwork.tex#L47-L58).
- FREDIS is described as a refinement/disambiguation method that updates the
  working candidate-set state across fusion rounds:
  [sec/2_relatedwork.tex:L70-L70](../sec/2_relatedwork.tex#L70-L70).
- PALS is described as influencing subsequent pseudo-labelling and partial-label
  augmentation, not as necessarily dominating later supervision:
  [sec/2_relatedwork.tex:L74-L74](../sec/2_relatedwork.tex#L74-L74).
- UPLLRS is described through reliable/unreliable split state and subsequent
  disambiguation/semi-supervised training:
  [sec/2_relatedwork.tex:L76-L76](../sec/2_relatedwork.tex#L76-L76).
- ALIM is described as importance-weight and adjusted-target based rather than
  a hard candidate-set rewrite:
  [sec/2_relatedwork.tex:L82-L82](../sec/2_relatedwork.tex#L82-L82).

These edits preserve the paper's extractive-supervision contrast while avoiding
unsupported claims that all baselines rewrite candidate sets in the same way.

Additional Cour 2009 correction:

- Cour et al. should not be described as a maximum-likelihood-estimation PLL
  method. In the active Related Work text, it is now used as an early
  ambiguous-label learning reference that formalizes candidate-label learning,
  introduces ambiguity-degree analysis, and proposes a convex surrogate loss:
  [sec/2_relatedwork.tex:L7-L23](../sec/2_relatedwork.tex#L7-L23).
- This use keeps Cour 2009 in the standard PLL setting where the true label is
  assumed to be contained in the candidate set. It should not be cited as a
  noisy-PLL method or as evidence for candidate-set purification/write-back.

Related-work taxonomy refinement:

- The paper should not imply that every PLL/NPLL method modifies the original
  PL/NPL source. The active Related Work text now separates source-fixed
  loss/risk/weight methods, soft pseudo-target or label-distribution states,
  hard candidate-membership/reconstruction updates, and sample-state carry-over:
  [sec/2_relatedwork.tex:L18-L75](../sec/2_relatedwork.tex#L18-L75).
- This taxonomy keeps RC/CC-style risk correction and LWS-style label weighting
  outside the destructive write-back bucket, treats ALIM as source-preserving
  label-importance adjustment, treats PiCO/VALEN-style methods as soft
  pseudo-target/distribution-state methods, and reserves the source-drift /
  wrong-write critique for persistent candidate membership, reconstructed
  source, pseudo-target, label-importance, or sample-reliability states that
  carry model-induced errors across epochs.

External verification of previously under-specified methods:

- PRODEN is the ICML/PMLR 2020 paper "Progressive identification of true labels
  for partial-label learning" and is already present in `main.bib` as
  `lv2020progressive`. It should be treated as a source-fixed PLL method with a
  dynamic candidate-supported label-confidence / soft-target state, not as hard
  candidate-set write-back. Source checked:
  [PMLR v119](https://proceedings.mlr.press/v119/lv20a.html).
- CRDPLL corresponds to "Revisiting consistency regularization for deep partial
  label learning", ICML/PMLR 2022, already present as `wu2022revisiting` and
  cited in the experiment baselines. It supports the consistency-regularized
  PLL baseline category, not a candidate-source rewrite claim. Sources checked:
  [PMLR v162](https://proceedings.mlr.press/v162/wu22l.html) and Google
  Scholar indexing for the author/title.
- The current "NPLL reconstruction" citation is `peng2025noise`, not a separate
  active `PLRC` bibkey. The verified formal record is "Noise Separation guided
  Candidate Label Reconstruction for Noisy Partial Label Learning", ICLR 2025.
  It is safe to describe it as candidate-label reconstruction / reconstructed
  candidate-set supervision. Sources checked:
  [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a8b879590adff2b1874f97db59b65518-Abstract-Conference.html)
  and [OpenReview](https://openreview.net/forum?id=TOahfjA3sP).
- The active Related Work text was updated to mention PRODEN and CRDPLL
  explicitly and to include PRODEN/LWS in the supervision-state table:
  [sec/2_relatedwork.tex:L15-L25](../sec/2_relatedwork.tex#L15-L25) and
  [sec/2_relatedwork.tex:L86-L89](../sec/2_relatedwork.tex#L86-L89).

## Symbol, Unit, And PDF Consistency Audit

The following notation issues were fixed:

- `math_commands.tex` no longer redefines lowercase `\eqref`, which had caused
  PDF text such as `Eq. equation 1`. The local helper is now `\eqnref`, while
  `amsmath` owns `\eqref`: [math_commands.tex:L35-L37](../math_commands.tex#L35-L37).
- The duplicate vector macro was corrected so `\rvi` maps to bold `i` and
  `\rvu` maps to bold `u`: [math_commands.tex:L108-L120](../math_commands.tex#L108-L120).
- Soft write-back now uses `\alpha_{\mathrm{wb}}`, keeping it separate from
  MixUp's `\alpha_{\mathrm{mix}}`:
  [sec/3_method.tex:L142-L153](../sec/3_method.tex#L142-L153) and
  [sec/3_method.tex:L574-L574](../sec/3_method.tex#L574-L574).
- The internal confidence scores in the reliability ratio were changed from
  `q_i` to `s_i`, avoiding collision with the partial-label rate `q`:
  [sec/3_method.tex:L329-L337](../sec/3_method.tex#L329-L337).
- The Hoeffding illustrative margin was changed from `\gamma_{i,c}` to
  `m_{i,c}`, avoiding collision with the masked-entropy hyperparameter
  `\gamma`: [sec/3_method.tex:L453-L471](../sec/3_method.tex#L453-L471).
- The salvage history length now uses `L_{\mathrm{hist}}`, avoiding collision
  with entropy notation and matching the hyperparameter table:
  [sec/3_method.tex:L531-L544](../sec/3_method.tex#L531-L544) and
  [sec/4_experiment.tex:L98-L98](../sec/4_experiment.tex#L98-L98).
- Source write-back diagnostics now define `Write`, `Wrong`, `SrcRec-N`, and
  `Damage` explicitly, including the `Wrong=--` case when no rows are written:
  [sec/4_experiment.tex:L58-L66](../sec/4_experiment.tex#L58-L66) and
  [sec/4_experiment.tex:L354-L362](../sec/4_experiment.tex#L354-L362).

## Ablation Design And Program Mapping Audit

The current component ablation table in the manuscript covers the main
mechanism families: two Topology-DAES passes, adaptive reliability arbitration,
candidate-prior-projected model evidence, model-evidence cap, and active
salvage. This is a reasonable core ablation set for the current claim, but it
is not yet a fully exhaustive source-restoration study.

C201 provenance must be kept precise:

- The active trusted unified-ablation code path remains
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py`.
  Its high-signal runtime anchors include `[AblationCheck] candidate_prior=`,
  `[AblationCheck] reliable_mixup=`, `detected_salvage`, and
  `promoted_salvage`.
- The paper's A0--A8 numeric table is an internal component-ablation table. It
  does not need to be tied to the later 2026-05-13 audit launcher. The only
  requirement is provenance clarity: cite the actual script, command, flags,
  seeds, and logs that produced A0--A8. A rerun is unnecessary if those existing
  artifacts already prove the intended component changes.
- The May-13 audit script accurately implements the important design checks for
  candidate-prior projection and salvage/no-salvage logic. The old invalid
  `--disable_salvage_training` and `--no_reliable_mixup` style should remain
  deprecated for paper evidence.
- The source-write-back code path
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/extractive_vs_destructive/bayes_unified_writeback_source_audit.py`
  supports `none`, `add_pseudo`, `replace_pseudo`, `remove_low_conf`, and
  `alpha_blend`, with `[SourceAudit]`, `[ExtractionAudit]`, and
  `[WriteBackAudit]` diagnostics.
- The broader 2026-05-18 MVP family script supports additional update modes
  such as hard insert, hard remove, add/remove, mix, soft evidence, replace, and
  top-k reconstruction. Those operators are useful for a stronger
  source-restoration ablation, but only verified C201/5080 rows should be used
  as paper evidence.

Current coverage judgment: the existing ablations are sufficient to support the
core NSE mechanism comparison, but a stronger paper-facing source-restoration
claim should add or verify rows for hard remove, add-remove, top-k
reconstruction, periodic reset, and soft-evidence/mix variants. The
write-back-result table must also keep its noise-rate and hardware provenance
aligned with the 5080-only result policy before it is treated as final paper
evidence.

## Source Write-Back Metric Semantics

The source write-back diagnostics should distinguish write events from the final
source state. This matters when a model writes an incorrect label in one epoch
and later writes the correct label, or when soft write-back gradually changes a
source distribution.

Recommended reporting convention:

- `Write` / `WriteCov`: write coverage, the fraction of source rows updated at
  an epoch or across a reporting window.
- `Wrong` / `WrongWrite`: event-level wrong-write rate. A wrong update at epoch
  `t` is counted as wrong even if a later epoch overwrites it correctly. If the
  table reports only the final epoch, this should be stated explicitly.
- `CumWrong`: cumulative wrong-write rate over all write events in the run. This
  is the best scalar summary of how often a write-back policy injects wrong
  source evidence.
- `SrcRec-N`: source-level recovery rate among initially noisy samples. For hard
  candidate-set writes, this means the true label becomes present in the final
  source support. For soft source distributions, support recovery must be tied
  to a threshold `\epsilon` and should be reported together with a mass or
  top-1 diagnostic.
- `SrcMass-N`: average true-label source mass among initially noisy samples.
  This is preferable for soft write-back because a correct label may receive
  some probability mass without becoming the largest source entry.
- `SrcTop1-N`: fraction of initially noisy samples whose final source top-1
  label is the true label. This is stricter than `SrcRec-N`.
- `Damage` / `CleanDamage`: source damage among initially clean samples. For
  hard support, this means the true label is removed from source support. For
  soft source distributions, also inspect true-label mass decrease or top-1
  demotion.
- `MassDamage`: average positive drop in true-label source mass among initially
  clean samples, e.g. `max(0, Omega_i^0[y_i] - \tilde{Omega}_i^T[y_i])`.
- `Top1Damage`: fraction of initially clean samples whose true label is demoted
  below another label in the final source distribution.
- `SourceDrift`: average `L1` distance between the current/final source and the
  native source. This captures distribution movement but not whether the
  movement is correct or incorrect, so it should not replace `Wrong`,
  `SrcRec-N`, or `Damage`.

For the manuscript, a compact table can keep `Write`, `Wrong`, `SrcRec-N`,
`Damage`, and `Drift`, but the caption or text should say whether these are
final-epoch values, cumulative event values, or final-source-state values. For
soft write-back variants, the most defensible appendix diagnostics are
`CumWrong`, `SrcMass-N`, `SrcTop1-N`, `MassDamage`, `Top1Damage`, and
`SourceDrift`.

Current C201 implementation semantics:

- In `train_nse_source_mvp_family_20260518.py`, `[MVPSource] WriteCov` is
  `write_count / N` for the current epoch's update mask.
- `[MVPSource] WrongWrite` is an event-level current-epoch measure. For insert,
  mix, soft-evidence, and replace modes, it is the fraction of updated rows
  whose model top-1 pseudo-label differs from the clean label. For hard-remove
  modes, it is instead whether the true label is removed from support. For
  top-k reconstruction, it is whether the true label is absent from the
  reconstructed top-k support.
- `[MVPSource] SrcRecN` is a current-source-state metric: among initially noisy
  samples, whether the true label is in the current source support
  `work > 1e-12`. It is not a top-1 recovery metric and is permissive for soft
  distributions.
- `[MVPSource] CleanDamage` is also a current-source-state metric: among
  initially clean samples, whether the true label is absent from current source
  support. It does not count true-label mass weakening or top-1 demotion if the
  true label still has positive support.
- `[MVPSource] SourceDrift` is the mean row-wise `L1` distance between the
  current mutable source and the native source.

Therefore, the current reported source-diagnostic table should be interpreted
as final-epoch write-event quality plus final-source-state recovery/damage,
unless a collector explicitly aggregates across epochs. It does not yet report
cumulative wrong-write exposure, wrong-write survival time, soft true-label mass
damage, or top-1 demotion damage.

Actual modes observed in the C201 `main_table_e500` run logs:

- `hard_insert` was used for `M1_PALSStyleWB`, `M2_FixedWB065All`,
  `M3_FixedWB065U`, and `M9_PeriodicReset10`.
- `mix` was used for `M4_MixWB03U`.
- `replace` was used for `M5_ReplaceWBU`.
- `hard_remove` and `add_remove` were used for the extended `M7_HardRemove` and
  `M8_AddRemove` diagnostics.
- `soft_evidence` is supported by the 2026-05-18 family launcher/code, but no
  completed `main_table_e500` result directory was observed for it in the C201
  logs checked here.

## 2026-05-23 Experiment Script Taxonomy

The current experiments were not all produced by one monolithic script. They
share the same NSE method family and many hyperparameters, but different
paper-facing rows come from different script versions:

| Experiment family | Paper role | Trusted script / version | Status |
|---|---|---|---|
| Main NSE result table | main method results on CIFAR/crowd datasets | remote draft-root `bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model.py` | historical main-result script; result paths are indexed in `docs/bayes_unified_collected_results_with_paths.csv` |
| Main component ablation A0--A8 | Table-style internal component ablation | curated C201 audit copy `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/active/audit/bayes_unified_unified_ablation_audit.py` | trusted for paper-facing component ablations |
| Old unified ablation | legacy / invalid trace | `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/scripts/deprecated_invalid_20260513/bayes_unified_unified_ablation.py` | deprecated; do not report old E5/E8 rows |
| R_i and model-belief contrasts | secondary sensitivity / analysis | remote draft-root `bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model_contrast.py` and related weak/model-belief scripts | useful for interpretation; keep separate from the trusted A0--A8 audit table |
| Early source write-back stress tests | M0--M6 style reset/write-back diagnostics | local release script [train_nse_source_mvp.py](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L513-L529), synced remotely as `train_nse_source_mvp.py` / extended variants | useful historical stress tests; verify hardware and mode before paper use |
| 2026-05-18 docx source-update family | broad operator sweep using `p2` evidence | local launcher [launch_c201_docx_family_20260518.sh](../experiments/nse_mvp_source_writeback_20260515_windows/launch_c201_docx_family_20260518.sh#L1-L90), synced as `train_nse_source_mvp_family_20260518.py` | broader family; not the same as the five final v2 proxies |
| Persistent-state v2 proxies | final conservative prior-aligned proxy table | stable c201 launchers `run_persistent_state_proxies_gpu0.sh` and `run_persistent_state_proxies_gpu1.sh`, using `train_nse_persistent_state_proxies.py`; local snapshots are stored under [reproducibility/code/persistent_state/](../reproducibility/code/persistent_state/train_nse_persistent_state_proxies_20260523.py#L1) | current five-proxy result source |

For manuscript wording, keep this distinction explicit:

- Main accuracy comparisons and component ablations are not generated by the
  same script as the persistent-state proxy table.
- The five v2 proxy rows are controlled mechanism transplants into the same NSE
  extractor, not exact reimplementations of FREDIS, IRNet, PALS/SARI, UPLLRS,
  or PiCO+.
- The local release script [train_nse_source_mvp.py](../experiments/nse_mvp_source_writeback_20260515_windows/train_nse_source_mvp.py#L513-L554)
  is the current reproducible implementation surface for source-update and
  persistent-state proxy operators; older remote timestamped copies should be
  identified by their launcher/result directory before being cited.

## 2026-05-23 Reproducibility Package

A compact paper-facing reproducibility package has been created under
[reproducibility/](../reproducibility/README.md). It stores script snapshots,
result indices, reproduction command templates, and archive boundaries:

- Main-result script snapshot:
  [bayes_unified_main_20260523.py](../reproducibility/code/main/bayes_unified_main_20260523.py#L1).
- Component-ablation audit snapshot:
  [bayes_unified_unified_ablation_audit_20260523.py](../reproducibility/code/component_ablation/bayes_unified_unified_ablation_audit_20260523.py#L1).
- Persistent-state v2 stable snapshot:
  [train_nse_persistent_state_proxies_20260523.py](../reproducibility/code/persistent_state/train_nse_persistent_state_proxies_20260523.py#L1).
- Result root index:
  [result_index.csv](../reproducibility/manifest/result_index.csv).
- Local result evidence snapshots:
  [summary_20260523.csv](../reproducibility/results/summary_20260523.csv) and
  copied paper-facing `master_log.txt` files under
  [reproducibility/results/](../reproducibility/results/README.md).
- Archive policy:
  [archive_policy.md](../reproducibility/archive_policy.md).

The package is intentionally a snapshot and manifest layer. It does not move
legacy files out of the repository, because older documents and result logs
still reference those paths. Instead, it identifies which scripts are active,
historical, internal, or deprecated.

Verification completed on 2026-05-23:

- Python 3.7.9 syntax check passed for all copied reproducibility Python
  snapshots under [reproducibility/code/](../reproducibility/code/main/bayes_unified_main_20260523.py#L1).
- `latexmk -pdf -interaction=nonstopmode main.tex` completed successfully and
  regenerated a 14-page `main.pdf`.
- `main.log` contained no undefined citation or undefined reference warnings.
- The IRNet entry in [main.bib](../main.bib#L115) and the canonical reference
  library entry now use the formal IEEE TPAMI 2026 citation with DOI
  `10.1109/TPAMI.2025.3620388`.
- On c201, timestamped persistent-state development scripts were archived under
  `/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_mvp_source_writeback_20260514/archive/deprecated_timestamped_20260523/`.
  The active stable entry points are `train_nse_persistent_state_proxies.py`,
  `run_persistent_state_proxies_gpu0.sh`, and
  `run_persistent_state_proxies_gpu1.sh`.

## Verification Commands

Commands run during this audit:

```powershell
latexmk -pdf -interaction=nonstopmode main.tex
pdftotext -layout -enc UTF-8 main.pdf main_current_pdf_text_20260518.txt
```

The PDF build succeeded and produced a 12-page `main.pdf`; the only LaTeX
warning observed was a float-only page warning. The post-build text audit
returned `LATEX_SYMBOL_AUDIT_GREEN`, and the active-reference audit returned
`REFERENCE_AUDIT_GREEN active_keys=21`.
