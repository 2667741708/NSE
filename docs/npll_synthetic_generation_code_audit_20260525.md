# Synthetic NPLL Generation Code Audit

Date: 2026-05-25

## Question

The manuscript states that the synthetic NPLL protocol follows PiCO+ and the
PALS-reported baseline setting. This audit checks the claim from code, not only
from paper text.

## Repositories Checked

- PiCO/PiCO+ official repository:
  <https://github.com/hbzju/PiCO>
- PALS official repository:
  <https://github.com/darshana1406/PALS>

## Local NSE Implementation

NSE invokes synthetic NPLL generation from the main training entry point:

- [main training seed and data call](../paper_reproducible_release/code/01_nse_main_table_train.py#L1910-L1933)
- [local CIFAR partial-noise wrappers](../data/dataset.py#L149-L169)
- [local CIFAR-100 partial-noise wrapper](../data/dataset.py#L212-L234)

The uniform synthetic NPLL rule is:

- [uniform candidate generation](../data/dataset.py#L352-L384)

The hierarchical CIFAR-100H rule is:

- [hierarchical candidate generation](../data/dataset.py#L266-L348)

## Code-Level Comparison

PiCO+ implements the same transition-matrix rule in
`utils_plus/utils_algo.py`:

```text
transition_matrix = np.eye(K) * (1 - noisy_rate)
transition_matrix[off-diagonal] = partial_rate
while partialY[j].sum() == 0:
    resample Bernoulli labels from transition_matrix[true_label]
if noisy_rate == 0:
    force the true label into the candidate set
```

PALS implements the same rule in `data/dataset.py`, including the same
non-empty candidate-set resampling loop and the same hierarchical CIFAR-100H
masking logic. PALS command examples use:

```text
CIFAR-100: partial_ratio=0.05, noise_ratio=0.3
CIFAR-10: partial_ratio=0.5, noise_ratio=0.3
CIFAR-100H: partial_ratio=0.5, noise_ratio=0.2, heirarchical=True
```

## Conclusion

The NSE synthetic NPLL generation algorithm is code-level consistent with the
PiCO+ and PALS synthetic NPLL protocol:

- each incorrect class is independently included with probability `q`;
- the ground-truth class is independently included with probability
  `1 - eta`;
- a sample is resampled until its observed candidate set is non-empty;
- for CIFAR-100H, candidate generation is restricted to the true class's
  CIFAR-100 superclass through the hierarchical mask.

The important caveat is random seeding. NSE sets `seed_dataset = seed` for each
run, so seeds `1 2 3` generate three dataset-noise realizations. The PALS
official example script fixes `seed_dataset=42`, while PiCO+ uses its `--seed`
to seed NumPy before data generation. Therefore the paper can safely claim that
NSE follows the PiCO+/PALS synthetic NPLL generation protocol, but it should not
claim that the exact sampled candidate masks are bit-for-bit identical to a
particular PALS or PiCO+ example run unless the dataset seed is also matched.
