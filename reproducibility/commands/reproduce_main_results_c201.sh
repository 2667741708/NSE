#!/usr/bin/env bash
set -euo pipefail

# Template for reproducing the main NSE result family on c201.
# Run from the draft project root:
# /home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409

cd /home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409

PY=${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}
SCRIPT=bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model.py

"${PY}" "${SCRIPT}" \
  --dataset CIFAR100 \
  --train_root ./data \
  --lpi 10 \
  --network R18 \
  --epochs 500 \
  --batch_size 256 \
  --lr 0.1 \
  --wd 0.001 \
  --momentum 0.9 \
  --lr_scheduler cosine \
  --mixup_alpha 1.0 \
  --lsr 0.0 \
  --consistency_weight 1.0 \
  --ema_alpha 0.999 \
  --k_val 15 \
  --delta 0.25 \
  --history_len 15 \
  --consensus_power 2.0 \
  --sim_mode_1 topology_daes \
  --sim_mode_2 topology_daes \
  --knn_heads 1 \
  --topology_rel_gamma 2.0 \
  --topology_rel_eps 1e-12 \
  --daes_entropy_coeff 0.5 \
  --max_w_model 0.5 \
  --model_warmup_epochs 20 \
  --pr 0.05 \
  --nr 0.3 \
  --seeds 1 2 3 \
  --cuda_dev 0 \
  --out ./out_ultimate \
  --exp_name bayes_unified_融合可靠_自适应R_i_双视图模型预测_分开model/c100_pr005_nr03_maxw05/reproduce_e500_seed123
