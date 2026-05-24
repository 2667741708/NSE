$ErrorActionPreference = "Stop"
$Root = "D:\文件\论文项目\CE泄露V3_Branch_Experiments - 副本"
$ExpDir = Join-Path $Root "experiments\nse_mvp_source_writeback_20260515_windows"
$Out = Join-Path $Root "results\nse_mvp_source_writeback_20260515_windows\smoke_e3"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$env:PYTHONPATH = $Root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:WANDB_MODE = "disabled"
$Python = "F:\anaconda\envs\flatgpu\python.exe"
& $Python (Join-Path $ExpDir "train_nse_source_mvp.py") `
  --dataset CIFAR100 --train_root (Join-Path $Root "data") `
  --pr 0.05 --nr 0.3 `
  --network R18 --epochs 3 --batch_size 128 `
  --lr 0.1 --wd 0.001 --momentum 0.9 --lr_scheduler cosine `
  --mixup_alpha 1.0 --lsr 0.0 `
  --k_val 15 --delta 0.25 --history_len 2 `
  --sim_mode_1 topology_daes --sim_mode_2 topology_daes --knn_heads 1 `
  --max_w_model 0.5 --model_warmup_epochs 1 `
  --out $Out --seeds 1 --cuda_dev 0 --exp_name M0_NSE `
  --source_update_mode none --source_update_scope none
