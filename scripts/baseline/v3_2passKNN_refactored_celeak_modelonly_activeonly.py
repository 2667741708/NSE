"""
Model-only Leaky Prior + Active-only Training.
"""

from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

def _resolve_base_script() -> Path:
    candidates = [
        THIS_DIR / "v3_2passKNN_refactored_bayes_unified.py",
        Path("/tmp/svc_mounts/c201/home_c201_公共_whm_PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/v3_2passKNN_refactored_bayes_unified.py"),
        Path("/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409/v3_2passKNN_refactored_bayes_unified.py"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.resolve() != THIS_FILE:
                return candidate.resolve()
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Could not locate base script.")

BASE_SCRIPT_PATH = _resolve_base_script()
BASE_SPEC = importlib.util.spec_from_file_location("base_runtime", BASE_SCRIPT_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load base module from {BASE_SCRIPT_PATH}")

base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules["base_runtime"] = base
BASE_SPEC.loader.exec_module(base)

torch = base.torch
F = base.F
np = base.np
wandb = base.wandb
copy = base.copy
setup_logger = base.setup_logger
run_single_experiment = base.run_single_experiment
GradScaler = getattr(base, "GradScaler", torch.cuda.amp.GradScaler)
autocast = getattr(base, "autocast", torch.cuda.amp.autocast)

def _extract_custom_args(argv: list[str]) -> tuple[list[str], dict[str, float]]:
    custom_values: dict[str, float] = {"prior_leakage": 0.01}
    cleaned_argv = [argv[0]]
    idx = 1
    while idx < len(argv):
        if argv[idx] == "--prior_leakage" and idx + 1 < len(argv):
            custom_values["prior_leakage"] = float(argv[idx + 1])
            idx += 2
            continue
        cleaned_argv.append(argv[idx])
        idx += 1
    return cleaned_argv, custom_values

def _build_leaky_prior(static_cand_mask, crowd_prior, num_classes, prior_leakage, eps):
    hard_support = static_cand_mask.float()
    base_prior = crowd_prior.float() if crowd_prior is not None else hard_support
    base_prior = base_prior / (base_prior.sum(dim=1, keepdim=True) + eps)
    uniform_prior = torch.full_like(base_prior, 1.0 / float(num_classes))
    leaky_prior = (1.0 - prior_leakage) * base_prior + prior_leakage * uniform_prior
    zero_mass_upper = prior_leakage * (hard_support <= 0).float().sum(dim=1) / float(num_classes)
    return leaky_prior, base_prior, zero_mass_upper

# --- OVERRIDE SELECTION ---
def reliable_pseudolabel_selection_advanced(logger, args, device, trainloader, features, epoch, state_manager, model_preds=None, proto_manager=None):
    N = features.shape[0]
    dataset = trainloader.dataset
    eps_stable = 1e-8

    if hasattr(dataset, "original_soft_labels"):
        static_cand_mask = torch.tensor(dataset.original_soft_labels, device=device, dtype=torch.float64)
    else:
        static_cand_mask = torch.tensor(dataset.soft_labels, device=device, dtype=torch.float64)

    is_crowd = getattr(args, "dataset", "") in ["Treeversity", "Benthic", "Plankton"]
    crowd_prior = None
    if is_crowd and hasattr(dataset, "weights"):
        crowd_prior = torch.tensor(dataset.weights, device=device, dtype=torch.float32) + eps_stable

    current_fixed_labels = torch.tensor(dataset.soft_labels, device=device).float()
    clean_labels = torch.tensor(dataset.clean_labels, device=device, dtype=torch.long)
    row_ids = torch.arange(N, device=device)
    clean_in_static_cand = (static_cand_mask[row_ids, clean_labels].to(torch.float32) > 0)

    def _filter_logic(soft_probs):
        prob_temp = torch.clamp(soft_probs, min=1e-6, max=1 - 1e-6)
        discrepancy = -torch.log(prob_temp)
        _, max_idx = soft_probs.max(dim=1)
        total_cand_mask = (static_cand_mask.gather(1, max_idx.unsqueeze(1)).squeeze(1) > 0)
        rel_mask = torch.zeros(N, device=device)
        counts = torch.bincount(max_idx[total_cand_mask], minlength=args.num_classes).double()
        limit = torch.quantile(counts, args.delta) if counts.numel() > 0 else 0

        for class_idx in range(args.num_classes):
            idx_c_mask = total_cand_mask & (max_idx == class_idx)
            if idx_c_mask.sum() == 0: continue
            k_val = min(limit.item(), idx_c_mask.sum().float().item())
            if k_val < 1: continue
            _, top_idx = torch.topk(discrepancy[idx_c_mask, class_idx], k=int(k_val), largest=False)
            rel_mask[idx_c_mask.nonzero().squeeze(1)[top_idx]] = 1.0

        n_sel = rel_mask.sum().item()
        acc = ((max_idx[rel_mask.bool()] == clean_labels[rel_mask.bool()]).float().mean().item() if n_sel > 0 else 0.0)
        return rel_mask, max_idx, acc, n_sel

    D_mh, neighbors_mh = base.knn_search_pytorch_chunked(features, args.k_val, num_heads=args.knn_heads)
    raw_sim = F.relu(D_mh).float()

    def _subset_accuracy(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
        if mask.sum().item() == 0: return 0.0
        return (predictions[mask] == labels[mask]).float().mean().item()

    curr_soft_out = crowd_prior.clone() if crowd_prior is not None else current_fixed_labels.clone()
    
    logger.info("=" * 80)
    logger.info(f"🚀 [Epoch {epoch}] Reliable Selection - ModelOnly Leaky")
    logger.info("=" * 80)

    with torch.no_grad():
        mask_init, _, acc_init, size_init = _filter_logic(curr_soft_out)
        overall_pred_init = curr_soft_out.argmax(dim=1)
        overall_acc_init = (overall_pred_init == clean_labels).float().mean().item()
        out_acc_init = _subset_accuracy(overall_pred_init, clean_labels, ~clean_in_static_cand)
        logger.info("📊 [Stage 0] Initial (Before Propagation)")
        logger.info(f"   ├─ Selected: {int(size_init)} samples | Acc: {acc_init * 100:.2f}% | Overall Acc: {overall_acc_init * 100:.2f}%")
        logger.info(f"   ├─ Clean-outside-candidate ratio: {(~clean_in_static_cand).float().mean().item() * 100:.2f}% | Outside Overall Acc: {out_acc_init * 100:.2f}%")

    propagation_input_1 = curr_soft_out
    refined_sim_1 = base.get_weight_matrix(args.sim_mode_1, raw_sim, neighbors_mh, propagation_input_1, args)
    neighbor_vals_1 = propagation_input_1[neighbors_mh]
    weighted_votes_1 = torch.einsum("nk,nkc->nc", refined_sim_1, neighbor_vals_1)
    curr_soft_out = F.softmax(weighted_votes_1, dim=1)

    with torch.no_grad():
        mask_s1, _, acc_s1, size_s1 = _filter_logic(curr_soft_out)
        overall_pred_s1 = curr_soft_out.argmax(dim=1)
        overall_acc_s1 = (overall_pred_s1 == clean_labels).float().mean().item()
        out_acc_s1 = _subset_accuracy(overall_pred_s1, clean_labels, ~clean_in_static_cand)
        logger.info("📊 [Stage 1] After 1st Propagation")
        logger.info(f"   ├─ Selected: {int(size_s1)} samples | Acc: {acc_s1 * 100:.2f}% | Overall Acc: {overall_acc_s1 * 100:.2f}%")
        logger.info(f"   ├─ Outside Overall Acc: {out_acc_s1 * 100:.2f}%")

    p_knn1 = curr_soft_out
    adap_eps = float(getattr(args, "adap_rel_eps", 1e-12))
    adap_gamma = float(getattr(args, "adap_rel_gamma", 2.0))
    ce_direction = getattr(args, "adap_ce_direction", "knn_over_model")
    model_warmup_epochs = float(getattr(args, "model_warmup_epochs", 20))
    max_w_model_val = float(getattr(args, "max_w_model", 1.0))
    w_model = min(max_w_model_val, epoch / max(model_warmup_epochs, 1.0))

    if model_preds is not None:
        p_model_raw = model_preds.float()
        p_model_effective = w_model * p_model_raw + (1.0 - w_model) * p_knn1.detach()
        p_model_effective = p_model_effective / (p_model_effective.sum(dim=1, keepdim=True) + adap_eps)

        if ce_direction == "knn_over_model":
            ce_i = -torch.sum(p_knn1 * torch.log(p_model_effective + adap_eps), dim=1)
        else:
            ce_i = -torch.sum(p_model_effective * torch.log(p_knn1 + adap_eps), dim=1)
        norm_ce = ce_i / (np.log(args.num_classes) + adap_eps)
        r_i = torch.exp(-adap_gamma * norm_ce.pow(2)).unsqueeze(1)
        logger.info(f"✨ [Stage 2.5] CE Reliability Fusion: w_model={w_model:.4f}")
    else:
        p_model_effective = p_knn1.clone()
        r_i = torch.ones(N, 1, device=p_knn1.device)

    prior_leakage = float(getattr(args, "prior_leakage", 0.01))
    omega, _, zero_mass_upper = _build_leaky_prior(
        static_cand_mask=static_cand_mask, crowd_prior=crowd_prior,
        num_classes=args.num_classes, prior_leakage=prior_leakage, eps=adap_eps
    )

    # 🌟 KEY DIFFERENCE: LEAKY OMEGA APPLIED ONLY TO MODEL 🌟
    model_prior_fused = p_model_effective * omega
    model_prior_fused = model_prior_fused / (model_prior_fused.sum(dim=1, keepdim=True) + adap_eps)
    
    adaptive_blend = r_i * p_knn1 + (1.0 - r_i) * model_prior_fused
    propagation_input_2 = adaptive_blend / (adaptive_blend.sum(dim=1, keepdim=True) + adap_eps)

    with torch.no_grad():
        mask_s28, _, acc_s28, size_s28 = _filter_logic(propagation_input_2)
        logger.info(f"📊 [Stage 2.8] Model-Only Leaky Prior Fusion")
        logger.info(f"   ├─ prior_leakage: {prior_leakage:.4f} | Selected: {int(size_s28)} samples")

    refined_sim_2 = base.get_weight_matrix(args.sim_mode_2, raw_sim, neighbors_mh, propagation_input_2, args)
    neighbor_vals_2 = propagation_input_2[neighbors_mh]
    weighted_votes_2 = torch.einsum("nk,nkc->nc", refined_sim_2, neighbor_vals_2)
    curr_soft_out = F.softmax(weighted_votes_2, dim=1)

    mask_final, pred_final, _, _ = _filter_logic(curr_soft_out)
    mask_final = mask_final.float()

    with torch.no_grad():
        mask_expanded = mask_final.view(-1, 1).expand(-1, args.k_val + 1).float()
        w_pruned = raw_sim * torch.gather(mask_expanded, 0, neighbors_mh)
        w_p_norm = w_pruned / (w_pruned.sum(dim=1).unsqueeze(1) + 1e-12)
        soft_out_pruned = torch.sum(F.embedding(neighbors_mh, curr_soft_out) * w_p_norm.view(N, -1, 1), dim=1)
        pruned_pl = soft_out_pruned.argmax(dim=1)

    model_hard_labels = model_preds.argmax(dim=1) if model_preds is not None else p_model_effective.argmax(dim=1)
    neighbor_model_onehot = F.one_hot(model_hard_labels[neighbors_mh], num_classes=args.num_classes).float()
    geo_soft_out = torch.sum(neighbor_model_onehot * raw_sim.unsqueeze(-1), dim=1)
    model_geo_pl = geo_soft_out.argmax(dim=1)

    proto_pl_all = proto_manager.predict(features)[1] if proto_manager is not None else None
    state_manager.update_history(mask_final, pruned_pl, model_geo_pl, proto_pl=proto_pl_all)
    return mask_final.float(), pred_final, pred_final, pred_final, curr_soft_out.float()

base.reliable_pseudolabel_selection_advanced = reliable_pseudolabel_selection_advanced

# --- OVERRIDE ACTIVE ONLY TRAIN ---
def train_unified_single_stream_active_only(args, encoder, classifier, device, unified_loader, optimizer, softmatch_manager, logger, num_classes, global_labels, global_is_reliable, knn_pl, model_pl, knn_scores, dynamic_consistency_weight, rebalance_factor, saved_softmatch_weights, proto_manager=None):
    encoder.train()
    classifier.train()
    scaler = GradScaler()
    total_loss_s, num_sup, skipped_unreliable = 0.0, 0, 0

    for batch_data in unified_loader:
        weak_imgs, strong_imgs, indices = batch_data
        weak_imgs, strong_imgs, indices = weak_imgs.to(device), strong_imgs.to(device), indices.to(device)

        labels = global_labels[indices]
        reliable_mask = global_is_reliable[indices].bool()

        if reliable_mask.sum().item() == 0:
            skipped_unreliable += int((~reliable_mask).sum().item())
            continue

        optimizer.zero_grad()
        rel_weak, rel_strong, rel_labels = weak_imgs[reliable_mask], strong_imgs[reliable_mask], labels[reliable_mask]
        batch_active = rel_weak.size(0)
        skipped_unreliable += int((~reliable_mask).sum().item())

        if proto_manager is not None:
            with torch.no_grad():
                proto_manager.update(encoder(rel_weak).detach(), torch.ones(batch_active, device=device, dtype=torch.bool), rel_labels)

        s_labels = F.one_hot(rel_labels.long(), num_classes).float()
        s_labels = s_labels * (1 - args.lsr) + args.lsr / num_classes

        lam_w = np.random.beta(args.mixup_alpha, args.mixup_alpha)
        perm_w = torch.randperm(batch_active, device=device)
        mix_w = lam_w * rel_weak + (1 - lam_w) * rel_weak[perm_w]
        mix_l_w = lam_w * s_labels + (1 - lam_w) * s_labels[perm_w]

        lam_s = np.random.beta(args.mixup_alpha, args.mixup_alpha)
        perm_s = torch.randperm(batch_active, device=device)
        mix_s = lam_s * rel_strong + (1 - lam_s) * rel_strong[perm_s]
        mix_l_s = lam_s * s_labels + (1 - lam_s) * s_labels[perm_s]

        with autocast():
            loss_w = -torch.sum(F.log_softmax(classifier(encoder(mix_w)), 1) * mix_l_w, 1).mean()
            loss_s_strong = -torch.sum(F.log_softmax(classifier(encoder(mix_s)), 1) * mix_l_s, 1).mean()
            loss_s = 0.5 * (loss_w + loss_s_strong)

        if loss_s > 0 and not torch.isnan(loss_s):
            scaler.scale(loss_s).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss_s += loss_s.item() * batch_active
        num_sup += batch_active

    logger.info(f"  -> [Train Loss] Sup={total_loss_s/num_sup if num_sup>0 else 0:.4f} | ActiveOnly=True | Skipped={skipped_unreliable}")
    return 0.0

base.train_unified_single_stream = train_unified_single_stream_active_only

def main():
    cleaned_argv, custom_values = _extract_custom_args(sys.argv)
    sys.argv = cleaned_argv
    args = base.parse_args()
    for k, v in custom_values.items(): setattr(args, k, v)

    wandb.init(mode="disabled")
    master_log_dir = os.path.join(args.out, args.exp_name)
    master_logger = setup_logger(master_log_dir, "master_log.txt", is_master=True)
    master_logger.info("========================= Starting Experiment Series =========================")
    master_logger.info(f"Wrapper Variant: ModelOnly Leaky Prior + ActiveOnly LSR")
    master_logger.info(f"Base Settings: {vars(args)}")

    for idx, seed in enumerate(args.seeds):
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        master_logger.info(f"--- Starting Run {idx + 1}/{len(args.seeds)} with Seed: {seed} ---")
        best, final, dur = run_single_experiment(run_args)
        master_logger.info(f"--- Run {idx + 1} Finished. Dur: {dur/60.0:.2f}min | Best: {best:.2f}% | Final: {final:.2f}% ---\n")

if __name__ == "__main__":
    main()
