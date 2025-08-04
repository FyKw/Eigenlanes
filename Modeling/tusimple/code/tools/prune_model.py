import os
import copy
import torch
import torch.nn.utils.prune as prune


def count_size(state_dict):
    """Return model size in MB"""
    return sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)) * 4 / (1024 ** 2)


def count_sparsity(state_dict):
    """Return sparsity (percentage of 0 weights)"""
    total, zeros = 0, 0
    for p in state_dict.values():
        if isinstance(p, torch.Tensor):
            total += p.numel()
            zeros += (p == 0).sum().item()
    return 100.0 * zeros / total if total else 0.0


def structured_prune(model, ratio):
    """
    Apply structured pruning (L2 norm) on Conv2d and Linear layers
    """

    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                prune.ln_structured(module, name="weight", amount=ratio, n=2, dim=0)
                prune.remove(module, "weight")  # Make pruning permanent
                print(f"✅ Pruned: {name}")
            except Exception as e:
                print(f"⚠️ Skipped {name}: {e}")
    return model


def run_prune(cfg, dict_DB):
    print("\n🔧 Running structured pruning...")

    model = dict_DB.get("model", None)
    if model is None:
        raise RuntimeError("❌ Model not loaded into dict_DB. Make sure `load_model_for_pruning()` was called.")

    model = model.cpu().eval()
    state_dict = model.state_dict()

    original_size = count_size(state_dict)
    original_sparsity = count_sparsity(state_dict)
    print(f"🔎 Original: {original_size:.2f} MB, {original_sparsity:.2f}% sparsity")

    pruned_model = structured_prune(model, cfg.prune_ratio)
    pruned_state = pruned_model.state_dict()

    pruned_size = count_size(pruned_state)
    pruned_sparsity = count_sparsity(pruned_state)
    print(f"📉 Pruned:   {pruned_size:.2f} MB, {pruned_sparsity:.2f}% sparsity")
    print(f"💾 Size saved: {100 * (original_size - pruned_size) / original_size:.2f}%")

    out_dir = os.path.join(cfg.dir["weight"], "pruned")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pruned_r{cfg.prune_ratio:.2f}.pth")
    torch.save({"model": pruned_state}, out_path)
    print(f"✅ Saved pruned model to: {out_path}")

