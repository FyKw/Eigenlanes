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


def structured_prune(model, group_ratios):
    """Apply structured pruning to model groups (encoder, decoder, etc.)."""
    model = copy.deepcopy(model)

    always_skip = [
        "encoder.conv1",  # very first layer
        "decoder.4",  # final output
        "w1", "w2"  # correlation heads
        "regression"
    ]

    def should_skip(name):
        return any(skip in name for skip in always_skip)

    def get_group_ratio(name):
        if name.startswith("encoder"):
            return group_ratios.get("encoder", 0.0)
        elif name.startswith("decoder"):
            return group_ratios.get("decoder", 0.0)
        # elif "classification" in name:
        #     return group_ratios.get("classification", 0.0)
        return 0.0

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and not should_skip(name):
            try:
                amount = get_group_ratio(name)
                if amount > 0:
                    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                    prune.remove(module, "weight")
                    print(f"✅ Pruned: {name} ({amount*100:.0f}%)")
                else:
                    print(f"➖ Skipped: {name} (0% pruning)")
            except Exception as e:
                print(f"⚠️ Error pruning {name}: {e}")
        elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"⏩ Skipped: {name} (sensitive or excluded)")
    return model



def run_prune(cfg, dict_DB, group_ratios, suffix=""):
    print("\n🔧 Running structured pruning...")

    model = dict_DB.get("model", None)
    if model is None:
        raise RuntimeError("❌ Model not loaded into dict_DB.")

    model = model.eval()
    state_dict = model.state_dict()

    original_size = count_size(state_dict)
    original_sparsity = count_sparsity(state_dict)
    print(f"🔎 Original: {original_size:.2f} MB, {original_sparsity:.2f}% sparsity")

    pruned_model = structured_prune(model, group_ratios)
    pruned_state = pruned_model.state_dict()

    pruned_size = count_size(pruned_state)
    pruned_sparsity = count_sparsity(pruned_state)
    print(f"📉 Pruned:   {pruned_size:.2f} MB, {pruned_sparsity:.2f}% sparsity")
    print(f"💾 Size saved: {100 * (original_size - pruned_size) / original_size:.2f}%")

    out_dir = os.path.join(cfg.dir["weight"], "pruned")
    os.makedirs(out_dir, exist_ok=True)

    if not suffix:
        suffix = "_".join([f"{k[:4]}{int(v*100)}" for k, v in group_ratios.items()])
    out_path = os.path.join(out_dir, f"checkpoint_tusimple_res_{cfg.backbone}_pruned_{suffix}")

    torch.save({
        "model": pruned_model.state_dict(),
        "epoch": 300,
        "val_result": 0.0,
        "optimizer": torch.optim.Adam(pruned_model.parameters()).state_dict(),
    }, out_path)


    # print(f"✅ Saved pruned model to: {out_path}")
    # print("=========original keys============")
    # for key in original_state.keys():
    #     print(key + "\n")
    # print("=========pruned keys============")
    # for key in pruned_state.keys():
    #     print(key + "\n")
    #
    # for key in original_state.keys():
    #     assert torch.equal(original_state[key], pruned_state[key]), f"{key} has changed during 0% pruning."

