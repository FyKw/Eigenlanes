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


def structured_prune(model, global_ratio):
    """
    Apply structured pruning (L2 norm) on Conv2d and Linear layers,
    excluding critical layers and adjusting pruning strength by depth.
    """

    model = copy.deepcopy(model)

    # Manual rules or patterns to control pruning
    layers_to_skip = [
        "encoder.conv1",               # first layer
        "decoder.4",                   # final output
        "classification", "regression", "w1", "w2"  # avoid prediction heads
    ]

    def should_skip(name):
        return any(skip in name for skip in layers_to_skip)

    def get_layer_ratio(name):
        # Custom logic per layer
        if "layer1" in name: return global_ratio
        if "layer2" in name: return global_ratio
        if "layer3" in name: return global_ratio
        if "layer4" in name: return global_ratio
        return global_ratio

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and not should_skip(name):
            try:
                amount = get_layer_ratio(name)
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                prune.remove(module, "weight")
                print(f"✅ Pruned: {name} with {amount * 100:.0f}%")
            except Exception as e:
                print(f"⚠️ Skipped {name}: {e}")
        else:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                print(f"⏩ Skipped: {name} (sensitive layer)")

    return model



def run_prune(cfg, dict_DB, prune_ratio):
    print("\n🔧 Running structured pruning...")

    model = dict_DB.get("model", None)
    if model is None:
        raise RuntimeError("❌ Model not loaded into dict_DB. Make sure `load_model_for_pruning()` was called.")

    model = model.eval()
    state_dict = model.state_dict()

    original_state = state_dict

    original_size = count_size(state_dict)
    original_sparsity = count_sparsity(state_dict)
    print(f"🔎 Original: {original_size:.2f} MB, {original_sparsity:.2f}% sparsity")

    pruned_model = structured_prune(model, prune_ratio)

    pruned_state = pruned_model.state_dict()

    pruned_size = count_size(pruned_state)
    pruned_sparsity = count_sparsity(pruned_state)
    print(f"📉 Pruned:   {pruned_size:.2f} MB, {pruned_sparsity:.2f}% sparsity")
    print(f"💾 Size saved: {100 * (original_size - pruned_size) / original_size:.2f}%")

    out_dir = os.path.join(cfg.dir["weight"], "pruned")
    os.makedirs(out_dir, exist_ok=True)
    prune_str = f"{int(prune_ratio * 100)}"
    out_path = os.path.join(out_dir, f"checkpoint_tusimple_res_{cfg.backbone}_pruned{prune_str}")
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

