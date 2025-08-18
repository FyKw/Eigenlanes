import os, copy, time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# ---------- helpers ----------

def count_size(state_dict):
    return sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)) * 4 / (1024 ** 2)

def count_sparsity(state_dict):
    total, zeros = 0, 0
    for p in state_dict.values():
        if isinstance(p, torch.Tensor):
            total += p.numel()
            zeros += (p == 0).sum().item()
    return 100.0 * zeros / total if total else 0.0

def load_checkpoint_strict(model, path, key="model"):
    """Strictly load weights and fail loud if keys mismatch."""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("⚠️ load_state_dict mismatches:")
        if missing:    print("  Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected: print("  Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
        raise RuntimeError("State dict mismatch — likely loading the wrong blob/key.")
    return ckpt

def tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype.is_floating_point:
        return torch.allclose(a, b, atol=0.0, rtol=0.0)  # exact for float weights
    return torch.equal(a, b)

def state_dicts_identical(sd1, sd2) -> bool:
    if sd1.keys() != sd2.keys(): return False
    for k in sd1.keys():
        if not tensors_equal(sd1[k], sd2[k]): return False
    return True

# ---------- pruning ----------

def structured_prune(model, group_ratios):
    """
    Mask-based structured pruning with per-group ratios.
    If all ratios are 0, return the original model unchanged (true no-op).
    """
    # 1) Hard no-op if everything is 0
    if not any(v > 0.0 for v in group_ratios.values()):
        return model  # do NOT deepcopy; keep it byte-identical

    # Otherwise, work on a copy so the caller still has the original intact
    model = copy.deepcopy(model)

    always_skip = [
        "encoder.conv1",
        "decoder.4",
    ]

    def should_skip(name: str) -> bool:
        return any(skip in name for skip in always_skip)

    def which_group(name: str):
        if name.startswith("encoder"): return "encoder"
        if name.startswith("decoder"): return "decoder"
        if name.startswith("feat_squeeze") or name.startswith("feat_combine"): return "squeeze"
        if (name.startswith("classification") or name.startswith("regression")
            or name.startswith("w1") or name.startswith("w2")): return "heads"
        return None

    PRUNE_TYPES = (nn.Conv2d, nn.Linear, nn.Conv1d)

    for name, module in model.named_modules():
        if isinstance(module, PRUNE_TYPES) and not should_skip(name):
            group = which_group(name)
            amount = float(group_ratios.get(group, 0.0)) if group is not None else 0.0
            if amount > 0.0:
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                prune.remove(module, "weight")  # make pruning permanent
    return model

# ---------- end-to-end prune + save ----------

def run_prune(cfg, dict_DB, group_ratios, suffix=""):
    print("\n🔧 Running structured pruning...")
    model = dict_DB.get("model", None)
    if model is None:
        raise RuntimeError("❌ Model not loaded into dict_DB.")
    model.eval()

    pruned_model = structured_prune(model, group_ratios)

    out_dir = os.path.join(cfg.dir["weight"], "pruned")
    os.makedirs(out_dir, exist_ok=True)

    if not suffix:
        suffix = "_".join([f"{k[:4]}{int(v*100)}" for k, v in group_ratios.items()])
    out_path = os.path.join(
        out_dir,
        f"checkpoint_tusimple_res_{cfg.backbone}_pruned_{suffix}"
    )

    # save in the SAME format as training
    torch.save({
        "epoch": 0,
        "val_result": 0.0,
        "model": pruned_model.state_dict(),  # << this is what your loader expects
        "optimizer": torch.optim.Adam(pruned_model.parameters()).state_dict(),
    }, out_path)
    return out_path

# ---------- optional: forward equivalence test for 0% ----------

@torch.no_grad()
def assert_noop_forward_equivalence(model, dataloader, device="cuda"):
    """Run 1–2 batches to ensure outputs match before/after 0% prune."""
    model = model.to(device).eval()
    # Make a pruned copy with 0 ratios
    m2 = structured_prune(model, {"encoder":0.0,"decoder":0.0,"squeeze":0.0,"heads":0.0}).to(device).eval()

    for i, batch in enumerate(dataloader):
        if i >= 2: break
        img = batch["img"].to(device)
        # your pipeline’s staged forward
        model.forward_for_encoding(img);      m2.forward_for_encoding(img)
        model.forward_for_squeeze();          m2.forward_for_squeeze()
        model.forward_for_lane_feat_extraction(); m2.forward_for_lane_feat_extraction()
        out1 = model.forward_for_lane_component_prediction()
        out2 = m2.forward_for_lane_component_prediction()

        # Compare a representative tensor
        k = next(iter(out1.keys()))
        diff = (out1[k] - out2[k]).abs().max().item()
        print(f"[noop-check] max abs diff on '{k}': {diff:.6f}")
        if diff != 0.0:
            raise RuntimeError("0% prune changed forward outputs — investigate load/path.")
