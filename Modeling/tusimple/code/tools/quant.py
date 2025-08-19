import os, torch, torch.nn as nn

def summarize_dtypes(model):
    dtypes = {}
    for n,p in model.named_parameters():
        dtypes.setdefault(str(p.dtype), 0)
        dtypes[str(p.dtype)] += p.numel()
    total = sum(dtypes.values())
    print("Param dtype breakdown:")
    for dt, cnt in dtypes.items():
        print(f"  {dt:<10}  {cnt/1e6:.2f}M params  ({cnt/total*100:.1f}%)")

def weight_stats(model, n=10):
    # peek min/max on a few big tensors
    picked = sorted(list(model.state_dict().items()), key=lambda kv: -kv[1].numel())[:n]
    for k, t in picked:
        if not torch.is_floating_point(t): continue
        print(f"{k:<50}  shape={tuple(t.shape)}  min={t.min().item():+.4e}  max={t.max().item():+.4e}")


def to_half_inference(model: nn.Module) -> nn.Module:
    model = model.eval()
    model.half()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            m.float()
    return model

def to_bf16_inference(model: nn.Module) -> nn.Module:
    model = model.eval().to(torch.bfloat16)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            m.float()
    return model

def save_quant_variant(cfg, model: nn.Module, tag: str, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(cfg.dir["weight"], "quant")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"checkpoint_tusimple_res_{cfg.backbone}_quant_{tag}")
    torch.save({
        "epoch": 0,
        "val_result": 0.0,
        "model_obj": model.cpu(),            # full module (so loader uses it)
        "model": model.state_dict(),         # legacy

    }, path)
    return path