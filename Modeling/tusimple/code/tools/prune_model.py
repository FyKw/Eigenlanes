import os, copy, torch
import torch.nn as nn
import torch.nn.functional as F
from tools.decoder_prune import prune_decoder_only_out_channels
from tools.bn_calib import recalibrate_decoder_bn
from tools.save_slim import save_slim_checkpoint
import torch_pruning as tp


def count_size(state_dict):
    return sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)) * 4 / (1024 ** 2)

def count_sparsity(state_dict):
    total, zeros = 0, 0
    for p in state_dict.values():
        if isinstance(p, torch.Tensor):
            total += p.numel()
            zeros += (p == 0).sum().item()
    return 100.0 * zeros / total if total else 0.0

# ---------- small helpers for heads width ----------
def _resize_conv1d_inout(conv: nn.Conv1d, new_c: int) -> nn.Conv1d:
    new = nn.Conv1d(
        in_channels=new_c, out_channels=new_c, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=1, bias=(conv.bias is not None),
    )
    with torch.no_grad():
        k_in  = min(new_c, conv.in_channels)
        k_out = min(new_c, conv.out_channels)
        new.weight.zero_()
        new.weight[:k_out, :k_in, ...] = conv.weight[:k_out, :k_in, ...].clone()
        if conv.bias is not None and k_out > 0:
            new.bias[:k_out].copy_(conv.bias[:k_out])
    return new

def _resize_bn1d(bn: nn.BatchNorm1d, new_c: int) -> nn.BatchNorm1d:
    new = nn.BatchNorm1d(new_c, eps=bn.eps, momentum=bn.momentum, affine=True, track_running_stats=True)
    with torch.no_grad():
        k = min(new_c, bn.num_features)
        if bn.affine:
            new.weight[:k].copy_(bn.weight[:k])
            new.bias[:k].copy_(bn.bias[:k])
        if bn.track_running_stats:
            new.running_mean[:k].copy_(bn.running_mean[:k])
            new.running_var[:k].copy_(bn.running_var[:k])
    return new

def _shrink_heads_width(model: nn.Module, new_c: int):
    """
    Reduce the lane head width (c_sq) structurally and resize dependent layers.
    Keeps final output sizes intact.
    Note: Do NOT touch w1/w2 — they operate on x_concat (c_feat2*3), not c_sq.
    """
    old_c = getattr(model, 'c_sq', None)
    assert old_c is not None, "Model missing c_sq"
    if new_c == old_c:
        return model

    # Adjust feat_combine last conv2d → out_channels = new_c
    fc_last = model.feat_combine[-1]
    assert isinstance(fc_last, nn.Conv2d), "feat_combine last must be Conv2d"
    if fc_last.out_channels != new_c:
        new_fc = nn.Conv2d(
            in_channels=fc_last.in_channels, out_channels=new_c,
            kernel_size=fc_last.kernel_size, stride=fc_last.stride,
            padding=fc_last.padding, dilation=fc_last.dilation, bias=(fc_last.bias is not None)
        )
        with torch.no_grad():
            k = min(new_c, fc_last.out_channels)
            new_fc.weight.zero_()
            new_fc.weight[:k, :, :, :] = fc_last.weight[:k, :, :, :].clone()
            if fc_last.bias is not None:
                new_fc.bias[:k].copy_(fc_last.bias[:k])
        model.feat_combine[-1] = new_fc

    def _shrink_head_block(head: nn.Sequential):
        # Must be [Conv1d, BN1d, ReLU, Conv1d] with hidden width = c_sq
        if not isinstance(head, nn.Sequential) or len(head) < 4:
            return
        conv1, bn, relu, conv2 = head[0], head[1], head[2], head[3]
        if not isinstance(conv1, nn.Conv1d) or not isinstance(bn, nn.BatchNorm1d) or not isinstance(conv2, nn.Conv1d):
            return

        if conv1.in_channels != new_c or conv1.out_channels != new_c:
            head[0] = _resize_conv1d_inout(conv1, new_c)
        if bn.num_features != new_c:
            head[1] = _resize_bn1d(bn, new_c)
        if conv2.in_channels != new_c:
            new_conv2 = nn.Conv1d(
                in_channels=new_c, out_channels=conv2.out_channels,
                kernel_size=conv2.kernel_size, stride=conv2.stride,
                padding=conv2.padding, dilation=conv2.dilation,
                groups=1, bias=(conv2.bias is not None)
            )
            with torch.no_grad():
                k_in = min(new_c, conv2.in_channels)
                new_conv2.weight.zero_()
                new_conv2.weight[:, :k_in, ...] = conv2.weight[:, :k_in, ...].clone()
                if conv2.bias is not None:
                    new_conv2.bias.copy_(conv2.bias)
            head[3] = new_conv2

    # Lane heads present in your model
    for name in ("classification1", "classification2", "regression1", "regression2", "regression3", "regression4"):
        if hasattr(model, name):
            _shrink_head_block(getattr(model, name))

    # Do NOT change w1/w2 here (they use x_concat channels = c_feat2*3)
    model.c_sq = new_c
    return model

def _resize_conv2d_in(conv: nn.Conv2d, new_in: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_channels=new_in, out_channels=conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=1, bias=(conv.bias is not None),
    )
    with torch.no_grad():
        k = min(new_in, conv.in_channels)
        new.weight.zero_()
        new.weight[:, :k, :, :] = conv.weight[:, :k, :, :].clone()
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new

@torch.no_grad()
def _forward_encoder_only(encoder, inputs):
    x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    # Return ONLY the deepest feature to build a single-output graph
    # (Assumes encoder(x) returns a tuple: (feat1, feat2, feat3))
    f1, f2, f3 = encoder(x)
    return f3


def prune_encoder_out_channels(model: nn.Module, ratio: float, H: int, W: int, device):
    """
    Prune Conv2d OUT channels inside model.encoder (dependency-graph built on encoder only),
    then align feat_squeeze first convs to the new feature widths.
    """
    model.to(device).eval()
    example = torch.randn(1, 3, H, W, device=device)

    # 1) Build graph on encoder ONLY
    enc = model.encoder
    DG = tp.DependencyGraph().build_dependency(
        enc,
        example_inputs=(example,),
        forward_fn=_forward_encoder_only,
    )

    # 2) Collect Conv2d inside encoder
    encoder_convs = [(name, m) for name, m in enc.named_modules() if isinstance(m, nn.Conv2d)]

    # 3) Build pruning ratio dict for modules present in the graph
    #    (handle API variants for module->node map)
    module_in_graph = set()
    if hasattr(DG, "module2node") and isinstance(DG.module2node, dict):
        module_in_graph = set(DG.module2node.keys())
    elif hasattr(DG, "_module_to_node") and isinstance(DG._module_to_node, dict):
        module_in_graph = set(DG._module_to_node.keys())

    pruning_ratio_dict = {}
    for name, conv in encoder_convs:
        if module_in_graph and conv not in module_in_graph:
            continue
        if conv.out_channels <= 1:
            continue
        pruning_ratio_dict[conv] = ratio

    if not pruning_ratio_dict:
        print("ℹ️ Encoder: no eligible Conv2d modules found in graph — skipping.")
        return model, 0

    # 4) Run pruner on encoder
    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model=enc,  # IMPORTANT: pruner over encoder only
        example_inputs=(example,),
        importance=importance,
        pruning_ratio_dict=pruning_ratio_dict,
        global_pruning=False,
        forward_fn=_forward_encoder_only,
    )
    pruner.step()

    # 5) One pass to get new encoder feature widths
    feat1, feat2, feat3 = enc(example)
    c1, c2, c3 = feat1.shape[1], feat2.shape[1], feat3.shape[1]

    # 6) Align feat_squeeze[1..3] first convs to match (c1, c2, c3)
    # feat_squeeze1[0] conv expects in_channels=c1
    fs1_0 = model.feat_squeeze1[0]
    if hasattr(fs1_0, "conv") and isinstance(fs1_0.conv, nn.Conv2d) and fs1_0.conv.in_channels != c1:
        fs1_0.conv = _resize_conv2d_in(fs1_0.conv, c1)

    fs2_0 = model.feat_squeeze2[0]
    if hasattr(fs2_0, "conv") and isinstance(fs2_0.conv, nn.Conv2d) and fs2_0.conv.in_channels != c2:
        fs2_0.conv = _resize_conv2d_in(fs2_0.conv, c2)

    fs3_0 = model.feat_squeeze3[0]
    if hasattr(fs3_0, "conv") and isinstance(fs3_0.conv, nn.Conv2d) and fs3_0.conv.in_channels != c3:
        fs3_0.conv = _resize_conv2d_in(fs3_0.conv, c3)

    # 7) Re-materialize x_concat once on the model to lock shapes
    model.forward_for_encoding(example)
    model.forward_for_squeeze()

    # Rough count of removed channels
    removed = 0
    for conv, r in pruning_ratio_dict.items():
        removed += int(round(conv.out_channels * r))
    return model, removed

def run_prune(cfg, dict_DB, group_ratios, suffix=""):
    """
    Accepts multiple groups via group_ratios, e.g. {'decoder': 0.15, 'heads': 0.10, 'encoder': 0.05}.
    Applies structural pruning where implemented, recalibrates BN, and saves slim checkpoint.
    """
    print("\n🔧 Running structural pruning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build fresh model from baseline each time
    base = dict_DB.get("model_base")
    if base is None:
        if "model" not in dict_DB or dict_DB["model"] is None:
            raise RuntimeError("❌ Model not loaded into dict_DB.")
        base = copy.deepcopy(dict_DB["model"]).cpu()
        dict_DB["model_base"] = base

    model = copy.deepcopy(dict_DB["model_base"]).to(device).eval()

    # Report before
    sd_before = model.state_dict()
    size_before = count_size(sd_before)
    spars_before = count_sparsity(sd_before)
    nparams_before = sum(p.numel() for p in model.parameters())
    print(f"🔎 Original: {size_before:.2f} MB, {spars_before:.2f}% sparsity, params={nparams_before}")

    # Decode ratios
    dec_r = float(group_ratios.get("decoder", 0.0))
    enc_r = float(group_ratios.get("encoder", 0.0))
    hd_r  = float(group_ratios.get("heads",   0.0))
    sq_r  = float(group_ratios.get("squeeze", 0.0))
    fc_r  = float(group_ratios.get("combine", 0.0))

    # 1) Encoder structured prune (affects accuracy)
    enc_removed = 0
    if enc_r > 0.0:
        print(f"🔧 Pruning encoder with ratio={enc_r:.2f}")
        model, enc_removed = prune_encoder_out_channels(model, enc_r, cfg.height, cfg.width, device)
        print(f"🧩 Encoder out-ch removed: {enc_removed}")

    # 2) Decoder structured prune (latency/seg visualization)
    dec_removed = 0
    if dec_r > 0.0:
        model, dec_removed = prune_decoder_only_out_channels(model, dec_r, cfg.height, cfg.width, device)
        print(f"🧩 Decoder out-ch removed: {dec_removed}")

    # 3) Heads width shrink (sensitive; directly changes logits)
    if hd_r > 0.0:
        old_c = int(model.c_sq)
        new_c = max(1, int(round(old_c * (1.0 - hd_r))))
        if new_c < old_c:
            print(f"⚠️ Shrinking heads width c_sq: {old_c} -> {new_c}")
            model = _shrink_heads_width(model, new_c)
        else:
            print(f"ℹ️ heads ratio {hd_r} produced no shrink (c_sq={old_c})")

    # 4) Squeeze / combine: not enabled (explicitly)
    if sq_r > 0.0:
        print("⚠️ Squeeze-path pruning requested but not enabled yet (skipped).")
    if fc_r > 0.0:
        print("⚠️ feat_combine pruning requested but not enabled yet (skipped).")

    # BN recalibration
    try:
        # If only decoder changed, recalibrate decoder BN
        if dec_r > 0.0 and enc_r == 0.0 and hd_r == 0.0:
            recalibrate_decoder_bn(model, dict_DB['testloader'], device, num_batches=200)
        else:
            # Light global BN recalibration to stabilize stats after encoder/heads pruning
            model.train()
            it = 0
            for batch in dict_DB['testloader']:
                imgs = batch['img'].to(device, non_blocking=True)
                model.forward_for_encoding(imgs)
                model.forward_for_squeeze()
                model.forward_for_lane_feat_extraction()
                it += 1
                if it >= 200:
                    break
            model.eval()
    except Exception as e:
        print(f"ℹ️ BN recalib skipped ({e}).")

    # After-report
    sd_after = model.state_dict()
    size_after = count_size(sd_after)
    spars_after = count_sparsity(sd_after)
    nparams_after = sum(p.numel() for p in model.parameters())
    print(f"📉 Slimmed:  {size_after:.2f} MB, {spars_after:.2f}% sparsity, params={nparams_after}")
    if nparams_before > 0:
        print(f"💾 Size saved (params): {(1 - nparams_after / nparams_before) * 100:.2f}%")

    # Save
    out_dir = os.path.join(cfg.dir["weight"], "pruned")
    os.makedirs(out_dir, exist_ok=True)
    if not suffix:
        alias = {'decoder': 'dec', 'encoder': 'enc', 'heads': 'hd', 'squeeze': 'sq', 'combine': 'fc'}
        parts = [f"{alias[k]}{int(group_ratios[k]*100)}" for k in sorted(group_ratios.keys())]
        suffix = "_".join(parts)
    out_path = os.path.join(out_dir, f"checkpoint_tusimple_res_{cfg.backbone}_pruned_{suffix}.pt")

    meta = {"backbone": cfg.backbone, **{k: float(v) for k, v in group_ratios.items()},
            "enc_removed": int(enc_removed), "dec_removed": int(dec_removed)}
    save_slim_checkpoint(model, out_path, meta=meta)

    try:
        fsz_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"🗃  File saved: {out_path} ({fsz_mb:.2f} MB)")
    except Exception:
        pass

    return out_path