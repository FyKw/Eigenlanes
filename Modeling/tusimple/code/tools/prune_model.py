import os, copy, time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from types import GeneratorType
import torch_pruning as tp

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

from contextlib import contextmanager

@contextmanager
def eval_no_grad(model):
    was_training = model.training
    try:
        model.eval()
        yield
    finally:
        if was_training:
            model.train()



def _synthetic_loader(cfg, device="cuda", num_batches=8, batch_size=1):
    H, W = int(cfg.height), int(cfg.width)
    def _gen():
        for _ in range(num_batches):
            yield {"img": torch.randn(batch_size, 3, H, W, device=device)}
    return _gen()

def _iterable_loader(obj):
    # Accept PyTorch DataLoader, lists, generators, etc.
    try:
        import torch.utils.data as tud
    except Exception:
        tud = None
    if isinstance(obj, (list, tuple, GeneratorType)):
        return obj
    if tud is not None and isinstance(obj, tud.DataLoader):
        return obj
    if hasattr(obj, "__iter__"):
        return obj
    return None

def _get_any_loader(dict_DB, cfg=None, device="cuda"):
    """
    Prefer your keys: 'trainloader', 'testloader'.
    Fallback to many common names. If none, make a tiny synthetic loader.
    """
    PREFERRED = ["trainloader", "testloader"]
    CANDIDATE_KEYS = PREFERRED + [
        "train_loader","train_dataloader","loader_train","training_loader",
        "val_loader","val_dataloader","validation_loader",
        "test_loader","test_dataloader","loader","dataloader",
        "data_loader","eval_loader",
    ]
    for k in CANDIDATE_KEYS:
        if k in dict_DB and dict_DB[k] is not None:
            it = _iterable_loader(dict_DB[k])
            if it is not None:
                print(f"ℹ️ Using dataloader from dict_DB['{k}'].")
                return it

    # Optional: dataset keys → wrap
    for k in ["train_dataset","dataset","val_dataset","test_dataset"]:
        if k in dict_DB and dict_DB[k] is not None:
            try:
                import torch.utils.data as tud
                print(f"ℹ️ Wrapping dict_DB['{k}'] in a DataLoader.")
                return tud.DataLoader(dict_DB[k], batch_size=1, shuffle=False, num_workers=0)
            except Exception:
                pass

    if cfg is None:
        raise RuntimeError("No dataloader found and no cfg provided for synthetic fallback.")
    print("⚠️ No dataloader found in dict_DB; using synthetic calibration batches.")
    return _synthetic_loader(cfg, device=device, num_batches=8, batch_size=1)


class TaylorCollector:
    """
    Collects first-order Taylor scores per-output-channel for target conv modules.
    Taylor score per channel j ~ sum_{N,H,W} | activation[j] * grad_out[j] |
    """
    def __init__(self, target_modules: dict, device="cuda"):
        # target_modules: { "name": conv_module }
        self.device = device
        self.targets = target_modules
        self.scores = {name: None for name in target_modules.keys()}
        self._handles = []

    def _make_hooks(self, name, module):
        # We store forward activations; then use a module-level backward hook
        # to read grad_output (same shape as activation).
        act_store = {}

        def fwd_hook(mod, inp, out):
            # Save activation for this module; detach so we don't bloat the graph.
            act_store["act"] = out.detach()

        def bwd_hook(mod, grad_input, grad_output):
            # grad_output[0] matches the module's output tensor gradient
            g = grad_output[0]
            a = act_store.get("act", None)
            if a is None or g is None:
                return
            with torch.no_grad():
                if a.device != g.device:
                    a = a.to(g.device)
                # First-order Taylor score per out-channel: sum_{N,H,W} |g * a|
                contrib = (g * a).abs().sum(dim=(0, 2, 3))  # [C_out]
                if self.scores[name] is None:
                    self.scores[name] = contrib
                else:
                    self.scores[name] += contrib

        # Use module-level hooks (works even if the output tensor doesn't require grad flag)
        self._handles.append(module.register_forward_hook(fwd_hook))
        self._handles.append(module.register_full_backward_hook(bwd_hook))

    def attach(self):
        for name, mod in self.targets.items():
            self._make_hooks(name, mod)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

def _get_squeeze_last_convs(model):
    """
    Returns the last convs of feat_squeeze1/2/3 as a dict name->module.
    This matches your Model definition:
      feat_squeeze1: 4 conv_bn_relu blocks → use index 3
      feat_squeeze2: 3 blocks → index 2
      feat_squeeze3: 2 blocks → index 1
    """
    return {
        "sq1_last": model.feat_squeeze1[3].conv,
        "sq2_last": model.feat_squeeze2[2].conv,
        "sq3_last": model.feat_squeeze3[1].conv,
    }

def _pick_keep_indices(score_vec: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    Keep top-k channels by score. Returns 1D LongTensor of kept indices (sorted ascending).
    """
    c = score_vec.numel()
    k = max(1, int(round((1.0 - keep_ratio) * c)))  # keep (1 - prune_ratio)
    k = min(k, c)
    topk = torch.topk(score_vec, k, largest=True).indices
    keep = torch.sort(topk).values
    return keep

def _slice_conv2d_out(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    """
    Build a new Conv2d with fewer out_channels by selecting filters by keep_idx.
    """
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=keep_idx.numel(),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[keep_idx, :, :, :])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[keep_idx])
    return new_conv

def _slice_bn2d_out(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    new_bn = nn.BatchNorm2d(
        num_features=keep_idx.numel(),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
        device=bn.weight.device,
        dtype=bn.weight.dtype,
    )
    if bn.affine:
        with torch.no_grad():
            new_bn.weight.copy_(bn.weight[keep_idx])
            new_bn.bias.copy_(bn.bias[keep_idx])
    if bn.track_running_stats:
        with torch.no_grad():
            new_bn.running_mean.copy_(bn.running_mean[keep_idx])
            new_bn.running_var.copy_(bn.running_var[keep_idx])
    return new_bn

def _slice_conv2d_in(conv: nn.Conv2d, keep_in_idx: torch.Tensor) -> nn.Conv2d:
    """
    Reduce in_channels by slicing weight along dim=1.
    """
    new_conv = nn.Conv2d(
        in_channels=keep_in_idx.numel(),
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,  # NOTE: assumes groups==1 or compatible with slicing
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[:, keep_in_idx, :, :])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv

def _slice_conv1d_in(conv: nn.Conv1d, keep_in_idx: torch.Tensor) -> nn.Conv1d:
    new_conv = nn.Conv1d(
        in_channels=keep_in_idx.numel(),
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[:, keep_in_idx, :])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv

def _replace_conv_bn_relu_block(block, new_conv: nn.Conv2d):
    """
    Your conv_bn_relu is conv->bn->relu. Replace conv & its BN with sliced versions.
    Assumes block is instance of conv_bn_relu.
    """
    assert hasattr(block, "conv") and hasattr(block, "bn") and hasattr(block, "relu")
    # out slicing has already been applied to conv, so we must also slice BN to match
    old_bn: nn.BatchNorm2d = block.bn
    keep_out = torch.arange(new_conv.out_channels, device=new_conv.weight.device)
    new_bn = _slice_bn2d_out(old_bn, keep_out)
    block.conv = new_conv
    block.bn = new_bn
    return block

def _concat_keep_indices(k1, k2, k3, c_each_original):
    """
    Build input-channel index list for concatenation [sq1 || sq2 || sq3],
    given per-branch keep indices in their own local coordinates [0..c_each_original-1].
    """
    k2_shift = k2 + c_each_original
    k3_shift = k3 + 2 * c_each_original
    return torch.cat([k1, k2_shift, k3_shift], dim=0)

@torch.no_grad()
@torch.no_grad()
def _ensure_img_in_batch(batch, device):
    img = batch.get("img", None)
    if img is None:
        # fall back to any tensor-like entry shaped like an image
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                img = v
                break
    if img is None:
        raise RuntimeError("No 'img' tensor found in batch for forward pass.")
    return img.to(device)

def _safe_mean(t):
    # helper: returns None if t is None
    if t is None:
        return None
    return t.mean()

def _sum_terms(terms):
    # sum only non-None; return None if all None
    terms = [t for t in terms if t is not None]
    if not terms:
        return None
    s = terms[0]
    for t in terms[1:]:
        s = s + t
    return s

def _requires_grad_scalar(x):
    return isinstance(x, torch.Tensor) and x.requires_grad

def _float_tensor(x):
    return x.float() if isinstance(x, torch.Tensor) else x

def _forward_with_loss(dict_DB, model, batch, device):
    """
    Build a scalar that *always* has a grad path to the squeeze stack,
    even without labels. Prefer real loss if available; otherwise use
    a surrogate that touches x_concat and sq_feat.
    """
    # ensure autograd ON for this function
    torch.set_grad_enabled(True)

    img = _ensure_img_in_batch(batch, device)
    img.requires_grad_(False)  # inputs need not require grad; we care about params

    # staged forward
    model.forward_for_encoding(img)
    model.forward_for_squeeze()
    model.forward_for_lane_feat_extraction()
    outs = model.forward_for_lane_component_prediction()

    loss_fn = dict_DB.get("loss_fn", None)
    has_labels = all(k in batch for k in ["prob", "height_prob", "offset", "is_pos_reg", "seg_label", "gt_edge_map"])

    if loss_fn is not None and has_labels:
        # True task loss (preferred). Ensure it's a scalar with grad.
        loss_dict = loss_fn(outs, batch)
        loss = loss_dict.get("sum", None)
        if loss is None:
            # fallback: sum of known sub-losses
            loss = _sum_terms([
                loss_dict.get("prob", None),
                loss_dict.get("seg", None),
                loss_dict.get("edge", None),
                loss_dict.get("prob_h", None),
            ])
        if loss is not None and _requires_grad_scalar(loss):
            return loss

    # ---------- Surrogate loss (guaranteed grad path) ----------
    # Touch tensors that *must* depend on squeeze outputs:
    #   - x_concat (cat of squeeze outputs)
    #   - sq_feat   (feat_combine result)
    # Also include logits if present (extra signal).
    probe_terms = []

    if hasattr(model, "x_concat") and isinstance(model.x_concat, torch.Tensor):
        # L2 energy couples to squeeze outputs through cat()
        probe_terms.append((model.x_concat ** 2).mean())

    if hasattr(model, "sq_feat") and isinstance(model.sq_feat, torch.Tensor):
        probe_terms.append(model.sq_feat.abs().mean())

    pl = outs.get("prob_logit", None)
    if isinstance(pl, torch.Tensor):
        probe_terms.append(_float_tensor(pl).mean())

    hl = outs.get("height_prob_logit", None)
    if isinstance(hl, torch.Tensor):
        probe_terms.append(_float_tensor(hl).mean())

    loss = _sum_terms(probe_terms)
    if loss is None or not _requires_grad_scalar(loss):
        raise RuntimeError("Calibration surrogate loss does not require grad; check grad mode/graph.")

    return loss


def taylor_prune_and_slim_squeeze(cfg, dict_DB, group_ratios, calib_batches=8, device="cuda"):
    """
    1) Collect Taylor scores on the last conv of sq1/sq2/sq3 over a few calibration batches.
    2) Build new (slim) convs for those outputs using top-K channels.
    3) Fix in_channels for first convs in feat_combine[0], decoder[0], w1[0].conv1, w2[0].conv1.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: nn.Module = dict_DB["model"].to(device)


    # Only apply if squeeze group ratio > 0
    prune_ratio = float(group_ratios.get("squeeze", 0.0))
    if prune_ratio <= 0.0:
        print("No squeeze pruning requested; returning original model.")
        return model

    # 1) target the last convs in the three squeeze blocks
    targets = _get_squeeze_last_convs(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Make double-sure params track grads during calibration
    for p in model.parameters():
        p.requires_grad_(True)

    loader = _get_any_loader(dict_DB, cfg=cfg, device=device)  # prefers 'testloader'

    collector = TaylorCollector(targets, device=device)
    collector.attach()


    with torch.enable_grad():
        seen = 0
        for batch in loader:
            loss = _forward_with_loss(dict_DB, model, batch, device)
            model.zero_grad(set_to_none=True)
            loss.backward()
            seen += 1
            if seen >= calib_batches:
                break

    collector.detach()

    # 2) choose keep indices per squeeze (keep = (1 - prune_ratio))
    # all three squeeze blocks originally output self.c_feat2 channels
    # (per your model: self.c_feat2 = 64)
    keep_idx = {}
    for name, scores in collector.scores.items():
        if scores is None:
            raise RuntimeError(f"No Taylor scores collected for {name}.")
        keep_idx[name] = _pick_keep_indices(scores, keep_ratio=prune_ratio)  # keep top-K

    # 3) Rebuild:
    # 3a) shrink last conv+BN in each squeeze block
    # feat_squeeze1[3], feat_squeeze2[2], feat_squeeze3[1] are conv_bn_relu blocks
    sq1_block = model.feat_squeeze1[3]
    sq2_block = model.feat_squeeze2[2]
    sq3_block = model.feat_squeeze3[1]

    sq1_new_conv = _slice_conv2d_out(sq1_block.conv, keep_idx["sq1_last"])
    sq2_new_conv = _slice_conv2d_out(sq2_block.conv, keep_idx["sq2_last"])
    sq3_new_conv = _slice_conv2d_out(sq3_block.conv, keep_idx["sq3_last"])

    _replace_conv_bn_relu_block(sq1_block, sq1_new_conv)
    _replace_conv_bn_relu_block(sq2_block, sq2_new_conv)
    _replace_conv_bn_relu_block(sq3_block, sq3_new_conv)

    # 3b) fix the first convs that consume x_concat = cat([sq1, sq2, sq3], dim=1)
    # Build concatenated input-channel indices for these consumers
    c_each_orig = cfg.c_feat2 if hasattr(cfg, "c_feat2") else model.c_feat2  # 64
    cat_keep = _concat_keep_indices(
        keep_idx["sq1_last"].to(device),
        keep_idx["sq2_last"].to(device),
        keep_idx["sq3_last"].to(device),
        c_each_orig,
    )

    # feat_combine[0] is conv_bn_relu; slice its conv's *input*
    fc0_block = model.feat_combine[0]
    fc0_new_conv = _slice_conv2d_in(fc0_block.conv, cat_keep)
    fc0_block.conv = fc0_new_conv  # BN unchanged (out_channels same)

    # decoder[0] also consumes x_concat
    dec0_block = model.decoder[0]
    dec0_new_conv = _slice_conv2d_in(dec0_block.conv, cat_keep)
    dec0_block.conv = dec0_new_conv

    # w1[0].conv1 and w2[0].conv1 are Conv1d on concatenated lane-pooled features (C = c_feat2*3)
    w1_conv1 = model.w1[0]  # Conv1d
    w2_conv1 = model.w2[0]  # Conv1d

    # For Conv1d inputs, the "channels" axis matches the same concatenation order;
    # just slice input channels accordingly.
    w1_new_conv1 = _slice_conv1d_in(w1_conv1, cat_keep)
    w2_new_conv1 = _slice_conv1d_in(w2_conv1, cat_keep)
    model.w1[0] = w1_new_conv1
    model.w2[0] = w2_new_conv1

    slim_meta = {
        "type": "taylor_squeeze",
        "c_feat2_original": int(getattr(model, "c_feat2", 64)),
        "sq1_keep_idx": keep_idx["sq1_last"].detach().cpu().tolist(),
        "sq2_keep_idx": keep_idx["sq2_last"].detach().cpu().tolist(),
        "sq3_keep_idx": keep_idx["sq3_last"].detach().cpu().tolist(),
    }
    return model, slim_meta

def _l2_out_channel_importance(conv: nn.Conv2d) -> torch.Tensor:
    # [C_out, C_in, kH, kW] -> norm over (C_in,kH,kW) per out channel
    w = conv.weight.detach()
    oc = w.shape[0]
    if oc == 0:
        return torch.zeros(0, device=w.device)
    return w.view(oc, -1).norm(p=2, dim=1)

def prune_encoder_structured_l2(model: nn.Module, encoder_ratio: float, example_inputs: torch.Tensor):
    """
    Structured channel pruning inside ResNet encoder (layer2/3/4).
    MUST run with autograd enabled (no torch.no_grad).
    """
    if encoder_ratio <= 0.0:
        return model

    # Make sure autograd is ON and example_inputs participates in the graph
    with torch.enable_grad():
        example_inputs = example_inputs.requires_grad_(True)

        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

        stages = [model.encoder.layer2, model.encoder.layer3, model.encoder.layer4]
        for stage in stages:
            for block in stage:
                conv = getattr(block, "conv3", None) or getattr(block, "conv2", None)
                if not isinstance(conv, nn.Conv2d):
                    continue
                c_out = conv.out_channels
                pruned = int(round(encoder_ratio * c_out))
                if pruned <= 0 or pruned >= c_out:
                    continue

                # L2 importance per out-channel
                w = conv.weight.detach()
                score = w.view(c_out, -1).norm(p=2, dim=1)
                prune_idx = torch.argsort(score)[:pruned].tolist()

                group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, prune_idx)
                group.exec()

    return model


def run_prune_taylor_slim(cfg, dict_DB, group_ratios, suffix="", calib_batches=8):
    """
    End-to-end: Taylor scores → slim → save checkpoint (like your run_prune).
    Only prunes 'squeeze' group (safe path). You can extend similarly to more blocks.
    """
    print("\n🔧 Running Taylor-FO pruning + slimming (squeeze blocks) ...")
    model = dict_DB.get("model", None)
    if model is None:
        raise RuntimeError("❌ Model not loaded into dict_DB.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    slim_model, slim_meta = taylor_prune_and_slim_squeeze(
        cfg, dict_DB, group_ratios, calib_batches=calib_batches, device=device
    )

    out_dir = os.path.join(cfg.dir["weight"], "pruned")
    os.makedirs(out_dir, exist_ok=True)

    if not suffix:
        suffix = "taylor_" + "_".join([f"{k[:4]}{int(v*100)}" for k, v in group_ratios.items()])
    out_path = os.path.join(
        out_dir,
        f"checkpoint_tusimple_res_{cfg.backbone}_pruned_{suffix}"
    )
    slim_model_cpu = slim_model.to("cpu")

    torch.save({
        "epoch": 0,
        "val_result": 0.0,
        "model": slim_model_cpu.state_dict(),  # keep for backwards-compat
        "model_obj": slim_model_cpu,  # <-- full module object (Option A)
        # (optional) also keep slim_meta if you implemented Option B earlier
        # "slim_meta": slim_meta,
        "optimizer": torch.optim.Adam(slim_model_cpu.parameters()).state_dict(),
    }, out_path)
    print(f"✅ Saved slim model object (CPU) to: {out_path}")
    print(f"New sparsity (just for ref): {count_sparsity(slim_model.state_dict()):.2f}% (structured slimming won’t look sparse)")
    return out_path

def run_prune_encoder_and_squeeze(cfg, dict_DB, ratios, suffix="", calib_batches=8):
    """
    ratios: dict like {"encoder": 0.30, "squeeze": 0.20}
    1) clone baseline
    2) prune encoder channels (structured L2) with torch-pruning
    3) Taylor-FO prune+slim squeeze (your existing routine)
    4) save full model object (Option A) for painless loading
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = dict_DB.get("model", None)
    if base is None:
        raise RuntimeError("❌ Model not loaded into dict_DB.")

    model = copy.deepcopy(base).to(device).eval()

    # 1) Encoder structured pruning
    enc_ratio = float(ratios.get("encoder", 0.0))
    example_inputs = torch.randn(1, 3, int(cfg.height), int(cfg.width), device=device, requires_grad=True)
    with torch.enable_grad():
        model = prune_encoder_structured_l2(model, enc_ratio, example_inputs)


    # 2) Squeeze Taylor-FO prune+slim (reuse your function; it expects dict_DB['model'])
    tmp_db = {**dict_DB, "model": model}
    sq_ratio = float(ratios.get("squeeze", 0.0))
    if sq_ratio > 0.0:
        model, _meta = taylor_prune_and_slim_squeeze(
            cfg, {**tmp_db, "model": model}, {"squeeze": sq_ratio},
            calib_batches=calib_batches, device=device
        )
    else:
        _meta = None

    # 3) Save (Option A: save full object on CPU)
    out_dir = os.path.join(cfg.dir["weight"], "pruned")
    os.makedirs(out_dir, exist_ok=True)
    if not suffix:
        suffix = f"enc{int(enc_ratio*100)}_sq{int(sq_ratio*100)}"
    out_path = os.path.join(out_dir, f"checkpoint_tusimple_res_{cfg.backbone}_pruned_{suffix}")

    model_cpu = model.to("cpu")
    torch.save({
        "epoch": 0,
        "val_result": 0.0,
        "model": model_cpu.state_dict(),    # keep for bc
        "model_obj": model_cpu,             # Option A: painless load
    }, out_path)
    print(f"✅ Saved encoder+squeeze-pruned model to: {out_path}")
    return out_path
