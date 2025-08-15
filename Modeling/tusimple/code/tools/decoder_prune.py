# tools/decoder_prune.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

@torch.no_grad()
def _warmup_for_decoder(model, H, W, device):
    x = torch.randn(1, 3, H, W, device=device)
    model.forward_for_encoding(x)
    x1 = model.feat_squeeze1(model.feat[model.sf[0]])
    x2 = model.feat_squeeze2(model.feat[model.sf[1]])
    x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
    x3 = model.feat_squeeze3(model.feat[model.sf[2]])
    x3 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
    model.x_concat = torch.cat([x1, x2, x3], dim=1)  # keep 192 ch
    _ = model.decoder(model.x_concat)

def _forward_fn(model, inputs):
    # inputs arrives as a tuple from torch-pruning — unpack
    x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    model.forward_for_encoding(x)
    x1 = model.feat_squeeze1(model.feat[model.sf[0]])
    x2 = model.feat_squeeze2(model.feat[model.sf[1]])
    x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
    x3 = model.feat_squeeze3(model.feat[model.sf[2]])
    x3 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
    model.x_concat = torch.cat([x1, x2, x3], dim=1)
    return model.decoder(model.x_concat)

def _l1_smallest_out_channels(conv: nn.Conv2d, n_prune: int):
    """
    Select indices of the n_prune smallest-L1 output channels.
    conv.weight shape: [out_ch, in_ch, kh, kw]
    """
    with torch.no_grad():
        scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))  # [out_ch]
        idx = torch.argsort(scores)[:n_prune].tolist()
    return idx

def prune_decoder_only_out_channels(model: nn.Module, ratio_per_layer: float, H: int, W: int, device):
    """
    Physically prune ONLY decoder conv OUT channels for layers [1..3].
    Keeps x_concat width (192) intact and avoids changing encoder/heads.
    """
    model.to(device).eval()
    _warmup_for_decoder(model, H, W, device)

    example = torch.randn(1, 3, H, W, device=device)
    DG = tp.DependencyGraph().build_dependency(
        model,
        example_inputs=(example,),
        forward_fn=_forward_fn
    )

    # tools/decoder_prune.py  (only the pruning loop shown; keep the rest as you have it)

    total_removed = 0

    # Only prune decoder blocks [1..3] (skip [0] to keep x_concat interface; usually skip [4] 1x1)
    targets = []
    for idx in [1, 2, 3]:
        block = model.decoder[idx]
        if hasattr(block, "conv") and isinstance(block.conv, nn.Conv2d):
            targets.append(block.conv)

    for conv in targets:
        out_ch = conv.out_channels
        n_prune = int(out_ch * ratio_per_layer)
        if n_prune <= 0 or n_prune >= out_ch:
            continue

        idxs_to_remove = _l1_smallest_out_channels(conv, n_prune)  # [int indices]

        # 1.x API → get_pruning_group + exec()
        group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs_to_remove)
        if group is None:
            continue
        # (Optional safety) if hasattr(DG, "check_pruning_group") and not DG.check_pruning_group(group): continue
        group.exec()
        total_removed += len(idxs_to_remove)

    return model, total_removed

