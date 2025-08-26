import torch, torch.nn as nn, torch.nn.functional as F
#depricated?
@torch.no_grad()
def apply_2to4_sparsity_inplace(model: nn.Module):
    """
    Enforce NVIDIA 2:4 semi-structured sparsity on all Conv2d/Linear weights.
    Groups of 4 along the flattened input dimension: keep the 2 largest by |w|.
    """
    n_layers = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            W = m.weight.data
            orig_shape = W.shape
            # reshape to (out, in*kH*kW)
            W2d = W.view(W.shape[0], -1)
            out, cols = W2d.shape
            pad = (-cols) % 4
            if pad:
                W2d = F.pad(W2d, (0, pad))  # pad columns to multiple of 4
            Wg = W2d.view(out, -1, 4)             # (out, groups, 4)
            idx = Wg.abs().argsort(dim=2, descending=True)
            mask = torch.zeros_like(Wg)
            top2 = idx[:, :, :2]
            mask.scatter_(2, top2, 1.0)
            Wg.mul_(mask)
            W2d = Wg.view(out, -1)[:, :cols]      # drop pad
            m.weight.data.copy_(W2d.view(orig_shape))
            n_layers += 1
    print(f"✅ Applied 2:4 sparsity to {n_layers} layers.")
