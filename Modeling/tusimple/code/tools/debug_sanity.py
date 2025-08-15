# tools/debug_sanity.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def debug_one_batch(model, loader, device):
    batch = next(iter(loader))
    img = batch['img'].to(device)
    model.eval()

    # encode + squeeze
    model.forward_for_encoding(img)
    model.forward_for_squeeze()
    x_concat = model.x_concat
    print(f"x_concat: {tuple(x_concat.shape)} device={x_concat.device}")

    # check masks & areas
    sf0 = model.sf[0]
    m = model.cand_mask[sf0]
    a = model.cand_area[sf0]
    print(f"cand_mask[{sf0}]: {tuple(m.shape)} device={m.device} | cand_area[{sf0}]: {tuple(a.shape)} device={a.device}")

    # lane feat (from sq_feat) + heads
    model.forward_for_lane_feat_extraction()
    l_feat = model.l_feat
    print(f"l_feat: {tuple(l_feat.shape)}")

    out = model.forward_for_lane_component_prediction()
    p = out['prob']                 # (B, 1, N)
    hp = out['height_prob']         # (B, Hc, N)
    print(f"prob: {tuple(p.shape)} logit_mean={out['prob_logit'].mean().item():.4f} | "
          f"height_prob: {tuple(hp.shape)} ent={-(hp*hp.clamp_min(1e-8).log()).sum(dim=1).mean().item():.3f}")

    # run the NMS init + run (should produce center_idx)
    from_idx = model.cfg.max_iter if hasattr(model.cfg, 'max_iter') else 100
    # your project’s forward_model uses batch/out; rely on your dict_DB wiring in real run
