import torch
import os
from networks.model import Model
from networks.loss import *
import torch.nn as nn
import torch.nn.functional as F

def load_model_for_test(cfg, dict_DB):
    # ---- pick checkpoint path ----
    if cfg.run_mode == 'test_paper':
        ckpt_path = os.path.join(cfg.dir['weight_paper'], f'checkpoint_tusimple_res_{cfg.backbone}')
    else:
        if cfg.param_name == 'trained_last':
            ckpt_path = os.path.join(cfg.dir['weight'], 'checkpoint_final')
        elif cfg.param_name == 'max':
            ckpt_path = os.path.join(cfg.dir['weight'], f'checkpoint_max_acc_tusimple_res_{cfg.backbone}')
        elif cfg.param_name == 'multi':
            ckpt_path = os.path.join(cfg.dir['weight'], 'pruned', cfg.dir['current'])
        else:
            raise ValueError(f"Unknown cfg.param_name: {cfg.param_name}")

    # ---- load to CPU (portable), then move to device explicitly ----
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preferred: full slimmed module
    if isinstance(ckpt, dict) and isinstance(ckpt.get('model_obj', None), torch.nn.Module):
        model = ckpt['model_obj']

        # 1) ALWAYS attach the current cfg (overwrite stale one)
        model.cfg = cfg

        # 2) RELOAD candidates from current cfg (since __init__ wasn’t run)
        if hasattr(model, 'reload_candidates_from_cfg'):
            model.reload_candidates_from_cfg()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()

        # move aux tensors to the correct device
        if hasattr(model, 'move_aux_to'):
            model.move_aux_to(device)

        ensure_channel_consistency(model, cfg, device)

        dict_DB['model'] = model
        return dict_DB

    # ---- Rare: module saved at top-level ----
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        if not hasattr(model, 'cfg'):
            model.cfg = cfg
        model = model.to(device).eval()
        if hasattr(model, 'move_aux_to'):
            model.move_aux_to(device)
        ensure_channel_consistency(model, cfg, device)
        dict_DB['model'] = model
        return dict_DB

    # ---- Legacy: state_dict for ORIGINAL (unslimmed) arch only ----
    model = Model(cfg=cfg)
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"ℹ️ load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device).eval()
    if hasattr(model, 'move_aux_to'):
        model.move_aux_to(device)
    # no need to ensure_channel_consistency for unpruned/original shapes
    dict_DB['model'] = model

    # more sanity check
    print("sf:", model.sf, "n_cand:", model.n_cand)
    print("cand_mask[sf0] device:", model.cand_mask[model.sf[0]].device)
    print("x_concat C:", getattr(model, 'x_concat', torch.empty(0)).shape[1] if hasattr(model, 'x_concat') else 'n/a')

    return dict_DB


import copy
import os
import torch
from networks.model import Model

def load_model_for_pruning(cfg, dict_DB):
    """
    Load the original, unpruned model as a CPU baseline for pruning.
    - dict_DB['model_base']: CPU baseline (deepcopy) used as the source for every grid combo
    - dict_DB['model']: CUDA eval copy (optional), in case callers need a device model
    """
    ckpt_path = os.path.join(cfg.dir['weight'], f'checkpoint_tusimple_res_{cfg.backbone}')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load raw checkpoint on CPU
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')

    print(f"✅ Checkpoint loaded from: {ckpt_path}")
    print(f"🔑 Checkpoint type: {type(ckpt)}")

    # Build fresh model and load state_dict
    model = Model(cfg=cfg)
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"ℹ️ load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    # Store a CPU baseline for the grid to copy from each time
    model.eval()
    dict_DB['model_base'] = copy.deepcopy(model).cpu()

    # Also provide a CUDA eval copy (optional)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_DB['model'] = model.to(device).eval()

    return dict_DB



def load_model_for_train(cfg, dict_DB):
    model = Model(cfg=cfg)
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=cfg.gamma)

    if cfg.resume == False:
        checkpoint = torch.load(cfg.dir['weight'] + 'checkpoint_final', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.milestones,
                                                         gamma=cfg.gamma,
                                                         last_epoch=checkpoint['epoch'])
        dict_DB['epoch'] = checkpoint['epoch'] + 1
        dict_DB['val_result'] = checkpoint['val_result']

    loss_fn = Loss_Function(cfg)

    dict_DB['model'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB




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
        new.weight[:k].copy_(bn.weight[:k])
        new.bias[:k].copy_(bn.bias[:k])
        if bn.track_running_stats:
            new.running_mean[:k].copy_(bn.running_mean[:k])
            new.running_var[:k].copy_(bn.running_var[:k])
    return new

def _build_x_concat(model, H, W, device):
    # minimal pass to materialize x_concat on the right device
    x = torch.randn(1, 3, H, W, device=device)
    model.forward_for_encoding(x)
    x1 = model.feat_squeeze1(model.feat[model.sf[0]])
    x2 = model.feat_squeeze2(model.feat[model.sf[1]]); x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
    x3 = model.feat_squeeze3(model.feat[model.sf[2]]); x3 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
    model.x_concat = torch.cat([x1, x2, x3], dim=1)
    return model.x_concat

def ensure_channel_consistency(model, cfg, device):
    model.to(device).eval()
    # 1) get current x_concat width
    x_concat = _build_x_concat(model, cfg.height, cfg.width, device)
    Cx = x_concat.shape[1]

    # 2) align feat_combine first conv (input = x_concat)
    fc0 = model.feat_combine[0]  # conv_bn_relu
    if fc0.conv.in_channels != Cx:
        fc0.conv = _resize_conv2d_in(fc0.conv, Cx)

    # 3) align decoder first conv (input = x_concat)
    dec0 = model.decoder[0]      # conv_bn_relu
    if dec0.conv.in_channels != Cx:
        dec0.conv = _resize_conv2d_in(dec0.conv, Cx)

    # 4) align w1, w2 (Conv1d-BN1d-ReLU-Conv1d) to Cx channels
    def _align_head(head: nn.Sequential):
        conv1, bn, relu, conv2 = head[0], head[1], head[2], head[3]
        if conv1.in_channels != Cx or conv1.out_channels != Cx:
            head[0] = _resize_conv1d_inout(conv1, Cx)
        if bn.num_features != Cx:
            head[1] = _resize_bn1d(bn, Cx)
        if conv2.in_channels != Cx or conv2.out_channels != Cx:
            head[3] = _resize_conv1d_inout(conv2, Cx)

    _align_head(model.w1)
    _align_head(model.w2)

    return model
