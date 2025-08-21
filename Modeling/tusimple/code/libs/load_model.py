import torch
import os
from networks.model import Model
from networks.loss import *

def load_model_for_test(cfg, dict_DB):
    # pick the checkpoint as you already do...
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(cfg.dir['weight_paper'] + f'checkpoint_tusimple_res_{cfg.backbone}', map_location="cpu")
    else:
        if cfg.param_name == 'trained_last':
            checkpoint = torch.load(cfg.dir['weight'] + 'checkpoint_final', map_location="cpu")
        elif cfg.param_name == 'max':
            checkpoint = torch.load(cfg.dir['weight'] + f'checkpoint_max_acc_tusimple_res_{cfg.backbone}', map_location="cpu")
        elif cfg.param_name == 'multi':
            checkpoint = torch.load(cfg.dir['weight'] + 'pruned/' + cfg.dir['current'], map_location="cpu", weights_only=False)

    # === Option A: if a full model object is present, use it ===
    if isinstance(checkpoint, dict) and "model_obj" in checkpoint:
        model = checkpoint["model_obj"]
        model = model.cuda().eval()
        dict_DB['model'] = model
        return dict_DB

    # === Fallback: old-style checkpoints (state_dict only) ===
    model = Model(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)  # or True if shapes match
    model = model.cuda().eval()
    dict_DB['model'] = model
    return dict_DB



def load_model_for_pruning(cfg, dict_DB):
    checkpoint_path = os.path.join(cfg.dir['weight'], f'checkpoint_tusimple_res_{cfg.backbone}')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    print(f"✅ Checkpoint loaded from: {checkpoint_path}")
    print(f"🔑 Checkpoint type: {type(checkpoint)}")

    model = Model(cfg=cfg)

    # Since it's a raw state_dict, load it directly
    model.load_state_dict(checkpoint['model'], strict=False)

    model.cuda()
    model.eval()
    dict_DB["model"] = model

    return dict_DB

def load_model_for_quant(cfg, dict_DB):
    checkpoint_path = os.path.join(cfg.dir['weight'], f'checkpoint_tusimple_res_{cfg.backbone}')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    print(f"✅ Checkpoint loaded from: {checkpoint_path}")
    print(f"🔑 Checkpoint type: {type(checkpoint)}")

    model = Model(cfg=cfg)

    # Since it's a raw state_dict, load it directly
    model.load_state_dict(checkpoint['model'], strict=False)

    model.cuda()
    model.eval()
    dict_DB["model"] = model

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