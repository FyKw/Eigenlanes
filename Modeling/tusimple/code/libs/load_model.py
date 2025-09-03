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
            cand = cfg.dir.get('current', '')
            # NEW: allow absolute file path
            if os.path.isabs(cand) and os.path.exists(cand):
                checkpoint = torch.load(cand, weights_only=False)
            else:
                # keep your previous behavior (e.g., from pruned/)
                checkpoint = torch.load(cfg.dir['weight'] + 'pruned/' + cand, weights_only=False)

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

def _normalize_val_result(v):
    """
    Accept old formats (float) and new (dict). Always return {'acc': float}.
    """
    if isinstance(v, dict):
        out = dict(v)
        out['acc'] = float(out.get('acc', 0.0))
        return out
    elif isinstance(v, (int, float)):
        return {'acc': float(v)}
    else:
        return {'acc': 0.0}

def load_model_from_checkpoint_for_prune(cfg, dict_DB, ckpt_path):
    """Load a checkpoint (pruned/finetuned/anything) into dict_DB['model'] for pruning."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if 'model_obj' in ckpt:
        model = ckpt['model_obj']
        # make sure cfg is attached (some pickles might not have it)
        model.cfg = cfg
    else:
        model = Model(cfg=cfg)
        model.load_state_dict(ckpt['model'], strict=False)
    model = model.cuda().eval()
    dict_DB['model'] = model
    return dict_DB

def _new_optimizer_and_sched(cfg, model):
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=cfg.gamma)
    return optimizer, scheduler

def _safe_load_optimizer(optimizer, sd):
    try:
        optimizer.load_state_dict(sd)
        return True
    except Exception as e:
        print(f"⚠️ Optimizer state not loaded (shape changed): {e}")
        return False

def _safe_last_epoch(scheduler, last_epoch):
    try:
        scheduler.last_epoch = last_epoch
    except Exception as e:
        print(f"⚠️ Could not set scheduler.last_epoch: {e}")

def load_model_for_train(cfg, dict_DB):
    """
    Priority:
      1) If cfg.resume_from provided:
           - load checkpoint
           - if 'model_obj' exists -> use it (shape-safe)
           - else try state_dict onto a fresh Model(cfg) (may fail if shapes differ)
           - optionally resume optimizer/scheduler if cfg.resume
      2) Else (no resume_from):
           - warm-start from '<weight>/checkpoint_final' if available
           - else random init
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _attach(model):
        model.cfg = cfg  # ensure cfg on the loaded object
        model = model.to(device)
        model.train()
        return model

    model = None
    optimizer = None
    scheduler = None
    start_epoch = 0
    val_result = dict()

    ckpt_path = None
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        ckpt_path = cfg.resume_from
    else:
        # optional warm-start if no explicit resume_from
        default_path = os.path.join(cfg.dir['weight'], 'checkpoint_final')
        if os.path.exists(default_path):
            ckpt_path = default_path

    if ckpt_path:
        print(f"🔄 Loading training checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if 'model_obj' in ckpt:
            # ✅ safest for pruned/slimmed shapes
            model = ckpt['model_obj']
            model = _attach(model)
        else:
            # fallback: init fresh arch and try state_dict
            model = _attach(Model(cfg=cfg))
            try:
                model.load_state_dict(ckpt['model'], strict=True)
            except Exception as e:
                print("❌ state_dict load failed (shape mismatch). "
                      "Please resume from a checkpoint that contains 'model_obj'.\n", e)
                raise

        # (re)build optimizer/scheduler on this model
        optimizer, scheduler = _new_optimizer_and_sched(cfg, model)

        if cfg.resume:
            # try load optimizer/scheduler states; if shapes changed, keep fresh ones
            if 'optimizer' in ckpt:
                _safe_load_optimizer(optimizer, ckpt['optimizer'])
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
                _safe_last_epoch(scheduler, ckpt['epoch'])
            if 'val_result' in ckpt:
                val_result = ckpt['val_result']

    else:
        # no checkpoint available → start fresh (or warm-start later yourself)
        print("ℹ️ No checkpoint provided; starting from scratch.")
        model = Model(cfg=cfg).to(device).train()
        optimizer, scheduler = _new_optimizer_and_sched(cfg, model)
        start_epoch = 0
        val_result = _normalize_val_result({'acc': 0.0})

    loss_fn = Loss_Function(cfg).to(device)

    val_result = _normalize_val_result(val_result)

    dict_DB['model'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn
    dict_DB['epoch'] = start_epoch
    dict_DB['val_result'] = val_result
    return dict_DB
