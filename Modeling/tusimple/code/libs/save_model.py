import torch
from libs.utils import *

def save_model(checkpoint, param, path):
    mkdir(path)
    torch.save({
        'epoch': checkpoint['epoch'],
        'val_result': checkpoint['val_result'],
        'model': checkpoint['model'].state_dict(),
        'optimizer': checkpoint['optimizer'].state_dict()},
        path + param)

def save_model_max(checkpoint, path, max_val, val, logger, logfile, model_name):
    if max_val < val:
        save_model(checkpoint, model_name, path)
        max_val = val
        logger("Epoch %03d => %s : %5f\n" % (checkpoint['epoch'], model_name, max_val), logfile)
        print(model_name)
    return max_val

def save_model_max_upper(checkpoint, path, max_val, val, val2, thresd, logger, logfile, model_name):
    mkdir(path)
    if max_val < val and val2 > thresd:
        save_model(checkpoint, model_name, path)
        max_val = val
        logger("Epoch %03d => %s : %5f\n" % (checkpoint['epoch'], model_name, max_val), logfile)
        print(model_name)
    return max_val

def _pop_aux(model):
    aux = {}
    for key in ("cand_c", "cand_iou", "cand_iou_upper", "cand_mask", "cand_area"):
        if hasattr(model, key):
            aux[key] = getattr(model, key)
            setattr(model, key, None)
    return aux

def _restore_aux(model, aux):
    for k, v in aux.items():
        setattr(model, k, v)

def save_slimmed_model(model, out_path, meta=None):
    """
    Save a physically slimmed model:
      - store the FULL MODULE as 'model_obj' (required because shapes changed),
      - strip cand_* tensors (they’ll be rebuilt via model.reload_candidates_from_cfg()),
      - optionally include a state_dict for legacy tools.
    """
    aux = _pop_aux(model)
    try:
        torch.save({
            "model_obj": model.cpu(),          # portable full module (no cand_*)
            "model": model.state_dict(),       # optional: legacy consumers
            "epoch": 0,
            "val_result": 0.0,
            "meta": meta or {},
        }, out_path)
    finally:
        _restore_aux(model, aux)