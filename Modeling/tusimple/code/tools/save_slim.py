# tools/save_slim.py
import torch

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

def save_slim_checkpoint(model, out_path, meta=None):
    """
    Save model_obj WITHOUT huge cand_* tensors. They will be rebuilt on load via reload_candidates_from_cfg().
    """
    aux = _pop_aux(model)
    try:
        torch.save({
            "model_obj": model.cpu(),              # portable module, but without cand_*
            "model": model.state_dict(),           # optional legacy
            "epoch": 0,
            "val_result": 0.0,
            "meta": meta or {}
        }, out_path)
    finally:
        _restore_aux(model, aux)
