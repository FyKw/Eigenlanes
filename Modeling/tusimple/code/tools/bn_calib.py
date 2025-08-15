# tools/bn_calib.py
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def recalibrate_decoder_bn(model, data_loader, device, num_batches=200):
    """
    Update BN running stats ONLY inside `model.decoder` after structural pruning.
    Leaves encoder/squeeze stats intact.
    """
    # Collect decoder BNs
    dec_bns = []
    for name, m in model.decoder.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            dec_bns.append(m)

    if not dec_bns:
        return

    # Reset decoder BN stats and set ONLY decoder to train mode
    for bn in dec_bns:
        bn.reset_running_stats()
        bn.train()

    # Keep whole model eval, but BN layers we care about are in train mode
    model.eval()

    it = 0
    for batch in data_loader:
        imgs = batch['img'].to(device, non_blocking=True)

        # drive the same path that touches decoder BN
        model.forward_for_encoding(imgs)
        model.forward_for_squeeze()
        _ = model.decoder(model.x_concat)

        it += 1
        if it >= num_batches:
            break

    # back to eval for all BNs
    for bn in dec_bns:
        bn.eval()
