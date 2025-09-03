import os, glob

from options.config import Config
from options.args import *
from tests.test import *
from trains.train import *
from libs.prepare import *
from tools.prune_model import run_prune, count_sparsity, run_prune_encoder_and_squeeze
from libs.load_model import load_model_for_pruning, load_model_for_test, load_model_for_quant, load_model_for_train, load_model_from_checkpoint_for_prune
from tools.quant import *
import torch
from itertools import product
from tools.export_onnx import export_onnx
from tools.bench_onnx import bench_onnx_cuda
from datasets.dataset_tusimple import Dataset_Train

def _ensure_trainloader_local(cfg, dict_DB):
    if 'trainloader' in dict_DB:
        return dict_DB
    dataset_train = Dataset_Train(cfg=cfg)
    trainloader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=cfg.batch_size['img'],
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    dict_DB['trainloader'] = trainloader
    return dict_DB

def _ft_latest_path_for_source(cfg, source_ckpt_path):
    tag = os.path.splitext(os.path.basename(source_ckpt_path))[0]
    return os.path.join(cfg.dir['weight'], 'finetuned', f"{tag}__ft_latest")

def _finetune_once(cfg, dict_DB, ckpt_path, extra_epochs=5, new_lr=None, freeze_encoder=False):
    # tell the loader what to resume from
    cfg.resume_from = ckpt_path
    cfg.resume = True

    # >>> add these two:
    cfg.finetune_out_subdir = "finetuned"                     # where to save
    cfg.finetune_tag = os.path.splitext(os.path.basename(ckpt_path))[0]  # name seed

    # make sure we have a trainloader
    dict_DB = _ensure_trainloader_local(cfg, dict_DB)

    # (re)load model/optimizer/scheduler/loss
    dict_DB = load_model_for_train(cfg, dict_DB)
    model = dict_DB['model']

    # optional LR override
    if new_lr is not None:
        for g in dict_DB['optimizer'].param_groups:
            g['lr'] = float(new_lr)

    # optional stage freezing
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # run only N more epochs from current starting epoch
    start_epoch = dict_DB['epoch']
    cfg.epochs = start_epoch + int(extra_epochs)

    # wire a Test_Process (validation)
    dict_DB['test_process'] = Test_Process(cfg, dict_DB)

    # train
    trainer = Train_Process(cfg, dict_DB)
    trainer.run()

    # unfreeze if needed
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = True

    return dict_DB

def run_finetune_from_paths(cfg, dict_DB, ckpt_paths, extra_epochs=5, new_lr=1e-4, freeze_encoder=False):
    """Fine-tune a list of explicit checkpoint files (full paths)."""
    for i, ckpt in enumerate(ckpt_paths, 1):
        if not os.path.exists(ckpt):
            print(f"⚠️ Skip (not found): {ckpt}")
            continue
        print(f"\n[{i}/{len(ckpt_paths)}] Fine-tuning: {ckpt}")
        dict_DB = _finetune_once(cfg, dict_DB, ckpt, extra_epochs, new_lr, freeze_encoder)
    return dict_DB

def run_finetune_from_dirs(cfg, dict_DB, dirs, pattern="checkpoint_tusimple*", extra_epochs=5, new_lr=1e-4, freeze_encoder=False):
    """Scan one or more directories for checkpoints by pattern, then fine-tune each."""
    all_ckpts = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"⚠️ Skip dir (not found): {d}")
            continue
        hits = sorted(glob.glob(os.path.join(d, pattern)))
        all_ckpts.extend(hits)
    if not all_ckpts:
        print("⚠️ No checkpoints matched; nothing to finetune.")
        return dict_DB

    return run_finetune_from_paths(cfg, dict_DB, all_ckpts, extra_epochs, new_lr, freeze_encoder)
def main_eval(cfg, dict_DB):
    test_process = Test_Process(cfg, dict_DB)
    test_process.evaluation(mode='test')

def main_test(cfg, dict_DB):
    test_process = Test_Process(cfg, dict_DB)
    print(f"Running test on model with sparsity: {count_sparsity(dict_DB['model'].state_dict()):.2f}%")
    test_process.run(dict_DB['model'], mode='test')

def main_train(cfg, dict_DB):
    dict_DB['test_process'] = Test_Process(cfg, dict_DB)
    train_process = Train_Process(cfg, dict_DB)
    train_process.run()

def main_prune(cfg, dict_DB):
    ratios = {"encoder": 0.0, "squeeze": 0.2}
    dict_DB = load_model_for_pruning(cfg, dict_DB)
    run_prune_encoder_and_squeeze(cfg, dict_DB, ratios, suffix="")

def quant_info(cfg, dict_DB):
    dict_DB = load_model_for_quant(cfg, dict_DB)
    model = dict_DB.get("model", None)
    summarize_dtypes(model)
    weight_stats(model)

def quant_fp_and_bf(cfg, dict_DB):
    dict_DB = load_model_for_quant(cfg, dict_DB)
    base = dict_DB.get("model", None)

    fp16_m = to_half_inference(base)
    save_quant_variant(cfg, fp16_m, "fp16")

    bf16_m = to_bf16_inference(base)
    save_quant_variant(cfg, bf16_m, "bf16")

def multi_quant(cfg, dict_DB):
    quant_dir =  os.path.join(cfg.dir['weight'], 'quant')
    model_files = [file for file in os.listdir(quant_dir) if file.startswith('checkpoint_tusimple')]

    for i, file in enumerate(model_files, start=1):
        print(f"Processing model {i}/{len(model_files)}: {file}")

        cfg.dir['current'] = file

        dict_DB = load_model_for_test(cfg, dict_DB)
        prune_config_str = file.replace("checkpoint_tusimple_res_18_quant_", "")

        test_process = Test_Process(cfg, dict_DB)
        test_process.run(dict_DB['model'], mode='test', prune_config_str=prune_config_str)

def multi_pruned(cfg, dict_DB):
    pruned_dir = os.path.join(cfg.dir['weight'], 'pruned')
    model_files = [file for file in os.listdir(pruned_dir) if file.startswith('checkpoint_tusimple')]

    for i, file in enumerate(model_files, start=1):
        print("-------------------------------------------------------")
        print(f"Processing model {i}/{len(model_files)}: {file}")

        cfg.dir['current'] = file
        dict_DB = load_model_for_test(cfg, dict_DB)

        prune_config_str = file.replace("checkpoint_tusimple_res_18_pruned_", "")

        test_process = Test_Process(cfg, dict_DB)
        test_process.run(dict_DB['model'], mode='test', prune_config_str=prune_config_str)

def run_auto_dir_grid(cfg, dict_DB):
    """
    Directory-wide grid:
      For each seed checkpoint found in START_DIRS:
        For each per-iteration ratio combo (RATIO_GRID):
          Repeat for LOOPS:
            prune (per-iteration on remaining) -> TEST
            finetune -> TEST
    """
    # ---- knobs live here (edit in-code) ----
    START_DIRS = [os.path.join(cfg.dir['weight'], 'seeds')]
    FILE_PATTERN = "checkpoint_tusimple*"
    LOOPS          = 5                                            # repeats per combo
    RATIO_GRID     = {                                            # per-iteration prune ratios
        "encoder": [0.0, 0.1],
        "squeeze": [0.0, 0.1],
    }
    FT_EPOCHS      = 2
    FT_LR          = 1e-4
    FREEZE_ENCODER = False

    # ---------------------------------------

    # collect seeds
    seeds = []

    # snapshot once; exclude anything that looks generated
    seeds = [
        s for s in seeds
        if "__" not in os.path.basename(s)  # our outputs have __G / __L / __ft
    ]
    print(f"Seeds: {len(seeds)} found")

    for d in START_DIRS:
        if os.path.isdir(d):
            seeds += sorted(glob.glob(os.path.join(d, FILE_PATTERN)))
    if not seeds:
        print("⚠️ No starting checkpoints found in dirs:", START_DIRS)
        return

    # build ratio combinations (skip the all-zero case)
    keys = list(RATIO_GRID.keys())
    combos = []
    for vals in product(*[RATIO_GRID[k] for k in keys]):
        ratios = dict(zip(keys, vals))
        if all(v == 0.0 for v in ratios.values()):
            continue
        combos.append(ratios)

    # ensure we can train & test
    dict_DB = _ensure_trainloader_local(cfg, dict_DB)
    tester  = Test_Process(cfg, dict_DB)

    for si, seed in enumerate(seeds, 1):
        print(f"\n====================  SEED [{si}/{len(seeds)}]  {seed}  ====================")
        for ci, ratios in enumerate(combos, 1):
            print(f"\n---- combo [{ci}/{len(combos)}]: {ratios} ----")
            # start each combo from the original seed (not previous combo’s result)
            current = seed

            for L in range(1, LOOPS + 1):
                # 1) PRUNE current
                dict_DB = load_model_from_checkpoint_for_prune(cfg, dict_DB, current)
                enc = int(100 * float(ratios.get("encoder", 0.0)))
                sq  = int(100 * float(ratios.get("squeeze", 0.0)))
                seed_tag = os.path.splitext(os.path.basename(current))[0]
                suffix   = f"{seed_tag}__G{ci}_L{L}_enc{enc}_sq{sq}"
                print(f"\n🪚 Loop {L}/{LOOPS} | per-iter ratios: {ratios} | suffix={suffix}")
                pruned_path = run_prune_encoder_and_squeeze(cfg, dict_DB, ratios, suffix=suffix)
                # ^ make sure run_prune_encoder_and_squeeze RETURNS the saved path

                # TEST (pruned)
                cfg.dir['current'] = pruned_path
                cfg.param_name     = 'multi'
                dict_DB = load_model_for_test(cfg, dict_DB)
                tester.run(dict_DB['model'], mode='test', prune_config_str=f"{suffix}__pruned")

                # 2) FINETUNE
                print(f"🎯 Finetuning: {pruned_path}")
                _ = _finetune_once(cfg, dict_DB, pruned_path, extra_epochs=FT_EPOCHS,
                                   new_lr=FT_LR, freeze_encoder=FREEZE_ENCODER)
                ft_latest = _ft_latest_path_for_source(cfg, pruned_path)

                # TEST (finetuned)
                cfg.dir['current'] = ft_latest
                cfg.param_name     = 'multi'
                dict_DB = load_model_for_test(cfg, dict_DB)
                tester.run(dict_DB['model'], mode='test', prune_config_str=f"{suffix}__finetuned")

                # next iteration continues from the fine-tuned result
                current = ft_latest

def run_onnx(cfg, dict_DB, iters=200, warmup=50, precision="fp32"):
    """
    Export all pruned checkpoints to <weight_dir>/onnx/ and benchmark them.
    Logs results to <weight_dir>/onnx/onnx_bench.csv
    """
    pruned_dir = os.path.join(cfg.dir['weight'], 'pruned')
    onnx_dir   = os.path.join(cfg.dir['weight'], 'onnx')
    os.makedirs(onnx_dir, exist_ok=True)
    csv_log_path = os.path.join(onnx_dir, "onnx_bench.csv")

    files = [f for f in os.listdir(pruned_dir) if f.startswith('checkpoint_tusimple')]
    if not files:
        print(f"⚠️ No pruned checkpoints found in {pruned_dir}")
        return

    for i, file in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] ONNX export+bench for: {file}")
        # load pruned checkpoint (your loader uses cfg.param_name='multi')
        cfg.dir['current'] = file
        cfg.param_name = 'multi'
        dict_DB = load_model_for_test(cfg, dict_DB)
        model = dict_DB['model']

        base = file  # filenames in your setup often have no extension
        onnx_path = os.path.join(onnx_dir, base + (f"_{precision}.onnx" if precision else ".onnx"))

        # export (precision: "fp32" now; later you can try "fp16")
        export_onnx(model, cfg.height, cfg.width, onnx_path, precision=precision)

        # bench (auto-match dtype in ONNX; pass model_name for logs)
        bench_onnx_cuda(
            onnx_path, cfg.height, cfg.width,
            iters=iters, warmup=warmup, fp16=None,
            csv_log_path=csv_log_path, model_name=base
        )


def run_group_grid_prune(cfg, dict_DB):
    encoder_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # layer-level encoder ratios
    squeeze_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # layer-level squeeze ratios

    for enc_r, sq_r in product(encoder_grid, squeeze_grid):
        ratios = {"encoder": enc_r, "squeeze": sq_r}
        suffix = f"enc{int(enc_r*100)}_sq{int(sq_r*100)}"
        print(f"\n🔍 Testing config: {ratios}")

        dict_DB = load_model_for_pruning(cfg, dict_DB)
        run_prune_encoder_and_squeeze(cfg, dict_DB, ratios, suffix=suffix)


def main():
    cfg = Config()
    cfg = parse_args(cfg)

    FINETUNE_FILES = [
        # examples:
        # os.path.join(cfg.dir['weight'], 'pruned', 'checkpoint_tusimple_res_18_pruned_enc30_sq5'),
        # os.path.join(cfg.dir['weight'], 'pruned', 'checkpoint_tusimple_res_18_pruned_enc40_sq10'),
    ]
    FINETUNE_DIRS = [
        os.path.join(cfg.dir['weight'], 'pruned'),
        # later add: os.path.join(cfg.dir['weight'], 'quant'), etc.
    ]

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)
    dict_DB = prepare_model(cfg, dict_DB)
    dict_DB = prepare_evaluation(cfg, dict_DB)
    dict_DB = prepare_post_processing(cfg, dict_DB)
    dict_DB = prepare_generator(cfg, dict_DB)
    dict_DB = prepare_training(cfg, dict_DB)

    print("Default CUDA device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")


    if "model" in dict_DB:
        model = dict_DB["model"]
        print("Model is on device:", next(model.parameters()).device)

    if 'quant_A' in cfg.run_mode:
        quant_fp_and_bf(cfg, dict_DB)
    if 'info_quant' in cfg.run_mode:
        quant_info(cfg, dict_DB)
    if 'prune' in cfg.run_mode:
        main_prune(cfg, dict_DB)
    if 'multi_pruned' in cfg.run_mode:
        multi_pruned(cfg, dict_DB)
    if 'multi_quant' in cfg.run_mode:
        multi_quant(cfg, dict_DB)
    if 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
    if 'eval' in cfg.run_mode:
        main_eval(cfg, dict_DB)
    if 'grid' in cfg.run_mode:
        run_group_grid_prune(cfg, dict_DB)
    if 'onnx' in cfg.run_mode:
        run_onnx(cfg, dict_DB)
    if 'finetune_files' in cfg.run_mode:
        run_finetune_from_paths(cfg, dict_DB, FINETUNE_FILES, extra_epochs=5, new_lr=1e-4, freeze_encoder=False)
    if 'finetune_dirs' in cfg.run_mode:
        # will scan the listed dirs for "checkpoint_tusimple*" files and finetune all matches
        run_finetune_from_dirs(cfg, dict_DB, FINETUNE_DIRS, pattern="checkpoint_tusimple*", extra_epochs=5, new_lr=1e-4, freeze_encoder=False)
    if 'auto' in cfg.run_mode:
        run_auto_dir_grid(cfg, dict_DB)


if __name__ == '__main__':
    print("is CUDA Available:" + str(torch.cuda.is_available()))
    main()
