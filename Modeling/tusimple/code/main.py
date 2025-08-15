import os

from options.config import Config
from options.args import *
from tests.test import *
from trains.train import *
from libs.prepare import *
from tools.prune_model import run_prune, count_sparsity
from libs.load_model import load_model_for_pruning, load_model_for_test
import torch
from itertools import product

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

def multi(cfg, dict_DB):
    pruned_dir = os.path.join(cfg.dir['weight'], 'pruned')
    model_files = sorted([f for f in os.listdir(pruned_dir) if f.endswith('.pt') and f.startswith('checkpoint_tusimple')])

    if not model_files:
        print(f"⚠️ No checkpoints found in: {pruned_dir}")
        return

    for i, file in enumerate(model_files, start=1):
        path = os.path.join(pruned_dir, file)

        # Peek once to see if it has model_obj; if not, we still try load_model_for_test (legacy support).
        try:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            has_obj = isinstance(ck, dict) and isinstance(ck.get('model_obj', None), torch.nn.Module)
        except Exception as e:
            print(f"⚠️ Skipping {file}: load error: {e}")
            continue

        print(f"\nProcessing model {i}/{len(model_files)}: {file} | model_obj={has_obj}")
        cfg.dir['current'] = file

        # Load (loader prefers model_obj, rejects slimmed state_dicts)
        dict_DB = load_model_for_test(cfg, dict_DB)
        model = dict_DB['model']

        # sanity check
        print("x_concat C:", model.x_concat.shape[1])
        print("feat_combine[0].conv.in:", model.feat_combine[0].conv.in_channels)
        print("decoder[0].conv.in:", model.decoder[0].conv.in_channels)

        # Extract neat config string
        prefix = f"checkpoint_tusimple_res_{cfg.backbone}_pruned_"
        prune_config_str = file[len(prefix):-3] if file.startswith(prefix) and file.endswith('.pt') else file

        # Sparsity + run
        sparsity = count_sparsity(model.state_dict())
        print(f"Running test on model '{file}' with sparsity: {sparsity:.2f}%")
        print(f"use_decoder={cfg.use_decoder}, disp_test_result={cfg.disp_test_result}")

        test_process = Test_Process(cfg, dict_DB)
        test_process.run(model, mode='test', prune_config_str=prune_config_str)
    if not model_files:
        print("⚠️ No slimmed checkpoints with 'model_obj' found in:", pruned_dir)
        return


def run_group_grid_prune(cfg, dict_DB):
    # Ensure grid is configured
    if not getattr(cfg, 'grid', {}).get('enabled', False):
        print("⚠️ Grid not enabled. Pass --grid to enable grid pruning.")
        return

    # Load baseline model once inside the loader (not here)
    dict_DB = load_model_for_pruning(cfg, dict_DB)

    layers = cfg.grid.get('layers', [])
    ratio_lists = cfg.grid.get('ratios', {})

    # Build a list of (layer_name, ratios_list) for selected layers
    selected = [(name, ratio_lists.get(name, [0.0])) for name in layers]
    if not selected:
        print("⚠️ No layers selected for grid. Nothing to do.")
        return

    # Prepare Cartesian product over selected layers
    names = [n for n, _ in selected]
    lists = [lst for _, lst in selected]

    for combo in product(*lists):
        group_ratios = dict(zip(names, combo))
        # Create a readable suffix, e.g. dec5_enc0_hd0_sq0_fc0
        parts = []
        alias = {'decoder': 'dec', 'encoder': 'enc', 'heads': 'hd', 'squeeze': 'sq', 'combine': 'fc'}
        for n in names:
            v = group_ratios[n]
            parts.append(f"{alias[n]}{int(v*100)}")
        suffix = "_".join(parts)

        print(f"\n🔍 Grid combo: {group_ratios}")
        run_prune(cfg, dict_DB, group_ratios, suffix=suffix)


def main():
    cfg = Config()
    cfg = parse_args(cfg)

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

    # Pruning now goes through grid-only path
    if 'prune' in cfg.run_mode or 'grid' in cfg.run_mode:
        run_group_grid_prune(cfg, dict_DB)
    if 'multi' in cfg.run_mode:
        multi(cfg, dict_DB)
    if 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
    if 'eval' in cfg.run_mode:
        main_eval(cfg, dict_DB)




if __name__ == '__main__':
    print("is CUDA Available:" + str(torch.cuda.is_available()))
    main()
