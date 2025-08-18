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


def main_prune(cfg, dict_DB):
    prune_ratio = 0.05
    dict_DB = load_model_for_pruning(cfg, dict_DB)

    run_prune(cfg, dict_DB, prune_ratio)

def long_run(cfg, dict_DB):
    steps = [i * 0.05 for i in range(20)]

    dict_DB = load_model_for_pruning(cfg, dict_DB)
    for prune_ratio in steps:
        run_prune(cfg, dict_DB, prune_ratio)


def multi(cfg, dict_DB):
    pruned_dir = os.path.join(cfg.dir['weight'], 'pruned')
    model_files = [file for file in os.listdir(pruned_dir) if file.startswith('checkpoint_tusimple')]

    for i, file in enumerate(model_files, start=1):
        print(f"Processing model {i}/{len(model_files)}: {file}")

        cfg.dir['current'] = file

        dict_DB = load_model_for_test(cfg, dict_DB)

        # Extract config string from filename (e.g., "_enc10_dec5_cls5_reg0.pt")
        prune_config_str = file.replace("checkpoint_tusimple_res_18_pruned_", "")

        # Initialize Test_Process
        test_process = Test_Process(cfg, dict_DB)

        # Print sparsity
        sparsity = count_sparsity(dict_DB['model'].state_dict())
        print(f"Running test on model '{file}' with sparsity: {sparsity:.2f}%")

        # Run the test with config string
        test_process.run(dict_DB['model'], mode='test', prune_config_str=prune_config_str)


def run_group_grid_prune(cfg, dict_DB):
    # small, illustrative grid (expand if you want)
    ratios = [0.0, 0.2, 0.3, 0.4]

    ratio_options = {
        "encoder": ratios,
        "decoder": ratios,
        "squeeze": ratios,
 #       "heads":   ratios,
    }

    keys = list(ratio_options.keys())
    value_combinations = list(product(*[ratio_options[k] for k in keys]))

    dict_DB = load_model_for_pruning(cfg, dict_DB)

    for combo in value_combinations:
        group_ratios = dict(zip(keys, combo))
        suffix = "_".join([f"{k[:4]}{int(v * 100)}" for k, v in group_ratios.items()])
        print(f"\n🔍 Testing config: {group_ratios}")
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

    if 'prune' in cfg.run_mode:
        main_prune(cfg, dict_DB)
    if 'long_run' in cfg.run_mode:
        long_run(cfg, dict_DB)
    if 'multi' in cfg.run_mode:
        multi(cfg, dict_DB)
    if 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
    if 'eval' in cfg.run_mode:
        main_eval(cfg, dict_DB)
    if 'grid' in cfg.run_mode:
        run_group_grid_prune(cfg, dict_DB)



if __name__ == '__main__':
    print("is CUDA Available:" + str(torch.cuda.is_available()))
    main()
