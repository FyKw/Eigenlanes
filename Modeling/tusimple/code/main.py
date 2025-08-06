import os

from options.config import Config
from options.args import *
from tests.test import *
from trains.train import *
from libs.prepare import *
from tools.prune_model import run_prune, count_sparsity
from libs.load_model import load_model_for_pruning
import torch

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
    if 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
    if 'eval' in cfg.run_mode:
        main_eval(cfg, dict_DB)


if __name__ == '__main__':
    print("is CUDA Available:" + str(torch.cuda.is_available()))
    main()
