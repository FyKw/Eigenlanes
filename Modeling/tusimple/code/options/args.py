import argparse

def _parse_float_list(s):
    if s is None or s == "":
        return []
    try:
        return [float(x) for x in s.split(",")]
    except Exception:
        raise argparse.ArgumentTypeError(f"Expected comma-separated floats, got: {s}")

def _parse_str_list(s):
    if s is None or s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default='test_paper', help='run mode (train, test, test_paper)')
    parser.add_argument('--pre_dir', type=str, default='--root/preprocessed/DATASET_NAME/', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default=None, help='dataset dir')
    parser.add_argument('--paper_weight_dir', default='--root/pretrained/DATASET_NAME/', help='pretrained weights dir (paper)')
    parser.add_argument('--weight_dir', default='../../../pretrained/tusimple/', help='pretrained weights dir (custom)')

    # GRID-ONLY pruning controls
    parser.add_argument('--grid', action='store_true', help='enable grid search over pruning ratios (grid-only mode)')
    parser.add_argument('--grid_layers', type=_parse_str_list, default="decoder,encoder,heads",
                        help='comma-separated layer groups to prune: decoder,encoder,heads,squeeze,combine')

    # Per-layer ratio lists (comma-separated floats, e.g. "0.0,0.05,0.15")
    parser.add_argument('--grid_decoder', type=_parse_float_list, default=None, help='decoder prune ratios')
    parser.add_argument('--grid_encoder', type=_parse_float_list, default=None, help='encoder prune ratios')
    parser.add_argument('--grid_heads',   type=_parse_float_list, default=None, help='heads prune ratios')
    parser.add_argument('--grid_squeeze', type=_parse_float_list, default=None, help='squeeze path ratios')
    parser.add_argument('--grid_combine', type=_parse_float_list, default=None, help='feat_combine ratios')

    args = parser.parse_args()
    cfg = args_to_config(cfg, args)
    return cfg

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
    if args.pre_dir is not None:
        cfg.dir['head_pre'] = args.pre_dir
        cfg.dir['weight'] = args.weight_dir
        cfg.dir['pre0_train'] = cfg.dir['pre0_train'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre0_test'] = cfg.dir['pre0_test'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre1'] = cfg.dir['pre1'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre2'] = cfg.dir['pre2'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre3'] = cfg.dir['pre3'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre4'] = cfg.dir['pre4'].replace('--preprocessed data path', args.pre_dir)

    cfg.dir['weight_paper'] = args.paper_weight_dir
    cfg.run_mode = args.run_mode

    # GRID configuration (grid-only; no single-run pruning)
    layers = args.grid_layers if isinstance(args.grid_layers, list) else _parse_str_list(args.grid_layers)
    layers = [l.lower() for l in layers]
    valid = {'decoder', 'encoder', 'heads', 'squeeze', 'combine'}
    layers = [l for l in layers if l in valid]

    # Compose per-layer lists and sanitize to [0,1]
    def _sanitize(lst, fallback):
        if lst is None:
            lst = fallback
        lst = [max(0.0, min(1.0, float(x))) for x in lst]
        return lst

    cfg.grid = {
        'enabled': bool(args.grid),
        'layers': layers,
        'ratios': {
            'decoder': _sanitize(args.grid_decoder, [0.0, 0.05]),
            'encoder': _sanitize(args.grid_encoder, [0.0]),
            'heads':   _sanitize(args.grid_heads,   [0.0]),
            'squeeze': _sanitize(args.grid_squeeze, [0.0]),
            'combine': _sanitize(args.grid_combine, [0.0]),
        }
    }

    # If grid is enabled but no layers selected, pick a safe default
    if cfg.grid['enabled'] and not cfg.grid['layers']:
        cfg.grid['layers'] = ['decoder']

    return cfg