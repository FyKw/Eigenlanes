import os
import numpy as np
from libs.utils import *
import time
import torch
import csv

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.testloader = dict_DB['testloader']
        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['post_process']
        self.eval_tusimple = dict_DB['eval_tusimple']
        self.visualize = dict_DB['visualize']
        self.csv_log_path = os.path.join(cfg.dir['out'], "test_log.csv")
        self.csv_layer_path = os.path.join(cfg.dir['out'], "test_layer_times.csv")  # NEW

        # Write CSV header only once (existing summary file, unchanged)
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "prune_config", "mode", "inference_time_sec",
                    "accuracy", "false_positives", "false_negatives"
                ])

        # NEW: per-stage averages file (separate to avoid breaking existing CSV)
        if not os.path.exists(self.csv_layer_path):
            with open(self.csv_layer_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "prune_config", "mode", "num_samples",
                    "h2d_ms", "encode_ms", "squeeze_ms", "feat_ms",
                    "predict_ms", "nms_init_ms", "nms_ms", "match_ms",
                    "post_update_ms", "post_run_ms",
                    "decode_ms", "visualize_ms"
                ])

    def init_data(self):
        self.result = {'out': {'mul': []}, 'gt': {'mul': []}, 'name': []}
        self.datalist = []
        self.stage_sums = {
            "h2d_ms": 0.0,
            "encode_ms": 0.0,
            "squeeze_ms": 0.0,
            "feat_ms": 0.0,
            "predict_ms": 0.0,
            "nms_init_ms": 0.0,
            "nms_ms": 0.0,
            "match_ms": 0.0,
            "post_update_ms": 0.0,
            "post_run_ms": 0.0,
            "decode_ms": 0.0,
            "visualize_ms": 0.0,
        }
        self.num_samples = 0

    def _time_block(self, fn):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            dt_ms = start.elapsed_time(end)
            return out, dt_ms
        else:
            import time
            t0 = time.perf_counter()
            out = fn()
            return out, (time.perf_counter() - t0) * 1000.0

    def run(self, model, mode='val', prune_config_str="default"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.init_data()

        torch.backends.cudnn.benchmark = True
        with torch.inference_mode():
            # Warmup
            for _ in range(10):
                _ = model(self.testloader.dataset[0]['img'].unsqueeze(0).to(device, non_blocking=True))
            torch.cuda.synchronize()

        with torch.no_grad():
            model.eval()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            for i, self.batch in enumerate(self.testloader):
                # H2D
                t0 = time.perf_counter()
                self.batch['img'] = self.batch['img'].to(device, non_blocking=True)
                self.batch['seg_label'] = self.batch['seg_label'].to(device, non_blocking=True)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                self.stage_sums["h2d_ms"] += (time.perf_counter() - t0) * 1000.0

                img_name = self.batch['img_name'][0]
                out = dict()

                # Forward stages (timed)
                _, dt = self._time_block(lambda: model.forward_for_encoding(self.batch['img']))
                self.stage_sums["encode_ms"] += dt

                _, dt = self._time_block(lambda: model.forward_for_squeeze())
                self.stage_sums["squeeze_ms"] += dt

                _, dt = self._time_block(lambda: model.forward_for_lane_feat_extraction())
                self.stage_sums["feat_ms"] += dt

                pred, dt = self._time_block(lambda: model.forward_for_lane_component_prediction())
                out.update(pred)
                self.stage_sums["predict_ms"] += dt

                init_nms, dt = self._time_block(lambda: self.forward_model.initialize_for_nms(
                    out, model, self.cfg.max_iter, self.cfg.thresd_nms_iou,
                    self.cfg.thresd_nms_iou_upper, thresd_score=self.cfg.thresd_score))
                out.update(init_nms)
                self.stage_sums["nms_init_ms"] += dt

                nms_out, dt = self._time_block(lambda: self.forward_model.run_for_nms(out, model))
                out.update(nms_out)
                self.stage_sums["nms_ms"] += dt

                match_out, dt = self._time_block(lambda: model.forward_for_matching(out['center_idx']))
                out.update(match_out)
                self.stage_sums["match_ms"] += dt

                # Post-process (split update/run to see where time goes)
                _, dt = self._time_block(lambda: self.post_process.update(self.batch, out, mode))
                self.stage_sums["post_update_ms"] += dt

                post, dt = self._time_block(lambda: self.post_process.run(out))
                out.update(post)
                self.stage_sums["post_run_ms"] += dt

                # Optional decode + visualize
                if self.cfg.disp_test_result:
                    if self.cfg.use_decoder:
                        dec_out, dt = self._time_block(lambda: model.forward_for_decoding())
                        out.update(dec_out)
                        self.stage_sums["decode_ms"] += dt
                    _, dt = self._time_block(lambda: self.visualize.display_for_test(
                        batch=self.batch, out=out, batch_idx=i, mode=mode))
                    self.stage_sums["visualize_ms"] += dt

                # Bookkeeping (unchanged)
                self.result['out']['mwcs'] = to_tensor(out['mwcs'])
                self.result['out']['mwcs_reg'] = to_tensor(out['mwcs_reg'])
                self.result['out']['mwcs_h_idx'] = to_tensor(out['mwcs_height_idx'])
                self.result['out']['mwcs_vp_idx'] = to_tensor(out['mwcs_reg_vp_idx'])
                self.result['name'] = img_name
                self.datalist.append(img_name)

                if self.cfg.save_pickle:
                    dir_name, file_name = os.path.split(img_name)
                    save_pickle(dir_name=os.path.join(self.cfg.dir['out'] + '{}/pickle/{}/'.format(mode, dir_name)),
                                file_name=file_name.replace('.jpg', ''), data=self.result)

                if i % 1000 == 1:
                    print(f'image {i} ---> {img_name} done!')

                self.num_samples += 1

            if self.cfg.save_pickle:
                save_pickle(dir_name=self.cfg.dir['out'] + mode + '/pickle/', file_name='datalist', data=self.datalist)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            process_time = time.time() - start

        # Evaluate + log results (existing CSV)
        metric = self.evaluation(mode, process_time, prune_config_str)

        # NEW: write per-stage averages (separate CSV, one row per model/config)
        self._write_stage_averages(mode, prune_config_str)

        return metric

    def _write_stage_averages(self, mode, prune_config_str):
        n = max(self.num_samples, 1)
        avg = {k: (v / n) for k, v in self.stage_sums.items()}
        with open(self.csv_layer_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                prune_config_str, mode, n,
                round(avg["h2d_ms"], 3),
                round(avg["encode_ms"], 3),
                round(avg["squeeze_ms"], 3),
                round(avg["feat_ms"], 3),
                round(avg["predict_ms"], 3),
                round(avg["nms_init_ms"], 3),
                round(avg["nms_ms"], 3),
                round(avg["match_ms"], 3),
                round(avg["post_update_ms"], 3),
                round(avg["post_run_ms"], 3),
                round(avg["decode_ms"], 3),
                round(avg["visualize_ms"], 3),
            ])

    def evaluation(self, mode, inference_time, prune_config_str="default"):
        acc, fp, fn = self.eval_tusimple.measure_accuracy(mode, mode_h=True)
        metric = {
            'acc': acc,
            'fp': fp,
            'fn': fn,
            'inference_time': inference_time
        }

        # Log to CSV (existing behavior)
        with open(self.csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                prune_config_str,
                mode,
                round(inference_time, 3),
                round(acc, 4),
                fp,
                fn
            ])

        print(f'⏱ Inference: {inference_time:.1f}s | Accuracy: {acc:.2f} | FP: {fp:.2f} | FN: {fn:.2f}')
        return metric
