import numpy as np
from libs.utils import *
import time
import torch
import csv
import hashlib
import os

def _tensor_checksum(t):
    # stable 64-bit checksum for a tensor's content on CPU
    if t is None: return "none"
    b = t.detach().float().cpu().numpy().tobytes()
    return hashlib.md5(b).hexdigest()[:8]

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.testloader = dict_DB['testloader']
        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['post_process']
        self.eval_tusimple = dict_DB['eval_tusimple']
        self.visualize = dict_DB['visualize']
        self.csv_log_path = os.path.join(cfg.dir['out'], "test_log.csv")

        # Write CSV header only once
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "prune_config", "mode", "inference_time_sec",
                    "accuracy", "false_positives", "false_negatives"
                ])

    def init_data(self):
        self.result = {'out': {'mul': []}, 'gt': {'mul': []}, 'name': []}
        self.datalist = []




    def run(self, model, mode='val', prune_config_str=""):
        def _log_once(msg, attr="_logged_cnt", limit=3):
            c = getattr(self, attr, 0)
            if c < limit:
                print(msg)
                setattr(self, attr, c + 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        self.init_data()

        # timing buckets
        t_encode = t_squeeze = t_lane = t_heads = t_nms = 0.0

        with torch.no_grad():
            start = time.time()
            for i, self.batch in enumerate(self.testloader):
                self.batch['img'] = self.batch['img'].to(device, non_blocking=True)
                self.batch['seg_label'] = self.batch['seg_label'].to(device, non_blocking=True)
                img_name = self.batch['img_name'][0]

                out = {}

                # 1) encoder
                t0 = time.time()
                model.forward_for_encoding(self.batch['img'])
                t_encode += time.time() - t0
                _log_once(f"[enc] feat scales: {[k for k in model.feat.keys()]}; "
                          f"shapes: {[tuple(v.shape) for v in model.feat.values()]}")

                # 2) squeeze
                t0 = time.time()
                model.forward_for_squeeze()
                t_squeeze += time.time() - t0
                _log_once(f"[sq] x_concat={tuple(model.x_concat.shape)}; "
                          f"sq_feat={tuple(model.sq_feat.shape)}")

                # 3) lane feat
                t0 = time.time()
                model.forward_for_lane_feat_extraction()
                t_lane += time.time() - t0
                _log_once(f"[lane] l_feat={tuple(model.l_feat.shape)}")

                # 4) heads
                t0 = time.time()
                out.update(model.forward_for_lane_component_prediction())
                t_heads += time.time() - t0
                _log_once(f"[heads] keys={sorted(list(out.keys()))}; "
                          f"prob={tuple(out['prob'].shape)}; height_prob={tuple(out['height_prob'].shape)}")

                # 5) NMS prep + run
                t0 = time.time()
                out.update(self.forward_model.initialize_for_nms(
                    out, model, self.cfg.max_iter, self.cfg.thresd_nms_iou,
                    self.cfg.thresd_nms_iou_upper, thresd_score=self.cfg.thresd_score))
                out.update(self.forward_model.run_for_nms(out, model))

                t_nms += time.time() - t0
                _log_once(f"[nms] keys now={sorted(list(out.keys()))}; "
                          f"has_nms={'nms' in out}; has_center_idx={'center_idx' in out}")

                _log_once(f"[nms] keys={sorted(out.keys())}  has_nms={'nms' in out}  "
                          f"center_idx={out.get('center_idx', None) if 'center_idx' in out else 'None'}")

                # 6) matching
                out.update(model.forward_for_matching(
                    out.get('center_idx', torch.zeros((1, 1), dtype=torch.long, device=device))))
                _log_once(f"[match] edge_map in out? {'edge_map' in out}")

                # (Optional) decoder: only affects visualization path
                if self.cfg.disp_test_result and self.cfg.use_decoder:
                    out.update(model.forward_for_decoding())
                    seg = out.get('seg_map', None)
                    if seg is not None:
                        _log_once(f"[dec] seg_map={tuple(seg.shape)}; "
                                  f"seg mean={seg.mean().item():.4f} min={seg.min().item():.4f} max={seg.max().item():.4f}")

                    # visualization (now resilient to missing 'nms')
                    self.visualize.display_for_test(batch=self.batch, out=out, batch_idx=i, mode=mode)

                # record minimal fields (if you need them)
                self.result['name'] = img_name
                self.datalist.append(img_name)

                if i % 1000 == 1:
                    print(f'image {i} ---> {img_name} done!')

            process_time = time.time() - start

        metric = self.evaluation(mode, inference_time=process_time)

        # quick fingerprints to show two models produce identical logits (optional)
        # sample the last batch heads for a checksum
        prob_logit = out.get('prob_logit', None)
        height_logit = out.get('height_prob_logit', None)
        chks_prob = _tensor_checksum(prob_logit)
        chks_h    = _tensor_checksum(height_logit)

        # print split timings
        print(f"[timing] encode={t_encode:.2f}s, squeeze={t_squeeze:.2f}s, lane={t_lane:.2f}s, "
              f"heads={t_heads:.2f}s, nms={t_nms:.2f}s, total={process_time:.2f}s")
        print(f"[fingerprints] prob_logit={chks_prob} height_logit={chks_h}")

        # write CSV row (append)
        csv_dir = os.path.join(self.cfg.dir['out'], 'metrics')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, 'prune_eval.csv')
        hdr = ['checkpoint','prune_config','acc','fp','fn','tot_time',
               't_encode','t_squeeze','t_lane','t_heads','t_nms',
               'prob_md5','height_md5']
        row = [
            getattr(self.cfg.dir, 'current', None) or getattr(self.cfg.dir, 'weight', ''),
            prune_config_str,
            metric.get('acc', 0), metric.get('fp', 0), metric.get('fn', 0),
            process_time, t_encode, t_squeeze, t_lane, t_heads, t_nms,
            chks_prob, chks_h
        ]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header: w.writerow(hdr)
            w.writerow(row)

        print(f"⏱ Inference: {process_time:.3f}s | Accuracy: {metric['acc']:.4f} | FP: {metric['fp']} | FN: {metric['fn']}")
        print(f"📝 Logged to: {csv_path}")
        return metric

    def evaluation(self, mode, inference_time, prune_config_str="default"):
        acc, fp, fn = self.eval_tusimple.measure_accuracy(mode, mode_h=True)
        metric = {
            'acc': acc,
            'fp': fp,
            'fn': fn,
            'inference_time': inference_time
        }

        # Log to CSV
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

        # Optional: print nicely
        print(f'⏱ Inference: {inference_time:.3f}s | Accuracy: {acc:.4f} | FP: {fp} | FN: {fn}')

        return metric
