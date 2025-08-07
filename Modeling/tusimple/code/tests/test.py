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

    def run(self, model, mode='val', prune_config_str="default"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.init_data()

        with torch.no_grad():
            model.eval()
            torch.cuda.synchronize()
            start = time.time()

            for i, self.batch in enumerate(self.testloader):
                self.batch['img'] = self.batch['img'].to(device)
                self.batch['seg_label'] = self.batch['seg_label'].to(device)

                img_name = self.batch['img_name'][0]

                out = dict()
                model.forward_for_encoding(self.batch['img'])
                model.forward_for_squeeze()
                model.forward_for_lane_feat_extraction()
                out.update(model.forward_for_lane_component_prediction())
                out.update(self.forward_model.initialize_for_nms(out, model, self.cfg.max_iter, self.cfg.thresd_nms_iou, self.cfg.thresd_nms_iou_upper, thresd_score=self.cfg.thresd_score))
                out.update(self.forward_model.run_for_nms(out, model))
                out.update(model.forward_for_matching(out['center_idx']))

                self.post_process.update(self.batch, out, mode)
                out.update(self.post_process.run(out))

                if self.cfg.disp_test_result:
                    if self.cfg.use_decoder:
                        out.update(model.forward_for_decoding())
                    self.visualize.display_for_test(batch=self.batch, out=out, batch_idx=i, mode=mode)

                self.result['out']['mwcs'] = to_tensor(out['mwcs'])
                self.result['out']['mwcs_reg'] = to_tensor(out['mwcs_reg'])
                self.result['out']['mwcs_h_idx'] = to_tensor(out['mwcs_height_idx'])
                self.result['out']['mwcs_vp_idx'] = to_tensor(out['mwcs_reg_vp_idx'])
                self.result['name'] = img_name
                self.datalist.append(img_name)

                if self.cfg.save_pickle:
                    dir_name, file_name = os.path.split(img_name)
                    save_pickle(dir_name=os.path.join(self.cfg.dir['out'] + '{}/pickle/{}/'.format(mode, dir_name)), file_name=file_name.replace('.jpg', ''), data=self.result)

                if i % 1000 == 1:
                    print(f'image {i} ---> {img_name} done!')

            if self.cfg.save_pickle:
                save_pickle(dir_name=self.cfg.dir['out'] + mode + '/pickle/', file_name='datalist', data=self.datalist)

            torch.cuda.synchronize()
            process_time = time.time() - start

        # Evaluate + log results
        metric = self.evaluation(mode, process_time, prune_config_str)
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
