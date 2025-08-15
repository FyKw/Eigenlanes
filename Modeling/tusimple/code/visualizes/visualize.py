import cv2
import math

import matplotlib.pyplot as plt
import numpy as np

from libs.utils import *

class Visualize_cv(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = {}

        # candidates
        self.candidates = load_pickle(self.cfg.dir['pre3'] + 'lane_candidates_' + str(self.cfg.n_clusters))
        self.cand_px_coord = to_np(self.candidates['px_coord'])
        self.cand_c = to_tensor(self.candidates['c'])
        self.U = load_pickle(self.cfg.dir['pre3'] + 'U')[:, :self.cfg.top_m]

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def update_label(self, label, name='label'):
        label = to_np(label)
        label = np.repeat(np.expand_dims(np.uint8(label != 0) * 255, axis=2), 3, 2)
        self.show[name] = label

    def update_data(self, data, name=None):
        self.show[name] = data

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name

    def b_map_to_rgb_image(self, data):
        data = np.repeat(np.uint8(to_np2(data.permute(1, 2, 0) * 255)), 3, 2)
        data = cv2.resize(data, (self.width, self.height))
        return data

    def draw_text(self, pred, label, name, ref_name='img', color=(255, 0, 0)):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        cv2.rectangle(img, (1, 1), (250, 120), color, 1)
        cv2.putText(img, 'pred : ' + str(pred), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, 'label : ' + str(label), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        self.show[name] = img

    def draw_polyline_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        pts = np.int32(data).reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False, color, s,
                            lineType=cv2.LINE_AA)
        self.show[name] = img

    def display_imglist(self, dir_name, file_name, list):
        # boundary line
        if self.show[list[0]].shape[0] != self.line.shape[0]:
            self.line = np.zeros((self.show[list[0]].shape[0], 3, 3), dtype=np.uint8)
            self.line[:, :, :] = 255
        disp = self.line

        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp)

    def display_for_train(self, batch, out, batch_idx):

        px_coord = dict()
        namelist = ['out_cls',
                    'out_cls_reg',
                    'out_nms',
                    'out_nms_reg',
                    'out_nms_reg_h',
                    'gt']

        self.update_image(batch['img'][0], name='img')
        self.update_image_name(batch['img_name'][0])

        self.show['seg_map'] = self.b_map_to_rgb_image(out['seg_map'][0])
        self.show['seg_map_gt'] = self.b_map_to_rgb_image(batch['seg_label'][0:1])

        idx_cls = to_np((out['prob'][0] > 0.5).nonzero()[:, 1])
        idx_s = idx_cls
        px_coord['out_cls'] = self.run_for_c_to_x_coord_conversion(idx=idx_cls)

        idx_nms = to_np(out['center_idx'][0])
        px_coord['out_nms'] = self.run_for_c_to_x_coord_conversion(idx=idx_nms)

        # out cls reg
        px_coord['out_cls_reg'] = self.run_for_c_to_x_coord_conversion(idx=idx_cls,
                                                                       offset=out['offset'][0][idx_s])
        # out nms reg
        px_coord['out_nms_reg'] = self.run_for_c_to_x_coord_conversion(idx=idx_nms,
                                                                       offset=out['offset'][0][idx_nms])
        # out nms reg h
        idx_h = torch.argmax(out['height_prob'][0][:, idx_nms], dim=0)
        height_node_idx = self.cfg.height_node_idx[to_np(idx_h)]
        px_coord['out_nms_reg_h'] = np.copy(px_coord['out_nms_reg'])

        # gt
        px_coord['gt'] = np.float32(to_np(batch['org_px_coord'][0][:batch['gt_num'][0]]))

        # draw polylines
        py_coord = np.float32(self.cfg.py_coord)
        for name in namelist:
            self.show[name] = np.copy(self.show['img'])
            for i in range(px_coord[name].shape[0]):

                if '_h' in name:
                    idx = height_node_idx[i]
                else:
                    idx = 0

                node_pts = np.concatenate((px_coord[name][i][idx:].reshape(-1, 1), py_coord[idx:].reshape(-1, 1)), axis=1)
                self.draw_polyline_cv(data=node_pts, name=name, ref_name=name, color=(0, 255, 0), s=3)

        save_namelist = ['img', 'seg_map', 'seg_map_gt'] + namelist
        # save result
        self.display_imglist(dir_name=self.cfg.dir['out'] + 'train/display/',
                             file_name=str(batch_idx) + '.jpg',
                             list=save_namelist)

    def display_for_train_edge_score(self, out, idx):
        py_coord = np.float32(self.cfg.py_coord)

        idx_node = to_np(out['node_idx'][0])
        px_coord_node = self.cand_px_coord[idx_node]
        self.show['node_set'] = np.copy(self.show['img'])
        for i in range(len(px_coord_node)):
            node_pts = np.concatenate((px_coord_node[i].reshape(-1, 1), py_coord.reshape(-1, 1)), axis=1)
            self.draw_polyline_cv(data=node_pts, name='node_set', ref_name='node_set', color=(0, 255, 0), s=3)

        pos_edge_idx = (out['gt_edge_map'][0] > 0).nonzero()
        for i in range(0, pos_edge_idx.shape[0], 3):

            px_coord = []
            n1 = pos_edge_idx[i, 0]
            n2 = pos_edge_idx[i, 1]
            idx1 = to_np(out['node_idx'][0][n1])
            idx2 = to_np(out['node_idx'][0][n2])
            px_coord.append(self.cand_px_coord[idx1])
            px_coord.append(self.cand_px_coord[idx2])

            self.show['pos_edge'] = np.copy(self.show['img'])
            for j in range(len(px_coord)):
                node_pts = np.concatenate((px_coord[j].reshape(-1, 1), py_coord.reshape(-1, 1)), axis=1)
                self.draw_polyline_cv(data=node_pts, name='pos_edge', ref_name='pos_edge', color=(0, 255, 0), s=3)

            self.draw_text(pred=np.round(to_np2(out['edge_map'][0][n1, n2]), 3),
                           label=np.round(to_np2(out['gt_edge_map'][0][n1, n2]), 3),
                           name='pos_edge', ref_name='pos_edge', color=(0, 255, 0))

            # save result
            self.display_imglist(dir_name=self.cfg.dir['out'] + 'train/display_pos_edge/',
                                 file_name=str(idx) + '_' + str(i) + '_edge.jpg',
                                 list=['img', 'node_set', 'pos_edge', 'gt'])

        neg_edge_idx = (out['gt_edge_map'][0] == 0).nonzero()
        for i in range(0, neg_edge_idx.shape[0], 10):

            px_coord = []
            n1 = neg_edge_idx[i, 0]
            n2 = neg_edge_idx[i, 1]
            if n1 == n2:
                continue
            # out masked line
            idx1 = to_np(out['node_idx'][0][n1])
            idx2 = to_np(out['node_idx'][0][n2])
            px_coord.append(self.cand_px_coord[idx1])
            px_coord.append(self.cand_px_coord[idx2])

            self.show['neg_edge'] = np.copy(self.show['img'])
            for j in range(len(px_coord)):
                node_pts = np.concatenate((px_coord[j].reshape(-1, 1), py_coord.reshape(-1, 1)), axis=1)
                self.draw_polyline_cv(data=node_pts, name='neg_edge', ref_name='neg_edge', color=(0, 255, 0), s=3)

            self.draw_text(pred=np.round(to_np2(out['edge_map'][0][n1, n2]), 3),
                           label=np.round(to_np2(out['gt_edge_map'][0][n1, n2]), 3),
                           name='neg_edge', ref_name='neg_edge', color=(0, 255, 0))

            # save result
            self.display_imglist(dir_name=self.cfg.dir['out'] + 'train/display_neg_edge/',
                                 file_name=str(idx) + '_' + str(i) + '_edge.jpg',
                                 list=['img', 'node_set', 'neg_edge', 'gt'])

    def display_for_test(self, batch, out, batch_idx, mode):
        import numpy as np
        import torch

        def _to_numpy(x):
            if x is None: return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        # names to draw and their source keys in `out`
        name_to_outkey = {
            'out_nms': 'nms',
            'out_nms_reg': 'nms_reg',
            'out_mwcs_reg': 'mwcs_reg',
            'out_mwcs_reg_gt': 'mwcs_reg',  # GT overlays reuse same coords
            'out_mwcs_reg_h_gt': 'mwcs_reg',  # … ditto, but uses height idx
        }
        namelist = list(name_to_outkey.keys())

        # --- update base tiles (image + label + name) ---
        self.update_image(batch['img'][0], name='img')
        self.update_label(batch['org_label'][0], name='org_label')
        self.update_image_name(batch['img_name'][0])

        # --- seg map is optional ---
        seg = out.get('seg_map', None)
        if seg is not None:
            # expects CHW tensor in [0,1] or logits; adapt if needed
            self.show['seg_map'] = self.b_map_to_rgb_image(seg[0])
        else:
            # create a placeholder so the grid saver still works
            self.show['seg_map'] = np.zeros_like(self.show['img'])

        # --- gather 2D coordinates safely ---
        px_coord = {}
        py_coord = np.float32(self.cfg.py_coord)  # (Hc,) vector

        for name, out_key in name_to_outkey.items():
            src = out.get(out_key, None)
            if src is None:
                # one-time warn per missing key
                warn_flag = f"_warn_missing_{out_key}"
                if not hasattr(self, warn_flag):
                    print(f"⚠️ visualize: '{out_key}' not in out — skipping {name}.")
                    setattr(self, warn_flag, True)
                continue

            src_np = _to_numpy(src)
            # normalize shapes: expect a list/array of lane polylines per candidate
            # typical shape: [num_lanes, Hc] or list of arrays with length Hc
            if isinstance(src_np, list):
                px_coord[name] = [np.asarray(a) for a in src_np]
            else:
                # assume shape (L, Hc) or (Hc,)
                arr = np.asarray(src_np)
                if arr.ndim == 1:
                    arr = arr[None, :]
                px_coord[name] = [arr[i] for i in range(arr.shape[0])]

        # --- draw each pane ---
        for name in namelist:
            # skip panes whose source was missing
            if name not in px_coord:
                continue

            # choose background
            if 'gt' in name:
                self.show[name] = np.copy(self.show['org_label'])
            else:
                self.show[name] = np.copy(self.show['img'])

            lanes = px_coord[name]
            if lanes is None or len(lanes) == 0:
                continue

            # choose height index per lane if requested
            h_idx_all = out.get('mwcs_height_idx', None)
            h_idx_all = _to_numpy(h_idx_all) if h_idx_all is not None else None

            for i, lane_x in enumerate(lanes):
                # lane_x expected shape (Hc,)
                lane_x = np.asarray(lane_x).reshape(-1)
                Hc = len(py_coord)
                if lane_x.shape[0] != Hc:
                    # align lengths if mismatch
                    m = min(Hc, lane_x.shape[0])
                    lane_x = lane_x[:m]
                    y = py_coord[:m]
                else:
                    y = py_coord

                # pick start index
                if '_h' in name and h_idx_all is not None and i < len(h_idx_all):
                    idx = int(h_idx_all[i])
                    if idx < 0: idx = 0
                    if idx >= len(y): idx = 0
                else:
                    idx = 0

                node_pts = np.concatenate(
                    (lane_x[idx:].reshape(-1, 1), y[idx:].reshape(-1, 1)),
                    axis=1
                )
                # draw
                self.draw_polyline_cv(
                    data=node_pts,
                    name=name,
                    ref_name=name,
                    color=(0, 255, 0),
                    s=3
                )

        # --- save grid ---
        self.display_imglist(
            dir_name=self.cfg.dir['out'] + mode + '/display/',
            file_name=str(batch_idx) + '.jpg',
            list=['img'] + [n for n in namelist if n in self.show] + ['seg_map']
        )

    def run_for_c_to_x_coord_conversion(self, idx, offset=None):
        if idx.shape[0] != 0:
            cand_c = self.cand_c[idx]
            if offset is not None:
                reg_c = cand_c + offset
            else:
                reg_c = cand_c
            px_coord = torch.matmul(self.U, reg_c.permute(1, 0)).permute(1, 0) * (self.width - 1)
            px_coord = to_np2(px_coord)
        else:
            px_coord = np.float32([])

        return px_coord