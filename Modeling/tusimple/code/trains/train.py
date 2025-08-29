import torch, copy, os
from libs.load_model import _normalize_val_result
from libs.save_model import *
from libs.utils import *

class Train_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']

        self.model = dict_DB['model']

        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']
        self.generator = dict_DB['generator']

        self.test_process = dict_DB['test_process']
        self.forward_model = dict_DB['forward_model']
        self.val_result = dict_DB['val_result']
        self.val_result = _normalize_val_result(self.val_result)

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']

        self.count = np.zeros(11, dtype=np.int32)

    def training(self):
        loss_t = dict()
        rmdir(path=self.cfg.dir['out'] + 'train/display_pos_edge/')
        rmdir(path=self.cfg.dir['out'] + 'train/display_neg_edge/')

        # train start
        self.model.train()
        print('train start')
        logger('train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):
            # load data
            for name in list(batch):
                if torch.is_tensor(batch[name]):
                    batch[name] = batch[name].cuda()

            # model
            out = dict()
            # ---------encoder--------- #
            self.model.forward_for_encoding(batch['img'])
            self.model.forward_for_squeeze()
            # --------SI module-------- #
            self.model.forward_for_lane_feat_extraction()
            out.update(self.model.forward_for_lane_component_prediction())
            # -----------NMS----------- #
            out.update(self.forward_model.initialize_for_nms(out, self.model, self.cfg.max_iter,
                                                             self.cfg.thresd_nms_iou, self.cfg.thresd_nms_iou_upper,
                                                             self.cfg.thresd_score))
            out.update(self.forward_model.run_for_nms(out, self.model))
            # --------IC module-------- #
            out.update(self.generator.run(batch, out))
            out.update(self.model.forward_for_matching(out['node_idx']))
            # ---------decoder--------- #
            if self.cfg.use_decoder == True:
                out.update(self.model.forward_for_decoding())

            self.count += out['edge_count']

            # loss
            loss = self.loss_fn(
                out=out,
                gt=batch
            )

            # optimize
            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()

            for l_name in loss:
                if l_name not in loss_t.keys():
                    loss_t[l_name] = 0
                loss_t[l_name] += loss[l_name].item()

            if i % self.cfg.disp_step == 0:
                print('img iter {} ==> {}'.format(i, batch['img_name'][0]))
                self.visualize.display_for_train(batch, out, i)
                self.visualize.display_for_train_edge_score(out, i)
                if i % self.cfg.disp_step == 0:
                    logger("%d ==> " % i, self.logfile)
                    for l_name in loss:
                        logger("Loss_{} : {}, ".format(l_name, round(loss[l_name].item(), 4)), self.logfile)
                    logger("||| {}\n".format(batch['img_name'][0]), self.logfile)

            if i > 2000 // self.cfg.batch_size['img']:
                break

        # logger
        logger('\nAverage Loss : ', self.logfile)
        print('\nAverage Loss : ', end='')
        for l_name in loss_t:
            logger("{}, ".format(round(loss_t[l_name] / i, 6)), self.logfile)
        for l_name in loss_t:
            print("{}, ".format(round(loss_t[l_name] / i, 6)), end='')

        logger("\nlabel distribution {}\n".format(np.round((self.count / np.sum(self.count)), 3) * 100), self.logfile)
        print("\nlabel distribution {}\n".format(np.round((self.count / np.sum(self.count)), 3) * 100))

        # --- after logging avg losses etc. ---

        # 1) Keep a runtime ckpt for validation() compatibility
        self.ckpt = {
            'epoch': self.epoch,
            'model': self.model,  # KEEP the live module here (as before)
            'optimizer': self.optimizer,
            'val_result': self.val_result,
        }

        # 2) On-disk save (our new shape-safe checkpoint) — unchanged
        ckpt_dir = self.cfg.dir['weight']
        os.makedirs(ckpt_dir, exist_ok=True)

        state_sd = self.model.state_dict()
        dev = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval();
        self.model.to('cpu')

        ckpt = {
            'epoch': self.epoch,
            'model': state_sd,
            'model_obj': self.model,  # full object for shape-changed resume
            'optimizer': self.optimizer.state_dict(),
            'val_result': self.val_result,
        }

        base_tag = getattr(self.cfg, "finetune_tag", f"model_e{self.epoch:03d}")
        out_subdir = getattr(self.cfg, "finetune_out_subdir", "finetuned")
        out_dir = os.path.join(self.cfg.dir['weight'], out_subdir)
        os.makedirs(out_dir, exist_ok=True)

   #     fname_epoch = f"{base_tag}__ft_e{self.epoch:03d}.pt"
        fname_latest = f"{base_tag}__ft_latest.pt"
    #    torch.save(ckpt, os.path.join(out_dir, fname_epoch))
        torch.save(ckpt, os.path.join(out_dir, fname_latest))
        print(f"💾 Saved finetuned: {os.path.join(out_dir, fname_latest)}")

        # restore live model
        self.model.to(dev)
        if was_training: self.model.train()

    def validation(self):
        metric = self.test_process.run(self.model, mode='val')

        logger("Epoch %3d ==> Validation result\n" % (self.ckpt['epoch']), self.logfile)
        logger("Epoch %03d => tusimple acc %5f / fp %5f / fn %5f \n " % (self.ckpt['epoch'], metric['acc'], metric['fp'], metric['fn']), self.logfile)
        name_a = 'acc'
        model_name = f'checkpoint_max_{name_a}_tusimple_res_{self.cfg.backbone}'
        self.val_result[name_a] = save_model_max(self.ckpt, self.cfg.dir['weight'],
                                                 self.val_result[name_a], metric[name_a],
                                                 logger, self.logfile, model_name)
    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch

            print('\nepoch %d\n' % epoch)
            logger("\nepoch %d\n" % epoch, self.logfile)

            self.training()
            if epoch > self.cfg.epoch_eval:
                self.validation()
            elif epoch < 60 and epoch % 10 == 1:
                self.validation()
            elif 60 <= epoch and epoch < self.cfg.epoch_eval and epoch % 10 == 1:
                self.validation()
            self.scheduler.step(self.epoch)
