import time
import torch

import numpy as np
import torchvision.utils as vutils
import os.path as osp
from time import strftime, localtime
import os
from torch.utils.tensorboard import SummaryWriter
from .common import is_list, is_tensor, ts2np, mkdir, Odict, NoOp
import logging
import wandb
import socket
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


class MessageManager:
    def __init__(self):
        self.info_dict = Odict()
        self.writer_hparams = ['image', 'scalar']
        self.time = time.time()
        self.cfgs = None

    def init_wandb(self, cfgs):
        self.cfgs = cfgs
        wandb_cfg = cfgs['trainer_cfg']['wandb']
        wandb.init(project=wandb_cfg['project_name'], name=wandb_cfg['exp_name'])
        wandb.watch_called = False
        wandb.config.address = get_host_ip() + ':' + os.getcwd()
        wandb.config.batch_size = cfgs['trainer_cfg']['sampler']['batch_size']
        wandb.config.dataset = cfgs['data_cfg']['dataset_name']
        wandb.config.loss_term_weight = cfgs['loss_cfg']
        wandb.config.lr = cfgs['optimizer_cfg']['lr']
        wandb.config.solver = cfgs['optimizer_cfg']['solver']
        wandb.config.weight_decay = cfgs['optimizer_cfg']['weight_decay']
        wandb.config.scheduler_cfg = cfgs['scheduler_cfg']['scheduler']
        wandb.config.milestone = cfgs['scheduler_cfg']['milestones']
        wandb.config.enable_float16 = cfgs['trainer_cfg']['enable_float16']
        wandb.config.total_iter = cfgs['trainer_cfg']['total_iter']
        wandb.config.model = cfgs['model_cfg']['model']
        



    def init_manager(self, save_path, log_to_file, log_iter, iteration=0):
        self.iteration = iteration
        self.log_iter = log_iter
        mkdir(osp.join(save_path, "summary/"))
        self.writer = SummaryWriter(
            osp.join(save_path, "summary/"), purge_step=self.iteration)
        self.init_logger(save_path, log_to_file)

    def init_logger(self, save_path, log_to_file):
        # init logger
        self.logger = logging.getLogger('opengait')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if log_to_file:
            mkdir(osp.join(save_path, "logs/"))
            vlog = logging.FileHandler(
                osp.join(save_path, "logs/", strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'))
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            self.logger.addHandler(vlog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        self.logger.addHandler(console)

    def append(self, info):
        for k, v in info.items():
            v = [v] if not is_list(v) else v
            v = [ts2np(_) if is_tensor(_) else _ for _ in v]
            info[k] = v
        self.info_dict.append(info)

    def flush(self):
        self.info_dict.clear()
        self.writer.flush()

    def write_to_tensorboard(self, summary):

        for k, v in summary.items():
            module_name = k.split('/')[0]
            if module_name not in self.writer_hparams:
                self.log_warning(
                    'Not Expected --Summary-- type [{}] appear!!!{}'.format(k, self.writer_hparams))
                continue
            board_name = k.replace(module_name + "/", '')
            writer_module = getattr(self.writer, 'add_' + module_name)
            v = v.detach() if is_tensor(v) else v
            v = vutils.make_grid(
                v, normalize=True, scale_each=True) if 'image' in module_name else v
            if module_name == 'scalar':
                try:
                    v = v.mean()
                except:
                    v = v
            writer_module(board_name, v, self.iteration)
    
    def write_to_wandb(self, summary):
        wandb.log(summary)

    def log_training_info(self):
        now = time.time()
        string = "Iteration {:0>5}, Cost {:.2f}s".format(
            self.iteration, now-self.time, end="")
        for i, (k, v) in enumerate(self.info_dict.items()):
            if 'scalar' not in k:
                continue
            k = k.replace('scalar/', '').replace('/', '_')
            end = "\n" if i == len(self.info_dict)-1 else ""
            string += ", {0}={1:.4f}".format(k, np.mean(v), end=end)
        self.log_info(string)
        self.reset_time()

    def reset_time(self):
        self.time = time.time()

    def train_step(self, info, summary):
        self.iteration += 1
        self.append(info)
        # print(summary.keys())
        if self.iteration % self.log_iter == 0:
            self.log_training_info()
            self.flush()
            # self.write_to_tensorboard(summary)
            if self.cfgs['trainer_cfg']['wandb']['enable_wandb']:
                self.write_to_wandb(summary)

    def log_debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def log_info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def log_warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)


msg_mgr = MessageManager()
noop = NoOp()


def get_msg_mgr():
    if torch.distributed.get_rank() > 0:
        return noop
    else:
        return msg_mgr
