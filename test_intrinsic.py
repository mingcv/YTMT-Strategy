import os

from torch.utils.data import DataLoader

from options.errnet.train_options import TrainOptions
from intrinsic_engine import Engine
import torch
import data.intrinsic_dataset as datasets
from tools import mutils
import util.util as util

opt = TrainOptions().parse()
print(opt)
torch.backends.cudnn.benchmark = True

opt.isTrain = False
#cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 9999
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

eval_dataset_real = datasets.MITIntrinsicDataset("./data_cover/", train=False)
eval_dataloader_real = DataLoader(dataset=eval_dataset_real, batch_size=1, shuffle=False,
                                                   num_workers=opt.nThreads, pin_memory=True)
"""
    Main Loop
"""

engine = Engine(opt)

result_dir = os.path.join(f'./checkpoints/{opt.name}/results',
                          mutils.get_formatted_time())


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

if opt.resume or opt.debug_eval:
    save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.eval(eval_dataloader_real, dataset_name='MITIntrinsic', savedir=save_dir)

# define training strategy
engine.model.opt.lambda_gan = 0

