import os

from options.net_options.train_options import TrainOptions
from engine import Engine
import torch
import data.intrinsic_dataset as datasets
from tools import mutils
import util.util as util

opt = TrainOptions().parse()
print(opt)
#torch.backends.cudnn.benchmark = True

opt.display_freq = 10
opt.debug = True

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

train_dataset_MITIntrinsic = datasets.MITIntrinsicDataset("./data_cover/",
                                                          train=True)
eval_dataset_MITIntrinsic = datasets.MITIntrinsicDataset("./data_cover/",
                                                         train=False)

train_dataloader_MITIntrinsic = datasets.DataLoader(train_dataset_MITIntrinsic, batch_size=opt.batchSize,
                                                    shuffle=not opt.serial_batches, num_workers=opt.nThreads,
                                                    pin_memory=True)

eval_dataloader_MITIntrinsic = datasets.DataLoader(eval_dataset_MITIntrinsic, batch_size=1, shuffle=False,
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
    engine.eval(eval_dataloader_MITIntrinsic, dataset_name='mit-intrinsic', savedir=save_dir, suffix='real20')

# define training strategy
engine.model.opt.lambda_gan = 0
# engine.model.opt.lambda_gan = 0.01
set_learning_rate(opt.lr)

while engine.epoch < 120:
    if opt.fixed_lr == 0:
        if engine.epoch >= 20:
            engine.model.opt.lambda_gan = 0.01  # gan loss is added after epoch 20
        if engine.epoch >= 60:
            set_learning_rate(opt.lr * 0.5)
        if engine.epoch >= 80:
            set_learning_rate(opt.lr * 0.2)
        if engine.epoch >= 100:
            set_learning_rate(opt.lr * 0.1)
    else:
        set_learning_rate(opt.fixed_lr)

    engine.train(train_dataloader_MITIntrinsic)
