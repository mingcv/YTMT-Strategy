import os
from os.path import join

import torch.backends.cudnn as cudnn

import data.sirs_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils

opt = TrainOptions().parse()
print(opt)
cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 20
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# modify the following code to 
datadir = os.path.join(os.path.expanduser('~'), 'datasets/reflection-removal')

datadir_syn = join(datadir, 'train/VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'train/real')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('data/VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True,
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True, if_align=opt.if_align)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_real = datasets.CEILTestDataset(join(datadir, f'test/real20_{opt.real20_size}'),
                                             fns=read_fns('data/real_test.txt'), if_align=opt.if_align)
eval_dataset_solidobject = datasets.CEILTestDataset(join(datadir, 'test/SIR2/SolidObjectDataset'),
                                                    if_align=opt.if_align)
eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'test/SIR2/PostcardDataset'), if_align=opt.if_align)
eval_dataset_wild = datasets.CEILTestDataset(join(datadir, 'test/SIR2/WildSceneDataset'), if_align=opt.if_align)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

"""Main Loop"""
engine = Engine(opt)
result_dir = os.path.join(f'./checkpoints/{opt.name}/results',
                          mutils.get_formatted_time())


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


if opt.resume:
    save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.eval(eval_dataloader_real, dataset_name='testdata_real20', savedir=save_dir, suffix='real20')
    engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=save_dir,
                suffix='solidobject')
    engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=save_dir, suffix='postcard')
    engine.eval(eval_dataloader_wild, dataset_name='testdata_wild', savedir=save_dir, suffix='wild')

set_learning_rate(opt.lr)
while engine.epoch < 120:
    engine.model.opt.lambda_gan = 0.01  # gan loss is added after epoch 20
    engine.train(train_dataloader_fusion)

    if engine.epoch % 1 == 0:
        save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
        os.makedirs(save_dir, exist_ok=True)
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20', savedir=save_dir, suffix='real20')
        engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=save_dir,
                    suffix='solidobject')
        engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=save_dir, suffix='postcard')
        engine.eval(eval_dataloader_wild, dataset_name='testdata_wild', savedir=save_dir, suffix='wild')
