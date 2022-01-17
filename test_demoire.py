import os
from os.path import join

import torch.backends.cudnn as cudnn

import data.sirs_dataset as datasets
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False

datadir = os.path.join(os.path.expanduser('~'), 'datasets/demoire', 'eval')

eval_dataset = datasets.MoireDataset(datadir, phase='eval')
# eval_dataset = datasets.RealDataset(test_real_dir)
eval_dataloader = datasets.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=opt.nThreads,
                                      pin_memory=True)

engine = Engine(opt)
# engine.save_eval(label='ytmt_ucs_demoire_opt_086_00860000')
# exit(0)
"""Main Loop"""
result_dir = os.path.join('./results', opt.name, mutils.get_formatted_time())

res = engine.eval(eval_dataloader, savedir=join(result_dir, 'demoire'), dataset_name='LCD2019')
print(res)
