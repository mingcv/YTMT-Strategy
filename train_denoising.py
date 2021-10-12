import argparse

import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models
from data.denoising_dataset import prepare_data, Dataset
from tools import saver, mutils
from util import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--inet", type=str, default="DnCNN", help='Model name')
parser.add_argument('--icnn_path', type=str, default=None, help='icnn checkpoint to use.')
parser.add_argument('--name', type=str, default='default', help='name of the experiment')

parser.add_argument('--log_path', type=str, default='logs/')
parser.add_argument('--saved_path', type=str, default='logs/')
parser.add_argument('--criterion2', type=str, default='MSELoss')
parser.add_argument('--ratio2', type=float, default=1.0)
opt = parser.parse_args()


def save_checkpoint(model, name):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.state_dict(), os.path.join(opt.saved_path, name))


def main():
    base_path = os.path.join(opt.log_path, opt.name, mutils.get_formatted_time())

    opt.saved_path = os.path.join(base_path, 'weights')
    opt.log_path = os.path.join(base_path, 'tensorboard')
    saver.base_url = os.path.join(base_path, 'samples')

    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    os.makedirs(saver.base_url, exist_ok=True)

    with open(os.path.join(base_path, 'args.txt'), mode='w') as fp:
        fp.write(' '.join(sys.argv))

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, name=opt.name)
    dataset_val = Dataset(train=False, name=opt.name)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = getattr(models, opt.inet)
    print(net)
    net = net(in_channels=1, out_channels=1, num_of_layers=opt.num_of_layers, act=False)

    print(net)

    # net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.log_path)
    step = 0
    noiseL_B = [0, 75]  # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            if isinstance(model.module, models.DnCNN):
                out_train = model(imgn_train)
                loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            else:
                out_train, out_train_b = model(imgn_train)
                loss1 = criterion(out_train, img_train) / (imgn_train.size()[0] * 2)
                loss2 = criterion(out_train_b, noise) / (imgn_train.size()[0] * 2)
                loss3 = criterion(out_train + out_train_b, imgn_train) / (imgn_train.size()[0] * 2)

                loss = loss1 + loss2 * 2 + loss3 * 3
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            if isinstance(model.module, models.DnCNN):
                out_train = torch.clamp(imgn_train - out_train, 0., 1.)
            else:
                out_train = torch.clamp(out_train, 0., 1.)

            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('learning_rate', current_lr, step)
            step += 1
        model.eval()
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
                imgn_val = img_val + noise
                img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
                if isinstance(model.module, models.DnCNN):
                    out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
                else:
                    Irecon, Imgn = model(imgn_val)
                    out_val = torch.clamp(Irecon, 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)

                if isinstance(model.module, models.DnCNN):
                    imgn_val = imgn_val.cpu().detach()
                    Nrecon = model(imgn_val).cpu().detach()
                    Irecon = torch.clamp(imgn_val - Nrecon, 0., 1.).cpu().detach()
                    Nrecon = torch.clamp(Nrecon, 0., 1.).cpu().detach()

                    Img = utils.make_grid(img_val, nrow=8, normalize=True, scale_each=True)
                    Imgn = utils.make_grid(imgn_val, nrow=8, normalize=True, scale_each=True)
                    Irecon = utils.make_grid(Irecon, nrow=8, normalize=True, scale_each=True)
                    Nrecon = utils.make_grid(Nrecon, nrow=8, normalize=True, scale_each=True)
                    writer.add_image('clean image', Img, epoch)
                    writer.add_image('noisy image', Imgn, epoch)
                    writer.add_image('reconstructed image', Irecon, epoch)
                    writer.add_image('reconstructed noise', Nrecon, epoch)
                else:
                    imgn_val = imgn_val.cpu().detach()

                    Irecon, Nrecon = model(imgn_val)
                    Irecon, Nrecon = Irecon.cpu().detach(), Nrecon.cpu().detach()
                    Irecon = torch.clamp(Irecon, 0., 1.).cpu().detach()
                    Nrecon = torch.clamp(Nrecon, 0., 1.).cpu().detach()

                    epc = '%03d' % epoch
                    idx = '%03d' % k

                    saver.save_image(img_val, f'ISource', split_dir=f'{epc}/{idx}')
                    saver.save_image(noise, f'noise', split_dir=f'{epc}/{idx}')
                    saver.save_image(imgn_val, f'INoisy', split_dir=f'{epc}/{idx}')
                    saver.save_image(Irecon, f'PImg', split_dir=f'{epc}/{idx}')
                    saver.save_image(Nrecon, f'PNoisy', split_dir=f'{epc}/{idx}')

            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # save model
        save_checkpoint(model, f'{opt.model}_{epoch}_{step}.pth')


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1,
                         name=opt.name)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2,
                         name=opt.name)
    main()
