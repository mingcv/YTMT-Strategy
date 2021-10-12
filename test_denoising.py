import argparse
import glob

import cv2
import torchvision.transforms as vutils
import tqdm
from torch.autograd import Variable

import models
import util.index as index
from tools import saver, mutils
from util import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=75, help='noise level used on test set')
parser.add_argument('--icnn_path', type=str, default=None, help='icnn checkpoint to use.')
parser.add_argument("--inet", type=str, default="DnCNN", help='Model name')
parser.add_argument('--name', type=str, default='default', help='name of the experiment')
parser.add_argument('--log_path', type=str, default='logs/')
parser.add_argument('--saved_path', type=str, default='logs/')

opt = parser.parse_args()


def normalize(data):
    return data / 255.


def main():
    topil = vutils.ToPILImage()
    # Build model
    base_path = os.path.join(opt.log_path, opt.name, mutils.get_formatted_time())

    opt.saved_path = os.path.join(base_path, 'weights')
    opt.log_path = os.path.join(base_path, 'tensorboard')
    saver.base_url = os.path.join(base_path, 'samples')

    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    os.makedirs(saver.base_url, exist_ok=True)

    with open(os.path.join(base_path, 'args.txt'), mode='w') as fp:
        fp.write(' '.join(sys.argv))

    print('Loading model ...\n')
    net = getattr(models, opt.inet)
    net = net(in_channels=1, out_channels=1, num_of_layers=opt.num_of_layers, act=False)

    print(net)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.module.load_state_dict(torch.load(opt.icnn_path))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('./data/datasets/denoising', opt.test_data, '*.png'))
    files_source.sort()
    avg_meters = util.AverageMeters()
    for i, f in enumerate(tqdm.tqdm(files_source)):
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        idx = '%03d' % i
        with torch.no_grad():
            if isinstance(model.module, models.DnCNN):
                PNoise = model(INoisy)
                PImg_Sub = torch.clamp(INoisy - PNoise, 0., 1.)
            else:
                PImg, PNoise = model(INoisy)
                PImg_Sub = torch.clamp(INoisy - PNoise, 0., 1.)
                saver.save_image(PImg, f'PImg', split_dir=idx)
                PImg = torch.clamp(PImg, 0., 1.)

                res = index.quality_assess(np.array(topil(PImg[0].repeat(3, 1, 1))),
                                           np.array(topil(ISource[0].repeat(3, 1, 1))))
                avg_meters.update(res)

        saver.save_image(ISource, f'ISource', split_dir=idx)
        saver.save_image(noise, f'noise', split_dir=idx)
        saver.save_image(INoisy, f'INoisy', split_dir=idx)
        saver.save_image(PNoise, f'PNoise', split_dir=idx)
        saver.save_image(PImg_Sub, f'PImg_Sub', split_dir=idx)

    print("\nMeters on test data: ", avg_meters)


if __name__ == "__main__":
    main()
