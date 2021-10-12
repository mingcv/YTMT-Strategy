import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


class YTMTNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)

        if len(self.gpu_ids) > 0:  # transfer data into gpu
            if not isinstance(input, tuple):
                input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])

        self.input = input

        self.target_t = target_t
        self.target_r = target_r
        self.data_name = data_name

        self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True

    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            out_l1, out_r1, out_l2, out_r2 = self.forward(mode='eval')

            out_l1 = tensor2im(out_l1)
            out_r1 = tensor2im(out_r1)
            out_l2 = tensor2im(out_l2)
            out_r2 = tensor2im(out_r2)
            target = tensor2im(self.target_t)
            target_r = tensor2im(self.target_r)

            if self.aligned:
                # res1 = index.quality_assess(out_l1, target)
                res = index.quality_assess(out_l2, target)

                # res = res1 if res1['PSNR'] > res2['PSNR'] else res2

            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    savedir = join(savedir, suffix, name)
                    os.makedirs(savedir, exist_ok=True)

                    Image.fromarray(out_l1.astype(np.uint8)).save(
                        join(savedir, '{}_l1.png'.format(self.opt.name)))
                    Image.fromarray(out_r1.astype(np.uint8)).save(
                        join(savedir, '{}_r1.png'.format(self.opt.name)))

                    Image.fromarray(out_l2.astype(np.uint8)).save(
                        join(savedir, '{}_l2.png'.format(self.opt.name)))
                    Image.fromarray(out_r2.astype(np.uint8)).save(
                        join(savedir, '{}_r2.png'.format(self.opt.name)))

                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 't_label.png'))
                    Image.fromarray(target_r.astype(np.uint8)).save(join(savedir, 'r_label.png'))
                    if isinstance(self.input, tuple):
                        Image.fromarray(tensor2im(self.input[0]).astype(np.uint8)).save(
                            join(savedir, 'input_l_stage2.png'))

                        Image.fromarray(tensor2im(self.input[1]).astype(np.uint8)).save(
                            join(savedir, 'input_r_stage2.png'))
                    else:
                        Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(
                            join(savedir, 'm_input.png'))

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return

        with torch.no_grad():
            out_l1, out_r1, out_l2, out_r2 = self.forward()
            out_l1 = tensor2im(out_l1)
            out_r1 = tensor2im(out_r1)
            out_l2 = tensor2im(out_l2)
            out_r2 = tensor2im(out_r2)
            if self.data_name is not None and savedir is not None:
                Image.fromarray(out_l1.astype(np.uint8)).save(join(savedir, name, '{}_l1.png'.format(self.opt.name)))
                Image.fromarray(out_r1.astype(np.uint8)).save(join(savedir, name, '{}_r1.png'.format(self.opt.name)))
                Image.fromarray(out_l2.astype(np.uint8)).save(join(savedir, name, '{}_l2.png'.format(self.opt.name)))
                Image.fromarray(out_r2.astype(np.uint8)).save(join(savedir, name, '{}_r2.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class TwoStageYTMTNetModel(YTMTNetBase):
    def name(self):
        return 'twostage_ytmtnet'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_stage2 = None

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i1)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i1.eval()
        self.net_i2.eval()
        torch.cuda.empty_cache()

    def _train(self):
        self.net_i1.train()
        self.net_i2.train()
        torch.cuda.empty_cache()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None

        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472

        self.net_i1 = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        self.net_i2 = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        networks.init_weights(self.net_i1, init_type=opt.init_type)  # using default initialization as EDSR
        networks.init_weights(self.net_i2, init_type=opt.init_type)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1, 0.1, 0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1, 0.1, 0.1, 0.1], indices=[8, 13, 22, 31],
                                                criterions=[losses.CX_loss] * 3 + [nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G1 = torch.optim.Adam(self.net_i1.parameters(),
                                                 lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
            self.optimizer_G2 = torch.optim.Adam(self.net_i2.parameters(),
                                                 lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G1, self.optimizer_G2])

        if opt.resume:
            self.load(self, opt.resume_epoch)

        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        loss_D_1, pred_fake_1, pred_real_1 = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i_stage2, self.target_t)

        loss_D_2, pred_fake_2, pred_real_2 = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_j_stage2, self.target_t)

        self.loss_D, self.pred_fake, self.pred_real = (
            loss_D_1, pred_fake_1, pred_real_1) if loss_D_1.item() < loss_D_2.item() else (
            loss_D_2, pred_fake_2, pred_real_2)

        (self.loss_D * self.opt.lambda_gan).backward(retain_graph=True)

    def get_loss(self, out_l, out_r):
        loss_G_GAN = self.loss_dic['gan'].get_g_loss(self.netD, self.input, out_l, self.target_t) * self.opt.lambda_gan
        loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(out_l, self.target_t) * 10
        loss_rcnn_pixel = self.loss_dic['r_pixel'].get_loss(out_r, self.target_r) * 10 * self.opt.r_pixel_weight
        loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(out_l, self.target_t) * self.opt.lambda_vgg
        return loss_G_GAN, loss_icnn_pixel, loss_rcnn_pixel, loss_icnn_vgg

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False

        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_rcnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        self.loss_contain = None

        loss_G_GAN_1, loss_icnn_pixel_1, loss_rcnn_pixel_1, \
        loss_icnn_vgg_1 = self.get_loss(self.output_i_stage2, self.output_j_stage2)

        loss_exclu = self.exclusion_loss(self.output_i_stage2, self.output_j_stage2, 3)
        loss_G_1 = loss_G_GAN_1 + loss_icnn_pixel_1 + loss_rcnn_pixel_1 + loss_icnn_vgg_1 + loss_exclu

        self.loss_G, self.loss_G_GAN, self.loss_icnn_pixel, self.loss_rcnn_pixel, self.loss_icnn_vgg = \
            (loss_G_1, loss_G_GAN_1, loss_icnn_pixel_1, loss_rcnn_pixel_1, loss_icnn_vgg_1)
        self.loss_G.backward()

    def hyper_column(self, input_img):
        hypercolumn = self.vgg(input_img)
        _, C, H, W = input_img.shape
        hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                       feature in hypercolumn]
        input_i = [input_img]
        input_i.extend(hypercolumn)
        input_i = torch.cat(input_i, dim=1)
        return input_i

    def _forward_stage1(self):
        # without edge
        input_i = self.input
        output_i, output_j = self.net_i1(self.hyper_column(input_i))
        self.output_i_stage1 = output_i
        self.output_j_stage1 = output_j

        return output_i, output_j

    def _forward_stage2(self):
        # without edge
        input_i, input_j = self.input_stage2
        output_i, output_j = self.net_i2(self.hyper_column(input_i), self.hyper_column(input_j))

        self.output_i_stage2 = output_i
        self.output_j_stage2 = output_j

        return output_i, output_j

    def forward(self, mode='train'):
        with torch.no_grad():
            out_l1, out_r1 = self._forward_stage1()
            self.input_stage2 = out_l1.detach(), out_r1.detach()

        out_l2, out_r2 = self._forward_stage2()

        return out_l1, out_r1, out_l2, out_r2

    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G1.zero_grad()
        self.optimizer_G2.zero_grad()
        self.backward_G()

        # self.optimizer_G1.step()
        self.optimizer_G2.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_rcnn_pixel is not None:
            ret_errors['RPixel'] = self.loss_rcnn_pixel.item()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.loss_contain is not None:
            ret_errors['Contain'] = self.loss_contain.item()

        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i_stage1'] = tensor2im(self.output_i_stage1).astype(np.uint8)
        ret_visuals['output_j_stage1'] = tensor2im(self.output_j_stage1).astype(np.uint8)
        ret_visuals['output_i_stage2'] = tensor2im(self.output_i_stage2).astype(np.uint8)
        ret_visuals['output_j_stage2'] = tensor2im(self.output_j_stage2).astype(np.uint8)

        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['reflection'] = tensor2im(self.target_r).astype(np.uint8)
        # ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals

    def exclusion_loss(self, img_T, img_R, level=3, eps=1e-6):
        grad_x_loss = []
        grad_y_loss = []

        for l in range(level):
            grad_x_T, grad_y_T = self.compute_grad(img_T)
            grad_x_R, grad_y_R = self.compute_grad(img_R)

            alphax = (2.0 * torch.mean(torch.abs(grad_x_T))) / (torch.mean(torch.abs(grad_x_R)) + eps)
            alphay = (2.0 * torch.mean(torch.abs(grad_y_T))) / (torch.mean(torch.abs(grad_y_R)) + eps)

            gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
            grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
            gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1

            grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + eps) ** 0.25)
            grad_y_loss.append(((torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2)))) + eps) ** 0.25)

            img_T = F.interpolate(img_T, scale_factor=0.5, mode='bilinear')
            img_R = F.interpolate(img_R, scale_factor=0.5, mode='bilinear')
        loss_gradxy = torch.sum(sum(grad_x_loss) / 3) + torch.sum(sum(grad_y_loss) / 3)

        return loss_gradxy / 2

    def contain_loss(self, img_T, img_R, img_I, eps=1e-6):
        pix_num = np.prod(img_I.shape)
        predict_tx, predict_ty = self.compute_grad(img_T)
        predict_rx, predict_ry = self.compute_grad(img_R)
        input_x, input_y = self.compute_grad(img_I)

        out = torch.norm(predict_tx / (input_x + eps), 2) ** 2 + \
              torch.norm(predict_ty / (input_y + eps), 2) ** 2 + \
              torch.norm(predict_rx / (input_x + eps), 2) ** 2 + \
              torch.norm(predict_ry / (input_y + eps), 2) ** 2

        return out / pix_num

    def compute_grad(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

    def load(self, model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            model_path = util.get_model_list(model.save_dir, self.opt.name, epoch=resume_epoch)
            state_dict = torch.load(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_i1.load_state_dict(state_dict['icnn'])
            model.net_i2.load_state_dict(state_dict['icnn2'] if 'icnn2' in state_dict else state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G1.load_state_dict(state_dict['opt_g'])
                model.optimizer_G2.load_state_dict(
                    state_dict['opt_g2'] if 'opt_g2' in state_dict else state_dict['opt_g'])
        else:
            state_dict = torch.load(icnn_path)
            model.net_i1.load_state_dict(state_dict['icnn'])
            model.net_i2.load_state_dict(state_dict['icnn2'] if 'icnn2' in state_dict else state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])

        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i1.state_dict(),
            'icnn2': self.net_i2.state_dict(),
            'opt_g': self.optimizer_G1.state_dict(),
            'opt_g2': self.optimizer_G2.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict

    def state_dict_eval(self):
        state_dict = {
            'icnn': self.net_i1.state_dict(),
            'icnn2': self.net_i2.state_dict(),
            'epoch': self.epoch,
            'iterations': self.iterations
        }

        return state_dict
