import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses_opt as losses
from data.intrinsic_dataset import linear_transform
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join
import math


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1 and len(image_numpy.shape) > 3:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    #print(image_numpy.shape)
    if len(image_numpy.shape) == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy * 255.0


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape

        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

        gradx = (img[..., 1:, :] - img[..., :-1, :]).abs().sum(dim=1, keepdim=True)
        grady = (img[..., 1:] - img[..., :-1]).abs().sum(dim=1, keepdim=True)

        gradX[..., :-1, :] += gradx
        gradX[..., 1:, :] += gradx
        gradX[..., 1:-1, :] /= 2

        gradY[..., :-1] += grady
        gradY[..., 1:] += grady
        gradY[..., 1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge




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
        identity = False
        mode = mode.lower()
        if mode == 'train':
            org_input, input, target_t, target_r, mask = data['org_input'], data['input'], data['albedo'], data['shading'], data['mask']
        elif mode == 'eval':
            org_input, input, target_t, target_r, mask, data_name = data['org_input'], data['input'], data['albedo'], data['shading'], data['mask'], data["fn"]
        elif mode == 'test':
            org_input, input, data_name = data['org_input'], data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        #print(data_name)
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0]).type(torch.cuda.FloatTensor)
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0]).type(torch.cuda.FloatTensor)
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0]).type(torch.cuda.FloatTensor)
        # print(target_t)
        # print(target_r)
        self.org_input = org_input
        self.input = torch.log(input)
        self.mask = mask.to(device=self.gpu_ids[0])
        # self.identity = identity
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.target_r = target_r
        self.data_name = data_name

        # self.issyn = False if 'real' in data else True
        self.issyn = True

        self.aligned = False if 'unaligned' in data else True

        if target_t is not None:
            self.target_edge = self.edge_map(self.target_t)

    def eval(self, data, savedir=None, suffix="test", pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            output_i = tensor2im(self.output_i)
            output_j = tensor2im(self.output_j)
            target_albedo = tensor2im(self.target_t)
            target_shading = tensor2im(self.target_r)

            recon = tensor2im(self.recon_img)
            if self.aligned:
                res = index.quality_assess(output_i, target_albedo)
                # res = index.quality_assess(output_j, target_r)
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    # CHANGED: name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    savedir = join(savedir, suffix, self.data_name[0])
                    os.makedirs(savedir, exist_ok=True)
                    Image.fromarray(output_i.astype(np.uint8)).save(
                        join(savedir, '{}_albedo.png'.format(self.opt.name)))
                    Image.fromarray(output_j.astype(np.uint8)).save(
                        join(savedir, '{}_shading.png'.format(self.opt.name)))
                    Image.fromarray(recon.astype(np.uint8)).save(
                        join(savedir, '{}_recon.png'.format(self.opt.name)))
                    Image.fromarray(target_albedo.astype(np.uint8)).save(join(savedir, 'gt_albedo.png'))
                    Image.fromarray(target_shading.astype(np.uint8)).save(join(savedir, 'gt_shading.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'gt_input.png'))
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
            output_i, output_j = self.forward()
            output_i = tensor2im(output_i)
            output_j = tensor2im(output_j)
            if self.data_name is not None and savedir is not None:
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}_l.png'.format(self.opt.name)))
                Image.fromarray(output_j.astype(np.uint8)).save(join(savedir, name, '{}_r.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class YTMTNetModel(YTMTNetBase):
    def name(self):
        return 'ytmtnet'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None

        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472

        self.net_i = arch.__dict__[self.opt.inet](in_channels, [3, 3]).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type)  # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            t_vggloss = losses.ContentLoss()
            t_vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = t_vggloss

            r_vggloss = losses.ContentLoss()
            r_vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['r_vgg'] = r_vggloss

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)

        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        loss_D_1, pred_fake_1, pred_real_1 = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        self.loss_D, self.pred_fake, self.pred_real = (loss_D_1, pred_fake_1, pred_real_1)

        (self.loss_D * self.opt.lambda_gan).backward(retain_graph=True)

    def get_loss(self, out_l, out_r):
        loss_G_GAN = self.loss_dic['gan'].get_g_loss(self.netD, self.input, out_l, self.target_t) * self.opt.lambda_gan
        loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(out_l, self.target_t) * self.opt.lambda_vgg
        loss_rcnn_vgg = self.loss_dic['r_vgg'].get_loss(out_r, self.target_r) * 0.2 * self.opt.lambda_vgg

        loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(out_l, self.target_t)
        loss_rcnn_pixel = self.loss_dic['r_pixel'].get_loss(out_r, self.target_r) * 1.5
        return loss_G_GAN, loss_icnn_pixel, loss_rcnn_pixel, loss_icnn_vgg + loss_rcnn_vgg

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False

        self.loss_G_GAN, self.loss_icnn_pixel, self.loss_rcnn_pixel, \
        self.loss_icnn_vgg = self.get_loss(self.output_i, self.output_j)

        self.loss_exclu = self.exclusion_loss(self.output_i, self.output_j, 3)

        self.loss_recons = self.loss_dic['recons'](self.output_i, self.output_j, self.input) * 0.2

        self.loss_G = self.loss_G_GAN + self.loss_icnn_pixel + self.loss_rcnn_pixel + \
                      self.loss_icnn_vgg + self.loss_recons
                      #self.loss_exclu + self.loss_recons

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

    def forward(self):
        # without edge
        input_i = self.input
        # print(input_i)
        # print(torch.sum(input_i > 0.0001))
        if self.vgg is not None:
            input_i = self.hyper_column(input_i)
        output_i, output_j = self.net_i(input_i, fn=self.data_name[0] if self.data_name else None)
        self.output_i = linear_transform(torch.exp(output_i * self.mask), 1 / (math.e - 1), 1 / (1 - math.e))
        self.output_j = linear_transform(torch.exp(output_j * self.mask), 1 / (math.e - 1), 1 / (1 - math.e))
        #self.output_j = torch.cat((self.output_j, self.output_j, self.output_j), dim=1)
        #print(self.output_j.shape)

        self.recon_img = (self.output_i * self.output_j) * self.mask
        return self.output_i, self.output_j

    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()

        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_rcnn_pixel is not None:
            ret_errors['RPixel'] = self.loss_rcnn_pixel.item()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.loss_G_GAN is not None:
            ret_errors['GAN'] = self.loss_G_GAN.item()
        #if self.loss_exclu is not None:
        #    ret_errors['Exclu'] = self.loss_exclu.item()
        if self.loss_recons is not None:
            ret_errors['Recons'] = self.loss_recons.item()

        ret_errors['lr'] = self.optimizer_G.param_groups[0]['lr']

        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        #ret_visuals['input'] = tensor2im(linear_transform(torch.exp(self.input), 1/(math.e-1), 1/(1-math.e))).astype(np.uint8)
        #ret_visuals['output_i'] = tensor2im(linear_transform(torch.exp(self.output_i),  1/(math.e-1), 1/(1-math.e))).astype(np.uint8)
        #ret_visuals['output_j'] = tensor2im(linear_transform(torch.exp(self.output_j), 1/(math.e-1), 1/(1-math.e))).astype(np.uint8)
        #ret_visuals['recon_img'] = tensor2im(linear_transform(torch.exp(self.recon_img), 1/(math.e-1), 1/(1-math.e))).astype(np.uint8)
        #ret_visuals['target'] = tensor2im(linear_transform(torch.exp(self.target_t),  1/(math.e-1), 1/(1-math.e))).astype(np.uint8)
        #ret_visuals['reflection'] = tensor2im(linear_transform(torch.exp(self.target_r), 1/(math.e-1), 1/(1-math.e))).astype(np.uint8)

        ret_visuals['input'] = tensor2im(self.org_input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)
        ret_visuals['output_j'] = tensor2im(self.output_j).astype(np.uint8)
        ret_visuals['recon_img'] = tensor2im(self.recon_img).astype(np.uint8)
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
        # predict_tx, predict_ty = self.compute_grad(img_T)
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
            model.net_i.load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = torch.load(icnn_path)
            model.net_i.load_state_dict(state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            # if model.isTrain:
            #     model.optimizer_G.load_state_dict(state_dict['opt_g'])

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])

        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict
