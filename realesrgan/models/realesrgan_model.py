import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from basicsr.losses import build_loss
from basicsr.losses.custom_loss import MteLoss


@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt.get('queue_size', 180)


        train_opt = self.opt.get('train')
        self.cri_mte = None
        if train_opt and train_opt.get('mte_opt'):
            self.cri_mte = build_loss(train_opt['mte_opt']).to(self.device)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'Queue size {self.queue_size} must be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0

        if self.queue_ptr == self.queue_size:
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]

            lq_dequeue = self.queue_lr[0:b].clone()
            gt_dequeue = self.queue_gt[0:b].clone()

            self.queue_lr[0:b] = self.lq.clone()
            self.queue_gt[0:b] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            self.queue_lr[self.queue_ptr:self.queue_ptr+b] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr+b] = self.gt.clone()
            self.queue_ptr += b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train and self.opt.get('high_order_degradation', True):
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # --- First Degradation ---
            out = filter2D(self.gt_usm, self.kernel1)
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            scale = np.random.uniform(1, self.opt['resize_range'][1]) if updown_type == 'up' \
                    else np.random.uniform(self.opt['resize_range'][0], 1) if updown_type == 'down' else 1
            out = F.interpolate(out, scale_factor=scale, mode=random.choice(['area', 'bilinear', 'bicubic']))
            # noise
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range'], clip=True,
                                                   rounds=False, gray_prob=self.opt['gray_noise_prob'])
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'],
                                                  gray_prob=self.opt['gray_noise_prob'], clip=True, rounds=False)
            # jpeg
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)

            # --- Second Degradation ---
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)

            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            scale = np.random.uniform(1, self.opt['resize_range2'][1]) if updown_type == 'up' \
                    else np.random.uniform(self.opt['resize_range2'][0], 1) if updown_type == 'down' else 1
            out = F.interpolate(out, size=(int(ori_h/self.opt['scale']*scale), int(ori_w/self.opt['scale']*scale)),
                                mode=random.choice(['area', 'bilinear', 'bicubic']))

            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range2'], clip=True,
                                                   rounds=False, gray_prob=self.opt['gray_noise_prob2'])
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range2'],
                                                  gray_prob=self.opt['gray_noise_prob2'], clip=True, rounds=False)

            if np.random.uniform() < 0.5:
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']),
                                    mode=random.choice(['area', 'bilinear', 'bicubic']))
                out = filter2D(out, self.sinc_kernel)
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']),
                                    mode=random.choice(['area', 'bilinear', 'bicubic']))
                out = filter2D(out, self.sinc_kernel)

            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            self._dequeue_and_enqueue()
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt_usm if self.opt['l1_gt_usm'] else self.gt
        percep_gt = self.gt_usm if self.opt['percep_gt_usm'] else self.gt
        gan_gt = self.gt_usm if self.opt['gan_gt_usm'] else self.gt

        # freeze net_d
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # MTE Loss
            if self.cri_mte:
                l_g_mte = self.cri_mte(self.output, percep_gt)
                l_g_total += l_g_mte
                loss_dict['l_g_mte'] = l_g_mte



            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()



        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        fake_d_pred = self.net_d(self.output.detach().clone())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
