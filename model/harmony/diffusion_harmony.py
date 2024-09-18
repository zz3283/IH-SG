import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core import metrics as Metrics

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas
# def extract(a, t, x_shape):
#         b, *_ = t.shape
#         out = a.gather(-1, t)
#         return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l2',   #l1
        # loss_type_mask='bce',
        loss_type_mask='l1',
        phase= 'train',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.loss_type_mask = loss_type_mask
        self.conditional = conditional
        self.phase=phase
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_loss_mask(self, device):
        if self.loss_type_mask == 'bce':
            # self.loss_func_mask = nn.BCEWithLogitsLoss(reduction='sum').to(device)
            self.loss_func_mask = nn.BCELoss(reduction='sum').to(device)
        elif self.loss_type_mask == 'l1':
            self.loss_func_mask = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type_mask == 'l2':
            self.loss_func_mask = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))  ##累计乘积
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others

        ##注解 onenote
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev',
                             to_torch(np.sqrt(1. - alphas_cumprod_prev)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))

        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))

        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
        # reserve for sr3
        # return self.sqrt_recip_alphas_cumprod[t] * x_t - \
        #     self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, et_1, x_t, t):
        mid = (x_t - self.sqrt_one_minus_alphas_cumprod[t] * et_1) / self.sqrt_alphas_cumprod[t]
        xt_1 = self.sqrt_alphas_cumprod_prev[t] * mid + self.sqrt_one_minus_alphas_cumprod_prev[t] * et_1
        return xt_1

        # origin for sr3
        # posterior_mean = self.posterior_mean_coef1[t] * \
        #     x_start + self.posterior_mean_coef2[t] * x_t        # 均值
        # posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]     # 方差
        # return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            mask,et_1= self.denoise_fn(torch.cat([x, condition_x], dim=1), noise_level)
            # x_recon = self.predict_start_from_noise(x, t=t, noise=et_1)
        else:
            mask,et_1 = self.denoise_fn(x, noise_level)
            # x_recon = self.predict_start_from_noise(x, t=t, noise=et_1)

        # if clip_denoised:
        #     et_1.clamp_(0., 1.)

        # here to reconstruct the xt-1
        xt_1 = self.q_posterior(
            et_1=et_1, x_t=x, t=t)
        if clip_denoised:
            xt_1.clamp_(0., 1.)

        # xt_1, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return mask,xt_1

    """ 使用vt"""
    # ##----## vt指的是初始mask？？(然后迭代）
    # #ori--有mask生成
    # def p_mean_variance(self, x, t, vt, clip_denoised: bool, condition_x=None):
    #     batch_size = x.shape[0]
    #     noise_level = torch.FloatTensor(
    #         [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
    #     if condition_x is not None:
    #         et_1, mask_t_1 = self.denoise_fn(torch.cat([x, condition_x, vt], dim=1), noise_level)
    #         # x_recon = self.predict_start_from_noise(x, t=t, noise=et_1)
    #     else:
    #         et_1, mask_t_1 = self.denoise_fn(x, noise_level)
    #         # x_recon = self.predict_start_from_noise(x, t=t, noise=et_1)
    #
    #     if clip_denoised:
    #         et_1.clamp_(0., 1.)
    #
    #     # here to reconstruct the xt-1
    #     xt_1 = self.q_posterior(
    #         et_1=et_1, x_t=x, t=t)
    #     if clip_denoised:
    #         xt_1.clamp_(0., 1.)
    #     # xt_1, posterior_log_variance = self.q_posterior(
    #     #     x_start=x_recon, x_t=x, t=t)
    #
    #     return xt_1, mask_t_1


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        mask,xt_1= self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x = condition_x)
        # model_mean is the xt; condition_x is the ori input,  x is xt
        return mask,xt_1 #

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:        # 先不考虑
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                mask,img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in['Input']   # x -> y input是 合成图像----->用作condition
            shape = x.shape
            b, c, h, w = x.shape
            zt = torch.randn(shape, device=device)     # zt, 纯噪声
            ret_img = x
            zt_img = zt
            # vt_img = vt
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                mask,xt_1= self.p_sample(zt, i, condition_x=x)
                # """ background 要加对应第i步的噪声"""
                noise = torch.randn_like(x)
                input_weight = torch.FloatTensor(
                    [self.sqrt_alphas_cumprod[i]]).repeat(b, 1).to(x.device)
                noise_weight = torch.FloatTensor(
                    [self.sqrt_one_minus_alphas_cumprod[i]]).repeat(b, 1).to(x.device)
                input_part = input_weight * x
                noise_part = noise_weight * noise
                wighted_input = input_part + noise_part
                zt = wighted_input* (1 - mask) + mask* xt_1

                # zt = xt_1
                # output= wighted_input * (1 - mask) + mask* xt_1
                zt.clamp_(0., 1.)
                # vt.clamp_(0., 1.)
                # vt = torch.where(vt>0.1, 1., 0.)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, xt_1], dim=0)      # concat img in the batch dim
                    zt_img = torch.cat([zt_img, zt], dim=0)      # concat img in the batch dim
                    # vt_img = torch.cat([vt_img, mask_t_1], dim=0)      # concat img in the batch dim

        if continous:
            return [ret_img,mask]
        else:
            # """post deal"""
            # ret_img=ret_img[-1]*x_in['Mask']+ (1 - x_in['Mask']) *x_in['Input']
            # print(ret_img[-1].shape)
            # return [ret_img, zt_img[-1]]
            """post deal"""
            print(ret_img[-1].shape)
            return [ret_img[-1],mask[-1]]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))       # if noise exists, return itself, otherwise create noise

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )


    def p_losses(self, x_in, noise=None):
        x_start = x_in['GT']  ##GT
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps ) #+1
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
                mask,x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
                mask,x_recon= self.denoise_fn(
                torch.cat([x_noisy, x_in['Input']], dim=1), continuous_sqrt_alpha_cumprod)

        loss_diff = self.loss_func(noise, x_recon)       # sr3 loss for x_recon and noise, x_recon is et_1


        x_remake = self.predict_start_from_noise(x_noisy, t=t, noise=x_recon)  # x_recon 在这里代表et_1
        loss_pix=self.loss_func(x_remake,x_start)
        loss = loss_diff+ 0.1*loss_pix
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
