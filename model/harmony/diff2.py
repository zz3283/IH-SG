"""
预测的是结果
和unet1搭配
"""

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

    #预测X_0 为了区分 用x_start 表达
    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    #预测x_t-1 mean var
    """
            Compute the mean and variance of the diffusion posterior
            q(x_{t-1} | x_t, x_0)
    """
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
             x_start + self.posterior_mean_coef2[t] * x_t        # 均值
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]     # 方差
        return posterior_mean, posterior_log_variance_clipped

    # 通过模型 预测x_start  通过x_start 计算 mean var 用于计算x_t-1
    def p_mean_variance(self, x, m, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_start,mask= self.denoise_fn(torch.cat([x, condition_x,m], dim=1), noise_level)
            # x_recon = self.predict_start_from_noise(x, t=t, noise=et_1)
            xt_prev_mean, xt_prev_var = self.q_posterior(x_start, x, t)
        else:
            x_start,mask = self.denoise_fn(x, noise_level)
            xt_prev_mean, xt_prev_var = self.q_posterior(x_start, x, t)

        return xt_prev_mean,  xt_prev_var, x_start , mask



    @torch.no_grad()
    def p_sample(self, x,m, t, clip_denoised=True, condition_x=None):
        mean,var,x_start,mask= self.p_mean_variance(
            x=x, m=m, t=t, clip_denoised=clip_denoised, condition_x = condition_x)
        # model_mean is the xt; condition_x is the ori input,  x is xt
        return mean,var,x_start,mask

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:        # 先不考虑
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                mean,var,x_start,mask = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, x_start], dim=0)
        if self.conditional:
            x = x_in['Input']   # x -> y input是 合成图像----->用作condition
            # ref=x_in['ref']
            shape = x.shape
            b, c, h, w = x.shape
            zt = torch.randn(shape, device=device)     # zt, 纯噪声
            ret_img = x
            zt_img = zt
            m=x_in['MM'] #细化的mask
            x_0=x #预测的x_0
            # h = torch.ones_like(x_in['Input'])
            # vt_img = vt
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                mean,var,x_start,mask= self.p_sample(zt, i, condition_x=x,m=m)
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                zt = mean + noise * (0.5 * var).exp()  ####zt-1
                zt.clamp_(0., 1.)

                if i % sample_inter == 0:
                    zt_img = torch.cat([zt_img, zt], dim=0)
                    x_0=torch.cat([x_0, zt], dim=0)
                    m= torch.cat([m, mask], dim=0)


        if continous:
            return [x_0,zt_img ,mask]
        else:
            """post deal"""
            # ret_img=ret_img[-1]*mask+ (1 - mask) *x_in['Input']
            # print(ret_img[-1].shape)
            return [x_0[-1],zt_img[-1],mask[-1]]

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
        x_0 = x_in['GT']  ##GT
        [b, c, h, w] = x_0.shape
        t = np.random.randint(1, self.num_timesteps ) #+1
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_0.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_0))
        x_noisy = self.q_sample(
            x_start=x_0, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        #因为直接预测结果
        if not self.conditional:
            x_recon,mask = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon,mask= self.denoise_fn(
                torch.cat([x_noisy, x_in['Input'],x_in['MM']], dim=1), continuous_sqrt_alpha_cumprod)
        loss_pix=self.loss_func(x_recon,x_0) ##
        print(mask.shape)
        loss_mask=self.loss_func(mask,x_in['fsm'])


        # loss = loss_diff + 0.5*loss_mask
        # loss_fore=self.loss_func(x_recon*x_in['Mask'],x_in['GT']*x_in['Mask'])
        # loss = loss_diff + loss_fore*0.5
        loss = loss_pix+0.2 * loss_mask
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
