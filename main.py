import random
from torch.utils.data import DataLoader
import Dataset
import lightning as L
from torch import nn
import torch
import math
from torch import Tensor

class DiffusionParams_DDPM:
    diffusion_name: str = 'ddpm'
    n_steps_training: int = 1000
    beta_min: float = 0.0001
    beta_max: float = 0.02
    batch: int = 32
    num_classes: int = 32


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    # wait to complete


class ddpm(L.LightningModule):
    def __init__(self, params: DiffusionParams_DDPM):
        super().__init__()
        self.p = params
        self.batch = params.batch
        self.n_steps_training = params.n_steps_training
        self.beta_min = params.beta_min
        self.beta_max = params.beta_max
        self.beta = torch.linspace(self.beta_min, self.beta_max,
                                   self.n_steps_training)  # linearly increasing variance schedule
        self.alpha = (1. - self.beta)
        self.alpha_bar = self.alpha.cumprod(dim=0)
        self.one_minus_alpha = self.beta
        self.sqrt_recip_alpha = (1. / self.alpha).sqrt()

        self.sqrt_alpha_bar = self.alpha_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1. - self.alpha_bar).sqrt()
        self.eps_coef = self.one_minus_alpha / self.sqrt_one_minus_alphas_bar

        self.sigma2 = self.beta
        self.sigma2_sqrt = self.beta.sqrt()
        self.num_classes = params.num_classes
        # for x0
        self.sqrt_recip_alpha_bar = (1. / self.alpha_bar).sqrt()
        self.sqrt_recip_m1_alpha_bar = (1. / self.alpha_bar - 1.).sqrt()
        self.model = Model
        self.betas, self.alphas, self.cumalphas = self.cosine_schedule(time_steps=self.n_steps_training)

    # def forward(self, batch):
    #
    #     return self.model(inputs, target)
    def cosine_schedule(self, time_steps, s: float = 8e-3):
        t = torch.arange(0, time_steps)
        s = 0.008
        cumalphas = torch.cos(((t / time_steps + s) / (1 + s)) * (math.pi / 2)) ** 2

        def func(t): return math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2

        betas_ = []
        for i in range(time_steps):
            t1 = i / time_steps
            t2 = (i + 1) / time_steps
            betas_.append(min(1 - func(t2) / func(t1), 0.999))
        betas = torch.tensor(betas_)
        alphas = 1 - betas
        return betas.to('cuda:0'), alphas.to('cuda:0'), cumalphas.to('cuda:0')

    # def theta_post(self, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
    #     t = t - 1
    #
    #     alphas_t = self.alphas[t][..., None]
    #     cumalphas_tm1 = self.cumalphas[t - 1][..., None]
    #     alphas_t[t == 0] = 0.0
    #     cumalphas_tm1[t == 0] = 1.0
    #     theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) *
    #              (cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes))
    #     return theta / theta.sum(dim=1, keepdim=True)
    #
    # def theta_post_prob(self, xt: Tensor, theta_x0: Tensor, t: Tensor) -> Tensor:
    #     """
    #     This is equivalent to calling theta_post with all possible values of x0
    #     from 0 to C-1 and multiplying each answer times theta_x0[:, c].
    #
    #     This should be used when x0 is unknown and what you have is a probability
    #     distribution over x0. If x0 is one-hot encoded (i.e., only 0's and 1's),
    #     use theta_post instead.
    #     """
    #     t = t - 1
    #
    #     alphas_t = self.alphas[t][..., None]
    #     cumalphas_tm1 = self.cumalphas[t - 1][..., None, None]
    #     alphas_t[t == 0] = 0.0
    #     cumalphas_tm1[t == 0] = 1.0
    #
    #     # We need to evaluate theta_post for all values of x0
    #     x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
    #     # theta_xt_xtm1.shape == [B, C, H, W]
    #     theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
    #     # theta_xtm1_x0.shape == [B, C1, C2, H, W]
    #     theta_xtm1_x0 = cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes
    #
    #     aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
    #     # theta_xtm1_xtx0 == [B, C1, C2, H, W]
    #     theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True)
    #
    #     # theta_x0.shape = [B, C, H, W]
    #
    #     return torch.einsum("bcdhw,bdhw->bchw", theta_xtm1_xtx0, theta_x0)

    def test_step(self, batch, batch_idx):
        images, masks, classes = batch

        # noise = torch.rand_like(masks.to(torch.float32))
        mask_pre_p = torch.ones(size=masks) / 2
        ##### awaiting test
        class_pre_p = torch.ones(size=classes.shape) / 80
        t = torch.ones(size=(batch_size,)) * self.n_steps_training
        points = torch.zeros([2, ]).to(torch.int)
        points[:] = random.choice(torch.nonzero(masks[0, :, :]))
        while (points[0] + 1 > 640) or (points[1] + 1 > 640) or (
                torch.sum(masks[0, points[0]:points[0] + 2, points[1]:points[1] + 2]) != 4):
            points[:] = random.choice(torch.nonzero(masks[0, :, :]))
        mask_pre_p[0, points[0]:points[0] + 2, points[1]:points[1] + 2] = 1
        for time_step in reversed(range(self.n_steps_training)):
            mask_pre_p, class_pre_p = self.model(images, mask_pre_p, class_pre_p, t)
            mask_pre_p = mask_pre_p
            #####waitng for complment
        # loss = nn.functional.kl_div(class_pre_p, classes) + 0.1 * nn.functional.kl_div(mask_pre_p, masks)
        return mask_pre_p, class_pre_p

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self._step(batch, batch_idx)
        return loss

    def _step(self, batch, batch_idx):

        images, masks, classes = batch
        print(images.shape)
        # images [32,3,640,640 ]
        # masks [32,640,640]
        # classes [32,]
        points = torch.zeros([self.batch, 2]).to(torch.int)

        t = torch.randint(1, self.n_steps_training + 1, size=(self.batch,))
        class_0 = torch.zeros(size=[self.batch, 80]).to('cuda:0')
        class_0.scatter_(1, classes.unsqueeze(1), 1)
        class_t = self.cumalphas[t].view(-1, 1) * class_0 + ((1 - self.cumalphas[t]) / 80).view(-1, 1)
        # noise = torch.rand_like(masks.to(torch.float32)) #use rand num to noise the pic
        noise = torch.ones(size=masks.shape) / 2  # use 0.5 to noise the pic
        ##### which one is better is awaiting test
        mask_t = self.cumalphas[t].view(-1, 1, 1) * masks + (1 - self.cumalphas[t]).view(-1, 1, 1) * noise
        for i in range(self.batch):
            points[i, :] = random.choice(torch.nonzero(masks[i, :, :]))
            while (points[i, 0] + 1 > 640) or (points[i, 1] + 1 > 640) or (
                    torch.sum(masks[i, points[i, 0]:points[i, 0] + 2, points[i, 1]:points[i, 1] + 2]) != 4):
                points[i, :] = random.choice(torch.nonzero(masks[i, :, :]))
            mask_t[i, points[i, 0]:points[i, 0] + 2, points[i, 1]:points[i, 1] + 2] = 1
        mask_pre, class_pre = self.model(images, mask_t, class_t, t)

        class_pre_1=self.theta_post_1(class_t, class_0, t)
        ######waiting for complment
        class_pre_2=self.theta_post_prob_1(class_t,class_pre,t)
        ######waiting for complment
        mask_pre_1=self.theta_post_2(mask_t, masks, t)
        ######waiting for complment
        mask_pre_2=self.theta_post_prob_2(mask_t, mask_pre, t)
        ######waiting for complment

        loss_kl = nn.functional.kl_div(
            torch.log(torch.clamp(class_pre_2, min=1e-12)),
            class_pre_1,
            reduction='none'
        )+0.1*nn.functional.kl_div(
            torch.log(torch.clamp(mask_pre_2, min=1e-12)),
            mask_pre_1,
            reduction='none'
        )
        loss=torch.sum(loss_kl)/32
        # class_pre_q = (self.alphas[t - 1].view(-1, 1) * class_t + ((1 - self.alphas[t - 1]) / 80).view(-1, 1)) * (
        #             self.cumalphas[t - 2].view(-1, 1) * class_0 + ((1 - self.cumalphas[t - 2]) / 80).view(-1, 1))
        # class_pre_q = class_pre_q / class_pre_q.sum(dim=1, keepdim=True)

        # mask_pre_q = self.cumalphas[t - 1].view(-1, 1, 1) * masks + (1 - self.cumlphas[t - 1]).view(-1, 1, 1) * noise
        # loss = nn.functional.kl_div(class_pre_p, class_pre_q) + 0.1 * nn.functional.kl_div(mask_pre_p, mask_pre_q)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
        #                                                        verbose=True)
        return None
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}


batch_size = 1
dataset = Dataset.COCODataset(root="../dataset/val2017", annFile="../dataset/instances_val2017.json")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

trainer = L.Trainer(devices=1, logger=False)
ddpm = ddpm(DiffusionParams_DDPM)
trainer.fit(model=ddpm, train_dataloaders=data_loader)
trainer.test(dataloaders=test_data_loader[0])
