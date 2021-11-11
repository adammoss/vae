import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA, MNIST, CIFAR10
from astrovision.datasets import LensChallengeSpace1
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 train_M_N: float,
                 val_M_N: float,
                 test_M_N: float) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        self.train_M_N = train_M_N
        self.val_M_N = val_M_N
        self.test_M_N = test_M_N

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        results = self.forward(real_img)
        loss, logs = self.model.loss_function(*results,
                                              M_N=self.train_M_N,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({f"train_{k}": v.detach() for k, v in logs.items()}, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        results = self.forward(real_img)
        loss, logs = self.model.loss_function(*results,
                                              M_N=self.val_M_N,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({f"val_{k}": v.detach() for k, v in logs.items()}, on_step=True, on_epoch=False)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
