from __future__ import annotations

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import torchvision
import wandb


class ImageSampler(Callback):
    def __init__(
            self,
            img_size: Tuple[int, ...] = None,
            num_samples: int = 64,
            normalize: bool = True,
    ):
        self.img_size = img_size
        self.num_samples = num_samples
        self.normalize = normalize

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        rand_v = torch.rand((self.num_samples, pl_module.model.latent_dim), device=pl_module.device)
        p = torch.distributions.Normal(torch.zeros_like(rand_v), torch.ones_like(rand_v))
        z = p.rsample()
        with torch.no_grad():
            pl_module.model.eval()
            samples = pl_module.model.sample(self.num_samples, pl_module.device)
        pl_module.model.train()

        grid = torchvision.utils.make_grid(samples, nrow=8, normalize=self.normalize)

        caption = "Samples"
        trainer.logger.experiment.log({
            "samples": wandb.Image(grid, caption=caption),
            "global_step": trainer.global_step
        })
