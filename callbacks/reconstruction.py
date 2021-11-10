import numpy as np
import torch
from torch import Tensor
from pytorch_lightning.callbacks import Callback
import wandb


class ReconstructionCallback(Callback):
    def __init__(
            self,
            val_samples: Tensor,
            max_samples: int = 32,
    ):
        self.val_imgs, _ = val_samples
        self.val_imgs = self.val_imgs[:max_samples]

    def on_validation_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        with torch.no_grad():
            pl_module.eval()
            outs = pl_module.model.generate(val_imgs)
        pl_module.train()

        mosaics = torch.cat([outs, val_imgs], dim=-2)
        caption = "Top: Output, Bottom: Input"
        trainer.logger.experiment.log({
            "val/reconstruction": [wandb.Image(mosaic, caption=caption)
                             for mosaic in mosaics],
            "global_step": trainer.global_step
        })

