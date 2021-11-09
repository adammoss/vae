from typing import List

import numpy as np
import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import torchvision
import wandb


class LatentDimInterpolator(Callback):
    """Interpolates the latent space for a model by setting all dims to zero and stepping through the first two
    dims increasing one unit at a time.
    Default interpolates between [-5, 5] (-5, -4, -3, ..., 3, 4, 5)
    Example::
        from pl_bolts.callbacks import LatentDimInterpolator
        Trainer(callbacks=[LatentDimInterpolator()])
    """

    def __init__(
        self,
        interpolate_epoch_interval: int = 1,
        range_start: int = -5,
        range_end: int = 5,
        steps: int = 11,
        num_samples: int = 2,
        normalize: bool = True,
    ):
        """
        Args:
            interpolate_epoch_interval: default 20
            range_start: default -5
            range_end: default 5
            steps: number of step between start and end
            num_samples: default 2
            normalize: default True (change image to (0, 1) range)
        """
        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(
                pl_module, latent_dim=pl_module.hparams.latent_dim  # type: ignore[union-attr]
            )
            images = torch.cat(images, dim=0)  # type: ignore[assignment]

            grid = torchvision.utils.make_grid(images, nrow=self.steps, normalize=self.normalize)
            caption = "Latent Interpolator"
            trainer.logger.experiment.log({
                "latent": wandb.Image(grid, caption=caption),
                "global_step": trainer.global_step
            })

    def interpolate_latent_space(self, pl_module: LightningModule, latent_dim: int) -> List[Tensor]:
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                    # set all dims to zero
                    z = torch.zeros(self.num_samples, latent_dim, device=pl_module.device)

                    # set the first 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # sample
                    with torch.no_grad():
                        img = pl_module.decode(z.to(pl_module.device)).cpu()

                    if len(img.size()) == 2:
                        img = img.view(self.num_samples, *pl_module.img_dim)

                    img = img[0]
                    img = img.unsqueeze(0)
                    images.append(img)

        pl_module.train()
        return images
