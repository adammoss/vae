from torch import nn
from torch.nn import functional as F
from pytorch_generative.models.vae.vd_vae import VeryDeepVAE as MyVeryDeepVAE
from .types_ import *


class VeryDeepVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim, input_resolution=32, **kwargs):
        super().__init__()

        self.model = MyVeryDeepVAE(in_channels=in_channels, out_channels=in_channels,
                                   input_resolution=input_resolution, latent_channels=16,
                                   hidden_channels=64, bottleneck_channels=32)

    def forward(self, x, **kwargs):
        preds, kl_div = self.model(x)
        return preds, x, kl_div

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        kl_loss = args[2]

        recons_loss = F.binary_cross_entropy_with_logits(recons, input, reduction="none")
        recons_loss = recons_loss.sum(dim=(1, 2, 3))
        #recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + kl_loss
        return loss, {'loss': loss.mean(),
                      'Reconstruction_Loss': recons_loss.mean(),
                      'KL_Loss': kl_loss.mean()}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]