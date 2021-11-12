from torch import nn
from torch.nn import functional as F
from pytorch_generative.models.vae.vq_vae import VectorQuantizedVAE as MyVQVAE
from .types_ import *


class VectorQuantizedVAE2(nn.Module):
    def __init__(self, in_channels, embedding_dim, K=512, beta=0.25, **kwargs):
        super().__init__()

        self.model = MyVQVAE(in_channels=in_channels, out_channels=in_channels,
                                        embedding_dim=embedding_dim)

    def forward(self, x, **kwargs):
        preds, vq_loss = self.model(x)
        return preds, x, vq_loss

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
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return loss, {'loss': loss,
                      'Reconstruction_Loss': recons_loss,
                      'VQ_Loss': vq_loss}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]