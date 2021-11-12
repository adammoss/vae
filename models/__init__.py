from .base import *
from .vanilla_ae import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *
from .wae_mmd import *
from .cvae import *
from .hvae import *
from .vampvae import *
from .iwae import *
from .dfcvae import *
from .mssim_vae import MSSIMVAE
from .fvae import *
from .cat_vae import *
from .joint_vae import *
from .info_vae import *
# from .twostage_vae import *
from .lvae import LVAE
from .logcosh_vae import *
from .swae import *
from .miwae import *
from .vq_vae import *
from .betatc_vae import *
from .dip_vae import *
from .resnet_vae import *
from .vq_vae2 import *
from .vq_vae_test import *
from .vq_vae_test2 import *
from .vd_vae import *

# Aliases
AE = VanillaAE
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE
VQVAETEST = VectorQuantizedVAE
VQVAETEST2 = VectorQuantizedVAE2
VDVAE = VeryDeepVAE

vae_models = {'HVAE': HVAE,
              'LVAE': LVAE,
              'IWAE': IWAE,
              'SWAE': SWAE,
              'MIWAE': MIWAE,
              'VQVAE': VQVAE,
              'DFCVAE': DFCVAE,
              'DIPVAE': DIPVAE,
              'BetaVAE': BetaVAE,
              'InfoVAE': InfoVAE,
              'WAE_MMD': WAE_MMD,
              'VampVAE': VampVAE,
              'GammaVAE': GammaVAE,
              'MSSIMVAE': MSSIMVAE,
              'JointVAE': JointVAE,
              'BetaTCVAE': BetaTCVAE,
              'FactorVAE': FactorVAE,
              'LogCoshVAE': LogCoshVAE,
              'VanillaAE': VanillaAE,
              'VanillaVAE': VanillaVAE,
              'ConditionalVAE': ConditionalVAE,
              'CategoricalVAE': CategoricalVAE,
              'ResNetVAE': ResNetVAE,
              'VQVAE2': VQVAE2,
              'VQVAETEST': VQVAETEST,
              'VQVAETEST2': VQVAETEST2,
              'VDVAE': VDVAE}
