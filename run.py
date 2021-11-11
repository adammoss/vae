import yaml
from jsonargparse import ArgumentParser
import numpy as np
import sys
import os

from models import *
from models.utils import Identity
from callbacks import ImageSampler, ReconstructionCallback, LatentDimInterpolator
from datamodules import CelebADataModule, LensChallengeSpace1DataModule
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary
from torchvision import transforms as transform_lib
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule, MNISTDataModule

parser = ArgumentParser()
parser.add_argument('-c', '--config',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('-m', '--model_params.name')
parser.add_argument('-d', '--exp_params.dataset')
parser.add_argument('--model_params.in_channels', type=int)
parser.add_argument('--model_params.latent_dim', type=int)
parser.add_argument('--trainer_params.gpus', type=int)
parser.add_argument('--wandb_api_key', type=str, default='')
args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
for k, v in vars(args).items():
    if k in config:
        for k_nested, v_nested in vars(v).items():
            if k_nested in config[k] and v_nested is not None:
                config[k][k_nested] = v_nested

if args.wandb_api_key:
    os.environ["WANDB_API_KEY"] = args.wandb_api_key

wandb_logger = WandbLogger(save_dir=config['logging_params']['save_dir'],
                           name=config['logging_params']['name'],
                           project='vae')

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

# Datasets
SetRange = transform_lib.Lambda(lambda X: 2 * X - 1.)
ArcSinh = transform_lib.Lambda(lambda X: torch.asinh(X))

if config['exp_params']['dataset'] == "celeba":
    # CelebA is already normalized between (0, 1)
    dm_cls = CelebADataModule
    output_activation = nn.Tanh()
    dm_cls.train_transforms = transform_lib.Compose([transform_lib.RandomHorizontalFlip(),
                                                     transform_lib.CenterCrop(148),
                                                     transform_lib.Resize(config['exp_params']['img_size']),
                                                     transform_lib.ToTensor(),
                                                     SetRange])
    dm_cls.val_transforms = transform_lib.Compose([transform_lib.RandomHorizontalFlip(),
                                                   transform_lib.CenterCrop(148),
                                                   transform_lib.Resize(config['exp_params']['img_size']),
                                                   transform_lib.ToTensor(),
                                                   SetRange])
elif config['exp_params']['dataset'] == "mnist":
    # MNIST is already normalized between (0, 1)
    dm_cls = MNISTDataModule
    output_activation = nn.Tanh()
    dm_cls.train_transforms = transform_lib.Compose([transform_lib.Resize(config['exp_params']['img_size']),
                                                     transform_lib.ToTensor(),
                                                     SetRange])
    dm_cls.val_transforms = transform_lib.Compose([transform_lib.Resize(config['exp_params']['img_size']),
                                                   transform_lib.ToTensor(),
                                                   SetRange])
elif config['exp_params']['dataset'] == "lens":
    dm_cls = LensChallengeSpace1DataModule
    output_activation = Identity()
    dm_cls.train_transforms = transform_lib.Compose([transform_lib.Resize(config['exp_params']['img_size']),
                                                     transform_lib.ToTensor(),
                                                     transform_lib.Normalize(1.0E-13, 1.0E-12),
                                                     ArcSinh])
    dm_cls.val_transforms = transform_lib.Compose([transform_lib.Resize(config['exp_params']['img_size']),
                                                   transform_lib.ToTensor(),
                                                   transform_lib.Normalize(1.0E-13, 1.0E-12),
                                                   ArcSinh])
else:
    raise ValueError(f"undefined dataset {config['exp_param']['dataset']}")

dm = dm_cls(data_dir=config['exp_params']['data_path'], batch_size=config['exp_params']['batch_size'])
args.input_size = dm.size()

dm.prepare_data()
dm.setup()
val_images = next(iter(dm.val_dataloader()))

config['model_params']['in_channels'] = val_images[0].size()[1]

# Model
model = vae_models[config['model_params']['name']](output_activation=output_activation, **config['model_params'])

summary(model, input_size=(config['exp_params']['batch_size'], config['model_params']['in_channels'],
                           config['exp_params']['img_size'], config['exp_params']['img_size']))

train_M_N = config['exp_params']['batch_size'] / len(dm.dataset_train)
val_M_N = config['exp_params']['batch_size'] / len(dm.dataset_val)
test_M_N = config['exp_params']['batch_size'] / len(dm.dataset_test)

experiment = VAEXperiment(model, config['exp_params'], train_M_N, val_M_N, test_M_N)
with torch.no_grad():
    experiment.model.eval()
    samples = experiment.model.sample(1, experiment.device)
experiment.model.train()
print(samples)


try:
    experiment = VAEXperiment(model, config['exp_params'], train_M_N, val_M_N, test_M_N)
    with torch.no_grad():
        experiment.model.eval()
        samples = experiment.model.sample(1, experiment.device)
    experiment.model.train()
except:
    samples = None

if samples is not None:
    callbacks = [ImageSampler(), ReconstructionCallback(val_images), LatentDimInterpolator()]
else:
    callbacks = [ReconstructionCallback(val_images)]

runner = Trainer(default_root_dir=f"{wandb_logger.save_dir}",
                 min_epochs=1,
                 logger=wandb_logger,
                 flush_logs_every_n_steps=100,
                 limit_train_batches=1.,
                 limit_val_batches=1.,
                 num_sanity_val_steps=5,
                 callbacks=callbacks,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=dm)
