import yaml
from jsonargparse import ArgumentParser
import numpy as np
import sys
import os

from models import *
from callbacks import ImageSampler, ReconstructionCallback, LatentDimInterpolator
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

parser = ArgumentParser()
parser.add_argument('-c', '--config',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('-m', '--model_params.name')
parser.add_argument('-d', '--exp_params.dataset')
parser.add_argument('--model_params.in_channels', type=int)
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

model = vae_models[config['model_params']['name']](**config['model_params'])

summary(model, input_size=(config['exp_params']['batch_size'], config['model_params']['in_channels'],
                           config['exp_params']['img_size'], config['exp_params']['img_size']))

experiment = VAEXperiment(model, config['exp_params'])

val_images = next(iter(experiment.val_dataloader()))

runner = Trainer(default_root_dir=f"{wandb_logger.save_dir}",
                 min_epochs=1,
                 logger=wandb_logger,
                 flush_logs_every_n_steps=100,
                 limit_train_batches=1.,
                 limit_val_batches=1.,
                 num_sanity_val_steps=5,
                 callbacks=[ImageSampler(), ReconstructionCallback(val_images), LatentDimInterpolator()],
                 ** config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
