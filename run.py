import yaml
import argparse
import numpy as np

from models import *
from callbacks import ImageSampler, ReconstructionCallback
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

wandb_logger = WandbLogger(save_dir=config['logging_params']['save_dir'],
                           name=config['logging_params']['name'],
                           project='vae')

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

val_images = next(iter(experiment.val_dataloader()))

runner = Trainer(default_root_dir=f"{wandb_logger.save_dir}",
                 min_epochs=1,
                 logger=wandb_logger,
                 flush_logs_every_n_steps=100,
                 limit_train_batches=1.,
                 limit_val_batches=1.,
                 num_sanity_val_steps=5,
                 callbacks=[ImageSampler(), ReconstructionCallback(val_images)],
                 ** config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
