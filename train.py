"""Base file for starting training
"""

import torch
import argparse
from src import config
import matplotlib

matplotlib.use('Agg')

n_epochs = 140000

parser = argparse.ArgumentParser(
    description='Train a Non-Linear 3D Face Morphable Model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified '
                         'number of seconds with exit code 2.')
args = parser.parse_args()
cfg = config.load_config(args.config, None)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
exit_after = args.exit_after

models = config.get_models(cfg, device=device)
optimizers = config.get_optimizers(models, cfg)

train_loader = config.get_dataloader('train', cfg)
val_loader = config.get_dataloader('val_eval', cfg)
vis_loader = None

trainer = config.get_trainer(models, optimizers, cfg, device=device)

trainer.train(train_loader, val_loader, vis_loader,
              exit_after=exit_after, n_epochs=n_epochs)
