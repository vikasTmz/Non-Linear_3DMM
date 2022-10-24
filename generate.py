import torch
import os
import argparse
from tqdm import tqdm
from src import data
from src import config
from src.checkpoints import CheckpointIO
import numpy as np

# Get arguments and Config
parser = argparse.ArgumentParser(
    description='Generate Color for given mesh.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()
cfg = config.load_config(args.config, None)

# Define device
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Read config
out_dir = cfg['training']['out_dir']
vis_dir = cfg['test']['vis_dir']
split = cfg['test']['dataset_split']
if split != 'test_vis' and split != 'test_eval':
    print('Are you sure not using test data?')
batch_size = cfg['generation']['batch_size']
gen_mode = cfg['test']['generation_mode']
model_url = cfg['model']['model_url']

# Dataset
dataset = config.get_dataset(split, cfg, input_sampling=False)
datasets = [dataset]

# Load Model
models = config.get_models(cfg, device=device, dataset=dataset)
model_g = models['generator']
checkpoint_io = CheckpointIO(out_dir, model_g=model_g)
if model_url is None:
    checkpoint_io.load(cfg['test']['model_file'])
else:
    checkpoint_io.load(cfg['model']['model_url'])

# Assign Generator
generator = config.get_generator(model_g, cfg, device)

batch_counter = 0

for i_ds, ds in enumerate(datasets):
    ds_id = ds.metadata.get('id', str(i_ds))
    ds_name = ds.metadata.get('name', 'n/a')

    out_dir = vis_dir    

    test_loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=12, shuffle=False, 
            collate_fn=data.collate_remove_none)
    

    def get_batch_size(batch):
        batch_size = next(iter(batch.values())).shape[0]
        return batch_size

    for batch in tqdm(test_loader):
        model_names = [
            ds.get_model(i) for i in batch['face.idx']
        ]

        batch_counter += get_batch_size(batch)

        if gen_mode == 'interpolate':
            generator.generate_faces_via_interpolation(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)

        elif gen_mode == 'test_vae':
            generator.generate_faces_testset(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)

        elif gen_mode == 'eval_vae':
            generator.evaluate_vae_testset(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)

        elif gen_mode == 'sample_vae':
            generator.generate_faces_via_random_sampling(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)

        elif gen_mode == 'iter_sample_vae':
            generator.generate_faces_via_iterative_sampling(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)
        
        elif gen_mode == 'tsne_viz':
            generator.visualize_latent_space(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)

        elif gen_mode == 'exp_interpolate':
            generator.expression_only_interpolation(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)

        elif gen_mode == 'latent_viz_interpolate':
            generator.generate_latent_visualization_interpolation(out_dir, cfg['model']['n_id'], 
                cfg['model']['n_exp'], batch, model_names)

        else:
            print('')

print("Metrics for %s, L1 = %f, Fscore = %f, Chamfer = %f " %(cfg['test']['model_file'], generator.l1 / batch_counter,
                                                                generator.fscore / batch_counter, generator.chamfer / batch_counter))
