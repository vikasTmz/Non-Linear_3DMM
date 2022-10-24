
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from src import data
from src.nonlinearmm import training, generation
from src.nonlinearmm import models

def weights_init(m):
    # NOTE: xavier seems to converge faster
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    if hasattr(m, 'weight') and classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_models(cfg, dataset=None, device=None):
    # Get configs
    disp_size = cfg['data']['disp_size']
    b_size = cfg['data']['b_size']

    decoder = cfg['model']['decoder']
    id_vae_encoder = cfg['model']['id_vae_encoder']
    exp_vae_encoder = cfg['model']['exp_vae_encoder']

    decoder_kwargs = cfg['model']['decoder_kwargs']
    vae_encoder_kwargs = cfg['model']['vae_encoder_kwargs']
    n_id = cfg['model']['n_id']
    n_exp = cfg['model']['n_exp']

    # Create generator

    id_vae_encoder = models.vae_encoder_dict[id_vae_encoder](
        disp_size=disp_size, n_id=n_id, **vae_encoder_kwargs
    ).to(device)
    id_vae_encoder.apply(weights_init)

    exp_vae_encoder = models.vae_encoder_dict[exp_vae_encoder](
        b_size=b_size, n_exp=n_exp, **vae_encoder_kwargs
    ).to(device)
    exp_vae_encoder.apply(weights_init)

    # Joint decoder
    decoder = models.decoder_dict[decoder](
        disp_size=disp_size, n_id=n_id, n_exp=n_exp, **decoder_kwargs
    ).to(device)
    decoder.apply(weights_init)

    p0_z_id = get_prior_z(n_id, device)
    p0_z_exp = get_prior_z(n_exp, device)

    if cfg['model']['main_network'] == 'version_1':
        generator = models.DisneyNetwork1(
            decoder, id_vae_encoder, exp_vae_encoder, p0_z_id, p0_z_exp
        )

    elif cfg['model']['main_network'] == 'version_2':
        generator = models.DisneyNetwork2(
            decoder, id_vae_encoder, exp_vae_encoder, p0_z_id, p0_z_exp
        )

    # Output dict
    models_out = {
        'generator': generator
    }

    return models_out


def get_optimizers(models, cfg):
    model_g = models['generator']

    lr = cfg['training']['lr']
    #optimizer_g = optim.RMSprop(model_g.parameters(), lr=lr)
    optimizer_g = optim.Adam(model_g.parameters(), lr=lr)

    optimizers = {
        'generator': optimizer_g
    }
    return optimizers


def get_dataset(mode, cfg, input_sampling=True):
    # Config
    path_shapes = cfg['data']['path_shapes']
    
    # Fields
    if mode == 'train':
        fields = {
            'face': data.Facescape('neutral_geom.npz','target_geom.npz',
                'target_albedo.npz','blendweight.npz')
        }
        mode_ = 'train'

    elif mode == 'val_eval' or mode == 'val_vis':
        fields = {
            'face': data.Facescape('neutral_geom.npz','target_geom.npz',
                'target_albedo.npz','blendweight.npz')
        }
        mode_ = 'val'

    elif mode == 'test_eval' or mode == 'test_vis':
        fields = {
            'face': data.Facescape('neutral_geom.npz','target_geom.npz',
                'target_albedo.npz','blendweight.npz')
        }
        mode_ = 'test'

    else:
        print('Invalid data loading mode')

    # Dataset
    ds_shapes = data.Faces3dDataset(
            path_shapes, fields, split=mode_, no_except=False,
        )

    if mode_ == 'val' or mode_ == 'test':
        ds = ds_shapes
    else:
        ds = data.CombinedDataset([ds_shapes, ds_shapes])

    return ds


def get_dataloader(mode, cfg):
    # Config
    batch_size = cfg['training']['batch_size']
    with_shuffle = cfg['data']['with_shuffle']

    ds_shapes = get_dataset(mode, cfg)
    data_loader = torch.utils.data.DataLoader(
        ds_shapes, batch_size=batch_size, num_workers=12, shuffle=with_shuffle)
        #gcollate_fn=data.collate_remove_none)

    return data_loader


def get_trainer(models, optimizers, cfg, device=None):
    out_dir = cfg['training']['out_dir']

    print_every = cfg['training']['print_every']
    visualize_every = cfg['training']['visualize_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    backup_every = cfg['training']['backup_every']

    model_selection_metric = cfg['training']['model_selection_metric']
    model_selection_mode = cfg['training']['model_selection_mode']

    ma_beta = cfg['training']['moving_average_beta']
    multi_gpu = cfg['training']['multi_gpu']
    gp_reg = cfg['training']['gradient_penalties_reg']
    w_vae = cfg['training']['weight_vaeloss']
    w_geom_l1 = cfg['training']['weight_geom_l1loss']
    w_albedo_l1 = cfg['training']['weight_albedo_l1loss']
    w_kld = cfg['training']['weight_kldloss']
    experiment = cfg['training']['experiment']
    model_url = cfg['model']['model_url']
    trainer = training.Trainer(
        models['generator'],
        optimizers['generator'],
        ma_beta=ma_beta,
        gp_reg=gp_reg,
        w_vae=w_vae,
        w_geom_l1=w_geom_l1,
        w_albedo_l1=w_albedo_l1,
        w_kld=w_kld,
        multi_gpu=multi_gpu,
        experiment=experiment,
        out_dir=out_dir,
        model_selection_metric=model_selection_metric,
        model_selection_mode=model_selection_mode,
        print_every=print_every,
        visualize_every=visualize_every,
        checkpoint_every=checkpoint_every,
        backup_every=backup_every,
        validate_every=validate_every,
        device=device,
        model_url=model_url
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):

    generator = generation.Generator3D(
        model,
        device=device,
    )
    return generator


def get_prior_z(dim, device, **kwargs):
    p0_z = dist.Normal(
        torch.zeros(dim, device=device),
        torch.ones(dim, device=device)
    )

    return p0_z
