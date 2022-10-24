import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import trimesh
from src.nonlinearmm.models import (
    decoder, vae_encoder
)

decoder_dict = {
    'geom_resnet': decoder.GeomDecoder_ResnetFC,
    'joint': decoder.JointDecoder_ResnetFC
}

vae_encoder_dict = {
    'id_resnet': vae_encoder.IdentityEncoder_ResnetFC,
    'exp_resnet': vae_encoder.ExpressionEncoder_ResnetFC,
}

class DisneyNetwork1(nn.Module):
    def __init__(self,
                decoder,
                id_vae_encoder, 
                exp_vae_encoder,
                p0_z_id,
                p0_z_exp):
        super().__init__()
        
        self.decoder = decoder
        self.id_vae_encoder = id_vae_encoder
        self.exp_vae_encoder = exp_vae_encoder
        self.p0_z_id = p0_z_id
        self.p0_z_exp = p0_z_exp

    def forward(self,
                neutral_geom_disp,
                target_geom_disp,
                target_albedo_disp, 
                mean_geom,
                mean_albedo,
                blendweight):
        """Synthesize face mesh .

        Args:
            neutral_geom_disp (torch.FloatTensor): tensor of size B x 3 x M
            target_geom_disp (torch.FloatTensor): tensor of size B x 3 x M
            target_albedo_disp (torch.FloatTensor): tensor of size B x 3 X M
            blendweight (torch.FloatTensor): tensor of size B X 1 X N
        Returns:
            
        """
        
        batch_size, _, M = neutral_geom_disp.size()
        _, _, N = blendweight.size()
        device = neutral_geom_disp.device

        # get latent codes, mean and std for identity and expression
        z, kl_id, kl_exp, mean_zid, logstd_zid, \
        mean_zexp, logstd_zexp = self.infer_z(neutral_geom_disp, blendweight)

        # decode combined latent code to get geom and albedo displacements
        pred_target_geom_disp, pred_target_albedo_disp = self.decode(z)

        # Compute losses
        losses = self.loss_function(pred_target_geom_disp,
                                    target_geom_disp,
                                    pred_target_albedo_disp,
                                    target_albedo_disp,
                                    mean_geom,
                                    mean_albedo,
                                    mean_zid,
                                    logstd_zid,
                                    mean_zexp,
                                    logstd_zexp,
                                    kl_id,
                                    kl_exp)

        return losses, pred_target_geom_disp, pred_target_albedo_disp


    def loss_function(self,
                    pred_target_geom_disp,
                    target_geom_disp,
                    pred_target_albedo_disp,
                    target_albedo_disp,
                    mean_geom,
                    mean_albedo,
                    mean_zid,
                    logstd_zid,
                    mean_zexp,
                    logstd_zexp,
                    kl_id,
                    kl_exp):

        # compute reconstruction loss

        recon_loss_geom = F.l1_loss(pred_target_geom_disp + mean_geom, 
                                        target_geom_disp + mean_geom)
        recon_loss_albedo = 0

        if pred_target_albedo_disp is not None:
            recon_loss_albedo = F.l1_loss(pred_target_albedo_disp + mean_albedo, 
                                            target_albedo_disp + mean_albedo)

        # kld_loss version 1
        kld_loss =  torch.mean(torch.abs(torch.sum(kl_id, axis=-1))) + \
           torch.mean(torch.abs(torch.sum(kl_exp, axis=-1)))

        losses = {'recon_loss_geom':recon_loss_geom,'recon_loss_albedo':recon_loss_albedo, 'kld_loss':kld_loss}

        return losses

    def decode(self, z):
        """
        Args:
            z (torch.FloatTensor): tensor of size B x Z with latent codes
        Returns:
            
        """
        pred_target_geom_disp, pred_target_albedo_disp = self.decoder(z)
        return pred_target_geom_disp, pred_target_albedo_disp

    def infer_z(self, neutral_geom, blendweight, **kwargs):
        """Get latent codes
        Args:
            neutral_geom_disp (torch.FloatTensor): tensor of size B x 3 x M
            blendweight (torch.FloatTensor): tensor of size B X 1 X N

        Returns:
            
        """

        # Call Identity VAE's encoder
        mean_zid, logstd_zid = self.id_vae_encoder(neutral_geom, **kwargs)
        q_zid = dist.Normal(mean_zid, torch.exp(logstd_zid)) #ToDo: shouldn't this be torch.exp(0.5 * logstd_zid)
        z_id = q_zid.rsample()
        kl_id = dist.kl_divergence(q_zid, self.p0_z_id).sum(dim=-1)

        # Call Expression VAE's encoder
        mean_zexp, logstd_zexp = self.exp_vae_encoder(blendweight, **kwargs)
        q_zexp = dist.Normal(mean_zexp, torch.exp(logstd_zexp))#ToDo: shouldn't this be torch.exp(0.5 * logstd_zexp)
        z_exp = q_zexp.rsample()
        kl_exp = dist.kl_divergence(q_zexp, self.p0_z_exp).sum(dim=-1)

        # Combine latent code
        z = torch.cat([z_id, z_exp], dim=1)

        return z, kl_id, kl_exp, \
                mean_zid, logstd_zid, mean_zexp, logstd_zexp

    def infer_z_transfer(self, neutral_geom, blendweight, **kwargs):
        '''
            <>
        '''
        mean_zid, logstd_zid = self.id_vae_encoder(neutral_geom, **kwargs)
        q_zid = dist.Normal(mean_zid, torch.exp(logstd_zid))
        z_id = q_zid.rsample()
        mean_zexp, logstd_zexp = self.exp_vae_encoder(blendweight, **kwargs)
        q_zexp = dist.Normal(mean_zexp, torch.exp(logstd_zexp))
        z_exp = q_zexp.rsample()

        verbose = {'mean_zid':mean_zid,'logstd_zid':logstd_zid,
                    'mean_zexp':mean_zexp,'logstd_zexp':logstd_zexp}

        return z_id, z_exp, verbose


class DisneyNetwork2(nn.Module):

    def __init__(self,
                decoder,
                id_vae_encoder, 
                exp_vae_encoder,
                p0_z_id,
                p0_z_exp):
        super().__init__()

        self.decoder = decoder
        self.id_vae_encoder = id_vae_encoder
        self.exp_vae_encoder = exp_vae_encoder
        self.p0_z_id = p0_z_id
        self.p0_z_exp = p0_z_exp


    def forward(self,
                neutral_geom_disp,
                target_geom_disp,
                target_albedo_disp, 
                mean_geom,
                mean_albedo,
                blendweight):
        """Generate an image .

        Args:
            neutral_geom_disp (torch.FloatTensor): tensor of size B x 1 x M
            target_geom_disp (torch.FloatTensor): tensor of size B x 1 x M
            target_albedo_disp (torch.FloatTensor): tensor of size B x 1 X M
            blendweight (torch.FloatTensor): tensor of size B X 1 X N
            z
        Returns:
            
        """
        batch_size, _, M = neutral_geom_disp.size()
        _, _, N = blendweight.size()
        device = neutral_geom_disp.device

        mu_id, log_var_id = self.id_vae_encoder(neutral_geom_disp)
        z_id = self.reparameterize(mu_id, log_var_id)

        mu_exp, log_var_exp = self.exp_vae_encoder(blendweight)
        z_exp = self.reparameterize(mu_exp, log_var_exp)

        z = torch.cat([z_id, z_exp], dim=1)

        pred_target_geom_disp, pred_target_albedo_disp = self.decoder(z)

        losses = self.loss_function(pred_target_geom_disp,
                                    target_geom_disp,
                                    pred_target_albedo_disp,
                                    target_albedo_disp,
                                    mean_geom,
                                    mean_albedo,
                                    mu_id,
                                    log_var_id,
                                    mu_exp,
                                    log_var_exp)

        return losses, pred_target_geom_disp, pred_target_albedo_disp

    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self,
                    pred_target_geom_disp,
                    target_geom_disp,
                    pred_target_albedo_disp,
                    target_albedo_disp,
                    mean_geom,
                    mean_albedo,
                    mu_id,
                    log_var_id,
                    mu_exp,
                    log_var_exp):

        # Reconstruction Loss
        recon_loss_geom = F.l1_loss(pred_target_geom_disp + mean_geom, 
                                        target_geom_disp + mean_geom)
        recon_loss_albedo = 0

        if pred_target_albedo_disp is not None:
            recon_loss_albedo = F.l1_loss(pred_target_albedo_disp + mean_albedo, 
                                            target_albedo_disp + mean_albedo)

        kld_loss_id = torch.mean(-0.5 * torch.sum(1 + log_var_id - mu_id ** 2 - log_var_id.exp(), dim = 1), dim = 0)
        kld_loss_exp = torch.mean(-0.5 * torch.sum(1 + log_var_exp - mu_exp ** 2 - log_var_exp.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss_id + kld_loss_exp

        losses = {'recon_loss_geom':recon_loss_geom,'recon_loss_albedo':recon_loss_albedo, 'kld_loss':kld_loss}

        return losses

    def infer_z_transfer(self, neutral_geom, blendweight, **kwargs):
        """
        Samples from the latent space and return the corresponding
        :return: (Tensor)
        """
        # n_id = kwargs['n_id']
        # n_exp = kwargs['n_exp']
        # z_id = torch.randn(1, n_id)
        # z_exp = torch.randn(1, n_exp)

        # device = neutral_geom.device
        # z_id = z_id.to(device)
        # z_exp = z_exp.to(device)
    
        mu_id, log_var_id = self.id_vae_encoder(neutral_geom)
        z_id = self.reparameterize(mu_id, log_var_id)

        mu_exp, log_var_exp = self.exp_vae_encoder(blendweight)
        z_exp = self.reparameterize(mu_exp, log_var_exp)

        verbose = {'mean_zid':mu_id,'logstd_zid':log_var_id,
                    'mean_zexp':mu_exp,'logstd_zexp':log_var_exp}

        return z_id, z_exp, verbose
