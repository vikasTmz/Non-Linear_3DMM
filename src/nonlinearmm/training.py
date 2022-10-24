import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from src.training import BaseTrainer

class Trainer(BaseTrainer):
    '''
    Subclass of Basetrainer for defining train_step, eval_step and visualize
    '''
    def __init__(self, model_g,
                 optimizer_g,
                 ma_beta=0.99, gp_reg=10.,
                 w_vae=0.,
                 w_geom_l1=0.,
                 w_albedo_l1=0.,
                 w_kld=0.,
                 experiment='conditional',
                 gan_type='standard',
                 loss_type='L1',
                 multi_gpu=False,
                 **kwargs):

        # Initialize base trainer
        super().__init__(**kwargs)

        # Models and optimizers
        self.model_g = model_g

        self.model_g_ma = copy.deepcopy(model_g)

        for p in self.model_g_ma.parameters():
            p.requires_grad = False
        self.model_g_ma.eval()

        self.optimizer_g = optimizer_g
        self.loss_type = loss_type
        self.experiment = experiment
        # Attributes
        self.gp_reg = gp_reg
        self.ma_beta = ma_beta
        self.gan_type = gan_type
        self.multi_gpu = multi_gpu
        self.w_vae = w_vae
        self.w_geom_l1 = w_geom_l1
        self.w_albedo_l1 = w_albedo_l1
        self.w_kld = w_kld
        self.vae_loss = w_vae != 0
        # Checkpointer
        self.checkpoint_io.register_modules(
            model_g=self.model_g,
            model_g_ma=self.model_g_ma,
            optimizer_g=self.optimizer_g,
        )

        print('w_geom_l1: %f, w_albedo_l1: %f, w_kld: %f' % (self.w_geom_l1, self.w_albedo_l1, self.w_kld))

    def train_step(self, batch, epoch_it, it):
        '''
        A single training step for the generative experiment
        Output:
            Losses
        '''
        batch_model0, batch_model1 = batch
        loss_g = self.train_step_g(batch_model0)
        losses = {
            'loss_g': loss_g
        }

        return losses

    def train_step_g(self, batch):
        '''
        A single train step of the generator part of generative model 
        '''
        model_g = self.model_g

        model_g.train()

        if self.multi_gpu:
            model_g = nn.DataParallel(model_g)

        self.optimizer_g.zero_grad()

        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        loss_vae = 0

        # Forward part and loss derivation for given experiment
        if self.vae_loss is True:
            losses, pred_target_geom_disp, pred_target_albedo_disp = \
            model_g(neutral_geom_disp,
                    target_geom_disp,
                    target_albedo_disp,
                    mean_geom,
                    mean_albedo,
                    blendweight)

            # print("Loss Recon = %.5f and KL = %.5f" %(self.w_l1 * losses['recon_loss'],self.w_kld * losses['kld_loss']))

        loss = self.w_geom_l1 * losses['recon_loss_geom'] + \
               self.w_albedo_l1 * losses['recon_loss_albedo'] + \
               self.w_kld * losses['kld_loss']
        loss.backward()

        # Gradient step
        self.optimizer_g.step()

        return loss.item()

    def eval_step(self, batch):
        '''
        Evaluation step with L1
        '''

        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        # Get model
        model_g = self.model_g
        model_g.eval()

        if self.multi_gpu:
            model_g = nn.DataParallel(model_g)

        # Predict
        with torch.no_grad():
            losses, pred_target_geom_disp, pred_target_albedo_disp = \
            model_g(neutral_geom_disp,
                    target_geom_disp,
                    target_albedo_disp,
                    mean_geom,
                    mean_albedo,
                    blendweight)
        
        loss = losses['recon_loss_geom'] + losses['recon_loss_albedo'] + losses['kld_loss']
        loss_val_dict = {'loss_val': loss}
        return loss_val_dict

    def visualize(self, batch):
        '''
        Visualization step
        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        print('Visualization: ToDo')
