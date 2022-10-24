import os
import glob
import random

import numpy as np
from trimesh.util import array_to_string

import torch
from torchvision.utils import save_image
from torch.nn.functional import interpolate
from torch import distributions as dist
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools

'''
    Source: https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/training/trainer_loss.py
'''
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from auxiliary.ChamferDistancePytorch.fscore import fscore

class Generator3D(object):
    def __init__(self, model, device=None):

        self.model = model
        self.device = device
        self.distChamfer = dist_chamfer_3D.chamfer_3DDist()
        self.chamfer = 0
        self.fscore = 0
        self.l1 = 0

    def compute_error_metrics(self, gt, recon):
        """
        Training loss of Atlasnet. The Chamfer Distance. Compute the f-score in eval mode.
        :return:
        """
        # dist1, dist2, idx1, idx2 = self.distChamfer(gt, recon)  # mean over points
        # # chamfer distance
        # self.chamfer += torch.mean(dist1) + torch.mean(dist2)  # mean over points
        # # fscore
        # fscore_error , _, _ = fscore(dist1, dist2)
        # self.fscore += fscore_error.mean()
        # L1
        l1_error = F.l1_loss(gt, recon)
        print(l1_error.item())
        self.l1 += l1_error

    def generate_faces_testset(self,
                                out_dir,
                                n_id,
                                n_exp,
                                batch,
                                model_names):
        '''
        Generate using the VAE

        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)
        # Define Output folders
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # batch loop
        for j in range(batch_size):
            # In eval mode
            self.model.eval()
            with torch.no_grad():
                losses, pred_target_geom_disp, \
                pred_target_albedo_disp = self.model(neutral_geom_disp,
                                                    target_geom_disp,
                                                    target_albedo_disp,
                                                    mean_geom,
                                                    mean_albedo,
                                                    blendweight)

            predicted_face_geom = pred_target_geom_disp + mean_geom

            gt_face_geom = target_geom_disp + mean_geom

            if pred_target_albedo_disp is not None:
                predicted_face_albedo = pred_target_albedo_disp + mean_albedo
                gt_face_albedo = target_albedo_disp + mean_albedo
            else:
                predicted_face_albedo = torch.zeros(pred_target_geom_disp.size())
                gt_face_albedo = torch.zeros(pred_target_geom_disp.size())

            np.savez(os.path.join(out_dir,model_names[j] + '_predicted_face.npz'),
                     **{'prediction_geom':predicted_face_geom[0].detach().cpu().numpy().T,
                     'prediction_albedo':predicted_face_albedo[0].detach().cpu().numpy().T})
            np.savez(os.path.join(out_dir,model_names[j] + '_gt_face.npz'),
                     **{'gt_geom':gt_face_geom[0].detach().cpu().numpy().T,'gt_albedo':gt_face_albedo[0].detach().cpu().numpy().T})

            self.compute_error_metrics(gt_face_geom, predicted_face_geom)

    def generate_faces_via_interpolation(self,
                                        out_dir,
                                        n_id,
                                        n_exp,
                                        batch,
                                        model_names):
        '''
        Interpolates between latent encoding 
        of first and second element of batch 

        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)
        assert(batch_size == 2)

        # Define Output folders
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # if not os.path.exists(out_dir + '/%s-%s'%(model_names[0],model_names[1])):
        #     os.makedirs(out_dir + '/%s-%s'%(model_names[0],model_names[1]))

        # In eval mode
        self.model.eval()

        with torch.no_grad():
            neutral_geom_disp_1 = neutral_geom_disp[0].view(1,3,-1)#interpolate(neutral_geom_disp[0], size=neutral_geom_disp[0].size())
            z_id_1, z_exp_1, _ = self.model.infer_z_transfer(neutral_geom_disp_1,
                                                        blendweight[0].view(1,-1))

            # Derive latent texture code as end point of interpolation
            neutral_geom_disp_2 = neutral_geom_disp[1].view(1,3,-1)#interpolate(neutral_geom_disp[1], size=neutral_geom_disp[1].size())
            z_id_2, z_exp_2, _ = self.model.infer_z_transfer(neutral_geom_disp_2,
                                                        blendweight[1].view(1,-1))

            mean_geom_1 = mean_geom[0]
            mean_albedo_1 = mean_albedo[0]
            target_geom_disp_1 = target_geom_disp[0]
            target_geom_disp_2 = target_geom_disp[1]
            target_albedo_disp_1 = target_albedo_disp[0]
            target_albedo_disp_2 = target_albedo_disp[1]

            # Derive stepsize
            steps = 5
            step1 = (z_id_2 - z_id_1) / steps
            step2 = (z_exp_2 - z_exp_1) / steps
            
            # steps loop
            for num in range(steps):
                inter_id = z_id_1 + step1 * num
                #inter_id = z_id_1
                inter_exp = z_exp_1 + step2 * num
                #inter_exp = z_exp_1

                z = torch.cat([inter_id.view(1,-1), inter_exp.view(1,-1)], dim=1)
                pred_target_geom_disp, pred_target_albedo_disp = self.model.decoder(z)

                predicted_face_geom = pred_target_geom_disp + mean_geom_1


                if pred_target_albedo_disp is not None:
                    predicted_face_albedo = pred_target_albedo_disp + mean_albedo_1
                    gt_face_albedo_1 = target_albedo_disp_1 + mean_albedo_1
                    gt_face_albedo_2 = target_albedo_disp_2 + mean_albedo_1

                else:
                    predicted_face_albedo = torch.zeros(pred_target_geom_disp.size())
                    gt_face_albedo_1 = torch.zeros(pred_target_geom_disp.size())
                    gt_face_albedo_2 = torch.zeros(pred_target_geom_disp.size())

                predicted_face_geom = predicted_face_geom.view(3,-1)
                predicted_face_albedo = predicted_face_albedo.view(3,-1)
                np.savez(os.path.join(out_dir,'%s-%s_%d_predicted_face.npz'%(model_names[0],model_names[1],num)),
                         **{'prediction_geom':predicted_face_geom.detach().cpu().numpy().T, 'prediction_albedo':predicted_face_albedo.detach().cpu().numpy().T})
            

            gt_face_geom_1 = target_geom_disp_1 + mean_geom_1
            gt_face_geom_2 = target_geom_disp_2 + mean_geom_1

            gt_face_geom_1 = gt_face_geom_1.view(3,-1)
            gt_face_albedo_1 = gt_face_albedo_1.view(3,-1)
            gt_face_geom_2 = gt_face_geom_2.view(3,-1)
            gt_face_albedo_2 = gt_face_albedo_2.view(3,-1)

            np.savez(os.path.join(out_dir,'%s-%s_%s_gt_face.npz'%(model_names[0],model_names[1],model_names[0])),
                     **{'gt_geom':gt_face_geom_1.detach().cpu().numpy().T,'gt_albedo':gt_face_albedo_1.detach().cpu().numpy().T})

            np.savez(os.path.join(out_dir,'%s-%s_%s_gt_face.npz'%(model_names[0],model_names[1],model_names[1])),
                     **{'gt_geom':gt_face_geom_2.detach().cpu().numpy().T,'gt_albedo':gt_face_albedo_2.detach().cpu().numpy().T})


    def evaluate_vae_testset(self,
                            out_dir,
                            n_id,
                            n_exp,
                            batch,
                            model_names):
        '''
        Evaluate the VAE

        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)
        # Define Output folders
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # batch loop
        print(model_names)
        for j in range(batch_size):

            self.model.eval()
            with torch.no_grad():
                losses, pred_target_geom_disp, \
                pred_target_albedo_disp = self.model(neutral_geom_disp,
                                                    target_geom_disp,
                                                    target_albedo_disp,
                                                    mean_geom,
                                                    mean_albedo,
                                                    blendweight)

            predicted_face = pred_target_geom_disp + mean_geom
            gt_face = target_geom_disp + mean_geom

            self.compute_error_metrics(gt_face, predicted_face)


    def generate_faces_via_random_sampling(self,
                                out_dir,
                                n_id,
                                n_exp,
                                batch,
                                model_names):
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)
        # Define Output folders
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # batch loop
        for j in range(batch_size):
            self.model.eval()
            with torch.no_grad():
                for i in range(0,5):
                    # Sample combined latent code
                    # q_z = dist.Normal(torch.zeros((1,n_id + n_exp)), torch.ones((1,n_id + n_exp)))
                    # z = q_z.rsample()
                    # z = z.to(self.device)

                    # Sample identity latent code
                    q_zid = dist.Normal(torch.zeros((1,n_id)), torch.ones((1,n_id)))
                    # torch.manual_seed(i) # ToDo: remove seed
                    z_id = q_zid.rsample()
                    z_id = z_id.to(self.device)


                    '''ToDo: should we sample the combined latent code or 
                            just identity latent code and derived expression from blendweights?
                    '''
                    mean_zexp, logstd_zexp = self.model.exp_vae_encoder(blendweight[j].view(1,-1))
                    '''version 1'''
                    # q_zexp = dist.Normal(mean_zexp, torch.exp(0.5 * logstd_zexp))
                    # z_exp = q_zexp.rsample()
                    '''version 2'''#TODO: decide a version
                    z_exp = self.model.reparameterize(mean_zexp, logstd_zexp)


                    z = torch.cat([z_id.view(1,-1), z_exp.view(1,-1)], dim=1)
                    pred_target_geom_disp, pred_target_albedo_disp = self.model.decoder(z)

                    predicted_face_geom = pred_target_geom_disp + mean_geom[j]

                    if pred_target_albedo_disp is not None:
                        predicted_face_albedo = pred_target_albedo_disp + mean_albedo[j]

                    else:
                        predicted_face_albedo = torch.zeros(pred_target_geom_disp.size())

                    predicted_face_geom = predicted_face_geom.view(3,-1)
                    predicted_face_albedo = predicted_face_albedo.view(3,-1)

                    np.savez(os.path.join(out_dir,'%s_%d_predicted_face.npz'%(model_names[j],i)),
                             **{'prediction_geom':predicted_face_geom.detach().cpu().numpy().T,
                             'prediction_albedo':predicted_face_albedo.detach().cpu().numpy().T})
                


    def generate_faces_via_iterative_sampling(self,
                                        out_dir,
                                        n_id,
                                        n_exp,
                                        batch,
                                        model_names):
        '''
        Generates new faces by iteratively sampling over a given latent code

        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)
        # Define Output folders
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # batch loop
        for j in range(batch_size):
            # In eval mode
            self.model.eval()
            with torch.no_grad():

                # q_zid = dist.Normal(torch.zeros((1,n_id)), torch.ones((1,n_id)))
                # z_id = q_zid.rsample()
                # z_id = z_id.to(self.device)


                # mean_zexp, logstd_zexp = self.model.exp_vae_encoder(blendweight[j].view(1,-1))
                # z_exp = self.model.reparameterize(mean_zexp, logstd_zexp)

                z_id, z_exp, _ = self.model.infer_z_transfer(neutral_geom_disp[j].view(1,3,-1),
                                                        blendweight[j].view(1,-1),
                                                        n_id=n_id,
                                                        n_exp=n_exp)

                # Derive stepsize
                print(z_id)
                for i in [0,6,10,11,14,16,21,28,31]:
                    steps = 5
                    step1 = torch.zeros(z_id.size())
                    step1 = step1.to(self.device)
                    if z_id[:,i] < 0:
                        step1[:,i] = 1
                    else:
                        step1[:,i] = -1

                    # steps loop
                    for num in range(steps):
                        inter_id = z_id + step1 * num
                        # inter_id = z_id_1
                        # inter_exp = z_exp_1 + step2 * num
                        inter_exp = z_exp

                        z = torch.cat([inter_id.view(1,-1), inter_exp.view(1,-1)], dim=1)

                        pred_target_geom_disp, pred_target_albedo_disp = self.model.decoder(z)

                        predicted_face_geom = pred_target_geom_disp + mean_geom[j]

                        if pred_target_albedo_disp is not None:
                            predicted_face_albedo = pred_target_albedo_disp + mean_albedo[j]
                            gt_face_albedo = target_albedo_disp[j] + mean_albedo[j]

                        else:
                            predicted_face_albedo = torch.zeros(pred_target_geom_disp.size())
                            gt_face_albedo = torch.zeros(pred_target_geom_disp.size())

                        predicted_face_geom = predicted_face_geom.view(3,-1)
                        predicted_face_albedo = predicted_face_albedo.view(3,-1)

                        np.savez(os.path.join(out_dir,'%s_param%d_%d_predicted_face.npz'%(model_names[j],i,num)),
                                 **{'prediction_geom':predicted_face_geom.detach().cpu().numpy().T,
                                 'prediction_albedo':predicted_face_albedo.detach().cpu().numpy().T})
                



    def visualize_latent_space(self,
                                out_dir,
                                n_id,
                                n_exp,
                                batch,
                                model_names):

        '''
        Generate using the VAE

        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)
        operation = 'latent_gif'
        checkpoint = out_dir.split('/')[-1]
        # In eval mode
        self.model.eval()
        with torch.no_grad():
            z_id, z_exp, verbose = self.model.infer_z_transfer(neutral_geom_disp,
                                                        blendweight,
                                                        n_id=n_id,
                                                        n_exp=n_exp)

            z = torch.cat([z_id, z_exp], dim=1)
            z = z.detach().cpu()

            tsne = TSNE(n_components=2, n_iter=6000)
            z_tsne = tsne.fit_transform(z.view(batch_size, -1))

            if operation == "plot_latent":
                import re

                colors = itertools.cycle(["r", "b", "g"])

                fig, ax = plt.subplots()
                # z_tsne_train = z_tsne[:30]
                # z_tsne_test = z_tsne[30:]

                for i, name in enumerate(model_names):
                    #if re.match(r"S_FORTINO_|A_MERANI_|J_HARRIS_", name):
                    #    ax.scatter(z_tsne.T[0,i], z_tsne.T[1,i], color="r")
                    ax.scatter(z_tsne.T[0,i], z_tsne.T[1,i], color="g")
                    # ax.annotate(name, (z_tsne.T[0,i], z_tsne.T[1,i]))

                ax.set_xlabel('$c_1$')
                ax.set_ylabel('$c_2$')

                # plt.show()
                fig.set_size_inches(18.5, 10.5)
                fig.savefig(os.path.join('tsne_data',checkpoint+'.png'))

                tsne_dict = {}
                for i in range(0,len(model_names)):
                    # print(list(z_tsne[i]))
                    tsne_dict[model_names[i]] = z_tsne[i].tolist()

                import json
                with open(os.path.join('tsne_data',checkpoint+'.json'), 'w') as fp:
                    json.dump(tsne_dict, fp)
            
            elif operation == "plot_meanstd":
                import re
                reference_circle = plt.Circle((0, 0), 1, color='g',alpha=0.5)
                colors = itertools.cycle(["r", "b", "g"])

                fig, ax = plt.subplots()

                ax.add_patch(reference_circle)

                mean_zid = verbose['mean_zid'].view(batch_size, -1).detach().cpu().numpy()
                logstd_zid = torch.exp(0.5 * verbose['logstd_zid'].view(batch_size, -1)).detach().cpu().numpy()
                mean_zexp = verbose['mean_zexp'].view(batch_size, -1).detach().cpu().numpy()
                logstd_zexp = torch.exp(0.5 * verbose['logstd_zexp'].view(batch_size, -1)).detach().cpu().numpy()

                for i, name in enumerate(model_names):
                    center = np.mean(mean_zid[i,:])
                    radius = np.mean(logstd_zid[i,:])
                    print('center=%.3f, radius=%.3f for subject = %s'%(center, radius, name))
                    if re.match(r"S_FORTINO_|A_MERANI_|J_HARRIS_", name): # "Y_SENAT_|G_BRAMBLE_|M_FINKELSTEIN_|D_VAZQUEZ_"
                        ax.add_patch(plt.Circle((center, center), radius, color='r',alpha=0.3))
                    else:
                        ax.add_patch(plt.Circle((center, center), radius, color='b',alpha=0.3))

                    ax.annotate(name, (center, center))

                fig.set_size_inches(18.5, 10.5)
                fig.savefig('./tsne_data/normal_distribution_identity_disney_L1_albedoMM_full_50k.png')
        
            else:
                tsne_dict = {}
                for i in range(0,len(model_names)):
                    print(list(z_tsne[i]))
                    tsne_dict[model_names[i]] = z_tsne[i].tolist()

                import json
                with open('./tsne_data/tsne_l1_train-interpolate_40k.json', 'w') as fp:
                    json.dump(tsne_dict, fp)


    def expression_only_interpolation(self,
                                        out_dir,
                                        n_id,
                                        n_exp,
                                        batch,
                                        model_names):
        '''

        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)

        # Define Output folders
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # In eval mode
        self.model.eval()
        with torch.no_grad():
            for j in range(batch_size):

                neutral_geom_disp_1 = neutral_geom_disp[j].view(1,3,-1)
                blendweight = blendweight[j].view(1,-1)

                # z_id_source, z_exp_source, verbose = self.model.infer_z_transfer(neutral_geom_disp_1,
                #                                             blendweight,
                #                                             n_id=n_id,
                #                                             n_exp=n_exp)

                mean_zexp, logstd_zexp = self.model.exp_vae_encoder(blendweight)
                z_exp_source = self.model.reparameterize(mean_zexp, logstd_zexp)

                q_zid = dist.Normal(torch.zeros((1,n_id)), torch.ones((1,n_id)))
                # torch.manual_seed(j+10) # ToDo: remove seed
                z_id_source = q_zid.rsample()
                z_id_source = z_id_source.to(self.device)

                mean_geom_1 = mean_geom[j]
                mean_albedo_1 = mean_albedo[j]
                target_geom_disp_1 = target_geom_disp[j]
                target_albedo_disp_1 = target_albedo_disp[j]

                # interpolation steps
                steps = 20
                i = 0
                for exp_index in [1,0,7,0,9]: # ,10,11,35,36
                    blendweight *= 0.0
                    blendweight[:,exp_index] = 1.0

                    mean_zexp, logstd_zexp = self.model.exp_vae_encoder(blendweight)
                    z_exp_target = self.model.reparameterize(mean_zexp, logstd_zexp)
                    # _, z_exp_target, _ = self.model.infer_z_transfer(neutral_geom_disp_1,
                    #                                         blendweight,
                    #                                         n_id=n_id,
                    #                                         n_exp=n_exp)
                    # Derive stepsize
                    step2 = (z_exp_target - z_exp_source) / steps
                
                    # steps loop
                    for num in range(steps):
                        inter_exp = z_exp_source + step2 * num

                        z = torch.cat([z_id_source.view(1,-1), inter_exp.view(1,-1)], dim=1)
                        pred_target_geom_disp, pred_target_albedo_disp = self.model.decoder(z)

                        predicted_face_geom = pred_target_geom_disp + mean_geom_1


                        if pred_target_albedo_disp is not None:
                            predicted_face_albedo = pred_target_albedo_disp + mean_albedo_1
                            gt_face_albedo_1 = target_albedo_disp_1 + mean_albedo_1

                        else:
                            predicted_face_albedo = torch.zeros(pred_target_geom_disp.size())
                            gt_face_albedo_1 = torch.zeros(pred_target_geom_disp.size())

                        predicted_face_geom = predicted_face_geom.view(3,-1)
                        predicted_face_albedo = predicted_face_albedo.view(3,-1)
                        np.savez(os.path.join(out_dir,'%s_%d_predicted_face.npz'%(model_names[j],i)),
                                 **{'prediction_geom':predicted_face_geom.detach().cpu().numpy().T, \
                                    'prediction_albedo':predicted_face_albedo.detach().cpu().numpy().T})

                        i+=1

                    z_exp_source = z_exp_target

    def generate_latent_visualization_interpolation(self,
                                        out_dir,
                                        n_id,
                                        n_exp,
                                        batch,
                                        model_names):
        '''
            Intern Expo visualization
        '''
        # Get data
        neutral_geom_disp = batch['face.neutral_geom_disp'].to(self.device)
        target_geom_disp = batch['face.target_geom_disp'].to(self.device)
        mean_geom = batch['face.mean_geom'].to(self.device)
        target_albedo_disp = batch['face.target_albedo_disp'].to(self.device)
        mean_albedo = batch['face.mean_albedo'].to(self.device)
        blendweight = batch['face.blendweight'].to(self.device)

        batch_size = neutral_geom_disp.size(0)

        # Define Output folders
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # For newer numpy version: np.random.default_rng().choice(batch_size, size=10, replace=False)
        interpolate_ids = [i for i in range(batch_size)]
        random.shuffle(interpolate_ids)
        interpolate_ids = interpolate_ids[:7]
        interpolate_names = []
        interpolate_pairs = {}

        checkpoint = out_dir.split('/')[-1]

        # In eval mode
        self.model.eval()
        with torch.no_grad():
            z_id, z_exp, verbose = self.model.infer_z_transfer(neutral_geom_disp,
                                                        blendweight,
                                                        n_id=n_id,
                                                        n_exp=n_exp)

            z_full = torch.cat([z_id, z_exp], dim=1) # ToDo: or just use z_id

            tsne = TSNE(n_components=2, n_iter=6000)

            for i in range(len(interpolate_ids)):
                start = interpolate_ids[i]
                if i+1 >= len(interpolate_ids):
                    end = 0
                else:
                    end = interpolate_ids[i+1]

                interpolate_pairs[model_names[start]] = model_names[end]

                neutral_geom_disp_1 = neutral_geom_disp[start].view(1,3,-1)
                z_id_1, z_exp_1, _ = self.model.infer_z_transfer(neutral_geom_disp_1,
                                                            blendweight[start].view(1,-1))

                # Derive latent texture code as end point of interpolation
                neutral_geom_disp_2 = neutral_geom_disp[end].view(1,3,-1)
                z_id_2, z_exp_2, _ = self.model.infer_z_transfer(neutral_geom_disp_2,
                                                            blendweight[end].view(1,-1))

                mean_geom_1 = mean_geom[start]
                mean_albedo_1 = mean_albedo[start]
                target_geom_disp_1 = target_geom_disp[start]
                target_geom_disp_2 = target_geom_disp[end]
                target_albedo_disp_1 = target_albedo_disp[start]
                target_albedo_disp_2 = target_albedo_disp[end]

                # Derive stepsize
                steps = 23
                step1 = (z_id_2 - z_id_1) / steps
                step2 = (z_exp_2 - z_exp_1) / steps
                
                # steps loop
                for num in range(steps):
                    name = '%s-%s_%d' %(model_names[start], model_names[end], num)
                    interpolate_names.append(name)
                    inter_id = z_id_1 + step1 * num
                    inter_exp = z_exp_1 + step2 * num

                    z = torch.cat([inter_id.view(1,-1), inter_exp.view(1,-1)], dim=1)
                    pred_target_geom_disp, pred_target_albedo_disp = self.model.decoder(z)

                    predicted_face_geom = pred_target_geom_disp + mean_geom_1

                    if pred_target_albedo_disp is not None:
                        predicted_face_albedo = pred_target_albedo_disp + mean_albedo_1
                        gt_face_albedo_1 = target_albedo_disp_1 + mean_albedo_1
                        gt_face_albedo_2 = target_albedo_disp_2 + mean_albedo_1

                    else:
                        predicted_face_albedo = torch.zeros(pred_target_geom_disp.size())
                        gt_face_albedo_1 = torch.zeros(pred_target_geom_disp.size())
                        gt_face_albedo_2 = torch.zeros(pred_target_geom_disp.size())

                    predicted_face_geom = predicted_face_geom.view(3,-1)
                    predicted_face_albedo = predicted_face_albedo.view(3,-1)
                    np.savez(os.path.join(out_dir,'%s_predicted_face.npz'%(name)),
                             **{'prediction_geom':predicted_face_geom.detach().cpu().numpy().T,
                             'prediction_albedo':predicted_face_albedo.detach().cpu().numpy().T})
                
                    z_full = torch.cat([z_full, z], dim=0)

                gt_face_geom_1 = target_geom_disp_1 + mean_geom_1
                gt_face_geom_2 = target_geom_disp_2 + mean_geom_1

                gt_face_geom_1 = gt_face_geom_1.view(3,-1)
                gt_face_albedo_1 = gt_face_albedo_1.view(3,-1)
                gt_face_geom_2 = gt_face_geom_2.view(3,-1)
                gt_face_albedo_2 = gt_face_albedo_2.view(3,-1)

                np.savez(os.path.join(out_dir,'%s_gt_face.npz'%(model_names[start])),
                         **{'gt_geom':gt_face_geom_1.detach().cpu().numpy().T,
                         'gt_albedo':gt_face_albedo_1.detach().cpu().numpy().T})

                np.savez(os.path.join(out_dir,'%s_gt_face.npz'%(model_names[end])),
                         **{'gt_geom':gt_face_geom_2.detach().cpu().numpy().T,
                         'gt_albedo':gt_face_albedo_2.detach().cpu().numpy().T})


            z_full = z_full.detach().cpu()
            z_tsne = tsne.fit_transform(z_full)

            """
            Plot t-SNE


            """

            tsne_train_data = {}
            tsne_interpolate_data = {}

            for i in range(batch_size):
                tsne_train_data[model_names[i]] = z_tsne[i].tolist()

            for i in range(batch_size, z_tsne.shape[0]):
                tsne_interpolate_data[interpolate_names[i-batch_size]] = z_tsne[i].tolist()

            import json
            with open(os.path.join('tsne_data',checkpoint+'.json'), 'w') as fp:
                data = {'tsne_train_data':tsne_train_data,
                        'tsne_interpolate_data':tsne_interpolate_data,
                        'interpolate_pairs':interpolate_pairs}
                json.dump(data, fp)