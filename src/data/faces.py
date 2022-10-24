import os
import glob
import random
import numpy as np
import trimesh
import imageio
from src.data.core import Field


# 3D Faces
class Facescape(Field):
    def __init__(self, file_neutral_geom_disp, file_target_geom, 
                    file_target_albedo, file_blendweight):
        self.file_neutral_geom_disp = file_neutral_geom_disp
        self.file_target_geom = file_target_geom
        self.file_target_albedo = file_target_albedo
        self.file_blendweight = file_blendweight

    def load(self, model_path, idx):
        # import os.path
        # if not os.path.isfile(os.path.join(model_path, self.file_neutral_geom_disp)):
            # print(model_path, idx)
        neutral_geom_disp = np.load(os.path.join(model_path, self.file_neutral_geom_disp))
        target_geom = np.load(os.path.join(model_path, self.file_target_geom))
        target_albedo = np.load(os.path.join(model_path, self.file_target_albedo))
        blendweight = np.load(os.path.join(model_path, self.file_blendweight))

        neutral_geom_disp = neutral_geom_disp['disp'].astype(np.float32)
        
        target_geom_disp = target_geom['disp'].astype(np.float32)
        mean_geom = target_geom['mean'].astype(np.float32)
        
        target_albedo_disp = target_albedo['disp'].astype(np.float32)
        mean_albedo = target_albedo['mean'].astype(np.float32)
        
        blendweight = blendweight['blendweight'].astype(np.float32)

        data = {
            'neutral_geom_disp': neutral_geom_disp.T,
            'target_geom_disp': target_geom_disp.T,
            'mean_geom': mean_geom.T,
            'target_albedo_disp': target_albedo_disp.T,
            'mean_albedo': mean_albedo.T,
            'blendweight': blendweight.T,
            'idx':idx
        }

        return data

    def check_complete(self, files):
        complete = (self.file_neutral_geom_disp in files)
        return complete

