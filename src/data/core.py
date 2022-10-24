import os
import logging
from torch.utils import data
import numpy as np
import yaml


logger = logging.getLogger(__name__)


# Fields
class Field(object):
    def load(self, data_path, idx):
        raise NotImplementedError

    def check_complete(self, files):
        raise NotImplementedError

class Faces3dDataset(data.Dataset):
    def __init__(self, dataset_folder, fields, split=None,
                 metadata=dict(), no_except=True):
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.metadata = metadata
        self.no_except = no_except
        # Get (filtered) model list
        if split is None:
            models = [
                f for f in os.listdir(dataset_folder)
                if os.path.isdir(os.path.join(dataset_folder, f))
            ]
        else:
            split_file = os.path.join(dataset_folder, split + '.lst')
            with open(split_file, 'r') as f:
                models = f.read().split('\n')

        self.models = list(filter(None, models))

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.dataset_folder, model)
        data = {}
        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise
            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        return data

    def get_model(self, idx):
        return self.models[idx]


class CombinedDataset(data.Dataset):
    def __init__(self, datasets, idx_main=0):
        self.datasets = datasets
        self.idx_main = idx_main

    def __len__(self):
        return len(self.datasets[self.idx_main])

    def __getitem__(self, idx):
        out = []
        for it, ds in enumerate(self.datasets):
            if it != self.idx_main:
                x_idx = np.random.randint(0, len(ds))
            else:
                x_idx = idx
            out.append(ds[x_idx])
        return out


# Collater
def collate_remove_none(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(check_element_valid, batch))
    return data.dataloader.default_collate(batch)


def check_element_valid(batch):
    if batch is None:
        return False
    elif isinstance(batch, list):
        for b in batch:
            if not check_element_valid(b):
                return False
    elif isinstance(batch, dict):
        for b in batch.values():
            if not check_element_valid(b):
                return False
    return True


# Worker initialization to ensure true randomeness
def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
