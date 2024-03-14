
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging as log
import time

class TracedPointsDataset(Dataset):
    """Base class for single mesh datasets with points sampled only at a given octree sampling region.
    fix_render_xs & fix_render_hits: shape=(subjects, n_views, width*width, 3)
    xs & hits: shape=(subjects, n_random_batches, n_random_views, width*width, 3)
    """

    def __init__(self, 
        code_idx        : int = 32,
    ):
        """Construct dataset. This dataset also needs to be initialized.
        """

        self.initialization_mode = None
        self.label_map = {
            15:0, 17:1, 1:0, 4:1, 22:0, 32:1, 34:2, 5:0, 8:1, 9:0, 33:1, 42:2, 97:3
            }
        self.mapped_subject_idx = self.label_map[code_idx]
        self.current_batch_data = None 

    def init_from_h5(self, dataset_path):
        """Initializes the dataset from a h5 file.
        """

        self.h5_path = dataset_path
        with h5py.File(dataset_path, "r") as f:
            try:
                self.num_subjects = f['num_subjects'][()]
                self.n_fix_views = f['fix_render_xs'].shape[1]
                self.n_regions = f['xs'].shape[1]
                self.n_random_batches = f['xs'].shape[2]
                self.n_random_views = f['xs'].shape[3]
            except:
                raise ValueError("[Error] Can't load from h5 dataset")
        self.resample_batch()
        self.initialization_mode = "h5"

    def set_fix(self, fix:bool):
        """Whether to sample the fix rendered data

        Args:
            fix (bool): _description_
        """
        self.fix = fix

    def resample_batch(self):
        """Resamples a new working set of indices.
        """
        self.batch_idx = np.random.randint(0, self.n_random_batches)
         # Preloading data
        with h5py.File(self.h5_path, "r") as f:
            self.current_batch_data = {
                'x': torch.tensor(f['xs'][self.mapped_subject_idx, :, self.batch_idx, :], device=torch.device('cuda')),
                'hit': torch.tensor(f['hits'][self.mapped_subject_idx, :, self.batch_idx, :], device=torch.device('cuda')) # 3, 5, 20, 10, 160000, 3 -> 5, 10, 160000, 3
            }
            if self.current_batch_data['hit'].shape[-1] == 3:  # Adjusting for dataset bug
                self.current_batch_data['hit'] = self.current_batch_data['hit'][..., 0].unsqueeze(-1)


    def _get_data_fix(self):
        with h5py.File(self.h5_path, "r") as f:
            x = torch.tensor(f['fix_render_xs'][self.mapped_subject_idx])
            hit = torch.tensor(f['fix_render_hits'][self.mapped_subject_idx])

        return x, hit

    def __getitem__(self, idx):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        if isinstance(idx, torch.Tensor):
            region_idx, view_idx = idx
            x = self.current_batch_data['x'][region_idx, view_idx, :] # 160000, 3
            hit = self.current_batch_data['hit'][region_idx, view_idx, :] # 160000, 1
            return x, hit
        elif isinstance(idx, int):
            return self._get_data_fix()
        else:
            raise TypeError("Index must be a tuple of (region_idx, view_idx)")
        
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")

        print(f'current mapped subject id {self.mapped_subject_idx}, random batches {self.n_random_batches}, each containing views {self.n_random_views}')
        return self.n_random_batches * self.n_random_views * self.n_regions
