import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from concurrent.futures import ThreadPoolExecutor


class SDFCalculator:
    def __init__(self, config, max_workers=None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.config = config
        if 'sdf_out_max' not in config:
            self.config['sdf_out_max'] = -1

    def compute_sdf_single(self, label_slice):
        posmask = label_slice.astype('bool')
        negmask = ~posmask
        if posmask.any():
            if self.config['sdf_out_max'] == -1:
                normalized_sdf = (distance(posmask)-distance(negmask))/np.max(distance(posmask))
            else:
                normalized_sdf = (distance(posmask)-distance(negmask))/self.config['sdf_out_max']
            normalized_sdf = np.clip(normalized_sdf, a_min=-1, a_max=1)
            return normalized_sdf
        else:
            return -np.ones_like(label_slice, dtype=np.float32)
            # return np.zeros_like(label_slice, dtype=np.float32)

    def compute_sdf(self, label):
        labels = label.detach().cpu().numpy()
        C = labels.shape[0]
        sdf_label = np.zeros_like(labels, dtype=np.float32)

        futures = []
        for c in range(C):
            futures.append(self.executor.submit(self.compute_sdf_single, labels[c, :, :]))

        for idx, future in enumerate(futures):
            sdf_label[idx, :, :] = future.result()

        sdf_label = torch.as_tensor(sdf_label).to(label.device)
        return sdf_label
