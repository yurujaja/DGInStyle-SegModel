# Obtained from https://github.com/lhoyer/HRDA
# Modification:
# - Add generated data as an additional dataset

from mmseg.datasets.uda_dataset import UDADataset
from .builder import DATASETS
import torch


@DATASETS.register_module()
class DGDataset(UDADataset):

    def __init__(self, source, cfg, source2=None):
        self.source = source
        self.source2 = source2
        self.CLASSES = source.CLASSES
        self.PALETTE = source.PALETTE
        self._setup_rcs(cfg)
        self.color_aug_only = True if cfg.get('color_aug') else False
        assert cfg.get('sync_crop_size') is None


    def get_gta_rcs_crop(self, idx):        
        sample = self.source[idx]
        c = int(sample['img_metas'].data["ori_filename"].split("_")[0])
        for _ in range(10):
            n_class = torch.sum(sample['gt_semantic_seg'].data == c)
            if n_class > 1500:
                break
            sample = self.source[idx]
        return sample        


    def __getitem__(self, idx):
        """
        if self.rcs_enabled:
            return self.get_rare_class_source_sample()
        else:
            # return self.source[idx]
            if idx < len(self.source):
                return self.get_rare_class_source_sample()
            else:                
                # return self.source2[idx - len(self.source)]
                # return self.get_gta_rcs_crop(idx - len(self.source))
                return self.source[idx - len(self.source)]
        """
        if self.color_aug_only:
            if idx < len(self.source):
                return self.source[idx]
            else:                
                return self.source2[idx - len(self.source)]
        elif self.rcs_enabled:
            if idx < len(self.source):
                return self.get_rare_class_source_sample()
            else:                
                return self.source2[idx - len(self.source)]
        else:
            if idx < len(self.source):
                return self.get_gta_rcs_crop(idx)
            else:                
                return self.source2[idx - len(self.source)]


    def __len__(self):
        if self.source2:
            return len(self.source) + len(self.source2) 
        else:
            return len(self.source)
