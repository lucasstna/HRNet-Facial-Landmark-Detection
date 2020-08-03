# ---------------------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Lucas Santana (lucasstn10@gmail.com), based on paper original implementations
# ---------------------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

class BK_dataset(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # ann loading
        self.bk_keypoints = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.bk_keypoints)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.bk_keypoints.iloc[idx, 0])

    


                