# ---------------------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Lucas Santana (lucasstn10@gmail.com), based on paper original implementations
# ---------------------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import json
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

class BK(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.ann = cfg.DATASET.TRAINSET
        else:
            self.ann = cfg.DATASET.TESTSET

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
        with open(self.ann, 'r') as file_handle:
            self.landmarks_frame = json.loads(file_handle.read())        

    def __len__(self):
        return len(self.landmarks_frame['annotations'])

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame['annotations'][idx]['image_id'] + '.jpeg')
        
        # retrieving bbox of the object
        bbox_x = self.landmarks_frame['annotations'][idx]['bbox'][0]
        bbox_y = self.landmarks_frame['annotations'][idx]['bbox'][1]
        bbox_w = self.landmarks_frame['annotations'][idx]['bbox'][2]
        bbox_h = self.landmarks_frame['annotations'][idx]['bbox'][3]
        
        # load full image and crop it to separate the example
        img = Image.open(image_path).convert('RGB').crop(bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h)
                

        

if __name__ == '__main__':
    pass

                