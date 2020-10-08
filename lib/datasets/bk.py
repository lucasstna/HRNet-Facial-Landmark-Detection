# -------------------------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Lucas Santana Escobar (lucasstn10@gmail.com), based on paper original implementations
# -------------------------------------------------------------------------------------------------

import os
import random

import json
import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
import numpy as np
import sys

sys.path.append("../..")

from lib.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


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

        print(image_path)                                  

        scale = 1 / 1.25                                  
        
        # retrieving bbox of the object
        bbox_x = self.landmarks_frame['annotations'][idx]['bbox'][0]
        bbox_y = self.landmarks_frame['annotations'][idx]['bbox'][1]
        bbox_w = self.landmarks_frame['annotations'][idx]['bbox'][2]
        bbox_h = self.landmarks_frame['annotations'][idx]['bbox'][3]
        
        # load full image and crop it to separate the example
        img = Image.open(image_path).convert('RGB').crop((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

        # bbox central coordinates to use in data augmentation
        center_w = (bbox_w) / 2
        center_h = (bbox_h) / 2
        center = torch.Tensor([center_w, center_h])  

        # used to make keypoint values fit bbox coordinates
        resize_mat = np.matrix([[bbox_x, bbox_y],
                                [bbox_x, bbox_y],
                                [bbox_x, bbox_y],
                                [bbox_x, bbox_y]])     
                
        pts = np.array(self.landmarks_frame['annotations'][idx]['keypoints'], dtype=np.float32)
        pts = pts.reshape((-1, 3))[:, :2] - resize_mat

        print(pts.shape)

        scale *= 1.25
        nparts = pts.shape[0]

#########  JUST FOR TEST PURPOSES ###############################
        draw = ImageDraw.Draw(img)

        for i in range(pts.shape[0]):
            draw.point((pts[i,0], pts[i,1]), fill='yellow')

        img.save('teste.jpg')
#################################################################

        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='BK')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta
        

if __name__ == '__main__':
    a = BK('/home/lucas/Documents/reps/HRNet-Facial-Landmark-Detection/experiments/BK-dataset/bumper_keypoints.yaml')

    a.__getitem__(0)

                