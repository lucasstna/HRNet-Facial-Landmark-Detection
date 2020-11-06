# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torch.utils.data import DataLoader
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
import cv2


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--pred-file', help='predictions file',
                        required=True, type=str)

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
                                              
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():

    args = parse_args()

    # load predictions file
    predictions = np.array(torch.load(args.pred_file))
    
    dataset_type = get_dataset(config)

    gpus = list(config.GPUS)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False
    )


    for idx, (batch, target, meta) in enumerate(test_loader):
        for img in batch:
            img = img.permute(1, 2, 0).numpy()

            img = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_BGR2RGB)

            for pt in predictions[idx]:
                image = cv2.circle(img, (pt[0], pt[1]), radius=5, color=(255, 0, 0), thickness=-1)

            if cv2.imwrite('./img_results_bk/' + str(idx) + '.jpeg', image):
                print('nice')
            else:
                print('not nice')



if __name__ == '__main__':
    main()