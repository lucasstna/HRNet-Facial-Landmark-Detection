'''
    THIS CODE SHOWS THE INPUT FED TO THE NETWORK AT TRAINING/TEST TIME
'''

from bk import BK
import sys
import argparse
from PIL import Image, ImageDraw

sys.path.append("../..")

from lib.config import config, update_config    

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args

def main():

    # args = parse_args()

    # data = BK(config)

    # for i in range(data.__len__()):
    #     print(i, end=', ')
    #     a = data.__getitem__(idx = data.__len__() - 1 - i)

    image_path = '/hd4t/home/lucass/tcc/anns-V3/stanford16.jpeg'

    img = Image.open(image_path).convert('RGB').crop((216, 176, 543, 378))

    img.save('AAAA-train.jpg')


if __name__ == '__main__':
    main()
