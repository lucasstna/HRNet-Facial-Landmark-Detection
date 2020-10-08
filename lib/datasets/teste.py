from bk import BK
import sys
import argparse

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

    args = parse_args()

    data = BK(config)

    a = data.__getitem__(0)

if __name__ == '__main__':
    main()
