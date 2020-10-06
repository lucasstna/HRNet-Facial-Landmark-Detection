from ..datasets.bk import BK

data = BK(cfg='../experiments/BK-dataset/bumper_keypoints.yaml')

a = data.__getitem__(0)
