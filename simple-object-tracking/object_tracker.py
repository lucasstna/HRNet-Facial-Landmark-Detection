# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker

import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image
import numpy as np
import argparse
import imutils
import time
import tqdm
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import lib.models as models
from lib.config import config, update_config
from lib.core.evaluation import get_preds

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--video', help='video to analyze',
                        required=True, type=str)

    # parser.add_argument('--loc_model', help='localization model file',
    #                     required=True, type=str)

    parser.add_argument('--kp_model', help='keypoints model file',
                        required=True, type=str)                        

    parser.add_argument("--confidence", type=float, default=0.8,
                          help="minimum probability to filter weak detections")

    parser.add_argument('--cfg', help='keypoint model configuration filename',
                        required=True, type=str)

    
    args = parser.parse_args()
    update_config(config, args)                                          
        
    return args

def get_car_dets(img, model, threshold=0.5):
    # read the image and apply transformations
    image = Image.fromarray(img).convert('L')

    transform = T.Compose([T.Grayscale(num_output_channels=3), T.ToTensor()])
    image = transform(image)

    bbox = []
    confidence = []

    # generate detections
    model.eval()
    detections = model([image])

    boxes = [[i[0], i[1], i[2], i[3]] for i in detections[0]['boxes'].detach().numpy()]
    prediction_score = detections[0]['scores'].detach().numpy().tolist()

    if True not in (np.array(prediction_score) >= threshold):
        print('não tem coisa')

    else:
        prediction_t = [prediction_score.index(x) for x in prediction_score if x >= threshold][-1]
        boxes = boxes[:prediction_t + 1]

        for i in range(len(boxes)): 
            if detections[0]['labels'].numpy()[i] == 3:
                bbox.append(boxes[i])
                confidence.append(prediction_score[i])

    return {'bbox' : np.array(bbox, dtype=int), 'confidence' : confidence}

def get_keypoints(img, bbox, model):
    # read the image and apply transformations
    image = Image.fromarray(img).convert('RGB')

    image = image.crop(bbox)

    image = image.resize((256, 256))

    image = np.array(image, dtype=np.float32)
    image = image.transpose([2, 0, 1])

    model.eval()
    output = model(torch.Tensor([image]))
    score_map = output.data.cpu()
    preds = get_preds(score_map)[0]

    # print(preds)

    # modifying to global coordinates
    pts = np.array(preds) + [bbox[0], bbox[1]]

    # print(pts)

    return pts

def main():

    args = parse_args()

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    kp_model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)

    # set default device according to config file
    torch.cuda.set_device('cuda:' + str(gpus[0]))    

    kp_model = nn.DataParallel(kp_model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.kp_model)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        kp_model.load_state_dict(state_dict)
    else:
        kp_model.module.load_state_dict(state_dict)
    
    # initialize our centroid tracker and frame dimensions
    tracker = CentroidTracker()
    (H, W) = (None, None)

    # load model used to do detect cars
    det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args.video)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('../output.avi',fourcc, vs.get(cv2.CAP_PROP_FPS), (1920, 1080))

    # loop over the frames from the video stream
    while vs.isOpened():
        # read the next frame from the video stream and resize it
        ret, frame = vs.read()

        if frame is None:
            
            print('frame is none')

            if not ret:
                break
            else: 
                continue

        else:

            print('frame is not none')
            frame = cv2.resize(frame, (1920, 1080))
                
            # if the frame dimensions are None, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            dets = get_car_dets(frame, det_model)

            if len(dets['bbox']) > 0:

                rects = []
                keypoints = []

                for i in range(len(dets['bbox'])):
                    
                    if dets['confidence'][i] >= args.confidence:
                        rects.append(dets['bbox'][i])

                        (startX, startY, endX, endY) = dets['bbox'][i]
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                        # get keypoint
                        keypoints = get_keypoints(frame, (startX, startY, endX, endY), kp_model)

                        for point in keypoints:
                            cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)

                # update our centroid tracker using the computed set of bounding
                # box rectangles
                objects = tracker.update(rects)

                # loop over the tracked objects
                for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                video_writer.write(frame)
            
            else:
                print('len bbox: ', len(dets['bbox']))
    
    print('[INFO] Video processing ended...')

    vs.release()
    video_writer.release()


if __name__ == '__main__':
    main()