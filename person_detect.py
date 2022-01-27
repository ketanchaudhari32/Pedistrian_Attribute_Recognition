# import the necessary packages
import sys
import os
import numpy as np
import random
import math
import cv2

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import pickle
import time
import argparse
from PIL import Image, ImageFont, ImageDraw

from baseline.model.DeepMAR import DeepMAR_ResNet50
from baseline.utils.utils import str2bool
from baseline.utils.utils import save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict 
from baseline.utils.utils import set_devices
from baseline.utils.utils import set_seed
 

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=())
        parser.add_argument('--set_seed', type=str2bool, default=False)
        # model
        parser.add_argument('--resize', type=eval, default=(224, 224))
        parser.add_argument('--last_conv_stride', type=int, default=2, choices=[1,2])
        # demo image
        parser.add_argument('--demo_image', type=str, default='./dataset/peta/images/00161.png')
        ## dataset parameter
        parser.add_argument('--dataset', type=str, default='peta',
                choices=['peta','rap', 'pa100k'])
        # utils
        parser.add_argument('--load_model_weight', type=str2bool, default=True)
        parser.add_argument('--model_weight_file', type=str, default='./exp/deepmar_resnet50/peta/partition0/run1/model/ckpt_epoch1.pth')
        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids

        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else: 
            self.rand_seed = None
        self.resize = args.resize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # utils
        self.load_model_weight = args.load_model_weight
        self.model_weight_file = args.model_weight_file
        if self.load_model_weight:
            if self.model_weight_file == '':
                print('Please input the model_weight_file if you want to load model weight')
                raise ValueError
        # dataset 
        datasets = dict()
        datasets['peta'] = './dataset/peta/peta_dataset.pkl'

        if args.dataset in datasets:
            dataset = pickle.load(open(datasets[args.dataset],'rb'))
        else:
            print('%s does not exist.'%(args.dataset))
            raise ValueError
        self.att_list = [dataset['att_name'][i] for i in dataset['selected_attribute']]
        
        # demo image
        self.demo_image = args.demo_image

        # model
        model_kwargs = dict()
        model_kwargs['num_att'] = len(self.att_list)
        model_kwargs['last_conv_stride'] = args.last_conv_stride
        self.model_kwargs = model_kwargs


### main function ###
cfg = Config()


# set the random seed
if cfg.set_seed:
    set_seed( cfg.rand_seed )
# init the gpu ids
set_devices(cfg.sys_device_ids)

# dataset 
normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
test_transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.ToTensor(),
        normalize,])

### Att model ###
model = DeepMAR_ResNet50(**cfg.model_kwargs)

# load model weight if necessary
if cfg.load_model_weight:
    map_location = (lambda storage, loc:storage)
    ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
    model.load_state_dict(ckpt['state_dicts'][0])

#model.cuda()
model.eval()



# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(os.getcwd()+'/example.mp4')

# the output will be written to output.avi
out = cv2.VideoWriter(
    os.getcwd()+'/output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride=(12,12) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
        img = Image.fromarray(frame[xA:xB,yA:yB])
        img_trans = test_transform( img ) 
        img_trans = torch.unsqueeze(img_trans, dim=0)
        img_var = Variable(img_trans)
        score = model(img_var).data.cpu().numpy()

        positive_cnt = 0
        for idx in range(len(cfg.att_list)):
            if score[0, idx] >= 0:
                txt = '%s: %.2f'%(cfg.att_list[idx], score[0, idx])
                cv2.putText(frame,txt,(xA+10,yA+10+10*positive_cnt),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0))
                positive_cnt+=1
                




    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)