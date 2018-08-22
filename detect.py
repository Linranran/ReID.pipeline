from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
import matplotlib.pyplot as plt


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()
    
    
def measure(img, dim):
    im_dim = torch.FloatTensor(dim).repeat(1,2)   
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    with torch.no_grad():   
        output = model(img, CUDA)
    output = write_results_person(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    
    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

    return output
#    
#def kalman_prediction(positions, velocity):
#    """
#    args: 
#        positions: [num_identities_1, 2]
#        velocity: [num_identities_2, 4]
#    """
#     positions = 
    


args = arg_parse()
scales = args.scales
CUDA = torch.cuda.is_available()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")
model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
num_classes = 80
classes = load_classes('data/coco.names') 
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()
#Set the model in evaluation mode
model.eval()

if __name__ ==  '__main__':
    images = args.images
    batch_size = int(args.bs)

    start = 0

    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    if not os.path.exists(args.det):
        os.makedirs(args.det)
        
    load_batch = time.time()
    
#    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
#    im_batches = [x[0] for x in batches]
#    orig_ims = [x[1] for x in batches]
#    im_dim_list = [x[2] for x in batches]
#    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    imlist.sort()

    
    leftover = 0
    
    if (len(imlist) % batch_size):
        leftover = 1
        
    num_batches = len(imlist) // batch_size + leftover       
    im_batches = [list(imlist[i*batch_size : min((i+1)*batch_size, len(imlist))]) for i in range(num_batches)]           


    i = 0
    
    start_det_loop = time.time()
    
    objs = {}

    for ibatch in range(num_batches):
        
        #load the image 
        start = time.time()
        
        batches = list(map(prep_image, im_batches[ibatch], [inp_dim for x in range(len(im_batches[ibatch]))]))
        batch = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
        
        batch = torch.cat(batch)
        
        
        if CUDA:
            batch = batch.cuda()
            im_dim_list = im_dim_list.cuda()
        

        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(batch, CUDA)
        
#        prediction = prediction[:,scale_indices]

        
        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
        prediction = write_results_person(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        
        if type(prediction) == int:
            i += 1
            continue

        end = time.time()
        

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            objs = [classes[int(x[-1])] for x in prediction if int(x[0]) == im_num]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        
        if CUDA:
            torch.cuda.synchronize()

            
        im_dim_list = torch.index_select(im_dim_list, 0, prediction[:,0].long())
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)    
        prediction[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        prediction[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2    
        prediction[:,1:5] /= scaling_factor
        

        
        for idx in range(prediction.shape[0]):
            prediction[idx, [1,3]] = torch.clamp(prediction[idx, [1,3]], 0.0, im_dim_list[idx,0])
            prediction[idx, [2,4]] = torch.clamp(prediction[idx, [2,4]], 0.0, im_dim_list[idx,1])

        
        colors = pkl.load(open("pallete", "rb"))
        
        def write(x, orig):
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            img = orig[int(x[0])]
            cls = int(x[-1])
            objectiveness = np.round(float(x[-2]), 2)
            label = "{0} {1}".format(classes[cls], objectiveness)
            color = random.choice(colors)
            cv2.rectangle(img, c1, c2,color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2,color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
            return img
        
        list(map(lambda x: write(x, orig_ims), prediction))
        
        det_names = pd.Series(im_batches[ibatch]).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
        list(map(cv2.imwrite, det_names, orig_ims))
    
        i += 1

    
    end = time.time()
    
    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
#    print()
#    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
#    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
#    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
#    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
#    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    
    torch.cuda.empty_cache()
    
    
        
        
    
    
