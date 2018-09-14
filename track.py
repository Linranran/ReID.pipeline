import time, cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from detect import *
from preprocess import prep_image

from kalman import kalman_prediction, kalman_update
from idclass import identity

import pdb

def arg_parse():
    """
    Parse arguements to the tracking module
    
    """
    
    parser = argparse.ArgumentParser(description='Tracking Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
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
    parser.add_argument("--interval", dest = "interval", help = "Number of frames to skip before detection",
                        default = 10)
    parser.add_argument("--Q", dest = "Q", help = "Process noise variance",
                        default = 1)
    parser.add_argument("--R", dest = "R", help = "Measurement uncertainty scale factor",
                        default = 0.4)
    
    return parser.parse_args()

            
def __show_results__(id_list, candidate_list, ogl):
    _img_ = ogl.copy()
    
    for i in range(len(id_list)):
        c1 = (id_list[i].upper_left[0].astype('int'), id_list[i].upper_left[1].astype('int'))
        c2 = (id_list[i].lower_right[0].astype('int'), id_list[i].lower_right[1].astype('int'))
        label = "{0}:{1}".format('Person', id_list[i].name)
        color = id_list[i].color
        cv2.rectangle(_img_, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(_img_, c1, c2, color, -1)
        cv2.putText(_img_, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    
    for i in range(len(candidate_list)):
        c1 = (candidate_list[i].upper_left[0].astype('int'), candidate_list[i].upper_left[1].astype('int'))
        c2 = (candidate_list[i].lower_right[0].astype('int'), candidate_list[i].lower_right[1].astype('int'))
        label = "{0}:{1}".format('Candidate', candidate_list[i].name)
        color = (255, 255, 255)
        cv2.rectangle(_img_, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(_img_, c1, c2, color, -1)
        cv2.putText(_img_, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    
    
    plt.imshow(_img_)
    plt.show()
    
    return


if __name__ ==  '__main__':
    np.set_printoptions(suppress=True)
    args = arg_parse()
    images = args.images
    interval = args.interval
    R = args.R
    Q = args.Q
    inp_dim = int(args.reso)
    model.net_info["height"] = args.reso
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    batch_size = 1

    start = 0

    read_dir = time.time()

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
    imlist.sort()
    ttl_num = len(imlist)
    
    inter = 0
    detect_flag = False
    center_pos = lambda x: (x[0] + x[1])/2
    
    # deal with initial identities
    
    offset = 0
    print('Computing initial identities...')
    
    while True:
        frame, ogl, dim = prep_image(imlist[offset], inp_dim)
        position_pre = measure(frame, dim)[:, 1:5].cpu().numpy()
        position_pre[:, 0] = [(x[0]+x[2])/2 for x in position_pre]
        position_pre[:, 1] = [(x[1]+x[3])/2 for x in position_pre]
        
        frame, ogl, dim = prep_image(imlist[offset + interval], inp_dim)
        position_post = measure(frame, dim)[:, 1:5].cpu().numpy()
        position_post[:, 0] = [(x[0]+x[2])/2 for x in position_post]
        position_post[:, 1] = [(x[1]+x[3])/2 for x in position_post]
        
        identity.max_dim = dim
        pos_map, paired = pair_position(position_pre, position_post)
        
        if paired:
            break
        else:
            offset += 1
    
    
    id_list = [identity() for x in range(len(pos_map))] 
    candidate_list = []
    
    velocity = np.zeros([len(pos_map), 4])
    for i in range(len(pos_map)):
        if pos_map[i] is not None:
            velocity[i, 0] = (position_post[pos_map[i], 0] - position_pre[i, 0]) / interval
            velocity[i, 1] = (position_post[pos_map[i], 1] - position_pre[i, 1]) / interval
        velocity[i, 2:4] = position_pre[i, :2]
    
    for i in range(len(pos_map)):
        id_list[i].pos = velocity[i, 2:4]
        id_list[i].pos_pre = id_list[i].pos.copy()
        if any(velocity[i, 0:2]!=0):
            id_list[i].v = velocity[i, 0:2]
        id_list[i].lower_right = position_pre[i, 2:4]
    
    print('Initial identities and velocities resolved...')    
                          
    # run tracking
    fps = 0
    fps_idx = 0
    process_time = 0
    
    for idx in range(offset, ttl_num):
        start = time.time()
        io_start = time.time()
        inter += 1
        frame, ogl, dim = prep_image(imlist[idx], inp_dim)
        io_time = time.time() - io_start
        if inter >= interval:
            detect_flag = True
            
        # run kalman filter
        process_start = time.time()
        id_list, candidate_list = kalman_prediction(id_list, candidate_list, Q)   
        
        if detect_flag:
            measured_position = measure(frame, dim)[:, 1:5].cpu().numpy()
            measured_position[:, 0] = [(x[0]+x[2])/2 for x in measured_position]
            measured_position[:, 1] = [(x[1]+x[3])/2 for x in measured_position]
            
            id_list, candidate_list = kalman_update(id_list, candidate_list, measured_position, R)    
            inter = 0
            
        if idx % 20 == 0:
            [id_list[x].add_hist() for x in range(len(id_list))]
            
        # calculate fps without IO
        process_time += time.time() - process_start
        fps_idx += 1
        if fps_idx >= 20:
            fps = fps_idx / process_time
            process_time = 0
            fps_idx = 0
            
        
        io_start = time.time()
        write_identity(ogl, id_list, candidate_list)
        det_names = pd.Series(imlist[idx]).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))[0]
        cv2.imwrite(det_names, ogl)
        io_time += time.time() - io_start
        
        end = time.time()
        objs = [ids.name for ids in id_list]
        
        detect_flag = False
    
        print("{0:20s} finished in {1:6.3f} seconds; IO took {2:6.3f} seconds".format(imlist[idx].split("/")[-1], end - start, io_time))
        print("Process speed without IO: {0:6.3f} per seconds.".format(fps))
        print("{0:20s} {1:s}".format("Identity Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    
    
