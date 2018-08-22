import time, cv2, uuid
import numpy as np
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

from detect import *
from preprocess import prep_image


import pdb


class identity(object):
    colors = pkl.load(open("pallete", "rb"))
    colors_used = np.full(len(colors), False)
    max_dim = None
    
    def __init__(self):
        colorid = np.random.choice(len(identity.colors))
        while identity.colors_used[colorid]:
            colorid = np.random.choice(len(identity.colors))
        self.colorid = colorid
        self.color = identity.colors[self.colorid]
        identity.colors_used[self.colorid] = True
        self.name = str(uuid.uuid4())[:8]
        self.hist = []
        self.pos = None
        self.pos_pre = None     #  used to calculate velocity
        self.interval = 0
        self.v = None
        self.lower_right = None
        self.P = 0
        self.K = 0
        self.missed = 0
    
    @property
    def upper_left(self):
        if self.lower_right is None or self.pos is None:
            return None
        else:
            upper_left_ = (self.pos - (self.lower_right - self.pos))
            upper_left_[0].clip(min=0, max=identity.max_dim[0])
            upper_left_[1].clip(min=0, max=identity.max_dim[1])
            return upper_left_
            
            
    def add_hist(self):
        self.hist.append(tuple(self.pos.astype('int')))
        return


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
                        default = 5)
    parser.add_argument("--R", dest = "R", help = "Measurement uncertainty",
                        default = 5)
    
    return parser.parse_args()


def L2distance(dist_matrix):
    try:
        return np.sum(np.square(dist_matrix), axis=1)
    except:
        return np.sum(np.square(dist_matrix))

def none_dups(a):
    seen = {}

    for i, x in enumerate(a):
        if x not in seen:
            seen[x] = i   # record first occurance
        else:
            a[i] = None
            a[seen[x]] = None
    return a

def pair_position(position_pre, position_post, threshold=5000):

    num_id_pre = position_pre.shape[0]
    num_id_post = position_post.shape[0]
    
    center_pre = position_pre[:, 0:2]
    center_post = position_post[:, 0:2]

#    center_pre = np.tile(center_pre, (num_id_post, 1))
    center_pre_t = np.repeat(center_pre, repeats=num_id_post, axis=0)
#    center_post = np.repeat(center_post, repeats=num_id_pre, axis=0)
    center_post_t = np.tile(center_post, (num_id_pre, 1))
    
    dist = center_pre_t - center_post_t
    
    pos_map = [np.argmin(L2distance(dist[i: i+num_id_post]) )
                    for i in range(0,center_pre_t.shape[0],num_id_post)]
    pos_map = [pos_map[i] if L2distance(center_pre[i]-center_post[pos_map[i]])<threshold else None for i in range(num_id_pre)]
    pos_map = none_dups(pos_map)
    
    if all(i is None for i in pos_map):
        pos_map = dict(enumerate(pos_map))
        return pos_map, False
    else:
        pos_map = dict(enumerate(pos_map))
        return pos_map, True

    

def kalman_prediction(id_list, candidate_list, Q):
    for i in range(len(id_list)):
        if id_list[i].v is not None:
            id_list[i].pos += id_list[i].v
            id_list[i].pos[0].clip(min=0, max=identity.max_dim[0])
            id_list[i].pos[1].clip(min=0, max=identity.max_dim[1])
            
            id_list[i].lower_right += id_list[i].v
            id_list[i].lower_right[0].clip(min=0, max=identity.max_dim[0])
            id_list[i].lower_right[1].clip(min=0, max=identity.max_dim[1])
            
            id_list[i].v = 1*id_list[i].v
            
        id_list[i].interval += 1    
        id_list[i].P += Q
    
    for i in range(len(candidate_list)):
        candidate_list[i].interval += 1
        candidate_list[i].P += Q
    
    return id_list, candidate_list
    
def kalman_update(id_list, candidate_list, measured_position, R):

    import pdb

    ttl_list = id_list + candidate_list 
    legit_num = len(id_list)
    cand_num = len(candidate_list)
    est_position = np.zeros((len(ttl_list), 2))
    for i in range(len(ttl_list)):
        est_position[i, :] = ttl_list[i].pos
    pos_map, pair = pair_position(est_position, measured_position)
    
    accounted = np.full(measured_position.shape[0], False)
    
    if not pair:
        return id_list, candidate_list
#    pdb.set_trace()
    for i in range(len(pos_map)):
        if pos_map[i] is not None:
            
            accounted[pos_map[i]] = True
        
            ttl_list[i].K = ttl_list[i].P / (ttl_list[i].P + R)
            ttl_list[i].P -= ttl_list[i].K * ttl_list[i].P
            
            if i < legit_num:
                ttl_list[i].pos += ttl_list[i].K * (measured_position[pos_map[i], :2] - ttl_list[i].pos)
            else:   # candidates
                ttl_list[i].pos = measured_position[pos_map[i], :2]
                
            ttl_list[i].pos[0].clip(min=0, max=identity.max_dim[0])
            ttl_list[i].pos[1].clip(min=0, max=identity.max_dim[1])
            
            ttl_list[i].lower_right += ttl_list[i].K * (measured_position[pos_map[i], 2:4] - ttl_list[i].lower_right)
            ttl_list[i].lower_right[0].clip(min=0, max=identity.max_dim[0])
            ttl_list[i].lower_right[1].clip(min=0, max=identity.max_dim[1])
            
            
            
            ttl_list[i].v = (ttl_list[i].pos - ttl_list[i].pos_pre) / ttl_list[i].interval
            ttl_list[i].pos_pre = ttl_list[i].pos.copy()
            ttl_list[i].interval = 0
            
            # update candidate_list and id_list
            
            if i >= legit_num:
                id_list.append(ttl_list[i])
                candidate_list.remove(ttl_list[i])
        else:
            # could be exit ids, add to candidates
            if i < legit_num:
                ttl_list[i].missed += 1
                if ttl_list[i].missed > 10:
                    candidate_list.append(ttl_list[i])
                    id_list.remove(ttl_list[i])
            else:
                # was in candidate and not detected this time, could be false alarm, or exit ids
                candidate_list.remove(ttl_list[i])
                identity.colors_used[ttl_list[i].colorid] = False
    
    # deal with new identities
    for i in range(measured_position.shape[0]):
        if not accounted[i]:
            can_id = identity()
            can_id.pos = measured_position[i, :2]
            can_id.pos_pre = can_id.pos.copy()
            can_id.lower_right = measured_position[i, 2:4]
            candidate_list.append(can_id)
                
    return id_list, candidate_list
            
            
def write_identity(frame, id_list):

    for i in range(len(id_list)):
        c1 = (id_list[i].upper_left[0].astype('int'), id_list[i].upper_left[1].astype('int'))
        c2 = (id_list[i].lower_right[0].astype('int'), id_list[i].lower_right[1].astype('int'))
        label = "{0}:{1}".format('Person', id_list[i].name)
        color = id_list[i].color
        
        cv2.rectangle(frame, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(frame, c1, c2, color, -1)
        cv2.putText(frame, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        
        for h in range(len(id_list[i].hist)):
            cv2.circle(frame, id_list[i].hist[h], 3, color, -1)
        cv2.circle(frame, tuple(id_list[i].pos.astype('int')), 3, color, -1)
        
    return frame
    
    
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

    inp_dim = int(model.net_info["height"])

    args = arg_parse()
    images = args.images
    interval = args.interval
    R = args.R
    Q = args.Q
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
    print('Computinng initial identities...')
    
    
    
    while True:
        frame, ogl, dim = prep_image(imlist[offset], inp_dim)
        position_pre = measure(frame, dim)[:, 1:5].numpy()
        position_pre[:, 0] = [(x[0]+x[2])/2 for x in position_pre]
        position_pre[:, 1] = [(x[1]+x[3])/2 for x in position_pre]
        
        frame, ogl, dim = prep_image(imlist[offset + interval], inp_dim)
        position_post = measure(frame, dim)[:, 1:5].numpy()
        position_post[:, 0] = [(x[0]+x[2])/2 for x in position_post]
        position_post[:, 1] = [(x[1]+x[3])/2 for x in position_post]
        
        pos_map, paired = pair_position(position_pre, position_post)
        
        identity.max_dim = dim
        
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
    for idx in range(offset, ttl_num):
        start = time.time()
        inter += 1
        frame, ogl, dim = prep_image(imlist[idx], inp_dim)

        
        
        if inter > interval:
            detect_flag = True
            
        # run kalman filter
        
        id_list, candidate_list = kalman_prediction(id_list, candidate_list, Q)   
        
        if detect_flag:
            measured_position = measure(frame, dim)[:, 1:5].numpy()
            measured_position[:, 0] = [(x[0]+x[2])/2 for x in measured_position]
            measured_position[:, 1] = [(x[1]+x[3])/2 for x in measured_position]
            
            id_list, candidate_list = kalman_update(id_list, candidate_list, measured_position, R)    
            inter = 0
            
        if idx % 20 == 0:
            [id_list[x].add_hist() for x in range(len(id_list))]
            
        write_identity(ogl, id_list)
        det_names = pd.Series(imlist[idx]).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))[0]
        cv2.imwrite(det_names, ogl)
        
        end = time.time()
        objs = [ids.name for ids in id_list]
        
        detect_flag = False
        
    
        print("{0:20s} finished in {1:6.3f} seconds".format(imlist[idx].split("/")[-1], end - start))
        print("{0:20s} {1:s}".format("Identity Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    
    
