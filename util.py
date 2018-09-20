from __future__ import division
import os
import sklearn.utils.linear_assignment_ as sk_assignment

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import tensorflow as tf
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from bbox import bbox_iou
from idclass import identity

import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# =============================================
# Appearance modeling
# =============================================

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
                 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out
        
def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (center x, center y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


def appearance_modeler(model_filename="mars-small128.pb", input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder
    
    
    
# =============================================
# SORT tracking
# =============================================

def L2distance(dist_matrix):
    try:
        return np.sqrt(np.sum(np.square(dist_matrix), axis=1))
    except:
        return np.sqrt(np.sum(np.square(dist_matrix)))    

def calibrate_unreliable_pos(measured_position, features):
    '''heuristically solve occlusion using body ratio prior.
    '''
    num_measure = len(measured_position)
    new_measured_position = np.zeros(measured_position.shape)
    new_features = np.zeros(features.shape)
    new_ptr = 0
    for i in range(num_measure):
        raw_measured_position = measured_position[i, :2]
        raw_lower_right = measured_position[i, 2:4]
        raw_upper_left = measured_position[i, :2] \
                            -(measured_position[i, 2:4] - measured_position[i, :2])
                                  
        raw_height = (raw_lower_right[1] - raw_measured_position[1]) * 2
        raw_width = (raw_lower_right[0] - raw_measured_position[0]) * 2
        
        cali_height = raw_height
        cali_width = raw_width
        if raw_height/raw_width < 2:  # occlusion
            cali_height = raw_width * 2
        if raw_height/raw_width > 4:    # false positive, unusable
            measured_position[i, :] = None
            continue
            
        cali_pos = np.array((raw_upper_left[0] + cali_width/2, 
                    raw_upper_left[1] + cali_height/2,
                    raw_upper_left[0] + cali_width,
                    raw_upper_left[1] + cali_height))
        new_measured_position[new_ptr, :] = cali_pos
        new_features[new_ptr, :] = features[i, :]
        new_ptr += 1
        
    return new_measured_position[:new_ptr, :], new_features[:new_ptr, :]
                

def argmin_top3(array):     # todo
    '''
        return argmin of the least 3 elements in asecending order.
    '''
#    pass  # note could return None, elements could be None
    
    if len(array) == 0:   
        return (None)
        
    argmin_res = [None for _ in range(3)]
    for i in range(len(array)):
        if argmin_res[0] is None or array[i] < array[argmin_res[0]]:
            argmin_res[2] = argmin_res[1]
            argmin_res[1] = argmin_res[0]
            argmin_res[0] = i
        elif argmin_res[1] is None or array[i] < array[argmin_res[1]]:
            argmin_res[2] = argmin_res[1]
            argmin_res[1] = i
        elif argmin_res[2] is None or array[i] < array[argmin_res[2]]:
            argmin_res[2] = i

    return tuple(argmin_res)
    
    
def assign(dist_map, pair_dist, pair_velocity, pair_feature, pre_num, post_num, appearance_veto_mask):
    ''' assign identity pair relationships using Hungarian algorithm
    '''
    cost_mat = np.full((pre_num, post_num), 100, dtype='float32')
    for i in range(pre_num):
        for j, x in enumerate(dist_map[i]):
            if x is None:
                continue
            
            # appearance feature can veto results
            if appearance_veto_mask[i][j]:
                if pair_feature[i][j] is None:
                    continue
                    
            cost_mat[i, x] = pair_dist[i][j] # heuristic weights
#            if pair_velocity[i][j] is None:
#                cost_mat[i, x] = cost_mat[i, x] + 10
            if pair_feature[i][j] is None:
                cost_mat[i, x] = cost_mat[i, x] + 10
            
            if pair_velocity[i][j] is not None:
                cost_mat[i, x] = cost_mat[i, x] + pair_velocity[i][j]  # heuristic weights
            if pair_feature[i][j] is not None:
               cost_mat[i, x] = cost_mat[i, x] + pair_feature[i][j] # heuristic weights
                
    assign_result_holder = sk_assignment.linear_assignment(cost_mat)
    assign_result = dict([i, None] for i in range(pre_num))
    for i in range(assign_result_holder.shape[0]):
        pre, post = assign_result_holder[i]
        if cost_mat[pre, post] < 99:       # meaning distance hard requirment is met
            assign_result[pre] = post
    
    return assign_result
    
    
def pair_position(position_pre, position_post, features=None, threshold_scale=0.65, id_list=None):

    if len(position_post) == 0:
        return None, False
    
    num_id_pre = position_pre.shape[0]
    num_id_post = position_post.shape[0]
    
    center_pre = position_pre[:, 0:2]
    center_post = position_post[:, 0:2]

    center_pre_t = np.repeat(center_pre, repeats=num_id_post, axis=0)
    center_post_t = np.tile(center_post, (num_id_pre, 1))
    
    dist = center_pre_t - center_post_t
    id_sizes = L2distance(position_pre[:,:2]-position_pre[:,2:4])
    
    dist_map_ = [argmin_top3(L2distance(dist[i: i+num_id_post])) 
                    for i in range(0,center_pre_t.shape[0],num_id_post)]
    
    pair_dist = []
    appearance_veto_flag = []
    pair_velocity = []
    pair_feature = []
    dist_map = []
    velocity_map = []
    if id_list is not None:
        occlusion_flag = np.full((len(id_list), 1), False)
        
    for i in range(num_id_pre):
        tmp_dist = [L2distance(center_pre[i]-center_post[x]) for x in dist_map_[i] if x is not None ]
        
            
        if id_list is not None:
            if id_list[i].missed > 0:
                occlusion_flag[i] = True # possible occlusion happened due to multiple missed frame labeled

        if id_list is not None:
            if occlusion_flag[i]:  # larger pairing threshold after possible occlusion and must take appearance feature into account
                
                thresholded_tmp_dist = []
                tmp_veto_flag = []
                len_y = len(tmp_dist)
                for y in range(len_y):
                    if tmp_dist[y] < threshold_scale*id_sizes[i]:
                        thresholded_tmp_dist.append(tmp_dist[y]/id_sizes[i])
                        tmp_veto_flag.append(False)
                    
                    # take appearance feature into account
                    elif tmp_dist[y] < 6*threshold_scale*id_sizes[i]:
                        thresholded_tmp_dist.append(tmp_dist[y]/id_sizes[i])
                        tmp_veto_flag.append(True)
                        
                    else:
                        thresholded_tmp_dist.append(None)
                        tmp_veto_flag.append(False)
                    
                pair_dist.append(tuple(thresholded_tmp_dist))
                appearance_veto_flag.append(tuple(tmp_veto_flag))
            
            # not in occlusion state    
            else:
                pair_dist.append(tuple(map(lambda x: x/id_sizes[i] if x<threshold_scale*id_sizes[i] else None, tmp_dist)))
                appearance_veto_flag.append(tuple([False]*3))
        else:
            # initial id pair doesn't take occlusion into account
            pair_dist.append(tuple(map(lambda x: x/id_sizes[i] if x<threshold_scale*id_sizes[i] else None, tmp_dist))) 
            appearance_veto_flag.append(tuple([False]*3))
            
        dist_map.append(tuple(x for j,x in enumerate(dist_map_[i]) if x is not None and pair_dist[i][j] is not None))    # within threshold
        
        # ---------------------
        # calculate velocity pairing cost
        # ---------------------
        if id_list is not None:     # initial id pair doesn't take velocity into account
            tmp_velocity = [(center_post[j] - id_list[i].pos_pre)/id_list[i].interval for j in dist_map[i]]
            pair_velocity_ = list(map(lambda x: angle_between(x, id_list[i].v)/(np.pi/4) 
                                                    if id_list[i].v is not None and angle_between(x, id_list[i].v)<np.pi/2 else None, 
                                      tmp_velocity))
                                           
            for x in range(len(pair_velocity_)):
                if L2distance(tmp_velocity[x])<id_sizes[i]*0.015*threshold_scale:  #eliminate small velocity
                    if id_list[i].v is not None and L2distance(id_list[i].v)<id_sizes[i]*0.045*threshold_scale:
                        pair_velocity_[x] = 0
                    elif id_list[i].v is None:
                        pair_velocity_[x] = 0     
                                
            pair_velocity.append(tuple(pair_velocity_))   # pair_velocity: 0:unable to decide; None: velocity requirment unsatisfied; otherwise: velocity pair cost
        # ---------------------
        # calculate feature pairing cost
        # ---------------------
        if id_list is not None:
            feature_dist_ = [L2distance(id_list[i].features - features[x]) for x in dist_map_[i] if x is not None]
            feature_dist_filtered = [x if x<0.5 else None for x in feature_dist_]
            pair_feature.append(tuple(feature_dist_filtered))    
        
    if id_list is not None:
        if features is not None:
            pos_map = assign(dist_map, pair_dist, pair_velocity, pair_feature, num_id_pre, num_id_post, appearance_veto_flag)
        else:
            pos_map = assign(dist_map, pair_dist, 
                             pair_velocity, 
                             np.full((num_id_pre, num_id_post), None), 
                             num_id_pre, num_id_post, 
                             np.full((num_id_pre, num_id_post), False))
    else:
        # initial ids
        pos_map = assign(dist_map, pair_dist, 
                         np.full((num_id_pre, num_id_post), None), 
                         np.full((num_id_pre, num_id_post), None), 
                         num_id_pre, num_id_post, 
                         np.full((num_id_pre, num_id_post), False))
    
    if all(pos_map[i] is None for i in pos_map):
        return pos_map, False
    else:
        return pos_map, True    

    
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



# =============================================
# YOLOv3 detection
# =============================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]



    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    

    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
   
    
    return prediction

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_im_dim(im):
    im = cv2.imread(im)
    w,h = im.shape[1], im.shape[0]
    return w,h

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
    

    
def write_identity(frame, id_list, candidate_list, trace_flag=True):

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
        if  trace_flag:
            for h in range(len(id_list[i].hist)):
                cv2.circle(frame, id_list[i].hist[h], 3, color, -1)
            cv2.circle(frame, tuple(id_list[i].pos.astype('int')), 3, color, -1)
        
    for i in range(len(candidate_list)):
        c1 = (candidate_list[i].upper_left[0].astype('int'), candidate_list[i].upper_left[1].astype('int'))
        c2 = (candidate_list[i].lower_right[0].astype('int'), candidate_list[i].lower_right[1].astype('int'))
        label = "{0}:{1}".format('Candidate', candidate_list[i].name)
        color = (255, 255, 255)
        cv2.rectangle(frame, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(frame, c1, c2, color, -1)
        cv2.putText(frame, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        
    return frame
    
    
def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False


    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

		
        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output


def write_results_person(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
#    import pdb
#    pdb.set_trace()

    person_cls_index = torch.nonzero((torch.argmax(prediction[:, :, 5:5+num_classes], 2) == 0).squeeze()).squeeze()
    prediction = torch.index_select(prediction, 1, person_cls_index.long())

    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False


    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score   
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        cls = 0    # only do detection for person class
        #get the detections with one particular class
        cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
        class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
        

        image_pred_class = image_pred_[class_mask_ind].view(-1,7)

	
    
         #sort the detections such that the entry with the maximum objectness
         #confidence is at the top
        conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
        image_pred_class = image_pred_class[conf_sort_index]
        idx = image_pred_class.size(0)
        
        #if nms has to be done
        if nms:
            #For each detection
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
    
                except IndexError:
                    break
                
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
                
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:12:16 2018

@author: ayooshmac
"""

def predict_transform_half(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    grid_size = inp_dim // stride

    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda().half()
        y_offset = y_offset.cuda().half()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.HalfTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:,:, 5 : 5 + num_classes])).data

    prediction[:,:,:4] *= stride
    
    
    return prediction


def write_results_half(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).half().unsqueeze(2)
    prediction = prediction*conf_mask
    
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    
    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:]
        except:
            continue
        
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1].long()).half()
        
        
        
                
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).half().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind]

        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).half().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind]
                    
                    
            
            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output
