import pickle as pkl
import uuid
import numpy as np

class identity(object):
    colors = pkl.load(open("pallete", "rb"))
    colors_used = np.full(len(colors), False)
    max_dim = None
    interval = None
    max_v = None
    
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
        self.v_pre = None
        self.volatile_flag = False
#        self.hypo_pos = None
        self.lower_right = None
        self.P = 0
        self.K = 0
        self.missed = 0
        self.feature_list = []
    
    @property
    def features(self):
        if self.feature_list is []:
            return None
        if len(self.feature_list) > 20:
            del self.feature_list[1:-21]   # keep the first one and the most recent 19 features
        return np.mean(np.array(self.feature_list), axis=0, keepdims=True)
    
    @property
    def upper_left(self):
        if self.lower_right is None or self.pos is None:
            return None
        else:
            upper_left_ = (self.pos - (self.lower_right - self.pos))
            upper_left_[0].clip(min=0, max=identity.max_dim[0])
            upper_left_[1].clip(min=0, max=identity.max_dim[1])
            return upper_left_
            
    @property
    def size(self):
        if self.lower_right is None or self.pos is None or identity.max_dim is None:
            return None
        return np.sqrt(np.sum(np.square(self.lower_right - self.pos))) / (np.sqrt(np.sum(np.square(identity.max_dim)))/2)
        
    @property
    def speed_upper_bound(self):
        height = (self.lower_right[1] - self.pos[1])*2
        return height * identity.max_v
            
    def add_hist(self):
        self.hist.append(tuple(self.pos.astype('int')))
        return
        
    @property
    def exiting(self):
        margin = 0.05
        if self.v is not None:
            if self.v[0] < 0 and self.pos[0] < margin * identity.max_dim[0]:
                return True
            if self.v[0] > 0 and self.pos[0] > (1-margin) * identity.max_dim[0]:
                return True
            if self.v[1] < 0 and self.pos[1] < margin * identity.max_dim[1]:
                return True
            if self.v[1] > 0 and self.pos[1] > (1-margin) * identity.max_dim[1]:    
                return True
        return False

        
        
