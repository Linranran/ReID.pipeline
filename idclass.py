import pickle as pkl
import uuid
import numpy as np

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
            
    @property
    def size(self):
        if self.lower_right is None or self.pos is None or identity.max_dim is None:
            return None
        return np.sqrt(np.sum(np.square(self.lower_right - self.pos))) / (np.sqrt(np.sum(np.square(identity.max_dim)))/2)
            
            
    def add_hist(self):
        self.hist.append(tuple(self.pos.astype('int')))
        return
