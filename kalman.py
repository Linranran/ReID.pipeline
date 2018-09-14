from idclass import identity
from util import (argmin_top3, 
                  L2distance, 
                  assign, 
                  unit_vector, 
                  angle_between, 
                  write_identity, 
                  pair_position)
                  
import numpy as np

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

    ttl_list = id_list + candidate_list 
    legit_num = len(id_list)
    cand_num = len(candidate_list)
    est_position = np.zeros((len(ttl_list), 4))
    for i in range(len(ttl_list)):
        est_position[i, :2] = ttl_list[i].pos
        est_position[i, 2:4] = ttl_list[i].lower_right
    
#    pdb.set_trace()
    
    pos_map, pair = pair_position(est_position, measured_position, id_list=ttl_list)
    
    accounted = np.full(measured_position.shape[0], False)
    
    if not pair:
        return id_list, candidate_list
    for i in range(len(pos_map)):
        if pos_map[i] is not None:
            
            accounted[pos_map[i]] = True
            
            
            ttl_list[i].K = ttl_list[i].P / (ttl_list[i].P + R*(1/ttl_list[i].size))
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
            
            
            if ttl_list[i].v is None:
                ttl_list[i].v = (ttl_list[i].pos - ttl_list[i].pos_pre) / ttl_list[i].interval
            else:
                ttl_list[i].v += ttl_list[i].K * ((ttl_list[i].pos - ttl_list[i].pos_pre) / ttl_list[i].interval - ttl_list[i].v)
            ttl_list[i].pos_pre = ttl_list[i].pos.copy()
            ttl_list[i].interval = 0
            ttl_list[i].missed = 0
            
            # update candidate_list and id_list
            
            if i >= legit_num:
                id_list.append(ttl_list[i])
                candidate_list.remove(ttl_list[i])
        else:
            # could be exit ids, add to candidates
            if i < legit_num:
                ttl_list[i].missed += 1
                if ttl_list[i].missed > 15:
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
