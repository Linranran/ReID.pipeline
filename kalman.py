from idclass import identity
from util import (argmin_top3, 
                  L2distance, 
                  assign, 
                  unit_vector, 
                  angle_between, 
                  write_identity, 
                  pair_position,
                  calibrate_unreliable_pos,
                  angle_between)
                  
import numpy as np
import pdb

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
    
    
def kalman_update(id_list, candidate_list, measured_position, features, R):
    # heuristically solve occlusion using body ratio prior
    measured_position, features = calibrate_unreliable_pos(measured_position, features)
    
    ttl_list = id_list + candidate_list 
    legit_num = len(id_list)
    cand_num = len(candidate_list)
    est_position = np.zeros((len(ttl_list), 4))
    for i in range(len(ttl_list)):
        est_position[i, :2] = ttl_list[i].pos
        est_position[i, 2:4] = ttl_list[i].lower_right
    
    pos_map, pair = pair_position(est_position, measured_position, features=features, id_list=ttl_list)  # pos_map: dict{pre_idx: post_idx}
    
    accounted = np.full(measured_position.shape[0], False)
    
    if not pair:
        return id_list, candidate_list
    for i in range(len(pos_map)):
        if pos_map[i] is not None:          # found pairs
            
            accounted[pos_map[i]] = True
            ttl_list[i].K = ttl_list[i].P / (ttl_list[i].P + R*(1/ttl_list[i].size))
            ttl_list[i].P -= ttl_list[i].K * ttl_list[i].P
            
            if i < legit_num:
                # confirmed ids
                # recalibrate measured position using history information
                raw_measured_position = measured_position[pos_map[i], :2]
                raw_lower_right = measured_position[pos_map[i], 2:4]
                raw_upper_left = measured_position[pos_map[i], :2] \
                                    - (measured_position[pos_map[i], 2:4] 
                                          - measured_position[pos_map[i], :2])
                
                raw_height = (raw_lower_right[1] - raw_measured_position[1]) * 2
                raw_width = (raw_lower_right[0] - raw_measured_position[0]) * 2
                
                pre_height = (ttl_list[i].lower_right[1] - ttl_list[i].pos[1]) * 2
                pre_width = (ttl_list[i].lower_right[0] - ttl_list[i].pos[0]) * 2
                
                cali_height = 0.8 * pre_height + 0.2 * raw_height
                #cali_width = 0.8 * pre_width + 0.2 * raw_width
                cali_width = raw_width
                
                # assume upper left point is always correct
                cali_pos = (raw_upper_left[0] + cali_width/2, 
                            raw_upper_left[1] + cali_height/2,
                            raw_upper_left[0] + cali_width,
                            raw_upper_left[1] + cali_height)
                
                ttl_list[i].pos += ttl_list[i].K * (cali_pos[:2] - ttl_list[i].pos)
            else:
                # candidates
                cali_pos = measured_position[pos_map[i], :]
                ttl_list[i].pos = cali_pos[:2]
                
            ttl_list[i].pos[0].clip(min=0, max=identity.max_dim[0])
            ttl_list[i].pos[1].clip(min=0, max=identity.max_dim[1])
            
            ttl_list[i].lower_right += ttl_list[i].K * (cali_pos[2:4] - ttl_list[i].lower_right)
            ttl_list[i].lower_right[0].clip(min=0, max=identity.max_dim[0])
            ttl_list[i].lower_right[1].clip(min=0, max=identity.max_dim[1])
            
            
            if ttl_list[i].v is None:
                ttl_list[i].v = (ttl_list[i].pos - ttl_list[i].pos_pre) / ttl_list[i].interval
            else:
                # recalibrate measured velocity
                if i < legit_num:
#                    pdb.set_trace()
                    current_v = ttl_list[i].v + ttl_list[i].K \
                                    * ((ttl_list[i].pos - ttl_list[i].pos_pre) \
                                        / ttl_list[i].interval - ttl_list[i].v)
                    if L2distance(current_v) < ttl_list[i].speed_upper_bound:
                        pre_v = ttl_list[i].v.copy()
                        
                        # abrupt velocity change, could be due to detection artifacts
                        if angle_between(current_v, pre_v)/(np.pi/4) < 0.85 \
                            and L2distance(current_v)<ttl_list[i].size*0.0225:
                            
                            if ttl_list[i].v_pre is None:
                                ttl_list[i].v_pre = pre_v
                            else:
                                ttl_list[i].v_pre = None
                                
                        ttl_list[i].v = current_v
                        ttl_list[i].volatile_flag = False
                    else:
                        ttl_list[i].volatile_flag = True
                    
            ttl_list[i].pos_pre = ttl_list[i].pos.copy()
            ttl_list[i].interval = 0
            ttl_list[i].missed = 0
            ttl_list[i].feature_list.append(features[pos_map[i]])
            
            # update candidate_list and id_list
            if i >= legit_num:
                id_list.append(ttl_list[i])
                candidate_list.remove(ttl_list[i])
        else:   
            # unpaired
            # could be exiting ids, add to candidates
            if i < legit_num:
                ttl_list[i].missed += 1
                
                if ttl_list[i].volatile_flag and ttl_list[i].missed > 1:
                    candidate_list.append(ttl_list[i])
                    id_list.remove(ttl_list[i])
                else:
                    # revert velocity updates if abrupt change happens and no further confirmation
                    if ttl_list[i].missed == 1 and ttl_list[i].v_pre is not None:
                        ttl_list[i].v = (ttl_list[i].v_pre + ttl_list[i].v)/2
                        
                    if ttl_list[i].missed == 2 and ttl_list[i].v_pre is not None:
                        ttl_list[i].v = ttl_list[i].v_pre
                        ttl_list[i].v_pre = None
                    
                    if ttl_list[i].missed > 15:
                        candidate_list.append(ttl_list[i])
                        id_list.remove(ttl_list[i])
                    else:
                        if ttl_list[i].exiting and ttl_list[i].missed > 5:
                            candidate_list.append(ttl_list[i])
                            id_list.remove(ttl_list[i])
            else:
                # was in candidate and not detected this time, could be false alarm, or exiting ids
                candidate_list.remove(ttl_list[i])
                identity.colors_used[ttl_list[i].colorid] = False
    
    # deal with new identities
    for i in range(measured_position.shape[0]):
        if not accounted[i]:
            can_id = identity()
            can_id.feature_list.append(features[i])
            can_id.pos = measured_position[i, :2]
            can_id.pos_pre = can_id.pos.copy()
            can_id.lower_right = measured_position[i, 2:4]
            candidate_list.append(can_id)
                
    return id_list, candidate_list
