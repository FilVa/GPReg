# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 09:59:52 2021

@author: filipavaldeira
"""
from scipy.spatial import KDTree
import numpy as np

### Compare both approaches
def mean_dist_between_shapes(shape1,shape2):
    #for each point of shape 2 we get the closest point of shape 1
    tree = KDTree(shape1)
    neighbor_dists, neighbor_indices = tree.query(shape2) #Query the kd-tree for nearest neighbors
    
    return np.mean(neighbor_dists), np.max(neighbor_dists), np.min(neighbor_dists)

def mean_dist_organized(shape1,shape2):
    diff = shape1-shape2
    mean_err = np.linalg.norm(diff,axis=1).mean()
    max_err = np.linalg.norm(diff,axis=1).max()
    min_err = np.linalg.norm(diff,axis=1).min()
    return mean_err, max_err, min_err

# Receives reg_metric and returns dictionaries
def get_distances(reg_obj, ids):
    dist_miss, dist_non_miss = dict(), dict()
    dist_by_point_miss, dist_by_point_non_miss = dict(), dict()    
    for id_ in ids:
        vec_gt = reg_obj.corr_gt[id_]
        def_template = reg_obj.reg_dataset.def_src_dict[id_]
        full_shape = reg_obj.complete_dataset.shapes_dict[id_].points


        mean_err, max_err, min_err=  mean_dist_organized(def_template[np.isnan(vec_gt),:],full_shape[np.isnan(vec_gt),:])
        dist_miss[id_] = (mean_err)
        mean_err, max_err, min_err=  mean_dist_organized(def_template[~np.isnan(vec_gt),:],full_shape[~np.isnan(vec_gt),:])
        dist_non_miss[id_] = (mean_err)

        dist_by_point_miss[id_] = np.linalg.norm(def_template[np.isnan(vec_gt),:] - full_shape[np.isnan(vec_gt),:],axis=1)
        dist_by_point_non_miss[id_] = np.linalg.norm(def_template[~np.isnan(vec_gt),:] - full_shape[~np.isnan(vec_gt),:],axis=1)
        
        
    return dist_miss, dist_non_miss, dist_by_point_miss, dist_by_point_non_miss