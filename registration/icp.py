# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 10:45:02 2022

@author: filipavaldeira
"""

import copy
import numpy as np
import pandas as pd

# Third party imports
import open3d as o3d # ICP registration

from registration.registration import ShapesRegistration
from utils.convert import numpy2PointCloud


###############################################################################        
## ---------------------------------- ICP ---------------------------------- ##         
############################################################################### 
        
class IcpRegistration(ShapesRegistration):
    def __init__(self,  reg_type, max_dist, initial_transf = None, scaling =False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.reg_type = reg_type 
        self.threshold = max_dist
        if(self.reg_type == 'point_to_point'):
            self.reg_method = 'ICP (point to point)' 
            self.reg_scale = scaling
        elif(self.reg_type == 'point_to_plane'):
            self.reg_method = 'ICP (point to plane)'
            self.reg_scale = None #TODO why not applicable 
        
        # Maximum distance between def source and target to allow a correspondence
        #self.max_dist = max_dist
        
        if(initial_transf is None):
            self.initial_transf = np.asarray([[1, 0,0, 0],
                             [0, 1,0,0],
                             [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
        else:
            self.initial_transf = initial_transf
        #print('Initial transform')
        #print(self.initial_transf)    
        
        self.parameters_str = 'Max dist = {}. Scaling = {}'.format(max_dist,scaling)
        
    def pair_registration(self,target,source):
        # how to perform registration between a target and a source
        # input : numpy matrices
        # Returns 
        # row_vec, deformed_source, target_non_assigned,
        # --- template_non_assigned_mask : boolean vector size template, with true at non assigned points of source #TODO is this same as NaN in correspond_veC???
        # --- corr_vec : vector size of template (source) with ids of target which correspond. If source point does not have correspondence with any target, id set to NaN

        src = numpy2PointCloud(source)
        target = numpy2PointCloud(target)
        #o3d.geometry.estimate_normals(src)
        #o3d.geometry.estimate_normals(target)
        src.estimate_normals()
        target.estimate_normals()
        
        
 
        template_shape = np.array(src.points).shape
        
        # ICP registration different types
        if(self.reg_type == 'point_to_point'):
            reg_p2p = o3d.pipelines.registration.registration_icp(
        src, target, self.threshold, self.initial_transf ,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=self.reg_scale))            
        elif(self.reg_type == 'point_to_plane'):
            reg_p2p = o3d.pipelines.registration.registration_icp(
        src, target, self.threshold, self.initial_transf ,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())                        
        
        # Get row vector into desired shape
        correspondence_vec = np.array(reg_p2p.correspondence_set)
        #correspondence_vec : column0 has the indices of the source and column 1 has the correspondent indices of target
        print('Fitness {}'.format(reg_p2p.fitness))
        print('Inlier RMSE {} '.format(reg_p2p.inlier_rmse))
        #print(correspondence_vec.shape)
        keep_id = correspondence_vec[:,1]
        # target_pts = np.array(target.points)
        # target_correspondence = np.zeros((correspondence_vec.shape[0],3))
        # print(target_pts.shape)
        # print(correspondence_vec.shape)
        # print(target_correspondence.shape)
        # target_correspondence[correspondence_vec[:,0],:] = target_pts[keep_id,:]
        # if(target_correspondence.shape[0] != template_shape[0]):
        #     print('---------------------------- NONE REG ----------------------------')
        #     print(target_correspondence)
        #     return None, None
        
        # Build corr_vec in desired shape: id of target in order (size of source)
        corr_vec = np.empty((template_shape[0]))
        corr_vec[:]  = np.nan
        corr_vec[correspondence_vec[:,0]] = keep_id
        #print(corr_vec)

        # obtain deformed source
        transf_source = copy.deepcopy(src)
        transf_source = transf_source.transform(reg_p2p.transformation)
        arr = np.asarray(transf_source.points)
        deformed_source = np.reshape(arr[:,0:self.dim],(1,-1)) #remove stub zeros and reshape
        
        # Maximum distance allowed for correspondences
        # reg_target=target_correspondence[:,0:self.dim]
        # def_src = arr[:,0:self.dim]
        # corr_vec = self.filter_max_dist(corr_vec,reg_target,def_src)
        
        return deformed_source,corr_vec
    
    def filter_max_dist(self,corr_vec,reg_target,def_src):
        diff = def_src-reg_target
        dist_pts = np.sum(np.square(diff),axis=1)
        mask = (dist_pts>self.max_dist)
        corr_vec[mask] = np.nan
        
        return corr_vec        
        
    def read_template(self,template_path):
        df = pd.read_csv(template_path,header=None )
        template = df.to_numpy()

        self.dim = template.shape[1]
        self.n_points = template.shape[0]
        self.template = template
