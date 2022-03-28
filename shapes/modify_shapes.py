# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:18:28 2020

@author: filipavaldeira
"""

import numpy as np
import random

import open3d as o3d

from utils.convert import find_complementary_id

# General class to transform a shapes dict
class ModifyShapes(object):
        
    def transform_shapes(self,id_list,shapes_dict,corresp_dict):
        
        new_pts_dict, new_corr_dict = dict(), dict()
        
        if id_list=='all':            
            id_list = list(shapes_dict.keys())
        
        for id_, shape in shapes_dict.items():            
            old_corr = corresp_dict[id_]
            
            if id_ in id_list:            
                # apply transformation depending on the class
                new_pts,new_corr = self.specific_transf(shape,old_corr)
            else:
                # return same shape and correspondence
                new_pts = shape.points
                new_corr = old_corr
            
            # Add shape and correspondence to new dataset
            new_pts_dict[id_] = new_pts
            new_corr_dict[id_]= new_corr
                        
        return new_pts_dict,new_corr_dict

###################### SPECIFIC TRANSFORM ############
# Should implement specific_transf function

# Creates new dataset with noisy data
# Same number of points with same id
class RandomNoise(ModifyShapes):
    
    def __init__(self, noise_var): 
        self.noise_var = noise_var
        
    def specific_transf(self,shape,old_corr):
        
        shape_ = shape.points.shape
        noise = np.random.normal(0, self.noise_var, shape_) 
        
        # Points updated with noise
        new_pts = shape.points+noise
        # Correspondence stays the same
        new_corr = old_corr
        
        return new_pts, new_corr


# Creates new dataset with random missing data
# Less points than before
class RandomMissing(ModifyShapes):
    
    def __init__(self, miss_ratio): 
        self.miss_ratio = miss_ratio
        
    def specific_transf(self,shape,old_corr):
        
        n_points = shape.points.shape[0]
        n_pts_picked = round(self.miss_ratio*n_points)
        id_array = np.arange(n_points)

        # chosen_id : ids to keep
        chosen_id = random.sample(list(id_array),n_pts_picked)
        new_pts = shape.points[chosen_id,:]
        new_corr = old_corr[chosen_id]
        
        return new_pts, new_corr

# Creates new dataset with structured missing data. Removes points inside area.
# Less points than before
class StructMissing(ModifyShapes):

    def __init__(self, miss_area_pts,dim): 
        # miss_area_pts are an array of pts defining the area to be removed
        self.miss_area = miss_area_pts
        self.bbox_miss_area = self.create_bbox()
        self.dim = dim
    
    def create_bbox(self):
        pcd = o3d.geometry.PointCloud()  
        pcd.points = o3d.utility.Vector3dVector(self.miss_area)
        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox_missing_area = bbox.create_from_points(pcd.points)
        return bbox_missing_area
    
    def specific_transf(self,shape,old_corr):
        if(self.dim==3):
            o3d_shape = o3d.utility.Vector3dVector(shape.points)
        elif(self.dim==2):
            # insert coordinatez as 0 in each point to use 3d tools
            shape_3d = np.zeros((shape.points.shape[0],3))
            shape_3d[:,0:2] = shape.points
            o3d_shape = o3d.utility.Vector3dVector(shape_3d)
            
            
        miss_ids = self.bbox_miss_area.get_point_indices_within_bounding_box(o3d_shape)               
        # Find ids to keep
        total_points =  shape.points.shape[0]
        total_id_vec = np.arange(shape.points.shape[0])
        keep_ids = find_complementary_id(miss_ids, total_points)

        # Update points and correspondences
        if(keep_ids.size==0):
            print(' Warning : After remove missing structure this shape has no points left!')
            # No points remain in the shape
            new_points = np.array([])
            new_corr = np.array([])
        
        else : 
            new_points = shape.points[keep_ids,:] 
            new_corr = old_corr[keep_ids]
                
        return new_points, new_corr

class StructMissingID(ModifyShapes):
    # Same as above but removes points by ids, for shapes that are registered

    def __init__(self, ids_miss, dim): 
        # miss_area_pts are an array of pts defining the area to be removed
        self.ids_miss = np.array(ids_miss).astype(int)
        self.dim = dim
    
    def specific_transf(self,shape,old_corr):                      
        miss_ids = self.ids_miss                         
        # Find ids to keep
        total_points =  shape.points.shape[0]
        total_id_vec = np.arange(shape.points.shape[0])
        keep_ids = find_complementary_id(miss_ids, total_points)

        # Update points and correspondences
        if(keep_ids.size==0):
            print(' Warning : After remove missing structure this shape has no points left!')
            # No points remain in the shape
            new_points = np.array([])
            new_corr = np.array([])
        
        else : 
            new_points = shape.points[keep_ids,:] 
            new_corr = old_corr[keep_ids]
                
        return new_points, new_corr




class StructOutliers(ModifyShapes):
    
    def __init__(self, out_ratio,low_bound,high_bound): 
        self.out_ratio = out_ratio
        self.low_bound = low_bound
        self.high_bound = high_bound
        
    def specific_transf(self,shape,old_corr):
        
        n_points = shape.points.shape[0]
        n_pts_outliers = round(self.out_ratio*n_points)
        
        new_pts = shape.points
        new_corr = old_corr
        for n_pts in np.arange(n_pts_outliers ):
            point = np.zeros((1,len(self.low_bound)))
            for dim in np.arange(len(self.low_bound) ):
                coor = random.uniform(self.low_bound[dim], self.high_bound[dim])
                point[0,dim] = coor
            new_pts = np.append(new_pts, point,axis=0)
            new_corr =np.append(new_corr,np.nan)
        
        
        print(shape.points)
        print(new_pts)
        return new_pts, new_corr
