# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:46:04 2019

@author: filipavaldeira
"""

from shapes.sample import Shape
from utils.plot_shapes import plot_dataset
import open3d as o3d
import numpy as np
import os
import random
from utils.transformations import bounding_box_from_center
from utils.convert import find_complementary_id,array_filter_id_nan
from shapes.modify_shapes import RandomNoise,RandomMissing,StructOutliers,StructMissing, StructMissingID
from utils.transformations import centroid_size
from utils.io import pts_to_mesh, read_fish_data
from scipy.io import loadmat
from utils.transformations import find_closest_pts

import time
import pandas as pd
# Cretaes new dataset from dictionary of points and dictionary of correspondences
def create_dataset(new_pts_dict,new_corr_dict,dim):
    
    new_dataset = ShapeDataset(dim)
    for id_, pts in new_pts_dict.items():
        new_dataset.add_shape(int(id_), pts)
        corr = new_corr_dict[id_]
        new_dataset.add_corresp(int(id_),corr)      
    
    return new_dataset


##############################################################################
# Miscelalneous
##############################################################################

def subsample_random_corrs(og_shape,dim,miss_ratio):
    dataset = ShapeDataset(dim)
    dataset.add_shape(int(1),og_shape)
    id_list='all'
    data_miss = dataset.random_missing_data(miss_ratio,id_list)
    template_sub = data_miss.shapes_dict[1].points
    id_corr = data_miss.corresp_dict[1]
    
    return template_sub,id_corr

##############################################################################
# Dataset creation
##############################################################################

# Specific function to read fish dataset into datset format
def read_folder_fish(src_path,dim=2):
    
    dataset = ShapeDataset(dim)
    complete_dataset = ShapeDataset(dim)

    id_subj = 0
    for file in os.listdir(src_path):
        
        print ("Reading file: {} ".format(file))
                   
        # read file and add to dataset
        full_file = os.path.join(src_path, file)
        
        data = loadmat(full_file)
        template = data['x1']
        target = data['y2a']
        def_template_goal = data['y2']
        
        
        
        dataset.add_shape(int(id_subj),target)
        dataset.shapes_dict[int(id_subj)].file_name = file

        complete_dataset.add_shape(int(id_subj),def_template_goal)
        complete_dataset.shapes_dict[int(id_subj)].file_name = file
        
        if id_subj == 0:
            true_template = template
            corr = np.arange(template.shape[0])
        else:
            dist,corr = find_closest_pts(template,true_template,max_dist=1e-6)
        
        # For each shape is an array with same number of points of the shape. 
        # If shape has outliers NaN, if missing points then the id is not present in the array
        
        n_template, n_target = template.shape[0], target.shape[0]
        if n_template == n_target:
            #print('One to one correspondence')
            dataset.add_corresp(int(id_subj),corr)
        elif n_template > n_target:
            # missing points in target
            #print('Missing points')
            dataset.add_corresp(int(id_subj),corr[:n_target])
        else:
            # outliers in target
            #print('Outliers')
            corr_arr = np.arange(n_target).astype(float)
            corr_arr[:n_template] = corr
            corr_arr[n_template:] = np.nan
            dataset.add_corresp(int(id_subj),corr_arr)
            

        id_subj += 1
    return dataset, true_template, complete_dataset


def read_data_matrix(matrix, id_vec = None):
    #matrix should be size (n_shapes,n_points,dim)
    # shapes necessarily have the same number of points correctly registered
    
    dim = matrix.shape[2]
    n_shapes = matrix.shape[0]
    if id_vec is None:
        id_vec = np.arange(1,n_shapes+1)
    
    dataset = ShapeDataset(dim)
    
    for id_subj, shape in zip(id_vec, matrix):

        dataset.add_shape(int(id_subj),shape)
        print ("Adding shape with id: {} ".format(id_subj)) 
        
    return dataset

def read_data_folder_obj(src_path, exclude_path,dim,flag_clean_shape=None,other_sep=None,separation=','):
    # other sep : use other symbol to retrieve name from shape file
    dataset = ShapeDataset(dim)
    if(flag_clean_shape):
        print(" All meshes will have duplicate traingles and vertices removed")
    
    for file in os.listdir(src_path):
        
        # read file and add to dataset
        full_file = os.path.join(src_path, file)
        if full_file == os.path.join(src_path, exclude_path):
            print('Excluding file from exclude_path')
            continue
        
        # Get id number
        sep = file.split('.')
        if(other_sep is not None):
            sep_2 = sep[0].split(other_sep)
            id_subj = sep_2[-1]
        else:  
            id_subj = sep[0]
            

        if file.endswith('.ply') | file.endswith('.stl') | file.endswith('.obj') :
            mesh = o3d.io.read_triangle_mesh(full_file)   
            if(flag_clean_shape):
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            dataset.add_mesh(int(id_subj),vertices,triangles)

        elif file.endswith('.csv')| file.endswith('.txt'):
            df = pd.read_csv(full_file,header=None,sep=separation)
            data = df.values[:,0:dim] 
            dataset.add_shape(int(id_subj),data)
        else:
            continue    
        print ("Reading file of subject: {} ".format(id_subj)) 
        
    return dataset


# Reads all files in src_path except for the ones in exclude_path
# returns a list of matrices, where each matrix corresponds to the points in one file
def read_data(src_path, exclude_path):

    start = time.time()

    matrix_list = list()
    id_vec = list()

    for file in os.listdir(src_path):
        if file.endswith(".csv"):
            source_path = os.path.join(src_path, file)
            if(source_path == exclude_path):
                print('Found template, do not register')
                continue
            sep = file.split('.')
            id_subj = int(sep[0])
            
            print ("Reading file of subject: {} ".format(id_subj))         
            df = pd.read_csv(source_path,header=None )
            #data = df.to_numpy()
            data = df.values
                        
            matrix_list.append(data)
            id_vec.append(id_subj)
        else:
            print('Found non .csv file')
        
    end = time.time() 
    print("Total reading time: {:.4f}".format(end-start))
    
    return matrix_list, np.asarray(id_vec)


        
#################### CLASS Dataset ####################
class ShapeDataset(object):
    
    def __init__(self, dim):       
           
        self.dim = dim
        self.n_samples = 0
        self.shapes_dict = dict() # list of shapes objects
        self.dataset_type = 'non_def' # point cloud or mesh, here is still not defined
        
        # For each shape is an array with same number of points of the shape. 
        # If shape has outliers NaN, if missing points then the id is not present in the array
        self.corresp_dict = dict() 
    
    # ------------- Add new shapes ------------- #

    # Adds mesh shape with id_, vertices and triangles
    def add_mesh(self,id_,vertices,triangles):
        if(self.dataset_type=='point_cloud'):
            print('Do not mix dataset of pointcloud and mesh. Will not add new shape')
        else :
            self.dataset_type='mesh'
            print('Adding new shape to dataset: Mesh with id {}'.format(id_))
            shape = Shape()
            shape.add_points(vertices), shape.add_triangles(triangles)
            
            # Update dataset info
            self.shapes_dict[id_] = shape
            self.n_samples = self.n_samples + 1       
            self.add_corresp(id_,np.arange(vertices.shape[0])) # Init correspondence with ids
    
    # Adds point cloud shape with id_ and points
    def add_shape(self,id_,points): 
        if(self.dataset_type=='mesh'):
            print('Do not mix dataset of pointcloud and mesh. Will not add new shape')
        else :
            self.dataset_type='point_cloud'
            print('Adding new shape to dataset: Point Cloud with id {}'.format(id_))
            shape = Shape()
            shape.add_points(points)
            
            # Update dataset info
            self.shapes_dict[id_] = shape
            self.n_samples = self.n_samples + 1 
            self.add_corresp(id_,np.arange(points.shape[0])) # Init correspondence with ids
            
    # Add list of shapes
    def add_list_shapes(self,id_list,pts_list):
        for id_,shape in zip(id_list,pts_list):
            self.add_shape(id_,shape)   
    
    
    # ------------- Correspondences ------------- #
    
    # Set correspondence dict for shape with id_
    def add_corresp(self,id_,points):
        self.corresp_dict[id_] = points

    # Update all correspondences given new template order
    def update_corr_after_template_change(self, new_template_ids):
        for id_, corr_vec in self.corresp_dict.items():
            new_corr = array_filter_id_nan(new_template_ids,corr_vec)
            self.corresp_dict[id_] = new_corr
            
    # ------------- Retrieve info ------------- #
    
    # Get ids of all shapes
    def get_id_list(self):
        return list(self.shapes_dict.keys())
    
    # Return shapes in id_list
    def get_shape_pts_list(self,id_list):
        # id_list : can be list, 'all' or int
        if(isinstance(id_list,list)):
            shape_list = [self.shapes_dict[x].points for x in id_list]
            ids = id_list
        elif(id_list=='all'):
            ids = list(self.shapes_dict.keys())
            shape_list = [self.shapes_dict[x].points for x in ids]
        elif(isinstance(id_list,int)):
            ids = id_list
            shape_list = [self.shapes_dict[ids].points]
            
        return ids, shape_list    
    
    # Returns list with points of id_points for every shape
    def get_shape_points_by_id(self,id_points):
        shape_list = [shape.points[id_points,:] for shape in self.shapes_dict.values()]
        return shape_list  
    
    
    # ------------- Modify shapes ------------- #
    def normalize_centroid_sz(self):
        for id_, shape in self.shapes_dict.items():
            s = centroid_size(shape.points)
            self.shapes_dict[id_].points = shape.points*(1/s)

    def set_origin_by_user_pt(self,point):
        for id_, shape in self.shapes_dict.items():
            shape.set_origin_by_user_pt(point)
    
    def apply_rigid_transform(self,tform, id_vec):
        for id_ in id_vec:
            shape = self.shapes_dict[id_]
            shape.apply_rigid_transform(tform)

    
    # ------------- Create noisy datasets ------------- #    
    def struct_missing_data(self,center,width,id_list):
        print('sssss')
        # by area 
        miss_area_pts = bounding_box_from_center(center,width,self.dim)        
        trf = StructMissing(miss_area_pts,self.dim)
        new_pts_dict,new_corr_dict = trf.transform_shapes(id_list,self.shapes_dict.copy(),self.corresp_dict.copy())
        # Create new dataset with transformed shapes and correspondences        
        new_dataset =  create_dataset(new_pts_dict,new_corr_dict,self.dim)
        
        return new_dataset
 
    def struct_missing_data_ID(self,id_miss,id_list):
        # sturct missing by id instead of area     
        # id_miss = ids to be removed
        trf = StructMissingID(id_miss,self.dim)
        new_pts_dict,new_corr_dict = trf.transform_shapes(id_list,self.shapes_dict.copy(),self.corresp_dict.copy())
        # Create new dataset with transformed shapes and correspondences        
        new_dataset =  create_dataset(new_pts_dict,new_corr_dict,self.dim)
        return new_dataset   
 
    # Returns new dataset with random missing data
    def random_missing_data(self,miss_ratio,id_list):  
        """ Inserts missing data leaving miss_ratio of points for each target 
        shape in id_list

            Parameters
            ----------
            miss_ratio : float
                percentage of points that will remain in the new dataset
                
            id_list : list
                list of shape ids where to apply transformation

            Returns
            ------
            new_dataset : Dataset
                new dataset with missing data

        """    

        trf = RandomMissing(miss_ratio)
        new_pts_dict,new_corr_dict = trf.transform_shapes(id_list,self.shapes_dict,self.corresp_dict)
        # Create new dataset with transformed shapes and correspondences        
        new_dataset =  create_dataset(new_pts_dict,new_corr_dict,self.dim)
        return new_dataset
                
    # Returns new dataset with noisy data        
    def random_noisy_data(self,noise_var,id_list):        
        trf = RandomNoise(noise_var)
        new_pts_dict,new_corr_dict = trf.transform_shapes(id_list,self.shapes_dict,self.corresp_dict)
        # Create new dataset with transformed shapes and correspondences        
        new_dataset =  create_dataset(new_pts_dict,new_corr_dict,self.dim)
        return new_dataset

    def outliers_struct(self,outlier_ratio,low_bound,high_bound,id_list):        
        trf = StructOutliers(outlier_ratio,low_bound,high_bound)
        new_pts_dict,new_corr_dict = trf.transform_shapes(id_list,self.shapes_dict,self.corresp_dict)
        # Create new dataset with transformed shapes and correspondences        
        new_dataset =  create_dataset(new_pts_dict,new_corr_dict,self.dim)
        return new_dataset

    
    # ------------- Plotting ------------- #
    def plot_shapes(self,id_list):       
        ids,shapes = self.get_shape_pts_list(id_list)
        title = 'Plot shapes'
        legend = ids
        ax=  plot_dataset(shapes,title,legend)
        return ax


    # ------------- IO ------------- #
    

    def save_to_files(self,dest_folder,file_type):
        print('Saving dataset to folder {}'.format(dest_folder))
        
        print('---------- Will estimate normals and create meshes from point cloud ------------')
        

        if(file_type == 'ply'):
            print('Saving files as .ply')
            
            for id_, shape in self.shapes_dict.items():
                shape_ref = shape.get_o3d_mesh()
                
                stl_path = os.path.join(dest_folder, str(id_)+'.ply')
                print('Saving {}'.format(stl_path))
                o3d.io.write_triangle_mesh(stl_path, shape_ref,print_progress =True)
                
        elif(file_type == 'csv'):
            print('Saving files as .csv')
            for id_, shape in self.shapes_dict.items():
                csv_path = os.path.join(dest_folder, str(id_)+'.csv')
                print('Saving {}'.format(csv_path))
                np.savetxt(csv_path, shape.points, delimiter=",")
        elif(file_type == 'txt'):
            print('Saving files as .csv')
            for id_, shape in self.shapes_dict.items():
                csv_path = os.path.join(dest_folder, str(id_)+'.txt')
                print('Saving {}'.format(csv_path))
                np.savetxt(csv_path, shape.points, delimiter=",")

                
        elif(file_type == 'stl'):
            print('Saving files as .stl')
            
            for id_, shape in self.shapes_dict.items():
                #print(shape.points.shape)
                shape_ref = pts_to_mesh(shape.points)
                #shape_ref = shape.get_o3d_mesh()
                shape_ref.compute_vertex_normals()
                stl_path = os.path.join(dest_folder, str(id_)+'.stl')
                print('Saving {}'.format(stl_path))
                o3d.io.write_triangle_mesh(stl_path, shape_ref,print_progress =True)
          