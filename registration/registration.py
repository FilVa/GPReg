# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:31:33 2019

@author: filipavaldeira
"""
# Standard library imports
import copy
import numpy as np
import sys
import pandas as pd
import time
from utils.io import read_mesh
# Local application imports
from utils.plot_shapes import plot_connected_shapes, plot_shapes
from utils.convert import flat2mat
#from registration.r_functions import r_Procrustes


from multiprocessing import Pool # Multiprocessing


from shapes.reg_dataset import RegDataset # REPLACE RegOutpu

###############################################################################
##################### General Class for Shape Registration ####################
###############################################################################


class ShapesRegistration(object):
    def __init__(self):
        self.id_vec = None # id in the same order as points are found on data matrices
        self.original_data = None # original data points as read from the files (list of arrays, each with possibly different number of points)
        
        self.dim = None
        self.n_points = None
        
        # Data matrices
        self.template = None
        self.landmark_matrix = None # samples with registered points
        self.deformed_source_matrix = None
        self.target_non_assigned_pts = None              

    # Register all files in src_path against template path
    def simple_registration(self,template, dataset, id_list,flag_parallel):
        if self.reg_method == 'BCPD' and flag_parallel:
            print('BCPD cannot be run in parallel mode - please change flag_parallel to False')
            quit()
        
        id_vec, data = dataset.get_shape_pts_list(id_list)        
        
        self.original_data = data
        self.id_vec = id_vec
        self.template = template
        self.dim = template.shape[1]
        self.n_points = template.shape[0]
        print('Start registration')
        # Register matrix
        deformed_src_matrix, corr_vec_list, reg_time = self.register_matrix(self.template,self.original_data,self.id_vec,flag_parallel)

        #TODO Update data matrices :WE DONT NEED THIS

        self.deformed_source_matrix = deformed_src_matrix
        #self.target_non_assigned_pts = non_assigned_target 
        if(self.reg_method == 'NICP'):
            src_vertices,src_faces = read_mesh(self.template_mesh_path)
            reg_output = RegDataset(dataset,src_vertices, deformed_src_matrix, corr_vec_list, data, id_vec, self.reg_method,reg_time,self.parameters_str)

        else:
            reg_output = RegDataset(dataset,self.template, deformed_src_matrix, corr_vec_list, data, id_vec, self.reg_method,reg_time,self.parameters_str)
        
        return reg_output      
    
    def register_new_shape(self, test_data, id_vec,flag_parallel):        
        deformed_src_matrix, non_assigned_target = self.register_matrix(self.template,test_data,id_vec, flag_parallel)        
        reg_output = RegDataset(self.template, deformed_src_matrix, test_data, id_vec,self.reg_method)
        
        return reg_output    
    
    def registration_iter_mean_shape():        
        print('to develop')
        # Register and use obtained mean shape as template in next iteration
        # But does not preform registration -> Maybe add if in registration_alignment_iter. TO just avoid procrustes
        
    def registration_alignment_iter(self,max_iter=50,tol = 0.01, flag_paralell=True):
        # TODO if landmark_matrix is None : do register_all
        initial_matching = self.landmark_matrix
        criterion = False
        old_mean_shape = None
        i = 0        
        print('DEPRECATED!!! IMPLEMENT AGAIN')
        
        # # Iterate Procrustes alignment and registration method until convergence or maximum iterations reached
        # matched_data = initial_matching
        # legend = ['shape 1','shape 2', 'shape 3','shape4']
        
        # while((criterion==False) and (i<max_iter)):
        #     print(i)
        #     #plot_shapes(flat2mat(matched_data,self.n_points,self.dim), 'Matched data', legend)
        #     # Generalized Procrustes Analysis
            
        #     proc = r_Procrustes(self.dim,self.n_points,initial_matching.shape[0])
        #     proc.r_procGPA(matched_data,'partial')
        #     mean_shape = proc.mean_shape
        #     aligned_data = proc.rotated_shapes
            
        #     plot_shapes(mean_shape, 'Procrustes mean shape', 'mean shape')
            
        #     plot_shapes(aligned_data, 'Rotated shapes', legend )
            
        #     # Match all rotated to mean shape from Procrustes
        #     final_matrix, deformed_src_matrix, non_assigned_target = self.register_matrix(mean_shape,aligned_data, self.id_vec,flag_paralell)
            
        #     matched_data = final_matrix
            
        #     # Check convergence criterion
        #     if(i>=1):
        #         diff = old_mean_shape-mean_shape
        #         dist_each_pt = np.linalg.norm(diff, axis=1)
        #         dist_sum = np.sum(dist_each_pt)
        #         print(dist_sum)
        #         if(dist_sum<=tol):
        #             criterion = True
                
        #     # Update cycle count and mean shape
        #     i += 1
        #     old_mean_shape = mean_shape
            
        # self.landmark_matrix = final_matrix
        # self.deformed_source_matrix = deformed_src_matrix
        # self.target_non_assigned_pts = non_assigned_target            
        # self.template = mean_shape
    
    def register_matrix(self,template,data,id_vec, flag_parallel):
        """ Registers a set of shapes in data, wrt template.
        If registration fails both deformed_src_matrix and corr_vec_list are set to nan

            Parameters
            ----------
            template : shape of N*dim
                
            data : 
                
            id_vec: list of len M
                ids of M shapes to be registered
                
            flag_parallel : bool
                if True performs registration in parallel with several processes, otherwise in series

            Returns
            ------
            deformed_src_matrix : numpy array len(id_vec)*(N dim)
                matrix with flattend deformed templates for each shape in order of id_vec
            
            corr_vec_list: list of M numpy arrays of shape N
                list of correspodences in template view
            
            reg_time: float
                total registration time
        """ 

        
        data_list, output = list(), list()
        num_processors = 3
        
        # create input list
        for n_sample in range(len(id_vec)):
            element_list = list()
            if(isinstance(data, list)):
                source = data[n_sample]
            else:
                source = data[:,:,n_sample] 
            element_list.append(source)
            element_list.append(template)
            element_list.append(id_vec[n_sample])
            data_list.append(element_list)
            
        start = time.time()
        if flag_parallel:
            # call registration in parallel
            print('Starting {} parallel processes'.format(num_processors))
            p = Pool(processes = num_processors)
            output = p.map(self.reg_process,[i for i in data_list])            
        else:
            print('Registration in series')
            # Registration in series
            for data_input in data_list:
                out = self.reg_process(data_input)
                output.append(out)
        end = time.time()
        
        print('End all registration {}'.format(end-start))
        reg_time = end-start
        
        # Handle output list
        if len(data) != len(output):
            print('SOMETHING IS WRONG - INPUT AND OUPUT DATA DO NOT HAVE SAME NUMBER OF ELMENTS - FIX CODE ')
            quit()#TODO pretty
        
        corr_vec_list = [x[1] for x in output]
        deformed_src_matrix = np.array([x[0].flatten() for x in output])

        return deformed_src_matrix,corr_vec_list,reg_time
    
    # PRocess for registration
    def reg_process(self,input_list):
        source = input_list[0]
        template = input_list[1]
        id_ = input_list[2]
        output_list = list()
        
        start = time.time()
        #print ("Starting registration with {} of subject: {} ".format(self.reg_method,id_))        
        deformed_src,corr_vec = self.pair_registration( source, template)
        end = time.time()
        print ("Registration time: {:.4f} ".format(end-start)) 
        if deformed_src is None or corr_vec is None:
            quit() #THIS SHOULD NOT AHPPEN

        output_list.append(deformed_src)
        output_list.append(corr_vec)
                
        return output_list
    
  