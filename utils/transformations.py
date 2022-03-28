# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:52:36 2019

@author: filipavaldeira
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import math
from scipy.linalg import orthogonal_procrustes
from utils.convert import numpy2PointCloud,mat2flat


def get_ids_in_area(shape,center,width):
    dim = shape.shape[1]
    miss_area_pts = bounding_box_from_center(center,width,dim)  
    
    # cREATE BBOX
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(miss_area_pts)
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox_missing_area = bbox.create_from_points(pcd.points)
    
    # geT IDS
    if dim==3:
        o3d_shape = o3d.utility.Vector3dVector(shape)
    elif dim==2:
        # insert coordinatez as 0 in each point to use 3d tools
        shape_3d = np.zeros((shape.shape[0],3))
        shape_3d[:,0:2] = shape
        o3d_shape = o3d.utility.Vector3dVector(shape_3d)
        
        
    ids = bbox_missing_area.get_point_indices_within_bounding_box(o3d_shape)
    
    return ids

def find_closest_pts(ref,target,max_dist):
    #for each point of template we get the closest point of target
    # returns the distances and the indices of target points
    tree = KDTree(target)
    neighbor_dists, neighbor_indices = tree.query(ref,distance_upper_bound=max_dist)
    return neighbor_dists, neighbor_indices


def centroid_size(X):
    mean_dim = np.mean(X,axis=0)
    s = np.mean(np.sqrt(np.sum(np.square(X - mean_dim),axis=1)))  
    return s

# Rotate P into Q
# Return rotated P 'solution' and rotation matrix 'R'
def OPA(Q,P) :
      res =orthogonal_procrustes(P,Q);
      R = res[0]
      solution = np.matmul(P, R);
      return solution, R


# Cost function of GPA
def getG(Xp,m,k,n):
    
    Xps = np.reshape(Xp,(m*k,n));
    G = 0;
    for i in range(n) :
        rept = np.tile(Xps[:,i],(n-i-1,1));
        others = Xps[:,i+1:n];
        diff = rept - np.transpose(others);
        squared = np.square(diff);
        G = G+ np.sum(squared);
    G = G/n;   
    return G
    
# Generalized Procrustes Analysis         
# Receives k*m*n matrix 'shapes' : k is number of landmarks, m is dimension, n is number of samples
# Applies GPA with scaling if scale_flag =1
def GPA(shapes, scale_flag, thresh_tol):
    print('Start Generalized Procrustes Analysis')
    [n_points,dim,n_samples] = shapes.shape
    
    transformations = np.zeros((dim+1,dim+1,n_samples))
    X_bar= np.zeros((n_points,dim))
    X_p = np.zeros((n_points,dim,n_samples))
    
    # Translations
    print('Translation step')
    C= np.identity(n_points)-(1/n_points)*(np.ones((n_points,1))*np.ones((1,n_points))) 
    for i in range(n_samples):
        X_p[:,:,i] = np.matmul(C,shapes[:,:,i])
        diff =  X_p[:,:,i] - shapes[:,:,i]
        transformations[0:3,3,i] = diff[0,:]
                    
    #check_matrix = np.zeros((n_points,dim+1,n_samples));
    for i in range(n_samples):
        transformations[0:3,0:3,i] = np.identity(dim)
        transformations[3,3,i] = 1
        #check_matrix[:,:,i] = np.matmul(np.append(shapes[:,:,i],np.ones((n_points,1)),axis=1), np.transpose(transformations[:,:,i]))
    
    # Initialize tolerance
    tol_old = 1e16
    tol_new = 1e15
    while (abs(tol_old - tol_new) > thresh_tol):
        
        # Rotation cycle
        print('Rotation cycle')
        while (abs(tol_old - tol_new) > thresh_tol):            
            tol_old = tol_new;            
            for i in range(n_samples):                
                # exclude sample from mean
                before_sample = X_p[:,:,0:i]
                after_samples = X_p[:,:,i+1:n_samples]
                join =  np.zeros((n_points,dim,n_samples-1))
                join[:,:,0:i] = before_sample
                join[:,:,i:n_samples] = after_samples
                
                X_bar = np.mean(join,2) # Mean  shape
                X_p[:,:,i],rotation = OPA(X_bar, X_p[:,:,i])
                # Save rotation matrix
                transformations[0:3,0:3,i] = np.matmul(transformations[0:3,0:3,i],np.transpose(rotation)) 
                # Update translation
                transl_rot= np.matmul(np.transpose(transformations[0:3,3,i]),rotation)
                transformations[0:3,3,i] = np.transpose(transl_rot)                
           
            tol_new = getG(X_p,dim,n_points,n_samples);
        
        # Scaling
        print('Scaling')
        if(scale_flag ==1) : 
            beta = np.zeros((n_samples))
            vecXp = np.reshape(X_p,(n_points*dim, n_samples)) # Columns are observations, each row is a coordinate
            Phi = np.corrcoef(vecXp, rowvar=False);
            [eigval,eigvec] = np.linalg.eig(Phi)
            sorted_id = np.argsort(eigval)
            largest_eigvec = eigvec[:,sorted_id[-1]] # eigenvector of largest eigenvalue of corr matrix
            
            sumOfSquaredNorms = 0
            for i in range(n_samples):
                sumOfSquaredNorms = sumOfSquaredNorms + math.pow(np.linalg.norm(X_p[:,:,i]),2);
            for i in range(n_samples):
                norm_sqr = math.pow(np.linalg.norm(X_p[:,:,i]),2)
                beta[i] = math.sqrt(sumOfSquaredNorms/norm_sqr)*abs(largest_eigvec[i]);
                
                # Update transformation matrix
                transformations[:,:,i] =  transformations[:,:,i]*beta[i]
                X_p[:,:,i] = beta[i] * X_p[:,:,i];
            tol_new = getG(X_p,dim,n_points,n_samples)
        
        print(abs(tol_old - tol_new))
    
    # Correct transformation matrix
    for i in range(n_samples):
        transformations[3,3,i] = 1
    X_bar = np.mean(X_p,2) # Final mean shape
    
    return transformations, X_bar


def GPA_shape(shape_list,scale,dim,threshold):
    
    eps=1e-8
    mean_dist=1e8
    n_landmarks = shape_list[0].shape[0]
    n_shapes = len(shape_list)

    # Reshape for approprite order in GPA function
    matrix_reshaped = np.zeros((n_landmarks,dim,n_shapes))
    for id_,shape in enumerate(shape_list):
        matrix_reshaped[:,:,id_] = shape
    
    transformations, mean_shape = GPA(matrix_reshaped,scale,threshold)
    
    transf_list_shape = list()
    for shape in shape_list:
        procrustes_res = procrustes(mean_shape, shape,scaling=scale,reflection=False)
        d,Z,tform = procrustes_res
        transf_shape = apply_transformations(tform,shape)
        transf_list_shape.append(transf_shape)
        full_mat_reg = mat2flat(transf_list_shape)

    return full_mat_reg, mean_shape

def subsample(data,id_vec, param=0.1):
    
    downsampled_data = list()

    if(len(id_vec)==1):
        point_cloud = numpy2PointCloud(data)
        down_pcd = point_cloud.voxel_down_sample(voxel_size=param)
        down_shape = np.asarray(down_pcd.points)
        return down_shape
    else:
        for n_sample in range(len(id_vec)):
        
            if(isinstance(data, list)):
                shape = data[n_sample]
            else:
                shape = data[:,:,n_sample]
        
            point_cloud = numpy2PointCloud(shape)
            #print('Before down sampling')
            #print(point_cloud)
            down_pcd = point_cloud.voxel_down_sample(voxel_size=param)

            down_shape = np.asarray(down_pcd.points)
            print(down_shape.shape)        
            downsampled_data.append(down_shape)
    
    return downsampled_data


def subsample_dataset_same_ids(data,ref_shape, radius=0.1):
    
    sub_shape = subsample(ref_shape,list([0]), param=radius)
    ref_tree = KDTree(ref_shape)
    dist,ids = ref_tree.query(sub_shape)
    
    list_shapes = list()
    
    for shape in data:
        list_shapes.append(shape[ids,:])
        
    return sub_shape,ids,list_shapes
    
    




def bounding_box_from_center(center,width,dim):    


    # center is 3d/2d vector
    # width is 3d/2d vector with width for each axis 
    if(dim==3):
        vertices = np.zeros((8,3))
        
        vertices[0,:] = center + width
        vertices[1,:] = center + np.array([width[0],width[1],-width[2]])
        vertices[2,:] = center + np.array([width[0],-width[1],width[2]])
        vertices[3,:] = center + np.array([-width[0],width[1],width[2]])
        vertices[4,:] = center + np.array([-width[0],-width[1],width[2]])
        vertices[5,:] = center + np.array([-width[0],width[1],-width[2]])
        vertices[6,:] = center + np.array([width[0],-width[1],-width[2]])
        vertices[7,:] = center + np.array([-width[0],-width[1],-width[2]])
        
    elif(dim==2):
        # coordinate z is left as 0 to work with 3d tools
        vertices = np.zeros((4,3))
       
        vertices[0,0:2] = center + width
        vertices[1,0:2] = center + np.array([-width[0],-width[1]])
        vertices[2,0:2] = center + np.array([width[0],-width[1]])
        vertices[3,0:2] = center + np.array([-width[0],width[1]])
        
        
    return vertices
    
def procrustes(X, Y, scaling=True, reflection='best'):
       """
       A port of MATLAB's `procrustes` function to Numpy.

       Procrustes analysis determines a linear transformation (translation,
       reflection, orthogonal rotation and scaling) of the points in Y to best
       conform them to the points in matrix X, using the sum of squared errors
       as the goodness of fit criterion.

           d, Z, [tform] = procrustes(X, Y)

       Inputs:
       ------------
       X, Y    
           matrices of target and input coordinates. they must have equal
           numbers of  points (rows), but Y may have fewer dimensions
           (columns) than X.

       scaling 
           if False, the scaling component of the transformation is forced
           to 1

       reflection
           if 'best' (default), the transformation solution may or may not
           include a reflection component, depending on which fits the data
           best. setting reflection to True or False forces a solution with
           reflection or no reflection respectively.

       Outputs
       ------------
       d       
           the residual sum of squared errors, normalized according to a
           measure of the scale of X, ((X - X.mean(0))**2).sum()

       Z
           the matrix of transformed Y-values

       tform   
           a dict specifying the rotation, translation and scaling that
           maps X --> Y

       """

       n,m = X.shape
       ny,my = Y.shape

       muX = X.mean(0)
       muY = Y.mean(0)

       X0 = X - muX
       Y0 = Y - muY

       ssX = (X0**2.).sum()
       ssY = (Y0**2.).sum()

       # centred Frobenius norm
       normX = np.sqrt(ssX)
       normY = np.sqrt(ssY)

       # scale to equal (unit) norm
       X0 /= normX
       Y0 /= normY

       if my < m:
           Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

       # optimum rotation matrix of Y
       A = np.dot(X0.T, Y0)
       U,s,Vt = np.linalg.svd(A,full_matrices=False)
       V = Vt.T
       T = np.dot(V, U.T)

       if reflection is not 'best':

           # does the current solution use a reflection?
           have_reflection = np.linalg.det(T) < 0

           # if that's not what was specified, force another reflection
           if reflection != have_reflection:
               V[:,-1] *= -1
               s[-1] *= -1
               T = np.dot(V, U.T)

       traceTA = s.sum()

       if scaling:

           # optimum scaling of Y
           b = traceTA * normX / normY

           # standarised distance between X and b*Y*T + c
           d = 1 - traceTA**2

           # transformed coords
           Z = normX*traceTA*np.dot(Y0, T) + muX

       else:
           b = 1
           d = 1 + ssY/ssX - 2 * traceTA * normY / normX
           Z = normY*np.dot(Y0, T) + muX

       # transformation matrix
       if my < m:
           T = T[:my,:]
       c = muX - b*np.dot(muY, T)

       #transformation values 
       tform = {'rotation':T, 'scale':b, 'translation':c}

       return d, Z, tform

def apply_transformations(tform,shape):
    
      # tform(as the output from procrustes function)
      #      a dict specifying the rotation, translation and scaling that
      #      maps X --> Y
      R = tform['rotation']
      s = tform['scale']
      t = tform['translation']
      
      trans_shape = s*np.matmul(shape,R)+t.reshape(-1,1).T
      return trans_shape
