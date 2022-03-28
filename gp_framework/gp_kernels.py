# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:39:40 2020

@author: filipavaldeira
"""

import gpflow
import tensorflow as tf
import numpy as np
from utils.convert import find_complementary_id,get_flat_ids
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

def tf_kron(a,b):
    a_shape = [a.shape[0],a.shape[1]]
    b_shape = [b.shape[0],b.shape[1]]
    return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])*tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),[a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])

def tf_isin(a,b):
    a0 = tf.expand_dims(a, 1)
    b0 = tf.expand_dims(b, 0)
    return tf.reduce_any(tf.equal(a0, b0), 1)


# ------------- KERNELS

# Class to merge together the PCA personal kernel with the standard kernels
# Only produces the sum of kernels
class TotalKernel(gpflow.kernels.Kernel):
    def __init__(self,dim,empirical_kernel=None,kernel_gpflow=None):
        # correspondence vector: size of data with index respective to refence shape. this is just used for optimization with pca kernel. for other purposes use without correspondence.
        super().__init__()            
        self.dim = dim
        
        # Check for EMpirircal kernel
        if empirical_kernel is None:
            self.flag_empirical = False
        else:
            self.k_empirical = empirical_kernel
            self.flag_empirical = True
            self.full_pca_cov = empirical_kernel.cov
            
        # Check for gpflow kernel
        if kernel_gpflow is None:
            self.flag_gpflow = False
        else:
            self.k_gpflow = kernel_gpflow
            self.flag_gpflow = True                         
        
    def K(self, X, X2=None, corr_vec=None): 
        if X2 is None:
            X2 = X
            flag_use_train = True
        else:
            flag_use_train = False

        if self.flag_gpflow:
            X_points = tf.reshape(X,(-1,self.dim))
            X2_points = tf.reshape(X2,(-1,self.dim))
            og_k = self.k_gpflow.K(X_points,X2_points)
            cov_gpflow =tf_kron(og_k, tf.eye(self.dim,dtype=tf.dtypes.double)) 
            
        if self.flag_empirical:
            
            if(corr_vec is not None):   
                print("Using hack for optimization")                 
                if(flag_use_train):
                    filter_X = self.full_pca_cov[corr_vec,:]
                    cov = filter_X[:,corr_vec]

                else: 
                    cov = self.cov_X_X2 # 
                     
            else:
                cov =self.k_empirical.K(X,X2)
            cov_empirical = cov
            
        if self.flag_gpflow & self.flag_empirical:
            full_cov =cov_gpflow+cov_empirical
        elif self.flag_gpflow:
            full_cov = cov_gpflow
        else:
            full_cov = cov_empirical
            

        
        return full_cov
                    
            
    def K_diag(self, X):

        if self.flag_gpflow:
            X_points = tf.reshape(X,(-1,self.dim))
            og_k = self.k_gpflow.K_diag(X_points)
            cov_gpflow = tf.repeat(og_k,self.dim)
        if self.flag_empirical:
            cov_empirical = self.k_empirical.K_diag(X)                                

        if self.flag_gpflow & self.flag_empirical:
            full_cov =cov_gpflow+cov_empirical
        elif self.flag_gpflow:
            full_cov = cov_gpflow
        else:
            full_cov = cov_empirical

        return tf.constant(full_cov, tf.float64)
        


class EmpiricalKernel(gpflow.kernels.Kernel):
    # PCA kernel : can be used as a normal gpflow.kernels.Kernel except for 
    # optimization with sklearn optimizer, in that case please resort to TotalKernel class
    # It takes corresponding points by searching the closest point in the reference shape,
    # when doing registration we are using the reference shape too so it should find the exact point
    
    def __init__(self,ref_shape,shape_list,dim,n_components=None,alpha=1):
        super().__init__()
        self.alpha = alpha
        self.ref_shape = ref_shape
        self.dim = dim
        self.shape_list = shape_list
        if n_components is None:
            self.n_comp = ref_shape.shape[1]
        else : self.n_comp = n_components
        
        self.n_train_samples = len(shape_list)
        self.n_points = ref_shape.shape[0]
        
        self.mean_def,self.train_mat = self.get_train_mat()
        self.cov = self.get_covariance()
        
        self.ref_tree = KDTree(ref_shape)
        self.mean_function = Mean_pca(ref_shape, dim, self.mean_def)       
        
        self.train_mat,self.shape_list = None, None
        
    def get_train_mat(self):
        
        train_mat = (self.shape_list - self.ref_shape).reshape(self.n_train_samples,-1)
        mean_def = train_mat.mean(axis=0)
        return mean_def,train_mat

    def get_covariance(self):

        self.pca = PCA(n_components=self.n_comp)
        pca_result = self.pca.fit_transform(self.train_mat)
        cov = self.pca.get_covariance()
        return cov

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
            
        # Turn into points
        X_points = tf.reshape(X,(-1,self.dim))
        X2_points = tf.reshape(X2,(-1,self.dim))
        
        # Find closest ids
        X_dist,X_ids = self.ref_tree.query(X_points)
        if(X_ids.shape[0]==0):
            # No points (usually because we are sampling from posterior). choose first point of ref shape
            X_ids = tf.where(tf_isin(self.ref_shape,self.ref_shape[0,:])[:,0])

        X2_dist,X2_ids = self.ref_tree.query(X2_points)
        if(X2_ids.shape[0]==0):
            X2_ids = tf.where(tf_isin(self.ref_shape,self.ref_shape[0,:])[:,0])
 
        # Ids for covariance matrix
        X_ids_flat = get_flat_ids(X_ids,self.dim,self.n_points)
        X2_ids_flat = get_flat_ids(X2_ids,self.dim,self.n_points)
        filter_X = self.cov[X_ids_flat,:]
        filter_X2 = filter_X[:,X2_ids_flat]
        
        # add jitter to avoid numerical problems with cholesky decompostision
        jitter = 1e-9

        final_mat = self.alpha*filter_X2
        cov_mat_tf = tf.constant(final_mat, tf.float64)  

        return cov_mat_tf
        

    def K_diag(self, X):
        
        X_points = np.reshape(X,(-1,self.dim))
        X_dist,X_ids = self.ref_tree.query(X_points)
        X_ids_flat = get_flat_ids(X_ids,self.dim,self.n_points)
        diag = self.cov.diagonal()
        
        cov_mat_tf = tf.constant(self.alpha*diag[X_ids_flat], tf.float64)

        return cov_mat_tf 

    
class Mean_pca(gpflow.mean_functions.MeanFunction):


    def __init__(self,ref_shape,dim,mean_def):

        gpflow.mean_functions.MeanFunction.__init__(self)
        self.ref_tree = KDTree(ref_shape.copy())
        self.u_mean = mean_def.reshape(-1,1).copy()
        self.dim  = dim
        self.n_points = ref_shape.shape[0]

    def __call__(self, X,corr_vec=None):
        
        if(corr_vec is not None): 
            print("Mean function with optimization hack")
            mean_vec = self.u_mean[corr_vec,:]
            
        else:
            X_points = np.reshape(X,(-1,self.dim))
            X_dist,X_ids = self.ref_tree.query(X_points)
            X_ids_flat = get_flat_ids(X_ids,self.dim,self.n_points)
            mean_vec = self.u_mean[X_ids_flat,:]
        
        return tf.constant(mean_vec.reshape(-1,1), tf.float64)    