# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:44:04 2022

@author: filipavaldeira
"""

# Standard library imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.convert import correspondence_switch
from gpflow.utilities import deepcopy

from registration.registration import ShapesRegistration
from gp_framework.gp_reg import GpModel

from utils.transformations import find_closest_pts

class GpregRegistration(ShapesRegistration): 
    def __init__(self, kernel, corr_method, prob_out, varReg_init, 
                 proba_thresh, flag_addPosterior= True,mean_function = None,flag_save_mat=False, MAXITER = 50,error_tol = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        """ Registration class for gp framework
    
            Parameters
            ----------
                
            corr_method : str 'closest_point' or 'our_proposal'
                type of correspondence method to use in the search for deformations
                
            prob_out : float 0,1
                outlier probability
            
            varReg_init : float
                initial value for finding correspondences. only applicable for 'our_proposal'.
                updated according to varReg_update
            
            var_obs : float
                if using 'closest_point' defines variance of observation
                
            max_dist : float
                if using 'closest_point' defines the maximum distance to search for neighbors
            
            flag_addPosterior : bool (default True)
                if True adds posterior in proba_matrix otherwise it doesn't'
            
            flag_FirstIter : bool
                if True First iteration in GP reg is personalized deformations
            
            def_FirstIter : numpy array 
                deformations for first iteration if flag_FirstIter == True (matrix format)
            
            pts_FirstIter : numpy array 
                template points where we have deformations for first iteration (matrix format)
            
                
            maxiter : int
                maximum iterations allowed for the outer iter
        """  
        
        
        self.reg_method = 'GPReg'
        
        # ---  GPReg Paramterers with default values from original code
        
        # Kernel
        self.kernel = kernel
        self.mean_function = mean_function

        # Correspondence
        self.corr_method = corr_method
        self.prob_out = prob_out
        self.proba_thresh = proba_thresh
        self.flag_addPosterior = flag_addPosterior
        
        # Variance 
        self.varReg_update = 'new_update'
        self.varReg_init = varReg_init
        
        # General
        self.MAXITER = MAXITER
        self.error_tol = error_tol
        self.flag_sv_mat = flag_save_mat
        
        # Parameters string
        #TODO complete this
        self.parameters_str = 'Var_update = {}'.format(self.varReg_update)
        
        self.param_dict = self.get_param_dict()
        self.parameters_df = pd.DataFrame.from_dict(self.param_dict, orient='index') # gets a df with summary of parameters for registration
   
    def get_param_dict(self):
        
        opt = dict()
        opt['Kernel']= self.kernel
        opt['Mean function']= self.mean_function
        
        opt['Corr method']= self.corr_method
        opt['Proba thresh']= self.proba_thresh
        opt['Prob out']= self.prob_out
        opt['Flag add posterior']= self.flag_addPosterior
        
        opt['varReg update'] = self.varReg_update
        opt['varReg init'] = self.varReg_init
        
        opt['MAXITER'] = self.MAXITER
        opt['Error tol'] = self.error_tol
        opt['Flag sv mat'] = self.flag_sv_mat
                
        return opt   
    
    def pair_registration(self,target,source):
        # how to perform registration between a target and a source
        # input : numpy matrices
        # Returns correspondence vector
        # --- corr_vec : vector size of template (source) with ids of target which correspond. 
        #               If source point does not have correspondence with any target, id set to NaN

        
        # Instanciate gp model
        full_kernel = deepcopy(self.kernel)
        mean_func = deepcopy(self.mean_function)
        gp_model = GpModel(source,self.kernel,mean_func)
        
        # Instanciate gp model
        corr_method = self.corr_method
        prob_out = self.prob_out
        MAXITER = self.MAXITER
        proba_t = self.proba_thresh
        varReg_update = self.varReg_update
        varReg_init = self.varReg_init     
        flag_addPosterior = self.flag_addPosterior
        error_tol = self.error_tol
        flag_sv_mat = self.flag_sv_mat
        
        def_templates, final_corr = gp_model.iterative_reg(target, corr_method, prob_out, 
                                               varReg_update = varReg_update, varReg_init = varReg_init,
                                               proba_thresh = proba_t, MAX_ITER = MAXITER,
                                               flag_addPosterior = flag_addPosterior,
                                               flag_SaveMat = flag_sv_mat, flag_compareBCPD=False, error_tol=error_tol)

        
        #TODO CHANGE
        deformed_source = def_templates[-1]    
        missing = gp_model.keep_miss_mask[-1]
        
        dim = target.shape[1]
        # Get correspondences as closest point except for classified as missing
        if dim == 3:
            dist, ids = find_closest_pts(deformed_source, target, max_dist = 10)
            ids = ids.astype(float)
            ids[missing] = np.nan
            ids[dist == np.inf] = np.nan
        # final corr is sum of proba over target
        if dim == 2:
            dist, ids = find_closest_pts(deformed_source, target, max_dist = 1e6)
            ids = ids.astype(float)            
            ids[final_corr<0.5] = np.nan
            ids[missing] = np.nan
            ids[dist == np.inf] = np.nan
        
        corr_vec = ids
        
        deformed_source = deformed_source.reshape(1,-1) # Reshape for registration procedure
        
        self.gp_model = gp_model
        
        return deformed_source, corr_vec



class GpClosestPtRegistration(ShapesRegistration): 
    def __init__(self, kernel,  varObs, max_dist,
                 flag_addPosterior= True,mean_function = None,flag_save_mat=False, MAXITER = 50,error_tol = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        """ Registration class for gp framework
    
            Parameters
            ----------
                
            corr_method : str 'closest_point' or 'our_proposal'
                type of correspondence method to use in the search for deformations
                
            prob_out : float 0,1
                outlier probability
                
            varReg_update : 'constant', 'bcpd_update' or 'gp_posterior'
                only applicable for 'our_proposal'. 
                if 'constant' then varReg_init is always used to get the correspondences
                if 'bcpd_update' then update as per bcpd computation is used
                if 'gp_posterior' then the posterior variance from the gp regression is used
                
            varReg_init : float
                initial value for finding correspondences. only applicable for 'our_proposal'.
                updated according to varReg_update
            
            var_obs : float
                if using 'closest_point' defines variance of observation
                
            max_dist : float
                if using 'closest_point' defines the maximum distance to search for neighbors
            
            flag_addPosterior : bool (default True)
                if True adds posterior in proba_matrix otherwise it doesn't'
            
            flag_FirstIter : bool
                if True First iteration in GP reg is personalized deformations
            
            def_FirstIter : numpy array 
                deformations for first iteration if flag_FirstIter == True (matrix format)
            
            pts_FirstIter : numpy array 
                template points where we have deformations for first iteration (matrix format)
            
                
            maxiter : int
                maximum iterations allowed for the outer iter
        """  
        
        
        self.reg_method = 'GPReg'
        
        # ---  GPReg Paramterers with default values from original code
        
        # Kernel
        self.kernel = kernel
        self.mean_function = mean_function

        # Correspondence
        self.corr_method = 'closest_point'
        self.max_dist = max_dist
        
        # Variance 
        self.varObs = varObs
        
        # General
        self.MAXITER = MAXITER
        self.error_tol = error_tol
        
        # Parameters string
        #TODO complete this
        self.parameters_str = 'Var_update = '
        
        self.param_dict = self.get_param_dict()
        self.parameters_df = pd.DataFrame.from_dict(self.param_dict, orient='index') # gets a df with summary of parameters for registration
   
    def get_param_dict(self):
        
        opt = dict()
        opt['Kernel']= self.kernel
        opt['Mean function']= self.mean_function
        
        opt['Corr method']= self.corr_method
        opt['Max dist'] = self.max_dist
        opt['varObs'] = self.varObs
        
        opt['MAXITER'] = self.MAXITER
        opt['Error tol'] = self.error_tol
                
        return opt   
    
    def pair_registration(self,target,source):
        # how to perform registration between a target and a source
        # input : numpy matrices
        # Returns correspondence vector
        # --- corr_vec : vector size of template (source) with ids of target which correspond. 
        #               If source point does not have correspondence with any target, id set to NaN

        
        # Instanciate gp model
        full_kernel = deepcopy(self.kernel)
        mean_func = deepcopy(self.mean_function)
        gp_model = GpModel(source,full_kernel,mean_func)
        
        # Instanciate gp model
        corr_method = self.corr_method
        max_dist = self.max_dist
        varObs = self.varObs
        MAXITER = self.MAXITER

        error_tol = self.error_tol

        self.gp_model = gp_model
        def_templates, final_corr = gp_model.iterative_reg(target, corr_method, 
                                                           max_dist = max_dist, var_obs = varObs,
                                               MAX_ITER = MAXITER,
                                               error_tol=error_tol)

        
        
        if len(def_templates) < 2:
            print('Registration has failed - shape and corr set to nan') 
            deformed_source = np.empty((source.shape)).reshape(1,-1)
            deformed_source.fill(np.nan)
            corr_vec = np.empty((source.shape[0]))
            corr_vec.fill(np.nan)
        else:    
            deformed_source = def_templates[-1]    
            dim = target.shape[1]
            if dim == 2:
                #missing = gp_model.keep_miss_mask[-1]

                dist, ids = find_closest_pts(deformed_source, target, max_dist = 1e6)
                ids = ids.astype(float)            
                ids[final_corr<0.5] = np.nan
                #ids[missing] = np.nan
                ids[dist == np.inf] = np.nan
                corr_vec = ids
            else:
            
                corr_vec = final_corr
            
            deformed_source = deformed_source.reshape(1,-1) # Reshape for registration procedure

            
        
        
        return deformed_source, corr_vec