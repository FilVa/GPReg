# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:23:22 2022

@author: filipavaldeira
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:46:12 2020

@author: filipavaldeira
"""


# Standard library imports
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils.convert import correspondence_switch
import matlab.engine
import os
from registration.registration import ShapesRegistration

class BcpdRegistration(ShapesRegistration): 
    def __init__(self,omega = 0.1, lmbd=1e4,beta =2.0,gamma = 3.0,cov_tol = 1e-4,
                 min_VB_loops = 30,max_VB_loops = 500,dist = 0.15, scaling = 'e',
                 flag_std_acc=0,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.reg_method = 'BCPD'
        
        # BCPD Paramterers with default values from original code
        self.omega = omega # outlier probability
        self.lmbd = lmbd # expected length of deformation, 1e9 for rigid deformation. Smaller is longer.
        self.beta = beta # parameter of gaussian kernel. It controls the range where deformation vectors are smoothed.
        self.gamma = gamma # randomness of pt matching (positive)
        # Set gamma around 2 to 10 if your target point set is largely rotated. If input shapes are roughly registered, use -g0.1 with an option -ux
        self.dist=dist #Maximum radius to search for neighbors (within matlab bcpd original function)
        
        # convergence BCPD parameters
        self.cov_tol = cov_tol
        self.max_VB_lopps = max_VB_loops
        self.min_VB_loops = min_VB_loops
        
        # acceleration
        self.acc = flag_std_acc # activates acceleration with standard parameters
        
        self.scaling = scaling # e  normalized separately (default)., x both with scale of x, y both with scale of y, n not normalized
        
        self.param_dict = self.get_param_dict()
        self.parameters_df = pd.DataFrame.from_dict(self.param_dict, orient='index') # gets a df with summary of parameters for registration

        self.parameters_str = 'Omega = {}. Lambda = {}'.format(self.omega,self.lmbd)
        
        # If input shapes are roughly registered, use -g0.1 with an option -ux
    def get_param_dict(self):
        
        opt = dict()
        # BCPD parameters
        # Matlab needs these to be floats
        opt['omg']= float(self.omega)
        opt['lmd'] = float(self.lmbd)
        opt['beta'] = float(self.beta)
        opt['gamma'] = float(self.gamma)
        opt['dist'] = float(self.dist)
        # Convergence param
        opt['tol']= float(self.cov_tol)
        opt['max_loops'] = float(self.max_VB_lopps)
        opt['min_loops'] = float(self.min_VB_loops)
        # acceleration
        opt['flag_acc']= float(self.acc)
        opt['scale']= self.scaling
        
        return opt        
        
    
    def pair_registration(self,target,source):
        # how to perform registration between a target and a source
        # input : numpy matrices
        # Returns correspondence vector
        # if registration fails return Nan filled vectors
        
        
        # Runs matlab code BCPD from original authors
        
        opt = self.param_dict 

        # Delete all files in temporary folder - this ensures we can detect 
        # if something goes wrong and new files are not created
        temp_dir = r'C:\Users\filipavaldeira\Documents\shapes_GP_repo\resources\temp_files'
        filelist = [ f for f in os.listdir(temp_dir) if f.endswith(".txt") ]
        for f in filelist:
            os.remove(os.path.join(temp_dir, f))
        
        target_path = r'C:\Users\filipavaldeira\Documents\shapes_GP_repo\resources\temp_files\shape_target.txt'
        np.savetxt(target_path, target,delimiter=',')
        source_path = r'C:\Users\filipavaldeira\Documents\shapes_GP_repo\resources\temp_files\shape_source.txt'
        np.savetxt(source_path, source,delimiter=',')
                   
        input_dict = {'X' : target_path, 'Y' : source_path, 'opt' : opt}
        output_dict = matlab_cpd_reg(input_dict)
        target_corr = output_dict['correp_vec']
        deformed_source = output_dict['deformed_source']
        

        #prob_matrix = None #TODO : matlab version does not compute P for faster computation
        # if registration succeded handle results
        if target_corr is not None:
            # Corr_vec returned for this method is the target one, we need to convert
            target_corr = np.reshape(target_corr,(1,-1))
            target_corr = target_corr[0]
            corr_vec = correspondence_switch(target_corr, source.shape[0])
            deformed_source =  np.reshape(deformed_source, (1,-1))
        else:
            print('Registration has failed - shape and corr set to nan') 
            deformed_source = np.empty((source.shape)).reshape(1,-1)
            deformed_source.fill(np.nan)
            corr_vec = np.empty((source.shape[0]))
            corr_vec.fill(np.nan)
        
        return deformed_source, corr_vec
    
    def read_template(self,template_path):
        df = pd.read_csv(template_path,header=None )
        template = df.to_numpy()
        self.dim = template.shape[1]
        self.n_points = template.shape[0]
        self.template = template
   


def matlab_cpd_reg(input_dict):
    """ Calls matlab function for CPD registration
    
        Parameters
        ----------
        input_dict : dictionary with elements X (target), Y (template) 
        and opt (dictionary with options)

        Returns
        ------
        output_dict : dictionary with elements deformed_source 
        (deformed template from registration) and correp_vec (correspondences).
        Returns None in both elements if registration fails.
        
    """     
    X, Y, opt = input_dict['X'], input_dict['Y'], input_dict['opt']

    eng = matlab.engine.start_matlab()
    #TODO change this path
    eng.addpath(eng.genpath( r'C:\Users\filipavaldeira\Documents\shapes_GP_repo\Other_methods\BCPD'))    
    eng.cd(r'C:\Users\filipavaldeira\Documents\shapes_GP_repo\resources\temp_files')
    transform_Y, correspondence_vec, success = eng.bcpd_register(X,Y,opt, nargout=3)
    eng.exit()

    output_dict = dict()
    # test if registration failed
    if success:
        correspondence_vec = np.asarray(correspondence_vec)-1
        output_dict['deformed_source'] = np.asarray(transform_Y)
        output_dict['correp_vec'] = correspondence_vec    
    else:            
        output_dict['deformed_source'], output_dict['correp_vec'] = None, None
        
    return output_dict