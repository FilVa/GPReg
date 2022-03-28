# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 09:46:36 2021

@author: filipavaldeira
"""
import numpy as np
import gpflow
import time
import open3d as o3d
import os
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import sys
from utils.transformations import find_closest_pts
from utils.convert import mat2flat
from utils.io import write_excel,excel_get_last_row, read_mesh
from metrics.general import mean_dist_organized
from utils.auxiliar_bcpd import compute_init_var,Nystrom_approx, normalize_shape, get_matrices,get_step_1_vars, get_step_2_vars, update_var, likelihood_CPD
from utils.plot_shapes import plot_deformations
from scipy.sparse import csr_matrix


from gpflow.utilities import print_summary, set_trainable

import seaborn as sns
from gp_framework.gpr_reimplementation import GPR_multi_noise,GPR_multiDim_finalShape
from gp_framework.gp_kernels import TotalKernel

# Auxiliary functions
def check_inputs(corr_method,var_obs,max_dist,varReg_update,varReg_init, proba_thresh):
    if corr_method == 'closest_point':
        if (var_obs is None)|(max_dist is None):
            print('If using corr_method = closest_point please define max_dist and var_obs')
            sys.exit()       
        if (varReg_update is not None)|(varReg_init is not None)|(proba_thresh is not None):
            print('varReg_update or varReg_init are defined but are not being used in closes_point method. Please change this.')
            sys.exit()       
    if corr_method == 'our_proposal':
        if (varReg_update is None)|(varReg_init is None)|(proba_thresh is None):
            print('If using corr_method = our_proposal please define varReg_update, varReg_init and proba_thresh')
            sys.exit()       
        if (var_obs is not None)|(max_dist is not None):
            print('var_obs or max_dist are defined but are not being used in closes_point method. Please change this.')
            sys.exit()    
        if (varReg_update != 'constant')&(varReg_update != 'bcpd_update')&(varReg_update != 'gp_posterior')&(varReg_update != 'new_update'):
            print('varReg_update not valid : please chose  constant, bcpd_update or gp_posterior')
            sys.exit()  

##############################################################################
# Different methods to get observations
##############################################################################

# Closest point
def get_standard_observations(ref_shape, target_shape, max_dist, original_ref_shape):
    # we get deformations with respect to the original reference shape    
    dist,ids = find_closest_pts(ref_shape,target_shape, max_dist)
    #dist is inf is morethan max_dist
    
    filtered_dist = dist[~np.isinf(dist)]
    filtered_ids_target = ids[~np.isinf(dist)]
    
    ref_shape_observations = original_ref_shape[~np.isinf(dist),:]
    target_shape_observation = target_shape[filtered_ids_target]
    deformations = target_shape_observation-ref_shape_observations
    
    ids = ids.astype(float)
    ids[np.isinf(dist)] = np.nan
    
    return ref_shape_observations, deformations,ids







# Our method
def get_observations_our_method(ref_shape, target_shape, og_ref_shape,
                                var, prob_out, flag_IterOgTemplate, posterior_var_gp,
                                flag_SaveMat, flag_old_var_eq, proba_threshold,flag_addPosterior):
    """ Given reference and target computes distance matrices

        Parameters
        ----------
        ref_shape : numpy array M*D
            reference shape
            
        target_shape : Numpy array N*D
            target shape

        og_ref_shape : numpy array M*D
            original referece shape where to compute the deformations from
        
        var : float
            noise / sigma^2
        
        prob_out : float 
            outlier probability
            
        posterior_var_gp : numpy array
            equivalent to sigma_m in registration
        
        flag_SaveMat : Bool
            if True returns all intermediate matrices use only for shapes with low number of points

        Returns
        ------
        
        ref_shape_obs : numpy array P*D
            points where to apply the deformations
        
        final_def : numpy array P*D
           deformations to be applied at ref_shape_obs
           
        final_var : numpy array P
            variance of each observation
            
        vi : numpy array M*1
            sum of P over rows = P*ones (size M)
            
        ss_num : -
            P @ S**2
        
        pt : numpy array M*dim
            P@ t
        
        proba_matrix : numpy array M*N
            final probabilities matrix
            
        dist_mat : numpy array M*N
            distances between each ppoint in template and target
        
        phi_mat : 
            bcpd phi mat
            
        phi_mat_posterior : 
            bcpd phi posterior mat
        
        likelihood_term : float
            bcpd likelihood
    """  


    
    # Init
    M, D = ref_shape.shape
    N =  target_shape.shape[0]
    x, y = target_shape.reshape(-1,1), ref_shape.reshape(-1,1)
    
    logging.info("-> Computing proba matrices")
    proba_matrix, dist_mat, phi_mat, phi_mat_post, likelihood_term = get_matrices(ref_shape, target_shape,var, prob_out,
                                                                                  posterior_var_gp, flag_SaveMat,
                                                                                  flag_addPosterior=flag_addPosterior)
    logging.info("-> Done computing proba matrices")
    
    # compute terms for variance update 
    v_i = proba_matrix.sum(axis=1)
    ss_num = np.sum(proba_matrix @ (target_shape**2),axis=1)
    pt = proba_matrix @ target_shape
    
    #with Threshold proba
    full_mask = proba_matrix>proba_threshold
    proba_matrix_masked = np.ma.array(proba_matrix,mask=~full_mask,fill_value=0).filled() # values below threshold are replaced by zeros
    
    # compute distance and proba matrices
    if flag_SaveMat:        
    
        # get delta hat and sigma_hat
        if flag_old_var_eq:
            noise_vec_with_mask = 1/(np.matmul(proba_matrix_masked**2,np.ones(N))*np.pi*2)
            a= np.matmul(proba_matrix_masked**2,np.ones(N))        
            b = np.matmul(np.diag(np.reciprocal(a,out=np.zeros_like(a),where=a!=0)),proba_matrix_masked**2)
            x_hat_with_mask = np.matmul(b,x.reshape(-1,D)).reshape(-1,1)
        else:
            # Noise from multilabel = var/porb sum in j
            noise_vec_with_mask = var/np.matmul(proba_matrix_masked,np.ones(N)) # size 1 * M
            
            # Deformations from multilabel
            a= np.matmul(proba_matrix_masked,np.ones(N))
            b = np.matmul(np.diag(np.reciprocal(a,out=np.zeros_like(a),where=a!=0)),proba_matrix_masked)
            x_hat_with_mask = np.matmul(b,x.reshape(-1,D)).reshape(-1,1)
            
        
        
    else:
        # COMPUTATION FOR LARGE MATRICS
        if flag_old_var_eq:
            print('Option does not exist')

        else:
            proba_matrix_masked = csr_matrix(proba_matrix_masked)
            vi_masked = proba_matrix_masked @ np.ones(N)
            
            # Noise from multilabel = var/porb sum in j
            noise_vec_with_mask = var/vi_masked # size 1 * M
            
            # Deformations from multilabel
            v_inv = np.diag(np.reciprocal(vi_masked,out=np.zeros_like(vi_masked),where=vi_masked!=0))
            b = v_inv @ proba_matrix_masked
            x_hat_with_mask = np.matmul(b,x.reshape(-1,D)).reshape(-1,1)
            
            del v_inv, b
    
    def_with_mask = x_hat_with_mask - y
    
    # Remove missing points
    miss_mask = full_mask.sum(axis=1)<1
    final_corr = proba_matrix.sum(axis=1)
    
    final_var = noise_vec_with_mask[~miss_mask]
    final_def = def_with_mask.reshape(-1,D)[~miss_mask,:]   
    
    # Depending on whether we want to take deformations with respect to original template or current
    if flag_IterOgTemplate:
        final_def = ref_shape[~miss_mask,:] + final_def - og_ref_shape[~miss_mask,:]  
        ref_shape_obs = og_ref_shape[~miss_mask,:]
    else:
        ref_shape_obs = ref_shape[~miss_mask,:]
    
    # Return different for bigdata
    if flag_SaveMat:
        sv_matrices = [proba_matrix,dist_mat, phi_mat,phi_mat_post]
        return ref_shape_obs, final_def, final_var,noise_vec_with_mask,miss_mask,v_i,ss_num, pt, likelihood_term, sv_matrices, final_corr
    else:
        return ref_shape_obs, final_def, final_var,noise_vec_with_mask, miss_mask,v_i,ss_num, pt, None,None, final_corr





def sigma_delta_paper(sigma,deformations):
    # vector with size =  number of observed deformations *1
    # dformations = number of observed deformations * dim
    
    sigma_hat_sqrd = 1/np.sum(1/(sigma**2))    # scalar

    sigma_sqrd= sigma**2
    delta_hat = sigma_hat_sqrd*np.sum(deformations/sigma_sqrd[:,None],axis=0)

    return sigma_hat_sqrd,delta_hat
    



def comp_bcpd(x,y,proba_matrix,dim,I_D,Q_gg,D_gg,var_reg):
    v_bcpd,v_line_bcpd,N_hat,x_hat = get_step_1_vars(proba_matrix,x,dim)            # we do not want to compute twice
    Sigma,v_hat ,likelihood_gp = get_step_2_vars(Q_gg,D_gg,var_reg,v_bcpd,1,x_hat,I_D,y,flag_return_like=True) #scale=1, lambda = 1

    var_post = np.diag(Sigma)
    y_hat = y+v_hat

    var_for_sigma_bar = np.diag(var_post)
    var_update = update_var(x,v_line_bcpd,proba_matrix,I_D,y_hat,v_bcpd,var_for_sigma_bar,N_hat,dim)
    return Sigma, v_hat, var_update


def get_Ginv(D,M,Q):
    I_M = np.eye(M)
    return I_M-( Q @ np.linalg.inv(np.diag(np.reciprocal(D)) + Q.T @ Q) @Q.T)


class GpModel(object):
    def __init__(self,ref_shape_path,kernel,mean_function=None,ref_shape_landmarks_ids=None,ref_curvature_path=None):
        if(isinstance(ref_shape_path, str)):
            logging.info("GP Model : Received path for template mesh ")
            self.ref_shape,self.ref_shape_faces = read_mesh(ref_shape_path,remove_degenrate_flag=False)
            if self.ref_shape.shape[0] > 8000:
                print('\n ATTENTION: Reference shape has more than 8000 points. This will cause problems for GP regression. Will not continue. ')
                sys.exit()
            
            self.ref_shape_path = ref_shape_path
        else:
            logging.info("GP Model : Received numpy array for template mesh ")
            self.ref_shape = ref_shape_path
            
        self.kernel = kernel # kernel to use in the registration
        self.mean_function = mean_function # mean function to use in the registration
        # check if we are using our class of kernel that takes care of dimensions, etc. Otherwise we need to exceute different functions
        if type(kernel) == TotalKernel:
            self.flagMyKernel = True
        else: 
            self.flagMyKernel = False
            print('Attention : did not receive our class TotalKernel, therefore there is not dependence across different dimensions')
        
        # landmarks
        if ref_curvature_path is not None:
            self.ref_shape_curv = np.loadtxt(ref_curvature_path)        
        self.ref_shape_landmarks = ref_shape_landmarks_ids

    def iterative_reg(self, target_shape, corr_method, prob_out=None, 
                      flag_old_var_eq = False,
                      K_nystrom = None, flag_normalize = False,
                      varReg_update = None, varReg_init = None,
                      flag_addPosterior = True,
                      max_dist = None, var_obs = None, proba_thresh= None,                   
                      flag_FirstIter = False, def_FirstIter=None, pts_FirstIter = None,
                      flag_OptModel=False, flag_SaveMat = False,
                      flag_IterOgTemplate = True, MAX_ITER=500,
                      flag_compareBCPD = False, error_tol = 1e-4 ):

        """ Iterative registration with GP framework
    
            Parameters
            ----------
            target_shape : numpy array N*dim
                target array to be registered
                
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
            
            flag_IterOgTemplate : bool
                apply the iterations to original template or the one from previous iteration
                
            flag_SaveMat : bool
                if True save intermediate matrices for posterior analysis
                
            flag_OptModel : bool 
                if True includes optimization of kernel
                
            flag_compareBCPD :bool
                if True computes necessary steps to compare with bcpd evolution
                
            maxiter : int
                maximum iterations allowed for the outer iter
    
            Returns
            ------
            
            def_shapes : list of deformed templates over iterations
            
            
        """  
        ######################################################################
        # Check inputs validitity  
        check_inputs(corr_method,var_obs,max_dist,varReg_update,varReg_init,proba_thresh)
        print('Inputs validated')
        # Get flags and parameters
        K = self.kernel
        error_tolerance = error_tol
            
        N = target_shape.shape[0]
        M = self.ref_shape.shape[0]
        dim = self.ref_shape.shape[1]
        
        ref_shape = self.ref_shape.copy()
        target_shape = target_shape.copy()
        
        if N > 100 or M > 100:
            if flag_SaveMat:
                print('Large matrices - please set do not set flag_SaveMat to True')
                quit()
            if flag_compareBCPD:
                print('Large matrices - please set do not set flag_compareBCPD to True')
                quit()
        
        # For bcpd comparison
        if varReg_update == 'bcpd_update':
            flag_compareBCPD = True
            flag_SaveMat = True
       
        if flag_SaveMat:
            print('flag_SaveMat requires flag_compareBCPD - will be activated')
            flag_compareBCPD = True
        
        if flag_compareBCPD:
            x, y = target_shape.reshape(-1,1), ref_shape.reshape(-1,1) # For bcpd
            I_D = np.eye(dim)
            var_bcpd = varReg_init
            G = self.kernel.k_gpflow.K(ref_shape).numpy() # we just take gpflow kernel
            if(K_nystrom is None):
                K_nystrom = ref_shape.shape[0]
            D_gg, Q_gg = Nystrom_approx(G,K_nystrom)    
            G_inv = get_Ginv(D_gg,M,Q_gg)

        # Scaling
        self.scale_x = 1
        
        # Initial settings for cycle
        flag_leave = False
        n_iter = 0
        current_ref_shape = ref_shape.copy()
        ref_shape_flat = mat2flat(ref_shape)
        posterior_var_gp = np.zeros(M)
        var_reg = varReg_init
        
        # ----- List variables to save results
        self.shapes_list = list()
        self.mean_err_to_prev, self.max_err_to_prev = list(), list() # Store error with respect to previous shape,
        self.iter_time, self.n_def, self.current_likelihood = list(), list(), list()
        self.var_obs = var_obs       
        self.keep_miss_mask = list()
        
        self.like_posterior = list()
        
        
        if flag_SaveMat:
            # Likelihood
            self.likelihood_t1, self.likelihood_t2, self.likelihood_t3, self.likelihood_real = list(), list(), list(), list()
            # PRoba matrices
            self.keep_proba,self.keep_dist_mat, self.keep_phi_mat, self.keep_phi_mat_post = list(),list(),list(),list()        
            # New variance terms
            self.var_term_vpost, self.var_term_st, self.var_term_tt,self.var_term_ss, self.var_term_sum = list(), list(), list(), list(), list()
            # store comparison between deformations from bcpd and variance
            self.def_close_bcpd, self.var_close_bcpd = list(),list() 

            self.keep_points, self.keep_def, self.keep_noise_vec, self.keep_noise_vec_full= list(), list(), list() , list()# Store points, deformations and obs noise for each step o GPR
            self.keep_var = list() # Store variance for registration
            self.keep_post_var = list()

        og_proba_thresh = proba_thresh
        ######################################################################
        # Iterations for GP registration
        logging.info(" --- Start Iterative Registration in GP --- ")
        ######################################################################
        
        while(flag_leave==False):
            logging.info('-----------------------------------------')
            logging.info(' -------------------> Iteration number : {}'.format(n_iter))
            start = time.time()
            
            ######################################################################
            #STEP 1 : Get poitns and respective deformations for GP regression
            ######################################################################
            if((flag_FirstIter == True)&(n_iter==0)):
                logging.info(" First iteration in GP reg is personalized deformations")
                points, deformations = pts_FirstIter, def_FirstIter
            else:
                if(corr_method=='closest_point'):
                    logging.info(" Step 1 --- Get deformations --- with closest point")
                    points,deformations, final_corr = get_standard_observations(current_ref_shape,target_shape,max_dist,ref_shape)
                    likelihood_term_2 = 0
                elif(corr_method=='our_proposal'):
                    logging.info(" Step 1 --- Get deformations --- with probablistic")
                    points, deformations, noise_vec, noise_vec_with_mask, miss_mask,v_i , ss_num, pt ,likelihood_term_2, sv_mats, final_corr =get_observations_our_method(current_ref_shape,target_shape,ref_shape,var_reg,prob_out,flag_IterOgTemplate,posterior_var_gp,flag_SaveMat,flag_old_var_eq,proba_thresh,flag_addPosterior)
                    self.keep_miss_mask.append(miss_mask)

                    if flag_SaveMat:
                        proba_matrix, dist_mat, phi_mat, phi_mat_post = sv_mats[0], sv_mats[1], sv_mats[2], sv_mats[3]
                        self.keep_proba.append(proba_matrix), self.keep_dist_mat.append(dist_mat)
                        self.keep_phi_mat.append(phi_mat), self.keep_phi_mat_post.append(phi_mat_post)
                        logging.info('Proba for this iter: {},{}'.format(proba_matrix[0,0],proba_matrix[1,1]))
            
            n_def = points.shape[0]   
            if n_def==0:
                print("ERROR : NO DEFORMATIONS OBTAINED. FINISH COMPUTATION. ITER : {} ".format(n_iter))
                return self.shapes_list.copy(), final_corr
            
            logging.info('We have found {} deformations '.format(n_def))
            
            ######################################################################
            # STEP 2 :  Compute GP regression
            ######################################################################
            logging.info(" Step 2 --- Compute Regression --- ")
            
            # Create regression model with observations
            X, Y = points.copy(), deformations.copy() 
            X_flat, Y_flat = mat2flat(X), mat2flat(Y)
            
            if self.flagMyKernel:
                
                #  Create model
                if(corr_method=='closest_point'):
                    model = gpflow.models.GPR(data=(X_flat.T, Y_flat.T), kernel=K, mean_function=self.mean_function,noise_variance= var_obs)
                elif(corr_method=='our_proposal'):
                    
                    model = GPR_multi_noise(dim,data=(X_flat.T, Y_flat.T), kernel=K, mean_function=self.mean_function,noise_variance= noise_vec)
    
                # Predict posterior            
                if flag_IterOgTemplate:
                    mean, var = model.predict_f(ref_shape_flat.T,full_cov=False) 
                    
                    var_post = var.numpy().reshape(-1,dim)[:,0]
                    mean = mean.numpy()
                    deformed_template = ref_shape + np.reshape(mean,(-1,dim))
                else:
                    mean, var = model.predict_f(mat2flat(current_ref_shape).T)           
                    deformed_template = current_ref_shape + np.reshape(mean,(-1,dim))
                    var_post = var.numpy().reshape(-1,dim)[:,0]
            else:
                # normal kernel we need to execute each dimension in separate
                deformed_template, mean, var_post = GPR_multiDim_finalShape(X,Y,dim,kernel=K,var_observations=noise_vec,ref_shape=ref_shape)
                var_post = var_post[:,0]
            
            logging.info('Regression completed : Time until the end of regression {}'.format(time.time()-start))

            #------- BCPD 
            #TODO if bcpd compare
            if flag_compareBCPD:
                Sigma,v_hat,var_bcpd_update = comp_bcpd(x,y,proba_matrix,dim,I_D,Q_gg,D_gg,var_reg)
                #Comparison with bcpd
                self.def_close_bcpd.append(np.allclose(v_hat,mean,rtol=1e-5,atol=1e-5))
                self.var_close_bcpd.append( np.allclose(var_post,np.diag(Sigma),rtol=1e-5,atol=1e-5)  ) 
                



            ######################################################################
            # STEP 3:  Update model parameters
            ######################################################################

            
            
            # update variance
            if varReg_update == 'constant': 
                var_reg = varReg_init
            elif varReg_update == 'bcpd_update':
                var_reg = var_bcpd_update[0]      
                self.keep_var.append(var_reg)
            elif varReg_update=='gp_posterior':
                var_reg = var_post
            elif varReg_update=='new_update':
                ss_term = ss_num/v_i
                temp2 = pt.reshape(1,-1)* deformed_template.reshape(1,-1)
                st_term=-2*temp2.reshape(-1,dim).sum(axis=1)/v_i
                tt_term = np.linalg.norm(deformed_template,axis=1)**2
                var_reg = ((var_post*dim)+st_term+tt_term+ss_term)/dim
                
                if flag_SaveMat:
                    self.var_term_vpost.append(var_post)
                    self.var_term_st.append(st_term/dim)
                    self.var_term_tt.append(tt_term/dim)
                    self.var_term_ss.append(ss_term/dim)
                    self.var_term_sum.append((st_term+tt_term+ss_term)/dim)                

                
            posterior_var_gp = var_post # posterior variance from GP
            
            
            ######################################################################
            # STEP 4:  Stopping criteria
            ######################################################################
            logging.info(" Step 3 --- Evaluate Error --- ")
            errors = np.linalg.norm(deformed_template-current_ref_shape,axis=1)
            mean_error = errors.mean()
            logging.info('Mean error with respect to previous iteration: {}'.format(mean_error))
            logging.info('Max error with respect to previous iteration: {}'.format(errors.max()))
            
            
            if((mean_error<error_tolerance)|(n_iter==MAX_ITER)):
                flag_leave = True
            
            current_ref_shape = deformed_template.copy()
            total_time = time.time()-start
            
            self.shapes_list.append(current_ref_shape)
            self.mean_err_to_prev.append(mean_error)
            self.max_err_to_prev.append(errors.max())
            self.iter_time.append(total_time)
            self.n_def.append(n_def)
            
            if(flag_SaveMat):
                self.keep_points.append(points)
                self.keep_def.append(deformations)
                self.keep_noise_vec.append(noise_vec) #full noise vec even withpoints that do not have deformations (missing)
                self.keep_noise_vec_full.append(noise_vec_with_mask)
                self.keep_post_var.append(posterior_var_gp)
                
                # Compute likelihood          
                t1, t2, t3, real = likelihood_CPD(proba_matrix, target_shape, deformed_template, var_reg,dim, mean,G_inv,I_D, varReg_update)
               
                self.current_likelihood.append(t1+t2+t3)
                self.likelihood_t1.append(t1)
                self.likelihood_t2.append(t2)
                self.likelihood_t3.append(t3)
                self.likelihood_real.append(real)

            
            logging.info('Time in this iterations {}'.format(total_time))
            logging.info('Number of deformations {}'.format(n_def))
            n_iter += 1
        
        self.target_shape = target_shape
        
        return self.shapes_list.copy(), final_corr
    
    ########################################################################
    # PLOTTING FUNCTIONS
    ########################################################################
    
    def plot_deformations(self,n_iter,vmin=None,vmax=None):
        ref_shape = self.ref_shape
        target_shape = self.target_shape
        points = self.keep_points[n_iter]
        noise = self.keep_noise_vec[n_iter]
        deform = self.keep_def[n_iter]
        
        if vmin is None : vmin = noise.min()
        if vmax is None : vmax = noise.max()
        # noise is th observation noise for the gp observations
        fig, ax = plot_deformations(ref_shape,target_shape,points,noise,deform,vmin=vmin,vmax=vmax)
        ax.set_title('Iteration '+ str(n_iter))
    
    
    def plot_dist_mat(self,n_iter,figsz=(10,8)):
        plt.figure(figsize=figsz)
        mat = self.keep_dist_mat[n_iter]
        sns.heatmap(mat,annot=True,fmt='.1f')
        plt.title('Distance matrix between current (deformed) template and target for iteration '+str(n_iter))
            
    
    def plot_phi_mat(self,n_iter,figsz=(10,8)):
        plt.figure(figsize=figsz)
        mat = self.keep_phi_mat[n_iter]
        sns.heatmap(mat,annot=True,fmt='.1f')
        plt.title('Phi matrix between current (deformed) template and target for iteration '+str(n_iter))
    
    def plot_phi_mat_post(self,n_iter,figsz=(10,8)):
        plt.figure(figsize=figsz)
        mat = self.keep_phi_mat_post[n_iter]
        sns.heatmap(mat,annot=True,fmt='.1f')
        plt.title('Phi matrix with posterior between current (deformed) template and target for iteration '+str(n_iter))
   
    def plot_proba_mat(self,n_iter,figsz=(10,8)):
        plt.figure(figsize=figsz)
        mat = self.keep_proba[n_iter]
        sns.heatmap(mat,annot=True,fmt='.1f')
        plt.title('Phi matrix with posterior between current (deformed) template and target for iteration '+str(n_iter))
    
    def plot_likelihood(self):
        # term 1 is likelihood of compatibility target template
        # term 2 is likelihood of variance
        # term 3 is prior on deformationss
        like_sum = self.current_likelihood
        t1, t2, t3 = self.likelihood_t1, self.likelihood_t2, self.likelihood_t3
        real = self.likelihood_real
        
        # fig, axs = plt.subplots(3,sharex=True)
        # axs[0].plot(like_sum, label = 'Complete likelihood')
        # axs[1].plot(t1, label = 'Term 1')
        # axs[1].plot(t2, label = 'Term 2 (variance) ')
        # axs[1].plot(t3, label = 'Term 3 (prior on deformations)')
        # axs[1].legend()
        # axs[2].plot(real, label = 'Negative log-likelihood')
        # axs[2].plot(np.array(real)+np.array(t3), label = 'Negative log-likelihood+prior')
        # axs[2].legend()
        
        fig, axs = plt.subplots(1,sharex=True)
        axs.plot(real, label = 'Without regularization')
        axs.plot(np.array(real)+np.array(t3), label = 'With regularization')
        axs.legend()
        axs.set_title('Negative log-likelihood (CPD)')
        
    def plot_gp_posterior(self,template_ids=None):
        
        template = self.ref_shape
        if template_ids is None:
            template_ids = np.arange(template.shape[0])

        fig = plt.figure(figsize=(8,8))
        for def_id in template_ids:
            noise_vec_pt = [a[def_id] for a in self.keep_post_var]
            plt.plot(noise_vec_pt,label='Point ID '+str(def_id))
            plt.yscale('log')
        plt.legend()
        plt.title('Variance of GP posterior at each point of the template over iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('Variance')

    def plot_var_term_SS(self,template_ids=None):
        
        template = self.ref_shape
        if template_ids is None:
            template_ids = np.arange(template.shape[0])

        fig = plt.figure(figsize=(8,8))
        for def_id in template_ids:
            noise_vec_pt = [a[def_id] for a in self.var_term_ss]
            plt.plot(noise_vec_pt,label='Point ID '+str(def_id))
            #plt.yscale('log')
        plt.legend()
        plt.title('Variance term - SS')
        plt.xlabel('Number of iterations')
        plt.ylabel('Variance')

    def plot_var_term_ST(self,template_ids=None):
        
        template = self.ref_shape
        if template_ids is None:
            template_ids = np.arange(template.shape[0])

        fig = plt.figure(figsize=(8,8))
        for def_id in template_ids:
            noise_vec_pt = [a[def_id] for a in self.var_term_st]
            plt.plot(noise_vec_pt,label='Point ID '+str(def_id))
            #plt.yscale('log')
        plt.legend()
        plt.title('Variance term - ST')
        plt.xlabel('Number of iterations')
        plt.ylabel('Variance')
        
    def plot_var_term_TT(self,template_ids=None):
        
        template = self.ref_shape
        if template_ids is None:
            template_ids = np.arange(template.shape[0])

        fig = plt.figure(figsize=(8,8))
        for def_id in template_ids:
            noise_vec_pt = [a[def_id] for a in self.var_term_tt]
            plt.plot(noise_vec_pt,label='Point ID '+str(def_id))
            #plt.yscale('log')
        plt.legend()
        plt.title('Variance term - TT')
        plt.xlabel('Number of iterations')
        plt.ylabel('Variance')


    def plot_var_term_sumVSpost(self,template_ids=None):
        
        template = self.ref_shape
        if template_ids is None:
            template_ids = np.arange(template.shape[0])

        
        for def_id in template_ids:
            fig = plt.figure(figsize=(8,8))
            term_vpost = [a[def_id] for a in self.var_term_vpost]
            term_sum= [a[def_id] for a in self.var_term_sum]
            plt.plot(term_vpost,label='Var posterior')
            plt.plot(term_sum,label='Var term sum')

            plt.yscale('log')
            plt.legend()
            plt.title('Point ID '+str(def_id))
            plt.xlabel('Number of iterations')
            plt.ylabel('Variance')
        
        
    def plot_var_terms(self,template_ids):
        
        template = self.ref_shape
        if template_ids is None:
            template_ids = np.arange(template.shape[0])

        
        for def_id in template_ids:
            fig = plt.figure(figsize=(8,8))
            term_vpost = [a[def_id] for a in self.var_term_vpost]
            term_st= [a[def_id] for a in self.var_term_st]
            term_tt = [a[def_id] for a in self.var_term_tt]
            term_ss = [a[def_id] for a in self.var_term_ss]
            plt.plot(term_vpost,label='Var posterior')
            plt.plot(term_st,label='Var term st')
            plt.plot(term_tt,label='Var term tt')
            plt.plot(term_ss,label='Var term ss')

            #plt.yscale('log')
            plt.legend()
            plt.title('Point ID '+str(def_id))
            plt.xlabel('Number of iterations')
            plt.ylabel('Variance')

        
    def plot_noise_vec(self,template_ids= None):
        
        template = self.ref_shape
        
        if template_ids is None:
            template_ids = np.arange(template.shape[0])
        fig = plt.figure(figsize=(8,8))
        
        for def_id in template_ids:
            noise_vec_pt = [a[def_id] for a in self.keep_noise_vec_full]
            plt.plot(noise_vec_pt,label='Point ID '+str(def_id))
            plt.yscale('log')
        plt.legend()
        plt.title('Observation noise at each point of the template over iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('Variance')
        
        
    def plot_like_posterior(self):
        
        fig = plt.figure()
        plt.plot(self.like_posterior)
        plt.ylabel('log det K')
        plt.xlabel('Iterations')

    
    def write_obj_results(self,folder,general_name):
        
        self.output_folder = folder
        self.output_files = list()
                
        for id_,shape in enumerate(self.shapes_list):

            mesh  =o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(shape*self.scale_x),triangles =o3d.utility.Vector3iVector(self.ref_shape_faces))
            mesh.compute_vertex_normals()

            file_name = general_name+'_iter_'+str(id_)+'.ply'
            path = os.path.join(folder, file_name)
            self.output_files.append(file_name)
            o3d.io.write_triangle_mesh(path, mesh)     
        
            
            
            
                   