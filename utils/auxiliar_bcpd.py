# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:30:40 2021

@author: filipavaldeira
"""

import numpy as np
from scipy.spatial.distance import cdist


# Auxiliar functions for BCPD
def compute_init_var(X,Y,N,M,D,gamma):
    #var = 0
    #for x_n in X:
    #    var += (np.linalg.norm(Y-x_n,axis=1)**2).sum()
    var = np.sum(X**2)*M+np.sum(Y**2)*N+np.sum(-2*Y.sum(axis=0)*X.sum(axis=0))
    var = np.sqrt(var/(N*M*D))*gamma #TODO different than paper!!
    return var**2

def normalize_shape(points):    
    scale = np.sqrt(np.sum((points-points.mean(axis=0))**2)/(points.shape[0]*points.shape[1]))
    normalized_shape = (points-points.mean(axis=0))/scale     
    return normalized_shape,scale

def Nystrom_approx(G,K_nystrom):
    
    D_gg, Q_gg = np.linalg.eig(G)
    D_gg, Q_gg = np.real(D_gg)[:int(K_nystrom)], np.real(Q_gg)[:,:int(K_nystrom)]
    return  D_gg, Q_gg




def get_matrices(ref_shape,target_shape,variance,prob_out,posterior_var_gp,flag_out,p_out_bcpd = None,s=1,flag_addPosterior = True):
    """ Given reference and target computes distance matrices

        Parameters
        ----------
        ref_shape : numpy array M*D
            reference shape
            
        target_shape : Numpy array N*D
            target shape
        
        variance : float
            noise / sigma^2
        
        prob_out : float 
            outlier probability
            
        posterior_var_gp : numpy array
            equivalent to sigma_m in registration
            
        scale : float
            scale, if not provided then we go for 1
        
        flag_out : Bool
            if True we return all intermediate matrices
            
        Returns
        ------
        
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

        
    # Dimensions
    M, D = ref_shape.shape
    N =  target_shape.shape[0]
    w = prob_out
    
    # distance matrix M by N
    den = np.sqrt(variance*(2*np.pi))**D
    dist_mat = cdist(ref_shape,target_shape) # norm between each point in template and target
    phi_mat = np.transpose(np.exp(-(dist_mat.T)**2/(2*variance))/den) # transpose twice to do column division reverts back at the end.
    exp_term = np.exp( (-s**2/(2*variance)) *(D*posterior_var_gp) ) # posterior of GP . M*1
    if flag_addPosterior:
        phi_mat_post = np.transpose(phi_mat.T*exp_term)
    else:
        phi_mat_post = phi_mat

    # Probability matrix
    # if no p_out is given we use the one from bcpd
    if (p_out_bcpd is None):
        p_out_bcpd = np.ones(N)*(1/N) 
    
    den_aux = w*M*p_out_bcpd + (1-w)*phi_mat_post.sum(axis=0)
    proba_matrix = (1-w)*phi_mat_post/den_aux
    
    
    if(flag_out):
        likelihood_term = -0.5*(1/variance)*np.multiply(dist_mat**2,proba_matrix).sum()

        return proba_matrix, dist_mat, phi_mat, phi_mat_post, likelihood_term
    else:
        return proba_matrix,None,None,None,None
    
    return None


def get_step_1_vars(P,x,D):
    
    M = P.shape[0]
    N = P.shape[1]
    I_D = np.eye(D)
    
    v = np.matmul(P,np.ones(N)) # this is w in c prog        
    v_line =  np.matmul(P.T,np.ones(M))

    N_hat = np.matmul(v.T,np.ones(M))
    
    x_hat= np.matmul(np.matmul(np.diag(np.reciprocal(v)),P),x.reshape(-1,D)).reshape(-1,1)
    
    
    return v,v_line,N_hat,x_hat
    
def get_step_2_vars(Q,D,var,v,lmbda,x_hat,I_D,y,R=None,t=None,s=1,flag_return_like=False):
    
    M = v.shape[0]
    I_M = np.eye(M)
    dim = I_D.shape[0]
    I_K = np.eye(Q.shape[1])
    
    #Sigma = np.linalg.inv(lmbda*G_inv+ ((s**2)/var)*np.diag(v))
    #2. v_hat
    #T_inv = np.matmul(np.linalg.inv(np.kron(I_M,R)),(x_hat - np.kron(np.ones((M,1)),t))*(1/s))
    #T_inv = x_hat
    #v_hat = (s**2/var)*np.matmul(np.matmul(np.kron(Sigma,I_D),np.kron(np.diag(v),I_D)) , (T_inv-y))

    S_aux = Q.T @ np.diag(v) @ Q
    inv_term = np.linalg.inv(np.diag(np.reciprocal(D))*(lmbda*var/s**2)+S_aux)
    Sigma = (1/lmbda)*Q @ np.diag(D)@ (I_K- S_aux @ inv_term) @ Q.T
    if (R is None):
        T_inv = x_hat
    else:
        #T_inv = np.matmul(np.linalg.inv(np.kron(I_M,R)),(x_hat - np.kron(np.ones((M,1)),t))*(1/s))
        T_inv = ((1/s)*(np.linalg.inv(R)@(x_hat.reshape(-1,dim) - t.reshape(-1,dim)).T).T).reshape(-1,1)
    #v_hat = (s**2/var)*np.matmul(np.matmul(np.kron(Sigma,I_D),np.kron(np.diag(v),I_D)) , (T_inv-y))
    v_hat = ((s**2/var)*Sigma @ np.diag(v) @ (T_inv-y).reshape(-1,dim)).reshape(-1,1)
    if(flag_return_like):
        G_inv = I_M-( Q @ np.linalg.inv(np.diag(np.reciprocal(D)) + Q.T @ Q) @Q.T)
        likelihood = -(lmbda/2)*v_hat.T @(G_inv @v_hat.reshape(-1,dim)).reshape(-1,1)
        return Sigma, v_hat,likelihood
    
    else:
        return Sigma, v_hat
    
    return None
    
def update_var(x,v_line,P,I_D,y_hat,v,Sigma,N_hat,D,s=1):
    
    term_1 = np.matmul(x.T,np.matmul(np.diag(v_line),x.reshape(-1,D)).reshape(-1,1)) # reshape replaces kronecker
    term_2 = -2*np.matmul(x.T,np.matmul(P.T,y_hat.reshape(-1,D)).reshape(-1,1))
    term_3 = np.matmul(y_hat.T,np.matmul(np.diag(v),y_hat.reshape(-1,D)).reshape(-1,1))
    # in bcpd code the addtional term should be:
    # (s**2)*np.diag(Sigma).sum()/N_hat £ without multiplying by v

    var = (1/(N_hat*D))*(term_1+term_2+term_3)+np.multiply(np.diag(Sigma),v).sum()*(s**2/N_hat)
    return var
    
    
def likelihood_CPD(proba_matrix, target, updated_template, updated_var,dim, deformations,G_inv,I_D,varReg_update):
    # term 1 is likelihood of compatibility target template
    # term 2 is likelihood of variance
    # term 3 is prior on deformationss
    dist_mat = cdist(updated_template,target)
    if varReg_update=='gp_posterior':
        # in this case var is a vector 
        term1 = -0.5* np.sum((1/updated_var)*np.multiply(dist_mat**2,proba_matrix).sum(axis=1))
        term2 = (dim/2) *np.sum( proba_matrix.sum(axis=1) *  np.log(updated_var))
        term3 = -0.5*np.matmul(np.transpose(deformations),np.matmul(np.kron(G_inv,I_D),deformations))
        term3 = term3[0][0]
        
        
    else:
        term1 = -0.5*(1/updated_var)*np.multiply(dist_mat**2,proba_matrix).sum()
        term2 = proba_matrix.sum() * (dim/2) * np.log(updated_var)
        term3 = 0.5*np.matmul(np.transpose(deformations),np.matmul(np.kron(G_inv,I_D),deformations))
        term1, term2, term3 = term1[0], term2[0], term3[0][0]
    
    real_like = -np.sum(np.log( np.transpose((1/(updated_var)**(dim/2))*np.exp(-(1/(2*updated_var))*np.transpose(dist_mat**2))).sum(axis=0)))
    
       
 
    return term1, term2, term3, real_like