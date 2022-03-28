# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:31:12 2021

@author: filipavaldeira
"""
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.kernels import Kernel
from typing import Optional
from gpflow.mean_functions import MeanFunction
import tensorflow as tf
from gpflow import likelihoods, conditionals
from gpflow.models.util import data_input_to_tensor
from gpflow.logdensities import multivariate_normal
from gpflow.utilities import deepcopy
from gpflow.models import GPR

import numpy as np

# High dimension regression where covariance is diagonal for the dimensions.
def GPR_multiDim(X,Y,dim,kernel,var_observations):
    #MAXITER_ = ci_niter(MAXITER)    
    model_list = list()
    for i in np.arange(dim):
        kernel_ = deepcopy(kernel) # otherwise the kernel gets changed
        model = GPR_multi_noise(dim=1,data=(X, Y[:,i].reshape(-1,1)), kernel=kernel_, mean_function=None,noise_variance= var_observations)
        model_list.append(model)
    return model_list

def GPR_multiDim_finalShape(X,Y,dim,kernel,var_observations,ref_shape):
    model_list = GPR_multiDim(X,Y,dim,kernel,var_observations)
    final_shape = ref_shape.copy()
    final_mean = np.zeros(ref_shape.shape)
    final_var =  np.zeros(ref_shape.shape)
    for i in np.arange(dim):
        mean_shape, var_shape = model_list[i].predict_f(ref_shape)
        final_shape[:,i] = final_shape[:,i]+mean_shape.numpy()[:,0]
        final_mean[:,i] = mean_shape.numpy()[:,0]
        final_var[:,i] = var_shape.numpy()[:,0]
    return final_shape, final_mean, final_var



class GPR_multi_noise(GPModel, InternalDataTrainingLossMixin):
    r"""
    Reimplementation of  Gaussian Process Regression from gpflow, except noise observation is not constant
    
    """

    def __init__(
        self,dim,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance = 1.0,
    ):
        likelihood = likelihoods.Gaussian(1)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)
        self.noise_vec = tf.repeat(noise_variance,dim)
        self.dim = dim

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()


    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        K = self.kernel(X)
        num_data = tf.shape(X)[0]
        k_diag = tf.linalg.diag_part(K)
        #s_diag = tf.fill([num_data], self.likelihood.variance)
        s_diag =  self.noise_vec
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)


    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        num_data = X_data.shape[0]
        
        #s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))
        s = tf.linalg.diag(self.noise_vec)

        conditional = conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var