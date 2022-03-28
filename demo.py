# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:31:55 2022

@author: filipavaldeira
"""

########################################################################

import os
import sys
import numpy as np
import pandas as pd

# Add code folder to path
my_path = os.path.abspath(os.path.dirname(__file__))
if my_path not in sys.path:
    sys.path.append(module_path)

# Local imports
from registration.reg_metrics import RegMetrics
from registration.gpreg import GpregRegistration

from gpflow.kernels import SquaredExponential
from gp_framework.gp_kernels import TotalKernel

from shapes.dataset import read_data_folder_obj
########################################################################

########################################################################
# STEP 1 - READ DATA
########################################################################

# Set paths
src_path_rel = r'resources\Ex_2d_rectangles'
template_path_rel = r'resources\Ex_2d_rectangles\1.csv'
src_path = os.path.abspath(os.path.join(my_path, src_path_rel))
template_path = os.path.join(my_path, template_path_rel)


# Read
dim=2
og_dataset = read_data_folder_obj(src_path,'',dim)

df = pd.read_csv(template_path,header=None )
template = df.values    
all_ids = og_dataset.get_id_list()

########################################################################
# STEP 2 - Example of transformations to the dataset
########################################################################

# Random missing data
id_list = [2,3]
miss_ratio = 0.6
dataset_t1 = og_dataset.random_missing_data(miss_ratio,id_list)

id_list = 'all'
noise_var = 0.01
dataset_miss = dataset_t1.random_noisy_data(noise_var, id_list)

########################################################################
# STEP 3 - Registration of the dataset
########################################################################

id_list = 'all' # which ids to register

# Define kernel
k_var = 1
k_l = 1
k_exp = SquaredExponential(variance=k_var,lengthscales=k_l)
full_kernel = TotalKernel(empirical_kernel=None,kernel_gpflow=k_exp,dim=dim)

# Parameters for registration
prob_out = 0.1 # prob_out
corr_method = 'our_proposal' # or closest_point
varReg_init = 2
proba_thresh = 1e-2
MAXITER = 100

gp_reg = GpregRegistration(full_kernel, corr_method, prob_out, varReg_init, proba_thresh,MAXITER=MAXITER)
reg_dataset = gp_reg.simple_registration(template, dataset_miss, id_list='all', flag_parallel = False)

########################################################################
# STEP 4 - Results and metrics
########################################################################

# Plot results
for i in all_ids:
    reg_dataset.plot_template_and_shape(i,0,'all')

# Obtain registration metrics
thresh_dist_failed_reg = 3
reg_metric = RegMetrics(dataset_miss,reg_dataset,thresh_dist_failed_reg,complete_dataset = og_dataset)
mean_error_target,frac_corr,frac_exact_corr= reg_metric.get_standard_metrics()
data_miss_out, data_fractions, data_dist = reg_metric.dataset_metrics_summary(plot_flag=True)
