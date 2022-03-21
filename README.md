# GPReg

# Usage

1. Define the desired kernel
The kernel to be used should be of class TotalKernel, an extension of the gpflow kernel class. This allows for the combination of the empirical kernel and a generic gpflow kernel if desired. To build an empirical kernel one needs a template shape a list of registered shapes, all with the same number of points of the template in a one to one correspondence.

k_pca =  EmpiricalKernel(ref_shape = template, shape_list = reg_shape_list, dim = 3)
k_exp = SquaredExponential(variance = 1, lengthscales = 1)
full_kernel = TotalKernel(empirical_kernel = k_pca, kernel_gpflow = k_exp, dim = 3)


The registration method can be used by importing the class GpModel in gp_reg.py and calling the method iterative_reg
