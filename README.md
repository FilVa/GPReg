# GPReg

# Usage

Define the desired kernel
--

The kernel to be used should be of class TotalKernel, an extension of the gpflow kernel class. This allows for the combination of the empirical kernel and a generic gpflow kernel if desired. To build an empirical kernel one needs a template shape a list of registered shapes, all with the same number of points of the template in a one to one correspondence.

k_pca =  EmpiricalKernel(ref_shape = template, shape_list = reg_shape_list, dim = 3)
k_exp = SquaredExponential(variance = 1, lengthscales = 1)
full_kernel = TotalKernel(empirical_kernel = k_pca, kernel_gpflow = k_exp, dim = 3)

Parameters for the registration
--

### Correspondence type

The class GpModel performs to kinds of registration - the trivial one with the closest point and the proposed method GPReg. Depending on the method chosen different parameters need to be set.

1. "Closest point correspondence"

corr_method = 'closest_point'
var_obs [float] : observation variance for GPR
max_dist [float] : maximum distance when searching for neighbours with closest point

varReg_update, var_obs and varReg_init are not applicable.


2. "Probabilistic correspondence"

corr_method = 'our_proposal'

varReg_init [float] : initial variance for the registration.
prob_out [float] : outlier probability $\omega$

### General parameters

maxiter [int] : maximum number of iterations allowed


Execute the registration
--

The registration can be then executed by calling the method iterative_reg of class GpModel.


