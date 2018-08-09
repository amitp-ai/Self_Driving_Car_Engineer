from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


#Implement Bayesian Optimization using Gaussian Processes
#Based upon Nando De Freitas UBC CS540 (Homework 3)

def kernel(a, b):
	""" GP squared exponential kernel """
	sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
	kernel_rtn = 5*np.exp(-0.5 * sqdist) #make sure its scaled properly such that it covers the full (y-axis) range of the function trying to model
	#kernel_rtn = np.zeros((a.shape[0],b.shape[0]))
	return kernel_rtn

#Prior
def prior_GP(X_prior, to_plot=True):

	n_prior = X_prior.shape[0]

	mu_prior = np.zeros((n_prior,1))
	K_prior = kernel(X_prior, X_prior) # Kernel at test points. (this is the covariance matrix)
	# draw samples from the prior at our test points.
	L_prior = np.linalg.cholesky(K_prior + 1e-10*np.eye(n_prior)) #add a small number for numerical stability
	#L_prior = 3*np.eye(n_prior)
	f_prior = mu_prior + np.dot(L_prior, np.random.normal(size=(n_prior,1000))) #generate 10 samples

	if(to_plot):
		plt.plot(X_prior, f_prior)
		plt.show()

	return f_prior, mu_prior, L_prior

n = 50 # number of test points.
X_prior = np.linspace(-5, 5, n).reshape(-1,1) # Test points.
f_prior, mu_prior, L_prior = prior_GP(X_prior)


def compute_and_plot_mu_var(x_dist, f_dist, mu_dist, L_dist, to_plot=True):

	#True Mean and Variance (analytical approach)
	mu = mu_dist
	L_dist_temp = L_dist*L_dist #elementwise multiplication
	var_temp = np.ones((n,1)) #variance (it is 1 in this example) of each discretized point (independent of each other)
	var = np.dot(L_dist_temp,var_temp)
	sigma = np.sqrt(var)
	#
	upper_bound = mu + sigma
	lower_bound = mu - sigma

	if(to_plot):
		plt.plot(x_dist, mu, 'g^', x_dist, upper_bound, 'b-', x_dist, lower_bound, 'r-')
		plt.xlabel('Log10 of Regularization Parameter Lambda')
		plt.ylabel('F-Score')
		plt.show()


	# #Debug only (for sanity check only) (Numerical approach)
	# #sample mean and variance
	# mu = np.mean(f_dist,axis=1).reshape(-1,1)
	# var = (f_dist-mu) * (f_dist-mu)
	# var = np.mean(var,axis=1).reshape(-1,1)
	# sigma = np.sqrt(var)
	# #
	# upper_bound = mu + sigma
	# lower_bound = mu - sigma
	# plt.plot(x_dist, mu, 'g^', x_dist, upper_bound, 'b-', x_dist, lower_bound, 'r-')
	# plt.show()

	return mu, sigma

compute_and_plot_mu_var(X_prior, f_prior,mu_prior,L_prior)


#The True Function (Unknown but can be sampled at individual x points)
def gt_function(Xin):
	Yout = 1*np.sin(Xin) + np.cos(2*Xin) #3 + 0.2*Xin + 0.2*Xin**2 + 0.01*Xin**3 + 0.01*Xin**4
	return Yout

y_gt = gt_function(X_prior)
plt.plot(X_prior,y_gt)
plt.show()

# Find the Posterior for a Gaussian Process Prior #
def train_GP(X_star, X_train, Y_train):
	#Training
	# X_train and Y_train are 1D vectors

	# X_train = np.array([X_star[0], X_star[10], X_star[20], X_star[30], X_star[40]]).reshape(-1,1) #N_train x 1
	# #X_train = np.array([-5,-3,-1,1,3]).reshape(-1,1) #This works but depending on the X_train resolution, the variance never goes to zero at sampled points even for noiseless GP regression
	# Y_train = gt_function(X_train) #N_train x 1
	
	#Posterior (Noiseless GP Regression)
	N_star = X_star.shape[0]
	N_train = X_train.shape[0]
	K_train = kernel(X_train, X_train) # Kernel at test points (N_train X N_train)
	L_train = np.linalg.cholesky(K_train + 1e-10*np.eye(N_train)) #add a small number for numerical stability
	L_train_T = np.transpose(L_train)
	# Mu Posterior
	K_inv = np.dot(np.linalg.pinv(L_train_T),np.linalg.pinv(L_train)) #(N_train X N_train)
	alpha = np.dot(K_inv,Y_train) #N_train X 1
	# print(alpha.shape)
	K_star = kernel(X_train,X_star) #N_train x N_star
	# print(K_star.shape)

	mu_posterior = np.dot(np.transpose(K_star),alpha)
	# mu_posterior = np.dot(np.dot(K_star.transpose(),np.linalg.pinv(K_train)),Y_train) #leff efficient numerically as it doesn't use Cholesky
	# print(mu_posterior)

	#Var Posterior
	v = np.dot(np.linalg.pinv(L_train),K_star) #N_train x N_star
	K_star_star = kernel(X_star,X_star) #N_star X N_star
	covar_posterior = K_star_star - np.dot(v.transpose(),v) #N_star X N_star
	# print(covar_posterior)

	L_posterior = np.linalg.cholesky(covar_posterior + 1e-10*np.eye(N_star)) #add a small number for numerical stability

	return mu_posterior, L_posterior

"""
# Test Gaussian Process Function #
N_star = 50 # number of test points.
X_star = np.linspace(-5, 5, N_star).reshape(-1,1) # Test points.

X_train = np.array([X_star[0], X_star[10], X_star[20], X_star[30], X_star[40]]).reshape(-1,1) #N_train x 1
Y_train = gt_function(X_train) #N_train x 1
mu_posterior, L_posterior = train_GP(X_star, X_train, Y_train)


# Draw samples from the posterior
print('Test Samples')
# draw samples from the prior at our test points.
f_posterior = mu_posterior + np.dot(L_posterior, np.random.normal(size=(N_star,1000))) #generate 10 samples
plt.plot(X_star, f_posterior)
plt.show()

compute_and_plot_mu_var(X_star, f_posterior, mu_posterior, L_posterior)


# Bayesian Optimization to find the maximum value of the gt_function() without finding the gradient and without requiring alot of samples (i.e. function evaluations)
# The objective is to minimize the number of sample evaluations and still be able to find the maximum value of the function from the limited function evaluations.
# It balances exploration vs exploitation using the upper confidence bound metric

n = 50 # number of test points.
X_prior = np.linspace(-5, 5, n).reshape(-1,1) # Test points.
f_prior, mu_prior, L_prior = prior_GP(X_prior, to_plot=False)
mu, sigma = compute_and_plot_mu_var(X_prior, f_prior, mu_prior, L_prior, to_plot=False)

upper_conf_bound = mu+sigma
ucb_pos = np.argmax(upper_conf_bound)
X_train = X_prior[ucb_pos].reshape(-1,1)
Y_train = gt_function(X_train)
# print(X_train)
# print(Y_train)
# print(X_train.shape)
# print(Y_train.shape)

num_epochs = 5
X_star = np.copy(X_prior)
for epoch in range(num_epochs):
	mu_posterior, L_posterior = train_GP(X_star, X_train, Y_train)
	mu, sigma = compute_and_plot_mu_var(X_star, 0, mu_posterior, L_posterior, to_plot=True)	

	# print(X_train,'\n')
	# print(Y_train,'\n')

	upper_conf_bound = mu+sigma
	ucb_pos = np.argmax(upper_conf_bound)
	X_train_new = X_star[ucb_pos].reshape(-1,1)
	X_train = np.vstack((X_train, X_train_new))
	Y_train_new = gt_function(X_train_new)
	Y_train = np.vstack((Y_train, Y_train_new))

	max_function_pos = np.argmax(Y_train)
	Y_max_val = Y_train[max_function_pos]
	X_for_Y_max_val = X_train[max_function_pos]
	print(X_for_Y_max_val, Y_max_val)

	# print(X_train,'\n')
	# print(Y_train,'\n')

	# break

"""

# Used for Hyperparameter Optimization for Neural Networks Training #
# Just make sure the variance of the prior is large enough to cover the average F score range of the FCN #
print("Test for FCN Hyperparameter Tuning Using Bayesian Optimization")
print("X is regularization parameter lambda and Y is the averaged F score")
n = 50 # number of test points.
#X is the log10 of reg_lambda (as the full range of lamda is on log scale)
X_star = np.linspace(np.log10(1e-5), np.log10(1e-1), n).reshape(-1,1) # Test points (Discretization of the Hyperparameter space)
#X_star = np.logspace(-5, -2, n).reshape(-1,1) # Test points (Discretization of the Hyperparameter space) for lambda_reg
X_train = np.array([[np.log10(2e-2)],[np.log10(7.54e-4)],[np.log10(1.758e-5)],[np.log10(0.1)],[np.log10(1.15e-4)]]) #samples to evaluate
Y_train = np.array([[0.876],[0.915],[0.879],[0.879],[0.938]]) #averaged F score from validation data for above samples

mu_posterior, L_posterior = train_GP(X_star, X_train, Y_train)
mu, sigma = compute_and_plot_mu_var(X_star, 0, mu_posterior, L_posterior, to_plot=True)	
upper_conf_bound = mu+sigma
ucb_pos = np.argmax(upper_conf_bound)
Next_X_to_Sample = X_star[ucb_pos][0]
Predicted_Y_of_Next_X_to_Sample = np.max(upper_conf_bound)
print('Next_X_to_Sample in log: ', Next_X_to_Sample, ' True Next_X_to_Sample: ', 10**Next_X_to_Sample)
print('Predicted_Y_of_Next_X_to_Sample: ', Predicted_Y_of_Next_X_to_Sample)
