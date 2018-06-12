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
n = 50 # number of test points.
X_Full_Span = np.linspace(-5, 5, n).reshape(-1,1) # Test points.
mu_prior = np.zeros((n,1))
K_prior = kernel(X_Full_Span, X_Full_Span) # Kernel at test points. (this is the covariance matrix)
# draw samples from the prior at our test points.
L_prior = np.linalg.cholesky(K_prior + 1e-10*np.eye(n)) #add a small number for numerical stability
#L_prior = 3*np.eye(n)
f_prior = mu_prior + np.dot(L_prior, np.random.normal(size=(n,100))) #generate 10 samples

plt.plot(X_Full_Span, f_prior)
plt.show()

mu = np.mean(f_prior,axis=1).reshape(-1,1)
var = (f_prior-mu) * (f_prior-mu)
var = np.mean(var,axis=1).reshape(-1,1)
print(var.shape)

var = np.sum(K_prior,axis=1).reshape(-1,1)
# var = np.dot(K_prior,np.ones((n,1)))
var = np.sqrt(var)
# var = np.dot(np.linalg.pinv(L_prior),var)

upper_bound = mu + var #np.dot(L_prior, np.ones((n,1)))
lower_bound = mu - var #np.dot(L_prior, np.ones((n,1))) #np.dot(L_prior*np.eye(n), np.ones((n,1)))
plt.plot(X_Full_Span,mu, 'g^', X_Full_Span, upper_bound, 'b-', X_Full_Span, lower_bound, 'r-')
plt.show()


exit()

#The True Function (Unknown but can be sampled at individual x points)
def gt_function(Xin):
	Yout = 1*np.sin(Xin) + np.cos(1*Xin) #3 + 0.2*Xin + 0.2*Xin**2 + 0.01*Xin**3 + 0.01*Xin**4
	return Yout

y_gt = gt_function(X_Full_Span)
plt.plot(X_Full_Span,y_gt)
plt.show()


#Training
#Posterior (Noiseless GP Regression)
N_star = 50 # number of test points.
X_star = np.linspace(-5, 5, N_star).reshape(-1,1) # Test points.

X_train = np.array([X_star[0], X_star[10], X_star[20], X_star[30], X_star[40]]).reshape(-1,1) #N_train x 1
N_train = X_train.shape[0]
Y_train = gt_function(X_train) #N_train x 1
# # No need to. Make X_train have N_star rows each having the same X_train above
# X_train_mat = X_train.reshape(1,-1)
# X_train_mat = np.zeros((N_star,N_train)) + X_train_mat #broadcast into N_star x N_train
# Y_train = gt_function(X_train_mat)
# print(Y_train.shape)

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

# Draw samples from the posterior
print('for N_train=2 (i.e. two sample points)')
# draw samples from the prior at our test points.
print(covar_posterior.shape)
L_posterior = np.linalg.cholesky(covar_posterior + 1e-10*np.eye(N_star)) #add a small number for numerical stability
f_posterior = mu_posterior + np.dot(L_posterior, np.random.normal(size=(N_star,10))) #generate 10 samples
plt.plot(X_star, f_posterior)
plt.show()

mu = np.mean(f_posterior,axis=1).reshape(-1,1)
var = (f_posterior-mu) * (f_posterior-mu)
var = np.mean(var,axis=1).reshape(-1,1)
print(var.shape)

upper_bound = mu + var #np.dot(L_prior, np.ones((n,1)))
lower_bound = mu - var #np.dot(L_prior, np.ones((n,1))) #np.dot(L_prior*np.eye(n), np.ones((n,1)))
plt.plot(X_star,mu, 'g^', X_star, upper_bound, 'b-', X_star, lower_bound, 'r-')
plt.show()
