from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def kernel(a, b):
	""" GP squared exponential kernel """
	sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
	kernel_rtn = np.exp(-0.5 * sqdist)
	#kernel_rtn = np.zeros((a.shape[0],b.shape[0]))
	return kernel_rtn

n = 50 # number of test points.
X_Full_Span = np.linspace(-5, 5, n).reshape(-1,1) # Test points.
mu_prior = np.zeros((n,1))
K_prior = kernel(X_Full_Span, X_Full_Span) # Kernel at test points.
# draw samples from the prior at our test points.
L_prior = np.linalg.cholesky(K_prior + 1e-10*np.eye(n)) #add a small number for numerical stability
f_prior = mu_prior + np.dot(L_prior, np.random.normal(size=(n,10))) #generate 10 samples
plt.plot(X_Full_Span, f_prior)
plt.show()


#The True Function (Unknown but can be sampled at individual x points)
def gt_function(Xin):
	Yout = 3 + 2*Xin + 1*Xin**2 + 0.8*Xin**3 + 0.01*Xin**4
	return Yout

y_gt = gt_function(X_Full_Span)
plt.plot(X_Full_Span,y_gt)
plt.show()

#Training
#Posterior (Noiseless GP Regression)
X_train = np.array([X_Full_Span[20], X_Full_Span[30]]).reshape(-1,1)
#Make X_train have n rows each having the same X_train above
X_train_mat = X_train.reshape(1,-1)
X_train_mat = np.zeros((n,X_train.shape[0])) + X_train_mat #broadcast into n x N_star
Y_train = gt_function(X_train_mat)
# print(Y_train.shape)
K_ = kernel(X_Full_Span, X_Full_Span) # Kernel at test points (n X n)
L = np.linalg.cholesky(K_ + 1e-10*np.eye(n)) #add a small number for numerical stability
L_T = np.transpose(L)
# Mu Posterior
K_inv = np.dot(np.linalg.pinv(L_T),np.linalg.pinv(L))
alpha = np.dot(K_inv,Y_train) #n X N_star
# print(alpha.shape)
K_star = kernel(X_Full_Span,X_train) #n x N_star
# print(K_star.shape)
mu_posterior = np.dot(np.transpose(K_star),alpha)
mu_posterior = mu_posterior[0,:] #all the rows are same. Pick the first one. N_star X 1
# mu_posterior = np.dot(np.dot(K_star.transpose(),np.linalg.pinv(K_)),Y_train)
print(mu_posterior)
print()

#Var Posterior
v = np.dot(np.linalg.pinv(L),K_star) #n X N_star
K_star_star = kernel(X_train,X_train) #N_star X N_star
var_posterior = K_star_star - np.dot(v.transpose(),v) #N_star X N_star
print(var_posterior)

# Inference
print('for N_star=2')
position0 = (X_train[0] == X_Full_Span).nonzero()[0]
mu_prior[position0] = mu_posterior[0]
position1 = (X_train[1] == X_Full_Span).nonzero()[0]
mu_prior[position1] = mu_posterior[0]
print(position0,position1)
# K_prior[position0,position0] = var_posterior[0,0]
# K_prior[position0,position1] = var_posterior[0,1]
# K_prior[position1,position0] = var_posterior[1,0]
# K_prior[position1,position1] = var_posterior[1,1]
K_prior[position0,:] = var_posterior[0,0]
K_prior[position1,:] = var_posterior[1,1]

#
# draw samples from the prior at our test points.
print(K_prior.shape)
#L_prior = np.linalg.cholesky(K_prior + 1e-8*np.eye(n)) #add a small number for numerical stability
import scipy.linalg
L_prior = scipy.linalg.sqrtm(K_prior)
f_prior = mu_prior + np.dot(L_prior, np.random.normal(size=(n,10))) #generate 10 samples
plt.plot(X_Full_Span, f_prior)
plt.show()