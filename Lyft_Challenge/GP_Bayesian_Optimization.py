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


# Used for Hyperparameter Optimization for Neural Networks #
# Just make sure the variance of the prior is large enough to cover the accuracy range of the FCN #
print("Test for FCN Hyperparameter Tuning Using Bayesian Optimization")
X_star = np.linspace(-5, 5, n).reshape(-1,1) # Test points (Discretization of the Hyperparameter space)
X_train = np.array([[-1],[0],[1],[4]]) #samples to evaluate
Y_train = np.array([[1.5],[1.9],[-2.5],[0]]) #accuracy results of the above sample evaluations

mu_posterior, L_posterior = train_GP(X_star, X_train, Y_train)
mu, sigma = compute_and_plot_mu_var(X_star, 0, mu_posterior, L_posterior, to_plot=True)	
upper_conf_bound = mu+sigma
ucb_pos = np.argmax(upper_conf_bound)
Next_X_to_Sample = X_star[ucb_pos][0]
Predicted_Y_of_Next_X_to_Sample = np.max(upper_conf_bound)
print(Next_X_to_Sample)
print(Predicted_Y_of_Next_X_to_Sample)


'''
### Writeup for medium article: ###

INTRODUCTION
Image segmentation is an important subfield of computer vision. With the advent of Deep Learning, most areas of computer vision have been significantly impacted. Image segementation is no exception. This article focuses on applying deep learning based image segmentation for autonomous vehicle applications. In particular, this article discusses my implementation of the Lyft-Udacity Image Segmentation challenge that took place in earlier June. Due to other priorities, I was not able to publish this article earlier.

The architecture used is based upon UC Berkeley's original FCN architecture. In addition to the core FCN architecture, this article also utilizes Bayesian Optimization for hyperparameter tunining, particularly for the regularization parameter.

The dataset provided for this competiion is from the Carla simulator. See here for more details. The dataset is preprocessed to only include three classes: vehicle, road, and everything else. The dataset is divided in to training, validation, and test dataset. To help the model generalize better (i.e. improve FScore for the test dataset), data augmentation is done. In particular, images are randomly flipped, rotated with a variance of 30 degrees, the image color is randomly changed. This allows the model to learn from a wider variety of images. Note that data preprocessing is done as part of the netwrok inside tensorflow.


ARCHITECTURE
As stated earlier, the architecture used for image segmentaiton is based upon UC Berkeley's FCN8 architecture. It basically takes a VGG16 network and replaces the fully-connected layers with fully-convolutional layers. The output of the fully convolutional layer is then upsampled back to the input image's dimension using a learnable kernel. To help improve the accuracy in detecting smaller scale components, there are a couple skip layers which upsamples the output from earlier layers in the VGG16 network and combines/fuses (using matrix addition) all of them together at different places. The below diagram illustrates this.

The performance metric used is the weighted FScore (which basically combines precision and recall according to the below equation). Given that there are fewer cars in the images compared to roads and fewer roads compared to everything else, the dataset is skewed in favor of everything else. To address it, a weighted cross entropy loss is used so that more importance is given to weigth updates improving vehicle F1 score compared to other categories.

BAYESIAN OPTIMIZATION
One of the challenges in training deep learning models (e.g. FCNs) is hyperparameter (e.g. learning rate, regularization co-efficient, etc) optimizations.
What is typically done is to either do a grid search on the hyperparameters or randomly sample different values of the hyperparameters and train the FCN for a fixed number of epochs. Thereafter test the trained model on a validation dataset, and the hyperparameter set with the highest validation accuracy is chosen as the best set of hyperparameters.

The major problem with the above approach is that it takes pretty long time to train FCNs. And so it is very expensive in terms of training time (and the incurred cost of compute time on the cloud) to try many different values of hyperparameters to find the optimal hyperparameter.

While using the backpropagation algorithm, we can in theory compute the derivative of the loss with respect to some of the hyperparameters (e.g. regularization factor, learning rate, etc), it requires multiple iterations of gradient descent to find the optimal hyperparameter. And as we have already discussed above, each iteration involves training the neural newtrok for a certain number of epochs, and thus will take a very long time to find the optimal hyperparameter in this manner.

Again, given it is very time consuming to train a neural network for each set of hyperparameters, what we would ideally like to do is find the optimal value of the hyperparameter without needing to re-train the network many number of times using various different hyperparameters.

One way to address it is using Bayesian Optimization [2]. In this method, we construct an auxiliary function that models the mathematical relationship between the hyperparameters and the trained network's accuracy, but one that is alot easier to evaluate. The auxiliary function is consturcted using a Gaussian process (i.e. a Gaussian distribution over functions). The impetus for using a Gaussian process is that it gives us both the mean and the variance (i.e. a proxy for confidence interval) of the auxiliary; and together, they can be fed into the Upper Confidence Bound (UCB) algorithm (or even Thompson sampling) to determine the next set of hyperparameters to evaluate. This allows us to find the optimal hyperparameter while minimizing the number of hyperparameter trials required. Because we are intelligently deciding which hyperparameters to try next, it is much more efficient (and thus faster) in finding the optimal hyperparameter than doing grid search or random trials.

One downside of Bayesian optimization using Gaussian processes is that it suffers from the curse of dimensionality for higher dimensional spaces. This is because we have to discretize the hyperparameter space. However, for many machine learning applications, the hyperparameter space is inherentlty low dimensional. So this is not a problem in most cases, but just wanted to make the reader aware of the method's limitations.

As an aside, we could theoretically use Bayesian optimization to update the weights of the neural network without having to compute its gradients and be able to find the global optimum much faster. However, because there are millions of weights involved, the curse of dimensionality alluded earlier makes it computationally infeasible. Hence, it is not used to update the weights of the neural network.

For those familiar with Reinforcement Learning, the above mentioned Bayesian Optimization algorithm using Gaussian Processes is basically similar to the multi-armed bandit problem studied in Reinforcement Learning, where each discretized hyperparameter is a bandit arm and adjacent bandit arms are strongly correlated.


To Do:
1. Show the hyperparameter values tried using Bayesian Optimization
2. Show a few examples of data augmentation
3. Show final result FScore and IOU
4. Show a few examples of the final result from the test set
5. checkout the Medium article by the person (Asad Zia) who won the challenge

Conclusion
A VGG16 based FCN8 network was trained on the Udacity-Lyft challenge dataset and a final FScore of XXX was achieved. The speed of the netwrok is about 4FPS and thus more improvement on the speed front is a future goal. Moreover, finding the optimal regularization parameter was performed using Bayesian Optimization. In the future, instead of using a UCB based acquisition function, Thompson sampling can be tried to more quickly find the optimal hyperparameter. Another thing that can be addressed in the future is to perform Bayesian Optimization on all the hyperparameters used (i.e. regularization factor, learning rate, and batch size). While we can run three independent Bayesian Optimization algorithms (one for each hyperparameter), it assumes no correlation between the hyperparameters when finding the optimal hyperparameter, which is not a valid assumption. Thus a better way to do it is to perform Bayesian optimization on the three variables (i.e. while taking into account their correlations). For this project on the regularization parameter was optimized using Bayesian optimization beause the learning rate and batch size were easier to fine tune by trial-and-error. But it can certainly be combined with the regularization parameter to perform Bayesian optimization on all the hyperparameters. To do: Bayesian optimization using multivariate Gaussian processes.

 All of the code is available here: <Link to my Github Lyft Challenge project page>.


REFERENCES:
[1] Berkeley FCN Paper
[2] Nando De Freitas CS540 Bayesian Optimization Notes


###code for data augmentation###
batch_size = 4
img_h = 32
img_w = 32
img_chnl = 3
image_batch = np.random.random([batch_size, img_h, img_w, img_chnl])

def augment(image, gt_label):
	#use cv2 to resize images and read in images as its faster than scipy
	a = np.random.random()
	if a < 0.33:
		image = np.fliplr(image) #maybe even use cv2 for this
		gt_label = np.fliplr(gt_label) #maybe even use cv2 for this
	elif a < 0.66:
		r_M = cv2.getRotationMatrix2d(image)
		image = cv2.warpaffine(image,r_M)
		r_M = cv2.getRotationMatrix2d(gt_label)
		gt_label = cv2.warpaffine(gt_label,r_M)
	else:
		noise = np.random.random(image.shape)*50
		image = image + noise
	return image, gt_label


def image_augmentation(image, gt_label):
	a = 0.2 #percent of images to augment
	if np.random.random() < a:
		image,gt_label = augment(image,gt)
	else:
		#don't augment
	return image, gt_label

'''

print("End of Program")
