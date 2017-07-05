1. The project can be run using cmake and make as per the instructions provided.

2. Hyperparameter tuning was achieved in ways shown in the lectures.

3. How PID controller works
	- While the plant (i.e. vehicle) dynamics model was not provided, I used my intuition in how a PID controller works to setup the initial choice for the 	PID parameters. I tried to first increase P (with the other parameters set to zero) and see how high it could go before the car starts to become
	unstable -- because the P parameter controls the very low frequency loop gain of the closed loop system. Then I increased the D parameter to improve 
	stability as well as response time of the controller. A higher D parameter means the controller will be more responsive to sudden turns. However, too 
	high of D can cause it to go unstable. I then added the I parameter to reduce steady state error while still being fairly responsive to turns. Given the 
	I parameter has memory, after a while the controller doesn't need to act according to the initial cross-track error. To address this, there are various 	anti-windup strategies employed. One I chose is to use the following eqution to determine the I_error:
	i_error = 0.90*i_error + 1.3*cte; //this way old cte values will be forgotten as they are iteratively multiplied by a value below 1

	- I also reduce the throttle for high steering angles. This allows for better control


4. Hyperparameter optimization	
	- Once I got decent initial values of the PID parameters using the above approach, I setup a twiddle algorithm to optimize the PID parameters. But given 
	it's a non-convex optimization problem, the	initial values are very important. Hence the above approach was helpful

	- So the hyperparameter optimization was a combination of both manual tuning as well as twiddle.

	- SGD was not implemented as to properly calculate the gradients I didn't have an accurate model of the vehicle dynamics and the transfer function of how 
	the cte was calculated.


5. As can be verified, no tire of the car leaves the drivable portion of the road.

