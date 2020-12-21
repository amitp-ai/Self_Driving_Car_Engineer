# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

The project can be run using cmake and make as per the instructions provided.

## 1. The model Description
**State**
Since the waypoints are in map coordinates, they are first transformed in to vehicle coordinates (by translation followed by rotation).

Given MPC is a constrained optimization problem at every time step over some finite time horizon, having a good vehicle dynamics model is very important. However, there is tradeoff between model accuracy and computation speed. And given the optimization problem is to be solved at every point in time, the model cannot be very complicated due to computational challenges. For this project, the model used is the basic kinematics model presented in the lectures. It is a state-space based kinematics model with the following components constituting the state:
	a. x position of the car
	b. y position of the car
	c. psi (orientation) of the car
	d. v (velocity/speed) of the car
	e. cte (cross track error) is the difference between the car position and the reference trajectory. This error needs to be minimized.
	f. epsi (psi error) is the difference between the car's orientation and the desired orientation. This is to be minimized.
The variables x, y, ψ and v are received from the simulator. The x, y position are in the map coordinates and are converted into the vehicle coordinate system. 

**Actuators**
The actuation includes throttle (controls acceleration) and steering angle (controls orientation) of the car. These are obtained from the result of the solver and passed to the simulator.

**Update Equations**
The update equations are basically kinematic equations of motion used to set constaints on the optimization problem so that the optimization takes in to account the vehicle dynamics.

	  x = x + v*cos(ψ)* dt
	  y = y + v sin(psi) dt
	  v=v+a∗dt
	  a in [-1,1]
	  ψ=ψ+(v/L_f)*δ∗dt

**Cost Function**
The key to solving MPC equation is to define the cost properly. This requires a fair bit of tuning. The solver tries to minimize the cost. 

The cost function to be minimized is a quadratic cost function of cte, epsi, difference between current and reference speed, actuations, as well as the rate of change of actuations. cte and epsi are given large weights as well as the rate of change of steeringing angle and throttle for a smooth and accurate controller.

 for (int i = 0; i < N; i++) {
      fg[0] += cost_cte* CppAD::pow(vars[cte_start + i] - ref_cte, 2);
      fg[1] += cost_eps* CppAD::pow(vars[epsi_start + i] - ref_epsi, 2);
      fg[2] += cost_v* CppAD::pow(vars[v_start + i] - ref_v, 2);
    }

The reference CTE is set to zero, reference epsi is set to zero and ref velocity set to 40. N is the no. of timesteps we are forecasting into the future. The errors were all multiplied by factors (cost_cte, cost_eps etc) which were very important for smooth driving in the simulator and these multipliers were tuned by trying various values

In order to speedup the optimization process, the the initial values of actuations (sent to the MPC solver) is the same as the last set of actuations. This is better than setting the initial actuations to zero or to randomly select them.

The cost also depends on the actuator values and the change in actuation from previous time step. This ensures smooth changes in actuator values. Again the multipliers associated with them were tuned by looking at performance in the simulator. 

for (int i = 0; i < N - 1; i++) {
      fg[0] += cost_current_delta*CppAD::pow(vars[delta_start + i], 2);
      fg[1] += cost_current_a*CppAD::pow(vars[a_start + i], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int i = 0; i < N - 2; i++) {
      fg[0] += cost_diff_delta* CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      fg[1] += cost_diff_a*CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
    }

**MPC Setup**
As described in the lectures, the following are the steps for executing MPC

1. Define the length of the trajectory, N, and duration of each timestep, dt.
        See below for definitions of N, dt, and discussion on parameter tuning.

2. Fit polynomial to way points and use that to set the intial cross track error and orientation error
3. Define vehicle dynamics and actuator limitations along with other constraints.
        See the state, actuators and update equations above.
4. Define the cost function (as explained above)

Once all the model is set and all the parameters are defined, 
1. We pass the current state as the initial state to the model predictive controller.

2. We call the optimization solver. Given the initial state, the solver will return the vector of control inputs that minimizes the cost function. The solver we'll use is called Ipopt.
3. We apply the first control input to the vehicle.
 and  Back to 1.

## 2. Timestep Length (N) and Frequency (dt)

* Timestep length (N) - Is the number of timesteps the model predicts ahead. As N increases, the model predicts further ahead. The further ahead that the model predicts, the more inaccurate those predictions will be. Also if N is large, then a lot more computations need to happen which can lead to inaccurate results from the solver or solver unable to provide the solution in real time. I found that N around 10-15 steps was the best.

* Frequency (dt) - Is the time gap between different time steps. I found dt to be a very important parameter for the overall performance in the simulator. If dt was too low, it led to the car oscillating to and fro around the center of the track. This happens probably because the actuator input is received very quickly and the vehicle is constantly responding. Also if dt is smaller than the latency (0.1 s) then the new actuator signal is received before the previous one has been executed. This leads to jerky motion of the car. On the other hand if dt is too large, the car covers too much distance before the actuator is received and while this leads to smooth performance along the straight portions of the track, it leads to car going off the road on the curves. 

* N and dt are two of the many parameters used for tuning the MPC. N and dt together determines the horizon over which optimization is to be performed. In addition to computational challenges of using a large N, given the model is not perfect, the optimization horizon should not be too long as model inaccuracies will cause erroneous actuation predictions for longer periods of time. However, on the other hand, to develop an optimal control policy, it is important to predict over a longer time horizon in order to better account for the effect of current acctuation on future cte. Moreover, the higher the desired speed of the car, the longer the prediciton horizon needs to be inorder to obtain the best control actions and stability. Hence, there is a tradeoff.

Tuning of N and dt
* dt - I tried dt values of 0.10, 0.12, 0.15 and 0.20. I found at 0.10 or below the car oscillated around the center of the track. And at 0.20 while the motion was smooth, the car rolled off the edge of the track around the curve. The best performance was at dt = 0.15.
* N - Once dt was locked, I tried very high N - 30 and motion of the car was very jerky. The car was not very sensitive to N as long as it was in the range between 10 and 15. I finally chosen N =10.


## 3. Polynomial Fitting and MPC Preprocessing

The x and y coordinates received from the simulator were in the map space. They were converted to vehicle space. Once the coordinates were in the vehicle space, a 3rd degree (cubic) polynomial was used to fit the waypoints data. This polynomial is then used to describe the reference trajectory at each time step over the horizon when performing the optimization. The waypoints were not preprocessed, instead, a polynomial curve is fitted in real time for best performance and robustness.


## 4. Dealing with latency

The simulator has been setup to have a latency of 100ms. This means that the actuator inputs sent to the simulator execute with a delay of 0.1 s. I found that the best approach to dealing with latency is to explicitely account for it in the MPC model. I incorporated latency in the MPC state before passing it to the solver. As a result the solution from the solver - steering and throttle better account for the current state of the vehicle. Also I found that having dt > latency ensured smooth motion in the simulator.

The controller is designed to work well with a latency of 100ms in the actuation commands.
MPC controllers are generally much better in handling latency due to the following two reasons:
	a. The initial state used by the MPC solver is generated by predicting the measured state forward in time (latency time) using the kinematic model of the 		   vehicle and asssuming the previous set of actuations hold over the latency period.
	b. Furthermore, the dt parameter is set to be slightly larger than the latency so that the model predictions implicitly take latency into account.
	   Together with these two approaches, the controller is designed to handle latncy of even hundreds of milli-seconds.

## 5. Compared to PID Control

The vehcle never leaves the drivable portion of the track while moving at a maximum speed of almost 80mph. Which much better than was able to achieve using the PID controller or even using behavioural cloning using CNNs.

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `sudo bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./
