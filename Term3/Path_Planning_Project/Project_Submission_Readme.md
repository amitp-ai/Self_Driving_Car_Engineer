Listed below is my response to the requests made in the rubric file. I have updated it per the resubmission request.

## The basic operation of the path planner is as follows:

a. The behavior planner decides which action to take amongt keep lane, lane change left, and lane change right. This is implemented using a finite state machine. The state transition is determined by various cost  functions to pick the behavior that is the safest, smoothest, and fastest. While in an actual Sself-driving car a predcition module (using some machine learning algorithm) will be used the predict the next state of the other vehicles (sensor fusion data), it was not implemented in this project as the Ego vehicle already drives so well that didn't seem necessary.

b. for each state transition, regardless of the cost to go to the next state, the maximum safe acceleration for that state transition is calculated so as the Ego car always maintains a safe buffer distance with the car infront and the closest car behind (for lane changes). This is implemented in the "max_accel_for_keep_lane" and "max_accel_for_lane_change" functions. Using the max. acceleration and the distance and speed of the closest vehicle in front and in the back, the cost of a particular state transition is calculated. This is implemented in the "calculate_cost_keep_lane" and "calculate_cost_lane_change" functions. The decision with the lowest cost is selected. I also keep track of the previous decision inorder to make sure that car doesn't go into an oscillatory behavior by just changing lanes all the time.

c. Once a decision is made, the trajectory generatory actually generates the actual path to be followed. This is implemented in the "generate_Trajectory" fucntion. I do most of the implementation in the Frenet coordinate system and use the getFrenet() and getXY() functions to go between the two coordinate systems. To keep the lane changes smooth, I increase the delta s when changing lanes. However, when maintaining a given lane, the delta s is made a litle smaller (i.e. predcition time of 2seconds versus 3seconds for lane changes) so that the car properly stays in the middle of the lane. In in order to generate a smooth trajectory, I use a spline function with two previous points and 3 points from the future (based upon the selected decision). Moreover, 50 points are sent to the simulator and the points are separated out by the target speed (determined by the state transition function) of the car. The points are separated out using a linear appproximation as has been detailed in the main.cpp file. 

## Rubric Requirements
1. The project can be run using cmake and make as per the instructions provided.

2. The car is able to drive more than 4.32miles without any incident.
I have driven the car multiple laps across the circuit and it works fine.

3. The car never drives faster than the speed limit and tries to go at about the speed limit if the traffic flow allows so.

4. The car does not violate the max. acceleration and max. jerk at all times.

5. The car never collides with any other car.

6. The car never stays outside lane for more than 3 seconds





