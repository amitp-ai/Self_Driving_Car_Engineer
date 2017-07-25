Implement a Behavior Planner

In this exercise you will implement a behavior planner for highway driving. It will use prediction data to set the state of the ego vehicle to one of 5 values:

    "KL" - Keep Lane
    "LCL" / "LCR"- Lane Change Left / Right
    "PLCL" / "PLCR" - Prepare Lane Change Left / Right

Instructions

    Implement the update_state method in the vehicle.cpp class.
    Hit Test Run and see how your car does! How fast can you get to the goal without colliding?
    
I guess no, and I think its better that way, for me at least, because when there was a solution, I never spend time working on it , maybe because I learn from code faster.

The "solution" consists of the following  steps.

1. define for each current state the possible next_states.
2. for each possible next state create cost functions for
  a. collision b. goal_lane c. acceleration d.. something else...
3. group them together and sort them by cost and select the one with the lowest cost and set this.state = that_state.

If you do it correct then when you run the program you will reach the goal


