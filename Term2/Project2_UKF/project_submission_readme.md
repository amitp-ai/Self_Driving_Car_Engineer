The project can be run using cmake and make as per the instructions provided.

The rmse values are:
Data 1
Accuracy - RMSE:
0.0724186
0.0781114
0.576855
0.568325


Data2
Accuracy - RMSE:
0.165146
0.184243
0.265167
0.281672

Removing the radar data worsens the accuracy of vx and vy and removing the Lidar data worsens the accuracy of px and py (as expected).
The covariance of P is better than the covariance of Radar or Lidar measurements (again as expected if the Kalman filter is working properly).

One thing I found is the filter has some numerical instability/sensitivity to the initial P_ matrix -- which it shouldn't in theory. I think it has to do with computing the square-root when the P_ matrix non positive semi-definite.

To address the instability to some extent, I have put realistic clamps on the speed and yaw rate -- i.e. not allow them to go to infinitely large values. Set practical bounds.

Given some of the model assumptions are based upon small-signal variations, for cases where the delta t is large than 0.1Sec, I make multiple calls to the predict function with smaller time steps. This has helped a little.

Lastly, I check for the size of the P_ matrix and if it goes beyond certain value (i.e. explodes) I reinitialize the state to the current measurement and output a warning message to the user. This only happens for data2 and using the radar data only. In all other cases the P_ matrix never explodes and thus no need for reinitialization.


