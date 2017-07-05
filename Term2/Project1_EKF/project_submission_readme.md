The project can be run using cmake and make as per the instructions provided.

The rmse values are:
Data 1
Accuracy - RMSE:
0.0651648
0.0605379
0.533212
0.544193

Data2
Accuracy - RMSE:
0.185496
0.190302
0.476754
0.804469


Removing the radar data worsens the accuracy of vx and vy and removing the Lidar data worsens the accuracy of px and py (as expected).
The covariance of P is better than the covariance of Radar or Lidar measurements (again as expected if the Kalman filter is working properly).

