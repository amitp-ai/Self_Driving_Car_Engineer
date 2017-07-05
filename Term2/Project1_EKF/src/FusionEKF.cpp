#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define PI 3.14159265

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  P_ = MatrixXd(4,4);
  //Initialized to some large value (100) otherwise it might just ignore the measurements
  P_ << 1,0,0,0,
        0,1,0,0,
        0,0,1000,0,
        0,0,0,1000;

  F_ = MatrixXd(4,4);
  //Initialized to zero for now
  //Will be modified in realtime as it is dependent on delta t
  F_ << 1,0,1,0,
         0,1,0,1,
         0,0,1,0,
         0,0,0,1;

  Q_ = MatrixXd(4,4);
  //Initialized to zero for now
  //Will be modified in realtime as it is dependent on delta t
  Q_ << 0,0,0,0,
         0,0,0,0,
         0,0,0,0,
         0,0,0,0;


  H_laser_ << 1,0,0,0,
              0,1,0,0;

  //Initialized to zero for now
  //Will be modified in realtime as it is dependent on the state
  Hj_ << 0,0,0,0,
         0,0,0,0,
         0,0,0,0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    previous_timestamp_ = measurement_pack.timestamp_;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      //Make phi be between -pi and pi. No need for this
      //while (phi/PI > 1) {phi -= 2*PI;}
      //while (phi/PI < -1) {phi += 2*PI;}

      //can get both position and velocity measurements with radar
      ekf_.x_(0) = rho*cos(phi);
      ekf_.x_(1) = rho*sin(phi);
      ekf_.x_(2) = rho_dot*cos(phi); //since rho_dot is the projection of the velocity onto rho
      ekf_.x_(3) = rho_dot*sin(phi); //since rho_dot is the projection of the velocity onto rho

      ekf_.Init(ekf_.x_,P_,F_,Hj_,R_radar_,Q_);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      //can only get posiiton measurements with lidar
      ekf_.x_(0) = measurement_pack.raw_measurements_(0);
      ekf_.x_(1) = measurement_pack.raw_measurements_(1);
      ekf_.x_(2) = 0; //just set to 0
      ekf_.x_(3) = 0; //just set to 0

      ekf_.Init(ekf_.x_,P_,F_,H_laser_,R_laser_,Q_);

    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return; //i.e. first frame results are from the measurements no predict/update steps
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */


  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // TODO: YOUR CODE HERE
  //1. Modify the F matrix so that the time is integrated
  //2. Set the process covariance matrix Q
  //
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  double dt2 = dt*dt;
  double dt3 = dt2*dt;
  double dt4 = dt3*dt;
  //set the acceleration noise components
  float Q_varax_ = 9;
  float Q_varay_ = 9;

  ekf_.Q_ << dt4/4*Q_varax_, 0, dt3/2*Q_varax_, 0,
             0, dt4/4*Q_varay_, 0, dt3/2*Q_varay_,
             dt3/2*Q_varax_, 0, dt2*Q_varax_, 0,
             0, dt3/2*Q_varay_, 0, dt2*Q_varay_;

  ekf_.Predict();



  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    //cout << "Radar\n";
    VectorXd meas_z = VectorXd(3);
    meas_z << measurement_pack.raw_measurements_;
    //Make phi be between -pi and pi (it already is in this range)
    //while (meas_z(1)/PI > 1) {meas_z(1) -= 2*PI;}
    //while (meas_z(1)/PI < -1) {meas_z(1) += 2*PI;}

    ekf_.R_ = R_radar_;

    //calculating Jacobian Hj_ and hx inside updateEKF as it seems more efficient.
    ekf_.UpdateEKF(meas_z);
  } else {
    // Laser updates
    //cout << "Laser\n";
    VectorXd meas_z = VectorXd(2);
    meas_z << measurement_pack.raw_measurements_;
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(meas_z);
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
