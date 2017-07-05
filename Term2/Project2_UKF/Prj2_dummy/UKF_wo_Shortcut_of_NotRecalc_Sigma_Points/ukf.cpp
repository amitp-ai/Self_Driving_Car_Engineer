#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  //rule of thumb: std_a = a_max/2
  std_a_ = 0.8; //5; //0.8; //2.5; //30; //will adjust this based upon NIS. 5m/s^2 => 18km/hr/s

  // Process noise standard deviation yaw acceleration in rad/s^2
  //rule of thumb: std_yawdd = yawdd_max/2
  std_yawdd_ = 0.6; //1.0; //0.6; //0.05; //30; //will adjust this based upon NIS. 0.1rad/s^2 => 5.7degrees/s/s

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15; //not to be adjusted as it's given by the sensor manufacturer

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15; //not to be adjusted as it's given by the sensor manufacturer

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3; //not to be adjusted as it's given by the sensor manufacturer

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03; //not to be adjusted as it's given by the sensor manufacturer

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3; //not to be adjusted as it's given by the sensor manufacturer

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  //yes the given values for std_a and std_yawdd are wildly off and impractical.
  //So their initial values are adjusted as shown above

  //initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  //time when the state is true, in us (it's really previous measurement's time in us)
  time_us_ = 0.0;

  //State dimension
  n_x_ = 5;

  //Augmented state dimension
  n_aug_ = 7;

  //Sigma point spreading parameter
  lambda_ = 3 - n_aug_; //3 is a design choice

  //predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_,2*n_aug_+1);

  //Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);
  weights_cov_ = VectorXd(2*n_aug_+1);

  //the current NIS for radar
  NIS_radar_ = 0.0;

  //the current NIS for laser
  NIS_laser_ = 0.0;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if(is_initialized_ == false)
  {
      if(meas_package.sensor_type_ == MeasurementPackage::LASER)
      {
          x_(0) = meas_package.raw_measurements_(0); //px
          x_(1) = meas_package.raw_measurements_(1); //py
          //Not directly measured
          x_(2) = 0.0; //v
          x_(3) = 0.0; //yaw
          x_(4) = 0.0; //yaw_dot
      }

      if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
      {
          double rho = meas_package.raw_measurements_(0);
          double phi = meas_package.raw_measurements_(1);
          double rho_dot = meas_package.raw_measurements_(2);
          x_(0) = rho*cos(phi); //px
          x_(1) = rho*sin(phi); //py
          x_(2) = rho_dot; //v
          x_(3) = phi; //yaw. Since phi is the direction of the velocity (rho_dot), it is same as yaw.
          //Not directly measured
          x_(4) = 0.0; //yaw_dot
      }
      time_us_ = meas_package.timestamp_;
      //initialize P to some large value so as to initially give more priority to measurements vs predictions
      P_ << 1.00,0,0,0,0,
            0,1.00,0,0,0,
            0,0,1.00,0,0,
            0,0,0,1.00,0,
            0,0,0,0,0.90;

      is_initialized_ = true;

      //no need to predict or do anything for the first frame
      return;
  }

  else
  {
      double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0; //make sure its 1000000.0 and not 1000000
      time_us_ = meas_package.timestamp_;


      //due to numerical instability in the squareroot function and negative values in P
      //P sometimes explodes in value. So reinitialize everything when that happens.
      if(P_.norm() > 1e+3){cout << "P Matrix is exploding... Re initialize!\n"; is_initialized_ = false;}


      //prediction
      //skip prediction if delta_t is 0
      /*
      if(delta_t > 0.001)
      {
         Prediction(delta_t);
      }
      */

      //Keep delta_t for prediction small as some of the model assumptions are violated for large deviations
      while(delta_t > 0.1)
      {
          Prediction(0.1);
          delta_t -= 0.1;
      }
      Prediction(delta_t);
      //cout << "X\n" << x_ << "\n\n";
      //cout << "P predict\n" << P_.norm() << "\n\n";

      //update
      //use_radar_ = false;
      use_laser_ = false;
      if((meas_package.sensor_type_ == MeasurementPackage::LASER) && (use_laser_ == true))
      {
          UpdateLidar(meas_package);
      }
      if((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_ == true))
      {
          UpdateRadar(meas_package);
          //cout << "P Radar\n" << P_.norm() << "\n\n";
      }
      return; //empty return
  }

}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);


  //Since the process non-linearly interacts with the noise (and not in a additive manner)
  //we create an augmented state (mean and covariance) by including noise parameters in it
  x_aug.fill(0.0); //initialize to 0.0
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0.0; //mean of longitudinal acceleration noise is 0
  x_aug(n_aug_-1) = 0.0; //mean of lateral acceleration noise is 0

  //cout << x_aug << endl << endl;



  P_aug.fill(0.0); //initialize to 0.0


  /*
  //to make sure P_ is always symmetric but doesn't help
  P_ = (P_ + P_.transpose())/2.0;

  //set negative elements in P_aug to 0
  //B'cse P_aug need to be positive semi definite inorder to calculate square root
  //but doesn't help actually
  for(int i=0; i<n_x_; i++)
  {
      for(int j=0; j<n_x_; j++)
      {
          if(P_(i,j) < 0) {P_(i,j) = 0;}
      }
  }
  */

  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_ * std_a_;
  P_aug(n_aug_-1,n_aug_-1) = std_yawdd_ * std_yawdd_;

  MatrixXd Xsig_aug = MatrixXd(n_aug_,2*n_aug_+1);


  //cout << "Paug\n" << P_aug.norm() << "\n\n";

  MatrixXd P_aug_sqrt = P_aug.llt().matrixL(); //square root of the matrix P_aug
  P_aug_sqrt = sqrt(lambda_+n_aug_) * P_aug_sqrt;

  //cout << "Paug sqrt\n" << P_aug_sqrt.norm() << "\n\n";

  //The below for loop does the following:
  //1. Calculate the weights
  //2. Calculate the (augmented) Sigma Points
  //3. Calculate the predicted sigma points in state space
  //4. Calculate the predicted mean in the state space

  //Tacking the special case of 0
  //calculate the weights
  weights_(0) = (float) lambda_/(lambda_+n_aug_); //actually casting not necessary but added for emphasizing the carefulness
  //calculate the sigma points
  Xsig_aug.col(0) = x_aug;
  //calculate the predicted sigmoid points
  motion_predict(Xsig_aug, Xsig_pred_, 0, delta_t);
  //calculate the predicted mean
  x_.fill(0.0); //initialize x_ for calculating the mean
  x_ += weights_(0)*Xsig_pred_.col(0);
  for(int i=0; i<n_aug_; i++)
  {
      //calculate the (augmented) sigmoid points
      Xsig_aug.col(i+1) = x_aug + P_aug_sqrt.col(i);
      Xsig_aug.col(i+n_aug_+1) = x_aug - P_aug_sqrt.col(i);

      //calculate the weights
      weights_(i+1) = 1.0/(2.0*(lambda_+n_aug_));
      weights_(i+n_aug_+1) = 1.0/(2.0*(lambda_+n_aug_));

      //calculate the predicted sigmoid points in state space
      motion_predict(Xsig_aug, Xsig_pred_, i+1, delta_t);
      motion_predict(Xsig_aug, Xsig_pred_, i+n_aug_+1, delta_t);

      //calculate the predicted mean in state space
      //no need for angle normalization for mean. It will be in the correct range since weights sum to 1.
      x_ += weights_(i+1)*Xsig_pred_.col(i+1);
      x_ += weights_(i+n_aug_+1)*Xsig_pred_.col(i+n_aug_+1);

  }

  //Weights for recovering the covariance matrix
  weights_cov_ = weights_;
  weights_cov_(0) = weights_(0); //+ 2.625;

  //cout << weights_ << "\n\n";
  //cout << "X_aug\n" << x_aug.norm() << "\n\n";
  //cout << "Xsig_aug\n" << Xsig_aug.norm() << "\n\n";
  //cout << "Xsig_pred\n" << Xsig_pred_.norm() << "\n\n";

    /*//printing matrix
    std::string sep = "\n----------------------------------------\n";
    //Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
    Eigen::IOFormat CleanFmt(3, 1, ", ", "\n", "[", "]");
    cout << Xsig_aug.format(CleanFmt) << "\n\n" << sep;*/


  //calculate the predicted covariance in state space
  P_.fill(0.0); //initialize to 0.0
  for(int i=0; i<2*n_aug_+1; i++)
  {
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      //angle normalization
      while(x_diff(3) > M_PI) {x_diff(3) -= 2.0*M_PI;}
      while(x_diff(3) < -M_PI) {x_diff(3) += 2.0*M_PI;}

      //limit the max yaw rate to keep predictions realistic
      double max_yawd = 5; //10rad/s = 1.6full turns/sec
      if(x_diff(4) > max_yawd) {x_diff(4) = max_yawd;}
      if(x_diff(4) < -max_yawd) {x_diff(4) = -max_yawd;}

      //limit the max velocity to keep predictions realistic
      double max_v = 10; //25m/s = 90km/hr
      if(x_diff(2) > max_v) {x_diff(2) = max_v;}
      if(x_diff(2) < -max_v) {x_diff(2) = -max_v;}

      P_ += weights_cov_(i)*x_diff*x_diff.transpose();

  }

  recalcSigmaPoints(); //to calculate Xsig2_pred_, weights2_, weights2_cov_

  //done as have calculated Xsig_pred_, x_, and P_.
  //No need to return them as they are class variables

  return;

}

void UKF::recalcSigmaPoints()
{
    Xsig2_pred_ = MatrixXd(n_x_, 2*n_x_+1);
    Xsig2_pred_.fill(0.0);
    weights2_ = VectorXd(2*n_x_+1);
    weights2_.fill(0.0);
    weights2_cov_ = VectorXd(2*n_x_+1);
    weights2_cov_.fill(0.0);

    double lambda2 = 3 - n_x_;
    MatrixXd P_sqrt = P_.llt().matrixL(); //square root of the matrix P_
    P_sqrt = sqrt(n_x_ + lambda2) * P_sqrt;

    Xsig2_pred_.col(0) = x_;
    weights2_(0) = (float) lambda2/(lambda2+n_x_);
    for(int i=0; i<n_x_; i++)
    {
        Xsig2_pred_.col(i+1) = x_ + P_sqrt.col(i);
        Xsig2_pred_.col(i+1+n_x_) = x_ - P_sqrt.col(i);

        weights2_(i+1) = 1.0/(2.0*(n_x_ + lambda2));
        weights2_(i+1+n_x_) = 1.0/(2.0*(n_x_ + lambda2));
    }

    //weights to recover the covariance
    weights2_cov_ = weights2_;
    weights2_cov_(0) = weights2_(0); //+ 4;

    //don't update x_ or P_. They need to be same as from the predictions

}

void UKF::motion_predict(const MatrixXd &Xsig_aug_ref, MatrixXd &Xsig_pred_ref, const int &clmn, const double &delta_t)
{
    double px = Xsig_aug_ref(0,clmn);
    double py = Xsig_aug_ref(1,clmn);
    double v = Xsig_aug_ref(2,clmn);
    double yaw = Xsig_aug_ref(3,clmn);
    double yawd = Xsig_aug_ref(4,clmn);
    double nu_a = Xsig_aug_ref(5,clmn);
    double nu_yawdd = Xsig_aug_ref(6,clmn);

    if(fabs(yawd) < 0.001)
    {
        Xsig_pred_ref(0,clmn) = px + v*cos(yaw)*delta_t;
        Xsig_pred_ref(1,clmn) = py + v*sin(yaw)*delta_t;
    }

    else
    {
        Xsig_pred_ref(0,clmn) = px + v/yawd*(sin(yaw+yawd*delta_t)-sin(yaw));
        Xsig_pred_ref(1,clmn) = py + v/yawd*(cos(yaw)-cos(yaw+yawd*delta_t));
    }

    Xsig_pred_ref(2,clmn) = v + 0;
    Xsig_pred_ref(3,clmn) = yaw + yawd*delta_t;
    Xsig_pred_ref(4,clmn) = yawd + 0;

    //add noise
    Xsig_pred_ref(0,clmn) += 1/2.0*delta_t*delta_t*cos(yaw)*nu_a;
    Xsig_pred_ref(1,clmn) += 1/2.0*delta_t*delta_t*sin(yaw)*nu_a;
    Xsig_pred_ref(2,clmn) += delta_t*nu_a;
    Xsig_pred_ref(3,clmn) += 1/2.0*delta_t*delta_t*nu_yawdd;
    Xsig_pred_ref(4,clmn) += delta_t*nu_yawdd;


    //angle normalization
    while(Xsig_pred_ref(3,clmn) > M_PI) {Xsig_pred_ref(3,clmn) -= 2.0*M_PI;}
    while(Xsig_pred_ref(3,clmn) < -M_PI) {Xsig_pred_ref(3,clmn) += 2.0*M_PI;}

    //limit the max yaw rate to keep predictions realistic
    double max_yawd = 5; //10rad/s = 1.6full turns/sec
    if(Xsig_pred_ref(4,clmn) > max_yawd) {Xsig_pred_ref(4,clmn) = max_yawd;}
    if(Xsig_pred_ref(4,clmn) < -max_yawd) {Xsig_pred_ref(4,clmn) = -max_yawd;}

    //limit the max velocity to keep predictions realistic
    double max_v = 10; //25m/s = 90km/hr
    if(Xsig_pred_ref(2,clmn) > max_v) {Xsig_pred_ref(2,clmn) = max_v;}
    if(Xsig_pred_ref(2,clmn) < -max_v) {Xsig_pred_ref(2,clmn) = -max_v;}

    //cout << clmn << endl << Xsig_pred_ref.col(clmn) << "\n\n";

    //no need to return anything as Xsig_pred_ref is passed by reference
    return;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  int n_lidar = 2;
  MatrixXd Zsig_pred = MatrixXd(n_lidar,2*n_x_+1);
  VectorXd zmean_pred = VectorXd(n_lidar);
  MatrixXd Zcov_pred = MatrixXd(n_lidar,n_lidar);
  MatrixXd R = MatrixXd(n_lidar,n_lidar); //measurement noise covariance matrix
  MatrixXd S = MatrixXd(n_lidar,n_lidar); //Measurement covariance matrix
  MatrixXd T = MatrixXd(n_x_,n_lidar); //cross-correlation matrix
  VectorXd z_innov = VectorXd(n_lidar);

  //initialize
  Zsig_pred.fill(0.0);
  zmean_pred.fill(0.0);

  //we can take a shortcut by reusing Xsig_pred_ (and weights_) instead of calculating sigma points again for the measurement
  //the result will be pretty close. But not doing it to see if it helps with numerical instability.
  //Moreover since the measurement noise is interacting in additive manner, we don't have to augment the state with it
  for(int i=0; i<2*n_x_+1; i++)
  {
      double px = Xsig2_pred_(0,i);
      double py = Xsig2_pred_(1,i);

      //calculate the predicted sigma points in measurement space
      //note: since lidar measurement is a linear process, we could just use matrix operation to compute Zsig_pred
      //but it won't be more efficient as we'd still need a for loop to compute zmean_pred
      Zsig_pred(0,i) = px;
      Zsig_pred(1,i) = py;

      //calculate the predicted mean in measurement space
      //no need for angle normalization as it will already be in the correct range
      zmean_pred += weights2_(i)*Zsig_pred.col(i);
  }

  //cout << Xsig2_pred_ << endl << endl;

  Zcov_pred.fill(0.0);
  T.fill(0.0);
  //calculate the predicted covariance in measurement space
  //and calculate the cross-correlation matrix
  for(int i=0; i<2*n_x_+1; i++)
  {
      //predicted covariance matrix in measurement space
      VectorXd z_diff = Zsig_pred.col(i) - zmean_pred;

      Zcov_pred += weights2_cov_(i) * z_diff * z_diff.transpose();

      //cross correlation matrix
      VectorXd x_diff = Xsig2_pred_.col(i) - x_;
      //angle normalization
      while(x_diff(1) > M_PI) {x_diff(1) -= 2.0*M_PI;}
      while(x_diff(1) < -M_PI) {x_diff(1) += 2.0*M_PI;}

      T += weights2_cov_(i) * x_diff * z_diff.transpose();

      //cout << z_diff << endl << endl;

  }

  //cout << endl << endl << endl << endl;
  //cout << Xsig2_pred_ << endl << endl;

  R.fill(0.0);
  R(0,0) = std_laspx_ * std_laspx_;
  R(1,1) = std_laspy_ * std_laspy_;

  S = Zcov_pred + R;


  z_innov = meas_package.raw_measurements_ - zmean_pred; //innovation

  //cout << "Laser only\n";
  //cout << P_ << "\n\n";
  updateState(T, S, z_innov); //update the state

  //calculate the NIS
  NIS_laser_ = z_innov.transpose() * S.inverse() * z_innov;

  //for debug
  //cout << Zsig_pred << endl;
  //cout << Zmean_pred << endl;
  //cout << S << endl;

  //cout << NIS_radar_ << endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  int n_radar = 3;
  MatrixXd Zsig_pred = MatrixXd(n_radar,2*n_x_+1);
  VectorXd zmean_pred = VectorXd(n_radar);
  MatrixXd Zcov_pred = MatrixXd(n_radar,n_radar);
  MatrixXd R = MatrixXd(n_radar,n_radar); //measurement noise covariance matrix
  MatrixXd S = MatrixXd(n_radar,n_radar); //Measurement covariance matrix
  MatrixXd T = MatrixXd(n_x_,n_radar); //cross-correlation matrix
  VectorXd z_innov = VectorXd(n_radar);

  //initialize
  Zsig_pred.fill(0.0);
  zmean_pred.fill(0.0);

  //we can take a shortcut by reusing Xsig_pred_ (and weights_) instead of calculating sigma points again for the measurement
  //the result will be pretty close. But not doing it to see if it helps with numerical instability.
  //Moreover since the measurement noise is interacting in additive manner, we don't have to augment the state with it
  for(int i=0; i<2*n_x_+1; i++)
  {
      double px = Xsig2_pred_(0,i);
      double py = Xsig2_pred_(1,i);
      double v = Xsig2_pred_(2,i);
      double yaw = Xsig2_pred_(3,i);

      //check to make sure both points are not near zero
      if((fabs(px) < 0.01) && (fabs(py) < 0.01))
      {
          px = 0.01;
          py = 0.01;
      }

      //calculate the predicted sigma points in measurement space
      Zsig_pred(0,i) = sqrt(px*px + py*py);
      Zsig_pred(1,i) = atan2(py,px); //returns angle between -pi and pi (i.e. 4 quadrant)
      Zsig_pred(2,i) = (px*v*cos(yaw) + py*v*sin(yaw))/Zsig_pred(0,i);

      //calculate the predicted mean in measurement space
      //no need for angle normalization as it will already be in the correct range
      zmean_pred += weights2_(i)*Zsig_pred.col(i);
  }

  Zcov_pred.fill(0.0);
  T.fill(0.0);
  //calculate the predicted covariance in measurement space
  //and calculate the cross-correlation matrix
  for(int i=0; i<2*n_x_+1; i++)
  {
      //predicted covariance matrix in measurement space
      VectorXd z_diff = Zsig_pred.col(i) - zmean_pred;
      //angle normalization
      while(z_diff(1) > M_PI) {z_diff(1) -= 2.0*M_PI;}
      while(z_diff(1) < -M_PI) {z_diff(1) += 2.0*M_PI;}

      Zcov_pred += weights2_cov_(i) * z_diff * z_diff.transpose();

      //cross correlation matrix
      VectorXd x_diff = Xsig2_pred_.col(i) - x_;
      //angle normalization
      while(x_diff(1) > M_PI) {x_diff(1) -= 2.0*M_PI;}
      while(x_diff(1) < -M_PI) {x_diff(1) += 2.0*M_PI;}

      //limit the max yaw rate to keep predictions realistic
      double max_yawd = 5; //10rad/s = 1.6full turns/sec
      if(x_diff(4) > max_yawd) {x_diff(4) = max_yawd;}
      if(x_diff(4) < -max_yawd) {x_diff(4) = -max_yawd;}

      //limit the max velocity to keep predictions realistic
      double max_v = 10; //25m/s = 90km/hr
      if(x_diff(2) > max_v) {x_diff(2) = max_v;}
      if(x_diff(2) < -max_v) {x_diff(2) = -max_v;}

      T += weights2_cov_(i) * x_diff * z_diff.transpose();

  }

  //cout << "P\n" << P_ << "\n\n";
  //cout << "Zcov\n" << Zcov_pred << "\n\n";

  R.fill(0.0);
  R(0,0) = std_radr_ * std_radr_;
  R(1,1) = std_radphi_ * std_radphi_;
  R(2,2) = std_radrd_ * std_radrd_;

  S = Zcov_pred + R;


  z_innov = meas_package.raw_measurements_ - zmean_pred; //innovation
  updateState(T, S, z_innov); //update the state

  //calculate the NIS
  NIS_radar_ = z_innov.transpose() * S.inverse() * z_innov;

  //for debug
  //cout << Zsig_pred << endl;
  //cout << Zmean_pred << endl;
  //cout << S << endl;

  //cout << NIS_radar_ << endl;

}

void UKF::updateState(const MatrixXd &T, const MatrixXd &S, const VectorXd &z_innov)
{
    //calculate the Kalman gain
    MatrixXd K = T * S.inverse();

    //mean update in state space
    x_ = x_ + K*z_innov;
    //angle normalization
    while(x_(3) > M_PI) {x_(3) -= 2.0*M_PI;}
    while(x_(3) < -M_PI) {x_(3) += 2.0*M_PI;}

    //covariance update in state space
    P_ = P_ - K*S*K.transpose(); //this line is sensitive to P_ initialization goes into infinite loop

    //for debug only
    //cout << S << "\n\n";
}
