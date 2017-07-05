#include "kalman_filter.h"
#include <math.h>
#include <iostream>

#define PI 3.14159265

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
  //cout << x_ << endl;

}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  MatrixXd S = H_*P_*H_.transpose();
  S = S + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  x_ = x_ + K*(z-H_*x_);
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size,x_size);
  P_ = (I-K*H_)*P_;
  //cout << "Lidar:\n" << x_ << endl << endl;


  /*
  //only use measurement data for x,y
  x_(0) = z(0);
  x_(1) = z(1);
  */


}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float dnm = sqrt(px*px+py*py);
    float dnm_s = dnm*dnm;
    float dnm_c = dnm*dnm*dnm;

    H_ = MatrixXd(3,4);
    VectorXd hx = VectorXd(3);
    //MatrixXd Hj_prev = MatrixXd(3,4);
    //VectorXd hx_prev = VectorXd(3);

    //if (fabs(dnm) < 0.001) {cout << "Divide by zero!\n"; dnm = 0.001; dnm_s = 0.0001; dnm_c = 0.00001;}
    if (fabs(dnm) < 0.001)
    {
        //cout << "Divide by zero!\n";
        //skip update all together
        //H_ = Hj_prev;
        //hx = hx_prev;
    }

    else
    {
        //calculate the Jacobian
        H_ << px/dnm, py/dnm, 0, 0,
               -py/dnm_s, px/dnm_s, 0, 0,
               py*(vx*py-vy*px)/dnm_c, px*(vy*px-vx*py)/dnm_c, px/dnm, py/dnm;

        hx << dnm, atan2(py,px), (px*vx+py*vy)/dnm;

        //Hj_prev = H_;
        //hx_prev = hx;


        MatrixXd S = H_*P_*H_.transpose();
        S = S + R_;
        MatrixXd K = P_ * (H_.transpose()) * (S.inverse());

        //cout << P_ << endl;

        x_ = x_ + K*(z-hx);
        int x_size = x_.size();
        MatrixXd I = MatrixXd::Identity(x_size,x_size);
        P_ = (I-(K*H_))*P_;
    }


    /*
    //only use measurement data
    float rho = z(0);
    float phi = z(1);
    float rho_dot = z(2);
    //Make phi be between -pi and pi
    //while (phi/PI > 1) {phi -= 2*PI;}
    //while (phi/PI < -1) {phi += 2*PI;}

    //can get both position and velocity measurements with radar
    x_(0) = rho*cos(phi);
    x_(1) = rho*sin(phi);
    x_(2) = rho_dot*cos(phi); //since rho_dot is the projection of the velocity onto rho
    x_(3) = rho_dot*sin(phi); //since rho_dot is the projection of the velocity onto rho
    */

    //cout << "Radar:\n" << x_ << endl << endl;
    //cout << atan2(py,px) << endl;


}
