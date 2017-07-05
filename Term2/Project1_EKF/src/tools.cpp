#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  VectorXd rmse = VectorXd(4);
  VectorXd residual;
  rmse << 0.0,0.0,0.0,0.0;
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()
          || estimations.size() == 0){
      cout << "Invalid estimation or ground_truth data" << endl;
      return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i; i<estimations.size(); i++)
  {
      residual = estimations[i]-ground_truth[i];
      //cout << "debug\n" << residual << endl;
      residual = residual.array()*residual.array(); //elementwise product
      rmse += residual;
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

VectorXd Tools::Calculate_Percent_Error(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  VectorXd pcent_error = VectorXd(4);
  VectorXd residual;
  pcent_error << 0.0,0.0,0.0,0.0;
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()
          || estimations.size() == 0){
      cout << "Invalid estimation or ground_truth data" << endl;
      return pcent_error;
  }

  //accumulate squared residuals
  for(unsigned int i; i<estimations.size(); i++)
  {
      residual = estimations[i].array()/ground_truth[i].array();
      if (residual(3) == NAN){cout << i << endl;}
      //cout << "debug\n" << residual << endl;
      pcent_error += residual;
  }

  pcent_error /= estimations.size();
  return pcent_error;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  return MatrixXd(2,2); //just as a placeholder remove it later on
}
