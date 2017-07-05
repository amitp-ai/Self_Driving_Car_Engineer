#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  VectorXd rmse = VectorXd(4);
  rmse.fill(0.0);

  //check if the sizes of the two and equal and non-zero
  if((estimations.size() != ground_truth.size()) || estimations.size() == 0)
  {
      std::cout << "Size mismatch in RMSE calculation!\n";
      return rmse;
  }


  for(unsigned int i=0; i<estimations.size(); i++)
  {
      VectorXd diff = estimations[i] - ground_truth[i];
      diff = diff.array() * diff.array();
      rmse += diff;
      //rmse = estimations[i].array() / ground_truth[i].array();
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;

}
