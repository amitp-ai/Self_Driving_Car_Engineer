#ifndef PID_H
#define PID_H

#include <math.h>
#include <uWS/uWS.h>

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */
  double Kp;
  double Ki;
  double Kd;

  // Variable added for other purposes
  int step_counter;
  double cum_sqrd_cte;

  /*
  * Variables and Methods for Twiddle
  */
  int counter;
  int param;
  int ops;
  double Kpar[3];
  double dKpar[3];
  double avg_cte;
  double best_cte;

  void Init_Twiddle(void);
  void Twiddle(double cte, uWS::WebSocket<uWS::SERVER> &ws);
  //

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kkp, double Kki, double Kkd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
