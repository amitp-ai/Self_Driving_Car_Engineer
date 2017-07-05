#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;

    p_error = 0.0;
    i_error = 0.0;
    d_error = 0.0;

    step_counter = 0;

    cum_sqrd_cte = 0;
}

void PID::UpdateError(double cte) {

    p_error = cte;
    i_error = 0.90*i_error + 1.3*cte;

    cum_sqrd_cte += cte*cte;

}

double PID::TotalError() {

    double st_angl = -Kp*p_error - Ki*i_error - Kd*(p_error-d_error);
    d_error = p_error; //d_error is prev_cte
	return st_angl;
}

void PID::Init_Twiddle()
{
    counter = 0;
    param = 0; //0=P, 1=I, 2=D
    ops = 1; //+1 is add and -1 is subtract
    Kpar[0] = Kp; Kpar[1] = Ki; Kpar[2] = Kd;
    dKpar[0] = fabs(Kp*0.1); dKpar[1] = fabs(Ki*0.1); dKpar[2] = fabs(Kd*0.1); //set to 10% of initial values
    avg_cte = 0.0;
    best_cte = 1000000; //some really large number
}

void PID::Twiddle(double cte, uWS::WebSocket<uWS::SERVER> &ws)
{
	counter += 1; // update counter
	avg_cte += cte*cte;


	//always add first then subtract (i.e. ops=1 and then ops=-1)
	if (counter > 1500)
    {
		avg_cte /= 1500;

		if (avg_cte < best_cte)
        {
			best_cte = avg_cte;

			if(ops == 1)
            {
				dKpar[param] *= 1.1;
				ops = 1;

				param += 1;
				param = fmod(param,3); //this will cycle param between 0,1,2
				Kpar[param] += dKpar[param]; //to add the new dk
            }

			if(ops == -1)
            {
				dKpar[param] *= 1.1;
				ops = 1;

				param += 1;
				param = fmod(param,3); //this will cycle param between 0,1,2
				Kpar[param] += dKpar[param]; //to add the new dk
            }


			counter=0;
			avg_cte=0;

			//reset car robot location;
            std::string msg = "42[\"reset\",{}]";
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }


		else
        {
			if (ops == 1)
            {
				ops = -1;
				Kpar[param] -= 2*dKpar[param]; //to recover from +dk and then go down dk from 0 delta
            }

			else
            {
				ops=1;
				Kpar[param] += dKpar[param]; //to recover from -dk
				dKpar[param] *= 0.9;

				param += 1;
				param = fmod(param,3); //this will cycle param between 0,1,2
				Kpar[param] += dKpar[param]; //to add the new dk
            }


			counter = 0;
			avg_cte = 0;

			//reset car robot location
            std::string msg = "42[\"reset\",{}]";
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }

        step_counter = 0;
        cum_sqrd_cte = 0.0;
    }

    Kp = Kpar[0]; Ki = Kpar[1]; Kd = Kpar[2]; //update the PID parameter for calculation
}
