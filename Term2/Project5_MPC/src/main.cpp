#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;
  vector<double> prev_actuation = {0.0,0.0}; //holds previous actuation (throttle,steer_angle)

  h.onMessage([&mpc, &prev_actuation](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];

          /*
          * TODO: Calculate steeering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */

          //AMIT'S CODE
          //convert v from mph to m/s (note v is really speed and not velocity)
          //px and py are in meters so need to convert v in m/s^2
          v = v*(1.61*1000/(3600));
          //throttle is in m/s/s units as it is acceleration from the kinematic equation in the MPC solve function (when v is in m/s).
          //but steering angle is a normalized number. need to convert from radians to normalized number. see below.

          //account for acctuation delay by using the state after tdelay (100mS)
          //previous acctuation value holds during the delay time
          //because the transformation between car and map coordinates is only rotation and translation but not scaling,
          //actuations (throttle and steer angle) will be same in both coordinates
          double tdelay = 0.1; //0.1 seconds is delay
          px = px + v*cos(psi)*tdelay;
          py = py + v*sin(psi)*tdelay;
          psi = psi + v*prev_actuation[1]*deg2rad(25) / Lf * tdelay;
          v = v + prev_actuation[0] * tdelay; //throttle is in m/s^2

          //convert the waypoints into car's coordinate system
          assert(ptsx.size() == ptsy.size());
          Eigen::VectorXd ptsx_cc(ptsx.size()); //waypoints in car coordinate system in vectorXd
          Eigen::VectorXd ptsy_cc(ptsy.size()); //waypoints in car coordinate system in vector Xd

          vector<double> ptsx_cc_vec(ptsx.size()); ////waypoints in car coordinate system in vector for plotting waypoints
          vector<double> ptsy_cc_vec(ptsy.size()); ////waypoints in car coordinate system in vector for plotting waypoints

          for (int i=0; i<ptsx.size(); i++)
          {
            //first translate then rotate
            double ptsx_trans = ptsx[i] - px;
            double ptsy_trans = ptsy[i] - py;
            ptsx_cc[i] = ptsx_trans * cos(psi) + ptsy_trans * sin(psi);
            ptsy_cc[i] = -ptsx_trans * sin(psi) + ptsy_trans * cos(psi);

            ptsx_cc_vec[i] = ptsx_cc[i];
            ptsy_cc_vec[i] = ptsy_cc[i];
          }

          //the car's position in its coordinate system is x=0,y=0,psi=0
          int px_cc = 0;
          int py_cc = 0;
          int psi_cc = 0;
          int v_cc = v; //v stays as is between coordinate transformation due to no scaling, only rotation and translation and v is scalar/magnitude


          //the waypoints (in car coordinates) are fitted with a 2nd order polynomial (try 3rd as suggested in lectures)
          //but how long is ptsx?? needs to be greater than 3
          auto coeffs = polyfit(ptsx_cc, ptsy_cc, 3); //coeffs are in car's coordinate system

          //cte = f(x) - y where f(x) is the desired y location at current x
          double cte = polyeval(coeffs, px_cc) - py_cc; //the car's position in its coordinate system is x=0,y=0,psi=0
          //f(x) = coeffs[0] + coeffs[1] * X + coeffs[2] * X^2 + coeffs[3] * X^3
          //f'(x) = coeffs[1] + 2*coeffs[2] * X + 3*coeffs[3] * X^2
          //epsi = psi - atan(f'(x))
          double epsi = psi_cc - atan(coeffs[1] + 2*coeffs[2] * px_cc + 3*coeffs[3] * px_cc*px_cc); //the car's position in its coordinate system is x=0,y=0,psi=0

          Eigen::VectorXd state(8);
          state << px_cc, py_cc, psi_cc, v_cc, cte, epsi, prev_actuation[1]*deg2rad(25), prev_actuation[0];

          //MPC_Solve_Return vars;
          auto vars = mpc.Solve(state, coeffs); //vars is MPC_Solve_Return struct containing first set of actuations and predicted x/y points
          //state << vars[0], vars[1], vars[2], vars[3], vars[4], vars[5];

          //negative sign cause there is difference in the steering angle between how its used in the model and the simulator
          double steer_value = -vars.steer_1st/deg2rad(25); //need to be normalized. see below
          double throttle_value = vars.throttle_1st;

          prev_actuation[0] = throttle_value;
          prev_actuation[1] = -steer_value; //its -ve of steer_value
          //END AMIT'S CODE


          //std::cout << steer_value << "\t" << throttle_value << "\n";
          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          mpc_x_vals = vars.xpred;
          mpc_y_vals = vars.ypred;


          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;



          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;


          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          next_x_vals = ptsx_cc_vec; //waypoints in car coordinates in vector
          next_y_vals = ptsy_cc_vec; //waypoints in car coordinates in vector


          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;

          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does not actuate the commands instantly.
          //
          // Feel free to play around with this value but should be able to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

        }

      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }

    }

  });


  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });


  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
