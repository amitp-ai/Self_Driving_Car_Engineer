//AMIT PATEL (amitpatel.gt@gmail.com) BOSCH CHALLENGE SUBMISSION
#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h" //added by Amit
#include <map> //added by Amit. It's like dictionary in Python. Use to store cost weights.

using namespace std;

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
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

	return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

struct vehicle_Data
{
    double x, y, yaw, s, d, speed;
    double prev_d = 0.0; //initialized to 0
    int decision_count = 0; //initialized to 0
    string prev_decision = "Invalid";
    int prev_target_lane = 1; //likely initial lane (but doesnt matter)
    string state;
};


struct target_Data
{
    double speed, accel;
    int lane;
    double avg_lane = 1; //initialized to middle lane (but doesn't matter)
    double max_speed = 49.0; //MPH
    double max_accel = 0.448; //MPH per (0.02second). //0.448 MPH/(0.02SEC) = 11MPH/SEC = 10m/s^2. This is acceleration
    int plan_horizon = 50; //number of points to plan ahead every iteration
    double preferred_buffer = 25; //30 meters //0.1*1.61*1000; //in meters (0.1Miles * 1600m/miles)
};

void generate_Trajectory(vector<double> &next_x_vals, vector<double> &next_y_vals, vehicle_Data &ego_car, vector<double> &previous_path_x, vector<double> &previous_path_y, target_Data &ego_target, vector<double> &map_waypoints_x, vector<double> &map_waypoints_y, vector<double> &map_waypoints_s)
{

    int prev_size = previous_path_x.size(); //previous path can be helpful with transitions
    //create a list of widely space (x,y) waypoints, evenly spaced at 30m
    //later will interpolate these waypoints with a spline and fill it in with more points that control speed
    vector<double> ptsx;
    vector<double> ptsy;

    //reference x,y,yaw states
    //either we will reference the starting point as where the car is or at the previous paths end points
    double ref_x = ego_car.x;
    double ref_y = ego_car.y;
    double ref_yaw = deg2rad(ego_car.yaw);

    //if the previous path is almost empty, then use the car as starting reference
    if(prev_size < 2)
    {
        //use two points that make the path tangent to the car
        double prev_car_x = ref_x - cos(ref_yaw); //should it be car_x-v*0.02*cos(car_yaw)??
        double prev_car_y = ref_y - sin(ref_yaw);

        //double prev2_car_x = prev_car_x - cos(ref_yaw); //should it be car_x-v*0.02*cos(car_yaw)??
        //double prev2_car_y = prev_car_y - sin(ref_yaw);

       //ptsx.push_back(prev2_car_x);
        ptsx.push_back(prev_car_x);
        ptsx.push_back(ref_x);

        //ptsy.push_back(prev2_car_y);
        ptsy.push_back(prev_car_y);
        ptsy.push_back(ref_y);
    }

    //otherwise use the previous path's end points as starting reference
    else
    {
        //redefine reference state as previous path end point
        ref_x = previous_path_x[prev_size-1];
        ref_y = previous_path_y[prev_size-1];

        double ref_x_prev = previous_path_x[prev_size-2];
        double ref_y_prev = previous_path_y[prev_size-2];
        ref_yaw = atan2(ref_y-ref_y_prev,ref_x-ref_x_prev);

        //double ref_x_prev_2 = previous_path_x[prev_size-3];
        //double ref_y_prev_2 = previous_path_y[prev_size-3];

        //use the two points that make the path tangent to the previous path's end point
        //ptsx.push_back(ref_x_prev_2);
        ptsx.push_back(ref_x_prev);
        ptsx.push_back(ref_x);

        //ptsy.push_back(ref_y_prev_2);
        ptsy.push_back(ref_y_prev);
        ptsy.push_back(ref_y);
    }

    //smoothing is achieved by using the end points from previous path in addition to future points
    //In Frenet add evenly 30m spaced points ahead of the car's current location. Add three such points.
    double delta_s = ego_car.speed/2.24*2; //2.24 to convert MPH to m/s and 2.5 is seconds.
    if (delta_s < 35)
        delta_s = 35;

    int curr_lane = (int)(ego_car.d/4);
    if(curr_lane != ego_target.lane)
        delta_s *= 1.5;

    ego_target.avg_lane = 0.4*ego_target.lane + 0.6*ego_target.avg_lane; //to smooth out any oscillations during lane changes

    vector<double> next_wp0 = getXY(ego_car.s+delta_s*1, (2+4*ego_target.avg_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
    vector<double> next_wp1 = getXY(ego_car.s+delta_s*2, (2+4*ego_target.avg_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
    vector<double> next_wp2 = getXY(ego_car.s+delta_s*3, (2+4*ego_target.avg_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

    ptsx.push_back(next_wp0[0]);
    ptsx.push_back(next_wp1[0]);
    ptsx.push_back(next_wp2[0]);

    ptsy.push_back(next_wp0[1]);
    ptsy.push_back(next_wp1[1]);
    ptsy.push_back(next_wp2[1]);

    //shift car reference angle to zero degrees.
    //simplifies the Math
    //local transformation so that the car starts at the origin
    for(int i=0; i<ptsx.size();i++)
    {
        double shift_x = ptsx[i] - ref_x;
        double shift_y = ptsy[i] - ref_y;

        ptsx[i] = (shift_x*cos(0-ref_yaw)) - (shift_y*sin(0-ref_yaw));
        ptsy[i] = (shift_x*sin(0-ref_yaw)) + (shift_y*cos(0-ref_yaw));

    }

    //create a spline
    tk::spline s; //this function interpolates properly but extrapolation is linear only

    //set (x,y) points to the spline
    s.set_points(ptsx,ptsy);

    //start with all of the previous path points from last time
    //this helps out with the transition/smoothing
    //instead of recreating the path from scratch every time, we just add points to it from last time
    //and work with what you had left from last time
    for(int i=0; i<previous_path_x.size(); i++)
    {
        next_x_vals.push_back(previous_path_x[i]);
        next_y_vals.push_back(previous_path_y[i]);
    }

    //calculate how to break up spline points so that we travel at our desired reference velocity
    double target_x = 30.0;
    double target_y = s(target_x);
    double target_dist = sqrt(target_x*target_x + target_y*target_y);

    double x_add_on = 0;

    //Fill up the rest of our path planner after filling it with previous points, here we will always output 50 points
    for(int i=1; i<= (ego_target.plan_horizon-previous_path_x.size()); i++)
    {
        double N = target_dist/(0.02*ego_target.speed/2.24); //2.24 is to convert MPH to m/s
        double x_point = x_add_on + target_x/N;
        double y_point = s(x_point);

        x_add_on = x_point;

        double x_ref = x_point;
        double y_ref = y_point;

        //rotate back to normal after rotating/transforming it earlier
        //go from local to global coordinates
        x_point = (x_ref*cos(ref_yaw)-y_ref*sin(ref_yaw));
        y_point = (x_ref*sin(ref_yaw)+y_ref*cos(ref_yaw));

        x_point += ref_x;
        y_point += ref_y;

        next_x_vals.push_back(x_point);
        next_y_vals.push_back(y_point);
    }
}

double validate_max_accel(double &available_acc, vehicle_Data &ego_car, target_Data &ego_target, double &temp_plan_horizon)
{
    //delta_v_till_target in the temp_plan_horizon time frame
    double delta_v_till_target = (ego_target.max_speed - ego_car.speed)/temp_plan_horizon*0.02; //in MPH per 0.02Sec
    double max_acc; //in MPH/0.02sec
    if(available_acc < delta_v_till_target)
        max_acc = available_acc;
    else
        max_acc = delta_v_till_target;

    if(max_acc < 0)
    {
        double min_acc_allowed = 0.2*ego_target.max_accel;
        max_acc = max_acc * (1 + 2*exp(-max_acc - min_acc_allowed)); //smooth function to minimize jerk
        if(max_acc < -3*ego_target.max_accel)
        {
            max_acc = -3*ego_target.max_accel;
            //cout << "MAX BRAKE\n\n";
        }
    }
    else if(max_acc > 0)
    {
        max_acc *= 2; //effectively reduces planning horizon by sqrt(3) while maintaining the benefit of longer horizon (i.e more realistic and smoother acceleration)
        if(max_acc > 2*ego_target.max_accel)
        {
            max_acc = 2*ego_target.max_accel;
        }

    }

    return max_acc;
}

double validate_min_accel(double &available_acc, target_Data &ego_target)
{
    double min_acc = available_acc;
    if(min_acc <= 0)
        min_acc = 0.0; //lowest possible min_accel in forward direction
    else //i.e. if greater than 0
    {
        min_acc *= 2; //to keep it same as validate_ax_acc function
        if(min_acc > 2*ego_target.max_accel)
            min_acc = 2*ego_target.max_accel;
    }

    return min_acc;
}

vector<double> max_accel_for_keep_lane(vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion)
{
    //returns max acceleration in MPH per 0.02 second
    //also returns the distance to the closest vehicle in front
    int closest_vehicle_infront = 0;
    double dist_to_closest_vehicle_infront = 10000; //some very large number
    for(int i=0; i<sensor_fusion.size(); i++)
    {
        double d = sensor_fusion[i][6];
        if(ego_car.d > (d-2.5) && ego_car.d < (d+2.5))
        {
            if((sensor_fusion[i][5] >= ego_car.s) && ((sensor_fusion[i][5] - ego_car.s) < dist_to_closest_vehicle_infront))
            {
                dist_to_closest_vehicle_infront = sensor_fusion[i][5] - ego_car.s; //in meters
                closest_vehicle_infront = i;
            }
        }
    }

    double temp_plan_horizon = 1.5; //1.75; //2.0; //1.5; //seconds
    double available_acc = ego_target.max_accel; //the largest possible value by default
    double separation_next_front = 10000;
    double closest_car_speed = 50; //max speed in MPH. Needs to be high if dist_to_closest_vehicle_infront is high
    closest_car_speed /= 2.24; //max speed in m/s
    if(dist_to_closest_vehicle_infront < 10000)
    {
        //closest_car_speed is in m/s
        double vx = sensor_fusion[closest_vehicle_infront][3];
        double vy = sensor_fusion[closest_vehicle_infront][4];
        closest_car_speed = sqrt(vx*vx+vy*vy); //in m/s
        double closest_car_next_pos = sensor_fusion[closest_vehicle_infront][5];
        closest_car_next_pos += closest_car_speed * temp_plan_horizon; //this is in meters

        double ego_next_pos = ego_car.s + (ego_car.speed/2.24) * temp_plan_horizon; //ego_car's speed is in MPH so divide by 2.24 to convert to m/s
        separation_next_front = closest_car_next_pos - ego_next_pos; //in meters
        double available_room = separation_next_front - ego_target.preferred_buffer; //in meters
        available_acc = 2*available_room/(temp_plan_horizon*temp_plan_horizon); //in m/s^2 //ut+1/2at^2; //the ut component has already been addressed in speration next
        //convert available_acc to MPH per 0.02sec
        available_acc *= 2.24; // convert m/s/s to MPH/s
        available_acc *= 0.02; //convert MPH/s to MPH/0.02sec
    }
    //no need for else as the initial value of available_acc satisfies else condition

    double max_acc = validate_max_accel(available_acc, ego_car, ego_target, temp_plan_horizon);

    closest_car_speed *= 2.24; //in MPH
    //cout << max_acc << "\t" << ego_car.speed << "\t" << dist_to_closest_vehicle_infront << "\n" ;
    return {max_acc, separation_next_front, closest_car_speed}; //max_acc is in MPH per 0.02sec
}


vector<vector<double>> max_accel_for_lane_change(vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion)
{
    //returns max acceleration in MPH per 0.02 second and min accelaration in MPH per 0.02sec
    //also returns the distance to the closest vehicle in the front and back of the target lane from the EGO Car
    int closest_vehicle_infront = 0;
    double dist_to_closest_vehicle_infront = 10000; //some very large number
    int closest_vehicle_back = 0;
    double dist_to_closest_vehicle_back = 10000; //some very large number
    for(int i=0; i<sensor_fusion.size(); i++)
    {
        double d = sensor_fusion[i][6];
        if(ego_car.d > (d-2.5) && ego_car.d < (d+2.5))
        {
            //closest front vehicle
            if((sensor_fusion[i][5] >= ego_car.s) && ((sensor_fusion[i][5] - ego_car.s) < dist_to_closest_vehicle_infront))
            {
                dist_to_closest_vehicle_infront = sensor_fusion[i][5] - ego_car.s; //in meters
                closest_vehicle_infront = i;
            }
            //closest back vehicle
            if((ego_car.s > sensor_fusion[i][5]) && ((ego_car.s - sensor_fusion[i][5]) < dist_to_closest_vehicle_back))
            {
                dist_to_closest_vehicle_back = ego_car.s - sensor_fusion[i][5]; //in meters
                closest_vehicle_back = i;
            }
        }
    }

    double temp_plan_horizon = 1.5; //1.75; //2.0; //1.5; //seconds
    double available_acc = ego_target.max_accel; //the largest possible value by default
    double separation_next_front = 10000;
    double closest_front_car_speed = 50; //max speed in MPH. Needs to be high if dist_to_closest_vehicle_infront is large
    closest_front_car_speed /= 2.24; //max speed in m/s

    if(dist_to_closest_vehicle_infront < 10000)
    {
        //closest_front_car_speed is in m/s
        double vx = sensor_fusion[closest_vehicle_infront][3];
        double vy = sensor_fusion[closest_vehicle_infront][4];
        closest_front_car_speed = sqrt(vx*vx+vy*vy); //in m/s
        double closest_car_next_pos = sensor_fusion[closest_vehicle_infront][5];
        closest_car_next_pos += closest_front_car_speed * temp_plan_horizon; //this is in meters

        double ego_next_pos = ego_car.s + (ego_car.speed/2.24) * temp_plan_horizon; //ego_car's speed is in MPH so divide by 2.24 to convert to m/s
        separation_next_front = closest_car_next_pos - ego_next_pos; //in meters
        double available_room = separation_next_front - ego_target.preferred_buffer; //in meters
        available_acc = 2*available_room/(temp_plan_horizon*temp_plan_horizon); //in m/s^2 //ut+1/2at^2; //the ut component has already been addressed in speration next
        //convert available_acc to MPH per 0.02sec
        available_acc *= 2.24; // convert m/s/s to MPH/s
        available_acc *= 0.02; //convert MPH/s to MPH/0.02sec
    }
    //no need for else as the initial value of available_acc satisfies else condition

    double max_acc = validate_max_accel(available_acc, ego_car, ego_target, temp_plan_horizon);


    //find the min. acceleration
    double back_temp_plan_horizon = 1.5; //1.75; //1.5; //this can be on a shorter time frame than front planning
    double back_preffered_buffer = ego_target.preferred_buffer*1.0; //can be a bit more aggressive on buffer than for front gap/buffer
    available_acc = 0.0; //in MPH/0.02sec //lowest possible forward acceleration
    double separation_next_back = 10000;
    double closest_backcar_speed = 0.0; //for back car this is 0 when dist_to_closest_vehicle_back is very large
    if(dist_to_closest_vehicle_back < 10000)
    {
        //closest_backcar_speed is in m/s
        double vx = sensor_fusion[closest_vehicle_back][3];
        double vy = sensor_fusion[closest_vehicle_back][4];
        closest_backcar_speed = sqrt(vx*vx+vy*vy); //in m/s
        double closest_car_next_pos = sensor_fusion[closest_vehicle_back][5];
        closest_car_next_pos += closest_backcar_speed * back_temp_plan_horizon; //this is in meters

        double ego_next_pos = ego_car.s + (ego_car.speed/2.24) * back_temp_plan_horizon; //ego_car's speed is in MPH so divide by 2.24 to convert to m/s
        separation_next_back = ego_next_pos - closest_car_next_pos; //in meters
        double available_room = back_preffered_buffer - separation_next_back; //in meters
        //if separation_next is much larger than back_preffered_buffer, then available_acc will be a very large negative number
        //if separation_next is equal to back_preffered_buffer, then available_acc will be 0
        //if separation_next is less than back_preffered_buffer, then available_acc will be a positive number
        available_acc = 2*available_room/(back_temp_plan_horizon*back_temp_plan_horizon); //in m/s^2 //ut+1/2at^2; //the ut component has already been addressed in speration next
        //convert available_acc to MPH per 0.02sec
        available_acc *= 2.24; // convert m/s/s to MPH/s
        available_acc *= 0.02; //convert MPH/s to MPH/0.02sec
    }

    //no need for else as the initial value of available_acc satisfies else condition

    double min_acc = validate_min_accel(available_acc, ego_target);

    //cout << max_acc << "\t" << dist_to_closest_vehicle_infront <<"\t" << min_acc << "\t" << dist_to_closest_vehicle_back << "\n";

    closest_front_car_speed *= 2.24; //in MPH
    closest_backcar_speed *= 2.24; //in MPH
    vector<vector<double>> rtn_var;
    rtn_var.push_back({max_acc, separation_next_front, closest_front_car_speed});
    rtn_var.push_back({min_acc, separation_next_back, closest_backcar_speed});

    //min_acc & max_acc are in MPH per 0.02sec and dist_to_closest_vehicle_back & dist_to_closest_vehicle_infront are in meters (Frenet s axis)
    return rtn_var;
}

void ego_target_speed_validation(target_Data &ego_target)
{
    if(ego_target.speed < 1)
    {
        ego_target.speed = 5.0; //spline function complains if this gets below 0 as it wants the points to be sorted in ascending order. so keep slightly above 0
        ego_target.accel = 0.001;
    }
    else if(ego_target.speed > ego_target.max_speed)
    {
        ego_target.speed = ego_target.max_speed;
        ego_target.accel = 0.0;
    }
}


//Global constant: cost function weights
struct cf_weights
{
    double collision = 6e4; //due to how the exponential is implemented, this is highest cost
    //double Danger = 2e3; //buffer check
    double closest_car = 5e9; //2e3;
    double reach_goal = 1e2; //reach max velocity
    double comfort = 8e4; //4e3; //penalizes lane changes. Otherwise the car can just keep changing lanes. See the way target_lane is calculated below.
};

cf_weights cost_func_weights;

double calculate_cost_keep_lane(vehicle_Data &ego_car, target_Data &ego_target, double &min_separation_dist_front_next, double &closest_car_speed)
{
    //cout << cost_func_weights["collision cost"] << "\t" << cost_func_weights["buffer cost"] << "\n";

    //Note: all the costs are from the standpoint of staying in lane or changing lanes, wont help much interms of speed in a given lane. That's handled in the max_accel_in_lane function
    //Collision and Danger Cost
    double collision_cost;
    //if(min_separation_dist_front < 40)
    //    collision_cost = exp((10-min_separation_dist_front));
    //else
    //    collision_cost = exp(-min_separation_dist_front*3);
    double front_car_speed_factor = exp(2*(ego_car.speed/closest_car_speed - 1));
    if(front_car_speed_factor < 0.7)
        front_car_speed_factor = 0.7;
    else if(front_car_speed_factor > 1.3)
        front_car_speed_factor = 1.3;

    front_car_speed_factor = 1; //ignore it
    double pred_time = 0; //seconds
    double min_separation_dist_front_next_next = min_separation_dist_front_next + (closest_car_speed-ego_car.speed)/2.24*pred_time;
    collision_cost = exp((front_car_speed_factor*25-min_separation_dist_front_next_next/2)); //separation distance is in meters
    //min_separation_dist_front_next/2 effectively divides the prediction horizon time by 2

    //distance to closest vehicle cost
    double closest_car_cost = (100-min_separation_dist_front_next)/100; //check 100m out
    if(closest_car_cost < 0)
        closest_car_cost = 0;

    //Reach Goal Cost
    //this is still necessary even though max_accel_in_lane takes care of part of it. This helps pick the decision which gets us closest to goal.
    double reach_goal_cost = exp((ego_target.max_speed - (ego_car.speed+ego_target.accel*50))*0.25) - 1; //50 = acceleration over 1 seconds

    //Comfort Cost
    double comfort_cost = 1*(1 - exp(-2*abs(ego_car.d - (ego_target.lane*4+2))));
    comfort_cost += 1*(1 - exp(-2*abs(ego_car.d - ego_car.prev_d))); //prev_d is updated in behavior planner
    //if((int)ego_car.prev_d/4 == ego_target.lane) //only do this for lane_change cost not for keep lane cost
    //    comfort_cost *= 50;
    if(ego_car.prev_target_lane != ego_target.lane)
        comfort_cost *= 1; //don't do this for KL

    //total cost
    double tot_cost = cost_func_weights.collision*collision_cost + cost_func_weights.reach_goal*reach_goal_cost + cost_func_weights.comfort*comfort_cost;
    tot_cost += cost_func_weights.closest_car*closest_car_cost;

    return tot_cost;

}

double realize_keep_lane(vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion)
{
    ego_target.lane = (int)(ego_car.d/4); //keep current lane

    vector<double> temp_vec = max_accel_for_keep_lane(ego_car, ego_target, sensor_fusion);
    ego_target.accel = temp_vec[0];

    ego_target.speed += ego_target.accel;
    ego_target_speed_validation(ego_target);

    double min_separation_dist_next = temp_vec[1]; //this is in meters (distance along Frenet s axis)
    double closest_car_speed = temp_vec[2]; //MPH Not being used at the moment
    double tot_cost = calculate_cost_keep_lane(ego_car, ego_target, min_separation_dist_next, closest_car_speed);

    return tot_cost;
}

double calculate_cost_lane_change(vector<double> &output_KL, vector<vector<double>> &output_LC, vehicle_Data &ego_car, target_Data &ego_target)
{
    //cout << cost_func_weights["collision cost"] << "\t" << cost_func_weights["buffer cost"] << "\n";

    //Note: all the costs are from the standpoint of staying in lane or changing lanes, wont help much interms of speed in a given lane. That's handled in the max_accel_in_lane function
    //Collision and Danger Cost

    double KL_max_acc = output_KL[0];
    double KL_dist_closest_front_next = output_KL[1];

    double LC_max_acc = output_LC[0][0];
    double LC_dist_closest_front_next = output_LC[0][1];
    double LC_dist_closest_frontcar_speed = output_LC[0][2];
    double LC_min_acc = output_LC[1][0];
    double LC_dist_closest_back_next = output_LC[1][1];
    double LC_dist_closest_backcar_speed = output_LC[1][2];

    double collision_cost=0;
    //for back
    //if(LC_min_acc > KL_max_acc) //check if I should use this instead of distance
//    if(LC_dist_closest_back < 10)
//        collision_cost = exp((5-LC_dist_closest_back)*1);
//    else
//        collision_cost = exp(-LC_dist_closest_back*3);

    //for front
//    if(LC_dist_closest_front < 20)
//        collision_cost = collision_cost + exp((5-LC_dist_closest_front)*1);
//    else
//        collision_cost = collision_cost + exp(-LC_dist_closest_front*3);
    double back_car_speed_factor = exp(1*(LC_dist_closest_backcar_speed/ego_car.speed - 1));
    if(back_car_speed_factor < 0.9)
        back_car_speed_factor = 0.9;
    else if(back_car_speed_factor > 1.7)
        back_car_speed_factor = 1.7;

    back_car_speed_factor = 1; //ignore it
    double pred_time = 0; //seconds
    double back_car_dist_next = LC_dist_closest_back_next + (ego_car.speed-LC_dist_closest_backcar_speed)/2.24*pred_time;
    collision_cost += exp(1.0*(back_car_speed_factor*10-back_car_dist_next/1)); //min separation is in meters

    double front_car_speed_factor = exp(2*(ego_car.speed/LC_dist_closest_frontcar_speed - 1));
    if(front_car_speed_factor < 0.7)
        front_car_speed_factor = 0.7;
    else if(front_car_speed_factor > 1.3)
        front_car_speed_factor = 1.3;

    front_car_speed_factor = 1; //ignore it
    //for comfort even when collision cost it high
    //if KL and LC have high total cost (e.g. near other cars) compared to comfort cost, then still we want to penalize the LC decision.
    //Otherwise it won't be penalized. use 25 (instead of 15 for keep lane)
    double front_car_dist_next = LC_dist_closest_front_next + (LC_dist_closest_frontcar_speed-ego_car.speed)/2.24*pred_time;
    collision_cost += exp((front_car_speed_factor*25*1.15-front_car_dist_next/2)); //min separation is in meters
    //collision_cost *= 1.5; //penalty for changing lane
    //LC_dist_closest_front_next/2 effectively divides the prediction horizon time by 2

    //distance to closest vehicle cost
    double closest_car_cost = (100-LC_dist_closest_front_next)/100; //check 100m out
    if(closest_car_cost < 0)
        closest_car_cost = 0;

   //Reach Goal Cost
    //this is still necessary even though max_accel_in_lane takes care of part of it. This helps pick the decision which gets us closest to goal.
    double reach_goal_cost = exp((ego_target.max_speed - (ego_car.speed+LC_max_acc*50))*0.25) - 1; //50 = acceleration over 1 seconds (50*0.02)

    //Comfort Cost
    double comfort_cost = 1*(1 - exp(-2*abs(ego_car.d - (ego_target.lane*4+2))));
    comfort_cost += 1*(1 - exp(-2*abs(ego_car.d - ego_car.prev_d))); //prev_d is updated in behavior planner
    if((int)ego_car.prev_d/4 == ego_target.lane)
        comfort_cost *= 1e5; //1e4; //50; //penalize oscillatory behavior
    if(ego_car.prev_target_lane != ego_target.lane) //penalize varying target lane. Even for lane change, the target lane should not change.
        comfort_cost *= 1e4;

    //total cost
    double LC_cost = cost_func_weights.collision*collision_cost + cost_func_weights.reach_goal*reach_goal_cost + cost_func_weights.comfort*comfort_cost;
    LC_cost += cost_func_weights.closest_car*closest_car_cost;

    return LC_cost;

}

double realize_lane_change(vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion, string str_turn)
{
    //this function updates the target lane and target acceleration (in MPH/0.02sec)
    int delta = 1;
    if(str_turn == "L")
    	delta = -1;

    //keep current lane
    vehicle_Data temp_ego_car = ego_car;
    target_Data temp_ego_target = ego_target;
    vector<double> output_KL = max_accel_for_keep_lane(temp_ego_car, temp_ego_target, sensor_fusion);

    //change lanes
    //reinitialize these variables;
    temp_ego_car = ego_car;
    temp_ego_target = ego_target;

    int temp_lane = (int)(ego_car.d/4) + delta;
    temp_ego_car.d = (double) (temp_lane*4 + 2); //teleport the car into the target lane, with everything else staying as is.
    //temp_ego_car.d = ego_car.d + delta*4; //TRY THIS TOO AFTERWARDS

    vector<vector<double>> output_LC = max_accel_for_lane_change(temp_ego_car, temp_ego_target, sensor_fusion);

    //update target lane
    ego_target.lane = (int)(ego_car.d/4) + delta; //change lane

    //update accel/speed
    double LC_cost = 0.0;
    ego_target.accel = output_KL[0]; //i.e. slow down so as not to collide with car infront of EGO in current lane before changing lanes
    //cout << "Min LC dist(m): " << output_LC[1][0]*50/2.24*1*1 + ego_car.speed/2.24*1 << "\t" << output_KL[1] << endl;
    //min distance travelled in KL state using minimum accel for 1 second
    double temp_horizon = 1.0; //0.1 sec
    double min_dist = output_LC[1][0]*50/2.24*temp_horizon*temp_horizon + ego_car.speed/2.24*temp_horizon; //in meters
    if(ego_target.accel < output_LC[1][0]) //maintain min. accel to keep space with the behind car in target lane
    {
        if(min_dist < (output_KL[1]+output_KL[2]*temp_horizon-13))  //(w/ 13m margin) if min_dist is less than distance to car infront in KL state
        {
            ego_target.accel = 1.0*output_LC[1][0] + 0.0*output_KL[0]; //maintain min acceleration for the target lane
        }
        else
        {
            LC_cost = 1e26; //don't change lane
            //keep accel from output_KL so as not to collide car infront
        }
    }
    if(ego_target.accel > output_LC[0][0]) //don't accelerate more than feasible based upon car in front in the target lane
        ego_target.accel = output_LC[0][0]; //don't accelerate more than feasible based upon car in front in the target lane

    ego_target.speed += ego_target.accel;
    ego_target_speed_validation(ego_target);

    //make sure there is enough distance to back car
    //LC_min_acc = output_LC[1][0];
    //LC_dist_closest_back_next = output_LC[1][1];
    //LC_dist_closest_backcar_speed = output_LC[1][2];
    temp_horizon = 0.25;
    double ego_pos = ego_target.accel*50/2.24*temp_horizon*temp_horizon + ego_car.speed/2.24*temp_horizon;
    ego_pos += output_LC[1][1]; //ego position relative to closest back car
    double back_car_pos = output_LC[1][2]/2.24*temp_horizon;
    if((ego_pos-back_car_pos) < 10)
    {
        LC_cost = 1e26;
        //cout << "Close Car Behind\n\n";
    }

    //calculate the cost of lane change
    LC_cost += calculate_cost_lane_change(output_KL, output_LC, ego_car, ego_target);

    return LC_cost;

}

double realize_prep_far_lane_change(vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion, string str_turn)
{
    //this function updates the target lane and target acceleration (in MPH/0.02sec)

    //The Behavior Planner is implemented such that this function will only be called if current lane is 0 or 2
    int delta_first_turn = -1; //for PLCFL, first turn left
    string str_second_turn = "L"; //for the second left turn
    if(str_turn == "FR")
    {
        delta_first_turn = 1; //for PLCFR, first turn right
        str_second_turn = "R"; //for the second right turn
    }

//********************Calculate Cost of going to far left/right lane********************************
    //reinitialize these variables and calculate cost going in the far left or far right lane
    vehicle_Data temp_ego_car = ego_car;
    target_Data temp_ego_target = ego_target;
    int first_turn_lane = (int)(temp_ego_car.d/4) + delta_first_turn;
    temp_ego_car.d = (double) (first_turn_lane*4 + 2); //teleport the car into the first target lane (for first turn), with everything else staying as is.
    //temp_ego_car.d = ego_car.d + delta*4; //TRY THIS TOO AFTERWARDS

    //calculate cost going in the far left or far right lane (i.e. another turn in addition to first_turn_lane)
    double cost_PLCF = realize_lane_change(temp_ego_car, temp_ego_target, sensor_fusion, str_second_turn);

    //increase cost if the car is in the center lane (just in case during transition the car is in middle lane and still in PLCFR/L
    int curr_lane = (int)(ego_car.d/4);
    if (curr_lane == 1)
    {
        cost_PLCF *= 1e5; //dont do PLCF if in middle lane
        //cout << "PLCF in Middle Lane\n"; //just for debug as it should not come to this place ever
    }
//****************************************************

//********************update state to prepare to change to adjacent lane********************************

    //keep current lane
    temp_ego_car = ego_car;
    temp_ego_target = ego_target;
    vector<double> output_KL = max_accel_for_keep_lane(temp_ego_car, temp_ego_target, sensor_fusion);

    //change lanes
    //reinitialize these variables;
    temp_ego_car = ego_car;
    temp_ego_target = ego_target;

    first_turn_lane = (int)(temp_ego_car.d/4) + delta_first_turn;
    temp_ego_car.d = (double) (first_turn_lane*4 + 2); //teleport the car into the first target lane (for first turn), with everything else staying as is.
    //temp_ego_car.d = ego_car.d + delta*4; //TRY THIS TOO AFTERWARDS
    vector<vector<double>> output_LC = max_accel_for_lane_change(temp_ego_car, temp_ego_target, sensor_fusion);

    //update target lane and speed/accel (to stay in current lane but prepare to change lane)
    ego_target.lane = (int)(ego_car.d/4); //maintain current lane

    //update accel/speed
    double LC_cost = 0.0;
    ego_target.accel = output_KL[0]; //i.e. slow down so as not to collide with car infront of EGO in current lane before changing lanes
    //cout << "Min LC dist(m): " << output_LC[1][0]*50/2.24*1*1 + ego_car.speed/2.24*1 << "\t" << output_KL[1] << endl;
    //min distance travelled in KL state using minimum accel for 1 second
    double temp_horizon = 1.0; //0.1 sec
    double min_dist = output_LC[1][0]*50/2.24*temp_horizon*temp_horizon + ego_car.speed/2.24*temp_horizon; //in meters
    if(ego_target.accel < output_LC[1][0]) //maintain min. accel to keep space with the front car in target lane
    {
        if(min_dist < (output_KL[1]+output_KL[2]*temp_horizon-25))  //(w/ 13m margin) if min_dist is less than distance to car infront in KL state
        {
            ego_target.accel = 1.0*output_LC[1][0] + 0.0*output_KL[0]; //maintain min acceleration for the target lane
        }
        else
        {
            LC_cost = 1e26; //don't change lane
            //keep accel from output_KL so as not to collide car infront
        }
    }
    if(ego_target.accel > output_LC[0][0]) //don't accelerate more than feasible based upon car in front in the target lane
        ego_target.accel = output_LC[0][0]; //don't accelerate more than feasible based upon car in front in the target lane

    ego_target.speed += ego_target.accel;
    ego_target_speed_validation(ego_target);

    //make sure there is enough distance to back car
    //LC_min_acc = output_LC[1][0];
    //LC_dist_closest_back_next = output_LC[1][1];
    //LC_dist_closest_backcar_speed = output_LC[1][2];
    temp_horizon = 0.25;
    double ego_pos = ego_target.accel*50/2.24*temp_horizon*temp_horizon + ego_car.speed/2.24*temp_horizon;
    ego_pos += output_LC[1][1]; //ego position relative to closest back car
    double back_car_pos = output_LC[1][2]/2.24*temp_horizon;
    if((ego_pos-back_car_pos) < 10)
    {
        LC_cost = 1e26;
        //cout << "PLCF Close Car Behind\n\n";
    }

    //end update target lane and speed/accel
//****************************************************
    return 1e26; //cost_PLCF;

}


void maintain_KL_State(vehicle_Data &ego_car, target_Data &ego_target, target_Data &KL_target)
{
    ego_target = KL_target;
    ego_car.state = "KL";
    ego_car.prev_d = ego_car.d;
}

int PLCF_counter = 0;
int PLCF_lane = 99; //an invalid lane

target_Data prev_ego_target;

void behavior_Planner(vehicle_Data &ego_car, vector<vector<double>> &sensor_fusion, target_Data &ego_target)
{
    //figure out the list of possible next states
    //don't allow for lane change if the resulting lane is less than 0 or more than 2. Could handle this in coft function too, but it's unnecessary as the cost of such a lane change will be extremely high (same as collision)
    int temp_ego_lane = (int)(ego_car.d/4);

    vector<string> possible_states;
    possible_states .push_back("KL");
    if (ego_car.state == "KL")
    {
        if(temp_ego_lane > 0)
        {
            possible_states.push_back("LCL");
        }
        else //lane=0
        {
            possible_states.push_back("PLCFR"); //prep lane change far right
        }

        if(temp_ego_lane < 2)
        {
            possible_states.push_back("LCR");
        }
        else //lane=2
        {
            possible_states.push_back("PLCFL");
        }
    }
    else if (ego_car.state == "LCL")
    {
        if(temp_ego_lane > 0)
        {
            possible_states.push_back("LCL");
        }
    }
    else if (ego_car.state == "LCR")
    {
        if(temp_ego_lane < 2)
        {
            possible_states.push_back("LCR");
        }
    }
    else if (ego_car.state == "PLCFL")
    {
        if(temp_ego_lane == 2) //only go to the below states if lane is far right, else only go to KL state
        {
            possible_states.push_back("PLCFL");
            possible_states.push_back("LCL");

        }
    }
    else if (ego_car.state == "PLCFR")
    {
        if(temp_ego_lane == 0) //only go to the below states if lane is far left, else only go to KL state
        {
            possible_states.push_back("PLCFR");
            possible_states.push_back("LCR");
        }
    }
    else
        cout << "Invalid State\n";

    //possible_states = {"KL"};

    //vectors of cost and target_states for possible states
    vector<double> cost_possible_states;
    vector<target_Data> target_data_possible_states;
    double cost_ith_traj;
    target_Data KL_target;
    for(int i=0; i<possible_states.size(); i++)
    {
        string temp_state = possible_states[i];
        vector<vector<double>> temp_sensor_fusion = sensor_fusion;
        vehicle_Data temp_ego_car = ego_car;
        target_Data temp_ego_target = ego_target;

        if (temp_state == "KL")
        {
            cost_ith_traj = realize_keep_lane(temp_ego_car, temp_ego_target, temp_sensor_fusion);
            KL_target = temp_ego_target;
            if(temp_ego_target.lane == PLCF_lane)
                cost_ith_traj = 1e26; //very large number so as not to go in KL if in PLCF
            //cout << "KL: " << cost_ith_traj << "\n";
        }
        else if (temp_state == "LCL")
        {
            cost_ith_traj = realize_lane_change(temp_ego_car, temp_ego_target, temp_sensor_fusion, "L");
            //cout << "LCL: " << cost_ith_traj << "\n";
        }
        else if (temp_state == "LCR")
        {
            cost_ith_traj = realize_lane_change(temp_ego_car, temp_ego_target, temp_sensor_fusion, "R");
            //cout << "LCR: " << cost_ith_traj << "\n";
        }
        else if (temp_state == "PLCFL")
        {
            cost_ith_traj = realize_prep_far_lane_change(temp_ego_car, temp_ego_target, temp_sensor_fusion, "FL");
            //cout << "PLCFL: " << cost_ith_traj << "\n";
        }
        else if (temp_state == "PLCFR")
        {
            cost_ith_traj = realize_prep_far_lane_change(temp_ego_car, temp_ego_target, temp_sensor_fusion, "FR");
            //cout << "PLCFR: " << cost_ith_traj << "\n";
        }
        else
        {
            cout << "Invalid State: Throw Error!\n";
        }

        cost_possible_states.push_back(cost_ith_traj);
        target_data_possible_states.push_back(temp_ego_target);
    }


    //for finding min. cost path
    double min_cost = 1e26; //a really large number
    string state_min_cost = "UNKNOWN";
    target_Data min_cost_target;
    for(int i=0; i<possible_states.size(); i++)
    {
        if (cost_possible_states[i] < (min_cost))
        {
            min_cost = cost_possible_states[i];
            min_cost_target = target_data_possible_states[i];
            state_min_cost = possible_states[i];
        }
    }

    //For PLCFL or PLCFR states
    double cost_KL, cost_LC, cost_PLCF;
    target_Data target_data_KL, target_data_LC, target_data_PLCF;
    string state_KL, state_LC, state_PLCF;
    if(state_min_cost == "PLCFL" || state_min_cost == "PLCFR")
    {
        for(int i=0; i<possible_states.size(); i++)
        {
            if(possible_states[i] == "PLCFL" || possible_states[i] == "PLCFR")
            {
                cost_PLCF = cost_possible_states[i];
                target_data_PLCF = target_data_possible_states[i];
                state_PLCF = possible_states[i];
            }
            else if(possible_states[i] == "LCL" || possible_states[i] == "LCR")
            {
                cost_LC = cost_possible_states[i];
                target_data_LC = target_data_possible_states[i];
                state_LC = possible_states[i];
            }
            else if(possible_states[i] == "KL")
            {
                cost_KL = cost_possible_states[i];
                target_data_KL = target_data_possible_states[i];
                state_KL = possible_states[i];
            }
            else
                cout << "Invalid State (PLCF Section): Throw Error\n";
        }

        if(cost_KL>1e11 && cost_PLCF<3e9) //&& cost_PLCF<cost_KL/3) // && cost_PLCF<cost_LC/3) //cost_KL > 1e9 && cost_LC > 1e9 &&
        {
            PLCF_counter +=1;
            if(PLCF_counter > 3)
            {
                if(cost_LC < 5e9) //if safe to change lane
                {
                    //then change lane
                    min_cost = cost_LC;
                    min_cost_target = target_data_LC;
                    state_min_cost = state_LC;
                    cout << "\nPLCF to LC: " << PLCF_counter << "\t" << min_cost << "\n";
                    //PLCF_lane will be updated next cycle
                }
                else
                {
                    //prepare for lane change
                    min_cost = cost_PLCF; //this is not necessary as its redundant, but added for code readability
                    min_cost_target = target_data_PLCF; //to adjust speed so the car fits in the gap in the adjacent lane
                    state_min_cost = state_PLCF; //state stays as PLCFL/PLCFR
                    cout << "\nPLCF: " << PLCF_counter << "\t" << min_cost << "\n";

                    PLCF_lane = target_data_PLCF.lane; //so as to ignore KL state when in PLCF (not need for PLCF to LC)
                }

            }
            else
            {
                min_cost = cost_KL;
                min_cost_target = target_data_KL;
                state_min_cost = state_KL;

                //PLCF_counter = 0; //don't reset PLCF
                PLCF_lane = 99; //a non-existent lane number so as not to increase KL cost
                cout << "Get enough PLCF counter: " << PLCF_counter << "\t" << min_cost << "\n";
            }

        }

        else //do PLCF if PLCF_counter > 3
        {
            if(PLCF_counter>3)
            {
                //prepare for lane change
                min_cost = cost_PLCF; //this is not necessary as its redundant, but added for code readability
                min_cost_target = target_data_PLCF; //to adjust speed so the car fits in the gap in the adjacent lane
                state_min_cost = state_PLCF; //state stays as PLCFL/PLCFR

                PLCF_lane = target_data_PLCF.lane; //so as to ignore KL state when in PLCF
                cout << "\n(wait before ignore) PLCF: " << PLCF_counter << "\t" << min_cost << "\n";
            }
            else
            {
                min_cost = cost_KL;
                min_cost_target = target_data_KL;
                state_min_cost = state_KL;

                PLCF_counter = 0; //reset PLCF
                PLCF_lane = 99; //a non-existent lane number so as not to increase KL cost
                cout << "\nIgnore PLCF and do KL: " << "\t" << "Cost: " << min_cost << "\n";
            }

            //PLCF_counter -= 1;
        }
    }


    else //if min cost state is not PLCFL/R then just do what we normally do and reset PLCF
    {
        PLCF_counter = 0;
        PLCF_lane = 99; //a non-existent lane number
        //cout << "\Min Cost is not PLCF: " << PLCF_counter << "\t" << min_cost << "\n";

    }


    //cout << possible_states[traj_min_cost] << "\n\n";
    if(min_cost < 1e25 && ego_car.speed > 25) //1e9) //basically always will come here and don't change lane if driving too slow as it'll time out
    {
        ego_target = min_cost_target;
        ego_car.state = state_min_cost;
        ego_car.prev_d = ego_car.d;

        //ego_car.prev_decision = state_min_cost;
        //ego_car.prev_target_lane = ego_target.lane;
    }

    else
    {
        maintain_KL_State(ego_car, ego_target, KL_target);

        //ego_car.prev_decision = "KL";
        //ego_car.prev_target_lane = ego_target.lane;
    }

    ego_car.prev_target_lane = ego_target.lane;

    cout << "Target: " << ego_target.lane << "\t" << (int)ego_car.d/4 << "\t" << ego_car.state << "\t" << min_cost << "\t\t" << ego_car.d << "\t\t" << ego_car.speed << endl;

}


int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map_bosch1_final.csv"; //Final Bosch Challenge
  //string map_file_ = "../data/highway_map_bosch1.csv"; //Bosch Challenge
  //string map_file_ = "../data/highway_map.csv"; //Path Planning Project
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  vehicle_Data ego_car; //no need to initialize s,d,speed,x,y,yaw as we get them from localization data
  ego_car.state = "KL";

  target_Data ego_target;
  ego_target.speed = 0.0; //MPH
  ego_target.lane = 1; //lane 0 is left most, 1 is middle lane, and 2 is right most

  h.onMessage([&ego_car, &ego_target, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	//auto previous_path_x = j[1]["previous_path_x"]; //can't use auto when passing as argument to generate_Trajectory function
          	//auto previous_path_y = j[1]["previous_path_y"]; //can't use auto when passing as argument to generate_Trajectory function
          	vector<double> previous_path_x = j[1]["previous_path_x"];
          	vector<double> previous_path_y = j[1]["previous_path_y"];

          	// Previous path's end s and d values
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	//auto sensor_fusion = j[1]["sensor_fusion"]; //can't use auto for function call
          	vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

          	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds

          	//Behavior planner to tell us what to do next
          	//tells us about target lane, target velocity, and target acceleration

            int prev_size = previous_path_x.size(); //previous path can be helpful with transitions

            ego_car.s = car_s;
            ego_car.d = car_d;
            ego_car.speed = car_speed; //speed from localization data for EGO is in MPH
            ego_car.x = car_x;
            ego_car.y = car_y;
            ego_car.yaw = car_yaw;

            //ego_car.state/accel are updated in the behavior_Planner function
            behavior_Planner(ego_car, sensor_fusion, ego_target);

            //define the actual (x,y) points we will use for the planner
            vector<double> next_x_vals;
            vector<double> next_y_vals;

            generate_Trajectory(next_x_vals, next_y_vals, ego_car, previous_path_x, previous_path_y, ego_target, map_waypoints_x, map_waypoints_y, map_waypoints_s);

            msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
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
