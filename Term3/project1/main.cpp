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
    double x, y, yaw, s, d, speed, accel;
    string state;
};

struct target_Data
{
    double speed, accel;
    int lane;
    double max_speed = 49.0; //MPH
    double max_accel = 0.224*2; //MPH per (0.02second). //0.224 MPH/(0.02SEC) = 11MPH/SEC = 5m/s^2. This is acceleration
    int plan_horizon = 50; //number of points to plan ahead every iteration
    double preferred_buffer = 30; //30 meters //0.1*1.61*1000; //in meters (0.1Miles * 1600m/miles)
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

        ptsx.push_back(prev_car_x);
        ptsx.push_back(ref_x);

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

        //use the two points that make the path tangent to the previous path's end point
        ptsx.push_back(ref_x_prev);
        ptsx.push_back(ref_x);

        ptsy.push_back(ref_y_prev);
        ptsy.push_back(ref_y);
    }

    //smoothing is achieved by using the end points from previous path in addition to future points
    //In Frenet add evenly 30m spaced points ahead of the car's current location. Add three such points.
    vector<double> next_wp0 = getXY(ego_car.s+30, (2+4*ego_target.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
    vector<double> next_wp1 = getXY(ego_car.s+60, (2+4*ego_target.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
    vector<double> next_wp2 = getXY(ego_car.s+90, (2+4*ego_target.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

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
    for(int i=1; i<= ego_target.plan_horizon-previous_path_x.size(); i++)
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


//bool check_collision(vector<double> &temp_ego_now, vector<double> &temp_ego_prev, vector<double> &sf_now, vector<double> &sf_prev)
//{
//   if( (sf_now[1] > (temp_ego_now[1]-2) && sf_now[1] < (temp_ego_now[1]+2)) || (sf_prev[1] > (temp_ego_now[1]-2) && sf_prev[1] < (temp_ego_now[1]+2)) )
//   {
//       if(sf_prev[0] < temp_ego_now[0])
//       {
//           if(sf_now[0] >= temp_ego_now[0])
//               return true;
//           else
//               return false;
//       }
//       if(sf_prev[0] > temp_ego_now[0])
//       {
//           if(sf_now[0] <= temp_ego_now[0])
//               return true;
//           else
//               return false;
//       }
//       if(sf_prev[0] == temp_ego_now[0])
//       {
//           if( (sf_now[0]-sf_prev[0]) < (temp_ego_now[0]-temp_ego_prev[0]) ) //if the speed of sf_car is greater than ego then its ok
//               return true;
//           else
//               return false;
//       }
//   }
//
//   else
//   {
//       cout << "SOMETHING IS WRONG WITH RELEVANT_SENSOR_FUSION IN COST_FUNC_HELPER FUNCTION\n";
//       cout << sf_now[1] << "\t" << temp_ego_now[1] << "\t" << sf_prev[1] << "\n";
//	   return 1.1; //to throw an error
//   }
//}

//vector<double> get_helper_data_for_cost_func(vector<double> &trajectory_x, vector<double> &trajectory_y, vector<vector<double>> &sensor_fusion, vehicle_Data &ego_car, target_Data &ego_target, vector<double> &map_waypoints_x, vector<double> &map_waypoints_y)
//{
//    //returns the distance to the closest approach
//    //returns the time to first collision
//    /////////////////////////////////////////////////////////////////////////////////////////
//    //From the sensorfusion variable, find the vehicles that are closest to EGO (infront and behind) in the current lane and proposed lane.
//    //so at most 4 such vehicles. Only include vehicles whose s distance is within certain threshold from Ego.
//    vector<vector<double>> relevant_sensor_fusion;
//    int temp_ego_lane = (int)(ego_car.d/4);
//    int temp_ego_tgt_lane = ego_target.lane;
//
//    vector<double> lanes_to_check;
//    lanes_to_check.push_back(temp_ego_lane*4.0+2);
//    if(temp_ego_lane != temp_ego_tgt_lane) //that's why kept temp_ego_lane & temp_ego_tgt_lane of type int so its more reliable to do this comparison
//        lanes_to_check.push_back(temp_ego_tgt_lane*4.0+2);
//
//    vector<double> min_dist_front(lanes_to_check.size(), 10000); //initialize to some large value
//    vector<int> idx_min_dist_front(lanes_to_check.size(), 0);
//
//    vector<double> min_dist_back(lanes_to_check.size(), 10000); //initialize to some large value
//    vector<int> idx_min_dist_back(lanes_to_check.size(), 0);
//
//
//    cout << lanes_to_check.size() << "\n\n\n" << endl;
//
//    for(int k=0; k<lanes_to_check.size(); k++)
//    {
//        for(int i=0; i<sensor_fusion.size(); i++)
//        {
//            if((sensor_fusion[i][6] > (lanes_to_check[k]-2)) && (sensor_fusion[i][6] < (lanes_to_check[k]+2)))
//            {
//                //front check
//                if( ((sensor_fusion[i][5] - ego_car.s) >= 0) && ((sensor_fusion[i][5] - ego_car.s) < min_dist_front[k]) )
//                {
//                    min_dist_front[k] = sensor_fusion[i][5] - ego_car.s; //in meters
//                    idx_min_dist_front[k] = i;
//                }
//
//                //back check
//                if( ((ego_car.s - sensor_fusion[i][5]) >= 0) && ((ego_car.s - sensor_fusion[i][5]) < min_dist_back[k]) )
//                {
//                    min_dist_back[k] = ego_car.s - sensor_fusion[i][5]; //in meters
//                    idx_min_dist_back[k] = i;
//                }
//            }
//        }
//        //for each lane to check, add the closest car infront and back of EGO
//        if(min_dist_front[k] < 10000)
//        {
//            relevant_sensor_fusion.push_back(sensor_fusion[idx_min_dist_front[k]]);
//        }
//        if(min_dist_back[k] < 10000)
//        {
//            relevant_sensor_fusion.push_back(sensor_fusion[idx_min_dist_back[k]]);
//        }
//    }
//    /////////////////////////////////////////////////////////////////////////////////////////
//
//
//    //returns the distance to the closest approach
//    //returns the time to first collision
//    bool collides = false;
//    double collides_at = 10000; //a very large number
//    double closest_approach = 10000; //a very large number
//    for(int i=1; i<trajectory_x.size(); i++)
//    {
//        for(int j=0; j<relevant_sensor_fusion.size(); j++)
//        {
//            double sf_theta = atan2(relevant_sensor_fusion[j][4], relevant_sensor_fusion[j][4]); //theta=atan(vy/vx)
//
//            double sf_x = relevant_sensor_fusion[j][1] + relevant_sensor_fusion[j][3]*0.02*i;
//            double sf_y = relevant_sensor_fusion[j][2] + relevant_sensor_fusion[j][4]*0.02*i;
//            vector<double> sf_now = getFrenet(sf_x, sf_y, sf_theta, map_waypoints_x, map_waypoints_y);
//
//            sf_x = relevant_sensor_fusion[j][1] + relevant_sensor_fusion[j][3]*0.02*(i-1);
//            sf_y = relevant_sensor_fusion[j][2] + relevant_sensor_fusion[j][4]*0.02*(i-1);
//            vector<double> sf_prev = getFrenet(sf_x, sf_y, sf_theta, map_waypoints_x, map_waypoints_y);
//
//            double temp_ego_theta = atan2(trajectory_y[i]-trajectory_y[i-1], trajectory_x[i]-trajectory_x[i-1]);
//            vector<double> temp_ego_now = getFrenet(trajectory_x[i], trajectory_y[i], temp_ego_theta, map_waypoints_x, map_waypoints_y);
//            vector<double> temp_ego_prev = getFrenet(trajectory_x[i-1], trajectory_y[i-1], temp_ego_theta, map_waypoints_x, map_waypoints_y); //assume theta doesn't change much with adjacent points
//
//            bool vehicle_collides = check_collision(temp_ego_now, temp_ego_prev, sf_now, sf_prev);
//            if(vehicle_collides && collides == false) //save the first time stamp it collides at
//            {
//                collides = true;
//                collides_at = i*0.02;
//            }
//
//            //check for closest approach
//            if(fabs(temp_ego_now[0] - sf_now[0]) < closest_approach)
//                closest_approach = temp_ego_now[0] - sf_now[0];
//        }
//    }
//
//    return {collides_at, closest_approach}; //if collides_at is too large then it means no collision
//}

double calculate_cost(vehicle_Data &ego_car, target_Data &ego_target, double &min_separation_dist_front)
{
    //cost function weights
    map<string, double> cost_func_weights;
    cost_func_weights["Collision"] = 5e3;
    //cost_func_weights["Danger"] = 2e3; //buffer check
    cost_func_weights["Reach_Goal"] = 3e2; //reach s_max (make sure negative velocities are not allowed)
    cost_func_weights["Comfort"] = 1e2; //penalizes lane changes. Otherwise the car can just keep changing lanes. See the way target_lane is calculated below.
    //cout << cost_func_weights["collision cost"] << "\t" << cost_func_weights["buffer cost"] << "\n";

    //Note: all the costs are from the standpoint of changing lanes, wont help much interms of speed in a given.
    //Collision and Danger Cost
    double collision_cost;
    if(min_separation_dist_front < 10)
        collision_cost = exp(-min_separation_dist_front);
    else
        collision_cost = exp(-min_separation_dist_front*3);

    //Reach Goal Cost
    //this is not necessary as max_accel_in_lane takes care of it
    //double reach_goal_cost = 1 - exp(-(ego_target.max_speed - ego_target.speed));

    //Comfort Cost
    double comfort_cost = 1 - exp(-2*abs(ego_car.s - (ego_target.lane*4+2)));

    //total cost
    double tot_cost = cost_func_weights["collision"]*collision_cost + cost_func_weights["Comfort"]*comfort_cost;

    return tot_cost;

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
        if(ego_car.d > (d-2) && ego_car.d < (d+2))
        {
            if((sensor_fusion[i][5] >= ego_car.s) && ((sensor_fusion[i][5] - ego_car.s) < dist_to_closest_vehicle_infront))
            {
                dist_to_closest_vehicle_infront = sensor_fusion[i][5] - ego_car.s;
                closest_vehicle_infront = i;
            }
        }
    }

    double temp_plan_horizon = (double) ego_target.plan_horizon * 0.2; //at 20% of plan horizon
    temp_plan_horizon = temp_plan_horizon*0.02; //in seconds

    //delta_v_till_target in the temp_plan_horizon time frame
    double delta_v_till_target = (ego_target.max_speed - ego_car.speed)/temp_plan_horizon*0.02; //in MPH per 0.02Sec
    double max_acc = ego_target.max_accel; //in MPH/0.02sec
    if(delta_v_till_target > 0 && delta_v_till_target < max_acc)
        max_acc = delta_v_till_target; //in MPH/0.02sec

    if(dist_to_closest_vehicle_infront < 10000)
    {
        //closest_car_speed is in m/s
        double vx = sensor_fusion[closest_vehicle_infront][3];
        double vy = sensor_fusion[closest_vehicle_infront][4];
        double closest_car_speed = sqrt(vx*vx+vy*vy); //in m/s
        double closest_car_next_pos = sensor_fusion[closest_vehicle_infront][5];
        closest_car_next_pos += closest_car_speed * temp_plan_horizon; //this is in meters

        double ego_next_pos = ego_car.s + (ego_car.speed/2.24) * temp_plan_horizon; //ego_car's speed is in MPH so divide by 2.24 to convert to m/s
        double separation_next = closest_car_next_pos - ego_next_pos; //in meters
        double available_room = separation_next - ego_target.preferred_buffer; //in meters
        double available_acc = 2*available_room/(temp_plan_horizon*temp_plan_horizon); //in m/s^2 //ut+1/2at^2; //the ut component has already been addressed in speration next
        //convert available_acc to MPH per 0.02sec
        available_acc *= 2.24; // convert m/s/s to MPH/s
        available_acc *= 0.02; //convert MPH/s to MPH/0.02sec

        //updates max_acc using value from above
        if(available_acc > 0 && available_acc < max_acc)
        {
            max_acc = available_acc;
        }

        else if(available_acc < 0)
        {
            max_acc = available_acc;
            if(max_acc < -ego_target.max_accel*2)
                max_acc = -ego_target.max_accel*2;
        }
    }

    cout << max_acc << "\t" << ego_car.speed << "\t" << dist_to_closest_vehicle_infront << "\n" ;
    return {max_acc, dist_to_closest_vehicle_infront}; //max_acc is in MPH per 0.02sec
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
        if(ego_car.d > (d-2) && ego_car.d < (d+2))
        {
	    //closest front vehicle
            if((sensor_fusion[i][5] >= ego_car.s) && ((sensor_fusion[i][5] - ego_car.s) < dist_to_closest_vehicle_infront))
            {
                dist_to_closest_vehicle_infront = sensor_fusion[i][5] - ego_car.s;
                closest_vehicle_infront = i;
            }
	    //closest back vehicle
            if((ego_car.s > sensor_fusion[i][5]) && ((ego_car.s - sensor_fusion[i][5]) < dist_to_closest_vehicle_back))
            {
                dist_to_closest_vehicle_back = ego_car.s - sensor_fusion[i][5];
                closest_vehicle_back = i;
            }
        }
    }

    double temp_plan_horizon = (double) ego_target.plan_horizon * 0.2; //at 20% of plan horizon
    temp_plan_horizon = temp_plan_horizon*0.02; //in seconds
	
    //find the max. acceleration
    //delta_v_till_target in the temp_plan_horizon time frame
    double max_acc = ego_target.max_accel; //in MPH/0.02sec
    double delta_v_till_target = (ego_target.max_speed - ego_car.speed)/temp_plan_horizon*0.02; //in MPH per 0.02Sec
    if(delta_v_till_target > 0 && delta_v_till_target < max_acc)
        max_acc = delta_v_till_target; //in MPH/0.02sec
    if(dist_to_closest_vehicle_infront < 10000)
    {
        //closest_car_speed is in m/s
        double vx = sensor_fusion[closest_vehicle_infront][3];
        double vy = sensor_fusion[closest_vehicle_infront][4];
        double closest_car_speed = sqrt(vx*vx+vy*vy); //in m/s
        double closest_car_next_pos = sensor_fusion[closest_vehicle_infront][5];
        closest_car_next_pos += closest_car_speed * temp_plan_horizon; //this is in meters

        double ego_next_pos = ego_car.s + (ego_car.speed/2.24) * temp_plan_horizon; //ego_car's speed is in MPH so divide by 2.24 to convert to m/s
        double separation_next = closest_car_next_pos - ego_next_pos; //in meters
        double available_room = separation_next - ego_target.preferred_buffer; //in meters
        double available_acc = 2*available_room/(temp_plan_horizon*temp_plan_horizon); //in m/s^2 //ut+1/2at^2; //the ut component has already been addressed in speration next
        //convert available_acc to MPH per 0.02sec
        available_acc *= 2.24; // convert m/s/s to MPH/s
        available_acc *= 0.02; //convert MPH/s to MPH/0.02sec

        if(available_acc > 0 && available_acc < max_acc)
        {
            max_acc = available_acc;
        }

        else if(available_acc < 0)
        {
            max_acc = available_acc;
            if(max_acc < -ego_target.max_accel*2)
                max_acc = -ego_target.max_accel*2;
        }
    }
	
    //find the min. acceleration
    double back_temp_plan_horizon = temp_plan_horizon * 0.5; //this needs to be on a shorter time frame than front planning
    double back_preffered_buffer = ego_target.preferred_buffer/2; //can be a bit more aggressive on buffer than for front gap/buffer
    double min_acc = 0.0; //in MPH/0.02sec
    if(dist_to_closest_vehicle_back < 10000)
    {
        //closest_car_speed is in m/s
        double vx = sensor_fusion[closest_vehicle_back][3];
        double vy = sensor_fusion[closest_vehicle_back][4];
        double closest_car_speed = sqrt(vx*vx+vy*vy); //in m/s
        double closest_car_next_pos = sensor_fusion[closest_vehicle_back][5];
        closest_car_next_pos += closest_car_speed * back_temp_plan_horizon; //this is in meters

        double ego_next_pos = ego_car.s + (ego_car.speed/2.24) * back_temp_plan_horizon; //ego_car's speed is in MPH so divide by 2.24 to convert to m/s
        double separation_next = ego_next_pos - closest_car_next_pos; //in meters
	double available_room = back_preffered_buffer - separation_next; //in meters
	//if separation_next is much larger than back_preffered_buffer, then available_acc will be a very large negative number
	//if separation_next is equal to back_preffered_buffer, then available_acc will be 0
	//if separation_next is less than back_preffered_buffer, then available_acc will be a positive number
        double available_acc = 2*available_room/(back_temp_plan_horizon*back_temp_plan_horizon); //in m/s^2 //ut+1/2at^2; //the ut component has already been addressed in speration next
        //convert available_acc to MPH per 0.02sec
        available_acc *= 2.24; // convert m/s/s to MPH/s
        available_acc *= 0.02; //convert MPH/s to MPH/0.02sec
	
	min_acc = available_acc; //MPH per 0.02sec
    }
	
    cout << max_acc << "\t" << dist_to_closest_vehicle_infront << min_acc << "\t" << dist_to_closest_vehicle_back << "\n";

    vector<vector<double>> rtn_var;
    rtn_var.push_back({max_acc, dist_to_closest_vehicle_infront});
    rtn_var.push_back({min_acc, dist_to_closest_vehicle_back});
	
    //min_acc & max_acc are in MPH per 0.02sec and dist_to_closest_vehicle_back & dist_to_closest_vehicle_infront are in meters (Frenet s axis)
    return rtn_var;
}

void ego_target_speed_validation(target_Data &ego_target)
{
    if(ego_target.speed < 0)
    {
        ego_target.speed = 1.0; //spline function complains if this gets below 0 as it wants the points to be sorted in ascending order. so keep slightly above 0
        ego_target.accel = 0.0;
    }
    else if(ego_target.speed > ego_target.max_speed)
    {
        ego_target.speed = ego_target.max_speed;
        ego_target.accel = 0.0;
    }
}

double realize_keep_lane(vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion)
{
    ego_target.lane = (int)(ego_car.d/4); //keep current lane
    vector<double> temp_vec = max_accel_for_keep_lane(ego_car, ego_target, sensor_fusion);
    ego_target.accel = temp_vec[0];
    //speed will be updated in the update_ego_state function.
    ego_target.speed += ego_target.accel;
    ego_target_speed_validation(ego_target);

    double min_separation_dist = temp_vec[1]; //this is in meters (distance along Frenet s axis)
    double tot_cost = calculate_cost(ego_car, ego_target, min_separation_dist);

    return tot_cost;
}

double realize_lane_change(vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion, string str_turn)
{
    //this function updates the target lane and target acceleration (in MPH/0.02sec)
    int delta = 1;
    if(str_turn == "L")
    	delta = -1;

    ego_target.lane = (int)(ego_car.d/4) + delta; //update lane
	temp_ego_car = ego_car;
	temp_ego_target = ego_target;
	max_accel_for_keep_lane(temp_ego_car, temp_ego_target, sensor_fusion);
	//reinitialize these variables;
	temp_ego_car = ego_car;
	temp_ego_target = ego_target;
	temp_ego_car.d = ego_target.lane*4 + 2; //teleport the car into the target lane, with everything else staying as is.
    	ego_target.accel = max_accel_for_lane_change(temp_ego_car, temp_ego_target, sensor_fusion);
    //eventually need to add prepare lane change state to make sure the car can change lane.
    //speed will be updated in the update_ego_state function.
    ego_target.speed += ego_target.accel;
    ego_target_speed_validation(ego_target);

    return 100000;

}

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
	    possible_states.push_back("LCL");
	if(temp_ego_lane < 2)
	    possible_states.push_back("LCR");
    }
    else if (ego_car.state == "LCL")
    {
	if(temp_ego_lane > 0)
	    possible_states.push_back("LCL");
    }
    else if (ego_car.state == "LCR")
    {
	if(temp_ego_lane < 2)
	    possible_states.push_back("LCR");
    }
    else
        cout << "Invalid State\n";


    //possible_states = {"KL"};

    //car_speed is the speed in s direction
    vector<vector<double>> trajectory_x;
    vector<vector<double>> trajectory_y;

    //for finding min. cost path
    double min_cost = 1e10; //a really large number
    double cost_ith_traj;
    string state_min_cost;
    target_Data min_cost_target;

    for(int i=0; i<possible_states.size(); i++)
    {
        string temp_state = possible_states[i];
        vector<vector<double>> temp_sensor_fusion = sensor_fusion;
        vehicle_Data temp_ego_car = ego_car;
        target_Data temp_ego_target = ego_target;
        vector<double> temp_x_vals;
        vector<double> temp_y_vals;

        if (temp_state == "KL")
        {
            cost_ith_traj = realize_keep_lane(temp_ego_car, temp_ego_target, temp_sensor_fusion);
        }
        else if (temp_state == "LCL")
        {
            cost_ith_traj = realize_lane_change(temp_ego_car, temp_ego_target, temp_sensor_fusion, "L");
        }
        else if (temp_state == "LCR")
        {
            cost_ith_traj = realize_lane_change(temp_ego_car, temp_ego_target, temp_sensor_fusion, "R");
        }
        else
        {
            cout << "Invalid State\n";
        }

        if (cost_ith_traj < min_cost)
        {
            min_cost = cost_ith_traj;
            min_cost_target = temp_ego_target;
            state_min_cost = possible_states[i];
        }
    }
    //cout << possible_states[traj_min_cost] << "\n\n";

    ego_target = min_cost_target;

}

//void update_ego_state(int &prev_size, vehicle_Data &ego_car, target_Data &ego_target, vector<vector<double>> &sensor_fusion, double &end_path_s, double &end_path_d, string &state)
//{
//    if(state == "LCL" && ego_car.d > 4)
//    {
//       ego_target.lane = (int)(ego_car.d/4) - 1;
//    }
//
//    else if(state == "LCR" && ego_car.d < 8)
//    {
//        ego_target.lane = (int)(ego_car.d/4) + 1;
//    }
//
//    else
//    {
//        ego_target.lane = (int)(ego_car.d/4);
//        cout << ego_target.lane << endl;
//    }
//
//    if(prev_size > 0)
//    {
//        ego_car.s = end_path_s;
//        ego_car.d = end_path_d;
//    }
//
//    bool too_close = false;
//
//    //find ref_v to use
//    for(int i=0; i<sensor_fusion.size(); i++)
//    {
//        //check if car is in my lane
//        float d = sensor_fusion[i][6];
//        //cout << d << "\t" << 2+target_lane*4 << "\t" << car_d << endl;
//        //if((d > (2+target_lane*4-2)) && (d < (2+target_lane*4+2))) //car could be off center in a lane so have to check in a range of values
//        if((d > (ego_car.d-2)) && (d < (ego_car.d+2)))
//        {
//            double vx = sensor_fusion[i][3];
//            double vy = sensor_fusion[i][4];
//            double check_speed = sqrt(vx*vx+vy*vy);
//            double check_car_s = sensor_fusion[i][5];
//
//            double s_gap = 30;
//            //using previous points can project the sensor fusion car out into future
//            check_car_s += (double)prev_size * 0.02 * check_speed; //use prev_size as we are saying car_s = end_path_s
//            //check s values grater than mine and s gap
//            if(check_car_s > ego_car.s && (check_car_s - ego_car.s) < s_gap)
//            {
//                //do some logic here
//                //lower reference velocity so we don't crash into the car infront of us
//                //could also flag to try to change lane
//                //target_vel = 25; //MPH
//                too_close = true;
//                ego_target.speed -= ego_target.accel;
//                //ego_target.lane = (int)(ego_car.d/4) - 1; //LCL
//
//
//                //if(state == "LCL" && ego_target.lane > 0)
//                //{
//                //   ego_target.lane -= 1;
//                //}
//
//                //if(state == "LCR" && ego_target.lane < 2)
//                //{
//                //    ego_target.lane += 1;
//                //}
//
//            }
//
//        }
//    }
//
//    if(too_close)
//    {
//        //ego_target.speed -= ego_target.accel;
//    }
//
//    else if(ego_target.speed < ego_target.max_speed)
//    {
//        ego_target.speed += ego_target.accel;
//    }
//}


int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
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
  ego_car.accel = 0.0; //initially set to zero

  target_Data ego_target;
  //ego_target.accel = 0.224; //MPH per (0.02second). //0.224 MPH/(0.02SEC) = 11MPH/SEC = 5m/s^2. This is acceleration
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
            //ego_car.state = "KL";
            //ego_car.state/accel are updated in the behavior_Planner function
            behavior_Planner(ego_car, sensor_fusion, ego_target);

            //update_ego_state(prev_size, ego_car, ego_target, sensor_fusion, end_path_s, end_path_d, ego_car.state);

            //define the actual (x,y) points we will use for the planner
            vector<double> next_x_vals;
            vector<double> next_y_vals;

            generate_Trajectory(next_x_vals, next_y_vals, ego_car, previous_path_x, previous_path_y, ego_target, map_waypoints_x, map_waypoints_y, map_waypoints_s);


//            //basic go forward in the lane without smoothing
//            double dist_inc = 0.5;
//            for(int i = 0; i < 50; i++)
//            {
//                double next_s = car_s + (i+1)*30/2.24*0.02; //(i+1)*dist_inc;
//                //frenet is from the double yellow lines. Lane width is 4 meters
//                //Being in the middle of second lane is 1.5 lanes from the double yellow lines
//                double next_d = 1.5 * 4;
//                vector<double> next_xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
//                next_x_vals.push_back(next_xy[0]);
//                next_y_vals.push_back(next_xy[1]);
//                //next_x_vals.push_back(car_x+(dist_inc*i)*cos(deg2rad(car_yaw))); //for straight line
//                //next_y_vals.push_back(car_y+(dist_inc*i)*sin(deg2rad(car_yaw))); //for straight line
//            }
//
//            cout << car_speed << endl;
            //end to do

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