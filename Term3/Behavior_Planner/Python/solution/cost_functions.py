from __future__ import division
from collections import namedtuple
from math import sqrt,exp
import pdb

TrajectoryData = namedtuple("TrajectoryData", [
	'proposed_lane',
	'avg_speed',
	'max_acceleration',
	'rms_acceleration',
	'closest_approach',
	'end_distance_to_goal',
	'end_lanes_from_goal',
	'collides',
	])

# priority levels for costs
COLLISION  = 10 ** 6
DANGER     = 10 ** 5
REACH_GOAL = 10 ** 5
COMFORT    = 10 ** 4
EFFICIENCY = 10 ** 2

DESIRED_BUFFER = 1.5 # timesteps
PLANNING_HORIZON = 2

DEBUG = False
# DEBUG = True

def change_lane_cost(vehicle, trajectory, predictions, data):
	"""
	Penalizes lane changes AWAY from the goal lane and rewards
	lane changes TOWARDS the goal lane.
	"""
	proposed_lanes = data.end_lanes_from_goal
	cur_lanes = trajectory[0].lane
	cost = 0
	if proposed_lanes > cur_lanes:
		cost = COMFORT
	if proposed_lanes < cur_lanes:
		cost = -COMFORT
	if cost != 0:
		print "!! \n \ncost for lane change is {}\n\n".format(cost)
	return cost

def distance_from_goal_lane(vehicle, trajectory, predictions, data):
	"""
	
	"""
	distance = abs(data.end_distance_to_goal)
	distance = max(distance,1.0)
	time_to_goal = float(distance) / data.avg_speed
	lanes = data.end_lanes_from_goal
	multiplier = float(5 * lanes / time_to_goal) 
	cost = multiplier * REACH_GOAL

	return cost
	pass

def inefficiency_cost(vehicle, trajectory, predictions, data):
	speed = data.avg_speed
	target_speed = vehicle.target_speed
	diff = target_speed - speed
	pct = float(diff) / target_speed
	multiplier = pct ** 2
	return multiplier * EFFICIENCY
	pass

def collision_cost(vehicle, trajectory, predictions, data ):
	if data.collides:
		time_til_collision = data.collides['at']
		exponent = (float(time_til_collision) ) ** 2
		mult = exp(-exponent)

		return mult * COLLISION
	return 0

def buffer_cost(vehicle, trajectory, predictions, data):
	closest = data.closest_approach
	if closest == 0:
		return 10 * DANGER

	timesteps_away = closest / data.avg_speed
	if timesteps_away > DESIRED_BUFFER:
		return 0.0
	
	multiplier = 1.0 - (timesteps_away / DESIRED_BUFFER)**2
	return multiplier * DANGER
	pass

def calculate_cost(vehicle, trajectory, predictions, verbose=False):
	trajectory_data = get_helper_data(vehicle, trajectory, predictions)
	cost = 0.0
	for cf in [
		distance_from_goal_lane,
		inefficiency_cost,
		collision_cost,
		buffer_cost,
		change_lane_cost]:
		new_cost = cf(vehicle, trajectory, predictions, trajectory_data)
		if DEBUG or verbose:
			print "{} has cost {} for lane {}".format(cf.__name__, new_cost, trajectory[-1].lane)
			# pdb.set_trace()
		cost += new_cost
	return cost

	pass

def get_helper_data(vehicle,trajectory,predictions):
	t = trajectory
	current_snapshot = t[0]
	first = t[1]
	last = t[-1]
	end_distance_to_goal = vehicle.goal_s - last.s
	end_lanes_from_goal = abs(vehicle.goal_lane - last.lane)
	dt = float(len(trajectory))
	proposed_lane = first.lane
	avg_speed = (last.s - current_snapshot.s) / dt

	# initialize a bunch of variables
	accels = []
	closest_approach = 999999
	collides = False
	last_snap = trajectory[0]
	filtered = filter_predictions_by_lane(predictions, proposed_lane)

	for i, snapshot in enumerate(trajectory[1:PLANNING_HORIZON+1], 1):
		lane, s, v, a = unpack_snapshot(snapshot)

		accels.append(a)
		for v_id, v in filtered.items():
			state = v[i]
			last_state = v[i-1]
			vehicle_collides = check_collision(snapshot, last_state['s'], state['s'])
			if vehicle_collides:
				collides = True
				collides = {"at" : i}
			dist = abs(state['s'] - s)
			if dist < closest_approach:
				closest_approach = dist
		last_snap = snapshot
	max_accel = max(accels, key=lambda a: abs(a))
	rms_accels = [a**2 for a in accels]
	num_accels = len(rms_accels)
	rms_acceleration = float(sum(rms_accels)) / num_accels

	return TrajectoryData(
		proposed_lane, 
		avg_speed, 
		max_accel, 
		rms_acceleration, 
		closest_approach, 
		end_distance_to_goal,
		end_lanes_from_goal,
		collides)


def check_collision(snapshot, s_previous, s_now):
	s = snapshot.s
	v = snapshot.v
	v_target = s_now - s_previous
	if s_previous < s:
		if s_now >= s:
			return True
		else:
			return False

	if s_previous > s:
		if s_now <= s:
			return True
		else:
			return False

	if s_previous == s:
		if v_target > v:
			return False
		else:
			return True
	raise ValueError 


def unpack_snapshot(snapshot):
	s = snapshot
	return s.lane, s.s, s.v, s.a

def filter_predictions_by_lane(predictions, lane):
	filtered = {}
	for v_id, predicted_traj in predictions.items():
		if predicted_traj[0]['lane'] == lane and v_id != -1:
			filtered[v_id] = predicted_traj
	return filtered