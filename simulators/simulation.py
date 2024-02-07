import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2
import neat

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.EvoDrones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.EvoDrones.controllers.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.EvoDrones.controllers.rand_action import build_action, build_action_forward, to_hover, \
    action_decision, action_dec
from gym_pybullet_drones.EvoDrones.utils.computer_vision import display_drone_image, red_mask, segment_image, \
    detect_objects, detect_circles
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="EvoDrones",
    group="NEAT",
    # track hyperparameters and run metadata
    config={
    "fitness_criterion"     : max,
    "fitness_threshold"     : 500,
    "pop_size"              : 10,
    "reset_on_extinction"   : False,
    "species_fitness_func" : max,
    "max_stagnation"       : 3,
    "species_elitism"      : 1,
    "elitism"            : 1,
    "survival_threshold" : 0.2,
    "activation_mutate_rate"  : 1.0,
    "aggregation_default"     : sum,
    "aggregation_mutate_rate" : 0.0,
    "aggregation_options"     : sum,
    "bias_init_mean"          : 3.0,
    "bias_init_stdev"         : 1.0,
    "bias_max_value"          : 30.0,
    "bias_min_value"          : -30.0,
    "bias_mutate_power"       : 0.5,
    "bias_mutate_rate"        : 0.7,
    "bias_replace_rate"       : 0.1,
    "compatibility_disjoint_coefficient" : 1.0,
    "compatibility_weight_coefficient"   : 0.5,
    "conn_add_prob"           : 0.5,
    "conn_delete_prob"        : 0.5,
    "enabled_default"         : True,
    "enabled_mutate_rate"     : 0.01,
    "feed_forward"            : True,
    "node_add_prob"           : 0.2,
    "node_delete_prob"        : 0.2,
    "num_hidden"              : 4,
    "num_inputs"              : 37,
    "num_outputs"             : 5,
    "response_init_mean"      : 1.0,
    "response_init_stdev"     : 0.0,
    "response_max_value"      : 30.0,
    "response_min_value"      : -30.0,
    "response_mutate_power"   : 0.0,
    "response_mutate_rate"    : 0.0,
    "response_replace_rate"   : 0.0,
    "weight_init_mean"        : 0.0,
    "weight_init_stdev"       : 1.0,
    "weight_max_value"        : 30,
    "weight_min_value"        : -30,
    "weight_mutate_power"     : 0.5,
    "weight_mutate_rate"      : 0.8,
    "weight_replace_rate"     : 0.1,
    "compatibility_threshold" : 3.0
    }
)

# Sim constants, do not change unless you really know what you are doing
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 3
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run_sim(
        genome,
        config,
        # individual: np.array,
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
):
    # Initialize the simulation
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array(
        [[R * np.cos((i / 6) * 2 * np.pi + np.pi / 2), R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R, H + i * H_STEP]
         for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, i * (np.pi / 2) / num_drones] for i in range(num_drones)])

    # Create the environment
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )

    # Set the current segment to the first in case drone does not start on it for some reason
    current_segment_idx = 0

    # For completion tracking
    drones_segments_completed = np.zeros((num_drones, env.num_segments))  # Tracks the segments completed by drone
    segment_completion = np.zeros((env.num_segments, 10))  # Tracks completed sections for all segments
    current_segment_completion = np.zeros(10)  # Tracks current segment completed sections

    # Drone state tracking
    taking_off = True
    following = False
    landing = False
    landed = False

    # Takeoff state rewards tracking
    takeoff_reward = 0
    takeoff_penalty = 5
    stable_penalty = 0
    speed_penalty = 0

    # Following state rewards tracking
    following_reward = 0
    first_pos = env._getDronePositions()[0]
    old_dist = env.distance_from_circle(first_pos)

    # Landing state rewards tracking
    landing_reward = 0

    # Run the simulation
    START = time.time()
    steps = 0
    genome.fitness = 0
    segments_completed = 0
    last_action = np.array([[0,0,0,0]])

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        drone_positions = env._getDronePositions()
        for _ in enumerate(drone_positions):

            # Computer Vision
            rgb_image, _, _ = env._getDroneImages(0)
            circle_image = rgb_image
            mask = red_mask(rgb_image)
            pixel_count = detect_objects(mask)
            circle, circle_i = detect_circles(circle_image)
            # display_drone_image(circle_i)
            pixel_count.append(circle)

            # Build and take action
            
            output = net.activate(pixel_count)
            # print(output)
            speed = output[0] * 10
            decision = output.index(max(output[1:8]))
            if i == 0:
                default = 15000
                prev = np.array([[default, default, default, default]])
                action = action_dec(prev, decision, speed)
                last_action = action
            else:
                action = action_dec(last_action, decision, speed)
                last_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            # print('---')
            pos = obs[0][:3]
            quat = obs[0][3:7]
            rpy = obs[0][7:10]
            vel = obs[0][10:13]
            ang_v = obs[0][13:16]
            last_clip_act = obs[0][16:19]
            # print('pos', pos)
            # print('quat', quat)
            # print('rpy', rpy)
            # print('vel', vel)
            # print('ang_v', ang_v)
            # print('last_clip_act', last_clip_act)

            # Update drone position and steps
            position = env._getDronePositions()[0]
            x, y, z = position
            steps += 1

            # Check if drone is moving towards the circle and update old distance for next step
            curr_dist_to_circle = env.distance_from_circle(position)
            moving_forward = True if old_dist - curr_dist_to_circle >= 0 else False
            old_dist = curr_dist_to_circle

            # Get segment information for tracking, should probably be compacted
            segment_idx = env.get_current_segment(position)
            if env.get_current_segment(position) is not None:
                current_segment_idx = env.get_current_segment(position) - 2
            current_segment_id = current_segment_idx + 2  # For some reason the dictionary starts at id 2
            current_segment_name = env.get_segment_name_by_id(current_segment_id)

            # Check the completion of the segment by the drone
            drone_segment_position = env.check_drone_position_in_sections(position, current_segment_name)
            segment_completion[current_segment_idx][drone_segment_position == 1] = 1  # All segments
            current_segment_completion[np.where(drone_segment_position == 1)] = 1  # Current Segment

            # If the drone has completed 90% of a segment then we consider it complete
            if np.sum(current_segment_completion) >= 9:
                drones_segments_completed[0][current_segment_idx] = 1

            # -------------------------------------- Drone's state machine --------------------------------------#
            if taking_off:
                # print("Taking Off")
                # Encourage going up by penalizing staying down
                if z < 0.2:
                    takeoff_reward -= takeoff_penalty
                elif z >= 0.2:
                    takeoff_reward += 20
                    taking_off = False
                    following = True
                    
                if z < 0.07:
                    env.close()
                    genome.fitness -= 300
                    return genome.fitness
            elif following:
                # print("Following")
                at_segments_end = np.sum(current_segment_completion) >= 8

                # Encourage moving towards the circle
                if moving_forward:
                    following_reward += takeoff_penalty
                else:
                    following_reward -= takeoff_penalty

                # Encourage staying within the current segments coordinates
                if not env.is_drone_over_line(position, current_segment_name):
                    following_reward -= takeoff_penalty
                    
                # Encourage stability
                stable_penalty -= (abs(sum(ang_v)) + 4)
                
                if not all(abs(element) < 0.8 for element in vel):
                    speed_penalty -= 4
                    

                # Check if the circle has been detected
                if (circle == 1 and at_segments_end) or at_segments_end:
                    following = False
                    landing = True
                elif z < 0.07:
                    env.close()
                    genome.fitness -= 300
                    return genome.fitness
            elif landing:
                # print("Landing")
                if not env.drone_landed(position):
                    # Encourage flying down
                    landing_reward -= takeoff_penalty

                    # Encourage moving towards the circle
                    if moving_forward:
                        landing_reward += takeoff_penalty
                    else:
                        landing_reward -= takeoff_penalty

                    # Encourage staying within the circles x and y coordinates
                    if env.drone_in_target_circle(position):
                        landing_reward += takeoff_penalty
                elif env.drone_landed(position):
                    # print("Landed")
                    landing_reward += 50
                    landing = True
                    landed = False
            # -------------------------------------- Drone's state machine END --------------------------------------#
        
        # print(i, z)
        # Render and sync the sim if needed
        if gui:
            env.render()
            sync(i, START, env.CTRL_TIMESTEP)

        if landed:
            break

    # Final fitness calculations
    segment_reward = np.sum(drones_segments_completed[0, :]) * 50
    sections_reward = np.sum(current_segment_completion) * 2
    genome.fitness += takeoff_reward + following_reward + landing_reward\
                      + segment_reward + sections_reward + stable_penalty + speed_penalty - (steps * 0.1)
    # print('fitness', genome.fitness)
    # Close the environment and return fitness
    wandb.log({'fitness': genome.fitness})
    env.close()

    return genome.fitness
