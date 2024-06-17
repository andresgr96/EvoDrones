import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import cv2
import neat
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.evo_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.evo_drones.controllers.nn import build_action
from gym_pybullet_drones.evo_drones.utils.computer_vision import display_drone_image, red_mask, segment_image, \
    detect_objects, detect_circles
from gym_pybullet_drones.utils.utils import sync, str2bool
# from gym_pybullet_drones.EvoDrones.controllers.DSLPIDControl import DSLPIDControl
# from gym_pybullet_drones.EvoDrones.controllers.rand_action import build_action, build_action_forward, to_hover, \
#     action_decision

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
DEFAULT_DURATION_SEC = 4.5
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
    done = False


    # Takeoff state rewards tracking
    takeoff_reward = 0
    takeoff_reward_value = 250
    takeoff_penalty = 1

    # Following state rewards tracking
    following_reward = 0
    out_of_track_penalty = 1

    # Landing state rewards tracking
    landing_reward = 0
    landing_reward_value = 500
    high_z_penalty = 1
    within_circle_reward = 1

    # General reward values
    segment_completed_value = 250
    section_reward_value = 20
    moving_forwards_value = 1.5
    unfeasible_action_penalty = 1
    moving_backwards_pen = 1
    stability_penalty = 1
    runtime_penalty = 1

    # Final settings
    steps = 0
    genome.fitness = 0
    first_pos = env._getDronePositions()[0]
    old_dist = env.distance_from_circle(first_pos)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    START = time.time()

    # Run the simulation
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        if done:
            break

        drone_positions = env._getDronePositions()
        for drone_id, _ in enumerate(drone_positions):

            # Computer Vision
            rgb_image, _, _ = env._getDroneImages(0)
            circle_image = rgb_image
            mask = red_mask(rgb_image)
            pixel_count = detect_objects(mask)
            circle, circle_i = detect_circles(circle_image)
            # display_drone_image(circle_i)
            pixel_count.append(circle)

            # Build and take action
            action = net.activate(pixel_count)
            if np.any(np.array(action) > 21000):  # Penalize unfeasible actions
                # print(f"Penalize action: {action}")
                genome.fitness -= unfeasible_action_penalty
            # print(f"Before: {action}")
            action = build_action(action)
            # print(f"After: {action}")
            _ = env.step(action)
            steps += 1

            # Update drone state information
            state_vector = env._getDroneStateVector(drone_id)
            position = state_vector[:3]
            vel = state_vector[10:13]
            ang_vel = state_vector[13:16]
            curr_dist_to_circle = env.distance_from_circle(position)
            x, y, z = position

            # Get segment information for tracking, should probably be compacted
            if env.get_current_segment(position) is not None:
                current_segment_idx = env.get_current_segment(position) - 2
            current_segment_id = current_segment_idx + 2  # For some reason the dictionary starts at id 2
            current_segment_name = env.get_segment_name_by_id(current_segment_id)

            # Update fitness function conditionals
            is_stable = True if env.is_stable(ang_vel, vel) else False
            within_track = True if env.is_drone_over_line(position, current_segment_name) else False
            in_target_circle = True if env.drone_in_target_circle(position) else False
            drone_landed = True if env.drone_landed(position, vel) else False
            moving_forward = True if old_dist - curr_dist_to_circle > 0 else False
            old_dist = curr_dist_to_circle

            # Check the completion of the segment by the drone
            drone_segment_position = env.check_drone_position_in_sections(position, current_segment_name)
            if is_stable:     # Do not count sections completed by flying in a way that leads to crashes
                segment_completion[current_segment_idx][drone_segment_position == 1] = 1  # All segments
                current_segment_completion[np.where(drone_segment_position == 1)] = 1  # Current Segment
            if np.sum(current_segment_completion) >= 9:
                drones_segments_completed[0][current_segment_idx] = 1

            # -------------------------------------- Drone's state machine --------------------------------------#
            if taking_off:
                # Encourage going up by penalizing staying down
                if z < 0.5:
                    takeoff_reward -= takeoff_penalty
                    # Encourage stable flight
                    if not is_stable:
                        following_reward -= stability_penalty
                    if not within_track:
                        following_reward = 0
                        genome.fitness = -1000
                        print("Out of Bounds")
                        done = True
                        break
                elif z >= 0.5:
                    takeoff_reward += takeoff_reward_value
                    print("Mission is Good")
                    done = True
                    break
            # -------------------------------------- Drone's state machine END --------------------------------------#

        # Render and sync the sim if needed
        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)


    # Final fitness calculations
    genome.fitness += takeoff_reward - (steps * runtime_penalty/100)
    print('Fitness', genome.fitness)
    print('Steps', steps)

    # Close the environment and return fitness
    env.close()

    return genome.fitness
