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
from gym_pybullet_drones.EvoDrones.controllers.rand_action import build_action, build_action_forward, to_hover, action_decision
from gym_pybullet_drones.EvoDrones.utils.computer_vision import display_drone_image, red_mask, segment_image,\
    detect_objects
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

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


def calculate_distance(target, position):
    tx, ty, tz = target
    x, y, z = position
    
    return -(abs(tx - x) + abs(ty - y) + abs(tz - z))
    

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
    drones_segments_completed = np.zeros((num_drones, env.num_segments))   # Tracks the segments completed by drone
    segment_completion = np.zeros((env.num_segments, 10))                  # Tracks completed sections for all segments
    current_segment_completion = np.zeros(10)                              # Tracks current segment completed sections

    # Run the simulation
    START = time.time()
    genome.fitness = 0
    segments_completed = 0
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        
        # At each time step we want to :
       
        drone_positions = env._getDronePositions()
        for z, position in enumerate(drone_positions):
            
            x, y, z = position
            
            # 0. Get to hover
            if z < 0.2:
                # print(z)
                action = to_hover()
                obs, reward, terminated, truncated, info = env.step(action)
                break
            
            # 1. Observe current state
            # Display the camera feed of drone 1
            rgb_image, _, _ = env._getDroneImages(0)
            segmented = segment_image(rgb_image)
            mask = red_mask(rgb_image)
            # display_drone_image(mask)  # Use mask here if binary mask, segmented for normal img with lines

            output = net.activate((x, y, z))
            decision = output.index(max(output))

            
            # 3. Pass output action to drone
            action = action_decision(decision)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 4. Observe segment number we are at and calculate fitness
            # From output (array of 4), before turning get to hover?
            # segment_idx = env.get_current_segment(position)
            
            
            # if segment_idx is None:
            #     distance = calculate_distance((2.5, 0.0, .001), position)
            #     genome.fitness += distance
            #     env.close()
            #     return distance
            
            
            # current_segment_idx = env.get_current_segment(position) - 2
            # print('id', current_segment_idx)
            # # print(current_segment_idx, position)
            
            # # Get the segments name (can probably refactor to only use id for everything)
            # current_segment_id = current_segment_idx + 2  # For some reason the dictionary starts at id 2
            # current_segment_name = env.get_segment_name_by_id(current_segment_id)
            
            # # Get the segment position to calculate its completion by drones position
            # drone_segment_position = env.check_drone_position_in_sections(position, current_segment_name)
            # segment_completion[current_segment_idx][drone_segment_position == 1] = 1  # All segments
            
            # current_segment_completion[np.where(drone_segment_position == 1)] = 1  # Current Segment
                
            # # If the drone has completed 80% of a segment then we consider it complete
            # if np.sum(current_segment_completion) >= 8:
            #     drones_segments_completed[z][current_segment_idx] = 1
            
            # fitness = np.sum(drones_segments_completed[0, :])
            
            distance = calculate_distance((2.5, 0.0, .001), position)
            if distance > -0.1:
                genome.fitness += distance
                env.close()
                return distance
                

        # Render and sync the sim if needed
        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    genome.fitness += distance
    print('fitness', distance)
    
    # Close the environment and return fitness
    env.close()

    
    return distance
