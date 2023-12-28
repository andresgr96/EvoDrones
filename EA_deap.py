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

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.EvoDrones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.EvoDrones.controllers.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.EvoDrones.controllers.rand_action import build_action, build_action_forward
from gym_pybullet_drones.EvoDrones.utils.computer_vision import display_drone_image, red_mask, segment_image,\
    detect_objects
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Sim constants, do not change unless you really know what you are doing
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 6
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def run_sim(
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
    env = CtrlAviary(drone_model=drone,
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
    # Helper Prints
    # print("[INFO] Action space:", env.action_space)
    # print("[INFO] Observation space:", env.observation_space)
    # print("[INFO] Max RPM:", env.MAX_RPM)

    # Obtain the PyBullet Client ID from the environment
    PYB_CLIENT = env.getPyBulletClient()

    # Test variables, segment tracking code currently only works for 1 drone
    line_position, _ = p.getBasePositionAndOrientation(env.segment_ids.get("segment_1")["id"])
    coord_line_covers = env.segment_ids.get("segment_1")["coordinates"]
    drones_segments_completed = np.zeros((num_drones, env.num_segments))  # Tracks the segments completed by drone
    current_segment_completion = np.zeros(10)                             # Tracks current segment completed sections
    current_segment_idx = 0                                               # Tracks the current segment


    # Run the simulation
    START = time.time()
    segments_completed = 0
    fitness = 0

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        # Build the action for each drone and take a step, action is random for now
        action = build_action_forward(num_drones)
        obs, reward, terminated, truncated, info = env.step(action)

        # Display the camera feed of drone 1
        rgb_image, _, _ = env._getDroneImages(0)
        print(rgb_image.shape)
        segmented = segment_image(rgb_image)
        mask = red_mask(rgb_image)
        display_drone_image(mask)  # Use mask here if binary mask, segmented for normal img with lines
        # print(detect_objects(mask))     # Use in combination with segmented above to test correct functionality

        # Calculate if the drones are over a segment, currently only checks for the same segment.
        drone_positions = env._getDronePositions()
        for z, position in enumerate(drone_positions):
            over_line = env.is_drone_over_line(position, line_position)
            over_last_10 = env.is_within_last_10_percent(position, "segment_1")
            drone_segment_position = env.check_drone_position_in_sections(position, "segment_1")
            current_segment_completion[np.where(drone_segment_position == 1)] = 1

            if np.sum(current_segment_completion) >= 8:
                drones_segments_completed[z][current_segment_idx] = 1


            # Print Line and Drone Position to test functionality
            # print(f"Custom Line Position: x={line_position[0]}, y={line_position[1]}, z={line_position[2]}")
            # print(f"Coordinates Covered: x range= {coord_line_covers[0]} to {coord_line_covers[1]}, "
            #       f"y range= {coord_line_covers[2]} to {coord_line_covers[3]}")
            # print(f"Drone {z + 1} Position: x={position[0]}, y={position[1]}, z={position[2]}")
            # print(f"Is drone {z + 1} over the line? {over_line}")
            # print(f"Is drone {z + 1} at end of segment? {over_last_10}")
            # print(f"Drone {z + 1} position in segment {drone_segment_position}")
            # print(f"Drones current segment completion: {current_segment_completion}")
            # print(f"Drones {z + 1} segments completed: {drones_segments_completed[z]}")

        # Render and sync the sim if needed
        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    # Close the environment and return fitness
    env.close()

    fitness = np.sum(drones_segments_completed[0, :])
    return fitness