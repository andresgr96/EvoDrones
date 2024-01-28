"""

Example
-------
In a terminal, run as:

    $ python control_tests.py

"""
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
import pickle
import neat

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.EvoDrones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.EvoDrones.controllers.DSLPIDControl import DSLPIDControl
# from gym_pybullet_drones.EvoDrones.controllers.rand_action import build_action, build_action_forward
from gym_pybullet_drones.EvoDrones.utils.computer_vision import display_drone_image, red_mask, segment_image,\
    detect_objects, detect_circles
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.EvoDrones.controllers.nn import build_action

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
DEFAULT_DURATION_SEC = 120
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def run(
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
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    print("[INFO] Max RPM:", env.MAX_RPM)

    # Obtain the PyBullet Client ID from the environment
    PYB_CLIENT = env.getPyBulletClient()

    # Initialize the logger
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    # Initialize the controllers
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    # Tracking of the current segments information
    current_segment_idx = 0
    current_segment_id = current_segment_idx + 2      # For some reason the dictionary starts at id 2
    current_segment_name = env.get_segment_name_by_id(current_segment_id)
    line_position, _ = p.getBasePositionAndOrientation(env.segment_ids.get(current_segment_name)["id"])

    # For completion tracking
    drones_segments_completed = np.zeros((num_drones, env.num_segments))  # Tracks the segments completed by drone
    current_segment_completion = np.zeros(10)                             # Tracks current segment completed sections

    # Run the simulation
    START = time.time()
    segments_completed = 0

    # Load the saved genome
    file_path = os.path.join(os.getcwd(), "../results/V2/best.pickle")
    with open(file_path, "rb") as f:
        winner = pickle.load(f)

    # Load NEAT configuration
    config_path = "../assets/config_rpms.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create a FeedForwardNetwork using the loaded genome and configuration
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

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
        print(action)
        action = build_action(action)
        _ = env.step(action)

        # Update drone state information
        state_vector = env._getDroneStateVector(0)
        position = state_vector[:3]
        quat = state_vector[3:7]
        rpy = state_vector[7:10]
        vel = state_vector[10:13]
        ang_vel = state_vector[13:16]
        last_clipped_action = state_vector[16:]
        x, y, z = position

        print(env.drone_landed(position, vel))

        # Calculate if the drones are over a segment, currently only checks for the same segment.
        drone_positions = env._getDronePositions()
        print(x, y, z)
        for z, position in enumerate(drone_positions):
            # over_line = env.is_drone_over_line(position, line_position)
            # over_last_10 = env.is_within_last_10_percent(position, "segment_1")
            drone_segment_position = env.check_drone_position_in_sections(position, "segment_1")
            current_segment_completion[np.where(drone_segment_position == 1)] = 1
            # print(env.is_drone_inside_circle(position))
            # print(env.distance_from_circle(position))

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

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)',
                        metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)',
                        metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
