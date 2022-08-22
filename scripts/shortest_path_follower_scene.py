import math
import os
import random
import sys

import cv2 
import git
import imageio
import magnum as mn
import numpy as np
import argparse
import tqdm 

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_sim.nav import GreedyGeodesicFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

from scipy.spatial.transform import Rotation as R


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	"--scene_id",
    	type=str,
    	default="./data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    	help="specify glb file ")
    parser.add_argument(
        "--out_dir",
        default=os.path.join("data_collection", "shortest_path_scene"),
        help="output directory to store recorded data ",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect data for",
    )
    parser.add_argument(
    	"--max_steps",
    	type=int,
    	default=128,
    	help="maximum steps allowed per episode")
    args = parser.parse_args()
    return args


args = get_args()



rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": args.scene_id,  # Scene path
    "default_agent": 0,
    "sensor_height": 0.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



cfg = make_cfg(sim_settings)
# Needed to handle out of order cell run in Colab
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)




def euler_to_quaternion(yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]



agent = sim.initialize_agent(sim_settings["default_agent"])
follower = GreedyGeodesicFollower(sim.pathfinder, agent, goal_radius=0.25)


sim.seed(sim_settings["seed"])
random.seed(sim_settings["seed"])
for episode in tqdm.tqdm(range(args.num_episodes)):
    sim.reset()
    start_state = habitat_sim.AgentState()
    goal_state = habitat_sim.AgentState()
    
    # Sample Random navigable points as start and goal states 
    start_state.position = sim.pathfinder.get_random_navigable_point()
    goal_state.position = sim.pathfinder.get_random_navigable_point()
    
    # Randomly sample the rotations for start and goal states
    start_state.rotation = euler_to_quaternion(0, random.randint(0,360), 0)
    # goal_state.rotation = R.random().as_quat() # Not being used right now. 
    
    # Set the agent start state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent.set_state(start_state)
    
    # Creating directories 
    try:
        os.system("mkdir -p %s"%os.path.join(args.out_dir, args.scene_id.split("/")[-1][:-4], "%03d"%episode, "images"))
    except:
        pass
    
    step = 0 
    images, actions = [], []
    while step < args.max_steps:
        best_action = follower.next_action_along(goal_state.position)
        if best_action is None:
            break
        obs = sim.step(best_action) 
        im = obs["color_sensor"]
        actions.append(best_action)
        cv2.imwrite(os.path.join(args.out_dir, args.scene_id.split("/")[-1][:-4], "%03d"%episode, "images", "%03d.png"%step), im)
        step+=1
    np.save(os.path.join(args.out_dir, args.scene_id.split("/")[-1][:-4], "%03d"%episode, "greedy_geodesic_follower_actions.npy"), np.array(actions))
        