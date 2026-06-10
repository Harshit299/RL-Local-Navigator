import gymnasium as gym
from typing import Optional
import numpy as np
from collections import deque

class RobotEscapeEnv(gym.Env):

    def __init__(self):
        super(RobotEscapeEnv, self).__init__()

        # =========== define observation and action spaces ==================
        
        # 24 lidar data and 2d goal pose (distance to goal and angle to goal) - normalised between 0 and 1
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (26,), dtype = np.float32)
        self.action_space = gym.spaces.Box(low = np.array([-0.2, -1.0], dtype = np.float32), # max -ve linear speed, max -ve angular speed
                                           high = np.array([1.5, 1.0], dtype = np.float32), # max +ve linear speed, max +ve angular speed
                                           dtype = np.float32)
        
        # ===================================================================
        
        self.stagnation_penalty = 0.0
        self.dt = 0.1 # time diff b/w state 1 and state 2
        self.robot_pose = np.array([0.0, 0.0, 0.0]) # x,y,theta
        self.robot_radius = 0.125
        self.obstacle_radius = 0.25
        self.goal_pose = np.array([0.0, 0.0]) # x,y
        self.obstacles = []
        self.prev_dist_to_goal = 0.0
        self.pose_history = deque(maxlen = 30)

        # Gap-trap clearance
        self.clear_gap = 0.27


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)

        # ================ 1. DEFINE ROOM BOUNDARIES ==================
        room_min_x = 0.0
        room_max_x = 14.0
        room_min_y = 0.0
        room_max_y = 20.0

        # Calculate the Safe Zone (Shrink room by the robot/obstacle radius)
        safe_min_x = room_min_x + self.robot_radius
        safe_max_x = room_max_x - self.robot_radius
        safe_min_y = room_min_y + self.robot_radius
        safe_max_y = room_max_y - self.robot_radius

        # ================ Move robot in safe zone ==================
        self.robot_pose = np.array([
            self.np_random.uniform(room_min_x, room_max_x),
            self.np_random.uniform(room_min_y, room_max_y),
            self.np_random.uniform(-np.pi, np.pi)
        ])

        dist_to_left_wall = self.robot_pose[0] - 0.0
        dist_to_right_wall = 14.0 - self.robot_pose[0]
        dist_to_bottom_wall = self.robot_pose[1] - 0.0
        dist_to_top_wall = 20.0 - self.robot_pose[1]

        # ================ Safe zone for obstacles ==================
        obs_min_x = room_min_x + self.obstacle_radius
        obs_max_x = room_max_x / 2 - self.obstacle_radius
        obs_min_y = room_min_y + self.obstacle_radius
        obs_max_y = room_max_y - self.obstacle_radius

        all_obs_coords = []

        # ==============  Generating 15 random gap traps (placed in bottom room) ==============

        x_coords_gap = [8.0, 10.5, 13.0]
        y_coords_gap = [1.2, 3.6, 6.0, 8.4, 8.8]

        # Center-to-center distance
        # 0.25 + 0.25 + 0.27 = 0.77 m
        dist_cc = 0.50 + self.clear_gap

        for y in y_coords_gap:
            for x in x_coords_gap:

                beta = np.random.uniform(0, 2 * np.pi)

                p1 = [
                    round(x - (dist_cc / 2) * np.cos(beta), 3),
                    round(y - (dist_cc / 2) * np.sin(beta), 3)
                ]
                p2 = [
                    round(x + (dist_cc / 2) * np.cos(beta), 3),
                    round(y + (dist_cc / 2) * np.sin(beta), 3)
                ]

                all_obs_coords.extend([p1, p2])

        # ============ Generating 12 V traps (placed in top room) ===============
        x_coords_v = [8.2, 10.5, 12.8]
        y_coords_v = [11.5, 14.167, 16.833, 18.5]

        angle_rad = np.radians(40)

        # Touching 0.25m radius obstacles
        dist_step = 0.50 

        for y in y_coords_v:
            for x in x_coords_v:

                alpha = np.random.uniform(0, 2 * np.pi)

                p0 = [round(x, 3), round(y, 3)]

                p1 = [
                    round(x + dist_step * np.cos(alpha + angle_rad), 3),
                    round(y + dist_step * np.sin(alpha + angle_rad), 3)
                ]
                p2 = [
                    round(x + 2 * dist_step * np.cos(alpha + angle_rad), 3),
                    round(y + 2 * dist_step * np.sin(alpha + angle_rad), 3)
                ]
                p3 = [
                    round(x + dist_step * np.cos(alpha - angle_rad), 3),
                    round(y + dist_step * np.sin(alpha - angle_rad), 3)
                ]
                p4 = [
                    round(x + 2 * dist_step * np.cos(alpha - angle_rad), 3),
                    round(y + 2 * dist_step * np.sin(alpha - angle_rad), 3)
                ]

                all_obs_coords.extend([p0, p1, p2, p3, p4])
        
        normal_obstacles = self.np_random.uniform(
            low=[obs_min_x, obs_min_y], 
            high=[obs_max_x, obs_max_y], 
            size=(5, 2)
        )

        # combine all obstacles
        all_obs_coords.extend(normal_obstacles.tolist())

        self.obstacles = np.array(all_obs_coords)

        self.goal_pose = np.array([
            self.np_random.uniform(safe_min_x, safe_max_x),
            self.np_random.uniform(safe_min_y, safe_max_y)
        ])
        
        # Ensure goal is at least 0.5 meters away from the robot's starting position
        while np.linalg.norm(self.robot_pose[:2] - self.goal_pose) < 0.5:
            self.goal_pose = np.array([
                self.np_random.uniform(safe_min_x, safe_max_x),
                self.np_random.uniform(safe_min_y, safe_max_y)
            ])

        # checking if the robot collides with walls
        while (dist_to_left_wall <= self.robot_radius or 
              dist_to_right_wall <= self.robot_radius or 
              dist_to_bottom_wall <= self.robot_radius or 
              dist_to_top_wall <= self.robot_radius):
            
            self.robot_pose = np.array([
                self.np_random.uniform(room_min_x, room_max_x),
                self.np_random.uniform(room_min_y, room_max_y),
                self.np_random.uniform(-np.pi, np.pi)
            ])

            dist_to_left_wall = self.robot_pose[0] - 0.0
            dist_to_right_wall = 14.0 - self.robot_pose[0]
            dist_to_bottom_wall = self.robot_pose[1] - 0.0
            dist_to_top_wall = 20.0 - self.robot_pose[1]

        # ================ Finalize reset ==================
        self.prev_dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pose)
        self.pose_history.clear() # clear buffer for next episode

        observation = self.get_obs()
        info = self.get_info()

        return observation, info
    
    def calculate_reward(self, lidar_scans, cur_dist_to_goal, terminated, stagnation_penalty):
        
        # Progress component
        delta_dist = self.prev_dist_to_goal - cur_dist_to_goal
        reward = -0.01 
        reward += delta_dist * 15.0
        self.prev_dist_to_goal = cur_dist_to_goal
        
        # Crash with obstacle rule
        if terminated and cur_dist_to_goal > 0.20:
            return -150.0
            
        # Success rule
        if cur_dist_to_goal < 0.05:
            return 200.0 
            
        # Danger zone warning
        if np.min(lidar_scans) < 0.15:
            reward -= 50.0  # -ve reward for getting too close to walls

        reward += stagnation_penalty  # add stagnation penalty
            
        return reward
    
    def step(self, action):

        truncated = False
        terminated = False

        linear_vel = action[0]
        angular_vel = action[1]

        # kinematics equations for differential drive robot
        self.robot_pose[0] += linear_vel * np.cos(self.robot_pose[2]) * self.dt
        self.robot_pose[1] += linear_vel * np.sin(self.robot_pose[2]) * self.dt
        self.robot_pose[2] += angular_vel * self.dt
        self.robot_pose[2] = (self.robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

        dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pose)
        reached_target = dist_to_goal < 0.20

        lidar_scans = self.simulate_lidar()

        self.pose_history.append(self.robot_pose[:2].copy())

        if (len(self.pose_history) == 30):

            dist_to_past = np.linalg.norm(self.robot_pose[:2] - self.pose_history[0])
            if dist_to_past < 0.15:
                self.stagnation_penalty = -0.5
            else: self.stagnation_penalty = 0.0


        dist_to_obstacle = np.linalg.norm(self.obstacles - self.robot_pose[:2], axis=1)
        is_collided = np.any(dist_to_obstacle < (self.robot_radius + self.obstacle_radius))

        # Wall collision check
        out_of_bounds = (
            self.robot_pose[0] < self.robot_radius or 
            self.robot_pose[0] > (7.0 - self.robot_radius) or 
            self.robot_pose[1] < self.robot_radius or 
            self.robot_pose[1] > (10.0 - self.robot_radius)
        )

        if reached_target or is_collided or out_of_bounds:
            terminated = True

        reward = self.calculate_reward(lidar_scans, dist_to_goal, terminated, self.stagnation_penalty)

        observation = self.get_obs()
        return observation, reward, terminated, truncated, {}

    def simulate_lidar(self):
        """
        Simulates 24 LiDAR beams. 
        Implements raycasting math against self.obstacles.
        """
        num_lidar_beams = 24
        lidar_range = 5.0
        lidar_scans = np.full(num_lidar_beams, 1.0, dtype=np.float32)
        _, _, theta = self.robot_pose
        angles = np.linspace(0, 2*np.pi, num_lidar_beams, endpoint=False)

        for ray_idx in range(num_lidar_beams):
            # Angle of the ray wrt the world
            ray_angle = angles[ray_idx]

            ray_direction = np.array([np.cos(theta + ray_angle), 
                                      np.sin(theta + ray_angle)])
            
            # Track the closest obstacle hit by this specific ray
            min_hit_dist = lidar_range 
            
            # Checking every ray against each obstacle
            for obs in self.obstacles:
                vector_to_obstacle = obs - self.robot_pose[:2]
                
                # projection_dist: how far along the ray the obstacle sits
                projection_dist = np.dot(vector_to_obstacle, ray_direction)
                
                # If projection is negative, the obstacle is physically behind the laser
                if projection_dist <= 0:
                    continue
                
                # Get the closest (x,y) point
                closest_point_on_ray = projection_dist * ray_direction
                perpendicular_dist = np.linalg.norm(vector_to_obstacle - closest_point_on_ray)

                # detecting whether a ray hits the obstacle or not
                if perpendicular_dist < self.obstacle_radius:
                    hit_distance = projection_dist - np.sqrt(self.obstacle_radius**2 - perpendicular_dist**2)
                    
                    if 0.0 < hit_distance < min_hit_dist:
                        min_hit_dist = hit_distance

            # Normalize the final distance between 0 & 1
            lidar_scans[ray_idx] = np.clip(min_hit_dist / lidar_range, 0.0, 1.0)
            
        return lidar_scans

    def get_obs(self):
        # Package the 26 inputs expected by observation space
        lidar = self.simulate_lidar()
        
        dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pose)
        angle_to_goal = np.arctan2(self.goal_pose[1] - self.robot_pose[1], 
                                   self.goal_pose[0] - self.robot_pose[0]) - self.robot_pose[2]
        
        # Normalize the relative angle to [-pi, pi]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        # Normalize inputs mathematically to [0.0, 1.0] for the neural network
        norm_dist = np.clip(dist_to_goal / 5.0, 0.0, 1.0) # Assuming an arbitrary 5m max distance
        norm_angle = (angle_to_goal + np.pi) / (2 * np.pi)
        
        return np.concatenate([lidar, [norm_dist, norm_angle]]).astype(np.float32)

    def get_info(self):
        return {"distance_to_goal": np.linalg.norm(self.robot_pose[:2] - self.goal_pose)}