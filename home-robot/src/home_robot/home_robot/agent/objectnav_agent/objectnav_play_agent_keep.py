# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple
import math
import numpy as np
import torch
from torch.nn import DataParallel
import habitat_sim
import home_robot.utils.pose as pu
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.navigation_planner.discrete_planner_play import DiscretePlanner

from .objectnav_agent_module import ObjectNavAgentModule
import matplotlib.pyplot as plt
import numpy as np
from numpy import array,float32
import cv2
import os
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
import pickle
import random
# For visualizing exploration issues
debug_frontier_map = False


class ObjectNavAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(
        self,
        config,
        device_id: int = 0,
        min_goal_distance_cm: float = 50.0,
        continuous_angle_tolerance: float = 30.0,
    ):
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                instance_association=getattr(
                    config.AGENT.SEMANTIC_MAP, "instance_association", "map_overlap"
                ),
                debug_visualize=config.PRINT_IMAGES,
            )

        self._module = ObjectNavAgentModule(
            config, instance_memory=self.instance_memory
        )

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=min_goal_distance_cm,
            continuous_angle_tolerance=continuous_angle_tolerance,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.verbose = config.AGENT.PLANNER.verbose

        self.evaluate_instance_tracking = getattr(
            config.ENVIRONMENT, "evaluate_instance_tracking", False
        )
        self.one_hot_instance_encoding = None
        if self.evaluate_instance_tracking:
            self.one_hot_instance_encoding = torch.eye(
                config.AGENT.SEMANTIC_MAP.max_instances + 1, device=self.device
            )
        self.config = config
        
        self.rgb_keep = []
        self.depth_keep = []
        self.turn_num  = 0

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        object_goal_category: torch.Tensor = None,
        start_recep_goal_category: torch.Tensor = None,
        end_recep_goal_category: torch.Tensor = None,
        nav_to_recep: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
        the main inference function of the agent that lets it interact with
        vectorized environments.

        This function assumes that the agent has been initialized.

        Args:
            obs: current frame containing (RGB, depth, segmentation) of shape
             (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (num_environments, 3)
            object_goal_category: semantic category of small object goals
            start_recep_goal_category: semantic category of start receptacle goals
            end_recep_goal_category: semantic category of end receptacle goals
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

        Returns:
            planner_inputs: list of num_environments planner inputs dicts containing
                obstacle_map: (M, M) binary np.ndarray local obstacle map
                 prediction
                sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                 and local map boundaries planning window (gx1, gx2, gy1, gy2)
                goal_map: (M, M) binary np.ndarray denoting goal location
            vis_inputs: list of num_environments visualization info dicts containing
                explored_map: (M, M) binary np.ndarray local explored map
                 prediction
                semantic_map: (M, M) np.ndarray containing local semantic map
                 predictions
        """
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
        if start_recep_goal_category is not None:
            start_recep_goal_category = start_recep_goal_category.unsqueeze(1)
        if end_recep_goal_category is not None:
            end_recep_goal_category = end_recep_goal_category.unsqueeze(1)
        (
            goal_map,
            found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            seq_object_goal_category=object_goal_category,
            seq_start_recep_goal_category=start_recep_goal_category,
            seq_end_recep_goal_category=end_recep_goal_category,
            seq_nav_to_recep=nav_to_recep,
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = goal_map.squeeze(1).cpu().numpy()
        found_goal = found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            if found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[e] = self.goal_update_steps
            self.timesteps[e] = self.timesteps[e] + 1
            self.timesteps_before_goal_update[e] = (
                self.timesteps_before_goal_update[e] - 1
            )

        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(131)
            plt.imshow(self.semantic_map.get_frontier_map(e))
            plt.subplot(132)
            plt.imshow(frontier_map[e][0])
            plt.subplot(133)
            plt.imshow(self.semantic_map.get_goal_map(e))
            plt.show()

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]
        
        if self.visualize:
            vis_inputs = [
                {
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "timestep": self.timesteps[e],
                }
                for e in range(self.num_environments)
            ]
            if self.record_instance_ids:
                for e in range(self.num_environments):
                    vis_inputs[e]["instance_map"] = self.semantic_map.get_instance_map(
                        e
                    )
        else:
            vis_inputs = [{} for e in range(self.num_environments)]
        return planner_inputs, vis_inputs

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        if self.record_instance_ids:
            self.instance_memory.reset()
        self.planner.reset()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        if self.record_instance_ids:
            self.instance_memory.reset_for_env(e)
        self.planner.reset()

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()
        self.rgb_keep = []
        self.depth_keep = []
        self.turn_num  = 0
        self.hidden_state = None
        if self.verbose:
            print("ObjectNavAgent reset")

    def get_nav_to_recep(self):
        return None
    
    def plot_path(self,path):
        # 提取 x 和 z 坐标
        x_coords = [point[0] for point in path]
        z_coords = [point[1] for point in path]

        # 创建图形并绘制路径
        plt.figure(figsize=(8, 6))
        plt.plot(x_coords, z_coords, marker='o', linestyle='-', color='b')
        plt.axis('equal')

        # 标注起点和终点
        plt.text(x_coords[0], z_coords[0], 'Start', fontsize=12, color='green', verticalalignment='bottom', horizontalalignment='right')
        plt.text(x_coords[-1], z_coords[-1], 'End', fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='left')

        # 高亮起点和终点
        plt.scatter(x_coords[0], z_coords[0], color='green', s=100, label='Start')
        plt.scatter(x_coords[-1], z_coords[-1], color='red', s=100, label='End')

        # 添加图形标签和标题
        plt.title('Path Visualization')
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')
        plt.grid(True)

        # 显示图形
        # plt.show()

        # 保存图像到内存而不是文件
        import io
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        # 获取当前图像
        fig = plt.gcf()

        # 使用 FigureCanvas 将 matplotlib 图像保存为 NumPy 数组
        canvas = FigureCanvas(fig)
        canvas.draw()

        # 获取图像尺寸并转换为 NumPy 数组
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))

        # 关闭 matplotlib 图像以避免干扰
        plt.close()

        # 将 RGB 转换为 BGR 格式（OpenCV 使用 BGR）
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image_bgr

    def sim_pos2gps_pos(self,obs,pos):
        relative_agent_position = quaternion_rotate_vector(
            obs.rotation_world_start.inverse(), pos - obs.origin
        )
        return np.array(
            [relative_agent_position[0], -relative_agent_position[2]], dtype=np.float32
        )
    
    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        # t0 = time.time()
        
        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            goal_name,
            camera_pose,
        ) = self._preprocess_obs(obs)

        # t1 = time.time()
        # print(f"[Agent] Obs preprocessing time: {t1 - t0:.2f}")

        # 2 - Semantic mapping + policy
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            object_goal_category=object_goal_category,
            start_recep_goal_category=start_recep_goal_category,
            end_recep_goal_category=end_recep_goal_category,
            camera_pose=camera_pose,
            nav_to_recep=None,
        )
        # the goal map in plannner_inputs is just fontier map (processed)

        # t2 = time.time()
        # print(f"[Agent] Semantic mapping and policy time: {t2 - t1:.2f}")
        
        now_sim = obs.now_sim
        agent_pos = now_sim.robot.base_pos
        candidate_objects = obs.episode.candidate_objects_hard[0]
        agent_target_pos = candidate_objects.view_points[0].agent_state.position
        
        view_points_list = [self.sim_pos2gps_pos(obs,point_tmp.agent_state.position) for point_tmp in candidate_objects.view_points]
        # print("view_points_list:",view_points_list)
        
        if len(obs.task.nav_goal_pos.shape) == 1:
            goals = np.expand_dims(obs.task.nav_goal_pos, axis=0)
        else:
            goals = obs.task.nav_goal_pos
        dis_target = obs.now_sim.geodesic_distance(
            obs.now_sim.robot.base_pos,
            goals,
            episode=None,
        )
        print("dis_target:",dis_target)
            
        # candidate_start_receps = obs.episode.candidate_start_receps
        path = self.get_oracle_action(obs,agent_pos,candidate_objects.position)
        # cv2.imshow("Path",self.plot_path(path))
        relative_stg = [path[-1][0] - path[0][0], path[-1][1] - path[0][1]]
        # if len(path)==2 and np.linalg.norm(relative_stg)<1:
        #     relative_stg = []
        # if len(path)-2 > 0:
        #     dist = 0
        #     for i in range(len(path)-1):
        #         dist = dist + np.linalg.norm([path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]])
        #         if dist > 1:
        #             relative_stg = [path[i+1][0] - path[0][0], path[i+1][1] - path[0][1]]
        #             break

        # 3 - Planning
        closest_goal_map = None
        short_term_goal = None
        dilated_obstacle_map = None
        action = None
        save_flag = False
        self.max_steps = 800
        if planner_inputs[0]["found_goal"]:
            self.episode_panorama_start_steps = 0
        if self.timesteps[0] < self.episode_panorama_start_steps:
            action = DiscreteNavigationAction.TURN_RIGHT
        elif self.timesteps[0] > self.max_steps:
            action = DiscreteNavigationAction.STOP
        else:
            (
                action,
                closest_goal_map,
                short_term_goal,
                dilated_obstacle_map,
                goal_map_save,
            ) = self.planner.plan(
                **planner_inputs[0],
                use_dilation_for_stg=self.use_dilation_for_stg,
                timestep=self.timesteps[0],
                debug=self.verbose,
                relative_stg = relative_stg,
                dis_target = dis_target,
            )
            save_flag = True
        
        terminate = False
        if dis_target < 0.1:
            ovmm_find_object_phase_success = obs.task.measurements.measures["ovmm_find_object_phase_success"].get_metric()
            if ovmm_find_object_phase_success:
                action = DiscreteNavigationAction.STOP 
            # elif not planner_inputs[0]["found_goal"]:
            # else:
            elif action == DiscreteNavigationAction.STOP or self.turn_num >0:
                self.turn_num += 1
                action = DiscreteNavigationAction.TURN_RIGHT 
                if self.turn_num > 12:
                    terminate = True
                    
        else:
            self.turn_num  = 0
            if action == DiscreteNavigationAction.STOP:
                action = DiscreteNavigationAction.MOVE_FORWARD_SMALL 
            elif action == DiscreteNavigationAction.MOVE_FORWARD and dis_target< 0.2:
                action = DiscreteNavigationAction.MOVE_FORWARD_SMALL 
        
        # self.rgb_keep.append(obs.rgb)
        # self.depth_keep.append(obs.depth)
        # if len(self.rgb_keep) > 3:
        #     self.rgb_keep.pop(0)
        #     self.depth_keep.pop(0)
            
        # target_pos = planner_inputs[0]["sensor_pose"][0:2]+relative_stg
        # goal_iou = (obs.semantic==1).astype(np.uint8)
        # dd_map  = self.planner.dd
        # if save_flag and goal_map_save is not None and action is not None and dd_map is not None:
        #     if len(np.unique(dd_map)) > 10000:
        #         if closest_goal_map is None:
        #             closest_goal_map = goal_map_save
        #         closest_goal_map = closest_goal_map.astype(np.float32)
        #         save_data = {
        #             "local_map": self.semantic_map.local_map[0].cpu().numpy(),  # NumPy 数组
        #             "goal_map": goal_map_save,              # NumPy 数组
        #             "closest_goal_map": closest_goal_map,
        #             "action": action,                  # 枚举值或字符串
        #             "goal_name": obs.task_observations["goal_name"],  # 字符串
        #             "rgb": self.rgb_keep,
        #             "depth": self.depth_keep,
        #             "sensor_pose": planner_inputs[0]["sensor_pose"],
        #             "target_pos":target_pos,
        #             "goal_iou":goal_iou,
        #             "dd_map":dd_map
        #         }

        #     # 保存到本地文件
        #     save_name = os.path.join(self.planner.local_map_dir, f"record_{self.timesteps[0]}.pkl")
        #     with open(save_name, 'wb') as f:
        #         pickle.dump(save_data, f)
        
    
        vis_inputs[0]["goal_name"] = obs.task_observations["goal_name"]
        if self.visualize:
            vis_inputs[0]["semantic_frame"] = obs.task_observations["semantic_frame"]
            vis_inputs[0]["closest_goal_map"] = closest_goal_map
            vis_inputs[0]["third_person_image"] = obs.third_person_image
            vis_inputs[0]["short_term_goal"] = None
            vis_inputs[0]["dilated_obstacle_map"] = dilated_obstacle_map
            vis_inputs[0]["semantic_map_config"] = self.config.AGENT.SEMANTIC_MAP
            vis_inputs[0]["instance_memory"] = self.instance_memory
        info = {**planner_inputs[0], **vis_inputs[0]}
        return action, info, terminate
    
    def get_finer_waypoints(self,path_points, min_distance=1):
        finer_points = []
        for i in range(len(path_points) - 1):
            if i < 5:
                start = np.array(path_points[i])
                end = np.array(path_points[i + 1])
                dist = np.linalg.norm(end - start)
                
                # 在两个点之间插入更多的waypoints
                num_points = max(int(dist / min_distance), 1)  # 按min_distance决定插入点数量
                for j in range(num_points):
                    finer_point = start + j / num_points * (end - start)
                    finer_points.append(finer_point)
            else:
                finer_points.append(path_points[i])
        
        finer_points.append(path_points[-1])  # 加上最后的终点
        return finer_points
    
    def safe2obstacle(self,point,now_sim,safety_distance):
        for  x in [-1,0,1]:
            for y in [-1,0,1]:
                point_move = point * 1.0
                point_move[0] = point[0]+x*safety_distance
                point_move[2] = point[2]+y*safety_distance
                if not now_sim.pathfinder.is_navigable(point_move):
                    return False
        return True

    def adjust_point(self,point,now_sim, safety_distance):
        for  x in [-1,0,1]:
            for y in [-1,0,1]:
                point_move = point * 1.0
                point_move[0] = point[0]+x*safety_distance
                point_move[2] = point[2]+y*safety_distance
                if self.safe2obstacle(point_move,now_sim,safety_distance):
                    return point_move
        return None

    def adjust_path_for_obstacles(self,path_points, now_sim, safety_distance=0.3):
        adjusted_points = [path_points[0]]
        for i in range(len(path_points)-1):
            point = path_points[i+1]
            if i < 2:
                # 如果某点太接近障碍物，稍微调整这个点的位置
                if self.safe2obstacle(point,now_sim,safety_distance):
                    adjusted_points.append(point)
                else:
                    # slight mode point
                    point_move = self.adjust_point(point,now_sim,safety_distance)
                    if point_move is not None:
                        adjusted_points.append(point_move)
            else:
                adjusted_points.append(point)
        return adjusted_points
    
    def get_oracle_action(self, obs: Observations, agent_pos, point):
        now_sim = obs.now_sim
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = now_sim.pathfinder.find_path(path)
        if not found_path:
            path = [agent_pos, point]
        path = path.points
        path = [self.sim_pos2gps_pos(obs,point_tmp) for point_tmp in path]
        path = self.get_finer_waypoints(path)
        # path = self.adjust_path_for_obstacles(path,now_sim)
        return path

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm
        if self.store_all_categories_in_map:
            # semantic = obs.semantic
            # obj_goal_idx = obs.task_observations["object_goal"]
            # if "start_recep_goal" in obs.task_observations:
            #     start_recep_idx = obs.task_observations["start_recep_goal"]
            # if "end_recep_goal" in obs.task_observations:
            #     end_recep_idx = obs.task_observations["end_recep_goal"]
                
            semantic = np.zeros((obs.semantic.shape[0],obs.semantic.shape[1],self.config.AGENT.SEMANTIC_MAP.num_sem_categories))
            obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
            semantic[:,:,obj_goal_idx][
                obs.semantic == obs.task_observations["object_goal"]
            ] = 1
            if "start_recep_goal" in obs.task_observations:
                semantic[:,:,start_recep_idx][
                    obs.semantic == obs.task_observations["start_recep_goal"]
                ] = 1
            if "end_recep_goal" in obs.task_observations:
                semantic[:,:,end_recep_idx][
                    obs.semantic == obs.task_observations["end_recep_goal"]
                ] = 1
            for i in range(semantic.shape[2]):
                if i > 3:
                    semantic[:,:,i][obs.semantic==i-2] = 1
            semantic = torch.from_numpy(semantic).to(self.device)
        else:
            semantic = np.full_like(obs.semantic, 4)
            obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
            semantic[
                obs.semantic == obs.task_observations["object_goal"]
            ] = obj_goal_idx
            if "start_recep_goal" in obs.task_observations:
                semantic[
                    obs.semantic == obs.task_observations["start_recep_goal"]
                ] = start_recep_idx
            if "end_recep_goal" in obs.task_observations:
                semantic[
                    obs.semantic == obs.task_observations["end_recep_goal"]
                ] = end_recep_idx

            semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        if self.evaluate_instance_tracking:
            gt_instance_ids = (
                torch.from_numpy(obs.task_observations["gt_instance_ids"])
                .to(self.device)
                .long()
            )
            gt_instance_ids = self.one_hot_instance_encoding[gt_instance_ids]
            obs_preprocessed = torch.cat([obs_preprocessed, gt_instance_ids], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose
        object_goal_category = None
        end_recep_goal_category = None
        if (
            "object_goal" in obs.task_observations
            and obs.task_observations["object_goal"] is not None
        ):
            if self.verbose:
                print("object goal =", obs.task_observations["object_goal"])
            object_goal_category = torch.tensor(obj_goal_idx).unsqueeze(0)
        start_recep_goal_category = None
        if (
            "start_recep_goal" in obs.task_observations
            and obs.task_observations["start_recep_goal"] is not None
        ):
            if self.verbose:
                print(
                    "start_recep goal =",
                    obs.task_observations["start_recep_goal"],
                )
            start_recep_goal_category = torch.tensor(start_recep_idx).unsqueeze(0)
        if (
            "end_recep_goal" in obs.task_observations
            and obs.task_observations["end_recep_goal"] is not None
        ):
            if self.verbose:
                print("end_recep goal =", obs.task_observations["end_recep_goal"])
            end_recep_goal_category = torch.tensor(end_recep_idx).unsqueeze(0)
        goal_name = [obs.task_observations["goal_name"]]
        if self.verbose:
            print("[ObjectNav] Goal name: ", goal_name)

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            goal_name,
            camera_pose,
        )
