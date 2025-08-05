# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
import json
import os
import time
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from omegaconf import DictConfig
from tqdm import tqdm
from utils.env_utils import create_ovmm_env_fn
from utils.metrics_utils import get_stats_from_episode_metrics

if TYPE_CHECKING:
    from habitat.core.dataset import BaseEpisode
    from habitat.core.vector_env import VectorEnv

    from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
    from home_robot.core.abstract_agent import Agent

from scipy.ndimage import label 
class EvaluationType(Enum):
    LOCAL = "local"
    LOCAL_VECTORIZED = "local_vectorized"
    REMOTE = "remote"


class OVMMEvaluator(PPOTrainer):
    """Class for creating vectorized environments, evaluating OpenVocabManipAgent on an episode dataset and returning metrics"""

    def __init__(self, eval_config: DictConfig) -> None:
        self.metrics_save_freq = eval_config.EVAL_VECTORIZED.metrics_save_freq
        self.results_dir = os.path.join(
            eval_config.DUMP_LOCATION, "results", eval_config.EXP_NAME
        )
        self.videos_dir = eval_config.habitat_baselines.video_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        super().__init__(eval_config)

    def local_evaluate_vectorized(self, agent, num_episodes_per_env=10):
        self._init_envs(
            config=self.config, is_eval=True, make_env_fn=create_ovmm_env_fn
        )
        self._evaluate_vectorized(
            agent,
            self.envs,
            num_episodes_per_env=num_episodes_per_env,
        )

    def _summarize_metrics(self, episode_metrics: Dict) -> Dict:
        """Gets stats from episode metrics"""
        # convert to a dataframe
        episode_metrics_df = pd.DataFrame.from_dict(episode_metrics, orient="index")
        episode_metrics_df["start_idx"] = 0
        stats = get_stats_from_episode_metrics(episode_metrics_df)
        return stats

    def _print_summary(self, summary: dict):
        """Prints the summary of metrics"""
        print("=" * 50)
        print("Averaged metrics")
        print("=" * 50)
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("=" * 50)

    def _check_set_planner_vis_dir(
        self, agent: "Agent", current_episode: "BaseEpisode"
    ):
        """
        Sets vis_dir for storing planner's debug visualisations if the agent has a planner.
        """
        if hasattr(agent, "planner"):
            agent.planner.set_vis_dir(
                current_episode.scene_id.split("/")[-1].split(".")[0],
                current_episode.episode_id,
            )
            
            agent.planner.set_local_map_dir(
                current_episode.scene_id.split("/")[-1].split(".")[0],
                current_episode.episode_id,
            )

    def _evaluate_vectorized(
        self,
        agent: "OpenVocabManipAgent",
        envs: "VectorEnv",
        num_episodes_per_env=None,
    ):
        # The stopping condition is either specified through
        # num_episodes_per_env (stop after each environment
        # finishes a certain number of episodes)
        print(f"Running eval on {envs.number_of_episodes} episodes")

        if num_episodes_per_env is None:
            num_episodes_per_env = envs.number_of_episodes
        else:
            num_episodes_per_env = [num_episodes_per_env] * envs.num_envs

        episode_metrics = {}

        def stop():
            return all(
                [
                    episode_idxs[i] >= num_episodes_per_env[i]
                    for i in range(envs.num_envs)
                ]
            )

        start_time = time.time()
        episode_idxs = [0] * envs.num_envs
        obs = envs.call(["reset"] * envs.num_envs)

        agent.reset_vectorized()
        self._check_set_planner_vis_dir(agent, self.envs.current_episodes()[0])
        while not stop():
            current_episodes_info = self.envs.current_episodes()
            # TODO: Currently agent can work with only 1 env, Parallelize act across envs
            actions, infos, _ = zip(*[agent.act(ob) for ob in obs])

            outputs = envs.call(
                ["apply_action"] * envs.num_envs,
                [{"action": a, "info": i} for a, i in zip(actions, infos)],
            )

            obs, dones, hab_infos = [list(x) for x in zip(*outputs)]
            for e, (done, info, hab_info) in enumerate(zip(dones, infos, hab_infos)):
                episode_key = (
                    f"{current_episodes_info[e].scene_id.split('/')[-1].split('.')[0]}_"
                    f"{current_episodes_info[e].episode_id}"
                )
                if episode_key not in episode_metrics:
                    episode_metrics[episode_key] = {}
                # Record metrics after each skill finishes. This is useful for debugging.
                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    episode_metrics[episode_key] = {
                        **metrics_at_skill_end,
                        **episode_metrics[episode_key],
                    }
                    if "goal_name" in episode_metrics[episode_key]:
                        episode_metrics[episode_key]["goal_name"] = info["goal_name"]
                if done:  # environment times out
                    metrics = self._extract_scalars_from_info(hab_info)
                    if episode_idxs[e] < num_episodes_per_env[e]:
                        metrics_at_episode_end = {
                            f"END." + k: v for k, v in metrics.items()
                        }
                        episode_metrics[episode_key] = {
                            **metrics_at_episode_end,
                            **episode_metrics[episode_key],
                        }
                        if "goal_name" in episode_metrics[episode_key]:
                            episode_metrics[episode_key]["goal_name"] = info[
                                "goal_name"
                            ]
                        episode_idxs[e] += 1
                        print(
                            f"Episode indexes {episode_idxs[e]} / {num_episodes_per_env[e]} "
                            f"after {round(time.time() - start_time, 2)} seconds"
                        )
                    if len(episode_metrics) % self.metrics_save_freq == 0:
                        aggregated_metrics = self._aggregate_metrics(episode_metrics)
                        self._write_results(episode_metrics, aggregated_metrics)
                    if not stop():
                        obs[e] = envs.call_at(e, "reset")
                        agent.reset_vectorized_for_env(e)
                        self._check_set_planner_vis_dir(
                            envs, envs.current_episodes()[e]
                        )

        envs.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def _aggregate_metrics(self, episode_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Aggregates metrics tracked by environment."""
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [
                k
                for metrics_per_episode in episode_metrics.values()
                for k in metrics_per_episode
                if k != "goal_name"
            ]
        )
        for v in episode_metrics.values():
            for k in metrics:
                if k in v:
                    aggregated_metrics[f"{k}/total"].append(v[k])

        aggregated_metrics = dict(
            sorted(
                {
                    k2: v2
                    for k1, v1 in aggregated_metrics.items()
                    for k2, v2 in {
                        f"{k1}/mean": np.mean(v1),
                        f"{k1}/min": np.min(v1),
                        f"{k1}/max": np.max(v1),
                    }.items()
                }.items()
            )
        )

        return aggregated_metrics

    def _write_results(
        self, episode_metrics: Dict[str, Dict], aggregated_metrics: Dict[str, float]
    ) -> None:
        """Writes metrics tracked by environment to a file."""
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)
    
    def collect_info(self, observations, episode, step):
        save_dir = "/aiarena/gpfs/code/code/OVMM/home-robot/train_data/collect_rep_obj_a12_large_11_13/"
        img_dir = save_dir + "images/" + episode
        label_dir = save_dir + "labels/" + episode
        if not os.path.exists(img_dir):  
            os.makedirs(img_dir)
        if not os.path.exists(label_dir):  
            os.makedirs(label_dir)
        
        rgb_image = observations.rgb
        semantic = observations.semantic
        semantic_obj = observations.semantic_obj

        # Get bounding boxes and class labels from semantic observations
        unique_recep = np.unique(semantic)
        boxes = []
        labels = []
        for obj_id in unique_recep:
            # Mask for current object
            if obj_id > 1 and obj_id < 23:
                mask = (semantic == obj_id)
                
                # Perform connected component analysis
                labeled_mask, num_features = label(mask)
                
                for instance_id in range(1, num_features + 1):
                    instance_mask = (labeled_mask == instance_id)
                    if instance_mask.sum() < 1000:
                        continue
                    ys, xs = np.where(instance_mask)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    if 0.1 < (x_max-x_min)/(y_max-y_min) < 10:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(obj_id-2)  # Use the object ID as the class indx
        unique_obj = np.unique(semantic_obj)
        for obj_id in unique_obj:
            # Mask for current object
            if obj_id > 0 and obj_id < 109:
                mask = (semantic_obj == obj_id)
                
                # Perform connected component analysis
                labeled_mask, num_features = label(mask)
                
                for instance_id in range(1, num_features + 1):
                    instance_mask = (labeled_mask == instance_id)
                    if instance_mask.sum() < 50:
                        continue
                    ys, xs = np.where(instance_mask)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    if 0.1 < (x_max-x_min)/(y_max-y_min) < 10:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(obj_id+20)  # Use the object ID as the class indx

        label_filename = os.path.join(label_dir, f"frame_{episode}_{step:04d}.txt")
        img_width=rgb_image.shape[1]
        img_height=rgb_image.shape[0]
        if len(labels) > 0:
            # Save RGB image
            img_filename = os.path.join(img_dir, f"frame_{episode}_{step:04d}.jpg")
            img = Image.fromarray(rgb_image)
            img.save(img_filename)
            
            with open(label_filename, 'w') as f:
                for box, indx in zip(boxes, labels):
                    # YOLO format requires center x, center y, width, and height normalized
                    x_center = (box[0] + box[2]) / 2.0 / img_width
                    y_center = (box[1] + box[3]) / 2.0 / img_height
                    # x_min = box[0] / img_width
                    # y_min = box[1] / img_height
                    box_width = (box[2] - box[0]) / img_width
                    box_height = (box[3] - box[1]) / img_height
                    f.write(f"{indx} {x_center} {y_center} {box_width} {box_height}\n")

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the local environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        env_num_episodes = self._env.number_of_episodes
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0
        
        scene_id_dict = {}

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = self._env.reset(), False
            current_episode = self._env.get_current_episode()
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)
            
            print("scene_id_dict.keys():",scene_id_dict.keys())
            if current_episode.scene_id in scene_id_dict.keys():
                if scene_id_dict[current_episode.scene_id] > 4:
                    count_episodes += 1
                    pbar.update(1)
                    continue 
                else:
                    scene_id_dict[current_episode.scene_id] += 1
            else:
                scene_id_dict[current_episode.scene_id] = 1

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            print("current_episode_key:",current_episode_key)
            # if '104348361_171513414_5' != current_episode_key:
            # if '106366410_174226806_7' != current_episode_key:
            #     count_episodes += 1
            #     pbar.update(1)
            #     continue 
            
            current_episode_metrics = {}
            step = 0
            while not done:
                action, info, _  = agent.act(observations)
                
                # self.collect_info(observations,current_episode_key,step)
                # if collect_over_flag:
                #     break
                
                observations, done, hab_info = self._env.apply_action(action, info)
                step = step + 1
                if step > 800:  # 收集大概收集800步就够探索差不多了
                    done = True

                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = self._extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        self._env.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
        evaluation_type: str = "local",
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        if evaluation_type == EvaluationType.LOCAL.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate(agent, num_episodes)
        elif evaluation_type == EvaluationType.LOCAL_VECTORIZED.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate_vectorized(agent, num_episodes)
        elif evaluation_type == EvaluationType.REMOTE.value:
            self._env = None
            return self.remote_evaluate(agent, num_episodes)
        else:
            raise ValueError(
                "Invalid evaluation type. Please choose from 'local', 'local_vectorized', 'remote'"
            )
