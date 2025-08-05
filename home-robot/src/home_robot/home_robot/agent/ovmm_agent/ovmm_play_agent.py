# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from datetime import datetime
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# from home_robot.agent.objectnav_agent.objectnav_play_action_agent_his import ObjectNavAgent
from home_robot.agent.objectnav_agent.objectnav_play_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ovmm_perception import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.manipulation import HeuristicPickPolicy, HeuristicPlacePolicy
from home_robot.perception.constants import RearrangeBasicCategories

import cv2


class Skill(IntEnum):
    NAV_TO_OBJ = auto()
    GAZE_AT_OBJ = auto()
    PICK = auto()
    NAV_TO_REC = auto()
    GAZE_AT_REC = auto()
    PLACE = auto()
    FALL_WAIT = auto()


class SemanticVocab(IntEnum):
    FULL = auto()
    SIMPLE = auto()
    ALL = auto()


def get_skill_as_one_hot_dict(curr_skill: Skill):
    skill_dict = {f"is_curr_skill_{skill.name}": 0 for skill in Skill}
    skill_dict[f"is_curr_skill_{Skill(curr_skill).name}"] = 1
    return skill_dict


class PlayOpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map."""

    def __init__(self, config, device_id: int = 0):
        super().__init__(config, device_id=device_id)
        self.states = None
        self.place_start_step = None
        self.pick_start_step = None
        self.gaze_at_obj_start_step = None
        self.fall_wait_start_step = None
        self.is_gaze_done = None
        self.place_done = None
        self.gaze_agent = None
        self.nav_to_obj_agent = None
        self.nav_to_rec_agent = None
        self.pick_agent = None
        self.place_agent = None
        self.pick_policy = None
        self.place_policy = None
        self.semantic_sensor = None 

        # if config.GROUND_TRUTH_SEMANTICS == 1 and self.store_all_categories_in_map:
        #     # currently we get ground truth semantics of only the target object category and all scene receptacles from the simulator
        #     raise NotImplementedError

        self.skip_skills = config.AGENT.skip_skills
        self.max_pick_attempts = 10
        if config.GROUND_TRUTH_SEMANTICS == 0:
            self.semantic_sensor = OvmmPerception(config, device_id, self.verbose)
            self.obj_name_to_id, self.rec_name_to_id = read_category_map_file(
                config.ENVIRONMENT.category_map_file
            )
        if (
            config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"
            and not self.skip_skills.nav_to_obj
        ):
            from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent

            self.nav_to_obj_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.NAV_TO_OBJ,
                device_id=device_id,
            )
        self._fall_wait_steps = getattr(config.AGENT, "fall_wait_steps", 0)
        self.config = config
        
        self.keys2actions = {
            'w':DiscreteNavigationAction.MOVE_FORWARD,
            'a':DiscreteNavigationAction.TURN_LEFT,
            'd':DiscreteNavigationAction.TURN_RIGHT,
            # DiscreteNavigationAction.EXTEND_ARM,
            # DiscreteNavigationAction.NAVIGATION_MODE,
            # DiscreteNavigationAction.MANIPULATION_MODE
            }

    def _get_info(self, obs: Observations) -> Dict[str, torch.Tensor]:
        """Get inputs for visual skill."""
        use_detic_viz = self.config.ENVIRONMENT.use_detic_viz

        if self.config.GROUND_TRUTH_SEMANTICS == 1 or use_detic_viz:
            semantic_category_mapping = None  # Visualizer handles mapping
        elif self.semantic_sensor.current_vocabulary_id == SemanticVocab.SIMPLE:
            semantic_category_mapping = RearrangeBasicCategories()
            # semantic_category_mapping = None  
        else:
            semantic_category_mapping = self.semantic_sensor.current_vocabulary
        if use_detic_viz:
            semantic_frame = obs.task_observations["semantic_frame"]
        else:
            semantic_frame = np.concatenate(
                [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
            ).astype(np.uint8)
        info = {
            "semantic_frame": semantic_frame,
            "semantic_category_mapping": semantic_category_mapping,
            "goal_name": obs.task_observations["goal_name"],
            "third_person_image": obs.third_person_image,
            "timestep": self.timesteps[0],
            "curr_skill": Skill(self.states[0].item()).name,
            "skill_done": "",  # Set if skill gets done
        }
        # only the current skill has corresponding value as 1
        info = {**info, **get_skill_as_one_hot_dict(self.states[0].item())}
        return info

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()

    def reset_vectorized(self):
        """Initialize agent state."""
        super().reset_vectorized()

        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized()
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized()
        if self.place_agent is not None:
            self.place_agent.reset_vectorized()
        if self.nav_to_rec_agent is not None:
            self.nav_to_rec_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.pick_start_step = torch.tensor([0] * self.num_environments)
        self.gaze_at_obj_start_step = torch.tensor([0] * self.num_environments)
        self.place_start_step = torch.tensor([0] * self.num_environments)
        self.gaze_at_obj_start_step = torch.tensor([0] * self.num_environments)
        self.fall_wait_start_step = torch.tensor([0] * self.num_environments)
        self.is_gaze_done = torch.tensor([0] * self.num_environments)
        self.place_done = torch.tensor([0] * self.num_environments)
        if self.place_policy is not None:
            self.place_policy.reset()
        if self.pick_policy is not None:
            self.pick_policy.reset()

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.place_start_step[e] = 0
        self.pick_start_step[e] = 0
        self.gaze_at_obj_start_step[e] = 0
        self.fall_wait_start_step[e] = 0
        self.is_gaze_done[e] = 0
        self.place_done[e] = 0
        if self.place_policy is not None:
            self.place_policy.reset()
        if self.pick_policy is not None:
            self.pick_policy.reset()
        super().reset_vectorized_for_env(e)
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized_for_env(e)
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized_for_env(e)
        if self.place_agent is not None:
            self.place_agent.reset_vectorized_for_env(e)
        if self.nav_to_rec_agent is not None:
            self.nav_to_rec_agent.reset_vectorized_for_env(e)

    def _init_episode(self, obs: Observations):
        """
        This method is called at the first timestep of every episode before any action is taken.
        """
        if self.verbose:
            print("Initializing episode...")
        if self.config.GROUND_TRUTH_SEMANTICS == 0:
            self._update_semantic_vocabs(obs)
            if self.store_all_categories_in_map:
                # self._set_semantic_vocab(SemanticVocab.ALL, force_set=True)
                self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=True)
            elif (
                self.config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"
                and not self.skip_skills.nav_to_obj
            ):
                self._set_semantic_vocab(SemanticVocab.FULL, force_set=True)
            else:
                self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=True)

    def _update_semantic_vocabs(
        self, obs: Observations, update_full_vocabulary: bool = True
    ):
        """
        Sets vocabularies for semantic sensor at the start of episode.
        Optional-
        :update_full_vocabulary: if False, only updates simple vocabulary
        True by default
        """
        obj_id_to_name = {
            0: obs.task_observations["object_name"],
        }
        simple_rec_id_to_name = {
            0: obs.task_observations["start_recep_name"],
            1: obs.task_observations["place_recep_name"],
        }

        # Simple vocabulary contains only object and necessary receptacles
        simple_vocab = build_vocab_from_category_map(
            obj_id_to_name, simple_rec_id_to_name
        )
        self.semantic_sensor.update_vocabulary_list(simple_vocab, SemanticVocab.SIMPLE)

        if update_full_vocabulary:
            # Full vocabulary contains the object and all receptacles
            full_vocab = build_vocab_from_category_map(
                obj_id_to_name, self.rec_name_to_id
            )
            self.semantic_sensor.update_vocabulary_list(full_vocab, SemanticVocab.FULL)

        # All vocabulary contains all objects and all receptacles
        all_vocab = build_vocab_from_category_map(
            self.obj_name_to_id, self.rec_name_to_id
        )
        self.semantic_sensor.update_vocabulary_list(all_vocab, SemanticVocab.ALL)

    def _set_semantic_vocab(self, vocab_id: SemanticVocab, force_set: bool):
        """
        Set active vocabulary for semantic sensor to use to the given ID.
        """
        if self.config.GROUND_TRUTH_SEMANTICS == 0 and (
            force_set or self.semantic_sensor.current_vocabulary_id != vocab_id
        ):
            self.semantic_sensor.set_vocabulary(vocab_id)
                

    def _heuristic_nav(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any]:
        action, planner_info, terminate = super().act(obs)
        # action, planner_info = super().act(obs)
        terminate = False
        # info overwrites planner_info entries for keys with same name
        info = {**planner_info, **info}
        self.timesteps[0] -= 1  # objectnav agent increments timestep
        info["timestep"] = self.timesteps[0]
        if action == DiscreteNavigationAction.STOP or terminate:
            terminate = True
        else:
            terminate = False
        return action, info, terminate

    """
    The following methods each correspond to a skill/state this agent can execute.
    They take sensor observations as input and return the action to take and
    the state to transition to. Either the action has a value and the new state doesn't,
    or the action has no value and the new state does. The latter case indicates
    a state transition.
    """

    def _nav_to_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        nav_to_obj_type = self.config.AGENT.SKILLS.NAV_TO_OBJ.type
        if self.skip_skills.nav_to_obj:
            terminate = True
        elif nav_to_obj_type == "heuristic":
            if self.verbose:
                print("[OVMM AGENT] step heuristic nav policy")
            action, info, terminate = self._heuristic_nav(obs, info)
        elif nav_to_obj_type == "rl":
            action, info, terminate = self.nav_to_obj_agent.act(obs, info)
        else:
            raise ValueError(
                f"Got unexpected value for NAV_TO_OBJ.type: {nav_to_obj_type}"
            )
        new_state = None
        if terminate:
            action = None
            new_state = Skill.GAZE_AT_OBJ
        return action, info, new_state
    
    def play_action(self,obs):
        cv2.imshow("RGB",cv2.cvtColor(obs.rgb,cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(0)  # 等待按键输入，不限制值在0-255之间  
        if key==27 or  key==126:  # Esc 键退出（ASCII码为27）  
            print("Exit")  
        action_name = self.keys2actions.get(chr(key & 0xFF), None)  # 对于ASCII字符，使用 & 0xFF  
        if action_name:
            print("action_name:",action_name)
        new_state = None
        return action_name, new_state

    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any], Observations]:
        """State machine"""
        if self.timesteps[0] == 0:
            self._init_episode(obs)

        if self.config.GROUND_TRUTH_SEMANTICS == 0:
            obs = self.semantic_sensor(obs) # with Detic to get semantic map
        else:
            obs.task_observations["semantic_frame"] = None
        info = self._get_info(obs)

        self.timesteps[0] += 1
        action = None
        print("self.timesteps[0]:",self.timesteps[0]," self.states[0]:",self.states[0])
        while action is None:
            # self.update_point_cloud(obs)
            if self.states[0] == Skill.NAV_TO_OBJ:
                action, info, new_state = self._nav_to_obj(obs, info)
                # action, new_state = self.play_action(obs)
            else:
                raise ValueError

            # Since heuristic nav is not properly vectorized, this agent currently only supports 1 env
            # _switch_to_next_skill is thus invoked with e=0
            if new_state:
                # mark the current skill as done
                info["skill_done"] = info["curr_skill"]
                assert (
                    action is None
                ), f"action must be None when switching states, found {action} instead"
                # action = self._switch_to_next_skill(0, new_state, info)
                action = DiscreteNavigationAction.STOP
        # update the curr skill to the new skill whose action will be executed
        info["curr_skill"] = Skill(self.states[0].item()).name
        if self.verbose:
            print(
                f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}'
            )
        return action, info, obs
