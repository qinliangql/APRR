import cv2
import numpy as np
from typing import List, Optional, Tuple

from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
from home_robot.perception.detection.utils import filter_depth, overlay_masks
from ultralytics import YOLOWorld, SAM
import matplotlib.pyplot as plt
import time

class YolowSAMPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        vocabulary="custom",
        custom_vocabulary="",
        checkpoint_file=None,
        sem_gpu_id='cuda:0',
        verbose: bool = False,
    ):
        
        self.yolow_model = YOLOWorld("/aiarena/gpfs/code/code/OVMM/home-robot/mytest/runs/detect/train5/weights/best.pt").to(sem_gpu_id)
        # self.yolow_model = YOLOWorld("/aiarena/gpfs/code/code/OVMM/home-robot/mytest/runs/detect/train6/weights/best.pt").to(sem_gpu_id)
        # self.yolow_model.confidence_thres=0.5

        self.sam2_model = SAM("/aiarena/gpfs/code/code/OVMM/home-robot/mytest/sam2_b.pt").to(sem_gpu_id)
        
    def reset_vocab(self, new_vocab: List[str], vocab_type="custom"):
        # self.yolow_model.set_classes(new_vocab) # for example: new_vocab: ['.', 'mouse_pad', 'chest_of_drawers', 'table', 'other']
        self.yolow_model.set_classes(new_vocab[1:-1])

        # vocab_dict = {0: 'bathtub', 1: 'bed', 2: 'bench', 3: 'cabinet', 4: 'chair', 5: 'chest_of_drawers', 6: 'couch', 7: 'counter', 8: 'filing_cabinet', 9: 'hamper', 10: 'serving_cart', 11: 'shelves', 12: 'shoe_rack', 13: 'sink', 14: 'stand', 15: 'stool', 16: 'table', 17: 'toilet', 18: 'trunk', 19: 'wardrobe', 20: 'washer_dryer'}
        # vocab = new_vocab[1:2] + [vocab_dict[id] for id in vocab_dict]
        # self.recep_index = [vocab.index(new_vocab[2]),vocab.index(new_vocab[3])]
        
        # self.yolow_model.set_classes(vocab)
        
         
    
    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        image = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
        # image = obs.rgb
        depth = obs.depth
        height, width, _ = image.shape
        
        time1 = time.time()
        yolow_results = self.yolow_model.predict(image,verbose=False,conf=0.3)
        # print("box time:",time.time()-time1)
        
        class_idcs = yolow_results[0].boxes.cls.cpu().numpy() + 1
        scores = yolow_results[0].boxes.conf.cpu().numpy()
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if len(scores) > 0:
            sam2_results = self.sam2_model(image, bboxes=yolow_results[0].boxes.xyxy,verbose=False)
            semantic_vis = sam2_results[0].plot()
            # plt.imshow(semantic_vis)
            # plt.show()
            # cv2.imwrite("semantic_vis.png",semantic_vis)
            masks = sam2_results[0].masks.data.cpu().numpy()
        else:
            semantic_vis = yolow_results[0].plot()
            masks = []
        # print("yolow time:",time.time()-time1)
        if obs.task_observations is None:
            obs.task_observations = {}

        if draw_instance_predictions:
            obs.task_observations["semantic_frame"] = semantic_vis
        else:
            obs.task_observations["semantic_frame"] = None
        
        if depth_threshold is not None and depth is not None:
            masks = np.array(
                [filter_depth(mask, depth, depth_threshold) for mask in masks]
            )

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        obs.semantic = semantic_map.astype(int)
        obs.instance = instance_map.astype(int)
        if obs.task_observations is None:
            obs.task_observations = dict()
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = class_idcs
        obs.task_observations["instance_scores"] = scores
        
        obs.semantic[obs.semantic==0] = 4
        obs.task_observations["object_goal"] = 1
        obs.task_observations["start_recep_goal"] = 2
        obs.task_observations["end_recep_goal"] = 3
        obs.task_observations["semantic_max_val"] = 4
        obs.task_observations["recep_idx"] = 2

        return obs
        
    def predict_train(
        self,
        rgb_img,
        depth_img,
        depth_threshold: Optional[float] = 0.5,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        depth = depth_img
        height, width, _ = image.shape
        
        # yolow_results = self.yolow_model.predict(image,verbose=False,conf=0.5)
        yolow_results = self.yolow_model.predict(image,verbose=False,conf=0.4)
        
        class_idcs = yolow_results[0].boxes.cls.cpu().numpy() + 1
        scores = yolow_results[0].boxes.conf.cpu().numpy()
        
        if len(scores) > 0:
            sam2_results = self.sam2_model(image, bboxes=yolow_results[0].boxes.xyxy,verbose=False)
            semantic_vis = sam2_results[0].plot()
            masks = sam2_results[0].masks.data.cpu().numpy()
        else:
            semantic_vis = yolow_results[0].plot()
            masks = []
        
        
        if depth_threshold is not None and depth is not None:
            masks = np.array(
                [filter_depth(mask, depth, depth_threshold) for mask in masks]
            )

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        semantic = semantic_map.astype(int)
        
        semantic[semantic==0] = 4

        return semantic,semantic_vis