import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.interpolate import splprep, splev
import time
import cv2
from scipy.spatial.distance import cdist
import random

class AStarPlanner:
    def __init__(self, factor=4, margin=0, smoothness=2, num_points=50):
        self.factor = factor
        self.margin = margin
        self.smoothness = smoothness
        self.num_points = num_points
        
    def find_nearest_goal_pixel(self, goal_map, agent_position):
        goal_map_min = goal_map[0::4,0::4]
        if not np.any(goal_map_min):
            return None,0
        # 找到所有目标点的坐标
        goal_positions = np.array(np.where(goal_map_min > 0)).T
        
        agent_position_min = [agent_position[0]/4,agent_position[1]/4]
        # 计算agent到所有目标点的距离
        distances = cdist([agent_position_min], goal_positions)[0]
        
        # 找到最小距离及其索引
        min_index = np.argmin(distances)
        
        # 获取最近目标点的坐标
        nearest_goal = tuple(goal_positions[min_index]*4)
        
        # vis_map = np.zeros((goal_map.shape[0], goal_map.shape[1], 3), dtype=np.uint8)
        # vis_map[goal_map>0] = [255, 255, 255]
        # goal_y, goal_x = nearest_goal
        # cv2.circle(vis_map, (goal_x, goal_y), 2, (0, 0, 255), -1)  # 红色
        # agent_y, agent_x = agent_position
        # cv2.circle(vis_map, (agent_x, agent_y), 3, (255, 0, 0), -1)  # blue
        # cv2.line(vis_map, (agent_x, agent_y), (goal_x, goal_y), (0, 255, 255), 1)  # 黄色
        # cv2.imshow("vis_map",np.flipud(vis_map))
        # cv2.waitKey(1)
        
        return nearest_goal, distances[min_index]*4
    
    def get_shortterm_goal(self,navigable_goal_map,state,traversible):
        stg_y, stg_x = None, None
        astar_flag, path = False, None
        near_goal_pos,near_dis = self.find_nearest_goal_pixel(navigable_goal_map,state)
        if near_dis > 15+5*random.random():  # enough far to use A star, 避免硬切换
            grid = (traversible + navigable_goal_map)
            end = [near_goal_pos[1],near_goal_pos[0]]
            path = self.get_global_path(grid,state,end)
            length = 0.0
            if len(path) > 1:
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i-1][0]
                    dy = path[i][1] - path[i-1][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    length += distance
                    if length > 0.2*20:
                        stg_y, stg_x = path[i]  
                        astar_flag = True
                        break
        return astar_flag, stg_y, stg_x, near_goal_pos, path
        
    def get_global_path(self, grid, start, end):
        # 添加安全间隔
        grid_with_margin = self.add_safety_margin(grid)
    
        # 下采样网格
        downsampled_grid = self.downsample_grid(grid_with_margin)
        
        start = (start[0] // self.factor, start[1] // self.factor)
        end = (end[0] // self.factor, end[1] // self.factor)
        
        path = self.plan_path(downsampled_grid, start, end)
        
        # 将路径映射回原始分辨率
        original_path = [(p.x * self.factor, p.y * self.factor) for p in path]
        
        smooth_path_result = original_path
        
        return smooth_path_result

    def add_safety_margin(self, grid):
        """为障碍物添加安全边距"""
        # print("self.margin:",self.margin)
        padded_grid = np.pad(grid, self.margin, mode='constant', constant_values=1)
        for _ in range(self.margin):
            padded_grid = np.logical_and(padded_grid, np.roll(padded_grid, 1, axis=0))
            padded_grid = np.logical_and(padded_grid, np.roll(padded_grid, -1, axis=0))
            padded_grid = np.logical_and(padded_grid, np.roll(padded_grid, 1, axis=1))
            padded_grid = np.logical_and(padded_grid, np.roll(padded_grid, -1, axis=1))
        return padded_grid[self.margin:grid.shape[1]-self.margin, self.margin:grid.shape[0]-self.margin].astype(int)

    def downsample_grid(self, grid):
        return grid[::self.factor, ::self.factor]

    def plan_path(self, grid, start, end):
        grid = Grid(matrix=grid)
        start = grid.node(*start)
        end = grid.node(*end)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, _ = finder.find_path(start, end, grid)
        return path
    
    def visualize_path(self, grid, path, start, end):
        # 将 grid 转换为 BGR 格式的图像
        img = cv2.cvtColor((grid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # 绘制路径
        if path:
            for i in range(1, len(path)):
                cv2.line(img, path[i-1], path[i], (0, 0, 255), 1)
        
        # 绘制起点和终点
        cv2.circle(img, start, 5, (0, 255, 0), -1)
        cv2.circle(img, end, 5, (255, 0, 0), -1)
        
        return img


if __name__ == "__main__":
    
    glbal_planner = GlobalPlanner(factor=5,margin=0,smoothness=0.1)
    
    # 创建原始网格
    # width, height = 500, 500
    # grid = np.ones((height, width), dtype=int)
    
    # # 添加矩形障碍物
    # grid[50:100, 100:150] = 0
    # grid[200:300, 300:350] = 0
    # grid[350:400, 100:450] = 0
    
    # # 添加圆形障碍物
    # center_x, center_y = 250, 250
    # radius = 50
    # y, x = np.ogrid[:height, :width]
    # mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    # grid[mask] = 0
    grid = np.load("/home/code/code/Hab3/V3_1/habitat-lab/PEANUT/test/grid.npy")
    print("grid",grid.shape,np.unique(grid))
    
    start = (240,240)
    end = (1, 236)
    
    # start = (10,10)
    # end = (400, 400)
    
    path = glbal_planner.get_global_path(grid,start,end)
    
    img = glbal_planner.visualize_path(grid, path, start, end)
    
    # 显示图像
    cv2.imshow('Optimized A* Pathfinding with Obstacles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()