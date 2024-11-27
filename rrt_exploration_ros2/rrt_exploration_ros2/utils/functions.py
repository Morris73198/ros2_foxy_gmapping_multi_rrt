#!/usr/bin/env python3
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from numpy.linalg import norm

def get_frontier(map_data: OccupancyGrid) -> list:
    """探索邊界點檢測"""
    data = map_data.data
    w = map_data.info.width
    h = map_data.info.height
    resolution = map_data.info.resolution
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y
    
    img = np.zeros((h, w, 1), np.uint8)
    
    for i in range(h):
        for j in range(w):
            if data[i*w+j] == 100:
                img[i,j] = 0
            elif data[i*w+j] == 0:
                img[i,j] = 255
            elif data[i*w+j] == -1:
                img[i,j] = 205
    
    # 邊界檢測
    edges = cv2.Canny(img, 0, 255)
    contours, _ = cv2.findContours(
        cv2.inRange(img, 0, 1),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    frontier_points = []
    
    # 提取邊界點
    if len(contours) > 0:
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                point = [
                    cx * resolution + origin_x,
                    cy * resolution + origin_y
                ]
                frontier_points.append(np.array(point))
    
    return frontier_points

def nearest(V: np.ndarray, x: np.ndarray) -> int:
    """找最近點"""
    distances = norm(V - x, axis=1)
    return np.argmin(distances)

def grid_value(map_data: OccupancyGrid, point: np.ndarray) -> int:
    """獲取地圖網格值"""
    resolution = map_data.info.resolution
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y
    width = map_data.info.width
    
    x = int((point[0] - origin_x) / resolution)
    y = int((point[1] - origin_y) / resolution)
    
    if (0 <= x < map_data.info.width and 
        0 <= y < map_data.info.height):
        return map_data.data[y * width + x]
    return 100  # 超出地圖範圍視為障礙物
