ros2 launch merge_map two_robot_merge_launch.py


ros2 launch merge_map merge_map_launch.py

記得在 RViz 中新增 :

MarkerArray —> /robot_markers    #以顯示機器人位置

ros2 launch rrt_exploration_ros2 boundary.launch.py

記得在 RViz 中新增 :

MarkerArray —> /boundary_circles    #以顯示點選的4個點

Marker —> /boundary_line    #以顯示boundary

ros2 launch rrt_exploration_ros2 global_detector.launch.py

記得在 RViz 中新增 :

MarkerArray —> visualization_marker_array   #以顯示 Global RRT 和其搜尋到的 frontiers
