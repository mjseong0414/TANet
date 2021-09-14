import numpy as np

h = 4
points_lidar = np.array([1, 2, 3])
print(points_lidar)]

points_camera = np.array([-points_lidar[1], -points_lidar[2]+h/2, points_lidar[0]])
print(points_camera)

x_lidar = points_camera[2]
y_lidar = -points_camera[0]
z_lidar = -points_camera[1] + 0.5 * h

points_camera_to_lidar = np.array([x_lidar, y_lidar, z_lidar])
print(points_camera_to_lidar)