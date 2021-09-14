"""
Lidar_to_panorama Calibration example. JRDB sensor setup file have this formulation and explanation
    Lidar point = X, Y, Z
    panorama point = u, v

formulation
    u = W_img{(arctan(X/Z) + pi)/(2*pi)}
    v = f^ * {Y/(sec(theta)* Z)} + (y^0 / h^img) * H_img

<sensor_4 parameter>
sensor_4:
    width: 752
    height: 480
    
    # distortion
    D: -0.34064 0.168338 0.000147292 0.000229372 -0.0516133
    
    # intrinsic parameter
    K: >
      485.046 0 368.864
      0 488.185 208.215
      0 0 1
    
    # extrinsic, rotation
    R: >
      0.310275 0.00160497 0.950645
      -0.00648686 0.999979 0.000428942
      -0.950625 -0.00629979 0.310279
    
    # extrinsic, translation
    T: -0.333857 -5.12974 -56.0573
"""

import torch
import numpy as np
import os
import cv2
import math
import pickle

# panorama image load part
image_dir = "/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/training/image_2/"
image_list = os.listdir(image_dir)
image_list_sort = sorted(image_list)
# image read
image = cv2.imread(image_dir + image_list_sort[2])

"""
3D bounding box to panorama image projection start
"""

"""
find points in GT 3D bbox
"""
def camera_to_lidar_roughV1(points, height = None, dimension = None, name=None):
    x_lidar = float(points[2])
    y_lidar = -float(points[0])
    z_lidar = -float(points[1]) + 0.5* float(height)
    xyz_lidar_rough = [str(x_lidar), str(y_lidar), str(z_lidar)]
    return xyz_lidar_rough

# label.txt load
label_dir = "/home/minjae/TANet/pointpillars_with_TANet/second/demo_open3d/label_JRDB/"
label_list = os.listdir(label_dir)
label_list_sort = sorted(label_list)
with open(label_dir + label_list_sort[0], 'r') as f:
    labels = f.readlines()

label = labels[11].split(' ')
box0 = int(label[4])
box1 = int(label[5])
box2 = int(label[6])
box3 = int(label[7])
# dim = label[8:11] # hwl
# loc = label[11:14] # x(l), y(h), z(w)
# import pdb; pdb.set_trace()
# loc_lidar = camera_to_lidar_roughV1(loc, dim[0])
# import pdb; pdb.set_trace()
# x_range_max = np.array(float(loc[0]) + float(dim[2]))
# x_range_min = np.array(float(loc[0]) - float(dim[2]))

# y_range_max = np.array(float(loc[1]) + float(dim[0]))
# y_range_min = np.array(float(loc[1]) - float(dim[0]))

# z_range_max = np.array(float(loc[2]) + float(dim[1]))
# z_range_min = np.array(float(loc[2]) - float(dim[1]))

# print(str(x_range_min), str(x_range_max))
# print(y_range_min, y_range_max)
# print(z_range_min, z_range_max)

# point cloud load part
# points_dir = '/home/minjae/TANet/pointpillars_with_TANet/second/demo_open3d/points/'
# points_list = os.listdir(points_dir)
# points_lists_sort = sorted(points_list)
# points = np.load(points_dir + points_lists_sort[0]).reshape(-1, 4)

# valid_points = []
# for i in range(len(points)):
#     if (x_range_min <= points[i][0] <= x_range_max) and (y_range_min <= points[i][1] <= y_range_max) and (z_range_min <= points[i][2] <= z_range_max):
#         valid_points.append(points[i])
        
with open("/home/minjae/TANet/pointpillars_with_TANet/second/toy_example/GT_points.pkl", "rb") as f:
    valid_points = pickle.load(f)

import pdb; pdb.set_trace()
GT_point = valid_points[0]
GT_point_x = torch.tensor(float(GT_point[0]))
GT_point_y = torch.tensor(float(GT_point[1]))
GT_point_z = torch.tensor(float(GT_point[2]))
W_img = 3760
H_img = 480

#################################### formulate u #########################################
u_value0 = W_img * ((torch.arctan(GT_point_x/GT_point_z) + torch.tensor(np.pi)) / (torch.tensor(np.pi * 2)))

# focal length
fx = 485.046
fy = 488.185

sqrt0 = GT_point_x ** 2 + GT_point_y ** 2 + GT_point_z** 2

optical_center_y = 208.215 # y^0 = cy
median_height = 480 # h^img

#################################### formulate v #########################################
# sec(theta) * Z = sqrt(x^2 + y^2 + z^2)
first_term0 = fy * (GT_point_y / (torch.sqrt(sqrt0)))


Second_term = H_img * (optical_center_y / median_height)

v_value0 = first_term0 + Second_term


print("point_x0 is {0} and point_y0 is {1} and point_z0 is {2}".format(round(float(GT_point_x)), round(float(GT_point_y)), round(float(GT_point_z))))
print("u0 is {0} and v0 is {1}".format(round(float(u_value0)), round(float(v_value0))))


# projection u and v to image
# cv2.circle(img, center, radius, color, thickness)
blue_color = (255, 0, 0)
red_color = (0, 0, 255)

cv2.circle(image, (round(float(u_value0)), round(float(v_value0-300))), 10, blue_color, 10)

# 2d bbox
cv2.circle(image, (box0, box1), 10, red_color, 5)
cv2.circle(image, (box0, box3), 10, red_color, 5)
cv2.circle(image, (box2, box1), 10, red_color, 5)
cv2.circle(image, (box2, box3), 10, red_color, 5)

cv2.line(image, ((box0, box1)), ((box0, box3)), red_color, 5)
cv2.line(image, ((box0, box1)), ((box2, box1)), red_color, 5)
cv2.line(image, ((box2, box3)), ((box0, box3)), red_color, 5)
cv2.line(image, ((box2, box3)), ((box2, box1)), red_color, 5)

cv2.imwrite('./demo.jpg', image)