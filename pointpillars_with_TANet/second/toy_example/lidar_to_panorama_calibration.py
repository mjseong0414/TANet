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
from torch import stack as tstack

# panorama image load part
image_dir = "/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/training/image_2/"
image_list = os.listdir(image_dir)
image_list_sort = sorted(image_list)
# image read
image = cv2.imread(image_dir + image_list_sort[0])
"""
# point cloud load part
points_dir = '/home/minjae/TANet/pointpillars_with_TANet/second/demo_open3d/points/'
points_list = os.listdir(points_dir)
points_lists_sort = sorted(points_list)
points = np.load(points_dir + points_lists_sort[0]).reshape(-1, 4)

W_img = 3760
H_img = 480

# X, Y, Z => point location
X, Y, Z = points[0, :3] # point cloud load
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)
Z_tensor = torch.tensor(Z)

u_value = W_img * ((torch.arctan(X_tensor/Z_tensor) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))

# focal length
fx = 483.911
fy = 486.466

x2y2z2 = X * X + Y * Y + Z * Z
x2y2z2_tensor = torch.tensor(x2y2z2)
optical_center_y = 223.023 # y^0 = cy
median_height = 480 # h^img

first_term = fy * (Y / (torch.sqrt(x2y2z2_tensor))) # sec(theta) * Z = sqrt(x^2 + y^2 + z^2)
optical_center_y = 223.023 # y^0 = cy
Second_term = H_img * (optical_center_y / median_height)
v_value = first_term + Second_term
print("point_x is {0} and point_y is {1} and point_z is {2}".format(round(X), round(Y), round(Z)))
print("u is {0} and v is {1}".format(round(float(u_value)), round(float(v_value))))

# projection u and v to image
# cv2.circle(img, center, radius, color, thickness)
blue_color = (255, 0, 0)
cv2.circle(image, (round(float(u_value)), round(float(v_value))), 10, blue_color, 5)
cv2.imwrite('./demo.jpg', image)
"""

"""
3D bounding box to panorama image projection start
"""
def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack([
            tstack([rot_cos, zeros, -rot_sin]),
            tstack([zeros, ones, zeros]),
            tstack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack([
            tstack([rot_cos, -rot_sin, zeros]),
            tstack([rot_sin, rot_cos, zeros]),
            tstack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = tstack([
            tstack([zeros, rot_cos, -rot_sin]),
            tstack([zeros, rot_sin, rot_cos]),
            tstack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")
    
    # return torch.einsum('aij,jka->aik', (points, rot_mat_T))
    return torch.einsum('aij,jka->aik', (points, torch.reshape(rot_mat_T, (-1,3, 1))))

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    
    ndim = int(dims.shape[0]) # JRDB is dims.shape[1]
    #dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(np.float32)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        # import pdb; pdb.set_trace()
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=np.float32)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    return corners # shape : (1, 8, 3)

def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=[-0.02, 0, 0.84],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners # shape : (1, 8, 3)

def camera_to_lidar_roughV1(points, height = None, dimension = None, name=None):
    if name == "gt":
        x_lidar = float(points[2])
        y_lidar = -float(points[0])
        z_lidar = -float(points[1]) + 0.5* float(height)
        xyz_lidar_rough = [str(x_lidar), str(y_lidar), str(z_lidar)]
        return xyz_lidar_rough
    elif name == "predicted":
        height = dimension[:, 0].reshape(-1,1)
        x_lidar = points[:,2].reshape(-1,1)
        y_lidar = -points[:,0].reshape(-1,1)
        z_lidar = -points[:,1].reshape(-1,1) + 0.5*height
        xyz_lidar_rough = np.concatenate([x_lidar, y_lidar, z_lidar], axis=1)
        return xyz_lidar_rough

def center_to_corner_box3d_rough(centers, dims, angles, origin=[-0.02, 0, 0.84], axis=2):
    # centers : [x, y, z]
    # dims : [l, h, w]
    corner0_x = torch.tensor(centers[0] - dims[0]/2).reshape(-1, 1)
    corner0_y = torch.tensor(centers[1] + dims[1]/2).reshape(-1, 1)
    corner0_z = torch.tensor(centers[2] + dims[2]/2).reshape(-1, 1)
    corner0 = torch.cat((corner0_x, corner0_y, corner0_z), dim=1)

    corner1_x = torch.tensor(centers[0] - dims[0]/2).reshape(-1, 1)
    corner1_y = torch.tensor(centers[1] + dims[1]/2).reshape(-1, 1)
    corner1_z = torch.tensor(centers[2] - dims[2]/2).reshape(-1, 1)
    corner1 = torch.cat((corner1_x, corner1_y, corner1_z), dim=1)

    corner2_x = torch.tensor(centers[0] - dims[0]/2).reshape(-1, 1)
    corner2_y = torch.tensor(centers[1] - dims[1]/2).reshape(-1, 1)
    corner2_z = torch.tensor(centers[2] + dims[2]/2).reshape(-1, 1)
    corner2 = torch.cat((corner2_x, corner2_y, corner2_z), dim=1)

    corner3_x = torch.tensor(centers[0] - dims[0]/2).reshape(-1, 1)
    corner3_y = torch.tensor(centers[1] - dims[1]/2).reshape(-1, 1)
    corner3_z = torch.tensor(centers[2] - dims[2]/2).reshape(-1, 1)
    corner3 = torch.cat((corner3_x, corner3_y, corner3_z), dim=1)

    corner4_x = torch.tensor(centers[0] + dims[0]/2).reshape(-1, 1)
    corner4_y = torch.tensor(centers[1] + dims[1]/2).reshape(-1, 1)
    corner4_z = torch.tensor(centers[2] + dims[2]/2).reshape(-1, 1)
    corner4 = torch.cat((corner4_x, corner4_y, corner4_z), dim=1)

    corner5_x = torch.tensor(centers[0] + dims[0]/2).reshape(-1, 1)
    corner5_y = torch.tensor(centers[1] + dims[1]/2).reshape(-1, 1)
    corner5_z = torch.tensor(centers[2] - dims[2]/2).reshape(-1, 1)
    corner5 = torch.cat((corner5_x, corner5_y, corner5_z), dim=1)

    corner6_x = torch.tensor(centers[0] + dims[0]/2).reshape(-1, 1)
    corner6_y = torch.tensor(centers[1] - dims[1]/2).reshape(-1, 1)
    corner6_z = torch.tensor(centers[2] + dims[2]/2).reshape(-1, 1)
    corner6 = torch.cat((corner6_x, corner6_y, corner6_z), dim=1)

    corner7_x = torch.tensor(centers[0] + dims[0]/2).reshape(-1, 1)
    corner7_y = torch.tensor(centers[1] - dims[1]/2).reshape(-1, 1)
    corner7_z = torch.tensor(centers[2] - dims[2]/2).reshape(-1, 1)
    corner7 = torch.cat((corner7_x, corner7_y, corner7_z), dim=1)
    
    corners = torch.cat((corner0, corner1, corner2, corner3, corner4, corner5, corner6, corner7), dim=1).reshape(1, 8, -1)
    
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    return corners


    
    
    

def camera_to_lidar_roughV2(points,h):
    x_lidar = points[2].reshape(-1,1)
    y_lidar = -points[0].reshape(-1,1)
    z_lidar = -points[1].reshape(-1,1) + 0.5*h
    xyz_lidar_rough = torch.cat([x_lidar, y_lidar, z_lidar], dim=-1)
    return xyz_lidar_rough

# label.txt load
label_dir = "/home/minjae/TANet/pointpillars_with_TANet/second/demo_open3d/label_JRDB/"
label_list = os.listdir(label_dir)
label_list_sort = sorted(label_list)
with open(label_dir + label_list_sort[0], 'r') as f:
    labels = f.readlines()

label = labels[2].split(' ')
loc = torch.tensor(list(map(float, label[11:14])))


h = float(label[8])
w = float(label[9])
l = float(label[10])
# dim = torch.tensor([l, h, w])
# dim = torch.tensor([w, l, h])
dim = torch.tensor(list(map(float, label[8:11]))) # hwl
loc_lidar = camera_to_lidar_roughV2(loc, dim[0])

angle = torch.tensor(float(label[14]))
box0 = int(label[4])
box1 = int(label[5])
box2 = int(label[6])
box3 = int(label[7])

# import pdb; pdb.set_trace()
# box_corners = center_to_corner_box3d(loc, dim, angle, axis=1) # torch.Size([1, 8, 3])
box_corners = center_to_corner_box3d_rough(loc, dim, angle, axis=1)

# box_corner : camera coordinate -> lidar coordinate
"""
loc0 = box_corners[0][0]
loc1 = box_corners[0][1]
loc2 = box_corners[0][2]
loc3 = box_corners[0][3]
loc4 = box_corners[0][4]
loc5 = box_corners[0][5]
loc6 = box_corners[0][6]
loc7 = box_corners[0][7]

loc_lidar0 = camera_to_lidar_roughV2(loc0, dim[1])
loc_lidar1 = camera_to_lidar_roughV2(loc1, dim[1])
loc_lidar2 = camera_to_lidar_roughV2(loc2, dim[1])
loc_lidar3 = camera_to_lidar_roughV2(loc3, dim[1])
loc_lidar4 = camera_to_lidar_roughV2(loc4, dim[1])
loc_lidar5 = camera_to_lidar_roughV2(loc5, dim[1])
loc_lidar6 = camera_to_lidar_roughV2(loc6, dim[1])
loc_lidar7 = camera_to_lidar_roughV2(loc7, dim[1])
"""
loc_lidar0 = box_corners[0][0]
loc_lidar1 = box_corners[0][1]
loc_lidar2 = box_corners[0][2]
loc_lidar3 = box_corners[0][3]
loc_lidar4 = box_corners[0][4]
loc_lidar5 = box_corners[0][5]
loc_lidar6 = box_corners[0][6]
loc_lidar7 = box_corners[0][7]

# eight corner point
x0 = torch.ravel(loc_lidar0)[0]
y0 = torch.ravel(loc_lidar0)[1]
z0 = torch.ravel(loc_lidar0)[2]
x1 = torch.ravel(loc_lidar1)[0]
y1 = torch.ravel(loc_lidar1)[1]
z1 = torch.ravel(loc_lidar1)[2]
x2 = torch.ravel(loc_lidar2)[0]
y2 = torch.ravel(loc_lidar2)[1]
z2 = torch.ravel(loc_lidar2)[2]
x3 = torch.ravel(loc_lidar3)[0]
y3 = torch.ravel(loc_lidar3)[1]
z3 = torch.ravel(loc_lidar3)[2]
x4 = torch.ravel(loc_lidar4)[0]
y4 = torch.ravel(loc_lidar4)[1]
z4 = torch.ravel(loc_lidar4)[2]
x5 = torch.ravel(loc_lidar5)[0]
y5 = torch.ravel(loc_lidar5)[1]
z5 = torch.ravel(loc_lidar5)[2]
x6 = torch.ravel(loc_lidar6)[0]
y6 = torch.ravel(loc_lidar6)[1]
z6 = torch.ravel(loc_lidar6)[2]
x7 = torch.ravel(loc_lidar7)[0]
y7 = torch.ravel(loc_lidar7)[1]
z7 = torch.ravel(loc_lidar7)[2]

W_img = 3760
H_img = 480

#import pdb; pdb.set_trace()
#################################### formulate u #########################################
u_value0 = W_img * ((2 * torch.arctan(x0/z0) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))
u_value1 = W_img * ((torch.arctan(x1/z1) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))
u_value2 = W_img * ((torch.arctan(x2/z2) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))
u_value3 = W_img * ((torch.arctan(x3/z3) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))
u_value4 = W_img * ((torch.arctan(x4/z4) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))
u_value5 = W_img * ((torch.arctan(x5/z5) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))
u_value6 = W_img * ((torch.arctan(x6/z6) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))
u_value7 = W_img * ((torch.arctan(x7/z7) + torch.tensor(np.pi)) / (2 * torch.tensor(np.pi)))

# focal length
fx = 485.046
fy = 488.185

sqrt0 = x0 ** 2 + y0 ** 2 + z0** 2
sqrt1 = x1 ** 2 + y1 ** 2 + z1** 2
sqrt2 = x2 ** 2 + y2 ** 2 + z2** 2
sqrt3 = x3 ** 2 + y3 ** 2 + z3** 2
sqrt4 = x4 ** 2 + y4 ** 2 + z4** 2
sqrt5 = x5 ** 2 + y5 ** 2 + z5** 2
sqrt6 = x6 ** 2 + y6 ** 2 + z6** 2
sqrt7 = x7 ** 2 + y7 ** 2 + z7** 2

sqrt0_tensor = sqrt0
sqrt1_tensor = sqrt1
sqrt2_tensor = sqrt2
sqrt3_tensor = sqrt3
sqrt4_tensor = sqrt4
sqrt5_tensor = sqrt5
sqrt6_tensor = sqrt6
sqrt7_tensor = sqrt7

optical_center_y = 208.215 # y^0 = cy
median_height = 480 # h^img

#################################### formulate v #########################################
# sec(theta) * Z = sqrt(x^2 + y^2 + z^2)
first_term0 = fy * (y0 / (torch.sqrt(sqrt0_tensor))) 
first_term1 = fy * (y1 / (torch.sqrt(sqrt1_tensor)))
first_term2 = fy * (y2 / (torch.sqrt(sqrt2_tensor)))
first_term3 = fy * (y3 / (torch.sqrt(sqrt3_tensor)))
first_term4 = fy * (y4 / (torch.sqrt(sqrt4_tensor)))
first_term5 = fy * (y5 / (torch.sqrt(sqrt5_tensor)))
first_term6 = fy * (y6 / (torch.sqrt(sqrt6_tensor)))
first_term7 = fy * (y7 / (torch.sqrt(sqrt7_tensor)))


Second_term = H_img * (optical_center_y / median_height)

v_value0 = first_term0 + Second_term
v_value1 = first_term1 + Second_term
v_value2 = first_term2 + Second_term
v_value3 = first_term3 + Second_term
v_value4 = first_term4 + Second_term
v_value5 = first_term5 + Second_term
v_value6 = first_term6 + Second_term
v_value7 = first_term7 + Second_term

print("point_x0 is {0} and point_y0 is {1} and point_z0 is {2}".format(round(float(x0)), round(float(y0)), round(float(z0))))
print("u0 is {0} and v0 is {1}".format(round(float(u_value0)), round(float(v_value0))))
print("point_x1 is {0} and point_y1 is {1} and point_z1 is {2}".format(round(float(x1)), round(float(y1)), round(float(z1))))
print("u1 is {0} and v1 is {1}".format(round(float(u_value1)), round(float(v_value1))))
print("point_x2 is {0} and point_y2 is {1} and point_z2 is {2}".format(round(float(x2)), round(float(y2)), round(float(z2))))
print("u2 is {0} and v2 is {1}".format(round(float(u_value2)), round(float(v_value2))))
print("point_x3 is {0} and point_y3 is {1} and point_z3 is {2}".format(round(float(x3)), round(float(y3)), round(float(z3))))
print("u3 is {0} and v3 is {1}".format(round(float(u_value3)), round(float(v_value3))))
print("point_x4 is {0} and point_y4 is {1} and point_z4 is {2}".format(round(float(x4)), round(float(y4)), round(float(z4))))
print("u4 is {0} and v4 is {1}".format(round(float(u_value4)), round(float(v_value4))))
print("point_x5 is {0} and point_y5 is {1} and point_z5 is {2}".format(round(float(x5)), round(float(y5)), round(float(z5))))
print("u5 is {0} and v5 is {1}".format(round(float(u_value5)), round(float(v_value5))))
print("point_x6 is {0} and point_y6 is {1} and point_z6 is {2}".format(round(float(x6)), round(float(y6)), round(float(z6))))
print("u6 is {0} and v6 is {1}".format(round(float(u_value6)), round(float(v_value6))))
print("point_x7 is {0} and point_y7 is {1} and point_z7 is {2}".format(round(float(x7)), round(float(y7)), round(float(z7))))
print("u7 is {0} and v7 is {1}".format(round(float(u_value7)), round(float(v_value7))))

# projection u and v to image
# cv2.circle(img, center, radius, color, thickness)
blue_color = (255, 0, 0)
red_color = (0, 0, 255)

cv2.circle(image, (round(float(u_value0)), round(float(v_value0))), 10, blue_color, 5)
cv2.circle(image, (round(float(u_value1)), round(float(v_value1))), 10, blue_color, 5)
cv2.circle(image, (round(float(u_value2)), round(float(v_value2))), 10, blue_color, 5)
cv2.circle(image, (round(float(u_value3)), round(float(v_value3))), 10, blue_color, 5)
cv2.circle(image, (round(float(u_value4)), round(float(v_value4))), 10, blue_color, 5)
cv2.circle(image, (round(float(u_value5)), round(float(v_value5))), 10, blue_color, 5)
cv2.circle(image, (round(float(u_value6)), round(float(v_value6))), 10, blue_color, 5)
cv2.circle(image, (round(float(u_value7)), round(float(v_value7))), 10, blue_color, 5)

cv2.line(image, (round(float(u_value0)), round(float(v_value0))), (round(float(u_value1)), round(float(v_value1))), blue_color, 5)
cv2.line(image, (round(float(u_value0)), round(float(v_value0))), (round(float(u_value3)), round(float(v_value3))), blue_color, 5)
cv2.line(image, (round(float(u_value0)), round(float(v_value0))), (round(float(u_value4)), round(float(v_value4))), blue_color, 5)
cv2.line(image, (round(float(u_value1)), round(float(v_value1))), (round(float(u_value2)), round(float(v_value2))), blue_color, 5)
cv2.line(image, (round(float(u_value1)), round(float(v_value1))), (round(float(u_value5)), round(float(v_value5))), blue_color, 5)
cv2.line(image, (round(float(u_value2)), round(float(v_value2))), (round(float(u_value3)), round(float(v_value3))), blue_color, 5)
cv2.line(image, (round(float(u_value2)), round(float(v_value2))), (round(float(u_value6)), round(float(v_value6))), blue_color, 5)
cv2.line(image, (round(float(u_value3)), round(float(v_value3))), (round(float(u_value7)), round(float(v_value7))), blue_color, 5)
cv2.line(image, (round(float(u_value4)), round(float(v_value4))), (round(float(u_value5)), round(float(v_value5))), blue_color, 5)
cv2.line(image, (round(float(u_value4)), round(float(v_value4))), (round(float(u_value7)), round(float(v_value7))), blue_color, 5)
cv2.line(image, (round(float(u_value5)), round(float(v_value5))), (round(float(u_value6)), round(float(v_value6))), blue_color, 5)
cv2.line(image, (round(float(u_value6)), round(float(v_value6))), (round(float(u_value7)), round(float(v_value7))), blue_color, 5)


# cv2.circle(image, (round(float(u_value0)), round(float(v_value0)-400)), 10, blue_color, 5)
# cv2.circle(image, (round(float(u_value1)), round(float(v_value1)-400)), 10, blue_color, 5)
# cv2.circle(image, (round(float(u_value2)), round(float(v_value2)-400)), 10, blue_color, 5)
# cv2.circle(image, (round(float(u_value3)), round(float(v_value3)-400)), 10, blue_color, 5)
# cv2.circle(image, (round(float(u_value4)), round(float(v_value4)-400)), 10, blue_color, 5)
# cv2.circle(image, (round(float(u_value5)), round(float(v_value5)-400)), 10, blue_color, 5)
# cv2.circle(image, (round(float(u_value6)), round(float(v_value6)-400)), 10, blue_color, 5)
# cv2.circle(image, (round(float(u_value7)), round(float(v_value7)-400)), 10, blue_color, 5)

# cv2.line(image, (round(float(u_value0)), round(float(v_value0)-400)), (round(float(u_value1)), round(float(v_value1)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value0)), round(float(v_value0)-400)), (round(float(u_value3)), round(float(v_value3)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value0)), round(float(v_value0)-400)), (round(float(u_value4)), round(float(v_value4)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value1)), round(float(v_value1)-400)), (round(float(u_value2)), round(float(v_value2)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value1)), round(float(v_value1)-400)), (round(float(u_value5)), round(float(v_value5)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value2)), round(float(v_value2)-400)), (round(float(u_value3)), round(float(v_value3)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value2)), round(float(v_value2)-400)), (round(float(u_value6)), round(float(v_value6)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value3)), round(float(v_value3)-400)), (round(float(u_value7)), round(float(v_value7)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value4)), round(float(v_value4)-400)), (round(float(u_value5)), round(float(v_value5)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value4)), round(float(v_value4)-400)), (round(float(u_value7)), round(float(v_value7)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value5)), round(float(v_value5)-400)), (round(float(u_value6)), round(float(v_value6)-400)), blue_color, 5)
# cv2.line(image, (round(float(u_value6)), round(float(v_value6)-400)), (round(float(u_value7)), round(float(v_value7)-400)), blue_color, 5)


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