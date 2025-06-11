import math
import os
import numpy as np

def linear_fitting_3D_points(points):
    '''
    x = k1 * z + b1
    y = k2 * z + b2
    '''

    Sum_X = 0.0
    Sum_Y = 0.0
    Sum_Z = 0.0
    Sum_XZ = 0.0
    Sum_YZ = 0.0
    Sum_Z2 = 0.0

    for i in range(0, len(points)):
        xi = points[i][0]
        yi = points[i][1]
        zi = points[i][2]

        Sum_X = Sum_X + xi
        Sum_Y = Sum_Y + yi
        Sum_Z = Sum_Z + zi
        Sum_XZ = Sum_XZ + xi * zi
        Sum_YZ = Sum_YZ + yi * zi
        Sum_Z2 = Sum_Z2 + zi ** 2

    num = len(points)
    den = num * Sum_Z2 - Sum_Z * Sum_Z
    k1 = (num * Sum_XZ - Sum_X * Sum_Z) / den
    b1 = (Sum_X - k1 * Sum_Z) / num
    k2 = (num * Sum_YZ - Sum_Y * Sum_Z) / den
    b2 = (Sum_Y - k2 * Sum_Z) / num

    z0 = points[0][2]
    x0 = b1 + k1 * z0
    y0 = b2 + k2 * z0
    M = np.array([x0, y0, z0])

    p = 0.1
    m = k1 * p
    n = k2 * p
    S = np.array([m, n, p])
    M_S = np.cross((points-M), S)
    MS_length = np.linalg.norm(M_S, axis=1)
    S_length = np.linalg.norm(S)
    residuals = (np.sum((MS_length/S_length)**2))/num

    # print("res")
    return k1, b1, k2, b2, residuals

def find_start_end(points, k1, b1, k2, b2, z0):
    x0 = b1 + k1 * z0
    y0 = b2 + k2 * z0
    M = np.array([x0, y0, z0])

    p = 0.1
    m = k1 * p
    n = k2 * p
    S = np.array([m, n, p])

    DotProduct = np.dot((points - M), S)
    max_i = np.where(DotProduct == np.max(DotProduct))
    min_i = np.where(DotProduct == np.min(DotProduct))
    max_index = max_i[0][0]
    min_index = min_i[0][0]

    z_start = points[max_index, 2]
    z_end = points[min_index, 2]

    start = points[max_index]
    end = points[min_index]

    ks = -(np.dot((M - start), S) / np.sum(S**2))
    line_start = ks*S + M

    ke = -(np.dot((M - end), S) / np.sum(S**2))
    line_end = ke*S + M

    dis = math.dist(points[min_index], points[max_index])
    return line_start, line_end

def sample_line(k1, b1, k2, b2, z0, z1):
    x0 = b1 + k1 * z0
    y0 = b2 + k2 * z0

    x1 = b1 + k1 * z1
    y1 = b2 + k2 * z1

    X = np.linspace(start=x0, stop=x1, num=200).reshape(200, 1)
    Y = np.linspace(start=y0, stop=y1, num=200).reshape(200, 1)
    Z = np.linspace(start=z0, stop=z1, num=200).reshape(200, 1)
    line = np.concatenate((X, Y, Z), axis=1)

    return line

def sample_line_new(point1, point2):
    X = np.linspace(start=point1[0], stop=point2[0], num=200).reshape(200, 1)
    Y = np.linspace(start=point1[1], stop=point2[1], num=200).reshape(200, 1)
    Z = np.linspace(start=point1[2], stop=point2[2], num=200).reshape(200, 1)
    line = np.concatenate((X, Y, Z), axis=1)

    return line

def fit_xyz_yzx_xzy(cloud):
    para_xyz = linear_fitting_3D_points(np.array(cloud))
    points_yzx = cloud[:, [1, 2, 0]]
    para_yzx = linear_fitting_3D_points(np.array(points_yzx))
    points_xzy = cloud[:, [0, 2, 1]]
    para_xzy = linear_fitting_3D_points(np.array(points_xzy))

    res_numbers = [para_xyz[4], para_yzx[4], para_xzy[4]]
    res_min_index = res_numbers.index(min(res_numbers))
    if res_min_index == 0:
        z = cloud[0, 2]
        print(para_xyz)
        z0, z1 = find_start_end(cloud, *para_xyz[0:4], z)
        line_points = sample_line_new(z0, z1)
        print(line_points.shape)
        np.savetxt(save_dir + fn + "_linefitxyz.txt", line_points, fmt="%.5f")
    elif res_min_index == 1:
        z = points_yzx[0, 2]
        print(para_yzx)
        z0, z1 = find_start_end(points_yzx, *para_yzx[0:4], z)
        line_points = sample_line_new(z0, z1)
        print(line_points.shape)
        points_1 = line_points[:, [2, 0, 1]]
        np.savetxt(save_dir + fn + "_linefityzx.txt", points_1, fmt="%.5f")
    elif res_min_index == 2:
        z = points_xzy[0, 2]
        print(para_xzy)
        z0, z1 = find_start_end(points_xzy, *para_xzy[0:4], z)
        line_points = sample_line_new(z0, z1)
        print(line_points.shape)
        points_2 = line_points[:, [0, 2, 1]]
        np.savetxt(save_dir + fn + "_linefitxzy.txt", points_2, fmt="%.5f")


if __name__=='__main__':

    partseg_res_dir = "./data_results/00011917"
    fns = [fn[:-4] for fn in os.listdir(partseg_res_dir) if fn.endswith(".txt")]
    print(fns)
    save_dir = "./data_results/00011917/"

    for fn in fns:
        print("===> predict", fn)
        pointcloud_v2_60k_path = "{}/{}.txt".format(partseg_res_dir, fn)
        cloud = np.loadtxt(pointcloud_v2_60k_path, delimiter=";", usecols=range(3))

        fit_xyz_yzx_xzy(cloud)







