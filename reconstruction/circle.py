import numpy as np
import random
import colorsys
import open3d as o3d
import re
import math
import os

# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
def rodrigues_rot(P, n0, n1):
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))

    return P_rot

# - Find center [xc, yc] and radius r of circle fitting to set of 2D points
# - Optionally specify weights for points
#
# - Implicit circle function:
#   (x-xc)^2 + (y-yc)^2 = r^2
#   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
#   c[0]*x + c[1]*y + c[2] = x^2+y^2
#
# - Solution by method of least squares:
#   A*c = b, c' = argmin(||A*c - b||^2)
#   A = [x y 1], b = [x^2+y^2]
def fit_circle_2d(x, y, w=[]):
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    solve = np.linalg.lstsq(A, b, rcond=None)
    print("solve", solve)
    c = solve[0]
    residuals = solve[1]
    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r, residuals

# Generate points on circle
# P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
def generate_circle_by_vectors(t, C, r, n, u):
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
    return P_circle

def circle_segmentation(cloud):
    # -------------------------------------------------------------------------------
    # (1) Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    # -------------------------------------------------------------------------------
    P_mean = cloud.mean(axis=0)
    P_centered = cloud - P_mean
    U, s, V = np.linalg.svd(P_centered)

    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2, :]
    d = -np.dot(P_mean, normal)  # d = -<p,n>

    # -------------------------------------------------------------------------------
    # (2) Project points to coords X-Y in 2D plane
    # -------------------------------------------------------------------------------
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

    # -------------------------------------------------------------------------------
    # (3) Fit circle in new 2D coords
    # -------------------------------------------------------------------------------
    xc, yc, r, residuals = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

    # --- Generate circle points in 2D
    # t = np.linspace(0, 2 * np.pi, 100)
    # xx = xc + r * np.cos(t)
    # yy = yc + r * np.sin(t)

    # -------------------------------------------------------------------------------
    # (4) Transform circle center back to 3D coords
    # -------------------------------------------------------------------------------
    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()

    # --- Generate points for fitting circle
    t = np.linspace(0, 2 * np.pi, 360)
    # print(t)
    u = cloud[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)
    return P_fitcircle, C, r, residuals, normal, u

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors

def get_str_range(string):
    result = np.empty(shape=(0, 2)).astype(int)
    n = 36
    pattern = '[0]{' + str(n) + ',}'
    re_list = list(re.finditer(pattern, string))
    # print("re_list", re_list)
    if len(re_list) == 0:
        result = np.array([[0, 360]]).astype(int)
    else:
        for i in range(len(re_list)+1):
            if i == 0:
                x = np.array([[0, re_list[i].start()]]).astype(int)
            elif i == len(re_list):
                x = np.array([[re_list[i-1].end(), 360]]).astype(int)
            else:
                x = np.array([[re_list[i-1].end(), re_list[i].start()]]).astype(int)
            result = np.append(result, x, axis=0)

        if (string[0] == '0') and (string[-1] == '0'):
            print("have 0 and 0!")
            pattern_s = '[0]+'
            re_list_s = list(re.finditer(pattern_s, string))
            # print("re_list_s", re_list_s)
            head = (re_list_s[0].end()-re_list_s[0].start())
            foot = (re_list_s[-1].end() - re_list_s[-1].start())
            if (head + foot) >= n:
                result[0, 0] = re_list_s[0].end()
                result[-1, 1] = re_list_s[-1].start()

    rows = result.shape[0]
    del_list = []
    for m in range(rows):
        if result[m, 0] > result[m, 1]:
            del_list.append(m)
    result = np.delete(result, del_list, axis=0)
    print(result)
    return result

def find_arc(points_input, points_fit, p2p_dis):
    r_ball = p2p_dis * 3000
    circle_string = ''

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_input)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    row = points_fit.shape[0]
    for r_index in range(row):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(points_fit[r_index], r_ball)  # r_ball
        neighborindex_list = idx[0:]
        if neighborindex_list:
            circle_string = circle_string + '1'
        else:
            circle_string = circle_string + '0'
    # print(circle_string)
    t_result = get_str_range(circle_string)
    return t_result

def deg2rad(result):
    rows, cols = result.shape
    result_s = np.empty(shape=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            result_s[i, j] = math.radians(result[i, j])
    return result_s

def sample_circle(t_rad, circle_center, radius, normal, u):
    fitcircle = np.empty(shape=(0, 3))
    for row in t_rad:
        t = np.linspace(row[0], row[1], 360)
        fitcircle_temp = generate_circle_by_vectors(t, circle_center, radius, normal, u)

        fitcircle = np.append(fitcircle, fitcircle_temp, axis=0)
    return fitcircle

def get_r_ball(points_input):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_input)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    index_random = random.randint(0, points_input.shape[0] - 1)
    [k, idx, dis] = pcd_tree.search_knn_vector_3d(pcd.points[10], 10)
    p2p_dis = dis[1:][0]
    # print("p2p_dis", p2p_dis)
    return p2p_dis

if __name__=='__main__':

    partseg_res_dir = "/data/circle"
    fns = [fn[:-4] for fn in os.listdir(partseg_res_dir) if fn.endswith(".txt")]  # GT_points.txt
    print(fns)
    save_dir = "/data/circle/"

    for fn in fns:
        print("===> predict", fn)
        pointcloud_v2_60k_path = "{}/{}.txt".format(partseg_res_dir, fn)
        cloud = np.loadtxt(pointcloud_v2_60k_path, delimiter=" ", usecols=range(3))

        circle, circle_center, radius, residuals, normal, u = circle_segmentation(np.array(cloud))
        print(circle_center)
        print(radius)
        print(residuals)
        p2p_dis = get_r_ball(circle)
        t = find_arc(cloud, circle, p2p_dis)
        t_rad = deg2rad(t)
        fitcircle = sample_circle(t_rad, circle_center, radius, normal, u)


