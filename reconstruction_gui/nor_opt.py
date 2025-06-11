import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
def read_points_file(pcd_path, pcd_name):

    points = np.loadtxt(pcd_path + "/" + pcd_name + "_pred_normals.txt", delimiter=";")

    file_path = pcd_path + "/" + pcd_name + "_pred_edge.txt"
    df = pd.read_csv(file_path, delimiter=';', header=None)
    data = df.iloc[:, 1].values if len(df.columns) > 1 else df.iloc[:, 0].values
    edge_probability = np.array(data)

    vis_inst = np.loadtxt(pcd_path + "/" + pcd_name + "_pred_Vis_I.txt", delimiter=";")
    if os.path.isfile(pcd_path + "/" + pcd_name + '_pred_inst.npy'):
        inst = np.load(pcd_path + "/" + pcd_name + '_pred_inst.npy').astype(int)  # npy
    elif os.path.isfile(pcd_path + "/" + pcd_name + '_pred_inst.txt'):
        inst = np.loadtxt(pcd_path + "/" + pcd_name + '_pred_inst.txt', dtype=int)  # txt
    return points, edge_probability, inst, vis_inst

def fit_plane_svd(data):
    points = data[:, :3]
    M = points - np.mean(points, axis=0)
    _, _, V = svd(M, full_matrices=False)
    normal = V[2, :]

    dtmp = np.mean(points @ normal)
    return np.concatenate([normal * np.sign(dtmp), [-dtmp * np.sign(dtmp)]])

def replace_normals(points, plane_params):
    A, B, C, _ = plane_params[:4]
    plane_normal = np.array([A, B, C])

    dot_products = np.dot(points[:, 3:], plane_normal)

    positive_count = np.sum(dot_products > 0)
    negative_count = np.sum(dot_products < 0)

    if positive_count >= negative_count:
        print("positive_count")
        points[:, 3:] = plane_normal
    else:
        print("negative_count")
        points[:, 3:] = -plane_normal

def fit_plane(points, inst, threshold):
    merged_points = np.copy(points)

    xyz = points

    unique_inst = np.unique(inst)
    # print(unique_inst)

    for category in unique_inst:
        mask = (inst == category)
        category_points = xyz[mask]

        # print(category_points.shape)
        if category_points.shape[0] < 3:
            continue
        plane_params = fit_plane_svd(category_points[:, :3])
        # visualize_fit(category_points, plane_params)
        _, residuals_sse = calculate_residuals(category_points[:, :3], plane_params)
        # print("res====", plane_params, residuals_sse)
        if residuals_sse < threshold:
            replace_normals(category_points, plane_params)
            print("plane", category)

        merged_points[mask, :] = category_points
    return merged_points

def calculate_residuals(points, plane_params):
    A, B, C, D = plane_params[:4]
    plane_normal_magnitude = np.sqrt(A**2 + B**2 + C**2)
    # print("plane_normal_magnitude", plane_normal_magnitude)

    distances = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / plane_normal_magnitude

    residuals_mse = np.mean(distances)
    residuals_sse = np.sum(distances)

    return residuals_mse, residuals_sse

if __name__ == '__main__':
    path = "./data/pred_normals.txt"
    filename = os.path.basename(path)
    pcd_path = os.path.dirname(path)
    pcd_name = filename[:-17]

    points, edge_probability, inst, vis_inst = read_points_file(pcd_path, pcd_name)
    merged_points = fit_plane(points, inst, 0.5)