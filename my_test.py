import sys
import os
import numpy as np
import torch.utils.data

from src.mean_shift import MeanShift
import torch
from src.myPointNet_v2 import PrimitivesEmbeddingDGCNGn
from read_config import Config
from src.segment_loss import *

def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U

def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + 1e-8)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + 1e-8)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
                  [sin, cos, 0],
                  [0, 0, 1]])
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R

def to_one_hot(target, maxx=50, device_id=0):
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target.astype(np.int64)).cuda(device_id)
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))

    target_one_hot = target_one_hot.cuda(device_id)
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot

def guard_mean_shift(ms, embedding, quantile, iterations, kernel_type="gaussian"):
    """
    Some times if band width is small, number of cluster can be larger than 50, that
    but we would like to keep max clusters 50 as it is the max number in our dataset.
    In that case you increase the quantile to increase the band width to decrease
    the number of clusters.
    """
    while True:
        _, center, bandwidth, cluster_ids = ms.mean_shift(
            embedding, 10000, quantile, iterations, kernel_type=kernel_type
        )
        if torch.unique(cluster_ids).shape[0] > 49:
            quantile *= 1.2
        else:
            break
    return center, bandwidth, cluster_ids

config = Config(sys.argv[1])
print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
if_normals = False # Flase
num_points = config.num_points
# edge_loss
edge_loss_method = config.edge_loss_method
print("edge_loss_method == ", edge_loss_method)

Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)
if config.mode == 0:
    # Just using points for training
    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        loss_function=Loss.triplet_loss,
        mode=config.mode,
        num_channels=3,
        edge_module=True,
        normal_module=True,
        kl_hist=edge_loss_method in [2, 3],
    )

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()

ms = MeanShift()

model.eval()

iterations = 50
quantile = 0.015

state_dict = torch.load(config.pretrain_model_path)
if torch.cuda.device_count() > 1:
    state_dict = {"module." + k: state_dict[k] for k in state_dict.keys()} if not list(state_dict.keys())[0].startswith(
        "module.") else state_dict
else:
    state_dict = {k[7:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith(
        "module.") else state_dict
model.load_state_dict(state_dict, False)


file_endswith = "_GT_points.txt"

partseg_res_dir = "/data/modelnet_gt_test" # load the test data dir
fns = [fn[:-len(file_endswith)] for fn in os.listdir(partseg_res_dir) if fn.endswith(file_endswith)]
# print(fns)
save_dir = "./logs/outputs/0501m/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for fn in fns:

    print("===> predict", fn)
    pointcloud_v2_60k_path = "{}/{}".format(partseg_res_dir, fn + file_endswith)
    dense_p = np.loadtxt(pointcloud_v2_60k_path, delimiter=";", usecols=range(3))
    pointcloud_gt_edge_path = "{}/{}".format(partseg_res_dir, fn + "_GT_edge.txt")
    dense_p_gt_edge = np.loadtxt(pointcloud_gt_edge_path)

    pointn = dense_p.shape[0]
    print('point_input_nums =', pointn)

    spare_p = dense_p
    spare_p_gt_edge = dense_p_gt_edge

    if True:
        points = spare_p
        print("points_output_nums =", points.shape)
        points = points[:, :3]

        mean = np.mean(points, axis=0, keepdims=True).astype(np.float32)
        points = points - mean

        S, U = pca_numpy(points)
        smallest_ev = U[:, np.argmin(S)]
        R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        points = (R @ points.T).T
        # normals = (R @ normals.T).T
        std = np.max(points, 0) - np.min(points, 0)
        points = points / (np.max(std) + 1e-9)

        points = torch.from_numpy(points.astype(np.float32)).cuda().unsqueeze(0)
        spare_p_gt_edge = torch.from_numpy(spare_p_gt_edge.astype(np.float32)).cuda().unsqueeze(0)

        with torch.no_grad():
            if not if_normals: #
                input = points # [1, n, 3]
                embedding, _, edges_pred, normal_pred = model(
                    input.permute(0, 2, 1), None, False)
            else:
                pass

        embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)

        _, _, cluster_ids = guard_mean_shift(
            ms, embedding, quantile, iterations, kernel_type="gaussian")
        weights = to_one_hot(cluster_ids, np.unique(cluster_ids.data.data.cpu().numpy()).shape[0])
        cluster_ids = cluster_ids.data.cpu().numpy()

        if edge_loss_method in [2, 3]:
            edges_pred_compute_metrics = logits_to_scalar(edges_pred)  # (B, N, 1)
            edges_pred = edges_pred_compute_metrics.squeeze(2)  # (B, N, 1) -> (B, N)

        edges_pred = edges_pred.squeeze().cpu().detach().numpy()  # [1, N] --> [N,]

        xyz = torch.cat([points, normal_pred], 2).cpu().squeeze(0).numpy()
        np.savetxt(save_dir + f"{fn}_pred_inst.txt", cluster_ids.astype(np.int32), fmt="%d", encoding="utf-8")
        np.savetxt(save_dir + f"{fn}_pred_edge.txt", edges_pred, delimiter="\n", fmt="%0.4f")

        R_1 = np.linalg.inv(R)
        p = xyz[:, 0:3]
        n = xyz[:, 3:6]
        p = p*(np.max(std) + 1e-9)
        p0 = (R_1 @ p.T).T
        p_ori = p0 + mean

        n_ori = (R_1 @ n.T).T
        p_n_ori = np.concatenate((p_ori, n_ori), axis=1)
        np.savetxt(save_dir + f"{fn}_pred_normals.txt", p_n_ori, fmt="%0.4f", delimiter=";")



