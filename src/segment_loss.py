"""
This script defines loss functions for AE based training.
"""
import numpy as np
import torch
from torch.nn import ReLU
from torch import nn
import torch.nn.functional as F
from torch import erf
from typing import List

from src.mean_shift import MeanShift

meanshift = MeanShift()
WEIGHT = False
relu = ReLU()

if WEIGHT:
    nllloss = torch.nn.NLLLoss(weight=old_weight)
else:
    nllloss = torch.nn.NLLLoss()


class EmbeddingLoss:
    def __init__(self, margin=1.0, if_mean_shift=False):
        """
        Defines loss function to train embedding network.
        :param margin: margin to be used in triplet loss.
        :param if_mean_shift: bool, whether to use mean shift
        iterations. This is only used in end to end training.
        """
        self.margin = margin
        self.if_mean_shift = if_mean_shift

    def triplet_loss(self, output, labels: np.ndarray, iterations=5):
        """
        Triplet loss
        :param output: output embedding from the network. size: B x 128 x N
        where B is the batch size, 128 is the dim size and N is the number of points.
        :param labels: B x N
        """
        max_segments = 5
        batch_size = output.shape[0]
        N = output.shape[2]
        loss_diff = torch.tensor([0.], requires_grad=True).cuda()
        relu = torch.nn.ReLU()

        output = output.permute(0, 2, 1)
        output = torch.nn.functional.normalize(output, p=2, dim=2)
        new_output = []

        if self.if_mean_shift:
            for b in range(batch_size):
                new_X, bw = meanshift.mean_shift(output[b], 4000,
                                                 0.015, iterations=iterations,
                                                 nms=False)
                new_output.append(new_X)
            output = torch.stack(new_output, 0)

        num_sample_points = {}
        sampled_points = {}
        for i in range(batch_size):
            sampled_points[i] = {}
            p = labels[i]
            unique_labels = np.unique(p)

            # number of points from each cluster.
            num_sample_points[i] = min([N // unique_labels.shape[0] + 1, 30])
            for l in unique_labels:
                ix = np.isin(p, l)
                sampled_indices = np.where(ix)[0]
                # point indices that belong to a certain cluster.
                sampled_points[i][l] = np.random.choice(
                    list(sampled_indices),
                    num_sample_points[i],
                    replace=True)

        sampled_predictions = {}
        for i in range(batch_size):
            sampled_predictions[i] = {}
            for k, v in sampled_points[i].items():
                pred = output[i, v, :]
                sampled_predictions[i][k] = pred

        all_satisfied = 0
        only_one_segments = 0
        for i in range(batch_size):
            len_keys = len(sampled_predictions[i].keys())
            keys = list(sorted(sampled_predictions[i].keys()))
            num_iterations = min([max_segments * max_segments, len_keys * len_keys])
            normalization = 0
            if len_keys == 1:
                only_one_segments += 1
                continue

            loss_shape = torch.tensor([0.], requires_grad=True).cuda()
            for _ in range(num_iterations):
                k1 = np.random.choice(len_keys, 1)[0]
                k2 = np.random.choice(len_keys, 1)[0]
                if k1 == k2:
                    continue
                else:
                    normalization += 1

                pred1 = sampled_predictions[i][keys[k1]]
                pred2 = sampled_predictions[i][keys[k2]]

                Anchor = pred1.unsqueeze(1)
                Pos = pred1.unsqueeze(0)
                Neg = pred2.unsqueeze(0)

                diff_pos = torch.sum(torch.pow((Anchor - Pos), 2), 2)
                diff_neg = torch.sum(torch.pow((Anchor - Neg), 2), 2)
                constraint = diff_pos - diff_neg + self.margin
                constraint = relu(constraint)

                # remove diagonals corresponding to same points in anchors
                loss = torch.sum(constraint) - constraint.trace()

                satisfied = torch.sum(constraint > 0) + 1.0
                satisfied = satisfied.type(torch.cuda.FloatTensor)

                loss_shape = loss_shape + loss / satisfied.detach()

            loss_shape = loss_shape / (normalization + 1e-8)
            loss_diff = loss_diff + loss_shape
        loss_diff = loss_diff / (batch_size - only_one_segments + 1e-8)
        return loss_diff

def evaluate_miou(gt_labels, pred_labels):
    N = gt_labels.shape[0]
    C = pred_labels.shape[2]
    pred_labels = np.argmax(pred_labels, 2)
    IoU_category = 0

    for n in range(N):
        label_gt = gt_labels[n]
        label_pred = pred_labels[n]
        IoU_part = 0.0

        for label_idx in range(C):
            locations_gt = (label_gt == label_idx)
            locations_pred = (label_pred == label_idx)
            I_locations = np.logical_and(locations_gt, locations_pred)
            U_locations = np.logical_or(locations_gt, locations_pred)
            I = np.sum(I_locations) + np.finfo(np.float32).eps
            U = np.sum(U_locations) + np.finfo(np.float32).eps
            IoU_part = IoU_part + I / U
        IoU_sample = IoU_part / C
        IoU_category += IoU_sample
    return IoU_category / N

def primitive_loss(pred, gt):
    return nllloss(pred, gt)

def edge_cls_loss(edges_pred, edges_label, bce_W):
    """
    对点云分类是否为边界的二分类软标签 Bce loss
    edges_pred: B x 2 x N , 无 softmax
    edges_label: B x N
    bce_W: B x N

    return: 一个标量值
    """
    BCEloss = nn.CrossEntropyLoss(reduction='none')

    _loss = (BCEloss(edges_pred, edges_label) * bce_W).mean(-1) # (B, )
    _loss[bce_W.sum(-1) == 0] = 0
    return torch.mean(_loss)

def rmse_loss(output, target):
    """
    用于回归评估
    output: B x N , 无 softmax
    target: B x N

    return: 一个标量值，代表所有 batch 的 RMSE 损失的平均值
    """
    # 计算均方根误差损失
    mse_loss = F.mse_loss(output, target, reduction='none') # b*n
    rmse_loss = torch.sqrt(mse_loss.mean(dim=-1)) # (B, )
    return torch.mean(rmse_loss)

def rmse_q95_loss(output, target, q=0.95):
    """
    用于回归评估 RMSE-q95 损失
    output: B x N , 无 softmax
    target: B x N
    q: 分位数，范围在 0 到 1 之间

    return: 一个标量值，代表所有 batch 的 RMSE-q95 损失的平均值
    """
    # 计算均方根误差损失
    mse_loss = F.mse_loss(output, target, reduction='none') # b*n
    # 计算分位数
    q_loss = torch.quantile(mse_loss, q=q, dim=-1) # (B, )
    rmse_q95_loss = torch.sqrt(q_loss.mean()) # 一个标量值
    return rmse_q95_loss

def edge_weighted_mse_loss(output, target, weight):
    """
    output: B x N , 无 softmax
    target: B x N
    weight: B x N

    return: 一个标量值
    """
    mse_loss = F.mse_loss(output, target, reduction='none') # b*n
    weighted_loss = mse_loss * weight # b*n
    return torch.mean(weighted_loss)

def edge_weighted_mae_loss(output, target, weight):
    """
    output: B x N , 无 softmax
    target: B x N
    weight: B x N

    return: 一个标量值
    """
    mae_loss = F.l1_loss(output, target, reduction='none') # b*n
    weighted_loss = mae_loss * weight # b*n
    return torch.mean(weighted_loss)

def edge_weighted_clean_std_huber_loss(output, target, weight):
    batch_size, num_points = output.size()
    weight_std = torch.std(weight, dim=1, unbiased=False)
    weight_std = torch.clamp(weight_std, min=1e-8)
    huber_loss = torch.zeros((batch_size, num_points), dtype=output.dtype, device=output.device)
    for i in range(batch_size):
        huber_loss[i] = F.smooth_l1_loss(output[i], target[i], reduction='none', beta=weight_std[i].item())
    weighted_loss = huber_loss * weight
    return torch.mean(weighted_loss)

def kl_div_loss(input: torch.Tensor, target: torch.Tensor,
                use_weight: bool = False, weight: torch.Tensor = None,
                a: float = 0.0, b: float = 1.0, discretization: int = 240, margin: int = 2):
    """
    Implementation of the KL divergence (histogram) loss with target scalar->distribution preprocessing

    Args:
        input (Tensor): of shape (B, discretization + 2 * marginm, N). Logits ---> edges_pred
        target (Tensor): of shape (B, N) ---> edges
        use_weight (bool): whether to use weight for balancing
        weight (Tensor): weight tensor of shape (B, N) ---> edges_W

        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        Tensor: loss value 一个标量
    """
    input = input.transpose(1, 2) # (B, N, discretization + 2 * marginm)
    assert margin >= 0 and input.size(-1) == discretization + 2 * margin

    with torch.no_grad():
        density = _calculate_density(target, a, b, discretization, margin)
    input = F.log_softmax(input, dim=-1)

    if use_weight and weight is not None:
        weight = weight.unsqueeze(-1).expand_as(input)
        weighted_kl_loss = F.kl_div(input, density, reduction='none') * weight
        return weighted_kl_loss.mean()

    return F.kl_div(input, density)

def _normal_cdf(x, mean, sigma):
    return 0.5 + 0.5 * erf((x - mean) / (torch.sqrt(torch.tensor(2.0)) * sigma))

def _calculate_density(mean: torch.Tensor, a: float, b: float, discretization: int, margin: int) -> torch.Tensor:
    """
    Args:
        mean (Tensor): of shape (B, *)
        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        Tensor: of shape (B, *, discretization + margin * 2)
    """
    bin_limits = _calculate_bin_limits(mean.shape, a, b, discretization, margin,
                                       device=mean.device)  # (B, *, discretization + margin * 2)
    mean = mean.unsqueeze(-1)  # (B, *, 1)
    sigma = 1.0 / discretization
    norm_coefficient = _normal_cdf(b, mean, sigma) - _normal_cdf(a, mean, sigma)  # (B, *, 1)
    density = (_normal_cdf(bin_limits[..., 1:], mean, sigma) - _normal_cdf(bin_limits[..., :-1], mean, sigma))
    density = density / norm_coefficient
    return density

def _calculate_bin_limits(shape: List, a: float, b: float, discretization: int, margin: int, device: torch.device) \
        -> torch.Tensor:
    """
    Args:
        shape (list): desired shape
        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        bin_limits (Tensor): of shape (*shape, discretization + margin * 2 + 1)
    """

    a = a - margin * (1 / discretization)
    b = b + margin * (1 / discretization)

    num_bins = discretization + margin * 2

    bin_limits = torch.linspace(a, b, num_bins + 1, device=device) \
        .view(*[1 for _ in range(len(shape))], num_bins + 1) \
        .expand(*shape, num_bins + 1)

    return bin_limits

def logits_to_scalar(input: torch.Tensor, a: float = 0.0, b: float = 1.0, discretization: int = 240, margin: int = 2):
    """
    Convert logits to scalar

    Args:
        input (Tensor): of shape (B, discretization + 2 * marginm, N). Logits ---> edges_pred
        a (float): left corner of the interval
        b (float): right corner of the interval
        discretization (int): the amount of bins inside the interval
        margin (int): how many bins to add to each corner (may improve performance)
    Returns:
        Tensor: scalar values of size (B, N, 1)
    """
    input = input.transpose(1, 2)  # (B, N, discretization + 2 * marginm)
    assert margin >= 0 and input.size(-1) == discretization + 2 * margin

    with torch.no_grad():
        bin_limits = _calculate_bin_limits(input.shape[:-1], a, b, discretization, margin, input.device)
        bin_centers = bin_limits[..., :-1] + (1.0 / discretization / 2.0)  # (B, *, discretization + 2 * margin)

    return (input.softmax(dim=-1) * bin_centers).sum(dim=-1, keepdim=True)