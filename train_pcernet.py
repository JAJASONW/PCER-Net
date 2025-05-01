import json
import logging
import os
import sys
from shutil import copyfile

import numpy as np
import torch.optim as optim
import torch.utils.data
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from read_config import Config
from src.myPointNet_v2 import PrimitivesEmbeddingDGCNGn
from src.dataset import generator_iter
from src.dataset_segments import Dataset
from src.segment_loss import *

config = Config(sys.argv[1])
print("config", config)
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

model_name = config.model_path.format(
    config.prefix,
    config.num_points,
    config.batch_size,
    config.lr,
    config.num_train,
    config.num_test,
)
print(model_name)
configure("logs/tensorboard/{}".format(model_name), flush_secs=5)

userspace = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(
    "logs/logs/{}.log".format(model_name), mode="w"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)

with open(
        "logs/configs/{}_config.json".format(model_name), "w"
) as file:
    json.dump(vars(config), file)
source_file = __file__
destination_file = "logs/scripts/{}_{}".format(
    model_name, __file__.split("/")[-1]
)
copyfile(source_file, destination_file)
if_normals = False
if_normal_noise = True
# edge_loss
edge_loss_method = config.edge_loss_method
print("edge_loss_method == ", edge_loss_method)
print("logs prepared!")

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

print("model got!")

model_bkp = model
model_bkp.l_permute = np.arange(7000)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

if config.pre_model_path != "":
    model.load_state_dict( # here
        torch.load("logs/pretrained_models/" + config.pre_model_path)
    )
print("config.pre_model_path-->", config.pre_model_path)

model.cuda()

dataset = Dataset(
    config.batch_size,
    config.num_train,
    config.num_test,
    config.num_points,
    primitives=False,
    normals=True,
    edges=True,
)

get_train_data = dataset.get_train(
    randomize=True, augment=True, align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise)
get_val_data = dataset.get_test(align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
if config.pre_model_path != "":
    optimizer.load_state_dict(torch.load("logs/pretrained_models/" +
                                         config.pre_opt_model_path))
print("config.pre_opt_model_path-->", config.pre_opt_model_path)


loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=2,
        pin_memory=False,
    )
)

loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=2,
        pin_memory=False,
    )
)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, verbose=True, min_lr=1e-4
)

model_bkp.triplet_loss = Loss.triplet_loss
prev_test_loss = 1e4
eval_inter = config.eval_T
cur_inter = 0

class sin_loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, pred, gt):
        return torch.norm(torch.cross(pred.reshape(-1, 3), gt.reshape(-1, 3), dim=-1), p=2, dim=-1).mean()
normal_pred_L1_loss = sin_loss()

for e in range(config.epochs):
    train_emb_losses = []
    train_losses = []
    train_edgeBce = []
    train_normalL1 = []
    model.train()

    # this is used for gradient accumulation because of small gpu memory.
    num_iter = 3
    for train_b_id in range(config.num_train // config.batch_size):
        optimizer.zero_grad()
        losses = 0
        embed_losses = 0
        normal_l1_losses = 0
        edge_cls_losses = 0

        torch.cuda.empty_cache()
        for _ in range(num_iter):
            # points, labels, normals, primitives = next(get_train_data)[0]
            points, labels, normals, edges, edges_W = next(get_train_data)[0]
            points = torch.from_numpy(points).cuda()
            normals = torch.from_numpy(normals).cuda()
            edges = torch.from_numpy(edges).cuda()
            edges_W = torch.from_numpy(edges_W).cuda()

            if if_normals: # Flase
                input = torch.cat([points, normals], 2)
                embedding, primitives_log_prob, embed_loss = model(
                    input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            else: #
                embedding, embed_loss, edges_pred, normal_pred = model(
                    points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )

            embed_loss = torch.mean(embed_loss)
            normal_loss = normal_pred_L1_loss(normal_pred, normals)
            if edge_loss_method == 2:
                use_hist_weight = False
                edge_loss = kl_div_loss(edges_pred, edges, use_hist_weight, edges_W)
            elif edge_loss_method == 3:
                use_hist_weight = True
                edge_loss = kl_div_loss(edges_pred, edges, use_hist_weight, edges_W)
            elif edge_loss_method == 1:
                edge_loss = edge_weighted_mae_loss(edges_pred, edges, edges_W)
            elif edge_loss_method == 0:
                edge_loss = edge_weighted_mse_loss(edges_pred, edges, edges_W)
            elif edge_loss_method == 7:
                edge_loss = edge_weighted_clean_std_huber_loss(edges_pred, edges, edges_W)

            loss = embed_loss + 0.25 * normal_loss + edge_loss
            loss.backward()

            losses += loss.data.cpu().numpy() / num_iter
            embed_losses += embed_loss.data.cpu().numpy() / num_iter
            normal_l1_losses += normal_loss.data.cpu().numpy() / num_iter
            edge_cls_losses += edge_loss.data.cpu().numpy() / num_iter

        optimizer.step()
        train_losses.append(losses)
        train_emb_losses.append(embed_losses)
        train_normalL1.append(normal_l1_losses)
        train_edgeBce.append(edge_cls_losses)

        cur_inter += 1
        print(
            "\rEpoch: {} iter: {}, normal loss: {}, emb loss: {}, edge loss: {}".format(
                e, train_b_id, normal_l1_losses, embed_losses, edge_cls_losses
            ), end="",)

        try:
            log_value("iter/emb_loss", embed_losses,
                              train_b_id + e * (config.num_train // config.batch_size))
            log_value(
                "iter/edge_loss", edge_cls_losses,
                              train_b_id + e * (config.num_train // config.batch_size))
            log_value(
                "iter/normal_loss", normal_l1_losses,
                              train_b_id + e * (config.num_train // config.batch_size))
        except OSError as e_name:
            print("\ntry-error", e_name)
        except:
            print("\nunknown-error")

        if cur_inter == eval_inter:
            torch.save(
                model.state_dict(),
                "logs/trained_models/{}_latest.pth".format(model_name),)
            torch.save(
                optimizer.state_dict(),
                "logs/trained_models/{}_optimizer_latest.pth".format(model_name),)

            cur_inter = 0
            test_emb_losses = []
            test_edgeBce = []
            test_normalL1 = []
            test_losses = []
            metrics_rmse = []
            metrics_rmse_q95 = []
            model.eval()

            for val_b_id in range(config.num_test // config.batch_size - 1):
                points, labels, normals, edges, edges_W = next(get_val_data)[0]
                points = torch.from_numpy(points).cuda()
                normals = torch.from_numpy(normals).cuda()
                edges = torch.from_numpy(edges).cuda()
                edges_W = torch.from_numpy(edges_W).cuda()
                # points, labels, normals, edges, edges_W = points.cuda(), labels.cuda(), normals.cuda(), edges.cuda(), edges_W.cuda()
                with torch.no_grad():
                    if if_normals: # Flase
                        input = torch.cat([points, normals], 2)
                        embedding, primitives_log_prob, embed_loss = model(
                            input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                        )
                    else: # embedding, embed_loss, edges_pred, normal_pred
                        embedding, embed_loss, edges_pred, normal_pred = model(
                            points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                        )

                    embed_loss = torch.mean(embed_loss)
                    normal_loss = normal_pred_L1_loss(normal_pred, normals)

                    if edge_loss_method == 2:
                        use_hist_weight = False
                        edge_loss = kl_div_loss(edges_pred, edges, use_hist_weight, edges_W)
                        edges_pred_compute_metrics = logits_to_scalar(edges_pred)  # (B, N, 1)
                        edges_pred_compute_metrics = edges_pred_compute_metrics.squeeze(2)  # (B, N, 1) -> (B, N)
                        edges_pred_compute_metrics_rmse = rmse_loss(edges_pred_compute_metrics, edges)
                        edges_pred_compute_metrics_rmse_q95 = rmse_q95_loss(edges_pred_compute_metrics, edges, q=0.95)
                    elif edge_loss_method == 3:
                        use_hist_weight = True
                        edge_loss = kl_div_loss(edges_pred, edges, use_hist_weight, edges_W)
                        edges_pred_compute_metrics = logits_to_scalar(edges_pred)  # (B, N, 1)
                        edges_pred_compute_metrics = edges_pred_compute_metrics.squeeze(2)  # (B, N, 1) -> (B, N)
                        edges_pred_compute_metrics_rmse = rmse_loss(edges_pred_compute_metrics, edges)
                        edges_pred_compute_metrics_rmse_q95 = rmse_q95_loss(edges_pred_compute_metrics, edges, q=0.95)
                    else:
                        if edge_loss_method == 1:
                            edge_loss = edge_weighted_mae_loss(edges_pred, edges, edges_W)
                        elif edge_loss_method == 0:
                            edge_loss = edge_weighted_mse_loss(edges_pred, edges, edges_W)
                        elif edge_loss_method == 7:  # sigma
                            edge_loss = edge_weighted_clean_std_huber_loss(edges_pred, edges, edges_W)

                        edges_pred_compute_metrics_rmse = rmse_loss(edges_pred, edges)
                        edges_pred_compute_metrics_rmse_q95 = rmse_q95_loss(edges_pred, edges, q=0.95)

                    loss = embed_loss + 0.25 * normal_loss + edge_loss

                test_edgeBce.append(edge_loss.data.cpu().numpy())
                test_emb_losses.append(embed_loss.data.cpu().numpy())
                test_normalL1.append(normal_loss.data.cpu().numpy())
                test_losses.append(loss.data.cpu().numpy())
                metrics_rmse.append(edges_pred_compute_metrics_rmse.data.cpu().numpy())
                metrics_rmse_q95.append(edges_pred_compute_metrics_rmse_q95.data.cpu().numpy())

            torch.cuda.empty_cache()
            print("\n")
            logger.info(
                "Epoch: {}/{} => TrL:{}, TsL:{}, TrE:{}, TsE:{}, TrN:{}, TsN:{}, TrE:{}, TsE:{}, TsE_rmse:{}, TsE_rmse_q95:{}".format(
                    e,
                    config.epochs,
                    np.mean(train_losses),
                    np.mean(test_losses),
                    np.mean(train_emb_losses),
                    np.mean(test_emb_losses),
                    np.mean(train_normalL1),
                    np.mean(test_normalL1),
                    np.mean(train_edgeBce),
                    np.mean(test_edgeBce),
                    np.mean(metrics_rmse),
                    np.mean(metrics_rmse_q95),
                )
            )

            try:
                log_value("epoch_train/emb_loss", np.mean(train_emb_losses), e)
                log_value("epoch_train/edge_loss", np.mean(train_edgeBce), e)
                log_value("epoch_train/normal_loss", np.mean(train_normalL1), e)
                log_value("epoch_train/loss", np.mean(train_losses), e)

                log_value("epoch_test/emb_loss", np.mean(test_emb_losses), e)
                log_value("epoch_test/edge_loss", np.mean(test_edgeBce), e)
                log_value("epoch_test/normal_loss", np.mean(test_normalL1), e)
                log_value("epoch_test/loss", np.mean(test_losses), e)
                log_value("epoch_test/metrics_rmse", np.mean(metrics_rmse), e)
                log_value("epoch_test/metrics_rmse_q95", np.mean(metrics_rmse_q95), e)
            except OSError as e_name:
                print("\ntry-error", e_name)
            except:
                print("\nunknown-error")


            my_crition = np.mean(test_emb_losses) + 0.2 * np.mean(test_normalL1) + np.mean(test_edgeBce)
            scheduler.step(my_crition)
            if prev_test_loss > my_crition:
                logger.info("improvement, saving model at epoch: {}".format(e))
                prev_test_loss = my_crition
                torch.save(
                    model.state_dict(),
                    "logs/trained_models/{}.pth".format(model_name),
                )
                torch.save(
                    optimizer.state_dict(),
                    "logs/trained_models/{}_optimizer.pth".format(model_name),
                )
