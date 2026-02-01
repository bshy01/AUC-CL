import argparse
import os
import random

import pandas as pd
import torch
import numpy as np
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.metrics import auc_roc_score
import torch.nn.functional as F

import scipy.spatial as sp
import utils
from model import Model
from sim import sim_matrix_calc, sim_matrix_calc2, sim_matrix_arc
from dataset import DatasetAcne04Class
import cv2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 물리적 GPU 1번만 보이게 함

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

# train for one epoch to learn unique features
def train(net, data_loader, optimizer, loss_type, temperature, batch_size, epoch, epochs, loss_fn=None, accum_steps=1):
    net.train()
    if loss_type in ['simclr', 'auc-cl']:
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader)

    optimizer.zero_grad()

    for step, batch_data in enumerate(train_bar):
        # 1. 데이터 언패킹 분리
        if loss_type == 'supervised':
            # CIFAR10Single 사용: (img, label)
            inputs, labels = batch_data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            _, _, logits = net(inputs)
            loss = F.cross_entropy(logits, labels)

        else:
            # CIFAR10Pair 사용: (pos_1, pos_2, label)
            pos_1, pos_2, _ = batch_data
            pos_1, pos_2 = pos_1.to(DEVICE), pos_2.to(DEVICE)

            feature_1, out_1, _ = net(pos_1)
            feature_2, out_2, _ = net(pos_2)

            # 2. 자가학습 로직 (SimCLR / AUC-CL)
            if loss_type == 'simclr':
                out = torch.cat([out_1, out_2], dim=0)
                sim = torch.matmul(out, out.T) / temperature

                # 자기 자신 제외 마스크
                bs = out_1.size(0)
                mask = torch.eye(2 * bs, device=DEVICE).bool()
                sim = sim.masked_fill(mask, -9e15)

                # 정답 레이블 (대각선 대칭 위치)
                sim_labels = torch.cat([
                    torch.arange(bs, 2 * bs),
                    torch.arange(0, bs)
                ]).to(DEVICE)
                loss = F.cross_entropy(sim, sim_labels)

            elif loss_type == 'auc-cl':
                out = torch.cat([out_1, out_2], dim=0)
                sim_matrix = sim_matrix_calc2(out, out, mode='train')

                bs = out_1.size(0)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=DEVICE)).bool()
                neg = torch.flatten(sim_matrix.masked_select(mask))

                pos_sim = sim_matrix_calc2(out_1, out_2, mode='train')
                bs = out_1.size(0)
                mask_pos = torch.eye(bs, device=DEVICE).bool()
                pos = torch.flatten(pos_sim.masked_select(mask_pos))
                pos = torch.cat([pos, pos], dim=0)

                y_pred = torch.cat([pos, neg], dim=0)
                y_true = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
                loss = loss_fn(y_pred, y_true)

        # loss 나누기
        loss = loss / accum_steps
        loss.backward()

        # step 조건
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if loss_type == 'supervised':
            bs = inputs.size(0)
        else:
            bs = pos_1.size(0)

        total_num += bs
        total_loss += loss.item() * bs
        train_bar.set_description(f'Train Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f}')

    if (step + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_num

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_list = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out, _ = net(data.to(DEVICE))
            feature_bank.append(feature)
            target_list.append(target.to(DEVICE)    )
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(target_list, dim=0)
        # print(feature_labels.shape)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            feature, out, _ = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = sim_matrix_calc2(feature, feature_bank.t(), mode='test')
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            # sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int,
                        help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int,
                        help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128,
                        type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--loss', default='simclr', choices=['supervised', 'simclr', 'auc-cl'],
                        help='loss type: supervised | simclr | auc-cl'
    )
    parser.add_argument('--accum_steps', default=16, type=int)
    parser.add_argument('--data_path', default='/shared/data/ACNE04_Total/Cropped Faces', type=str, help='Path to dataset')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    loss_type = args.loss
    accum_steps = args.accum_steps
    data_path = args.data_path

    # data prepare

    # supervised일 때
    if loss_type == 'supervised':
        train_data = DatasetAcne04Class(root=data_path, train=True,
                                        transform=utils.train_transform,
                                        training_mode='supervised',
                                        color=cv2.IMREAD_COLOR)
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        class_weights = torch.tensor([1.0, 1.0, 2.0, 2.0]).to(DEVICE)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    # simclr/auc-cl일 때
    else:
        # Map loss_type to dataset mode
        mode_str = 'auc_cl' if loss_type == 'auc-cl' else 'simclr'
        
        train_data = DatasetAcne04Class(root=data_path, train=True,
                                       transform=utils.train_transform,
                                       training_mode=mode_str,
                                       color=cv2.IMREAD_COLOR)
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # 평가용(Memory/Test)은 모드와 상관없이 무조건 Single로 통일
    memory_data = DatasetAcne04Class(root=data_path, train=True, 
                                     transform=utils.test_transform,
                                     training_mode='supervised',
                                     color=cv2.IMREAD_COLOR)
    test_data = DatasetAcne04Class(root=data_path, train=False, 
                                   transform=utils.test_transform,
                                   training_mode='supervised',
                                   color=cv2.IMREAD_COLOR)

    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim, num_classes=4).to(DEVICE)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(DEVICE),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    c = 4 # Hardcoded for Acne04
    weight_decay = 1e-4  # /batch_size
    lr = 1e-3

    log_dir = f'runs/{loss_type}_lr{lr}_bs{batch_size}'
    writer = SummaryWriter(log_dir=log_dir)

    if loss_type == 'supervised':
        loss_fn = None
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

    elif loss_type == 'simclr':
        loss_fn = None  # InfoNCE는 함수 안에서 계산
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-6
        )

    elif loss_type == 'auc-cl':
        loss_fn = AUCMLoss(device=DEVICE)
        learnable_params = list(model.parameters()) + [loss_fn.b]
        optimizer = optim.AdamW(
            learnable_params,
            lr=lr,
            weight_decay=weight_decay
        )

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = f'{loss_type}_lr{lr}_bs{batch_size}'
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_type,
                       temperature, batch_size, epoch, epochs, loss_fn=loss_fn, accum_steps=accum_steps)
        writer.add_scalar('Loss/train', train_loss, epoch)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch, epochs)
        writer.add_scalar('Acc/top1', test_acc_1, epoch)
        writer.add_scalar('Acc/top5', test_acc_5, epoch)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre),
                          index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(),
                       'results/{}_model.pth'.format(save_name_pre))

    writer.close()