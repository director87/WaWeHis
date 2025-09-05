import os
import numpy as np
import argparse
import importlib
import random
# from visdom import Visdom
import network.resnet38_cls

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from torchtoolbox.transform import Cutout
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
from tool.loss import KDLoss, PKT, Correlation, RKDLoss, Attention, SegmentationLosses
# from core.puzzle_utils import *
# from tools.ai.torch_utils import *
cudnn.enabled = True


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss



def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def train_phase(args):
    # viz = Visdom(env=args.env_name)
    model = getattr(importlib.import_module(args.network), 'Net')(args.init_gama, n_class=args.n_class)
    print(vars(args))
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.GaussianBlur(3),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                          # Cutout(),
                                          # transforms.RandomResizedCrop(size=224, scale=(0.9, 1)),
                                          transforms.ToTensor()])
                                          # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot,transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step, momentum=0.9)
    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        import network
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    else:
        print('random init')
    model = model.cuda()
    avg_meter = pyutils.AverageMeter(
            'loss',
            'avg_ep_EM',
            'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    wave = network.resnet38_cls.Class_Predictor_wavecam(6, 1568)
    # wave = network.resnet38_cls.Class_Predictor_wavecam(4, 1568)
    # wave = network.resnet38_cls.Class_Predictor_wavecam(2, 1568)
    # wave = network.resnet38_cls.Class_Predictor_wavecam(3, 1568)
    # wave = network.resnet38_cls.Class_Predictor_wavecam(2, 2048)
    wave = torch.nn.DataParallel(wave).cuda()
    for ep in range(args.max_epoches):
        model.train()
        args.ep_index = ep
        ep_count = 0
        ep_EM = 0
        ep_acc = 0
        for iter, (filename, data, label) in enumerate(train_data_loader):
            img = data
            # img2 = torchvision.transforms.RandomHorizontalFlip(p=1)(img)
            # img2 = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            # print(img, img2)
            label = label.cuda(non_blocking=True)
            # print(label, label2)
            if ep > 2:
                enable_PDA = 1
                enable_AMM = 1
                enable_NAEA = 1
                enable_MARS = 1
            else:
                enable_PDA = 0
                enable_AMM = 0
                enable_NAEA = 0
                enable_MARS = 0
            x, feature, y, cam1 = model(img.cuda(), enable_PDA=0, enable_AMM=0, enable_NAEA=0, enable_MARS=0)
            prob = y.cpu().data.numpy()
            # prob2 = y2.cpu().data.numpy()
            # prob = (prob + prob2) / 2
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls) == True:  # exact match
                    ep_EM += 1
                acc = compute_acc(pass_cls, true_cls)
                ep_acc += acc
            avg_ep_EM = round(ep_EM/ep_count, 4)
            avg_ep_acc = round(ep_acc/ep_count, 4)
            loss_cls = F.multilabel_soft_margin_loss(x, label, reduction='none')
            loss_wave = wave(feature, label, cam1)
            loss = loss_cls.mean() + 0.1*loss_wave
            avg_meter.add({'loss':loss.item(),
                            'avg_ep_EM':avg_ep_EM,
                            'avg_ep_acc':avg_ep_acc,
                           })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if (optimizer.global_step)%100 == 0 and (optimizer.global_step)!=0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'avg_ep_EM:%.4f' % (avg_meter.get('avg_ep_EM')),
                      'avg_ep_acc:%.4f' % (avg_meter.get('avg_ep_acc')),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), 
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)
                # viz.line([avg_meter.pop('loss')],[optimizer.global_step],win='loss',update='append',opts=dict(title='loss'))
                # viz.line([avg_meter.pop('avg_ep_EM')],[optimizer.global_step],win='Acc_exact',update='append',opts=dict(title='Acc_exact'))
                # viz.line([avg_meter.pop('avg_ep_acc')],[optimizer.global_step],win='Acc',update='append',opts=dict(title='Acc'))
        if model.gama > 0.65:
            model.gama = model.gama*0.98
        print('Gama of progressive dropout attention is: ',model.gama)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth'))

def test_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.cuda()
    args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth')
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    score = infer(model, args.testroot, args.n_class)
    print(score)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--session_name", default="Stage 1", type=str)
    parser.add_argument("--env_name", default="PDA", type=str)
    parser.add_argument("--model_name", default='res38d_wavecam', type=str)
    parser.add_argument("--n_class", default=6, type=int)
    parser.add_argument("--weights", default='E:/code/WSSS-Tissue-main/WSSS-Tissue-main/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='E:/code/WSSS-Tissue-main/WSSS-Tissue-main/datasets/GCSS/train/', type=str)
    parser.add_argument("--testroot", default='E:/code/WSSS-Tissue-main/WSSS-Tissue-main/datasets/GCSS/test/', type=str)
    parser.add_argument("--save_folder", default='E:/code/WSSS-Tissue-main/WSSS-Tissue-main/checkpoints/',  type=str)
    parser.add_argument("--init_gama", default=1, type=float)
    parser.add_argument("--dataset", default='gcss', type=str)
    args = parser.parse_args()
    torch.cuda.empty_cache()

    train_phase(args)
    test_phase(args)

