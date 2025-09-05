import argparse
import os
import time
import numpy as np
import math
import random
import cv2
from tqdm import tqdm
from PIL import Image
import torch
from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import *
from network.UNet import UNet
from tool.loss import SegmentationLosses, KDLoss, PKT, HintLoss, Correlation, RKDLoss, AT
from tool.lr_scheduler import LR_Scheduler, CosineAnnealingWarmRestarts
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator
import ml_collections
import segmentation_models_pytorch as smp
from collections import defaultdict
from scipy import stats
from torchvision import transforms
import timm

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 8
    config.transformer.num_layers = 8
    config.expand_ratio           = 16  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

def get_base_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 4
    config.activation = 'softmax'
    return config

def get_TFCNs_config():
    config = get_base_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 4
    config.n_skip = 3
    config.activation = 'softmax'

    return config

def voting(outputs_main, outputs_aux1, outputs_aux2, mask):
    n = outputs_main.shape[0]
    # e = math.sqrt(ep)
    loss_main = F.cross_entropy(
        outputs_main, mask.long(), reduction='none').view(n, -1)
    hard_aux1 = torch.argmax(outputs_aux1, dim=1).view(n, -1)
    hard_aux2 = torch.argmax(outputs_aux2, dim=1).view(n, -1)
    loss_select = 0
    for i in range(n):
        aux1_sample = hard_aux1[i]
        aux2_sample = hard_aux2[i]
        loss_sample = loss_main[i]
        agree_aux = (aux1_sample == aux2_sample)
        disagree_aux = (aux1_sample != aux2_sample)
        loss_select += 2*torch.sum(loss_sample[agree_aux]) + 0.5*torch.sum(loss_sample[disagree_aux])
        # loss_select += math.exp(-e)*torch.sum(loss_sample[agree_aux]) + (1 / math.exp(-e))*torch.sum(loss_sample[disagree_aux])
        # loss_select += math.pow(2, e)*torch.sum(loss_sample[agree_aux]) + (1 / math.pow(2, e))*torch.sum(loss_sample[disagree_aux])

    return loss_select / (n*loss_main.shape[1])

def joint_optimization(outputs_main, outputs_aux1, outputs_aux2, mask, kd_weight, kd_T, vote_weight):
    kd_loss = HintLoss()
    avg_aux = (outputs_aux1 + outputs_aux2) / 2

    L_kd1 = kd_loss(outputs_main.permute(0, 2, 3, 1).reshape(-1, 4),
                    outputs_aux1.permute(0, 2, 3, 1).reshape(-1, 4))
    L_kd2 = kd_loss(outputs_main.permute(0, 2, 3, 1).reshape(-1, 4),
                    outputs_aux2.permute(0, 2, 3, 1).reshape(-1, 4))
    L_kd3 = kd_loss(outputs_main.permute(0, 2, 3, 1).reshape(-1, 4),
                    avg_aux.permute(0, 2, 3, 1).reshape(-1, 4))
    L_kd = (L_kd1 + L_kd2 + L_kd3) / 3
    L_vote = voting(outputs_main, outputs_aux1, outputs_aux2, mask)
    L = vote_weight * L_vote + kd_weight * L_kd
    return L

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.gama = 1.0
        # Define
        self.saver = Saver(args)
        self.summary = TensorboardSummary('logs')
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        model = smp.PSPNet(encoder_name='timm-resnest200e', encoder_weights='imagenet', in_channels=3, classes=self.nclass)
        lr = args.lr
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Using cuda
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.cuda:
                W = checkpoint['state_dict']
                if not args.ft:
                    del W['decoder.last_conv.8.weight']
                    del W['decoder.last_conv.8.bias']
                self.model.module.load_state_dict(W, strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            if args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(args.resume))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target, target_a, target_b = sample['image'], sample['label'], sample['label_a'], sample['label_b']
            image_small = F.interpolate(image, scale_factor=0.75, mode='bilinear',
                                        align_corners=True,
                                        recompute_scale_factor=True)
            image_big = F.interpolate(image, scale_factor=1.25, mode='bilinear',
                                      align_corners=True,
                                      recompute_scale_factor=True)
            if self.args.cuda:
                image, target, target_a, target_b = image.cuda(), target.cuda(), target_a.cuda(), target_b.cuda()
                image_small, image_big = image_small.cuda(), image_big.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            self.optimizer.zero_grad()

            output = self.model(image)
            output2 = self.model(image_small)
            output3 = self.model(image_big)
            output2 = F.interpolate(output2, size=(224,224), mode='bilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(224,224), mode='bilinear', align_corners=True)
            one = torch.ones((output.shape[0],1,224,224)).cuda()
            # one2 = torch.ones((output2.shape[0],1,224,224)).cuda()
            # one3 = torch.ones((output3.shape[0],1,224,224)).cuda()
            # one = torch.ones((output[0].shape[0],1,224,224)).cuda()
            # print(output)
            # print((100 * one * (target==4).unsqueeze(dim = 1)).shape)
            # output = torch.cat([output,(100 * one * (target==0).unsqueeze(dim = 1))],dim = 1)
            # output = torch.cat([output,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)
            # output2 = torch.cat([output2,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)
            # output3 = torch.cat([output3,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)
            if self.args.dataset == 'wsss' or self.args.dataset == 'w4l':
                one = torch.ones((output.shape[0], 1, 224, 224)).cuda()
                output = torch.cat([output, (100 * one * (target == 3).unsqueeze(dim=1))], dim=1)
                # output2 = torch.cat([output2, (100 * one * (target == 3).unsqueeze(dim=1))], dim=1)
                # output3 = torch.cat([output3, (100 * one * (target == 3).unsqueeze(dim=1))], dim=1)

            loss_o = self.criterion(output, target, self.gama)
            loss_kd = joint_optimization(output, output2, output3, target, kd_weight=0.1, kd_T=30, vote_weight=0)
            loss = loss_o + loss_kd
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        # if (epoch + 1) % 3 == 0:
        # self.gama = self.gama * 0.95

    def validation(self, epoch):
        time.sleep(0.003)
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                # print(image.shape, target.shape)
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            # pred = output[0].data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            # pred[target==4]=4
            if self.args.dataset == 'wsss' or self.args.dataset == 'w4l':
                pred[target==3]=3
            elif self.args.dataset == 'bcss' or self.args.dataset == 'luad' or self.args.dataset == 'zjch':
                pred[target==4]=4
            elif self.args.dataset == 'gcss':
                pred[target == 0] = 0
            # pred[target==3]=3
            self.evaluator.add_batch(target, pred)

        # Fast test during the training (Validation)
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        mDice = self.evaluator.Mean_Dice_Similarity_Coefficient()
        dices = self.evaluator.Dice_Similarity_Coefficient()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/mDice', mDice, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, mDice: {}".format(Acc, Acc_class, mIoU, FWIoU, mDice))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)
        print('Dices: ', dices)

        if mIoU > self.best_pred:
            self.best_pred = mIoU
            self.saver.save_checkpoint({
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, 'stage2_checkpoint_trained_on_'+self.args.dataset+self.args.backbone+self.args.loss_type+'.pth')
    def load_the_best_checkpoint(self):
        checkpoint = torch.load('E:/code/WSSS-Tissue-main/WSSS-Tissue-main/checkpoints/stage2_checkpoint_trained_on_'+self.args.dataset+self.args.backbone+self.args.loss_type+'.pth')
        # checkpoint = torch.load(r'E:\code\WSSS-Tissue-main\WSSS-Tissue-main\checkpoints\stage2_checkpoint_trained_on_luadpistoseg_ce.pth')
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
    def test(self, epoch, Is_GM):
        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            # print(target.shape)
            # print(sample)
            image_name = sample[-1][0].split('/')[-1].replace('.png', '')
            # print(image_name, "----")
            # print(image_name)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if Is_GM:
                    output = self.model(image)
                    # print(output.shape)
                    _,y_cls = self.model_stage1.forward_cam(image)
                    y_cls = y_cls.cpu().data
                    # y_cls = y_cls.cpu().data
                    # print(y_cls)
                    pred_cls = (y_cls > 0.1)
            pred = output.data.cpu().numpy()
            # pred = output[0].data.cpu().numpy()
            # print(pred.shape)
            if Is_GM:
                pred = pred*(pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # print(pred)
            ## cls 4 is exclude
            # pred[target==4]=4
            if self.args.dataset == 'wsss' or self.args.dataset == 'w4l':
                pred[target==3]=3
            elif self.args.dataset == 'bcss' or self.args.dataset == 'luad' or self.args.dataset == 'zjch':
                pred[target==4]=4
            elif self.args.dataset == 'gcss':
                pred[target==0]=0
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        mDice = self.evaluator.Mean_Dice_Similarity_Coefficient()
        dices = self.evaluator.Dice_Similarity_Coefficient()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/DSC', mDice, epoch)
        print('Test:')
        print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, mDice: {}".format(Acc, Acc_class, mIoU, FWIoU, mDice))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)
        print('Dices: ', dices)

def main():
    parser = argparse.ArgumentParser(description="WSSS Stage2")
    parser.add_argument('--backbone', type=str, default='wavecam')
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--Is_GM', type=bool, default=False, help='Enable the Gate mechanism in test phase')
    parser.add_argument('--dataroot', type=str, default='E:/code/WSSS-Tissue-main/WSSS-Tissue-main/datasets/GCSS')
    parser.add_argument('--dataset', type=str, default='gcss')
    parser.add_argument('--savepath', type=str, default='E:/code/WSSS-Tissue-main/WSSS-Tissue-main/checkpoints/')
    parser.add_argument('--workers', type=int, default=1, metavar='N')
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--n_class', type=int, default=6)
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR')
    parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    parser.add_argument('--nesterov', action='store_true', default=True)
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    # checking point
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)

    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    print(args)
    trainer = Trainer(args)
    for epoch in range(trainer.args.epochs):
        # pass
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.test(epoch, args.Is_GM)
    trainer.writer.close()

if __name__ == "__main__":
    # torch.cuda.empty_cache()
    main()
