import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.ce_criterion = nn.CrossEntropyLoss()

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'camce':
            return self.CrossEntropyLoss
        elif mode == 'mvce':
            return self.CrossEntropyLoss
        elif mode == 'fsce':
            return self.CrossEntropyLoss
        elif mode == 'pdace':
            return self.CrossEntropyLoss
        elif mode == 'kdce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'se':
            return self.StructuralEntropyLoss
        elif mode == 'mi':
            return self.MutualInformationLoss
        elif mode == 'gp':
            return self.GravityPotentialLoss
        elif mode == 'ma':
            return self.MMDManifoldAlignmentLoss
        elif mode == 'oeem':
            return self.oeem
        elif mode == 'pus':
            return self.pus
        elif mode == 'urn':
            return self.urn
        elif mode == 'ns':
            return self.ns
        elif mode == 'gm':
            return self.GM_loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target, gama):
        # print(logit.size())
        n, c, h, w = logit.size()
        # n, c = logit.size()
        # print(n, c, h, w)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())
        # print(loss)

        if self.batch_average:
            loss /= n
        # print(loss.shape)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        # print(logit)
        # n, h, w = logit.size()
        n, c, h, w = logit.size()
        # loss = F.cross_entropy(logit, target.long(), weight=self.weight, ignore_index=self.ignore_index)
        # print(loss.shape[0])
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        # logpt = -criterion(logit, target)
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def SCELoss(self, logit, target, alpha=0.2, beta=0.8, num_classes=4):
        # print(logit)
        n, c, h, w = logit.size()
        # print(n, c, h, w)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        ce = criterion(logit, target.long())
        logit = F.softmax(logit, dim=1)
        logit = torch.clamp(logit, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(target.long(), num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_one_hot = label_one_hot.view(n, num_classes, 224, 224)
        # print(logit.shape)
        # print(label_one_hot.shape)
        rce = (-1 * torch.sum(logit * torch.log(label_one_hot), dim=1))
        loss = alpha * ce + beta * rce.mean()

        if self.batch_average:
            loss /= n

        return loss

    def gm(self, x, r):
        return torch.pow(torch.pow(x.reshape(x.shape[0], -1), r).mean(dim=1).unsqueeze(1), 1 / r)

    def GM_loss(self, logit, target, gama):
        # criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
        #                                 size_average=self.size_average)
        criterion = nn.BCELoss()
        if self.cuda:
            criterion = criterion.cuda()
        loss = None
        # print(self.gm(logit, 2), target.long())
        loss = loss + criterion(self.gm(logit, 2), target.long()) if loss else criterion(self.gm(logit, 2), target.long())
        return loss

    def StructuralEntropyLoss(self, logit, target, gama):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(reduction='mean')

        loss = criterion(logit, target.long())

        # 计算 softmax 概率
        prob = F.softmax(logit, dim=1)  # (n, c, h, w)

        # 计算结构熵
        adjacency_matrix = torch.einsum('nchw,nchw->nc', prob, prob)  # (n, c)
        prob_sum = torch.sum(adjacency_matrix, dim=1, keepdim=True) + 1e-10
        prob_distribution = adjacency_matrix / prob_sum

        structural_entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-10), dim=1)
        loss += 0.2 * structural_entropy.mean()

        return loss

    def MutualInformationLoss(self, logit, target, gama):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(reduction='mean')

        loss = criterion(logit, target.long())

        # 计算 softmax 概率
        prob = F.softmax(logit, dim=1)

        target = target.long()
        target_one_hot = F.one_hot(target, num_classes=5)  # (N, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # 转换为 (N, C, H, W)

        # 计算标签熵 H(Y)
        label_entropy = -torch.sum(target_one_hot * torch.log(target_one_hot + 1e-10), dim=1).mean()

        # 计算条件熵 H(Y | X)
        conditional_entropy = -torch.sum(prob * torch.log(prob + 1e-10) * target_one_hot, dim=1).mean()

        # 计算互信息
        mutual_information = label_entropy - conditional_entropy

        # 最小化互信息损失
        loss -= 0.5 * mutual_information

        return loss

    def GravityPotentialLoss(self, logit, target, gama=1.0):
        """
        引力场损失：
        假设图像中每个像素在高维特征空间中都有位置坐标，根据像素之间的距离产生“引力”或“斥力”。
        在弱监督场景中，希望同类像素之间相互吸引、异类像素相互排斥，以强化分割。

        数学思想：
        - 对每对像素 (i, j)，若预测属于同类，则引力势能 U_ij = 1 / d^2
        - 若预测不同类，则斥力势能 V_ij = d^2
        - 最终损失 = cross-entropy + gama * (Sum of potential)
        """
        n, c, h, w = logit.size()
        # cross entropy
        criterion = nn.CrossEntropyLoss(reduction='mean')

        loss = criterion(logit, target.long())

        # 计算 softmax 概率
        prob = F.softmax(logit, dim=1)  # (n, c, h, w)

        # 简化：将特征坐标设为 (row, col)
        # 生成像素坐标网格
        device = logit.device
        rows = torch.arange(h, device=device).float()
        cols = torch.arange(w, device=device).float()
        grid_y, grid_x = torch.meshgrid(rows, cols)  # (h, w), (h, w)

        # 仅示例：将 grid_x, grid_y 视为 (h*w, 2) 的坐标
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (h*w, 2)

        # 假设分类结果 pred_class = argmax(prob, dim=1) => (n, h, w)
        pred_class = torch.argmax(prob, dim=1)  # (n, h, w)

        # 引力/斥力势能 (简化实现：仅batch中第一个样本)
        # 大规模时应随机抽取像素对以降低计算量
        sample_idx = 0
        pc = pred_class[sample_idx].flatten()  # (h*w,)
        pot_loss = 0.0
        # 随机选取 M 个像素对
        M = 200  # 超参
        total_pixels = h * w
        idxs = torch.randperm(total_pixels)[:M * 2]
        idx_pairs = idxs.view(-1, 2)  # (M, 2)

        for i in range(M):
            i1, i2 = idx_pairs[i, 0], idx_pairs[i, 1]
            c1, c2 = pc[i1], pc[i2]
            dist = torch.norm(coords[i1] - coords[i2]) + 1e-5
            if c1 == c2:
                # 同类 => 引力势能 U = 1 / dist^2
                pot_loss += 1.0 / (dist ** 2)
            else:
                # 异类 => 斥力 V = dist^2
                pot_loss += dist ** 2

        pot_loss = pot_loss / M  # 平均势能

        # 总损失
        loss = loss + 1 * pot_loss
        return loss

    def rbf_kernel(self, x, y, gamma=1.0):
        dist = (x - y).pow(2).sum(dim=-1)
        return torch.exp(-dist * gamma)

    # class MMDManifoldAlignmentLoss(nn.Module):
    #     def __init__(self, lambda_mmd=0.1, gamma=1.0):
    #         super(MMDManifoldAlignmentLoss, self).__init__()
    #         self.lambda_mmd = lambda_mmd
    #         self.gamma = gamma
    #         self.ce_criterion = nn.CrossEntropyLoss()

    def MMDManifoldAlignmentLoss(self, logit_a, logit_b, target_a, target_b):
        """
        logit_a, logit_b: (N, C, H, W)
        target_a, target_b: (N, H, W)
        """
        ce_loss_a = self.ce_criterion(logit_a, target_a.long())
        ce_loss_b = self.ce_criterion(logit_b, target_b.long())
        ce_loss = (ce_loss_a + ce_loss_b) / 2

        # 提取特征(示例: 直接用softmax均值)
        prob_a = F.softmax(logit_a, dim=1).mean(dim=(2, 3))  # (N, C)
        prob_b = F.softmax(logit_b, dim=1).mean(dim=(2, 3))  # (N, C)

        # 计算 MMD
        mmd_loss = self._compute_mmd(prob_a, prob_b)
        return ce_loss + 0.1 * mmd_loss

    def _compute_mmd(self, xa, xb):
        # 简易 MMD
        # E[ka(xa,xa)] + E[kb(xb,xb)] - 2 E[kab(xa,xb)]
        ka = 0
        kb = 0
        kab = 0
        for i in range(xa.size(0)):
            for j in range(xa.size(0)):
                ka += self.rbf_kernel(xa[i], xa[j], 1)
                kb += self.rbf_kernel(xb[i], xb[j], 1)
        ka /= xa.size(0) * xa.size(0)
        kb /= xb.size(0) * xb.size(0)

        for i in range(xa.size(0)):
            for j in range(xb.size(0)):
                kab += self.rbf_kernel(xa[i], xb[j], 1)
        kab /= xa.size(0) * xb.size(0)
        return ka + kb - 2 * kab


    def weight_loss(self, loss):
        n = loss.shape[0]
        loss = loss.view(n, -1)
        loss_weight = F.softmax(loss.clone().detach(), dim=1) / torch.mean(
            F.softmax(loss.clone().detach(), dim=1), dim=1, keepdim=True
        )
        loss = torch.sum(loss * loss_weight) / (n * loss.shape[1])
        return loss

    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.

        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".

        Return:
            Tensor: Reduced loss tensor.
        """
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()

    def weight_reduce_loss(self, loss, weight=None, reduction='mean', avg_factor=None):
        """Apply element-wise weight and reduce loss.

        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Avarage factor when computing the mean of losses.

        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            assert weight.dim() == loss.dim()
            if weight.dim() > 1:
                assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def oeem(self, pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
        loss = F.cross_entropy(pred, label.long(), weight=class_weight, reduction='none', ignore_index=ignore_index)

        weight = torch.ones_like(loss)
        metric = -loss.detach().reshape((loss.shape[0], loss.shape[1] * loss.shape[2]))
        weight = F.softmax(metric, 1)  # sm(-L)
        weight = weight / weight.mean(1).reshape((-1, 1))  # sm(-L)/mean(sm(-L))
        weight = weight.reshape((loss.shape[0], loss.shape[1], loss.shape[2]))
        sm_x = F.softmax(pred.detach().reshape((pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])), 1)
        max_x = torch.max(sm_x, dim=1, keepdim=False)
        weight = torch.mul(weight, max_x[0])

        # apply onss on images of multiple labels
        for i in range(label.shape[0]):
            tag = set(label[i].reshape(label.shape[1] * label.shape[2]).tolist()) - {255}
            if len(tag) <= 1:
                weight[i] = 1

            # apply weights and reduction
        if weight is not None:
            weight = weight.float()
        loss = self.weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    def ns(self, pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
        loss = F.cross_entropy(pred, label.long(), weight=class_weight, reduction='none', ignore_index=ignore_index)
        # print(pred.shape)

        # if loss.mean() <= 0.5:

        weight = torch.ones_like(loss)
        metric = -loss.detach().reshape((loss.shape[0], loss.shape[1] * loss.shape[2]))
        weight = F.softmax(metric, 1)  # sm(-L)
        # print(torch.max(weight), torch.min(weight), torch.mean(weight))
        # print(torch.max(weight, dim=1, keepdim=False)[0])
        weight = weight / (torch.max(weight, dim=1, keepdim=False)[0] - torch.min(weight, dim=1, keepdim=False)[0]).reshape(-1, 1)
        # weight = weight / weight.mean(1).reshape((-1, 1))  # sm(-L)/mean(sm(-L))
        weight = weight.reshape((loss.shape[0], loss.shape[1], loss.shape[2]))
        sm_x = F.softmax(pred.detach().reshape((pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])), 1)
        max_x = torch.max(sm_x, dim=1, keepdim=False)
        weight = torch.mul(weight, max_x[0])

        # apply onss on images of multiple labels
        for i in range(label.shape[0]):
            tag = set(label[i].reshape(label.shape[1] * label.shape[2]).tolist()) - {255}
            if len(tag) <= 1:
                weight[i] = 1
        # else:
        #     weight = class_weight

        # apply weights and reduction
        if weight is not None:
            weight = weight.float()
        loss = self.weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    def pus(self, pred,
            label,
            gama,
            weight=None,
            class_weight=None,
            reduction='mean',
            avg_factor=None,
            pus_type='clamp',
            pus_beta=0.3,
            pus_k=0.3,
            ignore_index=255):
        """The wrapper function for :func:`F.cross_entropy`"""
        # class_weight is a manual rescaling weight given to each class.
        # If given, has to be a Tensor of size C element-wise losses
        loss = F.cross_entropy(
            pred,
            label.long(),
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)
        # print(loss.mean())
        # if gama > 14:
        #     pus_k = pus_beta = 0.5
        # gama *= 0.98

        # Pretended Under-fitting Strategy
        if pus_type != "" and loss.mean() < pus_beta:
            if pus_type == 'pow':
                loss = torch.pow(loss + 0.000000001, pus_k)
            elif pus_type == 'clamp':
                loss = torch.clamp(loss, 0, pus_k)
            elif pus_type == 'ignore':
                loss = loss * (loss <= pus_k).float()
            else:
                pass

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = self.weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    def urn(self, pred,
            label,
            gama,
            weight=None,
            class_weight=None,
            reduction='mean',
            avg_factor=None,
            pus_type='clamp',
            pus_beta=0.8,
            pus_k=0.8,
            weight_thresh=0.01,
            ignore_index=255):
        """The wrapper function for :func:`F.cross_entropy`"""

        # class_weight is a manual rescaling weight given to each class.
        # If given, has to be a Tensor of size C element-wise losses
        loss = F.cross_entropy(
            pred,
            label.long(),
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)
        # smw_loss = self.weight_loss(loss)
        # print(F.softmax(loss))
        # loss_sm = F.softmax(loss)
        # print(loss_sm)
        # proc_pred = pred.mean(dim=1)
        # diff = proc_pred - label
        # print(diff)
        max = torch.max(loss)
        min = torch.min(loss)
        # print(max.item(), min.item())
        condition = loss > ((max.item() - min.item()) / 2)
        # print(loss.mean())
        # condition = loss > (loss.mean())
        weight_mask = torch.where(condition.cuda(), loss, torch.tensor(0.0).cuda())
        # weight_mask = diff
        weight_mask = weight_mask.float() / 255
        # print(weight_mask)
        uncertain_mask = weight_mask >= weight_thresh
        weight_mask[uncertain_mask == 1] = weight_thresh * gama
        weight_mask[uncertain_mask == 0] = 1
        # print(weight_mask.shape)

        if weight_thresh >= 0:
            loss = loss * weight_mask

        # print(torch.max(loss).item(), torch.min(loss).item())
        # print(loss.mean())

        # Underfit Cheating
        if pus_type != "" and loss.mean() < pus_beta:
            if pus_type == 'pow':
                loss = torch.pow(loss + 0.000000001, pus_k)
            elif pus_type == 'clamp':
                loss = torch.clamp(loss, 0, pus_k)
            elif pus_type == 'ignore':
                loss = loss * (loss <= pus_k).float()
            elif pus_type == 'mix':
                if loss.mean() < pus_k:
                    # loss = loss * (loss <= pus_k).float()
                    loss = torch.pow(loss + 0.000000001, pus_k)
                elif pus_k < loss.mean() < pus_beta:
                    loss = torch.clamp(loss, 0, pus_k)
            else:
                pass
        # if loss.mean() > pus_beta:
        #     loss = torch.pow(loss + 0.000000001, pus_k)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = self.weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
                F.kl_div(F.log_softmax(out_s / self.T, dim=0),
                         F.softmax(out_t / self.T, dim=0), reduction="batchmean")
                * self.T
                * self.T
        )
        return loss


class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""

    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""

    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        # loss = self.crit(F.softmax(f_s / 1, dim=0), F.softmax(f_t / 1, dim=0))
        return loss


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""

    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""

    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        # return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        list = [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        list = torch.stack(list)
        sum = torch.sum(list)
        return sum

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class Correlation(nn.Module):
    """Similarity-preserving loss. My origianl own reimplementation
    based on the paper before emailing the original authors."""

    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        return self.similarity_loss(f_s, f_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = G_s / G_s.norm(2)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = G_t / G_t.norm(2)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, w_d=5, w_a=10):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

class AT(nn.Module):
    '''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am


def L2_norm(p, q, label_mask, eps):
    print((torch.norm(p + eps, dim=(-1, -2,), keepdim=True)).shape)
    loss = torch.mean(torch.pow((p / torch.norm(p + eps, dim=(-1, -2,), keepdim=True)) - (q / torch.norm(q + eps, dim=(-1, -2,), keepdim=True)), 2) * label_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    return loss


class MapLossL2Norm(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(MapLossL2Norm, self).__init__()
        self.epsilon = epsilon

    def forward(self, p, q, label_mask):
        return L2_norm(p, q, label_mask, self.epsilon)


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
